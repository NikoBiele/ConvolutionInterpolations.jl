# Eager's ghost rule for one side of one axis (see `create_convolutional_coefs`), always as
# (matrix, detect, reach) so that tuples of rules have one concrete type: the ghost matrix, or for
# :detect the polynomial matrix with detect = true (then decided per line between it and the linear
# matrix); and how many interior values along the axis, from the boundary inward, the rule reads.
function _lazy_side_rule(bc::Symbol, kernel::Symbol, n::Int, eqs::Int)
    poly = get_polynomial_ghost_coeffs(:poly, kernel)     # the kernel's polynomial ghost matrix
    few_points = n < size(poly, 2)                        # too few values for the polynomial matrix
    if bc === :linear || bc === :quadratic
        g = get_polynomial_ghost_coeffs(bc, kernel)
        return g, false, size(g, 2)
    elseif few_points                                     # eager falls back to the linear matrix
        g = get_polynomial_ghost_coeffs(:linear, kernel)
        return g, false, size(g, 2)
    elseif bc === :poly
        return poly, false, size(poly, 2)
    elseif bc === :detect                                 # decided per line; eager reads m values
        return poly, true, min(n, max(size(poly, 2) + 3, eqs))
    else
        throw(ArgumentError("unsupported boundary condition: $bc"))
    end
end

# Eager's :detect decision for one line: the m values nearest the boundary, in increasing index
# order as eager stores its slice, mean-centered exactly as eager centers them, then the same guard.
# `inward(p)` is the value p steps from the boundary (p = 1 at the boundary itself).
function _lazy_detect(slice::Vector{T}, inward, m::Int, left::Bool, poly::AbstractMatrix,
                      eqs::Int) where {T}
    length(slice) < m && resize!(slice, m)                # grown on first use only
    for j in 1:m
        # left: j steps inward; right: the last m values, entry j lies m − j + 1 steps from the boundary
        slice[j] = left ? inward(j) : inward(m - j + 1)
    end
    y_mean = zero(T)                                      # eager's mean: sequential sum, then divide
    for j in 1:m
        y_mean += slice[j]
    end
    y_mean /= m
    for j in 1:m
        slice[j] -= y_mean
    end
    return bc_accept_polynomial(T, poly, view(slice, 1:m), left ? 1 : m, left ? 1 : -1,
                                m, eqs - 1, one(T), nothing)
end

"""
    _lazy_patch_sum(itp, pos, w)

Stencil sum of a lazy interpolant in a boundary cell: the sum over the stencil offsets k (per axis
1 … 2·eqs[d], coefficient index pos[d] + k − eqs[d]) of coefficient · Π_d w[d][k_d]. Coefficients
outside the domain are the ghost values eager stores there, formed the same way on a small local
patch: the data values are copied into the patch, then the ghosts are filled axis by axis (1 to N)
with `_compensated_row_dot` in eager's term order, so each ghost is computed once, and a value
outside the domain in several axes is formed from the rounded ghosts of the lower axes, bit for
bit as in `create_convolutional_coefs`. Each line takes eager's ghost rule (`_lazy_side_rule`),
including the per-line :detect decision (`_lazy_detect`) and the linear matrix on short axes. Only
the lines the stencil needs are filled: in the pass of axis d, the lower axes range over the
stencil, the higher axes over the patch inside the domain.

The patch is a scratch vector in task-local storage, one per task and element type, grown on first
use: tasks never share it, so a lazy interpolant can be evaluated from several threads at once.
"""
function _lazy_patch_sum(itp::FastConvolutionInterpolation{T,N}, pos::NTuple{N,Int}, w) where {T,N}
    n = itp.domain_size                                   # data points per axis
    eqs = itp.eqs                                         # stencil half-width per axis
    ksym = _kernel_sym(itp.kernel_sym)                    # kernel per axis
    s_lo = ntuple(d -> pos[d] - eqs[d] + 1, Val(N))       # first stencil index per axis
    s_hi = ntuple(d -> pos[d] + eqs[d], Val(N))           # last stencil index per axis
    # eager's ghost rule of both boundaries of every axis (a side the stencil doesn't reach is never used)
    rule_left  = ntuple(d -> _lazy_side_rule(itp.bc[d][1], ksym[d], n[d], eqs[d]), Val(N))
    rule_right = ntuple(d -> _lazy_side_rule(itp.bc[d][2], ksym[d], n[d], eqs[d]), Val(N))

    # patch box per axis: the stencil, widened to the interior values the ghost rules read
    lo = ntuple(d -> s_hi[d] > n[d] ? min(s_lo[d], n[d] - rule_right[d][3] + 1) : s_lo[d], Val(N))
    hi = ntuple(d -> s_lo[d] < 1 ? max(s_hi[d], rule_left[d][3]) : s_hi[d], Val(N))
    len = ntuple(d -> hi[d] - lo[d] + 1, Val(N))          # patch length per axis
    stride = ntuple(d -> prod(e -> len[e], 1:d-1; init=1), Val(N))   # column-major strides
    # the patch, stored column-major, and the :detect slice: scratch vectors of this task
    buf = get!(() -> T[], task_local_storage(), (:ConvolutionInterpolations_patch, T))::Vector{T}
    length(buf) < prod(len) && resize!(buf, prod(len))    # grown on first use only
    slice = get!(() -> T[], task_local_storage(), (:ConvolutionInterpolations_slice, T))::Vector{T}
    lin(I) = 1 + sum(ntuple(d -> (I[d] - lo[d]) * stride[d], Val(N)))   # patch position of data index I

    # data values: every position of the patch box inside the domain
    @inbounds for I in CartesianIndices(ntuple(d -> max(lo[d], 1):min(hi[d], n[d]), Val(N)))
        buf[lin(Tuple(I))] = itp.coefs[I]
    end

    # ghosts, axis by axis as in eager: pass d fills the stencil positions outside the domain along d
    for d in 1:N, side in 1:2
        left = side == 1                                  # left or right boundary of axis d
        (left ? s_lo[d] < 1 : s_hi[d] > n[d]) || continue # the stencil doesn't reach this side
        matrix, detect, reach = left ? rule_left[d] : rule_right[d]   # the side's ghost rule
        linear = get_polynomial_ghost_coeffs(:linear, ksym[d])   # :detect's rejection matrix
        ghosts = left ? (s_lo[d]:0) : ((n[d] + 1):s_hi[d])   # ghost positions along d in the stencil
        sd = stride[d]                                    # patch step along d
        # lines along d: lower axes over the stencil, higher axes over the patch inside the domain
        lines = CartesianIndices(ntuple(e -> e < d ? (s_lo[e]:s_hi[e]) :
                                             e == d ? (lo[e]:lo[e]) :
                                             (max(lo[e], 1):min(hi[e], n[e])), Val(N)))
        for J in lines
            base = lin(Tuple(J))                          # patch position of the line at lo[d]
            # interior value p of this line (p = 1 at the boundary, increasing inward)
            inward(p) = @inbounds buf[base + ((left ? p : n[d] - p + 1) - lo[d]) * sd]
            g = if detect                                 # :detect: eager's per-line decision
                _lazy_detect(slice, inward, reach, left, matrix, eqs[d]) ? matrix : linear
            else
                matrix
            end
            ns = size(g, 2)                               # interior values per ghost
            for gpos in ghosts
                row = left ? 1 - gpos : gpos - n[d]       # ghost number: 1 next to the boundary
                value = _compensated_row_dot(g, row, inward, ns, T)
                @inbounds buf[base + (gpos - lo[d]) * sd] = value
            end
        end
    end

    # the stencil sum over the patch
    result = zero(T)
    @inbounds for K in CartesianIndices(ntuple(d -> 1:2eqs[d], Val(N)))
        weight = prod(ntuple(d -> w[d][K[d]], Val(N)))    # product of the per-axis weights
        result += buf[lin(ntuple(d -> pos[d] + K[d] - eqs[d], Val(N)))] * weight
    end
    return result
end

@inline function (itp::FastConvolutionInterpolation{T,N,0,TCoefs,Axs,KA,HigherDimension{N},
                    LowerOrderKernel{DG},EQ,KBC,DerivativeOrder{DO},FD,SD,Val{SG},
                    Val{true},Val{0}})(x::Vararg{Number,N}) where {T<:AbstractFloat,N,TCoefs<:AbstractArray{T,N},
                    KA<:NTuple{N,<:Nothing},Axs<:NTuple{N,<:AbstractVector},DG,
                    EQ<:NTuple{N,Int},KBC<:NTuple{N,Tuple{Symbol,Symbol}},
                    DO,FD,SD,SG}

    x = T.(x)
    # same as eager path
    if DG[1] == :a0
        return _eval_a0_nd(itp, x)
    elseif DG[1] == :a1
        return _eval_a1_nd(itp, x, Val{DO}())
    end
end

@inline function (itp::FastConvolutionInterpolation{T,N,0,TCoefs,Axs,KA,HigherDimension{N},
                    DG,EQ,KBC,DerivativeOrder{DO},FD,SD,Val{SG},
                    Val{true},Val{0}})(x::Vararg{Number,N}) where {T<:AbstractFloat,N,TCoefs<:AbstractArray{T,N},
                    KA<:NTuple{N,<:Nothing},Axs<:NTuple{N,<:AbstractVector},DG,
                    EQ<:NTuple{N,Int},
                    KBC<:NTuple{N,Tuple{Symbol,Symbol}},DO,FD,SD,SG}

    # specialized dispatch for N-dimensional higher-order kernel
    x = T.(x)

    # Compute i_float once per dimension
    i_floats = ntuple(d -> (x[d] - itp.knots[d][1]) / itp.h[d] + one(T), N)

    # Find knot indices for each dimension
    pos_ids = ntuple(d -> clamp(floor(Int, i_floats[d]), 1, length(itp.knots[d]) - 1), N)

    is_boundary = ntuple(d -> is_boundary_stencil(pos_ids[d], size(itp.coefs, d), itp.eqs[d]), N)

    if any(is_boundary) && itp.boundary_fallback

        # Compute normalized left distances - recompute from actual knot positions
        weights = ntuple(d -> (x[d] - itp.knots[d][pos_ids[d]]) / itp.h[d], N)

        # Build up: for each "slice" in remaining dimensions,
        # accumulate the interpolated result
        result = zero(T)

        # Iterate over all 2^(N-1) combinations of the last N-1 dimensions
        @inbounds for corner in 0:(2^(N-1) - 1)
            # Build indices for dimensions 2:N
            tail_indices = ntuple(d -> (corner >> (d-1)) & 1 == 0 ? pos_ids[d+1] : pos_ids[d+1]+1, N-1)

            # Get the two values along dimension 1
            idx0 = (pos_ids[1], tail_indices...)
            idx1 = (pos_ids[1]+1, tail_indices...)

            # Interpolate along dimension 1
            interp_val = (one(T) - weights[1]) * itp.coefs[idx0...] + weights[1] * itp.coefs[idx1...]

            # Weight by all the other dimensions
            tail_weight = prod(ntuple(d -> (corner >> (d-1)) & 1 == 0 ? (one(T) - weights[d+1]) : weights[d+1], N-1))

            result += tail_weight * interp_val
        end

        return @inbounds @fastmath result * prod((-one(T)/itp.h[d])^DO[d] for d in 1:N)

    else

        # Normalized positions within the cell, τ = diff_right per dimension
        diff_right = ntuple(d -> one(T) - (x[d] - itp.knots[d][pos_ids[d]]) / itp.h[d], N)

        # Kernel weights per dimension, each with its own kernel and derivative order
        w = _column_weights_per_dim(Val(_kernel_sym(itp.kernel_sym)), Val(DO), diff_right)

        result = if any(is_boundary)
            # the stencil reaches past the domain: ghosts formed on a local patch, as eager forms them
            _lazy_patch_sum(itp, pos_ids, w)
        else
            # the whole stencil lies inside the domain: the stencil sum over the data
            acc = zero(T)
            @inbounds for offsets in Iterators.product(ntuple(d -> -(itp.eqs[d]-1):itp.eqs[d], N)...)
                idxs = ntuple(d -> pos_ids[d] + offsets[d], N)
                coef = itp.coefs[idxs...]
                kernel_val = prod(ntuple(d -> w[d][offsets[d] + itp.eqs[d]], Val(N)))
                acc += coef * kernel_val
            end
            acc
        end

        return @inbounds @fastmath result * prod((-one(T)/itp.h[d])^DO[d] for d in 1:N)
    end
end