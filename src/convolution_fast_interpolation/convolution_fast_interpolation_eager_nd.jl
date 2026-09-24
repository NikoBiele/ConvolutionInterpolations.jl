"""
    (itp::FastConvolutionInterpolation{T,N,...})(x::Vararg{Number,N})

Evaluate N-dimensional (N > 3) fast convolution interpolation at coordinates `x`.

Dispatches on kernel type:

**Specialized kernels**:
- `:a0` — Nearest neighbor, ~20ns (4D). Selects nearest grid point per dimension.
- `:a1` — Multilinear interpolation, ~35ns (4D). Interpolates over 2^N corners
  using bit-manipulation to enumerate combinations efficiently.

**Higher-order kernels**: exact column polynomials. The kernel weights of each dimension are
evaluated once from compile-time polynomial coefficients (see `_column_rows`), and the N-D
kernel is their tensor product `K_N(x₁,...,x_N) = K₁(x₁)·...·K₁(x_N)` over all `(2*eqs)^N`
support points.

O(1) evaluation time with respect to grid size, allocation-free, exact to rounding for every
derivative order. The `(2*eqs)^N` support limits practical use to modest kernel orders in
high dimensions.

Results scaled by `∏ᵢ (-1/hᵢ)^derivative`.

See also: `FastConvolutionInterpolation`, `_column_weights_per_dim`.
"""

@inline function (itp::FastConvolutionInterpolation{T,N,0,TCoefs,Axs,KA,HigherDimension{N},
                    LowerOrderKernel{DG},EQ,KBC,DerivativeOrder{DO},FD,SD,Val{SG},
                    Val{false},Val{0}})(x::Vararg{Number,N}) where {T<:AbstractFloat,N,TCoefs<:AbstractArray{T,N},
                    KA<:NTuple{N,<:Nothing},Axs<:NTuple{N,<:AbstractVector},DG,EQ<:NTuple{N,Int},KBC<:NTuple{N,Tuple{Symbol,Symbol}},
                    DO,FD,SD,SG}

    x = T.(x)
    if DG[1] == :a0
        return _eval_a0_nd(itp, x)
    elseif DG[1] == :a1
        return _eval_a1_nd(itp, x, Val{DO}())
    end
end

@inline function _eval_a0_nd(itp::FastConvolutionInterpolation{T,N}, x::NTuple{N,<:Number}) where {T,N}

    # specialized dispatch for N-dimensional nearest neighbor kernel

    # Compute i_float once per dimension
    i_floats = ntuple(d -> (x[d] - itp.x0[d]) / itp.h[d] + one(T), N)
    
    # Find knot indices for each dimension
    pos_ids = ntuple(d -> clamp(floor(Int, i_floats[d]), itp.eqs[1], length(itp.knots[d]) - itp.eqs[1]), N)
    
    # Compute normalized left distances - recompute from actual knot positions
    diff_left = ntuple(d -> i_floats[d] - T(pos_ids[d]), N)

    # Nearest neighbor: return coefficient at nearest grid point
    nearest_ids = ntuple(d -> diff_left[d] < 0.5 ? pos_ids[d] : pos_ids[d]+1, N)
    return itp.coefs[nearest_ids...]
end

@inline function _eval_a1_nd(itp::FastConvolutionInterpolation{T,N,0}, x::NTuple{N,<:Number}, ::Val{DO}) where {T,N,DO}

    # specialized dispatch for N-dimensional linear kernel
    
    # Compute i_float once per dimension
    i_floats = ntuple(d -> (x[d] - itp.x0[d]) / itp.h[d] + one(T), N)
    
    # Find knot indices for each dimension
    pos_ids = ntuple(d -> clamp(floor(Int, i_floats[d]), itp.eqs[1], length(itp.knots[d]) - itp.eqs[1]), N)
    
    # Compute normalized left distances - recompute from actual knot positions
    weights = ntuple(d -> i_floats[d] - T(pos_ids[d]), N)
    
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
end

function (itp::FastConvolutionInterpolation{T,N,0,TCoefs,Axs,KA,HigherDimension{N},
            HigherOrderKernel{DG},EQ,KBC,DerivativeOrder{DO},FD,SD,Val{SG},
                    Val{false},Val{0}})(x::Vararg{Number,N}) where {T<:AbstractFloat,N,
                    TCoefs<:AbstractArray{T,N},KA<:NTuple{N,<:Nothing},
                    Axs<:NTuple{N,<:AbstractVector},DG,EQ<:NTuple{N,Int},KBC<:NTuple{N,Tuple{Symbol,Symbol}},
                    DO,FD,SD,SG}

    x = T.(x)
    # specialized dispatch for N-dimensional higher-order kernel

    # Compute i_float once per dimension
    i_floats = ntuple(d -> (x[d] - itp.x0[d]) / itp.h[d] + one(T), N)

    # Find knot indices for each dimension
    pos_ids = ntuple(d -> clamp(floor(Int, i_floats[d]), itp.eqs[1], length(itp.knots[d]) - itp.eqs[1]), N)

    # Normalized distances within the cell
    diff_left = ntuple(d -> i_floats[d] - T(pos_ids[d]), N)
    diff_right = ntuple(d -> one(T) - diff_left[d], N)

    # Kernel weights per dimension at τ = diff_right (column offsets[d] + eqs)
    w = _column_weights_per_dim(Val(DG), Val(DO), diff_right)

    result = zero(T)
    @inbounds for offsets in Iterators.product(ntuple(d -> -(itp.eqs[1]-1):itp.eqs[1], N)...) # same kernel in all directions
        coef = itp.coefs[(pos_ids .+ offsets)...]

        kernel_val = one(T)
        @inbounds for d in 1:N
            kernel_val *= w[d][offsets[d] + itp.eqs[1]]
        end

        result += coef * kernel_val
    end

    return @inbounds @fastmath result * prod((-one(T)/itp.h[d])^DO[d] for d in 1:N)
end