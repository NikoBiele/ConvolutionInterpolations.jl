"""
    FastConvolutionInterpolation(knots, vs::AbstractArray{T,N}; kwargs...) where {T,N}

Construct a fast convolution interpolation object with precomputed kernel tables for O(1) evaluation.
This is the lower-level fast constructor — most users should prefer `convolution_interpolation`,
which wraps this with extrapolation handling and automatic mode selection.

Only supports uniform grids. For nonuniform grids, use `ConvolutionInterpolation` directly.

# Arguments
- `knots`: Vector or range (1D) or tuple of vectors/ranges (N-D) of grid coordinates.
- `vs`: Array of values at the grid points.

# Keyword Arguments
- `kernel::Symbol=:b5`: Convolution kernel to use.
  - `a`-series: `:a0` (nearest), `:a1` (linear), `:a3` (cubic), `:a4` (quartic), `:a5` (quintic), `:a7` (septic)
  - `b`-series: `:b5`, `:b7`, `:b9`, `:b11`, `:b13`
- `precompute::Int=101`: Resolution of the precomputed kernel table. The default uses
  pre-shipped tables (zero computation). Automatically raised to at least 10,000 for
  `:linear` subgrid mode.
- `B=nothing`: If provided, uses Gaussian kernel with parameter `B` for C∞ smoothness.
  Forces eager mode.
- `bc=:auto`: Boundary condition for kernel evaluation at domain edges.
  Options: `:auto`, `:poly`, `:linear`, `:quadratic`, `:periodic`.
- `derivative::Int=0`: Derivative order to evaluate. Supported up to 6 for `b`-series
  kernels. For `a`-series kernels the top derivative automatically uses linear interpolation
  to match the kernel's continuity class.
- `subgrid::Symbol=:cubic`: Subgrid interpolation mode for the precomputed kernel table.
  Options: `:linear` (fastest, needs high `precompute`), `:cubic` (default), `:quintic`.
  Automatically downgraded to `:linear` for 3D+ and when evaluating the top derivative
  of a kernel. Validated against the kernel's continuity class.
- `lazy::Bool=false`: When `true`, skip ghost point expansion at construction time.
  The raw values are stored directly and ghost points are computed on the fly during
  evaluation near boundaries. Interior evaluation has zero overhead compared to eager mode.
  Automatically disabled for `:a0`, `:a1`, and Gaussian kernels.
- `boundary_fallback::Bool=false`: When `true`, near-boundary evaluations are linearly
  extrapolated from interior points rather than computing full ghost point stencils.
  Prevents expensive tensor-product ghost computation in high dimensions.
  Only active when `lazy=true`.

# Returns
A `FastConvolutionInterpolation` object callable at arbitrary points within the grid domain.
Does not handle extrapolation — use `convolution_interpolation` or wrap in
`ConvolutionExtrapolation` for that.

See also: [`convolution_interpolation`](@ref), [`ConvolutionInterpolation`](@ref), [`ConvolutionExtrapolation`](@ref).
"""

function FastConvolutionInterpolation(knots::Union{AbstractVector,NTuple{N,AbstractVector},AbstractRange,NTuple{N,AbstractRange}},
                                      vs::AbstractArray{T,N};
                                      kernel::Union{Symbol,NTuple{N,Symbol}}=:b5, precompute::Int=101, B=nothing,
                                      bc::Union{Symbol,Vector{Tuple{Symbol,Symbol}},NTuple{N,Tuple{Symbol,Symbol}}}=:auto,
                                      derivative::Union{Int,NTuple{N,Int}}=0,
                                      subgrid::Union{Symbol,NTuple{N,Symbol}}=:cubic,
                                      lazy::Bool=false, boundary_fallback::Bool=false) where {T,N}

    if kernel == :n3
        error("The :n3 kernel is not supported by FastConvolutionInterpolation. Use ConvolutionInterpolation instead.")
    end

    # convert inputs to expected types (if called directly)
    knots_tuple = knots isa NTuple{N,AbstractVector} ? knots : knots isa AbstractVector || knots isa AbstractRange ? (knots,) : knots
    kernels_tuple = kernel isa NTuple{N,Symbol} ? kernel : kernel isa Symbol ? ntuple(_ -> kernel,     N) : kernel
    derivatives = derivative isa NTuple{N,Int} ? derivative : derivative isa Int ? ntuple(_ -> derivative, N) : derivative

    # Normalize bc to per-dimension, per-side tuples (for direct calls)
    bc = if bc isa Symbol
        ntuple(_ -> (bc, bc), N)
    elseif bc isa Vector{Tuple{Symbol,Symbol}}
        ntuple(d -> bc[d], N)
    elseif bc isa NTuple{N,Tuple{Symbol,Symbol}}
        bc  # already a tuple of tuples
    else
        error("Unsupported bc type: $(typeof(bc)), must be Symbol, Vector{Tuple{Symbol,Symbol}}, or NTuple{N,Tuple{Symbol,Symbol}}.")
    end

    eqs = ntuple(d -> get_equations_for_degree(kernels_tuple[d]), N)

    n_integral = count(d -> derivatives[d] == -1, 1:N)

    # Normalise subgrid to per-dim tuple.
    subgrids = if subgrid isa Symbol
        ntuple(d -> subgrid, N)
    else
        ntuple(d -> subgrid[d], N)
    end

    # Validate and possibly downgrade per dimension
    subgrids = ntuple(N) do d
        if kernels_tuple[d] == :a0 && derivatives[d] >= 0
            :linear
        elseif kernels_tuple[d] == :a0 && derivatives[d] == -1
            :linear
        elseif kernels_tuple[d] == :a1 && derivatives[d] >= 0 && !(subgrids[d] == :linear)
            :linear
        elseif kernels_tuple[d] == :a1 && derivatives[d] == -1 && !(subgrids[d] in (:linear, :cubic))
            :cubic # default
        elseif any(d -> derivatives[d] == -1, 1:N) && any(d -> derivatives[d] >= 0, 1:N)
            :linear # mixed integral and derivative
        elseif N < 3
            validate_subgrid_compatibility(kernels_tuple[d], derivatives[d], subgrids[d])
        elseif N >= 3
            if n_integral == 3
                validate_subgrid_compatibility(kernels_tuple[d], derivatives[d], subgrids[d])
            else
                :linear
            end
        end
    end

    h   = ntuple(d -> knots_tuple[d][2] - knots_tuple[d][1], N)

    # precompute resolution
    precompute_actual = ntuple(d -> (subgrids[d] == :linear ? max(precompute, 10_000) : precompute), N)

    # ===== LAZY vs EAGER coefs/knots =====
    high_order_kernel_used = any(d -> kernels_tuple[d] in (:a3, :a4, :a5, :a7, :b5, :b7, :b9, :b11, :b13), 1:N)
    if (lazy || !high_order_kernel_used) && B === nothing
        # LAZY: store raw values directly, original knots (not expanded)
        coefs = vs
        knots_new = ntuple(i -> collect(eltype(h), knots_tuple[i]), N)
    else
        # EAGER: original path — expand knots and compute ghost points
        lazy = false  # ensure flag matches actual state
        knots_new = expand_knots(knots_tuple, ntuple(d -> eqs[d] - 1, N))
        coefs = create_convolutional_coefs(vs, h, eqs, bc, kernels_tuple)
    end

    # Per-dim kernel tables — one call per dim, unpack into 4 NTuples
    tables        = ntuple(d -> get_precomputed_kernel_and_range(kernels_tuple[d];
                                precompute=precompute_actual[d], float_type=T,
                                derivative=derivatives[d], subgrid=subgrids[d]), N)
    pre_range_d   = ntuple(d -> tables[d][1], N)
    kernel_pre_d  = ntuple(d -> tables[d][2], N)
    kd1_pre_d     = ntuple(d -> tables[d][3], N)
    kd2_pre_d     = ntuple(d -> tables[d][4], N)

    # anchor: knots_new[d][eqs[d]] for integral dims, zero for derivative dims
    anchor = ntuple(d -> derivatives[d] == -1 ? knots_new[d][eqs[d]] : zero(T), N)

    # left_values: computed for integral dims, placeholder for derivative dims
    int_dims = findall(d -> derivatives[d] == -1, 1:N)
    placeholder = Array{T,N}(undef, ntuple(_ -> 0, N)...)
    if n_integral >= 1
        left_values = ntuple(N) do d
            if derivatives[d] == -1
                _compute_left_values(T, eqs[d], size(coefs, d),
                    tables[d][2], tables[d][3], tables[d][4], Val(:linear))
            else
                [zero(T)]
            end
        end
        tail1_left = ntuple(N) do d
            if derivatives[d] == -1
                lv = reshape(left_values[d], ntuple(i -> i == d ? length(left_values[d]) : 1, N))
                cumsum(coefs .* (T(1//2) .- lv), dims=d)
            else
                placeholder
            end
        end
        tail1_right = ntuple(N) do d
            if derivatives[d] == -1
                lv = reshape(left_values[d], ntuple(i -> i == d ? length(left_values[d]) : 1, N))
                _suffix_sum(coefs .* (-T(1//2) .- lv), d)
            else
                placeholder
            end
        end
    else
        left_values = ntuple(_ -> [zero(T)], N)
        tail1_left = ntuple(_ -> placeholder, N)
        tail1_right = ntuple(_ -> placeholder, N)
    end    
    if n_integral >= 2
        lv1 = reshape(left_values[int_dims[1]], ntuple(i -> i == int_dims[1] ? size(coefs, int_dims[1]) : 1, N))
        lv2 = reshape(left_values[int_dims[2]], ntuple(i -> i == int_dims[2] ? size(coefs, int_dims[2]) : 1, N))
        wll = coefs .* (T(1//2) .- lv1) .* (T(1//2) .- lv2)
        wrl = coefs .* (-T(1//2) .- lv1) .* (T(1//2) .- lv2)
        wlr = coefs .* (T(1//2) .- lv1) .* (-T(1//2) .- lv2)
        wrr = coefs .* (-T(1//2) .- lv1) .* (-T(1//2) .- lv2)
        tail2_ll = cumsum(cumsum(wll, dims=int_dims[1]), dims=int_dims[2])
        tail2_rl = _suffix_sum(cumsum(wrl, dims=int_dims[2]), int_dims[1])
        tail2_lr = cumsum(_suffix_sum(wlr, int_dims[2]), dims=int_dims[1])
        tail2_rr = _suffix_sum(_suffix_sum(wrr, int_dims[2]), int_dims[1])
    else
        ph = Array{T,N}(undef, ntuple(_ -> 0, N)...)
        tail2_ll = ph
        tail2_rl = ph
        tail2_lr = ph
        tail2_rr = ph
    end
    if n_integral == 3
        lv1 = reshape(left_values[int_dims[1]], ntuple(i -> i == int_dims[1] ? size(coefs, int_dims[1]) : 1, N))
        lv2 = reshape(left_values[int_dims[2]], ntuple(i -> i == int_dims[2] ? size(coefs, int_dims[2]) : 1, N))
        lv3 = reshape(left_values[int_dims[3]], ntuple(i -> i == int_dims[3] ? size(coefs, int_dims[3]) : 1, N))
        wl1 = T(1//2) .- lv1;  wr1 = -T(1//2) .- lv1
        wl2 = T(1//2) .- lv2;  wr2 = -T(1//2) .- lv2
        wl3 = T(1//2) .- lv3;  wr3 = -T(1//2) .- lv3

        # faces: saturated in one dim, free in the other two
        # tail3_face_l[d] = prefix sum along dim d only, weighted by (1/2 - lv_d)
        tail3_face_l = (
            cumsum(coefs .* wl1, dims=int_dims[1]),
            cumsum(coefs .* wl2, dims=int_dims[2]),
            cumsum(coefs .* wl3, dims=int_dims[3]),
        )
        tail3_face_r = (
            _suffix_sum(coefs .* wr1, int_dims[1]),
            _suffix_sum(coefs .* wr2, int_dims[2]),
            _suffix_sum(coefs .* wr3, int_dims[3]),
        )

        # edges: saturated in two dims, free in one
        # tail3_edge_ll[d] = free in dim d, left×left in the other two (ascending order)
        # d=1: saturated in dims 2&3
        # d=2: saturated in dims 1&3
        # d=3: saturated in dims 1&2
        tail3_edge_ll = (
            cumsum(cumsum(coefs .* wl2 .* wl3, dims=int_dims[2]), dims=int_dims[3]),
            cumsum(cumsum(coefs .* wl1 .* wl3, dims=int_dims[1]), dims=int_dims[3]),
            cumsum(cumsum(coefs .* wl1 .* wl2, dims=int_dims[1]), dims=int_dims[2]),
        )
        tail3_edge_rl = (
            _suffix_sum(cumsum(coefs .* wr2 .* wl3, dims=int_dims[3]), int_dims[2]),
            _suffix_sum(cumsum(coefs .* wr1 .* wl3, dims=int_dims[3]), int_dims[1]),
            _suffix_sum(cumsum(coefs .* wr1 .* wl2, dims=int_dims[2]), int_dims[1]),
        )
        tail3_edge_lr = (
            cumsum(_suffix_sum(coefs .* wl2 .* wr3, int_dims[3]), dims=int_dims[2]),
            cumsum(_suffix_sum(coefs .* wl1 .* wr3, int_dims[3]), dims=int_dims[1]),
            cumsum(_suffix_sum(coefs .* wl1 .* wr2, int_dims[2]), dims=int_dims[1]),
        )
        tail3_edge_rr = (
            _suffix_sum(_suffix_sum(coefs .* wr2 .* wr3, int_dims[3]), int_dims[2]),
            _suffix_sum(_suffix_sum(coefs .* wr1 .* wr3, int_dims[3]), int_dims[1]),
            _suffix_sum(_suffix_sum(coefs .* wr1 .* wr2, int_dims[2]), int_dims[1]),
        )

        # corners: saturated in all three dims
        tail3_corner_lll = cumsum(cumsum(cumsum(coefs .* wl1 .* wl2 .* wl3, dims=int_dims[1]), dims=int_dims[2]), dims=int_dims[3])
        tail3_corner_rll = _suffix_sum(cumsum(cumsum(coefs .* wr1 .* wl2 .* wl3, dims=int_dims[2]), dims=int_dims[3]), int_dims[1])
        tail3_corner_lrl = cumsum(_suffix_sum(cumsum(coefs .* wl1 .* wr2 .* wl3, dims=int_dims[3]), int_dims[2]), dims=int_dims[1])
        tail3_corner_llr = cumsum(cumsum(_suffix_sum(coefs .* wl1 .* wl2 .* wr3, int_dims[3]), dims=int_dims[1]), dims=int_dims[2])
        tail3_corner_rrl = _suffix_sum(_suffix_sum(cumsum(coefs .* wr1 .* wr2 .* wl3, dims=int_dims[3]), int_dims[2]), int_dims[1])
        tail3_corner_rlr = _suffix_sum(cumsum(_suffix_sum(coefs .* wr1 .* wl2 .* wr3, int_dims[2]), dims=int_dims[1]), int_dims[3])
        tail3_corner_lrr = cumsum(_suffix_sum(_suffix_sum(coefs .* wl1 .* wr2 .* wr3, int_dims[3]), int_dims[2]), dims=int_dims[1])
        tail3_corner_rrr = _suffix_sum(_suffix_sum(_suffix_sum(coefs .* wr1 .* wr2 .* wr3, int_dims[3]), int_dims[2]), int_dims[1])
    else
        ph = Array{T,N}(undef, ntuple(_ -> 0, N)...)
        tail3_face_l = ntuple(_ -> ph, 3)
        tail3_face_r = ntuple(_ -> ph, 3)
        tail3_edge_ll = ntuple(_ -> ph, 3)
        tail3_edge_rl = ntuple(_ -> ph, 3)
        tail3_edge_lr = ntuple(_ -> ph, 3)
        tail3_edge_rr = ntuple(_ -> ph, 3)
        tail3_corner_lll = ph; tail3_corner_rll = ph
        tail3_corner_lrl = ph; tail3_corner_llr = ph
        tail3_corner_rrl = ph; tail3_corner_rlr = ph
        tail3_corner_lrr = ph; tail3_corner_rrr = ph
    end

    kernel_type = ntuple(d -> nothing, N)
    dimension = N <= 3 ? Val(N) : HigherDimension(Val(N))
    integral_dimension = n_integral <= 3 ? Val(n_integral) : HigherDimension(Val(n_integral))

    do_type = if n_integral == 0
        DerivativeOrder(Val(derivatives))
    elseif n_integral == N
        IntegralOrder()
    else # if n_integral <= 3 && n_integral < N
        FastMixedIntegralOrder{derivatives}()
    end

    low_order_kernel_used  = any(d -> kernels_tuple[d] in (:a0, :a1), 1:N)
    derivatives_equal = allequal(derivatives)
    kernels_equal = allequal(kernels_tuple)
    if derivatives_equal
        kernels = if high_order_kernel_used && low_order_kernel_used
            FullMixedOrderKernel(Val(kernels_tuple))
        elseif high_order_kernel_used && !low_order_kernel_used && kernels_equal
            HigherOrderKernel(Val(kernels_tuple))
        elseif high_order_kernel_used && !low_order_kernel_used && !kernels_equal
            HigherOrderMixedKernel(Val(kernels_tuple))
        elseif !high_order_kernel_used && low_order_kernel_used && kernels_equal
            LowerOrderKernel(Val(kernels_tuple))
        elseif !high_order_kernel_used && low_order_kernel_used && !kernels_equal
            LowerOrderMixedKernel(Val(kernels_tuple))
        end
    else
        kernels = if high_order_kernel_used && low_order_kernel_used
            FullMixedOrderKernel(Val(kernels_tuple))
        elseif high_order_kernel_used && !low_order_kernel_used
            HigherOrderMixedKernel(Val(kernels_tuple))
        elseif !high_order_kernel_used && low_order_kernel_used
            LowerOrderMixedKernel(Val(kernels_tuple))
        end
    end

    return FastConvolutionInterpolation{T,N,n_integral,typeof(coefs),typeof(knots_new),
                                        typeof(kernel_type),typeof(dimension),typeof(kernels),
                                        typeof(eqs),typeof(pre_range_d),typeof(kernel_pre_d),
                                        typeof(bc),typeof(do_type),
                                        typeof(kd1_pre_d),typeof(kd2_pre_d),typeof(Val(subgrids)),
                                        typeof(Val(lazy)),typeof(integral_dimension)}(
        coefs, knots_new, h, kernel_type, dimension, kernels, eqs,
        pre_range_d, kernel_pre_d, bc, do_type,
        kd1_pre_d, kd2_pre_d, Val(subgrids),
        Val(lazy), boundary_fallback, left_values, anchor, integral_dimension,
        tail1_left, tail1_right, tail2_ll, tail2_rl, tail2_lr, tail2_rr,
        # 3d arrrays
        tail3_edge_ll,  # free dim d, left×left in the other two
        tail3_edge_rl, tail3_edge_lr, tail3_edge_rr,
        tail3_face_l,   # tail3_face_l[d] = left-saturated in dim d
        tail3_face_r,
        tail3_corner_lll, tail3_corner_rll, tail3_corner_lrl, tail3_corner_llr,
        tail3_corner_rrl, tail3_corner_rlr, tail3_corner_lrr, tail3_corner_rrr,        
    )
end

function validate_subgrid_compatibility(kernel::Symbol, derivative::Int, subgrid::Symbol)
    if kernel == :a0 && subgrid == :linear
        return subgrid
    elseif kernel == :a0 && subgrid == :cubic
        return :linear
    elseif kernel == :a1 && derivative == -1 && (subgrid == :linear || subgrid == :cubic)
        return subgrid
    elseif kernel == :a1 && derivative == 0 && subgrid == :cubic
        return :linear
    elseif kernel == :a1 && subgrid == :quintic
        error("Cannot compute derivative $(derivative) for kernel $kernel with subgrid $subgrid." * 
                "Derivative is not available for this kernel and subgrid.")
    elseif kernel == :a1 && subgrid == :cubic && derivative > -1
        error("Cannot compute derivative $(derivative) for kernel $kernel with subgrid $subgrid." * 
                "Derivative is not available for this kernel and subgrid.")
    end 
    max_deriv = max_smooth_derivative[kernel]  # e.g., 3 for b5
    available_derivs = max_deriv - derivative

    if available_derivs == 0 && subgrid != :linear
        subgrid = :linear # force linear subgrid for top derivatives
    elseif available_derivs < 0
        error("Cannot compute derivative $(derivative) for kernel $kernel with subgrid $subgrid." * 
                "Maximum available derivative with this kernel and subgrid: $(available_derivs)."*
                "Try a different kernel or subgrid.")
    end
    
    return subgrid
end

function suggest_subgrid(available_derivs)
    available_derivs >= 2 && return :quintic
    available_derivs >= 1 && return :cubic
    return :linear
end

const required_derivs = Dict(
  :linear => 0,
  :cubic => 1,
  :quintic => 2
)

const max_smooth_derivative = Dict(
  :a0 => -1,
  :a1 => 0,
  :a3 => 1,
  :a4 => 1,
  :a5 => 1,
  :a7 => 1,
  :b5 => 3,
  :b7 => 4,
  :b9 => 5,
  :b11 => 6,
  :b13 => 6
)

function _compute_left_values(T, eqs, n_coefs_d, kernel_pre, kernel_d1_pre, kernel_d2_pre, sg)
    n_pre_loc = size(kernel_pre, 1)
    h_pre_loc = one(T) / T(n_pre_loc - 1)
    lv = zeros(T, n_coefs_d)
    @inbounds for j in 1:n_coefs_d
        sj = T(eqs - j)
        s_abs = abs(sj)
        if s_abs >= T(eqs)
            lv[j] = T(1//2) * T(sign(sj))
        else
            col_float = T(eqs) + sj
            col = clamp(floor(Int, col_float) + 1, 1, 2 * eqs)
            x_diff_right = col_float - T(col - 1)
            continuous_idx = x_diff_right * T(n_pre_loc - 1) + one(T)
            idx      = clamp(floor(Int, continuous_idx), 1, n_pre_loc - 1)
            idx_next = idx + 1
            t        = continuous_idx - T(idx)
            lv[j] = if sg == Val(:quintic)
                quintic_hermite(t,
                    kernel_pre[idx,    col], kernel_pre[idx_next,    col],
                    kernel_d1_pre[idx, col], kernel_d1_pre[idx_next, col],
                    kernel_d2_pre[idx, col], kernel_d2_pre[idx_next, col],
                    h_pre_loc)
            elseif sg == Val(:cubic)
                cubic_hermite(t,
                    kernel_pre[idx,    col], kernel_pre[idx_next,    col],
                    kernel_d1_pre[idx, col], kernel_d1_pre[idx_next, col],
                    h_pre_loc)
            else
                (one(T) - t) * kernel_pre[idx, col] + t * kernel_pre[idx_next, col]
            end
        end
    end
    lv
end

# Computes the suffix (reverse cumulative) sum of A along dimension `dims`.
# suffix_sum[i] = sum of A[i], A[i+1], ..., A[end] along that dimension.
function _suffix_sum(A::AbstractArray, dims::Int)
    reverse(cumsum(reverse(A, dims=dims), dims=dims), dims=dims)
end