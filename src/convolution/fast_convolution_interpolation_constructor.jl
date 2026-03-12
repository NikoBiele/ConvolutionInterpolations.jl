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
- `degree::Symbol=:a4`: Convolution kernel to use.
  - `a`-series: `:a0` (nearest), `:a1` (linear), `:a3` (cubic), `:a4` (quartic), `:a5` (quintic), `:a7` (septic)
  - `b`-series: `:b5`, `:b7`, `:b9`, `:b11`, `:b13`
- `precompute::Int=101`: Resolution of the precomputed kernel table. The default uses
  pre-shipped tables (zero computation). Automatically raised to at least 10,000 for
  `:linear` subgrid mode.
- `B=nothing`: If provided, uses Gaussian kernel with parameter `B` for C∞ smoothness.
  Forces eager mode.
- `kernel_bc=:auto`: Boundary condition for kernel evaluation at domain edges.
  Options: `:auto`, `:polynomial`, `:linear`, `:quadratic`, `:periodic`, `:detect`.
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
- `boundary_fallback::Bool=false`: When `true`, throw an error instead of computing ghost
  points when evaluating near boundaries in lazy mode. Prevents expensive tensor-product
  ghost computation in high dimensions. Only active when `lazy=true`.

# Returns
A `FastConvolutionInterpolation` object callable at arbitrary points within the grid domain.
Does not handle extrapolation — use `convolution_interpolation` or wrap in
`ConvolutionExtrapolation` for that.

See also: [`convolution_interpolation`](@ref), [`ConvolutionInterpolation`](@ref), [`ConvolutionExtrapolation`](@ref).
"""

function FastConvolutionInterpolation(knots::Union{AbstractVector,NTuple{N,AbstractVector},AbstractRange,NTuple{N,AbstractRange}},
                                      vs::AbstractArray{T,N};
                                      degree::Union{Symbol,NTuple{N,Symbol}}=:b5, precompute::Int=101, B=nothing,
                                      kernel_bc::Union{Symbol,Vector{Tuple{Symbol,Symbol}},NTuple{N,Tuple{Symbol,Symbol}}}=:auto,
                                      derivative::Union{Int,NTuple{N,Int}}=0,
                                      subgrid::Union{Symbol,NTuple{N,Symbol}}=:cubic,
                                      lazy::Bool=false, boundary_fallback::Bool=false) where {T,N}

    if degree == :n3
        error("The :n3 kernel is not supported by FastConvolutionInterpolation. Use ConvolutionInterpolation instead.")
    end

    is_perdim = degree isa NTuple{N,Symbol} || derivative isa NTuple{N,Int}
 
    if is_perdim
        degrees     = degree     isa Symbol ? ntuple(_ -> degree,     N) : degree
        derivatives = derivative isa Int    ? ntuple(_ -> derivative, N) : derivative
 
        n_integral = count(d -> derivatives[d] == -1, 1:N)
 
        # Normalise subgrid to per-dim tuple.
        # Integral dims are forced to :linear (K̃ uses linear table lookup).
        # Derivative dims use normal subgrid validation.
        subgrids = if subgrid isa Symbol
            ntuple(d -> derivatives[d] == -1 ? :linear : subgrid, N)
        else
            ntuple(d -> derivatives[d] == -1 ? :linear : subgrid[d], N)
        end
 
        # Validate and possibly downgrade per dimension (3D+ forced to linear)
        subgrids = if N >= 3
            ntuple(_ -> :linear, N)
        else
            ntuple(d -> derivatives[d] == -1 ? :linear :
                validate_subgrid_compatibility(degrees[d], derivatives[d], subgrids[d]), N)
        end
 
        eqs = ntuple(d -> get_equations_for_degree(degrees[d]), N)
        h   = map(k -> k[2] - k[1], knots)
        it  = ntuple(_ -> ConvolutionMethod(), N)
 
        # precompute resolution: max needed across dims
        precompute_actual = foldl(max,
            (subgrids[d] == :linear ? max(precompute, 10_000) : precompute for d in 1:N);
            init = precompute)
 
        # Eager only for per-dim fast path
        lazy      = false
        knots_new = expand_knots(knots, ntuple(d -> eqs[d] - 1, N))
        coefs     = create_convolutional_coefs(vs, h, eqs, kernel_bc, degrees, derivatives)
 
        # Per-dim kernel tables — one call per dim, unpack into 4 NTuples
        tables        = ntuple(d -> get_precomputed_kernel_and_range(degrees[d];
                                    precompute=precompute_actual, float_type=T,
                                    derivative=derivatives[d], subgrid=subgrids[d]), N)
        pre_range_d   = ntuple(d -> tables[d][1], N)
        kernel_pre_d  = ntuple(d -> tables[d][2], N)
        kd1_pre_d     = ntuple(d -> tables[d][3], N)
        kd2_pre_d     = ntuple(d -> tables[d][4], N)
 
        # anchor: knots_new[d][eqs[d]] for integral dims, zero for derivative dims
        anchor = ntuple(d -> derivatives[d] == -1 ? knots_new[d][eqs[d]] : zero(T), N)
 
        # left_values: computed for integral dims, placeholder for derivative dims
        left_values = ntuple(N) do d
            if derivatives[d] == -1
                _compute_left_values(T, eqs[d], size(coefs, d),
                    tables[d][2], tables[d][3], tables[d][4], Val(:linear))
            else
                [zero(T)]
            end
        end
 
        kernel    = ntuple(d -> degrees[d] in (:a0, :a1) ?
                                Val(degrees[d]) : HigherOrderKernel(Val(degrees[d])), N)
        dimension = N <= 3 ? Val(N) : HigherDimension(Val(N))
 
        do_type = if n_integral == 0
            DerivativeOrder(Val(derivatives))
        elseif n_integral == N
            IntegralOrder()
        else
            MixedIntegralOrder{derivatives}()
        end
 
        # SG dispatch tag: the subgrid tuple itself, e.g. Val{(:cubic,:quintic)}
        sg_tag = Val(subgrids)
 
        return FastConvolutionInterpolation{T,N,typeof(coefs),typeof(it),typeof(knots_new),
                                            typeof(kernel),typeof(dimension),typeof(Val(degrees)),
                                            typeof(eqs),typeof(pre_range_d),typeof(kernel_pre_d),
                                            typeof(kernel_bc),typeof(do_type),
                                            typeof(kd1_pre_d),typeof(kd2_pre_d),typeof(sg_tag)}(
            coefs, knots_new, it, h, kernel, dimension, Val(degrees), eqs,
            pre_range_d, kernel_pre_d, kernel_bc, do_type,
            kd1_pre_d, kd2_pre_d, sg_tag,
            lazy, boundary_fallback, left_values, anchor
        )
    end

    subgrid = if derivative == -1
        if degree == :a0
            :linear
        else
            subgrid
        end
    else
        validate_subgrid_compatibility(degree::Symbol, derivative::Int, subgrid::Symbol)
    end
    if N >= 3 && subgrid in (:cubic, :quintic)
        subgrid = :linear
    end
    precompute = subgrid == :linear ? max(precompute, 10_000) : precompute # minimum precompute for linear subgrid
    if knots isa AbstractVector || knots isa AbstractRange
        knots = (knots,) # Convert knots to tuple if needed (if called directly)
    end
    eqs = B === nothing ? get_equations_for_degree(degree) : 50
    h = map(k -> k[2] - k[1], knots)
    it = ntuple(_ -> ConvolutionMethod(), N)

    # ===== LAZY vs EAGER coefs/knots =====
    if lazy && degree != :a0 && degree != :a1 && B === nothing
        # LAZY: store raw values directly, original knots (not expanded)
        coefs = vs
        knots_new = ntuple(i -> collect(eltype(h), knots[i]), N)
    else
        # EAGER: original path — expand knots and compute ghost points
        lazy = false  # ensure flag matches actual state
        knots_new = expand_knots(knots, eqs-1)
        coefs = degree == :a0 || degree == :a1 ? vs : create_convolutional_coefs(vs, h, eqs, kernel_bc, degree, derivative)
    end
    kernel = B === nothing ? ConvolutionKernel(Val(degree), Val(derivative)) : GaussianConvolutionKernel(Val(B))
    dimension = N <= 3 ? Val(N) : HigherDimension(Val(N))
    pre_range, kernel_pre, kernel_d1_pre, kernel_d2_pre = 
                  get_precomputed_kernel_and_range(degree; precompute=precompute, float_type=T,
                  derivative=derivative, subgrid=subgrid)
    anchor = ntuple(d -> knots_new[d][eqs], N)
    if derivative == -1
        left_values = ntuple(d -> _compute_left_values(T, eqs, size(coefs, d),
                             kernel_pre, kernel_d1_pre, kernel_d2_pre, subgrid), N)
    else
        left_values = ntuple(d -> [zero(T)], N)
    end

    degree = degree == :a0 || degree == :a1 ? Val(degree) : HigherOrderKernel(Val(degree))
    do_type = derivative == -1 ? IntegralOrder() : DerivativeOrder(Val(derivative))

    return FastConvolutionInterpolation{T,N,typeof(coefs),typeof(it),typeof(knots_new),typeof(kernel),typeof(dimension),
                                        typeof(degree),typeof(eqs),typeof(pre_range),typeof(kernel_pre),typeof(kernel_bc),
                                        typeof(do_type), typeof(kernel_d1_pre),typeof(kernel_d2_pre),typeof(Val(subgrid))}(
        coefs, knots_new, it, h, kernel, dimension, degree, eqs, pre_range, kernel_pre, kernel_bc, do_type,
        kernel_d1_pre, kernel_d2_pre, Val(subgrid), lazy, boundary_fallback, left_values, anchor
    )
end

function validate_subgrid_compatibility(degree::Symbol, derivative::Int, subgrid::Symbol)
    if degree == :a0 || degree == :a1
        return :not_used
    end 
    max_deriv = max_smooth_derivative[degree]  # e.g., 3 for b5
    available_derivs = max_deriv - derivative

    if available_derivs == 0 && subgrid != :linear
        subgrid = :linear # force linear subgrid for top derivatives
    elseif available_derivs < 0
        error("Cannot compute derivative $(derivative) for kernel $degree with subgrid $subgrid." * 
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
  :nearest => 0,
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