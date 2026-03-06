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
                                      degree::Symbol=:b5, precompute::Int=101, B=nothing,
                                      kernel_bc::Union{Symbol,Vector{Tuple{Symbol,Symbol}},NTuple{N,Tuple{Symbol,Symbol}}}=:auto,
                                      derivative::Int=0,
                                      subgrid::Symbol=:cubic,
                                      lazy::Bool=false, boundary_fallback::Bool=false) where {T,N}

    if degree == :n3
        error("The :n3 kernel is not supported by FastConvolutionInterpolation. Use ConvolutionInterpolation instead.")
    end

    subgrid = validate_subgrid_compatibility(degree::Symbol, derivative::Int, subgrid::Symbol)
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
        coefs = degree == :a0 || degree == :a1 ? vs : create_convolutional_coefs(vs, h, eqs, kernel_bc, degree)
    end

    kernel = B === nothing ? ConvolutionKernel(Val(degree), Val(derivative)) : GaussianConvolutionKernel(Val(B))
    dimension = N <= 3 ? Val(N) : HigherDimension(Val(N))
    pre_range, kernel_pre, kernel_d1_pre, kernel_d2_pre = 
                  get_precomputed_kernel_and_range(degree; precompute=precompute, float_type=T,
                  derivative=derivative, subgrid=subgrid)
    degree = degree == :a0 || degree == :a1 ? Val(degree) : HigherOrderKernel(Val(degree))

    return FastConvolutionInterpolation{T,N,typeof(coefs),typeof(it),typeof(knots_new),typeof(kernel),typeof(dimension),
                                        typeof(degree),typeof(eqs),typeof(pre_range),typeof(kernel_pre),typeof(kernel_bc),
                                        typeof(Val(derivative)), typeof(kernel_d1_pre),typeof(kernel_d2_pre),typeof(Val(subgrid))}(
        coefs, knots_new, it, h, kernel, dimension, degree, eqs, pre_range, kernel_pre, kernel_bc, Val(derivative),
        kernel_d1_pre, kernel_d2_pre, Val(subgrid), lazy, boundary_fallback
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
  :a1 => -1,
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