"""
    FastConvolutionInterpolation(knots, vs::AbstractArray{T,N};
        degree::Symbol=:b5, precompute::Int=101, B=nothing, kernel_bc=:auto,
        derivative::Int=0, subgrid::Symbol=:cubic) where {T,N}

Construct a fast convolution interpolation object with precomputed kernel values for O(1) evaluation.

# Arguments
- `knots`: Vector, range, or tuple of vectors/ranges containing the grid points in each dimension
- `vs::AbstractArray{T,N}`: Values to interpolate at the grid points

# Keyword Arguments
- `degree::Symbol=:b5`: Convolution kernel to use. Available kernels:
  - `a`-series: `:a0` (nearest), `:a1` (linear), `:a3` (cubic), `:a5` (quintic), `:a7` (septic)
  - `b`-series (recommended): `:a4`, `:b5`, `:b7`, `:b9`, `:b11`, `:b13`
  - Default `:b5` provides quintic reproduction with 7th-order convergence
- `precompute::Int=101`: Resolution of the precomputed kernel table. The default (101) uses
  pre-shipped tables requiring zero computation. Higher values (e.g. 10_000) trigger on-demand
  computation and disk caching, useful for the `:linear` subgrid mode
- `B=nothing`: If provided, uses Gaussian kernel with parameter `B` instead of polynomial kernel
- `kernel_bc=:auto`: Boundary condition for kernel evaluation at domain boundaries.
  Options: `:auto`, `:polynomial`, `:linear`, `:quadratic`, `:periodic`, `:detect`
- `derivative::Int=0`: Order of derivative to evaluate (0 for interpolation, up to 6 for b-series)
- `subgrid::Symbol=:cubic`: Subgrid interpolation mode for precomputed kernel tables.
  Options: `:linear` (fastest, needs high precompute), `:cubic` (default), `:quintic` (most accurate).
  Availability depends on remaining smooth derivatives: `max_smooth_derivative[kernel] - derivative`

# Returns
`FastConvolutionInterpolation` object with O(1) evaluation time, independent of grid size.

# Performance
- Construction: ~5μs for 1D with 100 grid points (b5 kernel), using pre-shipped kernel tables
- Evaluation: allocation-free, O(1). Typical 1D timings: `:a0`/`:a1` ~6ns, `:b5` ~20ns
- Performance scales with dimensionality but not with grid size

# Details
The constructor expands the grid at boundaries, computes ghost point coefficients using
polynomial boundary conditions, and loads precomputed kernel tables. For the default
`precompute=101`, kernel tables are shipped with the package and loaded as constants
(no disk I/O). For higher resolutions or BigFloat precision, tables are computed on first
use and cached to disk via Scratch.jl.

The subgrid mode controls how kernel values between precomputed points are interpolated:
`:cubic` uses Hermite interpolation with analytically predifferentiated kernels (default),
`:quintic` adds second derivatives for higher accuracy, and `:linear` uses simple linear
interpolation requiring higher table resolution for equivalent accuracy.

# Examples
```julia
# 1D fast interpolation with quintic kernel
knots = (range(0, 1, 100),)
data = sin.(2π .* knots[1])
itp = FastConvolutionInterpolation(knots, data)
itp(0.5)  # O(1) evaluation at x=0.5

# 2D fast interpolation
knots = (range(0, 1, 50), range(0, 1, 50))
data = [sin(x) * cos(y) for x in knots[1], y in knots[2]]
itp = FastConvolutionInterpolation(knots, data, degree=:b7)
itp(0.3, 0.7)

# First derivative evaluation
itp = FastConvolutionInterpolation((range(0, 2π, 100),), sin.(range(0, 2π, 100)), derivative=1)
```

See also: `convolution_interpolation` for the recommended high-level interface with extrapolation.
"""

function FastConvolutionInterpolation(knots::Union{AbstractVector,NTuple{N,AbstractVector},AbstractRange,NTuple{N,AbstractRange}},
                                      vs::AbstractArray{T,N};
                                      degree::Symbol=:b5, precompute::Int=101, B=nothing,
                                      kernel_bc=:auto,
                                      derivative::Int=0,
                                      subgrid::Symbol=:cubic) where {T,N}

    subgrid = validate_subgrid_compatibility(degree::Symbol, derivative::Int, subgrid::Symbol)
    if knots isa AbstractVector || knots isa AbstractRange
        knots = (knots,) # Convert knots to tuple if needed (if called directly)
    end
    eqs = B === nothing ? get_equations_for_degree(degree) : 50
    h = map(k -> k[2] - k[1], knots)
    it = ntuple(_ -> ConvolutionMethod(), N)
    knots_new = expand_knots(knots, eqs-1) # expand boundaries
    coefs = degree == :a0 || degree == :a1 ? vs : create_convolutional_coefs(vs, h, eqs, kernel_bc, degree) # create boundaries
    kernel = B === nothing ? ConvolutionKernel(Val(degree), Val(derivative)) : GaussianConvolutionKernel(Val(B))
    dimension = N <= 3 ? Val(N) : HigherDimension(Val(N))
    if N >= 3 && subgrid in (:cubic, :quintic)
        subgrid = :linear
        precompute = max(precompute, 10_000)
    end
    pre_range, kernel_pre, kernel_d1_pre, kernel_d2_pre = 
                  get_precomputed_kernel_and_range(degree; precompute=precompute, float_type=T,
                  derivative=derivative, subgrid=subgrid)
    degree = degree == :a0 || degree == :a1 ? Val(degree) : HigherOrderKernel(Val(degree))

    return FastConvolutionInterpolation{T,N,typeof(coefs),typeof(it),typeof(knots_new),typeof(kernel),typeof(dimension),
                                        typeof(degree),typeof(eqs),typeof(pre_range),typeof(kernel_pre),typeof(kernel_bc),
                                        typeof(Val(derivative)), typeof(kernel_d1_pre),typeof(kernel_d2_pre),typeof(Val(subgrid))}(
        coefs, knots_new, it, h, kernel, dimension, degree, eqs, pre_range, kernel_pre, kernel_bc, Val(derivative),
        kernel_d1_pre, kernel_d2_pre, Val(subgrid)
    )
end

function validate_subgrid_compatibility(degree::Symbol, derivative::Int, subgrid::Symbol)
    if degree == :a0 || degree == :a1
        return :not_used
    end 
    max_deriv = max_smooth_derivative[degree]  # e.g., 3 for b5
    available_derivs = max_deriv - derivative
    
    if required_derivs[subgrid] > available_derivs
        throw(ArgumentError(
            "Subgrid strategy :$subgrid requires $(required_derivs[subgrid]) additional derivatives, " *
            "but only $available_derivs available for kernel :$degree with derivative order $derivative. " *
            "Use :$(suggest_subgrid(available_derivs)) subgrid or lower instead."
        ))
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