"""
    FastConvolutionInterpolation(knots::NTuple{N,AbstractVector}, vs::AbstractArray{T,N};
                                degree::Symbol=:b5, precompute::Int=100_000, B=nothing, kernel_bc=:auto) where {T,N}

Construct a fast convolution interpolation object with precomputed kernel values for O(1) evaluation.

# Arguments
- `knots::NTuple{N,AbstractVector}`: Tuple of vectors containing the grid points in each dimension
- `vs::AbstractArray{T,N}`: Values to interpolate at the grid points

# Keyword Arguments
- `degree::Symbol=:b5`: Convolution kernel to use. Available kernels:
  - `a`-series: `:a0` (nearest), `:a1` (linear), `:a3` (cubic), `:a5` (quintic), `:a7` (septic)
  - `b`-series (recommended): `:a4`, `:b5`, `:b7`, `:b9`, `:b11`, `:b13`
  - Default `:b5` provides quintic reproduction with 7th-order convergence (from Taylor series)
- `precompute::Int=100_000`: Number of kernel values to precompute. Higher values increase
  accuracy of kernel interpolation at the cost of initialization time and memory
- `B=nothing`: If provided, uses Gaussian kernel with parameter `B` instead of polynomial kernel
- `kernel_bc=:auto`: Boundary condition for kernel evaluation at domain boundaries

# Returns
`FastConvolutionInterpolation` object with O(1) evaluation time, independent of grid size

# Performance
Fast mode provides significant speedup over direct evaluation:
- Uses precomputed kernel values with linear interpolation for O(1) lookup
- Allocation-free evaluation across all dimensions
- Typical speedup: 3-10× faster than direct evaluation, especially for higher-order kernels
- Performance scales with dimensionality but not with grid size
- Example timings (1D, 100-point grid): `:a1` ~4ns, `:b5` ~14ns per evaluation

# Details
This constructor creates an optimized convolution interpolator that precomputes kernel values
at regularly spaced positions, enabling constant-time interpolation regardless of grid size.
During evaluation, kernel values are obtained via fast linear interpolation of precomputed data.

The grid is automatically expanded at boundaries and coefficients are computed to satisfy
the chosen boundary conditions. All `b`-series kernels provide 7th-order convergence with
polynomial reproduction properties (e.g., `:b5` exactly reproduces quintic polynomials).

The `precompute` parameter controls the resolution of the precomputed kernel grid. The default
value (100,000) provides excellent accuracy while keeping initialization time reasonable.

# Examples
```julia
# 1D fast interpolation with quintic kernel
knots = (range(0, 1, 100),)
data = sin.(2π .* knots[1])
itp = FastConvolutionInterpolation(knots, data)
itp(0.5)  # O(1) evaluation at x=0.5

# 2D fast interpolation with custom precompute resolution
knots = (range(0, 1, 50), range(0, 1, 50))
data = [sin(x) * cos(y) for x in knots[1], y in knots[2]]
itp = FastConvolutionInterpolation(knots, data, degree=:b5, precompute=50_000)
itp(0.3, 0.7)  # Fast evaluation at (x,y) = (0.3, 0.7)
```

See also: `convolution_interpolation` for automatic extrapolation at boundaries.
"""
function FastConvolutionInterpolation(knots::Union{AbstractVector,NTuple{N,AbstractVector},AbstractRange,NTuple{N,AbstractRange}},
                                      vs::AbstractArray{T,N};
                                      degree::Symbol=:b5, precompute::Int=100, B=nothing,
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
    pre_range, kernel_pre, kernel_d1_pre, kernel_d2_pre = 
                  get_precomputed_kernel_and_range(degree; precompute=big(precompute//1), float_type=T,
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