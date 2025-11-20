"""
    ConvolutionInterpolation(knots::NTuple{N,AbstractVector}, vs::AbstractArray{T,N};
                             degree::Symbol=:b5, B=nothing, kernel_bc=:auto) where {T,N}

Construct a convolution interpolation object for N-dimensional data using direct kernel evaluation.

# Arguments
- `knots::NTuple{N,AbstractVector}`: Tuple of vectors containing the grid points in each dimension
- `vs::AbstractArray{T,N}`: Values to interpolate at the grid points

# Keyword Arguments
- `degree::Symbol=:b5`: Convolution kernel to use. Available kernels:
  - `a`-series: `:a0` (nearest), `:a1` (linear), `:a3` (cubic), `:a5` (quintic), `:a7` (septic)
  - `b`-series (recommended): `:b3`, `:b5`, `:b7`, `:b9`, `:b11`, `:b13`
  - Default `:b5` provides quintic reproduction with 7th-order convergence (from Taylor series)
- `B=nothing`: If provided, uses Gaussian kernel with parameter `B` instead of polynomial kernel
- `kernel_bc=:auto`: Boundary condition for kernel evaluation at domain boundaries

# Returns
`ConvolutionInterpolation` object callable at arbitrary points within the grid domain

# Performance
For faster evaluation, consider `convolution_interpolation(..., fast=true)` which precomputes
kernel values and uses O(1) lookup with linear interpolation (typically 3-10× faster, especially
for higher-order kernels).

# Details
Creates a convolution-based interpolator with specified kernel and boundary handling.
The grid is automatically expanded at boundaries and coefficients are computed to satisfy
the chosen boundary conditions. All `b`-series kernels provide 7th-order convergence with
polynomial reproduction properties (e.g., `:b5` exactly reproduces quintic polynomials).

# Examples
```julia
# 1D interpolation with quintic kernel
knots = (range(0, 1, 100),)
data = sin.(2π .* knots[1])
itp = ConvolutionInterpolation(knots, data)
itp(0.5)  # Evaluate at x=0.5

# 2D interpolation with cubic kernel
knots = (range(0, 1, 50), range(0, 1, 50))
data = [sin(x) * cos(y) for x in knots[1], y in knots[2]]
itp = ConvolutionInterpolation(knots, data, degree=:a3)
itp(0.3, 0.7)  # Evaluate at (x,y) = (0.3, 0.7)
```

See also: `convolution_interpolation` for automatic extrapolation at boundaries.
"""
function ConvolutionInterpolation(knots::NTuple{N,AbstractVector}, vs::AbstractArray{T,N};
                                degree::Symbol=:b5, B=nothing, kernel_bc=:auto) where {T,N}
    eqs = B === nothing ? get_equations_for_degree(degree) : 50
    h = map(k -> k[2] - k[1], knots)
    it = ntuple(_ -> ConvolutionMethod(), N)
    knots_new = expand_knots(knots, eqs-1) # expand boundaries
    coefs = create_convolutional_coefs(vs, h, eqs, kernel_bc, degree) # create boundaries
    kernel = B === nothing ? ConvolutionKernel(Val(degree)) : GaussianConvolutionKernel(Val(B))
    dimension = N <= 3 ? Val(N) : HigherDimension(Val(N))
    
    ConvolutionInterpolation{T,N,typeof(coefs),typeof(it),typeof(knots_new),typeof(kernel),typeof(dimension),typeof(Val(degree)),typeof(eqs),typeof(kernel_bc)}(
        coefs, knots_new, it, h, kernel, dimension, Val(degree), eqs, kernel_bc
    )
end