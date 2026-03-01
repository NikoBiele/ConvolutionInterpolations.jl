"""
    convolution_interpolation(knots, values::AbstractArray{T,N}; 
        degree::Symbol=:b5, fast::Bool=true, precompute::Int=101, 
        B=nothing, extrapolation_bc=Throw(), kernel_bc=:auto,
        derivative::Int=0, subgrid::Symbol=:cubic) where {T,N}

Create a convolution-based interpolation object with automatic optimization and boundary handling.

# Arguments
- `knots`: Vector or range (for 1D) or tuple of vectors/ranges (for N-D) containing the grid coordinates
- `values`: Array containing the values at the grid points

# Keyword Arguments
- `degree::Symbol=:b5`: Convolution kernel to use. Available kernels:
  - `a`-series: `:a0` (nearest), `:a1` (linear), `:a3` (cubic), `:a5` (quintic), `:a7` (septic)
  - `b`-series (recommended): `:a4`, `:b5`, `:b7`, `:b9`, `:b11`, `:b13`
  - Default `:b5` provides quintic reproduction with 7th-order accuracy
- `fast::Bool=true`: Use fast mode with precomputed kernel values for O(1) evaluation.
  Automatically set to `false` for nonuniform grids
- `precompute::Int=101`: Resolution of the precomputed kernel table. The default uses
  pre-shipped tables (zero computation). Higher values (e.g. 10_000) trigger on-demand
  computation and disk caching for the `:linear` subgrid mode
- `B=nothing`: If provided, uses Gaussian kernel with parameter `B` for C∞ smoothness
- `extrapolation_bc=Throw()`: Boundary condition for extrapolation outside the grid domain.
  Options: `Throw()`, `Flat()`, `Line()`, `Periodic()`, or `Natural()`
- `kernel_bc=:auto`: Boundary condition for kernel evaluation at domain boundaries.
  Options: `:auto`, `:polynomial`, `:linear`, `:quadratic`, `:periodic`, `:detect`
- `derivative::Int=0`: Order of derivative to evaluate (0 for interpolation, up to 6 for b-series)
- `subgrid::Symbol=:cubic`: Subgrid interpolation mode for precomputed kernel tables.
  Options: `:linear` (fastest, needs high precompute), `:cubic` (default), `:quintic` (most accurate)

# Returns
`ConvolutionExtrapolation` object callable at arbitrary points, with extrapolation handling.

# Performance
Construction: ~3μs for 1D with 100 grid points (b5 kernel).
Evaluation (1D): `:a1` ~3ns, `:b5` ~14ns per query, O(1) and allocation-free.

# Kernel Properties
All `b`-series kernels provide 7th-order accuracy:
- `:a4`: C¹ continuity, cubic reproduction
- `:b5`: C³ continuity, quintic reproduction (recommended default)
- `:b7`: C⁴ continuity, septic reproduction
- `:b9`: C⁵ continuity, 7th-order accuracy
- `:b11`: C⁶ continuity, 7th-order accuracy
- `:b13`: C⁶ continuity, 7th-order accuracy

# Examples
```julia
# 1D interpolation with default quintic kernel
x = range(0, 2π, length=100)
y = sin.(x)
itp = convolution_interpolation(x, y)
itp(π/3)  # Evaluate at π/3

# 2D interpolation with high-order kernel
xs = range(-2, 2, length=50)
ys = range(-2, 2, length=50)
zs = [exp(-(x^2 + y^2)) for x in xs, y in ys]
itp = convolution_interpolation((xs, ys), zs, degree=:b7)
itp(0.5, -1.2)  # Evaluate at (0.5, -1.2)

# With extrapolation boundary condition
itp = convolution_interpolation(x, y, extrapolation_bc=Line())
itp(3π)  # Extrapolates linearly beyond domain

# First derivative
itp = convolution_interpolation(x, y, derivative=1)
itp(π/3)  # Evaluates cos(π/3) ≈ 0.5

# Direct evaluation mode (no precomputation)
itp = convolution_interpolation(x, y, fast=false)
```

See also: `FastConvolutionInterpolation`, `ConvolutionInterpolation`, `ConvolutionExtrapolation`.
"""

function convolution_interpolation(knots, values::AbstractArray{T,N}; 
        degree::Symbol=:b5, fast::Bool=true, precompute::Int=101,
        B=nothing, extrapolation_bc=Throw(), kernel_bc=:auto,
        derivative::Int=0, subgrid::Symbol=:cubic) where {T,N}
    
    knots_tuple = knots isa AbstractVector || knots isa AbstractRange ? (knots,) : knots
    if any(d -> !is_uniform_grid(knots_tuple[d]), 1:N)
        fast = false
    end

    if extrapolation_bc isa Natural
        return _build_natural(knots, values; degree, fast, precompute, B, kernel_bc, derivative, subgrid)
    elseif fast
        return _build_fast(knots, values; degree, precompute, B, kernel_bc, derivative, subgrid, extrapolation_bc)
    else
        return _build_slow(knots, values; degree, B, kernel_bc, derivative, extrapolation_bc)
    end
end

function _build_fast(knots, values; degree, precompute, B, kernel_bc, derivative, subgrid, extrapolation_bc)
    itp = FastConvolutionInterpolation(knots, values;
          degree, precompute, B, kernel_bc, derivative, subgrid)
    return ConvolutionExtrapolation(itp, extrapolation_bc)
end

function _build_slow(knots, values; degree, B, kernel_bc, derivative, extrapolation_bc)
    itp = ConvolutionInterpolation(knots, values; degree, B, kernel_bc, derivative)
    return ConvolutionExtrapolation(itp, extrapolation_bc)
end

function _build_natural(knots, values; degree, fast, precompute, B, kernel_bc, derivative, subgrid)
    itp = ConvolutionInterpolation(knots, values; degree, B, kernel_bc, derivative)
    if fast
        itp = FastConvolutionInterpolation(itp.knots, itp.coefs;
              degree, precompute, B, kernel_bc=:linear, derivative, subgrid)
    else
        itp = ConvolutionInterpolation(itp.knots, itp.coefs;
              degree, B, kernel_bc=:linear, derivative)
    end
    return ConvolutionExtrapolation(itp, Line())
end