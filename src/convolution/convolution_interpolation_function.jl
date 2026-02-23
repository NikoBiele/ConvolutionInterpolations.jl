"""
    convolution_interpolation(knots::Union{AbstractVector,NTuple{N,AbstractVector}}, values::AbstractArray{T,N}; 
        degree::Symbol=:b5, fast::Bool=true, precompute::Int=100_000, 
        B=nothing, extrapolation_bc=Throw(), kernel_bc=:auto) where {T,N}

Create a convolution-based interpolation object with automatic optimization and boundary handling.

# Arguments
- `knots`: Vector (for 1D) or tuple of vectors (for N-D) containing the grid coordinates
- `values`: Array containing the values at the grid points

# Keyword Arguments
- `degree::Symbol=:b5`: Convolution kernel to use. Available kernels:
  - `a`-series: `:a0` (nearest), `:a1` (linear), `:a3` (cubic), `:a5` (quintic), `:a7` (septic)
  - `b`-series (recommended): `:a4`, `:b5`, `:b7`, `:b9`, `:b11`, `:b13`
  - Default `:b5` provides quintic reproduction with 7th-order accuracy (from Taylor series)
- `fast::Bool=true`: Use fast mode with precomputed kernel values for O(1) evaluation
  - `true` (default): Fast mode with precomputed kernels (recommended for most cases)
  - `false`: Direct kernel evaluation (use for memory-constrained scenarios)
- `precompute::Int=100_000`: Number of kernel values to precompute (fast mode only).
  Default value provides maximum accuracy. Values below 100,000 degrade interpolation accuracy
- `B=nothing`: If provided, uses Gaussian kernel with parameter `B` for C∞ smoothness
- `extrapolation_bc=Throw()`: Boundary condition for extrapolation outside the grid domain.
  Options: `Throw()`, `Flat()`, `Line()`, `Periodic()`, or `Natural()`
- `kernel_bc=:auto`: Boundary condition for kernel evaluation at domain boundaries

# Returns
Interpolation object callable at arbitrary points, with extrapolation handling

# Performance
Fast mode (default) provides:
- O(1) evaluation time independent of grid size
- Allocation-free evaluation
- Example timings (1D): `:a1` ~4ns, `:b5` ~14ns per evaluation
- Recommended for all use cases unless memory is extremely limited

# Kernel Properties
All `b`-series kernels provide 7th-order accuracy:
- `:a4`: C¹ continuity, cubic reproduction
- `:b5`: C³ continuity, quintic reproduction (recommended default)
- `:b7`: C⁴ continuity, septic reproduction
- `:b9`: C⁵ continuity, 7th-order accuracy
- `:b11`: C⁶ continuity, 7th-order accuracy
- `:b13`: C⁶ continuity, 7th-order accuracy

Higher-order `b` kernels offer better accuracy on smooth functions, with `:b5` approaching
machine precision with sufficient grid resolution (e.g., `:b5` achieves ~10⁻¹⁴ error on
smooth functions with 1000+ grid points).

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

# Direct evaluation mode (no precomputation)
itp = convolution_interpolation(x, y, fast=false)

# Custom precompute resolution for memory-constrained scenarios
itp = convolution_interpolation(x, y, precompute=10_000)

# Gaussian smoothing kernel
itp = convolution_interpolation(x, y, B=0.1)  # Controlled smoothing
```

See also: `FastConvolutionInterpolation`, `ConvolutionInterpolation`, `ConvolutionExtrapolation`.
"""
function convolution_interpolation(knots::Union{NTuple{N, AbstractVector},AbstractVector,AbstractRange,NTuple{N,AbstractRange}},
                    values::AbstractArray{T,N}; degree::Symbol=:b5, fast::Bool=true, precompute::Int=100,
                    B=nothing, extrapolation_bc=Throw(), kernel_bc=:auto,
                    derivative::Int=0,
                    subgrid::Symbol=:cubic) where {T,N}
    
    # Force non-fast path for nonuniform grids
    knots_tuple = knots isa AbstractVector || knots isa AbstractRange ? (knots,) : knots
    if any(d -> !is_uniform_grid(knots_tuple[d]), 1:N)
        fast = false
    end

    # Handle Natural extrapolation boundary condition specially
    if extrapolation_bc isa Natural
        itp = ConvolutionInterpolation(knots, values;
              degree=degree, B=B, kernel_bc=kernel_bc, derivative=derivative)
        if fast
            itp = FastConvolutionInterpolation(itp.knots, itp.coefs;
                  degree=degree, precompute=precompute, B=B, kernel_bc=:linear,
                  base=base, derivative=derivative, integral=integral, subgrid=subgrid)
        else
            itp = ConvolutionInterpolation(itp.knots, itp.coefs;
                  degree=degree, B=B, kernel_bc=:linear, derivative=derivative)
        end 
        return ConvolutionExtrapolation(itp, Line())
    else
        # Standard case
        if fast
            itp = FastConvolutionInterpolation(knots, values;
                  degree=degree, precompute=precompute, B=B, kernel_bc=kernel_bc, 
                  derivative=derivative, subgrid=subgrid)
        else
            itp = ConvolutionInterpolation(knots, values; degree=degree, B=B,
                  kernel_bc=kernel_bc, derivative=derivative)
        end
        return ConvolutionExtrapolation(itp, extrapolation_bc)
    end
end