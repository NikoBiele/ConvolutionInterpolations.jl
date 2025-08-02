"""
    convolution_interpolation(knots::Union{AbstractVector,NTuple{N,AbstractVector}}, values::AbstractArray{T,N}; 
        degree::Symbol=:a3, fast::Bool=true, precompute::Int=1000, B=nothing, extrapolation_bc=Throw(), kernel_bc=:detect) where {T,N}

Create a convolution-based interpolation object for the given data.

# Arguments
- `knots`: Vector (for 1D) or tuple of vectors (for N-D) containing the grid coordinates
- `values`: Array containing the values at the knot points

# Keyword Arguments
- `degree::Symbol=:a3`: The degree of the polynomial kernel (':a3' for cubic, ':a5' for quintic, etc.)
- `fast::Bool=true`: Whether to use `FastConvolutionInterpolation` with precomputed kernels (faster but uses more memory)
- `precompute::Int=1000`: The number of positions to precompute for kernel evaluation (only used if fast=true)
- `B=nothing`: Parameter for Gaussian kernel (if not `nothing`, uses a Gaussian kernel instead of polynomial)
- `extrapolation_bc=Throw()`: The boundary condition to use for extrapolation outside the domain

# Returns
- An interpolation object that can evaluate the function at arbitrary points

# Details
This is the main entry point function for creating convolution interpolations. It creates
either a `FastConvolutionInterpolation` or `ConvolutionInterpolation` object based on the
`fast` parameter, and wraps it with the specified extrapolation boundary condition.

The `degree` parameter controls the accuracy and smoothness of the interpolation:
- :b3 (cubic): C1 continuity, 4th order accuracy
- :b5 (quintic): C3 continuity, 7th order accuracy
- :b7 (septic): C5 continuity, 7th order accuracy
- :b9 (nonic): C7 continuity, 7th order accuracy
- :b11 (decic): C9 continuity, 7th order accuracy
- :b13 (undecic): C11 continuity, 7th order accuracy

If `B` is provided, a Gaussian smoothing kernel with parameter `B` is used instead of a polynomial kernel,
providing infinite differentiability (C∞) but with some controlled blurring.

The returned object can be called as a function with real-valued coordinates to perform the
actual interpolation, e.g., `itp(x)` or `itp(x, y, z)`.

# Examples
```julia
# 1D interpolation with cubic kernel
x = range(0, 2π, length=100)
y = sin.(x)
itp = convolution_interpolation(x, y)
itp(π/3)  # Interpolated value at π/3

# 2D interpolation with quintic kernel and linear extrapolation
xs = range(-2, 2, length=30)
ys = range(-2, 2, length=30)
zs = [exp(-(x^2 + y^2)) for x in xs, y in ys]
itp = convolution_interpolation((xs, ys), zs; degree=:b5, extrapolation_bc=Line())
itp(0.5, -1.2)  # Interpolated value at (0.5, -1.2)
```
"""
function convolution_interpolation(knots::Union{AbstractVector,NTuple{N,AbstractVector}}, values::AbstractArray{T,N}; 
    degree::Symbol=:a3, fast::Bool=true, precompute::Int=1000, B=nothing, extrapolation_bc=Throw(), kernel_bc=:detect) where {T,N}
    if knots isa AbstractVector
        knots = (knots,)
    end
    if fast
        itp = FastConvolutionInterpolation(knots, values; degree=degree, precompute=precompute, B=B, kernel_bc=kernel_bc)
    else
        itp = ConvolutionInterpolation(knots, values; degree=degree, B=B, kernel_bc=kernel_bc)
    end
    if extrapolation_bc isa Natural
        if fast
            itp = FastConvolutionInterpolation(itp.knots, itp.coefs; degree=degree, precompute=precompute, B=B, kernel_bc=:linear)
        else
            itp = ConvolutionInterpolation(itp.knots, itp.coefs; degree=degree, B=B, kernel_bc=:linear)
        end 
        return ConvolutionExtrapolation(itp, Line())
    else
        return ConvolutionExtrapolation(itp, extrapolation_bc)
    end
end