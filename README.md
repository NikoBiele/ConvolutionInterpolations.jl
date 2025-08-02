# ConvolutionInterpolations.jl

A Julia package for smooth N-dimensional interpolation using separable convolution kernels.

![Performance of available kernels](fig/convolution_interpolation_a_and_b.png)

## Features

- N-dimensional interpolation (1D to arbitrary dimensions) using separable convolution kernels
- Multiple polynomial kernel options in the same framework: Nearest neighbor, linear, cubic, quintic etc.
- Two options for accuracy:
- (1) ':a' kernels: Published kernels from the literature (minimal boundary coefficient handling)
- (2) ':b' kernels: New superior high order of accuracy kernels developed for this package (but more boundary coefficients)
- Various extrapolation methods (linear, flat, periodic, reflection)
- Zero dependencies outside core Julia
- Fast implementation with precomputed kernel optimizations
- Sample data must be uniformly spaced in each dimension, but spacing can vary between dimensions
- Optional Gaussian convolution kernel for smoothing noisy data

## Installation

```julia
using Pkg
Pkg.add("ConvolutionInterpolations")
```

## Quick Start

```julia
using ConvolutionInterpolations

# 1D interpolation
x = range(0, 1, length=10)
y = sin.(2π .* x)
itp = convolution_interpolation(x, y)
itp(0.5) # Evaluate at a specific point

# 2D interpolation
xs = range(-2, 2, length=20)
ys = range(-2, 2, length=20)
zs = [exp(-(x^2 + y^2)) for x in xs, y in ys]
itp_2d = convolution_interpolation((xs, ys), zs)
itp_2d(0.5, -1.2) # Evaluate at a specific point
```

## Advanced Usage

### Kernel Selection

Control the precision/smoothness by selecting different kernel degrees:
```julia
# Default cubic interpolation (degree=:a3)
itp = convolution_interpolation(x, y) # cubic (C1 continuous)

# Higher-order interpolation for enhanced smoothness
itp = convolution_interpolation(x, y; degree=:b5)  # quintic (7th order accurate, C3 continuous)
itp = convolution_interpolation(x, y; degree=:b7)  # septic (7th order accurate, C5 continuous)

# Ultra-high-precision interpolation
itp = convolution_interpolation(x, y; degree=:b9)  # 9th degree (7th order accurate, C7 continuous)
itp = convolution_interpolation(x, y; degree=:b11) # 11th degree (7th order accurate, C9 continuous)
itp = convolution_interpolation(x, y; degree=:b13) # 13th degree (7th order accurate, C11 continuous)

# Gaussian convolution smoothing interpolation (does not hit data points)
itp = convolution_interpolation(x, y; B=1.0)     # B controls width
```
In general the ':b' type kernels should be preferred, since they show superior performance.

### Kernel Boundary Condition

Used internally to determine interpolation behaviour near boundaries (critical for high accuracy).
Default behaviour is to attempt to detect the most appropriate kernel boundary condition (linear, quadratic or periodic).

```julia
itp = convolution_interpolation(x, y; kernel_bc=:detect) # default

# other options include: linear, quadratic or periodic
itp = convolution_interpolation(x, y; kernel_bc=:linear) # linear boundary condition
itp = convolution_interpolation(x, y; kernel_bc=:quadratic) # adapts to linear or quadratic signals
itp = convolution_interpolation(x, y; kernel_bc=:periodic) # good with periodic signals
```
It is possible to specify the kernel boundary condition in a more detailed way, by specifying a vector of tuples, where each vector entry is a dimension, and the two tuple entries give the boundary condition at each end of this dimension:

```julia
x = range(0, 1, length=10)
y = range(0, 1, length=10)
z = [sin(2π*xi) for xi in x, yi in y]
kernel_bcs = [(:linear, :quadratic), # both ends of first dimension
             (:periodic, :detect)] # both ends of second dimension
itp = convolution_interpolation((x, y), z; kernel_bc=kernel_bcs)
```

### Extrapolation

Choose from different boundary conditions for extrapolating beyond the data domain:

```julia
# Error for out-of-bounds access (default)
itp = convolution_interpolation(x, y; extrapolation_bc=Throw())

# Linear extrapolation
itp = convolution_interpolation(x, y; extrapolation_bc=Line())

# Constant extrapolation at boundaries
itp = convolution_interpolation(x, y; extrapolation_bc=Flat())

# Natural extrapolation at boundaries (highest continuity, but more boundary handling)
itp = convolution_interpolation(x, y; extrapolation_bc=Natural())

# Periodic extension
itp = convolution_interpolation(x, y; extrapolation_bc=Periodic())

# Reflection at boundaries
itp = convolution_interpolation(x, y; extrapolation_bc=Reflect())
```

### High-Dimensional Data

ConvolutionInterpolations.jl handles smooth interpolation of high-dimensional data:

```julia
# 4D interpolation example (time series of 3D volumes)
x = range(0, 1, length=10)
y = range(0, 1, length=10)
z = range(0, 1, length=10)
t = range(0, 1, length=5)

# Create 4D data (e.g., time-evolving 3D scalar field)
data_4d = [sin(2π*xi)*cos(2π*yi)*exp(-zi)*sqrt(ti) for xi in x, yi in y, zi in z, ti in t]

# Create 4D interpolator
itp_4d = convolution_interpolation((x, y, z, t), data_4d)

# Evaluate at arbitrary 4D point
value = itp_4d(0.42, 0.33, 0.77, 0.51)
```
Works analogously for 5D, 6D, or higher dimensionality

## Performance Tips

- Use ```degree=:b``` type kernels whenever possible, as they perform best
- Use the highest degree kernel possible, since a higher degree only improves interpolation quality
- If the ```degree=:b``` constructors become too slow, switch to the ```degree=:a``` kernels
- Use ```degree=:b5``` to access the most efficient high order of accuracy kernel (C3 continuous, 7th order accurate)
- Use ```degree=:a``` only when the ```degree=:b``` constructors become too slow, to minimize boundary handling
- Use lower degrees for high dimensional data to minimize necessary boundary condition handling
- Set ```fast=true``` (default) to use precomputed kernels for better performance
- Use the ```Natural()``` extrapolation boundary condition for high continuity in lower dimensions (not recommended for higher dimensions)
- Adjust the ```precompute``` parameter to control the accuracy-memory tradeoff for kernel evaluation:
```julia
# More precomputed points = more accurate kernel evaluation but more memory usage
itp = convolution_interpolation(x, y; precompute=10_000) # Default is 1000

# Fewer points = less memory usage but slightly less accurate kernel evaluation
itp = convolution_interpolation(x, y; precompute=500)
```
- Increasing ```precompute``` increases accuracy at virtually no cost after the precompute step

## Comparison with Other Packages

Unlike other interpolation packages, ConvolutionInterpolations.jl:

- Provides a wide range of interpolation kernels (nearest neighbor up to 13th degree kernel)
- Combines the best of both worlds: A single framework for both low and high order of accuracy interpolation
- Uses separable convolution kernels, which extend naturally into higher dimensions
- Provides smooth interpolation in higher dimensions, with adjustable accuracy
- Requires zero dependencies outside core Julia

## Acknowledgments

ConvolutionInterpolations.jl draws significant inspiration from [Interpolations.jl](https://github.com/JuliaMath/Interpolations.jl) in terms of API design and interface. This package originally started as a PR for Interpolations.jl before evolving into a standalone implementation. I'm grateful to the Interpolations.jl maintainers and contributors for their excellent work, which provided a solid foundation for the design patterns used here.

The implementation of the convolution kernels is based on seminal work in the field:

- R. G. Keys, "Cubic Convolution Interpolation for Digital Image Processing," in IEEE Transactions on Acoustics, Speech, and Signal Processing, vol. 29, no. 6, pp. 1153-1160, December 1981.

- E. H. W. Meijering, K. J. Zuiderveld, and M. A. Viergever, "Image Reconstruction by Convolution with Symmetrical Piecewise nth-Order Polynomial Kernels," in IEEE Transactions on Image Processing, vol. 8, no. 2, pp. 192-201, February 1999.

These papers provided the mathematical foundation for the high-order polynomial kernels developed for this package.

## License

This package is available under the MIT License.