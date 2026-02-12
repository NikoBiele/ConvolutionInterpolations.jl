# docstring for polynomial convolution kernels
"""
    (::ConvolutionKernel{DG})(s::T) where {T,DG}

Evaluate a polynomial convolution kernel of degree `DG` at position `s`.

# Arguments
- `s::T`: The position at which to evaluate the kernel

# Returns
- The kernel value at position `s`

# Details
This function implements various polynomial convolution kernels of different degrees.
Each kernel is defined by a piecewise polynomial function that provides different levels
of accuracy and smoothness properties.

The following kernel degrees are supported:

## ':a' type kernels:

## Nearest Neighbor Kernel (DG=:a0)
The nearest neighbor convolution kernel with 1 equation and support [-0.5,0.5]:
- 0th order accuracy with no continuity
- 1 piecewise equation for |s| < 0.5

## Linear Kernel (DG=:a1)
The linear convolution kernel with 1 equation and support [-1,1]:
- 1st order accuracy with C0 continuity
- 1 piecewise equation for |s| < 1.0

## Cubic Kernel (DG=:a3)
The original cubic convolution kernel with 2 equations and support [-2,2]:
- 3rd order accuracy with C1 continuity
- 2 piecewise equations for |s| < 1.0 and 1.0 ≤ |s| < 2.0

## Cubic Kernel (DG=:a4)
3rd degree convolution kernel with 3 equations and support [-3,3]:
- 4th order accuracy with C1 continuity
- 3 piecewise equations for |s| < 1.0 and 1.0 ≤ |s| < 3.0

## Quintic Kernel (DG=:a5)
Quintic convolution kernel with 3 equations and support [-3,3]:
- Order of accuracy unknown (but less than the :b5 kernel) with C1 continuity
- 3 piecewise equations covering ranges from |s| < 1.0 to |s| < 3.0

## Septic Kernel (DG=:a7)
Septic convolution kernel with 4 equations and support [-4,4]:
- Order of accuracy unknown (but less than the :b7 kernel) with C1 continuity
- 4 piecewise equations covering ranges from |s| < 1.0 to |s| < 4.0

## ':b' type kernels:

## Quintic Kernel (DG=:b5)
Quintic convolution kernel with 5 equations and support [-5,5]:
- 7th order accuracy with C3 continuity
- 5 piecewise equations for |s| < 1.0 and 1.0 ≤ |s| < 5.0

## Septic Kernel (DG=:b7)
Septic convolution kernel with 6 equations and support [-6,6]:
- 7th order accuracy with C4 continuity
- 6 piecewise equations for |s| < 1.0 and 1.0 ≤ |s| < 6.0

## Nonic Kernel (DG=:b9)
Nonic convolution kernel with 7 equations and support [-7,7]:
- 7th order accuracy with C5 continuity
- 7 piecewise equations covering ranges from |s| < 1.0 to |s| < 7.0

## 11th Degree Kernel (DG=:b11)
11th degree convolution kernel with 8 equations and support [-8,8]:
- 7th order accuracy with C6 continuity
- 8 piecewise equations covering ranges from |s| < 1.0 to |s| < 8.0

## 13th Degree Kernel (DG=13)
13th degree convolution kernel with 9 equations and support [-9,9]:
- 7th order accuracy with C6 continuity
- 9 piecewise equations covering ranges from |s| < 1.0 to |s| < 9.0

For all kernels, the support range equals [-E,E] where E is the number of equations.
Higher degree kernels provide better smoothness and derivative properties at the cost
of computational complexity.

# See Also
- `GaussianConvolutionKernel`: Alternative kernel with C∞ continuity
"""

# docstring for gaussian convolution kernels
"""
    (::GaussianConvolutionKernel{B})(s) where B

Evaluate a Gaussian smoothing convolution kernel with parameter `B` at position `s`.

# Arguments
- `s`: The position at which to evaluate the kernel

# Returns
- The kernel value at position `s`

# Details
This function implements a smoothing convolution kernel based on a normalized Gaussian function.
The kernel is defined by:

    K(s) = (1 / θ(B)) * exp(-B * s²)

where θ(B) is the normalization factor calculated as:

    θ(B) = 1 + 2 * ∑(exp(-B * n²)) for n = 1 to ∞

## Kernel Parameter B
The parameter B controls the width of the Gaussian:
- Larger B values produce a narrower Gaussian with faster decay
- Smaller B values produce a wider Gaussian with slower decay
- Typical values range from 1.0 (very smooth) to 10.0 (more localized)

## Properties
Unlike the polynomial-based convolution kernels, the Gaussian kernel has:
- Infinite support (though practically truncated when values become negligible)
- C∞ continuity (infinitely differentiable)
- Very smooth interpolation results
- Controlled blurring which can be beneficial for noise reduction

## Applications
This kernel is particularly useful for:
- Smoothing noisy data
- Creating visually pleasing interpolations where sharp features are not critical
- Applications where differentiability at all orders is important
- Signal processing requiring a controlled frequency response

## Implementation Note
The normalization sum θ(B) is computed using a convergent series approximation,
truncated when terms become sufficiently small (below 1e-12).

# See Also
- `ConvolutionKernel`: Polynomial-based kernels with finite support
"""