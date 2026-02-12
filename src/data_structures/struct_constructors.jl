"""
    HigherDimension{N}

A type used for dispatch on higher-dimensional interpolation methods (N > 3).

This type provides specialized handling for interpolation in four or more dimensions,
where dimension-specific optimizations are no longer practical. It allows for a generic
implementation that works with arbitrary dimensionality.

# Type Parameters
- `N`: The number of dimensions in the interpolation

This type is used internally within the interpolation system and is not typically
manipulated directly by users.
"""
struct HigherDimension{N} end

"""
    HigherDimension(::Val{N}) where N

Construct a `HigherDimension{N}` type from a `Val{N}` for dispatch purposes.

This is a convenience constructor that creates a `HigherDimension{N}` instance from
a value type, used for implementing specialized methods based on dimensionality.
"""
HigherDimension(::Val{N}) where N = HigherDimension{N}()

"""
    HigherOrderKernel{DG}

A type representing higher-order convolution kernels (`:a3` and above) for dispatch.

This type is used to distinguish between simple kernels (`:a0` nearest, `:a1` linear)
and higher-order kernels that require precomputed kernel values and convolution
operations for evaluation.

# Type Parameters
- `DG`: The kernel degree symbol (`:a3`, `:a4`, `:b5`, `:b7`, `:b9`, `:b11`, `:b13`)

# Details
Higher-order kernels require:
- Precomputed kernel values stored in `kernel_pre` matrix
- Convolution over a support region of size `2*eqs-1`
- Linear interpolation between precomputed kernel positions
- Boundary condition handling via ghost points

This type enables specialized dispatch for the fast evaluation methods that use
precomputed kernels, distinguishing them from `:a0` and `:a1` which use simpler
direct evaluation.

# Usage
Used internally for method dispatch in `FastConvolutionInterpolation` evaluation.
Not typically constructed directly by users.

See also: `ConvolutionKernel`, `FastConvolutionInterpolation`.
"""

struct HigherOrderKernel{DG} end

"""
    HigherOrderKernel(::Val{DG}) where DG

Construct a `HigherOrderKernel{DG}` type from a `Val{DG}` for dispatch purposes.

This is a convenience constructor that creates a `HigherOrderKernel{DG}` instance from
a value type, used for implementing specialized kernel methods for higher-order
convolution interpolation.
"""

HigherOrderKernel(::Val{DG}) where DG = HigherOrderKernel{DG}()

"""
    ConvolutionKernel{DG}

A type representing a polynomial convolution kernel of degree `DG`.

This type encapsulates the behavior of a specific degree polynomial kernel used
for convolution interpolation. Different degrees provide different levels of
accuracy and smoothness in the interpolated result.

# Type Parameters
- `DG`: The degree of the polynomial kernel (3=cubic, 5=quintic, 7=septic, etc.)

The implementation defines specific behavior for different degrees through
method specialization.
"""
struct ConvolutionKernel{DG,DO} end

"""
    ConvolutionKernel(::Val{DG}) where {DG}

Construct a `ConvolutionKernel{DG}` type from a `Val{DG}` for dispatch purposes.

This is a convenience constructor that creates a `ConvolutionKernel{DG}` instance from
a value type, used for implementing specialized kernel methods based on degree.
"""
ConvolutionKernel(::Val{DG}, ::Val{DO}) where {DG,DO} = ConvolutionKernel{DG,DO}()

"""
    GaussianConvolutionKernel{B}

A type representing a Gaussian smoothing convolution kernel with parameter `B`.

This type encapsulates the behavior of a Gaussian kernel used for convolution
interpolation. The Gaussian kernel provides smooth interpolation with infinite
differentiability but introduces controlled blurring.

# Type Parameters
- `B`: The width parameter of the Gaussian (larger B = narrower Gaussian)

The implementation defines specific behavior through method specialization.
"""
struct GaussianConvolutionKernel{B} end

"""
    GaussianConvolutionKernel(::Val{B}) where B

Construct a `GaussianConvolutionKernel{B}` type from a `Val{B}` for dispatch purposes.

This is a convenience constructor that creates a `GaussianConvolutionKernel{B}` instance
from a value type, used for implementing specialized Gaussian kernel methods.
"""
GaussianConvolutionKernel(::Val{B}) where B = GaussianConvolutionKernel{B}()