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

"""
    HigherOrderKernel(::Val{DG}) where DG

Construct a `HigherOrderKernel{DG}` type from a `Val{DG}` for dispatch purposes.

This is a convenience constructor that creates a `HigherOrderKernel{DG}` instance from
a value type, used for implementing specialized kernel methods for higher-order
convolution interpolation.
"""

# the structs and abstract types defined here are to control functor dispatch

abstract type AbstractConvolutionKernel end

abstract type AbstractMixedConvolutionKernel end

struct HigherOrderKernel{DG} <: AbstractConvolutionKernel end

struct HigherOrderMixedKernel{DG} <: AbstractMixedConvolutionKernel end

HigherOrderKernel(::Val{DG}) where DG = HigherOrderKernel{DG}()

HigherOrderMixedKernel(::Val{DG}) where DG = HigherOrderMixedKernel{DG}()

struct LowerOrderKernel{DG} <: AbstractConvolutionKernel end
LowerOrderKernel(::Val{DG}) where DG = LowerOrderKernel{DG}()

struct LowerOrderMixedKernel{DG} <: AbstractMixedConvolutionKernel end
LowerOrderMixedKernel(::Val{DG}) where DG = LowerOrderMixedKernel{DG}()

struct FullMixedOrderKernel{DG} <: AbstractMixedConvolutionKernel end
FullMixedOrderKernel(::Val{DG}) where DG = FullMixedOrderKernel{DG}()

struct NonUniformMixedOrderKernel{DG} <: AbstractMixedConvolutionKernel end
NonUniformMixedOrderKernel(::Val{DG}) where DG = NonUniformMixedOrderKernel{DG}()

struct NonUniformNonMixedLowKernel{DG} <: AbstractConvolutionKernel end
NonUniformNonMixedLowKernel(::Val{DG}) where DG = NonUniformNonMixedLowKernel{DG}()

struct NonUniformNonMixedHighKernel{DG} <: AbstractConvolutionKernel end
NonUniformNonMixedHighKernel(::Val{DG}) where DG = NonUniformNonMixedHighKernel{DG}()

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

"""
    IntegralOrder

Dispatch type indicating that a `ConvolutionInterpolation` or
`FastConvolutionInterpolation` computes the antiderivative (indefinite integral)
of the interpolant. Stored in the derivative slot (position 11) of the interpolant
type when `derivative=-1` is passed at construction.

See also: `DerivativeOrder`, `convolution_interpolation`.
"""
struct IntegralOrder end

"""
    DerivativeOrder{DO}

Dispatch type indicating the derivative order of a `ConvolutionInterpolation` or
`FastConvolutionInterpolation`. Stored in the derivative slot (position 11) of the
interpolant type. `DO=0` corresponds to plain interpolation, `DO=1` to the first
derivative, and so on up to the kernel's maximum supported order.

# Type Parameters
- `DO`: Derivative order as an `Int`. Must satisfy `0 ≤ DO ≤ max_derivative[kernel]`.

See also: `IntegralOrder`, `convolution_interpolation`.
"""
struct DerivativeOrder{DO} end
DerivativeOrder(::Val{DO}) where DO = DerivativeOrder{DO}()

"""
    MixedIntegralOrder{DO}
 
Dispatch type for per-dimension mixed antiderivative/derivative orders.
`DO` is an `NTuple{N,Int}` where `DO[d] == -1` means integrate along
dimension `d`, and `DO[d] >= 0` means differentiate to that order.
 
Used when at least one but not all dimensions have `derivative == -1`.
`IntegralOrder` is still used when ALL dimensions are `-1`.
`DerivativeOrder{DO}` is still used when NO dimensions are `-1`.
 
See also: `IntegralOrder`, `DerivativeOrder`.
"""
struct MixedIntegralOrder{DO} end