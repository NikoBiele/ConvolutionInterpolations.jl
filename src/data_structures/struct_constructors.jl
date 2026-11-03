abstract type AbstractConvolutionKernel end

abstract type AbstractNonMixedConvolutionKernel <: AbstractConvolutionKernel end

abstract type AbstractMixedConvolutionKernel <: AbstractConvolutionKernel end

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
    HigherOrderKernel{K}

A type representing higher-order convolution kernels (`:a3` and above) for dispatch.

It distinguishes the simple kernels (`:a0` nearest, `:a1` linear), which evaluate directly,
from the higher-order kernels, which evaluate as a convolution over a stencil of `2*eqs`
coefficients around the point: each coefficient is weighted by a polynomial in the position
within the cell, derived exactly from the kernel's pieces (see `_column_rows`), with boundary
values supplied by ghost points.

# Type Parameters
- `K`: The kernel symbol (`:a3`, `:a4`, `:a5`, `:a7`, `:b5`, `:b7`, `:b9`, `:b11`, `:b13`)

# Usage
Used internally for method dispatch in `ConvolutionInterpolation` evaluation.
Not typically constructed directly by users.

See also: `ConvolutionKernel`, `ConvolutionInterpolation`.
"""
struct HigherOrderKernel{K} <: AbstractNonMixedConvolutionKernel end

struct HigherOrderMixedKernel{K} <: AbstractMixedConvolutionKernel end

"""
    HigherOrderKernel(::Val{K}) where K

Construct a `HigherOrderKernel{K}` type from a `Val{K}` for dispatch purposes.

This is a convenience constructor that creates a `HigherOrderKernel{K}` instance from
a value type, used for implementing specialized kernel methods for higher-order
convolution interpolation.
"""
HigherOrderKernel(::Val{K}) where K = HigherOrderKernel{K}()

HigherOrderMixedKernel(::Val{K}) where K = HigherOrderMixedKernel{K}()

struct LowerOrderKernel{K} <: AbstractNonMixedConvolutionKernel end
LowerOrderKernel(::Val{K}) where K = LowerOrderKernel{K}()

struct LowerOrderMixedKernel{K} <: AbstractMixedConvolutionKernel end
LowerOrderMixedKernel(::Val{K}) where K = LowerOrderMixedKernel{K}()

struct FullMixedOrderKernel{K} <: AbstractMixedConvolutionKernel end
FullMixedOrderKernel(::Val{K}) where K = FullMixedOrderKernel{K}()

struct NonUniformMixedOrderKernel{K} <: AbstractConvolutionKernel end
NonUniformMixedOrderKernel(::Val{K}) where K = NonUniformMixedOrderKernel{K}()

struct NonUniformNonMixedLowKernel{K} <: AbstractNonMixedConvolutionKernel end
NonUniformNonMixedLowKernel(::Val{K}) where K = NonUniformNonMixedLowKernel{K}()

struct NonUniformNonMixedHighKernel{K} <: AbstractNonMixedConvolutionKernel end
NonUniformNonMixedHighKernel(::Val{K}) where K = NonUniformNonMixedHighKernel{K}()

"""
    ConvolutionKernel{K,DO}

A piecewise-polynomial convolution kernel `K` evaluated at derivative order `DO`: an
instance is callable, `ConvolutionKernel{K,DO}()(s)`, and returns the `DO`-th derivative of
the kernel at the normalized offset `s` (`DO = -m` its m-fold antiderivative, see
`_kernel_antiderivative`). Each kernel is symmetric with compact support, cardinal (one at
`s = 0`, zero at the other integers) and built from exact rational coefficients, one
polynomial per unit piece of `|s|`; the derivative tables and the antiderivative pieces are
generated from the base coefficients at precompile time.

| Kernel | Degree | Continuity | Pieces | Uniform-grid order | Notes                                    |
|:-------|:------:|:----------:|:------:|:------------------:|:-----------------------------------------|
| `:a0`  | 0      | —          | 1      | 1                  | nearest neighbour, support `[-1/2, 1/2]` |
| `:a1`  | 1      | C⁰         | 1      | 2                  | linear                                   |
| `:a3`  | 3      | C¹         | 2      | 3                  | Keys cubic; `:n3` on nonuniform grids    |
| `:a4`  | 3      | C¹         | 3      | 4                  | cubic, 6-wide, reproduces degree 3       |
| `:a5`  | 5      | C³         | 3      | 3                  | quintic                                  |
| `:a7`  | 7      | C⁵         | 4      | 3                  | septic                                   |
| `:b5`  | 5      | C³         | 5      | 6                  | reproduces degree 5                      |
| `:b7`  | 7      | C⁵         | 6      | 7                  | reproduces degree 6; the default kernel  |
| `:b9`  | 9      | C⁷         | 7      | 7                  | reproduces degree 6                      |
| `:b11` | 11     | C⁹         | 8      | 7                  | reproduces degree 6                      |
| `:b13` | 13     | C¹¹        | 9      | 7                  | reproduces degree 6                      |

On nonuniform grids the `b` kernels are re-projected per interval and reproduce their full
degree `p`, converging at order `p + 1` (14 for `:b13`); `:n3` is the nonuniform form of
`:a3`. The supported derivative and antiderivative orders of each kernel on each grid type
are listed in the kernel tables of the README and enforced by the constructors; the
construction of the kernels, their reproduction conditions and the convergence results
are worked through on the documentation site, https://convinterp.org.

# Type Parameters
- `K` : the kernel symbol, one of the table above.
- `DO`: the derivative order, `0` for the kernel itself, positive for derivatives and `-m`
  for the m-fold antiderivative.
"""
struct ConvolutionKernel{K,DO} end

"""
    ConvolutionKernel(::Val{K}, ::Val{DO})

Construct the `ConvolutionKernel{K,DO}` instance for kernel `K` at derivative order `DO`
from value types, so that the interpolant's kernel and order are fixed at compile time.
"""
ConvolutionKernel(::Val{K}, ::Val{DO}) where {K,DO} = ConvolutionKernel{K,DO}()

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
struct GaussianConvolutionKernel{B,NT} end

"""
    GaussianConvolutionKernel(::Val{B}) where B

Construct a `GaussianConvolutionKernel{B}` type from a `Val{B}` for dispatch purposes.

This is a convenience constructor that creates a `GaussianConvolutionKernel{B}` instance
from a value type, used for implementing specialized Gaussian kernel methods.
"""
GaussianConvolutionKernel(::Val{B}, ::Val{NT}) where {B,NT} = GaussianConvolutionKernel{B,NT}()

"""
    IntegralOrders{DO}

Dispatch type for antiderivatives of any order in any number of dimensions, mixed with
derivatives. `DO` is the per-dimension order tuple: `DO[d] < 0` integrates along dimension `d`
(order `-DO[d]`), and `DO[d] ≥ 0` interpolates (0) or differentiates along it. Stored in the
derivative slot (position 11) of the interpolant type when any entry of `derivative` is
negative at construction.

See also: `DerivativeOrder`, `convolution_interpolation`.
"""
struct IntegralOrders{DO} end

"""
    DerivativeOrder{DO}

Dispatch type indicating the derivative order of a `ConvolutionInterpolation`. Stored in the
derivative slot (position 11) of the interpolant type. `DO=0` corresponds to plain
interpolation, `DO=1` to the first derivative, and so on up to the kernel's maximum
supported order on the grid of each axis (`_max_derivative_order`).

# Type Parameters
- `DO`: Derivative order as an `Int`, or one `Int` per dimension as a tuple.

See also: `IntegralOrders`, `convolution_interpolation`.
"""
struct DerivativeOrder{DO} end
DerivativeOrder(::Val{DO}) where DO = DerivativeOrder{DO}()

"""
    AbstractExtrapolation

Supertype of the extrapolation rules `Throw`, `Flat`, `Line` and `Natural`, which decide what an
interpolant returns outside its grid. Passed as `extrap` to `convolution_interpolation`, either
as an instance (`Line()`) or as the matching symbol (`:line`).
"""
abstract type AbstractExtrapolation end

"""
    Line()

Linear extrapolation: outside the grid, the interpolant continues along the slope it has at the
nearest boundary point, measured inward from that point. Symbol form `:line`.
"""
struct Line <: AbstractExtrapolation end

"""
    Flat()

Constant extrapolation: outside the grid, the interpolant returns its value at the nearest
boundary point. Symbol form `:flat`.
"""
struct Flat <: AbstractExtrapolation end

"""
    Throw()

No extrapolation: evaluating outside the grid is an error. The default. Symbol form `:throw`.
"""
struct Throw <: AbstractExtrapolation end

"""
    Natural()

Natural extrapolation: the kernel's own polynomial boundary extension is applied twice, so that
the interpolant is defined on a domain one ghost region wider than the grid and evaluated there
as an ordinary interpolant. Smoothest of the rules, uniform grids only, and the one that costs
most in many dimensions. Symbol form `:natural`.
"""
struct Natural <: AbstractExtrapolation end