"""
    ConvolutionInterpolation{T,N,TCoefs<:AbstractArray,IT<:NTuple{N,ConvolutionMethod},
                             Axs<:Tuple,KA,DT,DG,EQ,KBC} <: AbstractConvolutionInterpolation{T,N,TCoefs,IT,Axs,KA,DT,DG,EQ,KBC}

A structure that implements convolution-based interpolation on N-dimensional data.

# Type Parameters
- `T`: The element type of the interpolated values
- `N`: The number of dimensions
- `TCoefs`: The type of the coefficient array
- `IT`: The tuple of interpolation methods for each dimension
- `Axs`: The type of the knot points (typically a tuple of ranges)
- `KA`: The kernel type (ConvolutionKernel{DG} or GaussianConvolutionKernel{B})
- `DT`: The dimension type (Val{N} for N≤3, HigherDimension{N} otherwise)
- `DG`: The degree type (Val{degree})
- `EQ`: The equation order type

# Fields
- `coefs::TCoefs`: The coefficient array with boundary extensions
- `knots::Axs`: The knot points for each dimension
- `it::IT`: The tuple of interpolation methods
- `h::NTuple{N,Float64}`: The step size in each dimension
- `kernel::KA`: The convolution kernel
- `dimension::DT`: Type for dimension-specific dispatch
- `deg::DG`: The degree of the interpolation
- `eqs::EQ`: The number of equations used in the boundary conditions

This implementation evaluates the convolution kernel at each point directly,
providing accurate results but potentially slower performance than `FastConvolutionInterpolation`.
"""
struct ConvolutionInterpolation{T,N,TCoefs<:AbstractArray,IT<:NTuple{N,ConvolutionMethod},
                                Axs<:Tuple,KA,DT,DG,EQ,KBC} <: AbstractConvolutionInterpolation{T,N,TCoefs,IT,Axs,KA,DT,DG,EQ,KBC}
    coefs::TCoefs
    knots::Axs
    it::IT
    h::NTuple{N,T}
    kernel::KA
    dimension::DT
    deg::DG
    eqs::EQ
    kernel_bc::KBC
end