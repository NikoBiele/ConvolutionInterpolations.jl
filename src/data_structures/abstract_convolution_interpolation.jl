"""
    AbstractConvolutionInterpolation{T,N,TCoefs,IT<:Union{Tuple{Vararg{ConvolutionMethod}},ConvolutionMethod},Axs,KA,DT,DG,EQ}

Abstract supertype for all convolution-based interpolation types.

# Type Parameters
- `T`: The element type of the interpolated values
- `N`: The number of dimensions
- `TCoefs`: The type of the coefficient array
- `IT`: The interpolation type (or tuple of types for each dimension)
- `Axs`: The type of the knot points (typically a tuple of ranges)
- `KA`: The kernel type
- `DT`: The dimension type (Val{N} for N≤3, HigherDimension{N} otherwise)
- `DG`: The degree type (Val{degree})
- `EQ`: The equation order type

This abstract type serves as the parent type for the concrete implementations
`ConvolutionInterpolation` and `FastConvolutionInterpolation`.
"""
abstract type AbstractConvolutionInterpolation{T,N,TCoefs,IT<:Union{Tuple{Vararg{ConvolutionMethod}},ConvolutionMethod},Axs,KA,DT,DG,EQ} end