"""
    AbstractConvolutionInterpolation{T,N,NI,TCoefs,Axs,KA,DT,DG,EQ,KBC,DO,LZ,DI,SZ}

Abstract supertype of the two interpolation types, `ConvolutionInterpolation` (direct kernel
evaluation) and `FastConvolutionInterpolation` (exact polynomial weights on uniform grids).

# Type Parameters
- `T`: Element type of the interpolated values
- `N`: Number of dimensions
- `NI`: Number of integral dimensions (derivative order −1)
- `TCoefs`: Type of the coefficient array
- `Axs`: Type of the knots (a tuple of ranges or vectors)
- `KA`: Kernel type per dimension
- `DT`: Dimension type for dispatch (`Val{N}` for N ≤ 3, `HigherDimension{N}` otherwise)
- `DG`: Kernel symbol type for dispatch
- `EQ`: Kernel stencil half-widths per dimension
- `KBC`: Boundary condition type
- `DO`: Derivative order type
- `LZ`: Lazy mode flag (`Val{true}` or `Val{false}`)
- `DI`: Integral dimension type for dispatch
- `SZ`: Type of the data domain size
"""
abstract type AbstractConvolutionInterpolation{T,N,NI,TCoefs,Axs,KA,DT,DG,EQ,KBC,DO,LZ,DI,SZ} end