"""
    FastConvolutionInterpolation{T,N,TCoefs<:AbstractArray,IT<:NTuple{N,ConvolutionMethod},
                                Axs<:Tuple,KA,DT,DG,EQ,PR,KP} <: AbstractConvolutionInterpolation{T,N,TCoefs,IT,Axs,KA,DT,DG,EQ,KBC}

A structure that implements convolution-based interpolation with precomputed kernel values for speed.

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
- `PR`: The type of the precomputed range
- `KP`: The type of the precomputed kernel values

# Fields
- `coefs::TCoefs`: The coefficient array with boundary extensions
- `knots::Axs`: The knot points for each dimension
- `it::IT`: The tuple of interpolation methods
- `h::NTuple{N,Float64}`: The step size in each dimension
- `kernel::KA`: The convolution kernel
- `dimension::DT`: Type for dimension-specific dispatch
- `deg::DG`: The degree of the interpolation
- `eqs::EQ`: The number of equations used in the boundary conditions
- `pre_range::PR`: The precomputed range of positions for kernel evaluation
- `kernel_pre::KP`: The precomputed kernel values

This implementation uses precomputed kernel values for commonly used positions,
offering significantly faster performance than `ConvolutionInterpolation` at the cost
of some additional memory usage.
"""
struct FastConvolutionInterpolation{T,N,TCoefs<:AbstractArray,IT<:NTuple{N,ConvolutionMethod},
                                Axs<:Tuple,KA,DT,DG,EQ,PR,KP,KBC,DO,FD,SD,SG} <: 
                                AbstractConvolutionInterpolation{T,N,TCoefs,IT,Axs,KA,DT,DG,EQ,KBC,DO,FD,SD,SG}
    coefs::TCoefs
    knots::Axs
    it::IT
    h::NTuple{N,T}
    kernel::KA
    dimension::DT
    deg::DG
    eqs::EQ
    pre_range::PR
    kernel_pre::KP
    kernel_bc::KBC
    derivative_order::DO
    kernel_d1_pre::FD
    kernel_d2_pre::SD
    subgrid::SG
end