"""
    FastConvolutionInterpolation{T,N,TCoefs<:AbstractArray,IT<:NTuple{N,ConvolutionMethod},
                                Axs<:Tuple,KA,DT,DG,EQ,PR,KP,KBC,DO,FD,SD,SG} <: 
                                AbstractConvolutionInterpolation{T,N,TCoefs,IT,Axs,KA,DT,DG,EQ,KBC,DO,FD,SD,SG}

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
- `KBC`: The kernel boundary condition type
- `DO`: The derivative order type
- `FD`: The first derivative type
- `SD`: The second derivative type
- `SG`: The subgrid type

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
- `kernel_bc::KBC`: The kernel boundary condition
- `derivative_order::DO`: The derivative order
- `kernel_d1_pre::FD`: The precomputed first derivative
- `kernel_d2_pre::SD`: The precomputed second derivative
- `subgrid::SG`: The subgrid
- `lazy::Bool`: If `true`, no ghost points are computed, and are instead computed on the fly.
- `boundary_fallback::Bool`: When `true`, throws an error instead of computing ghost
  points when evaluating near boundaries in lazy mode.

This implementation uses precomputed kernel values,
offering significantly faster performance than `ConvolutionInterpolation`.
"""
struct FastConvolutionInterpolation{T,N,TCoefs<:AbstractArray,IT<:NTuple{N,ConvolutionMethod},
                                Axs<:Tuple,KA,DT,DG,EQ,PR,KP,KBC,DOT,FD,SD,SG} <: 
                                AbstractConvolutionInterpolation{T,N,TCoefs,IT,Axs,KA,DT,DG,EQ,KBC,DOT,FD,SD,SG}
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
    derivative_order::DOT
    kernel_d1_pre::FD
    kernel_d2_pre::SD
    subgrid::SG
    lazy::Bool
    boundary_fallback::Bool
    left_values::NTuple{N, Vector{T}}
    anchor::NTuple{N, T}
    cumcoefs_left::NTuple{N, Array{T}}
    cumcoefs_right::NTuple{N, Array{T}}
    cross_ll::NTuple{N, Array{T}}
    cross_rl::NTuple{N, Array{T}}
    cross_lr::NTuple{N, Array{T}}
    cross_rr::NTuple{N, Array{T}}
end