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
- `DG`: The kernel type (Val{degree})
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
struct FastConvolutionInterpolation{T,N,NI,TCoefs<:AbstractArray,
                                Axs<:Tuple,KA,DT,DG,EQ,PR,KP,KBC,DOT,FD,SD,SG,LZ,DI} <: 
                                AbstractConvolutionInterpolation{T,N,NI,TCoefs,Axs,KA,DT,DG,EQ,KBC,DOT,FD,SD,SG,LZ,DI}
    coefs::TCoefs
    knots::Axs
    h::NTuple{N,T}
    kernel::KA
    dimension::DT
    kernel_sym::DG
    eqs::EQ
    pre_range::PR
    kernel_pre::KP
    bc::KBC
    derivative_order::DOT
    kernel_d1_pre::FD
    kernel_d2_pre::SD
    subgrid::SG
    lazy::LZ
    boundary_fallback::Bool
    left_values::NTuple{N, Vector{T}}
    anchor::NTuple{N, T}
    dim_integral::DI
    # 1d and 2d integral tails
    tail1_left::NTuple{N, Array{T,N}}
    tail1_right::NTuple{N, Array{T,N}}
    tail2_ll::Array{T,N}  # or NTuple based approach
    tail2_rl::Array{T,N}
    tail2_lr::Array{T,N}
    tail2_rr::Array{T,N}
    # 3d integral tails
    tail3_edge_ll::NTuple{3, Array{T,N}}  # free dim d, left×left in the other two
    tail3_edge_rl::NTuple{3, Array{T,N}}
    tail3_edge_lr::NTuple{3, Array{T,N}}
    tail3_edge_rr::NTuple{3, Array{T,N}}
    tail3_face_l::NTuple{3, Array{T,N}}   # tail3_face_l[d] = left-saturated in dim d
    tail3_face_r::NTuple{3, Array{T,N}}
    tail3_corner_lll::Array{T,N}
    tail3_corner_rll::Array{T,N}
    tail3_corner_lrl::Array{T,N}
    tail3_corner_llr::Array{T,N}
    tail3_corner_rrl::Array{T,N}
    tail3_corner_rlr::Array{T,N}
    tail3_corner_lrr::Array{T,N}
    tail3_corner_rrr::Array{T,N}
end