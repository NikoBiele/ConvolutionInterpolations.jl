"""
    FastConvolutionInterpolation{T,N,NI,TCoefs<:AbstractArray{T,N},
                                 Axs<:Tuple,KA,DT,DG,EQ,KBC,DOT,FD,SD,SG,LZ,DI,SZ} <:
        AbstractConvolutionInterpolation{T,N,NI,TCoefs,Axs,KA,DT,DG,EQ,KBC,DOT,FD,SD,SG,LZ,DI,SZ}

Convolution interpolation on uniform grids with O(1) evaluation. Kernel weights are computed
exactly from each kernel's polynomial pieces (see `_column_rows`), so evaluation is exact to
rounding for every kernel, derivative and integral order.

# Type Parameters
- `T`: Element type of the interpolated values
- `N`: Number of dimensions
- `NI`: Number of integral dimensions (derivative order −1)
- `TCoefs`: Type of the coefficient array
- `Axs`: Type of the knots (typically a tuple of ranges)
- `KA`: Kernel type per dimension (`nothing` for polynomial kernels)
- `DT`: Dimension type for dispatch (`Val{N}` for N ≤ 3, `HigherDimension{N}` otherwise)
- `DG`: Kernel symbol type for dispatch (e.g. `HigherOrderKernel{(:b5, :b5)}`)
- `EQ`: Kernel stencil half-widths per dimension
- `KBC`: Boundary condition type
- `DOT`: Derivative order type
- `FD`, `SD`: Unused placeholders (`Nothing`), shared with `ConvolutionInterpolation`
- `SG`: Unused placeholder (`Val{:not_used}`), shared with `ConvolutionInterpolation`
- `LZ`: Lazy mode flag (`Val{true}` or `Val{false}`)
- `DI`: Integral dimension type for dispatch
- `SZ`: Type of the data domain size

# Fields
- `coefs::TCoefs`: Coefficients, extended with ghost points (eager mode) or raw values (lazy mode)
- `domain_size::SZ`: Size of the original data array
- `knots::Axs`: Knots of each dimension, extended with ghost knots in eager mode
- `h::NTuple{N,T}`: Grid spacing per dimension
- `x0::NTuple{N,T}`: First (extended) knot per dimension
- `kernel::KA`: Kernel type per dimension
- `dimension::DT`: Dimension type for dispatch
- `kernel_sym::DG`: Kernel symbols for dispatch
- `eqs::EQ`: Kernel stencil half-widths per dimension
- `bc::KBC`: Boundary conditions per dimension
- `derivative_order::DOT`: Derivative orders per dimension
- `kernel_d1_pre::FD`, `kernel_d2_pre::SD`: Unused placeholders (`nothing`)
- `subgrid::SG`: Unused placeholder (`Val(:not_used)`)
- `lazy::LZ`: If `Val(true)`, ghost points are computed on the fly near the boundaries
- `boundary_fallback::Bool`: In lazy mode, use linear interpolation near the boundaries
  instead of computing ghost points
- `left_values::NTuple{N,Vector{T}}`: Antiderivative kernel at the anchor, per coefficient,
  for integral dimensions
- `anchor::NTuple{N,T}`: Point where antiderivatives are zero, per integral dimension
- `dim_integral::DI`: Integral dimension type for dispatch
- `lazy_workspace::LazyBoundaryWorkspace{T,N}`: Scratch buffers for lazy boundary evaluation
- `tail1_left`, `tail1_right`, `tail2_*`, `tail3_*`: Prefix sums of the saturated
  antiderivative contributions outside the stencil, for O(1) integral evaluation
"""

struct LazyBoundaryWorkspace{T,N}
    ghost_buf::Vector{T}      # length 9, scratch for mul!
    stencil_buf::Array{T,N}   # size (2*eqs, 2*eqs, ...) for ND stencil
end

LazyBoundaryWorkspace(T::Type, ::Val{N}, eqs::Int) where N = 
    LazyBoundaryWorkspace{T,N}(zeros(T, 9), zeros(T, ntuple(_ -> 2*eqs, N)...))

struct FastConvolutionInterpolation{T,N,NI,TCoefs<:AbstractArray{T,N},
                                Axs<:Tuple,KA,DT,DG,EQ,KBC,DOT,FD,SD,SG,LZ,DI,SZ} <:
                                AbstractConvolutionInterpolation{T,N,NI,TCoefs,Axs,KA,DT,DG,EQ,KBC,DOT,FD,SD,SG,LZ,DI,SZ}
    coefs::TCoefs
    domain_size::SZ
    knots::Axs
    h::NTuple{N,T}
    x0::NTuple{N,T}
    kernel::KA
    dimension::DT
    kernel_sym::DG
    eqs::EQ
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
    lazy_workspace::LazyBoundaryWorkspace{T,N}
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