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
- `tail1_left`, `tail2_ll`, `tail3_edge_ll`, `tail3_corner_lll`: Prefix sums of the
  antiderivative contributions left of the stencil, for O(1) integral evaluation. Coefficients
  right of the stencil contribute exactly zero, so there are no right tails.
- `anchor_taylor::Matrix{T}`: For a 1D integral of order M ≥ 2, the exact values
  K_{M−r}(eqs − j) of the coefficients near the anchor (r = 0 … M−1), which anchor the M-fold
  integral and all lower integrals at zero
- `tail_polynomial::Vector{Array{T,N}}`: For a 1D integral of order M ≥ 2, the left tail as
  a polynomial in the position within the cell: array k holds the coefficient of t^(k−1)
- `integral_taylor::NTuple{N,Matrix{T}}`: For each integral dimension of order m, the exact values
  K_{m−r}(eqs − j) of the coefficients near the anchor (see `_anchor_taylor_table`)
- `integral_entries::NTuple{N,Matrix{T}}`: For each integral dimension, the exact polynomials
  (in the position within the cell) that near-anchor coefficients contribute left of the stencil
- `integral_tails::Vector{Vector{Array{T,N}}}`: For at most 3 integral dimensions, the left tails
  of every region: entry `mask` holds one array per combination of powers of the positions within
  the cell (see `_build_region_tails`)
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
    # integral tails: prefix sums over the coefficients left of the stencil (coefficients right
    # of it contribute exactly zero, so no right tails exist)
    tail1_left::NTuple{N, Array{T,N}}      # left-saturated in one integral dimension
    tail2_ll::Array{T,N}                   # left-saturated in both of 2 integral dimensions
    tail3_edge_ll::NTuple{3, Array{T,N}}   # 3 integral dims: free in dim k, left-saturated in the other two
    tail3_corner_lll::Array{T,N}           # 3 integral dims: left-saturated in all three
    # integrals of any order in any dimension (see FastIntegralOrders)
    integral_taylor::NTuple{N, Matrix{T}}      # per integral dimension: exact anchoring table
    integral_entries::NTuple{N, Matrix{T}}     # per integral dimension: near-anchor entry polynomials
    integral_tails::Vector{Vector{Array{T,N}}} # left tails of every region (≤ 3 integral dimensions)
end