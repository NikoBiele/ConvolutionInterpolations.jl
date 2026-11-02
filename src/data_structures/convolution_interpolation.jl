"""
    ConvolutionInterpolation{T,N,NI,TCoefs<:AbstractArray{T,N},
                                 Axs<:Tuple,KA,DT,K,EQ,KBC,DOT,LZ,DI,SZ,NU} <:
        AbstractConvolutionInterpolation{T,N,NI,TCoefs,Axs,KA,DT,K,EQ,KBC,DOT,LZ,DI,SZ}

Convolution interpolation with O(1) evaluation. On uniform axes the kernel weights are
computed exactly from each kernel's polynomial pieces (see `_column_rows`), so evaluation is
exact to rounding for every kernel, derivative and integral order. On nonuniform axes the
weights of the `b` kernels come from per-interval polynomial tables computed at construction
(see `precompute_nonuniform_b_all`), and the closed-form kernels (`:n3`, `:a0`, `:a1`) adapt
their weights to the local spacing at evaluation.

# Type Parameters
- `T`: Element type of the interpolated values
- `N`: Number of dimensions
- `NI`: Number of integral dimensions (derivative order -1)
- `TCoefs`: Type of the coefficient array
- `Axs`: Type of the knots (typically a tuple of ranges)
- `KA`: Kernel type per dimension (`nothing` for polynomial kernels)
- `DT`: Dimension type for dispatch (`Val{N}` for N ≤ 3, `HigherDimension{N}` otherwise)
- `K`: Kernel symbol type for dispatch (e.g. `HigherOrderKernel{(:b5, :b5)}`)
- `EQ`: Kernel stencil half-widths per dimension
- `KBC`: Boundary condition type
- `DOT`: Derivative order type
- `LZ`: Lazy mode flag (`Val{true}` or `Val{false}`)
- `DI`: Integral dimension type for dispatch
- `SZ`: Type of the data domain size
- `NU`: Type of the per-axis nonuniform data (a tuple of `Nothing` on a uniform grid)

# Fields
- `coefs::TCoefs`: Coefficients, extended with ghost points (eager mode) or raw values (lazy mode)
- `domain_size::SZ`: Size of the original data array
- `knots::Axs`: Knots of each dimension, extended with ghost knots in eager mode
- `h::NTuple{N,T}`: Grid spacing per dimension (one on nonuniform dimensions)
- `x0::NTuple{N,T}`: First (extended) knot per dimension
- `kernel::KA`: Kernel type per dimension
- `dimension::DT`: Dimension type for dispatch
- `kernel_sym::K`: Kernel symbols for dispatch
- `eqs::EQ`: Kernel stencil half-widths per dimension
- `bc::KBC`: Boundary conditions per dimension
- `derivative_order::DOT`: Derivative orders per dimension
- `lazy::LZ`: If `Val(true)`, ghost points are computed on the fly near the boundaries
- `anchor::NTuple{N,T}`: Point where antiderivatives are zero, per integral dimension
- `dim_integral::DI`: Integral dimension type for dispatch
- `integral_taylor::NTuple{N,Matrix{T}}`: For each integral dimension of order m, the exact values
  K_{m-r}(eqs - j) of the coefficients near the anchor (see `_anchor_taylor_table`)
- `integral_entries::NTuple{N,Matrix{T}}`: For each integral dimension, the exact polynomials
  (in the position within the cell) that near-anchor coefficients contribute left of the stencil
- `integral_tails::Vector{Vector{Array{T,N}}}`: For at most 3 integral dimensions, the left tails
  of every region: entry `mask` holds one array per combination of powers of the positions within
  the cell (see `_build_region_tails`)
- `nonuniform::NU`: Per axis, `nothing` on a uniform axis; on a nonuniform axis the per-interval
  weight tables of a `b` kernel (`Vector{Matrix{T}}`), or `Val(:nonuniform)` for a kernel whose
  weights are computed from the local spacing at evaluation (`:n3`, `:a0`, `:a1`)
"""
struct ConvolutionInterpolation{T,N,NI,TCoefs<:AbstractArray{T,N},
                                Axs<:Tuple,KA,DT,K,EQ,KBC,DOT,LZ,DI,SZ,NU} <:
                                AbstractConvolutionInterpolation{T,N,NI,TCoefs,Axs,KA,DT,K,EQ,KBC,DOT,LZ,DI,SZ}
    coefs::TCoefs
    domain_size::SZ
    knots::Axs
    h::NTuple{N,T}
    x0::NTuple{N,T}
    kernel::KA
    dimension::DT
    kernel_sym::K
    eqs::EQ
    bc::KBC
    derivative_order::DOT
    lazy::LZ
    anchor::NTuple{N, T}
    dim_integral::DI
    # integrals of any order in any dimension (see FastIntegralOrders)
    integral_taylor::NTuple{N, Matrix{T}}      # per integral dimension: exact anchoring table
    integral_entries::NTuple{N, Matrix{T}}     # per integral dimension: near-anchor entry polynomials
    integral_tails::Vector{Vector{Array{T,N}}} # left tails of every region (≤ 3 integral dimensions)
    # nonuniform axes (see the docstring): nothing on a uniform axis
    nonuniform::NU
end