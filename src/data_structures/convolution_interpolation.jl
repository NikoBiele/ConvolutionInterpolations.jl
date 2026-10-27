"""
    ConvolutionInterpolation{T,N,NI,TCoefs<:AbstractArray{T,N},
                             Axs<:Tuple,KA,DT,DG,EQ,KBC,DO,NB,LZ,DI,SZ} <:
        AbstractConvolutionInterpolation{T,N,NI,TCoefs,Axs,KA,DT,DG,EQ,KBC,DO,LZ,DI,SZ}

Convolution interpolation by direct kernel evaluation: the kernel is evaluated at each point of
the stencil, which is what makes nonuniform grids, the Gaussian kernel and per-dimension kernels
possible. Uniform grids with polynomial kernels use `FastConvolutionInterpolation` instead.

# Type Parameters
- `T`: Element type of the interpolated values
- `N`: Number of dimensions
- `NI`: Number of integral dimensions (derivative order −1)
- `TCoefs`: Type of the coefficient array
- `Axs`: Type of the knots (a tuple of ranges or vectors)
- `KA`: Kernel type per dimension
- `DT`: Dimension type for dispatch (`Val{N}` for N ≤ 3, `HigherDimension{N}` otherwise)
- `DG`: Kernel symbol type for dispatch (e.g. `Val{(:b5, :n3)}`)
- `EQ`: Kernel stencil half-widths per dimension
- `KBC`: Boundary condition type
- `DO`: Derivative order type
- `NB`: Type of the nonuniform `b`-kernel weight coefficients (`Nothing` when not used)
- `LZ`: Lazy mode flag (`Val{true}` or `Val{false}`)
- `DI`: Integral dimension type for dispatch
- `SZ`: Type of the data domain size

# Fields
- `coefs::TCoefs`: Coefficients, extended with ghost points (eager mode) or raw values (lazy mode)
- `domain_size::SZ`: Size of the original data array
- `knots::Axs`: Knots of each dimension, extended with ghost knots in eager mode
- `h::NTuple{N,T}`: Grid spacing per dimension (one on nonuniform dimensions)
- `kernel::KA`: Kernel type per dimension
- `dimension::DT`: Dimension type for dispatch
- `kernel_sym::DG`: Kernel symbols for dispatch
- `eqs::EQ`: Kernel stencil half-widths per dimension
- `bc::KBC`: Boundary conditions per dimension
- `derivative_order::DO`: Derivative orders per dimension
- `nb_weight_coeffs::NB`: For nonuniform `b` kernels, the per-interval polynomial weight
  coefficients of each dimension, computed exactly at construction; `nothing` otherwise
- `lazy::LZ`: If `Val(true)`, ghost points are computed on the fly near the boundaries
- `anchor::NTuple{N,T}`: Point where antiderivatives are zero, per integral dimension
- `dim_integral::DI`: Integral dimension type for dispatch
"""
struct ConvolutionInterpolation{T,N,NI,TCoefs<:AbstractArray{T,N},
                                Axs<:Tuple,KA,DT,DG,EQ,KBC,DO,NB,LZ,DI,SZ} <: 
                                AbstractConvolutionInterpolation{T,N,NI,TCoefs,
                                Axs,KA,DT,DG,EQ,KBC,DO,LZ,DI,SZ}
    coefs::TCoefs
    domain_size::SZ
    knots::Axs
    h::NTuple{N,T}
    kernel::KA
    dimension::DT
    kernel_sym::DG
    eqs::EQ
    bc::KBC
    derivative_order::DO
    nb_weight_coeffs::NB
    lazy::LZ
    anchor::NTuple{N, T}
    dim_integral::DI
end