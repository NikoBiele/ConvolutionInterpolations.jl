# Gaussian smoothing interpolants: the `kernel` field holds a `GaussianConvolutionKernel` per
# axis, whose weights are the kernel evaluated at the 2·eqs stencil knots (eqs = NT + 1 for a
# kernel with NT terms, a compile-time constant). The stencils of all axes are contracted with
# the coefficients in `_stencil_contract`, as for the polynomial kernels.

# Stencil of one axis: the index of the coefficient before it and the kernel weight of each entry
@inline function _gaussian_axis_stencil(kernel::GaussianConvolutionKernel{B,NT},
                                        knots::AbstractVector{T}, h::T, x::T) where {B,NT,T}
    eqs = NT + 1
    i_float = (x - knots[1]) / h + one(T)
    i = clamp(floor(Int, i_float), eqs, length(knots) - eqs)
    o = i - eqs
    w = ntuple(k -> kernel((x - knots[o + k]) / h), Val(2 * (NT + 1)))
    return o, w
end

# The stencils of every axis: offsets, weights and stencil lengths (the generated body may not
# contain closures, so the tuples are spelled out)
@generated function _gaussian_stencils(kernels::NTuple{N,GaussianConvolutionKernel}, knots::Tuple,
                                       h::NTuple{N,T}, x::NTuple{N,T}) where {N,T}
    calls = [:(_gaussian_axis_stencil(kernels[$d], knots[$d], h[$d], x[$d])) for d in 1:N]
    offsets = [:(st[$d][1]) for d in 1:N]
    weights = [:(st[$d][2]) for d in 1:N]
    lens = [:(length(st[$d][2])) for d in 1:N]
    return quote
        st = tuple($(calls...))
        return tuple($(offsets...)), tuple($(weights...)), tuple($(lens...))
    end
end

"""
    (itp::FastConvolutionInterpolation{T,N,...,KA<:NTuple{N,GaussianConvolutionKernel},...})(x...)

Evaluate a Gaussian smoothing interpolant: the weighted sum Σⱼ cⱼ·Π_d K((x_d − x_{j,d})/h_d)
over the 2·eqs nearest coefficients of every axis, with K the normalized Gaussian of the axis'
kernel. Allocation-free, O(eqs^N) per evaluation.
"""
function (itp::FastConvolutionInterpolation{T,N,0,TCoefs,Axs,KA,DT,K,EQ,KBC,
            DerivativeOrder{DO},Val{false},Val{0}})(x::Vararg{Number,N}) where
            {T<:AbstractFloat,N,TCoefs<:AbstractArray{T,N},Axs,KA<:NTuple{N,GaussianConvolutionKernel},
            DT,K,EQ,KBC,DO}
    x = T.(x)
    o, W, len = _gaussian_stencils(itp.kernel, itp.knots, itp.h, x)
    return _stencil_contract(itp.coefs, W, o, len)
end