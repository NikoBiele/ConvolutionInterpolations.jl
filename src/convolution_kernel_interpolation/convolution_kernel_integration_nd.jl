@inline function (itp::ConvolutionInterpolation{T,N,NI,TCoefs,Axs,KA,HigherDimension{N},
                    DG,EQ,KBC,IntegralOrder,FD,SD,Val{:not_used},NB,Val{false},
                    HigherDimension{N}})(x::Vararg{Number,N}) where 
                    {T<:AbstractFloat,N,NI,TCoefs<:AbstractArray{T,N},KA<:NTuple{N,ConvolutionKernel},
                    Axs<:NTuple{N,<:AbstractVector},DG,EQ<:NTuple{N,Int},
                    KBC<:NTuple{N,Tuple{Symbol,Symbol}},FD,SD,NB<:Nothing}

    result = zero(T)
    ns = ntuple(d -> size(itp.coefs, d), N)
    @inbounds for idx in Iterators.product(ntuple(d -> 1:ns[d], N)...)
        dx = prod(d -> begin
            eqs_d = _eqs_d(itp.eqs, d)
            xd = itp.knots[d][eqs_d] + (idx[d] - eqs_d) * itp.h[d]
            _kernel_d(itp.kernel, d)((x[d]          - xd) / itp.h[d]) -
            _kernel_d(itp.kernel, d)((itp.anchor[d] - xd) / itp.h[d])
        end, 1:N)
        result += itp.coefs[idx...] * dx
    end
    return result * prod(itp.h[d] for d in 1:N)
end