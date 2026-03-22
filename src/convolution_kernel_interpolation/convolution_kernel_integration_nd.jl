@inline function (itp::ConvolutionInterpolation{T,N,TCoefs,IT,Axs,KA,HigherDimension{N},
                    DG,EQ,KBC,IntegralOrder,FD,SD,SG,NB,Val{false}})(x::Vararg{Number,N}) where 
                    {T,N,TCoefs,IT,KA<:NTuple{N,<:ConvolutionKernel},Axs,DG,EQ<:NTuple{N,Int},KBC,FD,SD,SG,NB<:Nothing}

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

function (itp::ConvolutionInterpolation{T,N,TCoefs,IT,Axs,KA,HigherDimension{N},
                    DG,EQ,KBC,MixedIntegralOrder{DO},FD,SD,SG})(x::Vararg{Number,N}) where
                    {T,N,TCoefs,IT,Axs,KA<:NTuple{N,<:ConvolutionKernel},DG,EQ<:NTuple{N,Int},KBC,DO,FD,SD,SG}
 
    ns = ntuple(d -> size(itp.coefs, d), N)
 
    ids = ntuple(N) do d
        DO[d] != -1 ? clamp(floor(Int, (x[d] - itp.knots[d][1]) / itp.h[d] + one(T)),
                            itp.eqs[d], length(itp.knots[d]) - itp.eqs[d]) : 0
    end
 
    ranges = ntuple(N) do d
        DO[d] == -1 ? (1:ns[d]) : ((ids[d] - (itp.eqs[d]-1)):(ids[d] + itp.eqs[d]))
    end
 
    result = zero(T)
    @inbounds for idx in Iterators.product(ranges...)
        kval = one(T)
        for d in 1:N
            i_d = idx[d]
            if DO[d] == -1
                xj = itp.knots[d][itp.eqs[d]] + (i_d - itp.eqs[d]) * itp.h[d]
                kval *= itp.kernel[d]((x[d]          - xj) / itp.h[d]) -
                        itp.kernel[d]((itp.anchor[d] - xj) / itp.h[d])
            else
                kval *= itp.kernel[d]((x[d] - itp.knots[d][i_d]) / itp.h[d])
            end
        end
        result += itp.coefs[idx...] * kval
    end
 
    scale = one(T)
    for d in 1:N
        scale *= DO[d] == -1 ? itp.h[d] : one(T) / itp.h[d]^DO[d]
    end
    return @fastmath result * scale
end