function (itp::ConvolutionInterpolation{T,N,NI,TCoefs,Axs,KA,HigherDimension{N},
                    DG,EQ,KBC,MixedIntegralOrder{DO},FD,SD,Val{:not_used},NB,Val{false},DI})(x::Vararg{Number,N}) where
                    {T<:AbstractFloat,N,NI,TCoefs<:AbstractArray{T,N},Axs<:Tuple{Vararg{AbstractVector}},
                    KA<:Tuple{Vararg{ConvolutionKernel}},DG,EQ<:Tuple{Vararg{Int}},
                    KBC<:Tuple{Vararg{Tuple{Symbol,Symbol}}},DO,FD,SD,DI,NB<:Nothing}
 
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