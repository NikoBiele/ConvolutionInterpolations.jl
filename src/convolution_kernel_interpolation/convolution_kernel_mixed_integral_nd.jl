function (itp::ConvolutionInterpolation{T,N,NI,TCoefs,Axs,KA,HigherDimension{N},
                    K,EQ,KBC,MixedIntegralOrder{DO},NB,Val{false},DI})(x::Vararg{Number,N}) where
                    {T<:AbstractFloat,N,NI,TCoefs<:AbstractArray{T,N},Axs<:NTuple{N,<:AbstractVector},
                    KA<:NTuple{N,ConvolutionKernel},K,EQ<:NTuple{N,Int},
                    KBC<:NTuple{N,Tuple{Symbol,Symbol}},DO,DI,NB<:Nothing}

    x = T.(x)
    ns = ntuple(d -> size(itp.coefs, d), N)
 
    ids = ntuple(N) do d
        if DO[d] >= 0
            clamp(floor(Int, (x[d] - itp.knots[d][1]) / itp.h[d] + one(T)),
                  itp.eqs[d], length(itp.knots[d]) - itp.eqs[d])
        else
            0  # unused for integral dims
        end
    end
 
    ranges = ntuple(N) do d
        if DO[d] < 0
            1:ns[d]
        else
            (ids[d] - (itp.eqs[d]-1)):(ids[d] + itp.eqs[d])
        end
    end
 
    result = zero(T)
    @inbounds for idx in Iterators.product(ranges...)
        kval = one(T)
        for d in 1:N
            i_d = idx[d]
            if DO[d] < 0
                xj = itp.knots[d][itp.eqs[d]] + (i_d - itp.eqs[d]) * itp.h[d]
                kval *= _direct_antiderivative_weight(itp.kernel[d],
                                                      (x[d]          - xj) / itp.h[d],
                                                      (itp.anchor[d] - xj) / itp.h[d])
            else
                kval *= itp.kernel[d]((x[d] - itp.knots[d][i_d]) / itp.h[d])
            end
        end
        result += itp.coefs[idx...] * kval
    end
 
    # h^m per integral axis, 1/h^k per derivative axis
    scale = one(T)
    for d in 1:N
        scale *= itp.h[d]^(-DO[d])
    end
    return @fastmath result * scale
end