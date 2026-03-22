@inline function (itp::ConvolutionInterpolation{T,1,TCoefs,IT,Axs,KA,Val{1},
                    DG,EQ,KBC,DerivativeOrder{DO},FD,SD,Val{:not_used},NB,Val{false}})(x::Vararg{Number,1}) where 
                    {T,TCoefs,IT,Axs,KA<:Tuple{<:ConvolutionKernel},DG,EQ<:Tuple{Int},KBC,DO,FD,SD,NB<:Nothing}

    i_float = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T)
    i = clamp(floor(Int, i_float), itp.eqs[1], length(itp.knots[1]) - itp.eqs[1])

    result = zero(T)
    @inbounds for l = -(itp.eqs[1]-1):itp.eqs[1]
        result += itp.coefs[i+l] * itp.kernel[1]((x[1] - itp.knots[1][i+l]) / itp.h[1])
    end
    scale = one(T) / itp.h[1]^DO[1]
    return @fastmath result * scale
end