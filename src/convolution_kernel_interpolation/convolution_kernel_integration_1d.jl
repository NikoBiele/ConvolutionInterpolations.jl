# integral order m of a kernel carrying differentiation order −m
_integral_order(::ConvolutionKernel{K,DO}) where {K,DO} = -DO

@inline function (itp::ConvolutionInterpolation{T,1,1,TCoefs,Axs,KA,Val{1},
                    K,EQ,KBC,IntegralOrder,NB,Val{false},Val{1}})(x::Vararg{Number,1}) where 
                    {T<:AbstractFloat,TCoefs<:AbstractArray{T,1},Axs<:Tuple{<:AbstractVector},
                    KA<:Tuple{<:ConvolutionKernel},K,EQ<:Tuple{Int},
                    KBC<:Tuple{<:Tuple{Symbol,Symbol}},NB<:Nothing}

    x = T.(x)
    result = zero(T)
    @inbounds for j in 1:length(itp.coefs)
        xj = itp.knots[1][itp.eqs[1]] + (j - itp.eqs[1]) * itp.h[1]
        s_x = (x[1]          - xj) / itp.h[1]
        s_a = (itp.anchor[1] - xj) / itp.h[1]
        result += itp.coefs[j] * _direct_antiderivative_weight(itp.kernel[1], s_x, s_a)
    end
    return result * itp.h[1]^_integral_order(itp.kernel[1])
end