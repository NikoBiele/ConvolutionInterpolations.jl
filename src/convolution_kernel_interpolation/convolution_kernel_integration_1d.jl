@inline function (itp::ConvolutionInterpolation{T,1,1,TCoefs,Axs,KA,Val{1},
                    DG,EQ,KBC,IntegralOrder,FD,SD,Val{:not_used},NB,Val{false},Val{1}})(x::Vararg{Number,1}) where 
                    {T<:AbstractFloat,TCoefs<:AbstractArray{T,1},Axs<:Tuple{<:AbstractVector},
                    KA<:Tuple{<:ConvolutionKernel},DG,EQ<:Tuple{Int},
                    KBC<:Tuple{<:Tuple{Symbol,Symbol}},FD,SD,NB<:Nothing}

    result = zero(T)
    @inbounds for j in 1:length(itp.coefs)
        xj = itp.knots[1][itp.eqs[1]] + (j - itp.eqs[1]) * itp.h[1]
        kt_x    = itp.kernel[1]((x[1]       - xj) / itp.h[1])
        kt_left = itp.kernel[1]((itp.anchor[1] - xj) / itp.h[1])
        result += itp.coefs[j] * (kt_x - kt_left)
    end
    return result * itp.h[1]
end