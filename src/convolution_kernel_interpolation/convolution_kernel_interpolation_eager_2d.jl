@inline function (itp::ConvolutionInterpolation{T,2,0,TCoefs,Axs,KA,Val{2},
                    DG,EQ,KBC,DerivativeOrder{DO},FD,SD,Val{:not_used},NB,Val{false},Val{0}})(x::Vararg{Number,2}) where 
                    {T<:AbstractFloat,TCoefs<:AbstractArray{T,2},
                    Axs<:Tuple{<:AbstractVector,<:AbstractVector},
                    KA<:Tuple{<:ConvolutionKernel,<:ConvolutionKernel},
                    DG,EQ<:Tuple{Int,Int},KBC<:Tuple{<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol}},
                    DO,FD,SD,NB<:Nothing}

    i_float = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T)
    i = clamp(floor(Int, i_float), itp.eqs[1], length(itp.knots[1]) - itp.eqs[1])

    j_float = (x[2] - itp.knots[2][1]) / itp.h[2] + one(T)
    j = clamp(floor(Int, j_float), itp.eqs[2], length(itp.knots[2]) - itp.eqs[2])

    result = zero(T)
    @inbounds for l = -(itp.eqs[1]-1):itp.eqs[1]
        kx = itp.kernel[1]((x[1] - itp.knots[1][i+l]) / itp.h[1])
        for m = -(itp.eqs[2]-1):itp.eqs[2]
            result += itp.coefs[i+l, j+m] * kx * itp.kernel[2]((x[2] - itp.knots[2][j+m]) / itp.h[2])
        end
    end
    scale = prod(d -> one(T) / itp.h[d]^DO[d], 1:2)
    return @fastmath result * scale
end