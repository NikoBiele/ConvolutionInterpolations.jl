@inline function (itp::ConvolutionInterpolation{T,2,NI,TCoefs,Axs,KA,Val{2},
                    DG,EQ,KBC,MixedIntegralOrder{DO},FD,SD,Val{:not_used},NB,Val{false},Val{NI}})(x::Vararg{Number,2}) where
                    {T<:AbstractFloat,NI,TCoefs<:AbstractArray{T,2},Axs<:Tuple{<:AbstractVector,<:AbstractVector},
                    KA<:Tuple{<:ConvolutionKernel,<:ConvolutionKernel},
                    DG,EQ<:Tuple{Int,Int},KBC<:Tuple{<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol}},DO,FD,SD,NB<:Nothing}
 
    n1 = size(itp.coefs, 1)
    n2 = size(itp.coefs, 2)
 
    # For derivative dims, find the local stencil centre
    if DO[1] != -1
        i_float = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T)
        i = clamp(floor(Int, i_float), itp.eqs[1], length(itp.knots[1]) - itp.eqs[1])
    end
    if DO[2] != -1
        j_float = (x[2] - itp.knots[2][1]) / itp.h[2] + one(T)
        j = clamp(floor(Int, j_float), itp.eqs[2], length(itp.knots[2]) - itp.eqs[2])
    end
 
    result = zero(T)
 
    # Outer loop: all coefs if integrating dim 1, stencil if differentiating
    i1_range = DO[1] == -1 ? (1:n1) : ((i - (itp.eqs[1]-1)):(i + itp.eqs[1]))
    i2_range = DO[2] == -1 ? (1:n2) : ((j - (itp.eqs[2]-1)):(j + itp.eqs[2]))
 
    @inbounds for i1 in i1_range
        if DO[1] == -1
            xi = itp.knots[1][itp.eqs[1]] + (i1 - itp.eqs[1]) * itp.h[1]
            kx = itp.kernel[1]((x[1]          - xi) / itp.h[1]) -
                 itp.kernel[1]((itp.anchor[1] - xi) / itp.h[1])
        else
            kx = itp.kernel[1]((x[1] - itp.knots[1][i1]) / itp.h[1])
        end
        for i2 in i2_range
            if DO[2] == -1
                yj = itp.knots[2][itp.eqs[2]] + (i2 - itp.eqs[2]) * itp.h[2]
                ky = itp.kernel[2]((x[2]          - yj) / itp.h[2]) -
                     itp.kernel[2]((itp.anchor[2] - yj) / itp.h[2])
            else
                ky = itp.kernel[2]((x[2] - itp.knots[2][i2]) / itp.h[2])
            end
            result += itp.coefs[i1, i2] * kx * ky
        end
    end
 
    scale = one(T)
    scale *= DO[1] == -1 ? itp.h[1] : one(T) / itp.h[1]^DO[1]
    scale *= DO[2] == -1 ? itp.h[2] : one(T) / itp.h[2]^DO[2]
    return @fastmath result * scale
end