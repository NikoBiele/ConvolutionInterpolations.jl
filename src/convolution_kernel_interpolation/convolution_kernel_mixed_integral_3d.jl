@inline function (itp::ConvolutionInterpolation{T,3,NI,TCoefs,Axs,KA,Val{3},
                    DG,EQ,KBC,MixedIntegralOrder{DO},FD,SD,Val{:not_used},NB,Val{false},Val{NI}})(x::Vararg{Number,3}) where
                    {T<:AbstractFloat,NI,TCoefs<:AbstractArray{T,3},
                    Axs<:Tuple{<:AbstractVector,<:AbstractVector,<:AbstractVector},
                    KA<:Tuple{<:ConvolutionKernel,<:ConvolutionKernel,<:ConvolutionKernel},
                    DG,EQ<:Tuple{Int,Int,Int},KBC<:Tuple{<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol}},
                    DO,FD,SD,NB<:Nothing}
 
    ns = ntuple(d -> size(itp.coefs, d), 3)
 
    ids = ntuple(3) do d
        if DO[d] != -1
            clamp(floor(Int, (x[d] - itp.knots[d][1]) / itp.h[d] + one(T)),
                  itp.eqs[d], length(itp.knots[d]) - itp.eqs[d])
        else
            0  # unused for integral dims
        end
    end
 
    ranges = ntuple(3) do d
        if DO[d] == -1
            1:ns[d]
        else
            (ids[d] - (itp.eqs[d]-1)):(ids[d] + itp.eqs[d])
        end
    end
 
    result = zero(T)
    @inbounds for i1 in ranges[1]
        if DO[1] == -1
            xi = itp.knots[1][itp.eqs[1]] + (i1 - itp.eqs[1]) * itp.h[1]
            kx = itp.kernel[1]((x[1] - xi) / itp.h[1]) -
                 itp.kernel[1]((itp.anchor[1] - xi) / itp.h[1])
        else
            kx = itp.kernel[1]((x[1] - itp.knots[1][i1]) / itp.h[1])
        end
        for i2 in ranges[2]
            if DO[2] == -1
                yj = itp.knots[2][itp.eqs[2]] + (i2 - itp.eqs[2]) * itp.h[2]
                ky = itp.kernel[2]((x[2] - yj) / itp.h[2]) -
                     itp.kernel[2]((itp.anchor[2] - yj) / itp.h[2])
            else
                ky = itp.kernel[2]((x[2] - itp.knots[2][i2]) / itp.h[2])
            end
            for i3 in ranges[3]
                if DO[3] == -1
                    zk = itp.knots[3][itp.eqs[3]] + (i3 - itp.eqs[3]) * itp.h[3]
                    kz = itp.kernel[3]((x[3] - zk) / itp.h[3]) -
                         itp.kernel[3]((itp.anchor[3] - zk) / itp.h[3])
                else
                    kz = itp.kernel[3]((x[3] - itp.knots[3][i3]) / itp.h[3])
                end
                result += itp.coefs[i1, i2, i3] * kx * ky * kz
            end
        end
    end
 
    scale = one(T)
    for d in 1:3
        scale *= DO[d] == -1 ? itp.h[d] : one(T) / itp.h[d]^DO[d]
    end
    return @fastmath result * scale
end