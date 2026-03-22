@inline function (itp::ConvolutionInterpolation{T,3,TCoefs,IT,Axs,KA,Val{3},
                    DG,EQ,KBC,IntegralOrder,FD,SD,SG,NB,Val{false}})(x::Vararg{Number,3}) where 
                    {T,TCoefs,IT,Axs,KA<:NTuple{3,<:ConvolutionKernel},DG,EQ<:Tuple{Int,Int,Int},KBC,FD,SD,SG,NB<:Nothing}

    result = zero(T)
    n1 = size(itp.coefs, 1)
    n2 = size(itp.coefs, 2)
    n3 = size(itp.coefs, 3)
    @inbounds for i1 in 1:n1
        xi = itp.knots[1][_eqs_d(itp.eqs,1)] + (i1 - _eqs_d(itp.eqs,1)) * itp.h[1]
        dx = _kernel_d(itp.kernel,1)((x[1]          - xi) / itp.h[1]) -
             _kernel_d(itp.kernel,1)((itp.anchor[1] - xi) / itp.h[1])
        for i2 in 1:n2
            yj = itp.knots[2][_eqs_d(itp.eqs,2)] + (i2 - _eqs_d(itp.eqs,2)) * itp.h[2]
            dy = _kernel_d(itp.kernel,2)((x[2]          - yj) / itp.h[2]) -
                 _kernel_d(itp.kernel,2)((itp.anchor[2] - yj) / itp.h[2])
            for i3 in 1:n3
                zk = itp.knots[3][_eqs_d(itp.eqs,3)] + (i3 - _eqs_d(itp.eqs,3)) * itp.h[3]
                dz = _kernel_d(itp.kernel,3)((x[3]          - zk) / itp.h[3]) -
                     _kernel_d(itp.kernel,3)((itp.anchor[3] - zk) / itp.h[3])
                result += itp.coefs[i1, i2, i3] * dx * dy * dz
            end
        end
    end
    return result * itp.h[1] * itp.h[2] * itp.h[3]
end

@inline function (itp::ConvolutionInterpolation{T,3,TCoefs,IT,Axs,KA,Val{3},
                    DG,EQ,KBC,MixedIntegralOrder{DO},FD,SD,SG})(x::Vararg{Number,3}) where
                    {T,TCoefs,IT,Axs,KA<:NTuple{3,<:ConvolutionKernel},
                    DG,EQ<:Tuple{Int,Int,Int},KBC,DO,FD,SD,SG}
 
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