@inline function (itp::ConvolutionInterpolation{T,3,3,TCoefs,Axs,KA,Val{3},
                    DG,EQ,KBC,IntegralOrder,FD,SD,Val{:not_used},NB,Val{false},Val{3}})(x::Vararg{Number,3}) where 
                    {T<:AbstractFloat,TCoefs<:AbstractArray{T,3},
                    Axs<:Tuple{<:AbstractVector,<:AbstractVector,<:AbstractVector},
                    KA<:Tuple{<:ConvolutionKernel,<:ConvolutionKernel,<:ConvolutionKernel},DG,
                    EQ<:Tuple{Int,Int,Int},KBC<:Tuple{<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol}},
                    FD,SD,NB<:Nothing}

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