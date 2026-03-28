@inline function (itp::ConvolutionInterpolation{T,2,2,TCoefs,Axs,KA,Val{2},
                    DG,EQ,KBC,IntegralOrder,FD,SD,Val{:not_used},NB,Val{false},Val{2}})(x::Vararg{Number,2}) where 
                    {T<:AbstractFloat,TCoefs<:AbstractArray{T,2},Axs<:Tuple{<:AbstractVector,<:AbstractVector},
                    KA<:Tuple{<:ConvolutionKernel,<:ConvolutionKernel},DG,EQ<:Tuple{Int,Int},
                    KBC<:Tuple{<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol}},FD,SD,NB<:Nothing}

    result = zero(T)
    n1 = size(itp.coefs, 1)
    n2 = size(itp.coefs, 2)
    @inbounds for i1 in 1:n1
        xi = itp.knots[1][_eqs_d(itp.eqs,1)] + (i1 - _eqs_d(itp.eqs,1)) * itp.h[1]
        dx = _kernel_d(itp.kernel,1)((x[1]          - xi) / itp.h[1]) -
             _kernel_d(itp.kernel,1)((itp.anchor[1] - xi) / itp.h[1])
        for i2 in 1:n2
            yj = itp.knots[2][_eqs_d(itp.eqs,2)] + (i2 - _eqs_d(itp.eqs,2)) * itp.h[2]
            dy = _kernel_d(itp.kernel,2)((x[2]          - yj) / itp.h[2]) -
                 _kernel_d(itp.kernel,2)((itp.anchor[2] - yj) / itp.h[2])
            result += itp.coefs[i1, i2] * dx * dy
        end
    end
    return result * itp.h[1] * itp.h[2]
end