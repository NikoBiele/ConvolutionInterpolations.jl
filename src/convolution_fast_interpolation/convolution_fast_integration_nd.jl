function (itp::FastConvolutionInterpolation{T,N,N,TCoefs,Axs,KA,HigherDimension{N},
        DG,EQ,PR,KP,KBC,FastIntegralOrder,FD,SD,Val{SG},Val{false},HigherDimension{N}})(x::Vararg{Number,N}) where
        {T<:AbstractFloat,N,TCoefs<:AbstractArray{T,N},
        Axs<:NTuple{N,<:AbstractVector},
        KA<:NTuple{N,<:Nothing},DG,EQ<:NTuple{N,Int},PR<:NTuple{N,<:AbstractVector},
        KP,KBC<:NTuple{N,Tuple{Symbol,Symbol}},FD,SD,SG}    

    n_pre = ntuple(i -> length(itp.pre_range[i]), N)
    h_pre = ntuple(i -> one(T) / T(n_pre[i] - 1), N)

    return _nd_integral_eval(itp, Val(SG), h_pre, x)

end

@generated function _nd_integral_eval(itp::FastConvolutionInterpolation{T,N,N},
                            ::Val{SG}, h_pre::NTuple{N,T}, x::NTuple{N,<:Number}) where {T, N, SG}
    quote
        coefs = itp.coefs
        result = zero(T)
        Base.Cartesian.@nloops $N i dd -> 1:itp.domain_size[dd] begin
            kt_prod = one(T)
            Base.Cartesian.@nexprs $N dd -> begin
                kt_prod *= eval_sg_kt(itp, i_dd, Val(dd), Val(SG), h_pre[dd], x) - itp.left_values[dd][i_dd]
            end
            result += Base.Cartesian.@nref($N, coefs, i) * kt_prod
        end
        return result * prod(itp.h)
    end
end