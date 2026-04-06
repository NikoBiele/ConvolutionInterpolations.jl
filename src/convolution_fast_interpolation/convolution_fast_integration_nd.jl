function (itp::FastConvolutionInterpolation{T,N,N,TCoefs,Axs,KA,HigherDimension{N},
        DG,EQ,PR,KP,KBC,FastIntegralOrder,FD,SD,Val{SG},Val{false},HigherDimension{N}})(x::Vararg{Number,N}) where
        {T<:AbstractFloat,N,TCoefs<:AbstractArray{T,N},
        Axs<:NTuple{N,<:AbstractVector},
        KA<:NTuple{N,<:Nothing},DG,EQ<:NTuple{N,Int},PR<:NTuple{N,<:AbstractVector},
        KP,KBC<:NTuple{N,Tuple{Symbol,Symbol}},FD,SD,SG}    

    return _nd_integral_eval(itp, Val{SG}(), x)

end

@inline function eval_kt(itp::FastConvolutionInterpolation{T,N,N}, sj::Number, ::Val{D}, ::Val{SG}) where {T, N, D, SG}
    abs(sj) >= T(itp.eqs[D]) && return T(1//2) * T(sign(sj))
    col_float = T(itp.eqs[D]) + sj
    col = clamp(floor(Int, col_float) + 1, 1, 2 * itp.eqs[D])
    n_pre_d = length(itp.pre_range[D])
    h_pre_d = one(T) / T(n_pre_d - 1)
    t = (col_float - T(col - 1)) * T(length(itp.pre_range[D]) - 1) + one(T)
    idx = clamp(floor(Int, t), 1, length(itp.pre_range[D]) - 1); t -= T(idx)
    if SG[D] == :quintic
        quintic_hermite(t,
            itp.kernel_pre[D][idx, col],    itp.kernel_pre[D][idx+1, col],
            itp.kernel_d1_pre[D][idx, col], itp.kernel_d1_pre[D][idx+1, col],
            itp.kernel_d2_pre[D][idx, col], itp.kernel_d2_pre[D][idx+1, col], h_pre_d)
    elseif SG[D] == :cubic
        cubic_hermite(t,
            itp.kernel_pre[D][idx, col],    itp.kernel_pre[D][idx+1, col],
            itp.kernel_d1_pre[D][idx, col], itp.kernel_d1_pre[D][idx+1, col], h_pre_d)
    else
        (one(T) - t) * itp.kernel_pre[D][idx, col] + t * itp.kernel_pre[D][idx+1, col]
    end
end

@generated function _nd_integral_eval(itp::FastConvolutionInterpolation{T,N,N}, ::Val{SG}, x::NTuple{N,<:Number}) where {T, N, SG}
    quote
        coefs = itp.coefs
        result = zero(T)
        Base.Cartesian.@nloops $N i dd -> 1:itp.domain_size[dd] begin
            kt_prod = one(T)
            Base.Cartesian.@nexprs $N dd -> begin
                xjdd = itp.knots[dd][itp.eqs[dd]] + (i_dd - itp.eqs[dd]) * itp.h[dd]
                sjdd = (x[dd] - xjdd) / itp.h[dd]
                kt_prod *= eval_kt(itp, sjdd, Val(dd), Val{SG}()) - itp.left_values[dd][i_dd]
            end
            result += Base.Cartesian.@nref($N, coefs, i) * kt_prod
        end
        return result * prod(itp.h)
    end
end