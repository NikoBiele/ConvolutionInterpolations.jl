function (itp::FastConvolutionInterpolation{T,N,TCoefs,IT,Axs,KA,HigherDimension{N},
        CK,EQ,PR,KP,KBC,IntegralOrder,FD,SD,Val{SG},Val{false}})(x::Vararg{Number,N}) where
        {T,N,TCoefs,IT,KA<:NTuple{N,<:Nothing},Axs,CK<:AbstractConvolutionKernel,EQ,PR,KP,KBC,FD,SD,SG}

    result = zero(T)
    @inbounds for idx in Iterators.product(ntuple(d -> 1:size(itp.coefs, d), N)...)
        kt_prod = one(T)
        @inbounds for d in 1:N
            xjd = itp.knots[d][itp.eqs[d]] + (idx[d] - itp.eqs[d]) * itp.h[d]
            sjd = (x[d] - xjd) / itp.h[d]

            if abs(sjd) >= T(itp.eqs[d])
                ktd = T(1//2) * T(sign(sjd))
            else
                col_float = T(itp.eqs[d]) + sjd
                col = clamp(floor(Int, col_float) + 1, 1, 2 * itp.eqs[d])
                x_diff_right = col_float - T(col - 1)
                continuous_idx = x_diff_right * T(length(itp.pre_range[d]) - 1) + one(T)
                i      = clamp(floor(Int, continuous_idx), 1, length(itp.pre_range[d]) - 1)
                i_next = i + 1
                t      = continuous_idx - T(i)
                ktd = (one(T) - t) * itp.kernel_pre[d][i, col] + t * itp.kernel_pre[d][i_next, col]
            end

            kt_prod *= ktd - itp.left_values[d][idx[d]]
        end

        result += itp.coefs[idx...] * kt_prod
    end

    return result * prod(itp.h)
end