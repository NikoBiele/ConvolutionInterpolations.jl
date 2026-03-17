"""
(itp::FastConvolutionInterpolation{T,N,...})(x...) — MixedIntegralOrder
Evaluate mixed interpolation/derivative/antiderivative operator in N dimensions.

For each dimension d, applies either:
  K̃ lookup (DO[d] == -1): antiderivative contribution, scaled by h[d]
  K  lookup (DO[d] == 0):  interpolation contribution
  K  lookup (DO[d] >= 1):  derivative contribution, scaled by (-1/h[d])^DO[d]

Result is the tensor product over all dimensions, summed over all coefficient indices.
Evaluation is O(N^D) — no prefix sum optimization for mixed orders.
See also: FastConvolutionInterpolation, convolution_fast_integration_1d, convolution_fast_interpolation_perdim.
"""

function (itp::FastConvolutionInterpolation{T,N,TCoefs,IT,Axs,KA,DIM,DG,EQ,PR,KP,KBC,
            MixedIntegralOrder{DO},FD,SD,SG})(x::Vararg{Number,N}) where
            {T,N,TCoefs,IT,Axs,KA,DIM,DG,EQ<:NTuple{N,Int},PR,KP,KBC,DO,FD,SD,SG}

    n_pre = length(itp.pre_range[1])   # all dims share same precompute resolution
    result = zero(T)

    @inbounds for idx in Iterators.product(ntuple(d -> 1:size(itp.coefs, d), N)...)
        kt_prod = one(T)
        @inbounds for d in 1:N
            eqs_d = itp.eqs[d]
            if DO[d] == -1
                # ---- integral dimension: K̃ lookup ----
                xjd = itp.knots[d][eqs_d] + (idx[d] - eqs_d) * itp.h[d]
                sjd = (x[d] - xjd) / itp.h[d]

                if abs(sjd) >= T(eqs_d)
                    ktd = T(1//2) * T(sign(sjd))
                else
                    col_float = T(eqs_d) + sjd
                    col = clamp(floor(Int, col_float) + 1, 1, 2 * eqs_d)
                    x_diff_right = col_float - T(col - 1)
                    n_pre_d = length(itp.pre_range[d])
                    continuous_idx = x_diff_right * T(n_pre_d - 1) + one(T)
                    i      = clamp(floor(Int, continuous_idx), 1, n_pre_d - 1)
                    i_next = i + 1
                    t      = continuous_idx - T(i)
                    ktd = (one(T) - t) * itp.kernel_pre[d][i, col] + t * itp.kernel_pre[d][i_next, col]
                end

                kt_prod *= ktd - itp.left_values[d][idx[d]]

            else
                # ---- derivative dimension: normal K lookup ----
                i_float = (x[d] - itp.knots[d][1]) / itp.h[d] + one(T)
                pos = clamp(floor(Int, i_float), eqs_d, length(itp.knots[d]) - eqs_d)
                off = idx[d] - pos
                col = off + eqs_d   # column in kernel table: valid range 1:2*eqs_d
                if col < 1 || col > 2 * eqs_d
                    kt_prod = zero(T)
                    break
                end
                diff_right = one(T) - (x[d] - itp.knots[d][pos]) / itp.h[d]
                n_pre_d = length(itp.pre_range[d])
                continuous_idx = diff_right * T(n_pre_d - 1) + one(T)
                i      = clamp(floor(Int, continuous_idx), 1, n_pre_d - 1)
                i_next = i + 1
                t      = continuous_idx - T(i)
                ktd = (one(T) - t) * itp.kernel_pre[d][i, col] + t * itp.kernel_pre[d][i_next, col]
                kt_prod *= ktd
            end
        end

        result += itp.coefs[idx...] * kt_prod
    end

    # scale: h[d] for integral dims, (-1/h[d])^DO[d] for derivative dims
    scale = one(T)
    @inbounds for d in 1:N
        if DO[d] == -1
            scale *= itp.h[d]
        else
            scale *= (-one(T) / itp.h[d])^DO[d]
        end
    end

    return result * scale
end