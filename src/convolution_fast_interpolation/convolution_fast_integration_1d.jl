"""
(itp::FastConvolutionInterpolation{T,1,...})(x::Number) — IntegralOrder
Evaluate 1D fast antiderivative at coordinate x.
Dispatches on subgrid mode: :quintic, :cubic, :linear.

Evaluation decomposes into three parts:
  local stencil — O(eqs) K̃ lookups via Hermite/linear subgrid interpolation
  left tail     — O(1) prefix sum lookup (tail1_left)
  right tail    — O(1) suffix sum lookup (tail1_right)

Result is (local + left_tail + right_tail) * h, anchored to zero at the
leftmost interior knot. O(1) in grid size, allocation-free.
See also: FastConvolutionInterpolation, convolution_fast_integration_2d.
"""

@inline function (itp::FastConvolutionInterpolation{T,1,1,TCoefs,Axs,KA,Val{1},
                    DG,EQ,PR,KP,KBC,FastIntegralOrder,FD,SD,Val{SG},Val{false},Val{1}})(x::Vararg{Number,1}) where 
                    {T<:AbstractFloat,TCoefs<:AbstractArray{T,1},Axs<:Tuple{<:AbstractVector},
                    KA<:Tuple{<:Nothing},DG,EQ<:Tuple{Int},PR<:Tuple{<:AbstractVector},
                    KP,KBC<:Tuple{<:Tuple{Symbol,Symbol}},FD,SD,SG}

    eqs_int = itp.eqs[1]
    n_pre = length(itp.pre_range[1])
    h_pre = one(T) / T(n_pre - 1)
    result = zero(T)
    i_float  = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T)
    i        = clamp(floor(Int, i_float), eqs_int, length(itp.coefs) - eqs_int)

    @inbounds for j in (i - eqs_int + 1):(i + eqs_int)

        kt = eval_sg_kt(itp, j, Val(1), Val(SG), h_pre, x)

        result += itp.coefs[j] * (kt - itp.left_values[1][j])
    end

    left_tail  = (i - eqs_int) >= 1                      ? itp.tail1_left[1][i - eqs_int]      : zero(T)
    right_tail = (i + eqs_int + 1) <= length(itp.coefs)  ? itp.tail1_right[1][i + eqs_int + 1] : zero(T)

    return (result + left_tail + right_tail) * itp.h[1]
end