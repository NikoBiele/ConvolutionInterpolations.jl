"""
(itp::FastConvolutionInterpolation{T,1,...})(x::Number) — IntegralOrder
Evaluate 1D fast antiderivative at coordinate x.

Evaluation decomposes into three parts:
  local stencil — K̃ weights of all 2·eqs columns from exact column polynomials
                  (see `_kernel_weights`), anchored by `left_values`
  left tail     — O(1) prefix sum lookup (tail1_left)
  right tail    — O(1) suffix sum lookup (tail1_right)

Result is (local + left_tail + right_tail) * h, anchored to zero at the
leftmost interior knot. O(1) in grid size, allocation-free, exact to rounding.
See also: FastConvolutionInterpolation, convolution_fast_integration_2d.
"""

@inline function (itp::FastConvolutionInterpolation{T,1,1,TCoefs,Axs,KA,Val{1},
                    DG,EQ,KBC,FastIntegralOrder,FD,SD,Val{SG},Val{false},Val{1}})(x::Vararg{Number,1}) where 
                    {T<:AbstractFloat,TCoefs<:AbstractArray{T,1},Axs<:Tuple{<:AbstractVector},
                    KA<:Tuple{<:Nothing},DG,EQ<:Tuple{Int},KBC<:Tuple{<:Tuple{Symbol,Symbol}},FD,SD,SG}

    x = T.(x)
    eqs_int = itp.eqs[1]
    i_float  = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T)
    i        = clamp(floor(Int, i_float), eqs_int, length(itp.coefs) - eqs_int)
    t        = i_float - T(i)

    # K̃ weights of all 2·eqs columns at τ = t. Column c holds K̃(c − 1 − eqs + t), the weight
    # of coefficient j = i + eqs + 1 − c (the stencil in reversed order).
    w = _kernel_weights(Val(_kernel_sym(itp.kernel_sym)[1]), Val(-1), t)

    result = zero(T)
    @inbounds for c in 1:length(w)
        j = i + eqs_int + 1 - c
        result += itp.coefs[j] * (w[c] - itp.left_values[1][j])
    end

    left_tail  = (i - eqs_int) >= 1                      ? itp.tail1_left[1][i - eqs_int]      : zero(T)
    right_tail = (i + eqs_int + 1) <= length(itp.coefs)  ? itp.tail1_right[1][i + eqs_int + 1] : zero(T)

    return (result + left_tail + right_tail) * itp.h[1]
end