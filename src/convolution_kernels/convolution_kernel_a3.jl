"""
    (::ConvolutionKernel{Val{:a3},DO})(s)

Keys' cubic kernel (a = -1/2). Support [-2, 2], 2 pieces.
C1 continuous, 3rd-order accuracy. The classic cubic convolution kernel from Keys (1981).
"""

const a3_coefs = Dict(
    :eq1 => [1//1, 0//1, -5//2, 3//2],
    :eq2 => [2//1, -4//1, 5//2, -1//2]
)
const a3_coefs_d1 = Dict(
    :eq1 => [0//1*1//1, 0//1, 2//1*-5//2, 3//1*3//2],
    :eq2 => [0//1*2//1, -4//1, 2//1*5//2, 3//1*-1//2]
)

function (::ConvolutionKernel{:a3,DO})(s::T) where {T,DO}
    a3_coefs_in = if DO == 0
        a3_coefs
    elseif DO == 1
        a3_coefs_d1
    else
        error("kernel :a3 supports differentiation orders 0, 1, but got $DO")
    end
    s_abs = abs(s)
    if s_abs < 1.0
        return horner(s_abs, a3_coefs_in, :eq1, T, DO) * (isodd(DO) ? T(sign(s)) : one(T))
    elseif s_abs < 2.0
        return horner(s_abs, a3_coefs_in, :eq2, T, DO) * (isodd(DO) ? T(sign(s)) : one(T))
    else
        return zero(T)
    end
end