"""
    (::ConvolutionKernel{:a4,DO})(s)

Keys' 4th order cubic kernel. Support [-3, 3], 3 pieces.
C1 continuous, 4th-order accuracy. One extra stencil point over `:a3` for an additional
order of accuracy — the default kernel.
"""

const a4_coefs = Dict(
    :eq1 => [1//1, 0//1, -7//3, 4//3],
    :eq2 => [15//6, -59//12, 3//1, -7//12],
    :eq3 => [-3//2, 21//12, -2//3, 1//12]
)
const a4_coefs_d1 = Dict(
    :eq1 => [0//1*1//1, 0//1, 2//1*-7//3, 3//1*4//3],
    :eq2 => [0//1*15//6, -59//12, 2//1*3//1, 3//1*-7//12],
    :eq3 => [0//1*-3//2, 21//12, 2//1*-2//3, 3//1*1//12]
)

const a4_coefs_i1 = Dict(
    :eq1 => [0//1, 1//1, 0//1, -7//9, 1//3],
    :eq2 => [-49//144, 5//2, -59//24, 1//1, -7//48],
    :eq3 => [23//16, -3//2, 7//8, -2//9, 1//48],
)

function (::ConvolutionKernel{:a4,DO})(s::T) where {T,DO}
    s_abs = abs(s)
    if DO == -1
        if s_abs >= 3
            return T(1//2) * T(sign(s))
        elseif s_abs < 1
            return horner(s_abs, a4_coefs_i1, :eq1, T, 0) * T(sign(s))
        elseif s_abs < 2
            return horner(s_abs, a4_coefs_i1, :eq2, T, 0) * T(sign(s))
        else
            return horner(s_abs, a4_coefs_i1, :eq3, T, 0) * T(sign(s))
        end
    end
    a4_coefs_in = if DO == 0
        a4_coefs
    elseif DO == 1
        a4_coefs_d1
    else
        error("kernel :a4 supports differentiation orders -1, 0, 1, but got $DO")
    end
    if s_abs < 1.0
        return horner(s_abs, a4_coefs_in, :eq1, T, DO) * (isodd(DO) ? T(sign(s)) : one(T))
    elseif s_abs < 2.0
        return horner(s_abs, a4_coefs_in, :eq2, T, DO) * (isodd(DO) ? T(sign(s)) : one(T))
    elseif s_abs < 3.0
        return horner(s_abs, a4_coefs_in, :eq3, T, DO) * (isodd(DO) ? T(sign(s)) : one(T))
    else
        return zero(T)
    end
end