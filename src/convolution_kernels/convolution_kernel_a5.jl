"""
    (::ConvolutionKernel{:a5,DO})(s)

Quintic a-series kernel. Support [-3, 3], 3 pieces.
C1 continuous. Same support as `:a4` but higher polynomial degree.
"""

const a = 3//64
const a5_coefs = Dict(
    # 3 equation quintic
    :eq1 => [1//1, 0//1, 8//1*a-5//2, 0//1, -18//1*a+45//16, 10//1*a-21//16],
    :eq2 => [-66//1*a+5//1, 265//1*a-15//1, -392//1*a+35//2, 270//1*a-10//1, -88//1*a+45//16, 11//1*a-5//16],
    :eq3 => [-162//1*a, 297//1*a, -216//1*a, 78//1*a, -14//1*a, a]
)
const a5_coefs_d1 = Dict(
    # 3 equation quintic
    :eq1 => [0//1*1//1, 0//1, 2//1*(8//1*a-5//2), 3//1*0//1, 4//1*(-18//1*a+45//16), 5//1*(10//1*a-21//16)],
    :eq2 => [0//1*(-66//1*a+5//1), 265//1*a-15//1, 2//1*(-392//1*a+35//2), 3//1*(270//1*a-10//1), 4//1*(-88//1*a+45//16), 5//1*(11//1*a-5//16)],
    :eq3 => [0//1*-162//1*a, 297//1*a, 2//1*-216//1*a, 3//1*78//1*a, 4//1*-14//1*a, 5//1*a]
)

const a5_coefs_i1 = Dict(
    :eq1 => [0//1, 1//1, 0//1, -17//24, 0//1, 63//160, -9//64],
    :eq2 => [-83//384, 61//32, -165//128, -7//24, 85//128, -21//80, 13//384],
    :eq3 => [2507//640, -243//32, 891//128, -27//8, 117//128, -21//160, 1//128],
)

function (::ConvolutionKernel{:a5,DO})(s::T) where {T,DO}
    s_abs = abs(s)
    if DO == -1
        if s_abs >= 3
            return T(1//2) * T(sign(s))
        elseif s_abs < 1
            return horner(s_abs, a5_coefs_i1, :eq1, T, 0) * T(sign(s))
        elseif s_abs < 2
            return horner(s_abs, a5_coefs_i1, :eq2, T, 0) * T(sign(s))
        else
            return horner(s_abs, a5_coefs_i1, :eq3, T, 0) * T(sign(s))
        end
    end
    a5_coefs_in = if DO == 0
        a5_coefs
    elseif DO == 1
        a5_coefs_d1
    else
        error("kernel :a5 supports differentiation orders -1, 0, 1, but got $DO")
    end
    if s_abs < 1.0
        return horner(s_abs, a5_coefs_in, :eq1, T, DO) * (isodd(DO) ? T(sign(s)) : one(T))
    elseif s_abs < 2.0
        return horner(s_abs, a5_coefs_in, :eq2, T, DO) * (isodd(DO) ? T(sign(s)) : one(T))
    elseif s_abs < 3.0
        return horner(s_abs, a5_coefs_in, :eq3, T, DO) * (isodd(DO) ? T(sign(s)) : one(T))
    else
        return zero(T)
    end
end