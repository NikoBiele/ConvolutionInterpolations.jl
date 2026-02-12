# see 'docstring.jl' for documentation
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

function (::ConvolutionKernel{:a5,DO})(s::T) where {T,DO}
    a5_coefs_in = if DO == 0
        a5_coefs
    elseif DO == 1
        a5_coefs_d1
    else
        error("kernel :a5 supports differentiation orders 0, 1, but got $DO")
    end
    s_abs = abs(s)
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