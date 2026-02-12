# see 'docstring.jl' for documentation
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

function (::ConvolutionKernel{:a4,DO})(s::T) where {T,DO}
    a4_coefs_in = if DO == 0
        a4_coefs
    elseif DO == 1
        a4_coefs_d1
    elseif DO == -1
        a4_coefs_i1
    else
        error("kernel :a4 supports differentiation orders -1, 0, 1, but got $DO")
    end
    s_abs = abs(s)
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