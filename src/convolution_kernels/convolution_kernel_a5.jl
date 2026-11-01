const a = 3//64
const a5_coefs = Dict(
    # 3 equation quintic
    :eq1 => [1//1, 0//1, 8//1*a-5//2, 0//1, -18//1*a+45//16, 10//1*a-21//16],
    :eq2 => [-66//1*a+5//1, 265//1*a-15//1, -392//1*a+35//2, 270//1*a-10//1, -88//1*a+45//16, 11//1*a-5//16],
    :eq3 => [-162//1*a, 297//1*a, -216//1*a, 78//1*a, -14//1*a, a]
)

# Derivative coefficients, derived exactly from a5_coefs (see derive_kernel_coefs.jl)
const a5_coefs_d1 = _derivative_coefs(a5_coefs, 1)

function (::ConvolutionKernel{:a5,DO})(s::T) where {T,DO}
    s_abs = abs(s)
    if DO < 0
        # antiderivative of order −DO, exact pieces and far field (see column_polynomials.jl)
        return _kernel_antiderivative(Val(:a5), Val(-DO), s)
    end
    a5_coefs_in = if DO == 0
        a5_coefs
    elseif DO == 1
        a5_coefs_d1
    else
        error("kernel :a5 supports differentiation orders -$(_max_integral_order[:a5]) to $(_max_derivative_uniform[:a5]), but got $DO")
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