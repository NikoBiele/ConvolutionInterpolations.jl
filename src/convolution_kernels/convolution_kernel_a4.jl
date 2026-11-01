const a4_coefs = Dict(
    :eq1 => [1//1, 0//1, -7//3, 4//3],
    :eq2 => [15//6, -59//12, 3//1, -7//12],
    :eq3 => [-3//2, 21//12, -2//3, 1//12]
)

# Derivative coefficients, derived exactly from a4_coefs (see derive_kernel_coefs.jl)
const a4_coefs_d1 = _derivative_coefs(a4_coefs, 1)

function (::ConvolutionKernel{:a4,DO})(s::T) where {T,DO}
    s_abs = abs(s)
    if DO < 0
        # antiderivative of order −DO, exact pieces and far field (see column_polynomials.jl)
        return _kernel_antiderivative(Val(:a4), Val(-DO), s)
    end
    a4_coefs_in = if DO == 0
        a4_coefs
    elseif DO == 1
        a4_coefs_d1
    else
        error("kernel :a4 supports differentiation orders -$(_max_integral_order[:a4]) to $(_max_derivative_uniform[:a4]), but got $DO")
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