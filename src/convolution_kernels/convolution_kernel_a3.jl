const a3_coefs = Dict(
    :eq1 => [1//1, 0//1, -5//2, 3//2],
    :eq2 => [2//1, -4//1, 5//2, -1//2]
)

# Derivative coefficients, derived exactly from a3_coefs (see derive_kernel_coefs.jl)
const a3_coefs_d1 = _derivative_coefs(a3_coefs, 1)

function (::ConvolutionKernel{:a3,DO})(s::T) where {T,DO}
    s_abs = abs(s)
    if DO < 0
        # antiderivative of order −DO, exact pieces and far field (see column_polynomials.jl)
        return _kernel_antiderivative(Val(:a3), Val(-DO), s)
    end
    a3_coefs_in = if DO == 0
        a3_coefs
    elseif DO == 1
        a3_coefs_d1
    else
        error("kernel :a3 supports differentiation orders -$(_max_integral_order[:a3]) to $(_max_derivative_uniform[:a3]), but got $DO")
    end
    if s_abs < 1.0
        return horner(s_abs, a3_coefs_in, :eq1, T, DO) * (isodd(DO) ? T(sign(s)) : one(T))
    elseif s_abs < 2.0
        return horner(s_abs, a3_coefs_in, :eq2, T, DO) * (isodd(DO) ? T(sign(s)) : one(T))
    else
        return zero(T)
    end
end