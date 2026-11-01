const a7_coefs = Dict(
    # 4 equation septic
    :eq1 => [1//1, 0//1, -3611//1734, 0//1, 16775//10404, 0//1, -22039//27744, 22013//83232],
    :eq2 => [259//3468, 109739//20808, -243//17, 33145//2312, -5815//867, 9583//6936, -477//9248, -1127//83232],
    :eq3 => [114355//3468, -1983989//20808, 99572//867, -521045//6936, 301115//10404, -45703//6936, 79//96, -403//9248],
    :eq4 => [9088//867, -49984//2601, 13064//867, -5680//867, 8875//5202, -923//3468, 213//9248, -71//83232]
)

# Derivative coefficients, derived exactly from a7_coefs (see derive_kernel_coefs.jl)
const a7_coefs_d1 = _derivative_coefs(a7_coefs, 1)

function (::ConvolutionKernel{:a7,DO})(s::T) where {T,DO}
    s_abs = abs(s)
    if DO < 0
        # antiderivative of order −DO, exact pieces and far field (see column_polynomials.jl)
        return _kernel_antiderivative(Val(:a7), Val(-DO), s)
    end
    a7_coefs_in = if DO == 0
        a7_coefs
    elseif DO == 1
        a7_coefs_d1
    else
        error("kernel :a7 supports differentiation orders -$(_max_integral_order[:a7]) to $(_max_derivative_uniform[:a7]), but got $DO")
    end
    if s_abs < 1.0
        return horner(s_abs, a7_coefs_in, :eq1, T, DO) * (isodd(DO) ? T(sign(s)) : one(T))
    elseif s_abs < 2.0
        return horner(s_abs, a7_coefs_in, :eq2, T, DO) * (isodd(DO) ? T(sign(s)) : one(T))
    elseif s_abs < 3.0
        return horner(s_abs, a7_coefs_in, :eq3, T, DO) * (isodd(DO) ? T(sign(s)) : one(T))
    elseif s_abs < 4.0
        return horner(s_abs, a7_coefs_in, :eq4, T, DO) * (isodd(DO) ? T(sign(s)) : one(T))
    else
        return zero(T)
    end
end