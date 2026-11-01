const b5_coefs = Dict(
    # 5 equation quintic, 7th order accurate
    :eq1 => [1//1, 0//1, -731//384, 0//1, 11423//7680, -4483//7680], 
    :eq2 => [2597//3840, 16931//7680, -14371//1920, 25933//3840, -6337//2560, 313//960], 
    :eq3 => [1211//256, -96221//7680, 4549//384, -20011//3840, 1675//1536, -169//1920], 
    :eq4 => [1681//1280, -7843//15360, -1463//3840, 2071//7680, -439//7680, 21//5120],
    :eq5 => [-3625//768, 5075//1024, -1595//768, 667//1536, -29//640, 29//15360]
)

# Derivative coefficients, derived exactly from b5_coefs (see derive_kernel_coefs.jl)
const b5_coefs_d1 = _derivative_coefs(b5_coefs, 1)
const b5_coefs_d2 = _derivative_coefs(b5_coefs, 2)
const b5_coefs_d3 = _derivative_coefs(b5_coefs, 3)

function (::ConvolutionKernel{:b5,DO})(s::T) where {T,DO} # 5 equations 7th order accurate quintic
    s_abs = abs(s)
    if DO < 0
        # antiderivative of order −DO, exact pieces and far field (see column_polynomials.jl)
        return _kernel_antiderivative(Val(:b5), Val(-DO), s)
    end
    b5_coefs_in = if DO == 0
        b5_coefs
    elseif DO == 1
        b5_coefs_d1
    elseif DO == 2
        b5_coefs_d2
    elseif DO == 3
        b5_coefs_d3
    else
        error("kernel :b5 supports differentiation orders -$(_max_integral_order[:b5]) to $(_max_derivative_uniform[:b5]), but got $DO")
    end
    if s_abs < 1.0
        return horner(s_abs, b5_coefs_in, :eq1, T, DO) * (isodd(DO) ? sign(s) : one(T))
    elseif s_abs < 2.0
        return horner(s_abs, b5_coefs_in, :eq2, T, DO) * (isodd(DO) ? sign(s) : one(T))
    elseif s_abs < 3.0
        return horner(s_abs, b5_coefs_in, :eq3, T, DO) * (isodd(DO) ? sign(s) : one(T))
    elseif s_abs < 4.0
        return horner(s_abs, b5_coefs_in, :eq4, T, DO) * (isodd(DO) ? sign(s) : one(T))
    elseif s_abs < 5.0
        return horner(s_abs, b5_coefs_in, :eq5, T, DO) * (isodd(DO) ? sign(s) : one(T))
    else
        return zero(T)
    end
end