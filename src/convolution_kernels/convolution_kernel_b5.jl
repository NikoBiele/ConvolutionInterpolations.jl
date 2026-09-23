"""
    (::ConvolutionKernel{:b5,DO})(s)

Quintic b-series kernel. Support [-5, 5], 5 pieces.
C3 continuous, 7th-order accuracy. Derived by optimizing frequency response closeness
to the ideal sinc function. Supports derivatives up to order 3 with smooth evaluation.
"""

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

const b5_coefs_i1 = Dict(
    :eq1 => [0//1, 1//1, 0//1, -731//1152, 0//1, 11423//38400, -4483//46080],
    :eq2 => [881//25600, 2597//3840, 16931//15360, -14371//5760, 25933//15360, -6337//12800, 313//5760],
    :eq3 => [-54061//76800, 1211//256, -96221//15360, 4549//1152, -20011//15360, 335//1536, -169//11520],
    :eq4 => [-137039//153600, 1681//1280, -7843//30720, -1463//11520, 2071//30720, -439//38400, 7//10240],
    :eq5 => [78091//18432, -3625//768, 5075//2048, -1595//2304, 667//6144, -29//3200, 29//92160],
)

function (::ConvolutionKernel{:b5,DO})(s::T) where {T,DO} # 5 equations 7th order accurate quintic
    s_abs = abs(s)
    if DO == -1
        # Antiderivative K̃: odd function, saturates at ±1/2
        if s_abs >= 5
            return T(1//2) * T(sign(s))
        elseif s_abs < 1
            return horner(s_abs, b5_coefs_i1, :eq1, T, 0) * T(sign(s))
        elseif s_abs < 2
            return horner(s_abs, b5_coefs_i1, :eq2, T, 0) * T(sign(s))
        elseif s_abs < 3
            return horner(s_abs, b5_coefs_i1, :eq3, T, 0) * T(sign(s))
        elseif s_abs < 4
            return horner(s_abs, b5_coefs_i1, :eq4, T, 0) * T(sign(s))
        else
            return horner(s_abs, b5_coefs_i1, :eq5, T, 0) * T(sign(s))
        end
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
        error("kernel :b5 supports differentiation orders -1, 0, 1, 2, 3, but got $DO")
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