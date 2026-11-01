const b7_coefs = Dict(
    # 6 equation septic, 7th order accurate
    :eq1 => [1//1, 0//1, -1173431//624240, 0//1, 922243//749088, 0//1, -7587149//14981760, 258097//1664640],
    :eq2 => [106117//124848, 834167//1248480, -1709177//624240, -119839//249696, 2712769//749088, -214769//83232, 11023027//14981760, -384481//4993920], 
    :eq3 => [30587//2448, -35785081//1248480, 376423//15606, -126025//14688, 794957//1498176, 13571//27744, -796241//5992704, 105457//9987840], 
    :eq4 => [-2861771//41616, 325209229//2496960, -129943051//1248480, 22764947//499392, -17736439//1498176, 303937//166464, -4645469//29963520, 167719//29963520],
    :eq5 => [9845663//124848, -908278361//7490880, 99126901//1248480, -14343661//499392, 9292729//1498176, -399533//499392, 1710679//29963520, -52123//29963520], 
    :eq6 => [9711//1156, -29133//2890, 119769//23120, -20501//13872, 14027//55488, -1079//41616, 44239//29963520, -1079//29963520]
)

# Derivative coefficients, derived exactly from b7_coefs (see derive_kernel_coefs.jl)
const b7_coefs_d1 = _derivative_coefs(b7_coefs, 1)
const b7_coefs_d2 = _derivative_coefs(b7_coefs, 2)
const b7_coefs_d3 = _derivative_coefs(b7_coefs, 3)
const b7_coefs_d4 = _derivative_coefs(b7_coefs, 4)
const b7_coefs_d5 = _derivative_coefs(b7_coefs, 5)

function (::ConvolutionKernel{:b7,DO})(s::T) where {T,DO} # 7th order accurate 7th degree
    s_abs = abs(s)
    if DO < 0
        # antiderivative of order −DO, exact pieces and far field (see column_polynomials.jl)
        return _kernel_antiderivative(Val(:b7), Val(-DO), s)
    end
    b7_coefs_in = if DO == 0
        b7_coefs
    elseif DO == 1
        b7_coefs_d1
    elseif DO == 2
        b7_coefs_d2
    elseif DO == 3
        b7_coefs_d3
    elseif DO == 4
        b7_coefs_d4
    elseif DO == 5
        b7_coefs_d5
    else
        error("kernel :b7 supports differentiation orders -$(_max_integral_order[:b7]) to $(_max_derivative_uniform[:b7]), but got $DO")
    end
    if s_abs < 1.0
        return horner(s_abs, b7_coefs_in, :eq1, T, DO) * (isodd(DO) ? T(sign(s)) : one(T))
    elseif s_abs < 2.0
        return horner(s_abs, b7_coefs_in, :eq2, T, DO) * (isodd(DO) ? T(sign(s)) : one(T))
    elseif s_abs < 3.0
        return horner(s_abs, b7_coefs_in, :eq3, T, DO) * (isodd(DO) ? T(sign(s)) : one(T))
    elseif s_abs < 4.0
        return horner(s_abs, b7_coefs_in, :eq4, T, DO) * (isodd(DO) ? T(sign(s)) : one(T))
    elseif s_abs < 5.0
        return horner(s_abs, b7_coefs_in, :eq5, T, DO) * (isodd(DO) ? T(sign(s)) : one(T))
    elseif s_abs < 6.0
        return horner(s_abs, b7_coefs_in, :eq6, T, DO) * (isodd(DO) ? T(sign(s)) : one(T))
    else
        return zero(T)
    end
end