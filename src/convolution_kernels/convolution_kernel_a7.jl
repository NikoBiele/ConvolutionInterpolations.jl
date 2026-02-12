# see 'docstring.jl' for documentation
const a7_coefs = Dict(
    # 4 equation septic
    :eq1 => [1//1, 0//1, -3611//1734, 0//1, 16775//10404, 0//1, -22039//27744, 22013//83232],
    :eq2 => [259//3468, 109739//20808, -243//17, 33145//2312, -5815//867, 9583//6936, -477//9248, -1127//83232],
    :eq3 => [114355//3468, -1983989//20808, 99572//867, -521045//6936, 301115//10404, -45703//6936, 79//96, -403//9248],
    :eq4 => [9088//867, -49984//2601, 13064//867, -5680//867, 8875//5202, -923//3468, 213//9248, -71//83232]
)
const a7_coefs_d1 = Dict(
    # 4 equation septic
    :eq1 => [0//1*1//1, 0//1, 2//1*-3611//1734, 3//1*0//1, 4//1*16775//10404, 5//1*0//1, 6//1*-22039//27744, 7//1*22013//83232],
    :eq2 => [0//1*259//3468, 109739//20808, 2//1*-243//17, 3//1*33145//2312, 4//1*-5815//867, 5//1*9583//6936, 6//1*-477//9248, 7//1*-1127//83232],
    :eq3 => [0//1*114355//3468, -1983989//20808, 2//1*99572//867, 3//1*-521045//6936, 4//1*301115//10404, 5//1*-45703//6936, 6//1*79//96, 7//1*-403//9248],
    :eq4 => [0//1*9088//867, -49984//2601, 2//1*13064//867, 3//1*-5680//867, 4//1*8875//5202, 5//1*-923//3468, 6//1*213//9248, 7//1*-71//83232]
)

function (::ConvolutionKernel{:a7,DO})(s::T) where {T,DO}
    a7_coefs_in = if DO == 0
        a7_coefs
    elseif DO == 1
        a7_coefs_d1
    else
        error("kernel :a5 supports differentiation orders 0, 1, but got $DO")
    end
    s_abs = abs(s)
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