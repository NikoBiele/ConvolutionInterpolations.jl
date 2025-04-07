# see 'docstring.jl' for documentation
function (::ConvolutionKernel{:a7})(s::T) where {T}
    s_abs = abs(s)
    coef = Dict(
        # 4 equation septic
        :eq1 => [1, 0, -3611/1734, 0, 16775/10404, 0, -22039/27744, 22013/83232],
        :eq2 => [259/3468, 109739/20808, -243/17, 33145/2312, -5815/867, 9583/6936, -477/9248, -1127/83232],
        :eq3 => [114355/3468, -1983989/20808, 99572/867, -521045/6936, 301115/10404, -45703/6936, 79/96, -403/9248],
        :eq4 => [9088/867, -49984/2601, 13064/867, -5680/867, 8875/5202, -923/3468, 213/9248, -71/83232]
    )
    if s_abs < 1.0
        return horner(s_abs, coef, :eq1)
    elseif s_abs < 2.0
        return horner(s_abs, coef, :eq2)
    elseif s_abs < 3.0
        return horner(s_abs, coef, :eq3)
    elseif s_abs < 4.0
        return horner(s_abs, coef, :eq4)
    else
        return 0.0
    end
end