# see 'docstring.jl' for documentation
const a = 3//64
const a5_coef = Dict(
    # 3 equation quintic
    :eq1 => [1//1, 0//1, 8//1*a-5//2, 0//1, -18//1*a+45//16, 10//1*a-21//16],
    :eq2 => [-66//1*a+5//1, 265//1*a-15//1, -392//1*a+35//2, 270//1*a-10//1, -88//1*a+45//16, 11//1*a-5//16],
    :eq3 => [-162//1*a, 297//1*a, -216//1*a, 78//1*a, -14//1*a, a]
)

function (::ConvolutionKernel{:a5})(s::T) where {T}
    s_abs = abs(s)
    if s_abs < 1.0
        return horner(s_abs, a5_coef, :eq1, T)
    elseif s_abs < 2.0
        return horner(s_abs, a5_coef, :eq2, T)
    elseif s_abs < 3.0
        return horner(s_abs, a5_coef, :eq3, T)
    else
        return zero(T)
    end
end