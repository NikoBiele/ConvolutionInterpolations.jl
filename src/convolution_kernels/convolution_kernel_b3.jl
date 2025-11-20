# see 'docstring.jl' for documentation
const b3_coefs = Dict(
    :eq1 => [1//1, 0//1, -7//3, 4//3],
    :eq2 => [15//6, -59//12, 3//1, -7//12],
    :eq3 => [-3//2, 21//12, -2//3, 1//12]
)

function (::ConvolutionKernel{:b3})(s::T) where {T}
    s_abs = abs(s)
    if s_abs < 1.0
        return horner(s_abs, b3_coefs, :eq1, T)
    elseif s_abs < 2.0
        return horner(s_abs, b3_coefs, :eq2, T)
    elseif s_abs < 3.0
        return horner(s_abs, b3_coefs, :eq3, T)
    else
        return zero(T)
    end
end