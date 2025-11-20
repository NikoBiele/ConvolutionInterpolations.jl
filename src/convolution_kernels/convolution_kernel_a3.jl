# see 'docstring.jl' for documentation
const a3_coefs = Dict(
    :eq1 => [1//1, 0//1, -5//2, 3//2],
    :eq2 => [2//1, -4//1, 5//2, -1//2]
)

function (::ConvolutionKernel{:a3})(s::T) where {T}
    s_abs = abs(s)
    if s_abs < 1.0
        return horner(s_abs, a3_coefs, :eq1, T)
    elseif s_abs < 2.0
        return horner(s_abs, a3_coefs, :eq2, T)
    else
        return zero(T)
    end
end