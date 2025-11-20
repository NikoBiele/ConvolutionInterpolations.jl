# see 'docstring.jl' for documentation
const a1_coefs = Dict(
    :eq1 => [1//1, -1//1]
)

function (::ConvolutionKernel{:a1})(s::T) where {T}
    s_abs = abs(s)
    if s_abs < 1.0
        return horner(s_abs, a1_coefs, :eq1, T)
    else
        return zero(T)
    end
end