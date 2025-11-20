# see 'docstring.jl' for documentation
const a0_coefs = Dict(
    :eq1 => [1//1]
)

function (::ConvolutionKernel{:a0})(s::T) where {T}
    s_abs = abs(s)
    if s_abs < 0.5
        return horner(s_abs, a0_coefs, :eq1, T)
    else
        return zero(T)
    end
end