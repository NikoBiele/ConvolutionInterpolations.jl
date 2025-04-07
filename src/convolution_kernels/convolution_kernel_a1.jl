# see 'docstring.jl' for documentation
function (::ConvolutionKernel{:a1})(s::T) where {T}
    s_abs = abs(s)
    coefs = Dict(
        :eq1 => [1, -1]
    )
    if s_abs < 1.0
        return horner(s_abs, coefs, :eq1)
    else
        return 0.0
    end
end