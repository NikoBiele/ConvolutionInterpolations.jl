# see 'docstring.jl' for documentation
function (::ConvolutionKernel{:a0})(s::T) where {T}
    s_abs = abs(s)
    coefs = Dict(
        :eq1 => [1]
    )
    if s_abs < 0.5
        return horner(s_abs, coefs, :eq1)
    else
        return 0.0
    end
end