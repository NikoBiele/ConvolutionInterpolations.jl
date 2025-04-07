# see 'docstring.jl' for documentation
function (::ConvolutionKernel{:a3})(s::T) where {T}
    s_abs = abs(s)
    coefs = Dict(
        :eq1 => [1, 0, -5/2, 3/2],
        :eq2 => [2, -4, 5/2, -1/2],
    )
    if s_abs < 1.0
        return horner(s_abs, coefs, :eq1)
    elseif s_abs < 2.0
        return horner(s_abs, coefs, :eq2)
    else
        return 0.0
    end
end