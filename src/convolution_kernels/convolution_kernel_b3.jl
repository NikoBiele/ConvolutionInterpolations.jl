# see 'docstring.jl' for documentation
function (::ConvolutionKernel{:b3})(s::T) where {T}
    s_abs = abs(s)
    coefs = Dict(
        :eq1 => [1, 0, -7/3, 4/3],
        :eq2 => [15/6, -59/12, 3, -7/12],
        :eq3 => [-3/2, 21/12, -2/3, 1/12]
    )
    if s_abs < 1.0
        return horner(s_abs, coefs, :eq1)
    elseif s_abs < 2.0
        return horner(s_abs, coefs, :eq2)
    elseif s_abs < 3.0
        return horner(s_abs, coefs, :eq3)
    else
        return 0.0
    end
end