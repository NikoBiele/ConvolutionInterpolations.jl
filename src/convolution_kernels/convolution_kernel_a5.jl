# see 'docstring.jl' for documentation
function (::ConvolutionKernel{:a5})(s::T) where {T}
    s_abs = abs(s)
    a = 3/64
    coef = Dict(
        # 3 equation quintic
        :eq1 => [1, 0, 8*a-5/2, 0, -18*a+45/16, 10*a-21/16], 
        :eq2 => [-66*a+5, 265*a-15, -392*a+35/2, 270*a-10, -88*a+45/16, 11*a-5/16],
        :eq3 => [-162*a, 297*a, -216*a, 78*a, -14*a, a],
    )
    if s_abs < 1.0
        return horner(s_abs, coef, :eq1)
    elseif s_abs < 2.0
        return horner(s_abs, coef, :eq2)
    elseif s_abs < 3.0
        return horner(s_abs, coef, :eq3)
    else
        return 0.0
    end
end