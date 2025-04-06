# see 'docstring.jl' for documentation
function (::ConvolutionKernel{5})(s::T) where {T} # 5 equations 7th order accurate quintic
    s_abs = abs(s)
    coef = Dict(
        # 5 equation quintic, 7th order accurate
        :eq1 => [1, 0, -731/384, 0, 11423/7680, -4483/7680], 
        :eq2 => [2597/3840, 16931/7680, -14371/1920, 25933/3840, -6337/2560, 313/960], 
        :eq3 => [1211/256, -96221/7680, 4549/384, -20011/3840, 1675/1536, -169/1920], 
        :eq4 => [1681/1280, -7843/15360, -1463/3840, 2071/7680, -439/7680, 21/5120],
        :eq5 => [-3625/768, 5075/1024, -1595/768, 667/1536, -29/640, 29/15360]
    )
    if s_abs < 1.0
        return horner(s_abs, coef, :eq1)
    elseif s_abs < 2.0
        return horner(s_abs, coef, :eq2)
    elseif s_abs < 3.0
        return horner(s_abs, coef, :eq3)
    elseif s_abs < 4.0
        return horner(s_abs, coef, :eq4)
    elseif s_abs < 5.0
        return horner(s_abs, coef, :eq5)
    else
        return 0.0
    end
end