# see 'docstring.jl' for documentation
const a3_coefs = Dict(
    :eq1 => [1//1, 0//1, -5//2, 3//2],
    :eq2 => [2//1, -4//1, 5//2, -1//2]
)
const a3_coefs_d1 = Dict(
    :eq1 => [0//1*1//1, 0//1, 2//1*-5//2, 3//1*3//2],
    :eq2 => [0//1*2//1, -4//1, 2//1*5//2, 3//1*-1//2]
)
# const a3_coefs_i1 = Dict(
#     :eq1 => [0//1, 1//1*1//1, 0//1*1//2, -5//2*1//3, 3//2*1//4],
#     :eq2 => [-1//6, 2//1*1//1, -4//1*1//2, 5//2*1//3, -1//2*1//4]
# )

function (::ConvolutionKernel{:a3,DO})(s::T) where {T,DO}
    a3_coefs_in = if DO == 0
        a3_coefs
    elseif DO == 1
        a3_coefs_d1
    elseif DO == -1
        a3_coefs_i1
    else
        error("kernel :a3 supports differentiation orders -1, 0, 1, but got $DO")
    end
    s_abs = abs(s)
    # if DO >= 0
        if s_abs < 1.0
            return horner(s_abs, a3_coefs_in, :eq1, T, DO) * (isodd(DO) ? T(sign(s)) : one(T))
        elseif s_abs < 2.0
            return horner(s_abs, a3_coefs_in, :eq2, T, DO) * (isodd(DO) ? T(sign(s)) : one(T))
        else
            return zero(T)
        end
    # else
    #     if s_abs < 1.0
    #         return horner(s_abs, a3_coefs_in, :eq1, T, DO) * sign(s) #* (isodd(DO) ? T(sign(s)) : one(T))
    #     elseif s_abs < 2.0
    #         return horner(s_abs, a3_coefs_in, :eq2, T, DO) * sign(s) #* (isodd(DO) ? T(sign(s)) : one(T))
    #     else
    #         return one(T)/2 * sign(s)
    #     end
    # end
end