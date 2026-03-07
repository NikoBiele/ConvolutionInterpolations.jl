"""
    (::ConvolutionKernel{:a1,DO})(s)

Linear kernel. Support [-1, 1], 1 piece.
C0 continuous, 1st-order accuracy. Equivalent to standard linear interpolation.
"""

const a1_coefs = Dict(
    :eq1 => [1//1, -1//1]
)
const a1_coefs_i1 = Dict(
    :eq1 => [0//1, 1//1, -1//2]
)

function (::ConvolutionKernel{:a1,DO})(s::T) where {T,DO}
    if DO > 0
        error("kernel :a1 does not support differentiation orders > 0, but got $DO")
    end
    s_abs = abs(s)
    if DO == -1
        if s_abs >= 1
            return T(1//2) * T(sign(s))
        else
            return horner(s_abs, a1_coefs_i1, :eq1, T, 0) * T(sign(s))
        end
    end
    if s_abs < 1.0
        return horner(s_abs, a1_coefs, :eq1, T, DO)
    else
        return zero(T)
    end
end