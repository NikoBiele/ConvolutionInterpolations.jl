"""
    (::ConvolutionKernel{:a0,DO})(s)

Nearest neighbor kernel. Support [-0.5, 0.5], 1 piece.
Returns 1 inside the support, 0 outside. No continuity, 0th-order accuracy.
Antiderivative (DO=-1): linear ramp through origin, saturates to ±1/2 outside support.
"""

const a0_coefs = Dict(
    :eq1 => [1//1]
)

const a0_coefs_i1 = Dict(
    :eq1 => [0//1, 1//1]
)

function (::ConvolutionKernel{:a0,DO})(s::T) where {T,DO}
    if DO > 0
        error("kernel :a0 does not support differentiation orders > 0, but got $DO")
    end
    s_abs = abs(s)
    if DO == -1
        if s_abs >= T(1//2)
            return T(1//2) * T(sign(s))
        else
            return horner(s_abs, a0_coefs_i1, :eq1, T, 0) * T(sign(s))
        end
    end
    if s_abs < T(1//2)
        return horner(s_abs, a0_coefs, :eq1, T, DO)
    else
        return zero(T)
    end
end