"""
    (::ConvolutionKernel{Val{:a0},DO})(s)

Nearest neighbor kernel. Support [-0.5, 0.5], 1 piece.
Returns 1 inside the support, 0 outside. No continuity, 0th-order accuracy.
"""

const a0_coefs = Dict(
    :eq1 => [1//1]
)
# discontinuous

function (::ConvolutionKernel{:a0,DO})(s::T) where {T,DO}
    if DO > 0
        error("kernel :a0 does not supports differentiation orders > 0, but got $DO")
    end
    s_abs = abs(s)
    if s_abs < 0.5
        return horner(s_abs, a0_coefs, :eq1, T, DO)
    else
        return zero(T)
    end
end