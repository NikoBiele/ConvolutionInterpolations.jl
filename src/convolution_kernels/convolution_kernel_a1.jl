const a1_coefs = Dict(
    :eq1 => [1//1, -1//1]
)

function (::ConvolutionKernel{:a1,DO})(s::T) where {T,DO}
    if DO > 0
        error("kernel :a1 does not support differentiation orders > 0, but got $DO")
    end
    if DO < 0
        # antiderivative of order −DO, exact pieces and far field (see column_polynomials.jl)
        return _kernel_antiderivative(Val(:a1), Val(-DO), s)
    end
    s_abs = abs(s)
    if s_abs < 1.0
        return horner(s_abs, a1_coefs, :eq1, T, DO)
    else
        return zero(T)
    end
end