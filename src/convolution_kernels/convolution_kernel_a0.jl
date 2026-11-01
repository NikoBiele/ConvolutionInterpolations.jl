const a0_coefs = Dict(
    :eq1 => [1//1]
)

function (::ConvolutionKernel{:a0,DO})(s::T) where {T,DO}
    if DO > 0
        error("kernel :a0 does not support differentiation orders > 0, but got $DO")
    end
    if DO < 0
        # antiderivative of order −DO in closed form (see column_polynomials.jl)
        return _kernel_antiderivative(Val(:a0), Val(-DO), s)
    end
    s_abs = abs(s)
    if s_abs < T(1//2)
        return horner(s_abs, a0_coefs, :eq1, T, DO)
    else
        return zero(T)
    end
end