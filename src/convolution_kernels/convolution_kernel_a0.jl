# see 'docstring.jl' for documentation
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