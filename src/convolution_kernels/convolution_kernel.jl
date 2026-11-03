# The kernel functor: K(s), its derivatives and its antiderivatives at a normalized offset.
# Generated once per (kernel, order, element type) from the exact pieces of column_polynomials.jl.

"""
    (::ConvolutionKernel{K,DO})(s)

The kernel `K` at the normalized offset `s`: its `DO`-th derivative for `DO > 0`, the kernel
itself for `DO = 0`, and its m-fold antiderivative for `DO = -m` (see `_kernel_antiderivative`).
Generated once per (kernel, order, T) from the exact pieces of `_kernel_pieces_exact`,
differentiated exactly: Horner's scheme on each piece in its local coordinate, an if-chain over
the pieces from the innermost outwards, and the sign of `s` for odd derivatives (the kernel is
a function of |s|). For bits types the coefficients are rounded once to `T` and are compile-time
constants; other types such as `Rational{BigInt}` or `BigFloat` convert the exact rationals at
evaluation. `:a0`, whose single piece breaks at |s| = ½, is given directly.
"""
@generated function (::ConvolutionKernel{K,DO})(s::T) where {K,DO,T}
    if DO < 0
        return :(_kernel_antiderivative(Val($(QuoteNode(K))), Val($(-DO)), s))
    end
    if K === :a0
        DO == 0 || throw(ArgumentError("kernel :a0 has no derivatives, got derivative = $DO"))
        return :(abs(s) < T(1//2) ? one(T) : zero(T))
    end
    # an exact coefficient as a literal of T for bits types, converted at run time otherwise
    lit(c) = isbitstype(T) ? T(c) : :(T($c))
    pieces = _kernel_pieces_exact(K, 0)
    for _ in 1:DO
        pieces = map(_poly_derivative, pieces)
    end
    # Horner's scheme for one piece in its local coordinate t = s_abs − (i−1) ∈ [0, 1)
    horner(p, i) = begin
        q = _shift_polynomial(p, i - 1, 1)
        foldr((c, acc) -> :(muladd($acc, s_abs - $(i - 1), $(lit(c)))), q[1:end-1];
              init = lit(q[end]))
    end
    # piece i covers [i−1, i): an if/elseif chain from the innermost piece outwards, zero beyond
    value = :(zero(T))
    for i in length(pieces):-1:1
        value = Expr(:if, :(s_abs < $i), horner(pieces[i], i), value)
    end
    if isodd(DO)
        signed = :(sign(s) * value)
    else
        signed = :(value)
    end
    return quote
        s_abs = abs(s)
        value = $value
        return $signed
    end
end

# ---------------------------------------------------------------------------------------------
# Antiderivative kernels
# ---------------------------------------------------------------------------------------------

"""
    _kernel_antiderivative(Val(kernel), Val(m), s)

Kₘ(s), the m-fold antiderivative (m ≥ 1) of `kernel` in the symmetric Cauchy convention of
`_kernel_pieces_exact`: the exact piece polynomial inside the support, the far field
sgn(s)·|s|^(m−1)/(2(m−1)!) outside it, and Kₘ(−s) = (−1)^m Kₘ(s). The far field holds for m up
to `_max_integral_order[kernel]`. `:a0` is given in closed form, ½[(s+½)^m + (s−½)^m]/m! inside
|s| < ½. Generated once per (kernel, m, T): for bits types the coefficients are rounded once to
T and are compile-time constants; other types such as BigFloat convert the exact rationals at
evaluation, at the precision then in effect.
"""
@generated function _kernel_antiderivative(::Val{kernel}, ::Val{m}, s::T) where {kernel,m,T}
    m >= 1 || throw(ArgumentError("integral order must be at least 1, got $m"))
    # an exact coefficient as a literal of T for bits types, converted at run time otherwise
    lit(c) = isbitstype(T) ? T(c) : :(T($c))
    far = :(s_abs^$(m - 1) / $(lit(2 * factorial(big(m - 1)))))
    if kernel == :a0
        inside = :(((s_abs + T(1//2))^$m + (s_abs - T(1//2))^$m) / $(lit(2 * factorial(big(m)))))
        value = Expr(:if, :(s_abs < T(1//2)), inside, far)
    else
        pieces = _kernel_pieces_exact(kernel, m)
        # Horner's scheme for one piece in its local coordinate t = s − (i−1) ∈ [0, 1), which keeps
        # the coefficients small; in s itself the wide kernels cancel to ~1e-9 (entry k of the
        # shifted piece multiplies t^(k−1))
        horner(p, i) = begin
            q = _shift_polynomial(p, i - 1, 1)
            foldr((c, acc) -> :(muladd($acc, s_abs - $(i - 1), $(lit(c)))), q[1:end-1];
                  init = lit(q[end]))
        end
        # piece i covers [i−1, i): an if/elseif chain from the innermost piece outwards
        value = far
        for i in length(pieces):-1:1
            value = Expr(:if, :(s_abs < $i), horner(pieces[i], i), value)
        end
    end
    if isodd(m)
        signed = :(s < 0 ? -value : value)
    else
        signed = :(value)
    end
    return quote
        s_abs = abs(s)
        value = $value
        return $signed
    end
end