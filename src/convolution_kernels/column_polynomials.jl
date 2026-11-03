# Exact column polynomials of the convolution kernels.
#
# For an evaluation point with fractional position t ∈ [0, 1] within its grid cell, the
# coefficient at integer offset o ∈ {−eqs, …, eqs−1} is weighted by K(o + t). Kernel pieces
# break only at integer |s|, so on each such column K(o + t) is a single polynomial in t.
# Column i corresponds to o = i − 1 − eqs, the same ordering as the kernel tables.
#
# Everything is derived exactly in Rational{BigInt} from each kernel's base coefficients.
# Derivatives are t-derivatives of the columns (since s = o + t, d/ds = d/dt). Integrals use
# the m-fold antiderivative Kₘ in the symmetric Cauchy convention
#     Kₘ(s) = ½ [∫₋∞ˢ − ∫ₛ^∞] (s−t)^(m−1)/(m−1)! K(t) dt,
# for which Kₘ' = Kₘ₋₁ and Kₘ(−s) = (−1)^m Kₘ(s).
#
# :a0 is not covered: its pieces break at half-integers, so its columns are not polynomials.

# Evaluate polynomial p (entry k multiplies x^(k−1)) at x by Horner's scheme
function _poly_eval(p::AbstractVector, x)
    acc = zero(x)
    for k in length(p):-1:1
        acc = acc * x + p[k]
    end
    return acc
end

# Derivative of polynomial p
_poly_derivative(p::Vector{Rational{BigInt}}) =
    length(p) == 1 ? [zero(Rational{BigInt})] : [k * p[k+1] for k in 1:length(p)-1]

# Antiderivative of polynomial p with zero constant term
_poly_integrate(p::Vector{Rational{BigInt}}) =
    vcat(zero(Rational{BigInt}), [p[k] / k for k in eachindex(p)])

# Coefficients in t of p(a + σ t)
function _shift_polynomial(p::Vector{Rational{BigInt}}, a::Integer, σ::Integer)
    n = length(p)
    q = zeros(Rational{BigInt}, n)
    for i in 0:n-1, k in 0:i
        q[k+1] += p[i+1] * binomial(big(i), big(k)) * big(a)^(i - k) * big(σ)^k
    end
    return q
end

# Pieces of Kₘ on s ≥ 0 inside the support, piece i on [i−1, i), for m ≥ 0
function _kernel_pieces_exact(kernel::Symbol, m::Int)
    base = getfield(@__MODULE__, Symbol(kernel, "_coefs"))
    P = length(base)
    base_pieces = [Rational{BigInt}.(base[Symbol("eq", i)]) for i in 1:P]

    # Half moment ∫₀^P tᵏ K(t) dt, which fixes Kₘ(0) for even m
    half_moment(k) = sum(1:P) do i
        q = _poly_integrate(vcat(zeros(Rational{BigInt}, k), base_pieces[i]))
        _poly_eval(q, big(i) // 1) - _poly_eval(q, big(i - 1) // 1)
    end

    pieces = base_pieces
    for level in 1:m
        # Kₘ(0): zero for odd m (odd function), the half moment for even m
        start = isodd(level) ? zero(Rational{BigInt}) :
                               half_moment(level - 1) / factorial(big(level - 1))
        integrated = Vector{Vector{Rational{BigInt}}}(undef, P)
        for i in 1:P
            q = _poly_integrate(pieces[i])                  # integrate the previous level
            q[1] += start - _poly_eval(q, big(i - 1) // 1)  # continuity at the left break
            integrated[i] = q
            start = _poly_eval(q, big(i) // 1)              # value carried to the next piece
        end
        pieces = integrated
    end
    return pieces
end

"""
    _column_polynomials_exact(kernel, derivative)

Exact polynomials in t ∈ [0, 1] of the kernel derivative of order `derivative` (≥ 0), or of
the m-fold antiderivative Kₘ for `derivative = −m`, on each of the 2·eqs columns. Column i
belongs to offset o = i − 1 − eqs. Returns a vector of coefficient vectors (entry k
multiplies t^(k−1)).
"""
function _column_polynomials_exact(kernel::Symbol, derivative::Int)
    kernel == :a0 &&
        throw(ArgumentError("column polynomials are not defined for :a0 (pieces break at half-integers)"))
    eqs = get_equations_for_degree(kernel)
    m = max(-derivative, 0)
    pieces = _kernel_pieces_exact(kernel, m)
    columns = Vector{Vector{Rational{BigInt}}}(undef, 2 * eqs)
    for i in 1:2*eqs
        o = i - 1 - eqs
        if o >= 0
            column = _shift_polynomial(pieces[o + 1], o, 1)    # |s| = o + t
        else
            column = _shift_polynomial(pieces[-o], -o, -1)     # |s| = −o − t
            isodd(m) && (column = -column)                     # Kₘ is odd for odd m
        end
        for _ in 1:max(derivative, 0)
            column = _poly_derivative(column)                  # d/ds = d/dt
        end
        columns[i] = column
    end
    return columns
end

"""
    _kernel_value_exact(kernel, q, σ)

Exact value of K_q(σ), the q-fold antiderivative (q ≥ 1) of `kernel` in the symmetric Cauchy
convention, at an integer offset σ. Inside the support it is the exact integrated kernel piece;
outside (|σ| ≥ eqs) it is the far field sgn(σ)·σ^(q−1)/(2(q−1)!), which holds for q up to
`_max_integral_order[kernel]`. `:a0`, whose support has half-width ½, is given in closed form.
"""
function _kernel_value_exact(kernel::Symbol, q::Int, σ::Integer)
    s = big(σ)
    # far field: the q-fold integrated step
    far_field() = sign(s) * s^(q - 1) // (2 * factorial(big(q - 1)))
    if kernel == :a0
        # K_q(0): zero for odd q, the half moment (½)^q / q! for even q
        σ == 0 && return isodd(q) ? zero(Rational{BigInt}) : (big(1)//2)^q // factorial(big(q))
        return far_field()
    end
    eqs = get_equations_for_degree(kernel)
    abs(σ) >= eqs && return far_field()
    # inside the support: piece |σ| + 1 (piece i covers [i−1, i)) evaluated at |σ|
    piece = _kernel_pieces_exact(kernel, q)[abs(σ) + 1]
    value = _poly_eval(piece, Rational{BigInt}(abs(s)))
    return (σ < 0 && isodd(q)) ? -value : value
end

# ---------------------------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------------------------

"""
    _column_rows(Val(kernel), Val(derivative), T)

The column polynomials of `kernel` at `derivative` order, rounded once to type `T`, as
`rows[d][k]` = coefficient of τ^(d−1) for column k. Generated once per (kernel, derivative, T)
from the exact derivation, so the coefficients are compile-time constants in the calling code.
"""
@generated function _column_rows(::Val{kernel}, ::Val{derivative}, ::Type{T}) where {kernel,derivative,T}
    columns = _column_polynomials_exact(kernel, derivative)
    K = length(columns)
    D = maximum(length, columns)
    rows = ntuple(d -> ntuple(k -> d <= length(columns[k]) ? T(columns[k][d]) : zero(T), K), D)
    return :($rows)
end

# One Horner step for all K columns at once: w ← w·τ + row. Kept as a separate function so the
# closure captures only unchanging arguments (capturing a reassigned variable would box it).
@inline _horner_step(w::NTuple{K,T}, τ::T, row::NTuple{K,T}) where {K,T} =
    ntuple(k -> muladd(w[k], τ, row[k]), Val(K))

"""
    _column_weights(rows, τ)

Kernel weights of all K columns at position τ: Horner's scheme run across the columns at once,
one vector step per polynomial degree.
"""
@inline function _column_weights(rows::NTuple{D,NTuple{K,T}}, τ::T) where {D,K,T}
    w = rows[D]
    for d in D-1:-1:1
        w = _horner_step(w, τ, rows[d])
    end
    return w
end

"""
    _kernel_weights(Val(kernel), Val(derivative), τ)

Kernel weights of all 2·eqs columns of one dimension at τ = diff_right (column k ↔ coefficient
offset + k): exact column polynomials for every kernel except `:a0`, whose nearest-neighbour
weights are given directly.
"""
@inline _kernel_weights(::Val{kernel}, ::Val{derivative}, τ::T) where {kernel,derivative,T} =
    _column_weights(_column_rows(Val(kernel), Val(derivative), T), τ)

# :a0 (nearest neighbour) is not a polynomial on a column, but its weights are trivial. With
# τ = 1 − t, column 1 (coefficient i) is selected for t < 1/2 and column 2 (coefficient i+1)
# otherwise, matching the dedicated :a0 evaluators (`x_diff_left < 0.5` selects coefs[i]).
@inline _kernel_weights(::Val{:a0}, ::Val{0}, τ::T) where {T} =
    τ > T(1//2) ? (one(T), zero(T)) : (zero(T), one(T))

# :a0 antiderivative K̃(s) = clamp(s, −½, ½), which saturates inside each column:
# column 1 (o = −1) holds K̃(τ − 1), column 2 (o = 0) holds K̃(τ).
@inline function _kernel_weights(::Val{:a0}, ::Val{-1}, τ::T) where {T}
    half = T(1//2)
    return τ > half ? (τ - one(T), half) : (-half, τ)
end

# :a0 second antiderivative K₂(s) = s²/2 + 1/8 for |s| ≤ ½ and |s|/2 beyond:
# column 1 (o = −1) holds K₂(τ − 1), column 2 (o = 0) holds K₂(τ).
@inline function _kernel_weights(::Val{:a0}, ::Val{-2}, τ::T) where {T}
    half = T(1//2)
    eighth = T(1//8)
    k1 = τ > half ? (τ - one(T))^2 / 2 + eighth : (one(T) - τ) / 2
    k2 = τ <= half ? τ^2 / 2 + eighth : τ / 2
    return (k1, k2)
end

"""
    _column_weights_per_dim(Val(kernels), Val(derivatives), τ)

Kernel weights of every dimension, `(_kernel_weights(kernels[1], derivatives[1], τ[1]), …)`.
Generated so that each dimension's kernel and derivative order are literal constants, which
keeps the result type-stable even when kernels differ between dimensions.
"""
@generated function _column_weights_per_dim(::Val{KS}, ::Val{DO}, τ::NTuple{N,T}) where {KS,DO,N,T}
    calls = [:(_kernel_weights(Val($(QuoteNode(KS[d]))), Val($(DO[d])), τ[$d])) for d in 1:N]
    return :(tuple($(calls...)))
end