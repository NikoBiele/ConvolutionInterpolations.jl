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

"""
    _column_weights_per_dim(Val(kernels), Val(derivatives), τ)

Kernel weights of every dimension, `(_kernel_weights(kernels[1], derivatives[1], τ[1]), …)`.
Generated so that each dimension's kernel and derivative order are literal constants, which
keeps the result type-stable even when kernels differ between dimensions.
"""
@generated function _column_weights_per_dim(::Val{DG}, ::Val{DO}, τ::NTuple{N,T}) where {DG,DO,N,T}
    calls = [:(_kernel_weights(Val($(QuoteNode(DG[d]))), Val($(DO[d])), τ[$d])) for d in 1:N]
    return :(tuple($(calls...)))
end

"""
    _antiderivative_weight(w, i, eqs, j)

K̃ weight of coefficient j for a point in cell i, from the column weights `w` of that dimension
at τ = t (see `_kernel_weights` with derivative −1): column c = i + eqs + 1 − j inside the
stencil, and the saturated values outside it (+½ left of the stencil, −½ right of it).
"""
@inline function _antiderivative_weight(w::NTuple{K,T}, i::Int, eqs::Int, j::Int) where {K,T}
    c = i + eqs + 1 - j
    c > K && return T(1//2)
    c < 1 && return -T(1//2)
    return @inbounds w[c]
end

# Val((−1, −1, …, −1)) for N dimensions, as a compile-time constant
@generated _integral_orders(::Val{N}) where {N} = :(Val($(ntuple(_ -> -1, N))))

# Weight of stencil column `col`, or zero outside the stencil
@inline _stencil_weight(w::NTuple{K,T}, col::Int) where {K,T} =
    1 <= col <= K ? (@inbounds w[col]) : zero(T)

"""
    _mixed_weights(itp, x, i, Val(DO))

Kernel weights of every dimension of a mixed evaluation in cells `i`: integral dimensions
(DO[d] == −1) at τ = t, in the antiderivative orientation (see `_antiderivative_weight`); all
other dimensions at τ = diff_right, in the evaluation orientation (column k ↔ coefficient
i − eqs + k).
"""
@inline function _mixed_weights(itp, x::NTuple{N,T}, i::NTuple{N,Int}, ::Val{DO}) where {N,T,DO}
    τ = ntuple(Val(N)) do d
        DO[d] == -1 ? T((x[d] - itp.knots[d][1]) / itp.h[d] + one(T) - T(i[d])) :
                      T(one(T) - (x[d] - itp.knots[d][i[d]]) / itp.h[d])
    end
    return _column_weights_per_dim(Val(_kernel_sym(itp.kernel_sym)), Val(DO), τ)
end

"""
    _derivative_weight_product(w, idx, i, eqs, Val(DO))

Product of the kernel weights of all non-integral dimensions for coefficient index `idx`.
Generated so each dimension's weight tuple is accessed with a literal index.
"""
@generated function _derivative_weight_product(w::Tuple, idx::NTuple{N,Int}, i::NTuple{N,Int},
                                               eqs::NTuple{N,Int}, ::Val{DO}) where {N,DO}
    factors = [:(_stencil_weight(w[$d], idx[$d] - i[$d] + eqs[$d])) for d in 1:N if DO[d] != -1]
    return :(*($(factors...)))
end

"""
    _mixed_weight_product(w, idx, i, eqs, left_values, Val(DO))

Product over all dimensions for coefficient index `idx`: anchored K̃ weights in integral
dimensions, kernel weights (zero outside the stencil) in all others.
"""
@generated function _mixed_weight_product(w::Tuple, idx::NTuple{N,Int}, i::NTuple{N,Int},
                                          eqs::NTuple{N,Int}, left_values::Tuple, ::Val{DO}) where {N,DO}
    factors = map(1:N) do d
        DO[d] == -1 ?
            :(_antiderivative_weight(w[$d], i[$d], eqs[$d], idx[$d]) - left_values[$d][idx[$d]]) :
            :(_stencil_weight(w[$d], idx[$d] - i[$d] + eqs[$d]))
    end
    return :(*($(factors...)))
end

# Indices of the integral dimensions (DO[d] == −1), as compile-time constants
@generated _integral_dims(::Val{DO}) where {DO} = :($(Tuple(findall(==(-1), collect(DO)))))