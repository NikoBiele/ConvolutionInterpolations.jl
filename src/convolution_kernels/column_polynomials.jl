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

# ---------------------------------------------------------------------------------------------
# Higher integral orders
# ---------------------------------------------------------------------------------------------

# Highest integral order per kernel: m − 1 must not exceed the degree up to which the kernel's
# moments vanish, so that outside the support Kₘ is exactly the integrated step (the far field
# used by the anchoring and the tails). Validated exactly for every kernel up to these orders.
const _max_integral_order = Dict(:a0 => 2, :a1 => 2,
                                 :a3 => 4, :a4 => 4, :a5 => 4, :a7 => 4,
                                 :b5 => 6,
                                 :b7 => 8, :b9 => 8, :b11 => 8, :b13 => 8)

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

"""
    _anchor_taylor_table(T, kernel, m, eqs)

Anchoring data for integral order m: row j (1 ≤ j ≤ 2·eqs − 1, the coefficients within reach
of the anchor) holds K_{m−r}(eqs − j) for r = 0 … m−1, exact and rounded once to T. The anchored
weight of such a coefficient at index position u is K_m(u − j) − Σ_r K_{m−r}(eqs − j)·(u − eqs)^r / r!.
"""
function _anchor_taylor_table(::Type{T}, kernel::Symbol, m::Int, eqs::Int) where {T}
    table = Matrix{T}(undef, 2eqs - 1, m)
    for j in 1:2eqs-1, r in 0:m-1
        table[j, r+1] = T(_kernel_value_exact(kernel, m - r, eqs - j))
    end
    return table
end

"""
    _near_anchor_tail_entry(kernel, m, eqs, j)

Exact coefficients in t (entry k+1 multiplies t^k) of the left-tail weight of a coefficient j near
the anchor (j ≤ 2·eqs − 1), in the cell where it enters the left tail (i = j + eqs, so that
u = t + j + eqs): the far field (u − j)^(m−1) / (2(m−1)!) minus the Taylor polynomial
Σ_r K_{m−r}(eqs − j)·(u − eqs)^r / r!. Both parts are large far from the anchor, so their
difference is formed exactly.
"""
function _near_anchor_tail_entry(kernel::Symbol, m::Int, eqs::Int, j::Int)
    coef = zeros(Rational{BigInt}, m)
    fac = factorial(big(m - 1))
    # far field: (u − j)^(m−1) / (2(m−1)!) with u − j = t + eqs
    for k in 0:m-1
        coef[k+1] += binomial(big(m - 1), big(k)) * big(eqs)^(m - 1 - k) // (2 * fac)
    end
    # Taylor polynomial at the anchor: Σ_r K_{m−r}(eqs − j)·(u − eqs)^r / r! with u − eqs = t + j
    for r in 0:m-1
        value = _kernel_value_exact(kernel, m - r, eqs - j)
        for k in 0:r
            coef[k+1] -= value * binomial(big(r), big(k)) * big(j)^(r - k) // factorial(big(r))
        end
    end
    return coef
end

"""
    _left_tail_polynomial(coefs, kernel, m, eqs, d)

Left tail of integral order m along dimension d of `coefs`, as m arrays of the size of `coefs`:
array k+1 at position l (along d) holds the coefficient of t^k of Σ_{j ≤ l} c_j·W_j(t + l + eqs),
the contribution of all coefficients left of the stencil of cell i = l + eqs. Coefficients far
from the anchor have W_j = (u − j)^(m−1) / (m−1)!; those near it have the exact entry polynomial
of `_near_anchor_tail_entry`.

Built by the shifted prefix sum Λ(l) = S·Λ(l−1) + c_l·e_l, where S re-expands a polynomial in t
around the next cell (p(t) ↦ p(t + 1)). For m = 1 this is the plain prefix sum of c·(½ − K̃(eqs − l)).
"""
function _left_tail_polynomial(coefs::AbstractArray{T,N}, kernel::Symbol, m::Int, eqs::Int,
                               d::Int) where {T,N}
    n = size(coefs, d)
    fac = factorial(big(m - 1))
    # entry polynomial of a coefficient far from the anchor: (t + eqs)^(m−1) / (m−1)!
    far_entry = [T(binomial(big(m - 1), big(k)) * big(eqs)^(m - 1 - k) // fac) for k in 0:m-1]
    # entry polynomials of the coefficients near the anchor, exact and rounded once
    near_entry = [T.(_near_anchor_tail_entry(kernel, m, eqs, j)) for j in 1:min(2eqs - 1, n)]
    # Taylor shift by one cell: new coefficient k = Σ_{q ≥ k} binomial(q, k)·old coefficient q
    shift = [T(binomial(q, k)) for k in 0:m-1, q in 0:m-1]

    tails = [similar(coefs) for _ in 1:m]
    for l in 1:n
        entry = l <= length(near_entry) ? near_entry[l] : far_entry
        c_l = selectdim(coefs, d, l)
        for k in 1:m
            acc = entry[k] .* c_l
            if l > 1
                for q in k:m
                    acc = acc .+ shift[k, q] .* selectdim(tails[q], d, l - 1)
                end
            end
            selectdim(tails[k], d, l) .= acc
        end
    end
    return tails
end

"""
    _build_region_tails(coefs, kernels, orders, eqs)

Left tails of every region of an integral evaluation. The integral dimensions are those with
`orders[d] < 0` (of order −orders[d]). Region `mask` (1 ≤ mask ≤ 2^n − 1 for n integral
dimensions) contains the coefficients left of the stencil in the integral dimensions whose bit
is set (bit b ↔ the b-th integral dimension), and within the stencil in all others.

Each region is a vector of arrays of the size of `coefs`, one per combination of powers of the
positions within the cell: for the region's dimensions d₁ < d₂ < …, array index
1 + Σ kᵢ·Π_{i' > i} m_{d_i'} holds the coefficient of Π t_{dᵢ}^{kᵢ} (the last dimension's power
varying fastest). Built separably, one dimension of the region at a time.
"""
function _build_region_tails(coefs::AbstractArray{T,N}, kernels::NTuple{N,Symbol},
                             orders::NTuple{N,Int}, eqs::NTuple{N,Int}) where {T,N}
    int_dims = [d for d in 1:N if orders[d] < 0]
    n = length(int_dims)
    tails = Vector{Vector{Array{T,N}}}(undef, 2^n - 1)
    for mask in 1:2^n-1
        arrays = Array{T,N}[copy(coefs)]
        for (b, d) in enumerate(int_dims)
            (mask >> (b - 1)) & 1 == 1 || continue
            m = -orders[d]
            expanded = Array{T,N}[]
            for array in arrays
                append!(expanded, _left_tail_polynomial(array, kernels[d], m, eqs[d], d))
            end
            arrays = expanded
        end
        tails[mask] = arrays
    end
    return tails
end

"""
    _anchored_stencil_weights(Val(kernel), Val(M), t, i, eqs, taylor)

Anchored weights of the 2·eqs stencil coefficients of one integral dimension of order M, for a
point at position t within cell i, in eager order: entry k is the weight of coefficient
j = i − eqs + k. The anchored weight is K_M(u − j) − Σ_r K_{M−r}(eqs − j)·(u − eqs)^r / r!
(u = i + t): the column polynomial of K_M minus the Taylor polynomial at the anchor, which for
coefficients far from the anchor (j ≥ 2·eqs) is the far field −(u − j)^(M−1) / (2(M−1)!), and
for those near it comes from the exact table `taylor` (see `_anchor_taylor_table`).
"""
@inline function _anchored_stencil_weights(::Val{kernel}, ::Val{M}, t::T, i::Int, eqs::Int,
                                           taylor::AbstractMatrix{T}) where {kernel,M,T}
    # K_M weights of all columns: column c holds K_M(c − 1 − eqs + t), the weight of j = i + eqs + 1 − c
    w = _kernel_weights(Val(kernel), Val(-M), t)
    v = T(i - eqs) + t                                   # u − eqs: position relative to the anchor
    return ntuple(k -> _anchored_weight(w, k, t, i, eqs, v, taylor, Val(M)), Val(length(w)))
end

# Anchored weight of stencil entry k (coefficient j = i − eqs + k), see `_anchored_stencil_weights`
@inline function _anchored_weight(w::NTuple{K,T}, k::Int, t::T, i::Int, eqs::Int, v::T,
                                  taylor::AbstractMatrix{T}, ::Val{M}) where {K,T,M}
    c = K + 1 - k                                        # column holding coefficient j
    j = i - eqs + k
    if j >= 2eqs
        # far from the anchor: the Taylor polynomial is the far field −(u − j)^(M−1) / (2(M−1)!)
        s = t + T(c - 1 - eqs)                           # u − j, small within the stencil
        return w[c] + s^(M - 1) / T(2 * factorial(M - 1))
    else
        # near the anchor: Σ_r K_{M−r}(eqs − j)·v^r / r! from the exact table
        acc = zero(T)
        vr = one(T)
        for r in 0:M-1
            acc += taylor[j, r + 1] * vr / T(factorial(r))
            vr *= v
        end
        return w[c] - acc
    end
end

"""
    _near_anchor_entries(T, kernel, m, eqs)

The entry polynomials of `_near_anchor_tail_entry` for all near-anchor coefficients
j = 1 … 2·eqs − 1, rounded once to T: row j holds the coefficients of t^0 … t^(m−1). Used to
evaluate left-of-stencil weights directly where no tails are stored (more than 3 integral
dimensions).
"""
function _near_anchor_entries(::Type{T}, kernel::Symbol, m::Int, eqs::Int) where {T}
    entries = Matrix{T}(undef, 2eqs - 1, m)
    for j in 1:2eqs-1
        entries[j, :] .= T.(_near_anchor_tail_entry(kernel, m, eqs, j))
    end
    return entries
end