# Anchoring and stored tails of the antiderivatives.
#
# An m-fold antiderivative along an axis is anchored at zero (with its first m − 1
# derivatives) at the leftmost interior knot. Every coefficient left of a cell's stencil
# contributes through the far field of Kₘ, a polynomial in the position within the cell, so
# those contributions are stored per cell as polynomial tails (`_build_region_tails`) and
# evaluated in O(1); coefficients within the stencil get anchored weights
# (`_anchored_stencil_weights`). Everything is derived exactly from the kernel pieces of
# column_polynomials.jl and rounded once to the element type.

"""
    _anchor_tables_rounded(T, kernel, m)

The exact anchoring data of integral order m for the near-anchor coefficients j = 1 … 2·eqs − 1,
each entry rounded once to T: `taylor` (row j holds K_{m−r}(eqs − j) for r = 0 … m−1) and
`entries` (row j holds the entry polynomial of `_near_anchor_tail_entry`, the coefficients of
t^0 … t^(m−1)). Derived in Rational{BigInt}; see `_anchor_tables` for the cached version.
"""
function _anchor_tables_rounded(::Type{T}, kernel::Symbol, m::Int) where {T}
    eqs = get_equations_for_degree(kernel)
    taylor = Matrix{T}(undef, 2eqs - 1, m)
    entries = Matrix{T}(undef, 2eqs - 1, m)
    for j in 1:2eqs-1
        for r in 0:m-1
            taylor[j, r+1] = T(_kernel_value_exact(kernel, m - r, eqs - j))
        end
        entries[j, :] .= T.(_near_anchor_tail_entry(kernel, m, eqs, j))
    end
    return taylor, entries
end

"""
    _anchor_tables(Val(kernel), Val(m), T)

`_anchor_tables_rounded` derived once per (kernel, m, T): for bits types such as Float64 and
Float32 the tables are compile-time constants, shared by all interpolants, so callers must copy
before storing them. Non-bits types such as BigFloat derive them at run time instead, at the
precision in effect at construction.
"""
@generated function _anchor_tables(::Val{kernel}, ::Val{m}, ::Type{T}) where {kernel,m,T}
    isbitstype(T) || return :(_anchor_tables_rounded(T, kernel, m))
    tables = _anchor_tables_rounded(T, kernel, m)
    return :($tables)
end

"""
    _anchor_taylor_table(T, kernel, m, eqs)

Anchoring data for integral order m: row j (1 ≤ j ≤ 2·eqs − 1, the coefficients within reach
of the anchor) holds K_{m−r}(eqs − j) for r = 0 … m−1, exact and rounded once to T. The anchored
weight of such a coefficient at index position u is K_m(u − j) − Σ_r K_{m−r}(eqs − j)·(u − eqs)^r / r!.
"""
_anchor_taylor_table(::Type{T}, kernel::Symbol, m::Int, eqs::Int) where {T} =
    copy(_anchor_tables(Val(kernel), Val(m), T)[1])

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
    sz = size(coefs)
    n = sz[d]                                            # length along dimension d
    pre = prod(i -> sz[i], 1:d-1; init=1)                # elements before d (contiguous in memory)
    post = prod(i -> sz[i], d+1:N; init=1)               # elements after d
    fac = factorial(big(m - 1))
    # entry polynomial of a coefficient far from the anchor: (t + eqs)^(m−1) / (m−1)!
    far_entry = T[T(binomial(big(m - 1), big(k)) * big(eqs)^(m - 1 - k) // fac) for k in 0:m-1]
    # entry polynomials of the coefficients near the anchor, exact and rounded once
    entries = _anchor_tables(Val(kernel), Val(m), T)[2]
    near_entry = Vector{T}[entries[j, :] for j in 1:min(2eqs - 1, n)]
    # Taylor shift by one cell: new coefficient k = Σ_{q ≥ k} binomial(q, k)·old coefficient q
    shift = T[T(binomial(q, k)) for k in 0:m-1, q in 0:m-1]

    tails = Array{T,N}[Array{T,N}(undef, sz) for _ in 1:m]
    # the same three-index view of the coefficients and of every tail array: (before d, along d,
    # after d), so the recursion along d runs over plain loops with the contiguous index innermost
    A = reshape(coefs, pre, n, post)
    R = Array{T,3}[reshape(tails[k], pre, n, post) for k in 1:m]
    _left_tail_recursion!(R, A, near_entry, far_entry, shift)   # function barrier: concrete types
    return tails
end

# The shifted prefix sum of `_left_tail_polynomial` along the middle index of A and of every
# tail array in R (R[k] holds the coefficient of t^(k−1)): R(l) = S·R(l−1) + A(l)·entry(l).
# A separate function, so the loops are compiled for the concrete array types.
function _left_tail_recursion!(R::Vector{Array{T,3}}, A::AbstractArray{T,3},
                               near_entry::Vector{Vector{T}}, far_entry::Vector{T},
                               shift::Matrix{T}) where {T}
    pre, n, post = size(A)                               # (before d, along d, after d)
    m = length(R)                                        # number of powers of t
    @inbounds for c in 1:post, l in 1:n
        entry = l <= length(near_entry) ? near_entry[l] : far_entry
        for k in 1:m
            Rk = R[k]                                    # tail array of the coefficient of t^(k−1)
            for a in 1:pre
                acc = entry[k] * A[a, l, c]              # new coefficient c_l entering the tail
                if l > 1
                    for q in k:m                         # shifted tail of the previous position
                        acc = acc + shift[k, q] * R[q][a, l - 1, c]
                    end
                end
                Rk[a, l, c] = acc
            end
        end
    end
    return R
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
varying fastest). Built separably: each region extends the region without its highest
dimension (a smaller mask, built earlier; the coefficients themselves for single-dimension
regions) by the left tail along that dimension.
"""
function _build_region_tails(coefs::AbstractArray{T,N}, kernels::NTuple{N,Symbol},
                             orders::NTuple{N,Int}, eqs::NTuple{N,Int}) where {T,N}
    int_dims = [d for d in 1:N if orders[d] < 0]
    n = length(int_dims)
    tails = Vector{Vector{Array{T,N}}}(undef, 2^n - 1)
    for mask in 1:2^n-1
        b = 8 * sizeof(mask) - leading_zeros(mask)      # position of the highest set bit
        low = mask & ~(1 << (b - 1))                    # the region without that dimension
        d = int_dims[b]                                  # the dimension added in this region
        m = -orders[d]                                   # its integral order
        base = low == 0 ? Array{T,N}[coefs] : tails[low] # read only, never modified
        expanded = Array{T,N}[]
        for array in base
            append!(expanded, _left_tail_polynomial(array, kernels[d], m, eqs[d], d))
        end
        tails[mask] = expanded
    end
    return tails
end

"""
    _far_stencil_rows(Val(kernel), Val(M), T)

Anchored weights of a cell whose whole stencil lies beyond the anchor's reach (every coefficient
j ≥ 2·eqs), as polynomials in t, in eager order: entry k is the weight of coefficient
j = i − eqs + k. Each is the column polynomial of K_M with the far field
(u − j)^(M−1) / (2(M−1)!) folded in, where u − j = t + o and o is the column's offset. Derived
exactly in Rational{BigInt} and rounded once to `T`, in the layout of `_column_rows`
(rows[d][k] = coefficient of t^(d−1) for entry k), so such a cell costs one Horner evaluation.
Not defined for :a0, which has no column polynomials.
"""
@generated function _far_stencil_rows(::Val{kernel}, ::Val{M}, ::Type{T}) where {kernel,M,T}
    columns = _column_polynomials_exact(kernel, -M)      # column c holds K_M(o + t), o = c − 1 − eqs
    K = length(columns)                                  # 2·eqs columns
    eqs = K ÷ 2                                          # stencil half-width
    fac = 2 * factorial(big(M - 1))                      # divisor of the far field
    folded = map(1:K) do c
        o = big(c - 1 - eqs)                             # offset of column c, so u − j = t + o
        p = copy(columns[c])                             # exact coefficients of t^0, t^1, …
        length(p) < M && append!(p, zeros(Rational{BigInt}, M - length(p)))  # room for t^(M−1)
        for k in 0:M-1
            # (t + o)^(M−1) / (2(M−1)!): the coefficient of t^k is binomial(M−1, k)·o^(M−1−k) / fac
            p[k+1] += binomial(big(M - 1), big(k)) * o^(M - 1 - k) // fac
        end
        p
    end
    D = maximum(length, folded)                          # number of rows (highest power + 1)
    # entry k (coefficient j = i − eqs + k) is held by column K + 1 − k, as in `_anchored_weight`
    rows = ntuple(d -> ntuple(k -> d <= length(folded[K+1-k]) ? T(folded[K+1-k][d]) : zero(T), K), D)
    return :($rows)
end

"""
    _anchored_stencil_weights(Val(kernel), Val(M), t, i, eqs, taylor)

Anchored weights of the 2·eqs stencil coefficients of one integral dimension of order M, for a
point at position t within cell i, in eager order: entry k is the weight of coefficient
j = i − eqs + k. The anchored weight is K_M(u − j) − Σ_r K_{M−r}(eqs − j)·(u − eqs)^r / r!
(u = i + t): the column polynomial of K_M minus the Taylor polynomial at the anchor, which for
coefficients far from the anchor (j ≥ 2·eqs) is the far field −(u − j)^(M−1) / (2(M−1)!), and
for those near it comes from the exact table `taylor` (see `_anchor_taylor_table`).

In a cell whose whole stencil is far from the anchor (i ≥ 3·eqs − 1, almost every cell on a
large grid), the weights come directly from the exactly folded table `_far_stencil_rows`: one
branch per cell instead of one per coefficient, and one rounding per table entry.
"""
@inline function _anchored_stencil_weights(::Val{kernel}, ::Val{M}, t::T, i::Int, eqs::Int,
                                           taylor::AbstractMatrix{T}) where {kernel,M,T}
    # whole stencil beyond the anchor's reach (its first coefficient i − eqs + 1 ≥ 2·eqs): the
    # folded table. kernel is a compile-time constant, so the :a0 test disappears when compiled
    if kernel !== :a0 && i - eqs + 1 >= 2eqs
        return _column_weights(_far_stencil_rows(Val(kernel), Val(M), T), t)
    end
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
_near_anchor_entries(::Type{T}, kernel::Symbol, m::Int, eqs::Int) where {T} =
    copy(_anchor_tables(Val(kernel), Val(m), T)[2])