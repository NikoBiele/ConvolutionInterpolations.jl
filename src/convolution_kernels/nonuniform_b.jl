# ============================================================
# Nonuniform b-kernel interpolation — core precomputation
# 
# For any b-series kernel (b5, b7, b9, b11, b13), provides
# high-order interpolation and derivatives on nonuniform grids.
#
# Three key adaptations from uniform to nonuniform:
#   1. Binomial expansion of K^(n)(offset + sgn·s) with power shift
#      (derivative coefficients c[k] represent t^(k-derivative))
#   2. Sign correction sgn^derivative for odd-derivative kernels
#   3. Reproduction projection V'·coeffs = E to enforce polynomial
#      exactness, with E indexed as E[moment, s_power]
#
# The (1/h₀)^derivative chain rule factor is applied at evaluation
# time, matching the uniform-grid pattern.
#
# All precomputation uses Rational{BigInt} for exact arithmetic.
# The result is stored as Float64 polynomial coefficients for fast evaluation.
# ============================================================


"""
    nonuniform_b_params(degree::Symbol) -> (M_eqs, p_deg)

Return kernel parameters for a b-series kernel:
- `M_eqs`: number of piecewise pieces (= support radius)
- `p_deg`: polynomial degree of the kernel
"""
function nonuniform_b_params(degree::Symbol)
    degree == :b5  && return (5,  5)
    degree == :b7  && return (6,  7)
    degree == :b9  && return (7,  9)
    degree == :b11 && return (8, 11)
    degree == :b13 && return (9, 13)
    error("nonuniform_b_params: unsupported kernel $degree")
end


"""
    _get_kernel_rational_coefs(degree::Symbol, derivative::Int=0) -> Dict{Int, Vector{Rational{BigInt}}}

Return the kernel piece coefficients as exact rationals.
For derivative > 0, returns the differentiated kernel coefficients.
"""
function _get_kernel_rational_coefs(degree::Symbol, derivative::Int=0)
    coefs_dict = if degree == :b5
        derivative == 0 ? b5_coefs :
        derivative == 1 ? b5_coefs_d1 :
        derivative == 2 ? b5_coefs_d2 :
        derivative == 3 ? b5_coefs_d3 :
        error("b5: derivative order $derivative not available (max 3)")
    elseif degree == :b7
        derivative == 0 ? b7_coefs :
        derivative == 1 ? b7_coefs_d1 :
        derivative == 2 ? b7_coefs_d2 :
        derivative == 3 ? b7_coefs_d3 :
        derivative == 4 ? b7_coefs_d4 :
        error("b7: derivative order $derivative not available (max 4)")
    elseif degree == :b9
        derivative == 0 ? b9_coefs :
        derivative == 1 ? b9_coefs_d1 :
        derivative == 2 ? b9_coefs_d2 :
        derivative == 3 ? b9_coefs_d3 :
        derivative == 4 ? b9_coefs_d4 :
        derivative == 5 ? b9_coefs_d5 :
        error("b9: derivative order $derivative not available (max 5)")
    elseif degree == :b11
        derivative == 0 ? b11_coefs :
        derivative == 1 ? b11_coefs_d1 :
        derivative == 2 ? b11_coefs_d2 :
        derivative == 3 ? b11_coefs_d3 :
        derivative == 4 ? b11_coefs_d4 :
        derivative == 5 ? b11_coefs_d5 :
        derivative == 6 ? b11_coefs_d6 :
        error("b11: derivative order $derivative not available (max 6)")
    elseif degree == :b13
        derivative == 0 ? b13_coefs :
        derivative == 1 ? b13_coefs_d1 :
        derivative == 2 ? b13_coefs_d2 :
        derivative == 3 ? b13_coefs_d3 :
        derivative == 4 ? b13_coefs_d4 :
        derivative == 5 ? b13_coefs_d5 :
        derivative == 6 ? b13_coefs_d6 :
        error("b13: derivative order $derivative not available (max 6)")
    else
        error("_get_kernel_rational_coefs: unsupported kernel $degree")
    end

    # Convert from Dict{Symbol, Vector} to Dict{Int, Vector{Rational{BigInt}}}
    result = Dict{Int, Vector{Rational{BigInt}}}()
    for (sym, coeffs) in coefs_dict
        idx = parse(Int, string(sym)[3:end])  # :eq1 -> 1, :eq2 -> 2, etc.
        result[idx] = Rational{BigInt}.(coeffs)
    end
    return result
end


"""
    precompute_nonuniform_b_interval(knots_r, i, kernel_coefs, M_eqs, p_deg, derivative)

Precompute weight polynomial coefficients for one interval using exact arithmetic.

The three key operations for derivative support:

1. **Binomial expansion with power shift**: The stored derivative coefficients
   c[k] represent t^(k-derivative), not t^k (matching Julia's horner with DO skip).
   So we expand (offset + sgn·s)^(k-derivative) for k ≥ derivative.

2. **Sign correction**: For odd derivatives, multiply target by sgn (= sign(s - d_j)
   for all s ∈ [0,1]), since K^(n)(|t|) · sign(t)^n.

3. **Reproduction projection**: Enforce Σ w_j · node_j^m = E[m, :] · s_vec,
   where E[m, s_power] = falling(m, n) · h₀^m · δ(s_power, m-n).
   Note: E is indexed as E[moment_row, s_power_col].
   For d0 this is diagonal (E[m,m] = h₀^m).
   For d1 this is sub-diagonal (E[m, m-1] = m · h₀^m).
"""
function precompute_nonuniform_b_interval(knots_r::Vector{Rational{BigInt}}, i::Int,
                                           kernel_coefs::Dict{Int,Vector{Rational{BigInt}}},
                                           M_eqs::Int, p_deg::Int, derivative::Int)
    R = Rational{BigInt}
    n_stencil = 2 * M_eqs
    n_poly = p_deg + 1
    h0 = knots_r[i+1] - knots_r[i]

    # Node positions relative to knots[i]
    nodes = [knots_r[i - M_eqs + j] - knots_r[i] for j in 1:n_stencil]

    # ── Step 1: Target weights as exact polynomials in s ──
    target = zeros(R, n_stencil, n_poly)

    for j in 1:n_stencil
        d_j = nodes[j] // h0

        if d_j >= 1
            offset = d_j; sgn = R(-1)
        else
            offset = -d_j; sgn = R(1)
        end

        t_at_0, t_at_1 = offset, offset + sgn
        t_min, t_max = minmax(t_at_0, t_at_1)

        # Find kernel piece containing [t_min, t_max]
        piece = 0
        for p in 1:M_eqs
            if t_min >= p - 1 && t_max <= p
                piece = p
                break
            end
        end

        piece == 0 && continue  # outside support → zero weight

        # Binomial expansion of K^(n)(offset + sgn·s)
        # The stored d_n coefficients c[k] represent t^(k-derivative):
        #   K^(n)(t) = Σ_{k≥derivative} c[k] · t^(k-derivative)
        # Substituting t = offset + sgn·s and expanding via binomial theorem:
        c = kernel_coefs[piece]
        for k in derivative:(length(c)-1)   # 0-based coefficient index
            c_k = c[k+1]                    # 1-based Julia indexing
            c_k == 0 && continue
            actual_power = k - derivative    # true power of t in K^(n)
            for m in 0:actual_power
                binom = R(binomial(BigInt(actual_power), BigInt(m)))
                target[j, m+1] += c_k * binom * offset^(actual_power - m) * sgn^m
            end
        end

        # Sign correction for odd derivatives:
        # K^(n)(|t|) · sign(t)^n where sign(t) = sign(s - d_j) = sgn
        if isodd(derivative)
            for m in 1:n_poly
                target[j, m] *= sgn
            end
        end
    end

    # ── Step 2: Reproduction projection (exact) ──
    V = zeros(R, n_stencil, n_poly)
    for j in 1:n_stencil
        V[j,1] = R(1)
        for m in 2:n_poly
            V[j,m] = V[j,m-1] * nodes[j]
        end
    end

    # Build E: reproduction target
    # E[moment_row, s_power_col]: what Σ w_j · node_j^m should equal as polynomial in s
    #
    # For derivative=0: E[m, m] = h₀^m  (diagonal)
    #   → Σ w_j · node_j^m = (h₀·s)^m
    #
    # For derivative=n: E[m, m-n] = falling(m,n) · h₀^m  (for m ≥ n)
    #   → (1/h₀)^n · Σ w_j · node_j^m = falling(m,n) · (h₀·s)^(m-n)
    #   → Σ w_j · node_j^m = falling(m,n) · h₀^m · s^(m-n)
    E = zeros(R, n_poly, n_poly)

    for m_row in 1:n_poly   # m_row = moment + 1 (1-based, 0-based power = m_row-1)
        power = m_row - 1   # 0-based exponent
        if power < derivative
            continue         # falling factorial is zero for power < derivative
        end
        # falling(power, derivative) = power · (power-1) · ... · (power-derivative+1)
        falling = R(1)
        for k in 0:derivative-1
            falling *= (power - k)
        end
        s_power = power - derivative   # 0-based power of s
        s_col = s_power + 1            # 1-based column in E
        E[m_row, s_col] = falling * h0^power
    end

    # Projection: coeffs = target - V · Λ where Λ = (V'V)⁻¹ (V'·target - E)
    VtV = V' * V
    Λ = VtV \ (V' * target - E)
    coeffs_rational = target - V * Λ

    # Convert to Float64 for fast evaluation
    return Float64.(coeffs_rational)
end


"""
    precompute_nonuniform_b_all(knots, degree, derivative=0)

Precompute weight coefficients for ALL intervals of a 1D nonuniform grid.

# Arguments
- `knots`: original (unexpanded) knot vector
- `degree`: kernel symbol (`:b5`, `:b7`, etc.)
- `derivative`: derivative order — selects the appropriate kernel coefficients
  and reproduction target. The (1/h₀)^derivative post-multiply is applied
  at evaluation time.

# Returns
- `weight_coeffs`: `Vector{Matrix{Float64}}` of length `n_intervals`
- `knots_expanded`: expanded knot vector with ghost points
"""
function precompute_nonuniform_b_all(knots::AbstractVector{T}, degree::Symbol,
                                      derivative::Int=0) where T
    M_eqs, p_deg = nonuniform_b_params(degree)
    n = length(knots)
    n_intervals = n - 1

    # Convert knots to Rational{BigInt}
    knots_r = Rational{BigInt}.(knots)

    # Expand with M_eqs ghost points on each side (mirror spacing)
    knots_exp_r = copy(knots_r)
    for _ in 1:M_eqs
        pushfirst!(knots_exp_r, knots_exp_r[1] - (knots_exp_r[2] - knots_exp_r[1]))
        push!(knots_exp_r, knots_exp_r[end] + (knots_exp_r[end] - knots_exp_r[end-1]))
    end

    # Get kernel coefficients (with appropriate derivative)
    kernel_coefs = _get_kernel_rational_coefs(degree, derivative)

    # Precompute for each interval
    weight_coeffs = Vector{Matrix{Float64}}(undef, n_intervals)
    for k in 1:n_intervals
        i = k + M_eqs  # index in expanded knots (1-based)
        weight_coeffs[k] = precompute_nonuniform_b_interval(
            knots_exp_r, i, kernel_coefs, M_eqs, p_deg, derivative)
    end

    # Expanded knots as Float64
    knots_expanded = T.(knots_exp_r)

    return weight_coeffs, knots_expanded
end

"""
    precompute_uniform_b_all(knots, degree, derivative=0)

Precompute weight polynomial coefficients for uniform b-kernel interpolation.

For uniform grids, the kernel piece polynomials are converted from |t|-space to
s-space (local interval coordinate) via binomial expansion. No Vandermonde projection
is needed because the uniform kernel has exact polynomial reproduction built in.

All intervals produce identical weight matrices, so we compute once and copy.

# How it works
The kernel evaluates `K(|t|)` where `|t| = offset ± s` and offset is an integer.
The binomial theorem expands `(offset ± s)^k` to convert the kernel piece polynomial
from |t|-coordinates to s-coordinates suitable for Horner evaluation.

Piece selection uses the midpoint `|t(s=1/2)|` to avoid ambiguity at integer
boundaries where the kernel is zero.
"""
function precompute_uniform_b_all(knots::AbstractVector{T}, degree::Symbol,
                                      derivative::Int=0) where T
    M_eqs, p_deg = nonuniform_b_params(degree)
    n = length(knots)
    n_intervals = n - 1

    R = Rational{BigInt}
    h = R(knots[2] - knots[1])

    # Expand with ghost points
    knots_r = Rational{BigInt}.(knots)
    knots_exp_r = copy(knots_r)
    for _ in 1:M_eqs
        pushfirst!(knots_exp_r, knots_exp_r[1] - h)
        push!(knots_exp_r, knots_exp_r[end] + h)
    end

    kernel_coefs = _get_kernel_rational_coefs(degree, derivative)

    n_stencil = 2 * M_eqs
    n_poly = p_deg + 1

    # Representative interval: index M_eqs+1 in expanded knots
    i_ref = M_eqs + 1

    target = zeros(R, n_stencil, n_poly)

    for j in 1:n_stencil
        # Node position relative to interval start, normalized by h
        d_j = (knots_exp_r[i_ref - M_eqs + j] - knots_exp_r[i_ref]) // h

        # The kernel evaluates at |t| where t = (x - knot_j)/h
        # At position s ∈ [0,1] within interval: x = knot_i + s·h
        # So t_j = s - d_j, and |t_j| = |s - d_j|
        #
        # For d_j <= 0 (nodes to the left): |t_j| = s - d_j = s + |d_j|
        #   offset = |d_j|, sgn = +1, |t| = offset + s
        # For d_j >= 1 (nodes to the right): |t_j| = d_j - s
        #   offset = d_j, sgn = -1, |t| = offset - s

        if d_j >= 1
            offset = d_j
            sgn = R(-1)
        else
            offset = -d_j
            sgn = R(1)
        end

        # Find which kernel piece covers this range using midpoint
        # to avoid ambiguity at integer boundaries where kernel is zero.
        # Kernel pieces: eq_p covers |t| ∈ [p-1, p), with |t| >= M_eqs → zero
        t_mid = abs(offset + sgn * R(1//2))
        piece = 0
        for p in 1:M_eqs
            if t_mid >= p - 1 && t_mid < p
                piece = p
                break
            end
        end

        piece == 0 && continue

        c = kernel_coefs[piece]

        # Kernel evaluates: Σ_{k=0}^{deg} c[k+1] · |t|^k
        # With |t| = offset + sgn·s, expand via binomial theorem:
        # (offset + sgn·s)^k = Σ_{m=0}^{k} C(k,m) · offset^(k-m) · (sgn·s)^m
        #
        # So coeff of s^m = Σ_{k=m}^{deg} c[k+1] · C(k,m) · offset^(k-m) · sgn^m

        for k in 0:(length(c)-1)
            c_k = c[k+1]
            c_k == 0 && continue
            for m in 0:k
                binom = R(binomial(BigInt(k), BigInt(m)))
                target[j, m+1] += c_k * binom * offset^(k-m) * sgn^m
            end
        end

        # Sign correction for odd derivatives: kernel uses sign(t) factor
        if isodd(derivative)
            sgn_sign = (d_j >= 0) ? R(1) : R(-1)
            for m in 1:n_poly
                target[j, m] *= sgn_sign
            end
        end
    end

    ref = Float64.(target)
    weight_coeffs = [ref for _ in 1:n_intervals]
    knots_expanded = T.(knots_exp_r)
    return weight_coeffs, knots_expanded
end