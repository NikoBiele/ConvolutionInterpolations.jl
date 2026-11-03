# Exact derivation of kernel coefficient dictionaries from each kernel's base coefficients.
#
# A kernel's base dictionary maps :eq1, :eq2, … to the coefficients of its polynomial pieces
# in |s|, with entry i multiplying |s|^(i−1). Everything else is derived from it exactly.

# Falling factorial n (n−1) ⋯ (n−k+1) as a BigInt; zero when 0 ≤ n < k, and one when k = 0.
_falling_factorial(n::Integer, k::Integer) = prod((big(n - j) for j in 0:k-1); init=big(1))

"""
    _derivative_coefs(base, k)

Coefficients of the k-th derivative of a kernel's polynomial pieces, in the layout of the
base dictionary: entry i is `base[i] * (i−1)!/(i−1−k)!`, which is zero for i ≤ k, so the
first k entries of every piece are zero and the remaining ones multiply |s|^(i−1−k).
Computed exactly in Rational{BigInt}.
"""
function _derivative_coefs(base::Dict{Symbol,<:AbstractVector{<:Rational}}, k::Integer)
    exact = Dict{Symbol,Vector{Rational{BigInt}}}()
    for (key, coefs) in base
        exact[key] = [Rational{BigInt}(coefs[i]) * _falling_factorial(i - 1, k)
                      for i in eachindex(coefs)]
    end
    return exact
end