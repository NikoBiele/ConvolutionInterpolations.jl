# Exact derivation of kernel coefficient dictionaries from each kernel's base coefficients.
#
# A kernel's base dictionary maps :eq1, :eq2, … to the coefficients of its polynomial pieces
# in |s|, with entry i multiplying |s|^(i−1). Everything else is derived from it exactly.

# Falling factorial n (n−1) ⋯ (n−k+1) as a BigInt; zero when 0 ≤ n < k, and one when k = 0.
_falling_factorial(n::Integer, k::Integer) = prod((big(n - j) for j in 0:k-1); init=big(1))

# Return the dictionary as Rational{Int64} if every entry fits, else keep Rational{BigInt}.
# `horner` converts coefficients on every call, which is cheap from Rational{Int64} and costly
# from Rational{BigInt}, so the narrower type is used wherever it is exact.
function _narrowest_rational_dict(exact::Dict{Symbol,Vector{Rational{BigInt}}})
    fits(v) = typemin(Int64) <= numerator(v) <= typemax(Int64) && denominator(v) <= typemax(Int64)
    if all(fits, Iterators.flatten(values(exact)))
        return Dict{Symbol,Vector{Rational{Int64}}}(key => Rational{Int64}.(v) for (key, v) in exact)
    else
        return exact
    end
end

"""
    _derivative_coefs(base, k)

Coefficients of the k-th derivative of a kernel's polynomial pieces, in the layout the kernel
evaluation expects: entry i is `base[i] * (i−1)!/(i−1−k)!`, which is zero for i ≤ k, and
`horner(x, dict, key, T, k)` skips those first k entries. Computed exactly in
Rational{BigInt}, then narrowed to Rational{Int64} when every entry fits.
"""
function _derivative_coefs(base::Dict{Symbol,<:AbstractVector{<:Rational}}, k::Integer)
    exact = Dict{Symbol,Vector{Rational{BigInt}}}()
    for (key, coefs) in base
        exact[key] = [Rational{BigInt}(coefs[i]) * _falling_factorial(i - 1, k)
                      for i in eachindex(coefs)]
    end
    return _narrowest_rational_dict(exact)
end