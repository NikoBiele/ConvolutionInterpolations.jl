# Evaluation of interpolants with nonuniform axes. Every axis contributes a stencil: the index of
# the coefficient just before it, the kernel weights of its entries, and the factor that scales a
# derivative to the spacing. The axis' kind, the entry of the `nonuniform` field of
# `ConvolutionInterpolation`, decides where the weights come from:
#   nothing            uniform axis: the exact column polynomials (`_kernel_weights`)
#   Vector{Matrix{T}}  nonuniform b kernel: per-interval polynomial weights (`_nb_dim`)
#   Val(:nonuniform)   closed-form nonuniform kernel: :n3 (`_nonuniform_dim_ghost`), :a0 or :a1
# The stencils of all axes are then contracted with the coefficients in `_stencil_contract`.

# ---------------------------------------------------------------------------------------------
# Per-axis weights of the nonuniform kernels
# ---------------------------------------------------------------------------------------------

# Stencil half-width M_eqs of a nonuniform b kernel and the full stencil length as a Val, as
# compile-time constants: b5 → 5, b7 → 6, b9 → 7, b11 → 8, b13 → 9
@inline _nb_M_eqs(::Val{:b5})  = 5
@inline _nb_M_eqs(::Val{:b7})  = 6
@inline _nb_M_eqs(::Val{:b9})  = 7
@inline _nb_M_eqs(::Val{:b11}) = 8
@inline _nb_M_eqs(::Val{:b13}) = 9

@inline _nb_n_stencil(::Val{:b5})  = Val(10)
@inline _nb_n_stencil(::Val{:b7})  = Val(12)
@inline _nb_n_stencil(::Val{:b9})  = Val(14)
@inline _nb_n_stencil(::Val{:b11}) = Val(16)
@inline _nb_n_stencil(::Val{:b13}) = Val(18)

"""
    _nb_dim(knots_expanded, weight_coeffs, M_eqs, x_val, ::Val{NS})

Interval and stencil weights of a nonuniform b kernel along one axis. Returns `(i, w, h0)`: the
interval index in the expanded knots, the `NS` stencil weights (coefficients `i − M_eqs + 1` …
`i + M_eqs`) by Horner's scheme from the per-interval polynomial table `weight_coeffs[k]` of
the original interval `k = i − M_eqs`, and the interval's spacing.
"""
@inline function _nb_dim(knots_expanded::AbstractVector{T},
                          weight_coeffs::Vector{<:AbstractMatrix{<:AbstractFloat}},
                          M_eqs::Int, x_val::Number,
                          ::Val{NS}) where {T, NS}
    n_exp = length(knots_expanded)
    i = searchsortedlast(knots_expanded, x_val)
    i = clamp(i, M_eqs + 1, n_exp - M_eqs - 1)

    # Interval index in the original grid (1-based)
    k = i - M_eqs

    # Normalized coordinate s ∈ [0, 1]
    h0 = knots_expanded[i+1] - knots_expanded[i]
    s = (T(x_val) - knots_expanded[i]) / h0

    # Evaluate weights via Horner — ntuple with compile-time size
    coeffs = weight_coeffs[k]
    n_poly = size(coeffs, 2)

    w = ntuple(Val(NS)) do j
        @inbounds begin
            val = coeffs[j, n_poly]
            for p in (n_poly-1):-1:1
                val = val * s + coeffs[j, p]
            end
            val
        end
    end

    return i, w, h0
end

"""
    _nonuniform_dim_ghost(knots, x_val)

Interval and the four closed-form `:n3` weights along one axis, from the expanded knots (one
ghost per side): the interval index `i` and the weights of the stencil `i−1 … i+2`, computed
from the local spacings by `nonuniform_cubic_weights`.
"""
@inline function _nonuniform_dim_ghost(knots::AbstractVector{T}, x_val::Number) where T
    n = length(knots)
    i = searchsortedlast(knots, x_val)
    # with ghost knots the valid range is 2 … n−2 (the stencil needs i−1 … i+2)
    i = clamp(i, 2, n - 2)

    x_local = T(x_val) - knots[i]
    hm = knots[i]   - knots[i-1]
    h0 = knots[i+1] - knots[i]
    hp = knots[i+2] - knots[i+1]

    w = nonuniform_cubic_weights(x_local, hm, h0, hp)
    return i, w
end

# ---------------------------------------------------------------------------------------------
# Per-axis stencils
# ---------------------------------------------------------------------------------------------

# Uniform axis: cell i, column coordinate τ = diff_right, weights of the 2·eqs columns
@inline function _axis_stencil(::Nothing, ::Val{kernel}, ::Val{DO}, knots::AbstractVector{T},
                               h::T, eqs::Int, x::T) where {kernel,DO,T}
    i_float = (x - knots[1]) / h + one(T)
    i = clamp(floor(Int, i_float), eqs, length(knots) - eqs)
    τ = one(T) - (i_float - T(i))
    w = _kernel_weights(Val(kernel), Val(DO), τ)
    return i - eqs, w, (-one(T) / h)^DO
end

# Nonuniform b kernel: the per-interval weight tables, eqs is the kernel's M_eqs
@inline function _axis_stencil(tables::Vector{<:AbstractMatrix}, ::Val{kernel}, ::Val{DO},
                               knots::AbstractVector{T}, h::T, eqs::Int, x::T) where {kernel,DO,T}
    i, w, h0 = _nb_dim(knots, tables, eqs, x, _nb_n_stencil(Val(kernel)))
    return i - eqs, T.(w), (one(T) / h0)^DO
end

# Closed-form nonuniform kernels. :n3 reads the coefficients with one ghost per side (stencil
# i−1 … i+2); :a0 and :a1 read the raw values, so the knots carry one ghost per side and the
# coefficients none, and knot index i is coefficient i − 1.
@inline function _axis_stencil(::Val{:nonuniform}, ::Val{kernel}, ::Val{DO},
                               knots::AbstractVector{T}, h::T, eqs::Int, x::T) where {kernel,DO,T}
    if kernel === :n3
        i, w = _nonuniform_dim_ghost(knots, x)
        return i - 2, w, one(T)
    end
    i = clamp(searchsortedlast(knots, x), 2, length(knots) - 2)
    t = (x - knots[i]) / (knots[i+1] - knots[i])
    if kernel === :a0
        w = t <= T(1//2) ? (one(T), zero(T)) : (zero(T), one(T))
    else
        w = (one(T) - t, t)
    end
    return i - 2, w, one(T)
end

# The stencils of every axis: offsets, weights padded to one common length so that they can be
# indexed with a runtime axis, the true stencil length per axis, and the product of the scales.
# Generated so that each axis' kind and kernel are compile-time constants, which makes the
# stencil lengths constants too.
@generated function _axis_stencils(nonuniform::NU, ::Val{KS}, ::Val{DO}, knots::Tuple,
                                   h::NTuple{N,T}, eqs::NTuple{N,Int},
                                   x::NTuple{N,T}) where {NU,KS,DO,N,T}
    lens = map(1:N) do d
        kind = NU.parameters[d]
        if kind === Nothing
            2 * get_equations_for_degree(KS[d])
        elseif kind <: Val
            KS[d] === :n3 ? 4 : 2
        else
            2 * _nb_M_eqs(Val(KS[d]))
        end
    end
    Kmax = maximum(lens)
    # the generated body may not contain closures, so every tuple is spelled out here
    calls = [:(_axis_stencil(nonuniform[$d], Val($(QuoteNode(KS[d]))), Val($(DO[d])),
                             knots[$d], h[$d], eqs[$d], x[$d])) for d in 1:N]
    offsets = [:(st[$d][1]) for d in 1:N]
    weights = [:(_pad_weights(st[$d][2], Val($Kmax))) for d in 1:N]
    scales = [:(st[$d][3]) for d in 1:N]
    return quote
        st = tuple($(calls...))
        return tuple($(offsets...)), tuple($(weights...)), $(Tuple(lens)), *($(scales...))
    end
end

"""
    (itp::ConvolutionInterpolation{T,N,...,NonUniformMixedOrderKernel{K},...})(x...)

Evaluate an interpolant with nonuniform axes: the stencil of every axis from its kind and
kernel (`_axis_stencils`), contracted with the coefficients one axis at a time
(`_stencil_contract`), times the derivative scale. Any dimension, any kernel per axis.
"""
function (itp::ConvolutionInterpolation{T,N,0,TCoefs,Axs,KA,DT,NonUniformMixedOrderKernel{K},
            EQ,KBC,DerivativeOrder{DO},LZ,Val{0}})(x::Vararg{Number,N}) where
            {T<:AbstractFloat,N,TCoefs<:AbstractArray{T,N},Axs,KA,DT,K,EQ,KBC,DO,LZ}
    x = T.(x)
    o, W, len, scale = _axis_stencils(itp.nonuniform, Val(K), Val(DO), itp.knots, itp.h,
                                      itp.eqs, x)
    return _stencil_contract(itp.coefs, W, o, len) * scale
end