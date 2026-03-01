"""
    _nonuniform_dim_ghost(knots::AbstractVector{T}, x_val::Number) where T

Find interval index and compute cubic nonuniform weights for a single dimension.

Takes the expanded knot vector (including ghost points) and returns the interval index `i`
and a 4-element weight tuple `w` for the stencil `[i-1, i, i+1, i+2]`. Uses local spacing
ratios (hm, h0, hp) for the nonuniform cubic weight formula.
"""

@inline function _nonuniform_dim_ghost(knots::AbstractVector{T}, x_val::Number) where T
    # knots is the EXPANDED knot vector (includes ghost positions)
    # Data array also has ghost values at index 1 and end
    n = length(knots)
    i = searchsortedlast(knots, x_val)
    # With ghost points, valid range is 2 to n-2 (need i-1, i, i+1, i+2)
    i = clamp(i, 2, n - 2)
    
    x_local = T(x_val) - knots[i]
    hm = knots[i]   - knots[i-1]
    h0 = knots[i+1] - knots[i]
    hp = knots[i+2] - knots[i+1]
    
    w = nonuniform_weights(x_local, hm, h0, hp)
    return i, w
end

"""
    (itp::ConvolutionInterpolation{...,Val{:n3},...})(x...)

Evaluate nonuniform cubic (`:n3`) interpolation. Uses 4-point stencil per dimension with
locally adapted weights based on nonuniform grid spacing. Supports 1D through N-D via
tensor product of 1D weights. Ghost points at boundaries eliminate the need for clamping.

This is the fallback path for a-series kernels on nonuniform grids.
"""

# --- 1D with ghost points ---

@inline function (itp::ConvolutionInterpolation{T,1,TCoefs,IT,Axs,KA,Val{1},
                    Val{:n3},EQ,KBC,DO})(x::Vararg{Number,1}) where 
                    {T,TCoefs,IT,Axs,KA,EQ,KBC,DO}

    # itp.knots[1] is the expanded knot vector
    # itp.coefs includes ghost values
    i, w = _nonuniform_dim_ghost(itp.knots[1], x[1])
    
    @inbounds return w[1] * itp.coefs[i-1] + w[2] * itp.coefs[i] + 
                     w[3] * itp.coefs[i+1] + w[4] * itp.coefs[i+2]
end

# --- 2D with ghost points ---

@inline function (itp::ConvolutionInterpolation{T,2,TCoefs,IT,Axs,KA,Val{2},
                    Val{:n3},EQ,KBC,DO})(x::Vararg{Number,2}) where 
                    {T,TCoefs,IT,Axs,KA,EQ,KBC,DO}

    i, wi = _nonuniform_dim_ghost(itp.knots[1], x[1])
    j, wj = _nonuniform_dim_ghost(itp.knots[2], x[2])
    
    result = zero(T)
    @inbounds for (li, wli) in zip(-1:2, wi)
        for (lj, wlj) in zip(-1:2, wj)
            result += wli * wlj * itp.coefs[i+li, j+lj]
        end
    end
    return result
end

# --- 3D with ghost points ---

@inline function (itp::ConvolutionInterpolation{T,3,TCoefs,IT,Axs,KA,Val{3},
                    Val{:n3},EQ,KBC,DO})(x::Vararg{Number,3}) where 
                    {T,TCoefs,IT,Axs,KA,EQ,KBC,DO}

    i, wi = _nonuniform_dim_ghost(itp.knots[1], x[1])
    j, wj = _nonuniform_dim_ghost(itp.knots[2], x[2])
    k, wk = _nonuniform_dim_ghost(itp.knots[3], x[3])
    
    result = zero(T)
    @inbounds for (li, wli) in zip(-1:2, wi)
        for (lj, wlj) in zip(-1:2, wj)
            for (lk, wlk) in zip(-1:2, wk)
                result += wli * wlj * wlk * itp.coefs[i+li, j+lj, k+lk]
            end
        end
    end
    return result
end

# --- ND (N > 3) with ghost points ---

@inline function (itp::ConvolutionInterpolation{T,N,TCoefs,IT,Axs,KA,HigherDimension{N},
                    Val{:n3},EQ,KBC,DO})(x::Vararg{Number,N}) where 
                    {T,N,TCoefs,IT,Axs,KA,EQ,KBC,DO}

    iw = ntuple(d -> _nonuniform_dim_ghost(itp.knots[d], x[d]), N)
    indices = ntuple(d -> iw[d][1], N)
    weights = ntuple(d -> iw[d][2], N)
    
    result = zero(T)
    @inbounds for offsets in Iterators.product(ntuple(_ -> 1:4, N)...)
        w_prod = prod(weights[d][offsets[d]] for d in 1:N)
        idx = ntuple(d -> indices[d] + offsets[d] - 2, N)
        result += w_prod * itp.coefs[idx...]
    end
    return result
end

# ============================================================
# Nonuniform b-kernel interpolation
# ============================================================

"""
    _nb_M_eqs(::Val{degree})

Return the half-stencil size for a nonuniform b-kernel. Compile-time dispatch on kernel
degree: b5→5, b7→6, b9→7, b11→8, b13→9.
"""

# --- Stencil size from degree (compile-time) ---
@inline _nb_M_eqs(::Val{:b5})  = 5
@inline _nb_M_eqs(::Val{:b7})  = 6
@inline _nb_M_eqs(::Val{:b9})  = 7
@inline _nb_M_eqs(::Val{:b11}) = 8
@inline _nb_M_eqs(::Val{:b13}) = 9

"""
    _nb_n_stencil(::Val{degree})

Return the full stencil size as `Val(2*M_eqs)` for compile-time `ntuple` specialization.
"""

# Stencil size = 2 * M_eqs, returned as Val for ntuple
@inline _nb_n_stencil(::Val{:b5})  = Val(10)
@inline _nb_n_stencil(::Val{:b7})  = Val(12)
@inline _nb_n_stencil(::Val{:b9})  = Val(14)
@inline _nb_n_stencil(::Val{:b11}) = Val(16)
@inline _nb_n_stencil(::Val{:b13}) = Val(18)

"""
    _nb_dim(knots_expanded, weight_coeffs, M_eqs, x_val, ::Val{NS}) where {T, NS}

Find interval and evaluate precomputed polynomial weight coefficients for a single dimension
of nonuniform b-kernel interpolation.

Returns `(i, w, h0)`: interval index, stencil weights as `NTuple{NS,T}`, and local spacing.
Weights are evaluated via Horner's method from the precomputed coefficient matrices stored
in `weight_coeffs[k]`, where `k` is the interval index in the original (unexpanded) grid.
"""

# --- Core 1D helper: find interval + evaluate weight polynomials ---
# NS is the stencil size, known at compile time via Val{NS}

@inline function _nb_dim(knots_expanded::AbstractVector{T},
                          weight_coeffs::Vector{Matrix{Float64}},
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
    (itp::ConvolutionInterpolation{...,NB<:Tuple{Vararg{Vector{Matrix{Float64}}}}})(x...)

Evaluate nonuniform b-kernel interpolation. Uses precomputed polynomial weight coefficients
(exact Rational{BigInt} arithmetic, stored as Float64) for high-order convergence on
nonuniform grids. Stencil size is `2*M_eqs` per dimension.

Supports b5 through b13 kernels in 1D through N-D via tensor product of 1D weights.
Derivative scaling uses local interval spacing `h0` per dimension.
"""

# ── Nonuniform b-kernel 1D ───────────────────────────────────
# Dispatch: NB <: Tuple{Vector{Matrix{Float64}}} (not Nothing)

@inline function (itp::ConvolutionInterpolation{T,1,TCoefs,IT,Axs,KA,Val{1},
                    DG,EQ,KBC,DO,FD,SD,SG,NB})(x::Vararg{Number,1}) where 
                    {T,TCoefs,IT,Axs,KA,DG,EQ,KBC,DO,FD,SD,SG,
                     NB<:Tuple{Vector{Matrix{Float64}}}}

    M_eqs = _nb_M_eqs(itp.deg)
    ns = _nb_n_stencil(itp.deg)
    i, w, h0 = _nb_dim(itp.knots[1], itp.nb_weight_coeffs[1], M_eqs, x[1], ns)

    # Derivative order from type parameter
    deriv = DO.parameters[1]

    result = zero(T)
    @inbounds for (j, wj) in enumerate(w)
        result += T(wj) * itp.coefs[i - M_eqs + j]
    end
    return result * (one(T) / h0)^deriv
end

# ── Nonuniform b-kernel 2D ───────────────────────────────────

@inline function (itp::ConvolutionInterpolation{T,2,TCoefs,IT,Axs,KA,Val{2},
                    DG,EQ,KBC,DO,FD,SD,SG,NB})(x::Vararg{Number,2}) where 
                    {T,TCoefs,IT,Axs,KA,DG,EQ,KBC,DO,FD,SD,SG,
                     NB<:Tuple{Vector{Matrix{Float64}},Vector{Matrix{Float64}}}}

    M_eqs = _nb_M_eqs(itp.deg)
    ns = _nb_n_stencil(itp.deg)
    ix, wx, h0x = _nb_dim(itp.knots[1], itp.nb_weight_coeffs[1], M_eqs, x[1], ns)
    iy, wy, h0y = _nb_dim(itp.knots[2], itp.nb_weight_coeffs[2], M_eqs, x[2], ns)
    
    # Derivative order from type parameter
    deriv = DO.parameters[1]
    result = zero(T)
    
    @inbounds for (jx, wjx) in enumerate(wx)
        for (jy, wjy) in enumerate(wy)
            result += T(wjx) * T(wjy) * itp.coefs[ix - M_eqs + jx, iy - M_eqs + jy]
        end
    end
    return result * (one(T) / h0x)^deriv * (one(T) / h0y)^deriv
end

# ── Nonuniform b-kernel 3D ───────────────────────────────────

@inline function (itp::ConvolutionInterpolation{T,3,TCoefs,IT,Axs,KA,Val{3},
                    DG,EQ,KBC,DO,FD,SD,SG,NB})(x::Vararg{Number,3}) where 
                    {T,TCoefs,IT,Axs,KA,DG,EQ,KBC,DO,FD,SD,SG,
                     NB<:Tuple{Vector{Matrix{Float64}},Vector{Matrix{Float64}},Vector{Matrix{Float64}}}}

    M_eqs = _nb_M_eqs(itp.deg)
    ns = _nb_n_stencil(itp.deg)
    ix, wx, h0x = _nb_dim(itp.knots[1], itp.nb_weight_coeffs[1], M_eqs, x[1], ns)
    iy, wy, h0y = _nb_dim(itp.knots[2], itp.nb_weight_coeffs[2], M_eqs, x[2], ns)
    iz, wz, h0z = _nb_dim(itp.knots[3], itp.nb_weight_coeffs[3], M_eqs, x[3], ns)

    # Derivative order from type parameter
    deriv = DO.parameters[1]
    result = zero(T)

    @inbounds for (jx, wjx) in enumerate(wx)
        for (jy, wjy) in enumerate(wy)
            for (jz, wjz) in enumerate(wz)
                result += T(wjx) * T(wjy) * T(wjz) * 
                          itp.coefs[ix - M_eqs + jx, iy - M_eqs + jy, iz - M_eqs + jz]
            end
        end
    end
    return result * (one(T) / h0x)^deriv * (one(T) / h0y)^deriv * (one(T) / h0z)^deriv
end

# ── Nonuniform b-kernel ND (N > 3) ───────────────────────────

@inline function (itp::ConvolutionInterpolation{T,N,TCoefs,IT,Axs,KA,HigherDimension{N},
                    DG,EQ,KBC,DO,FD,SD,SG,NB})(x::Vararg{Number,N}) where 
                    {T,N,TCoefs,IT,Axs,KA,DG,EQ,KBC,DO,FD,SD,SG,
                     NB<:Tuple{Vararg{Vector{Matrix{Float64}}}}}

    M_eqs = _nb_M_eqs(itp.deg)
    ns = _nb_n_stencil(itp.deg)

    iw = ntuple(N) do d
        _nb_dim(itp.knots[d], itp.nb_weight_coeffs[d], M_eqs, x[d], ns)
    end
    indices = ntuple(d -> iw[d][1], N)
    weights = ntuple(d -> iw[d][2], N)
    n_stencil = length(weights[1])
    
    # Derivative order from type parameter
    deriv = DO.parameters[1]
    result = zero(T)
    @inbounds for offsets in Iterators.product(ntuple(_ -> 1:n_stencil, N)...)
        w_prod = prod(T(weights[d][offsets[d]]) for d in 1:N)
        idx = ntuple(d -> indices[d] - M_eqs + offsets[d], N)
        result += w_prod * itp.coefs[idx...]
    end
    return result * prod(one(T) / iw[d][3] for d in 1:N)^deriv
end