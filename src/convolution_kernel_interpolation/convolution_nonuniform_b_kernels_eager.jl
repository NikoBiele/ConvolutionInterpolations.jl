"""
Per-dimension nonuniform b-kernel evaluation functors for `ConvolutionInterpolation`.

Dispatches on `EQ<:NTuple{N,Int}` (per-dim stencil sizes), versus `EQ<:Int` for the
scalar path. Each dimension has its own `M_eqs`, stencil size, and derivative order,
allowing mixed kernel and derivative orders across dimensions on nonuniform grids.

Covers 2D, 3D, and ND paths.
See also: ConvolutionInterpolation, convolution_nonuniform_interpolation.
"""
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

# ── Per-dim helpers ──────────────────────────────────────────

@inline _nb_M_eqs_perdim(::Val{DGS}, d) where {DGS} = _nb_M_eqs(Val(DGS[d]))
@inline _nb_n_stencil_perdim(::Val{DGS}, d) where {DGS} = _nb_n_stencil(Val(DGS[d]))

@inline _nb_M_eqs_perdim(::NonUniformMixedOrderKernel{DGS}, d) where {DGS} = _nb_M_eqs(Val(DGS[d]))
@inline _nb_n_stencil_perdim(::NonUniformMixedOrderKernel{DGS}, d) where {DGS} = _nb_n_stencil(Val(DGS[d]))

# ── Nonuniform b-kernel 1D ───────────────────────────────────
# Dispatch: NB <: Tuple{Vector{Matrix{Float64}}} (not Nothing)

@inline function (itp::ConvolutionInterpolation{T,1,0,TCoefs,Axs,KA,Val{1},
                    NonUniformMixedOrderKernel{DG},EQ,KBC,DerivativeOrder{DO},
                    FD,SD,SG,NB,Val{false},Val{0}})(x::Vararg{Number,1}) where 
                    {T<:AbstractFloat,TCoefs<:AbstractArray{T,1},Axs<:Tuple{<:AbstractVector},
                    KA<:Tuple{<:Nothing},DG,EQ<:Tuple{Int},
                    KBC<:Tuple{<:Tuple{Symbol,Symbol}},DO,FD,SD,SG,
                    NB<:Tuple{Vector{Matrix{Float64}}}}

    M_eqs = _nb_M_eqs_perdim(itp.kernel_sym, 1)
    ns = _nb_n_stencil_perdim(itp.kernel_sym, 1)
    i, w, h0 = _nb_dim(itp.knots[1], itp.nb_weight_coeffs[1], M_eqs, x[1], ns)

    result = zero(T)
    @inbounds for (j, wj) in enumerate(w)
        result += T(wj) * itp.coefs[i - M_eqs + j]
    end
    return result * (one(T) / h0)^DO[1]
end

# ── 2D per-dim ───────────────────────────────────────────────

@inline function (itp::ConvolutionInterpolation{T,2,0,TCoefs,Axs,KA,Val{2},
                    NonUniformMixedOrderKernel{DG},EQ,KBC,DerivativeOrder{DO},
                    FD,SD,SG,NB,Val{false},Val{0}})(x::Vararg{Number,2}) where
                    {T<:AbstractFloat,TCoefs<:AbstractArray{T,2},
                    Axs<:Tuple{<:AbstractVector,<:AbstractVector},
                    KA<:Tuple{<:Nothing,<:Nothing},DG,EQ<:Tuple{Int,Int},
                    KBC<:Tuple{<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol}},DO,FD,SD,SG,
                    NB<:Tuple{Vector{Matrix{Float64}},Vector{Matrix{Float64}}}}

    M_eqs_x = _nb_M_eqs_perdim(itp.kernel_sym, 1)
    M_eqs_y = _nb_M_eqs_perdim(itp.kernel_sym, 2)
    ns_x    = _nb_n_stencil_perdim(itp.kernel_sym, 1)
    ns_y    = _nb_n_stencil_perdim(itp.kernel_sym, 2)
    ix, wx, h0x = _nb_dim(itp.knots[1], itp.nb_weight_coeffs[1], M_eqs_x, x[1], ns_x)
    iy, wy, h0y = _nb_dim(itp.knots[2], itp.nb_weight_coeffs[2], M_eqs_y, x[2], ns_y)

    result = zero(T)
    @inbounds for (jx, wjx) in enumerate(wx)
        for (jy, wjy) in enumerate(wy)
            result += T(wjx) * T(wjy) * itp.coefs[ix - M_eqs_x + jx, iy - M_eqs_y + jy]
        end
    end
    return result * (one(T) / h0x)^DO[1] * (one(T) / h0y)^DO[2]
end

# ── 3D per-dim ───────────────────────────────────────────────

@inline function (itp::ConvolutionInterpolation{T,3,0,TCoefs,Axs,KA,Val{3},
                    NonUniformMixedOrderKernel{DG},EQ,KBC,DerivativeOrder{DO},
                    FD,SD,SG,NB,Val{false},Val{0}})(x::Vararg{Number,3}) where
                    {T<:AbstractFloat,TCoefs<:AbstractArray{T,3},
                    Axs<:Tuple{<:AbstractVector,<:AbstractVector,<:AbstractVector},
                    KA<:Tuple{<:Nothing,<:Nothing,<:Nothing},DG,EQ<:Tuple{Int,Int,Int},
                    KBC<:Tuple{<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol}},
                    DO,FD,SD,SG,
                    NB<:Tuple{Vector{Matrix{Float64}},Vector{Matrix{Float64}},Vector{Matrix{Float64}}}}

    M_eqs_x = _nb_M_eqs_perdim(itp.kernel_sym, 1)
    M_eqs_y = _nb_M_eqs_perdim(itp.kernel_sym, 2)
    M_eqs_z = _nb_M_eqs_perdim(itp.kernel_sym, 3)
    ns_x    = _nb_n_stencil_perdim(itp.kernel_sym, 1)
    ns_y    = _nb_n_stencil_perdim(itp.kernel_sym, 2)
    ns_z    = _nb_n_stencil_perdim(itp.kernel_sym, 3)
    ix, wx, h0x = _nb_dim(itp.knots[1], itp.nb_weight_coeffs[1], M_eqs_x, x[1], ns_x)
    iy, wy, h0y = _nb_dim(itp.knots[2], itp.nb_weight_coeffs[2], M_eqs_y, x[2], ns_y)
    iz, wz, h0z = _nb_dim(itp.knots[3], itp.nb_weight_coeffs[3], M_eqs_z, x[3], ns_z)

    result = zero(T)
    @inbounds for (jx, wjx) in enumerate(wx)
        for (jy, wjy) in enumerate(wy)
            for (jz, wjz) in enumerate(wz)
                result += T(wjx) * T(wjy) * T(wjz) *
                          itp.coefs[ix - M_eqs_x + jx, iy - M_eqs_y + jy, iz - M_eqs_z + jz]
            end
        end
    end
    return result * (one(T) / h0x)^DO[1] * (one(T) / h0y)^DO[2] * (one(T) / h0z)^DO[3]
end

# ── ND per-dim (N > 3) ───────────────────────────────────────

@inline function (itp::ConvolutionInterpolation{T,N,0,TCoefs,Axs,KA,HigherDimension{N},
                    NonUniformMixedOrderKernel{DG},EQ,KBC,DerivativeOrder{DO},FD,SD,
                    SG,NB,Val{false},Val{0}})(x::Vararg{Number,N}) where
                    {T<:AbstractFloat,N,TCoefs<:AbstractArray{T,N},
                    Axs<:Tuple{Vararg{AbstractVector}},
                    KA<:Tuple{Vararg{Nothing}},DG,EQ<:Tuple{Vararg{Int}},
                    KBC<:Tuple{Vararg{Tuple{Symbol,Symbol}}},DO,FD,SD,SG,
                    NB<:Tuple{Vararg{Vector{Matrix{Float64}}}}}

    M_eqs_d = ntuple(d -> _nb_M_eqs_perdim(itp.kernel_sym, d), N)
    ns_d    = ntuple(d -> _nb_n_stencil_perdim(itp.kernel_sym, d), N)

    iw = ntuple(d -> _nb_dim(itp.knots[d], itp.nb_weight_coeffs[d], M_eqs_d[d], x[d], ns_d[d]), N)
    indices = ntuple(d -> iw[d][1], N)
    weights = ntuple(d -> iw[d][2], N)

    result = zero(T)
    @inbounds for offsets in Iterators.product(ntuple(d -> 1:length(weights[d]), N)...)
        w_prod = prod(T(weights[d][offsets[d]]) for d in 1:N)
        idx    = ntuple(d -> indices[d] - M_eqs_d[d] + offsets[d], N)
        result += w_prod * itp.coefs[idx...]
    end
    return result * prod((one(T) / iw[d][3])^DO[d] for d in 1:N)
end