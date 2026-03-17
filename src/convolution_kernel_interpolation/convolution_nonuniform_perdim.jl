"""
Per-dimension nonuniform b-kernel evaluation functors for `ConvolutionInterpolation`.

Dispatches on `EQ<:NTuple{N,Int}` (per-dim stencil sizes), versus `EQ<:Int` for the
scalar path. Each dimension has its own `M_eqs`, stencil size, and derivative order,
allowing mixed kernel and derivative orders across dimensions on nonuniform grids.

Covers 2D, 3D, and ND paths.
See also: ConvolutionInterpolation, convolution_nonuniform_interpolation.
"""

# ── Per-dim helpers ──────────────────────────────────────────

@inline _nb_M_eqs_perdim(::Val{DGS}, d) where {DGS} = _nb_M_eqs(Val(DGS[d]))
@inline _nb_n_stencil_perdim(::Val{DGS}, d) where {DGS} = _nb_n_stencil(Val(DGS[d]))

# ── 2D per-dim ───────────────────────────────────────────────

@inline function (itp::ConvolutionInterpolation{T,2,TCoefs,IT,Axs,KA,Val{2},
                    DG,EQ,KBC,DerivativeOrder{DO},FD,SD,SG,NB})(x::Vararg{Number,2}) where
                    {T,TCoefs,IT,Axs,KA,DG,EQ<:NTuple{2,Int},KBC,DO,FD,SD,SG,
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

@inline function (itp::ConvolutionInterpolation{T,3,TCoefs,IT,Axs,KA,Val{3},
                    DG,EQ,KBC,DerivativeOrder{DO},FD,SD,SG,NB})(x::Vararg{Number,3}) where
                    {T,TCoefs,IT,Axs,KA,DG,EQ<:NTuple{3,Int},KBC,DO,FD,SD,SG,
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

@inline function (itp::ConvolutionInterpolation{T,N,TCoefs,IT,Axs,KA,HigherDimension{N},
                    DG,EQ,KBC,DerivativeOrder{DO},FD,SD,SG,NB})(x::Vararg{Number,N}) where
                    {N,T,TCoefs,IT,Axs,KA,DG,EQ<:NTuple{N,Int},KBC,DO,FD,SD,SG,
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