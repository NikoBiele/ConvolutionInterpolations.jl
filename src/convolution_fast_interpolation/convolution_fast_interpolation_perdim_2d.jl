"""
Per-dimension fast convolution interpolation functors for `FastConvolutionInterpolation`.

Dispatches on per-dim eqs `EQ<:NTuple{N,Int}` and per-dim subgrid tuple `Val{SG}` where
`SG::NTuple{N,Symbol}`. Supports mixed kernel and derivative orders per dimension.

2D: 9 explicit functors covering all (sg_x, sg_y) subgrid combinations.
3D: 1 functor (`:linear` subgrid only).
ND: 1 functor (`:linear` subgrid only).

See also: FastConvolutionInterpolation, convolution_fast_interpolation_1d, convolution_fast_interpolation_2d.
"""
# ---------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------

@inline function _perdim_cidx(diff_right::T, pre::AbstractVector) where T
    n    = length(pre)
    cidx = diff_right * T(n - 1) + one(T)
    idx  = clamp(floor(Int, cidx), 1, n - 1)
    return idx, cidx - T(idx), one(T) / T(n - 1)
end

@inline function _perdim_scale(h::NTuple{N,T}, ::Val{DO}) where {N,T,DO}
    s = one(T)
    @inbounds for d in 1:N
        s *= (-one(T) / h[d])^DO[d]
    end
    s
end

# ---------------------------------------------------------------
# 2D position setup helper (shared by all 9 functors)
# Returns i, j, idx_x, t_x, h_pre_x, idx_y, t_y, h_pre_y
# ---------------------------------------------------------------
@inline function _perdim_2d_pos(itp, x::Vararg{Number,2})
    T = eltype(itp.h)
    eqs_x, eqs_y = itp.eqs

    i_float = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T)
    i = clamp(floor(Int, i_float), eqs_x, length(itp.knots[1]) - eqs_x)
    idx_x, t_x, h_pre_x = _perdim_cidx(one(T) - (x[1] - itp.knots[1][i]) / itp.h[1], itp.pre_range[1])

    j_float = (x[2] - itp.knots[2][1]) / itp.h[2] + one(T)
    j = clamp(floor(Int, j_float), eqs_y, length(itp.knots[2]) - eqs_y)
    idx_y, t_y, h_pre_y = _perdim_cidx(one(T) - (x[2] - itp.knots[2][j]) / itp.h[2], itp.pre_range[2])

    return i, j, idx_x, t_x, h_pre_x, idx_y, t_y, h_pre_y
end

# ---------------------------------------------------------------
# Macro to reduce boilerplate for dispatch signature
# ---------------------------------------------------------------
# All 9 2D functors share the same signature shape, differing only in Val{SG}.
# Written out explicitly for clarity and to match package style.

# ==============================================================
# 2D (linear, linear)
# ==============================================================
@inline function (itp::FastConvolutionInterpolation{T,2,0,TCoefs,Axs,KA,Val{2},DG,EQ,PR,KP,KBC,
            DerivativeOrder{DO},FD,SD,Val{(:linear,:linear)},Val{false},Val{0}})(x::Vararg{Number,2}) where
            {T<:AbstractFloat,TCoefs<:AbstractArray{T,2},Axs<:Tuple{<:AbstractVector,<:AbstractVector},
            KA<:Tuple{<:Nothing,<:Nothing},DG<:AbstractMixedConvolutionKernel,EQ<:Tuple{Int,Int},
            PR<:Tuple{<:AbstractVector,<:AbstractVector},KP,KBC<:Tuple{<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol}},DO,FD,SD}

    eqs_x, eqs_y = itp.eqs
    i, j, idx_x, t_x, _, idx_y, t_y, _ = _perdim_2d_pos(itp, x...)
    kp_x = itp.kernel_pre[1]; kp_y = itp.kernel_pre[2]
    r00 = zero(T); r10 = zero(T); r01 = zero(T); r11 = zero(T)
    @inbounds for m in -(eqs_y-1):eqs_y
        ky_0 = kp_y[idx_y,   m+eqs_y]; ky_1 = kp_y[idx_y+1, m+eqs_y]
        @inbounds for l in -(eqs_x-1):eqs_x
            c    = itp.coefs[i+l, j+m]
            kx_0 = kp_x[idx_x,   l+eqs_x]; kx_1 = kp_x[idx_x+1, l+eqs_x]
            r00 += c*kx_0*ky_0; r10 += c*kx_1*ky_0
            r01 += c*kx_0*ky_1; r11 += c*kx_1*ky_1
        end
    end
    ox = one(T)-t_x; oy = one(T)-t_y
    return @fastmath (ox*oy*r00 + t_x*oy*r10 + ox*t_y*r01 + t_x*t_y*r11) * _perdim_scale(itp.h, Val(DO))
end

# ==============================================================
# 2D (cubic, linear)
# ==============================================================
@inline function (itp::FastConvolutionInterpolation{T,2,0,TCoefs,Axs,KA,Val{2},DG,EQ,PR,KP,KBC,
            DerivativeOrder{DO},FD,SD,Val{(:cubic,:linear)},Val{false},Val{0}})(x::Vararg{Number,2}) where
            {T<:AbstractFloat,TCoefs<:AbstractArray{T,2},Axs<:Tuple{<:AbstractVector,<:AbstractVector},
            KA<:Tuple{<:Nothing,<:Nothing},DG<:AbstractMixedConvolutionKernel,EQ<:Tuple{Int,Int},
            PR<:Tuple{<:AbstractVector,<:AbstractVector},KP,KBC<:Tuple{<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol}},DO,FD,SD}

    eqs_x, eqs_y = itp.eqs
    i, j, idx_x, t_x, h_pre_x, idx_y, t_y, _ = _perdim_2d_pos(itp, x...)
    kp_x = itp.kernel_pre[1]; kd_x = itp.kernel_d1_pre[1]; kp_y = itp.kernel_pre[2]
    # accumulators: f=kernel, d=d1 for x; f only for y; 4 corners
    s_ff_00=zero(T); s_ff_10=zero(T); s_ff_01=zero(T); s_ff_11=zero(T)
    s_df_00=zero(T); s_df_10=zero(T); s_df_01=zero(T); s_df_11=zero(T)
    @inbounds for m in -(eqs_y-1):eqs_y
        ky_0 = kp_y[idx_y,   m+eqs_y]; ky_1 = kp_y[idx_y+1, m+eqs_y]
        @inbounds for l in -(eqs_x-1):eqs_x
            c = itp.coefs[i+l, j+m]; col_x = l+eqs_x
            kx_f0 = kp_x[idx_x, col_x]; kx_f1 = kp_x[idx_x+1, col_x]
            kx_d0 = kd_x[idx_x, col_x]; kx_d1 = kd_x[idx_x+1, col_x]
            s_ff_00+=c*kx_f0*ky_0; s_ff_10+=c*kx_f1*ky_0; s_ff_01+=c*kx_f0*ky_1; s_ff_11+=c*kx_f1*ky_1
            s_df_00+=c*kx_d0*ky_0; s_df_10+=c*kx_d1*ky_0; s_df_01+=c*kx_d0*ky_1; s_df_11+=c*kx_d1*ky_1
        end
    end
    oy = one(T)-t_y
    val_y0 = cubic_hermite(t_x, s_ff_00, s_ff_10, s_df_00, s_df_10, h_pre_x)
    val_y1 = cubic_hermite(t_x, s_ff_01, s_ff_11, s_df_01, s_df_11, h_pre_x)
    return @fastmath (oy*val_y0 + t_y*val_y1) * _perdim_scale(itp.h, Val(DO))
end

# ==============================================================
# 2D (quintic, linear)
# ==============================================================
@inline function (itp::FastConvolutionInterpolation{T,2,0,TCoefs,Axs,KA,Val{2},DG,EQ,PR,KP,KBC,
            DerivativeOrder{DO},FD,SD,Val{(:quintic,:linear)},Val{false},Val{0}})(x::Vararg{Number,2}) where
            {T<:AbstractFloat,TCoefs<:AbstractArray{T,2},Axs<:Tuple{<:AbstractVector,<:AbstractVector},
            KA<:Tuple{<:Nothing,<:Nothing},DG<:AbstractMixedConvolutionKernel,EQ<:Tuple{Int,Int},
            PR<:Tuple{<:AbstractVector,<:AbstractVector},KP,KBC<:Tuple{<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol}},DO,FD,SD}

    eqs_x, eqs_y = itp.eqs
    i, j, idx_x, t_x, h_pre_x, idx_y, t_y, _ = _perdim_2d_pos(itp, x...)
    kp_x = itp.kernel_pre[1]; kd_x = itp.kernel_d1_pre[1]; ke_x = itp.kernel_d2_pre[1]
    kp_y = itp.kernel_pre[2]
    s_ff_00=zero(T); s_ff_10=zero(T); s_ff_01=zero(T); s_ff_11=zero(T)
    s_df_00=zero(T); s_df_10=zero(T); s_df_01=zero(T); s_df_11=zero(T)
    s_ef_00=zero(T); s_ef_10=zero(T); s_ef_01=zero(T); s_ef_11=zero(T)
    @inbounds for m in -(eqs_y-1):eqs_y
        ky_0 = kp_y[idx_y,   m+eqs_y]; ky_1 = kp_y[idx_y+1, m+eqs_y]
        @inbounds for l in -(eqs_x-1):eqs_x
            c = itp.coefs[i+l, j+m]; col_x = l+eqs_x
            kx_f0=kp_x[idx_x,col_x]; kx_f1=kp_x[idx_x+1,col_x]
            kx_d0=kd_x[idx_x,col_x]; kx_d1=kd_x[idx_x+1,col_x]
            kx_e0=ke_x[idx_x,col_x]; kx_e1=ke_x[idx_x+1,col_x]
            s_ff_00+=c*kx_f0*ky_0; s_ff_10+=c*kx_f1*ky_0; s_ff_01+=c*kx_f0*ky_1; s_ff_11+=c*kx_f1*ky_1
            s_df_00+=c*kx_d0*ky_0; s_df_10+=c*kx_d1*ky_0; s_df_01+=c*kx_d0*ky_1; s_df_11+=c*kx_d1*ky_1
            s_ef_00+=c*kx_e0*ky_0; s_ef_10+=c*kx_e1*ky_0; s_ef_01+=c*kx_e0*ky_1; s_ef_11+=c*kx_e1*ky_1
        end
    end
    oy = one(T)-t_y
    val_y0 = quintic_hermite(t_x, s_ff_00,s_ff_10, s_df_00,s_df_10, s_ef_00,s_ef_10, h_pre_x)
    val_y1 = quintic_hermite(t_x, s_ff_01,s_ff_11, s_df_01,s_df_11, s_ef_01,s_ef_11, h_pre_x)
    return @fastmath (oy*val_y0 + t_y*val_y1) * _perdim_scale(itp.h, Val(DO))
end

# ==============================================================
# 2D (linear, cubic)
# ==============================================================
@inline function (itp::FastConvolutionInterpolation{T,2,0,TCoefs,Axs,KA,Val{2},DG,EQ,PR,KP,KBC,
            DerivativeOrder{DO},FD,SD,Val{(:linear,:cubic)},Val{false},Val{0}})(x::Vararg{Number,2}) where
            {T<:AbstractFloat,TCoefs<:AbstractArray{T,2},Axs<:Tuple{<:AbstractVector,<:AbstractVector},
            KA<:Tuple{<:Nothing,<:Nothing},DG<:AbstractMixedConvolutionKernel,EQ<:Tuple{Int,Int},
            PR<:Tuple{<:AbstractVector,<:AbstractVector},KP,KBC<:Tuple{<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol}},DO,FD,SD}

    eqs_x, eqs_y = itp.eqs
    i, j, idx_x, t_x, _, idx_y, t_y, h_pre_y = _perdim_2d_pos(itp, x...)
    kp_x = itp.kernel_pre[1]; kp_y = itp.kernel_pre[2]; kd_y = itp.kernel_d1_pre[2]
    s_ff_00=zero(T); s_ff_10=zero(T); s_ff_01=zero(T); s_ff_11=zero(T)
    s_fd_00=zero(T); s_fd_10=zero(T); s_fd_01=zero(T); s_fd_11=zero(T)
    @inbounds for m in -(eqs_y-1):eqs_y
        col_y = m+eqs_y
        ky_f0=kp_y[idx_y,col_y]; ky_f1=kp_y[idx_y+1,col_y]
        ky_d0=kd_y[idx_y,col_y]; ky_d1=kd_y[idx_y+1,col_y]
        @inbounds for l in -(eqs_x-1):eqs_x
            c = itp.coefs[i+l, j+m]
            kx_0=kp_x[idx_x,l+eqs_x]; kx_1=kp_x[idx_x+1,l+eqs_x]
            s_ff_00+=c*kx_0*ky_f0; s_ff_10+=c*kx_1*ky_f0; s_ff_01+=c*kx_0*ky_f1; s_ff_11+=c*kx_1*ky_f1
            s_fd_00+=c*kx_0*ky_d0; s_fd_10+=c*kx_1*ky_d0; s_fd_01+=c*kx_0*ky_d1; s_fd_11+=c*kx_1*ky_d1
        end
    end
    ox = one(T)-t_x
    val_y0 = ox*s_ff_00 + t_x*s_ff_10; dval_y0 = ox*s_fd_00 + t_x*s_fd_10
    val_y1 = ox*s_ff_01 + t_x*s_ff_11; dval_y1 = ox*s_fd_01 + t_x*s_fd_11
    return cubic_hermite(t_y, val_y0, val_y1, dval_y0, dval_y1, h_pre_y) * _perdim_scale(itp.h, Val(DO))
end

# ==============================================================
# 2D (cubic, cubic)  — the standard uniform case equivalent
# ==============================================================
@inline function (itp::FastConvolutionInterpolation{T,2,0,TCoefs,Axs,KA,Val{2},DG,EQ,PR,KP,KBC,
            DerivativeOrder{DO},FD,SD,Val{(:cubic,:cubic)},Val{false},Val{0}})(x::Vararg{Number,2}) where
            {T<:AbstractFloat,TCoefs<:AbstractArray{T,2},Axs<:Tuple{<:AbstractVector,<:AbstractVector},
            KA<:Tuple{<:Nothing,<:Nothing},DG<:AbstractMixedConvolutionKernel,EQ<:Tuple{Int,Int},
            PR<:Tuple{<:AbstractVector,<:AbstractVector},KP,KBC<:Tuple{<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol}},DO,FD,SD}

    eqs_x, eqs_y = itp.eqs
    i, j, idx_x, t_x, h_pre_x, idx_y, t_y, h_pre_y = _perdim_2d_pos(itp, x...)
    kp_x=itp.kernel_pre[1]; kd_x=itp.kernel_d1_pre[1]
    kp_y=itp.kernel_pre[2]; kd_y=itp.kernel_d1_pre[2]
    s_ff_00=zero(T); s_ff_10=zero(T); s_ff_01=zero(T); s_ff_11=zero(T)
    s_df_00=zero(T); s_df_10=zero(T); s_df_01=zero(T); s_df_11=zero(T)
    s_fd_00=zero(T); s_fd_10=zero(T); s_fd_01=zero(T); s_fd_11=zero(T)
    s_dd_00=zero(T); s_dd_10=zero(T); s_dd_01=zero(T); s_dd_11=zero(T)
    @inbounds for m in -(eqs_y-1):eqs_y
        col_y=m+eqs_y
        ky_f0=kp_y[idx_y,col_y]; ky_f1=kp_y[idx_y+1,col_y]
        ky_d0=kd_y[idx_y,col_y]; ky_d1=kd_y[idx_y+1,col_y]
        @inbounds for l in -(eqs_x-1):eqs_x
            c=itp.coefs[i+l,j+m]; col_x=l+eqs_x
            kx_f0=kp_x[idx_x,col_x]; kx_f1=kp_x[idx_x+1,col_x]
            kx_d0=kd_x[idx_x,col_x]; kx_d1=kd_x[idx_x+1,col_x]
            s_ff_00+=c*kx_f0*ky_f0; s_ff_10+=c*kx_f1*ky_f0; s_ff_01+=c*kx_f0*ky_f1; s_ff_11+=c*kx_f1*ky_f1
            s_df_00+=c*kx_d0*ky_f0; s_df_10+=c*kx_d1*ky_f0; s_df_01+=c*kx_d0*ky_f1; s_df_11+=c*kx_d1*ky_f1
            s_fd_00+=c*kx_f0*ky_d0; s_fd_10+=c*kx_f1*ky_d0; s_fd_01+=c*kx_f0*ky_d1; s_fd_11+=c*kx_f1*ky_d1
            s_dd_00+=c*kx_d0*ky_d0; s_dd_10+=c*kx_d1*ky_d0; s_dd_01+=c*kx_d0*ky_d1; s_dd_11+=c*kx_d1*ky_d1
        end
    end
    val_y0  = cubic_hermite(t_x, s_ff_00,s_ff_10, s_df_00,s_df_10, h_pre_x)
    val_y1  = cubic_hermite(t_x, s_ff_01,s_ff_11, s_df_01,s_df_11, h_pre_x)
    dval_y0 = cubic_hermite(t_x, s_fd_00,s_fd_10, s_dd_00,s_dd_10, h_pre_x)
    dval_y1 = cubic_hermite(t_x, s_fd_01,s_fd_11, s_dd_01,s_dd_11, h_pre_x)
    return cubic_hermite(t_y, val_y0,val_y1, dval_y0,dval_y1, h_pre_y) * _perdim_scale(itp.h, Val(DO))
end

# ==============================================================
# 2D (quintic, cubic)
# ==============================================================
@inline function (itp::FastConvolutionInterpolation{T,2,0,TCoefs,Axs,KA,Val{2},DG,EQ,PR,KP,KBC,
            DerivativeOrder{DO},FD,SD,Val{(:quintic,:cubic)},Val{false},Val{0}})(x::Vararg{Number,2}) where
            {T<:AbstractFloat,TCoefs<:AbstractArray{T,2},Axs<:Tuple{<:AbstractVector,<:AbstractVector},
            KA<:Tuple{<:Nothing,<:Nothing},DG<:AbstractMixedConvolutionKernel,EQ<:Tuple{Int,Int},
            PR<:Tuple{<:AbstractVector,<:AbstractVector},KP,KBC<:Tuple{<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol}},DO,FD,SD}

    eqs_x, eqs_y = itp.eqs
    i, j, idx_x, t_x, h_pre_x, idx_y, t_y, h_pre_y = _perdim_2d_pos(itp, x...)
    kp_x=itp.kernel_pre[1]; kd_x=itp.kernel_d1_pre[1]; ke_x=itp.kernel_d2_pre[1]
    kp_y=itp.kernel_pre[2]; kd_y=itp.kernel_d1_pre[2]
    s_ff_00=zero(T); s_ff_10=zero(T); s_ff_01=zero(T); s_ff_11=zero(T)
    s_df_00=zero(T); s_df_10=zero(T); s_df_01=zero(T); s_df_11=zero(T)
    s_ef_00=zero(T); s_ef_10=zero(T); s_ef_01=zero(T); s_ef_11=zero(T)
    s_fd_00=zero(T); s_fd_10=zero(T); s_fd_01=zero(T); s_fd_11=zero(T)
    s_dd_00=zero(T); s_dd_10=zero(T); s_dd_01=zero(T); s_dd_11=zero(T)
    s_ed_00=zero(T); s_ed_10=zero(T); s_ed_01=zero(T); s_ed_11=zero(T)
    @inbounds for m in -(eqs_y-1):eqs_y
        col_y=m+eqs_y
        ky_f0=kp_y[idx_y,col_y]; ky_f1=kp_y[idx_y+1,col_y]
        ky_d0=kd_y[idx_y,col_y]; ky_d1=kd_y[idx_y+1,col_y]
        @inbounds for l in -(eqs_x-1):eqs_x
            c=itp.coefs[i+l,j+m]; col_x=l+eqs_x
            kx_f0=kp_x[idx_x,col_x]; kx_f1=kp_x[idx_x+1,col_x]
            kx_d0=kd_x[idx_x,col_x]; kx_d1=kd_x[idx_x+1,col_x]
            kx_e0=ke_x[idx_x,col_x]; kx_e1=ke_x[idx_x+1,col_x]
            s_ff_00+=c*kx_f0*ky_f0; s_ff_10+=c*kx_f1*ky_f0; s_ff_01+=c*kx_f0*ky_f1; s_ff_11+=c*kx_f1*ky_f1
            s_df_00+=c*kx_d0*ky_f0; s_df_10+=c*kx_d1*ky_f0; s_df_01+=c*kx_d0*ky_f1; s_df_11+=c*kx_d1*ky_f1
            s_ef_00+=c*kx_e0*ky_f0; s_ef_10+=c*kx_e1*ky_f0; s_ef_01+=c*kx_e0*ky_f1; s_ef_11+=c*kx_e1*ky_f1
            s_fd_00+=c*kx_f0*ky_d0; s_fd_10+=c*kx_f1*ky_d0; s_fd_01+=c*kx_f0*ky_d1; s_fd_11+=c*kx_f1*ky_d1
            s_dd_00+=c*kx_d0*ky_d0; s_dd_10+=c*kx_d1*ky_d0; s_dd_01+=c*kx_d0*ky_d1; s_dd_11+=c*kx_d1*ky_d1
            s_ed_00+=c*kx_e0*ky_d0; s_ed_10+=c*kx_e1*ky_d0; s_ed_01+=c*kx_e0*ky_d1; s_ed_11+=c*kx_e1*ky_d1
        end
    end
    val_y0  = quintic_hermite(t_x, s_ff_00,s_ff_10, s_df_00,s_df_10, s_ef_00,s_ef_10, h_pre_x)
    val_y1  = quintic_hermite(t_x, s_ff_01,s_ff_11, s_df_01,s_df_11, s_ef_01,s_ef_11, h_pre_x)
    dval_y0 = quintic_hermite(t_x, s_fd_00,s_fd_10, s_dd_00,s_dd_10, s_ed_00,s_ed_10, h_pre_x)
    dval_y1 = quintic_hermite(t_x, s_fd_01,s_fd_11, s_dd_01,s_dd_11, s_ed_01,s_ed_11, h_pre_x)
    return cubic_hermite(t_y, val_y0,val_y1, dval_y0,dval_y1, h_pre_y) * _perdim_scale(itp.h, Val(DO))
end

# ==============================================================
# 2D (linear, quintic)
# ==============================================================
@inline function (itp::FastConvolutionInterpolation{T,2,0,TCoefs,Axs,KA,Val{2},DG,EQ,PR,KP,KBC,
            DerivativeOrder{DO},FD,SD,Val{(:linear,:quintic)},Val{false},Val{0}})(x::Vararg{Number,2}) where
            {T<:AbstractFloat,TCoefs<:AbstractArray{T,2},Axs<:Tuple{<:AbstractVector,<:AbstractVector},
            KA<:Tuple{<:Nothing,<:Nothing},DG<:AbstractMixedConvolutionKernel,EQ<:Tuple{Int,Int},
            PR<:Tuple{<:AbstractVector,<:AbstractVector},KP,KBC<:Tuple{<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol}},DO,FD,SD}

    eqs_x, eqs_y = itp.eqs
    i, j, idx_x, t_x, _, idx_y, t_y, h_pre_y = _perdim_2d_pos(itp, x...)
    kp_x=itp.kernel_pre[1]
    kp_y=itp.kernel_pre[2]; kd_y=itp.kernel_d1_pre[2]; ke_y=itp.kernel_d2_pre[2]
    s_ff_00=zero(T); s_ff_10=zero(T); s_ff_01=zero(T); s_ff_11=zero(T)
    s_fd_00=zero(T); s_fd_10=zero(T); s_fd_01=zero(T); s_fd_11=zero(T)
    s_fe_00=zero(T); s_fe_10=zero(T); s_fe_01=zero(T); s_fe_11=zero(T)
    @inbounds for m in -(eqs_y-1):eqs_y
        col_y=m+eqs_y
        ky_f0=kp_y[idx_y,col_y]; ky_f1=kp_y[idx_y+1,col_y]
        ky_d0=kd_y[idx_y,col_y]; ky_d1=kd_y[idx_y+1,col_y]
        ky_e0=ke_y[idx_y,col_y]; ky_e1=ke_y[idx_y+1,col_y]
        @inbounds for l in -(eqs_x-1):eqs_x
            c=itp.coefs[i+l,j+m]
            kx_0=kp_x[idx_x,l+eqs_x]; kx_1=kp_x[idx_x+1,l+eqs_x]
            s_ff_00+=c*kx_0*ky_f0; s_ff_10+=c*kx_1*ky_f0; s_ff_01+=c*kx_0*ky_f1; s_ff_11+=c*kx_1*ky_f1
            s_fd_00+=c*kx_0*ky_d0; s_fd_10+=c*kx_1*ky_d0; s_fd_01+=c*kx_0*ky_d1; s_fd_11+=c*kx_1*ky_d1
            s_fe_00+=c*kx_0*ky_e0; s_fe_10+=c*kx_1*ky_e0; s_fe_01+=c*kx_0*ky_e1; s_fe_11+=c*kx_1*ky_e1
        end
    end
    ox = one(T)-t_x
    val_y0  = ox*s_ff_00 + t_x*s_ff_10; d1_y0 = ox*s_fd_00 + t_x*s_fd_10; d2_y0 = ox*s_fe_00 + t_x*s_fe_10
    val_y1  = ox*s_ff_01 + t_x*s_ff_11; d1_y1 = ox*s_fd_01 + t_x*s_fd_11; d2_y1 = ox*s_fe_01 + t_x*s_fe_11
    return quintic_hermite(t_y, val_y0,val_y1, d1_y0,d1_y1, d2_y0,d2_y1, h_pre_y) * _perdim_scale(itp.h, Val(DO))
end

# ==============================================================
# 2D (cubic, quintic)
# ==============================================================
@inline function (itp::FastConvolutionInterpolation{T,2,0,TCoefs,Axs,KA,Val{2},DG,EQ,PR,KP,KBC,
            DerivativeOrder{DO},FD,SD,Val{(:cubic,:quintic)},Val{false},Val{0}})(x::Vararg{Number,2}) where
            {T<:AbstractFloat,TCoefs<:AbstractArray{T,2},Axs<:Tuple{<:AbstractVector,<:AbstractVector},
            KA<:Tuple{<:Nothing,<:Nothing},DG<:AbstractMixedConvolutionKernel,EQ<:Tuple{Int,Int},
            PR<:Tuple{<:AbstractVector,<:AbstractVector},KP,KBC<:Tuple{<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol}},DO,FD,SD}

    eqs_x, eqs_y = itp.eqs
    i, j, idx_x, t_x, h_pre_x, idx_y, t_y, h_pre_y = _perdim_2d_pos(itp, x...)
    kp_x=itp.kernel_pre[1]; kd_x=itp.kernel_d1_pre[1]
    kp_y=itp.kernel_pre[2]; kd_y=itp.kernel_d1_pre[2]; ke_y=itp.kernel_d2_pre[2]
    s_ff_00=zero(T); s_ff_10=zero(T); s_ff_01=zero(T); s_ff_11=zero(T)
    s_df_00=zero(T); s_df_10=zero(T); s_df_01=zero(T); s_df_11=zero(T)
    s_fd_00=zero(T); s_fd_10=zero(T); s_fd_01=zero(T); s_fd_11=zero(T)
    s_dd_00=zero(T); s_dd_10=zero(T); s_dd_01=zero(T); s_dd_11=zero(T)
    s_fe_00=zero(T); s_fe_10=zero(T); s_fe_01=zero(T); s_fe_11=zero(T)
    s_de_00=zero(T); s_de_10=zero(T); s_de_01=zero(T); s_de_11=zero(T)
    @inbounds for m in -(eqs_y-1):eqs_y
        col_y=m+eqs_y
        ky_f0=kp_y[idx_y,col_y]; ky_f1=kp_y[idx_y+1,col_y]
        ky_d0=kd_y[idx_y,col_y]; ky_d1=kd_y[idx_y+1,col_y]
        ky_e0=ke_y[idx_y,col_y]; ky_e1=ke_y[idx_y+1,col_y]
        @inbounds for l in -(eqs_x-1):eqs_x
            c=itp.coefs[i+l,j+m]; col_x=l+eqs_x
            kx_f0=kp_x[idx_x,col_x]; kx_f1=kp_x[idx_x+1,col_x]
            kx_d0=kd_x[idx_x,col_x]; kx_d1=kd_x[idx_x+1,col_x]
            s_ff_00+=c*kx_f0*ky_f0; s_ff_10+=c*kx_f1*ky_f0; s_ff_01+=c*kx_f0*ky_f1; s_ff_11+=c*kx_f1*ky_f1
            s_df_00+=c*kx_d0*ky_f0; s_df_10+=c*kx_d1*ky_f0; s_df_01+=c*kx_d0*ky_f1; s_df_11+=c*kx_d1*ky_f1
            s_fd_00+=c*kx_f0*ky_d0; s_fd_10+=c*kx_f1*ky_d0; s_fd_01+=c*kx_f0*ky_d1; s_fd_11+=c*kx_f1*ky_d1
            s_dd_00+=c*kx_d0*ky_d0; s_dd_10+=c*kx_d1*ky_d0; s_dd_01+=c*kx_d0*ky_d1; s_dd_11+=c*kx_d1*ky_d1
            s_fe_00+=c*kx_f0*ky_e0; s_fe_10+=c*kx_f1*ky_e0; s_fe_01+=c*kx_f0*ky_e1; s_fe_11+=c*kx_f1*ky_e1
            s_de_00+=c*kx_d0*ky_e0; s_de_10+=c*kx_d1*ky_e0; s_de_01+=c*kx_d0*ky_e1; s_de_11+=c*kx_d1*ky_e1
        end
    end
    val_y0 = cubic_hermite(t_x, s_ff_00,s_ff_10, s_df_00,s_df_10, h_pre_x)
    val_y1 = cubic_hermite(t_x, s_ff_01,s_ff_11, s_df_01,s_df_11, h_pre_x)
    d1_y0  = cubic_hermite(t_x, s_fd_00,s_fd_10, s_dd_00,s_dd_10, h_pre_x)
    d1_y1  = cubic_hermite(t_x, s_fd_01,s_fd_11, s_dd_01,s_dd_11, h_pre_x)
    d2_y0  = cubic_hermite(t_x, s_fe_00,s_fe_10, s_de_00,s_de_10, h_pre_x)
    d2_y1  = cubic_hermite(t_x, s_fe_01,s_fe_11, s_de_01,s_de_11, h_pre_x)
    return quintic_hermite(t_y, val_y0,val_y1, d1_y0,d1_y1, d2_y0,d2_y1, h_pre_y) * _perdim_scale(itp.h, Val(DO))
end

# ==============================================================
# 2D (quintic, quintic)
# ==============================================================
@inline function (itp::FastConvolutionInterpolation{T,2,0,TCoefs,Axs,KA,Val{2},DG,EQ,PR,KP,KBC,
            DerivativeOrder{DO},FD,SD,Val{(:quintic,:quintic)},Val{false},Val{0}})(x::Vararg{Number,2}) where
            {T<:AbstractFloat,TCoefs<:AbstractArray{T,2},Axs<:Tuple{<:AbstractVector,<:AbstractVector},
            KA<:Tuple{<:Nothing,<:Nothing},DG<:AbstractMixedConvolutionKernel,EQ<:Tuple{Int,Int},
            PR<:Tuple{<:AbstractVector,<:AbstractVector},KP,KBC<:Tuple{<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol}},DO,FD,SD}
            
    eqs_x, eqs_y = itp.eqs
    i, j, idx_x, t_x, h_pre_x, idx_y, t_y, h_pre_y = _perdim_2d_pos(itp, x...)
    kp_x=itp.kernel_pre[1]; kd_x=itp.kernel_d1_pre[1]; ke_x=itp.kernel_d2_pre[1]
    kp_y=itp.kernel_pre[2]; kd_y=itp.kernel_d1_pre[2]; ke_y=itp.kernel_d2_pre[2]
    s_ff_00=zero(T); s_ff_10=zero(T); s_ff_01=zero(T); s_ff_11=zero(T)
    s_df_00=zero(T); s_df_10=zero(T); s_df_01=zero(T); s_df_11=zero(T)
    s_ef_00=zero(T); s_ef_10=zero(T); s_ef_01=zero(T); s_ef_11=zero(T)
    s_fd_00=zero(T); s_fd_10=zero(T); s_fd_01=zero(T); s_fd_11=zero(T)
    s_dd_00=zero(T); s_dd_10=zero(T); s_dd_01=zero(T); s_dd_11=zero(T)
    s_ed_00=zero(T); s_ed_10=zero(T); s_ed_01=zero(T); s_ed_11=zero(T)
    s_fe_00=zero(T); s_fe_10=zero(T); s_fe_01=zero(T); s_fe_11=zero(T)
    s_de_00=zero(T); s_de_10=zero(T); s_de_01=zero(T); s_de_11=zero(T)
    s_ee_00=zero(T); s_ee_10=zero(T); s_ee_01=zero(T); s_ee_11=zero(T)
    @inbounds for m in -(eqs_y-1):eqs_y
        col_y=m+eqs_y
        ky_f0=kp_y[idx_y,col_y]; ky_f1=kp_y[idx_y+1,col_y]
        ky_d0=kd_y[idx_y,col_y]; ky_d1=kd_y[idx_y+1,col_y]
        ky_e0=ke_y[idx_y,col_y]; ky_e1=ke_y[idx_y+1,col_y]
        @inbounds for l in -(eqs_x-1):eqs_x
            c=itp.coefs[i+l,j+m]; col_x=l+eqs_x
            kx_f0=kp_x[idx_x,col_x]; kx_f1=kp_x[idx_x+1,col_x]
            kx_d0=kd_x[idx_x,col_x]; kx_d1=kd_x[idx_x+1,col_x]
            kx_e0=ke_x[idx_x,col_x]; kx_e1=ke_x[idx_x+1,col_x]
            s_ff_00+=c*kx_f0*ky_f0; s_ff_10+=c*kx_f1*ky_f0; s_ff_01+=c*kx_f0*ky_f1; s_ff_11+=c*kx_f1*ky_f1
            s_df_00+=c*kx_d0*ky_f0; s_df_10+=c*kx_d1*ky_f0; s_df_01+=c*kx_d0*ky_f1; s_df_11+=c*kx_d1*ky_f1
            s_ef_00+=c*kx_e0*ky_f0; s_ef_10+=c*kx_e1*ky_f0; s_ef_01+=c*kx_e0*ky_f1; s_ef_11+=c*kx_e1*ky_f1
            s_fd_00+=c*kx_f0*ky_d0; s_fd_10+=c*kx_f1*ky_d0; s_fd_01+=c*kx_f0*ky_d1; s_fd_11+=c*kx_f1*ky_d1
            s_dd_00+=c*kx_d0*ky_d0; s_dd_10+=c*kx_d1*ky_d0; s_dd_01+=c*kx_d0*ky_d1; s_dd_11+=c*kx_d1*ky_d1
            s_ed_00+=c*kx_e0*ky_d0; s_ed_10+=c*kx_e1*ky_d0; s_ed_01+=c*kx_e0*ky_d1; s_ed_11+=c*kx_e1*ky_d1
            s_fe_00+=c*kx_f0*ky_e0; s_fe_10+=c*kx_f1*ky_e0; s_fe_01+=c*kx_f0*ky_e1; s_fe_11+=c*kx_f1*ky_e1
            s_de_00+=c*kx_d0*ky_e0; s_de_10+=c*kx_d1*ky_e0; s_de_01+=c*kx_d0*ky_e1; s_de_11+=c*kx_d1*ky_e1
            s_ee_00+=c*kx_e0*ky_e0; s_ee_10+=c*kx_e1*ky_e0; s_ee_01+=c*kx_e0*ky_e1; s_ee_11+=c*kx_e1*ky_e1
        end
    end
    val_y0 = quintic_hermite(t_x, s_ff_00,s_ff_10, s_df_00,s_df_10, s_ef_00,s_ef_10, h_pre_x)
    d1_y0  = quintic_hermite(t_x, s_fd_00,s_fd_10, s_dd_00,s_dd_10, s_ed_00,s_ed_10, h_pre_x)
    d2_y0  = quintic_hermite(t_x, s_fe_00,s_fe_10, s_de_00,s_de_10, s_ee_00,s_ee_10, h_pre_x)
    val_y1 = quintic_hermite(t_x, s_ff_01,s_ff_11, s_df_01,s_df_11, s_ef_01,s_ef_11, h_pre_x)
    d1_y1  = quintic_hermite(t_x, s_fd_01,s_fd_11, s_dd_01,s_dd_11, s_ed_01,s_ed_11, h_pre_x)
    d2_y1  = quintic_hermite(t_x, s_fe_01,s_fe_11, s_de_01,s_de_11, s_ee_01,s_ee_11, h_pre_x)
    return quintic_hermite(t_y, val_y0,val_y1, d1_y0,d1_y1, d2_y0,d2_y1, h_pre_y) * _perdim_scale(itp.h, Val(DO))
end