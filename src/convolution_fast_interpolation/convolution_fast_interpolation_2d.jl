"""
    (itp::FastConvolutionInterpolation{T,2,...})(x::Number, y::Number)

Evaluate 2D fast convolution interpolation at coordinates `(x, y)`.

Dispatches on kernel type and subgrid parameter:

**Specialized kernels** (no precomputed table):
- `:a0` — Nearest neighbor, ~7ns
- `:a1` — Bilinear interpolation, ~8ns

**Higher-order kernels** with subgrid modes:
- `:linear` — Bilinear interpolation between four convolution results (needs `precompute≥10_000`), 4 tensor products per offset pair
- `:cubic` — Nested cubic Hermite using kernel first derivatives (default, 16 tensor products per offset pair)
- `:quintic` — Nested quintic Hermite using first and second derivatives (36 tensor products per offset pair, highest accuracy)

O(1) evaluation time, allocation-free. The 2D kernel is formed as a tensor product
of 1D kernels: `K_2D(x,y) = K_1D(x) * K_1D(y)`. Higher subgrid orders use
analytically predifferentiated kernel coefficients for nested Hermite interpolation —
first along x at each y-bracket, then along y.

Cubic and quintic are the highest subgrid modes available; 3D+ uses linear subgrid
as tensor product counts grow as `(order+1)^N`.

Results scaled by `(-1/h_x)^derivative * (-1/h_y)^derivative`.

See also: `FastConvolutionInterpolation`, `cubic_hermite`, `quintic_hermite`.
"""

@inline function (itp::FastConvolutionInterpolation{T,2,TCoefs,IT,Axs,KA,Val{2},Val{:a0},EQ,PR,KP})(x::Vararg{Number,2}) where {T,TCoefs,IT,Axs,KA,EQ,PR,KP}
    # specialized dispatch for 2d nearest neighbor kernel
    
    # First dimension (x)
    i_float = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T)
    i = clamp(floor(Int, i_float), itp.eqs, length(itp.knots[1]) - itp.eqs)
    x_diff_left = (x[1] - itp.knots[1][i]) / itp.h[1]  # Recompute from actual knot

    # Second dimension (y)
    j_float = (x[2] - itp.knots[2][1]) / itp.h[2] + one(T)
    j = clamp(floor(Int, j_float), itp.eqs, length(itp.knots[2]) - itp.eqs)
    y_diff_left = (x[2] - itp.knots[2][j]) / itp.h[2]  # Recompute from actual knot

    if x_diff_left < 0.5 && y_diff_left < 0.5
        return itp.coefs[i, j]
    elseif x_diff_left < 0.5 && y_diff_left >= 0.5
        return itp.coefs[i, j+1]
    elseif x_diff_left >= 0.5 && y_diff_left < 0.5
        return itp.coefs[i+1, j]
    else # if x_diff_left >= 0.5 && y_diff_left >= 0.5
        return itp.coefs[i+1, j+1]
    end
end

@inline function (itp::FastConvolutionInterpolation{T,2,TCoefs,IT,Axs,KA,Val{2},Val{:a1},EQ,PR,KP,KBC,DO})(x::Vararg{Number,2}) where {T,TCoefs,IT,Axs,KA,EQ,PR,KP,KBC,DO}
    # specialized dispatch for 2d linear kernel
    # First dimension (x)
    i_float = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T)
    i = clamp(floor(Int, i_float), itp.eqs, length(itp.knots[1]) - itp.eqs)
    x_diff_left = (x[1] - itp.knots[1][i]) / itp.h[1]  # Recompute from actual knot

    # Second dimension (y)
    j_float = (x[2] - itp.knots[2][1]) / itp.h[2] + one(T)
    j = clamp(floor(Int, j_float), itp.eqs, length(itp.knots[2]) - itp.eqs)
    y_diff_left = (x[2] - itp.knots[2][j]) / itp.h[2]  # Recompute from actual knot

    # Bilinear interpolation
    return @inbounds @fastmath ((1-x_diff_left)*(1-y_diff_left)*itp.coefs[i, j] + 
                        x_diff_left*(1-y_diff_left)*itp.coefs[i+1, j] + 
                        (1-x_diff_left)*y_diff_left*itp.coefs[i, j+1] + 
                        x_diff_left*y_diff_left*itp.coefs[i+1, j+1]) *
                        (-one(T)/itp.h[1])^(DO.parameters[1]) * (-one(T)/itp.h[2])^(DO.parameters[1])
end

function (itp::FastConvolutionInterpolation{T,2,TCoefs,IT,Axs,KA,Val{2},
            HigherOrderKernel{DG},EQ,PR,KP,KBC,DO,FD,SD,Val{:linear}})(x::Vararg{Number,2}) where 
            {T,TCoefs,IT,Axs,KA,DG,EQ,PR,KP,KBC,DO,FD,SD}
    # specialized dispatch for 2d higher-order kernels
    
    # First dimension (x)
    i_float = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T)
    i = clamp(floor(Int, i_float), itp.eqs, length(itp.knots[1]) - itp.eqs)
    x_diff_left = (x[1] - itp.knots[1][i]) / itp.h[1]  # Recompute from actual knot
    x_diff_right = one(T) - x_diff_left
    # Direct pre_range index - no search
    idx_x = clamp(floor(Int, x_diff_right * (length(itp.pre_range) - one(Int64))) + one(Int64), one(Int64), length(itp.pre_range) - one(Int64))
    # Linear interpolation weight
    t_x = (x_diff_right - itp.pre_range[idx_x]) / (itp.pre_range[idx_x+1] - itp.pre_range[idx_x])

    # Second dimension (y)
    j_float = (x[2] - itp.knots[2][1]) / itp.h[2] + one(T)
    j = clamp(floor(Int, j_float), itp.eqs, length(itp.knots[2]) - itp.eqs)
    y_diff_left = (x[2] - itp.knots[2][j]) / itp.h[2]  # Recompute from actual knot
    y_diff_right = one(T) - y_diff_left 
    # Direct pre_range index - no search
    idx_y = clamp(floor(Int, y_diff_right * (length(itp.pre_range) - one(T))) + one(Int64), one(Int64), length(itp.pre_range) - one(Int64))
    # Linear interpolation weight
    t_y = (y_diff_right - itp.pre_range[idx_y]) / (itp.pre_range[idx_y+1] - itp.pre_range[idx_y])
    
    # Initialize results
    result_00 = zero(T)
    result_10 = zero(T)
    result_01 = zero(T)
    result_11 = zero(T)

    # Single pass through the convolution support
    @inbounds for m in -(itp.eqs-1):itp.eqs
        ky_0 = itp.kernel_pre[idx_y, m+itp.eqs]
        ky_1 = itp.kernel_pre[idx_y+1, m+itp.eqs]
        
        @inbounds for l in -(itp.eqs-1):itp.eqs
            coef = itp.coefs[i+l, j+m]
            kx_0 = itp.kernel_pre[idx_x, l+itp.eqs]
            kx_1 = itp.kernel_pre[idx_x+1, l+itp.eqs]
            
            # Accumulate all 4 corners - pure tensor product
            result_00 += coef * kx_0 * ky_0
            result_10 += coef * kx_1 * ky_0
            result_01 += coef * kx_0 * ky_1
            result_11 += coef * kx_1 * ky_1
        end
    end

    # Bilinear interpolation
    return @fastmath ((one(T)-t_x)*(one(T)-t_y)*result_00 + t_x*(one(T)-t_y)*result_10 + (one(T)-t_x)*t_y*result_01 + t_x*t_y*result_11) *
                        (-one(T)/itp.h[1])^(DO.parameters[1]) * (-one(T)/itp.h[2])^(DO.parameters[1])
end

@inline function (itp::FastConvolutionInterpolation{T,2,TCoefs,IT,Axs,KA,Val{2},
                    HigherOrderKernel{DG},EQ,PR,KP,KBC,DO,FD,SD,Val{:cubic}})(x::Vararg{Number,2}) where 
                    {T,TCoefs,IT,Axs,KA,DG,EQ,PR,KP,KBC,DO,FD,SD}

    # First dimension (x)
    i_float = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T)
    i = clamp(floor(Int, i_float), itp.eqs, length(itp.knots[1]) - itp.eqs)
    x_diff_left = (x[1] - itp.knots[1][i]) / itp.h[1]
    x_diff_right = one(T) - x_diff_left
    n_pre = length(itp.pre_range)
    continuous_idx_x = x_diff_right * T(n_pre - 1) + one(T)
    idx_x = clamp(floor(Int, continuous_idx_x), 1, n_pre - 1)
    t_x = continuous_idx_x - T(idx_x)

    # Second dimension (y)
    j_float = (x[2] - itp.knots[2][1]) / itp.h[2] + one(T)
    j = clamp(floor(Int, j_float), itp.eqs, length(itp.knots[2]) - itp.eqs)
    y_diff_left = (x[2] - itp.knots[2][j]) / itp.h[2]
    y_diff_right = one(T) - y_diff_left
    continuous_idx_y = y_diff_right * T(n_pre - 1) + one(T)
    idx_y = clamp(floor(Int, continuous_idx_y), 1, n_pre - 1)
    t_y = continuous_idx_y - T(idx_y)

    # 16 accumulators: f and df for 2 x-positions × 2 y-positions
    # Naming: s_[x-quantity][y-quantity]_[x-idx][y-idx]
    # f = function value, d = derivative value
    s_ff_00 = zero(T); s_ff_10 = zero(T); s_ff_01 = zero(T); s_ff_11 = zero(T)
    s_df_00 = zero(T); s_df_10 = zero(T); s_df_01 = zero(T); s_df_11 = zero(T)
    s_fd_00 = zero(T); s_fd_10 = zero(T); s_fd_01 = zero(T); s_fd_11 = zero(T)
    s_dd_00 = zero(T); s_dd_10 = zero(T); s_dd_01 = zero(T); s_dd_11 = zero(T)

    @inbounds for m in -(itp.eqs-1):itp.eqs
        col_y = m + itp.eqs
        ky_f0 = itp.kernel_pre[idx_y, col_y]
        ky_f1 = itp.kernel_pre[idx_y+1, col_y]
        ky_d0 = itp.kernel_d1_pre[idx_y, col_y]
        ky_d1 = itp.kernel_d1_pre[idx_y+1, col_y]

        @inbounds for l in -(itp.eqs-1):itp.eqs
            coef = itp.coefs[i+l, j+m]
            col_x = l + itp.eqs
            kx_f0 = itp.kernel_pre[idx_x, col_x]
            kx_f1 = itp.kernel_pre[idx_x+1, col_x]
            kx_d0 = itp.kernel_d1_pre[idx_x, col_x]
            kx_d1 = itp.kernel_d1_pre[idx_x+1, col_x]

            # All 16 tensor products
            s_ff_00 += coef * kx_f0 * ky_f0
            s_ff_10 += coef * kx_f1 * ky_f0
            s_ff_01 += coef * kx_f0 * ky_f1
            s_ff_11 += coef * kx_f1 * ky_f1

            s_df_00 += coef * kx_d0 * ky_f0
            s_df_10 += coef * kx_d1 * ky_f0
            s_df_01 += coef * kx_d0 * ky_f1
            s_df_11 += coef * kx_d1 * ky_f1

            s_fd_00 += coef * kx_f0 * ky_d0
            s_fd_10 += coef * kx_f1 * ky_d0
            s_fd_01 += coef * kx_f0 * ky_d1
            s_fd_11 += coef * kx_f1 * ky_d1

            s_dd_00 += coef * kx_d0 * ky_d0
            s_dd_10 += coef * kx_d1 * ky_d0
            s_dd_01 += coef * kx_d0 * ky_d1
            s_dd_11 += coef * kx_d1 * ky_d1
        end
    end

    # Nested cubic Hermite: first interpolate along x for each y-bracket
    h_pre = one(T) / T(n_pre - 1)

    # At y-subgrid idx_y: Hermite along x using (ff, df) at x=0,1
    val_y0  = cubic_hermite(t_x, s_ff_00, s_ff_10, s_df_00, s_df_10, h_pre)
    # At y-subgrid idx_y+1: Hermite along x using (ff, df) at x=0,1
    val_y1  = cubic_hermite(t_x, s_ff_01, s_ff_11, s_df_01, s_df_11, h_pre)
    # d/dy at y-subgrid idx_y: Hermite along x using (fd, dd) at x=0,1
    dval_y0 = cubic_hermite(t_x, s_fd_00, s_fd_10, s_dd_00, s_dd_10, h_pre)
    # d/dy at y-subgrid idx_y+1: Hermite along x using (fd, dd) at x=0,1
    dval_y1 = cubic_hermite(t_x, s_fd_01, s_fd_11, s_dd_01, s_dd_11, h_pre)

    # Then Hermite along y
    result = cubic_hermite(t_y, val_y0, val_y1, dval_y0, dval_y1, h_pre)

    return result * (-one(T)/itp.h[1])^(DO.parameters[1]) * (-one(T)/itp.h[2])^(DO.parameters[1])
end

@inline function (itp::FastConvolutionInterpolation{T,2,TCoefs,IT,Axs,KA,Val{2},
                    HigherOrderKernel{DG},EQ,PR,KP,KBC,DO,FD,SD,Val{:quintic}})(x::Vararg{Number,2}) where 
                    {T,TCoefs,IT,Axs,KA,DG,EQ,PR,KP,KBC,DO,FD,SD}

    # First dimension (x)
    i_float = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T)
    i = clamp(floor(Int, i_float), itp.eqs, length(itp.knots[1]) - itp.eqs)
    x_diff_left = (x[1] - itp.knots[1][i]) / itp.h[1]
    x_diff_right = one(T) - x_diff_left
    n_pre = length(itp.pre_range)
    continuous_idx_x = x_diff_right * T(n_pre - 1) + one(T)
    idx_x = clamp(floor(Int, continuous_idx_x), 1, n_pre - 1)
    t_x = continuous_idx_x - T(idx_x)

    # Second dimension (y)
    j_float = (x[2] - itp.knots[2][1]) / itp.h[2] + one(T)
    j = clamp(floor(Int, j_float), itp.eqs, length(itp.knots[2]) - itp.eqs)
    y_diff_left = (x[2] - itp.knots[2][j]) / itp.h[2]
    y_diff_right = one(T) - y_diff_left
    continuous_idx_y = y_diff_right * T(n_pre - 1) + one(T)
    idx_y = clamp(floor(Int, continuous_idx_y), 1, n_pre - 1)
    t_y = continuous_idx_y - T(idx_y)

    # 36 accumulators: (f, d1, d2) × 2 x-brackets × (f, d1, d2) × 2 y-brackets
    # Naming: s_[x-type][y-type]_[x-idx][y-idx]
    # f = function, d = first derivative, e = second derivative
    s_ff_00 = zero(T); s_ff_10 = zero(T); s_ff_01 = zero(T); s_ff_11 = zero(T)
    s_df_00 = zero(T); s_df_10 = zero(T); s_df_01 = zero(T); s_df_11 = zero(T)
    s_ef_00 = zero(T); s_ef_10 = zero(T); s_ef_01 = zero(T); s_ef_11 = zero(T)
    s_fd_00 = zero(T); s_fd_10 = zero(T); s_fd_01 = zero(T); s_fd_11 = zero(T)
    s_dd_00 = zero(T); s_dd_10 = zero(T); s_dd_01 = zero(T); s_dd_11 = zero(T)
    s_ed_00 = zero(T); s_ed_10 = zero(T); s_ed_01 = zero(T); s_ed_11 = zero(T)
    s_fe_00 = zero(T); s_fe_10 = zero(T); s_fe_01 = zero(T); s_fe_11 = zero(T)
    s_de_00 = zero(T); s_de_10 = zero(T); s_de_01 = zero(T); s_de_11 = zero(T)
    s_ee_00 = zero(T); s_ee_10 = zero(T); s_ee_01 = zero(T); s_ee_11 = zero(T)

    @inbounds for m in -(itp.eqs-1):itp.eqs
        col_y = m + itp.eqs
        ky_f0 = itp.kernel_pre[idx_y, col_y]
        ky_f1 = itp.kernel_pre[idx_y+1, col_y]
        ky_d0 = itp.kernel_d1_pre[idx_y, col_y]
        ky_d1 = itp.kernel_d1_pre[idx_y+1, col_y]
        ky_e0 = itp.kernel_d2_pre[idx_y, col_y]
        ky_e1 = itp.kernel_d2_pre[idx_y+1, col_y]

        @inbounds for l in -(itp.eqs-1):itp.eqs
            coef = itp.coefs[i+l, j+m]
            col_x = l + itp.eqs
            kx_f0 = itp.kernel_pre[idx_x, col_x]
            kx_f1 = itp.kernel_pre[idx_x+1, col_x]
            kx_d0 = itp.kernel_d1_pre[idx_x, col_x]
            kx_d1 = itp.kernel_d1_pre[idx_x+1, col_x]
            kx_e0 = itp.kernel_d2_pre[idx_x, col_x]
            kx_e1 = itp.kernel_d2_pre[idx_x+1, col_x]

            # All 36 tensor products
            s_ff_00 += coef * kx_f0 * ky_f0;  s_ff_10 += coef * kx_f1 * ky_f0
            s_ff_01 += coef * kx_f0 * ky_f1;  s_ff_11 += coef * kx_f1 * ky_f1

            s_df_00 += coef * kx_d0 * ky_f0;  s_df_10 += coef * kx_d1 * ky_f0
            s_df_01 += coef * kx_d0 * ky_f1;  s_df_11 += coef * kx_d1 * ky_f1

            s_ef_00 += coef * kx_e0 * ky_f0;  s_ef_10 += coef * kx_e1 * ky_f0
            s_ef_01 += coef * kx_e0 * ky_f1;  s_ef_11 += coef * kx_e1 * ky_f1

            s_fd_00 += coef * kx_f0 * ky_d0;  s_fd_10 += coef * kx_f1 * ky_d0
            s_fd_01 += coef * kx_f0 * ky_d1;  s_fd_11 += coef * kx_f1 * ky_d1

            s_dd_00 += coef * kx_d0 * ky_d0;  s_dd_10 += coef * kx_d1 * ky_d0
            s_dd_01 += coef * kx_d0 * ky_d1;  s_dd_11 += coef * kx_d1 * ky_d1

            s_ed_00 += coef * kx_e0 * ky_d0;  s_ed_10 += coef * kx_e1 * ky_d0
            s_ed_01 += coef * kx_e0 * ky_d1;  s_ed_11 += coef * kx_e1 * ky_d1

            s_fe_00 += coef * kx_f0 * ky_e0;  s_fe_10 += coef * kx_f1 * ky_e0
            s_fe_01 += coef * kx_f0 * ky_e1;  s_fe_11 += coef * kx_f1 * ky_e1

            s_de_00 += coef * kx_d0 * ky_e0;  s_de_10 += coef * kx_d1 * ky_e0
            s_de_01 += coef * kx_d0 * ky_e1;  s_de_11 += coef * kx_d1 * ky_e1

            s_ee_00 += coef * kx_e0 * ky_e0;  s_ee_10 += coef * kx_e1 * ky_e0
            s_ee_01 += coef * kx_e0 * ky_e1;  s_ee_11 += coef * kx_e1 * ky_e1
        end
    end

    # Nested quintic Hermite: first interpolate along x, then along y
    h_pre = one(T) / T(n_pre - 1)

    # At y-bracket 0: quintic Hermite along x for f, df/dy, d²f/dy²
    val_y0   = quintic_hermite(t_x, s_ff_00, s_ff_10, s_df_00, s_df_10, s_ef_00, s_ef_10, h_pre)
    d1_y0    = quintic_hermite(t_x, s_fd_00, s_fd_10, s_dd_00, s_dd_10, s_ed_00, s_ed_10, h_pre)
    d2_y0    = quintic_hermite(t_x, s_fe_00, s_fe_10, s_de_00, s_de_10, s_ee_00, s_ee_10, h_pre)

    # At y-bracket 1: quintic Hermite along x for f, df/dy, d²f/dy²
    val_y1   = quintic_hermite(t_x, s_ff_01, s_ff_11, s_df_01, s_df_11, s_ef_01, s_ef_11, h_pre)
    d1_y1    = quintic_hermite(t_x, s_fd_01, s_fd_11, s_dd_01, s_dd_11, s_ed_01, s_ed_11, h_pre)
    d2_y1    = quintic_hermite(t_x, s_fe_01, s_fe_11, s_de_01, s_de_11, s_ee_01, s_ee_11, h_pre)

    # Final quintic Hermite along y
    result = quintic_hermite(t_y, val_y0, val_y1, d1_y0, d1_y1, d2_y0, d2_y1, h_pre)

    return result * (-one(T)/itp.h[1])^(DO.parameters[1]) * (-one(T)/itp.h[2])^(DO.parameters[1])
end