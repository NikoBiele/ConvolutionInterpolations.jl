@inline function (itp::FastConvolutionInterpolation{T,2,0,TCoefs,Axs,KA,Val{2},LowerOrderKernel{(:a0, :a0)},
                    EQ,PR,KP,KBC,DerivativeOrder{DO},FD,SD,Val{(:linear,:linear)},Val{true},Val{0}})(x::Vararg{Number,2}) where 
                    {T<:AbstractFloat,TCoefs<:AbstractArray{T,2},Axs<:Tuple{<:AbstractVector,<:AbstractVector},
                    KA<:Tuple{<:Nothing,<:Nothing},EQ<:Tuple{Int,Int},
                    PR<:Tuple{<:AbstractVector,<:AbstractVector},
                    KP,KBC<:Tuple{<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol}},DO,FD,SD}

    # specialized dispatch for 2d nearest neighbor kernel 
    # First dimension (x)
    i_float = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T)
    i = clamp(floor(Int, i_float), itp.eqs[1], length(itp.knots[1]) - itp.eqs[1])
    x_diff_left = (x[1] - itp.knots[1][i]) / itp.h[1]  # Recompute from actual knot

    # Second dimension (y)
    j_float = (x[2] - itp.knots[2][1]) / itp.h[2] + one(T)
    j = clamp(floor(Int, j_float), itp.eqs[2], length(itp.knots[2]) - itp.eqs[2])
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

@inline function (itp::FastConvolutionInterpolation{T,2,0,TCoefs,Axs,KA,Val{2},LowerOrderKernel{(:a1, :a1)},
                    EQ,PR,KP,KBC,DerivativeOrder{DO},FD,SD,Val{(:linear,:linear)},Val{true},Val{0}})(x::Vararg{Number,2}) where 
                    {T<:AbstractFloat,TCoefs<:AbstractArray{T,2},Axs<:Tuple{<:AbstractVector,<:AbstractVector},
                    KA<:Tuple{<:Nothing,<:Nothing},EQ<:Tuple{Int,Int},
                    PR<:Tuple{<:AbstractVector,<:AbstractVector},
                    KP,KBC<:Tuple{<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol}},DO,FD,SD}

    # specialized dispatch for 2d linear kernel
    # First dimension (x)
    i_float = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T)
    i = clamp(floor(Int, i_float), itp.eqs[1], length(itp.knots[1]) - itp.eqs[1])
    x_diff_left = (x[1] - itp.knots[1][i]) / itp.h[1]  # Recompute from actual knot

    # Second dimension (y)
    j_float = (x[2] - itp.knots[2][1]) / itp.h[2] + one(T)
    j = clamp(floor(Int, j_float), itp.eqs[2], length(itp.knots[2]) - itp.eqs[2])
    y_diff_left = (x[2] - itp.knots[2][j]) / itp.h[2]  # Recompute from actual knot

    # Bilinear interpolation
    return @inbounds @fastmath ((1-x_diff_left)*(1-y_diff_left)*itp.coefs[i, j] + 
                        x_diff_left*(1-y_diff_left)*itp.coefs[i+1, j] + 
                        (1-x_diff_left)*y_diff_left*itp.coefs[i, j+1] + 
                        x_diff_left*y_diff_left*itp.coefs[i+1, j+1]) *
                        (-one(T)/itp.h[1])^DO[1] * (-one(T)/itp.h[2])^DO[2]
end

@inline function (itp::FastConvolutionInterpolation{T,2,0,TCoefs,Axs,KA,Val{2},
            DG,EQ,PR,KP,KBC,DerivativeOrder{DO},FD,SD,Val{SG},Val{true},Val{0}})(x::Vararg{Number,2}) where 
            {T<:AbstractFloat,TCoefs<:AbstractArray{T,2},Axs<:Tuple{<:AbstractVector,<:AbstractVector},
            KA<:Tuple{<:Nothing,<:Nothing},DG,EQ<:Tuple{Int,Int},PR<:Tuple{<:AbstractVector,<:AbstractVector},
            KP,KBC<:Tuple{<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol}},DO,FD,SD,SG}

    # first dimension
    i_float = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T)
    i = clamp(floor(Int, i_float), 1, length(itp.knots[1]) - 1)
    is_boundary_x = is_boundary_stencil(i, size(itp.coefs, 1), itp.eqs[1])

    # second dimension
    j_float = (x[2] - itp.knots[2][1]) / itp.h[2] + one(T)
    j = clamp(floor(Int, j_float), 1, length(itp.knots[2]) - 1)
    is_boundary_y = is_boundary_stencil(j, size(itp.coefs, 2), itp.eqs[2])

    if (is_boundary_x || is_boundary_y) && itp.boundary_fallback
        
        x_diff_left = (x[1] - itp.knots[1][i]) / itp.h[1]  # Recompute from actual knot
        y_diff_left = (x[2] - itp.knots[2][j]) / itp.h[2]  # Recompute from actual knot

        # Bilinear interpolation
        return @inbounds @fastmath ((1-x_diff_left)*(1-y_diff_left)*itp.coefs[i, j] + 
                        x_diff_left*(1-y_diff_left)*itp.coefs[i+1, j] + 
                        (1-x_diff_left)*y_diff_left*itp.coefs[i, j+1] + 
                        x_diff_left*y_diff_left*itp.coefs[i+1, j+1]) *
                        (-one(T)/itp.h[1])^DO[1] * (-one(T)/itp.h[2])^DO[2]

    else

        ng1 = itp.eqs[1] - 1
        ng2 = itp.eqs[2] - 1
        n1 = itp.domain_size[1]
        n2 = itp.domain_size[2]
        kernel_type = _kernel_sym(itp.kernel_sym)

        if is_boundary_x || is_boundary_y
            ghost_matrix_xl = (is_boundary_x && i - ng1 < 1)          ? get_polynomial_ghost_coeffs(itp.bc[1][1], kernel_type[1]) : nothing
            ghost_matrix_xr = (is_boundary_x && i + itp.eqs[1] > n1)  ? get_polynomial_ghost_coeffs(itp.bc[1][2], kernel_type[1]) : nothing
            ns_xl = ghost_matrix_xl !== nothing ? size(ghost_matrix_xl, 2) : 0
            ns_xr = ghost_matrix_xr !== nothing ? size(ghost_matrix_xr, 2) : 0

            # Fill x-ghosts for each m in stencil range
            if is_boundary_x
                if i - ng1 < 1
                    for m in -(ng2):itp.eqs[2]
                        abs_m = j + m
                        if abs_m >= 1 && abs_m <= n2
                            y_slice = view(itp.coefs, 1:ns_xl, abs_m)
                            mul!(view(itp.lazy_workspace.ghost_buf, 1:ng1), view(ghost_matrix_xl, 1:ng1, :), y_slice)
                            for l in -(ng1):-i
                                itp.lazy_workspace.stencil_buf[l + itp.eqs[1], m + itp.eqs[2]] = itp.lazy_workspace.ghost_buf[1 - (i + l)]
                            end
                        end
                    end
                end
                if i + itp.eqs[1] > n1
                    for m in -(ng2):itp.eqs[2]
                        abs_m = j + m
                        if abs_m >= 1 && abs_m <= n2
                            for k in 1:ng1
                                acc = zero(T)
                                for p in 1:ns_xr
                                    acc += ghost_matrix_xr[k, p] * itp.coefs[n1 - p + 1, abs_m]
                                end
                                itp.lazy_workspace.ghost_buf[k] = acc
                            end
                            for l in (n1-i+1):itp.eqs[1]
                                itp.lazy_workspace.stencil_buf[l + itp.eqs[1], m + itp.eqs[2]] = itp.lazy_workspace.ghost_buf[i + l - n1]
                            end
                        end
                    end
                end
            end

            # Fill y-ghosts
            if is_boundary_y
                if j - ng2 < 1
                    ghost_matrix_yl = get_polynomial_ghost_coeffs(itp.bc[2][1], kernel_type[2])
                    ns_yl = size(ghost_matrix_yl, 2)
                    for l in -(ng1):itp.eqs[1]
                        abs_l = i + l
                        for k in 1:ng2
                            acc = zero(T)
                            for p in 1:ns_yl
                                val_p = if abs_l >= 1 && abs_l <= n1
                                    itp.coefs[abs_l, p]
                                elseif abs_l < 1
                                    inner = zero(T)
                                    for q in 1:ns_xl
                                        inner += ghost_matrix_xl[1 - abs_l, q] * itp.coefs[q, p]
                                    end
                                    inner
                                else
                                    inner = zero(T)
                                    for q in 1:ns_xr
                                        inner += ghost_matrix_xr[abs_l - n1, q] * itp.coefs[n1 - q + 1, p]
                                    end
                                    inner
                                end
                                acc += ghost_matrix_yl[k, p] * val_p
                            end
                            itp.lazy_workspace.ghost_buf[k] = acc
                        end
                        for m in -(ng2):-j
                            itp.lazy_workspace.stencil_buf[l + itp.eqs[1], m + itp.eqs[2]] = itp.lazy_workspace.ghost_buf[1 - (j + m)]
                        end
                    end
                end
                if j + itp.eqs[2] > n2
                    ghost_matrix_yr = get_polynomial_ghost_coeffs(itp.bc[2][2], kernel_type[2])
                    ns_yr = size(ghost_matrix_yr, 2)
                    for l in -(ng1):itp.eqs[1]
                        abs_l = i + l
                        for k in 1:ng2
                            acc = zero(T)
                            for p in 1:ns_yr
                                val_p = if abs_l >= 1 && abs_l <= n1
                                    itp.coefs[abs_l, n2 - p + 1]
                                elseif abs_l < 1
                                    inner = zero(T)
                                    for q in 1:ns_xl
                                        inner += ghost_matrix_xl[1 - abs_l, q] * itp.coefs[q, n2 - p + 1]
                                    end
                                    inner
                                else
                                    inner = zero(T)
                                    for q in 1:ns_xr
                                        inner += ghost_matrix_xr[abs_l - n1, q] * itp.coefs[n1 - q + 1, n2 - p + 1]
                                    end
                                    inner
                                end
                                acc += ghost_matrix_yr[k, p] * val_p
                            end
                            itp.lazy_workspace.ghost_buf[k] = acc
                        end
                        for m in (n2-j+1):itp.eqs[2]
                            itp.lazy_workspace.stencil_buf[l + itp.eqs[1], m + itp.eqs[2]] = itp.lazy_workspace.ghost_buf[j + m - n2]
                        end
                    end
                end
            end
        end

        if SG == (:linear, :linear)

            # specialized dispatch for 2d higher-order kernels
            # First dimension (x)
            x_diff_left = (x[1] - itp.knots[1][i]) / itp.h[1]
            x_diff_right = one(T) - x_diff_left
            idx_x = clamp(floor(Int, x_diff_right * (length(itp.pre_range[1]) - one(Int64))) + one(Int64), one(Int64), length(itp.pre_range[1]) - one(Int64))
            t_x = (x_diff_right - itp.pre_range[1][idx_x]) / (itp.pre_range[1][idx_x+1] - itp.pre_range[1][idx_x])

            # Second dimension (y)
            y_diff_left = (x[2] - itp.knots[2][j]) / itp.h[2]
            y_diff_right = one(T) - y_diff_left 
            idx_y = clamp(floor(Int, y_diff_right * (length(itp.pre_range[2]) - one(T))) + one(Int64), one(Int64), length(itp.pre_range[2]) - one(Int64))
            t_y = (y_diff_right - itp.pre_range[2][idx_y]) / (itp.pre_range[2][idx_y+1] - itp.pre_range[2][idx_y])
            
            # Initialize results
            result_00 = zero(T)
            result_10 = zero(T)
            result_01 = zero(T)
            result_11 = zero(T)

            # Single pass through the convolution support
            kernel_type = _kernel_sym(itp.kernel_sym)
            ng = itp.eqs[1] - 1 # lazy only for same kernel in all directions
            @inbounds for m in -(itp.eqs[2]-1):itp.eqs[2]
                ky_0 = itp.kernel_pre[2][idx_y, m+itp.eqs[2]]
                ky_1 = itp.kernel_pre[2][idx_y+1, m+itp.eqs[2]]
                
                @inbounds for l in -(itp.eqs[1]-1):itp.eqs[1]
                    abs_l = i + l
                    abs_m = j + m
                    coef = if abs_l >= 1 && abs_l <= n1 && abs_m >= 1 && abs_m <= n2
                        itp.coefs[abs_l, abs_m]
                    else
                        itp.lazy_workspace.stencil_buf[l + itp.eqs[1], m + itp.eqs[2]]
                    end
                    kx_0 = itp.kernel_pre[1][idx_x, l+itp.eqs[1]]
                    kx_1 = itp.kernel_pre[1][idx_x+1, l+itp.eqs[1]]
                    
                    result_00 += coef * kx_0 * ky_0
                    result_10 += coef * kx_1 * ky_0
                    result_01 += coef * kx_0 * ky_1
                    result_11 += coef * kx_1 * ky_1
                end
            end

            # Bilinear interpolation
            return @fastmath ((one(T)-t_x)*(one(T)-t_y)*result_00 + t_x*(one(T)-t_y)*result_10 + (one(T)-t_x)*t_y*result_01 + t_x*t_y*result_11) *
                                (-one(T)/itp.h[1])^DO[1] * (-one(T)/itp.h[2])^DO[2]

        elseif SG == (:cubic, :cubic)

            # First dimension (x)
            x_diff_left = (x[1] - itp.knots[1][i]) / itp.h[1]
            x_diff_right = one(T) - x_diff_left
            n_pre_x = length(itp.pre_range[1])
            continuous_idx_x = x_diff_right * T(n_pre_x - 1) + one(T)
            idx_x = clamp(floor(Int, continuous_idx_x), 1, n_pre_x - 1)
            t_x = continuous_idx_x - T(idx_x)

            # Second dimension (y)
            y_diff_left = (x[2] - itp.knots[2][j]) / itp.h[2]
            y_diff_right = one(T) - y_diff_left
            n_pre_y = length(itp.pre_range[2])
            continuous_idx_y = y_diff_right * T(n_pre_y - 1) + one(T)
            idx_y = clamp(floor(Int, continuous_idx_y), 1, n_pre_y - 1)
            t_y = continuous_idx_y - T(idx_y)

            # 16 accumulators: f and df for 2 x-positions × 2 y-positions
            s_ff_00 = zero(T); s_ff_10 = zero(T); s_ff_01 = zero(T); s_ff_11 = zero(T)
            s_df_00 = zero(T); s_df_10 = zero(T); s_df_01 = zero(T); s_df_11 = zero(T)
            s_fd_00 = zero(T); s_fd_10 = zero(T); s_fd_01 = zero(T); s_fd_11 = zero(T)
            s_dd_00 = zero(T); s_dd_10 = zero(T); s_dd_01 = zero(T); s_dd_11 = zero(T)

            kernel_type = _kernel_sym(itp.kernel_sym)
            ng = itp.eqs[1] - 1 # lazy only for same kernel in all directions
            @inbounds for m in -(itp.eqs[2]-1):itp.eqs[2]
                col_y = m + itp.eqs[2]
                ky_f0 = itp.kernel_pre[2][idx_y, col_y]
                ky_f1 = itp.kernel_pre[2][idx_y+1, col_y]
                ky_d0 = itp.kernel_d1_pre[2][idx_y, col_y]
                ky_d1 = itp.kernel_d1_pre[2][idx_y+1, col_y]

                @inbounds for l in -(itp.eqs[1]-1):itp.eqs[1]
                    abs_l = i + l
                    abs_m = j + m
                    coef = if abs_l >= 1 && abs_l <= n1 && abs_m >= 1 && abs_m <= n2
                        itp.coefs[abs_l, abs_m]
                    else
                        itp.lazy_workspace.stencil_buf[l + itp.eqs[1], m + itp.eqs[2]]
                    end
                    col_x = l + itp.eqs[1]
                    kx_f0 = itp.kernel_pre[1][idx_x, col_x]
                    kx_f1 = itp.kernel_pre[1][idx_x+1, col_x]
                    kx_d0 = itp.kernel_d1_pre[1][idx_x, col_x]
                    kx_d1 = itp.kernel_d1_pre[1][idx_x+1, col_x]

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
            h_pre_x = one(T) / T(n_pre_x - 1)
            h_pre_y = one(T) / T(n_pre_y - 1)

            val_y0  = cubic_hermite(t_x, s_ff_00, s_ff_10, s_df_00, s_df_10, h_pre_x)
            val_y1  = cubic_hermite(t_x, s_ff_01, s_ff_11, s_df_01, s_df_11, h_pre_x)
            dval_y0 = cubic_hermite(t_x, s_fd_00, s_fd_10, s_dd_00, s_dd_10, h_pre_x)
            dval_y1 = cubic_hermite(t_x, s_fd_01, s_fd_11, s_dd_01, s_dd_11, h_pre_x)

            result = cubic_hermite(t_y, val_y0, val_y1, dval_y0, dval_y1, h_pre_y)

            return result * (-one(T)/itp.h[1])^DO[1] * (-one(T)/itp.h[2])^DO[2]

        elseif DG == (:quintic, :quintic)

            # First dimension (x)
            x_diff_left = (x[1] - itp.knots[1][i]) / itp.h[1]
            x_diff_right = one(T) - x_diff_left
            n_pre_x = length(itp.pre_range[1])
            continuous_idx_x = x_diff_right * T(n_pre_x - 1) + one(T)
            idx_x = clamp(floor(Int, continuous_idx_x), 1, n_pre_x - 1)
            t_x = continuous_idx_x - T(idx_x)

            # Second dimension (y)
            y_diff_left = (x[2] - itp.knots[2][j]) / itp.h[2]
            y_diff_right = one(T) - y_diff_left
            n_pre_y = length(itp.pre_range[2])
            continuous_idx_y = y_diff_right * T(n_pre_y - 1) + one(T)
            idx_y = clamp(floor(Int, continuous_idx_y), 1, n_pre_y - 1)
            t_y = continuous_idx_y - T(idx_y)

            # 36 accumulators
            s_ff_00 = zero(T); s_ff_10 = zero(T); s_ff_01 = zero(T); s_ff_11 = zero(T)
            s_df_00 = zero(T); s_df_10 = zero(T); s_df_01 = zero(T); s_df_11 = zero(T)
            s_ef_00 = zero(T); s_ef_10 = zero(T); s_ef_01 = zero(T); s_ef_11 = zero(T)
            s_fd_00 = zero(T); s_fd_10 = zero(T); s_fd_01 = zero(T); s_fd_11 = zero(T)
            s_dd_00 = zero(T); s_dd_10 = zero(T); s_dd_01 = zero(T); s_dd_11 = zero(T)
            s_ed_00 = zero(T); s_ed_10 = zero(T); s_ed_01 = zero(T); s_ed_11 = zero(T)
            s_fe_00 = zero(T); s_fe_10 = zero(T); s_fe_01 = zero(T); s_fe_11 = zero(T)
            s_de_00 = zero(T); s_de_10 = zero(T); s_de_01 = zero(T); s_de_11 = zero(T)
            s_ee_00 = zero(T); s_ee_10 = zero(T); s_ee_01 = zero(T); s_ee_11 = zero(T)

            kernel_type = _kernel_sym(itp.kernel_sym)
            ng = itp.eqs[1] - 1 # lazy only for same kernel in all directions
            @inbounds for m in -(itp.eqs[2]-1):itp.eqs[2]
                col_y = m + itp.eqs[2]
                ky_f0 = itp.kernel_pre[2][idx_y, col_y]
                ky_f1 = itp.kernel_pre[2][idx_y+1, col_y]
                ky_d0 = itp.kernel_d1_pre[2][idx_y, col_y]
                ky_d1 = itp.kernel_d1_pre[2][idx_y+1, col_y]
                ky_e0 = itp.kernel_d2_pre[2][idx_y, col_y]
                ky_e1 = itp.kernel_d2_pre[2][idx_y+1, col_y]

                @inbounds for l in -(itp.eqs[1]-1):itp.eqs[1]
                    abs_l = i + l
                    abs_m = j + m
                    coef = if abs_l >= 1 && abs_l <= n1 && abs_m >= 1 && abs_m <= n2
                        itp.coefs[abs_l, abs_m]
                    else
                        itp.lazy_workspace.stencil_buf[l + itp.eqs[1], m + itp.eqs[2]]
                    end
                    col_x = l + itp.eqs[1]
                    kx_f0 = itp.kernel_pre[1][idx_x, col_x]
                    kx_f1 = itp.kernel_pre[1][idx_x+1, col_x]
                    kx_d0 = itp.kernel_d1_pre[1][idx_x, col_x]
                    kx_d1 = itp.kernel_d1_pre[1][idx_x+1, col_x]
                    kx_e0 = itp.kernel_d2_pre[1][idx_x, col_x]
                    kx_e1 = itp.kernel_d2_pre[1][idx_x+1, col_x]

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
            h_pre_x = one(T) / T(n_pre_x - 1)
            h_pre_y = one(T) / T(n_pre_y - 1)

            val_y0   = quintic_hermite(t_x, s_ff_00, s_ff_10, s_df_00, s_df_10, s_ef_00, s_ef_10, h_pre_x)
            d1_y0    = quintic_hermite(t_x, s_fd_00, s_fd_10, s_dd_00, s_dd_10, s_ed_00, s_ed_10, h_pre_x)
            d2_y0    = quintic_hermite(t_x, s_fe_00, s_fe_10, s_de_00, s_de_10, s_ee_00, s_ee_10, h_pre_x)

            val_y1   = quintic_hermite(t_x, s_ff_01, s_ff_11, s_df_01, s_df_11, s_ef_01, s_ef_11, h_pre_x)
            d1_y1    = quintic_hermite(t_x, s_fd_01, s_fd_11, s_dd_01, s_dd_11, s_ed_01, s_ed_11, h_pre_x)
            d2_y1    = quintic_hermite(t_x, s_fe_01, s_fe_11, s_de_01, s_de_11, s_ee_01, s_ee_11, h_pre_x)

            result = quintic_hermite(t_y, val_y0, val_y1, d1_y0, d1_y1, d2_y0, d2_y1, h_pre_y)

            return result * (-one(T)/itp.h[1])^DO[1] * (-one(T)/itp.h[2])^DO[2]
        end
    end
end