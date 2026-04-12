# ==============================================================
# 3D — cubic or linear per dim
# ==============================================================
function (itp::FastConvolutionInterpolation{T,3,0,TCoefs,Axs,KA,Val{3},DG,EQ,PR,KP,KBC,
            DerivativeOrder{DO},FD,SD,Val{SG},Val{false},Val{0}})(x::Vararg{Number,3}) where
            {T<:AbstractFloat,TCoefs<:AbstractArray{T,3},
            Axs<:Tuple{<:AbstractVector,<:AbstractVector,<:AbstractVector},
            KA<:Tuple{<:Nothing,<:Nothing,<:Nothing},DG<:AbstractMixedConvolutionKernel,EQ<:Tuple{Int,Int,Int},
            PR<:Tuple{<:AbstractVector,<:AbstractVector,<:AbstractVector},KP,
            KBC<:Tuple{<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol}},DO,FD,SD,SG}
            
    if SG == (:cubic,:cubic,:cubic)

        # Grid positions
        i_float = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T)
        i = clamp(floor(Int, i_float), itp.eqs[1], length(itp.knots[1]) - itp.eqs[1])
        x_diff_right = one(T) - (x[1] - itp.knots[1][i]) / itp.h[1]
        n_pre_x = length(itp.pre_range[1])
        cidx_x = x_diff_right * T(n_pre_x - 1) + one(T)
        idx_x = clamp(floor(Int, cidx_x), 1, n_pre_x - 1)
        t_x = cidx_x - T(idx_x)

        j_float = (x[2] - itp.knots[2][1]) / itp.h[2] + one(T)
        j = clamp(floor(Int, j_float), itp.eqs[2], length(itp.knots[2]) - itp.eqs[2])
        y_diff_right = one(T) - (x[2] - itp.knots[2][j]) / itp.h[2]
        n_pre_y = length(itp.pre_range[2])
        cidx_y = y_diff_right * T(n_pre_y - 1) + one(T)
        idx_y = clamp(floor(Int, cidx_y), 1, n_pre_y - 1)
        t_y = cidx_y - T(idx_y)

        k_float = (x[3] - itp.knots[3][1]) / itp.h[3] + one(T)
        k = clamp(floor(Int, k_float), itp.eqs[3], length(itp.knots[3]) - itp.eqs[3])
        z_diff_right = one(T) - (x[3] - itp.knots[3][k]) / itp.h[3]
        n_pre_z = length(itp.pre_range[3])
        cidx_z = z_diff_right * T(n_pre_z - 1) + one(T)
        idx_z = clamp(floor(Int, cidx_z), 1, n_pre_z - 1)
        t_z = cidx_z - T(idx_z)

        h_pre_x = one(T) / T(n_pre_x - 1)
        h_pre_y = one(T) / T(n_pre_y - 1)
        h_pre_z = one(T) / T(n_pre_z - 1)

        # 64 accumulators: [kx type][ky type][kz type][x bracket][y bracket][z bracket]
        # naming: s_{x-type}{y-type}{z-type}_{x-bracket}{y-bracket}{z-bracket}
        # types: f=function, d=derivative; brackets: 0=lower, 1=upper
        s_fff_000=zero(T); s_fff_100=zero(T); s_fff_010=zero(T); s_fff_110=zero(T)
        s_fff_001=zero(T); s_fff_101=zero(T); s_fff_011=zero(T); s_fff_111=zero(T)
        s_dff_000=zero(T); s_dff_100=zero(T); s_dff_010=zero(T); s_dff_110=zero(T)
        s_dff_001=zero(T); s_dff_101=zero(T); s_dff_011=zero(T); s_dff_111=zero(T)
        s_fdf_000=zero(T); s_fdf_100=zero(T); s_fdf_010=zero(T); s_fdf_110=zero(T)
        s_fdf_001=zero(T); s_fdf_101=zero(T); s_fdf_011=zero(T); s_fdf_111=zero(T)
        s_ddf_000=zero(T); s_ddf_100=zero(T); s_ddf_010=zero(T); s_ddf_110=zero(T)
        s_ddf_001=zero(T); s_ddf_101=zero(T); s_ddf_011=zero(T); s_ddf_111=zero(T)
        s_ffd_000=zero(T); s_ffd_100=zero(T); s_ffd_010=zero(T); s_ffd_110=zero(T)
        s_ffd_001=zero(T); s_ffd_101=zero(T); s_ffd_011=zero(T); s_ffd_111=zero(T)
        s_dfd_000=zero(T); s_dfd_100=zero(T); s_dfd_010=zero(T); s_dfd_110=zero(T)
        s_dfd_001=zero(T); s_dfd_101=zero(T); s_dfd_011=zero(T); s_dfd_111=zero(T)
        s_fdd_000=zero(T); s_fdd_100=zero(T); s_fdd_010=zero(T); s_fdd_110=zero(T)
        s_fdd_001=zero(T); s_fdd_101=zero(T); s_fdd_011=zero(T); s_fdd_111=zero(T)
        s_ddd_000=zero(T); s_ddd_100=zero(T); s_ddd_010=zero(T); s_ddd_110=zero(T)
        s_ddd_001=zero(T); s_ddd_101=zero(T); s_ddd_011=zero(T); s_ddd_111=zero(T)

        @inbounds for n in -(itp.eqs[3]-1):itp.eqs[3]
            col_z = n + itp.eqs[3]
            kz_f0 = itp.kernel_pre[3][idx_z, col_z]
            kz_f1 = itp.kernel_pre[3][idx_z+1, col_z]
            kz_d0 = itp.kernel_d1_pre[3][idx_z, col_z]
            kz_d1 = itp.kernel_d1_pre[3][idx_z+1, col_z]

            @inbounds for m in -(itp.eqs[2]-1):itp.eqs[2]
                col_y = m + itp.eqs[2]
                ky_f0 = itp.kernel_pre[2][idx_y, col_y]
                ky_f1 = itp.kernel_pre[2][idx_y+1, col_y]
                ky_d0 = itp.kernel_d1_pre[2][idx_y, col_y]
                ky_d1 = itp.kernel_d1_pre[2][idx_y+1, col_y]

                @inbounds for l in -(itp.eqs[1]-1):itp.eqs[1]
                    coef = itp.coefs[i+l, j+m, k+n]
                    col_x = l + itp.eqs[1]
                    kx_f0 = itp.kernel_pre[1][idx_x, col_x]
                    kx_f1 = itp.kernel_pre[1][idx_x+1, col_x]
                    kx_d0 = itp.kernel_d1_pre[1][idx_x, col_x]
                    kx_d1 = itp.kernel_d1_pre[1][idx_x+1, col_x]

                    # fff
                    s_fff_000 += coef*kx_f0*ky_f0*kz_f0; s_fff_100 += coef*kx_f1*ky_f0*kz_f0
                    s_fff_010 += coef*kx_f0*ky_f1*kz_f0; s_fff_110 += coef*kx_f1*ky_f1*kz_f0
                    s_fff_001 += coef*kx_f0*ky_f0*kz_f1; s_fff_101 += coef*kx_f1*ky_f0*kz_f1
                    s_fff_011 += coef*kx_f0*ky_f1*kz_f1; s_fff_111 += coef*kx_f1*ky_f1*kz_f1
                    # dff
                    s_dff_000 += coef*kx_d0*ky_f0*kz_f0; s_dff_100 += coef*kx_d1*ky_f0*kz_f0
                    s_dff_010 += coef*kx_d0*ky_f1*kz_f0; s_dff_110 += coef*kx_d1*ky_f1*kz_f0
                    s_dff_001 += coef*kx_d0*ky_f0*kz_f1; s_dff_101 += coef*kx_d1*ky_f0*kz_f1
                    s_dff_011 += coef*kx_d0*ky_f1*kz_f1; s_dff_111 += coef*kx_d1*ky_f1*kz_f1
                    # fdf
                    s_fdf_000 += coef*kx_f0*ky_d0*kz_f0; s_fdf_100 += coef*kx_f1*ky_d0*kz_f0
                    s_fdf_010 += coef*kx_f0*ky_d1*kz_f0; s_fdf_110 += coef*kx_f1*ky_d1*kz_f0
                    s_fdf_001 += coef*kx_f0*ky_d0*kz_f1; s_fdf_101 += coef*kx_f1*ky_d0*kz_f1
                    s_fdf_011 += coef*kx_f0*ky_d1*kz_f1; s_fdf_111 += coef*kx_f1*ky_d1*kz_f1
                    # ddf
                    s_ddf_000 += coef*kx_d0*ky_d0*kz_f0; s_ddf_100 += coef*kx_d1*ky_d0*kz_f0
                    s_ddf_010 += coef*kx_d0*ky_d1*kz_f0; s_ddf_110 += coef*kx_d1*ky_d1*kz_f0
                    s_ddf_001 += coef*kx_d0*ky_d0*kz_f1; s_ddf_101 += coef*kx_d1*ky_d0*kz_f1
                    s_ddf_011 += coef*kx_d0*ky_d1*kz_f1; s_ddf_111 += coef*kx_d1*ky_d1*kz_f1
                    # ffd
                    s_ffd_000 += coef*kx_f0*ky_f0*kz_d0; s_ffd_100 += coef*kx_f1*ky_f0*kz_d0
                    s_ffd_010 += coef*kx_f0*ky_f1*kz_d0; s_ffd_110 += coef*kx_f1*ky_f1*kz_d0
                    s_ffd_001 += coef*kx_f0*ky_f0*kz_d1; s_ffd_101 += coef*kx_f1*ky_f0*kz_d1
                    s_ffd_011 += coef*kx_f0*ky_f1*kz_d1; s_ffd_111 += coef*kx_f1*ky_f1*kz_d1
                    # dfd
                    s_dfd_000 += coef*kx_d0*ky_f0*kz_d0; s_dfd_100 += coef*kx_d1*ky_f0*kz_d0
                    s_dfd_010 += coef*kx_d0*ky_f1*kz_d0; s_dfd_110 += coef*kx_d1*ky_f1*kz_d0
                    s_dfd_001 += coef*kx_d0*ky_f0*kz_d1; s_dfd_101 += coef*kx_d1*ky_f0*kz_d1
                    s_dfd_011 += coef*kx_d0*ky_f1*kz_d1; s_dfd_111 += coef*kx_d1*ky_f1*kz_d1
                    # fdd
                    s_fdd_000 += coef*kx_f0*ky_d0*kz_d0; s_fdd_100 += coef*kx_f1*ky_d0*kz_d0
                    s_fdd_010 += coef*kx_f0*ky_d1*kz_d0; s_fdd_110 += coef*kx_f1*ky_d1*kz_d0
                    s_fdd_001 += coef*kx_f0*ky_d0*kz_d1; s_fdd_101 += coef*kx_f1*ky_d0*kz_d1
                    s_fdd_011 += coef*kx_f0*ky_d1*kz_d1; s_fdd_111 += coef*kx_f1*ky_d1*kz_d1
                    # ddd
                    s_ddd_000 += coef*kx_d0*ky_d0*kz_d0; s_ddd_100 += coef*kx_d1*ky_d0*kz_d0
                    s_ddd_010 += coef*kx_d0*ky_d1*kz_d0; s_ddd_110 += coef*kx_d1*ky_d1*kz_d0
                    s_ddd_001 += coef*kx_d0*ky_d0*kz_d1; s_ddd_101 += coef*kx_d1*ky_d0*kz_d1
                    s_ddd_011 += coef*kx_d0*ky_d1*kz_d1; s_ddd_111 += coef*kx_d1*ky_d1*kz_d1
                end
            end
        end

        # Reduce along x: 8 cubic Hermite calls (one per {y-type}{z-type}_{y-bracket}{z-bracket})
        v_ff_00 = cubic_hermite(t_x, s_fff_000, s_fff_100, s_dff_000, s_dff_100, h_pre_x)
        v_ff_10 = cubic_hermite(t_x, s_fff_010, s_fff_110, s_dff_010, s_dff_110, h_pre_x)
        v_ff_01 = cubic_hermite(t_x, s_fff_001, s_fff_101, s_dff_001, s_dff_101, h_pre_x)
        v_ff_11 = cubic_hermite(t_x, s_fff_011, s_fff_111, s_dff_011, s_dff_111, h_pre_x)
        v_df_00 = cubic_hermite(t_x, s_fdf_000, s_fdf_100, s_ddf_000, s_ddf_100, h_pre_x)
        v_df_10 = cubic_hermite(t_x, s_fdf_010, s_fdf_110, s_ddf_010, s_ddf_110, h_pre_x)
        v_df_01 = cubic_hermite(t_x, s_fdf_001, s_fdf_101, s_ddf_001, s_ddf_101, h_pre_x)
        v_df_11 = cubic_hermite(t_x, s_fdf_011, s_fdf_111, s_ddf_011, s_ddf_111, h_pre_x)

        # Reduce along y: 4 cubic Hermite calls (one per {z-type}_{z-bracket})
        w_f_0 = cubic_hermite(t_y, v_ff_00, v_ff_10, v_df_00, v_df_10, h_pre_y)
        w_f_1 = cubic_hermite(t_y, v_ff_01, v_ff_11, v_df_01, v_df_11, h_pre_y)

        # Now we need the z-derivative versions for the final z-Hermite
        v_fd_00 = cubic_hermite(t_x, s_ffd_000, s_ffd_100, s_dfd_000, s_dfd_100, h_pre_x)
        v_fd_10 = cubic_hermite(t_x, s_ffd_010, s_ffd_110, s_dfd_010, s_dfd_110, h_pre_x)
        v_fd_01 = cubic_hermite(t_x, s_ffd_001, s_ffd_101, s_dfd_001, s_dfd_101, h_pre_x)
        v_fd_11 = cubic_hermite(t_x, s_ffd_011, s_ffd_111, s_dfd_011, s_dfd_111, h_pre_x)
        v_dd_00 = cubic_hermite(t_x, s_fdd_000, s_fdd_100, s_ddd_000, s_ddd_100, h_pre_x)
        v_dd_10 = cubic_hermite(t_x, s_fdd_010, s_fdd_110, s_ddd_010, s_ddd_110, h_pre_x)
        v_dd_01 = cubic_hermite(t_x, s_fdd_001, s_fdd_101, s_ddd_001, s_ddd_101, h_pre_x)
        v_dd_11 = cubic_hermite(t_x, s_fdd_011, s_fdd_111, s_ddd_011, s_ddd_111, h_pre_x)

        w_d_0 = cubic_hermite(t_y, v_fd_00, v_fd_10, v_dd_00, v_dd_10, h_pre_y)
        w_d_1 = cubic_hermite(t_y, v_fd_01, v_fd_11, v_dd_01, v_dd_11, h_pre_y)

        # Reduce along z: 1 cubic Hermite call
        result = cubic_hermite(t_z, w_f_0, w_f_1, w_d_0, w_d_1, h_pre_z)

        return result * (-one(T)/itp.h[1])^DO[1] * (-one(T)/itp.h[2])^DO[2] * (-one(T)/itp.h[3])^DO[3]

    else # fallback to all linear

        eqs_x,eqs_y,eqs_z = itp.eqs
        kp_x=itp.kernel_pre[1]; kp_y=itp.kernel_pre[2]; kp_z=itp.kernel_pre[3]
        i_float = (x[1]-itp.knots[1][1])/itp.h[1] + one(T)
        i = clamp(floor(Int,i_float), eqs_x, length(itp.knots[1])-eqs_x)
        idx_x,t_x,_ = _perdim_cidx(one(T)-(x[1]-itp.knots[1][i])/itp.h[1], itp.pre_range[1])
        j_float = (x[2]-itp.knots[2][1])/itp.h[2] + one(T)
        j = clamp(floor(Int,j_float), eqs_y, length(itp.knots[2])-eqs_y)
        idx_y,t_y,_ = _perdim_cidx(one(T)-(x[2]-itp.knots[2][j])/itp.h[2], itp.pre_range[2])
        k_float = (x[3]-itp.knots[3][1])/itp.h[3] + one(T)
        k = clamp(floor(Int,k_float), eqs_z, length(itp.knots[3])-eqs_z)
        idx_z,t_z,_ = _perdim_cidx(one(T)-(x[3]-itp.knots[3][k])/itp.h[3], itp.pre_range[3])
        r000=zero(T); r100=zero(T); r010=zero(T); r110=zero(T)
        r001=zero(T); r101=zero(T); r011=zero(T); r111=zero(T)
        @inbounds for p in -(eqs_z-1):eqs_z
            kz_0=kp_z[idx_z,p+eqs_z]; kz_1=kp_z[idx_z+1,p+eqs_z]
            @inbounds for m in -(eqs_y-1):eqs_y
                ky_0=kp_y[idx_y,m+eqs_y]; ky_1=kp_y[idx_y+1,m+eqs_y]
                @inbounds for l in -(eqs_x-1):eqs_x
                    c=itp.coefs[i+l,j+m,k+p]
                    kx_0=kp_x[idx_x,l+eqs_x]; kx_1=kp_x[idx_x+1,l+eqs_x]
                    r000+=c*kx_0*ky_0*kz_0; r100+=c*kx_1*ky_0*kz_0
                    r010+=c*kx_0*ky_1*kz_0; r110+=c*kx_1*ky_1*kz_0
                    r001+=c*kx_0*ky_0*kz_1; r101+=c*kx_1*ky_0*kz_1
                    r011+=c*kx_0*ky_1*kz_1; r111+=c*kx_1*ky_1*kz_1
                end
            end
        end
        ox=one(T)-t_x; oy=one(T)-t_y; oz=one(T)-t_z
        result = @fastmath (ox*oy*oz*r000 + t_x*oy*oz*r100 + ox*t_y*oz*r010 + t_x*t_y*oz*r110 +
                            ox*oy*t_z*r001 + t_x*oy*t_z*r101 + ox*t_y*t_z*r011 + t_x*t_y*t_z*r111)
        return result * _perdim_scale(itp.h, Val(DO))
    end
end