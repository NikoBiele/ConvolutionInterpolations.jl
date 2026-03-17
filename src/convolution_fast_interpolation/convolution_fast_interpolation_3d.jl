"""
    (itp::FastConvolutionInterpolation{T,3,...})(x::Number, y::Number, z::Number)

Evaluate 3D fast convolution interpolation at coordinates `(x, y, z)`.

Dispatches on kernel type:

**Specialized kernels** (no precomputed table):
- `:a0` — Nearest neighbor, ~13ns
- `:a1` — Trilinear interpolation, ~15ns

**Higher-order kernels**: Uses linear subgrid interpolation between eight convolution
results (tensor product of three 1D linear brackets). The 3D kernel is formed as
`K_3D(x,y,z) = K_1D(x) * K_1D(y) * K_1D(z)`, with all eight corner results
accumulated simultaneously in a single triple loop.

O(1) evaluation time, allocation-free. Scales with `(2*eqs)³` kernel support, not
grid size. Cubic and quintic subgrid modes are not available in 3D due to the
exponential growth of tensor product counts; use higher `precompute` values instead.

Results scaled by `(-1/h_x)^derivative * (-1/h_y)^derivative * (-1/h_z)^derivative`.

See also: `FastConvolutionInterpolation`.
"""

@inline function (itp::FastConvolutionInterpolation{T,3,TCoefs,IT,Axs,KA,Val{3},Val{:a0},EQ,PR,KP})(x::Vararg{Number,3}) where {T,TCoefs,IT,Axs,KA,EQ,PR,KP}
    # specialized dispatch for 3d nearest neighbor kernel
        # First dimension (x)
    i_float = (x[1] - itp.knots[1][1]) / itp.h[1] + 1
    i = clamp(floor(Int, i_float), itp.eqs, length(itp.knots[1]) - itp.eqs)
    x_diff_left = (x[1] - itp.knots[1][i]) / itp.h[1]  # Recompute from actual knot

    # Second dimension (y)
    j_float = (x[2] - itp.knots[2][1]) / itp.h[2] + 1
    j = clamp(floor(Int, j_float), itp.eqs, length(itp.knots[2]) - itp.eqs)
    y_diff_left = (x[2] - itp.knots[2][j]) / itp.h[2]  # Recompute from actual knot

    # Third dimension (z)
    k_float = (x[3] - itp.knots[3][1]) / itp.h[3] + 1
    k = clamp(floor(Int, k_float), itp.eqs, length(itp.knots[3]) - itp.eqs)
    z_diff_left = (x[3] - itp.knots[3][k]) / itp.h[3]  # Recompute from actual knot

    if x_diff_left < 0.5 && y_diff_left < 0.5 && z_diff_left < 0.5
        return itp.coefs[i, j, k]
    elseif x_diff_left < 0.5 && y_diff_left < 0.5 && z_diff_left >= 0.5
        return itp.coefs[i, j, k+1]
    elseif x_diff_left < 0.5 && y_diff_left >= 0.5 && z_diff_left < 0.5
        return itp.coefs[i, j+1, k]
    elseif x_diff_left < 0.5 && y_diff_left >= 0.5 && z_diff_left >= 0.5
        return itp.coefs[i, j+1, k+1]
    elseif x_diff_left >= 0.5 && y_diff_left < 0.5 && z_diff_left < 0.5
        return itp.coefs[i+1, j, k]
    elseif x_diff_left >= 0.5 && y_diff_left < 0.5 && z_diff_left >= 0.5
        return itp.coefs[i+1, j, k+1]
    elseif x_diff_left >= 0.5 && y_diff_left >= 0.5 && z_diff_left < 0.5
        return itp.coefs[i+1, j+1, k]
    else # if x_diff_left >= 0.5 && y_diff_left >= 0.5 && z_diff_left >= 0.5
        return itp.coefs[i+1, j+1, k+1]
    end
end

@inline function (itp::FastConvolutionInterpolation{T,3,TCoefs,IT,Axs,KA,Val{3},Val{:a1},EQ,PR,KP,KBC,DerivativeOrder{DO}})(x::Vararg{Number,3}) where {T,TCoefs,IT,Axs,KA,EQ,PR,KP,KBC,DO}
    # specialized dispatch for 3d linear kernel
        # First dimension (x)
    i_float = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T)
    i = clamp(floor(Int, i_float), itp.eqs, length(itp.knots[1]) - itp.eqs)
    x_diff_left = (x[1] - itp.knots[1][i]) / itp.h[1]  # Recompute from actual knot

    # Second dimension (y)
    j_float = (x[2] - itp.knots[2][1]) / itp.h[2] + one(T)
    j = clamp(floor(Int, j_float), itp.eqs, length(itp.knots[2]) - itp.eqs)
    y_diff_left = (x[2] - itp.knots[2][j]) / itp.h[2]  # Recompute from actual knot

    # Third dimension (z)
    k_float = (x[3] - itp.knots[3][1]) / itp.h[3] + one(T)
    k = clamp(floor(Int, k_float), itp.eqs, length(itp.knots[3]) - itp.eqs)
    z_diff_left = (x[3] - itp.knots[3][k]) / itp.h[3]  # Recompute from actual knot

    # Trilinear interpolation formula
    return @inbounds @fastmath ((1-x_diff_left)*(1-y_diff_left)*(1-z_diff_left)*itp.coefs[i, j, k] + 
                    x_diff_left*(1-y_diff_left)*(1-z_diff_left)*itp.coefs[i+1, j, k] + 
                    (1-x_diff_left)*y_diff_left*(1-z_diff_left)*itp.coefs[i, j+1, k] + 
                    x_diff_left*y_diff_left*(1-z_diff_left)*itp.coefs[i+1, j+1, k] +
                    (1-x_diff_left)*(1-y_diff_left)*z_diff_left*itp.coefs[i, j, k+1] + 
                    x_diff_left*(1-y_diff_left)*z_diff_left*itp.coefs[i+1, j, k+1] + 
                    (1-x_diff_left)*y_diff_left*z_diff_left*itp.coefs[i, j+1, k+1] + 
                    x_diff_left*y_diff_left*z_diff_left*itp.coefs[i+1, j+1, k+1]) *
                    (-one(T)/itp.h[1])^DO *
                    (-one(T)/itp.h[2])^DO *
                    (-one(T)/itp.h[3])^DO
end

function (itp::FastConvolutionInterpolation{T,3,TCoefs,IT,Axs,KA,Val{3},HigherOrderKernel{DG},EQ,PR,KP,KBC,DerivativeOrder{DO}})(x::Vararg{Number,3}) where {T,TCoefs,IT,Axs,KA,DG,EQ,PR,KP,KBC,DO}
    # specialized dispatch for 3d higher order kernels (cubic and above)

    # First dimension (x)
    i_float = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T)
    i = if itp.lazy
        clamp(floor(Int, i_float), 1, length(itp.knots[1]) - 1)
    else
        clamp(floor(Int, i_float), itp.eqs, length(itp.knots[1]) - itp.eqs)
    end
    x_diff_left = (x[1] - itp.knots[1][i]) / itp.h[1]
    x_diff_right = one(T) - x_diff_left
    idx_x = clamp(floor(Int, x_diff_right * (length(itp.pre_range) - one(Int64))) + one(Int64), one(Int64), length(itp.pre_range) - one(Int64))
    idx_x_next = idx_x + one(Int64)
    t_x = (x_diff_right - itp.pre_range[idx_x]) / (itp.pre_range[idx_x_next] - itp.pre_range[idx_x])

    # Second dimension (y)
    j_float = (x[2] - itp.knots[2][1]) / itp.h[2] + one(T)
    j = if itp.lazy
        clamp(floor(Int, j_float), 1, length(itp.knots[2]) - 1)
    else
        clamp(floor(Int, j_float), itp.eqs, length(itp.knots[2]) - itp.eqs)
    end
    y_diff_left = (x[2] - itp.knots[2][j]) / itp.h[2]
    y_diff_right = one(T) - y_diff_left
    idx_y = clamp(floor(Int, y_diff_right * (length(itp.pre_range) - one(Int64))) + one(Int64), one(Int64), length(itp.pre_range) - one(Int64))
    idx_y_next = idx_y + one(Int64)
    t_y = (y_diff_right - itp.pre_range[idx_y]) / (itp.pre_range[idx_y_next] - itp.pre_range[idx_y])

    # Third dimension (z)
    k_float = (x[3] - itp.knots[3][1]) / itp.h[3] + one(T)
    k = if itp.lazy
        clamp(floor(Int, k_float), 1, length(itp.knots[3]) - 1)
    else
        clamp(floor(Int, k_float), itp.eqs, length(itp.knots[3]) - itp.eqs)
    end
    z_diff_left = (x[3] - itp.knots[3][k]) / itp.h[3]
    z_diff_right = one(T) - z_diff_left
    idx_z = clamp(floor(Int, z_diff_right * (length(itp.pre_range) - one(Int64))) + one(Int64), one(Int64), length(itp.pre_range) - one(Int64))
    idx_z_next = idx_z + one(Int64)
    t_z = (z_diff_right - itp.pre_range[idx_z]) / (itp.pre_range[idx_z_next] - itp.pre_range[idx_z])
    
    # Initialize all 8 corner results
    result_000 = zero(T)
    result_100 = zero(T)
    result_010 = zero(T)
    result_110 = zero(T)
    result_001 = zero(T)
    result_101 = zero(T)
    result_011 = zero(T)
    result_111 = zero(T)
    
    # Single triple loop - accumulate all 8 corners simultaneously
    if itp.lazy && (is_boundary_stencil(i, size(itp.coefs, 1), itp.eqs) ||
                    is_boundary_stencil(j, size(itp.coefs, 2), itp.eqs) ||
                    is_boundary_stencil(k, size(itp.coefs, 3), itp.eqs))
        kernel_type = _kernel_sym(itp.kernel_sym)
        ng = itp.eqs - 1
        @inbounds for n in -(itp.eqs-1):itp.eqs
            kz_0 = itp.kernel_pre[idx_z, n+itp.eqs]
            kz_1 = itp.kernel_pre[idx_z_next, n+itp.eqs]
            
            @inbounds for m in -(itp.eqs-1):itp.eqs
                ky_0 = itp.kernel_pre[idx_y, m+itp.eqs]
                ky_1 = itp.kernel_pre[idx_y_next, m+itp.eqs]
                
                @inbounds for l in -(itp.eqs-1):itp.eqs
                    coef = lazy_ghost_value(itp.coefs, (i + ng + l, j + ng + m, k + ng + n), itp.eqs, kernel_type)
                    kx_0 = itp.kernel_pre[idx_x, l+itp.eqs]
                    kx_1 = itp.kernel_pre[idx_x_next, l+itp.eqs]
                    
                    result_000 += coef * kx_0 * ky_0 * kz_0
                    result_100 += coef * kx_1 * ky_0 * kz_0
                    result_010 += coef * kx_0 * ky_1 * kz_0
                    result_110 += coef * kx_1 * ky_1 * kz_0
                    result_001 += coef * kx_0 * ky_0 * kz_1
                    result_101 += coef * kx_1 * ky_0 * kz_1
                    result_011 += coef * kx_0 * ky_1 * kz_1
                    result_111 += coef * kx_1 * ky_1 * kz_1
                end
            end
        end
    else
        @inbounds for n in -(itp.eqs-1):itp.eqs
            kz_0 = itp.kernel_pre[idx_z, n+itp.eqs]
            kz_1 = itp.kernel_pre[idx_z_next, n+itp.eqs]
            
            @inbounds for m in -(itp.eqs-1):itp.eqs
                ky_0 = itp.kernel_pre[idx_y, m+itp.eqs]
                ky_1 = itp.kernel_pre[idx_y_next, m+itp.eqs]
                
                @inbounds for l in -(itp.eqs-1):itp.eqs
                    coef = itp.coefs[i+l, j+m, k+n]
                    kx_0 = itp.kernel_pre[idx_x, l+itp.eqs]
                    kx_1 = itp.kernel_pre[idx_x_next, l+itp.eqs]
                    
                    result_000 += coef * kx_0 * ky_0 * kz_0
                    result_100 += coef * kx_1 * ky_0 * kz_0
                    result_010 += coef * kx_0 * ky_1 * kz_0
                    result_110 += coef * kx_1 * ky_1 * kz_0
                    result_001 += coef * kx_0 * ky_0 * kz_1
                    result_101 += coef * kx_1 * ky_0 * kz_1
                    result_011 += coef * kx_0 * ky_1 * kz_1
                    result_111 += coef * kx_1 * ky_1 * kz_1
                end
            end
        end
    end
    
    # Trilinear interpolation formula
    return @fastmath ((1-t_x)*(1-t_y)*(1-t_z)*result_000 + 
            t_x*(1-t_y)*(1-t_z)*result_100 + 
            (1-t_x)*t_y*(1-t_z)*result_010 + 
            t_x*t_y*(1-t_z)*result_110 +
            (1-t_x)*(1-t_y)*t_z*result_001 + 
            t_x*(1-t_y)*t_z*result_101 + 
            (1-t_x)*t_y*t_z*result_011 + 
            t_x*t_y*t_z*result_111) *
            (-one(T)/itp.h[1])^DO * 
            (-one(T)/itp.h[2])^DO * 
            (-one(T)/itp.h[3])^DO
end

@inline function (itp::FastConvolutionInterpolation{T,3,TCoefs,IT,Axs,KA,Val{3},
                    HigherOrderKernel{DG},EQ,PR,KP,KBC,IntegralOrder,FD,SD,SG})(x::Vararg{Number,3}) where 
                    {T,TCoefs,IT,Axs,KA,DG,EQ,PR,KP,KBC,FD,SD,SG}

    eqs_int = itp.eqs
    n_pre = length(itp.pre_range)
    result = zero(T)

    @inbounds for j3 in 1:size(itp.coefs, 3)
        xj3 = itp.knots[3][eqs_int] + (j3 - eqs_int) * itp.h[3]
        sj3 = (x[3] - xj3) / itp.h[3]

        if abs(sj3) >= T(eqs_int)
            kt3 = T(1//2) * T(sign(sj3))
        else
            col_float3 = T(eqs_int) + sj3
            col3 = clamp(floor(Int, col_float3) + 1, 1, 2 * eqs_int)
            x_diff_right3 = col_float3 - T(col3 - 1)
            continuous_idx3 = x_diff_right3 * T(n_pre - 1) + one(T)
            idx3      = clamp(floor(Int, continuous_idx3), 1, n_pre - 1)
            idx3_next = idx3 + 1
            t3        = continuous_idx3 - T(idx3)
            kt3 = (one(T) - t3) * itp.kernel_pre[idx3, col3] + t3 * itp.kernel_pre[idx3_next, col3]
        end

        lv3 = kt3 - itp.left_values[3][j3]

        @inbounds for j2 in 1:size(itp.coefs, 2)
            xj2 = itp.knots[2][eqs_int] + (j2 - eqs_int) * itp.h[2]
            sj2 = (x[2] - xj2) / itp.h[2]

            if abs(sj2) >= T(eqs_int)
                kt2 = T(1//2) * T(sign(sj2))
            else
                col_float2 = T(eqs_int) + sj2
                col2 = clamp(floor(Int, col_float2) + 1, 1, 2 * eqs_int)
                x_diff_right2 = col_float2 - T(col2 - 1)
                continuous_idx2 = x_diff_right2 * T(n_pre - 1) + one(T)
                idx2      = clamp(floor(Int, continuous_idx2), 1, n_pre - 1)
                idx2_next = idx2 + 1
                t2        = continuous_idx2 - T(idx2)
                kt2 = (one(T) - t2) * itp.kernel_pre[idx2, col2] + t2 * itp.kernel_pre[idx2_next, col2]
            end

            lv2 = (kt2 - itp.left_values[2][j2]) * lv3

            @inbounds for j1 in 1:size(itp.coefs, 1)
                xj1 = itp.knots[1][eqs_int] + (j1 - eqs_int) * itp.h[1]
                sj1 = (x[1] - xj1) / itp.h[1]

                if abs(sj1) >= T(eqs_int)
                    kt1 = T(1//2) * T(sign(sj1))
                else
                    col_float1 = T(eqs_int) + sj1
                    col1 = clamp(floor(Int, col_float1) + 1, 1, 2 * eqs_int)
                    x_diff_right1 = col_float1 - T(col1 - 1)
                    continuous_idx1 = x_diff_right1 * T(n_pre - 1) + one(T)
                    idx1      = clamp(floor(Int, continuous_idx1), 1, n_pre - 1)
                    idx1_next = idx1 + 1
                    t1        = continuous_idx1 - T(idx1)
                    kt1 = (one(T) - t1) * itp.kernel_pre[idx1, col1] + t1 * itp.kernel_pre[idx1_next, col1]
                end

                result += itp.coefs[j1, j2, j3] * (kt1 - itp.left_values[1][j1]) * lv2
            end
        end
    end

    return result * itp.h[1] * itp.h[2] * itp.h[3]
end

@inline function (itp::FastConvolutionInterpolation{T,3,TCoefs,IT,Axs,KA,Val{3},
                    Val{:a1},EQ,PR,KP,KBC,IntegralOrder,FD,SD,SG})(x::Vararg{Number,3}) where 
                    {T,TCoefs,IT,Axs,KA,EQ,PR,KP,KBC,FD,SD,SG}

    eqs_int = itp.eqs
    n_pre = length(itp.pre_range)
    result = zero(T)

    @inbounds for j3 in 1:size(itp.coefs, 3)
        xj3 = itp.knots[3][eqs_int] + (j3 - eqs_int) * itp.h[3]
        sj3 = (x[3] - xj3) / itp.h[3]

        if abs(sj3) >= T(eqs_int)
            kt3 = T(1//2) * T(sign(sj3))
        else
            col_float3 = T(eqs_int) + sj3
            col3 = clamp(floor(Int, col_float3) + 1, 1, 2 * eqs_int)
            x_diff_right3 = col_float3 - T(col3 - 1)
            continuous_idx3 = x_diff_right3 * T(n_pre - 1) + one(T)
            idx3      = clamp(floor(Int, continuous_idx3), 1, n_pre - 1)
            idx3_next = idx3 + 1
            t3        = continuous_idx3 - T(idx3)
            kt3 = (one(T) - t3) * itp.kernel_pre[idx3, col3] + t3 * itp.kernel_pre[idx3_next, col3]
        end

        lv3 = kt3 - itp.left_values[3][j3]

        @inbounds for j2 in 1:size(itp.coefs, 2)
            xj2 = itp.knots[2][eqs_int] + (j2 - eqs_int) * itp.h[2]
            sj2 = (x[2] - xj2) / itp.h[2]

            if abs(sj2) >= T(eqs_int)
                kt2 = T(1//2) * T(sign(sj2))
            else
                col_float2 = T(eqs_int) + sj2
                col2 = clamp(floor(Int, col_float2) + 1, 1, 2 * eqs_int)
                x_diff_right2 = col_float2 - T(col2 - 1)
                continuous_idx2 = x_diff_right2 * T(n_pre - 1) + one(T)
                idx2      = clamp(floor(Int, continuous_idx2), 1, n_pre - 1)
                idx2_next = idx2 + 1
                t2        = continuous_idx2 - T(idx2)
                kt2 = (one(T) - t2) * itp.kernel_pre[idx2, col2] + t2 * itp.kernel_pre[idx2_next, col2]
            end

            lv2 = (kt2 - itp.left_values[2][j2]) * lv3

            @inbounds for j1 in 1:size(itp.coefs, 1)
                xj1 = itp.knots[1][eqs_int] + (j1 - eqs_int) * itp.h[1]
                sj1 = (x[1] - xj1) / itp.h[1]

                if abs(sj1) >= T(eqs_int)
                    kt1 = T(1//2) * T(sign(sj1))
                else
                    col_float1 = T(eqs_int) + sj1
                    col1 = clamp(floor(Int, col_float1) + 1, 1, 2 * eqs_int)
                    x_diff_right1 = col_float1 - T(col1 - 1)
                    continuous_idx1 = x_diff_right1 * T(n_pre - 1) + one(T)
                    idx1      = clamp(floor(Int, continuous_idx1), 1, n_pre - 1)
                    idx1_next = idx1 + 1
                    t1        = continuous_idx1 - T(idx1)
                    kt1 = (one(T) - t1) * itp.kernel_pre[idx1, col1] + t1 * itp.kernel_pre[idx1_next, col1]
                end

                result += itp.coefs[j1, j2, j3] * (kt1 - itp.left_values[1][j1]) * lv2
            end
        end
    end

    return result * itp.h[1] * itp.h[2] * itp.h[3]
end