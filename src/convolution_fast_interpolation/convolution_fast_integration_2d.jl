"""
(itp::FastConvolutionInterpolation{T,2,...})(x::Number, y::Number) — IntegralOrder
Evaluate 2D fast antiderivative at coordinates (x, y).
Dispatches on subgrid mode: :quintic, :cubic, :linear.

The 2D domain decomposes into 9 quadrants around the local stencil box:
  center (K̃×K̃)         — O(eqs²) double loop over local stencil
  edge strips (K̃×tail)  — O(eqs) per strip via 1D prefix sum lookup
  corners (tail×tail)    — O(1) via precomputed 2D cross-sum arrays

Result is (center + strips + corners) * h[1] * h[2], anchored to zero at the
leftmost interior knot in each dimension. O(1) in grid size, allocation-free.
See also: FastConvolutionInterpolation, convolution_fast_integration_1d.
"""

@inline function (itp::FastConvolutionInterpolation{T,2,TCoefs,IT,Axs,KA,Val{2},
                    HigherOrderKernel{DG},EQ,PR,KP,KBC,IntegralOrder,FD,SD,Val{:cubic}})(x::Vararg{Number,2}) where 
                    {T,TCoefs,IT,Axs,KA,DG,EQ,PR,KP,KBC,FD,SD}

    eqs_int = itp.eqs
    n_pre = length(itp.pre_range)
    h_pre = one(T) / T(n_pre - 1)
    result = zero(T)

    i1_float = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T)
    i1       = clamp(floor(Int, i1_float), eqs_int, size(itp.coefs, 1) - eqs_int)
    i2_float = (x[2] - itp.knots[2][1]) / itp.h[2] + one(T)
    i2       = clamp(floor(Int, i2_float), eqs_int, size(itp.coefs, 2) - eqs_int)

    # tail boundary indices
    l1 = i1 - eqs_int;      l1_ok = l1 >= 1
    r1 = i1 + eqs_int + 1;  r1_ok = r1 <= size(itp.coefs, 1)
    l2 = i2 - eqs_int;      l2_ok = l2 >= 1
    r2 = i2 + eqs_int + 1;  r2_ok = r2 <= size(itp.coefs, 2)

    # ── center: K̃×K̃, and side strips: cumcoefs×K̃ ──────────────────
    @inbounds for j2 in (i2 - eqs_int + 1):(i2 + eqs_int)
        xj2 = itp.knots[2][eqs_int] + (j2 - eqs_int) * itp.h[2]
        sj2 = (x[2] - xj2) / itp.h[2]
        if abs(sj2) >= T(eqs_int)
            kt2 = T(1//2) * T(sign(sj2))
        else
            col_float2      = T(eqs_int) + sj2
            col2            = clamp(floor(Int, col_float2) + 1, 1, 2 * eqs_int)
            x_diff_right2   = col_float2 - T(col2 - 1)
            continuous_idx2 = x_diff_right2 * T(n_pre - 1) + one(T)
            idx2      = clamp(floor(Int, continuous_idx2), 1, n_pre - 1)
            idx2_next = idx2 + 1
            t2        = continuous_idx2 - T(idx2)
            kt2 = cubic_hermite(t2,
                    itp.kernel_pre[idx2,    col2], itp.kernel_pre[idx2_next,    col2],
                    itp.kernel_d1_pre[idx2, col2], itp.kernel_d1_pre[idx2_next, col2],
                    h_pre)
        end
        lv2 = kt2 - itp.left_values[2][j2]

        # center: K̃×K̃
        @inbounds for j1 in (i1 - eqs_int + 1):(i1 + eqs_int)
            xj1 = itp.knots[1][eqs_int] + (j1 - eqs_int) * itp.h[1]
            sj1 = (x[1] - xj1) / itp.h[1]
            if abs(sj1) >= T(eqs_int)
                kt1 = T(1//2) * T(sign(sj1))
            else
                col_float1      = T(eqs_int) + sj1
                col1            = clamp(floor(Int, col_float1) + 1, 1, 2 * eqs_int)
                x_diff_right1   = col_float1 - T(col1 - 1)
                continuous_idx1 = x_diff_right1 * T(n_pre - 1) + one(T)
                idx1      = clamp(floor(Int, continuous_idx1), 1, n_pre - 1)
                idx1_next = idx1 + 1
                t1        = continuous_idx1 - T(idx1)
                kt1 = cubic_hermite(t1,
                        itp.kernel_pre[idx1,    col1], itp.kernel_pre[idx1_next,    col1],
                        itp.kernel_d1_pre[idx1, col1], itp.kernel_d1_pre[idx1_next, col1],
                        h_pre)
            end
            result += itp.coefs[j1, j2] * (kt1 - itp.left_values[1][j1]) * lv2
        end

        # left strip: (-½)×K̃ — O(1) lookup weighted by lv2
        tl1 = l1_ok ? itp.cumcoefs_left[1][l1,  j2] : zero(T)
        # right strip: (+½)×K̃ — O(1) lookup weighted by lv2
        tr1 = r1_ok ? itp.cumcoefs_right[1][r1, j2] : zero(T)
        result += (tl1 + tr1) * lv2
    end

    # ── top strip: K̃×(-½) ────────────────────────────────────────────
    if l2_ok
        @inbounds for j1 in (i1 - eqs_int + 1):(i1 + eqs_int)
            xj1 = itp.knots[1][eqs_int] + (j1 - eqs_int) * itp.h[1]
            sj1 = (x[1] - xj1) / itp.h[1]
            if abs(sj1) >= T(eqs_int)
                kt1 = T(1//2) * T(sign(sj1))
            else
                col_float1      = T(eqs_int) + sj1
                col1            = clamp(floor(Int, col_float1) + 1, 1, 2 * eqs_int)
                x_diff_right1   = col_float1 - T(col1 - 1)
                continuous_idx1 = x_diff_right1 * T(n_pre - 1) + one(T)
                idx1      = clamp(floor(Int, continuous_idx1), 1, n_pre - 1)
                idx1_next = idx1 + 1
                t1        = continuous_idx1 - T(idx1)
                kt1 = cubic_hermite(t1,
                        itp.kernel_pre[idx1,    col1], itp.kernel_pre[idx1_next,    col1],
                        itp.kernel_d1_pre[idx1, col1], itp.kernel_d1_pre[idx1_next, col1],
                        h_pre)
            end
            result += itp.cumcoefs_left[2][j1, l2] * (kt1 - itp.left_values[1][j1])
        end
        result += l1_ok ? itp.cross_ll[2][l1, l2] : zero(T)
        result += r1_ok ? itp.cross_rl[2][r1, l2] : zero(T)
    end

    # ── bottom strip: K̃×(+½) ─────────────────────────────────────────
    if r2_ok
        @inbounds for j1 in (i1 - eqs_int + 1):(i1 + eqs_int)
            xj1 = itp.knots[1][eqs_int] + (j1 - eqs_int) * itp.h[1]
            sj1 = (x[1] - xj1) / itp.h[1]
            if abs(sj1) >= T(eqs_int)
                kt1 = T(1//2) * T(sign(sj1))
            else
                col_float1      = T(eqs_int) + sj1
                col1            = clamp(floor(Int, col_float1) + 1, 1, 2 * eqs_int)
                x_diff_right1   = col_float1 - T(col1 - 1)
                continuous_idx1 = x_diff_right1 * T(n_pre - 1) + one(T)
                idx1      = clamp(floor(Int, continuous_idx1), 1, n_pre - 1)
                idx1_next = idx1 + 1
                t1        = continuous_idx1 - T(idx1)
                kt1 = cubic_hermite(t1,
                        itp.kernel_pre[idx1,    col1], itp.kernel_pre[idx1_next,    col1],
                        itp.kernel_d1_pre[idx1, col1], itp.kernel_d1_pre[idx1_next, col1],
                        h_pre)
            end
            result += itp.cumcoefs_right[2][j1, r2] * (kt1 - itp.left_values[1][j1])
        end
        result += l1_ok ? itp.cross_lr[2][l1, r2] : zero(T)
        result += r1_ok ? itp.cross_rr[2][r1, r2] : zero(T)
    end

    return result * itp.h[1] * itp.h[2]
end

@inline function (itp::FastConvolutionInterpolation{T,2,TCoefs,IT,Axs,KA,Val{2},
                    Val{:a1},EQ,PR,KP,KBC,IntegralOrder,FD,SD,Val{:cubic}})(x::Vararg{Number,2}) where 
                    {T,TCoefs,IT,Axs,KA,EQ,PR,KP,KBC,FD,SD}

    eqs_int = itp.eqs
    n_pre = length(itp.pre_range)
    h_pre = one(T) / T(n_pre - 1)
    result = zero(T)

    i1_float = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T)
    i1       = clamp(floor(Int, i1_float), eqs_int, size(itp.coefs, 1) - eqs_int)
    i2_float = (x[2] - itp.knots[2][1]) / itp.h[2] + one(T)
    i2       = clamp(floor(Int, i2_float), eqs_int, size(itp.coefs, 2) - eqs_int)

    # tail boundary indices
    l1 = i1 - eqs_int;      l1_ok = l1 >= 1
    r1 = i1 + eqs_int + 1;  r1_ok = r1 <= size(itp.coefs, 1)
    l2 = i2 - eqs_int;      l2_ok = l2 >= 1
    r2 = i2 + eqs_int + 1;  r2_ok = r2 <= size(itp.coefs, 2)

    # ── center: K̃×K̃, and side strips: cumcoefs×K̃ ──────────────────
    @inbounds for j2 in (i2 - eqs_int + 1):(i2 + eqs_int)
        xj2 = itp.knots[2][eqs_int] + (j2 - eqs_int) * itp.h[2]
        sj2 = (x[2] - xj2) / itp.h[2]
        if abs(sj2) >= T(eqs_int)
            kt2 = T(1//2) * T(sign(sj2))
        else
            col_float2      = T(eqs_int) + sj2
            col2            = clamp(floor(Int, col_float2) + 1, 1, 2 * eqs_int)
            x_diff_right2   = col_float2 - T(col2 - 1)
            continuous_idx2 = x_diff_right2 * T(n_pre - 1) + one(T)
            idx2      = clamp(floor(Int, continuous_idx2), 1, n_pre - 1)
            idx2_next = idx2 + 1
            t2        = continuous_idx2 - T(idx2)
            kt2 = cubic_hermite(t2,
                    itp.kernel_pre[idx2,    col2], itp.kernel_pre[idx2_next,    col2],
                    itp.kernel_d1_pre[idx2, col2], itp.kernel_d1_pre[idx2_next, col2],
                    h_pre)
        end
        lv2 = kt2 - itp.left_values[2][j2]

        # center: K̃×K̃
        @inbounds for j1 in (i1 - eqs_int + 1):(i1 + eqs_int)
            xj1 = itp.knots[1][eqs_int] + (j1 - eqs_int) * itp.h[1]
            sj1 = (x[1] - xj1) / itp.h[1]
            if abs(sj1) >= T(eqs_int)
                kt1 = T(1//2) * T(sign(sj1))
            else
                col_float1      = T(eqs_int) + sj1
                col1            = clamp(floor(Int, col_float1) + 1, 1, 2 * eqs_int)
                x_diff_right1   = col_float1 - T(col1 - 1)
                continuous_idx1 = x_diff_right1 * T(n_pre - 1) + one(T)
                idx1      = clamp(floor(Int, continuous_idx1), 1, n_pre - 1)
                idx1_next = idx1 + 1
                t1        = continuous_idx1 - T(idx1)
                kt1 = cubic_hermite(t1,
                        itp.kernel_pre[idx1,    col1], itp.kernel_pre[idx1_next,    col1],
                        itp.kernel_d1_pre[idx1, col1], itp.kernel_d1_pre[idx1_next, col1],
                        h_pre)
            end
            result += itp.coefs[j1, j2] * (kt1 - itp.left_values[1][j1]) * lv2
        end

        # left strip: (-½)×K̃ — O(1) lookup weighted by lv2
        tl1 = l1_ok ? itp.cumcoefs_left[1][l1,  j2] : zero(T)
        # right strip: (+½)×K̃ — O(1) lookup weighted by lv2
        tr1 = r1_ok ? itp.cumcoefs_right[1][r1, j2] : zero(T)
        result += (tl1 + tr1) * lv2
    end

    # ── top strip: K̃×(-½) ────────────────────────────────────────────
    if l2_ok
        @inbounds for j1 in (i1 - eqs_int + 1):(i1 + eqs_int)
            xj1 = itp.knots[1][eqs_int] + (j1 - eqs_int) * itp.h[1]
            sj1 = (x[1] - xj1) / itp.h[1]
            if abs(sj1) >= T(eqs_int)
                kt1 = T(1//2) * T(sign(sj1))
            else
                col_float1      = T(eqs_int) + sj1
                col1            = clamp(floor(Int, col_float1) + 1, 1, 2 * eqs_int)
                x_diff_right1   = col_float1 - T(col1 - 1)
                continuous_idx1 = x_diff_right1 * T(n_pre - 1) + one(T)
                idx1      = clamp(floor(Int, continuous_idx1), 1, n_pre - 1)
                idx1_next = idx1 + 1
                t1        = continuous_idx1 - T(idx1)
                kt1 = cubic_hermite(t1,
                        itp.kernel_pre[idx1,    col1], itp.kernel_pre[idx1_next,    col1],
                        itp.kernel_d1_pre[idx1, col1], itp.kernel_d1_pre[idx1_next, col1],
                        h_pre)
            end
            result += itp.cumcoefs_left[2][j1, l2] * (kt1 - itp.left_values[1][j1])
        end
        result += l1_ok ? itp.cross_ll[2][l1, l2] : zero(T)
        result += r1_ok ? itp.cross_rl[2][r1, l2] : zero(T)
    end

    # ── bottom strip: K̃×(+½) ─────────────────────────────────────────
    if r2_ok
        @inbounds for j1 in (i1 - eqs_int + 1):(i1 + eqs_int)
            xj1 = itp.knots[1][eqs_int] + (j1 - eqs_int) * itp.h[1]
            sj1 = (x[1] - xj1) / itp.h[1]
            if abs(sj1) >= T(eqs_int)
                kt1 = T(1//2) * T(sign(sj1))
            else
                col_float1      = T(eqs_int) + sj1
                col1            = clamp(floor(Int, col_float1) + 1, 1, 2 * eqs_int)
                x_diff_right1   = col_float1 - T(col1 - 1)
                continuous_idx1 = x_diff_right1 * T(n_pre - 1) + one(T)
                idx1      = clamp(floor(Int, continuous_idx1), 1, n_pre - 1)
                idx1_next = idx1 + 1
                t1        = continuous_idx1 - T(idx1)
                kt1 = cubic_hermite(t1,
                        itp.kernel_pre[idx1,    col1], itp.kernel_pre[idx1_next,    col1],
                        itp.kernel_d1_pre[idx1, col1], itp.kernel_d1_pre[idx1_next, col1],
                        h_pre)
            end
            result += itp.cumcoefs_right[2][j1, r2] * (kt1 - itp.left_values[1][j1])
        end
        result += l1_ok ? itp.cross_lr[2][l1, r2] : zero(T)
        result += r1_ok ? itp.cross_rr[2][r1, r2] : zero(T)
    end

    return result * itp.h[1] * itp.h[2]
end

@inline function (itp::FastConvolutionInterpolation{T,2,TCoefs,IT,Axs,KA,Val{2},
                    HigherOrderKernel{DG},EQ,PR,KP,KBC,IntegralOrder,FD,SD,Val{:quintic}})(x::Vararg{Number,2}) where 
                    {T,TCoefs,IT,Axs,KA,DG,EQ,PR,KP,KBC,FD,SD}

    eqs_int = itp.eqs
    n_pre = length(itp.pre_range)
    h_pre = one(T) / T(n_pre - 1)
    result = zero(T)

    i1_float = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T)
    i1       = clamp(floor(Int, i1_float), eqs_int, size(itp.coefs, 1) - eqs_int)
    i2_float = (x[2] - itp.knots[2][1]) / itp.h[2] + one(T)
    i2       = clamp(floor(Int, i2_float), eqs_int, size(itp.coefs, 2) - eqs_int)

    l1 = i1 - eqs_int;      l1_ok = l1 >= 1
    r1 = i1 + eqs_int + 1;  r1_ok = r1 <= size(itp.coefs, 1)
    l2 = i2 - eqs_int;      l2_ok = l2 >= 1
    r2 = i2 + eqs_int + 1;  r2_ok = r2 <= size(itp.coefs, 2)

    @inbounds for j2 in (i2 - eqs_int + 1):(i2 + eqs_int)
        xj2 = itp.knots[2][eqs_int] + (j2 - eqs_int) * itp.h[2]
        sj2 = (x[2] - xj2) / itp.h[2]
        if abs(sj2) >= T(eqs_int)
            kt2 = T(1//2) * T(sign(sj2))
        else
            col_float2      = T(eqs_int) + sj2
            col2            = clamp(floor(Int, col_float2) + 1, 1, 2 * eqs_int)
            x_diff_right2   = col_float2 - T(col2 - 1)
            continuous_idx2 = x_diff_right2 * T(n_pre - 1) + one(T)
            idx2      = clamp(floor(Int, continuous_idx2), 1, n_pre - 1)
            idx2_next = idx2 + 1
            t2        = continuous_idx2 - T(idx2)
            kt2 = quintic_hermite(t2,
                    itp.kernel_pre[idx2,    col2], itp.kernel_pre[idx2_next,    col2],
                    itp.kernel_d1_pre[idx2, col2], itp.kernel_d1_pre[idx2_next, col2],
                    itp.kernel_d2_pre[idx2, col2], itp.kernel_d2_pre[idx2_next, col2],
                    h_pre)
        end
        lv2 = kt2 - itp.left_values[2][j2]

        @inbounds for j1 in (i1 - eqs_int + 1):(i1 + eqs_int)
            xj1 = itp.knots[1][eqs_int] + (j1 - eqs_int) * itp.h[1]
            sj1 = (x[1] - xj1) / itp.h[1]
            if abs(sj1) >= T(eqs_int)
                kt1 = T(1//2) * T(sign(sj1))
            else
                col_float1      = T(eqs_int) + sj1
                col1            = clamp(floor(Int, col_float1) + 1, 1, 2 * eqs_int)
                x_diff_right1   = col_float1 - T(col1 - 1)
                continuous_idx1 = x_diff_right1 * T(n_pre - 1) + one(T)
                idx1      = clamp(floor(Int, continuous_idx1), 1, n_pre - 1)
                idx1_next = idx1 + 1
                t1        = continuous_idx1 - T(idx1)
                kt1 = quintic_hermite(t1,
                        itp.kernel_pre[idx1,    col1], itp.kernel_pre[idx1_next,    col1],
                        itp.kernel_d1_pre[idx1, col1], itp.kernel_d1_pre[idx1_next, col1],
                        itp.kernel_d2_pre[idx1, col1], itp.kernel_d2_pre[idx1_next, col1],
                        h_pre)
            end
            result += itp.coefs[j1, j2] * (kt1 - itp.left_values[1][j1]) * lv2
        end

        tl1 = l1_ok ? itp.cumcoefs_left[1][l1,  j2] : zero(T)
        tr1 = r1_ok ? itp.cumcoefs_right[1][r1, j2] : zero(T)
        result += (tl1 + tr1) * lv2
    end

    if l2_ok
        @inbounds for j1 in (i1 - eqs_int + 1):(i1 + eqs_int)
            xj1 = itp.knots[1][eqs_int] + (j1 - eqs_int) * itp.h[1]
            sj1 = (x[1] - xj1) / itp.h[1]
            if abs(sj1) >= T(eqs_int)
                kt1 = T(1//2) * T(sign(sj1))
            else
                col_float1      = T(eqs_int) + sj1
                col1            = clamp(floor(Int, col_float1) + 1, 1, 2 * eqs_int)
                x_diff_right1   = col_float1 - T(col1 - 1)
                continuous_idx1 = x_diff_right1 * T(n_pre - 1) + one(T)
                idx1      = clamp(floor(Int, continuous_idx1), 1, n_pre - 1)
                idx1_next = idx1 + 1
                t1        = continuous_idx1 - T(idx1)
                kt1 = quintic_hermite(t1,
                        itp.kernel_pre[idx1,    col1], itp.kernel_pre[idx1_next,    col1],
                        itp.kernel_d1_pre[idx1, col1], itp.kernel_d1_pre[idx1_next, col1],
                        itp.kernel_d2_pre[idx1, col1], itp.kernel_d2_pre[idx1_next, col1],
                        h_pre)
            end
            result += itp.cumcoefs_left[2][j1, l2] * (kt1 - itp.left_values[1][j1])
        end
        result += l1_ok ? itp.cross_ll[2][l1, l2] : zero(T)
        result += r1_ok ? itp.cross_rl[2][r1, l2] : zero(T)
    end

    if r2_ok
        @inbounds for j1 in (i1 - eqs_int + 1):(i1 + eqs_int)
            xj1 = itp.knots[1][eqs_int] + (j1 - eqs_int) * itp.h[1]
            sj1 = (x[1] - xj1) / itp.h[1]
            if abs(sj1) >= T(eqs_int)
                kt1 = T(1//2) * T(sign(sj1))
            else
                col_float1      = T(eqs_int) + sj1
                col1            = clamp(floor(Int, col_float1) + 1, 1, 2 * eqs_int)
                x_diff_right1   = col_float1 - T(col1 - 1)
                continuous_idx1 = x_diff_right1 * T(n_pre - 1) + one(T)
                idx1      = clamp(floor(Int, continuous_idx1), 1, n_pre - 1)
                idx1_next = idx1 + 1
                t1        = continuous_idx1 - T(idx1)
                kt1 = quintic_hermite(t1,
                        itp.kernel_pre[idx1,    col1], itp.kernel_pre[idx1_next,    col1],
                        itp.kernel_d1_pre[idx1, col1], itp.kernel_d1_pre[idx1_next, col1],
                        itp.kernel_d2_pre[idx1, col1], itp.kernel_d2_pre[idx1_next, col1],
                        h_pre)
            end
            result += itp.cumcoefs_right[2][j1, r2] * (kt1 - itp.left_values[1][j1])
        end
        result += l1_ok ? itp.cross_lr[2][l1, r2] : zero(T)
        result += r1_ok ? itp.cross_rr[2][r1, r2] : zero(T)
    end

    return result * itp.h[1] * itp.h[2]
end

@inline function (itp::FastConvolutionInterpolation{T,2,TCoefs,IT,Axs,KA,Val{2},
                    HigherOrderKernel{DG},EQ,PR,KP,KBC,IntegralOrder,FD,SD,Val{:linear}})(x::Vararg{Number,2}) where 
                    {T,TCoefs,IT,Axs,KA,DG,EQ,PR,KP,KBC,FD,SD}

    eqs_int = itp.eqs
    n_pre = length(itp.pre_range)
    result = zero(T)

    i1_float = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T)
    i1       = clamp(floor(Int, i1_float), eqs_int, size(itp.coefs, 1) - eqs_int)
    i2_float = (x[2] - itp.knots[2][1]) / itp.h[2] + one(T)
    i2       = clamp(floor(Int, i2_float), eqs_int, size(itp.coefs, 2) - eqs_int)

    l1 = i1 - eqs_int;      l1_ok = l1 >= 1
    r1 = i1 + eqs_int + 1;  r1_ok = r1 <= size(itp.coefs, 1)
    l2 = i2 - eqs_int;      l2_ok = l2 >= 1
    r2 = i2 + eqs_int + 1;  r2_ok = r2 <= size(itp.coefs, 2)

    @inbounds for j2 in (i2 - eqs_int + 1):(i2 + eqs_int)
        xj2 = itp.knots[2][eqs_int] + (j2 - eqs_int) * itp.h[2]
        sj2 = (x[2] - xj2) / itp.h[2]
        if abs(sj2) >= T(eqs_int)
            kt2 = T(1//2) * T(sign(sj2))
        else
            col_float2      = T(eqs_int) + sj2
            col2            = clamp(floor(Int, col_float2) + 1, 1, 2 * eqs_int)
            x_diff_right2   = col_float2 - T(col2 - 1)
            continuous_idx2 = x_diff_right2 * T(n_pre - 1) + one(T)
            idx2      = clamp(floor(Int, continuous_idx2), 1, n_pre - 1)
            idx2_next = idx2 + 1
            t2        = continuous_idx2 - T(idx2)
            kt2 = (one(T) - t2) * itp.kernel_pre[idx2, col2] + t2 * itp.kernel_pre[idx2_next, col2]
        end
        lv2 = kt2 - itp.left_values[2][j2]

        @inbounds for j1 in (i1 - eqs_int + 1):(i1 + eqs_int)
            xj1 = itp.knots[1][eqs_int] + (j1 - eqs_int) * itp.h[1]
            sj1 = (x[1] - xj1) / itp.h[1]
            if abs(sj1) >= T(eqs_int)
                kt1 = T(1//2) * T(sign(sj1))
            else
                col_float1      = T(eqs_int) + sj1
                col1            = clamp(floor(Int, col_float1) + 1, 1, 2 * eqs_int)
                x_diff_right1   = col_float1 - T(col1 - 1)
                continuous_idx1 = x_diff_right1 * T(n_pre - 1) + one(T)
                idx1      = clamp(floor(Int, continuous_idx1), 1, n_pre - 1)
                idx1_next = idx1 + 1
                t1        = continuous_idx1 - T(idx1)
                kt1 = (one(T) - t1) * itp.kernel_pre[idx1, col1] + t1 * itp.kernel_pre[idx1_next, col1]
            end
            result += itp.coefs[j1, j2] * (kt1 - itp.left_values[1][j1]) * lv2
        end

        tl1 = l1_ok ? itp.cumcoefs_left[1][l1,  j2] : zero(T)
        tr1 = r1_ok ? itp.cumcoefs_right[1][r1, j2] : zero(T)
        result += (tl1 + tr1) * lv2
    end

    if l2_ok
        @inbounds for j1 in (i1 - eqs_int + 1):(i1 + eqs_int)
            xj1 = itp.knots[1][eqs_int] + (j1 - eqs_int) * itp.h[1]
            sj1 = (x[1] - xj1) / itp.h[1]
            if abs(sj1) >= T(eqs_int)
                kt1 = T(1//2) * T(sign(sj1))
            else
                col_float1      = T(eqs_int) + sj1
                col1            = clamp(floor(Int, col_float1) + 1, 1, 2 * eqs_int)
                x_diff_right1   = col_float1 - T(col1 - 1)
                continuous_idx1 = x_diff_right1 * T(n_pre - 1) + one(T)
                idx1      = clamp(floor(Int, continuous_idx1), 1, n_pre - 1)
                idx1_next = idx1 + 1
                t1        = continuous_idx1 - T(idx1)
                kt1 = (one(T) - t1) * itp.kernel_pre[idx1, col1] + t1 * itp.kernel_pre[idx1_next, col1]
            end
            result += itp.cumcoefs_left[2][j1, l2] * (kt1 - itp.left_values[1][j1])
        end
        result += l1_ok ? itp.cross_ll[2][l1, l2] : zero(T)
        result += r1_ok ? itp.cross_rl[2][r1, l2] : zero(T)
    end

    if r2_ok
        @inbounds for j1 in (i1 - eqs_int + 1):(i1 + eqs_int)
            xj1 = itp.knots[1][eqs_int] + (j1 - eqs_int) * itp.h[1]
            sj1 = (x[1] - xj1) / itp.h[1]
            if abs(sj1) >= T(eqs_int)
                kt1 = T(1//2) * T(sign(sj1))
            else
                col_float1      = T(eqs_int) + sj1
                col1            = clamp(floor(Int, col_float1) + 1, 1, 2 * eqs_int)
                x_diff_right1   = col_float1 - T(col1 - 1)
                continuous_idx1 = x_diff_right1 * T(n_pre - 1) + one(T)
                idx1      = clamp(floor(Int, continuous_idx1), 1, n_pre - 1)
                idx1_next = idx1 + 1
                t1        = continuous_idx1 - T(idx1)
                kt1 = (one(T) - t1) * itp.kernel_pre[idx1, col1] + t1 * itp.kernel_pre[idx1_next, col1]
            end
            result += itp.cumcoefs_right[2][j1, r2] * (kt1 - itp.left_values[1][j1])
        end
        result += l1_ok ? itp.cross_lr[2][l1, r2] : zero(T)
        result += r1_ok ? itp.cross_rr[2][r1, r2] : zero(T)
    end

    return result * itp.h[1] * itp.h[2]
end