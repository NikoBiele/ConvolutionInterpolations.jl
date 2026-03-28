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

@inline function (itp::FastConvolutionInterpolation{T,2,2,TCoefs,Axs,KA,Val{2},
                    DG,EQ,PR,KP,KBC,IntegralOrder,FD,SD,Val{SG},Val{false},Val{2}})(x::Vararg{Number,2}) where 
                    {T<:AbstractFloat,TCoefs<:AbstractArray{T,2},
                    Axs<:Tuple{<:AbstractVector,<:AbstractVector},
                    KA<:Tuple{<:Nothing,<:Nothing},DG,EQ<:Tuple{Int,Int},
                    PR<:Tuple{<:AbstractVector,<:AbstractVector},
                    KP,KBC<:Tuple{<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol}},
                    FD,SD,SG}

    eqs_1 = itp.eqs[1]
    eqs_2 = itp.eqs[2]
    n_pre = ntuple(i -> length(itp.pre_range[i]), 2)
    h_pre = ntuple(i -> one(T) / T(n_pre[i] - 1), 2)
    result = zero(T)

    i1_float = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T)
    i1       = clamp(floor(Int, i1_float), eqs_1, size(itp.coefs, 1) - eqs_1)
    i2_float = (x[2] - itp.knots[2][1]) / itp.h[2] + one(T)
    i2       = clamp(floor(Int, i2_float), eqs_2, size(itp.coefs, 2) - eqs_2)

    # tail boundary indices
    l1 = i1 - eqs_1;      l1_ok = l1 >= 1
    r1 = i1 + eqs_1 + 1;  r1_ok = r1 <= size(itp.coefs, 1)
    l2 = i2 - eqs_2;      l2_ok = l2 >= 1
    r2 = i2 + eqs_2 + 1;  r2_ok = r2 <= size(itp.coefs, 2)

    # ── center: K̃×K̃, and side strips: cumcoefs×K̃ ──────────────────
    @inbounds for j2 in (i2 - eqs_2 + 1):(i2 + eqs_2)
        xj2 = itp.knots[2][eqs_2] + (j2 - eqs_2) * itp.h[2]
        sj2 = (x[2] - xj2) / itp.h[2]
        if abs(sj2) >= T(eqs_2)
            kt2 = T(1//2) * T(sign(sj2))
        else
            col_float2      = T(eqs_2) + sj2
            col2            = clamp(floor(Int, col_float2) + 1, 1, 2 * eqs_2)
            x_diff_right2   = col_float2 - T(col2 - 1)
            continuous_idx2 = x_diff_right2 * T(n_pre[2] - 1) + one(T)
            idx2      = clamp(floor(Int, continuous_idx2), 1, n_pre[2] - 1)
            idx2_next = idx2 + 1
            t2        = continuous_idx2 - T(idx2)
            if SG[2] == :linear
                kt2 = (one(T) - t2) * itp.kernel_pre[2][idx2, col2] + t2 * itp.kernel_pre[2][idx2_next, col2]
            elseif SG[2] == :cubic
                kt2 = cubic_hermite(t2,
                    itp.kernel_pre[2][idx2,    col2], itp.kernel_pre[2][idx2_next,    col2],
                    itp.kernel_d1_pre[2][idx2, col2], itp.kernel_d1_pre[2][idx2_next, col2],
                    h_pre[2])
            elseif SG[2] == :quintic
                kt2 = quintic_hermite(t2,
                    itp.kernel_pre[2][idx2,    col2], itp.kernel_pre[2][idx2_next,    col2],
                    itp.kernel_d1_pre[2][idx2, col2], itp.kernel_d1_pre[2][idx2_next, col2],
                    itp.kernel_d2_pre[2][idx2, col2], itp.kernel_d2_pre[2][idx2_next, col2],
                    h_pre[2])
            end
        end
        lv2 = kt2 - itp.left_values[2][j2]

        # center: K̃×K̃
        @inbounds for j1 in (i1 - eqs_1 + 1):(i1 + eqs_1)
            xj1 = itp.knots[1][eqs_1] + (j1 - eqs_1) * itp.h[1]
            sj1 = (x[1] - xj1) / itp.h[1]
            if abs(sj1) >= T(eqs_1)
                kt1 = T(1//2) * T(sign(sj1))
            else
                col_float1      = T(eqs_1) + sj1
                col1            = clamp(floor(Int, col_float1) + 1, 1, 2 * eqs_1)
                x_diff_right1   = col_float1 - T(col1 - 1)
                continuous_idx1 = x_diff_right1 * T(n_pre[1] - 1) + one(T)
                idx1      = clamp(floor(Int, continuous_idx1), 1, n_pre[1] - 1)
                idx1_next = idx1 + 1
                t1        = continuous_idx1 - T(idx1)
                if SG[1] == :linear
                    kt1 = (one(T) - t1) * itp.kernel_pre[1][idx1, col1] + t1 * itp.kernel_pre[1][idx1_next, col1]
                elseif SG[1] == :cubic
                    kt1 = cubic_hermite(t1,
                        itp.kernel_pre[1][idx1,    col1], itp.kernel_pre[1][idx1_next,    col1],
                        itp.kernel_d1_pre[1][idx1, col1], itp.kernel_d1_pre[1][idx1_next, col1],
                        h_pre[1])
                elseif SG[1] == :quintic
                    kt1 = quintic_hermite(t1,
                        itp.kernel_pre[1][idx1,    col1], itp.kernel_pre[1][idx1_next,    col1],
                        itp.kernel_d1_pre[1][idx1, col1], itp.kernel_d1_pre[1][idx1_next, col1],
                        itp.kernel_d2_pre[1][idx1, col1], itp.kernel_d2_pre[1][idx1_next, col1],
                        h_pre[1])
                end
            end
            result += itp.coefs[j1, j2] * (kt1 - itp.left_values[1][j1]) * lv2
        end

        # left strip: (-½)×K̃ — O(1) lookup weighted by lv2
        tl1 = l1_ok ? itp.tail1_left[1][l1,  j2] : zero(T)
        # right strip: (+½)×K̃ — O(1) lookup weighted by lv2
        tr1 = r1_ok ? itp.tail1_right[1][r1, j2] : zero(T)
        result += (tl1 + tr1) * lv2
    end

    # ── top strip: K̃×(-½) ────────────────────────────────────────────
    if l2_ok
        @inbounds for j1 in (i1 - eqs_1 + 1):(i1 + eqs_1)
            xj1 = itp.knots[1][eqs_1] + (j1 - eqs_1) * itp.h[1]
            sj1 = (x[1] - xj1) / itp.h[1]
            if abs(sj1) >= T(eqs_1)
                kt1 = T(1//2) * T(sign(sj1))
            else
                col_float1      = T(eqs_1) + sj1
                col1            = clamp(floor(Int, col_float1) + 1, 1, 2 * eqs_1)
                x_diff_right1   = col_float1 - T(col1 - 1)
                continuous_idx1 = x_diff_right1 * T(n_pre[1] - 1) + one(T)
                idx1      = clamp(floor(Int, continuous_idx1), 1, n_pre[1] - 1)
                idx1_next = idx1 + 1
                t1        = continuous_idx1 - T(idx1)
                if SG[1] == :linear
                    kt1 = (one(T) - t1) * itp.kernel_pre[1][idx1, col1] + t1 * itp.kernel_pre[1][idx1_next, col1]
                elseif SG[1] == :cubic
                    kt1 = cubic_hermite(t1,
                        itp.kernel_pre[1][idx1,    col1], itp.kernel_pre[1][idx1_next,    col1],
                        itp.kernel_d1_pre[1][idx1, col1], itp.kernel_d1_pre[1][idx1_next, col1],
                        h_pre[1])
                elseif SG[1] == :quintic
                    kt1 = quintic_hermite(t1,
                        itp.kernel_pre[1][idx1,    col1], itp.kernel_pre[1][idx1_next,    col1],
                        itp.kernel_d1_pre[1][idx1, col1], itp.kernel_d1_pre[1][idx1_next, col1],
                        itp.kernel_d2_pre[1][idx1, col1], itp.kernel_d2_pre[1][idx1_next, col1],
                        h_pre[1])
                end
            end
            result += itp.tail1_left[2][j1, l2] * (kt1 - itp.left_values[1][j1])
        end
        result += l1_ok ? itp.tail2_ll[l1, l2] : zero(T)
        result += r1_ok ? itp.tail2_rl[r1, l2] : zero(T)
    end

    # ── bottom strip: K̃×(+½) ─────────────────────────────────────────
    if r2_ok
        @inbounds for j1 in (i1 - eqs_1 + 1):(i1 + eqs_1)
            xj1 = itp.knots[1][eqs_1] + (j1 - eqs_1) * itp.h[1]
            sj1 = (x[1] - xj1) / itp.h[1]
            if abs(sj1) >= T(eqs_1)
                kt1 = T(1//2) * T(sign(sj1))
            else
                col_float1      = T(eqs_1) + sj1
                col1            = clamp(floor(Int, col_float1) + 1, 1, 2 * eqs_1)
                x_diff_right1   = col_float1 - T(col1 - 1)
                continuous_idx1 = x_diff_right1 * T(n_pre[1] - 1) + one(T)
                idx1      = clamp(floor(Int, continuous_idx1), 1, n_pre[1] - 1)
                idx1_next = idx1 + 1
                t1        = continuous_idx1 - T(idx1)
                if SG[1] == :linear
                    kt1 = (one(T) - t1) * itp.kernel_pre[1][idx1, col1] + t1 * itp.kernel_pre[1][idx1_next, col1]
                elseif SG[1] == :cubic
                    kt1 = cubic_hermite(t1,
                        itp.kernel_pre[1][idx1,    col1], itp.kernel_pre[1][idx1_next,    col1],
                        itp.kernel_d1_pre[1][idx1, col1], itp.kernel_d1_pre[1][idx1_next, col1],
                        h_pre[1])
                elseif SG[1] == :quintic
                    kt1 = quintic_hermite(t1,
                        itp.kernel_pre[1][idx1,    col1], itp.kernel_pre[1][idx1_next,    col1],
                        itp.kernel_d1_pre[1][idx1, col1], itp.kernel_d1_pre[1][idx1_next, col1],
                        itp.kernel_d2_pre[1][idx1, col1], itp.kernel_d2_pre[1][idx1_next, col1],
                        h_pre[1])
                end
            end
            result += itp.tail1_right[2][j1, r2] * (kt1 - itp.left_values[1][j1])
        end
        result += l1_ok ? itp.tail2_lr[l1, r2] : zero(T)
        result += r1_ok ? itp.tail2_rr[r1, r2] : zero(T)
    end

    return result * itp.h[1] * itp.h[2]
end