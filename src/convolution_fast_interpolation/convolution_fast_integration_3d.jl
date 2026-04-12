@inline function (itp::FastConvolutionInterpolation{T,3,3,TCoefs,Axs,KA,Val{3},
                    DG,EQ,PR,KP,KBC,FastIntegralOrder,FD,SD,Val{SG},Val{false},Val{3}})(x::Vararg{Number,3}) where
                    {T<:AbstractFloat,TCoefs<:AbstractArray{T,3},
                    Axs<:Tuple{<:AbstractVector,<:AbstractVector,<:AbstractVector},
                    KA<:Tuple{<:Nothing,<:Nothing,<:Nothing},DG,EQ<:Tuple{Int,Int,Int},
                    PR<:Tuple{<:AbstractVector,<:AbstractVector,<:AbstractVector},
                    KP,KBC<:Tuple{<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol}},
                    FD,SD,SG}

    n_pre_d = ntuple(i -> length(itp.pre_range[i]), 3)
    h_pre_d = ntuple(i -> one(T) / T(n_pre_d[i] - 1), 3)

    result = zero(T)

    i1_float = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T)
    i1       = clamp(floor(Int, i1_float), itp.eqs[1], size(itp.coefs, 1) - itp.eqs[1])
    i2_float = (x[2] - itp.knots[2][1]) / itp.h[2] + one(T)
    i2       = clamp(floor(Int, i2_float), itp.eqs[2], size(itp.coefs, 2) - itp.eqs[2])
    i3_float = (x[3] - itp.knots[3][1]) / itp.h[3] + one(T)
    i3       = clamp(floor(Int, i3_float), itp.eqs[3], size(itp.coefs, 3) - itp.eqs[3])

    l1 = i1 - itp.eqs[1];      l1_ok = l1 >= 1
    r1 = i1 + itp.eqs[1] + 1;  r1_ok = r1 <= size(itp.coefs, 1)
    l2 = i2 - itp.eqs[2];      l2_ok = l2 >= 1
    r2 = i2 + itp.eqs[2] + 1;  r2_ok = r2 <= size(itp.coefs, 2)
    l3 = i3 - itp.eqs[3];      l3_ok = l3 >= 1
    r3 = i3 + itp.eqs[3] + 1;  r3_ok = r3 <= size(itp.coefs, 3)

    # ── center: K̃×K̃×K̃ + face contributions from tail1 ──────────────────────
    @inbounds for j3 in (i3 - itp.eqs[3] + 1):(i3 + itp.eqs[3])
        lv3 = eval_sg_kt(itp, j3, Val(3), Val(SG), h_pre_d[3], x) - itp.left_values[3][j3]

        @inbounds for j2 in (i2 - itp.eqs[2] + 1):(i2 + itp.eqs[2])
            lv2 = eval_sg_kt(itp, j2, Val(2), Val(SG), h_pre_d[2], x) - itp.left_values[2][j2]
            lv23 = lv2 * lv3

            # center: K̃₁ × K̃₂ × K̃₃
            @inbounds for j1 in (i1 - itp.eqs[1] + 1):(i1 + itp.eqs[1])
                result += itp.coefs[j1, j2, j3] *
                    (eval_sg_kt(itp, j1, Val(1), Val(SG), h_pre_d[1], x) - itp.left_values[1][j1]) * lv23
            end

            # face: tail1[1] × K̃₂ × K̃₃  (saturated in dim 1)
            result += (l1_ok ? itp.tail1_left[1][l1,  j2, j3] : zero(T)) * lv23
            result += (r1_ok ? itp.tail1_right[1][r1, j2, j3] : zero(T)) * lv23
        end

        # face: K̃₁ × tail1[2] × K̃₃  (saturated in dim 2)
        @inbounds for j1 in (i1 - itp.eqs[1] + 1):(i1 + itp.eqs[1])
            lv1 = eval_sg_kt(itp, j1, Val(1), Val(SG), h_pre_d[1], x) - itp.left_values[1][j1]
            result += (l2_ok ? itp.tail1_left[2][j1,  l2, j3] : zero(T)) * lv1 * lv3
            result += (r2_ok ? itp.tail1_right[2][j1, r2, j3] : zero(T)) * lv1 * lv3
        end

        # edge: tail3_edge[3] × K̃₃  (saturated in dims 1&2, free in dim 3)
        result += (l1_ok && l2_ok ? itp.tail3_edge_ll[3][l1, l2, j3] : zero(T)) * lv3
        result += (r1_ok && l2_ok ? itp.tail3_edge_rl[3][r1, l2, j3] : zero(T)) * lv3
        result += (l1_ok && r2_ok ? itp.tail3_edge_lr[3][l1, r2, j3] : zero(T)) * lv3
        result += (r1_ok && r2_ok ? itp.tail3_edge_rr[3][r1, r2, j3] : zero(T)) * lv3
    end

    # face: K̃₁ × K̃₂ × tail1[3]  (saturated in dim 3)
    @inbounds for j2 in (i2 - itp.eqs[2] + 1):(i2 + itp.eqs[2])
        lv2 = eval_sg_kt(itp, j2, Val(2), Val(SG), h_pre_d[2], x) - itp.left_values[2][j2]
        @inbounds for j1 in (i1 - itp.eqs[1] + 1):(i1 + itp.eqs[1])
            lv1 = eval_sg_kt(itp, j1, Val(1), Val(SG), h_pre_d[1], x) - itp.left_values[1][j1]
            result += (l3_ok ? itp.tail1_left[3][j1,  j2, l3] : zero(T)) * lv1 * lv2
            result += (r3_ok ? itp.tail1_right[3][j1, j2, r3] : zero(T)) * lv1 * lv2
        end
    end

    # edge: tail3_edge[2] × K̃₂  (saturated in dims 1&3, free in dim 2)
    @inbounds for j2 in (i2 - itp.eqs[2] + 1):(i2 + itp.eqs[2])
        lv2 = eval_sg_kt(itp, j2, Val(2), Val(SG), h_pre_d[2], x) - itp.left_values[2][j2]
        result += (l1_ok && l3_ok ? itp.tail3_edge_ll[2][l1, j2, l3] : zero(T)) * lv2
        result += (r1_ok && l3_ok ? itp.tail3_edge_rl[2][r1, j2, l3] : zero(T)) * lv2
        result += (l1_ok && r3_ok ? itp.tail3_edge_lr[2][l1, j2, r3] : zero(T)) * lv2
        result += (r1_ok && r3_ok ? itp.tail3_edge_rr[2][r1, j2, r3] : zero(T)) * lv2
    end

    # edge: tail3_edge[1] × K̃₁  (saturated in dims 2&3, free in dim 1)
    @inbounds for j1 in (i1 - itp.eqs[1] + 1):(i1 + itp.eqs[1])
        lv1 = eval_sg_kt(itp, j1, Val(1), Val(SG), h_pre_d[1], x) - itp.left_values[1][j1]
        result += (l2_ok && l3_ok ? itp.tail3_edge_ll[1][j1, l2, l3] : zero(T)) * lv1
        result += (r2_ok && l3_ok ? itp.tail3_edge_rl[1][j1, r2, l3] : zero(T)) * lv1
        result += (l2_ok && r3_ok ? itp.tail3_edge_lr[1][j1, l2, r3] : zero(T)) * lv1
        result += (r2_ok && r3_ok ? itp.tail3_edge_rr[1][j1, r2, r3] : zero(T)) * lv1
    end

    # ── corners: tail×tail×tail ─────────────────────────────────────────────────
    result += (l1_ok && l2_ok && l3_ok ? itp.tail3_corner_lll[l1, l2, l3] : zero(T))
    result += (r1_ok && l2_ok && l3_ok ? itp.tail3_corner_rll[r1, l2, l3] : zero(T))
    result += (l1_ok && r2_ok && l3_ok ? itp.tail3_corner_lrl[l1, r2, l3] : zero(T))
    result += (l1_ok && l2_ok && r3_ok ? itp.tail3_corner_llr[l1, l2, r3] : zero(T))
    result += (r1_ok && r2_ok && l3_ok ? itp.tail3_corner_rrl[r1, r2, l3] : zero(T))
    result += (r1_ok && l2_ok && r3_ok ? itp.tail3_corner_rlr[r1, l2, r3] : zero(T))
    result += (l1_ok && r2_ok && r3_ok ? itp.tail3_corner_lrr[l1, r2, r3] : zero(T))
    result += (r1_ok && r2_ok && r3_ok ? itp.tail3_corner_rrr[r1, r2, r3] : zero(T))

    return result * itp.h[1] * itp.h[2] * itp.h[3]
end