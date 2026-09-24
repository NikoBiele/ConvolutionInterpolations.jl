"""
(itp::FastConvolutionInterpolation{T,N,...})(x...) — FastMixedIntegralOrder, n_integral=3
Evaluate mixed antiderivative/derivative/interpolation in N dimensions,
with exactly 3 integral dimensions using fast tail lookups.

Integral dimensions: full 27-region decomposition (center + faces + edges + corners),
with K̃ weights from exact column polynomials.
Derivative dimensions: kernel weights from exact column polynomials.
"""
@inline function (itp::FastConvolutionInterpolation{T,N,3,TCoefs,Axs,KA,DIM,DG,EQ,KBC,
            FastMixedIntegralOrder{DO},FD,SD,Val{SG},Val{false},Val{3}})(x::Vararg{Number,N}) where
            {T<:AbstractFloat,N,TCoefs<:AbstractArray{T,N},Axs<:Tuple{Vararg{AbstractVector}},
            KA<:Tuple{Vararg{Nothing}},DIM,DG,EQ<:Tuple{Vararg{Int}},KBC<:Tuple{Vararg{Tuple{Symbol,Symbol}}},DO,FD,SD,SG}

    x = T.(x)
    # find the three integral dimensions (compile-time constants)
    int_dim1, int_dim2, int_dim3 = _integral_dims3(Val(DO))

    # ── per-dimension grid positions ─────────────────────────────────────
    i = ntuple(N) do d
        i_float = (x[d] - itp.knots[d][1]) / itp.h[d] + one(T)
        clamp(floor(Int, i_float), itp.eqs[d], size(itp.coefs, d) - itp.eqs[d])
    end

    # tail boundary indices
    l    = ntuple(d -> i[d] - itp.eqs[d],         N)
    r    = ntuple(d -> i[d] + itp.eqs[d] + 1,     N)
    l_ok = ntuple(d -> l[d] >= 1,                  N)
    r_ok = ntuple(d -> r[d] <= size(itp.coefs, d), N)

    # ── kernel weights of every dimension (exact column polynomials) ─────
    w = _mixed_weights(itp, x, i, Val(DO))
    # K̃ weights of the three integral dimensions
    w1 = w[int_dim1]
    w2 = w[int_dim2]
    w3 = w[int_dim3]

    result = zero(T)

    # ── main loop over derivative dimension stencils ─────────────────────
    deriv_ranges = ntuple(d -> (d == int_dim1 || d == int_dim2 || d == int_dim3) ? (1:1) :
                               ((i[d] - itp.eqs[d] + 1):(i[d] + itp.eqs[d])), N)

    @inbounds for idx_d in Iterators.product(deriv_ranges...)

        # product of the kernel weights of the non-integral dimensions
        kt_prod = _derivative_weight_product(w, idx_d, i, itp.eqs, Val(DO))

        # ── center + faces + edges + corners ─────────────────────────────
        @inbounds for j3 in (i[int_dim3] - itp.eqs[int_dim3] + 1):(i[int_dim3] + itp.eqs[int_dim3])
            lv3    = _antiderivative_weight(w3, i[int_dim3], itp.eqs[int_dim3], j3) - itp.left_values[int_dim3][j3]
            idx_d3 = Base.setindex(idx_d, j3, int_dim3)

            @inbounds for j2 in (i[int_dim2] - itp.eqs[int_dim2] + 1):(i[int_dim2] + itp.eqs[int_dim2])
                lv2    = _antiderivative_weight(w2, i[int_dim2], itp.eqs[int_dim2], j2) - itp.left_values[int_dim2][j2]
                lv23   = lv2 * lv3
                idx_d23 = Base.setindex(idx_d3, j2, int_dim2)

                # center: K̃₁ × K̃₂ × K̃₃
                @inbounds for j1 in (i[int_dim1] - itp.eqs[int_dim1] + 1):(i[int_dim1] + itp.eqs[int_dim1])
                    lv1      = _antiderivative_weight(w1, i[int_dim1], itp.eqs[int_dim1], j1) - itp.left_values[int_dim1][j1]
                    coef_idx = Base.setindex(idx_d23, j1, int_dim1)
                    result  += itp.coefs[coef_idx...] * lv1 * lv23 * kt_prod
                end

                # face: tail1[int_dim1] × K̃₂ × K̃₃
                idx_l1 = Base.setindex(idx_d23, l[int_dim1], int_dim1)
                idx_r1 = Base.setindex(idx_d23, r[int_dim1], int_dim1)
                tl1 = l_ok[int_dim1] ? itp.tail1_left[int_dim1][idx_l1...]  : zero(T)
                tr1 = r_ok[int_dim1] ? itp.tail1_right[int_dim1][idx_r1...] : zero(T)
                result += (tl1 + tr1) * lv23 * kt_prod
            end

            # face: K̃₁ × tail1[int_dim2] × K̃₃
            @inbounds for j1 in (i[int_dim1] - itp.eqs[int_dim1] + 1):(i[int_dim1] + itp.eqs[int_dim1])
                lv1     = _antiderivative_weight(w1, i[int_dim1], itp.eqs[int_dim1], j1) - itp.left_values[int_dim1][j1]
                idx_d13 = Base.setindex(idx_d3, j1, int_dim1)
                idx_l2  = Base.setindex(idx_d13, l[int_dim2], int_dim2)
                idx_r2  = Base.setindex(idx_d13, r[int_dim2], int_dim2)
                tl2 = l_ok[int_dim2] ? itp.tail1_left[int_dim2][idx_l2...]  : zero(T)
                tr2 = r_ok[int_dim2] ? itp.tail1_right[int_dim2][idx_r2...] : zero(T)
                result += (tl2 + tr2) * lv1 * lv3 * kt_prod
            end

            # edge: tail3_edge[3] × K̃₃ (saturated in int_dim1 & int_dim2, free in int_dim3)
            idx_l1l2 = Base.setindex(Base.setindex(idx_d3, l[int_dim1], int_dim1), l[int_dim2], int_dim2)
            idx_r1l2 = Base.setindex(Base.setindex(idx_d3, r[int_dim1], int_dim1), l[int_dim2], int_dim2)
            idx_l1r2 = Base.setindex(Base.setindex(idx_d3, l[int_dim1], int_dim1), r[int_dim2], int_dim2)
            idx_r1r2 = Base.setindex(Base.setindex(idx_d3, r[int_dim1], int_dim1), r[int_dim2], int_dim2)
            result += (l_ok[int_dim1] && l_ok[int_dim2] ? itp.tail3_edge_ll[3][idx_l1l2...] : zero(T)) * lv3 * kt_prod
            result += (r_ok[int_dim1] && l_ok[int_dim2] ? itp.tail3_edge_rl[3][idx_r1l2...] : zero(T)) * lv3 * kt_prod
            result += (l_ok[int_dim1] && r_ok[int_dim2] ? itp.tail3_edge_lr[3][idx_l1r2...] : zero(T)) * lv3 * kt_prod
            result += (r_ok[int_dim1] && r_ok[int_dim2] ? itp.tail3_edge_rr[3][idx_r1r2...] : zero(T)) * lv3 * kt_prod
        end

        # face: K̃₁ × K̃₂ × tail1[int_dim3]
        @inbounds for j2 in (i[int_dim2] - itp.eqs[int_dim2] + 1):(i[int_dim2] + itp.eqs[int_dim2])
            lv2 = _antiderivative_weight(w2, i[int_dim2], itp.eqs[int_dim2], j2) - itp.left_values[int_dim2][j2]
            @inbounds for j1 in (i[int_dim1] - itp.eqs[int_dim1] + 1):(i[int_dim1] + itp.eqs[int_dim1])
                lv1     = _antiderivative_weight(w1, i[int_dim1], itp.eqs[int_dim1], j1) - itp.left_values[int_dim1][j1]
                idx_d12 = Base.setindex(Base.setindex(idx_d, j1, int_dim1), j2, int_dim2)
                idx_l3  = Base.setindex(idx_d12, l[int_dim3], int_dim3)
                idx_r3  = Base.setindex(idx_d12, r[int_dim3], int_dim3)
                tl3 = l_ok[int_dim3] ? itp.tail1_left[int_dim3][idx_l3...]  : zero(T)
                tr3 = r_ok[int_dim3] ? itp.tail1_right[int_dim3][idx_r3...] : zero(T)
                result += (tl3 + tr3) * lv1 * lv2 * kt_prod
            end
        end

        # edge: tail3_edge[2] × K̃₂ (saturated in int_dim1 & int_dim3, free in int_dim2)
        @inbounds for j2 in (i[int_dim2] - itp.eqs[int_dim2] + 1):(i[int_dim2] + itp.eqs[int_dim2])
            lv2 = _antiderivative_weight(w2, i[int_dim2], itp.eqs[int_dim2], j2) - itp.left_values[int_dim2][j2]
            idx_d2 = Base.setindex(idx_d, j2, int_dim2)
            idx_l1l3 = Base.setindex(Base.setindex(idx_d2, l[int_dim1], int_dim1), l[int_dim3], int_dim3)
            idx_r1l3 = Base.setindex(Base.setindex(idx_d2, r[int_dim1], int_dim1), l[int_dim3], int_dim3)
            idx_l1r3 = Base.setindex(Base.setindex(idx_d2, l[int_dim1], int_dim1), r[int_dim3], int_dim3)
            idx_r1r3 = Base.setindex(Base.setindex(idx_d2, r[int_dim1], int_dim1), r[int_dim3], int_dim3)
            result += (l_ok[int_dim1] && l_ok[int_dim3] ? itp.tail3_edge_ll[2][idx_l1l3...] : zero(T)) * lv2 * kt_prod
            result += (r_ok[int_dim1] && l_ok[int_dim3] ? itp.tail3_edge_rl[2][idx_r1l3...] : zero(T)) * lv2 * kt_prod
            result += (l_ok[int_dim1] && r_ok[int_dim3] ? itp.tail3_edge_lr[2][idx_l1r3...] : zero(T)) * lv2 * kt_prod
            result += (r_ok[int_dim1] && r_ok[int_dim3] ? itp.tail3_edge_rr[2][idx_r1r3...] : zero(T)) * lv2 * kt_prod
        end

        # edge: tail3_edge[1] × K̃₁ (saturated in int_dim2 & int_dim3, free in int_dim1)
        @inbounds for j1 in (i[int_dim1] - itp.eqs[int_dim1] + 1):(i[int_dim1] + itp.eqs[int_dim1])
            lv1 = _antiderivative_weight(w1, i[int_dim1], itp.eqs[int_dim1], j1) - itp.left_values[int_dim1][j1]
            idx_d1 = Base.setindex(idx_d, j1, int_dim1)
            idx_l2l3 = Base.setindex(Base.setindex(idx_d1, l[int_dim2], int_dim2), l[int_dim3], int_dim3)
            idx_r2l3 = Base.setindex(Base.setindex(idx_d1, r[int_dim2], int_dim2), l[int_dim3], int_dim3)
            idx_l2r3 = Base.setindex(Base.setindex(idx_d1, l[int_dim2], int_dim2), r[int_dim3], int_dim3)
            idx_r2r3 = Base.setindex(Base.setindex(idx_d1, r[int_dim2], int_dim2), r[int_dim3], int_dim3)
            result += (l_ok[int_dim2] && l_ok[int_dim3] ? itp.tail3_edge_ll[1][idx_l2l3...] : zero(T)) * lv1 * kt_prod
            result += (r_ok[int_dim2] && l_ok[int_dim3] ? itp.tail3_edge_rl[1][idx_r2l3...] : zero(T)) * lv1 * kt_prod
            result += (l_ok[int_dim2] && r_ok[int_dim3] ? itp.tail3_edge_lr[1][idx_l2r3...] : zero(T)) * lv1 * kt_prod
            result += (r_ok[int_dim2] && r_ok[int_dim3] ? itp.tail3_edge_rr[1][idx_r2r3...] : zero(T)) * lv1 * kt_prod
        end

        # ── corners ───────────────────────────────────────────────────────
        idx_lll = Base.setindex(Base.setindex(Base.setindex(idx_d, l[int_dim1], int_dim1), l[int_dim2], int_dim2), l[int_dim3], int_dim3)
        idx_rll = Base.setindex(Base.setindex(Base.setindex(idx_d, r[int_dim1], int_dim1), l[int_dim2], int_dim2), l[int_dim3], int_dim3)
        idx_lrl = Base.setindex(Base.setindex(Base.setindex(idx_d, l[int_dim1], int_dim1), r[int_dim2], int_dim2), l[int_dim3], int_dim3)
        idx_llr = Base.setindex(Base.setindex(Base.setindex(idx_d, l[int_dim1], int_dim1), l[int_dim2], int_dim2), r[int_dim3], int_dim3)
        idx_rrl = Base.setindex(Base.setindex(Base.setindex(idx_d, r[int_dim1], int_dim1), r[int_dim2], int_dim2), l[int_dim3], int_dim3)
        idx_rlr = Base.setindex(Base.setindex(Base.setindex(idx_d, r[int_dim1], int_dim1), l[int_dim2], int_dim2), r[int_dim3], int_dim3)
        idx_lrr = Base.setindex(Base.setindex(Base.setindex(idx_d, l[int_dim1], int_dim1), r[int_dim2], int_dim2), r[int_dim3], int_dim3)
        idx_rrr = Base.setindex(Base.setindex(Base.setindex(idx_d, r[int_dim1], int_dim1), r[int_dim2], int_dim2), r[int_dim3], int_dim3)

        result += (l_ok[int_dim1] && l_ok[int_dim2] && l_ok[int_dim3] ? itp.tail3_corner_lll[idx_lll...] : zero(T)) * kt_prod
        result += (r_ok[int_dim1] && l_ok[int_dim2] && l_ok[int_dim3] ? itp.tail3_corner_rll[idx_rll...] : zero(T)) * kt_prod
        result += (l_ok[int_dim1] && r_ok[int_dim2] && l_ok[int_dim3] ? itp.tail3_corner_lrl[idx_lrl...] : zero(T)) * kt_prod
        result += (l_ok[int_dim1] && l_ok[int_dim2] && r_ok[int_dim3] ? itp.tail3_corner_llr[idx_llr...] : zero(T)) * kt_prod
        result += (r_ok[int_dim1] && r_ok[int_dim2] && l_ok[int_dim3] ? itp.tail3_corner_rrl[idx_rrl...] : zero(T)) * kt_prod
        result += (r_ok[int_dim1] && l_ok[int_dim2] && r_ok[int_dim3] ? itp.tail3_corner_rlr[idx_rlr...] : zero(T)) * kt_prod
        result += (l_ok[int_dim1] && r_ok[int_dim2] && r_ok[int_dim3] ? itp.tail3_corner_lrr[idx_lrr...] : zero(T)) * kt_prod
        result += (r_ok[int_dim1] && r_ok[int_dim2] && r_ok[int_dim3] ? itp.tail3_corner_rrr[idx_rrr...] : zero(T)) * kt_prod
    end

    # ── scaling ──────────────────────────────────────────────────────────
    scale = one(T)
    @inbounds for d in 1:N
        if (d == int_dim1 || d == int_dim2 || d == int_dim3)
            scale *= itp.h[d]
        else
            scale *= (-one(T) / itp.h[d])^DO[d]
        end
    end

    return result * scale
end

# Indices of the three integral dimensions (DO[d] == −1), as fixed numbers at compile time
_integral_dims3(::Val{DO}) where {DO} = _integral_dims(Val(DO))