"""
(itp::FastConvolutionInterpolation{T,N,...})(x...) — FastMixedIntegralOrder, n_integral=3
Evaluate mixed antiderivative/derivative/interpolation in N dimensions,
with exactly 3 integral dimensions using fast tail lookups.

Integral dimensions: center + left faces + left edges + left corner, with K̃ weights from
exact column polynomials. Coefficients right of the stencil in an integral dimension
contribute nothing (their anchored weight is exactly zero).
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

    # left tail boundary indices
    l    = ntuple(d -> i[d] - itp.eqs[d], N)
    l_ok = ntuple(d -> l[d] >= 1,          N)

    # ── kernel weights of every dimension (exact column polynomials) ─────
    w = _mixed_weights(itp, x, i, Val(DO))
    # K̃ weights of the three integral dimensions
    w1 = w[int_dim1]
    w2 = w[int_dim2]
    w3 = w[int_dim3]

    # stencil ranges of the three integral dimensions
    range1 = (i[int_dim1] - itp.eqs[int_dim1] + 1):(i[int_dim1] + itp.eqs[int_dim1])
    range2 = (i[int_dim2] - itp.eqs[int_dim2] + 1):(i[int_dim2] + itp.eqs[int_dim2])
    range3 = (i[int_dim3] - itp.eqs[int_dim3] + 1):(i[int_dim3] + itp.eqs[int_dim3])

    result = zero(T)

    # ── main loop over derivative dimension stencils ─────────────────────
    deriv_ranges = ntuple(d -> (d == int_dim1 || d == int_dim2 || d == int_dim3) ? (1:1) :
                               ((i[d] - itp.eqs[d] + 1):(i[d] + itp.eqs[d])), N)

    @inbounds for idx_d in Iterators.product(deriv_ranges...)

        # product of the kernel weights of the non-integral dimensions
        kt_prod = _derivative_weight_product(w, idx_d, i, itp.eqs, Val(DO))

        # ── center, left face in int_dim1, left face in int_dim2, left edge along int_dim3 ──
        @inbounds for j3 in range3
            lv3    = _antiderivative_weight(w3, i[int_dim3], itp.eqs[int_dim3], j3) - itp.left_values[int_dim3][j3]
            idx_d3 = Base.setindex(idx_d, j3, int_dim3)

            @inbounds for j2 in range2
                lv2     = _antiderivative_weight(w2, i[int_dim2], itp.eqs[int_dim2], j2) - itp.left_values[int_dim2][j2]
                lv23    = lv2 * lv3
                idx_d23 = Base.setindex(idx_d3, j2, int_dim2)

                # center: K̃₁ × K̃₂ × K̃₃
                @inbounds for j1 in range1
                    lv1      = _antiderivative_weight(w1, i[int_dim1], itp.eqs[int_dim1], j1) - itp.left_values[int_dim1][j1]
                    coef_idx = Base.setindex(idx_d23, j1, int_dim1)
                    result  += itp.coefs[coef_idx...] * lv1 * lv23 * kt_prod
                end

                # left face: tail1[int_dim1] × K̃₂ × K̃₃
                if l_ok[int_dim1]
                    idx_l1 = Base.setindex(idx_d23, l[int_dim1], int_dim1)
                    result += itp.tail1_left[int_dim1][idx_l1...] * lv23 * kt_prod
                end
            end

            if l_ok[int_dim2]
                # left face: K̃₁ × tail1[int_dim2] × K̃₃
                @inbounds for j1 in range1
                    lv1    = _antiderivative_weight(w1, i[int_dim1], itp.eqs[int_dim1], j1) - itp.left_values[int_dim1][j1]
                    idx_l2 = Base.setindex(Base.setindex(idx_d3, j1, int_dim1), l[int_dim2], int_dim2)
                    result += itp.tail1_left[int_dim2][idx_l2...] * lv1 * lv3 * kt_prod
                end

                # left edge along int_dim3: saturated in int_dim1 and int_dim2
                if l_ok[int_dim1]
                    idx_l1l2 = Base.setindex(Base.setindex(idx_d3, l[int_dim1], int_dim1), l[int_dim2], int_dim2)
                    result += itp.tail3_edge_ll[3][idx_l1l2...] * lv3 * kt_prod
                end
            end
        end

        if l_ok[int_dim3]
            # left face: K̃₁ × K̃₂ × tail1[int_dim3]
            @inbounds for j2 in range2
                lv2 = _antiderivative_weight(w2, i[int_dim2], itp.eqs[int_dim2], j2) - itp.left_values[int_dim2][j2]
                @inbounds for j1 in range1
                    lv1    = _antiderivative_weight(w1, i[int_dim1], itp.eqs[int_dim1], j1) - itp.left_values[int_dim1][j1]
                    idx_l3 = Base.setindex(Base.setindex(Base.setindex(idx_d, j1, int_dim1), j2, int_dim2), l[int_dim3], int_dim3)
                    result += itp.tail1_left[int_dim3][idx_l3...] * lv1 * lv2 * kt_prod
                end
            end

            # left edge along int_dim2: saturated in int_dim1 and int_dim3
            if l_ok[int_dim1]
                @inbounds for j2 in range2
                    lv2 = _antiderivative_weight(w2, i[int_dim2], itp.eqs[int_dim2], j2) - itp.left_values[int_dim2][j2]
                    idx_l1l3 = Base.setindex(Base.setindex(Base.setindex(idx_d, j2, int_dim2), l[int_dim1], int_dim1), l[int_dim3], int_dim3)
                    result += itp.tail3_edge_ll[2][idx_l1l3...] * lv2 * kt_prod
                end
            end

            if l_ok[int_dim2]
                # left edge along int_dim1: saturated in int_dim2 and int_dim3
                @inbounds for j1 in range1
                    lv1 = _antiderivative_weight(w1, i[int_dim1], itp.eqs[int_dim1], j1) - itp.left_values[int_dim1][j1]
                    idx_l2l3 = Base.setindex(Base.setindex(Base.setindex(idx_d, j1, int_dim1), l[int_dim2], int_dim2), l[int_dim3], int_dim3)
                    result += itp.tail3_edge_ll[1][idx_l2l3...] * lv1 * kt_prod
                end

                # ── left corner ───────────────────────────────────────────
                if l_ok[int_dim1]
                    idx_lll = Base.setindex(Base.setindex(Base.setindex(idx_d, l[int_dim1], int_dim1), l[int_dim2], int_dim2), l[int_dim3], int_dim3)
                    result += itp.tail3_corner_lll[idx_lll...] * kt_prod
                end
            end
        end
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