# """
# (itp::FastConvolutionInterpolation{T,N,...})(x...) — FastMixedIntegralOrder, n_integral=2
# Evaluate mixed antiderivative/derivative/interpolation in N dimensions,
# with exactly 2 integral dimensions using fast tail lookups.

# Integral dimensions: K̃×K̃ center loop from exact column polynomials + strip lookups (tail1)
# + corner lookups (tail2).
# Derivative dimensions: kernel weights from exact column polynomials.

# Cost: O(eqs^N) center + O(eqs^(N-1)) strips + O(eqs^(N-2)) corners.
# """
@inline function (itp::FastConvolutionInterpolation{T,N,2,TCoefs,Axs,KA,DIM,DG,EQ,KBC,
            FastMixedIntegralOrder{DO},FD,SD,Val{SG},Val{false},Val{2}})(x::Vararg{Number,N}) where
            {T<:AbstractFloat,N,TCoefs<:AbstractArray{T,N},Axs<:Tuple{Vararg{AbstractVector}},
            KA<:Tuple{Vararg{Nothing}},DIM,DG,EQ<:Tuple{Vararg{Int}},KBC<:Tuple{Vararg{Tuple{Symbol,Symbol}}},DO,FD,SD,SG}

    x = T.(x)
    # find the two integral dimensions (compile-time constants)
    int_dim1, int_dim2 = _integral_dims2(Val(DO))

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

    result = zero(T)

    # ── main loop over derivative dimension stencils ─────────────────────
    deriv_ranges = ntuple(d -> (d == int_dim1 || d == int_dim2) ? (1:1) :
                               ((i[d] - itp.eqs[d] + 1):(i[d] + itp.eqs[d])), N)

    @inbounds for idx_d in Iterators.product(deriv_ranges...)

        # product of the kernel weights of the non-integral dimensions
        kt_prod = _derivative_weight_product(w, idx_d, i, itp.eqs, Val(DO))

        # ── center: K̃×K̃ + strips ─────────────────────────────────────────
        @inbounds for j2 in (i[int_dim2] - itp.eqs[int_dim2] + 1):(i[int_dim2] + itp.eqs[int_dim2])
            lv2    = _antiderivative_weight(w[int_dim2], i[int_dim2], itp.eqs[int_dim2], j2) -
                     itp.left_values[int_dim2][j2]
            idx_d2 = Base.setindex(idx_d, j2, int_dim2)

            # center: K̃(int_dim1) × K̃(int_dim2)
            @inbounds for j1 in (i[int_dim1] - itp.eqs[int_dim1] + 1):(i[int_dim1] + itp.eqs[int_dim1])
                lv1      = _antiderivative_weight(w[int_dim1], i[int_dim1], itp.eqs[int_dim1], j1) -
                           itp.left_values[int_dim1][j1]
                coef_idx = Base.setindex(idx_d2, j1, int_dim1)
                result  += itp.coefs[coef_idx...] * lv1 * lv2 * kt_prod
            end

            # strip: tail1[int_dim1] × K̃(int_dim2)
            idx_l1 = Base.setindex(idx_d2, l[int_dim1], int_dim1)
            idx_r1 = Base.setindex(idx_d2, r[int_dim1], int_dim1)
            tl1 = l_ok[int_dim1] ? itp.tail1_left[int_dim1][idx_l1...]  : zero(T)
            tr1 = r_ok[int_dim1] ? itp.tail1_right[int_dim1][idx_r1...] : zero(T)
            result += (tl1 + tr1) * lv2 * kt_prod
        end

        # strip: K̃(int_dim1) × tail1[int_dim2]
        @inbounds for j1 in (i[int_dim1] - itp.eqs[int_dim1] + 1):(i[int_dim1] + itp.eqs[int_dim1])
            lv1    = _antiderivative_weight(w[int_dim1], i[int_dim1], itp.eqs[int_dim1], j1) -
                     itp.left_values[int_dim1][j1]
            idx_d1 = Base.setindex(idx_d, j1, int_dim1)
            idx_l2 = Base.setindex(idx_d1, l[int_dim2], int_dim2)
            idx_r2 = Base.setindex(idx_d1, r[int_dim2], int_dim2)
            tl2 = l_ok[int_dim2] ? itp.tail1_left[int_dim2][idx_l2...]  : zero(T)
            tr2 = r_ok[int_dim2] ? itp.tail1_right[int_dim2][idx_r2...] : zero(T)
            result += (tl2 + tr2) * lv1 * kt_prod
        end

        # ── corners: tail2 lookups ────────────────────────────────────────
        idx_l1l2 = Base.setindex(Base.setindex(idx_d, l[int_dim1], int_dim1), l[int_dim2], int_dim2)
        idx_r1l2 = Base.setindex(Base.setindex(idx_d, r[int_dim1], int_dim1), l[int_dim2], int_dim2)
        idx_l1r2 = Base.setindex(Base.setindex(idx_d, l[int_dim1], int_dim1), r[int_dim2], int_dim2)
        idx_r1r2 = Base.setindex(Base.setindex(idx_d, r[int_dim1], int_dim1), r[int_dim2], int_dim2)

        c_ll = (l_ok[int_dim1] && l_ok[int_dim2]) ? itp.tail2_ll[idx_l1l2...] : zero(T)
        c_rl = (r_ok[int_dim1] && l_ok[int_dim2]) ? itp.tail2_rl[idx_r1l2...] : zero(T)
        c_lr = (l_ok[int_dim1] && r_ok[int_dim2]) ? itp.tail2_lr[idx_l1r2...] : zero(T)
        c_rr = (r_ok[int_dim1] && r_ok[int_dim2]) ? itp.tail2_rr[idx_r1r2...] : zero(T)
        result += (c_ll + c_rl + c_lr + c_rr) * kt_prod
    end

    # ── scaling ──────────────────────────────────────────────────────────
    scale = one(T)
    @inbounds for d in 1:N
        if (d == int_dim1 || d == int_dim2)
            scale *= itp.h[d]
        else
            scale *= (-one(T) / itp.h[d])^DO[d]
        end
    end

    return result * scale
end

# Indices of the two integral dimensions (DO[d] == −1), as fixed numbers at compile time
_integral_dims2(::Val{DO}) where {DO} = _integral_dims(Val(DO))