# """
# (itp::FastConvolutionInterpolation{T,N,...})(x...) — FastMixedIntegralOrder, n_integral=2
# Evaluate mixed antiderivative/derivative/interpolation in N dimensions,
# with exactly 2 integral dimensions using fast tail lookups.

# Integral dimensions: K̃×K̃ center loop + strip lookups (tail1) + corner lookups (tail2).
# Derivative dimensions: linear subgrid kernel evaluation.

# Cost: O(eqs^N) center + O(eqs^(N-1)) strips + O(eqs^(N-2)) corners.
# """
@inline function (itp::FastConvolutionInterpolation{T,N,2,TCoefs,Axs,KA,DIM,DG,EQ,PR,KP,KBC,
            FastMixedIntegralOrder{DO},FD,SD,Val{SG},Val{false},Val{2}})(x::Vararg{Number,N}) where
            {T<:AbstractFloat,N,TCoefs<:AbstractArray{T,N},Axs<:Tuple{Vararg{AbstractVector}},
            KA<:Tuple{Vararg{Nothing}},DIM,DG,EQ<:Tuple{Vararg{Int}},PR<:Tuple{Vararg{AbstractVector}},
            KP,KBC<:Tuple{Vararg{Tuple{Symbol,Symbol}}},DO,FD,SD,SG}

    # find the two integral dimensions (compile-time constant)
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

    # ── subgrid fractional positions for derivative dimensions (linear) ──
    n_pre = ntuple(d -> length(itp.pre_range[d]), N)

    t_sg = ntuple(N) do d
        (d == int_dim1 || d == int_dim2) && return zero(T)
        i_float = (x[d] - itp.knots[d][1]) / itp.h[d] + one(T)
        pos = clamp(floor(Int, i_float), itp.eqs[d], length(itp.knots[d]) - itp.eqs[d])
        diff_right = one(T) - (x[d] - itp.knots[d][pos]) / itp.h[d]
        continuous_idx = diff_right * T(n_pre[d] - 1) + one(T)
        continuous_idx - T(clamp(floor(Int, continuous_idx), 1, n_pre[d] - 1))
    end

    idx_sg = ntuple(N) do d
        (d == int_dim1 || d == int_dim2) && return 0
        i_float = (x[d] - itp.knots[d][1]) / itp.h[d] + one(T)
        pos = clamp(floor(Int, i_float), itp.eqs[d], length(itp.knots[d]) - itp.eqs[d])
        diff_right = one(T) - (x[d] - itp.knots[d][pos]) / itp.h[d]
        continuous_idx = diff_right * T(n_pre[d] - 1) + one(T)
        clamp(floor(Int, continuous_idx), 1, n_pre[d] - 1)
    end

    # ── K̃ lookup for integral dimensions (with subgrid) ──────────────────
    @inline function eval_kt_int(d, jd)
        eqs_d   = itp.eqs[d]
        n_pre_d = n_pre[d]
        h_pre_d = one(T) / T(n_pre_d - 1)
        xjd     = itp.knots[d][eqs_d] + (jd - eqs_d) * itp.h[d]
        sj      = (x[d] - xjd) / itp.h[d]
        abs(sj) >= T(eqs_d) && return T(1//2) * T(sign(sj))
        col_float = T(eqs_d) + sj
        col       = clamp(floor(Int, col_float) + 1, 1, 2 * eqs_d)
        t         = (col_float - T(col - 1)) * T(n_pre_d - 1) + one(T)
        idx       = clamp(floor(Int, t), 1, n_pre_d - 1)
        t        -= T(idx)
        if SG[d] == :quintic
            quintic_hermite(t,
                itp.kernel_pre[d][idx,    col], itp.kernel_pre[d][idx+1,    col],
                itp.kernel_d1_pre[d][idx, col], itp.kernel_d1_pre[d][idx+1, col],
                itp.kernel_d2_pre[d][idx, col], itp.kernel_d2_pre[d][idx+1, col],
                h_pre_d)
        elseif SG[d] == :cubic
            cubic_hermite(t,
                itp.kernel_pre[d][idx,    col], itp.kernel_pre[d][idx+1,    col],
                itp.kernel_d1_pre[d][idx, col], itp.kernel_d1_pre[d][idx+1, col],
                h_pre_d)
        else
            (one(T) - t) * itp.kernel_pre[d][idx, col] + t * itp.kernel_pre[d][idx+1, col]
        end
    end

    result = zero(T)

    # ── main loop over derivative dimension stencils ─────────────────────
    deriv_ranges = ntuple(d -> (d == int_dim1 || d == int_dim2) ? (1:1) :
                               ((i[d] - itp.eqs[d] + 1):(i[d] + itp.eqs[d])), N)

    @inbounds for idx_d in Iterators.product(deriv_ranges...)

        # product of derivative kernel values (linear subgrid)
        kt_prod = one(T)
        skip    = false
        @inbounds for d in 1:N
            (d == int_dim1 || d == int_dim2) && continue
            pos = clamp(floor(Int, (x[d] - itp.knots[d][1]) / itp.h[d] + one(T)),
                        itp.eqs[d], length(itp.knots[d]) - itp.eqs[d])
            col = idx_d[d] - pos + itp.eqs[d]
            if col < 1 || col > 2 * itp.eqs[d]
                skip = true
                break
            end
            k0 = itp.kernel_pre[d][idx_sg[d],   col]
            k1 = itp.kernel_pre[d][idx_sg[d]+1, col]
            kt_prod *= (one(T) - t_sg[d]) * k0 + t_sg[d] * k1
        end
        skip && continue

        # ── center: K̃×K̃ + strips ─────────────────────────────────────────
        @inbounds for j2 in (i[int_dim2] - itp.eqs[int_dim2] + 1):(i[int_dim2] + itp.eqs[int_dim2])
            lv2    = eval_kt_int(int_dim2, j2) - itp.left_values[int_dim2][j2]
            idx_d2 = Base.setindex(idx_d, j2, int_dim2)

            # center: K̃(int_dim1) × K̃(int_dim2)
            @inbounds for j1 in (i[int_dim1] - itp.eqs[int_dim1] + 1):(i[int_dim1] + itp.eqs[int_dim1])
                lv1      = eval_kt_int(int_dim1, j1) - itp.left_values[int_dim1][j1]
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
            lv1    = eval_kt_int(int_dim1, j1) - itp.left_values[int_dim1][j1]
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

@inline function _integral_dims2(::Val{DO}) where {DO}
    d1 = findfirst(d -> DO[d] == -1, 1:length(DO))::Int
    d2 = findnext(d -> DO[d] == -1, 1:length(DO), d1+1)::Int
    return d1, d2
end