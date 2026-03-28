# """
# (itp::FastConvolutionInterpolation{T,N,...})(x...) — FastMixedIntegralOrder, n_integral=1
# Evaluate mixed antiderivative/derivative/interpolation in N dimensions,
# with exactly 1 integral dimension using fast tail lookups.

# Integral dimension: K̃ center loop with subgrid (linear/cubic/quintic) + O(1) tail1 lookups.
# Derivative dimensions: linear subgrid kernel evaluation.

# Cost: O(eqs^N) center + O(eqs^(N-1)) tails.
# """
@inline function (itp::FastConvolutionInterpolation{T,N,1,TCoefs,Axs,KA,DIM,DG,EQ,PR,KP,KBC,
            FastMixedIntegralOrder{DO},FD,SD,Val{SG},Val{false},Val{1}})(x::Vararg{Number,N}) where
            {T<:AbstractFloat,N,TCoefs<:AbstractArray{T,N},Axs<:Tuple{Vararg{AbstractVector}},
            KA<:Tuple{Vararg{Nothing}},DIM,DG,EQ<:Tuple{Vararg{Int}},PR<:Tuple{Vararg{AbstractVector}},
            KP,KBC<:Tuple{Vararg{Tuple{Symbol,Symbol}}},DO,FD,SD,SG}

    # find the single integral dimension (compile-time constant)
    int_dim = _integral_dim1(Val(DO))

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

    # ── subgrid fractional position for derivative dimensions (linear) ───
    n_pre = ntuple(d -> length(itp.pre_range[d]), N)

    t_sg = ntuple(N) do d
        d == int_dim && return zero(T)
        i_float = (x[d] - itp.knots[d][1]) / itp.h[d] + one(T)
        pos = clamp(floor(Int, i_float), itp.eqs[d], length(itp.knots[d]) - itp.eqs[d])
        diff_right = one(T) - (x[d] - itp.knots[d][pos]) / itp.h[d]
        continuous_idx = diff_right * T(n_pre[d] - 1) + one(T)
        continuous_idx - T(clamp(floor(Int, continuous_idx), 1, n_pre[d] - 1))
    end

    idx_sg = ntuple(N) do d
        d == int_dim && return 0
        i_float = (x[d] - itp.knots[d][1]) / itp.h[d] + one(T)
        pos = clamp(floor(Int, i_float), itp.eqs[d], length(itp.knots[d]) - itp.eqs[d])
        diff_right = one(T) - (x[d] - itp.knots[d][pos]) / itp.h[d]
        continuous_idx = diff_right * T(n_pre[d] - 1) + one(T)
        clamp(floor(Int, continuous_idx), 1, n_pre[d] - 1)
    end

    # ── K̃ lookup for integral dimension (with subgrid) ──────────────────
    @inline function eval_kt_int(jd)
        eqs_d   = itp.eqs[int_dim]
        n_pre_d = n_pre[int_dim]
        h_pre_d = one(T) / T(n_pre_d - 1)
        xjd     = itp.knots[int_dim][eqs_d] + (jd - eqs_d) * itp.h[int_dim]
        sj      = (x[int_dim] - xjd) / itp.h[int_dim]
        abs(sj) >= T(eqs_d) && return T(1//2) * T(sign(sj))
        col_float = T(eqs_d) + sj
        col       = clamp(floor(Int, col_float) + 1, 1, 2 * eqs_d)
        t         = (col_float - T(col - 1)) * T(n_pre_d - 1) + one(T)
        idx       = clamp(floor(Int, t), 1, n_pre_d - 1)
        t        -= T(idx)
        if SG[int_dim] == :quintic
            quintic_hermite(t,
                itp.kernel_pre[int_dim][idx,    col], itp.kernel_pre[int_dim][idx+1,    col],
                itp.kernel_d1_pre[int_dim][idx, col], itp.kernel_d1_pre[int_dim][idx+1, col],
                itp.kernel_d2_pre[int_dim][idx, col], itp.kernel_d2_pre[int_dim][idx+1, col],
                h_pre_d)
        elseif SG[int_dim] == :cubic
            cubic_hermite(t,
                itp.kernel_pre[int_dim][idx,    col], itp.kernel_pre[int_dim][idx+1,    col],
                itp.kernel_d1_pre[int_dim][idx, col], itp.kernel_d1_pre[int_dim][idx+1, col],
                h_pre_d)
        else
            (one(T) - t) * itp.kernel_pre[int_dim][idx, col] + t * itp.kernel_pre[int_dim][idx+1, col]
        end
    end

    result = zero(T)

    # ── main loop over derivative dimension stencils ─────────────────────
    deriv_ranges = ntuple(d -> d == int_dim ? (1:1) :
                               ((i[d] - itp.eqs[d] + 1):(i[d] + itp.eqs[d])), N)

    @inbounds for idx_d in Iterators.product(deriv_ranges...)

        # product of derivative kernel values (linear subgrid)
        kt_prod = one(T)
        skip    = false
        @inbounds for d in 1:N
            d == int_dim && continue
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

        # ── center: K̃ loop over integral dimension stencil ──────────────
        @inbounds for j_int in (i[int_dim] - itp.eqs[int_dim] + 1):(i[int_dim] + itp.eqs[int_dim])
            lv       = eval_kt_int(j_int) - itp.left_values[int_dim][j_int]
            coef_idx = Base.setindex(idx_d, j_int, int_dim)
            result  += itp.coefs[coef_idx...] * lv * kt_prod
        end

        # ── tails: O(1) lookups in integral dimension ────────────────────
        idx_l = Base.setindex(idx_d, l[int_dim], int_dim)
        idx_r = Base.setindex(idx_d, r[int_dim], int_dim)
        tl = l_ok[int_dim] ? itp.tail1_left[int_dim][idx_l...]  : zero(T)
        tr = r_ok[int_dim] ? itp.tail1_right[int_dim][idx_r...] : zero(T)
        result += (tl + tr) * kt_prod
    end

    # ── scaling ──────────────────────────────────────────────────────────
    scale = one(T)
    @inbounds for d in 1:N
        if d == int_dim
            scale *= itp.h[d]
        else
            scale *= (-one(T) / itp.h[d])^DO[d]
        end
    end

    return result * scale
end

@inline function _integral_dim1(::Val{DO}) where {DO}
    int_dim = findfirst(d -> DO[d] == -1, 1:length(DO))
    return int_dim::Int
end