# """
# (itp::FastConvolutionInterpolation{T,N,...})(x...) — FastMixedIntegralOrder, n_integral=1
# Evaluate mixed antiderivative/derivative/interpolation in N dimensions,
# with exactly 1 integral dimension using fast tail lookups.

# Integral dimension: K̃ center loop with subgrid (linear/cubic/quintic) + O(1) tail1 lookups.
# Derivative dimensions: linear subgrid kernel evaluation.

# Cost: O(eqs^N) center + O(eqs^(N-1)) tails.
# """
@inline function (itp::FastConvolutionInterpolation{T,N,1,TCoefs,Axs,KA,DIM,DG,EQ,KBC,
            FastMixedIntegralOrder{DO},FD,SD,Val{SG},Val{false},Val{1}})(x::Vararg{Number,N}) where
            {T<:AbstractFloat,N,TCoefs<:AbstractArray{T,N},Axs<:Tuple{Vararg{AbstractVector}},
            KA<:Tuple{Vararg{Nothing}},DIM,DG,EQ<:Tuple{Vararg{Int}},KBC<:Tuple{Vararg{Tuple{Symbol,Symbol}}},DO,FD,SD,SG}

    x = T.(x)
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

    # ── kernel weights of every dimension (exact column polynomials) ─────
    w = _mixed_weights(itp, x, i, Val(DO))   

    result = zero(T)

    # ── main loop over derivative dimension stencils ─────────────────────
    deriv_ranges = ntuple(d -> d == int_dim ? (1:1) :
                               ((i[d] - itp.eqs[d] + 1):(i[d] + itp.eqs[d])), N)

    @inbounds for idx_d in Iterators.product(deriv_ranges...)

        # product of derivative/interpolation kernel weights (exact column polynomials)
        kt_prod = _derivative_weight_product(w, idx_d, i, itp.eqs, Val(DO))

        # ── center: K̃ loop over integral dimension stencil ──────────────
        @inbounds for j_int in (i[int_dim] - itp.eqs[int_dim] + 1):(i[int_dim] + itp.eqs[int_dim])
            lv       = _antiderivative_weight(w[int_dim], i[int_dim], itp.eqs[int_dim], j_int) -
                       itp.left_values[int_dim][j_int]
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

# Index of the single integral dimension, as a compile-time constant
@inline _integral_dim1(::Val{DO}) where {DO} = _integral_dims(Val(DO))[1]