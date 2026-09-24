"""
(itp::FastConvolutionInterpolation{T,N,...})(x...) — FastIntegralOrders{DO}
Evaluate integrals of any order in any number of dimensions, possibly combined with
interpolation or derivatives in other dimensions.

The result is Σ_j c_j·Π_d w_d(j_d), scaled by Π_d h_d^m (integral dimensions of order m) and
(−1/h_d)^q (derivative dimensions of order q). The weight w_d depends on where j_d lies relative
to the stencil of x:
  in the stencil       — anchored integral weight, or kernel (derivative) weight
  left of the stencil  — integral dimensions only: a polynomial in the position t_d within the cell
  right of the stencil — zero
For at most 3 integral dimensions the sum splits into 2^n regions (a subset of the integral
dimensions left of the stencil, all others in it), whose left parts are precomputed as polynomial
tails (`integral_tails`), so evaluation is independent of grid size. With more integral
dimensions the sum runs directly over the coefficients up to the stencil.
See also: _build_region_tails, _anchored_stencil_weights.
"""
function (itp::FastConvolutionInterpolation{T,N,NI,TCoefs,Axs,KA,DT,DG,EQ,KBC,
            FastIntegralOrders{DO},FD,SD,Val{SG},Val{false},DI})(x::Vararg{Number,N}) where
            {T<:AbstractFloat,N,NI,TCoefs<:AbstractArray{T,N},Axs,KA,DT,DG,EQ,KBC,DO,FD,SD,SG,DI}

    x = T.(x)
    # cell and position within it, per dimension
    u = ntuple(d -> (x[d] - itp.knots[d][1]) / itp.h[d] + one(T), N)
    i = ntuple(d -> clamp(floor(Int, u[d]), itp.eqs[d], size(itp.coefs, d) - itp.eqs[d]), N)
    t = ntuple(d -> u[d] - T(i[d]), N)

    # stencil weights of every dimension, in eager order and padded to one common length
    W = _integral_stencil_weights(Val(_kernel_sym(itp.kernel_sym)), Val(DO), t, i, itp.eqs,
                                  itp.integral_taylor)

    result = _count_negative(Val(DO)) <= 3 ? _integral_sum_tails(itp, W, i, t, Val(DO)) :
                                             _integral_sum_direct(itp, W, i, t, Val(DO))
    return result * _integral_scale(itp.h, Val(DO))
end

# Number of integral dimensions (DO[d] < 0), as a compile-time constant
@generated _count_negative(::Val{DO}) where {DO} = count(<(0), DO)

# Rank of each integral dimension among the integral dimensions (1, 2, …), and 0 for the others
@generated function _integral_ranks(::Val{DO}) where {DO}
    ranks = Int[]
    r = 0
    for order in DO
        push!(ranks, order < 0 ? (r += 1) : 0)
    end
    return :($(Tuple(ranks)))
end

# Integral order of each dimension (m for DO[d] = −m, 0 for the others), compile-time
@generated _integral_orders_abs(::Val{DO}) where {DO} = :($(Tuple(o < 0 ? -o : 0 for o in DO)))

# Stencil weights of every dimension in eager order (entry k ↔ coefficient i − eqs + k), padded
# with zeros to one common length, so that they can be indexed with a runtime dimension
@generated function _integral_stencil_weights(::Val{KS}, ::Val{DO}, t::NTuple{N,T},
                                              i::NTuple{N,Int}, eqs::NTuple{N,Int},
                                              taylor::NTuple{N,Matrix{T}}) where {KS,DO,N,T}
    Kmax = 2 * maximum(get_equations_for_degree(k) for k in KS)
    calls = map(1:N) do d
        kernel = QuoteNode(KS[d])
        weights = DO[d] < 0 ?
            :(_anchored_stencil_weights(Val($kernel), Val($(-DO[d])), t[$d], i[$d], eqs[$d], taylor[$d])) :
            :(_kernel_weights(Val($kernel), Val($(DO[d])), one($T) - t[$d]))
        :(_pad_weights($weights, Val($Kmax)))
    end
    return :(tuple($(calls...)))
end

@inline _pad_weights(w::NTuple{K,T}, ::Val{Kmax}) where {K,T,Kmax} =
    ntuple(k -> k <= K ? w[k] : zero(T), Val(Kmax))

# Product of the stencil weights of all dimensions not in the tail, for coefficient index I
@inline function _stencil_weight_product(W::NTuple{N,NTuple{K,T}}, I::CartesianIndex{N},
                                         i::NTuple{N,Int}, eqs::NTuple{N,Int},
                                         in_tail::NTuple{N,Bool}) where {N,K,T}
    p = one(T)
    @inbounds for d in 1:N
        in_tail[d] || (p *= W[d][I[d] - i[d] + eqs[d]])
    end
    return p
end

# Π t_d^k_d for moment index `lin` of a region (last tail dimension's power varying fastest)
@inline function _moment_power(lin::Int, t::NTuple{N,T}, in_tail::NTuple{N,Bool},
                               orders::NTuple{N,Int}) where {N,T}
    r = lin - 1
    p = one(T)
    @inbounds for d in N:-1:1
        if in_tail[d]
            m = orders[d]
            p *= t[d]^(r % m)
            r ÷= m
        end
    end
    return p
end

# At most 3 integral dimensions: the center plus every region with precomputed tails
function _integral_sum_tails(itp::FastConvolutionInterpolation{T,N}, W, i::NTuple{N,Int},
                             t::NTuple{N,T}, ::Val{DO}) where {T,N,DO}
    eqs = itp.eqs
    ranks = _integral_ranks(Val(DO))
    orders = _integral_orders_abs(Val(DO))
    n_int = _count_negative(Val(DO))
    stencil = ntuple(d -> (i[d] - eqs[d] + 1):(i[d] + eqs[d]), N)
    result = zero(T)

    # region 0: every dimension within the stencil
    none = ntuple(_ -> false, N)
    @inbounds for I in CartesianIndices(stencil)
        result += itp.coefs[I] * _stencil_weight_product(W, I, i, eqs, none)
    end

    # regions 1 … 2^n − 1: the integral dimensions whose bit is set lie left of the stencil
    @inbounds for mask in 1:(1 << n_int) - 1
        in_tail = ntuple(d -> ranks[d] > 0 && ((mask >> (ranks[d] - 1)) & 1) == 1, N)
        # empty unless there are coefficients left of the stencil in all of the region's dimensions
        all(ntuple(d -> !in_tail[d] || i[d] - eqs[d] >= 1, N)) || continue
        ranges = ntuple(d -> in_tail[d] ? ((i[d] - eqs[d]):(i[d] - eqs[d])) : stencil[d], N)
        arrays = itp.integral_tails[mask]
        for I in CartesianIndices(ranges)
            acc = zero(T)
            for lin in eachindex(arrays)
                acc += arrays[lin][I] * _moment_power(lin, t, in_tail, orders)
            end
            result += _stencil_weight_product(W, I, i, eqs, in_tail) * acc
        end
    end
    return result
end

# More than 3 integral dimensions: direct sum over the coefficients up to the stencil
function _integral_sum_direct(itp::FastConvolutionInterpolation{T,N}, W, i::NTuple{N,Int},
                              t::NTuple{N,T}, ::Val{DO}) where {T,N,DO}
    eqs = itp.eqs
    orders = _integral_orders_abs(Val(DO))
    # right of the stencil the weight is zero; left of it only integral dimensions contribute
    ranges = ntuple(d -> DO[d] < 0 ? (1:(i[d] + eqs[d])) : ((i[d] - eqs[d] + 1):(i[d] + eqs[d])), N)
    result = zero(T)
    @inbounds for I in CartesianIndices(ranges)
        p = one(T)
        for d in 1:N
            k = I[d] - i[d] + eqs[d]
            p *= k >= 1 ? W[d][k] :
                 _left_weight(itp.integral_entries[d], I[d], t[d] + T(i[d] - I[d] - eqs[d]),
                              orders[d], eqs[d])
            iszero(p) && break
        end
        result += itp.coefs[I] * p
    end
    return result
end

# Weight of coefficient j left of the stencil in an integral dimension of order m, at position
# t0 = u − (j + eqs) relative to the cell where j enters the tail: the Cauchy weight
# (u − j)^(m−1)/(m−1)! far from the anchor, the exact entry polynomial near it
@inline function _left_weight(entries::Matrix{T}, j::Int, t0::T, m::Int, eqs::Int) where {T}
    j >= 2eqs && return (t0 + T(eqs))^(m - 1) / T(factorial(m - 1))
    acc = zero(T)
    @inbounds for k in m:-1:1
        acc = acc * t0 + entries[j, k]
    end
    return acc
end

# Π_d h_d^m (integral dimensions) · Π_d (−1/h_d)^q (derivative dimensions)
@inline function _integral_scale(h::NTuple{N,T}, ::Val{DO}) where {N,T,DO}
    s = one(T)
    @inbounds for d in 1:N
        s *= DO[d] < 0 ? h[d]^(-DO[d]) : (-one(T) / h[d])^DO[d]
    end
    return s
end