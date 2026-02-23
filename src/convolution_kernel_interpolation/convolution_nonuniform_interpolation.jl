# ============================================================
# Updated nonuniform evaluation — uses ghost points (no clamp)
# ============================================================

# --- Helper: find index and weights for one dimension (with ghost points) ---

@inline function _nonuniform_dim_ghost(knots::AbstractVector{T}, x_val::Number) where T
    # knots is the EXPANDED knot vector (includes ghost positions)
    # Data array also has ghost values at index 1 and end
    n = length(knots)
    i = searchsortedlast(knots, x_val)
    # With ghost points, valid range is 2 to n-2 (need i-1, i, i+1, i+2)
    i = clamp(i, 2, n - 2)
    
    x_local = T(x_val) - knots[i]
    hm = knots[i]   - knots[i-1]
    h0 = knots[i+1] - knots[i]
    hp = knots[i+2] - knots[i+1]
    
    w = nonuniform_weights(x_local, hm, h0, hp)
    return i, w
end

# --- 1D with ghost points ---

@inline function (itp::ConvolutionInterpolation{T,1,TCoefs,IT,Axs,KA,Val{1},
                    Val{:nonuniform},EQ,KBC,DO})(x::Vararg{Number,1}) where 
                    {T,TCoefs,IT,Axs,KA,EQ,KBC,DO}

    # itp.knots[1] is the expanded knot vector
    # itp.coefs includes ghost values
    i, w = _nonuniform_dim_ghost(itp.knots[1], x[1])
    
    @inbounds return w[1] * itp.coefs[i-1] + w[2] * itp.coefs[i] + 
                     w[3] * itp.coefs[i+1] + w[4] * itp.coefs[i+2]
end

# --- 2D with ghost points ---

@inline function (itp::ConvolutionInterpolation{T,2,TCoefs,IT,Axs,KA,Val{2},
                    Val{:nonuniform},EQ,KBC,DO})(x::Vararg{Number,2}) where 
                    {T,TCoefs,IT,Axs,KA,EQ,KBC,DO}

    i, wi = _nonuniform_dim_ghost(itp.knots[1], x[1])
    j, wj = _nonuniform_dim_ghost(itp.knots[2], x[2])
    
    result = zero(T)
    @inbounds for (li, wli) in zip(-1:2, wi)
        for (lj, wlj) in zip(-1:2, wj)
            result += wli * wlj * itp.coefs[i+li, j+lj]
        end
    end
    return result
end

# --- 3D with ghost points ---

@inline function (itp::ConvolutionInterpolation{T,3,TCoefs,IT,Axs,KA,Val{3},
                    Val{:nonuniform},EQ,KBC,DO})(x::Vararg{Number,3}) where 
                    {T,TCoefs,IT,Axs,KA,EQ,KBC,DO}

    i, wi = _nonuniform_dim_ghost(itp.knots[1], x[1])
    j, wj = _nonuniform_dim_ghost(itp.knots[2], x[2])
    k, wk = _nonuniform_dim_ghost(itp.knots[3], x[3])
    
    result = zero(T)
    @inbounds for (li, wli) in zip(-1:2, wi)
        for (lj, wlj) in zip(-1:2, wj)
            for (lk, wlk) in zip(-1:2, wk)
                result += wli * wlj * wlk * itp.coefs[i+li, j+lj, k+lk]
            end
        end
    end
    return result
end

# --- ND (N > 3) with ghost points ---

@inline function (itp::ConvolutionInterpolation{T,N,TCoefs,IT,Axs,KA,HigherDimension{N},
                    Val{:nonuniform},EQ,KBC,DO})(x::Vararg{Number,N}) where 
                    {T,N,TCoefs,IT,Axs,KA,EQ,KBC,DO}

    iw = ntuple(d -> _nonuniform_dim_ghost(itp.knots[d], x[d]), N)
    indices = ntuple(d -> iw[d][1], N)
    weights = ntuple(d -> iw[d][2], N)
    
    result = zero(T)
    @inbounds for offsets in Iterators.product(ntuple(_ -> 1:4, N)...)
        w_prod = prod(weights[d][offsets[d]] for d in 1:N)
        idx = ntuple(d -> indices[d] + offsets[d] - 2, N)
        result += w_prod * itp.coefs[idx...]
    end
    return result
end