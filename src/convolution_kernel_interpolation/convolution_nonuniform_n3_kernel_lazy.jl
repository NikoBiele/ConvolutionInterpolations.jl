"""
    _nonuniform_lazy_coef(values::AbstractArray{T,N}, knots_orig::NTuple{N}, 
                           idx::NTuple{N,Int}) where {T,N}

Resolve a coefficient at expanded-array index `idx` on the fly, without
materializing the expanded array.

Index convention matches the eager `create_nonuniform_coefs`:
    1               = left ghost
    2 : n+1         = interior (values[1:n])
    n+2             = right ghost

Uses iterative dimension-by-dimension resolution matching the eager
`for dim in 1:N` loop in `create_nonuniform_coefs`: dims are processed
sequentially, so a corner ghost (out of bounds in dims 1 and 2) resolves
dim 1 first, then dim 2 reads those resolved values — identical ordering
to the eager materialized fill.

For n3 (num_ghost=1, 3-point stencil), a D-dimensional corner expands to
at most 3^D weighted lookups into the raw data array.
"""
function _nonuniform_lazy_coef(values::AbstractArray{T,N}, knots_orig::NTuple{N}, 
                                idx::NTuple{N,Int}) where {T,N}
    # Start with a single term: weight=1.0 at the requested index
    # Each dimension that's out of bounds gets expanded into a weighted
    # sum using the same ghost coefficients as the eager path.
    
    # Represent as parallel vectors of (weight, raw_index_tuple)
    weights = T[one(T)]
    indices = NTuple{N,Int}[idx]
    
    for d in 1:N
        n_d = size(values, d)
        
        new_weights = T[]
        new_indices = NTuple{N,Int}[]
        
        for (w, ix) in zip(weights, indices)
            id = ix[d]
            
            if id >= 2 && id <= n_d + 1
                # Interior in this dim — keep as-is
                push!(new_weights, w)
                push!(new_indices, ix)
                
            elseif id < 2
                # Left ghost: 3-point extrapolation from first 3 interior points
                k = knots_orig[d]
                h0 = k[2] - k[1]
                h1 = k[3] - k[2]
                c1 =  2 * (2*h0 + h1) / (h0 + h1)
                c2 = -(2*h0 + h1) / h1
                c3 =  2 * h0^2 / (h1 * (h0 + h1))
                
                for (m, cm) in enumerate((c1, c2, c3))
                    new_ix = ntuple(dd -> dd == d ? 1 + m : ix[dd], Val(N))
                    push!(new_weights, w * cm)
                    push!(new_indices, new_ix)
                end
                
            else
                # Right ghost: 3-point extrapolation from last 3 interior points
                k = knots_orig[d]
                hm = k[end-1] - k[end-2]
                hl = k[end] - k[end-1]
                c1 =  2 * hl^2 / (hm * (hl + hm))
                c2 = -(2*hl + hm) / hm
                c3 =  2 * (2*hl + hm) / (hl + hm)
                
                for (m, cm) in enumerate((c1, c2, c3))
                    src = n_d - 3 + m + 1  # expanded coords: last 3 interior
                    new_ix = ntuple(dd -> dd == d ? src : ix[dd], Val(N))
                    push!(new_weights, w * cm)
                    push!(new_indices, new_ix)
                end
            end
        end
        
        weights = new_weights
        indices = new_indices
    end
    
    # All indices should now be interior — sum weighted raw values
    result = zero(T)
    @inbounds for (w, ix) in zip(weights, indices)
        raw = ntuple(d -> ix[d] - 1, Val(N))  # expanded → raw
        result += w * values[raw...]
    end
    return result
end

"""
For each dimension d, resolve index idx[d] in expanded coordinates:
- Interior: returns [(1.0, idx[d]-1)] (single raw index)
- Ghost: returns [(c1, raw1), (c2, raw2), (c3, raw3)] (3 weighted raw indices)

Computed once per dimension, reused across all stencil points.
"""
@inline function _resolve_dim_nonuniform(values, knots_orig_d, d, id, n_d)
    T = eltype(values)
    if id >= 2 && id <= n_d + 1
        # Interior
        return ((one(T), id - 1),)
    elseif id < 2
        k = knots_orig_d
        h0 = k[2] - k[1]
        h1 = k[3] - k[2]
        c1 =  2 * (2*h0 + h1) / (h0 + h1)
        c2 = -(2*h0 + h1) / h1
        c3 =  2 * h0^2 / (h1 * (h0 + h1))
        return ((c1, 1), (c2, 2), (c3, 3))
    else
        k = knots_orig_d
        hm = k[end-1] - k[end-2]
        hl = k[end] - k[end-1]
        c1 =  2 * hl^2 / (hm * (hl + hm))
        c2 = -(2*hl + hm) / hm
        c3 =  2 * (2*hl + hm) / (hl + hm)
        return ((c1, n_d - 2), (c2, n_d - 1), (c3, n_d))
    end
end

@inline function (itp::ConvolutionInterpolation{T,1,0,TCoefs,Axs,KA,Val{1},
                    NonUniformNonMixedHighKernel{DG},EQ,KBC,DerivativeOrder{DO},
                    FD,SD,SG,Val{:nonuniform},Val{true},Val{0}})(x::Vararg{Number,1}) where
                    {T<:AbstractFloat,TCoefs<:AbstractArray{T,1},Axs<:Tuple{<:AbstractVector},
                    KA<:Tuple{<:Nothing},DG,EQ<:Tuple{Int},
                    KBC<:Tuple{<:Tuple{Symbol,Symbol}},DO,FD,SD,SG}

    knots_exp = itp.knots[1]
    n = length(knots_exp)
    i = searchsortedlast(knots_exp, x[1])
    i = clamp(i, 2, n - 2)

    if itp.boundary_fallback && (i < 3 || i > n - 1)
        t = (T(x[1]) - knots_exp[i]) / (knots_exp[i+1] - knots_exp[i])
        return (one(T) - t) * itp.coefs[i - 1] + t * itp.coefs[i]
    end
    
    x_local = T(x[1]) - knots_exp[i]
    hm = knots_exp[i]   - knots_exp[i-1]
    h0 = knots_exp[i+1] - knots_exp[i]
    hp = knots_exp[i+2] - knots_exp[i+1]
    
    w = nonuniform_cubic_weights(x_local, hm, h0, hp)
    n_d = length(itp.coefs)
    knots_orig_d = knots_exp[2:end-1]
    
    # Resolve each stencil index once
    result = zero(T)
    for (si, wi) in zip((i-1, i, i+1, i+2), w)
        for (cw, ri) in _resolve_dim_nonuniform(itp.coefs, knots_orig_d, 1, si, n_d)
            result += wi * cw * itp.coefs[ri]
        end
    end
    return result
end

@inline function (itp::ConvolutionInterpolation{T,2,0,TCoefs,Axs,KA,Val{2},
                    NonUniformNonMixedHighKernel{DG},EQ,KBC,DerivativeOrder{DO},
                    FD,SD,SG,Val{:nonuniform},Val{true},Val{0}})(x::Vararg{Number,2}) where
                    {T<:AbstractFloat,TCoefs<:AbstractArray{T,2},
                    Axs<:Tuple{<:AbstractVector,<:AbstractVector},
                    KA<:Tuple{<:Nothing,<:Nothing},DG,EQ<:Tuple{Int,Int},
                    KBC<:Tuple{<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol}},DO,FD,SD,SG}

    i, wi = _nonuniform_dim_ghost(itp.knots[1], x[1])
    j, wj = _nonuniform_dim_ghost(itp.knots[2], x[2])
    
    if itp.boundary_fallback && (i < 3 || i > length(itp.knots[1]) - 2 ||
                                  j < 3 || j > length(itp.knots[2]) - 2)
        t1 = (T(x[1]) - itp.knots[1][i]) / (itp.knots[1][i+1] - itp.knots[1][i])
        t2 = (T(x[2]) - itp.knots[2][j]) / (itp.knots[2][j+1] - itp.knots[2][j])
        @inbounds return (one(T)-t1)*(one(T)-t2)*itp.coefs[i-1, j-1] +
                          t1*(one(T)-t2)*itp.coefs[i, j-1] +
                          (one(T)-t1)*t2*itp.coefs[i-1, j] +
                          t1*t2*itp.coefs[i, j]
    end

    n1 = size(itp.coefs, 1)
    n2 = size(itp.coefs, 2)
    k1 = itp.knots[1][2:end-1]
    k2 = itp.knots[2][2:end-1]
    
    # Precompute dim resolutions for all stencil indices
    res1 = ntuple(s -> _resolve_dim_nonuniform(itp.coefs, k1, 1, i + s - 2, n1), 4)
    res2 = ntuple(s -> _resolve_dim_nonuniform(itp.coefs, k2, 2, j + s - 2, n2), 4)
    
    result = zero(T)
    @inbounds for s1 in 1:4
        for (cw1, ri1) in res1[s1]
            w1 = wi[s1] * cw1
            for s2 in 1:4
                for (cw2, ri2) in res2[s2]
                    result += w1 * wj[s2] * cw2 * itp.coefs[ri1, ri2]
                end
            end
        end
    end
    return result
end

@inline function (itp::ConvolutionInterpolation{T,3,0,TCoefs,Axs,KA,Val{3},
                    NonUniformNonMixedHighKernel{DG},EQ,KBC,DerivativeOrder{DO},
                    FD,SD,SG,Val{:nonuniform},Val{true},Val{0}})(x::Vararg{Number,3}) where
                    {T<:AbstractFloat,TCoefs<:AbstractArray{T,3},
                    Axs<:Tuple{<:AbstractVector,<:AbstractVector,<:AbstractVector},
                    KA<:Tuple{<:Nothing,<:Nothing,<:Nothing},DG,EQ<:Tuple{Int,Int,Int},
                    KBC<:Tuple{<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol}},
                    DO,FD,SD,SG}

    i, wi = _nonuniform_dim_ghost(itp.knots[1], x[1])
    j, wj = _nonuniform_dim_ghost(itp.knots[2], x[2])
    k, wk = _nonuniform_dim_ghost(itp.knots[3], x[3])

    if itp.boundary_fallback && (i < 3 || i > length(itp.knots[1]) - 2 ||
                                  j < 3 || j > length(itp.knots[2]) - 2 ||
                                  k < 3 || k > length(itp.knots[3]) - 2)
        t1 = (T(x[1]) - itp.knots[1][i]) / (itp.knots[1][i+1] - itp.knots[1][i])
        t2 = (T(x[2]) - itp.knots[2][j]) / (itp.knots[2][j+1] - itp.knots[2][j])
        t3 = (T(x[3]) - itp.knots[3][k]) / (itp.knots[3][k+1] - itp.knots[3][k])
        result = zero(T)
        @inbounds for c3 in 0:1, c2 in 0:1, c1 in 0:1
            w = (c1 == 0 ? one(T) - t1 : t1) *
                (c2 == 0 ? one(T) - t2 : t2) *
                (c3 == 0 ? one(T) - t3 : t3)
            result += w * itp.coefs[i-1+c1, j-1+c2, k-1+c3]
        end
        return result
    end
    
    n1 = size(itp.coefs, 1)
    n2 = size(itp.coefs, 2)
    n3 = size(itp.coefs, 3)
    k1 = itp.knots[1][2:end-1]
    k2 = itp.knots[2][2:end-1]
    k3 = itp.knots[3][2:end-1]
    
    res1 = ntuple(s -> _resolve_dim_nonuniform(itp.coefs, k1, 1, i + s - 2, n1), 4)
    res2 = ntuple(s -> _resolve_dim_nonuniform(itp.coefs, k2, 2, j + s - 2, n2), 4)
    res3 = ntuple(s -> _resolve_dim_nonuniform(itp.coefs, k3, 3, k + s - 2, n3), 4)
    
    result = zero(T)
    @inbounds for s1 in 1:4
        for (cw1, ri1) in res1[s1]
            w1 = wi[s1] * cw1
            for s2 in 1:4
                for (cw2, ri2) in res2[s2]
                    w12 = w1 * wj[s2] * cw2
                    for s3 in 1:4
                        for (cw3, ri3) in res3[s3]
                            result += w12 * wk[s3] * cw3 * itp.coefs[ri1, ri2, ri3]
                        end
                    end
                end
            end
        end
    end
    return result
end

@inline function (itp::ConvolutionInterpolation{T,N,0,TCoefs,Axs,KA,HigherDimension{N},
        NonUniformNonMixedHighKernel{DG},EQ,KBC,DerivativeOrder{DO},
        FD,SD,SG,Val{:nonuniform},Val{true},Val{0}})(x::Vararg{Number,N}) where 
        {T<:AbstractFloat,N,TCoefs<:AbstractArray{T,N},Axs<:Tuple{Vararg{AbstractVector}},
        KA<:Tuple{Vararg{Nothing}},DG,EQ<:Tuple{Vararg{Int}},
        KBC<:Tuple{Vararg{Tuple{Symbol,Symbol}}},DO,FD,SD,SG}

    knots_orig = ntuple(d -> itp.knots[d][2:end-1], N)
    
    iw = ntuple(d -> _nonuniform_dim_ghost(itp.knots[d], x[d]), N)
    indices = ntuple(d -> iw[d][1], N)

    is_boundary = any(d -> let id = indices[d]
        id < 3 || id > length(itp.knots[d]) - 2
    end, 1:N)
    
    if itp.boundary_fallback && is_boundary
        ts = ntuple(N) do d
            kd = itp.knots[d]
            id = indices[d]
            (T(x[d]) - kd[id]) / (kd[id+1] - kd[id])
        end
        result = zero(T)
        @inbounds for offsets in Iterators.product(ntuple(_ -> 0:1, N)...)
            w = prod(d -> offsets[d] == 0 ? one(T) - ts[d] : ts[d], 1:N)
            idx = ntuple(d -> indices[d] - 1 + offsets[d], N)
            result += w * itp.coefs[idx...]
        end
        return result
    end

    weights = ntuple(d -> iw[d][2], N)
    
    result = zero(T)
    @inbounds for offsets in Iterators.product(ntuple(_ -> 1:4, N)...)
        w_prod = prod(weights[d][offsets[d]] for d in 1:N)
        idx = ntuple(d -> indices[d] + offsets[d] - 2, N)
        c = _nonuniform_lazy_coef(itp.coefs, knots_orig, idx)
        result += w_prod * c
    end
    return result
end