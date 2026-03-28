"""
    _nonuniform_dim_ghost(knots::AbstractVector{T}, x_val::Number) where T

Find interval index and compute cubic nonuniform weights for a single dimension.

Takes the expanded knot vector (including ghost points) and returns the interval index `i`
and a 4-element weight tuple `w` for the stencil `[i-1, i, i+1, i+2]`. Uses local spacing
ratios (hm, h0, hp) for the nonuniform cubic weight formula.
"""

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
    
    w = nonuniform_cubic_weights(x_local, hm, h0, hp)
    return i, w
end

"""
    (itp::ConvolutionInterpolation{...,Val{:n3},...})(x...)

Evaluate nonuniform cubic (`:n3`) interpolation. Uses 4-point stencil per dimension with
locally adapted weights based on nonuniform grid spacing. Supports 1D through N-D via
tensor product of 1D weights. Ghost points at boundaries eliminate the need for clamping.

This is the fallback path for a-series kernels on nonuniform grids.
"""

# --- 1D with ghost points ---

@inline function (itp::ConvolutionInterpolation{T,1,0,TCoefs,Axs,KA,Val{1},
                    NonUniformNonMixedHighKernel{DG},EQ,KBC,DerivativeOrder{DO},
                    FD,SD,SG,Val{:nonuniform},Val{false},Val{0}})(x::Vararg{Number,1}) where
                    {T<:AbstractFloat,TCoefs<:AbstractArray{T,1},Axs<:Tuple{<:AbstractVector},
                    KA<:Tuple{<:Nothing},DG,EQ<:Tuple{Int},
                    KBC<:Tuple{<:Tuple{Symbol,Symbol}},DO,FD,SD,SG}

    # itp.knots[1] is the expanded knot vector
    # itp.coefs includes ghost values
    i, w = _nonuniform_dim_ghost(itp.knots[1], x[1])
    @inbounds return w[1] * itp.coefs[i-1] + w[2] * itp.coefs[i] + 
                     w[3] * itp.coefs[i+1] + w[4] * itp.coefs[i+2]
end

# --- 2D with ghost points ---

@inline function (itp::ConvolutionInterpolation{T,2,0,TCoefs,Axs,KA,Val{2},
                    NonUniformNonMixedHighKernel{DG},EQ,KBC,DerivativeOrder{DO},
                    FD,SD,SG,Val{:nonuniform},Val{false},Val{0}})(x::Vararg{Number,2}) where
                    {T<:AbstractFloat,TCoefs<:AbstractArray{T,2},
                    Axs<:Tuple{<:AbstractVector,<:AbstractVector},
                    KA<:Tuple{<:Nothing,<:Nothing},DG,EQ<:Tuple{Int,Int},
                    KBC<:Tuple{<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol}},DO,FD,SD,SG}

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

@inline function (itp::ConvolutionInterpolation{T,3,0,TCoefs,Axs,KA,Val{3},
                    NonUniformNonMixedHighKernel{DG},EQ,KBC,DerivativeOrder{DO},
                    FD,SD,SG,Val{:nonuniform},Val{false},Val{0}})(x::Vararg{Number,3}) where
                    {T<:AbstractFloat,TCoefs<:AbstractArray{T,3},
                    Axs<:Tuple{<:AbstractVector,<:AbstractVector,<:AbstractVector},
                    KA<:Tuple{<:Nothing,<:Nothing,<:Nothing},DG,EQ<:Tuple{Int,Int,Int},
                    KBC<:Tuple{<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol}},
                    DO,FD,SD,SG}

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

@inline function (itp::ConvolutionInterpolation{T,N,0,TCoefs,Axs,KA,HigherDimension{N},
        NonUniformNonMixedHighKernel{DG},EQ,KBC,DerivativeOrder{DO},
        FD,SD,SG,Val{:nonuniform},Val{false}})(x::Vararg{Number,N}) where 
        {T<:AbstractFloat,N,TCoefs<:AbstractArray{T,N},Axs<:Tuple{Vararg{AbstractVector}},
        KA<:Tuple{Vararg{Nothing}},DG,EQ<:Tuple{Vararg{Int}},
        KBC<:Tuple{Vararg{Tuple{Symbol,Symbol}}},DO,FD,SD,SG}

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