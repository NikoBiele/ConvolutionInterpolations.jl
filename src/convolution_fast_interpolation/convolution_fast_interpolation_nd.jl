"""
    (itp::FastConvolutionInterpolation{T,N,...})(x::Vararg{Number,N})

Evaluate N-dimensional (N > 3) fast convolution interpolation at coordinates `x`.

Dispatches on kernel type:

**Specialized kernels** (no precomputed table):
- `:a0` — Nearest neighbor, ~20ns (4D). Selects nearest grid point per dimension.
- `:a1` — Multilinear interpolation, ~35ns (4D). Interpolates over 2^N corners
  using bit-manipulation to enumerate combinations efficiently.

**Higher-order kernels**: Uses linear subgrid interpolation over all `(2*eqs)^N`
kernel support points. The N-D kernel is formed as a tensor product of 1D kernels:
`K_N(x₁,...,x_N) = K₁(x₁)·...·K₁(x_N)`. Each 1D kernel value is linearly
interpolated from the precomputed table.

O(1) evaluation time with respect to grid size, allocation-free. Only linear subgrid
is available in N > 3 dimensions. The exponential growth in support points (e.g.
`(2*eqs)^N`) limits practical use to modest kernel orders in high dimensions.

Results scaled by `∏ᵢ (-1/hᵢ)^derivative`.

See also: `FastConvolutionInterpolation`.
"""

@inline function (itp::FastConvolutionInterpolation{T,N,TCoefs,IT,Axs,KA,HigherDimension{N},Val{:a0},EQ,PR,KP})(x::Vararg{Number,N}) where {T,N,TCoefs,IT,KA,Axs,EQ,PR,KP}
    # specialized dispatch for N-dimensional nearest neighbor kernel

    # Compute i_float once per dimension
    i_floats = ntuple(d -> (x[d] - itp.knots[d][1]) / itp.h[d] + one(T), N)
    
    # Find knot indices for each dimension
    pos_ids = ntuple(d -> clamp(floor(Int, i_floats[d]), itp.eqs, length(itp.knots[d]) - itp.eqs), N)
    
    # Compute normalized left distances - recompute from actual knot positions
    diff_left = ntuple(d -> (x[d] - itp.knots[d][pos_ids[d]]) / itp.h[d], N)

    # Nearest neighbor: return coefficient at nearest grid point
    nearest_ids = ntuple(d -> diff_left[d] < 0.5 ? pos_ids[d] : pos_ids[d]+1, N)
    return itp.coefs[nearest_ids...]
end

@inline function (itp::FastConvolutionInterpolation{T,N,TCoefs,IT,Axs,KA,HigherDimension{N},Val{:a1},EQ,PR,KP,KBC,DO})(x::Vararg{Number,N}) where {T,N,TCoefs,IT,KA,Axs,EQ,PR,KP,KBC,DO}
    # specialized dispatch for N-dimensional linear kernel
    
    # Compute i_float once per dimension
    i_floats = ntuple(d -> (x[d] - itp.knots[d][1]) / itp.h[d] + one(T), N)
    
    # Find knot indices for each dimension
    pos_ids = ntuple(d -> clamp(floor(Int, i_floats[d]), itp.eqs, length(itp.knots[d]) - itp.eqs), N)
    
    # Compute normalized left distances - recompute from actual knot positions
    weights = ntuple(d -> (x[d] - itp.knots[d][pos_ids[d]]) / itp.h[d], N)
    
    # Build up: for each "slice" in remaining dimensions, 
    # accumulate the interpolated result
    result = zero(T)
    
    # Iterate over all 2^(N-1) combinations of the last N-1 dimensions
    @inbounds for corner in 0:(2^(N-1) - 1)
        # Build indices for dimensions 2:N
        tail_indices = ntuple(d -> (corner >> (d-1)) & 1 == 0 ? pos_ids[d+1] : pos_ids[d+1]+1, N-1)
        
        # Get the two values along dimension 1
        idx0 = (pos_ids[1], tail_indices...)
        idx1 = (pos_ids[1]+1, tail_indices...)
        
        # Interpolate along dimension 1
        interp_val = (one(T) - weights[1]) * itp.coefs[idx0...] + weights[1] * itp.coefs[idx1...]
        
        # Weight by all the other dimensions
        tail_weight = prod(ntuple(d -> (corner >> (d-1)) & 1 == 0 ? (one(T) - weights[d+1]) : weights[d+1], N-1))
        
        result += tail_weight * interp_val
    end
    
    return @inbounds @fastmath result * prod((-one(T)/itp.h[i])^(DO.parameters[1]) for i in 1:N)
end

function (itp::FastConvolutionInterpolation{T,N,TCoefs,IT,Axs,KA,HigherDimension{N},HigherOrderKernel{DG},EQ,PR,KP,KBC,DO})(x::Vararg{Number,N}) where {T,N,TCoefs,IT,KA,Axs,DG,EQ,PR,KP,KBC,DO}
    # specialized dispatch for N-dimensional higher-order kernel
    
    # Compute i_float once per dimension
    i_floats = ntuple(d -> (x[d] - itp.knots[d][1]) / itp.h[d] + one(T), N)
    
    # Find knot indices for each dimension
    pos_ids = ntuple(d -> clamp(floor(Int, i_floats[d]), itp.eqs, length(itp.knots[d]) - itp.eqs), N)
    
    # Compute normalized left distances - recompute from actual knot positions
    diff_left = ntuple(d -> (x[d] - itp.knots[d][pos_ids[d]]) / itp.h[d], N)
    diff_right = ntuple(d -> one(T) - diff_left[d], N)

    idx_lower = ntuple(d -> clamp(floor(Int, diff_right[d] * (length(itp.pre_range) - one(Int64))) + one(Int64),
                                   one(Int64), length(itp.pre_range) - one(Int64)), N)
    
    idx_upper = ntuple(d -> idx_lower[d] + 1, N)
    t = ntuple(d -> (diff_right[d] - itp.pre_range[idx_lower[d]]) / 
                     (itp.pre_range[idx_upper[d]] - itp.pre_range[idx_lower[d]]), N)

    result = zero(T)
    
    @inbounds for offsets in Iterators.product(ntuple(_ -> -(itp.eqs-1):itp.eqs, N)...)
        coef = itp.coefs[(pos_ids .+ offsets)...]
        
        kernel_val = one(T)
        @inbounds for d in 1:N
            k_lower = itp.kernel_pre[idx_lower[d], offsets[d]+itp.eqs]
            k_upper = itp.kernel_pre[idx_upper[d], offsets[d]+itp.eqs]
            kernel_val *= (one(T) - t[d]) * k_lower + t[d] * k_upper
        end
        
        result += coef * kernel_val
    end
    
    return  @inbounds @fastmath result * prod((-one(T)/itp.h[i])^(DO.parameters[1]) for i in 1:N)
end
