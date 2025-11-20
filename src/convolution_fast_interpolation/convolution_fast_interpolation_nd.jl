"""
    (itp::FastConvolutionInterpolation{T,N,...,Val{:a0},...})(x::Vararg{Number,N})

Evaluate N-dimensional (N > 3) fast convolution interpolation with nearest neighbor kernel.

# Arguments
- `itp`: Fast convolution interpolation object with `:a0` kernel
- `x`: N coordinates at which to evaluate

# Returns
Value at the nearest grid point in N dimensions

# Performance
- O(1) evaluation time independent of grid size
- Allocation-free
- Typical timing (4D): ~20ns per evaluation

# Details
Generalizes nearest neighbor to arbitrary dimensions by finding the nearest grid point
in each dimension independently. The fractional distances are recomputed from actual knot
positions after clamping, ensuring correct behavior at boundaries.

# Algorithm
1. Compute floating-point indices for all N dimensions using `ntuple`
2. Clamp to valid interior ranges
3. Recompute fractional distances from actual knots (handles boundary clamping)
4. Select grid point based on which hyperoctant the evaluation point falls into
5. Return coefficient at nearest N-dimensional grid point

See also: `FastConvolutionInterpolation`.
"""

@inline function (itp::FastConvolutionInterpolation{T,N,TCoefs,IT,Axs,KA,HigherDimension{N},Val{:a0},EQ,PR,KP})(x::Vararg{Number,N}) where {T,N,TCoefs,IT,KA,Axs,EQ,PR,KP}
    # specialized dispatch for N-dimensional nearest neighbor kernel

    # Compute i_float once per dimension
    i_floats = ntuple(d -> (x[d] - itp.knots[d][1]) / itp.h[d] + 1, N)
    
    # Find knot indices for each dimension
    pos_ids = ntuple(d -> clamp(floor(Int, i_floats[d]), itp.eqs, length(itp.knots[d]) - itp.eqs), N)
    
    # Compute normalized left distances - recompute from actual knot positions
    diff_left = ntuple(d -> (x[d] - itp.knots[d][pos_ids[d]]) / itp.h[d], N)

    # Nearest neighbor: return coefficient at nearest grid point
    nearest_ids = ntuple(d -> diff_left[d] < 0.5 ? pos_ids[d] : pos_ids[d]+1, N)
    return itp.coefs[nearest_ids...]
end

"""
    (itp::FastConvolutionInterpolation{T,N,...,Val{:a1},...})(x::Vararg{Number,N})

Evaluate N-dimensional (N > 3) fast convolution interpolation with multilinear kernel.

# Arguments
- `itp`: Fast convolution interpolation object with `:a1` kernel
- `x`: N coordinates at which to evaluate

# Returns
Multilinearly interpolated value from the 2^N surrounding grid points

# Performance
- O(1) evaluation time independent of grid size
- Allocation-free
- Typical timing (4D): ~35ns per evaluation
- Scales with 2^N corners, not grid size

# Details
Uses multilinear interpolation with direct index calculation (no binary search). The
implementation uses a clever bit-manipulation approach to iterate over all 2^(N-1)
combinations efficiently. Fractional distances are recomputed from actual knot positions
after clamping to ensure correct interpolation at domain boundaries.

# Algorithm
1. Compute floating-point indices for all N dimensions
2. Clamp to valid interior ranges
3. Recompute fractional distances from actual knots (handles boundaries correctly)
4. Accumulation-based multilinear interpolation:
   - Iterate over 2^(N-1) combinations of dimensions 2:N using bit manipulation
   - For each combination, interpolate along dimension 1
   - Weight by products of (1-w) or w for remaining dimensions
5. Returns weighted sum over all 2^N corners

# Implementation Note
The bit-shifting approach `(corner >> (d-1)) & 1` efficiently enumerates all corner
combinations without nested loops, making the code dimension-agnostic.

See also: `FastConvolutionInterpolation`.
"""

@inline function (itp::FastConvolutionInterpolation{T,N,TCoefs,IT,Axs,KA,HigherDimension{N},Val{:a1},EQ,PR,KP})(x::Vararg{Number,N}) where {T,N,TCoefs,IT,KA,Axs,EQ,PR,KP}
    # specialized dispatch for N-dimensional linear kernel
    
    # Compute i_float once per dimension
    i_floats = ntuple(d -> (x[d] - itp.knots[d][1]) / itp.h[d] + 1, N)
    
    # Find knot indices for each dimension
    pos_ids = ntuple(d -> clamp(floor(Int, i_floats[d]), itp.eqs, length(itp.knots[d]) - itp.eqs), N)
    
    # Compute normalized left distances - recompute from actual knot positions
    weights = ntuple(d -> (x[d] - itp.knots[d][pos_ids[d]]) / itp.h[d], N)
    
    # Build up: for each "slice" in remaining dimensions, 
    # accumulate the interpolated result
    result = zero(T)
    
    # Iterate over all 2^(N-1) combinations of the last N-1 dimensions
    for corner in 0:(2^(N-1) - 1)
        # Build indices for dimensions 2:N
        tail_indices = ntuple(d -> (corner >> (d-1)) & 1 == 0 ? pos_ids[d+1] : pos_ids[d+1]+1, N-1)
        
        # Get the two values along dimension 1
        idx0 = (pos_ids[1], tail_indices...)
        idx1 = (pos_ids[1]+1, tail_indices...)
        
        # Interpolate along dimension 1
        interp_val = (1 - weights[1]) * itp.coefs[idx0...] + weights[1] * itp.coefs[idx1...]
        
        # Weight by all the other dimensions
        tail_weight = prod(ntuple(d -> (corner >> (d-1)) & 1 == 0 ? (1 - weights[d+1]) : weights[d+1], N-1))
        
        result += tail_weight * interp_val
    end
    
    return result
end

"""
    (itp::FastConvolutionInterpolation{T,N,...,HigherOrderKernel{DG},...})(x::Vararg{Number,N})

Evaluate N-dimensional (N > 3) fast convolution interpolation with higher-order kernel.

# Arguments
- `itp`: Fast convolution interpolation object with higher-order kernel
- `x`: N coordinates at which to evaluate

# Returns
Interpolated value using precomputed N-dimensional kernel (tensor product)

# Performance
- O(1) evaluation time independent of grid size
- Allocation-free
- Typical timing (4D, `:b5`): ~44ms per evaluation
- Performance scales with (2*eqs-1)^N kernel support, not grid size
- This is the fundamental curse of dimensionality for tensor product methods

# Details
Achieves high-order accuracy in N dimensions with O(1) grid scaling by:
1. Computing grid indices directly in all dimensions (no binary search)
2. Using precomputed 1D kernel values for each dimension
3. Forming N-D kernel as tensor product: K_N(x₁,...,x_N) = K₁(x₁)·...·K₁(x_N)
4. Iterating over all (2*eqs-1)^N kernel support points
5. Multilinear interpolation between 2^N precomputed kernel positions
6. Recomputing fractional distances from actual knots (correct boundary handling)

The tensor product structure keeps the algorithm dimension-agnostic, but the
exponential growth in support points (e.g., 6561 points for `:b5` in 4D) limits
practical use to modest kernel orders in high dimensions.

# Algorithm
1. Direct index calculation for all N dimensions using `ntuple`
2. Recompute fractional distances from actual knots (handles clamping)
3. Lookup precomputed kernel values at neighboring positions in all dimensions
4. Iterate over all (2*eqs-1)^N offsets using `Iterators.product`
5. For each offset:
   - Load coefficient at offset position
   - Compute N-D kernel value as product of 1D kernel values
   - Each 1D kernel is linearly interpolated from precomputed values
   - Accumulate weighted contribution
6. Returns final interpolated value

# Mathematical Detail
The evaluation computes:
```
result = Σ_{o₁,...,o_N} coef[pos + offset] * ∏ᵢ Kᵢ(oᵢ, tᵢ)
```
where each Kᵢ is linearly interpolated from precomputed kernel values, and the
sum runs over all (2*eqs-1)^N offset combinations.

# Performance Considerations
- 4D with `:b5`: 6561 iterations (81² in 2D, 729 in 3D by comparison)
- Consider using lower-order kernels (`:a3`, `:b3`) for 4D+ applications
- Grid size has zero impact on evaluation time

See also: `FastConvolutionInterpolation`, `ConvolutionInterpolation`.
"""

function (itp::FastConvolutionInterpolation{T,N,TCoefs,IT,Axs,KA,HigherDimension{N},HigherOrderKernel{DG},EQ,PR,KP})(x::Vararg{Number,N}) where {T,N,TCoefs,IT,KA,Axs,DG,EQ,PR,KP}
    # specialized dispatch for N-dimensional higher-order kernel
    
    # Compute i_float once per dimension
    i_floats = ntuple(d -> (x[d] - itp.knots[d][1]) / itp.h[d] + 1, N)
    
    # Find knot indices for each dimension
    pos_ids = ntuple(d -> clamp(floor(Int, i_floats[d]), itp.eqs, length(itp.knots[d]) - itp.eqs), N)
    
    # Compute normalized left distances - recompute from actual knot positions
    diff_left = ntuple(d -> (x[d] - itp.knots[d][pos_ids[d]]) / itp.h[d], N)
    diff_right = ntuple(d -> 1 - diff_left[d], N)

    idx_lower = ntuple(d -> clamp(floor(Int, diff_right[d] * (length(itp.pre_range) - 1)) + 1,
                                   1, length(itp.pre_range) - 1), N)
    
    idx_upper = ntuple(d -> idx_lower[d] + 1, N)
    t = ntuple(d -> (diff_right[d] - itp.pre_range[idx_lower[d]]) / 
                     (itp.pre_range[idx_upper[d]] - itp.pre_range[idx_lower[d]]), N)

    # Rest stays the same
    result = zero(T)
    
    for offsets in Iterators.product(ntuple(_ -> -(itp.eqs-1):itp.eqs, N)...)
        coef = itp.coefs[(pos_ids .+ offsets)...]
        
        kernel_val = one(T)
        for d in 1:N
            k_lower = itp.kernel_pre[idx_lower[d], offsets[d]+itp.eqs]
            k_upper = itp.kernel_pre[idx_upper[d], offsets[d]+itp.eqs]
            kernel_val *= (1 - t[d]) * k_lower + t[d] * k_upper
        end
        
        result += coef * kernel_val
    end
    
    return result
end
