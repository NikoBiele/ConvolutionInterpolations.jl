"""
    (itp::FastConvolutionInterpolation{T,1,...,Val{:a0},...})(x::Number)

Evaluate 1D fast convolution interpolation with nearest neighbor kernel.

# Arguments
- `itp`: Fast convolution interpolation object with `:a0` kernel
- `x`: Coordinate at which to evaluate

# Returns
Value at the nearest grid point

# Performance
- O(1) evaluation time independent of grid size
- Allocation-free
- Typical timing: ~4ns per evaluation

# Details
Uses direct index calculation to find the nearest grid point without binary search.
The implementation recomputes the fractional distance from the actual knot position
after clamping, ensuring correctness even at domain boundaries.

# Algorithm
1. Compute floating-point index directly from coordinate
2. Clamp to valid interior range
3. Recompute fractional distance from actual knot (handles boundary clamping correctly)
4. Round to nearest integer based on fractional part < 0.5
5. Return coefficient at nearest grid point

See also: `FastConvolutionInterpolation`.
"""

@inline function (itp::FastConvolutionInterpolation{T,1,TCoefs,IT,Axs,KA,Val{1},Val{:a0},EQ,PR,KP})(x::Vararg{Number,1}) where {T,TCoefs,IT,Axs,KA,EQ,PR,KP}
    # specialized dispatch for 1d nearest neighbor kernel
    i_float = (x[1] - itp.knots[1][1]) / itp.h[1] + 1 # +1 for 1-based indexing
    i = clamp(floor(Int, i_float), itp.eqs, length(itp.knots[1]) - itp.eqs)
    x_diff_left = (x[1] - itp.knots[1][i]) / itp.h[1]  # Guaranteed in [0,1]
    if x_diff_left < 0.5
        return itp.coefs[i]
    else
        return itp.coefs[i+1]
    end
end

"""
    (itp::FastConvolutionInterpolation{T,1,...,Val{:a1},...})(x::Number)

Evaluate 1D fast convolution interpolation with linear kernel.

# Arguments
- `itp`: Fast convolution interpolation object with `:a1` kernel
- `x`: Coordinate at which to evaluate

# Returns
Linearly interpolated value between adjacent grid points

# Performance
- O(1) evaluation time independent of grid size
- Allocation-free
- Typical timing: ~4ns per evaluation
- Competitive with or faster than other Julia interpolation packages

# Details
Uses direct index calculation without binary search, achieving constant-time evaluation.
The fractional distance is recomputed from the actual knot position after clamping,
ensuring correct interpolation even when points are very close to boundaries.

# Algorithm
1. Compute floating-point index directly from coordinate
2. Clamp to valid interior range
3. Recompute fractional distance from actual knot (handles boundary clamping correctly)
4. Linear interpolation: `(1-w)*f[i] + w*f[i+1]` where w is the fractional distance
5. Uses `@inbounds` and `@fastmath` for optimal performance

See also: `FastConvolutionInterpolation`.
"""

@inline function (itp::FastConvolutionInterpolation{T,1,TCoefs,IT,Axs,KA,Val{1},Val{:a1},EQ,PR,KP})(x::Vararg{Number,1}) where {T,TCoefs,IT,Axs,KA,EQ,PR,KP}
    # specialized dispatch for 1d linear kernel
    i_float = (x[1] - itp.knots[1][1]) / itp.h[1] + 1  # +1 for 1-based indexing
    i = clamp(floor(Int, i_float), itp.eqs, length(itp.knots[1]) - itp.eqs)
    x_diff_left = (x[1] - itp.knots[1][i]) / itp.h[1]  # Guaranteed in [0,1]
    return @inbounds @fastmath (1-x_diff_left) * itp.coefs[i] + x_diff_left * itp.coefs[i+1]
end

"""
    (itp::FastConvolutionInterpolation{T,1,...,HigherOrderKernel{DG},...})(x::Number)

Evaluate 1D fast convolution interpolation with higher-order kernel (`:a3`, `:b3`, `:b5`, etc.).

# Arguments
- `itp`: Fast convolution interpolation object with higher-order kernel
- `x`: Coordinate at which to evaluate

# Returns
Interpolated value using precomputed kernel values

# Performance
- O(1) evaluation time independent of grid size
- Allocation-free
- Typical timings: `:a3` ~12ns, `:b5` ~14ns per evaluation
- Scales with kernel support size (2*eqs-1 points), not grid size

# Details
This method achieves high-order accuracy with O(1) performance by:
1. Computing grid indices directly (no binary search)
2. Using precomputed kernel values stored in `kernel_pre`
3. Linear interpolation between precomputed kernel values for smoothness
4. Recomputing fractional distances from actual knot positions (correct boundary handling)

The precomputed kernel approach trades initialization time and memory for significantly
faster evaluation, typically 3-10× faster than direct kernel evaluation.

# Algorithm
1. Direct index calculation to find bracketing grid points
2. Recompute fractional distance from actual knot (handles clamping at boundaries)
3. Lookup precomputed kernel values at two neighboring positions
4. Compute weighted sum of coefficients with kernel values (convolution)
5. Linear interpolation between the two convolution results
6. Returns final interpolated value

# Mathematical Detail
For each kernel position `j`, the algorithm evaluates:
```
result = (1-t) * Σ(coef[i+j] * kernel[idx, j]) + t * Σ(coef[i+j] * kernel[idx+1, j])
```
where `t` is the linear interpolation weight between precomputed kernel positions.

See also: `FastConvolutionInterpolation`, `ConvolutionInterpolation`.
"""

@inline function (itp::FastConvolutionInterpolation{T,1,TCoefs,IT,Axs,KA,Val{1},HigherOrderKernel{DG},EQ,PR,KP})(x::Vararg{Number,1}) where {T,TCoefs,IT,Axs,KA,DG,EQ,PR,KP}
    # specialized dispatch for 1d higher-order kernels

    # Direct index calculation
    i_float = (x[1] - itp.knots[1][1]) / itp.h[1] + 1
    i = clamp(floor(Int, i_float), itp.eqs, length(itp.knots[1]) - itp.eqs)

    # Recompute diff_left from actual knot position, not from clamped index
    x_diff_left = (x[1] - itp.knots[1][i]) / itp.h[1]  # Guaranteed in [0,1]
    x_diff_right = 1 - x_diff_left

    # Rest stays the same
    idx = clamp(floor(Int, x_diff_right * length(itp.pre_range)) + 1, 1, length(itp.pre_range) - 1)
    idx_next = idx + 1
    t = (x_diff_right - itp.pre_range[idx]) / (itp.pre_range[idx_next] - itp.pre_range[idx])
    
    # Single pass: accumulate both results simultaneously
    result_lower = zero(T)
    result_upper = zero(T)
    
    @inbounds for j in -(itp.eqs-1):itp.eqs
        coef = itp.coefs[i+j]
        k_lower = itp.kernel_pre[idx, j+itp.eqs]
        k_upper = itp.kernel_pre[idx_next, j+itp.eqs]
        
        result_lower += coef * k_lower
        result_upper += coef * k_upper
    end

    return @fastmath (1-t) * result_lower + t * result_upper
end