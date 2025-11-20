"""
    (itp::FastConvolutionInterpolation{T,2,...,Val{:a0},...})(x::Number, y::Number)

Evaluate 2D fast convolution interpolation with nearest neighbor kernel.

# Arguments
- `itp`: Fast convolution interpolation object with `:a0` kernel
- `x, y`: Coordinates at which to evaluate

# Returns
Value at the nearest grid point in 2D

# Performance
- O(1) evaluation time independent of grid size
- Allocation-free
- Typical timing: ~7ns per evaluation

# Details
Extends 1D nearest neighbor to 2D by finding the nearest grid point in each dimension
independently. The fractional distances are recomputed from actual knot positions after
clamping, ensuring correct behavior at boundaries.

# Algorithm
1. Compute floating-point indices for both dimensions
2. Clamp to valid interior ranges
3. Recompute fractional distances from actual knots
4. Select grid point based on which quadrant the evaluation point falls into
5. Return coefficient at nearest 2D grid point

See also: `FastConvolutionInterpolation`.
"""

@inline function (itp::FastConvolutionInterpolation{T,2,TCoefs,IT,Axs,KA,Val{2},Val{:a0},EQ,PR,KP})(x::Vararg{Number,2}) where {T,TCoefs,IT,Axs,KA,EQ,PR,KP}
    # specialized dispatch for 2d nearest neighbor kernel
    
    # First dimension (x)
    i_float = (x[1] - itp.knots[1][1]) / itp.h[1] + 1
    i = clamp(floor(Int, i_float), itp.eqs, length(itp.knots[1]) - itp.eqs)
    x_diff_left = (x[1] - itp.knots[1][i]) / itp.h[1]  # Recompute from actual knot

    # Second dimension (y)
    j_float = (x[2] - itp.knots[2][1]) / itp.h[2] + 1
    j = clamp(floor(Int, j_float), itp.eqs, length(itp.knots[2]) - itp.eqs)
    y_diff_left = (x[2] - itp.knots[2][j]) / itp.h[2]  # Recompute from actual knot

    if x_diff_left < 0.5 && y_diff_left < 0.5
        return itp.coefs[i, j]
    elseif x_diff_left < 0.5 && y_diff_left >= 0.5
        return itp.coefs[i, j+1]
    elseif x_diff_left >= 0.5 && y_diff_left < 0.5
        return itp.coefs[i+1, j]
    else # if x_diff_left >= 0.5 && y_diff_left >= 0.5
        return itp.coefs[i+1, j+1]
    end
end

"""
    (itp::FastConvolutionInterpolation{T,2,...,Val{:a1},...})(x::Number, y::Number)

Evaluate 2D fast convolution interpolation with bilinear kernel.

# Arguments
- `itp`: Fast convolution interpolation object with `:a1` kernel
- `x, y`: Coordinates at which to evaluate

# Returns
Bilinearly interpolated value from the four surrounding grid points

# Performance
- O(1) evaluation time independent of grid size
- Allocation-free
- Typical timing: ~8ns per evaluation

# Details
Uses standard bilinear interpolation with direct index calculation (no binary search).
Fractional distances are recomputed from actual knot positions after clamping to ensure
correct interpolation at domain boundaries.

# Algorithm
1. Compute floating-point indices for both dimensions
2. Clamp to valid interior ranges
3. Recompute fractional distances from actual knots (handles boundaries correctly)
4. Bilinear interpolation from four corners:
```
   f(x,y) = (1-wx)(1-wy)*f[i,j] + wx(1-wy)*f[i+1,j] + 
            (1-wx)wy*f[i,j+1] + wx*wy*f[i+1,j+1]
```
5. Uses `@inbounds` and `@fastmath` for optimal performance

See also: `FastConvolutionInterpolation`.
"""

@inline function (itp::FastConvolutionInterpolation{T,2,TCoefs,IT,Axs,KA,Val{2},Val{:a1},EQ,PR,KP})(x::Vararg{Number,2}) where {T,TCoefs,IT,Axs,KA,EQ,PR,KP}
    # specialized dispatch for 2d linear kernel
    # First dimension (x)
    i_float = (x[1] - itp.knots[1][1]) / itp.h[1] + 1
    i = clamp(floor(Int, i_float), itp.eqs, length(itp.knots[1]) - itp.eqs)
    x_diff_left = (x[1] - itp.knots[1][i]) / itp.h[1]  # Recompute from actual knot

    # Second dimension (y)
    j_float = (x[2] - itp.knots[2][1]) / itp.h[2] + 1
    j = clamp(floor(Int, j_float), itp.eqs, length(itp.knots[2]) - itp.eqs)
    y_diff_left = (x[2] - itp.knots[2][j]) / itp.h[2]  # Recompute from actual knot

    # Bilinear interpolation
    return @inbounds @fastmath (1-x_diff_left)*(1-y_diff_left)*itp.coefs[i, j] + 
                        x_diff_left*(1-y_diff_left)*itp.coefs[i+1, j] + 
                        (1-x_diff_left)*y_diff_left*itp.coefs[i, j+1] + 
                        x_diff_left*y_diff_left*itp.coefs[i+1, j+1]
end

"""
    (itp::FastConvolutionInterpolation{T,2,...,HigherOrderKernel{DG},...})(x::Number, y::Number)

Evaluate 2D fast convolution interpolation with higher-order kernel (`:a3`, `:b3`, `:b5`, etc.).

# Arguments
- `itp`: Fast convolution interpolation object with higher-order kernel
- `x, y`: Coordinates at which to evaluate

# Returns
Interpolated value using precomputed 2D kernel (tensor product)

# Performance
- O(1) evaluation time independent of grid size
- Allocation-free
- Typical timings: `:a3` ~35ns, `:b5` ~88ns per evaluation
- Performance scales with (2*eqs-1)² kernel support, not grid size

# Details
Achieves high-order accuracy in 2D with O(1) performance by:
1. Computing grid indices directly in both dimensions (no binary search)
2. Using precomputed 1D kernel values for each dimension
3. Forming 2D kernel as tensor product of 1D kernels
4. Bilinear interpolation between four precomputed kernel positions
5. Recomputing fractional distances from actual knots (correct boundary handling)

The tensor product structure means the 2D kernel is formed by multiplying 1D kernel
values: `K_2D(x,y) = K_1D(x) * K_1D(y)`, enabling efficient evaluation.

# Algorithm
1. Direct index calculation for both dimensions
2. Recompute fractional distances from actual knots (handles clamping)
3. Lookup precomputed kernel values at neighboring positions in both dimensions
4. Compute 2D convolution at four kernel positions (tensor product):
   - `result_00`: kernels at (idx_x, idx_y)
   - `result_10`: kernels at (idx_x+1, idx_y)
   - `result_01`: kernels at (idx_x, idx_y+1)
   - `result_11`: kernels at (idx_x+1, idx_y+1)
5. Bilinear interpolation between the four results using weights `(t_x, t_y)`
6. Returns final interpolated value

# Mathematical Detail
The evaluation computes:
```
result = Σ_{l,m} coef[i+l, j+m] * K_x(l, t_x) * K_y(m, t_y)
```
where K_x and K_y are linearly interpolated from precomputed kernel values.

See also: `FastConvolutionInterpolation`, `ConvolutionInterpolation`.
"""

function (itp::FastConvolutionInterpolation{T,2,TCoefs,IT,Axs,KA,Val{2},HigherOrderKernel{DG},EQ,PR,KP})(x::Vararg{Number,2}) where {T,TCoefs,IT,Axs,KA,DG,EQ,PR,KP}
    # specialized dispatch for 2d higher-order kernels
    
    # First dimension (x)
    i_float = (x[1] - itp.knots[1][1]) / itp.h[1] + 1
    i = clamp(floor(Int, i_float), itp.eqs, length(itp.knots[1]) - itp.eqs)
    x_diff_left = (x[1] - itp.knots[1][i]) / itp.h[1]  # Recompute from actual knot
    x_diff_right = 1 - x_diff_left
    # Direct pre_range index - no search
    idx_x = clamp(floor(Int, x_diff_right * (length(itp.pre_range) - 1)) + 1, 
                1, length(itp.pre_range) - 1)
    # Linear interpolation weight
    t_x = (x_diff_right - itp.pre_range[idx_x]) / (itp.pre_range[idx_x+1] - itp.pre_range[idx_x])

    # Second dimension (y)
    j_float = (x[2] - itp.knots[2][1]) / itp.h[2] + 1
    j = clamp(floor(Int, j_float), itp.eqs, length(itp.knots[2]) - itp.eqs)
    y_diff_left = (x[2] - itp.knots[2][j]) / itp.h[2]  # Recompute from actual knot
    y_diff_right = 1 - y_diff_left 
    # Direct pre_range index - no search
    idx_y = clamp(floor(Int, y_diff_right * (length(itp.pre_range) - 1)) + 1, 
                1, length(itp.pre_range) - 1)
    # Linear interpolation weight
    t_y = (y_diff_right - itp.pre_range[idx_y]) / (itp.pre_range[idx_y+1] - itp.pre_range[idx_y])
    
    # Initialize results
    result_00 = zero(T)
    result_10 = zero(T)
    result_01 = zero(T)
    result_11 = zero(T)

    # Single pass through the convolution support
    for m in -(itp.eqs-1):itp.eqs
        ky_0 = itp.kernel_pre[idx_y, m+itp.eqs]
        ky_1 = itp.kernel_pre[idx_y+1, m+itp.eqs]
        
        for l in -(itp.eqs-1):itp.eqs
            coef = itp.coefs[i+l, j+m]
            kx_0 = itp.kernel_pre[idx_x, l+itp.eqs]
            kx_1 = itp.kernel_pre[idx_x+1, l+itp.eqs]
            
            # Accumulate all 4 corners - pure tensor product
            result_00 += coef * kx_0 * ky_0
            result_10 += coef * kx_1 * ky_0
            result_01 += coef * kx_0 * ky_1
            result_11 += coef * kx_1 * ky_1
        end
    end

    # Bilinear interpolation
    return @fastmath (1-t_x)*(1-t_y)*result_00 + t_x*(1-t_y)*result_10 + (1-t_x)*t_y*result_01 + t_x*t_y*result_11
end