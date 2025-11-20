"""
    (itp::FastConvolutionInterpolation{T,3,...,Val{:a0},...})(x::Number, y::Number, z::Number)

Evaluate 3D fast convolution interpolation with nearest neighbor kernel.

# Arguments
- `itp`: Fast convolution interpolation object with `:a0` kernel
- `x, y, z`: Coordinates at which to evaluate

# Returns
Value at the nearest grid point in 3D

# Performance
- O(1) evaluation time independent of grid size
- Allocation-free
- Typical timing: ~13ns per evaluation

# Details
Extends nearest neighbor to 3D by finding the nearest grid point in each dimension
independently. The fractional distances are recomputed from actual knot positions after
clamping, ensuring correct behavior at boundaries.

# Algorithm
1. Compute floating-point indices for all three dimensions
2. Clamp to valid interior ranges
3. Recompute fractional distances from actual knots
4. Select grid point based on which octant the evaluation point falls into
5. Return coefficient at nearest 3D grid point

See also: `FastConvolutionInterpolation`.
"""
@inline function (itp::FastConvolutionInterpolation{T,3,TCoefs,IT,Axs,KA,Val{3},Val{:a0},EQ,PR,KP})(x::Vararg{Number,3}) where {T,TCoefs,IT,Axs,KA,EQ,PR,KP}
    # specialized dispatch for 3d nearest neighbor kernel
        # First dimension (x)
    i_float = (x[1] - itp.knots[1][1]) / itp.h[1] + 1
    i = clamp(floor(Int, i_float), itp.eqs, length(itp.knots[1]) - itp.eqs)
    x_diff_left = (x[1] - itp.knots[1][i]) / itp.h[1]  # Recompute from actual knot

    # Second dimension (y)
    j_float = (x[2] - itp.knots[2][1]) / itp.h[2] + 1
    j = clamp(floor(Int, j_float), itp.eqs, length(itp.knots[2]) - itp.eqs)
    y_diff_left = (x[2] - itp.knots[2][j]) / itp.h[2]  # Recompute from actual knot

    # Third dimension (z)
    k_float = (x[3] - itp.knots[3][1]) / itp.h[3] + 1
    k = clamp(floor(Int, k_float), itp.eqs, length(itp.knots[3]) - itp.eqs)
    z_diff_left = (x[3] - itp.knots[3][k]) / itp.h[3]  # Recompute from actual knot

    if x_diff_left < 0.5 && y_diff_left < 0.5 && z_diff_left < 0.5
        return itp.coefs[i, j, k]
    elseif x_diff_left < 0.5 && y_diff_left < 0.5 && z_diff_left >= 0.5
        return itp.coefs[i, j, k+1]
    elseif x_diff_left < 0.5 && y_diff_left >= 0.5 && z_diff_left < 0.5
        return itp.coefs[i, j+1, k]
    elseif x_diff_left < 0.5 && y_diff_left >= 0.5 && z_diff_left >= 0.5
        return itp.coefs[i, j+1, k+1]
    elseif x_diff_left >= 0.5 && y_diff_left < 0.5 && z_diff_left < 0.5
        return itp.coefs[i+1, j, k]
    elseif x_diff_left >= 0.5 && y_diff_left < 0.5 && z_diff_left >= 0.5
        return itp.coefs[i+1, j, k+1]
    elseif x_diff_left >= 0.5 && y_diff_left >= 0.5 && z_diff_left < 0.5
        return itp.coefs[i+1, j+1, k]
    else # if x_diff_left >= 0.5 && y_diff_left >= 0.5 && z_diff_left >= 0.5
        return itp.coefs[i+1, j+1, k+1]
    end
end

"""
    (itp::FastConvolutionInterpolation{T,3,...,Val{:a1},...})(x::Number, y::Number, z::Number)

Evaluate 3D fast convolution interpolation with trilinear kernel.

# Arguments
- `itp`: Fast convolution interpolation object with `:a1` kernel
- `x, y, z`: Coordinates at which to evaluate

# Returns
Trilinearly interpolated value from the eight surrounding grid points

# Performance
- O(1) evaluation time independent of grid size
- Allocation-free
- Typical timing: ~15ns per evaluation

# Details
Uses standard trilinear interpolation with direct index calculation (no binary search).
Fractional distances are recomputed from actual knot positions after clamping to ensure
correct interpolation at domain boundaries.

# Algorithm
1. Compute floating-point indices for all three dimensions
2. Clamp to valid interior ranges
3. Recompute fractional distances from actual knots (handles boundaries correctly)
4. Trilinear interpolation from eight corners of the containing cube:
```
   f(x,y,z) = Σ w_i * f[corner_i]
```
   where weights are products of (1-w) or w for each dimension
5. Uses `@inbounds` and `@fastmath` for optimal performance

See also: `FastConvolutionInterpolation`.
"""

@inline function (itp::FastConvolutionInterpolation{T,3,TCoefs,IT,Axs,KA,Val{3},Val{:a1},EQ,PR,KP})(x::Vararg{Number,3}) where {T,TCoefs,IT,Axs,KA,EQ,PR,KP}
    # specialized dispatch for 3d linear kernel
        # First dimension (x)
    i_float = (x[1] - itp.knots[1][1]) / itp.h[1] + 1
    i = clamp(floor(Int, i_float), itp.eqs, length(itp.knots[1]) - itp.eqs)
    x_diff_left = (x[1] - itp.knots[1][i]) / itp.h[1]  # Recompute from actual knot

    # Second dimension (y)
    j_float = (x[2] - itp.knots[2][1]) / itp.h[2] + 1
    j = clamp(floor(Int, j_float), itp.eqs, length(itp.knots[2]) - itp.eqs)
    y_diff_left = (x[2] - itp.knots[2][j]) / itp.h[2]  # Recompute from actual knot

    # Third dimension (z)
    k_float = (x[3] - itp.knots[3][1]) / itp.h[3] + 1
    k = clamp(floor(Int, k_float), itp.eqs, length(itp.knots[3]) - itp.eqs)
    z_diff_left = (x[3] - itp.knots[3][k]) / itp.h[3]  # Recompute from actual knot

    # Trilinear interpolation formula
    return @inbounds @fastmath (1-x_diff_left)*(1-y_diff_left)*(1-z_diff_left)*itp.coefs[i, j, k] + 
                    x_diff_left*(1-y_diff_left)*(1-z_diff_left)*itp.coefs[i+1, j, k] + 
                    (1-x_diff_left)*y_diff_left*(1-z_diff_left)*itp.coefs[i, j+1, k] + 
                    x_diff_left*y_diff_left*(1-z_diff_left)*itp.coefs[i+1, j+1, k] +
                    (1-x_diff_left)*(1-y_diff_left)*z_diff_left*itp.coefs[i, j, k+1] + 
                    x_diff_left*(1-y_diff_left)*z_diff_left*itp.coefs[i+1, j, k+1] + 
                    (1-x_diff_left)*y_diff_left*z_diff_left*itp.coefs[i, j+1, k+1] + 
                    x_diff_left*y_diff_left*z_diff_left*itp.coefs[i+1, j+1, k+1]
end

"""
    (itp::FastConvolutionInterpolation{T,3,...,HigherOrderKernel{DG},...})(x::Number, y::Number, z::Number)

Evaluate 3D fast convolution interpolation with higher-order kernel (`:a3`, `:b3`, `:b5`, etc.).

# Arguments
- `itp`: Fast convolution interpolation object with higher-order kernel
- `x, y, z`: Coordinates at which to evaluate

# Returns
Interpolated value using precomputed 3D kernel (tensor product)

# Performance
- O(1) evaluation time independent of grid size
- Allocation-free
- Typical timings: `:a3` ~155ns, `:b5` ~1773ns per evaluation
- Performance scales with (2*eqs-1)³ kernel support, not grid size

# Details
Achieves high-order accuracy in 3D with O(1) performance by:
1. Computing grid indices directly in all dimensions (no binary search)
2. Using precomputed 1D kernel values for each dimension
3. Forming 3D kernel as tensor product of 1D kernels: K₃(x,y,z) = K₁(x)·K₁(y)·K₁(z)
4. Trilinear interpolation between eight precomputed kernel positions
5. Recomputing fractional distances from actual knots (correct boundary handling)

The tensor product structure enables efficient evaluation despite the high-dimensional
convolution operation.

# Algorithm
1. Direct index calculation for all three dimensions
2. Recompute fractional distances from actual knots (handles clamping)
3. Lookup precomputed kernel values at neighboring positions in all dimensions
4. Compute 3D convolution at eight kernel positions (tensor product):
   - Eight corners: (idx_x±1, idx_y±1, idx_z±1)
   - Each corner: `result = Σ_{l,m,n} coef[i+l,j+m,k+n] * Kₓ(l) * Kᵧ(m) * Kᵤ(n)`
5. Trilinear interpolation between the eight results using weights `(t_x, t_y, t_z)`
6. Returns final interpolated value

# Mathematical Detail
The evaluation computes:
```
result = Σ_{l,m,n} coef[i+l, j+m, k+n] * K_x(l, t_x) * K_y(m, t_y) * K_z(n, t_z)
```
where each K is linearly interpolated from precomputed kernel values.

The algorithm accumulates all 8 corner results simultaneously in a single triple loop,
minimizing memory accesses and maximizing cache efficiency.

See also: `FastConvolutionInterpolation`, `ConvolutionInterpolation`
"""

function (itp::FastConvolutionInterpolation{T,3,TCoefs,IT,Axs,KA,Val{3},HigherOrderKernel{DG},EQ,PR,KP})(x::Vararg{Number,3}) where {T,TCoefs,IT,Axs,KA,DG,EQ,PR,KP}
    # specialized dispatch for 3d higher order kernels (cubic and above)

    # First dimension (x)
    i_float = (x[1] - itp.knots[1][1]) / itp.h[1] + 1
    i = clamp(floor(Int, i_float), itp.eqs, length(itp.knots[1]) - itp.eqs)
    x_diff_left = (x[1] - itp.knots[1][i]) / itp.h[1]  # Recompute from actual knot
    x_diff_right = 1 - x_diff_left
    idx_x = clamp(floor(Int, x_diff_right * (length(itp.pre_range) - 1)) + 1, 
                1, length(itp.pre_range) - 1)
    idx_x_next = idx_x + 1
    t_x = (x_diff_right - itp.pre_range[idx_x]) / (itp.pre_range[idx_x_next] - itp.pre_range[idx_x])

    # Second dimension (y)
    j_float = (x[2] - itp.knots[2][1]) / itp.h[2] + 1
    j = clamp(floor(Int, j_float), itp.eqs, length(itp.knots[2]) - itp.eqs)
    y_diff_left = (x[2] - itp.knots[2][j]) / itp.h[2]  # Recompute from actual knot
    y_diff_right = 1 - y_diff_left
    idx_y = clamp(floor(Int, y_diff_right * (length(itp.pre_range) - 1)) + 1, 
                1, length(itp.pre_range) - 1)
    idx_y_next = idx_y + 1
    t_y = (y_diff_right - itp.pre_range[idx_y]) / (itp.pre_range[idx_y_next] - itp.pre_range[idx_y])

    # Third dimension (z)
    k_float = (x[3] - itp.knots[3][1]) / itp.h[3] + 1
    k = clamp(floor(Int, k_float), itp.eqs, length(itp.knots[3]) - itp.eqs)
    z_diff_left = (x[3] - itp.knots[3][k]) / itp.h[3]  # Recompute from actual knot
    z_diff_right = 1 - z_diff_left
    idx_z = clamp(floor(Int, z_diff_right * (length(itp.pre_range) - 1)) + 1, 
                1, length(itp.pre_range) - 1)
    idx_z_next = idx_z + 1
    t_z = (z_diff_right - itp.pre_range[idx_z]) / (itp.pre_range[idx_z_next] - itp.pre_range[idx_z])
    
    # Initialize all 8 corner results
    result_000 = zero(T)
    result_100 = zero(T)
    result_010 = zero(T)
    result_110 = zero(T)
    result_001 = zero(T)
    result_101 = zero(T)
    result_011 = zero(T)
    result_111 = zero(T)
    
    # Single triple loop - accumulate all 8 corners simultaneously
    for n in -(itp.eqs-1):itp.eqs
        kz_0 = itp.kernel_pre[idx_z, n+itp.eqs]
        kz_1 = itp.kernel_pre[idx_z_next, n+itp.eqs]
        
        for m in -(itp.eqs-1):itp.eqs
            ky_0 = itp.kernel_pre[idx_y, m+itp.eqs]
            ky_1 = itp.kernel_pre[idx_y_next, m+itp.eqs]
            
            for l in -(itp.eqs-1):itp.eqs
                coef = itp.coefs[i+l, j+m, k+n]
                kx_0 = itp.kernel_pre[idx_x, l+itp.eqs]
                kx_1 = itp.kernel_pre[idx_x_next, l+itp.eqs]
                
                # Accumulate all 8 corners
                result_000 += coef * kx_0 * ky_0 * kz_0
                result_100 += coef * kx_1 * ky_0 * kz_0
                result_010 += coef * kx_0 * ky_1 * kz_0
                result_110 += coef * kx_1 * ky_1 * kz_0
                result_001 += coef * kx_0 * ky_0 * kz_1
                result_101 += coef * kx_1 * ky_0 * kz_1
                result_011 += coef * kx_0 * ky_1 * kz_1
                result_111 += coef * kx_1 * ky_1 * kz_1
            end
        end
    end
    
    # Trilinear interpolation formula
    return @fastmath (1-t_x)*(1-t_y)*(1-t_z)*result_000 + 
            t_x*(1-t_y)*(1-t_z)*result_100 + 
            (1-t_x)*t_y*(1-t_z)*result_010 + 
            t_x*t_y*(1-t_z)*result_110 +
            (1-t_x)*(1-t_y)*t_z*result_001 + 
            t_x*(1-t_y)*t_z*result_101 + 
            (1-t_x)*t_y*t_z*result_011 + 
            t_x*t_y*t_z*result_111
end