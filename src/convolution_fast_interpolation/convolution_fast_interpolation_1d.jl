"""
(itp::FastConvolutionInterpolation{T,1,...})(x::Number)
Evaluate 1D fast convolution interpolation at coordinate x.
Dispatches on kernel type and subgrid parameter:
Specialized kernels (no precomputed table):

:a0 — Nearest neighbor, ~4ns
:a1 — Linear interpolation, ~4ns

Higher-order kernels with subgrid modes:

:nearest — Nearest precomputed convolution result
:linear — Linear interpolation between two convolution results (needs precompute≥10_000)
:cubic — Cubic Hermite using kernel first derivatives (default, 4 sums)
:quintic — Quintic Hermite using first and second derivatives (6 sums, highest accuracy)

O(1) evaluation time, allocation-free. Higher subgrid orders use analytically
predifferentiated kernel coefficients for Hermite interpolation of the convolution
results. Available subgrid depends on remaining smooth derivatives:
max_smooth_derivative[kernel] - derivative.
Results scaled by (-1/h)^derivative for derivative evaluation.
See also: FastConvolutionInterpolation, cubic_hermite, quintic_hermite.
"""

@inline function (itp::FastConvolutionInterpolation{T,1,TCoefs,IT,Axs,KA,Val{1},Val{:a0},
                EQ,PR,KP,KBC,DO,FD,SD,Val{:not_used}})(x::Vararg{Number,1}) where
                {T,TCoefs,IT,Axs,KA,EQ,PR,KP,KBC,DO,FD,SD}

    # specialized dispatch for 1d nearest neighbor kernel
    i_float = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T) # +1 for 1-based indexing
    i = clamp(floor(Int, i_float), itp.eqs, length(itp.knots[1]) - itp.eqs)
    x_diff_left = (x[1] - itp.knots[1][i]) / itp.h[1]  # Guaranteed in [0,1]
    if x_diff_left < 0.5
        return itp.coefs[i]
    else
        return itp.coefs[i+1]
    end
end

@inline function (itp::FastConvolutionInterpolation{T,1,TCoefs,IT,Axs,KA,Val{1},Val{:a1},
                    EQ,PR,KP,KBC,DO,FD,SD,Val{:not_used}})(x::Vararg{Number,1}) where 
                    {T,TCoefs,IT,Axs,KA,EQ,PR,KP,KBC,DO,FD,SD}

    # specialized dispatch for 1d linear kernel
    i_float = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T)  # +1 for 1-based indexing
    i = clamp(floor(Int, i_float), itp.eqs, length(itp.knots[1]) - itp.eqs)
    x_diff_left = (x[1] - itp.knots[1][i]) / itp.h[1]  # Guaranteed in [0,1]
    return @inbounds ((1-x_diff_left) * itp.coefs[i] + x_diff_left * itp.coefs[i+1])
end

@inline function (itp::FastConvolutionInterpolation{T,1,TCoefs,IT,Axs,KA,Val{1},
                    HigherOrderKernel{DG},EQ,PR,KP,KBC,DO,FD,SD,Val{:nearest}})(x::Vararg{Number,1}) where 
                    {T,TCoefs,IT,Axs,KA,DG,EQ,PR,KP,KBC,DO,FD,SD}
 
    # Direct index calculation
    i_float = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T)
    i = clamp(floor(Int, i_float), itp.eqs, length(itp.knots[1]) - itp.eqs)
    x_diff_left = (x[1] - itp.knots[1][i]) / itp.h[1]
    x_diff_right = one(T) - x_diff_left

    # Index into precomputed table
    n_pre = length(itp.pre_range)
    continuous_idx = x_diff_right * T(n_pre - 1) + one(T)
    idx = clamp(floor(Int, continuous_idx), 1, n_pre - 1)
    t = continuous_idx - T(idx)

    if t > T(1)/T(2)
        idx = idx + one(Int64)
    end

    # convolve nearest neighbor with kernel
    result = zero(T)
    
    @inbounds for j in -(itp.eqs-1):itp.eqs
        coef = itp.coefs[i+j]
        kernel = itp.kernel_pre[idx, j+itp.eqs]
        
        result += coef * kernel
    end

    # use the nearest neighbor directly
    return result * (-one(T)/itp.h[1])^(DO.parameters[1])
end

@inline function (itp::FastConvolutionInterpolation{T,1,TCoefs,IT,Axs,KA,Val{1},
                    HigherOrderKernel{DG},EQ,PR,KP,KBC,DO,FD,SD,Val{:linear}})(x::Vararg{Number,1}) where 
                    {T,TCoefs,IT,Axs,KA,DG,EQ,PR,KP,KBC,DO,FD,SD}

    # Direct index calculation
    i_float = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T)
    i = clamp(floor(Int, i_float), itp.eqs, length(itp.knots[1]) - itp.eqs)
    x_diff_left = (x[1] - itp.knots[1][i]) / itp.h[1]
    x_diff_right = one(T) - x_diff_left

    # Index into precomputed table
    n_pre = length(itp.pre_range)
    continuous_idx = x_diff_right * T(n_pre - 1) + one(T)
    idx = clamp(floor(Int, continuous_idx), 1, n_pre - 1)
    idx_next = idx + 1
    t = continuous_idx - T(idx)
    
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

    # linear interpolation
    return ((one(T)-t) * result_lower + t * result_upper) * (-one(T)/itp.h[1])^(DO.parameters[1])
end

@inline function (itp::FastConvolutionInterpolation{T,1,TCoefs,IT,Axs,KA,Val{1},
                    HigherOrderKernel{DG},EQ,PR,KP,KBC,DO,FD,SD,Val{:cubic}})(x::Vararg{Number,1}) where 
                    {T,TCoefs,IT,Axs,KA,DG,EQ,PR,KP,KBC,DO,FD,SD}

    # Direct index calculation
    i_float = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T)
    i = clamp(floor(Int, i_float), itp.eqs, length(itp.knots[1]) - itp.eqs)
    x_diff_left = (x[1] - itp.knots[1][i]) / itp.h[1]

    # move on to higher order subgrids
    x_diff_right = one(T) - x_diff_left

    # Index into precomputed table
    n_pre = length(itp.pre_range)
    continuous_idx = x_diff_right * T(n_pre - 1) + one(T)
    idx = clamp(floor(Int, continuous_idx), 1, n_pre - 1)
    idx_next = idx + 1
    t = continuous_idx - T(idx)
    
    # convolve kernel values and derivatives with data
    sum_f0 = zero(T)
    sum_f1 = zero(T)
    sum_di0 = zero(T)
    sum_di1 = zero(T)

    # Single pass: accumulate both results simultaneously
    @inbounds for j in -(itp.eqs-1):itp.eqs
        coef = itp.coefs[i+j]
        col = j + itp.eqs
        
        sum_f0 += coef * itp.kernel_pre[idx, col]
        sum_f1 += coef * itp.kernel_pre[idx_next, col]
        sum_di0 += coef * itp.kernel_d1_pre[idx, col]
        sum_di1 += coef * itp.kernel_d1_pre[idx_next, col]
    end
    # Cubic Hermite interpolation
    h_pre = one(T) / T(n_pre - 1)
    return cubic_hermite(t, sum_f0, sum_f1, sum_di0, sum_di1, h_pre) * (-one(T)/itp.h[1])^(DO.parameters[1])
end

@inline function (itp::FastConvolutionInterpolation{T,1,TCoefs,IT,Axs,KA,Val{1},
                    HigherOrderKernel{DG},EQ,PR,KP,KBC,DO,FD,SD,Val{:quintic}})(x::Vararg{Number,1}) where 
                    {T,TCoefs,IT,Axs,KA,DG,EQ,PR,KP,KBC,DO,FD,SD}

    # Direct index calculation
    i_float = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T)
    i = clamp(floor(Int, i_float), itp.eqs, length(itp.knots[1]) - itp.eqs)
    x_diff_left = (x[1] - itp.knots[1][i]) / itp.h[1]

    # move on to higher order subgrids
    x_diff_right = one(T) - x_diff_left

    # Index into precomputed table
    n_pre = length(itp.pre_range)
    continuous_idx = x_diff_right * T(n_pre - 1) + one(T)
    idx = clamp(floor(Int, continuous_idx), 1, n_pre - 1)
    idx_next = idx + 1
    t = continuous_idx - T(idx)

    # convolve kernel values and derivatives with data
    sum_f0 = zero(T)
    sum_f1 = zero(T)
    sum_d0 = zero(T)
    sum_d1 = zero(T)
    sum_dd0 = zero(T)
    sum_dd1 = zero(T)

    @inbounds for j in -(itp.eqs-1):itp.eqs
        coef = itp.coefs[i+j]
        col = j + itp.eqs
        
        sum_f0 += coef * itp.kernel_pre[idx, col]
        sum_f1 += coef * itp.kernel_pre[idx_next, col]
        sum_d0 += coef * itp.kernel_d1_pre[idx, col]
        sum_d1 += coef * itp.kernel_d1_pre[idx_next, col]
        sum_dd0 += coef * itp.kernel_d2_pre[idx, col]
        sum_dd1 += coef * itp.kernel_d2_pre[idx_next, col]
    end

    # Quintic Hermite interpolation
    h_pre = one(T) / T(n_pre - 1)
    return quintic_hermite(t, sum_f0, sum_f1, sum_d0, sum_d1, sum_dd0, sum_dd1, h_pre) * (-one(T)/itp.h[1])^abs(DO.parameters[1])
end