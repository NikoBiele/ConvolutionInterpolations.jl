@inline function (itp::FastConvolutionInterpolation{T,1,TCoefs,IT,Axs,KA,Val{1},
                    HigherOrderKernel{DG},EQ,PR,KP,KBC,IntegralOrder,FD,SD,Val{:quintic}})(x::Vararg{Number,1}) where 
                    {T,TCoefs,IT,Axs,KA,DG,EQ,PR,KP,KBC,FD,SD}

    eqs_int = itp.eqs
    n_pre = length(itp.pre_range)
    h_pre = one(T) / T(n_pre - 1)
    result = zero(T)
    i_float  = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T)
    i        = clamp(floor(Int, i_float), eqs_int, length(itp.coefs) - eqs_int)

    @inbounds for j in (i - eqs_int + 1):(i + eqs_int)
    # @inbounds for j in 1:length(itp.coefs)
        xj = itp.knots[1][eqs_int] + (j - eqs_int) * itp.h[1]
        sj = (x[1] - xj) / itp.h[1]

        if abs(sj) >= T(eqs_int)
            kt = T(1//2) * T(sign(sj))
        else
            col_float = T(eqs_int) + sj
            col = clamp(floor(Int, col_float) + 1, 1, 2 * eqs_int)
            x_diff_right = col_float - T(col - 1)
            continuous_idx = x_diff_right * T(n_pre - 1) + one(T)
            idx      = clamp(floor(Int, continuous_idx), 1, n_pre - 1)
            idx_next = idx + 1
            t        = continuous_idx - T(idx)
            kt = quintic_hermite(t,
                    itp.kernel_pre[idx,    col], itp.kernel_pre[idx_next,    col],
                    itp.kernel_d1_pre[idx, col], itp.kernel_d1_pre[idx_next, col],
                    itp.kernel_d2_pre[idx, col], itp.kernel_d2_pre[idx_next, col],
                    h_pre)
        end

        result += itp.coefs[j] * (kt - itp.left_values[1][j])
    end

    left_tail  = (i - eqs_int) >= 1                      ? itp.cumcoefs_left[1][i - eqs_int]      : zero(T)
    right_tail = (i + eqs_int + 1) <= length(itp.coefs)  ? itp.cumcoefs_right[1][i + eqs_int + 1] : zero(T)

    return (result + left_tail + right_tail) * itp.h[1]
end

@inline function (itp::FastConvolutionInterpolation{T,1,TCoefs,IT,Axs,KA,Val{1},
                    HigherOrderKernel{DG},EQ,PR,KP,KBC,IntegralOrder,FD,SD,Val{:cubic}})(x::Vararg{Number,1}) where 
                    {T,TCoefs,IT,Axs,KA,DG,EQ,PR,KP,KBC,FD,SD}

    eqs_int = itp.eqs
    n_pre = length(itp.pre_range)
    h_pre = one(T) / T(n_pre - 1)
    result = zero(T)
    i_float  = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T)
    i        = clamp(floor(Int, i_float), eqs_int, length(itp.coefs) - eqs_int)

    @inbounds for j in (i - eqs_int + 1):(i + eqs_int)
    # @inbounds for j in 1:length(itp.coefs)
        xj = itp.knots[1][eqs_int] + (j - eqs_int) * itp.h[1]
        sj = (x[1] - xj) / itp.h[1]

        if abs(sj) >= T(eqs_int)
            kt = T(1//2) * T(sign(sj))
        else
            col_float = T(eqs_int) + sj
            col = clamp(floor(Int, col_float) + 1, 1, 2 * eqs_int)
            x_diff_right = col_float - T(col - 1)
            continuous_idx = x_diff_right * T(n_pre - 1) + one(T)
            idx      = clamp(floor(Int, continuous_idx), 1, n_pre - 1)
            idx_next = idx + 1
            t        = continuous_idx - T(idx)
            kt = cubic_hermite(t,
                    itp.kernel_pre[idx,    col], itp.kernel_pre[idx_next,    col],
                    itp.kernel_d1_pre[idx, col], itp.kernel_d1_pre[idx_next, col],
                    h_pre)
        end

        result += itp.coefs[j] * (kt - itp.left_values[1][j])
    end

    left_tail  = (i - eqs_int) >= 1                      ? itp.cumcoefs_left[1][i - eqs_int]      : zero(T)
    right_tail = (i + eqs_int + 1) <= length(itp.coefs)  ? itp.cumcoefs_right[1][i + eqs_int + 1] : zero(T)

    return (result + left_tail + right_tail) * itp.h[1]
end

@inline function (itp::FastConvolutionInterpolation{T,1,TCoefs,IT,Axs,KA,Val{1},
                    Val{:a1},EQ,PR,KP,KBC,IntegralOrder,FD,SD,Val{:cubic}})(x::Vararg{Number,1}) where 
                    {T,TCoefs,IT,Axs,KA,EQ,PR,KP,KBC,FD,SD}

    eqs_int = itp.eqs
    n_pre = length(itp.pre_range)
    h_pre = one(T) / T(n_pre - 1)
    result = zero(T)
    i_float  = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T)
    i        = clamp(floor(Int, i_float), eqs_int, length(itp.coefs) - eqs_int)

    @inbounds for j in (i - eqs_int + 1):(i + eqs_int)
    # @inbounds for j in 1:length(itp.coefs)
        xj = itp.knots[1][eqs_int] + (j - eqs_int) * itp.h[1]
        sj = (x[1] - xj) / itp.h[1]

        if abs(sj) >= T(eqs_int)
            kt = T(1//2) * T(sign(sj))
        else
            col_float = T(eqs_int) + sj
            col = clamp(floor(Int, col_float) + 1, 1, 2 * eqs_int)
            x_diff_right = col_float - T(col - 1)
            continuous_idx = x_diff_right * T(n_pre - 1) + one(T)
            idx      = clamp(floor(Int, continuous_idx), 1, n_pre - 1)
            idx_next = idx + 1
            t        = continuous_idx - T(idx)
            kt = cubic_hermite(t,
                    itp.kernel_pre[idx,    col], itp.kernel_pre[idx_next,    col],
                    itp.kernel_d1_pre[idx, col], itp.kernel_d1_pre[idx_next, col],
                    h_pre)
        end

        result += itp.coefs[j] * (kt - itp.left_values[1][j])
    end

    left_tail  = (i - eqs_int) >= 1                      ? itp.cumcoefs_left[1][i - eqs_int]      : zero(T)
    right_tail = (i + eqs_int + 1) <= length(itp.coefs)  ? itp.cumcoefs_right[1][i + eqs_int + 1] : zero(T)

    return (result + left_tail + right_tail) * itp.h[1]
end

@inline function (itp::FastConvolutionInterpolation{T,1,TCoefs,IT,Axs,KA,Val{1},
                    HigherOrderKernel{DG},EQ,PR,KP,KBC,IntegralOrder,FD,SD,Val{:linear}})(x::Vararg{Number,1}) where 
                    {T,TCoefs,IT,Axs,KA,DG,EQ,PR,KP,KBC,FD,SD}

    eqs_int = itp.eqs
    n_pre   = length(itp.pre_range)
    result  = zero(T)

    i_float = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T)
    i       = clamp(floor(Int, i_float), eqs_int, length(itp.coefs) - eqs_int)

    @inbounds for j in (i - eqs_int + 1):(i + eqs_int)
        xj = itp.knots[1][eqs_int] + (j - eqs_int) * itp.h[1]
        sj = (x[1] - xj) / itp.h[1]

        if abs(sj) >= T(eqs_int)
            kt = T(1//2) * T(sign(sj))
        else
            col_float      = T(eqs_int) + sj
            col            = clamp(floor(Int, col_float) + 1, 1, 2 * eqs_int)
            x_diff_right   = col_float - T(col - 1)
            continuous_idx = x_diff_right * T(n_pre - 1) + one(T)
            idx            = clamp(floor(Int, continuous_idx), 1, n_pre - 1)
            idx_next       = idx + 1
            t              = continuous_idx - T(idx)
            kt = (one(T) - t) * itp.kernel_pre[idx, col] + t * itp.kernel_pre[idx_next, col]
        end

        result += itp.coefs[j] * (kt - itp.left_values[1][j])
    end

    left_tail  = (i - eqs_int) >= 1                      ? itp.cumcoefs_left[1][i - eqs_int]      : zero(T)
    right_tail = (i + eqs_int + 1) <= length(itp.coefs)  ? itp.cumcoefs_right[1][i + eqs_int + 1] : zero(T)

    return (result + left_tail + right_tail) * itp.h[1]
end