@inline function (itp::FastConvolutionInterpolation{T,3,0,TCoefs,Axs,KA,Val{3},LowerOrderKernel{(:a0,:a0,:a0)},
                    EQ,PR,KP,KBC,DerivativeOrder{DO},FD,SD,Val{(:linear,:linear,:linear)},
                    Val{true},Val{0}})(x::Vararg{Number,3}) where {T<:AbstractFloat,TCoefs<:AbstractArray{T,3},
                    Axs<:Tuple{<:AbstractVector,<:AbstractVector,<:AbstractVector},
                    KA<:Tuple{<:Nothing,<:Nothing,<:Nothing},EQ<:Tuple{Int,Int,Int},
                    PR<:Tuple{<:AbstractVector,<:AbstractVector,<:AbstractVector},KP,
                    KBC<:Tuple{<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol}},DO,FD,SD}

    # specialized dispatch for 3d nearest neighbor kernel
    # First dimension (x)
    i_float = (x[1] - itp.knots[1][1]) / itp.h[1] + 1
    i = clamp(floor(Int, i_float), itp.eqs[1], length(itp.knots[1]) - itp.eqs[1])
    x_diff_left = (x[1] - itp.knots[1][i]) / itp.h[1]  # Recompute from actual knot

    # Second dimension (y)
    j_float = (x[2] - itp.knots[2][1]) / itp.h[2] + 1
    j = clamp(floor(Int, j_float), itp.eqs[2], length(itp.knots[2]) - itp.eqs[2])
    y_diff_left = (x[2] - itp.knots[2][j]) / itp.h[2]  # Recompute from actual knot

    # Third dimension (z)
    k_float = (x[3] - itp.knots[3][1]) / itp.h[3] + 1
    k = clamp(floor(Int, k_float), itp.eqs[3], length(itp.knots[3]) - itp.eqs[3])
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

@inline function (itp::FastConvolutionInterpolation{T,3,0,TCoefs,Axs,KA,Val{3},LowerOrderKernel{(:a1,:a1,:a1)},
                    EQ,PR,KP,KBC,DerivativeOrder{DO},FD,SD,Val{(:linear,:linear,:linear)},
                    Val{true},Val{0}})(x::Vararg{Number,3}) where {T<:AbstractFloat,
                    TCoefs<:AbstractArray{T,3},Axs<:Tuple{<:AbstractVector,<:AbstractVector,<:AbstractVector},
                    KA<:Tuple{<:Nothing,<:Nothing,<:Nothing},EQ<:Tuple{Int,Int,Int},
                    PR<:Tuple{<:AbstractVector,<:AbstractVector,<:AbstractVector},
                    KP,KBC<:Tuple{<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol}},DO,FD,SD}

    # specialized dispatch for 3d linear kernel
    # First dimension (x)
    i_float = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T)
    i = clamp(floor(Int, i_float), itp.eqs[1], length(itp.knots[1]) - itp.eqs[1])
    x_diff_left = (x[1] - itp.knots[1][i]) / itp.h[1]  # Recompute from actual knot

    # Second dimension (y)
    j_float = (x[2] - itp.knots[2][1]) / itp.h[2] + one(T)
    j = clamp(floor(Int, j_float), itp.eqs[2], length(itp.knots[2]) - itp.eqs[2])
    y_diff_left = (x[2] - itp.knots[2][j]) / itp.h[2]  # Recompute from actual knot

    # Third dimension (z)
    k_float = (x[3] - itp.knots[3][1]) / itp.h[3] + one(T)
    k = clamp(floor(Int, k_float), itp.eqs[3], length(itp.knots[3]) - itp.eqs[3])
    z_diff_left = (x[3] - itp.knots[3][k]) / itp.h[3]  # Recompute from actual knot

    # Trilinear interpolation formula
    return @inbounds @fastmath ((1-x_diff_left)*(1-y_diff_left)*(1-z_diff_left)*itp.coefs[i, j, k] + 
                    x_diff_left*(1-y_diff_left)*(1-z_diff_left)*itp.coefs[i+1, j, k] + 
                    (1-x_diff_left)*y_diff_left*(1-z_diff_left)*itp.coefs[i, j+1, k] + 
                    x_diff_left*y_diff_left*(1-z_diff_left)*itp.coefs[i+1, j+1, k] +
                    (1-x_diff_left)*(1-y_diff_left)*z_diff_left*itp.coefs[i, j, k+1] + 
                    x_diff_left*(1-y_diff_left)*z_diff_left*itp.coefs[i+1, j, k+1] + 
                    (1-x_diff_left)*y_diff_left*z_diff_left*itp.coefs[i, j+1, k+1] + 
                    x_diff_left*y_diff_left*z_diff_left*itp.coefs[i+1, j+1, k+1]) *
                    (-one(T)/itp.h[1])^DO[1] *
                    (-one(T)/itp.h[2])^DO[2] *
                    (-one(T)/itp.h[3])^DO[3]
end

function (itp::FastConvolutionInterpolation{T,3,0,TCoefs,Axs,KA,Val{3},DG,
                EQ,PR,KP,KBC,DerivativeOrder{DO},FD,SD,Val{SG},Val{true},Val{0}})(x::Vararg{Number,3}) where 
                {T<:AbstractFloat,TCoefs<:AbstractArray{T,3},Axs<:NTuple{3,<:AbstractVector},
                KA<:NTuple{3,<:Nothing},DG,EQ<:NTuple{3,Int},PR<:NTuple{3,<:AbstractVector},
                KP,KBC<:Tuple{<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol}},
                DO,FD,SD,SG}
                
    # specialized dispatch for 3d higher order kernels (cubic and above)

    # First dimension (x)
    i_float = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T)
    i = clamp(floor(Int, i_float), 1, length(itp.knots[1]) - 1)
    x_diff_left = (x[1] - itp.knots[1][i]) / itp.h[1]
    x_diff_right = one(T) - x_diff_left
    idx_x = clamp(floor(Int, x_diff_right * (length(itp.pre_range[1]) - one(Int64))) + one(Int64), one(Int64), length(itp.pre_range[1]) - one(Int64))
    idx_x_next = idx_x + one(Int64)
    t_x = (x_diff_right - itp.pre_range[1][idx_x]) / (itp.pre_range[1][idx_x_next] - itp.pre_range[1][idx_x])

    # Second dimension (y)
    j_float = (x[2] - itp.knots[2][1]) / itp.h[2] + one(T)
    j = clamp(floor(Int, j_float), 1, length(itp.knots[2]) - 1)
    y_diff_left = (x[2] - itp.knots[2][j]) / itp.h[2]
    y_diff_right = one(T) - y_diff_left
    idx_y = clamp(floor(Int, y_diff_right * (length(itp.pre_range[2]) - one(Int64))) + one(Int64), one(Int64), length(itp.pre_range[2]) - one(Int64))
    idx_y_next = idx_y + one(Int64)
    t_y = (y_diff_right - itp.pre_range[2][idx_y]) / (itp.pre_range[2][idx_y_next] - itp.pre_range[2][idx_y])

    # Third dimension (z)
    k_float = (x[3] - itp.knots[3][1]) / itp.h[3] + one(T)
    k = clamp(floor(Int, k_float), 1, length(itp.knots[3]) - 1)
    z_diff_left = (x[3] - itp.knots[3][k]) / itp.h[3]
    z_diff_right = one(T) - z_diff_left
    idx_z = clamp(floor(Int, z_diff_right * (length(itp.pre_range[3]) - one(Int64))) + one(Int64), one(Int64), length(itp.pre_range[3]) - one(Int64))
    idx_z_next = idx_z + one(Int64)
    t_z = (z_diff_right - itp.pre_range[3][idx_z]) / (itp.pre_range[3][idx_z_next] - itp.pre_range[3][idx_z])
    
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
    kernel_type = _kernel_sym(itp.kernel_sym)
    ng = itp.eqs[1] - 1 # lazy only for same kernel in all directions
    @inbounds for n in -(itp.eqs[3]-1):itp.eqs[3]
        kz_0 = itp.kernel_pre[3][idx_z, n+itp.eqs[3]]
        kz_1 = itp.kernel_pre[3][idx_z_next, n+itp.eqs[3]]
        
        @inbounds for m in -(itp.eqs[2]-1):itp.eqs[2]
            ky_0 = itp.kernel_pre[2][idx_y, m+itp.eqs[2]]
            ky_1 = itp.kernel_pre[2][idx_y_next, m+itp.eqs[2]]
            
            @inbounds for l in -(itp.eqs[1]-1):itp.eqs[1]
                coef = lazy_ghost_value(itp.coefs, (i + ng + l, j + ng + m, k + ng + n), itp.eqs[1], kernel_type[1])
                kx_0 = itp.kernel_pre[1][idx_x, l+itp.eqs[1]]
                kx_1 = itp.kernel_pre[1][idx_x_next, l+itp.eqs[1]]
                
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
    return @fastmath ((1-t_x)*(1-t_y)*(1-t_z)*result_000 + 
            t_x*(1-t_y)*(1-t_z)*result_100 + 
            (1-t_x)*t_y*(1-t_z)*result_010 + 
            t_x*t_y*(1-t_z)*result_110 +
            (1-t_x)*(1-t_y)*t_z*result_001 + 
            t_x*(1-t_y)*t_z*result_101 + 
            (1-t_x)*t_y*t_z*result_011 + 
            t_x*t_y*t_z*result_111) *
            (-one(T)/itp.h[1])^DO[1] * 
            (-one(T)/itp.h[2])^DO[2] * 
            (-one(T)/itp.h[3])^DO[3]
end