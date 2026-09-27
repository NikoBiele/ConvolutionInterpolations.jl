@inline function (itp::FastConvolutionInterpolation{T,3,0,TCoefs,Axs,KA,Val{3},LowerOrderKernel{(:a0,:a0,:a0)},
                    EQ,KBC,DerivativeOrder{DO},FD,SD,Val{SG},
                    Val{true},Val{0}})(x::Vararg{Number,3}) where {T<:AbstractFloat,TCoefs<:AbstractArray{T,3},
                    Axs<:Tuple{<:AbstractVector,<:AbstractVector,<:AbstractVector},
                    KA<:Tuple{<:Nothing,<:Nothing,<:Nothing},EQ<:Tuple{Int,Int,Int},
                    KBC<:Tuple{<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol}},DO,FD,SD,SG}

    x = T.(x)
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
                    EQ,KBC,DerivativeOrder{DO},FD,SD,Val{SG},
                    Val{true},Val{0}})(x::Vararg{Number,3}) where {T<:AbstractFloat,
                    TCoefs<:AbstractArray{T,3},Axs<:Tuple{<:AbstractVector,<:AbstractVector,<:AbstractVector},
                    KA<:Tuple{<:Nothing,<:Nothing,<:Nothing},EQ<:Tuple{Int,Int,Int},
                    KBC<:Tuple{<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol}},DO,FD,SD,SG}

    x = T.(x)
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

@inline function (itp::FastConvolutionInterpolation{T,3,0,TCoefs,Axs,KA,Val{3},DG,
                EQ,KBC,DerivativeOrder{DO},FD,SD,Val{SG},Val{true},Val{0}})(x::Vararg{Number,3}) where 
                {T<:AbstractFloat,TCoefs<:AbstractArray{T,3},Axs<:NTuple{3,<:AbstractVector},
                KA<:NTuple{3,<:Nothing},DG,EQ<:NTuple{3,Int},
                KBC<:Tuple{<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol}},
                DO,FD,SD,SG}

    x = T.(x)
    # specialized dispatch for 3d higher order kernels (cubic and above)
    # First dimension (x)
    i_float = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T)
    i = clamp(floor(Int, i_float), 1, length(itp.knots[1]) - 1)
    is_boundary_x = is_boundary_stencil(i, size(itp.coefs, 1), itp.eqs[1])

    # Second dimension (y)
    j_float = (x[2] - itp.knots[2][1]) / itp.h[2] + one(T)
    j = clamp(floor(Int, j_float), 1, length(itp.knots[2]) - 1)
    is_boundary_y = is_boundary_stencil(j, size(itp.coefs, 2), itp.eqs[2])

    # Third dimension (z)
    k_float = (x[3] - itp.knots[3][1]) / itp.h[3] + one(T)
    k = clamp(floor(Int, k_float), 1, length(itp.knots[3]) - 1)
    is_boundary_z = is_boundary_stencil(k, size(itp.coefs, 3), itp.eqs[3])
    
    if (is_boundary_x || is_boundary_y || is_boundary_z) && itp.boundary_fallback 

        x_diff_left = (x[1] - itp.knots[1][i]) / itp.h[1]  # Recompute from actual knot
        y_diff_left = (x[2] - itp.knots[2][j]) / itp.h[2]  # Recompute from actual knot
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
    else

        # Normalized positions within the cell, τ = diff_right per dimension
        x_diff_right = one(T) - (x[1] - itp.knots[1][i]) / itp.h[1]
        y_diff_right = one(T) - (x[2] - itp.knots[2][j]) / itp.h[2]
        z_diff_right = one(T) - (x[3] - itp.knots[3][k]) / itp.h[3]

                # Kernel weights per dimension at τ = diff_right (column c ↔ stencil offset c − eqs)
        wx, wy, wz = _column_weights_per_dim(Val(_kernel_sym(itp.kernel_sym)), Val(DO),
                                            (x_diff_right, y_diff_right, z_diff_right))

        result = if is_boundary_x || is_boundary_y || is_boundary_z
            # the stencil reaches past the domain: ghosts formed on a local patch, as eager forms them
            _lazy_patch_sum(itp, (i, j, k), (wx, wy, wz))
        else
            # the whole stencil lies inside the domain: a plain tensor sum, as in eager mode
            ox = i - itp.eqs[1]                           # stencil offset along x
            oy = j - itp.eqs[2]                           # stencil offset along y
            oz = k - itp.eqs[3]                           # stencil offset along z
            acc = zero(T)
            @inbounds for kz in 1:length(wz)
                plane = zero(T)
                for ky in 1:length(wy)
                    row = zero(T)
                    @simd for kx in 1:length(wx)
                        row += itp.coefs[ox + kx, oy + ky, oz + kz] * wx[kx]
                    end
                    plane += row * wy[ky]
                end
                acc += plane * wz[kz]
            end
            acc
        end

        return result * (-one(T)/itp.h[1])^DO[1] * (-one(T)/itp.h[2])^DO[2] * (-one(T)/itp.h[3])^DO[3]    
    end
end