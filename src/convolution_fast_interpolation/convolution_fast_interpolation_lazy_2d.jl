@inline function (itp::FastConvolutionInterpolation{T,2,0,TCoefs,Axs,KA,Val{2},LowerOrderKernel{(:a0, :a0)},
                    EQ,KBC,DerivativeOrder{DO},FD,SD,Val{SG},Val{true},Val{0}})(x::Vararg{Number,2}) where 
                    {T<:AbstractFloat,TCoefs<:AbstractArray{T,2},Axs<:Tuple{<:AbstractVector,<:AbstractVector},
                    KA<:Tuple{<:Nothing,<:Nothing},EQ<:Tuple{Int,Int},KBC<:Tuple{<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol}},DO,FD,SD,SG}

    x = T.(x)
    # specialized dispatch for 2d nearest neighbor kernel 
    # First dimension (x)
    i_float = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T)
    i = clamp(floor(Int, i_float), itp.eqs[1], length(itp.knots[1]) - itp.eqs[1])
    x_diff_left = (x[1] - itp.knots[1][i]) / itp.h[1]  # Recompute from actual knot

    # Second dimension (y)
    j_float = (x[2] - itp.knots[2][1]) / itp.h[2] + one(T)
    j = clamp(floor(Int, j_float), itp.eqs[2], length(itp.knots[2]) - itp.eqs[2])
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

@inline function (itp::FastConvolutionInterpolation{T,2,0,TCoefs,Axs,KA,Val{2},LowerOrderKernel{(:a1, :a1)},
                    EQ,KBC,DerivativeOrder{DO},FD,SD,Val{SG},Val{true},Val{0}})(x::Vararg{Number,2}) where 
                    {T<:AbstractFloat,TCoefs<:AbstractArray{T,2},Axs<:Tuple{<:AbstractVector,<:AbstractVector},
                    KA<:Tuple{<:Nothing,<:Nothing},EQ<:Tuple{Int,Int},KBC<:Tuple{<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol}},DO,FD,SD,SG}

    x = T.(x)
    # specialized dispatch for 2d linear kernel
    # First dimension (x)
    i_float = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T)
    i = clamp(floor(Int, i_float), itp.eqs[1], length(itp.knots[1]) - itp.eqs[1])
    x_diff_left = (x[1] - itp.knots[1][i]) / itp.h[1]  # Recompute from actual knot

    # Second dimension (y)
    j_float = (x[2] - itp.knots[2][1]) / itp.h[2] + one(T)
    j = clamp(floor(Int, j_float), itp.eqs[2], length(itp.knots[2]) - itp.eqs[2])
    y_diff_left = (x[2] - itp.knots[2][j]) / itp.h[2]  # Recompute from actual knot

    # Bilinear interpolation
    return @inbounds @fastmath ((1-x_diff_left)*(1-y_diff_left)*itp.coefs[i, j] + 
                        x_diff_left*(1-y_diff_left)*itp.coefs[i+1, j] + 
                        (1-x_diff_left)*y_diff_left*itp.coefs[i, j+1] + 
                        x_diff_left*y_diff_left*itp.coefs[i+1, j+1]) *
                        (-one(T)/itp.h[1])^DO[1] * (-one(T)/itp.h[2])^DO[2]
end

@inline function (itp::FastConvolutionInterpolation{T,2,0,TCoefs,Axs,KA,Val{2},
            DG,EQ,KBC,DerivativeOrder{DO},FD,SD,Val{SG},Val{true},Val{0}})(x::Vararg{Number,2}) where 
            {T<:AbstractFloat,TCoefs<:AbstractArray{T,2},Axs<:Tuple{<:AbstractVector,<:AbstractVector},
            KA<:Tuple{<:Nothing,<:Nothing},DG,EQ<:Tuple{Int,Int},KBC<:Tuple{<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol}},DO,FD,SD,SG}

    x = T.(x)
    # first dimension
    i_float = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T)
    i = clamp(floor(Int, i_float), 1, length(itp.knots[1]) - 1)
    is_boundary_x = is_boundary_stencil(i, size(itp.coefs, 1), itp.eqs[1])

    # second dimension
    j_float = (x[2] - itp.knots[2][1]) / itp.h[2] + one(T)
    j = clamp(floor(Int, j_float), 1, length(itp.knots[2]) - 1)
    is_boundary_y = is_boundary_stencil(j, size(itp.coefs, 2), itp.eqs[2])

    if (is_boundary_x || is_boundary_y) && itp.boundary_fallback
        
        x_diff_left = (x[1] - itp.knots[1][i]) / itp.h[1]  # Recompute from actual knot
        y_diff_left = (x[2] - itp.knots[2][j]) / itp.h[2]  # Recompute from actual knot

        # Bilinear interpolation
        return @inbounds @fastmath ((1-x_diff_left)*(1-y_diff_left)*itp.coefs[i, j] + 
                        x_diff_left*(1-y_diff_left)*itp.coefs[i+1, j] + 
                        (1-x_diff_left)*y_diff_left*itp.coefs[i, j+1] + 
                        x_diff_left*y_diff_left*itp.coefs[i+1, j+1]) *
                        (-one(T)/itp.h[1])^DO[1] * (-one(T)/itp.h[2])^DO[2]

    else

        # Kernel weights per dimension at τ = diff_right (column c ↔ stencil offset c − eqs)
        x_diff_right = one(T) - (x[1] - itp.knots[1][i]) / itp.h[1]
        y_diff_right = one(T) - (x[2] - itp.knots[2][j]) / itp.h[2]
        wx, wy = _column_weights_per_dim(Val(_kernel_sym(itp.kernel_sym)), Val(DO), (x_diff_right, y_diff_right))

        result = if is_boundary_x || is_boundary_y
            # the stencil reaches past the domain: ghosts formed on a local patch, as eager forms them
            _lazy_patch_sum(itp, (i, j), (wx, wy))
        else
            # the whole stencil lies inside the domain: a plain tensor sum, as in eager mode
            ox = i - itp.eqs[1]                           # stencil offset along x
            oy = j - itp.eqs[2]                           # stencil offset along y
            acc = zero(T)
            @inbounds for ky in 1:length(wy)
                row = zero(T)
                @simd for kx in 1:length(wx)
                    row += itp.coefs[ox + kx, oy + ky] * wx[kx]
                end
                acc += row * wy[ky]
            end
            acc
        end

        return result * (-one(T)/itp.h[1])^DO[1] * (-one(T)/itp.h[2])^DO[2]
    end
end