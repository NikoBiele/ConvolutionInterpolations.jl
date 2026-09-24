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

        ng1 = itp.eqs[1] - 1
        ng2 = itp.eqs[2] - 1
        n1 = itp.domain_size[1]
        n2 = itp.domain_size[2]
        kernel_type = _kernel_sym(itp.kernel_sym)

        if is_boundary_x || is_boundary_y
            ghost_matrix_xl = (is_boundary_x && i - ng1 < 1)          ? get_polynomial_ghost_coeffs(itp.bc[1][1], kernel_type[1]) : nothing
            ghost_matrix_xr = (is_boundary_x && i + itp.eqs[1] > n1)  ? get_polynomial_ghost_coeffs(itp.bc[1][2], kernel_type[1]) : nothing
            ns_xl = ghost_matrix_xl !== nothing ? size(ghost_matrix_xl, 2) : 0
            ns_xr = ghost_matrix_xr !== nothing ? size(ghost_matrix_xr, 2) : 0

            # Fill x-ghosts for each m in stencil range
            if is_boundary_x
                if i - ng1 < 1
                    for m in -(ng2):itp.eqs[2]
                        abs_m = j + m
                        if abs_m >= 1 && abs_m <= n2
                            y_slice = view(itp.coefs, 1:ns_xl, abs_m)
                            mul!(view(itp.lazy_workspace.ghost_buf, 1:ng1), view(ghost_matrix_xl, 1:ng1, :), y_slice)
                            for l in -(ng1):-i
                                itp.lazy_workspace.stencil_buf[l + itp.eqs[1], m + itp.eqs[2]] = itp.lazy_workspace.ghost_buf[1 - (i + l)]
                            end
                        end
                    end
                end
                if i + itp.eqs[1] > n1
                    for m in -(ng2):itp.eqs[2]
                        abs_m = j + m
                        if abs_m >= 1 && abs_m <= n2
                            for k in 1:ng1
                                acc = zero(T)
                                for p in 1:ns_xr
                                    acc += ghost_matrix_xr[k, p] * itp.coefs[n1 - p + 1, abs_m]
                                end
                                itp.lazy_workspace.ghost_buf[k] = acc
                            end
                            for l in (n1-i+1):itp.eqs[1]
                                itp.lazy_workspace.stencil_buf[l + itp.eqs[1], m + itp.eqs[2]] = itp.lazy_workspace.ghost_buf[i + l - n1]
                            end
                        end
                    end
                end
            end

            # Fill y-ghosts
            if is_boundary_y
                if j - ng2 < 1
                    ghost_matrix_yl = get_polynomial_ghost_coeffs(itp.bc[2][1], kernel_type[2])
                    ns_yl = size(ghost_matrix_yl, 2)
                    for l in -(ng1):itp.eqs[1]
                        abs_l = i + l
                        for k in 1:ng2
                            acc = zero(T)
                            for p in 1:ns_yl
                                val_p = if abs_l >= 1 && abs_l <= n1
                                    itp.coefs[abs_l, p]
                                elseif abs_l < 1
                                    inner = zero(T)
                                    for q in 1:ns_xl
                                        inner += ghost_matrix_xl[1 - abs_l, q] * itp.coefs[q, p]
                                    end
                                    inner
                                else
                                    inner = zero(T)
                                    for q in 1:ns_xr
                                        inner += ghost_matrix_xr[abs_l - n1, q] * itp.coefs[n1 - q + 1, p]
                                    end
                                    inner
                                end
                                acc += ghost_matrix_yl[k, p] * val_p
                            end
                            itp.lazy_workspace.ghost_buf[k] = acc
                        end
                        for m in -(ng2):-j
                            itp.lazy_workspace.stencil_buf[l + itp.eqs[1], m + itp.eqs[2]] = itp.lazy_workspace.ghost_buf[1 - (j + m)]
                        end
                    end
                end
                if j + itp.eqs[2] > n2
                    ghost_matrix_yr = get_polynomial_ghost_coeffs(itp.bc[2][2], kernel_type[2])
                    ns_yr = size(ghost_matrix_yr, 2)
                    for l in -(ng1):itp.eqs[1]
                        abs_l = i + l
                        for k in 1:ng2
                            acc = zero(T)
                            for p in 1:ns_yr
                                val_p = if abs_l >= 1 && abs_l <= n1
                                    itp.coefs[abs_l, n2 - p + 1]
                                elseif abs_l < 1
                                    inner = zero(T)
                                    for q in 1:ns_xl
                                        inner += ghost_matrix_xl[1 - abs_l, q] * itp.coefs[q, n2 - p + 1]
                                    end
                                    inner
                                else
                                    inner = zero(T)
                                    for q in 1:ns_xr
                                        inner += ghost_matrix_xr[abs_l - n1, q] * itp.coefs[n1 - q + 1, n2 - p + 1]
                                    end
                                    inner
                                end
                                acc += ghost_matrix_yr[k, p] * val_p
                            end
                            itp.lazy_workspace.ghost_buf[k] = acc
                        end
                        for m in (n2-j+1):itp.eqs[2]
                            itp.lazy_workspace.stencil_buf[l + itp.eqs[1], m + itp.eqs[2]] = itp.lazy_workspace.ghost_buf[j + m - n2]
                        end
                    end
                end
            end
        end

        # Kernel weights per dimension at τ = diff_right (column c ↔ stencil offset c − eqs)
        x_diff_right = one(T) - (x[1] - itp.knots[1][i]) / itp.h[1]
        y_diff_right = one(T) - (x[2] - itp.knots[2][j]) / itp.h[2]
        wx, wy = _column_weights_per_dim(Val(kernel_type), Val(DO), (x_diff_right, y_diff_right))

        # Tensor sum over the stencil; out-of-domain coefficients come from the ghost stencil
        result = zero(T)
        @inbounds for ky in 1:length(wy)
            abs_m = j + ky - itp.eqs[2]
            row = zero(T)
            for kx in 1:length(wx)
                abs_l = i + kx - itp.eqs[1]
                coef = if abs_l >= 1 && abs_l <= n1 && abs_m >= 1 && abs_m <= n2
                    itp.coefs[abs_l, abs_m]
                else
                    itp.lazy_workspace.stencil_buf[kx, ky]
                end
                row += coef * wx[kx]
            end
            result += row * wy[ky]
        end

        return result * (-one(T)/itp.h[1])^DO[1] * (-one(T)/itp.h[2])^DO[2]
    end
end