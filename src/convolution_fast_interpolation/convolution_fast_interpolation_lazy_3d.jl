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

@inline function (itp::FastConvolutionInterpolation{T,3,0,TCoefs,Axs,KA,Val{3},DG,
                EQ,PR,KP,KBC,DerivativeOrder{DO},FD,SD,Val{SG},Val{true},Val{0}})(x::Vararg{Number,3}) where 
                {T<:AbstractFloat,TCoefs<:AbstractArray{T,3},Axs<:NTuple{3,<:AbstractVector},
                KA<:NTuple{3,<:Nothing},DG,EQ<:NTuple{3,Int},PR<:NTuple{3,<:AbstractVector},
                KP,KBC<:Tuple{<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol}},
                DO,FD,SD,SG}
                
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

        # First dimension (x)
        x_diff_left = (x[1] - itp.knots[1][i]) / itp.h[1]
        x_diff_right = one(T) - x_diff_left
        idx_x = clamp(floor(Int, x_diff_right * (length(itp.pre_range[1]) - one(Int64))) + one(Int64), one(Int64), length(itp.pre_range[1]) - one(Int64))
        idx_x_next = idx_x + one(Int64)
        t_x = (x_diff_right - itp.pre_range[1][idx_x]) / (itp.pre_range[1][idx_x_next] - itp.pre_range[1][idx_x])

        # Second dimension (y)
        y_diff_left = (x[2] - itp.knots[2][j]) / itp.h[2]
        y_diff_right = one(T) - y_diff_left
        idx_y = clamp(floor(Int, y_diff_right * (length(itp.pre_range[2]) - one(Int64))) + one(Int64), one(Int64), length(itp.pre_range[2]) - one(Int64))
        idx_y_next = idx_y + one(Int64)
        t_y = (y_diff_right - itp.pre_range[2][idx_y]) / (itp.pre_range[2][idx_y_next] - itp.pre_range[2][idx_y])

        # Third dimension (z)
        z_diff_left = (x[3] - itp.knots[3][k]) / itp.h[3]
        z_diff_right = one(T) - z_diff_left
        idx_z = clamp(floor(Int, z_diff_right * (length(itp.pre_range[3]) - one(Int64))) + one(Int64), one(Int64), length(itp.pre_range[3]) - one(Int64))
        idx_z_next = idx_z + one(Int64)
        t_z = (z_diff_right - itp.pre_range[3][idx_z]) / (itp.pre_range[3][idx_z_next] - itp.pre_range[3][idx_z])

        ng1 = itp.eqs[1] - 1
        ng2 = itp.eqs[2] - 1
        ng3 = itp.eqs[3] - 1
        n1 = itp.domain_size[1]
        n2 = itp.domain_size[2]
        n3 = itp.domain_size[3]
        kernel_type = _kernel_sym(itp.kernel_sym)

        if is_boundary_x || is_boundary_y || is_boundary_z
            ghost_matrix_xl = (is_boundary_x && i - ng1 < 1)          ? get_polynomial_ghost_coeffs(itp.bc[1][1], kernel_type[1]) : nothing
            ghost_matrix_xr = (is_boundary_x && i + itp.eqs[1] > n1)  ? get_polynomial_ghost_coeffs(itp.bc[1][2], kernel_type[1]) : nothing
            ghost_matrix_yl = (is_boundary_y && j - ng2 < 1)          ? get_polynomial_ghost_coeffs(itp.bc[2][1], kernel_type[2]) : nothing
            ghost_matrix_yr = (is_boundary_y && j + itp.eqs[2] > n2)  ? get_polynomial_ghost_coeffs(itp.bc[2][2], kernel_type[2]) : nothing
            ns_xl = ghost_matrix_xl !== nothing ? size(ghost_matrix_xl, 2) : 0
            ns_xr = ghost_matrix_xr !== nothing ? size(ghost_matrix_xr, 2) : 0
            ns_yl = ghost_matrix_yl !== nothing ? size(ghost_matrix_yl, 2) : 0
            ns_yr = ghost_matrix_yr !== nothing ? size(ghost_matrix_yr, 2) : 0

            # Step 1: Fill x-ghosts (from interior only)
            if is_boundary_x
                if i - ng1 < 1
                    for n in -(ng3):itp.eqs[3]
                        abs_n = k + n
                        if abs_n >= 1 && abs_n <= n3
                            for m in -(ng2):itp.eqs[2]
                                abs_m = j + m
                                if abs_m >= 1 && abs_m <= n2
                                    y_slice = view(itp.coefs, 1:ns_xl, abs_m, abs_n)
                                    mul!(view(itp.lazy_workspace.ghost_buf, 1:ng1), view(ghost_matrix_xl, 1:ng1, :), y_slice)
                                    for l in -(ng1):-i
                                        itp.lazy_workspace.stencil_buf[l + itp.eqs[1], m + itp.eqs[2], n + itp.eqs[3]] = itp.lazy_workspace.ghost_buf[1 - (i + l)]
                                    end
                                end
                            end
                        end
                    end
                end
                if i + itp.eqs[1] > n1
                    for n in -(ng3):itp.eqs[3]
                        abs_n = k + n
                        if abs_n >= 1 && abs_n <= n3
                            for m in -(ng2):itp.eqs[2]
                                abs_m = j + m
                                if abs_m >= 1 && abs_m <= n2
                                    for kk in 1:ng1
                                        acc = zero(T)
                                        for p in 1:ns_xr
                                            acc += ghost_matrix_xr[kk, p] * itp.coefs[n1 - p + 1, abs_m, abs_n]
                                        end
                                        itp.lazy_workspace.ghost_buf[kk] = acc
                                    end
                                    for l in (n1-i+1):itp.eqs[1]
                                        itp.lazy_workspace.stencil_buf[l + itp.eqs[1], m + itp.eqs[2], n + itp.eqs[3]] = itp.lazy_workspace.ghost_buf[i + l - n1]
                                    end
                                end
                            end
                        end
                    end
                end
            end

            # Step 2: Fill y-ghosts (from interior or x-ghosts)
            if is_boundary_y
                if j - ng2 < 1
                    for n in -(ng3):itp.eqs[3]
                        abs_n = k + n
                        if abs_n >= 1 && abs_n <= n3
                            for l in -(ng1):itp.eqs[1]
                                abs_l = i + l
                                for kk in 1:ng2
                                    acc = zero(T)
                                    for p in 1:ns_yl
                                        val_p = if abs_l >= 1 && abs_l <= n1
                                            itp.coefs[abs_l, p, abs_n]
                                        elseif abs_l < 1
                                            inner = zero(T)
                                            for q in 1:ns_xl
                                                inner += ghost_matrix_xl[1 - abs_l, q] * itp.coefs[q, p, abs_n]
                                            end
                                            inner
                                        else
                                            inner = zero(T)
                                            for q in 1:ns_xr
                                                inner += ghost_matrix_xr[abs_l - n1, q] * itp.coefs[n1 - q + 1, p, abs_n]
                                            end
                                            inner
                                        end
                                        acc += ghost_matrix_yl[kk, p] * val_p
                                    end
                                    itp.lazy_workspace.ghost_buf[kk] = acc
                                end
                                for m in -(ng2):-j
                                    itp.lazy_workspace.stencil_buf[l + itp.eqs[1], m + itp.eqs[2], n + itp.eqs[3]] = itp.lazy_workspace.ghost_buf[1 - (j + m)]
                                end
                            end
                        end
                    end
                end
                if j + itp.eqs[2] > n2
                    for n in -(ng3):itp.eqs[3]
                        abs_n = k + n
                        if abs_n >= 1 && abs_n <= n3
                            for l in -(ng1):itp.eqs[1]
                                abs_l = i + l
                                for kk in 1:ng2
                                    acc = zero(T)
                                    for p in 1:ns_yr
                                        val_p = if abs_l >= 1 && abs_l <= n1
                                            itp.coefs[abs_l, n2 - p + 1, abs_n]
                                        elseif abs_l < 1
                                            inner = zero(T)
                                            for q in 1:ns_xl
                                                inner += ghost_matrix_xl[1 - abs_l, q] * itp.coefs[q, n2 - p + 1, abs_n]
                                            end
                                            inner
                                        else
                                            inner = zero(T)
                                            for q in 1:ns_xr
                                                inner += ghost_matrix_xr[abs_l - n1, q] * itp.coefs[n1 - q + 1, n2 - p + 1, abs_n]
                                            end
                                            inner
                                        end
                                        acc += ghost_matrix_yr[kk, p] * val_p
                                    end
                                    itp.lazy_workspace.ghost_buf[kk] = acc
                                end
                                for m in (n2-j+1):itp.eqs[2]
                                    itp.lazy_workspace.stencil_buf[l + itp.eqs[1], m + itp.eqs[2], n + itp.eqs[3]] = itp.lazy_workspace.ghost_buf[j + m - n2]
                                end
                            end
                        end
                    end
                end
            end

            # Step 3: Fill z-ghosts (from interior, x-ghosts, y-ghosts, or xy-corner ghosts)
            if is_boundary_z
                if k - ng3 < 1
                    ghost_matrix_zl = get_polynomial_ghost_coeffs(itp.bc[3][1], kernel_type[3])
                    ns_zl = size(ghost_matrix_zl, 2)
                    for m in -(ng2):itp.eqs[2]
                        abs_m = j + m
                        for l in -(ng1):itp.eqs[1]
                            abs_l = i + l
                            for kk in 1:ng3
                                acc = zero(T)
                                for p in 1:ns_zl
                                    val_p = if abs_l >= 1 && abs_l <= n1 && abs_m >= 1 && abs_m <= n2
                                        itp.coefs[abs_l, abs_m, p]
                                    elseif abs_m >= 1 && abs_m <= n2
                                        if abs_l < 1
                                            inner = zero(T)
                                            for q in 1:ns_xl
                                                inner += ghost_matrix_xl[1 - abs_l, q] * itp.coefs[q, abs_m, p]
                                            end
                                            inner
                                        else
                                            inner = zero(T)
                                            for q in 1:ns_xr
                                                inner += ghost_matrix_xr[abs_l - n1, q] * itp.coefs[n1 - q + 1, abs_m, p]
                                            end
                                            inner
                                        end
                                    elseif abs_l >= 1 && abs_l <= n1
                                        if abs_m < 1
                                            inner = zero(T)
                                            for q in 1:ns_yl
                                                inner += ghost_matrix_yl[1 - abs_m, q] * itp.coefs[abs_l, q, p]
                                            end
                                            inner
                                        else
                                            inner = zero(T)
                                            for q in 1:ns_yr
                                                inner += ghost_matrix_yr[abs_m - n2, q] * itp.coefs[abs_l, n2 - q + 1, p]
                                            end
                                            inner
                                        end
                                    else
                                        if abs_l < 1 && abs_m < 1
                                            inner = zero(T)
                                            for qx in 1:ns_xl
                                                for qy in 1:ns_yl
                                                    inner += ghost_matrix_xl[1 - abs_l, qx] * ghost_matrix_yl[1 - abs_m, qy] * itp.coefs[qx, qy, p]
                                                end
                                            end
                                            inner
                                        elseif abs_l < 1 && abs_m > n2
                                            inner = zero(T)
                                            for qx in 1:ns_xl
                                                for qy in 1:ns_yr
                                                    inner += ghost_matrix_xl[1 - abs_l, qx] * ghost_matrix_yr[abs_m - n2, qy] * itp.coefs[qx, n2 - qy + 1, p]
                                                end
                                            end
                                            inner
                                        elseif abs_l > n1 && abs_m < 1
                                            inner = zero(T)
                                            for qx in 1:ns_xr
                                                for qy in 1:ns_yl
                                                    inner += ghost_matrix_xr[abs_l - n1, qx] * ghost_matrix_yl[1 - abs_m, qy] * itp.coefs[n1 - qx + 1, qy, p]
                                                end
                                            end
                                            inner
                                        else
                                            inner = zero(T)
                                            for qx in 1:ns_xr
                                                for qy in 1:ns_yr
                                                    inner += ghost_matrix_xr[abs_l - n1, qx] * ghost_matrix_yr[abs_m - n2, qy] * itp.coefs[n1 - qx + 1, n2 - qy + 1, p]
                                                end
                                            end
                                            inner
                                        end
                                    end
                                    acc += ghost_matrix_zl[kk, p] * val_p
                                end
                                itp.lazy_workspace.ghost_buf[kk] = acc
                            end
                            for n in -(ng3):-k
                                itp.lazy_workspace.stencil_buf[l + itp.eqs[1], m + itp.eqs[2], n + itp.eqs[3]] = itp.lazy_workspace.ghost_buf[1 - (k + n)]
                            end
                        end
                    end
                end
                if k + itp.eqs[3] > n3
                    ghost_matrix_zr = get_polynomial_ghost_coeffs(itp.bc[3][2], kernel_type[3])
                    ns_zr = size(ghost_matrix_zr, 2)
                    for m in -(ng2):itp.eqs[2]
                        abs_m = j + m
                        for l in -(ng1):itp.eqs[1]
                            abs_l = i + l
                            for kk in 1:ng3
                                acc = zero(T)
                                for p in 1:ns_zr
                                    val_p = if abs_l >= 1 && abs_l <= n1 && abs_m >= 1 && abs_m <= n2
                                        itp.coefs[abs_l, abs_m, n3 - p + 1]
                                    elseif abs_m >= 1 && abs_m <= n2
                                        if abs_l < 1
                                            inner = zero(T)
                                            for q in 1:ns_xl
                                                inner += ghost_matrix_xl[1 - abs_l, q] * itp.coefs[q, abs_m, n3 - p + 1]
                                            end
                                            inner
                                        else
                                            inner = zero(T)
                                            for q in 1:ns_xr
                                                inner += ghost_matrix_xr[abs_l - n1, q] * itp.coefs[n1 - q + 1, abs_m, n3 - p + 1]
                                            end
                                            inner
                                        end
                                    elseif abs_l >= 1 && abs_l <= n1
                                        if abs_m < 1
                                            inner = zero(T)
                                            for q in 1:ns_yl
                                                inner += ghost_matrix_yl[1 - abs_m, q] * itp.coefs[abs_l, q, n3 - p + 1]
                                            end
                                            inner
                                        else
                                            inner = zero(T)
                                            for q in 1:ns_yr
                                                inner += ghost_matrix_yr[abs_m - n2, q] * itp.coefs[abs_l, n2 - q + 1, n3 - p + 1]
                                            end
                                            inner
                                        end
                                    else
                                        if abs_l < 1 && abs_m < 1
                                            inner = zero(T)
                                            for qx in 1:ns_xl
                                                for qy in 1:ns_yl
                                                    inner += ghost_matrix_xl[1 - abs_l, qx] * ghost_matrix_yl[1 - abs_m, qy] * itp.coefs[qx, qy, n3 - p + 1]
                                                end
                                            end
                                            inner
                                        elseif abs_l < 1 && abs_m > n2
                                            inner = zero(T)
                                            for qx in 1:ns_xl
                                                for qy in 1:ns_yr
                                                    inner += ghost_matrix_xl[1 - abs_l, qx] * ghost_matrix_yr[abs_m - n2, qy] * itp.coefs[qx, n2 - qy + 1, n3 - p + 1]
                                                end
                                            end
                                            inner
                                        elseif abs_l > n1 && abs_m < 1
                                            inner = zero(T)
                                            for qx in 1:ns_xr
                                                for qy in 1:ns_yl
                                                    inner += ghost_matrix_xr[abs_l - n1, qx] * ghost_matrix_yl[1 - abs_m, qy] * itp.coefs[n1 - qx + 1, qy, n3 - p + 1]
                                                end
                                            end
                                            inner
                                        else
                                            inner = zero(T)
                                            for qx in 1:ns_xr
                                                for qy in 1:ns_yr
                                                    inner += ghost_matrix_xr[abs_l - n1, qx] * ghost_matrix_yr[abs_m - n2, qy] * itp.coefs[n1 - qx + 1, n2 - qy + 1, n3 - p + 1]
                                                end
                                            end
                                            inner
                                        end
                                    end
                                    acc += ghost_matrix_zr[kk, p] * val_p
                                end
                                itp.lazy_workspace.ghost_buf[kk] = acc
                            end
                            for n in (n3-k+1):itp.eqs[3]
                                itp.lazy_workspace.stencil_buf[l + itp.eqs[1], m + itp.eqs[2], n + itp.eqs[3]] = itp.lazy_workspace.ghost_buf[k + n - n3]
                            end
                        end
                    end
                end
            end
        end

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
        @inbounds for n in -(itp.eqs[3]-1):itp.eqs[3]
            kz_0 = itp.kernel_pre[3][idx_z, n+itp.eqs[3]]
            kz_1 = itp.kernel_pre[3][idx_z_next, n+itp.eqs[3]]
            
            @inbounds for m in -(itp.eqs[2]-1):itp.eqs[2]
                ky_0 = itp.kernel_pre[2][idx_y, m+itp.eqs[2]]
                ky_1 = itp.kernel_pre[2][idx_y_next, m+itp.eqs[2]]
                
                @inbounds for l in -(itp.eqs[1]-1):itp.eqs[1]
                    abs_l = i + l
                    abs_m = j + m
                    abs_n = k + n
                    coef = if abs_l >= 1 && abs_l <= n1 && abs_m >= 1 && abs_m <= n2 && abs_n >= 1 && abs_n <= n3
                        itp.coefs[abs_l, abs_m, abs_n]
                    else
                        itp.lazy_workspace.stencil_buf[l + itp.eqs[1], m + itp.eqs[2], n + itp.eqs[3]]
                    end
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
end