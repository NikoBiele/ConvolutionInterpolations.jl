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

        if SG == (:linear,:linear,:linear)

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

        elseif SG == (:cubic,:cubic,:cubic)

            n_pre_x = length(itp.pre_range[1])
            n_pre_y = length(itp.pre_range[2])
            n_pre_z = length(itp.pre_range[3])
            h_pre_x = one(T) / T(n_pre_x - 1)
            h_pre_y = one(T) / T(n_pre_y - 1)
            h_pre_z = one(T) / T(n_pre_z - 1)

            # 64 accumulators: [kx type][ky type][kz type][x bracket][y bracket][z bracket]
            # naming: s_{x-type}{y-type}{z-type}_{x-bracket}{y-bracket}{z-bracket}
            # types: f=function, d=derivative; brackets: 0=lower, 1=upper
            s_fff_000=zero(T); s_fff_100=zero(T); s_fff_010=zero(T); s_fff_110=zero(T)
            s_fff_001=zero(T); s_fff_101=zero(T); s_fff_011=zero(T); s_fff_111=zero(T)
            s_dff_000=zero(T); s_dff_100=zero(T); s_dff_010=zero(T); s_dff_110=zero(T)
            s_dff_001=zero(T); s_dff_101=zero(T); s_dff_011=zero(T); s_dff_111=zero(T)
            s_fdf_000=zero(T); s_fdf_100=zero(T); s_fdf_010=zero(T); s_fdf_110=zero(T)
            s_fdf_001=zero(T); s_fdf_101=zero(T); s_fdf_011=zero(T); s_fdf_111=zero(T)
            s_ddf_000=zero(T); s_ddf_100=zero(T); s_ddf_010=zero(T); s_ddf_110=zero(T)
            s_ddf_001=zero(T); s_ddf_101=zero(T); s_ddf_011=zero(T); s_ddf_111=zero(T)
            s_ffd_000=zero(T); s_ffd_100=zero(T); s_ffd_010=zero(T); s_ffd_110=zero(T)
            s_ffd_001=zero(T); s_ffd_101=zero(T); s_ffd_011=zero(T); s_ffd_111=zero(T)
            s_dfd_000=zero(T); s_dfd_100=zero(T); s_dfd_010=zero(T); s_dfd_110=zero(T)
            s_dfd_001=zero(T); s_dfd_101=zero(T); s_dfd_011=zero(T); s_dfd_111=zero(T)
            s_fdd_000=zero(T); s_fdd_100=zero(T); s_fdd_010=zero(T); s_fdd_110=zero(T)
            s_fdd_001=zero(T); s_fdd_101=zero(T); s_fdd_011=zero(T); s_fdd_111=zero(T)
            s_ddd_000=zero(T); s_ddd_100=zero(T); s_ddd_010=zero(T); s_ddd_110=zero(T)
            s_ddd_001=zero(T); s_ddd_101=zero(T); s_ddd_011=zero(T); s_ddd_111=zero(T)

            @inbounds for n in -(itp.eqs[3]-1):itp.eqs[3]
                col_z = n + itp.eqs[3]
                kz_f0 = itp.kernel_pre[3][idx_z, col_z]
                kz_f1 = itp.kernel_pre[3][idx_z+1, col_z]
                kz_d0 = itp.kernel_d1_pre[3][idx_z, col_z]
                kz_d1 = itp.kernel_d1_pre[3][idx_z+1, col_z]

                @inbounds for m in -(itp.eqs[2]-1):itp.eqs[2]
                    col_y = m + itp.eqs[2]
                    ky_f0 = itp.kernel_pre[2][idx_y, col_y]
                    ky_f1 = itp.kernel_pre[2][idx_y+1, col_y]
                    ky_d0 = itp.kernel_d1_pre[2][idx_y, col_y]
                    ky_d1 = itp.kernel_d1_pre[2][idx_y+1, col_y]

                    @inbounds for l in -(itp.eqs[1]-1):itp.eqs[1]
                        abs_l = i + l
                        abs_m = j + m
                        abs_n = k + n
                        coef = if abs_l >= 1 && abs_l <= n1 && abs_m >= 1 && abs_m <= n2 && abs_n >= 1 && abs_n <= n3
                            itp.coefs[abs_l, abs_m, abs_n]
                        else
                            itp.lazy_workspace.stencil_buf[l + itp.eqs[1], m + itp.eqs[2], n + itp.eqs[3]]
                        end
                        col_x = l + itp.eqs[1]
                        kx_f0 = itp.kernel_pre[1][idx_x, col_x]
                        kx_f1 = itp.kernel_pre[1][idx_x+1, col_x]
                        kx_d0 = itp.kernel_d1_pre[1][idx_x, col_x]
                        kx_d1 = itp.kernel_d1_pre[1][idx_x+1, col_x]

                        # fff
                        s_fff_000 += coef*kx_f0*ky_f0*kz_f0; s_fff_100 += coef*kx_f1*ky_f0*kz_f0
                        s_fff_010 += coef*kx_f0*ky_f1*kz_f0; s_fff_110 += coef*kx_f1*ky_f1*kz_f0
                        s_fff_001 += coef*kx_f0*ky_f0*kz_f1; s_fff_101 += coef*kx_f1*ky_f0*kz_f1
                        s_fff_011 += coef*kx_f0*ky_f1*kz_f1; s_fff_111 += coef*kx_f1*ky_f1*kz_f1
                        # dff
                        s_dff_000 += coef*kx_d0*ky_f0*kz_f0; s_dff_100 += coef*kx_d1*ky_f0*kz_f0
                        s_dff_010 += coef*kx_d0*ky_f1*kz_f0; s_dff_110 += coef*kx_d1*ky_f1*kz_f0
                        s_dff_001 += coef*kx_d0*ky_f0*kz_f1; s_dff_101 += coef*kx_d1*ky_f0*kz_f1
                        s_dff_011 += coef*kx_d0*ky_f1*kz_f1; s_dff_111 += coef*kx_d1*ky_f1*kz_f1
                        # fdf
                        s_fdf_000 += coef*kx_f0*ky_d0*kz_f0; s_fdf_100 += coef*kx_f1*ky_d0*kz_f0
                        s_fdf_010 += coef*kx_f0*ky_d1*kz_f0; s_fdf_110 += coef*kx_f1*ky_d1*kz_f0
                        s_fdf_001 += coef*kx_f0*ky_d0*kz_f1; s_fdf_101 += coef*kx_f1*ky_d0*kz_f1
                        s_fdf_011 += coef*kx_f0*ky_d1*kz_f1; s_fdf_111 += coef*kx_f1*ky_d1*kz_f1
                        # ddf
                        s_ddf_000 += coef*kx_d0*ky_d0*kz_f0; s_ddf_100 += coef*kx_d1*ky_d0*kz_f0
                        s_ddf_010 += coef*kx_d0*ky_d1*kz_f0; s_ddf_110 += coef*kx_d1*ky_d1*kz_f0
                        s_ddf_001 += coef*kx_d0*ky_d0*kz_f1; s_ddf_101 += coef*kx_d1*ky_d0*kz_f1
                        s_ddf_011 += coef*kx_d0*ky_d1*kz_f1; s_ddf_111 += coef*kx_d1*ky_d1*kz_f1
                        # ffd
                        s_ffd_000 += coef*kx_f0*ky_f0*kz_d0; s_ffd_100 += coef*kx_f1*ky_f0*kz_d0
                        s_ffd_010 += coef*kx_f0*ky_f1*kz_d0; s_ffd_110 += coef*kx_f1*ky_f1*kz_d0
                        s_ffd_001 += coef*kx_f0*ky_f0*kz_d1; s_ffd_101 += coef*kx_f1*ky_f0*kz_d1
                        s_ffd_011 += coef*kx_f0*ky_f1*kz_d1; s_ffd_111 += coef*kx_f1*ky_f1*kz_d1
                        # dfd
                        s_dfd_000 += coef*kx_d0*ky_f0*kz_d0; s_dfd_100 += coef*kx_d1*ky_f0*kz_d0
                        s_dfd_010 += coef*kx_d0*ky_f1*kz_d0; s_dfd_110 += coef*kx_d1*ky_f1*kz_d0
                        s_dfd_001 += coef*kx_d0*ky_f0*kz_d1; s_dfd_101 += coef*kx_d1*ky_f0*kz_d1
                        s_dfd_011 += coef*kx_d0*ky_f1*kz_d1; s_dfd_111 += coef*kx_d1*ky_f1*kz_d1
                        # fdd
                        s_fdd_000 += coef*kx_f0*ky_d0*kz_d0; s_fdd_100 += coef*kx_f1*ky_d0*kz_d0
                        s_fdd_010 += coef*kx_f0*ky_d1*kz_d0; s_fdd_110 += coef*kx_f1*ky_d1*kz_d0
                        s_fdd_001 += coef*kx_f0*ky_d0*kz_d1; s_fdd_101 += coef*kx_f1*ky_d0*kz_d1
                        s_fdd_011 += coef*kx_f0*ky_d1*kz_d1; s_fdd_111 += coef*kx_f1*ky_d1*kz_d1
                        # ddd
                        s_ddd_000 += coef*kx_d0*ky_d0*kz_d0; s_ddd_100 += coef*kx_d1*ky_d0*kz_d0
                        s_ddd_010 += coef*kx_d0*ky_d1*kz_d0; s_ddd_110 += coef*kx_d1*ky_d1*kz_d0
                        s_ddd_001 += coef*kx_d0*ky_d0*kz_d1; s_ddd_101 += coef*kx_d1*ky_d0*kz_d1
                        s_ddd_011 += coef*kx_d0*ky_d1*kz_d1; s_ddd_111 += coef*kx_d1*ky_d1*kz_d1
                    end
                end
            end

            # Reduce along x: 8 cubic Hermite calls (one per {y-type}{z-type}_{y-bracket}{z-bracket})
            v_ff_00 = cubic_hermite(t_x, s_fff_000, s_fff_100, s_dff_000, s_dff_100, h_pre_x)
            v_ff_10 = cubic_hermite(t_x, s_fff_010, s_fff_110, s_dff_010, s_dff_110, h_pre_x)
            v_ff_01 = cubic_hermite(t_x, s_fff_001, s_fff_101, s_dff_001, s_dff_101, h_pre_x)
            v_ff_11 = cubic_hermite(t_x, s_fff_011, s_fff_111, s_dff_011, s_dff_111, h_pre_x)
            v_df_00 = cubic_hermite(t_x, s_fdf_000, s_fdf_100, s_ddf_000, s_ddf_100, h_pre_x)
            v_df_10 = cubic_hermite(t_x, s_fdf_010, s_fdf_110, s_ddf_010, s_ddf_110, h_pre_x)
            v_df_01 = cubic_hermite(t_x, s_fdf_001, s_fdf_101, s_ddf_001, s_ddf_101, h_pre_x)
            v_df_11 = cubic_hermite(t_x, s_fdf_011, s_fdf_111, s_ddf_011, s_ddf_111, h_pre_x)

            # Reduce along y: 4 cubic Hermite calls (one per {z-type}_{z-bracket})
            w_f_0 = cubic_hermite(t_y, v_ff_00, v_ff_10, v_df_00, v_df_10, h_pre_y)
            w_f_1 = cubic_hermite(t_y, v_ff_01, v_ff_11, v_df_01, v_df_11, h_pre_y)

            # Now we need the z-derivative versions for the final z-Hermite
            v_fd_00 = cubic_hermite(t_x, s_ffd_000, s_ffd_100, s_dfd_000, s_dfd_100, h_pre_x)
            v_fd_10 = cubic_hermite(t_x, s_ffd_010, s_ffd_110, s_dfd_010, s_dfd_110, h_pre_x)
            v_fd_01 = cubic_hermite(t_x, s_ffd_001, s_ffd_101, s_dfd_001, s_dfd_101, h_pre_x)
            v_fd_11 = cubic_hermite(t_x, s_ffd_011, s_ffd_111, s_dfd_011, s_dfd_111, h_pre_x)
            v_dd_00 = cubic_hermite(t_x, s_fdd_000, s_fdd_100, s_ddd_000, s_ddd_100, h_pre_x)
            v_dd_10 = cubic_hermite(t_x, s_fdd_010, s_fdd_110, s_ddd_010, s_ddd_110, h_pre_x)
            v_dd_01 = cubic_hermite(t_x, s_fdd_001, s_fdd_101, s_ddd_001, s_ddd_101, h_pre_x)
            v_dd_11 = cubic_hermite(t_x, s_fdd_011, s_fdd_111, s_ddd_011, s_ddd_111, h_pre_x)

            w_d_0 = cubic_hermite(t_y, v_fd_00, v_fd_10, v_dd_00, v_dd_10, h_pre_y)
            w_d_1 = cubic_hermite(t_y, v_fd_01, v_fd_11, v_dd_01, v_dd_11, h_pre_y)

            # Reduce along z: 1 cubic Hermite call
            result = cubic_hermite(t_z, w_f_0, w_f_1, w_d_0, w_d_1, h_pre_z)

            return result * (-one(T)/itp.h[1])^DO[1] * (-one(T)/itp.h[2])^DO[2] * (-one(T)/itp.h[3])^DO[3]
        end
    end
end