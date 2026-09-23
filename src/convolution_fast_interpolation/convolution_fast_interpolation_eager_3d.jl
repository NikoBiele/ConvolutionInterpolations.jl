"""
    (itp::FastConvolutionInterpolation{T,3,...})(x::Number, y::Number, z::Number)

Evaluate 3D fast convolution interpolation at coordinates `(x, y, z)`.

Dispatches on kernel type:

**Specialized kernels**:
- `:a0` — Nearest neighbor, ~13ns
- `:a1` — Trilinear interpolation, ~15ns

**Higher-order kernels**: exact column polynomials. The kernel weights of each dimension are
evaluated once from compile-time polynomial coefficients (see `_column_rows`), and the 3D kernel
is their tensor product `K_3D(x,y,z) = K_1D(x) * K_1D(y) * K_1D(z)`: one product per
coefficient in the `(2*eqs)^3` support.

O(1) evaluation time, allocation-free, exact to rounding for every derivative order.

Results scaled by `(-1/h_x)^derivative * (-1/h_y)^derivative * (-1/h_z)^derivative`.

See also: `FastConvolutionInterpolation`, `_column_weights_per_dim`.
"""

@inline function (itp::FastConvolutionInterpolation{T,3,0,TCoefs,Axs,KA,Val{3},LowerOrderKernel{(:a0,:a0,:a0)},
                    EQ,PR,KP,KBC,DerivativeOrder{DO},FD,SD,Val{(:linear,:linear,:linear)},
                    Val{false},Val{0}})(x::Vararg{Number,3}) where {T<:AbstractFloat,TCoefs<:AbstractArray{T,3},
                    Axs<:Tuple{<:AbstractVector,<:AbstractVector,<:AbstractVector},
                    KA<:Tuple{<:Nothing,<:Nothing,<:Nothing},EQ<:Tuple{Int,Int,Int},
                    PR<:Tuple{<:AbstractVector,<:AbstractVector,<:AbstractVector},KP,
                    KBC<:Tuple{<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol}},DO,FD,SD}

    x = T.(x)
    # specialized dispatch for 3d nearest neighbor kernel
    # First dimension (x)
    i_float = (x[1] - itp.x0[1]) / itp.h[1] + one(T)
    i = clamp(floor(Int, i_float), itp.eqs[1], length(itp.knots[1]) - itp.eqs[1])
    x_diff_left = i_float - T(i)

    # Second dimension (y)
    j_float = (x[2] - itp.x0[2]) / itp.h[2] + one(T)
    j = clamp(floor(Int, j_float), itp.eqs[2], length(itp.knots[2]) - itp.eqs[2])
    y_diff_left = j_float - T(j)

    # Third dimension (z)
    k_float = (x[3] - itp.x0[3]) / itp.h[3] + one(T)
    k = clamp(floor(Int, k_float), itp.eqs[3], length(itp.knots[3]) - itp.eqs[3])
    z_diff_left = k_float - T(k)

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
                    Val{false},Val{0}})(x::Vararg{Number,3}) where {T<:AbstractFloat,TCoefs<:AbstractArray{T,3},
                    Axs<:NTuple{3,<:AbstractVector},KA<:NTuple{3,<:Nothing},EQ<:NTuple{3,Int},
                    PR<:NTuple{3,<:AbstractVector},KP,
                    KBC<:Tuple{<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol}},DO,FD,SD}
                    
    x = T.(x)
    # specialized dispatch for 3d linear kernel
    # First dimension (x)
    i_float = (x[1] - itp.x0[1]) / itp.h[1] + one(T)
    i = clamp(floor(Int, i_float), itp.eqs[1], length(itp.knots[1]) - itp.eqs[1])
    x_diff_left = i_float - T(i)

    # Second dimension (y)
    j_float = (x[2] - itp.x0[2]) / itp.h[2] + one(T)
    j = clamp(floor(Int, j_float), itp.eqs[2], length(itp.knots[2]) - itp.eqs[2])
    y_diff_left = j_float - T(j)

    # Third dimension (z)
    k_float = (x[3] - itp.x0[3]) / itp.h[3] + one(T)
    k = clamp(floor(Int, k_float), itp.eqs[3], length(itp.knots[3]) - itp.eqs[3])
    z_diff_left = k_float - T(k)

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

function (itp::FastConvolutionInterpolation{T,3,0,TCoefs,Axs,KA,Val{3},HigherOrderKernel{DG},
                EQ,PR,KP,KBC,DerivativeOrder{DO},FD,SD,Val{SG},Val{false},Val{0}})(x::Vararg{Number,3}) where 
                {T<:AbstractFloat,TCoefs<:AbstractArray{T,3},
                Axs<:NTuple{3,<:AbstractVector},KA<:NTuple{3,<:Nothing},DG,EQ<:NTuple{3,Int},
                PR<:NTuple{3,<:AbstractVector},KP,
                KBC<:Tuple{<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol}},
                DO,FD,SD,SG}

    x = T.(x)
    # Grid positions
    i_float = (x[1] - itp.x0[1]) / itp.h[1] + one(T)
    i = clamp(floor(Int, i_float), itp.eqs[1], length(itp.knots[1]) - itp.eqs[1])
    x_diff_right = one(T) + T(i) - i_float

    j_float = (x[2] - itp.x0[2]) / itp.h[2] + one(T)
    j = clamp(floor(Int, j_float), itp.eqs[2], length(itp.knots[2]) - itp.eqs[2])
    y_diff_right = one(T) + T(j) - j_float

    k_float = (x[3] - itp.x0[3]) / itp.h[3] + one(T)
    k = clamp(floor(Int, k_float), itp.eqs[3], length(itp.knots[3]) - itp.eqs[3])
    z_diff_right = one(T) + T(k) - k_float

    # Kernel weights per dimension at τ = diff_right (column c ↔ coefficient offset + c)
    wx, wy, wz = _column_weights_per_dim(Val(DG), Val(DO), (x_diff_right, y_diff_right, z_diff_right))

    # Tensor sum: one product per coefficient
    ox = i - itp.eqs[1]
    oy = j - itp.eqs[2]
    oz = k - itp.eqs[3]
    result = zero(T)
    @inbounds for kz in 1:length(wz)
        plane = zero(T)
        for ky in 1:length(wy)
            row = zero(T)
            @simd for kx in 1:length(wx)
                row += itp.coefs[ox + kx, oy + ky, oz + kz] * wx[kx]
            end
            plane += row * wy[ky]
        end
        result += plane * wz[kz]
    end

    return result * (-one(T)/itp.h[1])^DO[1] * (-one(T)/itp.h[2])^DO[2] * (-one(T)/itp.h[3])^DO[3]
end