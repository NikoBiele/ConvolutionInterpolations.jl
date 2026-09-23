"""
    (itp::FastConvolutionInterpolation{T,2,...})(x::Number, y::Number)

Evaluate 2D fast convolution interpolation at coordinates `(x, y)`.

Dispatches on kernel type:

**Specialized kernels**:
- `:a0` — Nearest neighbor, ~7ns
- `:a1` — Bilinear interpolation, ~8ns

**Higher-order kernels**: exact column polynomials. The kernel weights of each dimension are
evaluated once from compile-time polynomial coefficients (see `_column_rows`), and the 2D kernel
is their tensor product `K_2D(x,y) = K_1D(x) * K_1D(y)`: one product per coefficient in the
`(2*eqs)^2` support.

O(1) evaluation time, allocation-free, exact to rounding for every derivative order.

Results scaled by `(-1/h_x)^derivative * (-1/h_y)^derivative`.

See also: `FastConvolutionInterpolation`, `_column_weights_per_dim`.
"""

@inline function (itp::FastConvolutionInterpolation{T,2,0,TCoefs,Axs,KA,Val{2},LowerOrderKernel{(:a0, :a0)},
                    EQ,PR,KP,KBC,DerivativeOrder{DO},FD,SD,Val{(:linear,:linear)},Val{false},Val{0}})(x::Vararg{Number,2}) where 
                    {T<:AbstractFloat,TCoefs<:AbstractArray{T,2},Axs<:Tuple{<:AbstractVector,<:AbstractVector},
                    KA<:Tuple{<:Nothing,<:Nothing},EQ<:Tuple{Int,Int},PR<:Tuple{<:AbstractVector,<:AbstractVector},
                    KP,KBC<:Tuple{<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol}},DO,FD,SD}

    x = T.(x)
    # specialized dispatch for 2d nearest neighbor kernel 
    # First dimension (x)
    i_float = (x[1] - itp.x0[1]) / itp.h[1] + one(T)
    i = clamp(floor(Int, i_float), itp.eqs[1], length(itp.knots[1]) - itp.eqs[1])
    x_diff_left = i_float - T(i)

    # Second dimension (y)
    j_float = (x[2] - itp.x0[2]) / itp.h[2] + one(T)
    j = clamp(floor(Int, j_float), itp.eqs[2], length(itp.knots[2]) - itp.eqs[2])
    y_diff_left = j_float - T(j)

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
                    EQ,PR,KP,KBC,DerivativeOrder{DO},FD,SD,Val{(:linear,:linear)},Val{false},Val{0}})(x::Vararg{Number,2}) where 
                    {T<:AbstractFloat,TCoefs<:AbstractArray{T,2},Axs<:Tuple{<:AbstractVector,<:AbstractVector},
                    KA<:Tuple{<:Nothing,<:Nothing},EQ<:Tuple{Int,Int},PR<:Tuple{<:AbstractVector,<:AbstractVector},
                    KP,KBC<:Tuple{<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol}},DO,FD,SD}

    x = T.(x)
    # specialized dispatch for 2d linear kernel
    # First dimension (x)
    i_float = (x[1] - itp.x0[1]) / itp.h[1] + one(T)
    i = clamp(floor(Int, i_float), itp.eqs[1], length(itp.knots[1]) - itp.eqs[1])
    x_diff_left = i_float - T(i)

    # Second dimension (y)
    j_float = (x[2] - itp.x0[2]) / itp.h[2] + one(T)
    j = clamp(floor(Int, j_float), itp.eqs[2], length(itp.knots[2]) - itp.eqs[2])
    y_diff_left = j_float - T(j)

    # Bilinear interpolation
    return @inbounds @fastmath ((1-x_diff_left)*(1-y_diff_left)*itp.coefs[i, j] + 
                        x_diff_left*(1-y_diff_left)*itp.coefs[i+1, j] + 
                        (1-x_diff_left)*y_diff_left*itp.coefs[i, j+1] + 
                        x_diff_left*y_diff_left*itp.coefs[i+1, j+1]) *
                        (-one(T)/itp.h[1])^DO[1] * (-one(T)/itp.h[2])^DO[2]
end

function (itp::FastConvolutionInterpolation{T,2,0,TCoefs,Axs,KA,Val{2},
            HigherOrderKernel{DG},EQ,PR,KP,KBC,DerivativeOrder{DO},FD,SD,Val{SG},Val{false},Val{0}})(x::Vararg{Number,2}) where 
            {T<:AbstractFloat,TCoefs<:AbstractArray{T,2},Axs<:Tuple{<:AbstractVector,<:AbstractVector},
            KA<:Tuple{<:Nothing,<:Nothing},DG,EQ<:Tuple{Int,Int},PR<:Tuple{<:AbstractVector,<:AbstractVector},
            KP,KBC<:Tuple{<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol}},DO,FD,SD,SG}

    x = T.(x)
    # First dimension (x)
    i_float = (x[1] - itp.x0[1]) / itp.h[1] + one(T)
    i = clamp(floor(Int, i_float), itp.eqs[1], length(itp.knots[1]) - itp.eqs[1])
    x_diff_left = i_float - T(i)
    x_diff_right = one(T) - x_diff_left

    # Second dimension (y)
    j_float = (x[2] - itp.x0[2]) / itp.h[2] + one(T)
    j = clamp(floor(Int, j_float), itp.eqs[2], length(itp.knots[2]) - itp.eqs[2])
    y_diff_left = j_float - T(j)
    y_diff_right = one(T) - y_diff_left

    # Kernel weights per dimension at τ = diff_right (column k ↔ coefficient offset + k)
    wx, wy = _column_weights_per_dim(Val(DG), Val(DO), (x_diff_right, y_diff_right))

    # Tensor sum: one product per coefficient
    ox = i - itp.eqs[1]
    oy = j - itp.eqs[2]
    result = zero(T)
    @inbounds for ky in 1:length(wy)
        row = zero(T)
        @simd for kx in 1:length(wx)
            row += itp.coefs[ox + kx, oy + ky] * wx[kx]
        end
        result += row * wy[ky]
    end

    return result * (-one(T)/itp.h[1])^DO[1] * (-one(T)/itp.h[2])^DO[2]
end