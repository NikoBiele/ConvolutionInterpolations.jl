"""
(itp::FastConvolutionInterpolation{T,1,...})(x::Number)
Evaluate 1D fast convolution interpolation at coordinate x.
Dispatches on kernel type:

:a0 — Nearest neighbor, ~4ns
:a1 — Linear interpolation, ~4ns
Higher-order kernels — exact column polynomials: on each of the 2·eqs columns the kernel is a
single polynomial in the fractional cell position, so the weights of all coefficients are
evaluated at once from compile-time constant coefficients (see `_column_rows`), then combined
with the coefficients in one dot product.

O(1) evaluation time, allocation-free, exact to rounding for every derivative order.
Results scaled by (-1/h)^derivative for derivative evaluation.
See also: FastConvolutionInterpolation, _column_rows, _column_weights.
"""

@inline function (itp::FastConvolutionInterpolation{T,1,0,TCoefs,Axs,KA,Val{1},LowerOrderKernel{(:a0,)},
                    EQ,KBC,DerivativeOrder{DO},FD,SD,Val{SG},Val{false},Val{0}})(x::Vararg{Number,1}) where 
                    {T<:AbstractFloat,TCoefs<:AbstractArray{T,1},Axs<:Tuple{<:AbstractVector},
                    KA<:Tuple{<:Nothing},EQ<:Tuple{Int},KBC<:Tuple{<:Tuple{Symbol,Symbol}},DO,FD,SD,SG}

    x = T.(x)
    # specialized dispatch for 1d nearest neighbor kernel
    i_float = (x[1] - itp.x0[1]) / itp.h[1] + one(T)
    i = clamp(floor(Int, i_float), itp.eqs[1], length(itp.knots[1]) - itp.eqs[1])
    x_diff_left = i_float - T(i)
    if x_diff_left < 0.5
        return itp.coefs[i]
    else
        return itp.coefs[i+1]
    end
end

@inline function (itp::FastConvolutionInterpolation{T,1,0,TCoefs,Axs,KA,Val{1},LowerOrderKernel{(:a1,)},
                    EQ,KBC,DerivativeOrder{DO},FD,SD,Val{SG},Val{false},Val{0}})(x::Vararg{Number,1}) where 
                    {T<:AbstractFloat,TCoefs<:AbstractArray{T,1},Axs<:Tuple{<:AbstractVector},
                    KA<:Tuple{<:Nothing},EQ<:Tuple{Int},KBC<:Tuple{<:Tuple{Symbol,Symbol}},DO,FD,SD,SG}

    x = T.(x)
    # specialized dispatch for 1d linear kernel
    i_float = (x[1] - itp.x0[1]) / itp.h[1] + one(T)
    i = clamp(floor(Int, i_float), itp.eqs[1], length(itp.knots[1]) - itp.eqs[1])
    x_diff_left = i_float - T(i)
    return @inbounds ((1-x_diff_left) * itp.coefs[i] + x_diff_left * itp.coefs[i+1])
end

@inline function (itp::FastConvolutionInterpolation{T,1,0,TCoefs,Axs,KA,Val{1},
                    HigherOrderKernel{DG},EQ,KBC,DerivativeOrder{DO},FD,SD,Val{SG},Val{false},Val{0}})(x::Vararg{Number,1}) where 
                    {T<:AbstractFloat,TCoefs<:AbstractArray{T,1},Axs<:Tuple{<:AbstractVector},
                    KA<:Tuple{<:Nothing},DG,EQ<:Tuple{Int},KBC<:Tuple{<:Tuple{Symbol,Symbol}},DO,FD,SD,SG}

    x = T.(x)
    # Direct index calculation
    i_float = (x[1] - itp.x0[1]) / itp.h[1] + one(T)
    i = clamp(floor(Int, i_float), itp.eqs[1], length(itp.knots[1]) - itp.eqs[1])
    x_diff_left = i_float - T(i)
    x_diff_right = one(T) - x_diff_left

    # Kernel weights of all 2·eqs columns at τ = x_diff_right (column k ↔ coefficient i+k−eqs)
    rows = _column_rows(Val(DG[1]), Val(DO[1]), T)
    w = _column_weights(rows, x_diff_right)

    # One dot product with the coefficients in the stencil
    offset = i - itp.eqs[1]
    result = zero(T)
    @inbounds @simd for k in 1:length(w)
        result += itp.coefs[offset + k] * w[k]
    end

    scale = (-one(T)/itp.h[1])^DO[1]
    return result * scale
end