"""
(itp::FastConvolutionInterpolation{T,1,...})(x::Number)
Evaluate 1D lazy fast convolution interpolation at coordinate x.
Dispatches on kernel type:

:a0 — Nearest neighor, ~4ns
:a1 — Linear interpolation, ~4ns
Higher-order kernels — exact column polynomials (see `_kernel_weights`). Near the boundaries,
ghost values are computed on the fly from the raw data, exactly as eager mode stores them
(see `_lazy_patch_sum`).

O(1) evaluation time, allocation-free, exact to rounding for every derivative order.
Results scaled by (-1/h)^derivative for derivative evaluation.
See also: FastConvolutionInterpolation, _kernel_weights.
"""

@inline function (itp::FastConvolutionInterpolation{T,1,0,TCoefs,Axs,KA,Val{1},LowerOrderKernel{DG},
                    EQ,KBC,DerivativeOrder{DO},FD,SD,Val{SG},Val{true},Val{0}})(x::Vararg{Number,1}) where 
                    {T<:AbstractFloat,TCoefs<:AbstractArray{T,1},Axs<:Tuple{<:AbstractVector},
                    KA<:Tuple{<:Nothing},DG,EQ<:Tuple{Int},KBC<:Tuple{<:Tuple{Symbol,Symbol}},DO,FD,SD,SG}

    x = T.(x)
    if DG[1] == :a1
        # specialized dispatch for 1d linear kernel
        i_float = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T)  # +1 for 1-based indexing
        i = clamp(floor(Int, i_float), itp.eqs[1], length(itp.knots[1]) - itp.eqs[1])
        x_diff_left = (x[1] - itp.knots[1][i]) / itp.h[1]  # Guaranteed in [0,1]
        return @inbounds ((1-x_diff_left) * itp.coefs[i] + x_diff_left * itp.coefs[i+1])
    elseif DG[1] == :a0
        # specialized dispatch for 1d nearest neighbor kernel
        i_float = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T) # +1 for 1-based indexing
        i = clamp(floor(Int, i_float), itp.eqs[1], length(itp.knots[1]) - itp.eqs[1])
        x_diff_left = (x[1] - itp.knots[1][i]) / itp.h[1]  # Guaranteed in [0,1]
        if x_diff_left < 0.5
            return itp.coefs[i]
        else
            return itp.coefs[i+1]
        end
    end
end

@inline function (itp::FastConvolutionInterpolation{T,1,0,TCoefs,Axs,KA,Val{1},
                    HigherOrderKernel{DG},EQ,KBC,DerivativeOrder{DO},
                    FD,SD,Val{SG},Val{true},Val{0}})(x::Vararg{Number,1}) where
                    {T<:AbstractFloat,TCoefs<:AbstractArray{T,1},
                    Axs<:Tuple{<:AbstractVector},KA<:Tuple{<:Nothing},DG,
                    EQ<:Tuple{Int},
                    KBC<:Tuple{<:Tuple{Symbol,Symbol}},DO,FD,SD,SG}

    x = T.(x)
    # Direct index calculation
    i_float = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T)
    i = clamp(floor(Int, i_float), 1, length(itp.knots[1]) - 1)
    is_boundary = is_boundary_stencil(i, size(itp.coefs, 1)[1], itp.eqs[1])

    x_diff_left = (x[1] - itp.knots[1][i]) / itp.h[1]
    x_diff_right = one(T) - x_diff_left

    # Kernel weights of all 2·eqs columns at τ = x_diff_right (column k ↔ index i + k − eqs)
    w = _kernel_weights(Val(DG[1]), Val(DO[1]), x_diff_right)

    result = if is_boundary
        # the stencil reaches past the domain: ghosts formed on a local patch, as eager forms them
        _lazy_patch_sum(itp, (i,), (w,))
    else
        # the whole stencil lies inside the domain: a plain dot product, as in eager mode
        offset = i - itp.eqs[1]                           # stencil offset
        acc = zero(T)
        @inbounds @simd for k in 1:length(w)
            acc += itp.coefs[offset + k] * w[k]
        end
        acc
    end

    return result * (-one(T)/itp.h[1])^DO[1]
end