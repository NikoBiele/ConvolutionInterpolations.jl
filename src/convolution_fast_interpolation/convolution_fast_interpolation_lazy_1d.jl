"""
(itp::FastConvolutionInterpolation{T,1,...})(x::Number)
Evaluate 1D lazy fast convolution interpolation at coordinate x.
Dispatches on kernel type:

:a0 — Nearest neighor, ~4ns
:a1 — Linear interpolation, ~4ns
Higher-order kernels — exact column polynomials (see `_kernel_weights`). Near the boundaries,
ghost values are computed on the fly from the raw data (lazy mode); with `boundary_fallback`,
boundary stencils use linear interpolation instead.

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

    if is_boundary && itp.boundary_fallback

        # Specialized linear dispatch for boundary points
        x_diff_left = (x[1] - itp.knots[1][i]) / itp.h[1]  # Guaranteed in [0,1]
        return @inbounds ((1-x_diff_left) * itp.coefs[i] + x_diff_left * itp.coefs[i+1])

    else # internal point or boundary_fallback=false

        x_diff_left = (x[1] - itp.knots[1][i]) / itp.h[1]
        x_diff_right = one(T) - x_diff_left

        ng = itp.eqs[1] - 1
        n = itp.domain_size[1]
        kernel_type = _kernel_sym(itp.kernel_sym)

        if is_boundary
            if i - ng < 1
                ghost_matrix_l = get_polynomial_ghost_coeffs(itp.bc[1][1], kernel_type[1])
                y_slice_l = view(itp.coefs, 1:size(ghost_matrix_l, 2))
                mul!(view(itp.lazy_workspace.ghost_buf, 1:ng), view(ghost_matrix_l, 1:ng, :), y_slice_l)
            end
            if i + itp.eqs[1] > n
                ghost_matrix_r = get_polynomial_ghost_coeffs(itp.bc[1][2], kernel_type[1])
                ns_r = size(ghost_matrix_r, 2)
                for k in 1:ng
                    itp.lazy_workspace.ghost_buf[k] = sum(ghost_matrix_r[k, m] * itp.coefs[n - m + 1] for m in 1:ns_r)
                end
            end
        end

        # Kernel weights of all 2·eqs columns at τ = x_diff_right (column k ↔ index i + k − eqs)
        w = _kernel_weights(Val(DG[1]), Val(DO[1]), x_diff_right)

        # Dot product over the stencil; out-of-domain coefficients come from the ghost buffer
        result = zero(T)
        @inbounds for k in 1:length(w)
            abs_idx = i + k - itp.eqs[1]
            coef = if abs_idx < 1
                itp.lazy_workspace.ghost_buf[1 - abs_idx]
            elseif abs_idx > n
                itp.lazy_workspace.ghost_buf[abs_idx - n]
            else
                itp.coefs[abs_idx]
            end
            result += coef * w[k]
        end

        return result * (-one(T)/itp.h[1])^DO[1]
    end
end