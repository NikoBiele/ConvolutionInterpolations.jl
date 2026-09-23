"""
Per-dimension fast convolution interpolation functors for `FastConvolutionInterpolation`.

Dispatch on a mixed kernel type (`AbstractMixedConvolutionKernel`): kernels and/or derivative
orders differ per dimension. Each dimension's kernel weights come from its own exact column
polynomials (see `_kernel_weights`), so every combination is evaluated exactly, with one product
per coefficient in the tensor-product support.

2D and 3D: nested reductions over the support. ND: tensor product over all support points.

See also: FastConvolutionInterpolation, _column_weights_per_dim.
"""
# ---------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------

@inline function _perdim_scale(h::NTuple{N,T}, ::Val{DO}) where {N,T,DO}
    s = one(T)
    @inbounds for d in 1:N
        s *= (-one(T) / h[d])^DO[d]
    end
    s
end

# ==============================================================
# 2D
# ==============================================================
@inline function (itp::FastConvolutionInterpolation{T,2,0,TCoefs,Axs,KA,Val{2},DG,EQ,PR,KP,KBC,
            DerivativeOrder{DO},FD,SD,Val{SG},Val{false},Val{0}})(x::Vararg{Number,2}) where
            {T<:AbstractFloat,TCoefs<:AbstractArray{T,2},Axs<:Tuple{<:AbstractVector,<:AbstractVector},
            KA<:Tuple{<:Nothing,<:Nothing},DG<:AbstractMixedConvolutionKernel,EQ<:Tuple{Int,Int},
            PR<:Tuple{<:AbstractVector,<:AbstractVector},KP,KBC<:Tuple{<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol}},
            DO,FD,SD,SG}

    x = T.(x)
    eqs_x, eqs_y = itp.eqs

    # Grid positions, and τ = diff_right within the cell
    i_float = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T)
    i = clamp(floor(Int, i_float), eqs_x, length(itp.knots[1]) - eqs_x)
    x_diff_right = one(T) - (x[1] - itp.knots[1][i]) / itp.h[1]

    j_float = (x[2] - itp.knots[2][1]) / itp.h[2] + one(T)
    j = clamp(floor(Int, j_float), eqs_y, length(itp.knots[2]) - eqs_y)
    y_diff_right = one(T) - (x[2] - itp.knots[2][j]) / itp.h[2]

    # Kernel weights per dimension, each with its own kernel and derivative order
    wx, wy = _column_weights_per_dim(Val(_kernel_sym(itp.kernel_sym)), Val(DO),
                                     (x_diff_right, y_diff_right))

    # Tensor sum: one product per coefficient
    ox = i - eqs_x
    oy = j - eqs_y
    result = zero(T)
    @inbounds for ky in 1:length(wy)
        row = zero(T)
        @simd for kx in 1:length(wx)
            row += itp.coefs[ox + kx, oy + ky] * wx[kx]
        end
        result += row * wy[ky]
    end

    return result * _perdim_scale(itp.h, Val(DO))
end