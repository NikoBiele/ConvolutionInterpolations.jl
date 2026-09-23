# ==============================================================
# 3D — per-dimension kernels and derivative orders
# ==============================================================
function (itp::FastConvolutionInterpolation{T,3,0,TCoefs,Axs,KA,Val{3},DG,EQ,PR,KP,KBC,
            DerivativeOrder{DO},FD,SD,Val{SG},Val{false},Val{0}})(x::Vararg{Number,3}) where
            {T<:AbstractFloat,TCoefs<:AbstractArray{T,3},
            Axs<:Tuple{<:AbstractVector,<:AbstractVector,<:AbstractVector},
            KA<:Tuple{<:Nothing,<:Nothing,<:Nothing},DG<:AbstractMixedConvolutionKernel,EQ<:Tuple{Int,Int,Int},
            PR<:Tuple{<:AbstractVector,<:AbstractVector,<:AbstractVector},KP,
            KBC<:Tuple{<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol}},DO,FD,SD,SG}

    x = T.(x)
    eqs_x, eqs_y, eqs_z = itp.eqs

    # Grid positions, and τ = diff_right within the cell
    i_float = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T)
    i = clamp(floor(Int, i_float), eqs_x, length(itp.knots[1]) - eqs_x)
    x_diff_right = one(T) - (x[1] - itp.knots[1][i]) / itp.h[1]

    j_float = (x[2] - itp.knots[2][1]) / itp.h[2] + one(T)
    j = clamp(floor(Int, j_float), eqs_y, length(itp.knots[2]) - eqs_y)
    y_diff_right = one(T) - (x[2] - itp.knots[2][j]) / itp.h[2]

    k_float = (x[3] - itp.knots[3][1]) / itp.h[3] + one(T)
    k = clamp(floor(Int, k_float), eqs_z, length(itp.knots[3]) - eqs_z)
    z_diff_right = one(T) - (x[3] - itp.knots[3][k]) / itp.h[3]

    # Kernel weights per dimension, each with its own kernel and derivative order
    wx, wy, wz = _column_weights_per_dim(Val(_kernel_sym(itp.kernel_sym)), Val(DO),
                                         (x_diff_right, y_diff_right, z_diff_right))

    # Tensor sum: one product per coefficient
    ox = i - eqs_x
    oy = j - eqs_y
    oz = k - eqs_z
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

    return result * _perdim_scale(itp.h, Val(DO))
end