# ==============================================================
# ND — per-dimension kernels and derivative orders
# ==============================================================
function (itp::FastConvolutionInterpolation{T,N,0,TCoefs,Axs,KA,HigherDimension{N},DG,EQ,PR,KP,KBC,
            DerivativeOrder{DO},FD,SD,Val{SG},Val{false},Val{0}})(x::Vararg{Number,N}) where
            {T<:AbstractFloat,N,TCoefs<:AbstractArray{T,N},
            Axs<:NTuple{N,<:AbstractVector},
            KA<:NTuple{N,<:Nothing},DG<:AbstractMixedConvolutionKernel,EQ<:NTuple{N,Int},
            PR<:NTuple{N,<:AbstractVector},KP,KBC<:NTuple{N,Tuple{Symbol,Symbol}},
            DO,FD,SD,SG}

    x = T.(x)
    # Grid positions, and τ = diff_right within the cell
    pos_ids = ntuple(d -> clamp(floor(Int, (x[d]-itp.knots[d][1])/itp.h[d]+one(T)),
                                itp.eqs[d], length(itp.knots[d])-itp.eqs[d]), N)
    diff_right = ntuple(d -> one(T) - (x[d] - itp.knots[d][pos_ids[d]]) / itp.h[d], N)

    # Kernel weights per dimension, each with its own kernel and derivative order
    w = _column_weights_per_dim(Val(_kernel_sym(itp.kernel_sym)), Val(DO), diff_right)

    result = zero(T)
    @inbounds for offsets in Iterators.product(ntuple(d -> -(itp.eqs[d]-1):itp.eqs[d], N)...)
        c = itp.coefs[(pos_ids .+ offsets)...]
        kernel_val = prod(ntuple(d -> w[d][offsets[d] + itp.eqs[d]], Val(N)))
        result += c * kernel_val
    end
    return result * _perdim_scale(itp.h, Val(DO))
end