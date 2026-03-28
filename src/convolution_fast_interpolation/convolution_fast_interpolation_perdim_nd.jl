# ==============================================================
# ND — linear only per dim
# ==============================================================
function (itp::FastConvolutionInterpolation{T,N,0,TCoefs,Axs,KA,HigherDimension{N},DG,EQ,PR,KP,KBC,
            DerivativeOrder{DO},FD,SD,Val{SG},Val{false},Val{0}})(x::Vararg{Number,N}) where
            {T<:AbstractFloat,N,TCoefs<:AbstractArray{T,N},
            Axs<:Tuple{Vararg{AbstractVector}},
            KA<:Tuple{Vararg{Nothing}},DG<:AbstractMixedConvolutionKernel,EQ<:Tuple{Vararg{Int}},
            PR<:Tuple{Vararg{AbstractVector}},KP,KBC<:Tuple{Vararg{Tuple{Symbol,Symbol}}},
            DO,FD,SD,SG}
            
    pos_ids = ntuple(d -> clamp(floor(Int, (x[d]-itp.knots[d][1])/itp.h[d]+one(T)),
                                itp.eqs[d], length(itp.knots[d])-itp.eqs[d]), N)
    idx_lower = ntuple(N) do d
        dr = one(T)-(x[d]-itp.knots[d][pos_ids[d]])/itp.h[d]
        n  = length(itp.pre_range[d])
        clamp(floor(Int, dr*T(n-1))+1, 1, n-1)
    end
    t = ntuple(N) do d
        dr   = one(T)-(x[d]-itp.knots[d][pos_ids[d]])/itp.h[d]
        pre  = itp.pre_range[d]
        cidx = dr*T(length(pre)-1)+one(T)
        cidx - T(clamp(floor(Int,cidx), 1, length(pre)-1))
    end
    result = zero(T)
    @inbounds for offsets in Iterators.product(ntuple(d -> -(itp.eqs[d]-1):itp.eqs[d], N)...)
        c = itp.coefs[(pos_ids .+ offsets)...]
        kernel_val = one(T)
        @inbounds for d in 1:N
            il  = idx_lower[d]; col = offsets[d]+itp.eqs[d]
            k_lo = itp.kernel_pre[d][il, col]; k_hi = itp.kernel_pre[d][il+1, col]
            kernel_val *= (one(T)-t[d])*k_lo + t[d]*k_hi
        end
        result += c * kernel_val
    end
    return result * _perdim_scale(itp.h, Val(DO))
end