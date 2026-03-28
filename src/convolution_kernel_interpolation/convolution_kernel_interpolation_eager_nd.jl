@inline function (itp::ConvolutionInterpolation{T,N,0,TCoefs,Axs,KA,HigherDimension{N},
                    DG,EQ,KBC,DerivativeOrder{DO},FD,SD,Val{:not_used},NB,Val{false},Val{0}})(x::Vararg{Number,N}) where 
                    {T<:AbstractFloat,N,TCoefs<:AbstractArray{T,N},KA<:Tuple{Vararg{ConvolutionKernel}},
                    Axs<:Tuple{Vararg{AbstractVector}},DG,EQ<:Tuple{Vararg{Int}},
                    KBC<:Tuple{Vararg{Tuple{Symbol,Symbol}}},DO,FD,SD,NB<:Nothing}

    i_floats = ntuple(d -> (x[d] - itp.knots[d][1]) / itp.h[d] + one(T), N)
    pos_ids = ntuple(d -> clamp(floor(Int, i_floats[d]), itp.eqs[d], length(itp.knots[d]) - itp.eqs[d]), N)

    result = zero(T)
    @inbounds for offsets in Iterators.product(ntuple(d -> -(itp.eqs[d]-1):itp.eqs[d], N)...)
        result += itp.coefs[(pos_ids .+ offsets)...] *
                  prod(itp.kernel[d]((x[d] - itp.knots[d][pos_ids[d] + offsets[d]]) / itp.h[d]) for d in 1:N)
    end
    scale = prod(d -> one(T) / itp.h[d]^DO[d], 1:N)
    return @inbounds @fastmath result * scale
end