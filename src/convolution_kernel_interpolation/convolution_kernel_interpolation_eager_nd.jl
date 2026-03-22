@inline function (itp::ConvolutionInterpolation{T,N,TCoefs,IT,Axs,KA,HigherDimension{N},
                    DG,EQ,KBC,DerivativeOrder{DO},FD,SD,SG,NB,Val{false}})(x::Vararg{Number,N}) where 
                    {T,N,TCoefs,IT,KA<:NTuple{N,<:ConvolutionKernel},Axs,DG,EQ<:NTuple{N,Int},KBC,DO,FD,SD,SG,NB<:Nothing}

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