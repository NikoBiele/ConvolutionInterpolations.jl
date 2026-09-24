function (itp::FastConvolutionInterpolation{T,N,N,TCoefs,Axs,KA,HigherDimension{N},
        DG,EQ,KBC,FastIntegralOrder,FD,SD,Val{SG},Val{false},HigherDimension{N}})(x::Vararg{Number,N}) where
        {T<:AbstractFloat,N,TCoefs<:AbstractArray{T,N},
        Axs<:NTuple{N,<:AbstractVector},
        KA<:NTuple{N,<:Nothing},DG,EQ<:NTuple{N,Int},KBC<:NTuple{N,Tuple{Symbol,Symbol}},FD,SD,SG}    

    x = T.(x)
    # Cell and position within it, per dimension
    i_floats = ntuple(d -> (x[d] - itp.knots[d][1]) / itp.h[d] + one(T), N)
    cells = ntuple(d -> clamp(floor(Int, i_floats[d]), itp.eqs[d], size(itp.coefs, d) - itp.eqs[d]), N)

    # K̃ weights per dimension at τ = t (column c ↔ coefficient j = i + eqs + 1 − c)
    w = _column_weights_per_dim(Val(_kernel_sym(itp.kernel_sym)), _integral_orders(Val(N)),
                                ntuple(d -> i_floats[d] - T(cells[d]), N))

    return _nd_integral_eval(itp, w, cells, x)

end

@generated function _nd_integral_eval(itp::FastConvolutionInterpolation{T,N,N}, w::Tuple,
                            cells::NTuple{N,Int}, x::NTuple{N,<:Number}) where {T, N}
    quote
        coefs = itp.coefs
        result = zero(T)
        # Coefficients right of the stencil contribute exactly zero (anchored weight −½ − (−½)),
        # so each dimension stops at the stencil's right end, cells[dd] + eqs[dd]
        Base.Cartesian.@nloops $N i dd -> 1:(cells[dd] + itp.eqs[dd]) begin
            kt_prod = one(T)
            Base.Cartesian.@nexprs $N dd -> begin
                kt_prod *= _antiderivative_weight(w[dd], cells[dd], itp.eqs[dd], i_dd) - itp.left_values[dd][i_dd]
            end
            result += Base.Cartesian.@nref($N, coefs, i) * kt_prod
        end
        return result * prod(itp.h)
    end
end