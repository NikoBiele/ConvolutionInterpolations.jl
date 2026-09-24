"""
(itp::FastConvolutionInterpolation{T,N,...})(x...) — MixedIntegralOrder
Evaluate mixed interpolation/derivative/antiderivative operator in N dimensions.

For each dimension d, applies either:
  K̃ weight (DO[d] == -1): antiderivative contribution, scaled by h[d]
  K  weight (DO[d] == 0):  interpolation contribution
  K  weight (DO[d] >= 1):  derivative contribution, scaled by (-1/h[d])^DO[d]

All weights come from exact column polynomials, computed once per evaluation.
Result is the tensor product over all dimensions, summed over all coefficient indices.
Evaluation is O(N^D) — no prefix sum optimization for mixed orders.
See also: FastConvolutionInterpolation, convolution_fast_integration_1d, convolution_fast_interpolation_perdim.
"""

function (itp::FastConvolutionInterpolation{T,N,NI,TCoefs,Axs,KA,HigherDimension{N},DG,EQ,KBC,
            FastMixedIntegralOrder{DO},FD,SD,Val{SG},Val{false},HigherDimension{NI}})(x::Vararg{Number,N}) where
            {T<:AbstractFloat,N,NI,TCoefs<:AbstractArray{T,N},Axs<:NTuple{N,<:AbstractVector},
            KA<:NTuple{N,<:Nothing},DG,EQ<:NTuple{N,Int},KBC<:NTuple{N,Tuple{Symbol,Symbol}},DO,FD,SD,SG}

    x = T.(x)
    # cells per dimension
    i = ntuple(d -> clamp(floor(Int, (x[d] - itp.knots[d][1]) / itp.h[d] + one(T)),
                          itp.eqs[d], size(itp.coefs, d) - itp.eqs[d]), N)
    # kernel weights of every dimension (exact column polynomials)
    w = _mixed_weights(itp, x, i, Val(DO))

    result = zero(T)
    @inbounds for idx in Iterators.product(ntuple(d -> 1:size(itp.coefs, d), N)...)
        # anchored K̃ weights in integral dimensions, kernel weights (zero outside the stencil) elsewhere
        kt_prod = _mixed_weight_product(w, idx, i, itp.eqs, itp.left_values, Val(DO))
        result += itp.coefs[idx...] * kt_prod
    end

    # scale: h[d] for integral dims, (-1/h[d])^DO[d] for derivative dims
    scale = one(T)
    @inbounds for d in 1:N
        if DO[d] == -1
            scale *= itp.h[d]
        else
            scale *= (-one(T) / itp.h[d])^DO[d]
        end
    end

    return result * scale
end