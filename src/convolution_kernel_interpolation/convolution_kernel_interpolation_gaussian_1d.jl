"""
    (itp::ConvolutionInterpolation{T,1,0,TCoefs,Axs,KA,Val{1},DG,EQ,KBC,
            DerivativeOrder{DO},FD,SD,Val{:not_used},NB,Val{false},Val{0}})(x::Vararg{Number,1}) where 
            {KA<:Tuple{<:GaussianConvolutionKernel}}

Evaluate a 1D Gaussian smoothing interpolant at position `x`.

Computes the weighted sum:

    f(x) = Σⱼ cⱼ · exp(-B·((x-xⱼ)/h)²) / θ(B)

where the sum runs over the `2*eqs` nearest coefficients and `θ(B)` normalizes
the weights to unity. The stencil half-width `eqs = ceil(sqrt(-log(5e-13)/B))`
is determined by `B` at construction time.

Evaluation is allocation-free and O(eqs) in cost.

# Example
```julia
x = range(0.0, 2π, length=100)
itp = convolution_gaussian(x, sin.(x) .+ 0.1.*randn(100), 0.05)
itp(1.5)
```

See also: [`convolution_gaussian`](@ref), [`convolution_smooth`](@ref)
"""
@inline function (itp::ConvolutionInterpolation{T,1,0,TCoefs,Axs,KA,Val{1},
                    DG,EQ,KBC,DerivativeOrder{DO},FD,SD,Val{:not_used},NB,Val{false},Val{0}})(x::Vararg{Number,1}) where 
                    {T<:AbstractFloat,TCoefs<:AbstractArray{T,1},Axs<:Tuple{<:AbstractVector},
                    KA<:Tuple{<:GaussianConvolutionKernel},DG,EQ<:Tuple{Int},
                    KBC<:Tuple{<:Tuple{Symbol,Symbol}},DO,FD,SD,NB<:Nothing}

    i_float = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T)
    i = clamp(floor(Int, i_float), itp.eqs[1], length(itp.knots[1]) - itp.eqs[1])

    result = zero(T)
    @inbounds for l = -(itp.eqs[1]-1):itp.eqs[1]
        result += itp.coefs[i+l] * itp.kernel[1]((x[1] - itp.knots[1][i+l]) / itp.h[1])
    end
    scale = one(T) / itp.h[1]^DO[1]
    return @fastmath result * scale
end