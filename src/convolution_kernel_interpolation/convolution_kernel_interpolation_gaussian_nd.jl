"""
    (itp::ConvolutionInterpolation{T,N,0,TCoefs,Axs,KA,HigherDimension{N},DG,EQ,KBC,
            DerivativeOrder{DO},FD,SD,Val{:not_used},NB,Val{false},Val{0}})(x::Vararg{Number,N}) where 
            {KA<:NTuple{N,GaussianConvolutionKernel}}

Evaluate an N-dimensional Gaussian smoothing interpolant at position `(x[1], ..., x[N])`.

Computes the fully separable weighted sum over all N dimensions:

    f(x₁,...,xₙ) = Σⱼ₁...Σⱼₙ cⱼ₁...ⱼₙ · ∏ᵈ exp(-B·((xᵈ-xᵈⱼ)/hᵈ)²) / θ(B)ᴺ

where the sums run over the `2*eqs` nearest coefficients in each dimension.
The N-dimensional Gaussian is the product of N independent 1D Gaussians.

Used for N≥4. Evaluation is allocation-free and O(eqs^N) in cost.

# Example
```julia
xs = ntuple(_ -> range(0.0, 2π, length=20), 4)
vs = [sin(x)*cos(y)*sin(z)*cos(t) for x in xs[1], y in xs[2], z in xs[3], t in xs[4]] .+ 0.05.*randn(20,20,20,20)
itp = convolution_gaussian(xs, vs, 0.05)
itp(1.0, 2.0, 3.0, 4.0)
```

See also: [`convolution_gaussian`](@ref), [`convolution_smooth`](@ref)
"""
@inline function (itp::ConvolutionInterpolation{T,N,0,TCoefs,Axs,KA,HigherDimension{N},
                    DG,EQ,KBC,DerivativeOrder{DO},FD,SD,Val{:not_used},NB,Val{false},Val{0}})(x::Vararg{Number,N}) where 
                    {T<:AbstractFloat,N,TCoefs<:AbstractArray{T,N},KA<:NTuple{N,GaussianConvolutionKernel},
                    Axs<:NTuple{N,<:AbstractVector},DG,EQ<:NTuple{N,Int},
                    KBC<:NTuple{N,Tuple{Symbol,Symbol}},DO,FD,SD,NB<:Nothing}

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