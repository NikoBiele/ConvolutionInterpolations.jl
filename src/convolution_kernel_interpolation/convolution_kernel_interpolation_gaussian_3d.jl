"""
    (itp::ConvolutionInterpolation{T,3,0,TCoefs,Axs,KA,Val{3},DG,EQ,KBC,
            DerivativeOrder{DO},FD,SD,Val{:not_used},NB,Val{false},Val{0}})(x::Vararg{Number,3}) where 
            {KA<:Tuple{<:GaussianConvolutionKernel,<:GaussianConvolutionKernel,<:GaussianConvolutionKernel}}

Evaluate a 3D Gaussian smoothing interpolant at position `(x[1], x[2], x[3])`.

Computes the separable weighted sum:

    f(x,y,z) = Σⱼ Σₖ Σₗ cⱼₖₗ · exp(-B·((x-xⱼ)/h₁)²) · exp(-B·((y-yₖ)/h₂)²) · exp(-B·((z-zₗ)/h₃)²) / θ(B)³

where the sums run over the `2*eqs` nearest coefficients in each dimension.
The kernel is fully separable — the 3D Gaussian is the product of three 1D Gaussians.

Evaluation is allocation-free and O(eqs³) in cost.

# Example
```julia
xs = range(0.0, 2π, length=30)
ys = range(0.0, 2π, length=30)
zs = range(0.0, 2π, length=30)
vs = [sin(x)*cos(y)*sin(z) for x in xs, y in ys, z in zs] .+ 0.05.*randn(30,30,30)
itp = convolution_gaussian((xs, ys, zs), vs, 0.05)
itp(1.0, 2.0, 3.0)
```

See also: [`convolution_gaussian`](@ref), [`convolution_smooth`](@ref)
"""
@inline function (itp::ConvolutionInterpolation{T,3,0,TCoefs,Axs,KA,Val{3},
                    DG,EQ,KBC,DerivativeOrder{DO},FD,SD,Val{:not_used},NB,Val{false},Val{0}})(x::Vararg{Number,3}) where 
                    {T<:AbstractFloat,TCoefs<:AbstractArray{T,3},
                    Axs<:Tuple{<:AbstractVector,<:AbstractVector,<:AbstractVector},
                    KA<:Tuple{<:GaussianConvolutionKernel,<:GaussianConvolutionKernel,<:GaussianConvolutionKernel},DG,
                    EQ<:Tuple{Int,Int,Int},KBC<:Tuple{<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol}},
                    DO,FD,SD,NB<:Nothing}

    i_float = (x[1] - itp.knots[1][1]) / itp.h[1] + one(T)
    i = clamp(floor(Int, i_float), itp.eqs[1], length(itp.knots[1]) - itp.eqs[1])

    j_float = (x[2] - itp.knots[2][1]) / itp.h[2] + one(T)
    j = clamp(floor(Int, j_float), itp.eqs[2], length(itp.knots[2]) - itp.eqs[2])

    k_float = (x[3] - itp.knots[3][1]) / itp.h[3] + one(T)
    k = clamp(floor(Int, k_float), itp.eqs[3], length(itp.knots[3]) - itp.eqs[3])

    result = zero(T)
    @inbounds for l = -(itp.eqs[1]-1):itp.eqs[1],
                  m = -(itp.eqs[2]-1):itp.eqs[2],
                  n = -(itp.eqs[3]-1):itp.eqs[3]
        result += itp.coefs[i+l, j+m, k+n] *
                  itp.kernel[1]((x[1] - itp.knots[1][i+l]) / itp.h[1]) *
                  itp.kernel[2]((x[2] - itp.knots[2][j+m]) / itp.h[2]) *
                  itp.kernel[3]((x[3] - itp.knots[3][k+n]) / itp.h[3])
    end
    scale = prod(d -> one(T) / itp.h[d]^DO[d], 1:3)
    return @fastmath result * scale
end