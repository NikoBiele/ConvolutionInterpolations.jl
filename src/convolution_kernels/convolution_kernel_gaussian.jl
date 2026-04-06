"""
    (::GaussianConvolutionKernel{B})(s) where B

Evaluate a Gaussian smoothing kernel at position `s`.

Unlike the polynomial convolution kernels which interpolate (pass exactly through data points),
this kernel smooths: it produces a weighted average of nearby values, controlled by the
parameter `B`. Larger `B` gives a narrower Gaussian (less smoothing), smaller `B` gives
a wider one (more smoothing).

The kernel is defined as `K(s) = exp(-B s²) / θ(B)`, where `θ(B)` is a normalization
factor ensuring the discrete kernel sums to unity.

This kernel has infinite support (truncated in practice) and C∞ continuity, but it does
not reproduce polynomials — it blurs the signal by design. Useful for noisy data where
exact interpolation is undesirable.
"""

@inline function (::GaussianConvolutionKernel{B,NT})(s) where {B,NT} # Gaussian style smoothing kernel
    return f(s, B, NT)
end

@inline function θ(B::G, terms::Int) where G
    q = exp(-B)
    sum = 1.0
    for n in 1:terms
        term = 2 * q^(n^2)
        sum += term
    end
    return sum
end

@inline function f(x::T, B::G, terms::Int) where {G,T}
    return 1 / θ(B, terms) * exp(-B * x^2) # 
end