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

function (::GaussianConvolutionKernel{B})(s) where B # Gaussian style smoothing kernel
    function θ(B::G, terms::Int=1000) where G
        q = exp(-B)
        sum = 1.0
        for n in 1:terms
            term = 2 * q^(n^2)
            sum += term
            if term < 1e-12  # Break if the term is very small
                break
            end
        end
        return sum
    end
    function f(x::T, B::G) where {G,T}
        return 1 / θ(B) * exp(-B * x^2) # 
    end
    return f(s, B)
end