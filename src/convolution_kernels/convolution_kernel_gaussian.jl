"""
    (::GaussianConvolutionKernel{B})(s) where B

Evaluate a Gaussian-style smoothing convolution kernel at position `s`.

# Arguments
- `s`: The position at which to evaluate the kernel

# Returns
- The kernel value at position `s`

# Details
This implements a smoothing convolution kernel based on a normalized Gaussian function.
The kernel is defined by:

    K(s) = (1 / θ(B)) * exp(-B * s²)

where θ(B) is the normalization factor calculated as:

    θ(B) = 1 + 2 * ∑(exp(-B * n²)) for n = 1 to ∞

The parameter B controls the width of the Gaussian:
- Larger B values produce a narrower Gaussian with faster decay
- Smaller B values produce a wider Gaussian with slower decay

Unlike the polynomial-based convolution kernels, this kernel:
- Has infinite support (though practically truncated when values become negligible)
- Provides C∞ continuity (infinitely differentiable)
- Produces very smooth interpolation results
- Introduces controlled blurring which can be beneficial for noise reduction

This kernel is particularly useful for:
- Smoothing noisy data
- Creating visually pleasing interpolations where sharp features are not critical
- Applications where differentiability at all orders is important
- Signal processing requiring a controlled frequency response

Note: The normalization sum θ(B) is computed using a convergent series approximation,
truncated when terms become sufficiently small (below 1e-12).
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