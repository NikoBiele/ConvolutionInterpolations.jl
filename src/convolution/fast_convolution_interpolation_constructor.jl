"""
    FastConvolutionInterpolation(knots::NTuple{N,AbstractVector}, vs::AbstractArray{T,N};
                                degree::Int=3, eqs::Int=2, precompute::Int=1000, B=nothing) where {T,N}

Construct a fast convolution interpolation object with precomputed kernel values.

# Arguments
- `knots::NTuple{N,AbstractVector}`: A tuple of vectors containing the knot points in each dimension
- `vs::AbstractArray{T,N}`: The values to be interpolated, at the knot positions

# Keyword Arguments
- `degree::Int=3`: The degree of the polynomial kernel (3=cubic, 5=quintic, etc.)
- `eqs::Int=2`: The number of equations to use for boundary handling
- `precompute::Int=1000`: The number of points to precompute for kernel evaluation
- `B=nothing`: Parameter for Gaussian kernel (if not `nothing`, uses a Gaussian kernel instead of polynomial)

# Returns
- A `FastConvolutionInterpolation` object that can be used for interpolation

# Details
This constructor creates a fast convolution interpolation object that precomputes kernel
values at regularly spaced positions to accelerate interpolation. The boundaries are
handled by expanding the knots and creating boundary conditions based on signal characteristics.

If `B` is provided, a Gaussian kernel with parameter `B` is used instead of a polynomial
kernel, and the `eqs` parameter is set to 50 to accommodate the wider support of the Gaussian.

The `precompute` parameter controls how many positions are precomputed for the kernel.
Higher values may provide more accuracy but use more memory.
"""
function FastConvolutionInterpolation(knots::NTuple{N,AbstractVector}, vs::AbstractArray{T,N};
                                degree::Symbol=:a3, precompute::Int=1000, B=nothing) where {T,N}
    
    eqs = B === nothing ? get_equations_for_degree(degree) : 50
    h = map(k -> k[2] - k[1], knots)
    it = ntuple(_ -> ConvolutionMethod(), N)

    knots_new = expand_knots(knots, eqs-1) # expand boundaries
    coefs = create_convolutional_coefs(vs, h, eqs) # create boundaries
    kernel = B === nothing ? ConvolutionKernel(Val(degree)) : GaussianConvolutionKernel(Val(B))
    dimension = N <= 3 ? Val(N) : HigherDimension(Val(N))
    degree = Val(degree)
    pre_range = range(0.0, 1.0, length=precompute)
    kernel_pre = zeros(T, precompute, 2*eqs)
    for i = 1:2*eqs
        kernel_pre[:,i] .= kernel.(pre_range .- eqs .+ i .- 1)
    end

    FastConvolutionInterpolation{T,N,typeof(coefs),typeof(it),typeof(knots_new),typeof(kernel),typeof(dimension),typeof(degree),typeof(eqs),typeof(pre_range),typeof(kernel_pre)}(
        coefs, knots_new, it, h, kernel, dimension, degree, eqs, pre_range, kernel_pre
    )
end
