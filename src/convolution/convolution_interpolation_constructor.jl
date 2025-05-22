"""
    ConvolutionInterpolation(knots::NTuple{N,AbstractVector}, vs::AbstractArray{T,N};
                             degree::Symbol=:a3, B=nothing, kernel_bc=:detect) where {T,N}

Construct a convolution interpolation object for N-dimensional data.

# Arguments
- `knots::NTuple{N,AbstractVector}`: A tuple of vectors containing the knot points in each dimension
- `vs::AbstractArray{T,N}`: The values to be interpolated, at the knot positions

# Keyword Arguments
- `degree::Symbol=:a3`: The degree of the polynomial kernel (':a3' for cubic, ':a5' for quintic, etc.)
- `B=nothing`: Parameter for Gaussian kernel (if not `nothing`, uses a Gaussian kernel instead of polynomial)
- `kernel_bc=:detect`: The boundary condition (or boundary conditions) to use for kernel evaluation

# Returns
- A `ConvolutionInterpolation` object that can be used for interpolation

# Details
This constructor creates a convolution interpolation object that can evaluate the function
at arbitrary points within the domain defined by the knot points. The boundaries are
handled by expanding the knots and creating boundary conditions based on signal characteristics.

If `B` is provided, a Gaussian kernel with parameter `B` is used instead of a polynomial
kernel, and the `eqs` parameter is set to 50 to accommodate the wider support of the Gaussian.
"""
function ConvolutionInterpolation(knots::NTuple{N,AbstractVector}, vs::AbstractArray{T,N};
    degree::Symbol=:a3, B=nothing, kernel_bc=:detect) where {T,N}

    eqs = B === nothing ? get_equations_for_degree(degree) : 50
    h = map(k -> k[2] - k[1], knots)
    it = ntuple(_ -> ConvolutionMethod(), N)

    knots_new = expand_knots(knots, eqs-1) # expand boundaries
    coefs = create_convolutional_coefs(vs, h, eqs, kernel_bc) # create boundaries
    kernel = B === nothing ? ConvolutionKernel(Val(degree),Val(eqs)) : GaussianConvolutionKernel(Val(B))
    dimension = N <= 3 ? Val(N) : HigherDimension(Val(N))
    degree = Val(degree)
    
    ConvolutionInterpolation{T,N,typeof(coefs),typeof(it),typeof(knots_new),typeof(kernel),typeof(dimension),typeof(degree),typeof(eqs),typeof(kernel_bc)}(
        coefs, knots_new, it, h, kernel, dimension, degree, eqs, kernel_bc
    )
end