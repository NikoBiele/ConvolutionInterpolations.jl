"""
    (etp::ConvolutionExtrapolation{T,N,ITPT,ET,O})(x::Vararg{Number,N}) where {T,N,ITPT,ET,O}

Evaluate the interpolation/extrapolation at the given point.

# Arguments
- `etp::ConvolutionExtrapolation`: The extrapolation object
- `x::Vararg{Number,N}`: The coordinates at which to evaluate the function

# Returns
The interpolated or extrapolated value at the given point

# Details
This function is the main evaluation method for extrapolation objects. It first determines
whether the given point lies within or outside the valid interpolation domain by checking
each coordinate against the domain boundaries.

The valid interpolation domain is defined as the region where:
```julia
knots[d][eqs] ≤ x[d] ≤ knots[d][end-(eqs-1)] for all dimensions d
```

If the point is within bounds, the function delegates to the underlying interpolation
object for efficient evaluation. If the point is outside bounds (in any dimension),
it calls `extrapolate_point` to apply the appropriate boundary condition.

This approach ensures optimal performance for points inside the domain, while still
providing accurate extrapolation for points outside it, based on the specified
boundary condition (`Line`, `Flat`, `Periodic`, `Reflect`, or `Throw`).

# Broadcasting Support
The extrapolation object supports efficient broadcasting over arrays of coordinates,
adapted from [Interpolations.jl](https://github.com/JuliaMath/Interpolations.jl).
This enables vectorized evaluation for improved performance.

# Examples
```julia
# Create an interpolation with extrapolation
knots = range(0, 1, length=10)
values = sin.(2π .* knots)
etp = convolution_interpolation(knots, values, extrapolation_bc=Line())

# Evaluate at a point inside the domain
etp(0.5)  # Uses direct interpolation

# Evaluate at a point outside the domain
etp(1.2)  # Uses extrapolation based on the Line boundary condition

# Broadcasting (vectorized evaluation)
etp.([0.3, 0.5, 0.7, 1.2])  # Evaluates at multiple points efficiently
```

See also: [`extrapolate_point`](@ref), [`convolution_interpolation`](@ref)
"""

# For broadcasting (Adapted from Interpolations.jl)
@inline function (etp::ConvolutionExtrapolation{T,N,AbstractConvolutionInterpolation,ET,DG})(x::Vararg{Union{Number,AbstractVector,LinearAlgebra.Adjoint},N}) where {T,N,AbstractConvolutionInterpolation,ET,DG}
    itp = etp.itp
    knots = itp.knots
    sh = shape(x...)
    
    # Handle scalar case separately
    if sh == ()
        y = x  # x is already a tuple of scalars
        within_bounds = true
        for d in 1:N
            if y[d] > knots[d][end-(itp.eqs-1)] || y[d] < knots[d][itp.eqs]
                within_bounds = false
                break
            end
        end
        
        if within_bounds
            return itp(y...)
        else
            return extrapolate_point(etp, y)
        end
    end
    
    # Array case
    ret = zeros(T, sh)
    for (i, y) in zip(eachindex(ret), Iterators.product(x...))    
        within_bounds = true
        for d in 1:N
            if y[d] > knots[d][end-(itp.eqs-1)] || y[d] < knots[d][itp.eqs]
                within_bounds = false
                break
            end
        end

        if within_bounds
            ret[i] = itp(y...)
        else
            ret[i] = extrapolate_point(etp, y)
        end
    end
    return ret
end

"""
    shape(args...)

Compute the output shape for broadcasting over interpolation coordinates.

# Details
Utility functions for handling broadcasting with different input types (scalars, vectors,
adjoints, ranges). These enable efficient vectorized evaluation of interpolation objects.

Adapted from [Interpolations.jl](https://github.com/JuliaMath/Interpolations.jl).

# Implementation
- `shape(::Number, rest...)`: Scalars contribute nothing to output shape
- `shape(::AbstractVector, rest...)`: Vectors contribute their axis to output shape
- `shape(::Adjoint, rest...)`: Transposed vectors contribute their axis
- `shape(::StepRangeLen, rest...)`: Ranges contribute their axis
- `shape()`: Base case returns empty tuple

See also: `ConvolutionExtrapolation`.
"""

# Vector indexing utilities for broadcasting (Adapted from Interpolations.jl)
shape(i::Number, rest...)         = shape(rest...)
shape(v::AbstractVector, rest...) = (axes(v, 1), shape(rest...)...)
shape(v::LinearAlgebra.Adjoint, rest...) = (axes(v, 1), shape(rest...)...)
shape(v::StepRangeLen, rest...) = (axes(v, 1), shape(rest...)...)
shape() = ()