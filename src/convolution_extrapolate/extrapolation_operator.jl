"""
    (etp::ConvolutionExtrapolation)(x::Vararg{Number,N})

Evaluate the extrapolation object at the given coordinates.

For points inside the interpolation domain, delegates directly to the underlying
interpolation object. For points outside, applies the extrapolation boundary condition
(`Throw()`, `Flat()`, `Line()`, `Periodic()`).

Domain bounds are determined by `_domain_bounds`, which accounts for lazy mode and
boundary fallback settings — in lazy mode with `boundary_fallback=true`, the valid
domain is narrowed to avoid expensive ghost point computation near boundaries.

Supports broadcasting over arrays of coordinates for vectorized evaluation,
adapted from [Interpolations.jl](https://github.com/JuliaMath/Interpolations.jl).

# Examples
```julia
knots = range(0, 2π, length=50)
etp = convolution_interpolation(knots, sin.(knots), extrapolation_bc=Line())

etp(1.0)                         # interior: direct interpolation
etp(7.0)                         # exterior: linear extrapolation
etp.([0.3, 0.5, 0.7, 7.0])      # broadcasting over multiple points
```

See also: [`convolution_interpolation`](@ref), [`extrapolate_point`](@ref), [`_domain_bounds`](@ref).
"""

# For broadcasting (Adapted from Interpolations.jl)
@inline function (etp::ConvolutionExtrapolation{T,N,AbstractConvolutionInterpolation,ET,DG,DO})(x::Vararg{Union{Number,AbstractVector,LinearAlgebra.Adjoint},N}) where {T,N,AbstractConvolutionInterpolation,ET,DG,DO}
    itp = etp.itp
    knots = itp.knots
    sh = shape(x...)
    
    # Handle scalar case separately
    if sh == ()
        y = x  # x is already a tuple of scalars
        within_bounds = true
        for d in 1:N
            lo, hi = _domain_bounds(itp, d)
            if y[d] > hi || y[d] < lo
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
            lo, hi = _domain_bounds(itp, d)
            if y[d] > hi || y[d] < lo
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