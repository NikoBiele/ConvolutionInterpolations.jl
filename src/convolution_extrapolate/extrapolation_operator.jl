"""
    (etp::ConvolutionExtrapolation{T,N,ITPT,ET,O})(x::Vararg{Number,N}) where {T,N,ITPT,ET,O}

Evaluate the interpolation/extrapolation at the given point.

# Arguments
- `etp::ConvolutionExtrapolation`: The extrapolation object
- `x::Vararg{Number,N}`: The coordinates at which to evaluate the function

# Returns
- The interpolated or extrapolated value at the given point

# Details
This function is the main evaluation method for extrapolation objects. It first determines
whether the given point lies within or outside the valid interpolation domain by checking
each coordinate against the domain boundaries.

The valid interpolation domain is defined as the region where:
```
knots[d][eqs] ≤ x[d] ≤ knots[d][end-(eqs-1)] for all dimensions d
```

If the point is within bounds, the function delegates to the underlying interpolation
object for efficient evaluation. If the point is outside bounds (in any dimension),
it calls `extrapolate_point` to apply the appropriate boundary condition.

This approach ensures optimal performance for points inside the domain, while still
providing accurate extrapolation for points outside it, based on the specified
boundary condition (Line, Flat, Periodic, Reflect, or Throw).

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
```
"""
function (etp::ConvolutionExtrapolation{T,N,ITPT,ET,O})(x::Vararg{Number,N}) where {T,N,ITPT,ET,O}
    itp = etp.itp
    knots = getknots(itp)
    
    # Check if the point is within bounds
    within_bounds = true
    for d in 1:N
        if x[d] > knots[d][end-(itp.eqs-1)] || x[d] < knots[d][itp.eqs]
            within_bounds = false
            break
        end
    end

    if within_bounds
        return itp(x...)
    else
        return extrapolate_point(etp, x)
    end
end