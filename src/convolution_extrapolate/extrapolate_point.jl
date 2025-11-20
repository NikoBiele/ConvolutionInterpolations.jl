"""
    extrapolate_point(etp::ConvolutionExtrapolation{T,N,ITPT,ET,O}, x::NTuple{N,Number}) where {T,N,ITPT,ET,O}

Apply the appropriate extrapolation technique for a point outside the interpolation domain.

# Arguments
- `etp::ConvolutionExtrapolation`: The extrapolation object
- `x::NTuple{N,Number}`: The coordinates outside the interpolation domain

# Returns
- The extrapolated value at the given point

# Details
This function handles extrapolation for points outside the domain using different strategies
based on the extrapolation type. It first determines which dimensions need extrapolation by
comparing each coordinate to the valid interpolation domain boundaries.

For each dimension requiring extrapolation, the function records:
- The clamped coordinate value (at the boundary)
- The direction of extrapolation (-1 for left boundary, 1 for right boundary)

The function then applies one of the following extrapolation techniques:

## Line Extrapolation
For points outside the domain, computes high-order accurate gradients at the nearest
boundary point and performs linear extrapolation using the formula:
```
value = f(boundary_point) + ∑ gradient[d] * (x[d] - boundary_point[d])
```
This provides a first-order continuous extension beyond the boundaries.

## Flat Extrapolation
Returns the value at the nearest boundary point without modification:
```
value = f(boundary_point)
```
This creates a constant extension beyond the boundaries.

## Periodic Extrapolation
Wraps the coordinates using the domain's period:
```
x_periodic[d] = lower_bound[d] + mod(x[d] - lower_bound[d], period[d])
```
This is useful for inherently periodic data like angles or cyclic signals.

## Reflect Extrapolation
Reflects the coordinates across the boundary:
```
x_reflect[d] = 2*boundary - x[d]
```
where boundary is either the lower or upper bound depending on which was crossed.
This preserves symmetry and avoids discontinuities at boundaries.

The function handles each dimension independently.
"""

function extrapolate_point(etp::ConvolutionExtrapolation{T,N,ITPT,ET,O}, x::NTuple{N,Number}) where {T,N,ITPT,ET,O}
    itp = etp.itp
    knots = itp.knots
    
    function reflect(y, l, u)
        yr = mod(y - l, 2(u-l)) + l
        return ifelse(yr > u, 2u-yr, yr)
    end

    # Determine which dimensions need extrapolation
    clamped_x = zeros(T, N)
    clamped_dir = zeros(T, N)
    needs_extrapolation = false
    
    for d in 1:N
        if x[d] < knots[d][itp.eqs]
            clamped_dir[d] = -1
            clamped_x[d] = knots[d][itp.eqs]
            needs_extrapolation = true
        elseif x[d] > knots[d][end-(itp.eqs-1)]
            clamped_dir[d] = 1
            clamped_x[d] = knots[d][end-(itp.eqs-1)]
            needs_extrapolation = true
        else
            clamped_dir[d] = 0
            clamped_x[d] = x[d]
        end
    end
    
    if !needs_extrapolation
        return itp(x...)
    end
    
    clamped_x = ntuple(i -> clamped_x[i], N)
    
    # Handle different extrapolation types
    if etp.et isa Line
        # For Line extrapolation, compute the gradient and apply it
        grad = calculate_gradient(etp, clamped_x, clamped_dir)
        val = itp(clamped_x...)
        
        # Apply linear extrapolation for all dimensions at once
        for d in 1:N
            if !isapprox(x[d], clamped_x[d], atol=1e-6)
                val += grad[d] * (x[d] - clamped_x[d])
            end
        end
        return val
        
    elseif etp.et isa Flat
        # Simply return the value at the boundary
        return itp(clamped_x...)
        
    elseif etp.et isa Periodic
        # Handle periodic boundaries
        x_periodic = ntuple(d -> begin
            if !isapprox(x[d], clamped_x[d], atol=1e-6)
                period = knots[d][end-(itp.eqs-1)] - knots[d][itp.eqs]
                knots[d][itp.eqs] + mod(x[d] - knots[d][itp.eqs], period)
            else
                x[d]
            end
        end, N)
        return itp(x_periodic...)
        
    elseif etp.et isa Reflect
        # Handle reflection at boundaries
        x_reflect = ntuple(d -> begin
            if !isapprox(x[d], clamped_x[d], atol=1e-6)
                reflect(x[d], knots[d][itp.eqs], knots[d][end-(itp.eqs-1)])
            else
                x[d]
            end
        end, N)
        return itp(x_reflect...)
        
    else # if etp.et isa Throw
        error("Unsupported extrapolation type: $(etp.et).
        Please set 'extrapolation_bc =' Line(), Flat(), Natural(), Periodic(), or Reflect() to enable extrapolation.")
    end
end