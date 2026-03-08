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

function extrapolate_point(etp::ConvolutionExtrapolation{T,N,ITPT,ET}, x::NTuple{N,Number}) where {T,N,ITPT,ET}
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
        lo, hi = _domain_bounds(itp, d)
        if x[d] < lo
            clamped_dir[d] = -1
            clamped_x[d] = lo
            needs_extrapolation = true
        elseif x[d] > hi
            clamped_dir[d] = 1
            clamped_x[d] = hi
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
        # 2-point directional derivative: 2 evaluations total regardless of dimension
        dx = ntuple(d -> x[d] - clamped_x[d], N)
        dist_sq = sum(d -> dx[d]^2, 1:N)

        if dist_sq < eps(T)
            return itp(clamped_x...)
        end

        # Step inward along the extrapolation direction
        h_min = minimum(d -> itp.h[d], 1:N)
        ε = h_min / T(100)
        dist = sqrt(dist_sq)
        x_inward = ntuple(d -> clamped_x[d] - ε * dx[d] / dist, N)

        f_boundary = itp(clamped_x...)
        f_inward = itp(x_inward...)
        slope = (f_boundary - f_inward) / ε

        return f_boundary + slope * dist
        
    elseif etp.et isa Flat
        # Simply return the value at the boundary
        return itp(clamped_x...)
        
    elseif etp.et isa Periodic
        x_periodic = ntuple(d -> begin
            lo, hi = _domain_bounds(itp, d)
            if !isapprox(x[d], clamped_x[d], atol=1e-6)
                period = hi - lo
                lo + mod(x[d] - lo, period)
            else
                x[d]
            end
        end, N)
        return itp(x_periodic...)
        
    elseif etp.et isa Reflect
        x_reflect = ntuple(d -> begin
            lo, hi = _domain_bounds(itp, d)
            if !isapprox(x[d], clamped_x[d], atol=1e-6)
                reflect(x[d], lo, hi)
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

"""
    _domain_bounds(itp, d) -> (lo, hi)

Return the valid interpolation bounds for dimension `d`, accounting for ghost points.
In lazy mode with `boundary_fallback=true`, the domain is narrowed by `eqs-1` knots
on each side to avoid boundary ghost computation. In eager mode, bounds correspond
to the first and last non-ghost knot positions.
"""

@inline function _domain_bounds(itp, d)
    if itp.lazy && itp.boundary_fallback
        k = itp.knots[d]
        n_k = length(k)
        n_d = size(itp.coefs, d)
        ng = itp.eqs - 1
        return k[1 + ng], k[end - ng]
    elseif itp.lazy
        k = itp.knots[d]
        n_k = length(k)
        n_d = size(itp.coefs, d)
        # If knots are expanded (nonuniform lazy), skip ghost knots
        # Uniform lazy: n_k == n_d (knots not expanded)
        # Nonuniform lazy n3: n_k == n_d + 2 (1 ghost knot per side)
        ng = (n_k - n_d) ÷ 2
        return k[1 + ng], k[end - ng]
    else
        return itp.knots[d][itp.eqs], itp.knots[d][end-(itp.eqs-1)]
    end
end