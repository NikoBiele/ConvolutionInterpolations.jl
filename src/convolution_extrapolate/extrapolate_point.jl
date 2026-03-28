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
"""

# 1d case
@inline function extrapolate_point(etp::ConvolutionExtrapolation{T,1,ITPT,ET}, x::Number) where 
                        {T<:AbstractFloat,ITPT,ET<:AbstractExtrapolation}
    itp = etp.itp
    knots = itp.knots

    # Determine which dimensions need extrapolation
    clamped_x = zero(T)
    needs_extrapolation = false
    
    lo, hi = _domain_bounds(itp, 1)
    if x < lo
        clamped_x = lo
        needs_extrapolation = true
    elseif x > hi
        clamped_x = hi
        needs_extrapolation = true
    else
        clamped_x = x
    end
    
    if !needs_extrapolation
        return itp(x)
    end
        
    # Handle different extrapolation types
    if ET === Line
        # 2-point directional derivative: 2 evaluations total regardless of dimension
        dx = x - clamped_x
        dist_sq = dx^2

        if dist_sq < eps(T)
            return itp(clamped_x)
        end

        # Step inward along the extrapolation direction
        h_min = itp.h[1]
        ε = h_min / T(100)
        dist = sqrt(dist_sq)
        x_inward = clamped_x - ε * dx / dist

        f_boundary = itp(clamped_x)
        f_inward = itp(x_inward)
        slope = (f_boundary - f_inward) / ε

        return f_boundary + slope * dist
        
    elseif ET === Flat
        # Simply return the value at the boundary
        return itp(clamped_x)
        
    else # if etp.et == :throw
        error("Unsupported extrapolation type: $(etp.et).
        Please set 'extrap =' Line(), Flat() or Natural() to enable extrapolation.")
    end
end

# >1d case
@inline function extrapolate_point(etp::ConvolutionExtrapolation{T,N,ITPT,ET}, x::NTuple{N,Number}) where 
                        {T<:AbstractFloat,N,ITPT,ET<:AbstractExtrapolation}
    itp = etp.itp
    knots = itp.knots

    # Determine which dimensions need extrapolation
    clamped_x = zeros(T, N)
    needs_extrapolation = false
    
    for d in 1:N
        lo, hi = _domain_bounds(itp, d)
        if x[d] < lo
            clamped_x[d] = lo
            needs_extrapolation = true
        elseif x[d] > hi
            clamped_x[d] = hi
            needs_extrapolation = true
        else
            clamped_x[d] = x[d]
        end
    end
    
    if !needs_extrapolation
        return itp(x...)
    end
    
    clamped_x = ntuple(i -> clamped_x[i], N)
    
    # Handle different extrapolation types
    if ET === Line
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
        
    elseif ET === Flat
        # Simply return the value at the boundary
        return itp(clamped_x...)
        
    else # if etp.et == :throw
        error("Unsupported extrapolation type: $(etp.et).
        Please set 'extrap =' Line(), Flat() or Natural() to enable extrapolation.")
    end
end

"""
    _domain_bounds(itp, d) -> (lo, hi)

Return the valid interpolation bounds for dimension `d`, accounting for ghost points.
In lazy mode with `boundary_fallback=true`, the domain is narrowed by `eqs-1` knots
on each side to avoid boundary ghost computation. In eager mode, bounds correspond
to the first and last non-ghost knot positions.
"""

@inline function _domain_bounds(itp::AbstractConvolutionInterpolation{T,N,NI,TCoefs,
                    Axs,KA,DT,DG,EQ,KBC,DO,FD,SD,SG,Val{false},DI}, d) where {T,N,NI,TCoefs,Axs,KA,DT,DG,EQ,KBC,DO,FD,SD,SG,DI}
    eqs_d = itp.eqs isa Tuple ? itp.eqs[d] : itp.eqs
    return itp.knots[d][eqs_d], itp.knots[d][end-(eqs_d-1)]
end

@inline function _domain_bounds(itp::AbstractConvolutionInterpolation{T,N,NI,TCoefs,
                    Axs,KA,DT,DG,EQ,KBC,DO,FD,SD,SG,Val{true},DI}, d) where {T,N,NI,TCoefs,Axs,KA,DT,DG,EQ,KBC,DO,FD,SD,SG,DI}
    eqs_d = itp.eqs isa Tuple ? itp.eqs[d] : itp.eqs
    k = itp.knots[d]
    n_k = length(k)
    n_d = size(itp.coefs, d)
    ng = itp.boundary_fallback ? eqs_d - 1 : (n_k - n_d) ÷ 2
    return k[1 + ng], k[end - ng]
end