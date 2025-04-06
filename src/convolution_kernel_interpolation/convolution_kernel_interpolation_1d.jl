"""
    (itp::ConvolutionInterpolation{T,1,TCoefs,IT,Axs,KA,Val{1},Val{DG},EQ})(x::Vararg{T,1}) where {T,TCoefs,IT,Axs,KA,DG,EQ}

Evaluate a one-dimensional convolution interpolation at the given point.

# Arguments
- `itp`: The one-dimensional interpolation object
- `x`: The coordinate at which to evaluate the interpolation

# Returns
- The interpolated value at the specified coordinate

# Details
This method implements specialized, efficient convolution interpolation for the one-dimensional case.
It finds the nearest knot point, then computes a weighted sum of neighboring values using the
convolution kernel. The number of neighboring points used depends on the equation order (eqs).

The interpolation uses the formula:
```
result = ∑ coefs[i+l] * kernel((x - knots[i+l])/h) for l = -(eqs-1):eqs
```

where `i` is the index of the nearest knot point less than or equal to `x`.
"""
function (itp::ConvolutionInterpolation{T,1,TCoefs,IT,Axs,KA,Val{1},Val{DG},EQ})(x::Vararg{T,1}) where {T,TCoefs,IT,Axs,KA,DG,EQ}

    i = limit_convolution_bounds(1, searchsortedlast(itp.knots[1], x[1]), itp)

    result = zero(T)
    for l = -(itp.eqs-1):itp.eqs
        result += itp.coefs[i+l] * 
                  itp.kernel( (x[1] - itp.knots[1][i+l]) / itp.h[1] )
    end
    
    return result
end