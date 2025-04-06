"""
    (itp::ConvolutionInterpolation{T,2,TCoefs,IT,Axs,KA,Val{2},Val{DG},EQ})(x::Vararg{T,2}) where {T,TCoefs,IT,Axs,KA,DG,EQ}

Evaluate a two-dimensional convolution interpolation at the given point.

# Arguments
- `itp`: The two-dimensional interpolation object
- `x`: The coordinates at which to evaluate the interpolation (x₁, x₂)

# Returns
- The interpolated value at the specified coordinates

# Details
This method implements specialized, efficient convolution interpolation for the two-dimensional case.
It finds the nearest knot points in each dimension, then computes a weighted sum of neighboring values
using the convolution kernel. The number of neighboring points used depends on the equation order (eqs).

The interpolation uses the formula:
```
result = ∑∑ coefs[i+l, j+m] * kernel((x₁ - knots₁[i+l])/h₁) * kernel((x₂ - knots₂[j+m])/h₂)
```

for `l, m = -(eqs-1):eqs`, where `i` and `j` are the indices of the nearest knot points less than
or equal to `x₁` and `x₂` respectively.
"""
function (itp::ConvolutionInterpolation{T,2,TCoefs,IT,Axs,KA,Val{2},Val{DG},EQ})(x::Vararg{T,2}) where {T,TCoefs,IT,Axs,KA,DG,EQ}

    i = limit_convolution_bounds(1, searchsortedlast(itp.knots[1], x[1]), itp)
    j = limit_convolution_bounds(2, searchsortedlast(itp.knots[2], x[2]), itp)

    result = zero(T)
    for l = -(itp.eqs-1):itp.eqs, 
        m = -(itp.eqs-1):itp.eqs
        result += itp.coefs[i+l, j+m] * 
                  itp.kernel((x[1] - itp.knots[1][i+l]) / itp.h[1]) * 
                  itp.kernel((x[2] - itp.knots[2][j+m]) / itp.h[2])
    end
    
    return result
end