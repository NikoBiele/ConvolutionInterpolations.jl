"""
    (itp::ConvolutionInterpolation{T,3,TCoefs,IT,Axs,KA,Val{3},Val{DG},EQ})(x::Vararg{T,3}) where {T,TCoefs,IT,Axs,KA,DG,EQ}

Evaluate a three-dimensional convolution interpolation at the given point.

# Arguments
- `itp`: The three-dimensional interpolation object
- `x`: The coordinates at which to evaluate the interpolation (x₁, x₂, x₃)

# Returns
- The interpolated value at the specified coordinates

# Details
This method implements specialized, efficient convolution interpolation for the three-dimensional case.
It finds the nearest knot points in each dimension, then computes a weighted sum of neighboring values
using the convolution kernel. The number of neighboring points used depends on the equation order (eqs).

The interpolation uses the formula:
```
result = ∑∑∑ coefs[i+l, j+m, k+n] * kernel((x₁ - knots₁[i+l])/h₁) * 
                                      kernel((x₂ - knots₂[j+m])/h₂) * 
                                      kernel((x₃ - knots₃[k+n])/h₃)
```

for `l, m, n = -(eqs-1):eqs`, where `i`, `j`, and `k` are the indices of the nearest knot points
less than or equal to `x₁`, `x₂`, and `x₃` respectively.
"""
function (itp::ConvolutionInterpolation{T,3,TCoefs,IT,Axs,KA,Val{3},DG,EQ})(x::Vararg{Number,3}) where {T,TCoefs,IT,Axs,KA,DG,EQ}

    # First dimension (x)
    i_float = (x[1] - itp.knots[1][1]) / itp.h[1] + 1
    i = clamp(floor(Int, i_float), itp.eqs, length(itp.knots[1]) - itp.eqs)
    # Second dimension (y)
    j_float = (x[2] - itp.knots[2][1]) / itp.h[2] + 1
    j = clamp(floor(Int, j_float), itp.eqs, length(itp.knots[2]) - itp.eqs)
    # Third dimension (z)
    k_float = (x[3] - itp.knots[3][1]) / itp.h[3] + 1
    k = clamp(floor(Int, k_float), itp.eqs, length(itp.knots[3]) - itp.eqs)

    result = zero(T)
    for l = -(itp.eqs-1):itp.eqs, 
        m = -(itp.eqs-1):itp.eqs, 
        n = -(itp.eqs-1):itp.eqs
        result += itp.coefs[i+l, j+m, k+n] * 
                  itp.kernel((x[1] - itp.knots[1][i+l]) / itp.h[1]) * 
                  itp.kernel((x[2] - itp.knots[2][j+m]) / itp.h[2]) *
                  itp.kernel((x[3] - itp.knots[3][k+n]) / itp.h[3])
    end
    
    return result
end