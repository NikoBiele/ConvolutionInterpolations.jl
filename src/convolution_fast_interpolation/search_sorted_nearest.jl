"""
    searchsortednearest(a, x)

Find the index of the element in a sorted array `a` that is closest to value `x`.

# Arguments
- `a`: A sorted array
- `x`: The value to find the closest match for

# Returns
- The index of the element in `a` that is closest to `x`

# Details
This function finds the element in a sorted array that is closest to a given value,
returning its index. It uses a combination of `searchsortedfirst` and direct comparison
of neighboring values to efficiently find the closest match.

Special cases:
- If `x` is less than or equal to the first element, returns index 1
- If `x` is greater than the last element, returns the last index
- If `x` is exactly equal to an element in `a`, returns that element's index
- Otherwise, compares distances to the elements before and after the insertion point
  and returns the index of the closer one

This function is used in fast convolution interpolation to find the closest precomputed
kernel value for a given normalized distance.

Reference: "https://discourse.julialang.org/t/findnearest-function/4143/5"
"""
function searchsortednearest(a,x)
    idx = searchsortedfirst(a,x)
    if (idx==1); return idx; end
    if (idx>length(a)); return length(a); end
    if (a[idx]==x); return idx; end
    if (abs(a[idx]-x) < abs(a[idx-1]-x))
        return idx
    else
        return idx-1
    end
end