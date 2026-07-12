"""
    scattered_to_grid(points, values, knots; k=1) -> Array

Grid scattered data onto a regular grid by nearest-neighbor assignment.

- `points`: d×n matrix of scattered point coordinates
- `values`: length-n vector of data values
- `knots`: tuple of d ranges defining the target grid
- `k`: number of nearest neighbors to average (default 1)

Each grid node takes the value of its nearest scattered point (`k=1`),
or the unweighted mean of its `k` nearest points. Small `k` (3–5) can
mildly reduce noise; large `k` blurs the signal and introduces boundary
bias. Intended as a gridding step before `convolution_smooth` or
`convolution_interpolation` for noisy scattered data.

# Example
```julia
grid = scattered_to_grid(points, values, (xs, ys); k=1)
smooth = convolution_smooth((xs, ys), grid, 0.1)
itp = convolution_interpolation((xs, ys), smooth)
```
"""
function scattered_to_grid(points::AbstractMatrix, values::AbstractVector{T},
                           knots::Tuple; k::Int=1) where T
    size(points, 2) == length(values) ||
        throw(ArgumentError("number of points ($(size(points,2))) must match number of values ($(length(values)))"))
    size(points, 1) == length(knots) ||
        throw(ArgumentError("point dimension ($(size(points,1))) must match number of knot vectors ($(length(knots)))"))
    1 <= k <= length(values) ||
        throw(ArgumentError("k=$k must be between 1 and the number of points ($(length(values)))"))

    tree = KDTree(points)
    grid = Array{float(T)}(undef, length.(knots)...)
    q = Vector{Float64}(undef, length(knots))
    for I in CartesianIndices(grid)
        for d in eachindex(q)
            q[d] = knots[d][I[d]]
        end
        idxs, _ = knn(tree, q, k)
        s = zero(float(T))
        for i in idxs
            s += values[i]
        end
        grid[I] = s / k
    end
    return grid
end