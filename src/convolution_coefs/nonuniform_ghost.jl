"""
    nonuniform_ghost_coefficients(knots::AbstractVector{T}, num_ghost::Int,
                                  poly_degree::Int, side::Symbol) where T

Compute ghost point coefficients for nonuniform grid boundaries using closed-form
expressions derived from symbolic Vandermonde solutions.

# Arguments
- `knots`: Full knot vector for this dimension
- `num_ghost`: Number of ghost points needed on this side (currently 1)
- `poly_degree`: Polynomial reproduction degree (2 for cubic nonuniform weights)
- `side`: `:left` or `:right`

# Returns
`Matrix{T}` of size `(num_ghost, poly_degree+1)` where row `j` contains coefficients
for computing ghost point `g_{-j}` from interior values. Same format as
`POLYNOMIAL_GHOST_COEFFS` entries but with Float64 coefficients.

# Ghost point placement
Ghost positions mirror the nearest interval spacing:
- Left:  `x_{-j} = x_1 - j * (x_2 - x_1)`
- Right: `x_{n+j} = x_n + j * (x_n - x_{n-1})`

# Closed-form expressions (degree 2)
For the left boundary with spacings h₀ = x₂ - x₁ and h₁ = x₃ - x₂:

    α₀ = 2(2h₀ + h₁) / (h₀ + h₁)
    α₁ = -(2h₀ + h₁) / h₁
    α₂ = 2h₀² / (h₁(h₀ + h₁))

For the right boundary with spacings hₘ = x_{n-1} - x_{n-2} and hₗ = x_n - x_{n-1}:

    α₀ = 2hₗ² / (hₘ(hₗ + hₘ))
    α₁ = -(2hₗ + hₘ) / hₘ
    α₂ = 2(2hₗ + hₘ) / (hₗ + hₘ)

These recover the integer coefficients [3, -3, 1] on uniform grids.

Derived symbolically via SymPy from the Vandermonde system enforcing polynomial
reproduction of degree ≤ 2 at the ghost point position.

# Properties
- Reproduces polynomials up to degree `poly_degree` exactly at ghost positions
- Recovers integer coefficients `[3, -3, 1]` on uniform grids
- No linear system solve at runtime — pure arithmetic from closed-form expressions
- Compatible with `fill_ghost_points_polynomial!` (same matrix format)

See also: `POLYNOMIAL_GHOST_COEFFS`, `nonuniform_weights`.
"""
function nonuniform_ghost_coefficients(knots::AbstractVector{T}, num_ghost::Int,
                                       poly_degree::Int, side::Symbol) where T
    
    coef_matrix = zeros(T, num_ghost, poly_degree + 1)
    
    if side == :left
        # Spacings from the left boundary
        h0 = knots[2] - knots[1]   # first interval
        h1 = knots[3] - knots[2]   # second interval
        
        # Closed-form from symbolic Vandermonde solve:
        #   ghost at x_1 - h0, using points x_1, x_2, x_3
        coef_matrix[1, 1] =  2 * (2*h0 + h1) / (h0 + h1)
        coef_matrix[1, 2] = -(2*h0 + h1) / h1
        coef_matrix[1, 3] =  2 * h0^2 / (h1 * (h0 + h1))
    else  # :right
        # Spacings from the right boundary
        hm = knots[end-1] - knots[end-2]   # second-to-last interval
        hl = knots[end]   - knots[end-1]   # last interval
        
        # Closed-form from symbolic Vandermonde solve:
        #   ghost at x_n + hl, using points x_{n-2}, x_{n-1}, x_n
        coef_matrix[1, 1] =  2 * hl^2 / (hm * (hl + hm))
        coef_matrix[1, 2] = -(2*hl + hm) / hm
        coef_matrix[1, 3] =  2 * (2*hl + hm) / (hl + hm)
    end
    
    return coef_matrix
end

"""
    create_nonuniform_coefs(vs::AbstractArray{T,N}, knots::NTuple{N},
                            poly_degree::Int=2) where {T,N}

Create expanded coefficient array with ghost points for nonuniform grid interpolation.

# Arguments
- `vs`: Input data values on the interior grid
- `knots`: Tuple of knot vectors, one per dimension
- `poly_degree`: Polynomial degree for ghost point computation (default: 2)

# Returns
Tuple `(coefs, knots_expanded)` where:
- `coefs`: Array with 1 ghost point per side per dimension, size = `size(vs) .+ 2`
- `knots_expanded`: Tuple of expanded knot vectors including ghost positions

# Details
Applies Vandermonde-based ghost point computation dimension-by-dimension, mirroring
the approach used by `create_convolutional_coefs` for uniform grids. Each dimension
is processed independently using the separable tensor product structure.

Ghost points are placed by mirroring the nearest boundary interval spacing.
Coefficients reproduce polynomials up to `poly_degree` at ghost positions.

# Example
```julia
knots = ([0.0, 0.3, 0.7, 1.0, 1.8], [0.0, 0.5, 1.2, 2.0])
data = rand(5, 4)
coefs, knots_exp = create_nonuniform_coefs(data, knots)
# coefs is 7×6, knots_exp has 7 and 6 entries
```
"""
function create_nonuniform_coefs(vs::AbstractArray{T,N}, knots::NTuple{N};
                                  poly_degree::Int=2) where {T,N}
    num_ghost = 1  # 1 ghost point per side for 4-point cubic stencil
    
    # Expand array: add 1 ghost point on each side per dimension
    new_dims = size(vs) .+ 2 * num_ghost
    c = zeros(T, new_dims)
    
    # Copy interior values
    inner = ntuple(d -> (1+num_ghost):(new_dims[d]-num_ghost), N)
    c[inner...] = vs
    
    # Expand knot vectors
    knots_expanded = ntuple(N) do d
        k = knots[d]
        h_first = k[2] - k[1]
        h_last = k[end] - k[end-1]
        vcat(k[1] - h_first, k, k[end] + h_last)
    end
    
    # Apply ghost points dimension by dimension
    for dim in 1:N
        k = knots[dim]
        
        # Compute ghost coefficients for this dimension
        gc_left = nonuniform_ghost_coefficients(k, num_ghost, poly_degree, :left)
        gc_right = nonuniform_ghost_coefficients(k, num_ghost, poly_degree, :right)
        
        n_interior = size(gc_left, 2)  # poly_degree + 1
        
        # Build index ranges for all other dimensions (full expanded range)
        other_ranges = ntuple(N) do d
            d == dim ? (1:1) : (1:new_dims[d])  # placeholder for dim
        end
        
        # Process all slices along this dimension
        for idx in CartesianIndices(ntuple(d -> d == dim ? (1:1) : (1:new_dims[d]), N))
            # Left ghost: g_{-1} = sum(coef * interior_values)
            for j in 1:num_ghost
                ghost_idx = ntuple(d -> d == dim ? num_ghost + 1 - j : idx[d], N)
                val = zero(T)
                for k_idx in 1:n_interior
                    src_idx = ntuple(d -> d == dim ? num_ghost + k_idx : idx[d], N)
                    val += gc_left[j, k_idx] * c[src_idx...]
                end
                c[ghost_idx...] = val
            end
            
            # Right ghost: g_{n+1} = sum(coef * interior_values)
            n_dim = size(vs, dim)
            for j in 1:num_ghost
                ghost_idx = ntuple(d -> d == dim ? num_ghost + n_dim + j : idx[d], N)
                val = zero(T)
                for k_idx in 1:n_interior
                    src_idx = ntuple(d -> d == dim ? num_ghost + n_dim - n_interior + k_idx : idx[d], N)
                    val += gc_right[j, k_idx] * c[src_idx...]
                end
                c[ghost_idx...] = val
            end
        end
    end
    
    return c, knots_expanded
end