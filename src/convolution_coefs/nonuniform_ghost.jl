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
for computing ghost point `g_{-j}` from interior values.

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
"""
function nonuniform_ghost_coefficients(knots::AbstractVector{T}, num_ghost::Int,
                                       poly_degree::Int, side::Symbol) where T
    
    coef_matrix = zeros(T, num_ghost, poly_degree + 1)
    
    if side == :left
        h0 = knots[2] - knots[1]
        h1 = knots[3] - knots[2]
        
        coef_matrix[1, 1] =  2 * (2*h0 + h1) / (h0 + h1)
        coef_matrix[1, 2] = -(2*h0 + h1) / h1
        coef_matrix[1, 3] =  2 * h0^2 / (h1 * (h0 + h1))
    else  # :right
        hm = knots[end-1] - knots[end-2]
        hl = knots[end]   - knots[end-1]
        
        coef_matrix[1, 1] =  2 * hl^2 / (hm * (hl + hm))
        coef_matrix[1, 2] = -(2*hl + hm) / hm
        coef_matrix[1, 3] =  2 * (2*hl + hm) / (hl + hm)
    end
    
    return coef_matrix
end

"""
    create_nonuniform_coefs(vs::AbstractArray{T,N}, knots::NTuple{N};
                            degree::Symbol=:n3) where {T,N}

Create expanded coefficient array with ghost points for nonuniform cubic (:n3) interpolation.

# Returns
Tuple `(coefs, knots_expanded)` where:
- `coefs`: Array with 1 ghost point per side per dimension
- `knots_expanded`: Tuple of expanded knot vectors including ghost positions
"""
function create_nonuniform_coefs(vs::AbstractArray{T,N}, knots::NTuple{N};
                                degree::Symbol=:n3) where {T,N}
    @assert degree == :n3 "create_nonuniform_coefs only supports :n3, got $degree"

    num_ghost = 1
    poly_degree = 2
                                  
    new_dims = size(vs) .+ 2 * num_ghost
    c = zeros(T, new_dims)
    
    inner = ntuple(d -> (1+num_ghost):(new_dims[d]-num_ghost), N)
    c[inner...] = vs
    
    knots_expanded = ntuple(N) do d
        k = knots[d]
        h_first = k[2] - k[1]
        h_last = k[end] - k[end-1]
        vcat(k[1] - h_first, k, k[end] + h_last)
    end
    
    for dim in 1:N
        k = knots[dim]
        gc_left = nonuniform_ghost_coefficients(k, num_ghost, poly_degree, :left)
        gc_right = nonuniform_ghost_coefficients(k, num_ghost, poly_degree, :right)
        n_interior = size(gc_left, 2)
        
        for idx in CartesianIndices(ntuple(d -> d == dim ? (1:1) : (1:new_dims[d]), N))
            for j in 1:num_ghost
                ghost_idx = ntuple(d -> d == dim ? num_ghost + 1 - j : idx[d], N)
                val = zero(T)
                for k_idx in 1:n_interior
                    src_idx = ntuple(d -> d == dim ? num_ghost + k_idx : idx[d], N)
                    val += gc_left[j, k_idx] * c[src_idx...]
                end
                c[ghost_idx...] = val
            end
            
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


# ══════════════════════════════════════════════════════════════
# Ghost points for nonuniform b-kernels
#
# For b5 (M_eqs=5): 5 ghost points per side
# For b7 (M_eqs=6): 6 ghost points per side, etc.
#
# Uses general Vandermonde solve (not closed-form) since we need
# up to 9 ghost points with polynomial reproduction up to degree 13.
# ══════════════════════════════════════════════════════════════

"""
    nonuniform_b_ghost_coefficients(knots::AbstractVector{T}, num_ghost::Int,
                                     poly_degree::Int, side::Symbol) where T

Compute ghost point extrapolation coefficients for nonuniform b-kernel boundaries.

Uses Vandermonde system to express each ghost point as a linear combination of
`poly_degree + 1` nearest interior points, enforcing polynomial reproduction.
The system is set up in the local coordinate `(x - x_boundary) / h_end`, which leaves
the coefficients unchanged and keeps it well conditioned on grids far from the origin.

# Arguments
- `knots`: Original (unexpanded) knot vector
- `num_ghost`: Number of ghost points per side (= M_eqs)
- `poly_degree`: Polynomial reproduction degree (= p_deg of the kernel)
- `side`: `:left` or `:right`

# Returns
`Matrix{T}` of size `(num_ghost, poly_degree+1)`.
Row `g` gives coefficients for ghost point `g` (g=1 is closest to boundary).
"""
function nonuniform_b_ghost_coefficients(knots::AbstractVector{T}, num_ghost::Int,
                                          poly_degree::Int, side::Symbol) where T
    n = length(knots)
    n_stencil = poly_degree + 1

    coef_matrix = zeros(T, num_ghost, n_stencil)

    # Positions are mapped to ξ = (x - x_boundary) / h_end, with x_boundary the boundary knot
    # and h_end the spacing of the end interval (the ghost spacing). Polynomial reproduction is
    # invariant under this affine map, so the coefficients are unchanged, but the Vandermonde
    # entries become O(1) … O(p) instead of x^p for wherever the grid happens to lie, which is
    # what keeps the solve well conditioned for the wide kernels (p up to 13).
    if side == :left
        x0 = knots[1]
        h  = knots[2] - knots[1]
        ξ  = [(knots[k] - x0) / h for k in 1:n_stencil]          # 0, h1/h, …
        ghost_ξ = g -> -T(g)                                      # ghost g at x_1 - g*h
    else  # :right
        x0 = knots[end]
        h  = knots[end] - knots[end-1]
        ξ  = [(knots[k] - x0) / h for k in (n - n_stencil + 1):n] # …, -h_n/h, 0
        ghost_ξ = g -> T(g)                                       # ghost g at x_n + g*h
    end

    # Vandermonde in ξ: V[k, m] = ξ[k]^(m-1), the same for every ghost
    V = zeros(T, n_stencil, n_stencil)
    for k in 1:n_stencil
        V[k, 1] = one(T)
        for m in 2:n_stencil
            V[k, m] = V[k, m-1] * ξ[k]
        end
    end
    Vt = factorize(V')

    for g in 1:num_ghost
        # Target: ghost_ξ^(m-1) for m = 1, ..., n_stencil
        target = zeros(T, n_stencil)
        gp_pow = one(T)
        for m in 1:n_stencil
            target[m] = gp_pow
            gp_pow *= ghost_ξ(g)
        end
        coef_matrix[g, :] = Vt \ target
    end

    return coef_matrix
end


"""
    create_nonuniform_b_coefs(vs::AbstractArray{T,N}, knots::NTuple{N},
                               degree::Symbol) where {T,N}

Create expanded coefficient array with ghost points for nonuniform b-kernel interpolation.

# Arguments
- `vs`: Input data values on the interior grid
- `knots`: Tuple of knot vectors, one per dimension
- `degree`: B-kernel symbol (`:b5`, `:b7`, `:b9`, `:b11`, `:b13`)

# Returns
Tuple `(coefs, knots_expanded)` where:
- `coefs`: Array with `M_eqs` ghost points per side per dimension
- `knots_expanded`: Tuple of expanded knot vectors including ghost positions
"""

function create_nonuniform_b_coefs_perdim(vs::AbstractArray{T,N},
                                           knots::NTuple{N,AbstractVector},
                                           degrees::NTuple{N,Symbol}) where {T,N}
    params  = ntuple(d -> nonuniform_b_params(degrees[d]), N)
    M_eqs_d = ntuple(d -> params[d][1], N)
    p_deg_d = ntuple(d -> params[d][2], N)
 
    # Each dim padded by its own M_eqs
    new_dims = ntuple(d -> size(vs, d) + 2 * M_eqs_d[d], N)
    c = zeros(T, new_dims...)
 
    # Copy interior values — offset is M_eqs_d[d] per dim
    inner = ntuple(d -> (1 + M_eqs_d[d]):(new_dims[d] - M_eqs_d[d]), N)
    c[inner...] = vs
 
    # Fill ghost points dimension by dimension
    for dim in 1:N
        ng = M_eqs_d[dim]
        k  = knots[dim]
        gc_left  = nonuniform_b_ghost_coefficients(k, ng, p_deg_d[dim], :left)
        gc_right = nonuniform_b_ghost_coefficients(k, ng, p_deg_d[dim], :right)
        n_interior_pts = size(gc_left, 2)
        n_dim = size(vs, dim)
 
        # Iterate over all slices perpendicular to dim
        for idx in CartesianIndices(ntuple(d -> d == dim ? (1:1) : (1:new_dims[d]), N))
            for g in 1:ng
                ghost_idx = ntuple(d -> d == dim ? ng + 1 - g : idx[d], N)
                val = zero(T)
                for k_idx in 1:n_interior_pts
                    src_idx = ntuple(d -> d == dim ? ng + k_idx : idx[d], N)
                    val += gc_left[g, k_idx] * c[src_idx...]
                end
                c[ghost_idx...] = val
            end
 
            for g in 1:ng
                ghost_idx = ntuple(d -> d == dim ? ng + n_dim + g : idx[d], N)
                val = zero(T)
                for k_idx in 1:n_interior_pts
                    src_idx = ntuple(d -> d == dim ? ng + n_dim - n_interior_pts + k_idx : idx[d], N)
                    val += gc_right[g, k_idx] * c[src_idx...]
                end
                c[ghost_idx...] = val
            end
        end
    end
 
    return c
end