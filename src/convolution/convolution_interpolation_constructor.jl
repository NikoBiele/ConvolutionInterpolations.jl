"""
    ConvolutionInterpolation(knots, vs::AbstractArray{T,N};
        degree::Symbol=:b5, B=nothing, kernel_bc=:auto, derivative::Int=0) where {T,N}

Construct a convolution interpolation object for N-dimensional data using direct kernel evaluation.

# Arguments
- `knots`: Vector, range, or tuple of vectors/ranges containing the grid points in each dimension
- `vs::AbstractArray{T,N}`: Values to interpolate at the grid points

# Keyword Arguments
- `degree::Symbol=:b5`: Convolution kernel to use. Available kernels:
  - `a`-series: `:a0` (nearest), `:a1` (linear), `:a3` (cubic), `:a5` (quintic), `:a7` (septic)
  - `b`-series (recommended): `:a4`, `:b5`, `:b7`, `:b9`, `:b11`, `:b13`
  - Nonuniform-specific: `:n3` (cubic)
  - Default `:b5` provides quintic reproduction with 7th-order convergence
- `B=nothing`: If provided, uses Gaussian kernel with parameter `B` instead of polynomial kernel
- `kernel_bc=:auto`: Boundary condition for kernel evaluation at domain boundaries.
  Options: `:auto`, `:polynomial`, `:linear`, `:quadratic`, `:periodic`, `:detect`
- `derivative::Int=0`: Order of derivative to evaluate (0 for interpolation)

# Returns
`ConvolutionInterpolation` object callable at arbitrary points within the grid domain.

# Nonuniform grid handling
When any dimension has nonuniform spacing:
- **b-series kernels** (`:b5` through `:b13`): Uses precomputed polynomial weight
  coefficients with exact `Rational{BigInt}` arithmetic. Provides O(h^(p_deg+1))
  convergence where p_deg is the kernel's polynomial degree. Supports derivatives.
- **a-series kernels** (`:a0` through `:a7`): Falls back to nonuniform cubic (`:n3`)
  with O(h³) convergence.

# Performance
Direct kernel evaluation without precomputed tables. For faster evaluation, use
`FastConvolutionInterpolation` or `convolution_interpolation(..., fast=true)` which
provides O(1) lookup via precomputed kernel tables.

# Examples
```julia
# 1D interpolation with quintic kernel
knots = (range(0, 1, 100),)
data = sin.(2π .* knots[1])
itp = ConvolutionInterpolation(knots, data)
itp(0.5)

# Nonuniform grid with b5 kernel
knots = ([0.0, 0.3, 0.7, 1.0, 1.8, 2.5, 3.0],)
data = sin.(knots[1])
itp = ConvolutionInterpolation(knots, data, degree=:b5)
itp(0.5)
```

See also: `convolution_interpolation` for the recommended high-level interface with extrapolation.
"""

function ConvolutionInterpolation(knots::Union{NTuple{N,AbstractVector},
                                        AbstractVector,AbstractRange,NTuple{N,AbstractRange}},
                                  vs::AbstractArray{T,N};
                                  degree::Symbol=:b5, B=nothing, kernel_bc=:auto,
                                  derivative::Int=0) where {T,N}
    # Convert knots to tuple if needed (if called directly)
    if knots isa AbstractVector || knots isa AbstractRange
        knots = (knots,)
    end

    # === Nonuniform path: if ANY dimension is nonuniform ===
    nonuniform_dims = [!is_uniform_grid(knots[d]) for d in 1:N]
    if any(nonuniform_dims)

        is_b_kernel = degree in (:b5, :b7, :b9, :b11, :b13)

        if is_b_kernel
            # Nonuniform b-kernel path: precomputed polynomial weights
            M_eqs, p_deg = nonuniform_b_params(degree)
            eqs = M_eqs
            h = ntuple(d -> one(T), N) # dummy
            it = ntuple(_ -> ConvolutionMethod(), N)
            knots_new = ntuple(d -> collect(T, knots[d]), N)

            # Precompute weight coefficients for each dimension
            # Uses the derivative order to select the appropriate kernel coefficients
            nb_data = ntuple(N) do d
                if nonuniform_dims[d]
                    precompute_nonuniform_b_all(knots_new[d], degree, derivative)
                else
                    precompute_uniform_b_all(knots_new[d], degree, derivative)
                end
            end

            knots_expanded = ntuple(d -> nb_data[d][2], N)
            nb_wc = ntuple(d -> nb_data[d][1], N)

            # Build coefficient array with ghost points
            coefs, _ = create_nonuniform_b_coefs(vs, knots_new, degree)

            kernel = ConvolutionKernel(Val(degree), Val(derivative))
            dimension = N <= 3 ? Val(N) : HigherDimension(Val(N))
            kernel_d1_pre, kernel_d2_pre, subgrid = (nothing, nothing, :not_used)

            return ConvolutionInterpolation{T,N,typeof(coefs),typeof(it),typeof(knots_expanded),typeof(kernel),
                                    typeof(dimension),typeof(Val(degree)),typeof(eqs),typeof(kernel_bc),
                                    typeof(Val(derivative)),typeof(kernel_d1_pre),typeof(kernel_d2_pre),
                                    typeof(Val(subgrid)),typeof(nb_wc)}(
                coefs, knots_expanded, it, h, kernel, dimension, Val(degree), eqs, kernel_bc, Val(derivative),
                kernel_d1_pre, kernel_d2_pre, Val(subgrid), nb_wc
            )
        else
            # nonuniform path: a-series → :n3, or explicit :n3
            nu_degree = if degree in (:n3, :a0, :a1, :a3, :a4, :a5, :a7)
                :n3   # fall back to nonuniform cubic for all a-series kernels
            else
                error("Nonuniform interpolation not supported for degree=$degree. " *
                      "Use a b-series kernel (:b5, :b7, ...) or :n3.")
            end

            eqs = 2
            h = ntuple(d -> one(T), N)
            it = ntuple(_ -> ConvolutionMethod(), N)
            knots_new = ntuple(d -> collect(T, knots[d]), N)
            coefs, knots_expanded = create_nonuniform_coefs(vs, knots_new, degree=nu_degree)
            kernel = ConvolutionKernel(Val(nu_degree), Val(derivative))
            dimension = N <= 3 ? Val(N) : HigherDimension(Val(N))
            kernel_d1_pre, kernel_d2_pre, subgrid = (nothing, nothing, :not_used)
            nb_wc = nothing

            return ConvolutionInterpolation{T,N,typeof(coefs),typeof(it),typeof(knots_expanded),typeof(kernel),
                                    typeof(dimension),typeof(Val(nu_degree)),typeof(eqs),typeof(kernel_bc),
                                    typeof(Val(derivative)),typeof(kernel_d1_pre),typeof(kernel_d2_pre),
                                    typeof(Val(subgrid)),typeof(nb_wc)}(
                coefs, knots_expanded, it, h, kernel, dimension, Val(nu_degree), eqs, kernel_bc, Val(derivative),
                kernel_d1_pre, kernel_d2_pre, Val(subgrid), nb_wc
            )
        end
    end

    # === Existing uniform path ===
    eqs = B === nothing ? get_equations_for_degree(degree) : 50
    h = map(k -> k[2] - k[1], knots)
    it = ntuple(_ -> ConvolutionMethod(), N)
    knots_new = expand_knots(knots, eqs-1) # expand boundaries
    coefs = degree == :a0 || degree == :a1 ? vs : create_convolutional_coefs(vs, h, eqs, kernel_bc, degree) # create boundaries
    kernel = B === nothing ? ConvolutionKernel(Val(degree), Val(derivative)) : GaussianConvolutionKernel(Val(B))
    dimension = N <= 3 ? Val(N) : HigherDimension(Val(N))
    kernel_d1_pre, kernel_d2_pre, subgrid = (nothing, nothing, nothing, nothing, :not_used)
    nb_wc = nothing
    
    ConvolutionInterpolation{T,N,typeof(coefs),typeof(it),typeof(knots_new),typeof(kernel),
                            typeof(dimension),typeof(Val(degree)),typeof(eqs),typeof(kernel_bc),
                            typeof(Val(derivative)),typeof(kernel_d1_pre),typeof(kernel_d2_pre),
                            typeof(Val(subgrid)),typeof(nb_wc)}(
        coefs, knots_new, it, h, kernel, dimension, Val(degree), eqs, kernel_bc, Val(derivative),
        kernel_d1_pre, kernel_d2_pre, Val(subgrid), nb_wc
    )
end