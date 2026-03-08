"""
    ConvolutionInterpolation(knots, vs::AbstractArray{T,N}; kwargs...) where {T,N}

Construct a convolution interpolation object for N-dimensional data using direct kernel evaluation.
This is the lower-level constructor — most users should prefer `convolution_interpolation`, which
wraps this with extrapolation handling.

# Arguments
- `knots`: Vector or range (1D) or tuple of vectors/ranges (N-D) of grid coordinates.
- `vs`: Array of values at the grid points.

# Keyword Arguments
- `degree::Symbol=:a4`: Convolution kernel to use.
  - `a`-series (uniform): `:a0` (nearest), `:a1` (linear), `:a3` (cubic), `:a4` (quartic), `:a5` (quintic), `:a7` (septic)
  - `b`-series (uniform and nonuniform): `:b5`, `:b7`, `:b9`, `:b11`, `:b13`
  - Nonuniform-only: `:n3` (cubic)
  `:a0`, `:a1`, and all `b`-series kernels work on both uniform and nonuniform grids.
  Higher `a`-series kernels (`:a3`, `:a4`, `:a5`, `:a7`) fall back to `:n3` on nonuniform grids.
- `B=nothing`: If provided, uses Gaussian kernel with parameter `B` for C∞ smoothness.
  Forces eager mode.
- `kernel_bc=:auto`: Boundary condition for kernel evaluation at domain edges.
  Options: `:auto`, `:polynomial`, `:linear`, `:quadratic`, `:periodic`, `:detect`.
- `derivative::Int=0`: Derivative order to evaluate. Supported up to 6 for `b`-series
  kernels. For `a`-series kernels the top derivative automatically uses linear interpolation
  to match the kernel's continuity class.
- `lazy::Bool=false`: When `true`, skip ghost point expansion at construction time.
  The raw values are stored directly and ghost points are computed on the fly during
  evaluation near boundaries. This saves memory and speeds up construction, especially
  in high dimensions. Automatically disabled for `:a0`, `:a1`, Gaussian kernels, and
  nonuniform `b`-series paths.
- `boundary_fallback::Bool=false`: When `true`, throw an error instead of computing ghost
  points when evaluating near boundaries in lazy mode. Prevents expensive tensor-product
  ghost computation in high dimensions. Only active when `lazy=true`.

# Grid handling
Uniform and nonuniform grids are detected automatically per dimension. On nonuniform grids:
- `:a0` and `:a1` work natively (no ghost points needed).
- `:a3`, `:a4`, `:a5`, `:a7` silently fall back to `:n3`.
- `b`-series kernels use precomputed polynomial weights with full ghost point expansion.

# Returns
A `ConvolutionInterpolation` object callable at arbitrary points within the grid domain.
Does not handle extrapolation — use `convolution_interpolation` or wrap in
`ConvolutionExtrapolation` for that.

See also: [`convolution_interpolation`](@ref), [`FastConvolutionInterpolation`](@ref), [`ConvolutionExtrapolation`](@ref).
"""

function ConvolutionInterpolation(knots::Union{NTuple{N,AbstractVector},
                                        AbstractVector,AbstractRange,NTuple{N,AbstractRange}},
                                  vs::AbstractArray{T,N};
                                  degree::Symbol=:b5, B=nothing,
                                  kernel_bc::Union{Symbol,Vector{Tuple{Symbol,Symbol}},NTuple{N,Tuple{Symbol,Symbol}}}=:auto,
                                  derivative::Int=0,
                                  lazy::Bool=false, boundary_fallback::Bool=false) where {T,N}
    # Convert knots to tuple if needed (if called directly)
    if knots isa AbstractVector || knots isa AbstractRange
        knots = (knots,)
    end

    # === Nonuniform path: if ANY dimension is nonuniform ===
    nonuniform_dims = [!is_uniform_grid(knots[d]) for d in 1:N]
    if any(nonuniform_dims)

        if degree in (:b5, :b7, :b9, :b11, :b13)
            # Nonuniform b-kernels always uses eager mode
            lazy = false

            # Nonuniform b-kernel path: precomputed polynomial weights
            M_eqs, p_deg = nonuniform_b_params(degree)
            eqs = M_eqs
            h = ntuple(d -> one(T), N) # dummy
            it = ntuple(_ -> ConvolutionMethod(), N)
            knots_new = ntuple(d -> collect(T, knots[d]), N)

            # Precompute weight coefficients for each dimension
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
            anchor = ntuple(d -> zero(T), N)
            do_type = derivative == -1 ? IntegralOrder() : DerivativeOrder(Val(derivative))

            return ConvolutionInterpolation{T,N,typeof(coefs),typeof(it),typeof(knots_expanded),typeof(kernel),
                                    typeof(dimension),typeof(Val(degree)),typeof(eqs),typeof(kernel_bc),
                                    typeof(do_type),typeof(kernel_d1_pre),typeof(kernel_d2_pre),
                                    typeof(Val(subgrid)),typeof(nb_wc)}(
                coefs, knots_expanded, it, h, kernel, dimension, Val(degree), eqs, kernel_bc, do_type,
                kernel_d1_pre, kernel_d2_pre, Val(subgrid), nb_wc, lazy, boundary_fallback, anchor
            )
        else
            # nonuniform path: a-series → :n3, or explicit :n3
            nu_degree, lazy = if degree in (:a0, :a1)
                degree, true
            elseif degree in (:n3, :a3, :a4, :a5, :a7)
                :n3, lazy
            else
                error("Nonuniform interpolation not supported for degree=$degree. " *
                      "Use a b-series kernel (:b5, :b7, ...) or :n3.")
            end

            eqs = 2
            h = ntuple(d -> one(T), N)
            it = ntuple(_ -> ConvolutionMethod(), N)
            knots_new = ntuple(d -> collect(T, knots[d]), N)
            coefs, knots_expanded = if lazy
                vs, ntuple(N) do d
                    k = knots_new[d]
                    h_first = k[2] - k[1]
                    h_last = k[end] - k[end-1]
                    vcat(k[1] - h_first, k, k[end] + h_last)
                end
            else
                create_nonuniform_coefs(vs, knots_new, degree=nu_degree)
            end

            kernel = ConvolutionKernel(Val(nu_degree), Val(derivative))
            dimension = N <= 3 ? Val(N) : HigherDimension(Val(N))
            kernel_d1_pre, kernel_d2_pre, subgrid = (nothing, nothing, :not_used)
            nb_wc = nothing
            anchor = ntuple(d -> zero(T), N)
            do_type = derivative == -1 ? IntegralOrder() : DerivativeOrder(Val(derivative))

            return ConvolutionInterpolation{T,N,typeof(coefs),typeof(it),typeof(knots_expanded),typeof(kernel),
                                    typeof(dimension),typeof(Val(nu_degree)),typeof(eqs),typeof(kernel_bc),
                                    typeof(do_type),typeof(kernel_d1_pre),typeof(kernel_d2_pre),
                                    typeof(Val(subgrid)),typeof(nb_wc)}(
                coefs, knots_expanded, it, h, kernel, dimension, Val(nu_degree), eqs, kernel_bc, do_type,
                kernel_d1_pre, kernel_d2_pre, Val(subgrid), nb_wc, lazy, boundary_fallback, anchor
            )
        end
    end

    # === Uniform path ===
    eqs = B === nothing ? get_equations_for_degree(degree) : 50
    h = map(k -> k[2] - k[1], knots)
    it = ntuple(_ -> ConvolutionMethod(), N)

    # ===== LAZY vs EAGER coefs/knots =====
    if lazy && degree != :a0 && degree != :a1 && B === nothing
        # LAZY: store raw values, original knots (not expanded)
        coefs = vs
        knots_new = ntuple(i -> collect(eltype(h), knots[i]), N)
    else
        # EAGER: original path
        lazy = false
        knots_new = expand_knots(knots, eqs-1)
        coefs = degree == :a0 || degree == :a1 ? vs : create_convolutional_coefs(vs, h, eqs, kernel_bc, degree, derivative)
    end
    if derivative == -1
        anchor = ntuple(d -> knots_new[d][eqs], N)
    else
        anchor = ntuple(d -> zero(T), N)
    end

    kernel = B === nothing ? ConvolutionKernel(Val(degree), Val(derivative)) : GaussianConvolutionKernel(Val(B))
    dimension = N <= 3 ? Val(N) : HigherDimension(Val(N))
    do_type = derivative == -1 ? IntegralOrder() : DerivativeOrder(Val(derivative))
    kernel_d1_pre, kernel_d2_pre, subgrid = (nothing, nothing, :not_used)
    nb_wc = nothing
    
    ConvolutionInterpolation{T,N,typeof(coefs),typeof(it),typeof(knots_new),typeof(kernel),
                            typeof(dimension),typeof(Val(degree)),typeof(eqs),typeof(kernel_bc),
                            typeof(do_type),typeof(kernel_d1_pre),typeof(kernel_d2_pre),
                            typeof(Val(subgrid)),typeof(nb_wc)}(
        coefs, knots_new, it, h, kernel, dimension, Val(degree), eqs, kernel_bc, do_type,
        kernel_d1_pre, kernel_d2_pre, Val(subgrid), nb_wc, lazy, boundary_fallback, anchor
    )
end