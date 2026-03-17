"""
    ConvolutionInterpolation(knots, vs::AbstractArray{T,N}; kwargs...) where {T,N}

Construct a convolution interpolation object for N-dimensional data using direct kernel evaluation.
This is the lower-level constructor — most users should prefer `convolution_interpolation`, which
wraps this with extrapolation handling.

# Arguments
- `knots`: Vector or range (1D) or tuple of vectors/ranges (N-D) of grid coordinates.
- `vs`: Array of values at the grid points.

# Keyword Arguments
- `kernel::Symbol=:b5`: Convolution kernel to use.
  - `a`-series (uniform): `:a0` (nearest), `:a1` (linear), `:a3` (cubic), `:a4` (quartic), `:a5` (quintic), `:a7` (septic)
  - `b`-series (uniform and nonuniform): `:b5`, `:b7`, `:b9`, `:b11`, `:b13`
  - Nonuniform-only: `:n3` (cubic)
  `:a0`, `:a1`, and all `b`-series kernels work on both uniform and nonuniform grids.
  Higher `a`-series kernels (`:a3`, `:a4`, `:a5`, `:a7`) fall back to `:n3` on nonuniform grids.
- `B=nothing`: If provided, uses Gaussian kernel with parameter `B` for C∞ smoothness.
  Forces eager mode.
- `bc=:auto`: Boundary condition for kernel evaluation at domain edges.
  Options: `:auto`, `:polynomial`, `:linear`, `:quadratic`, `:periodic`.
- `derivative::Int=0`: Derivative order to evaluate. Supported up to 6 for `b`-series
  kernels. For `a`-series kernels the top derivative automatically uses linear interpolation
  to match the kernel's continuity class.
- `lazy::Bool=false`: When `true`, skip ghost point expansion at construction time.
  The raw values are stored directly and ghost points are computed on the fly during
  evaluation near boundaries. This saves memory and speeds up construction, especially
  in high dimensions. Automatically disabled for `:a0`, `:a1`, Gaussian kernels, and
  nonuniform `b`-series paths.
- `boundary_fallback::Bool=false`: When `true`, near-boundary evaluations are linearly
  extrapolated from interior points rather than computing full ghost point stencils.
  Prevents expensive tensor-product ghost computation in high dimensions.
  Only active when `lazy=true`.
  
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
                                  kernel::Union{Symbol,NTuple{N,Symbol}}=:b5, B=nothing,
                                  bc::Union{Symbol,Vector{Tuple{Symbol,Symbol}},NTuple{N,Tuple{Symbol,Symbol}}}=:auto,
                                  derivative::Union{Int,NTuple{N,Int}}=0,
                                  lazy::Bool=false, boundary_fallback::Bool=false) where {T,N}
    # Convert knots to tuple if needed (if called directly)
    if knots isa AbstractVector || knots isa AbstractRange
        knots = (knots,)
    end

    # === Nonuniform path: if ANY dimension is nonuniform ===
    nonuniform_dims = [!is_uniform_grid(knots[d]) for d in 1:N]
    if any(nonuniform_dims)

        if (derivative isa Int && derivative < 0) || (derivative isa Tuple && any(d -> d < 0, derivative))
            error("Nonuniform antiderivative (derivative=-1) is not directly supported.\n" *
                "Recommended workflow:\n" *
                "       (1) construct a nonuniform derivative=0 interpolation,\n" *
                "       (2) resample to a uniform grid,\n" *
                "       (3) construct a uniform derivative=-1 interpolation.")
        end

        is_perdim_kernel = kernel isa NTuple{N,Symbol}
        is_perdim_deriv  = derivative isa NTuple{N,Int}
 
        if (is_perdim_kernel || is_perdim_deriv)

            if N == 1
                error("Per-dimension kernel/derivative specification is not meaningful in 1D. Use scalar values.")
            end
            
            kernels     = kernel     isa Symbol ? ntuple(_ -> kernel,     N) : kernel
            derivatives = derivative isa Int    ? ntuple(_ -> derivative, N) : derivative
 
            if any(d -> derivatives[d] == -1, 1:N)
                error("Nonuniform antiderivative (derivative=-1) is not supported.")
            end
 
            # All per-dim kernels must be b-series for nonuniform
            if !all(d -> kernels[d] in (:b5, :b7, :b9, :b11, :b13), 1:N)
                error("Per-dimension nonuniform interpolation requires b-series kernels " *
                      "in all dimensions. Got: $kernels")
            end
 
            lazy = false  # nonuniform b always eager
 
            h = ntuple(d -> one(T), N)
            it = ntuple(_ -> ConvolutionMethod(), N)
            knots_new = ntuple(d -> collect(T, knots[d]), N)
 
            # Per-dim params
            params = ntuple(d -> nonuniform_b_params(kernels[d]), N)
            M_eqs_d = ntuple(d -> params[d][1], N)
 
            # Per-dim weight coefficients + expanded knots
            nb_data = ntuple(N) do d
                if nonuniform_dims[d]
                    precompute_nonuniform_b_all(knots_new[d], kernels[d], derivatives[d])
                else
                    precompute_uniform_b_all(knots_new[d], kernels[d], derivatives[d])
                end
            end
 
            knots_expanded = ntuple(d -> nb_data[d][2], N)
            nb_wc = ntuple(d -> nb_data[d][1], N)
 
            # Ghost coefs: use per-dim M_eqs
            coefs = create_nonuniform_b_coefs_perdim(vs, knots_new, kernels)
 
            eqs = M_eqs_d   # NTuple{N,Int}
 
            kernel_type    = ntuple(d -> ConvolutionKernel(Val(kernels[d]), Val(derivatives[d])), N)
            dimension = N <= 3 ? Val(N) : HigherDimension(Val(N))
            kernel_d1_pre, kernel_d2_pre, subgrid = (nothing, nothing, :not_used)
            anchor = ntuple(d -> zero(T), N)
 
            n_integral = count(d -> derivatives[d] == -1, 1:N)
            do_type = if n_integral == 0
                DerivativeOrder(Val(derivatives))
            elseif n_integral == N
                IntegralOrder()
            else
                MixedIntegralOrder{derivatives}()
            end
 
            kernels_val = Val(kernels)   # Val{(:b5,:b7,...)} for dispatch
 
            return ConvolutionInterpolation{T,N,typeof(coefs),typeof(it),typeof(knots_expanded),
                                    typeof(kernel_type),typeof(dimension),typeof(kernels_val),typeof(eqs),
                                    typeof(bc),typeof(do_type),typeof(kernel_d1_pre),
                                    typeof(kernel_d2_pre),typeof(Val(subgrid)),typeof(nb_wc)}(
                coefs, knots_expanded, it, h, kernel_type, dimension, kernels_val, eqs, bc, do_type,
                kernel_d1_pre, kernel_d2_pre, Val(subgrid), nb_wc, lazy, boundary_fallback, anchor
            )
        end

        if kernel in (:b5, :b7, :b9, :b11, :b13)
            # Nonuniform b-kernels always uses eager mode
            lazy = false

            # Nonuniform b-kernel path: precomputed polynomial weights
            M_eqs, p_deg = nonuniform_b_params(kernel)
            eqs = M_eqs
            h = ntuple(d -> one(T), N) # dummy
            it = ntuple(_ -> ConvolutionMethod(), N)
            knots_new = ntuple(d -> collect(T, knots[d]), N)

            # Precompute weight coefficients for each dimension
            nb_data = ntuple(N) do d
                if nonuniform_dims[d]
                    precompute_nonuniform_b_all(knots_new[d], kernel, derivative)
                else
                    precompute_uniform_b_all(knots_new[d], kernel, derivative)
                end
            end

            knots_expanded = ntuple(d -> nb_data[d][2], N)
            nb_wc = ntuple(d -> nb_data[d][1], N)

            # Build coefficient array with ghost points
            coefs, _ = create_nonuniform_b_coefs(vs, knots_new, kernel)

            kernel_type = ConvolutionKernel(Val(kernel), Val(derivative))
            dimension = N <= 3 ? Val(N) : HigherDimension(Val(N))
            kernel_d1_pre, kernel_d2_pre, subgrid = (nothing, nothing, :not_used)
            anchor = ntuple(d -> zero(T), N)
            do_type = derivative == -1 ? IntegralOrder() : DerivativeOrder(Val(derivative))

            return ConvolutionInterpolation{T,N,typeof(coefs),typeof(it),typeof(knots_expanded),typeof(kernel_type),
                                    typeof(dimension),typeof(Val(kernel)),typeof(eqs),typeof(bc),
                                    typeof(do_type),typeof(kernel_d1_pre),typeof(kernel_d2_pre),
                                    typeof(Val(subgrid)),typeof(nb_wc)}(
                coefs, knots_expanded, it, h, kernel_type, dimension, Val(kernel), eqs, bc, do_type,
                kernel_d1_pre, kernel_d2_pre, Val(subgrid), nb_wc, lazy, boundary_fallback, anchor
            )
        else
            # nonuniform path: a-series → :n3, or explicit :n3
            nu_kernel, lazy = if kernel in (:a0, :a1)
                kernel, true
            elseif kernel in (:n3, :a3, :a4, :a5, :a7)
                :n3, lazy
            else
                error("Nonuniform interpolation not supported for kernel=$kernel. " *
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
                create_nonuniform_coefs(vs, knots_new, degree=nu_kernel)
            end

            kernel_type = ConvolutionKernel(Val(nu_kernel), Val(derivative))
            dimension = N <= 3 ? Val(N) : HigherDimension(Val(N))
            kernel_d1_pre, kernel_d2_pre, subgrid = (nothing, nothing, :not_used)
            nb_wc = nothing
            anchor = ntuple(d -> zero(T), N)
            do_type = derivative == -1 ? IntegralOrder() : DerivativeOrder(Val(derivative))

            return ConvolutionInterpolation{T,N,typeof(coefs),typeof(it),typeof(knots_expanded),typeof(kernel_type),
                                    typeof(dimension),typeof(Val(nu_kernel)),typeof(eqs),typeof(bc),
                                    typeof(do_type),typeof(kernel_d1_pre),typeof(kernel_d2_pre),
                                    typeof(Val(subgrid)),typeof(nb_wc)}(
                coefs, knots_expanded, it, h, kernel_type, dimension, Val(nu_kernel), eqs, bc, do_type,
                kernel_d1_pre, kernel_d2_pre, Val(subgrid), nb_wc, lazy, boundary_fallback, anchor
            )
        end
    end
    
    # === Uniform path ===
    kernel     = N == 1 ? kernel     : kernel     isa Symbol ? ntuple(_ -> kernel,     N) : kernel
    derivative = N == 1 ? derivative : derivative isa Int    ? ntuple(_ -> derivative, N) : derivative
    
    eqs = B === nothing ? N == 1 ? get_equations_for_degree(kernel) : ntuple(d -> get_equations_for_degree(kernel[d]), N) : ntuple(_ -> 50, N)
    h = map(k -> k[2] - k[1], knots)
    it = ntuple(_ -> ConvolutionMethod(), N)

    # ===== LAZY vs EAGER coefs/knots =====
    if N == 1
        if lazy && kernel != :a0 && kernel != :a1 && B === nothing
            coefs = vs
            knots_new = knots
        else
            lazy = false
            knots_new = expand_knots(knots, eqs-1)
            coefs = kernel == :a0 || kernel == :a1 ? vs : create_convolutional_coefs(vs, h, eqs, bc, kernel)
        end
    else
        if lazy && all(d -> kernel[d] != :a0 && kernel[d] != :a1, 1:N) && B === nothing
            coefs = vs
            knots_new = ntuple(i -> collect(eltype(h), knots[i]), N)
        else
            lazy = false
            knots_new = expand_knots(knots, ntuple(d -> eqs[d]-1, N))
            coefs = if all(d -> kernel[d] in (:a0, :a1), 1:N)
                vs
            else
                create_convolutional_coefs(vs, h, eqs, bc, kernel)
            end
        end
    end
    if N == 1
        if derivative == -1
            anchor = (knots_new[1][eqs],)
        else
            anchor = (zero(T),)
        end
    else
        if any(d -> derivative[d] == -1, 1:N)
            anchor = ntuple(d -> knots_new[d][eqs[d]], N)
        else
            anchor = ntuple(d -> zero(T), N)
        end
    end

    kernel_type = B === nothing ? N == 1 ? ConvolutionKernel(Val(kernel), Val(derivative)) : 
                            ntuple(d -> ConvolutionKernel(Val(kernel[d]), Val(derivative[d])), N) :
                            GaussianConvolutionKernel(Val(B))
    dimension = N <= 3 ? Val(N) : HigherDimension(Val(N))
    do_type = if N == 1
        derivative == -1 ? IntegralOrder() : DerivativeOrder(Val(derivative))
    else
        n_int = count(d -> derivative[d] == -1, 1:N)
        if n_int == 0
            DerivativeOrder(Val(derivative))
        elseif n_int == N
            IntegralOrder()
        else
            MixedIntegralOrder{derivative}()
        end
    end
    kernel_d1_pre, kernel_d2_pre, subgrid = (nothing, nothing, :not_used)
    nb_wc = nothing

    ConvolutionInterpolation{T,N,typeof(coefs),typeof(it),typeof(knots_new),typeof(kernel_type),
                            typeof(dimension),typeof(Val(kernel)),typeof(eqs),typeof(bc),
                            typeof(do_type),typeof(kernel_d1_pre),typeof(kernel_d2_pre),
                            typeof(Val(subgrid)),typeof(nb_wc)}(
        coefs, knots_new, it, h, kernel_type, dimension, Val(kernel), eqs, bc, do_type,
        kernel_d1_pre, kernel_d2_pre, Val(subgrid), nb_wc, lazy, boundary_fallback, anchor
    )
end