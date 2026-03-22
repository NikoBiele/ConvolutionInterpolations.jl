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
  Options: `:auto`, `:poly`, `:linear`, `:quadratic`, `:periodic`.
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

    # convert inputs to expected types (if called directly)
    knots_tuple = knots isa NTuple{N,AbstractVector} ? knots : knots isa AbstractVector || knots isa AbstractRange ? (knots,) : knots
    kernels_tuple = kernel isa NTuple{N,Symbol} ? kernel : kernel isa Symbol ? ntuple(_ -> kernel, N) : kernel
    derivatives_tuple = derivative isa NTuple{N,Int} ? derivative : derivative isa Int ? ntuple(_ -> derivative, N) : derivative

    # Normalize bc to per-dimension, per-side tuples (for direct calls)
    bc = if bc isa Symbol
        ntuple(_ -> (bc, bc), N)
    elseif bc isa Vector{Tuple{Symbol,Symbol}}
        ntuple(d -> bc[d], N)
    elseif bc isa NTuple{N,Tuple{Symbol,Symbol}}
        bc  # already a tuple of tuples
    else
        error("Unsupported bc type: $(typeof(bc)), must be Symbol, Vector{Tuple{Symbol,Symbol}}, or NTuple{N,Tuple{Symbol,Symbol}}.")
    end

    # === Uniform path: if all dimensions are uniform ===
    uniform_dims = ntuple(d -> is_uniform_grid(knots_tuple[d]), N)
    
    if all(uniform_dims)
        # === Uniform path ===

        eqs = ntuple(N) do d
            if B === nothing
                get_equations_for_degree(kernels_tuple[d])
            else
                ntuple(_ -> 50, N)
            end
        end
        h = map(k -> k[2] - k[1], knots_tuple)
        it = ntuple(_ -> ConvolutionMethod(), N)

        # ===== LAZY vs EAGER coefs/knots =====
        high_order_kernel_used = any(d -> kernels_tuple[d] in (:a3, :a4, :a5, :a7, :b5, :b7, :b9, :b11, :b13), 1:N)
        if (lazy || !high_order_kernel_used) && B === nothing
            coefs = vs
            knots_new = knots_tuple
        else
            lazy = false
            knots_new = expand_knots(knots_tuple, ntuple(d -> eqs[d]-1, N))
            coefs = create_convolutional_coefs(vs, h, eqs, bc, kernels_tuple)
        end
        if N == 1
            if derivatives_tuple[1] == -1
                anchor = (knots_new[1][eqs[1]],)
            else
                anchor = (zero(T),)
            end
        else
            if any(d -> derivatives_tuple[d] == -1, 1:N)
                anchor = ntuple(d -> knots_new[d][eqs[d]], N)
            else
                anchor = ntuple(d -> zero(T), N)
            end
        end

        kernel_type = B === nothing ? ntuple(d -> ConvolutionKernel(Val(kernels_tuple[d]), Val(derivatives_tuple[d])), N) :
                                ntuple(d -> GaussianConvolutionKernel(Val(B)), N)
        dimension = N <= 3 ? Val(N) : HigherDimension(Val(N))
        do_type = if N == 1
            derivatives_tuple[1] == -1 ? IntegralOrder() : DerivativeOrder(Val(derivatives_tuple))
        else
            n_int = count(d -> derivatives_tuple[d] == -1, 1:N)
            if n_int == 0
                DerivativeOrder(Val(derivatives_tuple))
            elseif n_int == N
                IntegralOrder()
            else
                MixedIntegralOrder{derivatives_tuple}()
            end
        end
        n_integral = count(d -> derivatives_tuple[d] == -1, 1:N)
        integral_dims = ntuple(d -> derivatives_tuple[d] == -1, N)

        kernel_d1_pre, kernel_d2_pre, subgrid = (nothing, nothing, (:not_used))
        nb_wc = nothing

        ConvolutionInterpolation{T,N,typeof(coefs),typeof(it),typeof(knots_new),typeof(kernel_type),
                                typeof(dimension),typeof(Val(kernel)),typeof(eqs),typeof(bc),
                                typeof(do_type),typeof(kernel_d1_pre),typeof(kernel_d2_pre),
                                typeof(Val(subgrid)),typeof(nb_wc),typeof(Val(lazy)),typeof(n_integral)}(
            coefs, knots_new, it, h, kernel_type, dimension, Val(kernel), eqs, bc, do_type,
            kernel_d1_pre, kernel_d2_pre, Val(subgrid), nb_wc, Val(lazy), boundary_fallback,
            anchor, n_integral, integral_dims
        )

    else
        # === Nonuniform path ===

        if any(d -> derivatives_tuple[d] == -1, 1:N)
            error("Nonuniform antiderivative (derivative=-1) is not directly supported.\n" *
                "Recommended workflow:\n" *
                "       (1) construct a nonuniform derivative=0 interpolation,\n" *
                "       (2) resample to a uniform grid,\n" *
                "       (3) construct a uniform derivative=-1 interpolation.")
        end

        # check inputs
        all_b_kernels = all(d -> kernels_tuple[d] in (:b5, :b7, :b9, :b11, :b13), 1:N)
        all_n3_kernels = allequal(kernels_tuple) && (kernels_tuple[1] == :n3 || kernels_tuple[1] == :a3)
        all_a0_kernels = all(d -> kernels_tuple[d] == :a0, 1:N)
        all_a1_kernels = all(d -> kernels_tuple[d] == :a1, 1:N)
                     
        if !all_b_kernels && !all_n3_kernels && !all_a0_kernels && !all_a1_kernels
            error("Nonuniform interpolation requires all kernels in all dimensions to be either:\n" *
                  "          - all :b kernels (:b5, :b7, :b9, :b11, :b13),\n"*
                  "          - all :n3 kernels,\n" *
                  "          - all :a0 kernels or\n"*
                  "          - all :a1 kernels.\n" *
                  "Got: $kernels_tuple.")
        end

        if all_b_kernels

            # Nonuniform b-kernel path: precomputed polynomial weights
            nu_params = ntuple(d -> nonuniform_b_params(kernels_tuple[d]), N)
            M_eqs = ntuple(d -> nu_params[d][1], N)
            p_deg = ntuple(d -> nu_params[d][2], N)
            h = ntuple(d -> one(T), N) # dummy
            it = ntuple(_ -> ConvolutionMethod(), N)
            knots_new = ntuple(d -> collect(T, knots_tuple[d]), N)
            lazy = false # b kernels always use eager mode

            # Precompute weight coefficients for each dimension
            nb_data = ntuple(N) do d
                if uniform_dims[d]
                    precompute_uniform_b_all(knots_new[d], kernels_tuple[d], derivatives_tuple[d])
                else
                    precompute_nonuniform_b_all(knots_new[d], kernels_tuple[d], derivatives_tuple[d])
                end
            end

            knots_expanded = ntuple(d -> nb_data[d][2], N)
            nb_wc = ntuple(d -> nb_data[d][1], N)

            # Build coefficient array with ghost points
            coefs = create_nonuniform_b_coefs_perdim(vs, knots_new, kernels_tuple)

            kernel_type = ntuple(d -> nothing, N) # not used, uses nb_wc
            dimension = N <= 3 ? Val(N) : HigherDimension(Val(N))
            anchor = ntuple(d -> zero(T), N)
            do_type = DerivativeOrder(Val(derivatives_tuple)) # nonuniform integrals not supported
            kernel_d1_pre, kernel_d2_pre, subgrid = (nothing, nothing, (:not_used))
            kernel_sym = NonUniformMixedOrderKernel(Val(kernels_tuple))
            n_integral = 0
            integral_dims = ntuple(d -> false, N)

            return ConvolutionInterpolation{T,N,typeof(coefs),typeof(it),typeof(knots_expanded),typeof(kernel_type),
                                    typeof(dimension),typeof(kernel_sym),typeof(M_eqs),typeof(bc),
                                    typeof(do_type),typeof(kernel_d1_pre),typeof(kernel_d2_pre),
                                    typeof(Val(subgrid)),typeof(nb_wc),typeof(Val(lazy)),typeof(n_integral)}(
                coefs, knots_expanded, it, h, kernel_type, dimension, kernel_sym, M_eqs, bc, do_type,
                kernel_d1_pre, kernel_d2_pre, Val(subgrid), nb_wc, Val(lazy), boundary_fallback,
                anchor, n_integral, integral_dims
            )
        
        elseif all_n3_kernels || all_a0_kernels || all_a1_kernels

            kernels_tuple = kernels_tuple[1] == :a3 ? ntuple(d -> :n3, N) : kernels_tuple
            eqs = (all_a0_kernels || all_a1_kernels) ? ntuple(d -> 1, N) : ntuple(d -> 2, N)
            h = ntuple(d -> one(T), N)
            it = ntuple(_ -> ConvolutionMethod(), N)
            knots_new = ntuple(d -> collect(T, knots_tuple[d]), N)
            coefs, knots_expanded = if lazy || all_a0_kernels || all_a1_kernels
                vs, ntuple(N) do d
                    k = knots_new[d]
                    h_first = k[2] - k[1]
                    h_last = k[end] - k[end-1]
                    vcat(k[1] - h_first, k, k[end] + h_last)
                end
            else
                create_nonuniform_coefs(vs, knots_new, degree=:n3) # also expands knots
            end

            kernel_type = ntuple(d -> nothing, N)
            dimension = N <= 3 ? Val(N) : HigherDimension(Val(N))
            anchor = ntuple(d -> zero(T), N)
            do_type = N == 1 ? DerivativeOrder(Val(derivatives_tuple[1])) : DerivativeOrder(Val(derivatives_tuple))
            kernel_d1_pre, kernel_d2_pre, subgrid = (nothing, nothing, (:not_used))
            kernels_tuple = kernels_tuple[1] # nonuniform perdim only supported for :b kernels
            lazy = kernels_tuple == :a0 || kernels_tuple == :a1 ? true : lazy
            nb_wc = Val(:nonuniform)
            kernel_sym = kernels_tuple == :n3 ? NonUniformNonMixedHighKernel(Val(kernels_tuple)) :
                                     NonUniformNonMixedLowKernel(Val(kernels_tuple))
            n_integral = 0
            integral_dims = ntuple(d -> false, N)

            return ConvolutionInterpolation{T,N,typeof(coefs),typeof(it),typeof(knots_expanded),typeof(kernel_type),
                                    typeof(dimension),typeof(kernel_sym),typeof(eqs),typeof(bc),
                                    typeof(do_type),typeof(kernel_d1_pre),typeof(kernel_d2_pre),
                                    typeof(Val(subgrid)),typeof(nb_wc),typeof(Val(lazy)),typeof(n_integral)}(
                coefs, knots_expanded, it, h, kernel_type, dimension, kernel_sym, eqs, bc, do_type,
                kernel_d1_pre, kernel_d2_pre, Val(subgrid), nb_wc, Val(lazy), boundary_fallback,
                anchor, n_integral, integral_dims
            )
        end
    end
    
end