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
- `bc=:detect`: Boundary condition for kernel evaluation at domain edges.
  Options: `:detect`, `:poly`, `:linear`, `:quadratic`.
- `derivative::Int=0`: Derivative order to evaluate. Supported up to 6 for `b`-series
  kernels. For `a`-series kernels the top derivative automatically uses linear interpolation
  to match the kernel's continuity class.
- `lazy::Bool=false`: When `true`, skip ghost point expansion at construction time.
  The raw values are stored directly and ghost points are computed on the fly during
  evaluation near boundaries. This saves memory and speeds up construction, especially
  in high dimensions. Automatically disabled for `:a0`, `:a1`, Gaussian kernels, and
  nonuniform `b`-series paths.
- `boundary_fallback::Bool=false`: When `true`, near-boundary evaluations use a linear
  kernel rather than computing full ghost point stencils — correct throughout the domain
  at the cost of reduced smoothness in the boundary stencil region. Derivatives are not
  supported in this mode. Required for `N≥4`; only active when `lazy=true`.
  
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

function ConvolutionInterpolation(knots::Union{AbstractVector,NTuple{N,AbstractVector}},
                                  vs::AbstractArray{T,N};
                                  kernel::Union{Symbol,NTuple{N,Symbol}}=:b5,
                                  bc::Union{Symbol,Tuple{Symbol,Symbol},NTuple{N,Tuple{Symbol,Symbol}}}=:detect,
                                  derivative::Union{Int,NTuple{N,Int}}=0,
                                  lazy::Bool=false, boundary_fallback::Bool=false) where {T,N}

    # check and normalize inputs
    knots_tuple = knots isa AbstractVector ? (T.(knots),) : 
                  knots isa NTuple{N,AbstractVector} ? ntuple(d -> T.(knots[d]), N) :
                    error("Invalid knots specification: $knots.")
    kernels_tuple = kernel isa NTuple{N,Symbol} ? kernel :
                    kernel isa Symbol ? ntuple(_ -> kernel, N) :
                    kernel isa NTuple{1,Symbol} ? ntuple(_ -> kernel[1], N) :
                    error("Invalid kernel specification: $kernel.")
    derivatives_tuple = derivative isa NTuple{N,Int} ? derivative :
                    derivative isa Int ? ntuple(_ -> derivative, N) :
                    derivative isa NTuple{1,Int} ? ntuple(_ -> derivative[1], N) : 
                    error("Invalid derivative specification: $derivative.")
    bcs_tuple = bc isa NTuple{N,Tuple{Symbol,Symbol}} ? bc :
                    bc isa Tuple{Symbol,Symbol} ? ntuple(_ -> bc, N) :
                    bc isa NTuple{1,Tuple{Symbol,Symbol}} ? ntuple(_ -> bc[1], N) :
                    bc isa Symbol ? ntuple(_ -> (bc, bc), N) :
                    error("Invalid bc specification: $bc.")

    if lazy && N >= 4 && !boundary_fallback
        error("Lazy mode requires 'boundary_fallback=true' for dimensions >= 4.")
    end
    if lazy && boundary_fallback && any(d -> derivatives_tuple[d] != 0, 1:N)
        error("Derivatives not supported in lazy mode with 'boundary_fallback=true'.")
    end
    if lazy && !boundary_fallback && !allequal(kernels_tuple)
        error("Lazy non-fast mode with 'boundary_fallback=false' requires the same kernel for all dimensions, but got $(kernel).")
    end

    uniform_dims = ntuple(d -> is_uniform_grid(knots_tuple[d]), N)

    if all(uniform_dims)
        return _build_uniform_convolution(knots_tuple, vs, bcs_tuple,
                boundary_fallback, Val(kernel), Val(lazy), Val(derivatives_tuple))
    else
        all_b_kernels = all(d -> kernels_tuple[d] in (:b5, :b7, :b9, :b11, :b13), 1:N)
        all_n3_kernels = allequal(kernels_tuple) && (kernels_tuple[1] == :n3 || kernels_tuple[1] == :a3)
        all_a0_kernels = all(d -> kernels_tuple[d] == :a0, 1:N)
        all_a1_kernels = all(d -> kernels_tuple[d] == :a1, 1:N)

        if !all_b_kernels && !all_n3_kernels && !all_a0_kernels && !all_a1_kernels
            error("Nonuniform interpolation requires all kernels in all dimensions to be either:\n" *
                  "          - all :b kernels (:b5, :b7, :b9, :b11, :b13),\n" *
                  "          - all :n3 kernels,\n" *
                  "          - all :a0 kernels or\n" *
                  "          - all :a1 kernels.\n" *
                  "Got: $kernel.")
        end
        if any(d -> derivative[d] == -1, 1:N)
            error("Nonuniform antiderivative (derivative=-1) is not directly supported.\n" *
                "Recommended workflow:\n" *
                "       (1) construct a nonuniform derivative=0 interpolation,\n" *
                "       (2) resample to a uniform grid,\n" *
                "       (3) construct a uniform derivative=-1 interpolation.")
        end

        if all_b_kernels
            return _build_nonuniform_b_convolution(knots_tuple, vs, bcs_tuple, boundary_fallback, uniform_dims,
                                    Val(kernels_tuple), Val(derivatives_tuple))
        else
            return _build_nonuniform_n3_convolution(knots_tuple, vs, bcs_tuple, boundary_fallback,
                Val(kernels_tuple), Val(lazy), Val(derivatives_tuple), Val(all_a0_kernels), Val(all_a1_kernels))
        end
    end
end

function _build_uniform_convolution(knots::Tuple{Vararg{AbstractVector}},
                                    vs::AbstractArray{T,N},
                                    bc::BCT,
                                    boundary_fallback::Bool,
                                    ::Val{KS},
                                    ::Val{LZ},
                                    ::Val{DV}) where {T,N,BCT<:Tuple,KS,LZ,DV}
    kernel = KS
    derivative = DV

    domain_size = ntuple(d -> size(vs, d), N)

    eqs = ntuple(d -> get_equations_for_degree(kernel[d]), N)
    h = map(k -> k[2] - k[1], knots)

    all_kernels_low_order = all(d -> kernel[d] == :a0 || kernel[d] == :a1, 1:N)
    coefs, knots_new = if LZ || all_kernels_low_order
        _build_lazy_coefs(knots, vs)
    else
        _build_eager_coefs(knots, vs, eqs, bc, kernel, h, Val(false))
    end

    if N == 1
        if derivative[1] == -1
            anchor = (knots_new[1][eqs[1]],)
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

    kernel_type = _build_convolution_kernel_type(Val{KS}(), Val{DV}())
    dimension = N <= 3 ? Val(N) : HigherDimension(Val(N))
    do_type = _build_do_type(Val{DV}())
    n_integral = _count_integrals(Val{DV}())

    kernel_d1_pre, kernel_d2_pre, subgrid = (nothing, nothing, (:not_used))
    nb_wc = nothing
    integral_dimension = n_integral <= 3 ? Val(n_integral) : HigherDimension(Val(n_integral))

    return ConvolutionInterpolation{T,N,n_integral,typeof(coefs),typeof(knots_new),typeof(kernel_type),
                            typeof(dimension),typeof(Val{KS}()),typeof(eqs),typeof(bc),
                            typeof(do_type),typeof(kernel_d1_pre),typeof(kernel_d2_pre),
                            typeof(Val(subgrid)),typeof(nb_wc),typeof(Val{LZ}()),
                            typeof(integral_dimension),typeof(domain_size)}(
        coefs, domain_size, knots_new, h, kernel_type, dimension, Val{KS}(), eqs, bc, do_type,
        kernel_d1_pre, kernel_d2_pre, Val(subgrid), nb_wc, Val{LZ}(), boundary_fallback,
        anchor, integral_dimension
    )
end

function _build_nonuniform_b_convolution(knots::Tuple{Vararg{AbstractVector}},
                                         vs::AbstractArray{T,N},
                                         bc::BCT,
                                         boundary_fallback::Bool,
                                         uniform_dims::NTuple{N,Bool},
                                         ::Val{KS},
                                         ::Val{DV}) where {T,N,BCT<:Tuple,KS,DV}
    kernel = KS
    derivative = DV
    domain_size = ntuple(d -> size(vs, d), N)
    nu_params = ntuple(d -> nonuniform_b_params(kernel[d]), N)
    M_eqs = ntuple(d -> nu_params[d][1], N)
    h = ntuple(d -> one(T), N)
    knots_new = ntuple(d -> collect(T, knots[d]), N)

    nb_data = ntuple(N) do d
        if uniform_dims[d]
            precompute_uniform_b_all(knots_new[d], kernel[d], derivative[d])
        else
            precompute_nonuniform_b_all(knots_new[d], kernel[d], derivative[d])
        end
    end

    knots_expanded = ntuple(d -> nb_data[d][2], N)
    nb_wc = ntuple(d -> nb_data[d][1], N)
    coefs = create_nonuniform_b_coefs_perdim(vs, knots_new, kernel)
    kernel_type = ntuple(d -> nothing, N)
    dimension = N <= 3 ? Val(N) : HigherDimension(Val(N))
    anchor = ntuple(d -> zero(T), N)
    do_type = _build_do_type(Val{DV}())
    n_integral = _count_integrals(Val{DV}())
    kernel_d1_pre, kernel_d2_pre, subgrid = (nothing, nothing, (:not_used))
    kernel_sym = NonUniformMixedOrderKernel(Val(KS))
    integral_dimension = n_integral <= 3 ? Val(n_integral) : HigherDimension(Val(n_integral))

    return ConvolutionInterpolation{T,N,n_integral,typeof(coefs),typeof(knots_expanded),typeof(kernel_type),
                            typeof(dimension),typeof(kernel_sym),typeof(M_eqs),typeof(bc),
                            typeof(do_type),typeof(kernel_d1_pre),typeof(kernel_d2_pre),
                            typeof(Val(subgrid)),typeof(nb_wc),typeof(Val(false)),
                            typeof(integral_dimension),typeof(domain_size)}(
        coefs, domain_size, knots_expanded, h, kernel_type, dimension, kernel_sym, M_eqs, bc, do_type,
        kernel_d1_pre, kernel_d2_pre, Val(subgrid), nb_wc, Val(false), boundary_fallback,
        anchor, integral_dimension
    )
end

function _build_nonuniform_n3_convolution(knots::Tuple{Vararg{AbstractVector}},
                                          vs::AbstractArray{T,N},
                                          bc::BCT,
                                          boundary_fallback::Bool,
                                          ::Val{KS},
                                          ::Val{LZ},
                                          ::Val{DV},
                                          ::Val{A0},
                                          ::Val{A1}) where {T,N,BCT<:Tuple,KS,LZ,DV,A0,A1}
    lazy_final = A0 || A1 || LZ
    domain_size = ntuple(d -> size(vs, d), N)
    kernel = KS[1] == :a3 ? ntuple(d -> :n3, N) : KS
    eqs = (A0 || A1) ? ntuple(d -> 1, N) : ntuple(d -> 2, N)
    h = ntuple(d -> one(T), N)
    knots_new = ntuple(d -> collect(T, knots[d]), N)

    coefs, knots_expanded = if LZ || A0 || A1
        vs, ntuple(N) do d
            k = knots_new[d]
            h_first = k[2] - k[1]
            h_last = k[end] - k[end-1]
            vcat(k[1] - h_first, k, k[end] + h_last)
        end
    else
        create_nonuniform_coefs(vs, knots_new, degree=:n3)
    end

    kernel_type = ntuple(d -> nothing, N)
    dimension = N <= 3 ? Val(N) : HigherDimension(Val(N))
    anchor = ntuple(d -> zero(T), N)
    do_type = _build_do_type(Val{DV}())
    n_integral = _count_integrals(Val{DV}())
    kernel_d1_pre, kernel_d2_pre, subgrid = (nothing, nothing, (:not_used))
    kernel_scalar = kernel[1]
    nb_wc = Val(:nonuniform)
    kernel_sym = kernel_scalar == :n3 ? NonUniformNonMixedHighKernel(Val(kernel_scalar)) :
                                        NonUniformNonMixedLowKernel(Val(kernel_scalar))
    integral_dimension = n_integral <= 3 ? Val(n_integral) : HigherDimension(Val(n_integral))

    return ConvolutionInterpolation{T,N,n_integral,typeof(coefs),typeof(knots_expanded),typeof(kernel_type),
                            typeof(dimension),typeof(kernel_sym),typeof(eqs),typeof(bc),
                            typeof(do_type),typeof(kernel_d1_pre),typeof(kernel_d2_pre),
                            typeof(Val(subgrid)),typeof(nb_wc),typeof(Val(lazy_final)),
                            typeof(integral_dimension),typeof(domain_size)}(
        coefs, domain_size, knots_expanded, h, kernel_type, dimension, kernel_sym, eqs, bc, do_type,
        kernel_d1_pre, kernel_d2_pre, Val(subgrid), nb_wc, Val(lazy_final), boundary_fallback,
        anchor, integral_dimension
    )
end

function _build_lazy_coefs(knots::NTuple{N,AbstractVector}, vs::AbstractArray{T,N}) where {T,N}
    return vs, knots
end

function _build_eager_coefs(knots::NTuple{N,AbstractVector}, vs::AbstractArray{T,N},
                            eqs, bc, kernel, h, ::Val{UG}) where {T,N,UG}
    knots_new = expand_knots(knots, ntuple(d -> eqs[d]-1, N))
    coefs = create_convolutional_coefs(vs, h, eqs, bc, kernel, Val(UG))
    return coefs, knots_new
end

@generated function _build_do_type(::Val{D}) where D
    n_int = count(==(-1), D)
    N = length(D)
    if n_int == 0
        return :(DerivativeOrder(Val($D)))
    elseif n_int == N
        return :(IntegralOrder())
    else
        return :(MixedIntegralOrder{$D}())
    end
end

@generated function _count_integrals(::Val{D}) where D
    return :($(count(==(-1), D)))
end

@generated function _build_convolution_kernel_type(::Val{KS}, ::Val{DV}) where {KS, DV}
    N = length(KS)
    exprs = [:(ConvolutionKernel(Val($(QuoteNode(KS[d]))), Val($(DV[d])))) for d in 1:N]
    return :(tuple($(exprs...)))
end