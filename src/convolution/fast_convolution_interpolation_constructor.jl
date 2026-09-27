"""
    FastConvolutionInterpolation(knots, vs::AbstractArray{T,N}; kwargs...) where {T,N}

Construct a fast convolution interpolation object with exact kernel weights for O(1) evaluation.
This is the lower-level fast constructor — most users should prefer `convolution_interpolation`,
which wraps this with extrapolation handling and automatic mode selection.

Only supports uniform grids. For nonuniform grids, use `ConvolutionInterpolation` directly.

# Arguments
- `knots`: Vector or range (1D) or tuple of vectors/ranges (N-D) of grid coordinates.
- `vs`: Array of values at the grid points.

# Keyword Arguments
- `kernel::Symbol=:auto`: Convolution kernel to use (default is N-dependent to reflect tensor product cost).
  - `a`-series: `:a0` (nearest), `:a1` (linear), `:a3` (cubic), `:a4` (quartic), `:a5` (quintic), `:a7` (septic)
  - `b`-series: `:b5`, `:b7`, `:b9`, `:b11`, `:b13`
- `precompute`: Deprecated, has no effect, and will be removed in a future release. Kernels
  are evaluated exactly, so no kernel tables are precomputed.
- `B::Float64=-1.0`: If a positive value is provided, uses Gaussian kernel with parameter `B` for C∞ smoothness.
  Forces eager mode.
- `bc=:detect`: Boundary condition for kernel evaluation at domain edges.
  Options: `:detect`, `:poly`, `:linear`, `:quadratic`.
- `derivative::Union{Int,NTuple{N,Int}}=0`: Derivative order to evaluate, one value for all
  dimensions or one per dimension. Supported up to 6 for `b`-series kernels. `derivative=-m`
  evaluates the m-fold antiderivative, anchored at zero at the leftmost interior knot, up to
  order 2 for `:a0`/`:a1`, 4 for the other `a`-series kernels, 6 for `:b5` and 8 for
  `:b7`–`:b13`. Orders may be mixed per dimension, e.g. `(-2, 0)` or `(-1, 1)`.
  - `subgrid`: Deprecated, has no effect, and will be removed in a future release. Kernels
  are evaluated exactly, so there is no subgrid interpolation.
- `lazy::Union{Nothing,Bool}=nothing`: When `true`, skip ghost point expansion at construction
  time. The raw values are stored directly and ghost points are computed on the fly during
  evaluation near boundaries, with exactly the values eager expansion would store. When `false`,
  all ghost points are expanded at construction. The default, `nothing`, chooses lazy mode when
  eager expansion would build more than 2²⁷ coefficients (1 GiB in `Float64`) or compute more
  than 2²⁴ ghost points (roughly a second of construction) and there are no antiderivatives, and
  eager mode otherwise. Automatically disabled for `:a0`, `:a1`, and Gaussian kernels.
- `boundary_fallback`: Deprecated, has no effect; will be removed in v1.0. Lazy interpolants
  compute the same boundary ghost values as eager ones, in every dimension.

# Returns
A `FastConvolutionInterpolation` object callable at arbitrary points within the grid domain.
Does not handle extrapolation — use `convolution_interpolation` or wrap in
`ConvolutionExtrapolation` for that.

See also: [`convolution_interpolation`](@ref), [`ConvolutionInterpolation`](@ref), [`ConvolutionExtrapolation`](@ref).
"""

function FastConvolutionInterpolation(knots::Union{AbstractVector,NTuple{N,AbstractVector}},
                                      vs::AbstractArray{T,N};
                                      kernel::Union{Symbol,NTuple{N,Symbol}}=:auto,
                                      precompute=nothing,
                                      bc::Union{Symbol,Tuple{Symbol,Symbol},NTuple{N,Tuple{Symbol,Symbol}}}=:detect,
                                      derivative::Union{Int,NTuple{N,Int}}=0,
                                      subgrid=nothing,
                                      lazy::Union{Nothing,Bool}=nothing, boundary_fallback=nothing) where {T,N}

    # deprecated keywords: warn if set, then ignore
    _warn_deprecated_table_keywords(precompute, subgrid)
    _warn_deprecated_boundary_fallback(boundary_fallback)

    # check and normalize inputs
    knots_tuple = knots isa AbstractVector ?
                    (eltype(knots) === T ? knots : T.(knots),) :
                    knots isa NTuple{N,AbstractVector} ?
                    ntuple(d -> eltype(knots[d]) === T ? knots[d] : T.(knots[d]), N) :
                    error("Invalid knots specification: $knots.")
    kernel = kernel === :auto ? _default_kernel(N) : kernel
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
                    
    if any(==(:n3), kernels_tuple)
        error("The :n3 kernel is not supported by FastConvolutionInterpolation.")
    end
    if !all(d -> is_uniform_grid(knots_tuple[d]), 1:N)
        error("FastConvolutionInterpolation requires a uniform grid in every dimension.\n" *
              "The fast kernel weights assume uniform spacing and cannot adapt \n" *
              "to nonuniform spacing.\n" *
              "For nonuniform knots use either:\n" *
              "       - convolution_interpolation(knots, values; ...), which selects the " *
              "correct path automatically, or\n" *
              "       - ConvolutionInterpolation(knots, values; ...), the slow constructor, " *
              "whose kernels adjust their weights to nonuniform spacing.")
    end

    # lazy by default when eager expansion would build a large coefficient array (no
    # antiderivatives; see `_default_lazy`)
    lazy = lazy === nothing ? _default_lazy(size(vs), kernels_tuple, derivatives_tuple) : lazy
    
    # lazy mode deliberately skips, so the two cannot be combined
    if lazy && any(d -> derivatives_tuple[d] < 0, 1:N)
        error("Antiderivatives (derivative < 0) are not supported in lazy mode.")
    end

    # antiderivatives of order 2 and higher (derivative = -M, M ≥ 2): up to each kernel's highest
    # integral order, in every dimension
    for d in 1:N
        derivatives_tuple[d] < -1 || continue
        M = -derivatives_tuple[d]
        max_order = _max_integral_order[kernels_tuple[d]]
        M <= max_order || error("Kernel :$(kernels_tuple[d]) supports antiderivatives up to order " *
                                "$max_order (derivative = -$max_order). Got derivative = -$M.")
    end

    return _build_fast_uniform_convolution(knots_tuple, vs, bcs_tuple,
                                           Val(kernels_tuple), Val(lazy),
                                           Val(derivatives_tuple))
end

function _build_fast_uniform_convolution(knots::NTuple{N,AbstractVector},
                                         vs::AbstractArray{T,N},
                                         bc::BCT,
                                         ::Val{KS},
                                         ::Val{LZ},
                                         ::Val{DV}) where {T,N,BCT<:Tuple,KS,LZ,DV}

    kernel = KS
    derivative = DV

    eqs = ntuple(d -> get_equations_for_degree(kernel[d]), N)
    n_integral = _count_integrals(Val{DV}())

    h = ntuple(d -> (last(knots[d]) - first(knots[d]))/(length(knots[d]) - 1), N)
    
    all_kernels_low_order = all(d -> kernel[d] == :a0 || kernel[d] == :a1, 1:N)
    coefs, knots_new = if LZ || all_kernels_low_order
        _build_lazy_coefs(knots, vs)
    else
        _build_eager_coefs(knots, vs, eqs, bc, kernel, h, Val(false))
    end
    x0 = ntuple(d -> T(first(knots_new[d])), N)

    anchor = ntuple(d -> derivative[d] < 0 ? knots_new[d][eqs[d]] : zero(T), N)

    kernel_type = ntuple(d -> nothing, N)
    dimension = N <= 3 ? Val(N) : HigherDimension(Val(N))
    integral_dimension = n_integral <= 3 ? Val(n_integral) : HigherDimension(Val(n_integral))

    do_type = _build_fast_do_type(Val{DV}())

    kernels = _build_kernel_sym(Val{KS}(), Val{DV}())

    domain_size = ntuple(d -> size(vs, d), N)

    # integrals of any order in any dimension: exact anchoring and entry tables per integral
    # dimension, and the left tails of every region (tails only for at most 3 integral dimensions)
    if any(<(0), derivative)
        integral_taylor = ntuple(d -> derivative[d] < 0 ?
                                 _anchor_taylor_table(T, kernel[d], -derivative[d], eqs[d]) :
                                 Matrix{T}(undef, 0, 0), N)
        integral_entries = ntuple(d -> derivative[d] < 0 ?
                                  _near_anchor_entries(T, kernel[d], -derivative[d], eqs[d]) :
                                  Matrix{T}(undef, 0, 0), N)
        integral_tails = count(<(0), derivative) <= 3 ?
                         _build_region_tails(coefs, kernel, derivative, eqs) : Vector{Array{T,N}}[]
    else
        integral_taylor = ntuple(_ -> Matrix{T}(undef, 0, 0), N)
        integral_entries = ntuple(_ -> Matrix{T}(undef, 0, 0), N)
        integral_tails = Vector{Array{T,N}}[]
    end

    return FastConvolutionInterpolation{T,N,n_integral,typeof(coefs),typeof(knots_new),
                                        typeof(kernel_type),typeof(dimension),typeof(kernels),
                                        typeof(eqs),
                                        typeof(bc),typeof(do_type),
                                        Nothing,Nothing,Val{:not_used},
                                        typeof(Val{LZ}()),typeof(integral_dimension),typeof(domain_size)}(
        coefs, domain_size, knots_new, h, x0, kernel_type, dimension, kernels, eqs,
        bc, do_type, nothing, nothing, Val(:not_used),
        Val{LZ}(), anchor, integral_dimension,
        integral_taylor, integral_entries, integral_tails,
    )
end

@generated function _build_fast_do_type(::Val{D}) where D
    if any(<(0), D)
        # every integral: any orders, possibly mixed with interpolation or derivatives
        return :(FastIntegralOrders{$D}())
    else
        return :(DerivativeOrder(Val($D)))
    end
end

@generated function _build_kernel_sym(::Val{KS}, ::Val{DV}) where {KS, DV}
    N = length(KS)
    high = any(k -> k in (:a3,:a4,:a5,:a7,:b5,:b7,:b9,:b11,:b13), KS)
    low  = any(k -> k in (:a0,:a1), KS)
    keq  = allequal(KS)
    deq  = allequal(DV)
    T = if deq
        if high && low;        :(FullMixedOrderKernel(Val{$KS}()))
        elseif high && keq;    :(HigherOrderKernel(Val{$KS}()))
        elseif high;           :(HigherOrderMixedKernel(Val{$KS}()))
        elseif low && keq;     :(LowerOrderKernel(Val{$KS}()))
        else;                  :(LowerOrderMixedKernel(Val{$KS}()))
        end
    else
        if high && low;        :(FullMixedOrderKernel(Val{$KS}()))
        elseif high;           :(HigherOrderMixedKernel(Val{$KS}()))
        else;                  :(LowerOrderMixedKernel(Val{$KS}()))
        end
    end
    return T
end