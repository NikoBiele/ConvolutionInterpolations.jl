"""
    ConvolutionInterpolation(knots, vs::AbstractArray{T,N}; kwargs...) where {T,N}

Construct a convolution interpolation object with exact kernel weights for O(1) evaluation.
This is the lower-level constructor — most users should prefer `convolution_interpolation`,
which wraps this with extrapolation handling and the default choices of kernel and mode.

On a uniform grid every kernel, derivative and antiderivative order is available. On a grid with
a nonuniform axis the kernels must be all `b` kernels (whose weights are projected onto the
knots at construction), all `:n3` (or `:a3`, which evaluates as `:n3` there), all `:a0` or all
`:a1`; antiderivatives and lazy ghosts are not available there.

# Arguments
- `knots`: Vector or range (1D) or tuple of vectors/ranges (N-D) of grid coordinates.
- `vs`: Array of values at the grid points.

# Keyword Arguments
- `kernel::Symbol=:auto`: Convolution kernel to use (default is N-dependent to reflect tensor product cost).
  - `a`-series: `:a0` (nearest), `:a1` (linear), `:a3` (cubic), `:a4` (quartic), `:a5` (quintic), `:a7` (septic)
  - `b`-series: `:b5`, `:b7`, `:b9`, `:b11`, `:b13`
  - `:n3`: the cubic kernel of nonuniform grids (`:a3` on a uniform grid)
- `bc=nothing`: Boundary condition for the ghost values at the domain edges: `:detect`, `:poly`,
  `:linear` or `:quadratic`, as a Symbol, a `(left, right)` pair or one pair per dimension.
  `nothing` means `:detect` on a uniform grid and `:kernel` on a nonuniform grid, where the
  ghost values are the kernel's own extrapolation and `:kernel` is the only rule.
- `derivative::Union{Int,NTuple{N,Int}}=0`: Derivative order to evaluate, one value for all
  dimensions or one per dimension, up to 1 for the `a` kernels, 3 for `:b5` and 5 for
  `:b7`–`:b13`: on a uniform grid derivative `k` of a `b` kernel converges at order `7 - k`,
  and orders above 5 are excluded as sub-second-order (see `convolution_interpolation` and
  the README kernel tables). On a nonuniform axis a `b` kernel of degree `p` supports
  derivatives up to `p - 2`. `derivative=-m` evaluates the m-fold antiderivative, anchored at
  zero at the leftmost interior knot, up to order 2 for `:a0`/`:a1`, 4 for the other
  `a`-series kernels, 6 for `:b5` and 8 for `:b7`–`:b13`. Orders may be mixed per dimension,
  e.g. `(-2, 0)` or `(-1, 1)`.
- `lazy::Union{Nothing,Bool}=nothing`: When `true`, skip ghost point expansion at construction
  time. The raw values are stored directly and ghost points are computed on the fly during
  evaluation near boundaries, with exactly the values eager expansion would store. When `false`,
  all ghost points are expanded at construction. The default, `nothing`, chooses lazy mode when
  eager expansion would build more than 2²⁷ coefficients (1 GiB in `Float64`) or compute more
  than 2²⁴ ghost points (roughly a second of construction) and there are no antiderivatives, and
  eager mode otherwise. Automatically disabled for `:a0` and `:a1`, and not available on
  nonuniform grids.
- `fast::Bool=true`: Whether antiderivatives store their region tails at construction (the
  left-of-stencil sums of every region, a few arrays the size of the coefficients per integral
  dimension), which makes every evaluation independent of the grid size. With `fast = false`
  nothing beyond the coefficients is stored and each antiderivative evaluation sums over the
  coefficients up to the stencil instead: O(data) per query, no extra memory. Interpolation
  and derivatives are identical either way.

# Returns
A `ConvolutionInterpolation` object callable at arbitrary points within the grid domain.
Does not handle extrapolation — use `convolution_interpolation` or wrap in
`ConvolutionExtrapolation` for that.

See also: [`convolution_interpolation`](@ref), [`ConvolutionInterpolation`](@ref), [`ConvolutionExtrapolation`](@ref).
"""
function ConvolutionInterpolation(knots::Union{AbstractVector,NTuple{N,AbstractVector}},
                                      vs::AbstractArray{T,N};
                                      kernel::Union{Symbol,NTuple{N,Symbol}}=:auto,
                                      bc::Union{Nothing,Symbol,Tuple{Symbol,Symbol},NTuple{N,Tuple{Symbol,Symbol}}}=nothing,
                                      derivative::Union{Int,NTuple{N,Int}}=0,
                                      lazy::Union{Nothing,Bool}=nothing,
                                      fast::Bool=true) where {T,N}

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

    # the grid picks the cubic implementation: :n3 is :a3 on a uniform grid, :a3 is :n3 on a
    # nonuniform one
    uniform_dims = ntuple(d -> is_uniform_grid(knots_tuple[d]), N)
    is_nonuniform = !all(uniform_dims)
    if is_nonuniform
        kernels_tuple = ntuple(d -> kernels_tuple[d] === :a3 ? :n3 : kernels_tuple[d], N)
    else
        kernels_tuple = n3_to_a3_on_uniform(kernels_tuple, knots_tuple)
    end

    # the boundary condition: `nothing` lets the grid decide — `:detect` on a uniform grid, and
    # on a nonuniform grid `:kernel`, the kernel's own ghost extrapolation (the Vandermonde ghosts
    # of the b-kernels, the three-knot rule of :n3), which is the only rule there
    bc = bc === nothing ? (is_nonuniform ? :kernel : :detect) : bc
    bcs_tuple = bc isa NTuple{N,Tuple{Symbol,Symbol}} ? bc :
                    bc isa Tuple{Symbol,Symbol} ? ntuple(_ -> bc, N) :
                    bc isa NTuple{1,Tuple{Symbol,Symbol}} ? ntuple(_ -> bc[1], N) :
                    bc isa Symbol ? ntuple(_ -> (bc, bc), N) :
                    error("Invalid bc specification: $bc.")
    if is_nonuniform && any(d -> bcs_tuple[d] != (:kernel, :kernel), 1:N)
        throw(ArgumentError("bc = $bc is not supported on nonuniform grids: the ghost values are " *
                            "the kernel's own extrapolation there (bc = :kernel). Leave bc at its default."))
    elseif !is_nonuniform && any(d -> :kernel in bcs_tuple[d], 1:N)
        throw(ArgumentError("bc = :kernel is the ghost rule of nonuniform grids; on a uniform grid " *
                            "use :detect, :poly, :quadratic or :linear."))
    end

    if is_nonuniform
        all_b_kernels = all(k -> k in (:b5, :b7, :b9, :b11, :b13), kernels_tuple)
        all_closed_form = allequal(kernels_tuple) && kernels_tuple[1] in (:n3, :a0, :a1)
        if !all_b_kernels && !all_closed_form
            error("Nonuniform interpolation requires all kernels in all dimensions to be either:\n" *
                  "          - all :b kernels (:b5, :b7, :b9, :b11, :b13),\n" *
                  "          - all :n3 kernels,\n" *
                  "          - all :a0 kernels or\n" *
                  "          - all :a1 kernels.\n" *
                  "Got: $kernel.")
        end
        if any(d -> derivatives_tuple[d] < 0, 1:N)
            error("Antiderivatives (derivative < 0) are not supported on nonuniform grids.")
        end
        if lazy === true
            error("Lazy mode is not supported on nonuniform grids: their ghost values are " *
                  "computed at construction.")
        end
        # derivatives: up to each kernel's highest order on the grid of its own axis
        for d in 1:N
            derivatives_tuple[d] > 0 || continue
            k = kernels_tuple[d]
            table = uniform_dims[d] ? _max_derivative_uniform : _max_derivative_nonuniform
            haskey(table, k) || continue
            max_order = _max_derivative_order(k, !uniform_dims[d])
            grid = uniform_dims[d] ? "uniform" : "nonuniform"
            derivatives_tuple[d] <= max_order ||
                error("Kernel :$k supports derivatives up to order $max_order on a $grid grid " *
                      "(dimension $d). Got derivative = $(derivatives_tuple[d]).")
        end
        return _build_nonuniform_convolution(knots_tuple, vs, bcs_tuple, uniform_dims,
                                                  Val(kernels_tuple), Val(derivatives_tuple))
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

    # derivatives: up to each kernel's highest order on a uniform grid (second-order convergence
    # or better, and a continuous derivative), in every dimension
    for d in 1:N
        derivatives_tuple[d] > 0 || continue
        max_order = _max_derivative_order(kernels_tuple[d], false)
        derivatives_tuple[d] <= max_order ||
            error("Kernel :$(kernels_tuple[d]) supports derivatives up to order $max_order on a " *
                  "uniform grid (derivative k converges at order 7 - k for the b kernels, so " *
                  "higher orders are not meaningful there). Got derivative = $(derivatives_tuple[d]).")
    end

    return _build_uniform_convolution(knots_tuple, vs, bcs_tuple,
                                           Val(kernels_tuple), Val(lazy),
                                           Val(derivatives_tuple), Val(fast))
end

function _build_uniform_convolution(knots::NTuple{N,AbstractVector},
                                         vs::AbstractArray{T,N},
                                         bc::BCT,
                                         ::Val{KS},
                                         ::Val{LZ},
                                         ::Val{DV},
                                         ::Val{FAST}=Val(true)) where {T,N,BCT<:Tuple,KS,LZ,DV,FAST}

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

    do_type = _build_do_type(Val{DV}())

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
        # region tails only when asked for (fast = true) and for at most 3 integral dimensions;
        # without them every antiderivative evaluation sums over the coefficients directly
        if FAST && count(<(0), derivative) <= 3
            integral_tails = _build_region_tails(coefs, kernel, derivative, eqs)
        else
            integral_tails = Vector{Array{T,N}}[]
        end
    else
        integral_taylor = ntuple(_ -> Matrix{T}(undef, 0, 0), N)
        integral_entries = ntuple(_ -> Matrix{T}(undef, 0, 0), N)
        integral_tails = Vector{Array{T,N}}[]
    end

    # every axis is uniform: no nonuniform data
    nonuniform = ntuple(_ -> nothing, N)

    return ConvolutionInterpolation{T,N,n_integral,typeof(coefs),typeof(knots_new),
                                        typeof(kernel_type),typeof(dimension),typeof(kernels),
                                        typeof(eqs),
                                        typeof(bc),typeof(do_type),
                                        typeof(Val{LZ}()),typeof(integral_dimension),typeof(domain_size),
                                        typeof(nonuniform)}(
        coefs, domain_size, knots_new, h, x0, kernel_type, dimension, kernels, eqs,
        bc, do_type, Val{LZ}(), anchor, integral_dimension,
        integral_taylor, integral_entries, integral_tails, nonuniform,
    )
end

# Interpolant on a grid with nonuniform axes (no antiderivatives, no lazy ghosts). The kernels
# are all b kernels (uniform axes among them get the b tables of their own spacing, so one
# ghost rule holds on every axis), all :n3, all :a0 or all :a1; the keyword constructor has
# checked this. Each axis stores what `_axis_stencil` needs in `nonuniform`.
function _build_nonuniform_convolution(knots::NTuple{N,AbstractVector},
                                            vs::AbstractArray{T,N},
                                            bc::BCT,
                                            uniform_dims::NTuple{N,Bool},
                                            ::Val{KS},
                                            ::Val{DV}) where {T,N,BCT<:Tuple,KS,DV}
    kernel = KS
    derivative = DV
    domain_size = ntuple(d -> size(vs, d), N)
    knots_t = ntuple(d -> eltype(knots[d]) === T ? knots[d] : collect(T, knots[d]), N)
    h = ntuple(_ -> one(T), N)                            # derivatives scale by the local spacing

    if all(k -> k in (:b5, :b7, :b9, :b11, :b13), kernel)
        # per-interval weight tables and M_eqs ghost knots per side on every axis
        nb = ntuple(N) do d
            if uniform_dims[d]
                precompute_uniform_b_all(knots_t[d], kernel[d], derivative[d])
            else
                precompute_nonuniform_b_all(knots_t[d], kernel[d], derivative[d])
            end
        end
        nonuniform = ntuple(d -> nb[d][1], N)
        knots_new = ntuple(d -> nb[d][2], N)
        eqs = ntuple(d -> nonuniform_b_params(kernel[d])[1], N)
        coefs = create_nonuniform_b_coefs_perdim(vs, knots_t, kernel)
    elseif kernel[1] === :n3
        # closed-form cubic weights from the local spacing, one ghost per side
        nonuniform = ntuple(_ -> Val(:nonuniform), N)
        eqs = ntuple(_ -> 2, N)
        coefs, knots_new = create_nonuniform_coefs(vs, knots_t, degree = :n3)
    else
        # :a0 / :a1 read the raw values; the knots get one ghost per side for the cell search
        nonuniform = ntuple(_ -> Val(:nonuniform), N)
        eqs = ntuple(_ -> 1, N)
        coefs = vs
        knots_new = ntuple(N) do d
            k = knots_t[d]
            vcat(k[1] - (k[2] - k[1]), k, k[end] + (k[end] - k[end-1]))
        end
    end

    x0 = ntuple(d -> T(first(knots_new[d])), N)
    anchor = ntuple(_ -> zero(T), N)
    kernel_type = ntuple(_ -> nothing, N)
    dimension = N <= 3 ? Val(N) : HigherDimension(Val(N))
    integral_dimension = Val(0)
    do_type = _build_do_type(Val{DV}())
    kernels = NonUniformMixedOrderKernel(Val(KS))
    integral_taylor = ntuple(_ -> Matrix{T}(undef, 0, 0), N)
    integral_entries = ntuple(_ -> Matrix{T}(undef, 0, 0), N)
    integral_tails = Vector{Array{T,N}}[]

    return ConvolutionInterpolation{T,N,0,typeof(coefs),typeof(knots_new),
                                        typeof(kernel_type),typeof(dimension),typeof(kernels),
                                        typeof(eqs),
                                        typeof(bc),typeof(do_type),
                                        typeof(Val(false)),typeof(integral_dimension),typeof(domain_size),
                                        typeof(nonuniform)}(
        coefs, domain_size, knots_new, h, x0, kernel_type, dimension, kernels, eqs,
        bc, do_type, Val(false), anchor, integral_dimension,
        integral_taylor, integral_entries, integral_tails, nonuniform,
    )
end

@generated function _build_do_type(::Val{D}) where D
    if any(<(0), D)
        # every integral: any orders, possibly mixed with interpolation or derivatives
        return :(IntegralOrders{$D}())
    else
        return :(DerivativeOrder(Val($D)))
    end
end

# Number of integral dimensions (negative derivative orders), as a compile-time constant
@generated function _count_integrals(::Val{D}) where D
    return :($(count(<(0), D)))
end

# Lazy ghosts: the raw values and knots are stored as they are
function _build_lazy_coefs(knots::NTuple{N,AbstractVector}, vs::AbstractArray{T,N}) where {T,N}
    return vs, knots
end

# Eager ghosts: the knots extended by eqs − 1 per side and the coefficients with the ghost
# values of the boundary rule `bc` (see `create_convolutional_coefs`)
function _build_eager_coefs(knots::NTuple{N,AbstractVector}, vs::AbstractArray{T,N},
                            eqs, bc, kernel, h, ::Val{UG}) where {T,N,UG}
    knots_new = expand_knots(knots, ntuple(d -> eqs[d]-1, N))
    coefs = create_convolutional_coefs(vs, h, eqs, bc, kernel, Val(UG))
    return coefs, knots_new
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