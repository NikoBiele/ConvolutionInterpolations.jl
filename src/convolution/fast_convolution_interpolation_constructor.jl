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
- `derivative::Int=0`: Derivative order to evaluate. Supported up to 6 for `b`-series
  kernels. Negative values evaluate antiderivatives.
- `subgrid`: Deprecated, has no effect, and will be removed in a future release. Kernels
  are evaluated exactly, so there is no subgrid interpolation.
- `lazy::Bool=false`: When `true`, skip ghost point expansion at construction time.
  The raw values are stored directly and ghost points are computed on the fly during
  evaluation near boundaries. Interior evaluation has zero overhead compared to eager mode.
  Automatically disabled for `:a0`, `:a1`, and Gaussian kernels.
- `boundary_fallback::Bool=false`: When `true`, near-boundary evaluations use a linear
  kernel rather than computing full ghost point stencils — correct throughout the domain
  at the cost of reduced smoothness in the boundary stencil region. Derivatives are not
  supported in this mode. Required for `N≥4`; only active when `lazy=true`.

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
                                      lazy::Bool=false, boundary_fallback::Bool=false) where {T,N}

    # deprecated keywords: warn if set, then ignore
    _warn_deprecated_table_keywords(precompute, subgrid)

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
    if lazy && N >= 4 && !boundary_fallback
        error("Lazy mode requires 'boundary_fallback=true' for dimensions >= 4.")
    end
    if lazy && boundary_fallback && any(d -> derivatives_tuple[d] != 0, 1:N)
        error("Derivatives not supported in lazy mode with 'boundary_fallback=true'.")
    end

    return _build_fast_uniform_convolution(knots_tuple, vs, bcs_tuple, boundary_fallback,
                                           Val(kernels_tuple), Val(lazy),
                                           Val(derivatives_tuple))
end

function _build_fast_uniform_convolution(knots::NTuple{N,AbstractVector},
                                         vs::AbstractArray{T,N},
                                         bc::BCT,
                                         boundary_fallback::Bool,
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

    anchor = ntuple(d -> derivative[d] == -1 ? knots_new[d][eqs[d]] : zero(T), N)

    left_values, tail1_left, tail2_ll, tail3_edge_ll, tail3_corner_lll =
                                    _build_tails(coefs, kernel, Val{DV}(), eqs, knots_new)

    kernel_type = ntuple(d -> nothing, N)
    dimension = N <= 3 ? Val(N) : HigherDimension(Val(N))
    integral_dimension = n_integral <= 3 ? Val(n_integral) : HigherDimension(Val(n_integral))

    do_type = _build_fast_do_type(Val{DV}())

    kernels = _build_kernel_sym(Val{KS}(), Val{DV}())

    domain_size = ntuple(d -> size(vs, d), N)
    lazy_workspace = LazyBoundaryWorkspace(T, Val(N), maximum(eqs))

    return FastConvolutionInterpolation{T,N,n_integral,typeof(coefs),typeof(knots_new),
                                        typeof(kernel_type),typeof(dimension),typeof(kernels),
                                        typeof(eqs),
                                        typeof(bc),typeof(do_type),
                                        Nothing,Nothing,Val{:not_used},
                                        typeof(Val{LZ}()),typeof(integral_dimension),typeof(domain_size)}(
        coefs, domain_size, knots_new, h, x0, kernel_type, dimension, kernels, eqs,
        bc, do_type, nothing, nothing, Val(:not_used),
        Val{LZ}(), boundary_fallback, left_values, anchor, integral_dimension, lazy_workspace,
        tail1_left, tail2_ll, tail3_edge_ll, tail3_corner_lll,
    )
end

function _build_tails(coefs::AbstractArray{T,N}, kernel, ::Val{DV}, eqs, knots_new) where {T,N,DV}
    n_integral = _count_integrals(Val{DV}())
    derivative = DV
    integral_type = n_integral <= 3 ? Val(n_integral) : HigherDimension(Val(n_integral))
    return _build_tails_dispatch(coefs, kernel, derivative, eqs, knots_new, integral_type)
end

# Anchoring constants (antiderivative kernel at the anchor) of every integral dimension,
# and a one-element placeholder in the other dimensions
_all_left_values(coefs::AbstractArray{T,N}, kernel, derivative, eqs) where {T,N} =
    ntuple(N) do d
        derivative[d] == -1 ? _compute_left_values(T, kernel[d], eqs[d], size(coefs, d)) : [zero(T)]
    end

# Weights ½ − left value of the coefficients left of the stencil in dimension d, shaped to
# broadcast along dimension d of an N-dimensional array
_left_weights(left_values::Vector{T}, d::Int, ::Val{N}) where {T,N} =
    T(1//2) .- reshape(left_values, ntuple(i -> i == d ? length(left_values) : 1, N))

# Left tail in each integral dimension: prefix sums along that dimension
_all_tail1_left(coefs::AbstractArray{T,N}, left_values, derivative, placeholder) where {T,N} =
    ntuple(N) do d
        derivative[d] == -1 ? cumsum(coefs .* _left_weights(left_values[d], d, Val(N)), dims=d) :
                              placeholder
    end

function _build_tails_dispatch(coefs::AbstractArray{T,N}, kernel, derivative, eqs, knots_new, ::Val{0}) where {T,N}
    # no integral dimensions: placeholders only
    placeholder = Array{T,N}(undef, ntuple(_ -> 0, N)...)
    left_values = ntuple(_ -> [zero(T)], N)
    return left_values, ntuple(_ -> placeholder, N), placeholder,
           ntuple(_ -> placeholder, 3), placeholder
end

function _build_tails_dispatch(coefs::AbstractArray{T,N}, kernel, derivative, eqs, knots_new, ::Val{1}) where {T,N}
    # one integral dimension: left tail only
    placeholder = Array{T,N}(undef, ntuple(_ -> 0, N)...)
    left_values = _all_left_values(coefs, kernel, derivative, eqs)
    tail1_left = _all_tail1_left(coefs, left_values, derivative, placeholder)
    return left_values, tail1_left, placeholder,
           ntuple(_ -> placeholder, 3), placeholder
end

function _build_tails_dispatch(coefs::AbstractArray{T,N}, kernel, derivative, eqs, knots_new, ::Val{2}) where {T,N}
    # two integral dimensions: left tails, and the corner left in both
    placeholder = Array{T,N}(undef, ntuple(_ -> 0, N)...)
    int_dims = findall(d -> derivative[d] == -1, 1:N)
    left_values = _all_left_values(coefs, kernel, derivative, eqs)
    tail1_left = _all_tail1_left(coefs, left_values, derivative, placeholder)
    wl1 = _left_weights(left_values[int_dims[1]], int_dims[1], Val(N))
    wl2 = _left_weights(left_values[int_dims[2]], int_dims[2], Val(N))
    tail2_ll = cumsum(cumsum(coefs .* wl1 .* wl2, dims=int_dims[1]), dims=int_dims[2])
    return left_values, tail1_left, tail2_ll,
           ntuple(_ -> placeholder, 3), placeholder
end

function _build_tails_dispatch(coefs::AbstractArray{T,N}, kernel, derivative, eqs, knots_new, ::Val{3}) where {T,N}
    # three integral dimensions: left tails, the three left-left edges, and the left corner
    placeholder = Array{T,N}(undef, ntuple(_ -> 0, N)...)
    int_dims = findall(d -> derivative[d] == -1, 1:N)
    left_values = _all_left_values(coefs, kernel, derivative, eqs)
    tail1_left = _all_tail1_left(coefs, left_values, derivative, placeholder)
    wl1 = _left_weights(left_values[int_dims[1]], int_dims[1], Val(N))
    wl2 = _left_weights(left_values[int_dims[2]], int_dims[2], Val(N))
    wl3 = _left_weights(left_values[int_dims[3]], int_dims[3], Val(N))
    # edge k is free in the k-th integral dimension and left-saturated in the other two
    tail3_edge_ll = (cumsum(cumsum(coefs .* wl2 .* wl3, dims=int_dims[2]), dims=int_dims[3]),
                     cumsum(cumsum(coefs .* wl1 .* wl3, dims=int_dims[1]), dims=int_dims[3]),
                     cumsum(cumsum(coefs .* wl1 .* wl2, dims=int_dims[1]), dims=int_dims[2]))
    tail3_corner_lll = cumsum(cumsum(cumsum(coefs .* wl1 .* wl2 .* wl3, dims=int_dims[1]),
                                     dims=int_dims[2]), dims=int_dims[3])
    return left_values, tail1_left, placeholder, tail3_edge_ll, tail3_corner_lll
end

function _build_tails_dispatch(coefs::AbstractArray{T,N}, kernel, derivative, eqs, knots_new, ::HigherDimension{NI}) where {T,N,NI}
    # more than 3 integral dimensions: evaluated by direct summation, no tail arrays needed
    placeholder = Array{T,N}(undef, ntuple(_ -> 0, N)...)
    left_values = _all_left_values(coefs, kernel, derivative, eqs)
    return left_values, ntuple(_ -> placeholder, N), placeholder,
           ntuple(_ -> placeholder, 3), placeholder
end

@generated function _build_fast_do_type(::Val{D}) where D
    n_int = count(==(-1), D)
    N = length(D)
    if n_int == 0
        return :(DerivativeOrder(Val($D)))
    elseif n_int == N
        return :(FastIntegralOrder())
    else
        return :(FastMixedIntegralOrder{$D}())
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

# K̃ at the integer offset eqs − j of every coefficient j: the anchoring constants of an
# integral dimension. Exact: the constant terms of the column polynomials of K̃ (their value
# at τ = 0), rounded once to T, and the saturated values ±½ outside the kernel support.
function _compute_left_values(T, kernel::Symbol, eqs::Int, n_coefs_d::Int)
    lv = zeros(T, n_coefs_d)
    columns = kernel == :a0 ? nothing : _column_polynomials_exact(kernel, -1)
    for j in 1:n_coefs_d
        s = eqs - j                                   # integer offset of coefficient j
        if abs(s) >= eqs
            lv[j] = T(1//2) * T(sign(s))              # outside the support: saturated
        elseif kernel == :a0
            lv[j] = zero(T)                           # :a0 (eqs = 1): only s = 0 is inside, K̃(0) = 0
        else
            lv[j] = T(columns[s + eqs + 1][1])        # column c = s + eqs + 1, at τ = 0
        end
    end
    return lv
end