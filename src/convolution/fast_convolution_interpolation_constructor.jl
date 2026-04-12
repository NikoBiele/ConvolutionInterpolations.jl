"""
    FastConvolutionInterpolation(knots, vs::AbstractArray{T,N}; kwargs...) where {T,N}

Construct a fast convolution interpolation object with precomputed kernel tables for O(1) evaluation.
This is the lower-level fast constructor — most users should prefer `convolution_interpolation`,
which wraps this with extrapolation handling and automatic mode selection.

Only supports uniform grids. For nonuniform grids, use `ConvolutionInterpolation` directly.

# Arguments
- `knots`: Vector or range (1D) or tuple of vectors/ranges (N-D) of grid coordinates.
- `vs`: Array of values at the grid points.

# Keyword Arguments
- `kernel::Symbol=:b5`: Convolution kernel to use.
  - `a`-series: `:a0` (nearest), `:a1` (linear), `:a3` (cubic), `:a4` (quartic), `:a5` (quintic), `:a7` (septic)
  - `b`-series: `:b5`, `:b7`, `:b9`, `:b11`, `:b13`
- `precompute::Int=101`: Resolution of the precomputed kernel table. The default uses
  pre-shipped tables (zero computation). Automatically raised to at least 10,000 for
  `:linear` subgrid mode.
- `B::Float64=-1.0`: If a positive value is provided, uses Gaussian kernel with parameter `B` for C∞ smoothness.
  Forces eager mode.
- `bc=:detect`: Boundary condition for kernel evaluation at domain edges.
  Options: `:detect`, `:poly`, `:linear`, `:quadratic`.
- `derivative::Int=0`: Derivative order to evaluate. Supported up to 6 for `b`-series
  kernels. For `a`-series kernels the top derivative automatically uses linear interpolation
  to match the kernel's continuity class.
- `subgrid::Symbol=:cubic`: Subgrid interpolation mode for the precomputed kernel table.
  Options: `:linear` (fastest, needs high `precompute`), `:cubic` (default), `:quintic`.
  Automatically downgraded to `:linear` for 3D+ and when evaluating the top derivative
  of a kernel. Validated against the kernel's continuity class.
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
                                      kernel::Union{Symbol,NTuple{N,Symbol}}=:b5,
                                      precompute::Union{Int,NTuple{N,Int}}=101,
                                      bc::Union{Symbol,Tuple{Symbol,Symbol},NTuple{N,Tuple{Symbol,Symbol}}}=:detect,
                                      derivative::Union{Int,NTuple{N,Int}}=0,
                                      subgrid::Union{Symbol,NTuple{N,Symbol}}=:cubic,
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
    subgrids_tuple = subgrid isa NTuple{N,Symbol} ? subgrid :
                    subgrid isa Symbol ? ntuple(_ -> subgrid, N) : 
                    subgrid isa NTuple{1,Symbol} ? ntuple(_ -> subgrid[1], N) :
                    error("Invalid subgrid specification: $subgrid.")
    bcs_tuple = bc isa NTuple{N,Tuple{Symbol,Symbol}} ? bc :
                    bc isa Tuple{Symbol,Symbol} ? ntuple(_ -> bc, N) :
                    bc isa NTuple{1,Tuple{Symbol,Symbol}} ? ntuple(_ -> bc[1], N) :
                    bc isa Symbol ? ntuple(_ -> (bc, bc), N) :
                    error("Invalid bc specification: $bc.")
    precompute_tuple = precompute isa NTuple{N,Int} ? precompute :
                    precompute isa Int ? ntuple(_ -> precompute, N) :
                    precompute isa NTuple{1,Int} ? ntuple(_ -> precompute[1], N) :
                    error("Invalid precompute specification $precompute")

    if any(==(:n3), kernels_tuple)
        error("The :n3 kernel is not supported by FastConvolutionInterpolation.")
    end
    if lazy && N >= 4 && !boundary_fallback
        error("Lazy mode requires 'boundary_fallback=true' for dimensions >= 4.")
    end
    if lazy && boundary_fallback && any(d -> derivatives_tuple[d] != 0, 1:N)
        error("Derivatives not supported in lazy mode with 'boundary_fallback=true'.")
    end

    return _build_fast_uniform_convolution(knots_tuple, vs, bcs_tuple, precompute_tuple, boundary_fallback,
                                           Val(kernels_tuple), Val(lazy),
                                           Val(derivatives_tuple), Val(subgrids_tuple))
end

function _build_fast_uniform_convolution(knots::NTuple{N,AbstractVector},
                                         vs::AbstractArray{T,N},
                                         bc::BCT,
                                         precompute::NTuple{N,Int},
                                         boundary_fallback::Bool,
                                         ::Val{KS},
                                         ::Val{LZ},
                                         ::Val{DV},
                                         ::Val{SG}) where {T,N,BCT<:Tuple,KS,LZ,DV,SG}

    kernel = KS
    derivative = DV

    eqs = ntuple(d -> get_equations_for_degree(kernel[d]), N)
    n_integral = _count_integrals(Val{DV}())

    subgrids = _build_subgrids(Val{KS}(), Val{DV}(), Val{SG}(), Val{N}())
    
    precompute_actual = ntuple(d -> (subgrids[d] == :linear && !(kernel[d] in (:a0, :a1)) ?
                                max(precompute[d], 10_000) : precompute[d]), N)

    h = ntuple(d -> knots[d][2] - knots[d][1], N)

    all_kernels_low_order = all(d -> kernel[d] == :a0 || kernel[d] == :a1, 1:N)
    coefs, knots_new = if LZ || all_kernels_low_order
        _build_lazy_coefs(knots, vs)
    else
        _build_eager_coefs(knots, vs, eqs, bc, kernel, h, Val(false))
    end

    tables       = ntuple(d -> get_precomputed_kernel_and_range(kernel[d],
                                        precompute_actual[d], T,
                                        derivative[d], subgrids[d]), N)
    pre_range_d  = ntuple(d -> tables[d][1], N)
    kernel_pre_d = ntuple(d -> tables[d][2], N)
    kd1_pre_d    = ntuple(d -> tables[d][3], N)
    kd2_pre_d    = ntuple(d -> tables[d][4], N)

    anchor = ntuple(d -> derivative[d] == -1 ? knots_new[d][eqs[d]] : zero(T), N)

    left_values, tail1_left, tail1_right,
    tail2_ll, tail2_rl, tail2_lr, tail2_rr,
    tail3_edge_ll, tail3_edge_rl, tail3_edge_lr, tail3_edge_rr,
    tail3_face_l, tail3_face_r,
    tail3_corner_lll, tail3_corner_rll, tail3_corner_lrl, tail3_corner_llr,
    tail3_corner_rrl, tail3_corner_rlr, tail3_corner_lrr, tail3_corner_rrr = 
                                    _build_tails(coefs, tables, Val{DV}(), eqs, knots_new, subgrids)

    kernel_type = ntuple(d -> nothing, N)
    dimension = N <= 3 ? Val(N) : HigherDimension(Val(N))
    integral_dimension = n_integral <= 3 ? Val(n_integral) : HigherDimension(Val(n_integral))

    do_type = _build_fast_do_type(Val{DV}())

    kernels = _build_kernel_sym(Val{KS}(), Val{DV}())

    domain_size = ntuple(d -> size(vs, d), N)
    lazy_workspace = LazyBoundaryWorkspace(T, Val(N), maximum(eqs))

    return FastConvolutionInterpolation{T,N,n_integral,typeof(coefs),typeof(knots_new),
                                        typeof(kernel_type),typeof(dimension),typeof(kernels),
                                        typeof(eqs),typeof(pre_range_d),typeof(kernel_pre_d),
                                        typeof(bc),typeof(do_type),
                                        typeof(kd1_pre_d),typeof(kd2_pre_d),typeof(Val(subgrids)),
                                        typeof(Val{LZ}()),typeof(integral_dimension),typeof(domain_size)}(
        coefs, domain_size, knots_new, h, kernel_type, dimension, kernels, eqs,
        pre_range_d, kernel_pre_d, bc, do_type,
        kd1_pre_d, kd2_pre_d, Val(subgrids),
        Val{LZ}(), boundary_fallback, left_values, anchor, integral_dimension, lazy_workspace,
        tail1_left, tail1_right, tail2_ll, tail2_rl, tail2_lr, tail2_rr,
        tail3_edge_ll, tail3_edge_rl, tail3_edge_lr, tail3_edge_rr,
        tail3_face_l, tail3_face_r,
        tail3_corner_lll, tail3_corner_rll, tail3_corner_lrl, tail3_corner_llr,
        tail3_corner_rrl, tail3_corner_rlr, tail3_corner_lrr, tail3_corner_rrr,
    )
end

function _build_tails(coefs::AbstractArray{T,N}, tables, ::Val{DV}, eqs, knots_new, subgrids) where {T,N,DV}
    n_integral = _count_integrals(Val{DV}())
    derivative = DV
    integral_type = n_integral <= 3 ? Val(n_integral) : HigherDimension(Val(n_integral))
    return _build_tails_dispatch(coefs, tables, derivative, eqs, knots_new, integral_type, subgrids)
end

function _build_tails_dispatch(coefs::AbstractArray{T,N}, tables, derivative, eqs, knots_new, ::Val{0}, subgrids) where {T,N}
    placeholder = Array{T,N}(undef, ntuple(_ -> 0, N)...)
    left_values = ntuple(_ -> [zero(T)], N)
    tail1_left  = ntuple(_ -> placeholder, N)
    tail1_right = ntuple(_ -> placeholder, N)
    tail2_ll = placeholder; tail2_rl = placeholder
    tail2_lr = placeholder; tail2_rr = placeholder
    ph3 = ntuple(_ -> placeholder, 3)
    return left_values, tail1_left, tail1_right,
           tail2_ll, tail2_rl, tail2_lr, tail2_rr,
           ph3, ph3, ph3, ph3, ph3, ph3,
           placeholder, placeholder, placeholder, placeholder,
           placeholder, placeholder, placeholder, placeholder
end

function _build_tails_dispatch(coefs::AbstractArray{T,N}, tables, derivative, eqs, knots_new, ::Val{1}, subgrids) where {T,N}
    placeholder = Array{T,N}(undef, ntuple(_ -> 0, N)...)
    int_dims = findall(d -> derivative[d] == -1, 1:N)
    left_values = ntuple(N) do d
        if derivative[d] == -1
            _compute_left_values(T, eqs[d], size(coefs, d),
                tables[d][2], tables[d][3], tables[d][4], Val(subgrids[d]))
        else
            [zero(T)]
        end
    end
    tail1_left = ntuple(N) do d
        if derivative[d] == -1
            lv = reshape(left_values[d], ntuple(i -> i == d ? length(left_values[d]) : 1, N))
            cumsum(coefs .* (T(1//2) .- lv), dims=d)
        else
            placeholder
        end
    end
    tail1_right = ntuple(N) do d
        if derivative[d] == -1
            lv = reshape(left_values[d], ntuple(i -> i == d ? length(left_values[d]) : 1, N))
            _suffix_sum(coefs .* (-T(1//2) .- lv), d)
        else
            placeholder
        end
    end
    tail2_ll = placeholder; tail2_rl = placeholder
    tail2_lr = placeholder; tail2_rr = placeholder
    ph3 = ntuple(_ -> placeholder, 3)
    return left_values, tail1_left, tail1_right,
           tail2_ll, tail2_rl, tail2_lr, tail2_rr,
           ph3, ph3, ph3, ph3, ph3, ph3,
           placeholder, placeholder, placeholder, placeholder,
           placeholder, placeholder, placeholder, placeholder
end

function _build_tails_dispatch(coefs::AbstractArray{T,N}, tables, derivative, eqs, knots_new, ::Val{2}, subgrids) where {T,N}
    placeholder = Array{T,N}(undef, ntuple(_ -> 0, N)...)
    int_dims = findall(d -> derivative[d] == -1, 1:N)
    left_values = ntuple(N) do d
        if derivative[d] == -1
            _compute_left_values(T, eqs[d], size(coefs, d),
                tables[d][2], tables[d][3], tables[d][4], Val(subgrids[d]))
        else
            [zero(T)]
        end
    end
    tail1_left = ntuple(N) do d
        if derivative[d] == -1
            lv = reshape(left_values[d], ntuple(i -> i == d ? length(left_values[d]) : 1, N))
            cumsum(coefs .* (T(1//2) .- lv), dims=d)
        else
            placeholder
        end
    end
    tail1_right = ntuple(N) do d
        if derivative[d] == -1
            lv = reshape(left_values[d], ntuple(i -> i == d ? length(left_values[d]) : 1, N))
            _suffix_sum(coefs .* (-T(1//2) .- lv), d)
        else
            placeholder
        end
    end
    lv1 = reshape(left_values[int_dims[1]], ntuple(i -> i == int_dims[1] ? size(coefs, int_dims[1]) : 1, N))
    lv2 = reshape(left_values[int_dims[2]], ntuple(i -> i == int_dims[2] ? size(coefs, int_dims[2]) : 1, N))
    wll = coefs .* (T(1//2) .- lv1) .* (T(1//2) .- lv2)
    wrl = coefs .* (-T(1//2) .- lv1) .* (T(1//2) .- lv2)
    wlr = coefs .* (T(1//2) .- lv1) .* (-T(1//2) .- lv2)
    wrr = coefs .* (-T(1//2) .- lv1) .* (-T(1//2) .- lv2)
    tail2_ll = cumsum(cumsum(wll, dims=int_dims[1]), dims=int_dims[2])
    tail2_rl = _suffix_sum(cumsum(wrl, dims=int_dims[2]), int_dims[1])
    tail2_lr = cumsum(_suffix_sum(wlr, int_dims[2]), dims=int_dims[1])
    tail2_rr = _suffix_sum(_suffix_sum(wrr, int_dims[2]), int_dims[1])
    ph3 = ntuple(_ -> placeholder, 3)
    return left_values, tail1_left, tail1_right,
           tail2_ll, tail2_rl, tail2_lr, tail2_rr,
           ph3, ph3, ph3, ph3, ph3, ph3,
           placeholder, placeholder, placeholder, placeholder,
           placeholder, placeholder, placeholder, placeholder
end

function _build_tails_dispatch(coefs::AbstractArray{T,N}, tables, derivative, eqs, knots_new, ::Val{3}, subgrids) where {T,N}
    placeholder = Array{T,N}(undef, ntuple(_ -> 0, N)...)
    int_dims = findall(d -> derivative[d] == -1, 1:N)
    left_values = ntuple(N) do d
        if derivative[d] == -1
            _compute_left_values(T, eqs[d], size(coefs, d),
                tables[d][2], tables[d][3], tables[d][4], Val(subgrids[d]))
        else
            [zero(T)]
        end
    end
    tail1_left = ntuple(N) do d
        if derivative[d] == -1
            lv = reshape(left_values[d], ntuple(i -> i == d ? length(left_values[d]) : 1, N))
            cumsum(coefs .* (T(1//2) .- lv), dims=d)
        else
            placeholder
        end
    end
    tail1_right = ntuple(N) do d
        if derivative[d] == -1
            lv = reshape(left_values[d], ntuple(i -> i == d ? length(left_values[d]) : 1, N))
            _suffix_sum(coefs .* (-T(1//2) .- lv), d)
        else
            placeholder
        end
    end
    lv1 = reshape(left_values[int_dims[1]], ntuple(i -> i == int_dims[1] ? size(coefs, int_dims[1]) : 1, N))
    lv2 = reshape(left_values[int_dims[2]], ntuple(i -> i == int_dims[2] ? size(coefs, int_dims[2]) : 1, N))
    lv3 = reshape(left_values[int_dims[3]], ntuple(i -> i == int_dims[3] ? size(coefs, int_dims[3]) : 1, N))
    wl1 = T(1//2) .- lv1;  wr1 = -T(1//2) .- lv1
    wl2 = T(1//2) .- lv2;  wr2 = -T(1//2) .- lv2
    wl3 = T(1//2) .- lv3;  wr3 = -T(1//2) .- lv3
    lv1b = reshape(left_values[int_dims[1]], ntuple(i -> i == int_dims[1] ? size(coefs, int_dims[1]) : 1, N))
    lv2b = reshape(left_values[int_dims[2]], ntuple(i -> i == int_dims[2] ? size(coefs, int_dims[2]) : 1, N))
    wll = coefs .* (T(1//2) .- lv1b) .* (T(1//2) .- lv2b)
    wrl = coefs .* (-T(1//2) .- lv1b) .* (T(1//2) .- lv2b)
    wlr = coefs .* (T(1//2) .- lv1b) .* (-T(1//2) .- lv2b)
    wrr = coefs .* (-T(1//2) .- lv1b) .* (-T(1//2) .- lv2b)
    tail2_ll = cumsum(cumsum(wll, dims=int_dims[1]), dims=int_dims[2])
    tail2_rl = _suffix_sum(cumsum(wrl, dims=int_dims[2]), int_dims[1])
    tail2_lr = cumsum(_suffix_sum(wlr, int_dims[2]), dims=int_dims[1])
    tail2_rr = _suffix_sum(_suffix_sum(wrr, int_dims[2]), int_dims[1])
    tail3_face_l = (cumsum(coefs .* wl1, dims=int_dims[1]), cumsum(coefs .* wl2, dims=int_dims[2]), cumsum(coefs .* wl3, dims=int_dims[3]))
    tail3_face_r = (_suffix_sum(coefs .* wr1, int_dims[1]), _suffix_sum(coefs .* wr2, int_dims[2]), _suffix_sum(coefs .* wr3, int_dims[3]))
    tail3_edge_ll = (cumsum(cumsum(coefs .* wl2 .* wl3, dims=int_dims[2]), dims=int_dims[3]), cumsum(cumsum(coefs .* wl1 .* wl3, dims=int_dims[1]), dims=int_dims[3]), cumsum(cumsum(coefs .* wl1 .* wl2, dims=int_dims[1]), dims=int_dims[2]))
    tail3_edge_rl = (_suffix_sum(cumsum(coefs .* wr2 .* wl3, dims=int_dims[3]), int_dims[2]), _suffix_sum(cumsum(coefs .* wr1 .* wl3, dims=int_dims[3]), int_dims[1]), _suffix_sum(cumsum(coefs .* wr1 .* wl2, dims=int_dims[2]), int_dims[1]))
    tail3_edge_lr = (cumsum(_suffix_sum(coefs .* wl2 .* wr3, int_dims[3]), dims=int_dims[2]), cumsum(_suffix_sum(coefs .* wl1 .* wr3, int_dims[3]), dims=int_dims[1]), cumsum(_suffix_sum(coefs .* wl1 .* wr2, int_dims[2]), dims=int_dims[1]))
    tail3_edge_rr = (_suffix_sum(_suffix_sum(coefs .* wr2 .* wr3, int_dims[3]), int_dims[2]), _suffix_sum(_suffix_sum(coefs .* wr1 .* wr3, int_dims[3]), int_dims[1]), _suffix_sum(_suffix_sum(coefs .* wr1 .* wr2, int_dims[2]), int_dims[1]))
    tail3_corner_lll = cumsum(cumsum(cumsum(coefs .* wl1 .* wl2 .* wl3, dims=int_dims[1]), dims=int_dims[2]), dims=int_dims[3])
    tail3_corner_rll = _suffix_sum(cumsum(cumsum(coefs .* wr1 .* wl2 .* wl3, dims=int_dims[2]), dims=int_dims[3]), int_dims[1])
    tail3_corner_lrl = cumsum(_suffix_sum(cumsum(coefs .* wl1 .* wr2 .* wl3, dims=int_dims[3]), int_dims[2]), dims=int_dims[1])
    tail3_corner_llr = cumsum(cumsum(_suffix_sum(coefs .* wl1 .* wl2 .* wr3, int_dims[3]), dims=int_dims[1]), dims=int_dims[2])
    tail3_corner_rrl = _suffix_sum(_suffix_sum(cumsum(coefs .* wr1 .* wr2 .* wl3, dims=int_dims[3]), int_dims[2]), int_dims[1])
    tail3_corner_rlr = _suffix_sum(cumsum(_suffix_sum(coefs .* wr1 .* wl2 .* wr3, int_dims[2]), dims=int_dims[1]), int_dims[3])
    tail3_corner_lrr = cumsum(_suffix_sum(_suffix_sum(coefs .* wl1 .* wr2 .* wr3, int_dims[3]), int_dims[2]), dims=int_dims[1])
    tail3_corner_rrr = _suffix_sum(_suffix_sum(_suffix_sum(coefs .* wr1 .* wr2 .* wr3, int_dims[3]), int_dims[2]), int_dims[1])
    return left_values, tail1_left, tail1_right,
           tail2_ll, tail2_rl, tail2_lr, tail2_rr,
           tail3_edge_ll, tail3_edge_rl, tail3_edge_lr, tail3_edge_rr,
           tail3_face_l, tail3_face_r,
           tail3_corner_lll, tail3_corner_rll, tail3_corner_lrl, tail3_corner_llr,
           tail3_corner_rrl, tail3_corner_rlr, tail3_corner_lrr, tail3_corner_rrr
end

function _build_tails_dispatch(coefs::AbstractArray{T,N}, tables, derivative, eqs, knots_new, ::HigherDimension{NI}, subgrids) where {T,N,NI}
    # n_integral > 3: HigherDimension path, no tail arrays needed
    placeholder = Array{T,N}(undef, ntuple(_ -> 0, N)...)
    left_values = ntuple(N) do d
        if derivative[d] == -1
            _compute_left_values(T, eqs[d], size(coefs, d),
                tables[d][2], tables[d][3], tables[d][4], Val(subgrids[d]))
        else
            [zero(T)]
        end
    end
    tail1_left  = ntuple(_ -> placeholder, N)
    tail1_right = ntuple(_ -> placeholder, N)
    ph3 = ntuple(_ -> placeholder, 3)
    return left_values, tail1_left, tail1_right,
           placeholder, placeholder, placeholder, placeholder,
           ph3, ph3, ph3, ph3, ph3, ph3,
           placeholder, placeholder, placeholder, placeholder,
           placeholder, placeholder, placeholder, placeholder
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

const max_smooth_derivative = Dict(
  :a0 => -1,
  :a1 => 0,
  :a3 => 1,
  :a4 => 1,
  :a5 => 1,
  :a7 => 1,
  :b5 => 3,
  :b7 => 4,
  :b9 => 5,
  :b11 => 6,
  :b13 => 6
)

function _compute_left_values(T, eqs, n_coefs_d, kernel_pre, kernel_d1_pre, kernel_d2_pre, sg)
    n_pre_loc = size(kernel_pre, 1)
    h_pre_loc = one(T) / T(n_pre_loc - 1)
    lv = zeros(T, n_coefs_d)
    @inbounds for j in 1:n_coefs_d
        sj = T(eqs - j)
        s_abs = abs(sj)
        if s_abs >= T(eqs)
            lv[j] = T(1//2) * T(sign(sj))
        else
            col_float = T(eqs) + sj
            col = clamp(floor(Int, col_float) + 1, 1, 2 * eqs)
            x_diff_right = col_float - T(col - 1)
            continuous_idx = x_diff_right * T(n_pre_loc - 1) + one(T)
            idx      = clamp(floor(Int, continuous_idx), 1, n_pre_loc - 1)
            idx_next = idx + 1
            t        = continuous_idx - T(idx)
            lv[j] = if sg == Val(:quintic)
                quintic_hermite(t,
                    kernel_pre[idx,    col], kernel_pre[idx_next,    col],
                    kernel_d1_pre[idx, col], kernel_d1_pre[idx_next, col],
                    kernel_d2_pre[idx, col], kernel_d2_pre[idx_next, col],
                    h_pre_loc)
            elseif sg == Val(:cubic)
                cubic_hermite(t,
                    kernel_pre[idx,    col], kernel_pre[idx_next,    col],
                    kernel_d1_pre[idx, col], kernel_d1_pre[idx_next, col],
                    h_pre_loc)
            else
                (one(T) - t) * kernel_pre[idx, col] + t * kernel_pre[idx_next, col]
            end
        end
    end
    lv
end

# Computes the suffix (reverse cumulative) sum of A along dimension `dims`.
# suffix_sum[i] = sum of A[i], A[i+1], ..., A[end] along that dimension.
function _suffix_sum(A::AbstractArray, dims::Int)
    reverse(cumsum(reverse(A, dims=dims), dims=dims), dims=dims)
end

@generated function _build_subgrids(::Val{KS}, ::Val{DV}, ::Val{SG}, ::Val{N}) where {KS,DV,SG,N}
    a0_a1_kernel_present = any(k -> k == :a0 || k == :a1, KS)
    all_cubic_subgrid = allequal(SG) && SG[1] == :cubic
    has_integral_dim = any(dv -> dv == -1, DV)
    result = ntuple(N) do d
        k = KS[d]
        dv = DV[d]
        sg = SG[d]
        # integral direction
        if dv == -1
            if k == :a0 || k == :a1 || N >= 4
                :linear
            elseif k in (:a3, :a4, :a5, :a7)
                :cubic
            else
                sg
            end
        # interpolation direction
        elseif dv == 0
            if a0_a1_kernel_present || (N == 3 && !all_cubic_subgrid) || 
                                    N >= 4 || has_integral_dim
                :linear
            else
                sg
            end
        # derivative direction
        else # if dv >= 1
            max_deriv = get(ConvolutionInterpolations.max_smooth_derivative, k, 0)
            available = max_deriv - dv
            if available <= 0 || (N == 3 && !all_cubic_subgrid) || 
                        N >= 4 || a0_a1_kernel_present || has_integral_dim
                :linear
            else
                sg
            end
        end
    end
    return :($(result))
end