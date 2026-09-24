"""
    convolution_interpolation(knots, values::AbstractArray{T,N}; kwargs...) where {T,N}

Create a convolution-based interpolation object with automatic optimization and boundary handling.

# Arguments
- `knots`: Vector or range (1D) or tuple of vectors/ranges (N-D) of grid coordinates.
- `values`: Array of values at the grid points.

# Keyword Arguments
- `kernel::Symbol=:auto`: Convolution kernel to use (default is N-dependent to reflect tensor product cost).
  - `a`-series: `:a0` (nearest), `:a1` (linear), `:a3` (cubic), `:a4` (quartic), `:a5` (quintic), `:a7` (septic)
  - `b`-series: `:b5`, `:b7`, `:b9`, `:b11`, `:b13`
  - `:n3` (cubic): the nonuniform implementation of `:a3`. `:a3` and `:n3` are interchangeable —
    a uniform grid always evaluates with the fast `:a3` kernel, a nonuniform grid with the
    `:n3` weights, whichever name is given.
  `:a0`, `:a1`, and all `b`-series kernels work on both uniform and nonuniform grids.
  `:a3` falls back to `:n3` on nonuniform grids (same order, nonuniform weights).
  `:a4`, `:a5` and `:a7` are uniform-only and raise an error on nonuniform grids.
- `fast::Bool=true`: Use the fast uniform-grid evaluation (exact kernel weights from the
  kernels' polynomial pieces, O(1) per evaluation). Automatically disabled for nonuniform grids.
- `precompute`: Deprecated, has no effect, and will be removed in a future release. Kernels
  are evaluated exactly, so no kernel tables are precomputed.
- `B::Float64=-1.0`: If a positive value is provided, uses Gaussian kernel with parameter `B` for C∞ smoothness.
- `extrap=:throw`: Behavior outside the grid domain.
  Options: `:throw`, `:flat`, `:line` or `:natural`.
- `bc=:detect`: Boundary condition for kernel evaluation at domain edges.
  Options: `:detect`, `:poly`, `:linear`, `:quadratic`.
- `derivative::Int=0`: Derivative order to evaluate. Supported up to 6 for `b`-series
  kernels. Negative values evaluate antiderivatives (uniform grids, fast path).
- `subgrid`: Deprecated, has no effect, and will be removed in a future release. Kernels
  are evaluated exactly, so there is no subgrid interpolation.
- `lazy::Bool=false`: When `true`, skip ghost point expansion at construction time.
  Ghost values are computed on the fly only when evaluating near boundaries, saving memory
  and speeding up construction — especially in high dimensions. Interior evaluation is
  unaffected. Set to `false` (default) for eager expansion.
- `boundary_fallback::Bool=false`: When `true`, near-boundary evaluations use a linear
  kernel rather than computing full ghost point stencils — correct throughout the domain
  at the cost of reduced smoothness in the boundary stencil region. Derivatives are not
  supported in this mode. Required for `N≥4`; only active when `lazy=true`.

# Returns
A `ConvolutionExtrapolation` object callable at arbitrary points within (or, depending on
`extrap`, outside) the grid domain.

# Performance
- Construction: ~3μs for 1D with 100 points (`:a4` kernel, `lazy=false`).
  With `lazy=true`, construction is faster as ghost point expansion is skipped.
- Evaluation (1D): `:a1` ~3ns, `:a4` ~10ns, `:b5` ~14ns per query, O(1) and allocation-free.
  Boundary evaluation in lazy mode is slightly slower.

# Examples
```julia
# 1D interpolation
x = range(0, 2π, length=50)
itp = convolution_interpolation(x, sin.(x))
itp(1.0)

# 2D with lazy construction
x = range(0, 1, length=100)
y = range(0, 1, length=100)
vals = [sin(2π*xi)*cos(2π*yi) for xi in x, yi in y]
itp = convolution_interpolation((x, y), vals, lazy=true)
itp(0.3, 0.7)

# First derivative
itp_d = convolution_interpolation(x, sin.(x), derivative=1)
itp_d(1.0)  # ≈ cos(1.0)

# High-dimensional with lazy mode (recommended)
knots_5d = ntuple(_ -> range(0, 1, length=20), 5)
vals_5d = rand(20,20,20,20,20)
itp = convolution_interpolation(knots_5d, vals_5d, lazy=true)
itp(0.5, 0.5, 0.5, 0.5, 0.5)
```

See also: [`FastConvolutionInterpolation`](@ref), [`ConvolutionInterpolation`](@ref), [`ConvolutionExtrapolation`](@ref).
"""

function convolution_interpolation(knots::Union{AbstractVector,NTuple{N,AbstractVector}},
        values::AbstractArray{T,N};
        kernel::Union{Symbol,NTuple{N,Symbol}}=:auto, fast::Bool=true, precompute=nothing,
        extrap::Union{Symbol,AbstractExtrapolation}=Throw(),
        bc::Union{Symbol,Tuple{Symbol,Symbol},NTuple{N,Tuple{Symbol,Symbol}}}=:detect,
        derivative::Union{Int,NTuple{N,Int}}=0, subgrid=nothing,
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
    kernels_tuple = n3_to_a3_on_uniform(kernels_tuple, knots_tuple)
    derivatives_tuple = derivative isa NTuple{N,Int} ? derivative :
                    derivative isa Int ? ntuple(_ -> derivative, N) :
                    derivative isa NTuple{1,Int} ? ntuple(_ -> derivative[1], N) : 
                    error("Invalid derivative specification: $derivative.")
    bcs_tuple = bc isa NTuple{N,Tuple{Symbol,Symbol}} ? bc :
                    bc isa Tuple{Symbol,Symbol} ? ntuple(_ -> bc, N) :
                    bc isa NTuple{1,Tuple{Symbol,Symbol}} ? ntuple(_ -> bc[1], N) :
                    bc isa Symbol ? ntuple(_ -> (bc, bc), N) :
                    error("Invalid bc specification: $bc.")

    is_integral = any(d -> derivatives_tuple[d] < 0, 1:N)
    is_nonuniform = any(d -> !is_uniform_grid(knots_tuple[d]), 1:N) || any(d -> kernels_tuple[d] == :n3, 1:N)

    if is_integral && is_nonuniform
      error("Antiderivatives (derivative < 0) are not supported on nonuniform grids.")
    elseif is_integral && !is_nonuniform
      if lazy && fast
        error("Lazy mode not supported in fast mode for antiderivatives.")
      end
      if !fast && any(d -> derivatives_tuple[d] < -1, 1:N)
        error("Antiderivatives of order 2 and higher (derivative < -1) are only available " *
              "on the fast path (fast=true).")
      end
    elseif !is_integral && is_nonuniform
      fast = false # nonuniform grids not supported in fast mode
    end

    if lazy && N>=4 && !boundary_fallback && !is_nonuniform
        error("Lazy uniform mode requires 'boundary_fallback=true' for dimensions >= 4.")
    end

    if lazy && any(d -> derivatives_tuple[d] != 0, 1:N) && (boundary_fallback || is_nonuniform)
        error("In lazy mode, derivatives are only supported in the uniform fast path with 'boundary_fallback=false'.")
    end

    if lazy && !fast && !is_nonuniform
        error("For uniform grids, lazy mode ('lazy=true') is only supported for the fast path ('fast=true').")
    end

    # Check sufficiency per dimension
    bcs_tuple = ntuple(N) do d
        if :poly in bcs_tuple[d] && size(values, d) < minimum_polynomial_bc_points[kernels_tuple[d]]
            # replace :poly with :linear
            bcs_tuple[d] == (:poly,:poly) ? (:linear,:linear) : 
                    bcs_tuple[d][1] == :poly ? (:linear, bcs_tuple[d][2]) : (bcs_tuple[d][1], :linear)
        else
            bcs_tuple[d]
        end
    end

    if extrap == :natural || extrap == Natural()
        return _build_natural(knots_tuple, values, kernels_tuple, fast,
                              bcs_tuple, derivatives_tuple,
                              lazy, boundary_fallback)
    elseif fast
        return _build_fast(knots_tuple, values, kernels_tuple, bcs_tuple,
                          derivatives_tuple, extrap,
                          lazy, boundary_fallback)
    else
        return _build_slow(knots_tuple, values, kernels_tuple, bcs_tuple, derivatives_tuple,
                          extrap, lazy, boundary_fallback)
    end
end

function _build_fast(knots::NTuple{N,AbstractVector}, values::AbstractArray{T,N},
                    kernel::NTuple{N,Symbol},
                    bc::NTuple{N,Tuple{Symbol,Symbol}},
                    derivative::NTuple{N,Int},
                    extrap::Union{Symbol,AbstractExtrapolation},
                    lazy::Bool, boundary_fallback::Bool) where {T,N}
    itp = FastConvolutionInterpolation(knots, values;
                                  kernel, bc, derivative,
                                  lazy, boundary_fallback)
    return ConvolutionExtrapolation(itp, _extrap_type(extrap))
end

function _build_slow(knots::NTuple{N,AbstractVector}, values::AbstractArray{T,N}, 
                    kernel::NTuple{N,Symbol},
                    bc::NTuple{N,Tuple{Symbol,Symbol}},
                    derivative::NTuple{N,Int},
                    extrap::Union{Symbol,AbstractExtrapolation},
                    lazy::Bool, boundary_fallback::Bool) where {T,N}
    itp = ConvolutionInterpolation(knots, values; kernel, bc,
                                  derivative, lazy, boundary_fallback)
    return ConvolutionExtrapolation(itp, _extrap_type(extrap))
end

function _build_natural(knots::NTuple{N,AbstractVector}, values::AbstractArray{T,N}, 
                        kernel::NTuple{N,Symbol}, fast::Bool,
                        bc::NTuple{N,Tuple{Symbol,Symbol}},
                        derivative::NTuple{N,Int},
                        lazy::Bool, boundary_fallback::Bool) where {T,N}
    # Natural extrapolation always uses eager mode (needs double-extrapolation)
    itp = ConvolutionInterpolation(knots, values; kernel, bc, derivative, 
                                    lazy=false, boundary_fallback)
    bc = ntuple(_ -> (:linear,:linear), N) # overwrites
    if fast
        itp = FastConvolutionInterpolation(itp.knots, itp.coefs;
                        kernel, bc, derivative,
                        lazy, boundary_fallback)
    else
        itp = ConvolutionInterpolation(itp.knots, itp.coefs;
                        kernel, bc, derivative, 
                        lazy, boundary_fallback)
    end
    return ConvolutionExtrapolation(itp, Line())
end

const minimum_polynomial_bc_points = Dict(
    :a0 => 2,
    :a1 => 2,
    :a3 => 4,
    :a4 => 4,
    :a5 => 4,
    :a7 => 4,
    :b5 => 6,
    :b7 => 8,
    :b9 => 8,
    :b11 => 8,
    :b13 => 8
)

function _extrap_type(s::AbstractExtrapolation)
    return s
end

function _extrap_type(s::Symbol)
    if s == :throw
        return Throw()
    elseif s == :line
        return Line()
    elseif s == :flat
        return Flat()
    elseif s == :natural
        return Natural()
    else
        error("Unknown extrapolation type: $s, must be :throw, :line, :flat or :natural.")
    end
end