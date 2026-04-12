"""
    convolution_interpolation(knots, values::AbstractArray{T,N}; kwargs...) where {T,N}

Create a convolution-based interpolation object with automatic optimization and boundary handling.

# Arguments
- `knots`: Vector or range (1D) or tuple of vectors/ranges (N-D) of grid coordinates.
- `values`: Array of values at the grid points.

# Keyword Arguments
- `kernel::Symbol=:b5`: Convolution kernel to use.
  - `a`-series: `:a0` (nearest), `:a1` (linear), `:a3` (cubic), `:a4` (quartic), `:a5` (quintic), `:a7` (septic)
  - `b`-series: `:b5`, `:b7`, `:b9`, `:b11`, `:b13`
  - Nonuniform-only: `:n3` (cubic)
  `:a0`, `:a1`, and all `b`-series kernels work on both uniform and nonuniform grids.
  Higher `a`-series kernels (`:a3`, `:a4`, `:a5`, `:a7`) fall back to `:n3` on nonuniform grids.
- `fast::Bool=true`: Use precomputed kernel tables for O(1) evaluation. Automatically
  disabled for nonuniform grids.
- `precompute::Int=101`: Resolution of the precomputed kernel table. The default uses
  pre-shipped tables (zero computation). Higher values (e.g. 10_000) trigger on-demand
  computation and disk caching for the `:linear` subgrid mode.
- `B::Float64=-1.0`: If a positive value is provided, uses Gaussian kernel with parameter `B` for C∞ smoothness.
- `extrap=:throw`: Behavior outside the grid domain.
  Options: `:throw`, `:flat`, `:line` or `:natural`.
- `bc=:detect`: Boundary condition for kernel evaluation at domain edges.
  Options: `:detect`, `:poly`, `:linear`, `:quadratic`.
- `derivative::Int=0`: Derivative order to evaluate. Supported up to 6 for `b`-series
  kernels. For `a`-series kernels the top derivative automatically uses linear interpolation
  to match the kernel's continuity class.
- `subgrid::Symbol=:cubic`: Subgrid interpolation mode for precomputed kernel tables.
  Options: `:linear` (fastest, needs high `precompute`), `:cubic` (default), `:quintic`.
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
        kernel::Union{Symbol,NTuple{N,Symbol}}=:b5, fast::Bool=true, precompute::Int=101,
        extrap::Union{Symbol,AbstractExtrapolation}=Throw(),
        bc::Union{Symbol,Tuple{Symbol,Symbol},NTuple{N,Tuple{Symbol,Symbol}}}=:detect,
        derivative::Union{Int,NTuple{N,Int}}=0, subgrid::Union{Symbol,NTuple{N,Symbol}}=:cubic,
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

    is_integral = count(d -> derivatives_tuple[d] == -1, 1:N) > 0
    is_nonuniform = any(d -> !is_uniform_grid(knots_tuple[d]), 1:N) || any(d -> kernels_tuple[d] == :n3, 1:N)

    if is_integral && is_nonuniform
      error("Nonuniform antiderivative (derivative=-1) is not directly supported.\n")
    elseif is_integral && !is_nonuniform
      if lazy && fast
        error("Lazy mode not supported in fast mode for antiderivatives.")
      end
    elseif !is_integral && is_nonuniform
      fast = false # nonuniform grids not supported in fast mode
    end

    if lazy && N>=4 && !boundary_fallback
        error("Lazy mode requires 'boundary_fallback=true' for dimensions >= 4.")
    end

    if lazy && boundary_fallback && any(d -> derivatives_tuple[d] != 0, 1:N)
        error("Derivatives not supported in lazy mode with 'boundary_fallback=true'.")
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
        return _build_natural(knots_tuple, values, kernels_tuple, fast, precompute,
                              bcs_tuple, derivatives_tuple, subgrids_tuple,
                              lazy, boundary_fallback)
    elseif fast
        return _build_fast(knots_tuple, values, kernels_tuple, precompute, bcs_tuple,
                          derivatives_tuple, subgrids_tuple, extrap,
                          lazy, boundary_fallback)
    else
        return _build_slow(knots_tuple, values, kernels_tuple, bcs_tuple, derivatives_tuple,
                          extrap, lazy, boundary_fallback)
    end
end

function _build_fast(knots::NTuple{N,AbstractVector}, values::AbstractArray{T,N},
                    kernel::NTuple{N,Symbol}, precompute::Int,
                    bc::NTuple{N,Tuple{Symbol,Symbol}},
                    derivative::NTuple{N,Int},
                    subgrid::NTuple{N,Symbol},
                    extrap::Union{Symbol,AbstractExtrapolation},
                    lazy::Bool, boundary_fallback::Bool) where {T,N}
    itp = FastConvolutionInterpolation(knots, values;
                                  kernel, precompute, bc, derivative,
                                  subgrid, lazy, boundary_fallback)
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
    else
        error("Unknown extrapolation type: $s, must be :throw, :line, :flat or :natural.")
    end
end