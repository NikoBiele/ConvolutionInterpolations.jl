"""
    convolution_interpolation(knots, values::AbstractArray{T,N}; kwargs...) where {T,N}

Create a convolution-based interpolation object with automatic optimization and boundary handling.

# Arguments
- `knots`: Vector or range (1D) or tuple of vectors/ranges (N-D) of grid coordinates.
- `values`: Array of values at the grid points.

# Keyword Arguments
- `degree::Symbol=:a4`: Convolution kernel to use.
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
- `B=nothing`: If provided, uses Gaussian kernel with parameter `B` for C∞ smoothness.
- `extrapolation_bc=Throw()`: Behavior outside the grid domain.
  Options: `Throw()`, `Flat()`, `Line()`, `Periodic()`, or `Natural()`.
- `kernel_bc=:auto`: Boundary condition for kernel evaluation at domain edges.
  Options: `:auto`, `:polynomial`, `:linear`, `:quadratic`, `:periodic`, `:detect`.
- `derivative::Int=0`: Derivative order to evaluate. Supported up to 6 for `b`-series
  kernels. For `a`-series kernels the top derivative automatically uses linear interpolation
  to match the kernel's continuity class.
- `subgrid::Symbol=:cubic`: Subgrid interpolation mode for precomputed kernel tables.
  Options: `:linear` (fastest, needs high `precompute`), `:cubic` (default), `:quintic`.
- `lazy::Bool=false`: When `true`, skip ghost point expansion at construction time.
  Ghost values are computed on the fly only when evaluating near boundaries, saving memory
  and speeding up construction — especially in high dimensions. Interior evaluation is
  unaffected. Set to `false` (default) for eager expansion.
- `boundary_fallback::Bool=false`: When `true`, throw an error instead of computing ghost
  points when evaluating near boundaries in lazy mode. This prevents expensive tensor-product
  ghost computation in high dimensions. Automatically forced to `true` for `b`-series kernels
  in 4D and for all kernels in 5D+. Only active when `lazy=true`.

# Returns
A `ConvolutionExtrapolation` object callable at arbitrary points within (or, depending on
`extrapolation_bc`, outside) the grid domain.

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
itp(0.5, 0.5, 0.5, 0.5, 0.5)  # boundary_fallback automatically active
```

See also: [`FastConvolutionInterpolation`](@ref), [`ConvolutionInterpolation`](@ref), [`ConvolutionExtrapolation`](@ref).
"""

function convolution_interpolation(knots, values::AbstractArray{T,N}; 
        degree::Union{Symbol,NTuple{N,Symbol}}=:b5, fast::Bool=true, precompute::Int=101,
        B=nothing, extrapolation_bc=Throw(),
        kernel_bc::Union{Symbol,Vector{Tuple{Symbol,Symbol}},NTuple{N,Tuple{Symbol,Symbol}}}=:auto,
        derivative::Union{Int,NTuple{N,Int}}=0, subgrid::Symbol=:cubic,
        lazy::Bool=false, boundary_fallback::Bool=false) where {T,N}
    
    knots_tuple = knots isa AbstractVector || knots isa AbstractRange ? (knots,) : knots
    if any(d -> !is_uniform_grid(knots_tuple[d]), 1:N) || degree == :n3
        fast = false
    end
    degrees_tuple = degree isa Symbol ? ntuple(_ -> degree, N) : degree
    if any(d -> !is_uniform_grid(knots_tuple[d]), 1:N) || any(d -> degrees_tuple[d] == :n3, 1:N)
        fast = false
    end

    if lazy == true && N == 4 && degree in (:b5, :b7, :b9, :b11, :b13)
      boundary_fallback = true # in lazy mode for 4D, force extrapolation near boundaries for high kernels
    elseif lazy == true && N >= 5
      boundary_fallback = true # in lazy mode for 5D+, force extrapolation near boundaries for all kernels
    end

    # Normalize kernel_bc to per-dimension, per-side tuples
    kernel_bcs = if kernel_bc isa Symbol
        ntuple(_ -> (kernel_bc, kernel_bc), N)
    elseif kernel_bc isa Vector{Tuple{Symbol,Symbol}}
        ntuple(d -> kernel_bc[d], N)
    elseif kernel_bc isa NTuple{N,Tuple{Symbol,Symbol}}
        kernel_bc  # already a tuple of tuples
    else
        error("Unsupported kernel_bc type: $(typeof(kernel_bc)), must be Symbol, Vector{Tuple{Symbol,Symbol}}, or NTuple{N,Tuple{Symbol,Symbol}}.")
    end

    # Check sufficiency per dimension
    if lazy && boundary_fallback
        eqs = get_equations_for_degree(degree)
        extrapolation_bc = ConvolutionInterpolations.Line()
        kernel_bcs = ntuple(N) do d
            if size(values, d) < 3*eqs-2
                (:detect, :detect)
            else
                kernel_bcs[d]
            end
        end
    end

    if extrapolation_bc isa Natural
        return _build_natural(knots, values; degree=degree, fast=fast, precompute=precompute, B=B,
                              kernel_bc=kernel_bcs, derivative=derivative, subgrid=subgrid,
                              lazy=lazy, boundary_fallback=boundary_fallback)
    elseif fast
        return _build_fast(knots, values; degree=degree, precompute=precompute, B=B, kernel_bc=kernel_bcs,
                          derivative=derivative, subgrid=subgrid, extrapolation_bc=extrapolation_bc,
                          lazy=lazy, boundary_fallback=boundary_fallback)
    else
        return _build_slow(knots, values; degree=degree, B=B, kernel_bc=kernel_bcs, derivative=derivative,
                          extrapolation_bc=extrapolation_bc, lazy=lazy, boundary_fallback=boundary_fallback)
    end
end

function _build_fast(knots, values::AbstractArray{T,N}; degree::Union{Symbol,NTuple{N,Symbol}}=:b5, precompute::Int=101, B::Union{Nothing,Float64}=B,
                    kernel_bc::NTuple{N,Tuple{Symbol,Symbol}}=:auto,
                    derivative::Union{Int,NTuple{N,Int}}=0, subgrid::Symbol=:cubic, extrapolation_bc=Throw(),
                    lazy::Bool=false, boundary_fallback::Bool=false) where {T,N}
    itp = FastConvolutionInterpolation(knots, values;
          degree=degree, precompute=precompute, B=B, kernel_bc=kernel_bc, derivative=derivative,
          subgrid=subgrid, lazy=lazy, boundary_fallback=boundary_fallback)
    return ConvolutionExtrapolation(itp, extrapolation_bc)
end

function _build_slow(knots, values::AbstractArray{T,N}; degree::Union{Symbol,NTuple{N,Symbol}}=:b5, B::Union{Nothing,Float64}=nothing,
                    kernel_bc::NTuple{N,Tuple{Symbol,Symbol}}=:auto,
                    derivative::Union{Int,NTuple{N,Int}}=0, extrapolation_bc=Throw(),
                    lazy::Bool=false, boundary_fallback::Bool=false) where {T,N}
    itp = ConvolutionInterpolation(knots, values; degree=degree, B=B, kernel_bc=kernel_bc,
                                    derivative=derivative, lazy=lazy, boundary_fallback=boundary_fallback)
    return ConvolutionExtrapolation(itp, extrapolation_bc)
end

function _build_natural(knots, values::AbstractArray{T,N}; degree::Union{Symbol,NTuple{N,Symbol}}=:b5, fast::Bool=true,
                        precompute::Int=101, B::Union{Nothing,Float64}=nothing,
                        kernel_bc::NTuple{N,Tuple{Symbol,Symbol}}=:auto,
                        derivative::Union{Int,NTuple{N,Int}}=0, subgrid::Symbol=:cubic, lazy::Bool=false,
                        boundary_fallback::Bool=false) where {T,N}
    # Natural extrapolation always uses eager mode (needs double-extrapolation)
    itp = ConvolutionInterpolation(knots, values; degree, B, kernel_bc, derivative, lazy=false)
    if fast
        itp = FastConvolutionInterpolation(itp.knots, itp.coefs;
              degree=degree, precompute=precompute, B=B, kernel_bc=:linear,
              derivative=derivative, subgrid=subgrid, lazy=lazy,
              boundary_fallback=boundary_fallback)
    else
        itp = ConvolutionInterpolation(itp.knots, itp.coefs;
              degree=degree, B=B, kernel_bc=:linear, derivative=derivative, lazy=lazy,
              boundary_fallback=boundary_fallback)
    end
    return ConvolutionExtrapolation(itp, Line())
end