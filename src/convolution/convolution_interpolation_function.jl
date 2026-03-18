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
- `B=nothing`: If provided, uses Gaussian kernel with parameter `B` for C∞ smoothness.
- `extrap=:throw`: Behavior outside the grid domain.
  Options: `:throw`, `:flat`, `:line` or `:natural`.
- `bc=:auto`: Boundary condition for kernel evaluation at domain edges.
  Options: `:auto`, `:poly`, `:linear`, `:quadratic`, `:periodic`.
- `derivative::Int=0`: Derivative order to evaluate. Supported up to 6 for `b`-series
  kernels. For `a`-series kernels the top derivative automatically uses linear interpolation
  to match the kernel's continuity class.
- `subgrid::Symbol=:cubic`: Subgrid interpolation mode for precomputed kernel tables.
  Options: `:linear` (fastest, needs high `precompute`), `:cubic` (default), `:quintic`.
- `lazy::Bool=false`: When `true`, skip ghost point expansion at construction time.
  Ghost values are computed on the fly only when evaluating near boundaries, saving memory
  and speeding up construction — especially in high dimensions. Interior evaluation is
  unaffected. Set to `false` (default) for eager expansion.
- `boundary_fallback::Bool=false`: When `true`, near-boundary evaluations are linearly
  extrapolated from interior points rather than computing full ghost point stencils.
  Prevents expensive tensor-product ghost computation in high dimensions.
  Only active when `lazy=true`.

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

function convolution_interpolation(knots, values::AbstractArray{T,N}; 
        kernel::Union{Symbol,NTuple{N,Symbol}}=:b5, fast::Bool=true, precompute::Int=101,
        B=nothing, extrap=:throw,
        bc::Union{Symbol,Vector{Tuple{Symbol,Symbol}},NTuple{N,Tuple{Symbol,Symbol}}}=:auto,
        derivative::Union{Int,NTuple{N,Int}}=0, subgrid::Union{Symbol,NTuple{N,Symbol}}=:cubic,
        lazy::Bool=false, boundary_fallback::Bool=false) where {T,N}
    
    knots_tuple = knots isa AbstractVector || knots isa AbstractRange ? (knots,) : knots
    if any(d -> !is_uniform_grid(knots_tuple[d]), 1:N) || kernel == :n3
        fast = false
    end
    kernels_tuple = kernel isa Symbol ? ntuple(_ -> kernel, N) : kernel
    if any(d -> !is_uniform_grid(knots_tuple[d]), 1:N) || any(d -> kernels_tuple[d] == :n3, 1:N)
        fast = false
    end
    if kernel == :a0
      subgrid = :linear
    end

    # Normalize bc to per-dimension, per-side tuples
    bcs = if bc isa Symbol
        ntuple(_ -> (bc, bc), N)
    elseif bc isa Vector{Tuple{Symbol,Symbol}}
        ntuple(d -> bc[d], N)
    elseif bc isa NTuple{N,Tuple{Symbol,Symbol}}
        bc  # already a tuple of tuples
    else
        error("Unsupported bc type: $(typeof(bc)), must be Symbol, Vector{Tuple{Symbol,Symbol}}, or NTuple{N,Tuple{Symbol,Symbol}}.")
    end

    # Check sufficiency per dimension
    if lazy && boundary_fallback
        eqs = get_equations_for_degree(kernel)
        extrap = :line
        bcs = ntuple(N) do d
            if size(values, d) < 3*eqs-2
                (:detect, :detect)
            else
                bcs[d]
            end
        end
    end

    if extrap == :natural
        return _build_natural(knots, values; kernel=kernel, fast=fast, precompute=precompute, B=B,
                              bc=bcs, derivative=derivative, subgrid=subgrid,
                              lazy=lazy, boundary_fallback=boundary_fallback)
    elseif fast
        return _build_fast(knots, values; kernel=kernel, precompute=precompute, B=B, bc=bcs,
                          derivative=derivative, subgrid=subgrid, extrap=extrap,
                          lazy=lazy, boundary_fallback=boundary_fallback)
    else
        return _build_slow(knots, values; kernel=kernel, B=B, bc=bcs, derivative=derivative,
                          extrap=extrap, lazy=lazy, boundary_fallback=boundary_fallback)
    end
end

function _build_fast(knots, values::AbstractArray{T,N}; kernel::Union{Symbol,NTuple{N,Symbol}}=:b5, precompute::Int=101, B::Union{Nothing,Float64}=B,
                    bc::Union{Symbol,Vector{Tuple{Symbol,Symbol}},NTuple{N,Tuple{Symbol,Symbol}}}=:auto,
                    derivative::Union{Int,NTuple{N,Int}}=0, subgrid::Union{Symbol,NTuple{N,Symbol}}=:cubic, extrap=:throw,
                    lazy::Bool=false, boundary_fallback::Bool=false) where {T,N}
    itp = FastConvolutionInterpolation(knots, values;
          kernel=kernel, precompute=precompute, B=B, bc=bc, derivative=derivative,
          subgrid=subgrid, lazy=lazy, boundary_fallback=boundary_fallback)
    return ConvolutionExtrapolation(itp, extrap)
end

function _build_slow(knots, values::AbstractArray{T,N}; kernel::Union{Symbol,NTuple{N,Symbol}}=:b5, B::Union{Nothing,Float64}=nothing,
                    bc::Union{Symbol,Vector{Tuple{Symbol,Symbol}},NTuple{N,Tuple{Symbol,Symbol}}}=:auto,
                    derivative::Union{Int,NTuple{N,Int}}=0, extrap=:throw,
                    lazy::Bool=false, boundary_fallback::Bool=false) where {T,N}
    itp = ConvolutionInterpolation(knots, values; kernel=kernel, B=B, bc=bc,
                                    derivative=derivative, lazy=lazy, boundary_fallback=boundary_fallback)
    return ConvolutionExtrapolation(itp, extrap)
end

function _build_natural(knots, values::AbstractArray{T,N}; kernel::Union{Symbol,NTuple{N,Symbol}}=:b5, fast::Bool=true,
                        precompute::Int=101, B::Union{Nothing,Float64}=nothing,
                        bc::Union{Symbol,Vector{Tuple{Symbol,Symbol}},NTuple{N,Tuple{Symbol,Symbol}}}=:auto,
                        derivative::Union{Int,NTuple{N,Int}}=0, subgrid::Symbol=:cubic, lazy::Bool=false,
                        boundary_fallback::Bool=false) where {T,N}
    # Natural extrapolation always uses eager mode (needs double-extrapolation)
    itp = ConvolutionInterpolation(knots, values; kernel, B, bc, derivative, lazy=false)
    if fast
        itp = FastConvolutionInterpolation(itp.knots, itp.coefs;
              kernel=kernel, precompute=precompute, B=B, bc=:linear,
              derivative=derivative, subgrid=subgrid, lazy=lazy,
              boundary_fallback=boundary_fallback)
    else
        itp = ConvolutionInterpolation(itp.knots, itp.coefs;
              kernel=kernel, B=B, bc=:linear, derivative=derivative, lazy=lazy,
              boundary_fallback=boundary_fallback)
    end
    return ConvolutionExtrapolation(itp, :line)
end