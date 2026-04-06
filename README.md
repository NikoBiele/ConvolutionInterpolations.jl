# ConvolutionInterpolations.jl

High-order interpolation, differentiation, integration and smoothing on discrete grids in arbitrary dimensions.

[![Performance Comparison](fig/convolution_interpolation_kernels.png)](fig/convolution_interpolation_kernels.png)

## Why ConvolutionInterpolations.jl?

ConvolutionInterpolations.jl uses a new family of high-order convolution kernels to provide a single unified interface for interpolation, differentiation, integration and smoothing - from nearest-neighbor to C⁶ smooth 13th-degree polynomial kernels, on uniform and non-uniform grids, in any number of dimensions.

- **High accuracy by default**: The default `:b5` kernel gives 7th-order convergence
- **Uniform grids**: Uniform grids use optimized precomputed kernels
- **Non-uniform grids**: Non-uniform grids are detected automatically
- **O(1) evaluation (uniform)**: Query time is independent of grid size with allocation-free evaluation
- **N-dimensional**: Separable kernel design scales naturally from 1D to arbitrary dimensions
- **Simple API**: A single interface covers nearest-neighbor through 13th-degree polynomial kernels
- **Derivatives up to 6th order**: Analytically differentiated kernels, stable and allocation-free
- **Antiderivative support (uniform)**: Compute 7th order accurate smooth indefinite integrals
- **Gaussian smoothing**: Recover clean signals from noisy data with `convolution_smooth`
- **Grid resampling**: High-order separable resampling with `convolution_resample`

## Installation

```julia
using Pkg
Pkg.add("ConvolutionInterpolations")
```

## Quick Start

### 1D Interpolation

```julia
using ConvolutionInterpolations, Plots

# Sparse sampling: 4 samples sine wave
x = range(0, 2π, length=4)
y = sin.(x)
itp = convolution_interpolation(x, y);  # default :b5 kernel

x_fine = range(0, 2π, length=200)
p1 = plot(x_fine, sin.(x_fine), label="True function: sin(x)")
scatter!(p1, x, y, label="4 samples")
plot!(p1, x_fine, itp.(x_fine), label="Interpolated (4 samples)")
```

[![1D sine wave interpolation](fig/simple_sine_wave_1D_demonstration.png)](fig/simple_sine_wave_1D_demonstration.png)

### 2D Interpolation

```julia
using ConvolutionInterpolations, Plots

xs = range(-π, π, length=5)
ys = range(-π, π, length=5)
zs = sin.(xs) .* sin.(ys')

itp_2d = convolution_interpolation((xs, ys), zs);

xs_fine = range(-π, π, length=100)
ys_fine = range(-π, π, length=100)

p1 = contourf(xs, ys, zs', title="Original Data")
p2 = contourf(xs_fine, ys_fine, itp_2d.(xs_fine, ys_fine')', title="Interpolated")
plot(p1, p2, layout=(1,2), size=(800,300))
```

[![2D random interpolation](fig/smooth_2D_interpolation.png)](fig/smooth_2D_interpolation.png)

### Smoothing Noisy Data

`convolution_smooth` applies separable Gaussian kernel smoothing to recover clean signals from noisy data. 
The kernel is normalized to sum to unity, so the mean signal level is preserved.
Works in any number of dimensions and is allocation-free in the inner loop.
The parameter `B` controls the Gaussian width: larger `B` means a narrower kernel (less smoothing),
smaller `B` means a wider kernel (more smoothing).
For effective smoothing `B ≤ 0.1` is recommended, as visible in the figure below.

```julia
using ConvolutionInterpolations, Plots

x = range(0.0, 2π, length=200)
y = range(0.0, 2π, length=200)
z_noisy = [cos(xi)*sin(yi) for xi in x, yi in y] .+ 0.1.*randn(200,200);

B_values = (0.02, 0.05, 0.1, 0.5, 1.0, 2.0)

# call as 'z_smooth = convolution_smooth((x,y), z_noisy, B)'
plots = [contourf(x, y, convolution_smooth((x,y), z_noisy, B)',
                  title="B=$B", colorbar=false, 
                  clims=(-1.1,1.1)) for B in B_values]

plot(plots..., layout=(2,3), size=(1200,700),
     plot_title="Gaussian smoothing with different B values")
```

[![Gaussian smoothing 2D](fig/gaussian_smooth_2d.png)](fig/gaussian_smooth_2d.png)

Smoothing is performed separably, one dimension at a time, making it fast even in high dimensions:

| Grid | B=0.02 | B=0.05 | B=0.1 |
|------|--------|--------|-------|
| 1D n=1000 | 22 μs | 13 μs | 9.6 μs |
| 2D 200² | 1.7 ms | 0.99 ms | 0.72 ms |
| 3D 50³ | 9.3 ms | 5.7 ms | 4.4 ms |
| 4D 40⁴ | 273 ms | 206 ms | 169 ms |

For point-wise Gaussian smoothing that also supports evaluation at arbitrary locations,
use `convolution_gaussian` directly:

```julia
itp = convolution_gaussian(x, y_noisy, 0.1)  # returns an interpolant
itp(1.5)                                       # evaluate anywhere
```

### Grid Resampling

`convolution_resample` resamples data from one uniform grid to another using high-order convolution interpolation.
It is significantly faster than constructing a full interpolant when only a new grid of values is needed.

```julia
using ConvolutionInterpolations, Plots

x_coarse = range(0.0, 2π, length=8)
y_coarse = range(0.0, 2π, length=8)
z_coarse = [sin(x)*cos(y) for x in x_coarse, y in y_coarse];

x_fine = range(0.0, 2π, length=200)
y_fine = range(0.0, 2π, length=200)
z_fine = convolution_resample((x_coarse, y_coarse), (x_fine, y_fine), z_coarse)

p1 = contourf(x_coarse, y_coarse, z_coarse', title="Coarse grid (8²)")
p2 = contourf(x_fine, y_fine, z_fine', title="Resampled (200²)")
plot(p1, p2, layout=(1,2), size=(800,350))
```

[![Grid resampling 2D](fig/resample_2d.png)](fig/resample_2d.png)

The default kernel is `:b9`, a 9th degree polynomial kernel with 7th order convergence.
Resampling is separable and allocation-free in the inner loop, with alloc count independent of grid size:

| Grid | Time |
|------|------|
| 1D 50→100 | 2.8 μs |
| 2D 30²→60² | 185 μs |
| 3D 20³→40³ | 2.8 ms |

Derivatives can be resampled simultaneously:

```julia
# resample and differentiate in one pass
z_dx = convolution_resample((x_coarse, y_coarse), (x_fine, y_fine), z_coarse;
                             kernel=:b9, derivative=(1, 0))  # d/dx only
```

Antiderivatives (`derivative=-1`) are not supported with resampling, use `convolution_interpolation` with `derivative=-1` instead.

### Scattered Data Pipeline

For truly scattered (unstructured) data, resample onto a uniform grid first using a radial basis functions (RBF) package such as [ScatteredInterpolation.jl](https://github.com/eljungsk/ScatteredInterpolation.jl), then apply ConvolutionInterpolations.jl for Gaussian smoothing, and high-order convolution interpolation:

```julia
using ScatteredInterpolation, ConvolutionInterpolations

# Scattered noisy data
points = rand(2, 500) .* 2π
values = sin.(points[1,:]) .* cos.(points[2,:]) .+ 0.05.*randn(500)

# Step 1: RBF onto uniform grid
itp_rbf = interpolate(Multiquadratic(), points, values)
xs = ys = range(0, 2π, length=100)
grid_values = [evaluate(itp_rbf, [x, y])[1] for x in xs, y in ys]

# Step 2: Remove noise from RBF artefacts and measurement noise
grid_smooth = convolution_smooth((xs, ys), grid_values, 0.05)

# Step 3: High-order interpolant — evaluate, differentiate, integrate
itp = convolution_interpolation((xs, ys), grid_smooth)           # f(x,y)
itp_dx = convolution_interpolation((xs, ys), grid_smooth; derivative=(1,0))  # ∂f/∂x
itp_int = convolution_interpolation((xs, ys), grid_smooth; derivative=-1)    # ∫∫f dx dy
```

This pipeline enables high-order evaluation, differentiation, and integration of noisy
scattered data in arbitrary dimensions. The three steps are independent and composable:
use RBF alone if data is clean, add smoothing if it is noisy, and choose any combination
of interpolation, derivatives, and antiderivatives for downstream analysis.

### Non-uniform Grid Interpolation

Non-uniform grids are detected automatically if an irregular vector of knots is passed.
All `:b` kernels support non-uniform grids with full 7th-order accuracy and derivatives up to 6th order.

```julia
x = [0.0, 0.15, 0.4, 0.7, 1.5, 2.5, 3.8, 4.2, 4.6, 4.8, 5.0,
     5.3, 5.6, 6.0, 6.5, 7.0, 7.3, 7.8, 8.2, 8.5, 9.0, 9.5, 10.0]
y = sin.(x) .* exp.(-x/5)

itp   = convolution_interpolation(x, y; kernel=:b11);                # f(x)
itp_d1 = convolution_interpolation(x, y; kernel=:b11, derivative=1);  # f'(x)
itp_d2 = convolution_interpolation(x, y; kernel=:b11, derivative=2);  # f''(x)
itp_d3 = convolution_interpolation(x, y; kernel=:b11, derivative=3);  # f'''(x)
itp_d4 = convolution_interpolation(x, y; kernel=:b11, derivative=4);  # f⁴(x)
itp_d5 = convolution_interpolation(x, y; kernel=:b11, derivative=5);  # f⁵(x)
```

[![Nonuniform derivatives](fig/nonuniform_derivatives.png)](fig/nonuniform_derivatives.png)

Dots are analytical values, solid lines are the `:b11` interpolant.
Non-uniform grids work in any number of dimensions via tensor products, though precomputation
cost scales with the number of intervals and grows rapidly in high dimensions. For
performance-critical or high-dimensional applications, consider constructing the non-uniform
interpolant once to evaluate onto a uniform grid, then using efficient uniform kernels for
repeated evaluation.

## Accuracy

### Runge Function Benchmark

The 1D Runge function demonstrates convergence behavior across kernel families:

[![Runge function convergence](fig/convergence_interpolation_1d_runge.png)](fig/convergence_interpolation_1d_runge.png)

- `:b` kernels reach machine precision (~10⁻¹⁴) by 1000 sample points
- `:b`-series kernels show 7th order convergence (slope ≈ -7 on the log-log plot)
- The `:a4` kernel show 4th order convergence, and is similar to cubic splines
- The discretized kernel approach is both faster and more numerically stable than direct polynomial evaluation
- Chebyshev interpolation shown for reference (requires non-uniform grid points)

### Frequency Response

The `:b`-series kernels approximate the ideal sinc interpolation kernel more closely than previously published convolution kernels, with flatter passbands and steeper stopband rolloff.

[![Kernels spectra](fig/FFT_kernels_spectra.png)](fig/FFT_kernels_spectra.png)

## Speed

Performance across dimensions and kernel families:

[![Kernel performance heatmap](fig/kernel_performance_comparison.png)](fig/kernel_performance_comparison.png)

**Initialization** (left panel): One-time setup cost in eager mode (`lazy=false`). Ranges from ~2 μs for linear kernels to ~0.5 s for 4D `:b` kernels. For kernels higher than `:a1`, setup time scales with the number of boundary points. Benchmarks use 50 grid points per dimension (50, 50², 50³, 50⁴). With `lazy=true`, construction is constant-time regardless of grid size or dimension — see [High-Dimensional Interpolation](#high-dimensional-interpolation) for benchmarks.

**Evaluation** (right panel): Cost per interpolation call with default settings (`:cubic` subgrid, extrapolation wrapper).
Lower times are achievable with lower order kernels, `:linear` subgrid or by bypassing the extrapolation wrapper (`itp.itp(x)`).

Evaluation cost scales as (stencil)ᴺ across dimensions due to tensor product structure.

### Resampling Performance

`convolution_resample` is faster than constructing a full interpolant and evaluating
at each output point, with the advantage growing with dimension. Note that the default
kernel for `convolution_resample` is `:b9` while `convolution_interpolation` defaults
to `:b5`, so resampling also provides higher accuracy by default:

| Grid | `convolution_resample` (:b9) | construct + eval (:b5) | Speedup |
|------|----------------------|------------------|---------|
| 1D 500→1000 | 22 μs | 29 μs | 1.3× |
| 2D 100²→200² | 1.9 ms | 7.0 ms | 3.6× |
| 3D 30³→60³ | 8.9 ms | 127 ms | 14× |

## Kernel Reference

### Available Kernels

Uniform grid kernels

| Kernel | Degree | Continuity | Derivative range | Convergence | stencil |
|--------|--------|------------|------------------|-------------|-----|
| `:a0`  | 0      | —          | -1..0            | 1st order   | 2ᴺ   |
| `:a1`  | 1      | C⁰         | -1..0            | 2nd order   | 2ᴺ   |
| `:a3`  | 3      | C¹         | -1..1            | ~3rd order  | 4ᴺ   |
| `:a4`  | 3      | C¹         | -1..1            | ~4th order  | 6ᴺ   |
| `:a5`  | 5      | C¹         | -1..1            | ~3rd order  | 6ᴺ   |
| `:a7`  | 7      | C¹         | -1..1            | ~3rd order  | 8ᴺ   |
| `:b5`  | 5      | C³         | -1..3            | 7th order   | 10ᴺ   |
| `:b7`  | 7      | C⁴         | -1..4            | 7th order   | 12ᴺ   |
| `:b9`  | 9      | C⁵         | -1..5            | 7th order   | 14ᴺ   |
| `:b11` | 11     | C⁶         | -1..6            | 7th order   | 16ᴺ   |
| `:b13` | 13     | C⁶         | -1..6            | 7th order   | 18ᴺ   |

Non-uniform grid kernels

| Kernel | Degree | Continuity | Derivative range | Convergence | stencil |
|--------|--------|------------|------------------|-------------|-----|
| `:a0`  | 0      | —          | —                | 1st order   | 2ᴺ  |
| `:a1`  | 1      | C⁰         | 0..0             | 2nd order   | 2ᴺ  |
| `:n3`  | 3      | C¹         | 0..0             | ~3rd order  | 4ᴺ  |
| `:b5`  | 5      | C³         | 0..3             | 7th order   | 10ᴺ |
| `:b7`  | 7      | C⁴         | 0..4             | 7th order   | 12ᴺ |
| `:b9`  | 9      | C⁵         | 0..5             | 7th order   | 14ᴺ |
| `:b11` | 11     | C⁶         | 0..6             | 7th order   | 16ᴺ |
| `:b13` | 13     | C⁶         | 0..6             | 7th order   | 18ᴺ |

The default kernel is `:b5`, which provides 7th-order accuracy and C³ continuity.
It works across all modes: uniform, non-uniform, derivatives, antiderivative, lazy, and
arbitrary dimensions. The non-uniform `:n3` kernel uses cubic weights equivalent to
non-uniform Catmull-Rom splines with ~3rd order convergence.

### Boundary Conditions

Control how ghost point values are computed near domain edges:

```julia
itp = convolution_interpolation(x, y; bc=:detect);        # Default
itp = convolution_interpolation(x, y; bc=:poly);   # Optimal for b-series
itp = convolution_interpolation(x, y; bc=:linear);
itp = convolution_interpolation(x, y; bc=:quadratic);
itp = convolution_interpolation(x, y; bc=:periodic);
```

The default `:detect` prioritizes `:poly`, which preserves each kernel's polynomial reproduction properties at domain edges.
It falls back to `:linear` when:

- There are insufficient grid points or
- if sign-changes or strong curvature near boundaries are detected.

Per-dimension and per-direction boundary conditions are supported:

```julia
bcs = [
    (:linear, :quadratic),   # First dimension: linear at start, quadratic at end
    (:periodic, :poly)     # Second dimension: periodic at start, polynomial at end
]
itp = convolution_interpolation((x, y), z; bc=bcs);
```

### Derivatives

Derivatives are computed through analytically differentiated kernel coefficients, providing stable results without step-size tuning.

```julia
x = range(0, 2π, length=100)
y = sin.(x)

itp_d1 = convolution_interpolation(x, y; derivative=1);  # cos(x)
itp_d2 = convolution_interpolation(x, y; derivative=2);  # -sin(x)
```

The maximum supported derivative order is determined by the kernel's continuity class (see [Kernel Reference](#available-kernels)).

In multiple dimensions, `derivative=1` applies the derivative kernel along all dimensions simultaneously, producing the mixed partial derivative:

```julia
# 2D mixed partial derivative ∂²f/∂x∂y
x = range(0, 2π, length=100)
y = range(0, 2π, length=100)
data = [sin(xi) * sin(yi) for xi in x, yi in y]
itp_d1 = convolution_interpolation((x, y), data; derivative=1);
itp_d1(1.0, 2.0)  # Returns cos(1.0) * cos(2.0)
```

[![Derivative convergence](fig/kernel_derivatives_1d_runge.png)](fig/kernel_derivatives_1d_runge.png)

Approximately one order of convergence is lost per derivative order.

Switching from `subgrid=:cubic` (default) to `subgrid=:linear` makes one additional derivative order available.

Per-dimension derivative orders are supported by passing a tuple to `derivative`:

```julia
# d/dx only (derivative order 1 in x, 0 in y)
itp = convolution_interpolation((x, y), data; derivative=(1, 0))

# d²/dx² in x, d/dy in y
itp = convolution_interpolation((x, y), data; derivative=(2, 1))
```

Per-dimension kernels are also supported, allowing different accuracy/cost tradeoffs per axis:

```julia
itp = convolution_interpolation((x, y), data; kernel=(:b9, :b5))
```

This works on both uniform and non-uniform grids. On non-uniform grids, each dimension
independently selects its kernel and derivative order, and the ghost point arrays are
padded per-dimension by each kernel's own stencil radius.

### Antiderivative

Pass `derivative=-1` to compute the indefinite integral of the interpolant:

```julia
x = range(0.0, 2π, length=100)
y = sin.(x)

itp = convolution_interpolation(x, y; derivative=-1);
itp(1.0)   # ≈ 0.4597  (= 1 - cos(1), anchored at x=0)
```

The result is zero-anchored at the leftmost interior knot: `F(anchor) = 0` exactly.
The fast path (default) precomputes the anchor-side contribution at construction time.
In 1D, 2D and 3D, prefix sums further reduce evaluation to O(stencil), O(stencil²) and and O(stencil³)
respectively, independent of grid size after construction. In higher dimensions, evaluation falls back to
a general path that sums over the full coefficient array — accurate but slower for large grids.

In N dimensions the result is the iterated antiderivative:

```julia
xs = range(0.0, 1.0, length=50)
ys = range(0.0, 1.0, length=50)
vs = [x + y for x in xs, y in ys]

itp2 = convolution_interpolation((xs, ys), vs; derivative=-1);
itp2(0.5, 0.5)   # ≈ ∫₀^0.5 ∫₀^0.5 (x+y) dx dy = 0.125
```

Antiderivative support is available for all uniform-grid kernels.
It is not yet supported on non-uniform grids. 

Mixed antiderivative/derivative orders are supported on uniform grids via a tuple,
enabling e.g. Leibniz-rule-style operations — integrate in one variable, differentiate in another:

```julia
xs = range(0.0, 2π, length=100)
ys = range(0.0, 2π, length=100)
vs = [sin(x) * cos(y) for x in xs, y in ys]

# ∂/∂x ∫ f dy  (differentiate x, integrate y)
itp = convolution_interpolation((xs, ys), vs; kernel=:b7, derivative=(1, -1))
itp(1.0, 2.0)  # ≈ cos(1.0) * sin(2.0)
```

The figure below shows convergence of the antiderivative in 1D of Runge's function:

[![Antiderivative convergence](fig/convergence_integration_1d_runge.png)](fig/convergence_integration_1d_runge.png)

### Discontinuous Antiderivatives

The `:a0` kernel supports antiderivatives of piecewise constant and discontinuous functions.
A delta-like spike integrates to a Heaviside step:

```julia
using ConvolutionInterpolations, Plots

n = 1001
x = range(0.0, 2π, length=n)
h = x[2] - x[1]
y = [abs(xi - π) ≈ 0.0 ? 1.0/h : 0.0 for xi in x]

itp = convolution_interpolation(x, y; kernel=:a0, derivative=-1)

x_fine = range(0.0, 2π, length=500)
p1 = plot(x, y, title="Delta-like spike", lw=2, label="f(x)")
p2 = plot(x_fine, itp.(x_fine), title="∫f dx ≈ Heaviside", lw=2, label="F(x)")
plot!(x_fine, [xi < π ? 0.0 : 1.0 for xi in x_fine], lw=2, ls=:dash, label="exact")
plot(p1, p2, layout=(1,2), size=(900,400))
```

[![Delta spike and Heaviside](fig/delta_heaviside.png)](fig/delta_heaviside.png)

### Multidimensional Integration and Mixed Operations

For general smooth integrands, the antiderivative converges at approximately the 1D rate (~7th order for `:b` kernels) across all dimensions.

Beyond pure antiderivatives, mixed integral/derivative operations are supported on uniform grids - integrate along some dimensions while differentiating or interpolating along others, all in a single O(stencilᴺ) operator:

```julia
xs = range(0.0, 2π, length=100)
ys = range(0.0, 2π, length=100)
zs = range(0.0, 2π, length=100)
vs = [sin(x)*cos(y)*exp(z/4) for x in xs, y in ys, z in zs]

# antiderivative in x, derivative in y, interpolation in z
itp = convolution_interpolation((xs,ys,zs), vs; derivative=(-1, 1, 0))

# definite integrals via differences cancel the unknown constant:
x1, x2 = 1.0, 2.0
itp(x2, 1.5, 1.0) - itp(x1, 1.5, 1.0)  # ≈ (-cos(x2)+cos(x1)) * (-sin(1.5)) * exp(1.0/4)
```

The fast path uses tail sums to reduce the integral dimensions to O(1) lookups per axis, so evaluation cost depends only on the stencil width of the derivative/interpolation dimensions, independent of grid size. Up to 3 integral dimensions are accelerated this way; beyond that the package falls back to a general path that is accurate but slower for large grids.

This enables compact expressions for operators like the Leibniz integral rule, iterated integrals, and mixed differential-integral equations in arbitrary dimensions.

### Extrapolation

Define behavior outside the data domain:

```julia
itp = convolution_interpolation(x, y; extrap=Throw());     # Error (default)
itp = convolution_interpolation(x, y; extrap=Line());      # Linear
itp = convolution_interpolation(x, y; extrap=Flat());      # Constant
itp = convolution_interpolation(x, y; extrap=Natural());   # Smooth boundary preservation
```

The `Natural()` mode transforms extrapolation into interpolation by expanding the domain with boundary coefficients before applying linear extrapolation. This preserves the kernel's full smoothness across the boundary region, rather than abruptly transitioning at the domain edge.
`Natural()` is not recommended in high dimensions due to the double construction cost.

### High-Dimensional Interpolation

The separable kernel design scales to arbitrary dimensions.
For large three-dimensional grids or any four-dimensional and higher data, `lazy=true` is recommended,
as construction is constant-time (~0.12 ms) regardless of grid size; eager scales with the number of boundary points.

#### Construction timings with `lazy=true` (kernel `:b7`, warm start)

| Grid | eager | lazy | speedup |
|------|------:|-----:|--------:|
| 1D 100 | 2.7 μs | 1.6 μs | 1.7× |
| 2D 100² | 81 μs | 2.3 μs | 36× |
| 3D 100³ | 21 ms | 0.42 ms | 49× |
| 4D 30⁴ | 79 ms | 0.59 ms | 130× |
| 5D 12⁵ | 280 ms | 1.2 ms | 240× |

Lazy construction cost is essentially independent of dimensionality (~1 ms from 3D onward),
while eager scales with the number of ghost points, which grows
explosively with dimension and kernel width. For wider kernels than `:b7` the eager cost is even more
extreme, making `lazy=true` increasingly attractive.

#### Lazy mode constraints

`lazy=true` has two boundary modes controlled by `boundary_fallback`:

**`boundary_fallback=false` (default, N≤3 only)**: full ghost point resolution at boundaries,
identical accuracy and derivative support to eager mode. If near-boundary or corner evaluations are frequent, prefer `lazy=false` (eager
mode) instead, construction cost is quickly amortized.

**`boundary_fallback=true` (required for N≥4)**: near-boundary evaluation uses a linear
kernel — fast, allocation-free, and correct throughout the domain, at the cost of reduced
smoothness in the boundary stencil region. Per-dimension mixed kernels are supported;
derivatives are not.

The `:n3` kernel additionally supports lazy evaluation on nonuniform grids.

Interior evaluation is identical in speed and accuracy to eager mode in both boundary modes.

```julia
# boundary_fallback=false (default): full accuracy, derivatives supported (N≤3)
itp = convolution_interpolation((x, y, z), data; kernel=:b7, derivative=(1, 0, 2), lazy=true)

# boundary_fallback=true: per-dim kernels, no derivatives, required for N≥4
itp = convolution_interpolation((x, y, z, t), data; kernel=(:a3,:a3,:b5,:b7), lazy=true, boundary_fallback=true)
```

For full boundary accuracy with mixed kernels in any dimension, use `lazy=false` (eager mode).

### Subgrid Interpolation

The fast evaluation mode convolves data with precomputed kernel values, then interpolates between convolution results. 
The `subgrid` parameter controls this inner interpolation:

```julia
itp = convolution_interpolation(x, y; subgrid=:cubic);                    # Default, high accuracy
itp = convolution_interpolation(x, y; subgrid=:quintic);                  # Even higher accuracy
itp = convolution_interpolation(x, y; subgrid=:linear, precompute=10_000); # Fastest evaluation
```

| Subgrid    | Method               | Kernel derivatives used | Speed   | Accuracy |
|------------|----------------------|------------------------|---------|----------|
| `:linear`  | Linear interpolation | 0                      | Fastest | Lowest   |
| `:cubic`   | Cubic Hermite        | 1                      | Middle  | High     |
| `:quintic` | Quintic Hermite      | 2                      | Slowest | Highest  |

Cubic and quintic subgrids use analytically predifferentiated kernels for Hermite interpolation, achieving high accuracy with far fewer precomputed points than linear subgrid requires. The default `:cubic` with `precompute=101` uses pre-shipped kernel tables requiring zero computation at startup. For linear subgrid, increase `precompute` to at least 10,000.

The available subgrid order depends on remaining smooth derivatives: `max_derivative[kernel] - derivative`. For example, b5 with `derivative=3` has no remaining derivatives, so only `:linear` is available.

Cubic and quintic subgrids are implemented for 1D and 2D. Higher dimensions use `:linear` subgrid, as tensor products per cell grow as `(order+1)ᴺ`.

Benchmarks in the [Speed](#speed) section use the default `:cubic` subgrid.

## Performance Guidelines

- **Default `:b5` works everywhere**: 7th-order accuracy on uniform grids, non-uniform grids, high-order derivatives
- **Use `lazy=true` in high dimensions**: Skips ghost point expansion, reducing construction time and memory
- **Use `:a0`, `:a1` or `:a3` in high dimensions**: Evaluation time of narrower kernels scale better with dimensions
- **Pre-shipped kernel tables**: The default `precompute=101` with `:cubic` or `:quintic` subgrid loads precomputed constants
- **Orthogonal grids assumption**: The separable kernel design requires mutually orthogonal grid axes.
- **Use `convolution_resample` for grid-to-grid operations**: Faster than construct+eval, exploits separability
- **Use `convolution_smooth` before interpolating noisy data**: Separable Gaussian smoothing with B ≤ 0.1 recommended

## Technical Background

This package introduces five main contributions:

**b-series kernel family.** A new family of high-order convolution kernels (b5, b7, b9, b11, b13) discovered through systematic analytical search using symbolic computation (SymPy), generalizing the approach of R. G. Keys (1981). All b-series kernels achieve 7th order convergence. Kernel coefficients are stored as exact rational numbers, enabling extended precision arithmetic with BigFloat.

**Hermite multilevel interpolation.** Rather than evaluating kernel polynomials directly, kernels are discretized at a small number of points (default 101) and shipped as package constants. Higher resolutions or non-standard precisions are computed on demand and cached to disk via Scratch.jl. During evaluation, data is convolved with these precomputed values, and the results are interpolated using cubic or quintic Hermite subgrid interpolation. This approach is both faster (`O(1)`) and more numerically stable than direct polynomial evaluation.

**Polynomial boundary conditions.** A boundary handling method that computes optimal ghost point values that preserve each kernel's polynomial reproduction properties. This maintains convergence order across the entire domain rather than degrading near boundaries.

**Non-uniform b-kernel extension.** On non-uniform grids, each interval requires its own weights adapted to the local grid geometry. This is achieved by expanding the kernel in a binomial series around each interval's local coordinate, then projecting through a Vandermonde system to enforce polynomial reproduction up to the kernel's design order. The result is a compact set of polynomial coefficients per interval, evaluated via Horner's method at query time. All weight generation uses exact `Rational{BigInt}` arithmetic to avoid floating-point contamination, with conversion to `Float64` only at the final storage step. The same framework extends to derivatives by applying the binomial expansion to analytically differentiated kernel coefficients. On mixed grids (some dimensions uniform, some not), each dimension independently
selects the appropriate path through the separable tensor product. Per-dimension kernels are supported on both uniform and non-uniform grids: ghost point arrays are padded per-dimension by each kernel's own stencil radius, and derivative scaling factors are applied independently per axis.

**Antiderivative via kernel integration.** For `derivative=-1`, each kernel `K` is
analytically integrated to produce `K̃`, the antiderivative kernel, with coefficients
stored as exact rational numbers alongside the derivative kernel tables. The interpolant
antiderivative is then `F(x) = h · Σⱼ cⱼ · [K̃((x − xⱼ)/h) − K̃((anchor − xⱼ)/h)]`,
where `anchor` is the leftmost interior knot and the subtracted term enforces
`F(anchor) = 0`. In the fast path, the anchor-side sum `Σⱼ cⱼ · K̃((anchor − xⱼ)/h)`
is computed once at construction (using exact arithmetic for b-series kernels, converted
to Float64 on completion) and stored as `left_values`.
At evaluation time only the `K̃(x)` half is computed per dimension. In 1D, 2D, and 3D,
additional prefix sums reduce the tail contributions to O(1) lookups, giving O(stencil),
O(stencil²) and O(stencil³) total evaluation independent of grid size. In higher dimensions (4D+) the tail
contributions are summed directly over the full coefficient array, accurate but slower
for large grids. In N dimensions the antiderivative is the tensor product of
per-dimension antiderivatives, with `left_values` stored per dimension.

## Comparison with Other Packages

Key differences from existing interpolation packages:

- `:b`-series kernels with 7th order convergence on uniform grids
- Automatic non-uniform grid support in arbitrary dimensions
- Persistent kernel caching for near-instant subsequent initialization
- Hermite multilevel interpolation for combined speed and stability
- Single interface from nearest-neighbor to 13th-degree kernels
- Minimal dependencies (LinearAlgebra, Serialization, Scratch.jl)
- Separable Gaussian smoothing for noisy data in any number of dimensions
- Grid-to-grid resampling faster than construct+eval, with alloc count independent of grid size
- Mixed interpolation, differentiation, and integration in a single operator across arbitrary dimensions

## Acknowledgments

ConvolutionInterpolations.jl draws significant inspiration from [Interpolations.jl](https://github.com/JuliaMath/Interpolations.jl) in terms of API design and interface patterns. This package originally started as a PR for Interpolations.jl before evolving into a standalone implementation.

The theoretical foundation builds on:

- R. G. Keys, "Cubic Convolution Interpolation for Digital Image Processing," *IEEE Trans. Acoustics, Speech, and Signal Processing*, vol. 29, no. 6, 1981.
- E. H. W. Meijering, K. J. Zuiderveld, and M. A. Viergever, "Image Reconstruction by Convolution with Symmetrical Piecewise nth-Order Polynomial Kernels," *IEEE Trans. Image Processing*, vol. 8, no. 2, 1999.

## Declaration of AI Assistance

Parts of this package and its documentation were developed with assistance from Claude (Anthropic). All code, methods, and scientific content have been verified and validated by the author.