# ConvolutionInterpolations.jl

A high-performance Julia package for smooth N-dimensional interpolation on uniform grids using separable convolution kernels.

[![Performance Comparison](fig/convolution_interpolation_a_and_b.png)](fig/convolution_interpolation_a_and_b.png)

## Table of Contents

- [Why ConvolutionInterpolations.jl?](#why-convolutioninterpolationsjl)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [1D Interpolation](#1d-interpolation)
  - [2D Interpolation](#2d-interpolation)
- [Performance](#performance)
  - [Accuracy: Runge Function Benchmark](#accuracy-runge-function-benchmark)
  - [Frequency Response of Convolution Kernels](#frequency-response-of-convolution-kernels)
  - [Speed: Initialization and Evaluation](#speed-initialization-and-evaluation)
- [Advanced Usage](#advanced-usage)
  - [Kernel Selection](#kernel-selection)
  - [Kernel Boundary Conditions](#kernel-boundary-conditions)
  - [Derivatives](#derivatives)
  - [Extrapolation Methods](#extrapolation-methods)
  - [High-Dimensional Interpolation](#high-dimensional-interpolation)
- [Performance Optimization](#performance-optimization)
  - [General Guidelines](#general-guidelines)
  - [Subgrid Interpolation](#subgrid-interpolation)
- [Comparison with Other Packages](#comparison-with-other-packages)
- [Technical Background](#technical-background)
- [Novel Contributions](#novel-contributions)
- [Acknowledgments](#acknowledgments)
- [License](#license)

## Why ConvolutionInterpolations.jl?

**Uniform Grids**: Designed specifically for uniformly-spaced data. Grid spacing can vary between dimensions, but must be uniform within each dimension.

**Performance**: Achieves true O(1) interpolation time independent of grid size through elimination of binary search operations. Novel discretized kernel approach with linear interpolation provides both speed and numerical stability. Allocation-free evaluation for maximum performance.

**Accuracy**: Novel b-series kernels (b5, b7, b9, b11, b13) discovered through systematic analytical search. The b5 kernel provides dramatically better accuracy than cubic splines at comparable computational cost - achieving 7th order convergence with quintic polynomial reproduction.

**Simplicity**: Minimal dependencies (LinearAlgebra, Serialization, Scratch for kernel caching). Clean API that scales naturally from 1D to arbitrary dimensions using separable convolution kernels.

## Features

- **N-dimensional interpolation**: Handles 1D to arbitrary dimensions using separable convolution kernels
- **Multiple kernel options**: From nearest neighbor to 13th-degree polynomials
- **Two kernel families**:
  - `:a` kernels: Literature-based kernels with minimal boundary handling
  - `:b` kernels: Novel high-accuracy kernels with 7th order convergence (b5 is default)
- **Flexible extrapolation**: Linear, flat, natural, periodic, and reflection boundary conditions
- **O(1) performance**: True constant-time interpolation independent of grid size
- **Performance optimized**: Precomputed kernels eliminate polynomial evaluation overhead
- **Uniform grids**: Sample data must be uniformly spaced in each dimension (spacing can vary between dimensions)
- **Optional smoothing**: Gaussian convolution kernel for noisy data
- **Accurate higher order derivatives**: Convolution kernels offer up to 5 smooth derivatives

## Installation

```julia
using Pkg
Pkg.add("ConvolutionInterpolations")
```

## Quick Start

### 1D Interpolation

Interpolate a sine wave with just 3-4 samples per period:

```julia
using ConvolutionInterpolations
using Plots

# Sparse sampling: 3 samples per period
x = range(0, 2π, length=4)
y = sin.(x)
itp = convolution_interpolation(x, y)

x_fine = range(0, 2π, length=200)
p1 = plot(x_fine, sin.(x_fine), label="True function: sin(x)")
scatter!(p1, x, y, label="3 samples per period")
plot!(p1, x_fine, itp.(x_fine), label="Interpolated (3 samples)")

# Slightly denser: 4 samples per period
x = range(0, 2π, length=5)
y = sin.(x)
itp = convolution_interpolation(x, y)

p2 = plot(x_fine, sin.(x_fine), label="True function: sin(x)")
scatter!(p2, x, y, label="4 samples per period")
plot!(p2, x_fine, itp.(x_fine), label="Interpolated (4 samples)")

plot(p1, p2, layout=(1,2), size=(800,300), dpi=1000)
```

[![1D sine wave interpolation](fig/simple_sine_wave_1D_demonstration.png)](fig/simple_sine_wave_1D_demonstration.png)

### 2D Interpolation

Smooth interpolation of random data:

```julia
using Random
using ConvolutionInterpolations
using Plots

Random.seed!(123)
xs = range(-2, 2, length=10)
ys = range(-2, 2, length=10)
zs = rand(10, 10)

itp_2d = convolution_interpolation((xs, ys), zs)

xs_fine = range(-2, 2, length=100)
ys_fine = range(-2, 2, length=100)

p1 = contourf(xs, ys, zs', title="Original Data")
p2 = contourf(xs_fine, ys_fine, itp_2d.(xs_fine, ys_fine')', title="Interpolated")
plot(p1, p2, layout=(1,2), size=(800,300), dpi=1000)
```

[![2D random interpolation](fig/Smooth_random_2D_interpolation.png)](fig/Smooth_random_2D_interpolation.png)

## Performance

ConvolutionInterpolations.jl delivers exceptional performance across dimensions while maintaining high accuracy.

### Accuracy: Runge Function Benchmark

The challenging 1D Runge function demonstrates the superior convergence of the b-series kernels:

[![Runge function convergence](fig/interpolation_1d_runge.png)](fig/interpolation_1d_runge.png)

**Key observations:**
- **b5 (fast)** achieves machine precision (~10⁻¹⁴) by 1000 sample points
- All b-series kernels significantly outperform cubic splines
- The "fast" implementation (with discretized kernels) exceeds the "slow" (direct polynomial evaluation) in accuracy
- **7th order convergence verified**: The b-series kernels show a slope of approximately -7 on this log-log plot in their linear convergence region, confirming the 7th order accuracy derived from Taylor series analysis
- Cubic splines show the expected 4th order convergence (slope ~-4)
- Chebyshev interpolation shown for reference (requires non-uniform grid points)

### Frequency Response of Convolution Kernels

The b-series kernels are significantly more sinc-like than previously published convolution kernels, with flatter passbands and steeper stopband rolloff. This more faithful approximation of the ideal interpolation kernel explains their 7th order convergence.

[![Kernels spectra](fig/FFT_kernels_spectra.png)](fig/FFT_kernels_spectra.png)

### Speed: Initialization and Evaluation

Performance across dimensions and available kernels:

[![Kernel performance heatmap](fig/kernel_performance_comparison.png)](fig/kernel_performance_comparison.png)

**Initialization time** (left panel): One-time cost for setting up the interpolator
- Dominated by boundary condition computation
- Linear kernels (a0, a1) initialize in ~76-78 μs
- Higher-order kernels scale with kernel complexity and boundary handling
- 4D b13 initialization: ~91 ms (still practical for most applications)

**Evaluation time** (right panel): Cost per interpolation
- O(1) scaling with respect to grid size - independent of number of sample points
- All fast kernel evaluations are allocation-free for optimal performance
- 1D b5: 15 ns per evaluation
- 2D b5: 88 ns per evaluation  
- 3D b5: 1.8 μs per evaluation
- Tensor product scaling: evaluation cost grows as (2×eqs)^N across dimensions

The b5 kernel offers the best accuracy-to-performance ratio for most applications.

## Advanced Usage

### Kernel Selection

Choose kernels based on your accuracy and smoothness requirements:

```julia
# Default: b5 kernel (excellent accuracy/performance balance)
itp = convolution_interpolation(x, y)  # degree=:b5, C3 continuous, 7th order accurate

# Low-order kernels
itp = convolution_interpolation(x, y; degree=:a0)  # Nearest neighbor
itp = convolution_interpolation(x, y; degree=:a1)  # Linear (C0 continuous)
itp = convolution_interpolation(x, y; degree=:a3)  # Cubic (C1 continuous)
itp = convolution_interpolation(x, y; degree=:a4)  # Cubic (C1 continuous)

# Higher-order a-series kernels (less boundary handling than b-series)
itp = convolution_interpolation(x, y; degree=:a5)  # Quintic
itp = convolution_interpolation(x, y; degree=:a7)  # Septic

# b-series kernels (superior accuracy, 7th order convergence)
itp = convolution_interpolation(x, y; degree=:b5)  # Quintic (C3 continuous, 7th order accurate)
itp = convolution_interpolation(x, y; degree=:b7)  # Septic (C4 continuous, 7th order accurate)
itp = convolution_interpolation(x, y; degree=:b9)  # 9th degree (C5 continuous, 7th order accurate)
itp = convolution_interpolation(x, y; degree=:b11) # 11th degree (C6 continuous, 7th order accurate)
itp = convolution_interpolation(x, y; degree=:b13) # 13th degree (C6 continuous, 7th order accurate)

# Gaussian smoothing (does not hit data points exactly)
itp = convolution_interpolation(x, y; B=1.0)  # B controls smoothing width
```

**Performance note**: The `:b` kernels (b5, b7, b9, b11, b13) all achieve 7th order convergence from a Taylor series perspective. For polynomial reproduction: b5 reproduces quintics exactly, b7 reproduces septics (degree 7) exactly, while higher order kernels cannot reproduce their own degrees. The b5 kernel represents the optimal balance - achieving excellent accuracy at reasonable computational cost, making it the default choice. The b7 kernel is optimal if exact polynomial reproduction up to degree 7 is required.

### Kernel Boundary Conditions

Control interpolation behavior near boundaries for optimal accuracy:

```julia
# Automatic (default, prioritizes :polynomial with fallback to :detect)
itp = convolution_interpolation(x, y; kernel_bc=:auto)

# Polynomial reproduction (optimal, requires sufficient grid points)
itp = convolution_interpolation(x, y; kernel_bc=:polynomial)

# Automatic detection (detects periodic signals, adaptive fallback)
itp = convolution_interpolation(x, y; kernel_bc=:detect)

# Manual selection
itp = convolution_interpolation(x, y; kernel_bc=:linear)    # Linear signals
itp = convolution_interpolation(x, y; kernel_bc=:quadratic) # Quadratic signals
itp = convolution_interpolation(x, y; kernel_bc=:periodic)  # Periodic signals
```

**Note**: The default `:auto` option prioritizes the `:polynomial` boundary condition (a novel contribution that computes optimal ghost point values by solving Vandermonde systems to preserve each kernel's polynomial reproduction properties), automatically falling back to `:detect` when there are insufficient grid points. The `:detect` option includes sophisticated periodic signal detection using precomputed linear predictors mixed with quadratic methods. The `:polynomial` method ensures b-series kernels maintain their 7th order convergence all the way to domain edges.

Per-dimension boundary conditions:

```julia
x = range(0, 1, length=10)
y = range(0, 1, length=10)
z = [sin(2π*xi) for xi in x, yi in y]

# Different boundary conditions for each dimension and boundary
kernel_bcs = [
    (:linear, :quadratic),  # First dimension: linear at start, quadratic at end
    (:periodic, :detect)    # Second dimension: periodic at start, auto-detect at end
]
itp = convolution_interpolation((x, y), z; kernel_bc=kernel_bcs)
```

### Derivatives

Compute derivatives of interpolated functions using predifferentiated convolution kernels.
Unlike finite difference methods, derivatives are computed through analytically differentiated kernel coefficients, providing stable and accurate results without step-size tuning.

```julia
# 1D first derivative
x = range(0, 2π, length=100)
y = sin.(x)
itp_d1 = convolution_interpolation(x, y; derivative=1)
itp_d1(1.0)  # Returns cos(1.0) ≈ 0.5403

# Higher-order derivatives
itp_d2 = convolution_interpolation(x, y; derivative=2)  # -sin(x)
itp_d3 = convolution_interpolation(x, y; derivative=3)  # -cos(x)
```

The maximum supported derivative order depends on the kernel's continuity class:

| Kernel | Continuity | Max derivative |
|--------|-----------|----------------|
| `:b5`  | C³        | 3              |
| `:b7`  | C⁴        | 4              |
| `:b9`  | C⁵        | 5              |
| `:b11` | C⁶        | 6              |
| `:b13` | C⁶        | 6              |

In multiple dimensions, `derivative=1` applies the derivative kernel along all dimensions simultaneously, producing the mixed partial derivative:

```julia
# 2D mixed partial derivative ∂²f/∂x∂y
x = range(0, 2π, length=100)
y = range(0, 2π, length=100)
data = [sin(xi) * sin(yi) for xi in x, yi in y]
itp_d1 = convolution_interpolation((x, y), data; derivative=1)
itp_d1(1.0, 2.0)  # Returns cos(1.0) * cos(2.0)
```

[![Derivative convergence](fig/kernel_derivatives_1d_runge.png)](fig/kernel_derivatives_1d_runge.png)

Approximately 1 order of convergence rate is lost per derivative, allowing the 7th order b-kernels to converge reliably for higher derivatives.
The predifferentiated kernel approach means derivative accuracy is limited only by the grid resolution, not by any finite difference approximation.

### Extrapolation Methods

Define behavior outside the data domain:

```julia
# Error on out-of-bounds (default)
itp = convolution_interpolation(x, y; extrapolation_bc=Throw())

# Linear extrapolation
itp = convolution_interpolation(x, y; extrapolation_bc=Line())

# Constant (flat) extrapolation
itp = convolution_interpolation(x, y; extrapolation_bc=Flat())

# Periodic extension
itp = convolution_interpolation(x, y; extrapolation_bc=Periodic())

# Reflection at boundaries
itp = convolution_interpolation(x, y; extrapolation_bc=Reflect())
```

#### Natural Extrapolation: High-Order Boundary Preservation

The `Natural()` extrapolation mode provides the smoothest boundary behavior by leveraging your chosen kernel's boundary coefficients. Rather than abruptly transitioning to linear extrapolation at the domain edge, it:

1. Applies your kernel's boundary condition (e.g., `:polynomial`) to the original domain
2. Takes the resulting coefficients and wraps them in a new interpolator with `:linear` boundary conditions
3. Finally applies `Line()` extrapolation to the expanded domain

This approach transforms extrapolation into interpolation. The key insight: your original kernel's smoothness is fully preserved because you're interpolating through the boundary region rather than extrapolating across it. A b5 kernel maintains its C³ continuity, a b7 kernel maintains C⁵, and so on - all the way through what would traditionally be considered the "extrapolation" region.
```julia
# Compare extrapolation quality
using ConvolutionInterpolations
x = range(0, 2π, length=20)
y = sin.(x)

# Standard linear extrapolation - kernel smoothness lost at boundary
itp_linear = convolution_interpolation(x, y; extrapolation_bc=Line())

# Natural extrapolation - kernel smoothness preserved across boundary
itp_natural = convolution_interpolation(x, y; extrapolation_bc=Natural())

# Evaluate beyond original domain edge
x_test = 2π + 0.1
itp_natural(x_test)  # Maintains full kernel smoothness (C³ for b5, C⁵ for b7, etc.)
```

#### Advanced: Manual Extrapolation Nesting

For complete control over boundary behavior, you can manually nest multiple interpolators using the package's compositional design. Each nesting layer expands the domain according to its kernel's equation count (eqs), but only the final kernel is used for actual evaluation.

**Note**: Manual nesting uses the struct constructors (`ConvolutionInterpolation`, `ConvolutionExtrapolation`) directly, not the convenience function `convolution_interpolation`.

**Domain expansion by kernel degree:**
```julia
:a0  => eqs=1  → expands by 0 nodes per side (nearest neighbor)
:a1  => eqs=1  → expands by 0 nodes per side (linear)
:a3  => eqs=2  → expands by 1 node per side (cubic)
:a5  => eqs=3  → expands by 2 nodes per side (quintic)
:a7  => eqs=4  → expands by 3 nodes per side (septic)
:a4  => eqs=3  → expands by 2 nodes per side (cubic)
:b5  => eqs=5  → expands by 4 nodes per side (quintic)
:b7  => eqs=6  → expands by 5 nodes per side (septic)
:b9  => eqs=7  → expands by 6 nodes per side (nonic)
:b11 => eqs=8  → expands by 7 nodes per side (11th degree)
:b13 => eqs=9  → expands by 8 nodes per side (13th degree)
```

Each layer adds (eqs-1) boundary nodes per side, where the `kernel_bc` parameter controls how those coefficient values are computed:
```julia
# Start with your data
x = range(0, 1, length=20)
y = sin.(2π * x)

# Layer 1: Add 4 boundary nodes per side using b5 with polynomial boundary conditions
itp1 = ConvolutionInterpolation(x, y; degree=:b5, kernel_bc=:polynomial)

# Layer 2: Add 2 boundary nodes per side using b3 with quadratic boundary conditions
# Uses itp1.coefs (which now includes the first layer's boundary nodes)
itp2 = ConvolutionInterpolation(itp1.knots, itp1.coefs; degree=:a4, kernel_bc=:quadratic)

# Layer 3: Add 5 boundary nodes per side using b7 with linear boundary conditions
itp3 = ConvolutionInterpolation(itp2.knots, itp2.coefs; degree=:b7, kernel_bc=:linear)

# Final extrapolation layer
extrap = ConvolutionExtrapolation(itp3, Line())

# Alternatively, call itp3 directly without the extrapolation wrapper
value = itp3(1.5)  # Evaluates using only the final (b7) kernel
```

**Key principle**: 

- **Intermediate kernels** (`degree`) determine how many boundary nodes are added at each layer (eqs-1 per side)
- **Intermediate boundary conditions** (`kernel_bc`) determine the coefficient values for those added nodes
- **Final kernel** performs all actual interpolation evaluation across the expanded domain
- **Final extrapolation** (optional) defines behavior beyond the fully-expanded domain

#### Per-Dimension and Per-Direction Boundary Control

The `kernel_bc` parameter accepts either:
- A single `Symbol` applied to all dimensions and boundaries: `:polynomial`, `:linear`, `:quadratic`, `:periodic`, `:detect`, `:auto`
- A `Vector` of tuples, one per dimension, each tuple specifying `(left_bc, right_bc)` for that dimension's boundaries
```julia
x = range(0, 1, length=20)
y = range(0, 1, length=20)
z = [sin(2π*xi) * cos(2π*yi) for xi in x, yi in y]

# Different boundary conditions per dimension and direction
kernel_bcs = [
    (:periodic, :linear),     # X-dimension: periodic at left, linear at right
    (:polynomial, :quadratic) # Y-dimension: polynomial at left, quadratic at right
]

itp = ConvolutionInterpolation((x, y), z; degree=:b5, kernel_bc=kernel_bcs)
```

Multi-stage nesting with per-dimension control:
```julia
# Stage 1: X-dimension gets periodic boundaries (adds 4 nodes per side with b5)
itp1 = ConvolutionInterpolation((x, y), z; degree=:b5, 
                                 kernel_bc=[(:periodic, :periodic), (:polynomial, :polynomial)])

# Stage 2: Expand further with asymmetric boundaries (adds 2 nodes per side with b3)
itp2 = ConvolutionInterpolation(itp1.knots, itp1.coefs; degree=:a4,
                                 kernel_bc=[(:linear, :quadratic), (:periodic, :linear)])

# Stage 3: Final high-order kernel for smooth evaluation (adds 5 nodes per side with b7)
itp3 = ConvolutionInterpolation(itp2.knots, itp2.coefs; degree=:b7,
                                 kernel_bc=:polynomial)

# Optional final extrapolation wrapper
extrap = ConvolutionExtrapolation(itp3, Line())
```

This composability allows you to craft extrapolation strategies that match your problem's physics:

- **Periodic systems**: Use `:periodic` boundaries with seamless wrapping
- **Symmetric problems**: Stage multiple layers to build symmetric transition zones
- **Far-field behavior**: Stage transitions from `:polynomial` → `:linear` → `Line()` for smooth far-field decay, where the linear extrapolation naturally continues the linear boundary layer
- **Anisotropic domains**: Different strategies per dimension for problems with distinct behavior in each direction
- **Staged expansion**: Build up boundary zones layer by layer, each with its own expansion size and coefficient behavior

**Performance note**: The `Natural()` extrapolation mode is a pre-packaged version of this nesting strategy (original `kernel_bc` → `:linear` → `Line()`), optimized for the common case where you want smooth boundary preservation without manual configuration. Manual nesting adds even more boundary nodes through multiple expansion stages, making both approaches most practical in lower dimensions (1D-3D) where boundary node counts remain manageable. In higher dimensions (4D+), simpler extrapolation modes like `Line()` or `Flat()` are preferred to avoid the exponential growth in boundary handling overhead.

### High-Dimensional Interpolation

Seamlessly handle data of arbitrary dimensionality:

```julia
# 4D interpolation (e.g., time-evolving 3D scalar field)
x = range(0, 1, length=10);
y = range(0, 1, length=10);
z = range(0, 1, length=10);
t = range(0, 1, length=5);

data_4d = [sin(2π*xi)*cos(2π*yi)*exp(-zi)*sqrt(ti) 
           for xi in x, yi in y, zi in z, ti in t];

itp_4d = convolution_interpolation((x, y, z, t), data_4d);

# Evaluate at arbitrary point
value = itp_4d(0.42, 0.33, 0.77, 0.51)
```

The separable kernel approach scales naturally to 5D, 6D, and beyond.

## Performance Optimization

### General Guidelines

1. **Default to `:b5`**: Optimal balance of accuracy and computational cost with 7th order convergence
2. **Higher `:b` kernels offer smoothness**: b5 (C³), b7 (C⁴), b9 (C⁵), b11 (C⁶), b13 (C⁶) all maintain 7th order convergence while providing increasingly smooth derivatives
3. **All kernels scale O(1)**: Interpolation time is independent of grid size thanks to elimination of binary search operations
4. **Switch to `:a` if needed**: Use `:a` kernels only if `:b` construction becomes too slow for very fine grids
5. **Lower degrees for high dimensions**: Minimize boundary handling overhead in 4D+ applications

### Subgrid Interpolation

The fast evaluation mode works by convolving the data with precomputed kernel values at discrete positions, then interpolating between the convolution results. The `subgrid` parameter controls how this interpolation between convolution results is performed — essentially, it is interpolation of the interpolation itself.

```julia
# Cubic Hermite interpolation (default for :b5, uses kernel first derivative)
itp = convolution_interpolation(x, y; subgrid=:cubic)

# Quintic Hermite interpolation (uses kernel first and second derivatives)
itp = convolution_interpolation(x, y; subgrid=:quintic)

# Linear interpolation between precomputed kernel values
itp = convolution_interpolation(x, y; subgrid=:linear, precompute=10_000)
```

This is a form of **multilevel interpolation**: the outer level convolves your data with kernel values, while the inner level interpolates the kernel values themselves between precomputed points. Higher subgrid orders use the analytically predifferentiated kernels to perform Hermite interpolation, producing a smoother and more accurate kernel lookup.

Using derivative information at the subgrid level drastically reduces the number of precomputed points needed to achieve a given accuracy. The cubic subgrid (using first derivative information) and quintic subgrid (using first and second derivatives) not only require far fewer precomputed points but also improve accuracy beyond what linear subgrid can achieve at comparable resolution. This means you can simultaneously reduce memory usage and improve interpolation quality. The default `precompute=100` with cubic subgrid achieves excellent accuracy with minimal memory.

| Subgrid    | Method                  | Kernel derivatives used | Tensor products (2D) |
|------------|-------------------------|------------------------|----------------------|
| `:linear`  | Linear interpolation    | 0                      | 4                    |
| `:cubic`   | Cubic Hermite           | 1                      | 16                   |
| `:quintic` | Quintic Hermite         | 2                      | 36                   |

The available subgrid strategy depends on how many smooth derivatives remain after accounting for the requested derivative order:

```
available = max_smooth_derivative[kernel] - derivative
```

For example, `:b5` (C³) with `derivative=0` has 3 remaining derivatives, supporting up to `:quintic` subgrid. With `derivative=1`, 2 remain, still supporting `:quintic`. With `derivative=3`, no derivatives remain, so only `:linear` is available.

**Dimension availability**: Cubic and quintic subgrid modes are implemented for 1D and 2D. Higher dimensions (3D+) use `:linear` subgrid, as the number of tensor products grows as `(order+1)^N` per cell, making higher-order subgrids impractical.

**Practical guidance**: The default cubic subgrid with `precompute=100` provides excellent accuracy for most applications. Quintic subgrid offers even higher accuracy at the same precompute resolution. If using `:linear` subgrid, increase `precompute` to at least 10,000 to maintain accuracy.

```julia
# Default settings — cubic subgrid with 100 precomputed points
itp = convolution_interpolation(x, y)

# Maximum accuracy — quintic subgrid
itp = convolution_interpolation(x, y; subgrid=:quintic)

# Linear subgrid requires more precomputed points to compensate
itp = convolution_interpolation(x, y; subgrid=:linear, precompute=10_000)
```

## Comparison with Other Packages

ConvolutionInterpolations.jl offers several advantages:

- **Novel b-series kernels**: Discovered through systematic analytical search, these kernels achieve 7th order convergence. The b7 kernel is optimal for polynomial reproduction (reproducing its own degree 7), while higher-degree kernels (b9-b13) provide increased smoothness without additional polynomial reproduction capability
- **Persistent kernel caching**: Discretized kernels are computed once and cached to disk using Scratch.jl, making subsequent loads nearly instantaneous
- **Hermite multilevel interpolation**: Novel approach using cubic/quintic Hermite subgrid interpolation of precomputed kernel convolutions
- **Unified framework**: Single interface from nearest-neighbor to 13th-degree interpolation
- **Minimal dependencies**: Built primarily on Julia standard library (LinearAlgebra, Serialization) plus Scratch.jl for caching

## Technical Background

The b-series kernels (b5, b7, b9, b11, b13) represent original contributions discovered through systematic analytical search using symbolic computation. These kernels share several key properties:

- **7th order convergence**: All b-series kernels achieve 7th order accuracy from a Taylor series perspective
- **Polynomial reproduction**: b5 reproduces quintics, b7 reproduces septics (degree 7) and is the last kernel to reproduce its own degree. Higher-degree kernels (b9, b11, b13) cannot reproduce polynomials of their own degree.
- **High continuity**: Ranging from C³ (b5) to C⁶ (b13), providing extremely smooth interpolated functions
- **Optimal boundary handling**: Novel polynomial boundary condition method that solves Vandermonde systems to compute ghost point values, preserving polynomial reproduction properties at domain edges

The b5 kernel is the default due to its excellent balance of computational efficiency and accuracy. Higher-degree kernels offer increased smoothness while maintaining 7th order convergence.

**Extended precision**: For applications requiring accuracy beyond Float64 machine precision (~10⁻¹⁵), all kernels support BigFloat arithmetic. The analytical kernel coefficients are stored as exact rational numbers, enabling convergence well beyond standard machine precision when using extended precision types.

## Novel Contributions

This package introduces three main technical contributions beyond the implementation itself:

**1. B-series kernel family**

A new family of high-order convolution kernels (b5, b7, b9, b11, b13) discovered through systematic analytical search using symbolic computation, generalizing the approach of R. G. Keys (1981). All b-series kernels achieve 7th order convergence from Taylor series analysis, with continuity classes ranging from C³ (b5) to C⁶ (b13). The b5 kernel provides dramatically better accuracy than cubic splines at comparable computational cost, while higher-degree kernels offer increasingly smooth derivatives. Kernel coefficients are stored as exact rational numbers, enabling extended precision arithmetic with BigFloat for convergence beyond machine precision.

**2. Hermite multilevel interpolation**

Rather than evaluating kernel polynomials directly via Horner's method, kernels are discretized at a small number of points (default 100) and stored in a persistent disk cache via Scratch.jl. During interpolation, data is convolved with these precomputed kernel values, and the results are interpolated using cubic or quintic Hermite subgrid interpolation. This multilevel approach — interpolation of the interpolation — leverages analytically predifferentiated kernel coefficients to achieve high accuracy with very few precomputed points. The result is both faster and more numerically stable than direct polynomial evaluation, while the persistent cache makes subsequent initialization nearly instantaneous.

**3. Polynomial boundary conditions via Vandermonde systems**

A novel boundary handling method that computes optimal ghost point values by solving Vandermonde systems, preserving each kernel's polynomial reproduction properties at domain edges. This ensures b-series kernels maintain their 7th order convergence all the way to domain boundaries, enabling convergence to machine precision across the entire domain rather than degrading near edges as is typical with simpler boundary treatments.

## Acknowledgments

ConvolutionInterpolations.jl draws significant inspiration from [Interpolations.jl](https://github.com/JuliaMath/Interpolations.jl) in terms of API design and interface patterns. This package originally started as a PR for Interpolations.jl before evolving into a standalone implementation with novel kernel algorithms. I'm grateful to the Interpolations.jl maintainers and contributors for their excellent work.

The theoretical foundation builds on seminal work in convolution interpolation:

- R. G. Keys, "Cubic Convolution Interpolation for Digital Image Processing," *IEEE Transactions on Acoustics, Speech, and Signal Processing*, vol. 29, no. 6, pp. 1153-1160, December 1981.

- E. H. W. Meijering, K. J. Zuiderveld, and M. A. Viergever, "Image Reconstruction by Convolution with Symmetrical Piecewise nth-Order Polynomial Kernels," *IEEE Transactions on Image Processing*, vol. 8, no. 2, pp. 192-201, February 1999.

The novel b-series kernels (b5, b7, b9, b11, b13) represent original contributions discovered through a prolonged effort of systematic analytical search using symbolic computation (SymPy), by generalizing the approach of R. G. Keys. These kernels achieve 7th order convergence with varying degrees of polynomial reproduction and continuity classes. The performance optimizations, including O(1) interpolation through elimination of binary search operations and efficient precomputation strategies, were developed specifically for this package.

## License

This package is available under the MIT License.