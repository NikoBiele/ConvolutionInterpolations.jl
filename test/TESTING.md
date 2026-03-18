# ConvolutionInterpolations.jl — Test Suite Documentation

This document describes the test suite for ConvolutionInterpolations.jl. The suite validates correctness across all supported kernels, grid types, dimensions, derivative orders, and evaluation modes.

## Test Structure

The tests are organized into thematic files, all included from `runtests.jl`:

| File | Scope |
|------|-------|
| `test_uniform_interpolation.jl` | Grid point reproduction, midpoint accuracy, and extrapolation for uniform grids in 1D–4D |
| `test_uniform_derivatives.jl` | First and second derivative accuracy on uniform grids in 1D–4D |
| `test_uniform_convergence.jl` | Error reduction under grid refinement for function values and derivatives |
| `test_nonuniform_interpolation.jl` | Grid point reproduction and midpoint accuracy on nonuniform grids in 1D–4D |
| `test_nonuniform_derivatives.jl` | First and second derivative accuracy on nonuniform grids in 1D–2D |
| `test_nonuniform_convergence.jl` | Convergence rates on nonuniform grids, plus nearly-uniform regression tests |
| `test_lazy.jl` | Lazy vs eager agreement (uniform and nonuniform), construction speed, and boundary fallback behavior in 1D–5D |
| `test_nonuniform_a0_a1.jl` | Nearest-neighbor and linear interpolation on nonuniform grids in 1D–3D |

Shared helpers (e.g. `make_nonuniform_grid`) are defined in `runtests.jl` before the includes.

## Kernels Under Test

### Uniform grid kernels

| Kernel | Degree | Support | Continuity | Order of accuracy | Pieces |
|--------|--------|---------|------------|-------------------|--------|
| `:a0` | 0 | [-0.5, 0.5] | — | 0 | 1 |
| `:a1` | 1 | [-1, 1] | C0 | 1 | 1 |
| `:a3` | 3 | [-2, 2] | C1 | 3 | 2 |
| `:a4` | 3 | [-3, 3] | C1 | 4 | 3 |
| `:a5` | 5 | [-3, 3] | C1 | — | 3 |
| `:a7` | 7 | [-4, 4] | C1 | — | 4 |
| `:b5` | 5 | [-5, 5] | C3 | 7 | 5 |
| `:b7` | 7 | [-6, 6] | C4 | 7 | 6 |
| `:b9` | 9 | [-7, 7] | C5 | 7 | 7 |
| `:b11` | 11 | [-8, 8] | C6 | 7 | 8 |
| `:b13` | 13 | [-9, 9] | C6 | 7 | 9 |

### Nonuniform grid kernels

| Kernel | Notes |
|--------|-------|
| `:a0` | Works natively on nonuniform grids (no ghost points needed) |
| `:a1` | Works natively on nonuniform grids (no ghost points needed) |
| `:n3` | Cubic kernel for nonuniform grids; also used as fallback for `:a3`, `:a4`, `:a5`, `:a7` |
| `:b5`–`:b13` | Full b-series works on nonuniform grids via precomputed polynomial weights |

### Evaluation modes

Each uniform kernel is tested in two modes:

- **Direct** (`fast=false`): Evaluates the piecewise polynomial kernel directly.
- **Fast** (`fast=true`): Uses precomputed kernel tables with subgrid interpolation for O(1) evaluation.

Nonuniform kernels use direct evaluation (fast mode is automatically disabled).

## Test Strategies

### Grid point reproduction

The most fundamental correctness check: interpolation must pass exactly through the input data.

For every kernel, dimension (1D–4D), and evaluation mode, the test constructs an interpolator from random data on a regular grid and verifies that evaluating at grid points recovers the original values within tolerance (1e-4).

This test is applied to both uniform and nonuniform grids.

### Midpoint accuracy for linear data

Validates that the kernel reproduces polynomials of at least degree 1. For a linear (or multilinear) test function, the interpolated value at cell midpoints must equal the average of the surrounding corners.

The test functions scale with dimensionality:

| Dimension | Test function | Midpoint average over |
|-----------|--------------|----------------------|
| 1D | f(x) = x | 2 endpoints |
| 2D | f(x,y) = x + y | 4 corners |
| 3D | f(x,y,z) = x + y + z | 8 corners |
| 4D | f(x,y,z,w) = x + y + z + w | 16 hypercube corners |

Skipped for `:a0` (nearest neighbor cannot reproduce linear functions).

### Extrapolation boundary conditions

For each kernel and dimension, two extrapolation modes are tested:

- **`Line()`**: Linear extrapolation. For the test function f(x) = x, evaluating outside the domain should continue the linear trend (e.g. `etp(-0.2) ≈ -0.2`).
- **`Flat()`**: Constant extrapolation. Values outside the domain clamp to the nearest boundary value (e.g. `etp(-0.2) ≈ 0.0`).

### Derivative accuracy

Derivatives are tested on sin/cos functions where analytical values are known:

| Order | Test function | Analytical derivative |
|-------|--------------|----------------------|
| d0 | sin(x) | sin(x) |
| d1 | sin(x) | cos(x) |
| d2 | sin(x) | -sin(x) |

For multidimensional tests, the separable product structure is used: e.g. `sin(x)·sin(y)` has mixed derivative `cos(x)·cos(y)`.

Derivative tests use b-series kernels (`:b5`, `:b7`, `:b9`, `:b11`) which have sufficient continuity for smooth derivative evaluation. The a-series kernels automatically switch to linear interpolation at their top derivative order, which is tested implicitly through the convergence tests.

Tolerance for derivative tests is looser (0.01–0.1) than for interpolation (1e-4), reflecting the inherently lower accuracy of numerical differentiation.

### Convergence under refinement

The most important quantitative test: errors must decrease as the grid is refined. For grid sizes n = 12, 24, 48 (uniform) or n = 20, 40, 80 (nonuniform), the test verifies that the maximum error decreases by at least a factor of 2 per grid doubling.

This is tested for:

- Function values (d0): all kernels
- First derivatives (d1): b-series kernels
- Second derivatives (d2): b-series kernels

Both uniform and nonuniform grids are tested. The factor-of-2 threshold is deliberately conservative — most kernels converge much faster (e.g. b5 achieves 7th-order, meaning ~128× error reduction per doubling).

### Nearly-uniform regression

A regression test verifying that the nonuniform code path produces the same results as the fast uniform path when given a nearly-uniform grid (perturbation strength 1e-8). This catches bugs where the nonuniform path might silently produce different results from the uniform path on grids that are uniform in practice.

## Lazy Mode Tests

### Lazy vs eager agreement

For kernels `:a3`, `:b5`, `:b7` the test constructs both eager (`lazy=false`) and lazy (`lazy=true`) interpolators from identical data and verifies that they produce the same results at interior, boundary, and near-boundary points. This is tested for:

- Uniform grids in 1D and 2D
- Nonuniform grids (`:n3` path) in 1D, 2D, and 3D

The test also verifies structural properties: `itp.lazy == true`, `itp.coefs === vs` (lazy stores a reference to the original data, not a copy with ghost points).

### Lazy construction speed

Verifies that constructing a lazy interpolator for a 50×50×50 array with `:b5` completes in under 100ms, confirming that ghost point expansion is actually being skipped.

### Boundary fallback

Tests the `boundary_fallback` flag across dimensions:

| Dimension | Kernel | `lazy=true` behavior |
|-----------|--------|---------------------|
| 1D | all | `boundary_fallback=false`: evaluates near boundary. `boundary_fallback=true`: throws `ErrorException` |
| 4D | `:a3` | Evaluates normally (low-order kernel, manageable cost) |
| 4D | `:b5`, `:b7` | Automatically forces `boundary_fallback=true`, throws near boundary |
| 5D | all | Automatically forces `boundary_fallback=true`, throws near boundary |

### Nonuniform b-kernels force eager

Verifies that constructing a nonuniform interpolator with a b-series kernel and `lazy=true` silently sets `lazy=false`, since the nonuniform b-kernel path requires precomputed ghost points.

## Nonuniform `:a0`/`:a1` Tests

These kernels work natively on nonuniform grids without ghost points:

- **`:a0`**: Grid point reproduction in 1D and 2D. No midpoint test (nearest neighbor).
- **`:a1`**: Grid point reproduction, midpoint averaging, exact linear reproduction in 1D. Bilinear exactness (`f(x,y) = x + y`) in 2D. Trilinear exactness in 3D.
- Both verify `itp.lazy == true` and `itp.coefs === vs` (no data copying).

## Test Parameters

| Parameter | Value | Used in |
|-----------|-------|---------|
| `N = 4` | Grid points per dimension | Uniform interpolation/extrapolation |
| `tolerance = 1e-4` | Tight numerical tolerance | Interpolation, grid reproduction |
| `N_deriv = 10` | Grid points for derivative tests | Uniform derivatives |
| `tolerance_deriv = 0.01` | Looser tolerance for derivatives | Uniform derivatives |
| `N_nu = 12` | Grid points for nonuniform tests | Nonuniform interpolation |
| `tolerance_nu = 1e-4` | Tolerance for nonuniform tests | Nonuniform interpolation |
| `tolerance_nu_deriv = 0.1` | Tolerance for nonuniform derivatives | Nonuniform derivatives |
| `kernel_bc = :linear` | Kernel boundary condition | Uniform interpolation |
| `kernel_bc_deriv = :poly` | Kernel boundary condition for derivatives | Derivative and convergence tests |

## Running the Tests

```bash
# Full suite
julia --project -e 'using Pkg; Pkg.test()'

# Single file during development
julia --project -e 'using ConvolutionInterpolations, Test; include("test/runtests.jl")'
```
