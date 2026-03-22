# ConvolutionInterpolations.jl — Test Suite Documentation

This document describes the test suite for ConvolutionInterpolations.jl. The suite validates correctness across all supported kernels, grid types, dimensions, derivative orders, and evaluation modes.

## Test Structure

The tests are organized into thematic files, all included from `runtests.jl`:

| File | Scope |
|------|-------|
| `test_constructors.jl` | Low-level constructor tests for all kernels; 2D and 3D per-dim kernel combination coverage |
| `test_uniform_interpolation.jl` | Grid point reproduction and midpoint accuracy for uniform grids in 1D–4D |
| `test_uniform_derivatives.jl` | First derivative accuracy on uniform grids in 1D–4D |
| `test_uniform_convergence.jl` | Error reduction under grid refinement for function values and 1st/2nd derivatives in 1D |
| `test_nonuniform_interpolation.jl` | Grid point reproduction and midpoint accuracy on nonuniform grids in 1D–4D |
| `test_nonuniform_derivatives.jl` | First and second derivative accuracy on nonuniform grids in 1D–2D |
| `test_nonuniform_convergence.jl` | Convergence rates on nonuniform grids in 1D; nearly-uniform regression tests |
| `test_nonuniform_perdim_derivatives.jl` | Per-dim b-kernel derivatives on nonuniform grids in 2D |
| `test_nonuniform_perdim_kernels.jl`  |  Per-dim b-kernels on nonuniform grids in 2D and 3D |
| `test_lazy.jl` | Lazy vs eager agreement, construction speed, and boundary fallback in 1D–5D |
| `test_nonuniform_a0_a1.jl` | Nearest-neighbor and linear interpolation on nonuniform grids in 1D–3D |
| `test_antiderivative.jl` | Antiderivative convergence in 1D–3D; fast vs direct agreement; anchor correctness |
| `test_perdim_derivatives.jl` | Per-dim derivative orders in 2D and 3D, fast and direct |
| `test_mixed_integral.jl` | Mixed integral/derivative orders in 2D, fast and direct paths |
| `test_perdim_kernel_derivatives.jl` | Per-dim kernel combinations with derivatives in 2D and 3D |
| `test_boundary_condition.jl` | BC downgrade with insufficient points in 1D–3D |
| `test_extrapolation.jl` | `:line` and `:flat` extrapolation modes for uniform grids in 1D–4D |
| `test_subgrid_downgrade.jl` | Subgrid mode downgrade at top smooth derivative order |
| `test_gaussian.jl` | Gaussian kernel construction and evaluation in 1D and 2D |

## Kernel Coverage Strategy

Not all kernels are exercised in every test. The constructor test verifies that all kernels construct correctly. Subsequent tests use representative kernels that exercise distinct dispatch paths:

| Kernel | Reason for inclusion |
|--------|---------------------|
| `:a0` | Nearest-neighbor path, no ghost points, no table lookup |
| `:a1` | Linear path, no ghost points, no table lookup |
| `:a3` | Lower-order higher kernel path, C1 continuity, top derivative forces `:linear` subgrid |
| `:b5` | b-series higher kernel path, C3 continuity, 7th-order accuracy |

This covers all major dispatch paths while keeping the suite fast (under 4 minutes).

## Supported Kernels

### Uniform grid kernels

| Kernel | Degree | Support | Continuity | Order of accuracy |
|--------|--------|---------|------------|-------------------|
| `:a0` | 0 | [-0.5, 0.5] | — | 0 |
| `:a1` | 1 | [-1, 1] | C0 | 1 |
| `:a3` | 3 | [-2, 2] | C1 | 3 |
| `:a4` | 3 | [-3, 3] | C1 | 4 |
| `:a5` | 5 | [-3, 3] | C1 | — |
| `:a7` | 7 | [-4, 4] | C1 | — |
| `:b5` | 5 | [-5, 5] | C3 | 7 |
| `:b7` | 7 | [-6, 6] | C4 | 7 |
| `:b9` | 9 | [-7, 7] | C5 | 7 |
| `:b11` | 11 | [-8, 8] | C6 | 7 |
| `:b13` | 13 | [-9, 9] | C6 | 7 |

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

Nonuniform kernels use direct evaluation (fast mode is automatically disabled for nonuniform grids).

## Test Strategies

### Constructor tests

Verifies that all 11 kernels (`:a0` through `:b13`) construct correctly in 1D via both `ConvolutionInterpolation` and `FastConvolutionInterpolation`, and that direct and fast paths agree within `1e-6`. Also tests lazy construction variants and `ConvolutionExtrapolation` wrapping.

In addition, all 16 per-dim kernel combinations in 2D and all 64 in 3D are constructed and evaluated at a single point to verify dispatch is callable. The kernels tested per-dim are `:a0`, `:a1`, `:a3`, `:b5`.

### Grid point reproduction

The most fundamental correctness check: interpolation must pass exactly through the input data. For every kernel, dimension (1D–4D), and evaluation mode, the test constructs an interpolator from random data on a regular grid and verifies that evaluating at grid points recovers the original values within tolerance `1e-6`.

### Midpoint accuracy for linear data

Validates that the kernel reproduces polynomials of at least degree 1. For a linear (or multilinear) test function, the interpolated value at cell midpoints must equal the average of the surrounding corners. Skipped for `:a0`.

| Dimension | Test function | Midpoint average over |
|-----------|--------------|----------------------|
| 1D | f(x) = x | 2 endpoints |
| 2D | f(x,y) = x + y | 4 corners |
| 3D | f(x,y,z) = x + y + z | 8 corners |
| 4D | f(x,y,z,w) = x + y + z + w | 16 hypercube corners |

### Derivative accuracy

First derivatives are tested on separable sin/sin functions where analytical values are known. Grid size is `N=20` per dimension, tolerance `0.01`. Tested in 1D–4D for both fast and direct paths using `:b5`.

For the 4D direct path, only the midpoint of the grid is evaluated (to keep runtime reasonable).

### Per-dim kernel and derivative combinations

Tests all combinations of `(:a3, :b5)` kernels per dimension:
- 2D: 4 kernel combinations × 3 derivative orders `(1,0)`, `(0,1)`, `(1,1)` — fast and direct
- 3D: 8 kernel combinations × 4 derivative orders `(1,0,0)`, `(0,1,0)`, `(0,0,1)`, `(1,1,1)` — fast and direct

Verifies that all combinations produce finite values.

### Antiderivative (integral) convergence

Tests `derivative=-1` for kernels `:a3` and `:b5`. Grid sizes `n = 24, 48, 96` (1D) and `n = 20, 40, 80` (2D/3D). Checks:
- Convergence to the analytical antiderivative (minimum ratio `2^(order-1)` per grid doubling)
- Fast vs direct agreement within `1e-6`
- Anchor value is exactly zero
- 2D even/odd function correctness (sin×cos, cos×sin, sin×sin, cos×cos)
- `:a0` delta spike → Heaviside step function

Expected convergence orders: `:a3` → 3rd order, `:b5` → 5th order.

### Mixed integral/derivative orders

Tests `MixedIntegralOrder` in 2D for `derivative` combinations `(-1,0)`, `(0,-1)`, `(-1,1)`, `(1,-1)` with `:b5`. Tests both fast and direct paths for convergence and exact anchor-is-zero property. Grid sizes `n = 20, 40, 80`.

### Per-dim derivative orders

Tests `DerivativeOrder{(d1,d2,...)}` dispatch for `:b5` in 2D and 3D, fast and direct:

- 2D: `(1,0)`, `(0,1)`, `(1,1)`, `(2,0)`, `(0,2)` using `f(x,y) = sin(x)·sin(y)`
- 3D: `(1,0,0)`, `(0,1,0)`, `(0,0,1)`, `(1,1,0)`, `(1,0,1)`, `(0,1,1)` using `f(x,y,z) = sin(x)·sin(y)·sin(z)`

Tolerance `1e-4`, grid size `N=50`.

### Nonuniform grid convergence

Grid sizes `n = 14, 28, 56` (1D). Tests 0th, 1st, and 2nd derivative convergence for `:b5` on nonuniform grids with sinusoidal perturbation strength `0.3`. Minimum convergence ratio `2^(order-1)` per grid doubling.

### Nearly-uniform regression

Verifies that the nonuniform code path produces the same results as the fast uniform path when given a nearly-uniform grid (perturbation strength `1e-8`). Tolerance `1e-4`.

### Nonuniform per-dim derivatives

Tests per-dim b-kernel derivatives on nonuniform 2D grids for `f(x,y) = sin(x)·cos(y)`:
- `(0,0)`, `(1,0)`, `(0,1)`, `(1,1)` using `:b5`
- `(2,0)` using `:b7`, `:b9`, `:b11` (`:b5` marginally accurate at N=40)
- Consistency: tuple vs scalar same-kernel results agree within `1e-10`

### Nonuniform per-dim b kernels

Tests per-dim b-kernel selection on nonuniform grids in 2D and 3D.

### Lazy mode

- **Lazy vs eager agreement**: Constructs both lazy and eager interpolators and verifies agreement at interior, boundary, and near-boundary points for uniform grids in 1D and 2D (`:a3`, `:b5`), and nonuniform grids in 1D–3D (`:a3`).
- **Structural checks**: Verifies `itp.lazy == Val{true}()` and `itp.coefs === vs` (lazy stores a reference, not a copy).
- **Construction speed**: A 50×50×50 `:b5` lazy construction must complete in under 2 seconds.
- **Boundary fallback in high dimensions**: 4D lazy evaluates correctly near boundaries; 5D with `boundary_fallback=true` returns a finite value without throwing.
- **Nonuniform b-kernels force eager**: Constructing with `kernel=:b5, lazy=true` on a nonuniform grid silently sets `lazy=false`.

### Nonuniform `:a0`/`:a1`

- **`:a0`**: Grid point reproduction in 1D and 2D.
- **`:a1`**: Grid point reproduction, midpoint averaging, exact linear reproduction in 1D; exact bilinear in 2D; exact trilinear in 3D.
- Both verify `itp.lazy == Val{true}()` and `itp.coefs === vs`.

### Extrapolation boundary conditions

For kernels `:a1`, `:a3`, `:b5` and dimensions 1D–4D, tests two extrapolation modes:
- **`:line`**: Linear extrapolation. For `f(x) = x`, evaluating outside `[0,1]` continues the linear trend.
- **`:flat`**: Constant extrapolation. Values outside the domain clamp to the nearest boundary value.

Tolerance `1e-6`.

### Boundary condition downgrade

Verifies that `bc=:poly` with only 4 grid points is automatically downgraded to `(:linear, :linear)` per dimension. Checked by inspecting `itp.itp.bc` directly. Tested in 1D, 2D, and 3D with `:b5`.

### Subgrid downgrade at top derivative

Verifies that requesting `subgrid=:cubic` at the kernel's top smooth derivative is automatically downgraded to `:linear` for that dimension. Checked by inspecting `itp.itp.subgrid` directly.

| Kernel | Top derivative | Expected subgrid for `(d, 0)` |
|--------|----------------|-------------------------------|
| `:a3` | 1 | `(:linear, :cubic)` |
| `:b5` | 3 | `(:linear, :cubic)` |

### Gaussian kernel

Verifies construction and evaluation for `B ∈ (1.0, 2.0, 5.0)` in 1D and `B=2.0` in 2D. Tolerance `0.1` (Gaussian smoothing intentionally blurs the data). Uses `f(x) = sin(x)` and `f(x,y) = sin(x)·cos(y)`.

## Test Parameters

| Parameter | Value | Used in |
|-----------|-------|---------|
| `N` | 4 | Uniform interpolation, extrapolation (grid points per dimension) |
| `tolerance` | `1e-6` | Uniform interpolation, extrapolation |
| `N_deriv` | 20 | Uniform derivative test points |
| `tolerance_deriv` | `0.01` | Uniform derivatives |
| `N_nu` | 40 (1D/2D), 10 (3D), 6 (4D) | Nonuniform interpolation |
| `tolerance_nu` | `1e-6` | Nonuniform interpolation |
| `N_nu_deriv` | 40 | Nonuniform derivative test points |
| `tolerance_nu_deriv` | `1e-4` | Nonuniform derivatives |
| `N_pd` | 50 | Per-dim derivative test points |
| `tolerance_pd` | `1e-4` | Per-dim derivatives |
| `bc` | `:linear` | Uniform interpolation, extrapolation |
| `bc_deriv` | `:poly` | Derivative and convergence tests |
| Convergence grids (uniform) | `[12, 24, 48]` | Uniform convergence |
| Convergence grids (nonuniform) | `[14, 28, 56]` | Nonuniform convergence |
| Convergence grids (antiderivative 1D) | `[24, 48, 96]` | 1D antiderivative |
| Convergence grids (antiderivative 2D/3D) | `[20, 40, 80]` (2D), `[12, 24, 48]` (3D) | 2D/3D antiderivative |

## Running the Tests

```bash
# Full suite (~4 minutes)
julia --project -e 'using Pkg; Pkg.test()'

# Single file during development
julia --project -e 'using ConvolutionInterpolations, Test; include("test/runtests.jl")'
```