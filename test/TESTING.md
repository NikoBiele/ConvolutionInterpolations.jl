# ConvolutionInterpolations.jl — Test Suite Documentation

This document describes the test suite for ConvolutionInterpolations.jl. The suite validates correctness across all supported kernels, grid types, dimensions, derivative orders, and evaluation modes.

## Test Structure

The tests are organized into thematic files, all included from `runtests.jl`, grouped here by topic.

### Exact kernels

| File | Scope |
|------|-------|
| `test_column_polynomials.jl` | Column polynomials equal the kernels exactly for every kernel and order; exact integral kernel values at integer offsets |
| `test_constructors.jl` | Constructors for all kernels; 2D and 3D per-dimension kernel combinations |

### Uniform grids

| File | Scope |
|------|-------|
| `test_uniform_interpolation.jl` | Grid point reproduction and midpoint accuracy, 1D–4D |
| `test_uniform_derivatives.jl` | First derivatives, 1D–4D |
| `test_uniform_convergence.jl` | Convergence of values and 1st/2nd derivatives in 1D; dense-grid rounding floor |
| `test_perdim_derivatives.jl` | Per-dimension derivative orders in 2D and 3D, fast and direct |
| `test_perdim_kernel_derivatives.jl` | Per-dimension kernels combined with derivatives in 2D and 3D |
| `test_uniform_lazy.jl` | Lazy vs eager agreement, construction speed, boundary fallback, 1D–4D |

### Nonuniform grids

| File | Scope |
|------|-------|
| `test_nonuniform_interpolation.jl` | Grid point reproduction and midpoint accuracy, 1D–4D |
| `test_nonuniform_derivatives.jl` | First and second derivatives, 1D–2D |
| `test_nonuniform_convergence.jl` | Convergence rates in 1D; nearly-uniform regression |
| `test_nonuniform_perdim_derivatives.jl` | Per-dimension b-kernel derivatives in 2D |
| `test_nonuniform_perdim_kernels.jl` | Per-dimension b-kernels in 2D and 3D |
| `test_nonuniform_a0_a1.jl` | Nearest-neighbor and linear interpolation, 1D–3D |
| `test_nonuniform_lazy.jl` | Lazy vs eager agreement and boundary fallback for `:n3`, 1D–3D |

### Integrals

| File | Scope |
|------|-------|
| `test_antiderivative.jl` | First-order antiderivatives, 1D–3D: convergence, fast vs direct, anchoring |
| `test_nd_integral_separable.jl` | 4D integral of separable data equals the product of 1D integrals |
| `test_higher_integrals.jl` | Orders 2 and higher: exactness, consistency between orders, convergence, precision, N-D separability, errors |
| `test_mixed_integral_1D_2D.jl` | Integrals mixed with derivatives in 1D and 2D, fast and direct |
| `test_mixed_integral_3D_4D.jl` | Integrals mixed with derivatives in 3D and 4D |
| `test_mixed_integral_5D.jl` | Integrals mixed with derivatives in 5D; fast vs direct; anchoring |

### Boundaries, precision and options

| File | Scope |
|------|-------|
| `test_boundary_condition.jl` | Boundary condition downgrade with too few points, 1D–3D |
| `test_extrapolation.jl` | `:line` and `:flat` extrapolation, 1D–4D |
| `test_bigfloat_precision.jl` | BigFloat type preservation and machine precision |
| `test_float32.jl` | Float32 type preservation and accuracy across features |
| `test_allocations.jl` | Non-lazy evaluation is allocation-free |

### Further features

| File | Scope |
|------|-------|
| `test_gaussian.jl` | Gaussian smoothing kernel and `convolution_smooth` |
| `test_scattered_to_grid.jl` | Nearest-neighbor gridding of scattered data |
| `test_fit_scattered.jl` | Scattered-data fitting: exactness, convergence, derivatives, box integrals |
| `test_resample.jl` | `convolution_resample` accuracy, output size, derivatives |
| `test_show.jl` | `show` output for all interpolant types and options |
| `test_deprecations.jl` | Deprecated `precompute`/`subgrid` keywords warn and have no effect |

## Kernel Coverage Strategy

Not all kernels are exercised in every test. The constructor test verifies that all kernels construct correctly. Subsequent tests use representative kernels that exercise distinct dispatch paths:

| Kernel | Reason for inclusion |
|--------|---------------------|
| `:a0` | Nearest-neighbor path, no ghost points, dedicated evaluators |
| `:a1` | Linear path, no ghost points, dedicated evaluators |
| `:a3` | Lower-order higher kernel path, C1 continuity, top derivative 1 |
| `:b5` | b-series higher kernel path, C3 continuity, 7th-order accuracy |

This covers all major dispatch paths while keeping the suite fast.

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
- **Fast** (`fast=true`): Evaluates exact column polynomials of the kernel for O(1) evaluation.

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

Expected convergence orders: `:a3` → 3rd order, `:b5` → 7th order.

### Higher-order integrals

`test_higher_integrals.jl` covers `derivative = -m` for m ≥ 2, evaluated by the generic integral evaluator:

- **Exactness on reproduced data**: every kernel reproduces linear data (`:a0` constants), so every order up to each kernel's cap must match the closed-form m-fold integral to rounding (tolerance `1e-13`).
- **Consistency between orders**: F_m(x) must equal ∫ₐˣ F_(m−1)(t) dt, computed by 12-point Gauss–Legendre quadrature of the package's own order-(m−1) interpolant on every half grid step, where it is a polynomial. Checks anchoring and tails on smooth, non-polynomial data (tolerance `1e-12`).
- **Convergence**: on `sin` data against its analytic m-fold integral, each order converges at the kernel's rate (with a margin of one order).
- **Precision**: Float32 in gives Float32 out; BigFloat is exact on linear data far beyond Float64 precision.
- **N-D**: separable data, where the result must equal the product of 1D results.
- **Errors**: orders beyond a kernel's cap, lazy mode (for all integral orders), the direct path, and nonuniform grids.
- **Construction links to order 1**: for order 1, the polynomial left tails, the region tails and the anchored stencil weights must reproduce the existing order-1 construction.

### Mixed integral/derivative orders

`test_mixed_integral_3D_4D.jl` extends coverage to 3D and 4D. All tests construct in eager mode (`lazy=false`, which integrals require), with most evaluation points well inside the domain (at least 5 grid spacings from boundaries to stay clear of the b5 stencil width of 10). Convergence is verified on grids `n = 20, 40` for all combinations of 1, 2, and 3 integral dimensions in 3D, including mixed cases with derivatives such as `(-1,1,0)` and `(1,-1,0)`. A small set of near-boundary tests verify correct behavior close to domain edges. 4D tests verify construction and callability for all supported derivative combinations, plus numerical correctness for `(-1,-1,-1,0)` and `(-1,0,0,1)`.

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

### Exact column polynomials

`test_column_polynomials.jl` checks the foundation of the fast path: on every column `c` of every kernel, the column polynomial evaluated at `τ` equals the kernel itself at offset `c − 1 − eqs + τ`. Both sides are evaluated in exact `Rational{BigInt}` arithmetic, so the comparison is exact equality, not a tolerance. Covers every kernel and every derivative order from −1 (antiderivative) up, at points across each column including both ends.

### Separable N-D integral

`test_nd_integral_separable.jl` builds a 4D antiderivative interpolant of separable data `g₁(x)·g₂(y)·g₃(z)·g₄(w)`, which must equal the product of four 1D antiderivative interpolants exactly (to rounding), since the ghost-point extension acts per dimension. Evaluation points include the far corner and points just inside the far ends, where missing coefficients in the N-D summation would show first.

### Deprecated keywords

`test_deprecations.jl` verifies that setting `precompute` or `subgrid` produces a deprecation warning and a result identical to not setting them, and that nothing warns when neither is set. Covers `convolution_interpolation`, `FastConvolutionInterpolation` and the scattered-data `convolution_interpolation`; `convolution_resample` is covered in `test_resample.jl`.

### Gaussian kernel

Verifies construction and evaluation for `B ∈ (1.0, 2.0, 5.0)` in 1D and `B=2.0` in 2D. Tolerance `0.1` (Gaussian smoothing intentionally blurs the data). Uses `f(x) = sin(x)` and `f(x,y) = sin(x)·cos(y)`.

### BigFloat precision preservation

`test_bigfloat_precision.jl` verifies that `BigFloat` data flows correctly through construction and evaluation without silent downcast to `Float64`. Two test tiers:

- **Type preservation** (all kernels, coarse grid n=50): Verifies `result isa BigFloat` and errors are within coarse-grid tolerance for interpolation, derivative, and antiderivative.
- **Machine precision** (b-kernels only, dense grid n=2000): Verifies that errors fall below `1e-16` — well below `Float64` precision — proving that `BigFloat` arithmetic is used throughout. Note that b-kernels require dense grids to enter the asymptotic convergence regime beyond `Float64` precision; a-kernels converge too slowly to demonstrate this on practical grid sizes.

### Allocations

Non-lazy kernels should all be non-allocating.

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
| Convergence grids (mixed 3D/4D) | `[20, 40]` | 3D/4D mixed integral |
| BigFloat coarse grid | 50 | Type preservation |
| BigFloat dense grid | 2000 | Machine precision |
| Convergence grids (uniform) | `[12, 24, 48]` | Uniform convergence |
| Convergence grids (nonuniform) | `[14, 28, 56]` | Nonuniform convergence |
| Convergence grids (antiderivative 1D) | `[24, 48, 96]` | 1D antiderivative |
| Convergence grids (antiderivative 2D/3D) | `[20, 40, 80]` (2D), `[12, 24, 48]` (3D) | 2D/3D antiderivative |

## Running the Tests

```bash
# Full suite
julia --project -e 'using Pkg; Pkg.test()'

# Single file during development
julia --project -e 'using ConvolutionInterpolations, Test; include("test/runtests.jl")'
```