"""
Test Float32 support: eltype preservation, mixed-type handling, and accuracy.

The kernel tables are shipped as Float64 and converted to the working type
at construction. These tests guard the conversion, the scalar-parameter
signatures (B need not match the knot type), and query-type promotion
(queries convert to the interpolant's type at evaluation entry).
"""

println("\n" * "-"^60)
println("Testing Float32 support...")
println("-"^60)

# shared Float32 test data
x32_f = range(0.0f0, Float32(2π), length=50)
y32_f = sin.(x32_f)
xs32_f = range(0.0f0, Float32(2π), length=30)
z32_2d = [sin(a)*cos(b) for a in xs32_f, b in xs32_f]
z32_3d = [sin(a)*cos(b)*sin(c) for a in xs32_f, b in xs32_f, c in xs32_f]

@testset "Float32 1D interpolation eltype" begin
    for kernel in (:a0, :a1, :a3, :a4, :b5, :b7, :b13)
        println("    - 1D Float32 kernel: $kernel")
        itp = convolution_interpolation(x32_f, y32_f; kernel=kernel)
        @test itp(1.5f0) isa Float32
    end
end

@testset "Float32 1D accuracy" begin
    println("    - 1D Float32 accuracy (b5)")
    itp = convolution_interpolation(x32_f, y32_f)
    err = maximum(abs(itp(xi) - sin(xi)) for xi in range(0.5f0, 5.0f0, length=100))
    @test err < 1f-4          # Float32-appropriate, not degraded
    @test err > 1f-12         # genuinely Float32: Float64 leakage would go below this
end

@testset "Float32 derivatives and antiderivatives" begin
    println("    - 1D Float32 derivative / antiderivative")
    @test convolution_interpolation(x32_f, y32_f; derivative=1)(1.5f0) isa Float32
    @test convolution_interpolation(x32_f, y32_f; derivative=2)(1.5f0) isa Float32
    @test convolution_interpolation(x32_f, y32_f; derivative=-1)(1.5f0) isa Float32
end

@testset "Float32 options: bc, extrap, subgrid" begin
    println("    - 1D Float32 bc/extrap/subgrid combinations")
    for bc in (:detect, :poly, :linear, :quadratic)
        @test convolution_interpolation(x32_f, y32_f; bc=bc)(1.5f0) isa Float32
    end
    for ex in (:line, :flat, :natural)
        @test convolution_interpolation(x32_f, y32_f; extrap=ex)(7.0f0) isa Float32
    end
    for sg in (:linear, :cubic, :quintic)
        @test convolution_interpolation(x32_f, y32_f; subgrid=sg)(1.5f0) isa Float32
    end
end

@testset "Float32 multi-dimensional" begin
    println("    - 2D/3D Float32, per-dim kernels, per-dim derivatives, mixed orders")
    @test convolution_interpolation((xs32_f, xs32_f), z32_2d)(1.5f0, 2.0f0) isa Float32
    @test convolution_interpolation((xs32_f, xs32_f), z32_2d; kernel=(:b5, :a3))(1.5f0, 2.0f0) isa Float32
    @test convolution_interpolation((xs32_f, xs32_f), z32_2d; derivative=(1, 0))(1.5f0, 2.0f0) isa Float32
    @test convolution_interpolation((xs32_f, xs32_f), z32_2d; derivative=(-1, 1))(1.5f0, 2.0f0) isa Float32
    @test convolution_interpolation((xs32_f, xs32_f, xs32_f), z32_3d)(1.5f0, 2.0f0, 1.0f0) isa Float32
end

@testset "Float32 lazy mode" begin
    println("    - 2D lazy, 4D lazy boundary_fallback")
    @test convolution_interpolation((xs32_f, xs32_f), z32_2d; lazy=true)(1.5f0, 2.0f0) isa Float32
    xs4 = range(0.0f0, Float32(2π), length=10)
    z4 = [sin(a)*cos(b)*sin(c)*cos(d) for a in xs4, b in xs4, c in xs4, d in xs4]
    itp4 = convolution_interpolation((xs4, xs4, xs4, xs4), z4; lazy=true, boundary_fallback=true)
    @test itp4(1.5f0, 2.0f0, 1.0f0, 2.5f0) isa Float32
end

@testset "Float32 nonuniform grids" begin
    println("    - 1D nonuniform :b5 and :n3")
    xnu = Float32[0.0, 0.15, 0.4, 0.7, 1.5, 2.5, 3.8, 4.2, 4.6, 5.0, 5.5, 6.0]
    ynu = sin.(xnu)
    @test convolution_interpolation(xnu, ynu; kernel=:b5)(2.0f0) isa Float32
    @test convolution_interpolation(xnu, ynu; kernel=:n3)(2.0f0) isa Float32
end

@testset "Float32 smoothing, resampling, gridding" begin
    println("    - convolution_smooth / gaussian / resample / scattered_to_grid")
    ynoisy = y32_f .+ 0.1f0 .* randn(Float32, 50)

    # regression: Float32 range internals are Float64 (StepRangeLen{Float32,Float64,...});
    # B must be accepted as any Real, in both matching and non-matching types
    @test eltype(convolution_smooth(x32_f, ynoisy, 0.1f0)) == Float32
    @test eltype(convolution_smooth(x32_f, ynoisy, 0.1)) == Float32
    @test convolution_gaussian(x32_f, ynoisy, 0.1f0)(1.5f0) isa Float32

    x_fine = range(0.0f0, Float32(2π), length=200)
    @test eltype(convolution_resample(x32_f, x_fine, y32_f)) == Float32
    xf2 = range(0.0f0, Float32(2π), length=60)
    @test eltype(convolution_resample((xs32_f, xs32_f), (xf2, xf2), z32_2d)) == Float32

    pts = rand(Float32, 2, 200) .* Float32(2π)
    vals = sin.(pts[1,:]) .* cos.(pts[2,:])
    @test eltype(scattered_to_grid(pts, vals, (xs32_f, xs32_f))) == Float32
end

@testset "Float32 mixed-type handling" begin
    println("    - query promotion and mixed knot/data construction")
    itp32 = convolution_interpolation(x32_f, y32_f)

    # regression: Float64 query on Float32 interpolant converts at entry
    @test itp32(1.5) isa Float32
    @test itp32(1.5) ≈ itp32(1.5f0)

    # mixed construction follows the data/promotion contract
    @test convolution_interpolation(x32_f, Float64.(y32_f))(1.5) isa Float64
    x64 = range(0.0, 2π, length=50)
    @test convolution_interpolation(x64, y32_f)(1.5f0) isa Float32
end