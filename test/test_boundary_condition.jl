
println("Testing boundary condition downgrading for 1D, 2D, 3D...")
@testset "BC downgrade with insufficient points" begin
    println("Testing BC downgrade in 1D")
    xs_short = range(0.0, 2π, length=5)
    vs_short = sin.(xs_short)
    
    itp = convolution_interpolation(xs_short, vs_short; kernel=:b5, bc=:poly)
    # check both left and right bc were downgraded
    @test itp.itp.bc == ((:linear, :linear),)
end

@testset "2D BC downgrade with insufficient points" begin
    println("Testing BC downgrade in 2D")
    xs_short = range(0.0, 2π, length=5)
    ys_short = range(0.0, 2π, length=5)
    vs_short = [sin(x)*cos(y) for x in xs_short, y in ys_short]
    
    itp = convolution_interpolation((xs_short, ys_short), vs_short; kernel=:b5, bc=:poly)
    @test itp.itp.bc == ((:linear, :linear), (:linear, :linear))
end

@testset "3D BC downgrade with insufficient points" begin
    println("Testing BC downgrade in 3D")
    xs_short = range(0.0, 2π, length=5)
    ys_short = range(0.0, 2π, length=5)
    zs_short = range(0.0, 2π, length=5)
    vs_short = [sin(x)*cos(y)*sin(z) for x in xs_short, y in ys_short, z in zs_short]
    
    itp = convolution_interpolation((xs_short, ys_short, zs_short), vs_short; kernel=:b5, bc=:poly)
    @test itp.itp.bc == ((:linear, :linear), (:linear, :linear), (:linear, :linear))
end