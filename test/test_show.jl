println("\n" * "-"^60)
println("Testing show/display correctness...")
println("-"^60)

const ci = convolution_interpolation

const OUTER_MARKER = "ConvolutionExtrapolation("
const INNER_MARKER_direct = "ConvolutionInterpolation("
const INNER_MARKER_fast = "FastConvolutionInterpolation("

function check_show(obj; inner=true)
    s2 = sprint(show, obj)
    s3 = sprint(show, MIME("text/plain"), obj)
    @test s2 isa String && !isempty(s2)
    @test s3 isa String && !isempty(s3)
    @test occursin(OUTER_MARKER, s2) # wrapper + delegation actually fired
    @test sprint(show, [obj, obj]) isa String
    @test sprint(show, MIME("text/plain"), [obj, obj]) isa String
    @test sprint(show, obj; context = :compact => true) isa String
    if inner && hasproperty(obj, :itp)
        @test occursin(INNER_MARKER_direct, sprint(show, obj.itp)) || occursin(INNER_MARKER_fast, sprint(show, obj.itp))
        @test sprint(show, MIME("text/plain"), obj.itp) isa String
    end
end

# ---- fixtures ----
x1u = range(0, 2π, 20);  y1u = sin.(x1u)
x1n = [0.0, 0.3, 0.6, 1.0, 1.5, 2.1, 2.5, 3.0, 3.6, 4.1, 4.5, 5.0, 5.4, 5.9, 6.283]
y1n = sin.(x1n)
xg = range(0, 2π, 20);  yg = range(0, 2π, 20);  zg = range(0, 2π, 20)
z2 = [sin(a)*sin(b)        for a in xg, b in yg]
z3 = [sin(a)*sin(b)*sin(c) for a in xg, b in yg, c in zg]
x2n = [0.0, 0.4, 0.9, 1.5, 2.1, 2.8, 3.4, 4.0, 4.5, 5.1, 5.7, 6.283];  y2n = copy(x2n)
z2n = [sin(a)*sin(b) for a in x2n, b in y2n]
xg4 = range(0, 1, 8);  z4 = [a+b+c+d for a in xg4, b in xg4, c in xg4, d in xg4]

@testset "show" begin

    @testset "1D uniform kernels" begin
        @testset "kernel=$k" for k in (:a0, :a1, :a3, :a4, :a5, :a7, :b5)
            println("    - 1D uniform kernel: ", k)
            check_show(ci(x1u, y1u; kernel=k))
        end
    end

    @testset "1D nonuniform kernels" begin
        @testset "kernel=$k" for k in (:a0, :a1, :n3, :b5)
            println("    - 1D nonuniform kernel: ", k)
            check_show(ci(x1n, y1n; kernel=k))
        end
    end

    @testset "1D uniform derivatives" begin
        @testset "antideriv kernel=$k" for k in (:a0, :a1, :a3, :b5)   # -1 valid for all uniform
            println("    - 1D uniform antiderivative: ", k)
            check_show(ci(x1u, y1u; kernel=k, derivative=-1))
        end
        @testset "d1 kernel=$k" for k in (:a3, :a4, :a5, :a7, :b5)     # a0/a1 don't reach order 1
            println("    - 1D uniform first derivative: ", k)
            check_show(ci(x1u, y1u; kernel=k, derivative=1))
        end
        @testset "d$d b5" for d in (2, 3)
            println("    - 1D uniform b5 kernel, derivative order: ", d)
            check_show(ci(x1u, y1u; kernel=:b5, derivative=d))
        end
    end

    @testset "1D nonuniform derivatives" begin   # antiderivative not supported on nonuniform
        @testset "d$d b5" for d in (1, 2, 3)
            println("    - 1D nonuniform b5 kernel, derivative order: ", d)
            check_show(ci(x1n, y1n; kernel=:b5, derivative=d))
        end
    end

    @testset "extrapolation" begin
        @testset "extrap=$e" for e in (:line, :flat, :natural, Line(), Flat(), Natural())
            println("    - Extrapolation: ", e)
            check_show(ci(x1u, y1u; kernel=:b5, extrap=e))
            check_show(ci(x1n, y1n; kernel=:b5, extrap=e))
        end
    end

    @testset "boundary conditions" begin
        @testset "bc=$bc" for bc in (:detect, :poly, :linear, :quadratic)
            println("    - Boundary condition: ", bc)
            check_show(ci(x1u, y1u; kernel=:b5, bc=bc))
        end
        check_show(ci((xg, yg), z2; kernel=:b5,
                      bc=((:linear, :quadratic), (:detect, :poly))))   # per-dim/per-direction
    end

    @testset "dimensions" begin
        println("    - Dimensions")
        check_show(ci((xg, yg), z2; kernel=:b5))           # 2D uniform
        check_show(ci((x2n, y2n), z2n; kernel=:n3))        # 2D nonuniform
        check_show(ci((xg, yg, zg), z3; kernel=:b5))       # 3D uniform
    end

    @testset "per-dimension kernels" begin
        println("    - Per-dimension kernels")
        check_show(ci((xg, yg), z2; kernel=(:b5, :b7)))
        check_show(ci((xg, yg, zg), z3; kernel=(:b5, :b5, :b7)))
        check_show(ci((x2n, y2n), z2n; kernel=(:b7, :b5)))
    end

    @testset "per-dim & mixed derivatives" begin
        println("    - Per-dimension mixed derivatives")
        check_show(ci((xg, yg), z2; kernel=:b5, derivative=(1, 0)))
        check_show(ci((xg, yg), z2; kernel=:b5, derivative=(2, 1)))
        check_show(ci((xg, yg), z2; kernel=:b5, derivative=(-1, 1)))       # mixed integral/deriv
        check_show(ci((xg, yg, zg), z3; kernel=:b5, derivative=(-1, 1, 0)))
    end

    @testset "lazy" begin
        println("    - Lazy mode")
        check_show(ci((xg, yg, zg), z3; kernel=:b5, lazy=true))                       # N≤3 full
        check_show(ci((xg, yg, zg), z3; kernel=:b5, lazy=true, derivative=(1, 0, 2))) # N≤3 + deriv
        check_show(ci((xg4, xg4, xg4, xg4), z4; kernel=(:b5, :b5, :b5, :b5),
                      lazy=true, boundary_fallback=true))                             # N≥4 fallback
    end

    @testset "bigfloat types" begin
        println("    - BigFloat types")
        check_show(ci(BigFloat.(x1u), BigFloat.(y1u); kernel=:b5))
    end
    
    @testset "convolution_gaussian" begin
        println("    - convolution_gaussian")
        check_show(convolution_gaussian(x1u, y1u .+ 0.05 .* sin.(10 .* x1u), 0.1))   # 1D
        check_show(convolution_gaussian((xg, yg), z2, 0.1))                          # 2D
        check_show(convolution_gaussian((xg, yg, zg), z3, 0.1))                      # 3D
    end
end