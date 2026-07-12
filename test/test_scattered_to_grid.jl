println("\n" * "-"^60)
println("Testing scattered_to_grid...")
println("-"^60)

# Test scattered_to_grid: nearest-neighbor gridding of scattered data.
# Covers only this package's responsibilities — index bookkeeping,
# dimension conventions, averaging arithmetic, and argument guards.
# Nearest-neighbor search itself is delegated to NearestNeighbors.jl
# and is not re-verified here beyond brute-force equivalence, which
# primarily validates the query construction and output indexing.

@testset "scattered_to_grid argument guards" begin
    println("    - Argument guards")
    pts = rand(2, 10)
    vals = rand(10)
    xs = range(0.0, 1.0, length=5)

    @test_throws ArgumentError scattered_to_grid(pts, rand(9), (xs, xs))       # value count mismatch
    @test_throws ArgumentError scattered_to_grid(pts, vals, (xs,))             # too few knot dims
    @test_throws ArgumentError scattered_to_grid(rand(3, 10), vals, (xs, xs))  # too many point dims
    @test_throws ArgumentError scattered_to_grid(pts, vals, (xs, xs); k=0)     # k below range
    @test_throws ArgumentError scattered_to_grid(pts, vals, (xs, xs); k=11)    # k above n
end

@testset "scattered_to_grid exact cases" begin
    println("    - Exact cases")
    xs = range(0.0, 1.0, length=4)

    # single data point: whole grid takes its value
    g = scattered_to_grid(reshape([0.5, 0.5], 2, 1), [7.0], (xs, xs))
    @test all(g .== 7.0)
    @test size(g) == (4, 4)

    # k = n: every node gets the global mean (isolates averaging path)
    pts = rand(2, 8)
    vals = randn(8)
    gk = scattered_to_grid(pts, vals, (xs, xs); k=8)
    @test all(isapprox.(gk, sum(vals)/8; atol=1e-14))
end

@testset "scattered_to_grid non-square grid orientation" begin
    println("    - Non-square grid orientation")
    # Different lengths per axis; data point near a known node.
    # A transposed index mapping cannot pass this test.
    xs = range(0.0, 1.0, length=3)   # dim 1: 3 nodes
    ys = range(0.0, 1.0, length=7)   # dim 2: 7 nodes
    pts = reshape([0.0, 1.0], 2, 1)  # near node (xs[1], ys[7])
    g = scattered_to_grid(pts, [5.0], (xs, ys))
    @test size(g) == (3, 7)

    # two points distinguish the axes: value depends on position in BOTH dims
    pts2 = [0.0 1.0; 0.0 1.0]        # points at (0,0) and (1,1)
    vals2 = [1.0, 2.0]
    g2 = scattered_to_grid(pts2, vals2, (xs, ys))
    @test g2[1, 1] == 1.0            # node (0.0, 0.0) → nearest is point 1
    @test g2[3, 7] == 2.0            # node (1.0, 1.0) → nearest is point 2
end

@testset "scattered_to_grid brute-force equivalence $(d)D" for d in 2:3
    println("    - Brute-force equivalence $(d)D")
    n = 50
    pts = rand(d, n)
    vals = randn(n)
    knots = ntuple(i -> range(0.0, 1.0, length=4 + i), d)  # deliberately non-square

    for k in (1, 3)
        g = scattered_to_grid(pts, vals, knots; k=k)
        @test size(g) == length.(knots)

        ref = similar(g)
        for I in CartesianIndices(ref)
            q = [knots[dim][I[dim]] for dim in 1:d]
            dists = [sum(abs2, pts[:, j] .- q) for j in 1:n]
            idxs = partialsortperm(dists, 1:k)
            ref[I] = sum(vals[idxs]) / k
        end
        @test g ≈ ref atol=1e-14
    end
end

@testset "scattered_to_grid duplicate points" begin
    println("    - Duplicate points")
    # repeated measurement at the same location
    xs = range(0.0, 1.0, length=3)
    pts = [0.5 0.5; 0.5 0.5]         # two points at (0.5, 0.5)
    vals = [1.0, 3.0]
    g2 = scattered_to_grid(pts, vals, (xs, xs); k=2)
    @test all(g2 .== 2.0)            # mean of the duplicates everywhere
    g1 = scattered_to_grid(pts, vals, (xs, xs); k=1)
    @test all(v -> v == 1.0 || v == 3.0, g1)  # tie-break unspecified, either valid
end

@testset "scattered_to_grid types and determinism" begin
    println("    - Types and determinism")
    xs = range(0.0, 1.0, length=5)
    pts = rand(2, 20)

    # Int values promote to float
    g_int = scattered_to_grid(pts, collect(1:20), (xs, xs))
    @test eltype(g_int) <: AbstractFloat

    # Float32 values follow float(T)
    g_f32 = scattered_to_grid(pts, rand(Float32, 20), (xs, xs))
    @test eltype(g_f32) == Float32

    # identical inputs → identical outputs
    vals = randn(20)
    @test scattered_to_grid(pts, vals, (xs, xs); k=3) ==
          scattered_to_grid(pts, vals, (xs, xs); k=3)
end