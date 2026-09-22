# gen/build_kernel_artifact.jl
#
# Builds the "kernel_tables" artifact for ConvolutionInterpolations.jl.
#
# Run from the package root:
#     julia --project=. gen/build_kernel_artifact.jl
#
# What it does:
#   1. Recomputes every shipped (kernel, derivative) table at exact rational precision
#      with the package's own precompute_kernel_and_range, rounded to Float64.
#   2. Compares each table with what the package currently ships (_build_kernel_tables)
#      and aborts, writing nothing, if any table differs.
#   3. Writes the tables as binary files into a new artifact in the local artifact store
#      (~/.julia/artifacts/<tree-hash>).
#   4. Archives that artifact to a tarball in tempdir() for upload to a GitHub release.
#   5. Binds the artifact in the package's Artifacts.toml with the release URL and the
#      tarball's SHA256 (overwriting any existing binding of the same name).
#
# Binary format of every .bin file (little-endian, as on every platform Julia runs on):
#   Int64 rows, Int64 cols, then rows*cols Float64 values in column-major order.
#   An empty table is stored as rows = cols = 0 followed by no data.
#
# File names: "pre_range.bin" (101×1), and "<kernel>_<tag>_<part>.bin" where
#   tag  = "i1" for derivative -1, "d0", "d1", ... otherwise
#   part = "kp", "kd1" or "kd2"

using ConvolutionInterpolations
using Pkg.Artifacts

const CI = ConvolutionInterpolations

# Artifact identity and download location
const ARTIFACT_NAME = "kernel_tables"
const RELEASE_TAG   = "kernel-tables-v1"
const TARBALL_NAME  = "kernel_tables-v1.tar.gz"
const DOWNLOAD_URL  = "https://github.com/NikoBiele/ConvolutionInterpolations.jl/releases/download/$(RELEASE_TAG)/$(TARBALL_NAME)"

# Resolution of the shipped tables
const PRECOMPUTE = 101

# Kernels with shipped tables
const KERNELS = (:a0, :a1, :a3, :a4, :a5, :a7, :b5, :b7, :b9, :b11, :b13)

# Derivatives shipped per kernel: -1 (antiderivative) up to the highest shipped derivative.
# :a0 is listed as -1 in _max_shipped_derivative but also ships a derivative-0 table,
# hence the max(..., 0).
shipped_derivatives(kernel::Symbol) = -1:max(CI._max_shipped_derivative[kernel], 0)

# "i1" for the antiderivative, "d<k>" for derivative k
table_tag(derivative::Int) = derivative == -1 ? "i1" : "d$(derivative)"

# File name of one table part inside the artifact
table_filename(kernel::Symbol, derivative::Int, part::String) =
    "$(kernel)_$(table_tag(derivative))_$(part).bin"

# Write one matrix in the format described in the header
function write_table(path::String, M::Matrix{Float64})
    open(path, "w") do io
        write(io, Int64(size(M, 1)), Int64(size(M, 2)))
        write(io, M)
    end
    return nothing
end

# Describe how two matrices differ, or return nothing if they are equal
function describe_difference(fresh::Matrix{Float64}, shipped::Matrix{Float64})
    if size(fresh) != size(shipped)
        return "size $(size(fresh)) vs shipped $(size(shipped))"
    elseif fresh != shipped
        n_diff  = count(fresh .!= shipped)
        max_abs = maximum(abs.(fresh .- shipped))
        return "$(n_diff) entries differ, max |Δ| = $(max_abs)"
    else
        return nothing
    end
end

function main()
    # 1. Recompute every table exactly
    println("Recomputing $(sum(length(shipped_derivatives(k)) for k in KERNELS)) tables at exact rational precision...")
    pre_range = Float64[]
    tables = Dict{Tuple{Symbol,Int},NTuple{3,Matrix{Float64}}}()
    for kernel in KERNELS, derivative in shipped_derivatives(kernel)
        r, kp, kd1, kd2 = CI.precompute_kernel_and_range(kernel;
                                precompute=PRECOMPUTE, F=Float64, derivative=derivative)
        pre_range = r
        tables[(kernel, derivative)] = (kp, kd1, kd2)
    end

    # 2. Compare with what the package currently ships
    println("Comparing against the currently shipped tables...")
    mismatches = String[]
    for kernel in KERNELS, derivative in shipped_derivatives(kernel)
        r_old, kp_old, kd1_old, kd2_old = CI._build_kernel_tables(kernel, derivative, Float64)
        kp, kd1, kd2 = tables[(kernel, derivative)]
        if pre_range != r_old
            push!(mismatches, "$(kernel) $(table_tag(derivative)) pre_range differs")
        end
        for (part, fresh, shipped) in (("kp", kp, kp_old), ("kd1", kd1, kd1_old), ("kd2", kd2, kd2_old))
            msg = describe_difference(fresh, shipped)
            msg === nothing || push!(mismatches, "$(kernel) $(table_tag(derivative)) $(part): $(msg)")
        end
    end
    if !isempty(mismatches)
        println("MISMATCHES — nothing was written:")
        foreach(m -> println("  ", m), mismatches)
        error("Recomputed tables differ from the shipped tables ($(length(mismatches)) mismatches).")
    end
    println("All tables identical to the shipped constants.")

    # 3. Write the artifact into the local artifact store
    tree_hash = create_artifact() do dir
        write_table(joinpath(dir, "pre_range.bin"), reshape(pre_range, :, 1))
        for ((kernel, derivative), (kp, kd1, kd2)) in tables
            write_table(joinpath(dir, table_filename(kernel, derivative, "kp")),  kp)
            write_table(joinpath(dir, table_filename(kernel, derivative, "kd1")), kd1)
            write_table(joinpath(dir, table_filename(kernel, derivative, "kd2")), kd2)
        end
    end

    # 4. Archive it for upload
    tarball_path = joinpath(tempdir(), TARBALL_NAME)
    rm(tarball_path; force=true)
    tarball_sha256 = archive_artifact(tree_hash, tarball_path)

    # 5. Bind it in the package's Artifacts.toml
    artifacts_toml = joinpath(pkgdir(CI), "Artifacts.toml")
    bind_artifact!(artifacts_toml, ARTIFACT_NAME, tree_hash;
                   download_info=[(DOWNLOAD_URL, tarball_sha256)], force=true)

    # Summary
    artifact_dir = artifact_path(tree_hash)
    n_files = length(readdir(artifact_dir))
    n_bytes = sum(filesize(joinpath(artifact_dir, f)) for f in readdir(artifact_dir))
    println()
    println("Artifact tree hash : ", tree_hash)
    println("Artifact contents  : $(n_files) files, $(n_bytes) bytes")
    println("Tarball            : ", tarball_path, " ($(filesize(tarball_path)) bytes)")
    println("Tarball SHA256     : ", tarball_sha256)
    println("Artifacts.toml     : ", artifacts_toml)
    println("Upload the tarball to release tag '$(RELEASE_TAG)' so that this URL resolves:")
    println("  ", DOWNLOAD_URL)
    return nothing
end

main()