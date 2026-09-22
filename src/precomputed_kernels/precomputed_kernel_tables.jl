# Lookup for the precomputed kernel tables shipped as the "kernel_tables" artifact
# (bound in Artifacts.toml, built by gen/build_kernel_artifact.jl).
# The tables are computed once at exact rational precision and rounded to Float64.
# For :linear subgrid (high resolution), top derivatives and BigFloat, tables are
# computed and cached on demand instead (see get_precomputed_kernel_and_range.jl).
#
# Binary format of every .bin file (little-endian):
#   Int64 rows, Int64 cols, then rows*cols Float64 values in column-major order.
#   An empty table is stored as rows = cols = 0 followed by no data.
#
# File names: "pre_range.bin" (101×1), and "<kernel>_<tag>_<part>.bin" where
#   tag  = "i1" for derivative -1, "d0", "d1", ... otherwise
#   part = "kp", "kd1" or "kd2"

# Derivatives shipped per kernel: -1 (antiderivative) up to the highest shipped derivative.
# :a0 is listed as -1 in _max_shipped_derivative but also ships a derivative-0 table,
# hence the max(..., 0).
_shipped_derivatives(degree::Symbol) = -1:max(_max_shipped_derivative[degree], 0)

# True if the artifact holds a table for this kernel and derivative
_has_shipped_table(degree::Symbol, derivative::Int) =
    haskey(_max_shipped_derivative, degree) && derivative in _shipped_derivatives(degree)

# "i1" for the antiderivative, "d<k>" for derivative k
_kernel_table_tag(derivative::Int) = derivative == -1 ? "i1" : "d$(derivative)"

# File name of one table part inside the artifact
_kernel_table_filename(degree::Symbol, derivative::Int, part::String) =
    "$(degree)_$(_kernel_table_tag(derivative))_$(part).bin"

# Read one table in the format described above
function _read_kernel_table(path::String)
    open(path, "r") do io
        rows = read(io, Int64)
        cols = read(io, Int64)
        expected_bytes = 16 + 8 * rows * cols
        filesize(path) == expected_bytes ||
            error("Corrupt kernel table $path: expected $expected_bytes bytes, found $(filesize(path))")
        M = Matrix{Float64}(undef, rows, cols)
        read!(io, M)
        return M
    end::Matrix{Float64}
end

function _build_kernel_tables(degree::Symbol, derivative::Int, ::Type{T}) where T
    _has_shipped_table(degree, derivative) ||
        error("No shipped kernel table for kernel=$degree, derivative=$derivative")
    dir = artifact"kernel_tables"
    pre_range = T.(vec(_read_kernel_table(joinpath(dir, "pre_range.bin"))))
    kp  = T.(_read_kernel_table(joinpath(dir, _kernel_table_filename(degree, derivative, "kp"))))
    kd1 = T.(_read_kernel_table(joinpath(dir, _kernel_table_filename(degree, derivative, "kd1"))))
    kd2 = T.(_read_kernel_table(joinpath(dir, _kernel_table_filename(degree, derivative, "kd2"))))
    return pre_range, kp, kd1, kd2
end

const _KERNEL_TABLE_CACHE = Dict{Tuple{Symbol,Int,DataType},Any}()
const _KERNEL_TABLE_LOCK = ReentrantLock()

function get_shipped_kernel_tables(degree::Symbol, derivative::Int, ::Type{T}) where T
    key = (degree, derivative, T)
    tbl = lock(_KERNEL_TABLE_LOCK) do
        get!(() -> _build_kernel_tables(degree, derivative, T), _KERNEL_TABLE_CACHE, key)
    end
    return tbl::Tuple{Vector{T}, Matrix{T}, Matrix{T}, Matrix{T}}
end