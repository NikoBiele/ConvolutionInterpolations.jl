# Deprecated keywords: still accepted, ignored, and removed in a future release.
#
# Kernels are evaluated exactly from their polynomial pieces, so there are no precomputed
# kernel tables and no subgrid interpolation between table entries. The `precompute` and
# `subgrid` keywords therefore no longer have any effect.

"""
    _warn_deprecated_table_keywords(precompute, subgrid)

Warn (at most once per session) if `precompute` or `subgrid` was set, i.e. is anything
other than `nothing`. Both keywords are ignored.
"""
function _warn_deprecated_table_keywords(precompute, subgrid)
    if precompute !== nothing
        @warn "`precompute` no longer has any effect and will be removed in a future release. " *
              "Kernels are now evaluated exactly from their polynomial pieces, so no kernel " *
              "tables are precomputed, cached or written to disk." maxlog=1
    end
    if subgrid !== nothing
        @warn "`subgrid` no longer has any effect and will be removed in a future release. " *
              "Kernels are now evaluated exactly from their polynomial pieces, so there are no " *
              "precomputed tables to interpolate between; results are exact to rounding for " *
              "every kernel and derivative order." maxlog=1
    end
    return nothing
end