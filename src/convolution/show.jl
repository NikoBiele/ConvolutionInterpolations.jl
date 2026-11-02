function Base.show(io::IO, itp::ConvolutionInterpolation)
    N = length(itp.knots)
    T = eltype(itp.coefs)
    deg  = _show_deg(itp.kernel_sym)
    dord = _show_do(itp.derivative_order)
    grid = all(a -> a === nothing, itp.nonuniform) ? "uniform" : "nonuniform"
    sz   = join(size(itp.coefs), "×")
    kbc = _show_kbc(itp.bc)
    print(io, "ConvolutionInterpolation($T, $(N)D, coefs=$(sz), kernel=$deg, $dord, $grid, bc=$kbc)")
end

function Base.show(io::IO, etp::ConvolutionExtrapolation)
    bc_str = etp.et == :throw ? ":throw" :
             etp.et == :flat  ? ":flat"  :
             etp.et == :line  ? ":line"  : string(typeof(etp.et))
    print(io, "ConvolutionExtrapolation(")
    show(io, etp.itp)
    print(io, ", extrap=$bc_str)")
end

# --- Helpers ---

_show_S(S) = S isa Symbol ? ":$(S)" : allequal(S) ? ":$(S[1])" : "(" * join((":$s" for s in S), ", ") * ")"
_show_deg(::Val{S}) where S = _show_S(S)
_show_deg(::HigherOrderKernel{S}) where S = _show_S(S)
_show_deg(::HigherOrderMixedKernel{S}) where S = _show_S(S)
_show_deg(::LowerOrderKernel{S}) where S = _show_S(S)
_show_deg(::LowerOrderMixedKernel{S}) where S = _show_S(S)
_show_deg(::FullMixedOrderKernel{S}) where S = _show_S(S)
_show_deg(::NonUniformMixedOrderKernel{S}) where S = _show_S(S)

_show_deg(other) = other

_show_do(::DerivativeOrder{DO}) where DO = DO isa Tuple && length(DO) == 1 ? "derivative=$(DO[1])" : allequal(DO) ? "derivative=$(DO[1])" : "derivative=$DO"
_show_do(::FastIntegralOrders{DO}) where DO = length(DO) == 1 ? "derivative=$(DO[1])" : "derivative=$DO"

function _show_kbc(kbc)
    kbc isa Tuple || return "$kbc"
    # flatten all symbols
    all_syms = collect(Iterators.flatten(kbc))
    if allequal(all_syms)
        return ":$(all_syms[1])"
    else
        return string(kbc)
    end
end