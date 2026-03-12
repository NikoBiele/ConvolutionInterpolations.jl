function Base.show(io::IO, itp::ConvolutionInterpolation)
    N = length(itp.knots)
    T = eltype(itp.coefs)
    deg  = _show_deg(itp.deg)
    dord = _show_do(itp.derivative_order)
    grid = itp.nb_weight_coeffs === nothing ? "uniform" : "nonuniform"
    sz   = join(size(itp.coefs), "×")
    print(io, "ConvolutionInterpolation($T, $(N)D, coefs=$(sz), degree=$deg, $dord, $grid)")
end
 
function Base.show(io::IO, etp::ConvolutionExtrapolation)
    bc_str = etp.et isa Throw ? "Throw" :
             etp.et isa Flat  ? "Flat"  :
             etp.et isa Line  ? "Line"  : string(typeof(etp.et))
    print(io, "ConvolutionExtrapolation(")
    show(io, etp.itp)
    print(io, ", bc=$bc_str)")
end

function Base.show(io::IO, itp::FastConvolutionInterpolation)
    N = length(itp.knots)
    T = eltype(itp.coefs)
    deg  = _show_deg(itp.deg)
    dord = _show_do(itp.derivative_order)
    sz   = join(size(itp.coefs), "×")
    print(io, "FastConvolutionInterpolation($T, $(N)D, coefs=$(sz), degree=$deg, $dord)")
end

# --- Helpers ---
 
_show_deg(::Val{S}) where S = S
_show_deg(other)             = other
 
_show_do(::DerivativeOrder{DO}) where DO = "derivative=$DO"
_show_do(::IntegralOrder)                = "antiderivative"
_show_do(::MixedIntegralOrder{DO}) where DO = "mixed=$DO"