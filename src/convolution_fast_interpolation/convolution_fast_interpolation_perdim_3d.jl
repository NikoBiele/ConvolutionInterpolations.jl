# ==============================================================
# 3D — linear only per dim
# ==============================================================
function (itp::FastConvolutionInterpolation{T,3,0,TCoefs,Axs,KA,Val{3},DG,EQ,PR,KP,KBC,
            DerivativeOrder{DO},FD,SD,Val{(:linear,:linear,:linear)},Val{false},Val{0}})(x::Vararg{Number,3}) where
            {T<:AbstractFloat,TCoefs<:AbstractArray{T,3},
            Axs<:Tuple{<:AbstractVector,<:AbstractVector,<:AbstractVector},
            KA<:Tuple{<:Nothing,<:Nothing,<:Nothing},DG<:AbstractMixedConvolutionKernel,EQ<:Tuple{Int,Int,Int},
            PR<:Tuple{<:AbstractVector,<:AbstractVector,<:AbstractVector},KP,
            KBC<:Tuple{<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol},<:Tuple{Symbol,Symbol}},DO,FD,SD}
            
    eqs_x,eqs_y,eqs_z = itp.eqs
    kp_x=itp.kernel_pre[1]; kp_y=itp.kernel_pre[2]; kp_z=itp.kernel_pre[3]
    i_float = (x[1]-itp.knots[1][1])/itp.h[1] + one(T)
    i = clamp(floor(Int,i_float), eqs_x, length(itp.knots[1])-eqs_x)
    idx_x,t_x,_ = _perdim_cidx(one(T)-(x[1]-itp.knots[1][i])/itp.h[1], itp.pre_range[1])
    j_float = (x[2]-itp.knots[2][1])/itp.h[2] + one(T)
    j = clamp(floor(Int,j_float), eqs_y, length(itp.knots[2])-eqs_y)
    idx_y,t_y,_ = _perdim_cidx(one(T)-(x[2]-itp.knots[2][j])/itp.h[2], itp.pre_range[2])
    k_float = (x[3]-itp.knots[3][1])/itp.h[3] + one(T)
    k = clamp(floor(Int,k_float), eqs_z, length(itp.knots[3])-eqs_z)
    idx_z,t_z,_ = _perdim_cidx(one(T)-(x[3]-itp.knots[3][k])/itp.h[3], itp.pre_range[3])
    r000=zero(T); r100=zero(T); r010=zero(T); r110=zero(T)
    r001=zero(T); r101=zero(T); r011=zero(T); r111=zero(T)
    @inbounds for p in -(eqs_z-1):eqs_z
        kz_0=kp_z[idx_z,p+eqs_z]; kz_1=kp_z[idx_z+1,p+eqs_z]
        @inbounds for m in -(eqs_y-1):eqs_y
            ky_0=kp_y[idx_y,m+eqs_y]; ky_1=kp_y[idx_y+1,m+eqs_y]
            @inbounds for l in -(eqs_x-1):eqs_x
                c=itp.coefs[i+l,j+m,k+p]
                kx_0=kp_x[idx_x,l+eqs_x]; kx_1=kp_x[idx_x+1,l+eqs_x]
                r000+=c*kx_0*ky_0*kz_0; r100+=c*kx_1*ky_0*kz_0
                r010+=c*kx_0*ky_1*kz_0; r110+=c*kx_1*ky_1*kz_0
                r001+=c*kx_0*ky_0*kz_1; r101+=c*kx_1*ky_0*kz_1
                r011+=c*kx_0*ky_1*kz_1; r111+=c*kx_1*ky_1*kz_1
            end
        end
    end
    ox=one(T)-t_x; oy=one(T)-t_y; oz=one(T)-t_z
    result = @fastmath (ox*oy*oz*r000 + t_x*oy*oz*r100 + ox*t_y*oz*r010 + t_x*t_y*oz*r110 +
                        ox*oy*t_z*r001 + t_x*oy*t_z*r101 + ox*t_y*t_z*r011 + t_x*t_y*t_z*r111)
    return result * _perdim_scale(itp.h, Val(DO))
end