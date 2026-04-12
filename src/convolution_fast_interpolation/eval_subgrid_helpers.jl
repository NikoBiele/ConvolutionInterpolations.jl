# helper to evaluate subgrid kernel
@inline @inbounds function eval_sg_k(itp::FastConvolutionInterpolation, t_d::Number,
                                    idx_d::Int, col::Int, ::Val{D}, ::Val{SG}, h_pre_d::T) where {D,SG,T}
    if SG[D] == :quintic
        quintic_hermite(t_d,
            itp.kernel_pre[D][idx_d, col], itp.kernel_pre[D][idx_d+1, col],
            itp.kernel_d1_pre[D][idx_d, col], itp.kernel_d1_pre[D][idx_d+1, col],
            itp.kernel_d2_pre[D][idx_d, col], itp.kernel_d2_pre[D][idx_d+1, col],
            h_pre_d)
    elseif SG[D] == :cubic
        cubic_hermite(t_d,
            itp.kernel_pre[D][idx_d, col], itp.kernel_pre[D][idx_d+1, col],
            itp.kernel_d1_pre[D][idx_d, col], itp.kernel_d1_pre[D][idx_d+1, col],
            h_pre_d)
    else
        (one(T) - t_d) * itp.kernel_pre[D][idx_d, col] +
                        t_d * itp.kernel_pre[D][idx_d+1, col]
    end
end

# helper to evaluate subgrid integral kernel
@inline @inbounds function eval_sg_kt(itp::FastConvolutionInterpolation, jd::Int,
                                ::Val{D}, ::Val{SG}, h_pre_d::T, x::NTuple{N,Number}) where {D,SG,T,N}
    xjd = itp.knots[D][itp.eqs[D]] + (jd - itp.eqs[D]) * itp.h[D]
    sj = (x[D] - xjd) / itp.h[D]
    abs(sj) >= T(itp.eqs[D]) && return T(1//2) * T(sign(sj))
    col_float = T(itp.eqs[D]) + sj
    col = clamp(floor(Int, col_float) + 1, 1, 2 * itp.eqs[D])
    t_d = (col_float - T(col - 1)) * T(length(itp.pre_range[D]) - 1) + one(T)
    idx_d = clamp(floor(Int, t_d), 1, length(itp.pre_range[D]) - 1)
    t_d -= T(idx_d)
    eval_sg_k(itp, t_d, idx_d, col, Val(D), Val(SG), h_pre_d)
end