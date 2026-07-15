_p(g,i,k)=gt_entry(g,i,k)+k-i
function _positive_ratio(num,den)
    den==0 && throw(DomainError(den,"singular CG product"))
    num//den
end
"""Real GT Clebsch–Gordan coefficient of equations (B.10)–(B.11)."""
function cgc(mu::GTPattern{D}, j::Integer, lam::GTPattern{D}; T::Type{<:AbstractFloat}=Float64) where D
    weight(shape(lam)) == weight(shape(mu))+1 || return zero(T)
    tau=triangular_shift(mu,lam,j); tau===nothing && return zero(T)
    # tau[k-j] is tau_k for k=j+1,...,D
    tj=tau[1]
    q=big(1)//big(1)
    for k in 1:j
        q *= _positive_ratio(big(_p(mu,tj,j+1)-_p(mu,k,j)),1)
    end
    for k in 1:j+1
        k==tj && continue
        q /= big(_p(mu,tj,j+1)-_p(mu,k,j+1))
    end
    sign=1
    for l in j+2:D
        tprev=tau[l-j-1]; tl=tau[l-j]
        sign *= tprev < tl ? -1 : 1 # sgn(tprev-tl), sgn(0)=1
        for k in 1:l
            k==tl && continue
            q *= _positive_ratio(big(_p(mu,tprev,l-1)-_p(mu,k,l)+1),
                                 big(_p(mu,tl,l)-_p(mu,k,l)))
        end
        for k in 1:l-1
            k==tprev && continue
            q *= _positive_ratio(big(_p(mu,tl,l)-_p(mu,k,l-1)),
                                 big(_p(mu,tprev,l-1)-_p(mu,k,l-1)+1))
        end
    end
    q < 0 && abs(Float64(q)) < 1e-14 && (q=abs(q))
    q < 0 && throw(DomainError(q,"negative squared CG coefficient"))
    T(sign)*_checked_sqrt_exact_ratio(
        T,numerator(q),denominator(q);context="squared CG coefficient")
end

"""
    partition_triangle(l, m, r)

Return whether `l` and `r` are both obtained from `m` by adding one box. This
is the partition selection rule for a one-body three-sector contraction.
"""
partition_triangle(l::Partition,m::Partition,r::Partition) =
    weight(l)==weight(m)+1 && weight(r)==weight(m)+1 &&
    m in [remove_corner(l,i) for i in removable_corners(l)] &&
    m in [remove_corner(r,i) for i in removable_corners(r)]

"""
    three_nu_symbol(WL, Wm, WR; T=Float64)

Return the `D x D` dyadic of real one-box `U(D)` Clebsch--Gordan coefficients
associated with the patterns `Wm -> WL` and `Wm -> WR`. The local labels use
the paper's zero-based convention internally, and the returned matrix uses
ordinary one-based Julia indices.
"""
function three_nu_symbol(WL::GTPattern{D}, Wm::GTPattern{D}, WR::GTPattern{D}; T=Float64) where D
    a=[cgc(Wm,j,WL;T=T) for j in 0:D-1]
    b=[cgc(Wm,j,WR;T=T) for j in 0:D-1]
    a*b'
end
