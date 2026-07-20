_p(g,i,k)=gt_entry(g,i,k)+k-i
function _positive_ratio(num,den)
    den==0 && throw(DomainError(den,"singular CG product"))
    num//den
end
function _cgc_uncached(mu::GTPattern{D},j::Integer,lam::GTPattern{D},
                       ::Type{T}) where {D,T<:AbstractFloat}
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

_cgc_cache_scalar_type(::Nothing)=Float64

_resolve_cgc_scalar_type(::Nothing,::Nothing)=Float64
_resolve_cgc_scalar_type(::Nothing,cache)=_cgc_cache_scalar_type(cache)
function _resolve_cgc_scalar_type(::Type{T},cache) where T<:AbstractFloat
    isconcretetype(T)||throw(ArgumentError(
        "T must be a concrete AbstractFloat type, got $T"))
    # Cache compatibility is checked by `_cached_cgc`; keeping the return
    # value tied to the type parameter makes explicit-T hot queries inferable.
    T
end
_resolve_cgc_scalar_type(requested,cache)=throw(ArgumentError(
    "T must be a concrete AbstractFloat type, got $requested"))

"""
    cgc(mu, j, lam; T=nothing, cache=nothing)

Return the real GT Clebsch--Gordan coefficient of equations (B.10)--(B.11).
Without a cache, omitting `T` preserves the historical `Float64` result.  A
[`OneBoxCGCache`](@ref) supplies precomputed one-box coefficients and, when
`T` is omitted, selects its stored floating-point type.  An explicit `T` must
match the cache type; cached requests outside its exact basis-owned domain or
its configured depth raise instead of silently falling back to recomputation.
"""
function cgc(mu::GTPattern{D},j::Integer,lam::GTPattern{D};
             T=nothing,cache=nothing) where D
    R=_resolve_cgc_scalar_type(T,cache)
    cache===nothing ? _cgc_uncached(mu,j,lam,R) :
        _cached_cgc(cache,mu,j,lam,R)
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
    three_nu_symbol(WL, Wm, WR; T=nothing, cache=nothing)

Return the `D x D` dyadic of real one-box `U(D)` Clebsch--Gordan coefficients
associated with the patterns `Wm -> WL` and `Wm -> WR`. The local labels use
the paper's zero-based convention internally, and the returned matrix uses
ordinary one-based Julia indices.

With `cache=OneBoxCGCache(...)`, both coefficient vectors are read from the
precomputed sparse one-box transitions.  The returned matrix always owns
detached storage and may therefore be modified without changing the cache.
"""
function three_nu_symbol(WL::GTPattern{D},Wm::GTPattern{D},
                         WR::GTPattern{D};T=nothing,cache=nothing) where D
    R=_resolve_cgc_scalar_type(T,cache)
    a=[cgc(Wm,j,WL;T=R,cache=cache) for j in 0:D-1]
    b=[cgc(Wm,j,WR;T=R,cache=cache) for j in 0:D-1]
    a*b'
end
