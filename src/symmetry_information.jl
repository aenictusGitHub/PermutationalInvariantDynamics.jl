_xlogx(x,atol)=x>zero(x) ? x*log(x) : zero(x)

"""Schur-sector probabilities and conditional entropy contributions."""
function sector_resolved_entropy(rho::PIState;base::Real=2,
                                 atol::Real=_analysis_atol(rho))
    base>0&&base!=1||throw(ArgumentError("invalid logarithm base"))
    Rtype=_real_float_type(eltype(rho.data))
    zeroR=zero(Rtype)
    baseR=Rtype(base)
    out=NamedTuple[]
    for p in rho.basis.sectors
        E=_weighted_sector_eigen(rho,p;atol=atol,
            operation="sector-resolved entropy")
        prob=real(LinearAlgebra.tr(E.block))
        prob>=-atol||throw(ArgumentError("negative sector probability"))
        prob=max(zeroR,prob)
        multiplicity_entropy=_log_schur_multiplicity(Rtype,p)/log(baseR)
        if prob<=zeroR
            push!(out,(sector=p,probability=prob,irrep_entropy=zeroR,
                       multiplicity_entropy,conditional_entropy=zeroR,
                       contribution=zeroR))
            continue
        end
        vals=E.values./prob
        sir=-sum(x->_xlogx(x,atol),vals)/log(baseR)
        cond=sir+multiplicity_entropy
        push!(out,(sector=p,probability=prob,irrep_entropy=sir,
                   multiplicity_entropy,conditional_entropy=cond,
                   contribution=prob*cond))
    end
    out
end

"""Decompose entropy into sector-label, conditional-irrep, and multiplicity parts."""
function entropy_decomposition(rho::PIState;base::Real=2,
                               atol::Real=_analysis_atol(rho))
    Rtype=_real_float_type(eltype(rho.data))
    rs=sector_resolved_entropy(rho;base=base,atol=atol)
    l=log(Rtype(base))
    classical=-sum(x->_xlogx(x.probability,atol),rs)/l
    intra=sum(x->x.probability*x.irrep_entropy,rs)
    mult=sum(x->x.probability*x.multiplicity_entropy,rs)
    (classical=classical,intra_sector=intra,multiplicity=mult,
     total=classical+intra+mult,sectors=rs)
end

"""Relative-entropy coherence in the stored GT basis, resolved by Schur sector."""
function sector_resolved_coherence(rho::PIState;base::Real=2,
                                   atol::Real=_analysis_atol(rho))
    Rtype=_real_float_type(eltype(rho.data))
    zeroR=zero(Rtype)
    baseR=Rtype(base)
    out=NamedTuple[]
    for e in sector_resolved_entropy(rho;base=base,atol=atol)
        if e.probability<=zeroR
            push!(out,(sector=e.sector,probability=e.probability,
                       coherence=zeroR,contribution=zeroR))
            continue
        end
        B=_multiplicity_weighted_block(rho,e.sector)/e.probability
        sd=-sum(x->_xlogx(real(x),atol),diag(B))/log(baseR)
        c=max(zeroR,sd-e.irrep_entropy)
        push!(out,(sector=e.sector,probability=e.probability,coherence=c,
                   contribution=e.probability*c))
    end
    out
end
"""Total relative-entropy coherence in the stored Gelfand--Tsetlin basis."""
relative_entropy_of_coherence(rho::PIState;kwargs...)=
    sum(x->x.contribution,sector_resolved_coherence(rho;kwargs...))

function _generator_operator(rho,G)
    A=if G isa AbstractMatrix
        _check_local_observable(rho,G)
        T=promote_type(_real_float_type(eltype(rho.data)),
                       _real_float_type(eltype(G)))
        CollectiveObservablePlan(rho.basis,G;
            cache=OneBodyGeometry(rho.basis;T=T))
    else
        G
    end
    if A isa CollectiveObservablePlan
        _check_collective_plan(A,rho.basis)
        LinearAlgebra.ishermitian(A.local_operator)||throw(ArgumentError(
            "generator must be Hermitian"))
        return A
    end
    A isa PIOperator||throw(ArgumentError(
        "generator must be a local matrix, CollectiveObservablePlan, or PIOperator"))
    _samebasis(rho,A);ishermitian(A)||throw(ArgumentError(
        "generator must be Hermitian"))
    A
end

_symmetry_generator_eltype(A::CollectiveObservablePlan)=eltype(A.local_operator)
_symmetry_generator_eltype(A::PIOperator)=eltype(A.data)
_symmetry_generator_block(A::CollectiveObservablePlan,p)=
    A.blocks[_sidx(A.basis,p)]
_symmetry_generator_block(A::PIOperator,p)=physical_block(A,p)

"""Pinch a PI state in eigenspaces of a local, prepared collective, or PI symmetry generator."""
function symmetry_twirl(rho::PIState,G;atol::Real=_analysis_atol(rho))
    A=_generator_operator(rho,G)
    Rtype=promote_type(_real_float_type(eltype(rho.data)),
                       _real_float_type(_symmetry_generator_eltype(A)))
    out=PIState(rho.basis;T=Rtype)
    for p in rho.basis.sectors
        H=Matrix(_symmetry_generator_block(A,p))
        E=_hermitian_eigen(Hermitian((H+H')/2);operation="symmetry twirl")
        C=E.vectors'*Matrix(coefficient_block(rho,p))*E.vectors
        for i in axes(C,1),j in axes(C,2)
            abs(E.values[i]-E.values[j])>atol&&(C[i,j]=0)
        end
        coefficient_block(out,p).=E.vectors*C*E.vectors'
    end
    out
end

"""Relative entropy of asymmetry, equal to `S(twirl(rho))-S(rho)`."""
function relative_entropy_of_asymmetry(rho::PIState,G;base::Real=2,
                                       atol::Real=_analysis_atol(rho),
                                       rtol::Real=_state_rtol(rho))
    base>0&&base!=1||throw(ArgumentError("invalid logarithm base"))
    A=_generator_operator(rho,G)
    source_entropy=von_neumann_entropy(rho;base=base,atol=atol,rtol=rtol)
    Rtype=promote_type(_real_float_type(eltype(rho.data)),
                       _real_float_type(_symmetry_generator_eltype(A)))
    twirled_entropy=zero(Rtype)
    for p in rho.basis.sectors
        H=Matrix(_symmetry_generator_block(A,p))
        E=_hermitian_eigen(Hermitian((H+H')/2);
            operation="relative entropy of asymmetry")

        # Pinch the multiplicity-weighted density block rather than the
        # stored equation-(7) coefficient.  Both differ by a scalar and hence
        # have the same generator-eigenbasis pinching, but the weighted block
        # remains O(1) when C_nu is subnormal at very large N.  Computing the
        # entropy before converting back therefore avoids a needless loss of
        # relative trace accuracy.  Ordinary sectors still use the direct
        # one-multiply path inside `_multiplicity_weighted_block`.
        B=E.vectors'*_multiplicity_weighted_block(rho,p)*E.vectors
        for i in axes(B,1),j in axes(B,2)
            abs(E.values[i]-E.values[j])>atol&&(B[i,j]=0)
        end
        values=_hermitian_eigvals(Hermitian((B+B')/2);
            operation="relative entropy of asymmetry")
        scale=maximum(abs,values;init=zero(Rtype))
        tolerance=Rtype(atol)+Rtype(rtol)*scale
        minimum(values;init=zero(Rtype))>=-tolerance||throw(ArgumentError(
            "symmetry-twirled state has a negative eigenvalue in sector $p"))
        probability=zero(Rtype)
        for q in values
            q>zero(q)||continue
            probability+=q
            twirled_entropy-=q*log(q)
        end
        twirled_entropy+=probability*_log_schur_multiplicity(Rtype,p)
    end
    twirled_entropy/log(Rtype(base))-Rtype(source_entropy)
end
"""Alias for [`relative_entropy_of_asymmetry`](@ref)."""
relative_entropy_of_symmetry(args...;kwargs...)=relative_entropy_of_asymmetry(args...;kwargs...)

"""Wigner--Yanase skew information for a local, prepared collective, or PI generator."""
function wigner_yanase_asymmetry(rho::PIState,G;atol::Real=_analysis_atol(rho))
    A=_generator_operator(rho,G)
    Rtype=promote_type(_real_float_type(eltype(rho.data)),
                       _real_float_type(_symmetry_generator_eltype(A)))
    z=zero(Rtype);half=one(Rtype)/2
    for p in rho.basis.sectors
        E=_weighted_sector_eigen(rho,p;atol=atol,rtol=0,
            operation="Wigner--Yanase asymmetry")
        Ge=E.vectors'*Matrix(_symmetry_generator_block(A,p))*E.vectors
        for i in eachindex(E.values),j in eachindex(E.values)
            # Square roots amplify roundoff eigenvalues of an exactly
            # rank-deficient state; the positivity tolerance already defines
            # these accepted values as zero.
            root_i=E.values[i]>E.zero_tolerance ? sqrt(E.values[i]) : zero(Rtype)
            root_j=E.values[j]>E.zero_tolerance ? sqrt(E.values[j]) : zero(Rtype)
            z+=half*(root_i-root_j)^2*abs2(Ge[i,j])
        end
    end
    max(zero(Rtype),z)
end

"""QFI contributions of normalized conditional sectors for a local, prepared collective, or PI generator."""
function sector_resolved_qfi(rho::PIState,G;atol::Real=_analysis_atol(rho))
    A=_generator_operator(rho,G)
    Rtype=promote_type(_real_float_type(eltype(rho.data)),
                       _real_float_type(_symmetry_generator_eltype(A)))
    zeroR=zero(Rtype);out=NamedTuple[]
    for p in rho.basis.sectors
        E=_weighted_sector_eigen(rho,p;atol=atol,rtol=0,
            operation="sector-resolved quantum Fisher information")
        prob=real(tr(E.block))
        if prob<=zeroR
            push!(out,(sector=p,probability=max(prob,zeroR),qfi=zeroR,
                       contribution=zeroR))
            continue
        end
        Ge=E.vectors'*Matrix(_symmetry_generator_block(A,p))*E.vectors
        s=zero(Rtype)
        for i in eachindex(E.values),j in eachindex(E.values)
            den=E.values[i]+E.values[j]
            den>E.zero_tolerance&&
                (s+=2*(E.values[i]-E.values[j])^2/den*abs2(Ge[i,j]))
        end
        contribution=s
        push!(out,(sector=p,probability=prob,qfi=contribution/prob,
                   contribution=contribution))
    end
    out
end

"""
    relative_entropy_decomposition(rho, sigma; base=2, atol, rtol)

Classical sector-label and conditional quantum parts of `D(rho||sigma)`.
Conditional support is tested with a sector-level projector onto sigma's
numerical nullspace; positive rho weight above the documented roundoff floor
returns `Inf` rather than being discarded overlap by overlap.
"""
function relative_entropy_decomposition(rho::PIState,sigma::PIState;base::Real=2,
        atol::Real=max(_analysis_atol(rho),_analysis_atol(sigma)),
        rtol::Real=max(_state_rtol(rho),_state_rtol(sigma)))
    _samebasis(rho,sigma)
    base>0&&base!=1||throw(ArgumentError("invalid logarithm base"))
    validate_state(rho;atol=atol,rtol=rtol)
    validate_state(sigma;atol=atol,rtol=rtol)
    Rtype=promote_type(_real_float_type(eltype(rho.data)),
                       _real_float_type(eltype(sigma.data)))
    zeroR=zero(Rtype)
    baseR=Rtype(base)
    classical=zeroR
    intra=zeroR
    sectors=NamedTuple[]
    for p in rho.basis.sectors
        R=_multiplicity_weighted_block(rho,p);S=_multiplicity_weighted_block(sigma,p)
        pr=real(tr(R));ps=real(tr(S))
        if pr<=zeroR
            push!(sectors,(sector=p,rho_probability=max(pr,zeroR),
                           sigma_probability=max(ps,zeroR),
                           conditional_relative_entropy=zeroR,
                           contribution=zeroR))
            continue
        end
        ps<=zeroR&&return (classical=Rtype(Inf),intra_sector=Rtype(Inf),
                          total=Rtype(Inf),sectors=sectors)
        classical+=pr*log(pr/ps)/log(baseR)
        d=_relative_entropy_block(R/pr,S/ps,Rtype,atol,rtol;
            operation="conditional quantum relative entropy")
        isinf(d)&&return (classical=Rtype(Inf),
            intra_sector=Rtype(Inf),total=Rtype(Inf),sectors=sectors)
        d/=log(baseR)
        intra+=pr*d
        push!(sectors,(sector=p,rho_probability=pr,sigma_probability=ps,
                       conditional_relative_entropy=d,contribution=pr*d))
    end
    (classical=classical,intra_sector=intra,total=classical+intra,sectors=sectors)
end

"""Split tangent-state QFIM into classical sector-population and intra-sector parts."""
function qfim_sector_decomposition(rho::PIState,derivatives;
                                    atol::Real=_analysis_atol(rho))
    ds=collect(derivatives)
    all(d->d isa PIState&&d.basis===rho.basis,ds)||throw(ArgumentError(
        "derivatives must be PIState tangents on the same basis"))
    Rtype=foldl(promote_type,(_real_float_type(eltype(d.data)) for d in ds);
                    init=_real_float_type(eltype(rho.data)))
    zeroR=zero(Rtype)
    m=length(ds)
    classical=zeros(Rtype,m,m)
    sector_terms=NamedTuple[]
    for p in rho.basis.sectors
        prob=real(tr(_multiplicity_weighted_block(rho,p)))
        dp=Rtype[real(tr(_multiplicity_weighted_block(d,p))) for d in ds]
        C=zeros(Rtype,m,m)
        if prob>zeroR
            for mu in 1:m,nu in mu:m
                C[mu,nu]=C[nu,mu]=dp[mu]*dp[nu]/prob
            end
            classical .+= C
        elseif any(abs.(dp).>atol)
            throw(ArgumentError("a tangent changes a zero-probability sector, giving singular Fisher information"))
        end
        push!(sector_terms,(sector=p,probability=max(prob,zeroR),
                            probability_derivatives=dp,classical_qfim=C))
    end
    total=qfim_from_derivatives(rho,ds;atol=atol)
    intra=total-classical
    intra=(intra+intra')/2
    (classical=classical,intra_sector=intra,total=total,sectors=sector_terms)
end
