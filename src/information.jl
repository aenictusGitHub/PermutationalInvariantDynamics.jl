"""Return the von Neumann entropy of a PI state, with logarithm base `base`."""
function von_neumann_entropy(rho::PIState;base::Real=2,
                            atol::Real=_analysis_atol(rho),
                            rtol::Real=_state_rtol(rho))
    base>0&&base!=1||throw(ArgumentError("invalid logarithm base"))
    validate_state(rho;atol=atol,rtol=rtol)
    Rtype=_real_float_type(eltype(rho.data))
    s=zero(Rtype)
    for p in rho.basis.sectors
        E=_weighted_sector_eigvals(rho,p;atol=atol,rtol=rtol,
            operation="von Neumann entropy")
        probability=zero(Rtype)
        for q in E.values
            q>zero(q)||(continue)
            probability+=q
            s-=q*log(q)
        end
        s+=probability*_log_schur_multiplicity(Rtype,p)
    end
    s/log(Rtype(base))
end

"""Return the order-`alpha` Rényi entropy of a PI state."""
function renyi_entropy(rho::PIState,alpha::Real;base::Real=2,
                       atol::Real=_analysis_atol(rho),
                       rtol::Real=_state_rtol(rho))
    alpha>0||throw(ArgumentError("Rényi order must be positive"))
    alpha==1&&return von_neumann_entropy(rho;base=base,atol=atol,rtol=rtol)
    base>0&&base!=1||throw(ArgumentError("invalid logarithm base"))
    validate_state(rho;atol=atol,rtol=rtol)
    Rtype=_real_float_type(eltype(rho.data))
    base_log=log(Rtype(base))
    if isinf(alpha)
        maximum_log_eigenvalue=Rtype(-Inf)
        for p in rho.basis.sectors
            E=_weighted_sector_eigvals(rho,p;atol=atol,rtol=rtol,
                operation="infinite-order Rényi entropy")
            logf=_log_schur_multiplicity(Rtype,p)
            for q in E.values
                q>zero(q)&&(maximum_log_eigenvalue=max(maximum_log_eigenvalue,
                                                       log(q)-logf))
            end
        end
        isfinite(maximum_log_eigenvalue)||throw(ArgumentError(
            "Rényi entropy found no positive density eigenvalue"))
        return -maximum_log_eigenvalue/base_log
    end
    alphaR=Rtype(alpha)
    isfinite(alphaR)||throw(ArgumentError(
        "Rényi order is not representable in $Rtype"))
    # log Tr(rho^alpha) is accumulated with streaming log-sum-exp.  This
    # avoids both f^(1-alpha) overflow and the corresponding underflow of
    # physical block eigenvalues.
    maximum_term=Rtype(-Inf)
    scaled_sum=zero(Rtype)
    for p in rho.basis.sectors
        E=_weighted_sector_eigvals(rho,p;atol=atol,rtol=rtol,
            operation="Rényi entropy")
        logf=_log_schur_multiplicity(Rtype,p)
        for q in E.values
            q>zero(q)||continue
            term=(one(Rtype)-alphaR)*logf+alphaR*log(q)
            if term>maximum_term
                scaled_sum=isfinite(maximum_term) ?
                    scaled_sum*exp(maximum_term-term)+one(Rtype) : one(Rtype)
                maximum_term=term
            else
                scaled_sum+=exp(term-maximum_term)
            end
        end
    end
    isfinite(maximum_term)||throw(ArgumentError(
        "Rényi entropy found no positive density eigenvalue"))
    logz=maximum_term+log(scaled_sum)
    logz/((one(Rtype)-alphaR)*base_log)
end

"""Entropy of the `k`-particle reduced state."""
function reduced_entropy(rho::PIState,k::Integer;base::Real=2,
                         atol::Real=_analysis_atol(rho),rtol::Real=_state_rtol(rho))
    reduced=reduced_state(rho,k;atol=atol,rtol=rtol)
    von_neumann_entropy(reduced;base=base,atol=atol,rtol=rtol)
end

"""Mutual information across the `k | N-k` bipartition."""
function mutual_information(rho::PIState,k::Integer;kwargs...)
    reduced_entropy(rho,k;kwargs...)+reduced_entropy(rho,rho.basis.N-k;kwargs...)-von_neumann_entropy(rho;kwargs...)
end

"""Conditional entropy `S(A|B)` for `A` containing `k` particles."""
conditional_entropy(rho::PIState,k::Integer;kwargs...)=
    von_neumann_entropy(rho;kwargs...)-
    reduced_entropy(rho,rho.basis.N-k;kwargs...)

"""Hilbert--Schmidt distance between compatible PI states."""
function hilbert_schmidt_distance(rho::PIState,sigma::PIState)
    _samebasis(rho,sigma);norm(rho.data-sigma.data)
end

"""Trace distance between compatible PI states."""
function trace_distance(rho::PIState,sigma::PIState;
                        atol::Real=max(_analysis_atol(rho),_analysis_atol(sigma)),
                        rtol::Real=max(_state_rtol(rho),_state_rtol(sigma)))
    _samebasis(rho,sigma)
    validate_state(rho;atol=atol,rtol=rtol)
    validate_state(sigma;atol=atol,rtol=rtol)
    Rtype=promote_type(_real_float_type(eltype(rho.data)),
                       _real_float_type(eltype(sigma.data)))
    z=zero(Rtype)
    for p in rho.basis.sectors
        D=_multiplicity_weighted_block(rho,p)-_multiplicity_weighted_block(sigma,p)
        z+=sum(abs,_hermitian_eigvals(
            Hermitian((D+D')/2);operation="trace distance"))
    end
    z/2
end

function _validated_psd_tolerance(values,::Type{R},atol,rtol;
                                  context::AbstractString) where R<:AbstractFloat
    scale=maximum(abs,values;init=zero(R))
    tolerance=R(atol)+R(rtol)*scale
    minimum(values;init=zero(R))>=-tolerance||throw(ArgumentError(
        "$context is not positive semidefinite within tolerance $tolerance"))
    tolerance
end

# Projection is deliberately confined to a square-root input after the full
# spectrum has passed the requested PSD validation. The state and returned
# spectrum are never modified or exposed as clipped data.
function _psd_square_roots(values,::Type{R},tolerance) where R<:AbstractFloat
    map(values) do value
        value>=-tolerance||error("internal error: unvalidated PSD spectrum")
        value>zero(R) ? sqrt(value) : zero(R)
    end
end

function _unit_interval_roundoff(value::R,atol,rtol;
                                 context::AbstractString) where R<:AbstractFloat
    isfinite(value)||throw(ArgumentError("$context is not finite"))
    scale=max(one(R),abs(value))
    # Two independently validated unit-trace inputs contribute to this bound;
    # the final term covers arithmetic after validation.
    correction=R(4)*(R(atol)+R(rtol)*scale)+R(64)*eps(R)*scale
    if value<zero(R)
        value>=-correction||throw(ArgumentError(
            "$context=$value lies below zero beyond roundoff tolerance $correction"))
        return zero(R)
    elseif value>one(R)
        value<=one(R)+correction||throw(ArgumentError(
            "$context=$value exceeds one beyond roundoff tolerance $correction"))
        return one(R)
    end
    value
end

function _numerical_support_tolerance(values,::Type{R}) where R<:AbstractFloat
    scale=maximum(abs,values;init=zero(R))
    R(8)*R(max(length(values),1))*eps(R)*scale
end

function _relative_entropy_block(Rblock,Sblock,::Type{R},atol,rtol;
                                 operation::AbstractString) where R<:AbstractFloat
    RH=Hermitian((Rblock+Rblock')/2);SH=Hermitian((Sblock+Sblock')/2)
    er=_hermitian_eigen(RH;operation)
    es=_hermitian_eigen(SH;operation)
    rho_validation_tolerance=_validated_psd_tolerance(er.values,R,atol,rtol;
        context="$operation rho block")
    _validated_psd_tolerance(es.values,R,atol,rtol;
        context="$operation sigma block")

    support_tolerance=_numerical_support_tolerance(es.values,R)
    weight_tolerance=_numerical_support_tolerance(er.values,R)
    # A negative projector weight can only come from the tolerance-sized
    # negative spectrum already accepted above (plus arithmetic roundoff).
    # This bound is deliberately one-sided: a positive nullspace weight is a
    # support violation and is compared only with the numerical-rank floor.
    negative_weight_tolerance=max(weight_tolerance,
        R(max(length(er.values),1))*rho_validation_tolerance)
    null_indices=findall(value->value<=support_tolerance,es.values)
    Rmatrix=Matrix(RH)
    if !isempty(null_indices)
        null_vectors=@view es.vectors[:,null_indices]
        null_weight=real(tr(adjoint(null_vectors)*Rmatrix*null_vectors))
        null_weight>weight_tolerance&&return R(Inf)
        null_weight>=-negative_weight_tolerance||throw(ArgumentError(
            "$operation produced a negative rho weight on sigma's numerical nullspace"))
    end

    result=zero(R)
    for value in er.values
        value>zero(R)&&(result+=value*log(value))
    end
    for index in eachindex(es.values)
        sigma_value=es.values[index]
        sigma_value>support_tolerance||continue
        vector=@view es.vectors[:,index]
        weight=real(dot(vector,Rmatrix*vector))
        weight>=-negative_weight_tolerance||throw(ArgumentError(
            "$operation produced a negative rho support weight"))
        weight>zero(R)&&(result-=weight*log(sigma_value))
    end
    result
end

"""
    fidelity(rho, sigma; atol, rtol)

Squared Uhlmann fidelity between compatible PI states. Input and intermediate
PSD spectra are validated before tolerance-sized negative roundoff is replaced
by zero solely for matrix square roots. A final value outside `[0,1]` is
corrected only within a scale-aware roundoff bound; a larger discrepancy
throws.
"""
function fidelity(rho::PIState,sigma::PIState;
                  atol::Real=max(_analysis_atol(rho),_analysis_atol(sigma)),
                  rtol::Real=max(_state_rtol(rho),_state_rtol(sigma)))
    _samebasis(rho,sigma)
    validate_state(rho;atol=atol,rtol=rtol)
    validate_state(sigma;atol=atol,rtol=rtol)
    Rtype=promote_type(_real_float_type(eltype(rho.data)),
                       _real_float_type(eltype(sigma.data)))
    root=zero(Rtype)
    for p in rho.basis.sectors
        R0=_multiplicity_weighted_block(rho,p)
        S0=_multiplicity_weighted_block(sigma,p)
        R=Hermitian((R0+R0')/2)
        S=Hermitian((S0+S0')/2)
        er=_hermitian_eigen(R;operation="fidelity")
        rtolerance=_validated_psd_tolerance(
            er.values,Rtype,atol,rtol;context="fidelity rho block")
        esvalues=_hermitian_eigvals(S;operation="fidelity")
        _validated_psd_tolerance(
            esvalues,Rtype,atol,rtol;context="fidelity sigma block")
        sqrtR=er.vectors*Diagonal(
            _psd_square_roots(er.values,Rtype,rtolerance))*er.vectors'
        kernel=sqrtR*Matrix(S)*sqrtR
        M=Hermitian((kernel+kernel')/2)
        vals=_hermitian_eigvals(M;operation="fidelity")
        mtolerance=_validated_psd_tolerance(
            vals,Rtype,atol,rtol;context="fidelity kernel")
        root+=sum(_psd_square_roots(vals,Rtype,mtolerance))
    end
    _unit_interval_roundoff(root^2,atol,rtol;context="squared fidelity")
end

"""Bures distance corresponding to the squared-fidelity convention."""
function bures_distance(rho::PIState,sigma::PIState;kwargs...)
    f=fidelity(rho,sigma;kwargs...);R=typeof(f)
    squared=R(2)-R(2)*sqrt(f)
    if squared<zero(R)
        tolerance=R(64)*eps(R)*R(2)
        squared>=-tolerance||throw(ArgumentError(
            "Bures distance squared is negative beyond roundoff tolerance"))
        squared=zero(R)
    end
    sqrt(squared)
end

"""
    quantum_relative_entropy(rho, sigma; base=2, atol, rtol)

Quantum relative entropy `D(rho || sigma)` in the requested logarithm base.
Support inclusion is decided once per Schur sector with the projector onto
sigma's numerical nullspace. Positive rho weight above the scale-aware
roundoff floor returns `Inf`; it is never discarded through separate
eigenvalue or eigenvector-overlap cutoffs.
"""
function quantum_relative_entropy(rho::PIState,sigma::PIState;base::Real=2,
                                  atol::Real=max(_analysis_atol(rho),_analysis_atol(sigma)),
                                  rtol::Real=max(_state_rtol(rho),_state_rtol(sigma)))
    _samebasis(rho,sigma)
    base>0&&base!=1||throw(ArgumentError("invalid logarithm base"))
    validate_state(rho;atol=atol,rtol=rtol)
    validate_state(sigma;atol=atol,rtol=rtol)
    Rtype=promote_type(_real_float_type(eltype(rho.data)),
                       _real_float_type(eltype(sigma.data)))
    D=zero(Rtype)
    for p in rho.basis.sectors
        R0=_multiplicity_weighted_block(rho,p)
        S0=_multiplicity_weighted_block(sigma,p)
        contribution=_relative_entropy_block(
            R0,S0,Rtype,atol,rtol;operation="quantum relative entropy")
        isinf(contribution)&&return Rtype(Inf)
        D+=contribution
    end
    D/log(Rtype(base))
end
