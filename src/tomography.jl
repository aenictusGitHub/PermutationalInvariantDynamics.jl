function _prepare_pi_povm(rho_or_basis,effects;atol,rtol,complete::Bool)
    basis=rho_or_basis isa PIBasis ? rho_or_basis : rho_or_basis.basis
    prepared=collect(effects);isempty(prepared)&&throw(ArgumentError(
        "a PI POVM must contain at least one effect"))
    all(E->E isa PIOperator&&E.basis===basis,prepared)||throw(ArgumentError(
        "PI POVM effects must be PIOperators on the exact same basis"))
    all(E->ishermitian(E;atol,rtol)&&ispositive(E;atol,rtol),prepared)||
        throw(ArgumentError("every PI POVM effect must be Hermitian positive semidefinite"))
    if complete
        R=foldl(promote_type,(_real_float_type(eltype(E.data)) for E in prepared))
        total=zeros(Complex{R},length(basis))
        for E in prepared;total.+=E.data;end
        identity=identity_operator(basis;T=R).data
        scale=max(norm(identity),one(R));tolerance=R(atol)+R(rtol)*scale
        norm(total-identity)<=tolerance||throw(ArgumentError(
            "PI POVM effects do not sum to the retained-algebra identity"))
    end
    prepared
end

"""
    pi_povm_probabilities(rho, effects; complete=true, atol=..., rtol=...)

Return Born probabilities for PI POVM effects without reconstructing a
`d^N` density matrix.  Effects are required to be Hermitian positive
semidefinite; with `complete=true` they must sum to the retained PI identity.
No negative value is clipped: values outside the stated tolerance raise, while
roundoff-sized negative values are returned as exact zero only inside this
probability-only output buffer.
"""
function pi_povm_probabilities(rho::PIState,effects;complete::Bool=true,
        atol::Real=_analysis_atol(rho),rtol::Real=_state_rtol(rho))
    validate_state(rho;atol,rtol)
    prepared=_prepare_pi_povm(rho,effects;atol,rtol,complete)
    R=foldl(promote_type,(_real_float_type(eltype(E.data)) for E in prepared);
            init=_real_float_type(eltype(rho.data)))
    probabilities=Vector{R}(undef,length(prepared))
    tolerance=R(atol)+R(rtol)
    for index in eachindex(prepared)
        value=expectation(rho,prepared[index])
        abs(imag(value))<=tolerance||throw(ArgumentError(
            "PI POVM probability $index has a non-negligible imaginary part"))
        probability=R(real(value));probability>=-tolerance||throw(ArgumentError(
            "PI POVM probability $index is negative: $probability"))
        probabilities[index]=probability<zero(R) ? zero(R) : probability
    end
    if complete
        abs(sum(probabilities)-one(R))<=R(atol)+R(rtol)*max(one(R),sum(abs,probabilities))||
            throw(ArgumentError("PI POVM probabilities do not sum to one"))
    end
    probabilities
end

"""Outcome counts and exact model probabilities from [`sample_pi_povm`](@ref)."""
struct PIPOVMSample{R}
    counts::Vector{Int}
    probabilities::Vector{R}
    shots::Int
end

"""
    sample_pi_povm(rho, effects, shots; rng=Random.default_rng(), kwargs...)

Draw categorical PI POVM outcomes.  The cumulative table is built once and
sampling uses binary search, so cost is `O(number_of_effects + shots*log M)`.
"""
function sample_pi_povm(rho::PIState,effects,shots::Integer;
        rng=Random.default_rng(),kwargs...)
    shots>=0||throw(ArgumentError("shot count must be nonnegative"))
    probabilities=pi_povm_probabilities(rho,effects;kwargs...)
    cumulative=cumsum(probabilities);cumulative[end]=one(eltype(cumulative))
    counts=zeros(Int,length(probabilities))
    for _ in 1:shots
        outcome=searchsortedfirst(cumulative,rand(rng,eltype(probabilities)))
        counts[outcome]+=1
    end
    PIPOVMSample(counts,probabilities,Int(shots))
end

"""Result of constrained PI maximum-likelihood tomography."""
struct PITomographyResult{S,R}
    state::S
    loglikelihood::R
    iterations::Int
    converged::Bool
    residual::R
    probabilities::Vector{R}
    algorithm::Symbol
end

function Base.show(io::IO,result::PITomographyResult)
    print(io,"PITomographyResult(converged=$(result.converged), " *
        "iterations=$(result.iterations), loglikelihood=$(result.loglikelihood))")
end

function _povm_loglikelihood(counts,probabilities)
    R=eltype(probabilities);value=zero(R)
    for index in eachindex(counts)
        count=counts[index];iszero(count)&&continue
        probabilities[index]>zero(R)||return R(-Inf)
        value+=R(count)*log(probabilities[index])
    end
    value
end

"""
    maximum_likelihood_tomography(basis, effects, counts;
        initial_state=nothing, maxiter=10_000, atol=1e-10, rtol=1e-8,
        dilution=1, require_convergence=false)

Estimate a PI state with the positivity- and trace-preserving diluted
`R*rho*R` maximum-likelihood iteration.  Every iterate is represented in
Schur blocks; the full Hilbert space is never formed.  `dilution` in `(0,1]`
mixes the likelihood operator with the identity and can stabilize incomplete
or ill-conditioned POVMs.

The returned [`PITomographyResult`](@ref) always exposes convergence.  Set
`require_convergence=true` to raise instead of returning the best final
iterate.  This is a constrained MLE within the retained PI algebra, not a
tomography model for permutation-breaking states.
"""
function maximum_likelihood_tomography(basis::PIBasis,effects,counts;
        initial_state=nothing,maxiter::Integer=10_000,atol::Real=1e-10,
        rtol::Real=1e-8,dilution::Real=1,require_convergence::Bool=false)
    maxiter>0||throw(ArgumentError("maxiter must be positive"))
    atol>=0&&rtol>=0||throw(ArgumentError("tomography tolerances must be nonnegative"))
    0<dilution<=1||throw(ArgumentError("dilution must lie in (0,1]"))
    observed=collect(counts);all(c->c isa Integer&&c>=0,observed)||
        throw(ArgumentError("tomography counts must be nonnegative integers"))
    shots=sum(observed);shots>0||throw(ArgumentError(
        "tomography requires at least one observed shot"))
    reference=initial_state===nothing ? maximally_mixed_state(basis) : initial_state
    reference isa PIState&&reference.basis===basis||throw(ArgumentError(
        "initial tomography state must use the exact supplied basis"))
    R=_real_float_type(eltype(reference.data));absolute=R(atol);relative=R(rtol)
    prepared=_prepare_pi_povm(reference,effects;atol=absolute,rtol=relative,
                              complete=true)
    length(prepared)==length(observed)||throw(DimensionMismatch(
        "tomography count and effect lengths differ"))
    rho=copy(reference);validate_state(rho;atol=absolute,rtol=relative);normalize!(rho)
    identity=identity_operator(basis;T=R);residual=R(Inf);converged=false
    probabilities=pi_povm_probabilities(rho,prepared;atol=absolute,
                                         rtol=relative)
    loglikelihood=_povm_loglikelihood(observed,probabilities)
    iteration=0
    for current_iteration in 1:maxiter
        iteration=current_iteration
        Rop=PIOperator(basis;T=R)
        for index in eachindex(prepared)
            count=observed[index];iszero(count)&&continue
            probability=probabilities[index]
            probability>zero(R)||throw(ArgumentError(
                "a positive-count tomography outcome has zero model probability; choose a full-rank initial_state or nonzero dilution"))
            coefficient=R(count)/(R(shots)*probability)
            @. Rop.data=Rop.data+coefficient*prepared[index].data
        end
        rho_operator=PIOperator(basis,rho.data)
        accepted=false;step=R(dilution);updated=rho
        next_probabilities=probabilities;next_loglikelihood=loglikelihood
        likelihood_tolerance=R(100)*eps(R)*max(one(R),abs(loglikelihood))
        # Undiluted RrhoR can overshoot for projective or incomplete POVMs.
        # Monotone backtracking preserves the constrained-likelihood ascent
        # without changing the user-selected maximum trial step.
        for _ in 1:48
            trial=PIOperator(basis;T=R)
            @. trial.data=step*Rop.data+(one(R)-step)*identity.data
            updated_operator=trial*rho_operator*adjoint(trial)
            candidate=PIState(basis,updated_operator.data);normalize!(candidate)
            candidate_probabilities=pi_povm_probabilities(candidate,prepared;
                atol=absolute,rtol=relative)
            candidate_loglikelihood=_povm_loglikelihood(
                observed,candidate_probabilities)
            if candidate_loglikelihood+likelihood_tolerance>=loglikelihood
                updated=candidate;next_probabilities=candidate_probabilities
                next_loglikelihood=candidate_loglikelihood;accepted=true;break
            end
            step/=R(2)
        end
        accepted||throw(ErrorException(
            "diluted RrhoR tomography could not find a nondecreasing likelihood step; widen precision or inspect the POVM"))
        residual=R(norm(updated.data-rho.data))
        likelihood_change=abs(next_loglikelihood-loglikelihood)
        rho=updated;probabilities=next_probabilities
        loglikelihood=next_loglikelihood
        tolerance=absolute+relative*max(norm(rho.data),one(R))
        # Compare the change itself rather than scaling by the extensive
        # log-likelihood magnitude: the latter can stop many-shot fits well
        # before their probabilities are stationary.
        likelihood_converged=likelihood_change<=absolute+relative
        if residual<=tolerance||likelihood_converged
            converged=true;break
        end
    end
    require_convergence&&!converged&&throw(ArgumentError(
        "PI maximum-likelihood tomography did not converge in $maxiter iterations; residual=$residual"))
    PITomographyResult(rho,loglikelihood,iteration,converged,residual,
                       probabilities,:diluted_RrhoR)
end

maximum_likelihood_tomography(sample::PIPOVMSample,basis::PIBasis,effects;
                              kwargs...)=
    maximum_likelihood_tomography(basis,effects,sample.counts;kwargs...)
