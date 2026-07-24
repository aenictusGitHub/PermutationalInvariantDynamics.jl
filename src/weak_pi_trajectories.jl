"""
    WeakPIPseudoKet(basis, data; atol, rtol)

Normalized pseudo-ket in the direct sum of the retained Schur irreps,
``directsum_nu U_nu``.  Its sector slices are denoted ``psi_nu`` and map to a
physical PI density operator through

``C_nu = psi_nu psi_nu^dagger / sqrt(f^nu)``.

This is an auxiliary weak-PI trajectory state, not a pure state of labeled
particles and not a vector in the full ``d^N`` Hilbert space.  In particular,
relative phases between different Schur sectors have no physical meaning.
Construction checks dimensions, finiteness, and unit norm and never
normalizes the supplied vector silently.
"""
struct WeakPIPseudoKet{T<:AbstractFloat,B<:PIBasis}
    basis::B
    data::Vector{Complex{T}}
    function WeakPIPseudoKet(b::B,data::AbstractVector{Complex{T}};
            atol::Real=max(T(1e-10),T(100)*eps(T)),
            rtol::Real=max(T(1e-10),T(100)*eps(T))) where
            {T<:AbstractFloat,B<:PIBasis}
        expected=sum(length,b.patterns;init=0)
        length(data)==expected||throw(DimensionMismatch(
            "weak-PI pseudo-ket data have length $(length(data)); expected $expected"))
        atol>=0||throw(ArgumentError("atol must be nonnegative"))
        rtol>=0||throw(ArgumentError("rtol must be nonnegative"))
        all(isfinite,data)||throw(ArgumentError(
            "weak-PI pseudo-ket amplitudes must be finite"))
        value=norm(data)
        isfinite(value)||throw(ArgumentError(
            "weak-PI pseudo-ket norm is nonfinite"))
        abs(value-one(T))<=T(atol)+T(rtol)||throw(ArgumentError(
            "weak-PI pseudo-ket must have unit norm; norm=$value"))
        new{T,B}(b,collect(data))
    end
end

copy(state::WeakPIPseudoKet)=WeakPIPseudoKet(state.basis,state.data)
eltype(state::WeakPIPseudoKet)=eltype(state.data)
length(state::WeakPIPseudoKet)=length(state.data)
show(io::IO,state::WeakPIPseudoKet)=print(io,
    "WeakPIPseudoKet(N=$(state.basis.N), d=$(state.basis.d), dimension=$(length(state)))")

"""Return ``sum_nu dim(U_nu)``, the direct-sum weak-PI pseudo-ket dimension."""
weak_pi_dimension(b::PIBasis)=sum(length,b.patterns;init=0)

function _weak_pi_offsets(b::PIBasis)
    offsets=Vector{Int}(undef,length(b.sectors)+1)
    offsets[1]=1
    for sector in eachindex(b.sectors)
        offsets[sector+1]=offsets[sector]+length(b.patterns[sector])
    end
    offsets
end

@inline _weak_sector_range(offsets,sector)=offsets[sector]:offsets[sector+1]-1

# Preparing the exact sector scale once is important for ensemble
# postprocessing: a trajectory average visits every sector once per saved
# state.  The ordinary path deliberately retains the division used by
# `_divide_by_schur_multiplicity_scale`; the prepared inverse-square-root path
# is only used when the standalone divisor or the divided component is not
# representable in the working precision.
struct _WeakPIDensityScale{T<:AbstractFloat}
    divisor::T
    inverse::_PreparedExactScale{T,true}
    context::String
end

function _weak_pi_density_scales(b::PIBasis,::Type{T}) where T<:AbstractFloat
    map(b.sectors) do partition
        multiplicity=symmetric_group_dimension(partition)
        context="physical Schur-block scaling for $partition"
        divisor=try
            _checked_sqrt_exact_integer(T,multiplicity;
                context="square root of the sector multiplicity for $partition")
        catch error
            error isa ArgumentError||rethrow()
            zero(T)
        end
        inverse=_prepare_exact_scale(T,one(BigInt),multiplicity,Val(true);
                                     context)
        _WeakPIDensityScale(divisor,inverse,context)
    end
end

@inline function _weak_density_outer_value(left::Complex{T},right::Complex{T},
        scale::_WeakPIDensityScale{T}) where T<:AbstractFloat
    value=left*conj(right)
    if !iszero(scale.divisor)
        result=value/scale.divisor
        _ordinary_scaled_value_safe(result,value)&&return result
    end
    _apply_prepared_exact_scale(value,scale.inverse;context=scale.context)
end

function _weak_density_outer!(destination,psi,scale;accumulate::Bool)
    axes(destination,1)==axes(destination,2)==axes(psi,1)||
        throw(DimensionMismatch("weak-PI density block dimensions are incompatible"))
    @inbounds for column in axes(destination,2),row in axes(destination,1)
        value=_weak_density_outer_value(psi[row],psi[column],scale)
        if accumulate
            destination[row,column]+=value
        else
            destination[row,column]=value
        end
    end
    destination
end

# History-free physical-density reduction for weak-PI paths. The sampled
# object is quadratic in the pseudo-ket, so it must be formed before the
# within-path average. `mean` uses the orthonormal equation-(7) PI coefficient
# order and can therefore feed the ordinary Hilbert--Schmidt Welford reducer.
mutable struct _WeakPIDensitySampler{V,O,S,A}
    mean::V
    batch_mean::V
    count::Int
    batch_count::Int
    batch_size::Int
    first_output_index::Int
    coefficient_offsets::O
    pseudo_offsets::O
    scales::S
    batch_accumulator::A
end

function _WeakPIDensitySampler(plan,state::WeakPIPseudoKet{T},scales;
        first_output_index::Integer=2,
        batch_size::Integer=0) where T<:AbstractFloat
    first_output_index>0||throw(ArgumentError(
        "first weak-PI density sampling index must be positive"))
    batch_size>=0||throw(ArgumentError(
        "weak-PI density batch size must be nonnegative"))
    b=plan.model.basis
    state.basis===b||throw(ArgumentError(
        "pseudo-ket and weak-PI density sampler use incompatible bases"))
    length(scales)==length(b.sectors)||throw(DimensionMismatch(
        "weak-PI density sampler has the wrong number of sector scales"))
    mean=Vector{Complex{T}}(undef,length(b))
    batch_mean=batch_size>0 ? similar(mean) : similar(mean,0)
    batch_accumulator=batch_size>0 ? _OnlineStateAccumulator(mean) : nothing
    _WeakPIDensitySampler(mean,batch_mean,0,0,Int(batch_size),
        Int(first_output_index),b.offsets,plan.offsets,scales,
        batch_accumulator)
end

function _reset_weak_density_sampler!(sampler::_WeakPIDensitySampler)
    fill!(sampler.mean,zero(eltype(sampler.mean)))
    fill!(sampler.batch_mean,zero(eltype(sampler.batch_mean)))
    sampler.count=0
    sampler.batch_count=0
    sampler
end

function _record_weak_density!(sampler::_WeakPIDensitySampler,data,
                               output_index::Integer)
    output_index<sampler.first_output_index&&return sampler
    sampler.count+=1
    sampler.batch_size>0&&(sampler.batch_count+=1)
    R=_real_float_type(eltype(sampler.mean))
    count=_checked_statistics_count(R,sampler.count,
                                    "weak-PI trajectory time sample")
    batch_count=sampler.batch_size>0 ? _checked_statistics_count(
        R,sampler.batch_count,"weak-PI trajectory batch sample") : zero(R)
    @inbounds for sector in eachindex(sampler.scales)
        pseudo_first=sampler.pseudo_offsets[sector]
        dimension=sampler.pseudo_offsets[sector+1]-pseudo_first
        coordinate=sampler.coefficient_offsets[sector]
        scale=sampler.scales[sector]
        for column_offset in 0:dimension-1
            right=data[pseudo_first+column_offset]
            for row_offset in 0:dimension-1
                value=_weak_density_outer_value(
                    data[pseudo_first+row_offset],right,scale)
                sampler.mean[coordinate]+=
                    (value-sampler.mean[coordinate])/count
                sampler.batch_size>0&&
                    (sampler.batch_mean[coordinate]+=
                        (value-sampler.batch_mean[coordinate])/batch_count)
                coordinate+=1
            end
        end
    end
    if sampler.batch_size>0&&sampler.batch_count==sampler.batch_size
        accumulator=sampler.batch_accumulator
        accumulator===nothing&&throw(ErrorException(
            "internal weak-PI batch sampler has no accumulator"))
        _accumulate_state!(accumulator,sampler.batch_mean)
        fill!(sampler.batch_mean,zero(eltype(sampler.batch_mean)))
        sampler.batch_count=0
    end
    sampler
end

"""
    WeakPIBatchMeansDiagnostics

Autocorrelation-aware Hilbert--Schmidt uncertainty diagnostic optionally
attached to `weak_pi_trajectory_steady_state(...; batch_size=...)` as
`result.metadata.batch_means`.  `batch_size` consecutive post-settling time
samples are averaged before the variance is formed, and
`effective_independent_samples` is the number of resulting batch means.

The reported standard error assumes those batch means are approximately
independent.  Increase `batch_size` and the total sampling window until the
estimate stabilizes.  The diagnostic does not certify burn-in or finite-window
bias and never changes the returned state or the primary independent-path
standard error.
"""
struct WeakPIBatchMeansDiagnostics{T<:AbstractFloat,A}
    batch_size::Int
    batch_count::Int
    effective_independent_samples::Int
    sample_spread::T
    standard_error::T
    assumptions::A
end

function Base.show(io::IO,result::WeakPIBatchMeansDiagnostics)
    print(io,"WeakPIBatchMeansDiagnostics($(result.batch_count) batches x ",
          "$(result.batch_size) samples, HS standard error=",
          result.standard_error,")")
end

"""
    weak_pi_density(state)

Convert a [`WeakPIPseudoKet`](@ref) into its physical `PIState`.  Only
sector-diagonal outer products are retained, as required by the PI algebra;
no full-Hilbert-space object is constructed.
"""
function weak_pi_density(state::WeakPIPseudoKet{T}) where T
    b=state.basis;offsets=_weak_pi_offsets(b);rho=PIState(b;T=T)
    scales=_weak_pi_density_scales(b,T)
    for (sector,partition) in pairs(b.sectors)
        psi=view(state.data,_weak_sector_range(offsets,sector))
        _weak_density_outer!(coefficient_block(rho,partition),psi,
                             scales[sector];accumulate=false)
    end
    rho
end

"""
    weak_pi_pseudoket(rho; atol, rtol)

Construct a direct-sum weak-PI pseudo-ket from a `PIState` whose
multiplicity-weighted block in every occupied Schur sector has numerical rank
one.  General mixed PI states are rejected.  Sector phases are fixed by the
eigensolver and are physically irrelevant.
"""
function weak_pi_pseudoket(rho::PIState{T};
        atol::Real=_analysis_atol(rho),rtol::Real=_state_rtol(rho)) where T
    validate_state(rho;atol,rtol)
    b=rho.basis;offsets=_weak_pi_offsets(b)
    data=zeros(Complex{T},offsets[end]-1)
    for (sector,partition) in pairs(b.sectors)
        block=_multiplicity_weighted_block(rho,partition)
        scale=norm(block,Inf)
        hermiticity_error=norm(block-block',Inf)
        hermiticity_error<=T(atol)+T(rtol)*scale||throw(ArgumentError(
            "multiplicity-weighted block in sector $partition is not Hermitian"))
        iszero(scale)&&continue
        eig=_hermitian_eigen(Hermitian((block+block')/2);
            operation="weak-PI pseudo-ket conversion")
        values=eig.values
        tolerance=T(atol)+T(rtol)*maximum(abs,values)
        minimum(values)>=-tolerance||throw(ArgumentError(
            "sector $partition has a negative multiplicity-weighted eigenvalue"))
        largest=last(values)
        largest>tolerance||continue
        length(values)>1&&maximum(abs,view(values,1:length(values)-1))>tolerance&&
            throw(ArgumentError(
                "sector $partition is not rank one and cannot be represented by one weak-PI pseudo-ket"))
        view(data,_weak_sector_range(offsets,sector)).=
            sqrt(max(zero(T),largest)).*view(eig.vectors,:,length(values))
    end
    reconstructed=WeakPIPseudoKet(b,data;atol=max(T(atol),T(10)*eps(T)),
                                   rtol=max(T(rtol),T(10)*eps(T)))
    residual=norm(weak_pi_density(reconstructed).data-rho.data)
    residual<=T(atol)+T(rtol)*max(norm(rho.data),one(T))||throw(ArgumentError(
        "state is not representable by one weak-PI pseudo-ket; reconstruction residual=$residual"))
    reconstructed
end

function _weak_physical_blocks(operator::PIOperator{T}) where T
    b=operator.basis
    [_divide_by_schur_multiplicity_scale(
        Matrix(coefficient_block(operator,partition)),T,partition)
     for partition in b.sectors]
end

function _weak_block_expectation(data,offsets,blocks)
    value=zero(promote_type(eltype(data),eltype(first(blocks))))
    for sector in eachindex(blocks)
        psi=view(data,_weak_sector_range(offsets,sector));A=blocks[sector]
        @inbounds for column in eachindex(psi),row in eachindex(psi)
            value+=conj(psi[row])*A[row,column]*psi[column]
        end
    end
    value
end

"""
    weak_pi_expectation(state, observable)

Evaluate an observable on a weak-PI pseudo-ket without first constructing a
density matrix.  `observable` may be a compatible `PIOperator`, a local matrix
(interpreted as its collective sum), or a prepared
[`CollectiveObservablePlan`](@ref).
"""
function weak_pi_expectation(state::WeakPIPseudoKet,A::PIOperator)
    state.basis===A.basis||throw(ArgumentError(
        "pseudo-ket and observable use incompatible PI bases"))
    _weak_block_expectation(state.data,_weak_pi_offsets(state.basis),
                            _weak_physical_blocks(A))
end
function weak_pi_expectation(state::WeakPIPseudoKet,
                             plan::CollectiveObservablePlan)
    state.basis===plan.basis||throw(ArgumentError(
        "pseudo-ket and observable plan use incompatible PI bases"))
    _weak_block_expectation(state.data,_weak_pi_offsets(state.basis),plan.blocks)
end
weak_pi_expectation(state::WeakPIPseudoKet,A::AbstractMatrix)=
    weak_pi_expectation(state,CollectiveObservablePlan(state.basis,A))

"""One Kraus branch used by a weak-PI jump channel."""
struct WeakPIKrausBranch{T,D}
    channel::Int
    source_sector::Int
    target_sector::Int
    child_partition::Union{Nothing,Partition{D}}
    operator::Matrix{T}
end

"""
    WeakPIJumpRecord

Metadata for one sampled weak-PI Kraus branch.  `source_partition` and
`target_partition` are the Schur sectors before and after the jump;
`child_partition` identifies the one-box subduction branch of a local jump
and is `nothing` for a sector-preserving collective/direct jump.
"""
struct WeakPIJumpRecord{D}
    channel::Int
    branch::Int
    source_partition::Partition{D}
    target_partition::Partition{D}
    child_partition::Union{Nothing,Partition{D}}
end

"""One saved weak-PI pseudo-ket trajectory and its refined jump records."""
struct WeakPIQuantumTrajectory{T,S,R}
    times::Vector{T}
    states::Vector{S}
    jump_times::Vector{T}
    jump_channels::Vector{Int}
    jump_records::Vector{R}
end

"""
    WeakPITrajectoryPlan(model; T=nothing)
    WeakPITrajectoryPlan(compiled; T=nothing)

Prepare the opt-in direct-sum Schur-irrep pseudo-ket unraveling.  Fixed
collective, direct-PI, and collective p-body jumps are split by source Schur
sector.  A fixed one-body `LocalJump` is decomposed into sector-changing Kraus
maps using one-box Schur subduction.  The decomposition is valid for qubits
and qudits and is checked against every channel's prepared ``K^dagger K``
block at construction.

Operator-valued schedules and `LocalPBodyJump` are rejected.  Scalar jump
rates may depend on time and parameters but must evaluate to finite,
nonnegative real values in the prepared precision.  This plan never constructs
a ``d^N`` vector or matrix.
"""
struct WeakPITrajectoryPlan{M,P,H,E,J,B,O,R}
    model::M
    density_plan::P
    hamiltonians::H
    hamiltonian_terms::E
    jumps::J
    branches::B
    branch_ranges::Vector{UnitRange{Int}}
    offsets::O
    Rtype::Type{R}
end

isautonomous(plan::WeakPITrajectoryPlan)=isautonomous(plan.density_plan)

function _weak_local_branch_operator(cache::OneBodyGeometry,operator,
        output_sector,child,input_sector,::Type{CT}) where CT
    b=cache.basis
    noutput=length(b.patterns[output_sector])
    ninput=length(b.patterns[input_sector])
    matrix=zeros(CT,noutput,ninput)
    contractions=cache.contractions[(output_sector,child,input_sector)]
    @inbounds for output_index in 1:noutput,input_index in 1:ninput
        matrix[output_index,input_index]=
            _contract(contractions[output_index,input_index],operator)
    end
    R=_real_float_type(CT)
    output_multiplicity=symmetric_group_dimension(b.sectors[output_sector])
    input_multiplicity=symmetric_group_dimension(b.sectors[input_sector])
    coefficient_scale=cache.scales[(output_sector,child,input_sector)]
    strength=_checked_mul_sqrt_exact_ratio(R,coefficient_scale,
        output_multiplicity,input_multiplicity;
        context="weak-PI local-jump Kraus strength")
    isfinite(strength)&&strength>=zero(R)||throw(ArgumentError(
        "weak-PI local-jump Kraus strength is invalid"))
    matrix .*= sqrt(strength)
    matrix
end

function _weak_kraus_tolerance(::Type{R},dimension,scale) where R<:AbstractFloat
    max(R(1e-10),R(128max(1,dimension))*eps(R))*max(scale,one(R))
end

function _validate_weak_kraus_channel!(branches,range,kernel,b,::Type{CT}) where CT
    R=_real_float_type(CT)
    for source_sector in eachindex(b.sectors)
        dimension=length(b.patterns[source_sector])
        closure=zeros(CT,dimension,dimension)
        for branch_index in range
            branch=branches[branch_index]
            branch.source_sector==source_sector||continue
            mul!(closure,adjoint(branch.operator),branch.operator,
                 one(CT),one(CT))
        end
        target=kernel.qblocks[source_sector]
        error=norm(closure-target,Inf);scale=norm(target,Inf)
        error<=_weak_kraus_tolerance(R,dimension,scale)||throw(ArgumentError(
            "weak-PI Kraus decomposition failed completeness in sector $(b.sectors[source_sector]): error=$error"))
    end
    nothing
end

function _weak_pi_trajectory_plan(model::PIModel,density_plan::TrajectoryPlan)
    b=model.basis;CT=density_plan.liouvillian.Ttype
    CT<:Complex||throw(ArgumentError(
        "weak-PI trajectories require a complex prepared scalar type"))
    R=_real_float_type(CT)
    hamiltonian_terms=filter(
        term->term_process(term) isa Val{:hamiltonian},model.terms)
    all(term->term isa _HamiltonianPITerm,hamiltonian_terms)||
        throw(ArgumentError(
            "weak-PI trajectories require built-in Hamiltonian terms"))
    length(hamiltonian_terms)==length(density_plan.hamiltonians)||
        throw(ArgumentError("weak-PI Hamiltonian term/kernel alignment failed"))
    terms=filter(term->term_process(term) isa Val{:jump},model.terms)
    length(terms)==length(density_plan.jumps)||throw(ArgumentError(
        "weak-PI trajectory term/kernel alignment failed"))
    any(term->term isa LocalPBodyJump,terms)&&throw(ArgumentError(
        "weak-PI trajectories do not yet support LocalPBodyJump; use fixed one-body LocalJump channels"))
    cache=any(term->term isa LocalJump,terms) ? OneBodyGeometry(b,R) : nothing
    D=length(first(b.sectors).parts)
    Branch=WeakPIKrausBranch{CT,D}
    branches=Branch[];ranges=Vector{UnitRange{Int}}(undef,length(terms))
    for channel in eachindex(terms)
        term=terms[channel];kernel=density_plan.jumps[channel]
        first_branch=length(branches)+1
        if kernel isa DissipatorPIKernel
            for sector in eachindex(b.sectors)
                operator=Matrix{CT}(kernel.blocks[sector])
                iszero(norm(operator))&&continue
                push!(branches,Branch(channel,sector,sector,nothing,operator))
            end
        elseif kernel isa Union{LocalJumpPIKernel,
                                FactorizedLocalJumpPIKernel}&&
               term isa LocalJump
            operator=term_operator(term)
            operator isa AbstractMatrix||throw(ArgumentError(
                "weak-PI LocalJump operators must be fixed matrices"))
            for input_sector in eachindex(b.sectors),
                output_sector in eachindex(b.sectors),
                child in cache.connections[(output_sector,input_sector)]
                kraus=_weak_local_branch_operator(cache,operator,
                    output_sector,child,input_sector,CT)
                iszero(norm(kraus))&&continue
                push!(branches,Branch(channel,input_sector,output_sector,
                                      child,kraus))
            end
        else
            throw(ArgumentError(
                "weak-PI trajectories support fixed collective/direct jumps and fixed one-body LocalJump channels"))
        end
        ranges[channel]=first_branch:length(branches)
        _validate_weak_kraus_channel!(branches,ranges[channel],kernel,b,CT)
    end
    WeakPITrajectoryPlan(model,density_plan,density_plan.hamiltonians,
        hamiltonian_terms,density_plan.jumps,branches,ranges,
        _weak_pi_offsets(b),R)
end

function WeakPITrajectoryPlan(model::PIModel;T=nothing)
    _weak_pi_trajectory_plan(model,TrajectoryPlan(model;T))
end
function WeakPITrajectoryPlan(compiled::CompiledPIModel;T=nothing)
    _weak_pi_trajectory_plan(compiled.model,TrajectoryPlan(compiled;T))
end

function _weak_plan_for_state(model::PIModel,state::WeakPIPseudoKet)
    isempty(model.terms) ?
        WeakPITrajectoryPlan(model;T=_real_float_type(eltype(state.data))) :
        WeakPITrajectoryPlan(model)
end
function _weak_plan_for_state(compiled::CompiledPIModel,state::WeakPIPseudoKet)
    isempty(compiled.model.terms) ?
        WeakPITrajectoryPlan(compiled;T=_real_float_type(eltype(state.data))) :
        WeakPITrajectoryPlan(compiled)
end
_weak_plan_for_state(plan::WeakPITrajectoryPlan,state::WeakPIPseudoKet)=plan

"""
    WeakPITrajectoryWorkspace(plan, state; mode=:full)

Caller-owned integration and Kraus-selection scratch for one weak-PI
trajectory.  Reuse sequentially; concurrent paths require distinct
workspaces. The workspace caches the rate-weighted jump-loss blocks at the
current integration node; driven-rate caches are invalidated before a new
trajectory. The default `mode=:full` supports fixed-step and adaptive
event-driven paths. Fixed-step RK4 uses three full-vector registers. Use
`mode=:fixed` to omit `k3`, `k4`, and the six Dormand--Prince and event-root
vectors when the workspace will only be used with `algorithm=:fixed`.
"""
struct WeakPITrajectoryWorkspace{V,R,P,E}
    tmp::V
    k1::V
    k2::V
    k3::V
    k4::V
    k5::V
    k6::V
    k7::V
    trial::V
    embedded::V
    start::V
    current::V
    operator_scratch::V
    jump_output::V
    channel_intensities::Vector{R}
    branch_intensities::Vector{R}
    jump_scales::Vector{R}
    hazard_stages::Vector{R}
    dense_hazard::Vector{R}
    plan::P
    effective_qblocks::E
    effective_cache::_EffectiveJumpNodeCache{R}
    mode::Symbol
end

function WeakPITrajectoryWorkspace(plan::WeakPITrajectoryPlan,
                                   state::WeakPIPseudoKet;
                                   mode::Symbol=:full)
    mode in (:full,:fixed)||throw(ArgumentError(
        "weak-PI trajectory workspace mode must be :full or :fixed"))
    state.basis===plan.model.basis||throw(ArgumentError(
        "pseudo-ket and weak-PI trajectory plan use incompatible bases"))
    eltype(state.data)===plan.density_plan.liouvillian.Ttype||throw(ArgumentError(
        "pseudo-ket scalar type $(eltype(state.data)) does not match prepared weak-PI scalar type $(plan.density_plan.liouvillian.Ttype)"))
    vector=similar(state.data);R=_real_float_type(eltype(vector))
    adaptive_vector()=mode===:full ? similar(vector) : similar(vector,0)
    full_rk4_vector()=mode===:full ? similar(vector) : similar(vector,0)
    effective_qblocks=isempty(plan.jumps) ? Matrix{eltype(vector)}[] :
        [zeros(eltype(vector),
            length(plan.model.basis.patterns[sector]),
            length(plan.model.basis.patterns[sector]))
         for sector in eachindex(plan.model.basis.sectors)]
    WeakPITrajectoryWorkspace(similar(vector),similar(vector),similar(vector),
        full_rk4_vector(),full_rk4_vector(),adaptive_vector(),adaptive_vector(),
        adaptive_vector(),adaptive_vector(),adaptive_vector(),adaptive_vector(),
        vector,similar(vector),similar(vector),
        zeros(R,length(plan.jumps)),zeros(R,length(plan.branches)),
        zeros(R,length(plan.jumps)),zeros(R,7),zeros(R,4),plan,
        effective_qblocks,_EffectiveJumpNodeCache(zero(R),false),mode)
end
WeakPITrajectoryWorkspace(model::PIModel,state::WeakPIPseudoKet;kwargs...)=
    WeakPITrajectoryWorkspace(_weak_plan_for_state(model,state),state;kwargs...)
WeakPITrajectoryWorkspace(compiled::CompiledPIModel,state::WeakPIPseudoKet;
                          kwargs...)=
    WeakPITrajectoryWorkspace(_weak_plan_for_state(compiled,state),state;kwargs...)

"""
    WeakPITrajectoryBatchWorkspace(plan, state;
                                   workers=Threads.nthreads(), mode=:full)

Reusable task-owned workspace/RNG pool for
[`weak_pi_quantum_trajectories`](@ref) and
[`weak_pi_trajectory_steady_state`](@ref).  The immutable Kraus plan is
shared; every worker owns all mutable numerical scratch.
"""
struct WeakPITrajectoryBatchWorkspace{P,W,G,S}
    plan::P
    workers::W
    rngs::G
    seeds::S
end

function WeakPITrajectoryBatchWorkspace(plan::WeakPITrajectoryPlan,
        state::WeakPIPseudoKet;workers::Integer=Threads.nthreads(),
        mode::Symbol=:full)
    workers>0||throw(ArgumentError("worker count must be positive"))
    workspaces=[WeakPITrajectoryWorkspace(plan,state;mode)
                for _ in 1:Int(workers)]
    rngs=[MersenneTwister(0) for _ in 1:Int(workers)]
    WeakPITrajectoryBatchWorkspace(plan,workspaces,rngs,UInt64[])
end
WeakPITrajectoryBatchWorkspace(model::PIModel,state::WeakPIPseudoKet;kwargs...)=
    WeakPITrajectoryBatchWorkspace(_weak_plan_for_state(model,state),state;kwargs...)
WeakPITrajectoryBatchWorkspace(compiled::CompiledPIModel,
        state::WeakPIPseudoKet;kwargs...)=
    WeakPITrajectoryBatchWorkspace(_weak_plan_for_state(compiled,state),state;kwargs...)

function _weak_source_matches(plan::WeakPITrajectoryPlan,source::PIModel)
    plan.model===source
end
function _weak_source_matches(plan::WeakPITrajectoryPlan,
                              source::CompiledPIModel)
    plan.model===source.model&&
        (plan.density_plan.liouvillian===source.plan||isempty(source.model.terms))
end
_weak_source_matches(plan::WeakPITrajectoryPlan,
                     source::WeakPITrajectoryPlan)=plan===source

function _check_weak_workspace(work::WeakPITrajectoryWorkspace,source,state)
    _weak_source_matches(work.plan,source)||throw(ArgumentError(
        "weak-PI trajectory workspace was prepared for a different source"))
    state.basis===work.plan.model.basis||throw(ArgumentError(
        "pseudo-ket and weak-PI trajectory workspace use incompatible bases"))
    eltype(work.current)===eltype(state.data)||throw(ArgumentError(
        "weak-PI trajectory workspace has an incompatible scalar type"))
    work
end

function _require_weak_workspace_mode(work::WeakPITrajectoryWorkspace,
                                      algorithm::Symbol)
    algorithm===:fixed||work.mode===:full||throw(ArgumentError(
        "algorithm=:event requires WeakPITrajectoryWorkspace(mode=:full); " *
        "the supplied fixed-only workspace omits adaptive stages"))
    work
end

function _check_weak_batch_workspace(batch::WeakPITrajectoryBatchWorkspace,
                                     source,state)
    _weak_source_matches(batch.plan,source)||throw(ArgumentError(
        "weak-PI batch workspace was prepared for a different source"))
    isempty(batch.workers)&&throw(ArgumentError(
        "weak-PI batch workspace has no workers"))
    length(batch.workers)==length(batch.rngs)||throw(ArgumentError(
        "weak-PI batch workspace has inconsistent worker storage"))
    batch.seeds isa Vector{UInt64}||throw(ArgumentError(
        "weak-PI batch workspace has incompatible seed storage"))
    all(rng->rng isa AbstractRNG,batch.rngs)||throw(ArgumentError(
        "weak-PI batch workspace has incompatible RNG storage"))
    for worker in batch.workers
        worker.plan===batch.plan||throw(ArgumentError(
            "weak-PI batch workers do not share its plan"))
        _check_weak_workspace(worker,batch.plan,state)
    end
    batch
end

function _validate_weak_initial_state(plan,state::WeakPIPseudoKet)
    state.basis===plan.model.basis||throw(ArgumentError(
        "pseudo-ket and weak-PI trajectory plan use incompatible bases"))
    all(isfinite,state.data)||throw(ArgumentError(
        "initial weak-PI pseudo-ket amplitudes must be finite"))
    R=_real_float_type(eltype(state.data))
    tolerance=max(R(1e-10),R(100)*eps(R))
    value=norm(state.data)
    isfinite(value)&&abs(value-one(R))<=tolerance||throw(ArgumentError(
        "initial weak-PI pseudo-ket must have unit norm; norm=$value"))
    nothing
end

function _weak_hamiltonian_scale(term,t,p,::Type{R}) where R<:AbstractFloat
    rate=_weak_real_input(
        R,value_at(term_rate(term),t,p),"Hamiltonian rate")
    hbar=_weak_real_input(R,term_hbar(term),"Hamiltonian hbar")
    iszero(hbar)&&throw(ArgumentError("Hamiltonian hbar must be nonzero"))
    scale=rate/hbar
    isfinite(scale)||throw(ArgumentError("Hamiltonian rate/hbar must be finite"))
    scale
end

# Avoid setup-only BigInt round trips for the overwhelmingly common small
# machine-integer rates while retaining the exact check for arbitrary Integer
# implementations and the no-narrowing checks for floating inputs.
function _weak_real_input(::Type{R},value,label) where R<:AbstractFloat
    if value isa Base.BitInteger&&!(value isa BigInt)
        converted=R(value)
        isfinite(converted)&&converted==value||throw(ArgumentError(
            "$label=$value is not exactly representable in weak-PI trajectory precision $R"))
        return converted
    end
    _trajectory_real_input(R,value,label)
end

function _weak_jump_scale(kernel,t,p,::Type{R}) where R<:AbstractFloat
    scale=_weak_real_input(R,value_at(kernel.scale,t,p),"jump rate")
    scale>=zero(R)||throw(ArgumentError(
        "weak-PI quantum trajectories require nonnegative jump rates"))
    scale
end

function _weak_apply_block!(destination,operator,source,offsets,sector,alpha)
    range=_weak_sector_range(offsets,sector)
    mul!(view(destination,range),operator,view(source,range),alpha,one(alpha))
    destination
end

@inline _weak_apply_hamiltonians!(y,x,w,t,p,::Tuple{},::Tuple{})=nothing
@inline function _weak_apply_hamiltonians!(y,x,w,t,p,
        kernels::Tuple{K,Vararg{Any}},
        terms::Tuple{E,Vararg{Any}}) where {K,E}
    kernel=first(kernels);term=first(terms)
    R=eltype(w.channel_intensities)
    scale=_weak_hamiltonian_scale(term,t,p,R)
    for sector in eachindex(w.plan.model.basis.sectors)
        _weak_apply_block!(y,kernel.blocks[sector],x,w.plan.offsets,sector,
                           convert(eltype(y),-1im*scale))
    end
    _weak_apply_hamiltonians!(y,x,w,t,p,Base.tail(kernels),Base.tail(terms))
end

@inline function _weak_channel_intensities_recursive!(w,x,t,p,destination,
        ::Val{A},::Tuple{},channel::Int) where A
    zero(eltype(w.channel_intensities))
end
@inline function _weak_channel_intensities_recursive!(w,x,t,p,destination,
        ::Val{A},jumps::Tuple{K,Vararg{Any}},channel::Int) where {A,K}
    R=eltype(w.channel_intensities);b=w.plan.model.basis
    kernel=first(jumps);scale=_weak_jump_scale(kernel,t,p,R)
    w.jump_scales[channel]=scale
    value=zero(R)
    for sector in eachindex(b.sectors)
        range=_weak_sector_range(w.plan.offsets,sector)
        psi=view(x,range);scratch=view(w.operator_scratch,range)
        Q=kernel.qblocks[sector]
        mul!(scratch,Q,psi)
        contribution=real(dot(psi,scratch))
        tolerance=_intensity_tolerance(R)*
            max(norm(psi)*norm(scratch),one(R))
        contribution>=-tolerance||throw(ArgumentError(
            "weak-PI jump intensity is negative in sector $(b.sectors[sector])"))
        value+=max(zero(R),contribution)
        if A&&!iszero(scale)
            alpha=convert(eltype(destination),-scale/2)
            @inbounds for index in range
                destination[index]+=alpha*w.operator_scratch[index]
            end
        end
    end
    intensity=scale*value
    isfinite(intensity)||throw(ArgumentError(
        "weak-PI jump intensity is nonfinite"))
    w.channel_intensities[channel]=intensity
    intensity+_weak_channel_intensities_recursive!(w,x,t,p,destination,
        Val(A),Base.tail(jumps),channel+1)
end

function _weak_channel_intensities!(w,x,t,p;apply_generator::Bool=false,
                                    destination=nothing)
    apply_generator&&destination===nothing&&throw(ArgumentError(
        "a destination is required when applying the no-jump generator"))
    total=if apply_generator
        _weak_apply_effective_jump_drift_and_intensity!(
            destination,x,w,t,p)
    else
        # Prepare every driven scale once at this physical node, then retain
        # those same values for the channel-resolved selection.
        _weak_prepare_effective_jump_blocks!(w,t,p)
        _weak_channel_intensities_from_scales_recursive!(
            w,x,w.plan.jumps,1)
    end
    isfinite(total)||throw(ArgumentError(
        "total weak-PI jump intensity is nonfinite"))
    total
end

@inline _weak_channel_intensities_from_scales_recursive!(
    w,x,::Tuple{},channel::Int)=zero(eltype(w.channel_intensities))
@inline function _weak_channel_intensities_from_scales_recursive!(
        w,x,jumps::Tuple{K,Vararg{Any}},channel::Int) where K
    R=eltype(w.channel_intensities);b=w.plan.model.basis
    kernel=first(jumps);scale=w.jump_scales[channel]
    value=zero(R)
    for sector in eachindex(b.sectors)
        range=_weak_sector_range(w.plan.offsets,sector)
        psi=view(x,range);scratch=view(w.operator_scratch,range)
        mul!(scratch,kernel.qblocks[sector],psi)
        contribution=real(dot(psi,scratch))
        tolerance=_intensity_tolerance(R)*
            max(norm(psi)*norm(scratch),one(R))
        contribution>=-tolerance||throw(ArgumentError(
            "weak-PI jump intensity is negative in sector $(b.sectors[sector])"))
        value+=max(zero(R),contribution)
    end
    intensity=scale*value
    isfinite(intensity)||throw(ArgumentError(
        "weak-PI jump intensity is nonfinite"))
    w.channel_intensities[channel]=intensity
    intensity+_weak_channel_intensities_from_scales_recursive!(
        w,x,Base.tail(jumps),channel+1)
end

@inline _weak_accumulate_effective_jump_blocks!(w,t,p,::Tuple{},index)=
    nothing
@inline function _weak_accumulate_effective_jump_blocks!(w,t,p,
        jumps::Tuple{K,Vararg{Any}},index) where K
    kernel=first(jumps);R=eltype(w.channel_intensities)
    scale=_weak_jump_scale(kernel,t,p,R)
    w.jump_scales[index]=scale
    if !iszero(scale)
        @inbounds for sector in eachindex(w.effective_qblocks)
            effective=w.effective_qblocks[sector]
            qblock=kernel.qblocks[sector]
            for block_index in eachindex(effective,qblock)
                effective[block_index]+=scale*qblock[block_index]
            end
        end
    end
    _weak_accumulate_effective_jump_blocks!(
        w,t,p,Base.tail(jumps),index+1)
end

function _weak_prepare_effective_jump_blocks!(w,t,p)
    if w.effective_cache.valid&&
       (_trajectory_jump_rates_autonomous(w.plan.jumps)||
        isequal(w.effective_cache.time,t))
        return w.effective_qblocks
    end
    for block in w.effective_qblocks
        fill!(block,zero(eltype(block)))
    end
    _weak_accumulate_effective_jump_blocks!(w,t,p,w.plan.jumps,1)
    w.effective_cache.time=t
    # A failed schedule evaluation leaves the cache invalid, so partially
    # accumulated blocks are never reused.
    w.effective_cache.valid=true
    w.effective_qblocks
end

@inline function _weak_reset_effective_jump_cache!(
        w::WeakPITrajectoryWorkspace)
    _trajectory_jump_rates_autonomous(w.plan.jumps)||
        (w.effective_cache.valid=false)
    w
end

function _weak_effective_jump_intensity_from_blocks(w,x)
    R=eltype(w.channel_intensities);total=zero(R)
    b=w.plan.model.basis
    for sector in eachindex(b.sectors)
        range=_weak_sector_range(w.plan.offsets,sector)
        psi=view(x,range);scratch=view(w.operator_scratch,range)
        mul!(scratch,w.effective_qblocks[sector],psi)
        contribution=real(dot(psi,scratch))
        tolerance=_intensity_tolerance(R)*
            max(norm(psi)*norm(scratch),one(R))
        contribution>=-tolerance||throw(ArgumentError(
            "combined weak-PI jump intensity is negative in sector $(b.sectors[sector])"))
        total+=max(zero(R),contribution)
    end
    isfinite(total)||throw(ArgumentError(
        "total weak-PI jump intensity is nonfinite"))
    total
end

function _weak_effective_jump_intensity!(w,x,t,p)
    isempty(w.plan.jumps)&&return zero(eltype(w.channel_intensities))
    _weak_prepare_effective_jump_blocks!(w,t,p)
    _weak_effective_jump_intensity_from_blocks(w,x)
end

function _weak_apply_effective_jump_drift_and_intensity!(destination,x,w,t,p)
    isempty(w.plan.jumps)&&return zero(eltype(w.channel_intensities))
    _weak_prepare_effective_jump_blocks!(w,t,p)
    R=eltype(w.channel_intensities);total=zero(R)
    b=w.plan.model.basis
    for sector in eachindex(b.sectors)
        range=_weak_sector_range(w.plan.offsets,sector)
        psi=view(x,range);scratch=view(w.operator_scratch,range)
        mul!(scratch,w.effective_qblocks[sector],psi)
        contribution=real(dot(psi,scratch))
        tolerance=_intensity_tolerance(R)*
            max(norm(psi)*norm(scratch),one(R))
        contribution>=-tolerance||throw(ArgumentError(
            "combined weak-PI jump intensity is negative in sector $(b.sectors[sector])"))
        total+=max(zero(R),contribution)
        alpha=convert(eltype(destination),-one(R)/2)
        @inbounds for index in range
            destination[index]+=alpha*w.operator_scratch[index]
        end
    end
    isfinite(total)||throw(ArgumentError(
        "total weak-PI jump intensity is nonfinite"))
    total
end

function _weak_conditional_action_and_intensity!(y,x,w,t,p)
    fill!(y,zero(eltype(y)))
    _weak_apply_hamiltonians!(y,x,w,t,p,w.plan.hamiltonians,
                              w.plan.hamiltonian_terms)
    total=_weak_apply_effective_jump_drift_and_intensity!(y,x,w,t,p)
    alpha=convert(eltype(y),total/2)
    @inbounds for index in eachindex(y)
        y[index]+=alpha*x[index]
    end
    total
end

function _weak_conditional_action!(y,x,w,t,p)
    _weak_conditional_action_and_intensity!(y,x,w,t,p)
    y
end

function _weak_conditional_rk4!(x,w,t,h,p)
    _weak_conditional_action!(w.k1,x,w,t,p)
    copyto!(w.k2,w.k1)
    @. w.tmp=x+(h/2)*w.k1
    _weak_conditional_action!(w.k1,w.tmp,w,t+h/2,p)
    @. w.k2=w.k2+2w.k1
    @. w.tmp=x+(h/2)*w.k1
    _weak_conditional_action!(w.k1,w.tmp,w,t+h/2,p)
    @. w.k2=w.k2+2w.k1
    @. w.tmp=x+h*w.k1
    _weak_conditional_action!(w.k1,w.tmp,w,t+h,p)
    @. x=x+(h/6)*(w.k2+w.k1)
    value=norm(x);R=eltype(w.channel_intensities)
    isfinite(value)&&value>eps(R)||throw(ArgumentError(
        "weak-PI conditional pseudo-ket acquired zero or nonfinite norm"))
    x./=value
    x
end

# One Dormand--Prince 5(4) trial for the normalized weak-PI pseudo-ket and
# the integrated jump hazard.  The state and hazard share every stage, so an
# accepted event-driven step controls both errors without a Bernoulli time
# discretization.
function _weak_conditional_dopri_trial!(w,x,t,h,p,abstol,reltol)
    R=typeof(h)
    l1=_weak_conditional_action_and_intensity!(w.k1,x,w,t,p)
    @. w.tmp=x+h*(1//5)*w.k1
    l2=_weak_conditional_action_and_intensity!(
        w.k2,w.tmp,w,t+h*(R(1)/R(5)),p)
    @. w.tmp=x+h*((3//40)*w.k1+(9//40)*w.k2)
    l3=_weak_conditional_action_and_intensity!(
        w.k3,w.tmp,w,t+h*(R(3)/R(10)),p)
    @. w.tmp=x+h*((44//45)*w.k1-(56//15)*w.k2+(32//9)*w.k3)
    l4=_weak_conditional_action_and_intensity!(
        w.k4,w.tmp,w,t+h*(R(4)/R(5)),p)
    @. w.tmp=x+h*((19372//6561)*w.k1-(25360//2187)*w.k2+
                  (64448//6561)*w.k3-(212//729)*w.k4)
    l5=_weak_conditional_action_and_intensity!(
        w.k5,w.tmp,w,t+h*(R(8)/R(9)),p)
    @. w.tmp=x+h*((9017//3168)*w.k1-(355//33)*w.k2+
                  (46732//5247)*w.k3+(49//176)*w.k4-
                  (5103//18656)*w.k5)
    l6=_weak_conditional_action_and_intensity!(w.k6,w.tmp,w,t+h,p)
    @. w.trial=x+h*((35//384)*w.k1+(500//1113)*w.k3+
                    (125//192)*w.k4-(2187//6784)*w.k5+
                    (11//84)*w.k6)
    l7=_weak_conditional_action_and_intensity!(w.k7,w.trial,w,t+h,p)
    w.hazard_stages[1]=l1;w.hazard_stages[2]=l2
    w.hazard_stages[3]=l3;w.hazard_stages[4]=l4
    w.hazard_stages[5]=l5;w.hazard_stages[6]=l6
    w.hazard_stages[7]=l7
    @. w.embedded=x+h*((5179//57600)*w.k1+(7571//16695)*w.k3+
                       (393//640)*w.k4-(92097//339200)*w.k5+
                       (187//2100)*w.k6+(1//40)*w.k7)

    hazard5=h*((35//384)*l1+(500//1113)*l3+(125//192)*l4-
               (2187//6784)*l5+(11//84)*l6)
    hazard4=h*((5179//57600)*l1+(7571//16695)*l3+(393//640)*l4-
               (92097//339200)*l5+(187//2100)*l6+(1//40)*l7)
    state_scale=abstol+reltol*max(norm(x),norm(w.trial),one(R))
    @. w.tmp=w.trial-w.embedded
    state_error=norm(w.tmp)/(sqrt(R(length(x)))*state_scale)
    hazard_scale=abstol+reltol*max(abs(hazard5),one(hazard5))
    error=max(state_error,abs(hazard5-hazard4)/hazard_scale)
    hazard5,error
end

function _weak_branch_intensities!(w,x)
    R=eltype(w.branch_intensities)
    fill!(w.branch_intensities,zero(R))
    for (branch_index,branch) in pairs(w.plan.branches)
        source=_weak_sector_range(w.plan.offsets,branch.source_sector)
        target=_weak_sector_range(w.plan.offsets,branch.target_sector)
        output=view(w.jump_output,target)
        mul!(output,branch.operator,view(x,source))
        value=w.jump_scales[branch.channel]*real(dot(output,output))
        isfinite(value)&&value>=zero(R)||throw(ArgumentError(
            "weak-PI Kraus-branch intensity is invalid"))
        w.branch_intensities[branch_index]=value
    end
    branch_total=sum(w.branch_intensities)
    channel_total=sum(w.channel_intensities)
    tolerance=_weak_kraus_tolerance(R,length(x),channel_total)
    abs(branch_total-channel_total)<=tolerance||throw(ArgumentError(
        "weak-PI branch intensities do not sum to the channel intensity: branch=$branch_total, channel=$channel_total"))
    branch_total
end

function _weak_apply_branch!(x,w,branch_index)
    branch=w.plan.branches[branch_index]
    source=_weak_sector_range(w.plan.offsets,branch.source_sector)
    target=_weak_sector_range(w.plan.offsets,branch.target_sector)
    fill!(w.jump_output,zero(eltype(w.jump_output)))
    mul!(view(w.jump_output,target),branch.operator,view(x,source))
    value=norm(w.jump_output);R=eltype(w.branch_intensities)
    isfinite(value)&&value>eps(R)||throw(ArgumentError(
        "selected weak-PI Kraus branch has zero or nonfinite probability"))
    @. x=w.jump_output/value
    branch
end

function _weak_jump_record(plan,branch_index)
    branch=plan.branches[branch_index];b=plan.model.basis
    WeakPIJumpRecord(branch.channel,branch_index,
        b.sectors[branch.source_sector],b.sectors[branch.target_sector],
        branch.child_partition)
end

function _weak_observable_blocks(ops)
    Tuple(_weak_physical_blocks(last(pair)) for pair in ops)
end

function _record_weak_observables!(values::AbstractMatrix,ops,blocks,data,
                                   offsets,output_index::Integer)
    size(values,1)==length(ops)||throw(DimensionMismatch(
        "weak-PI observable buffer has the wrong observable count"))
    @inbounds for observable_index in eachindex(ops)
        value=_weak_block_expectation(data,offsets,blocks[observable_index])
        values[observable_index,output_index]=real(value)
    end
    values
end

function _weak_event_driven_trajectory(plan,state0,ts,w,rng,options;
        density_sampler=nothing,observable_ops=nothing,
        observable_blocks=nothing,observable_values=nothing,
        save_states::Bool=true,record_jumps::Bool=true)
    save_states&&!record_jumps&&throw(ArgumentError(
        "saved weak-PI trajectories require recorded jump histories"))
    x=w.current;copyto!(x,state0.data)
    states=save_states ? Vector{typeof(state0)}(undef,length(ts)) : nothing
    save_states&&(states[1]=copy(state0))
    density_sampler===nothing||_record_weak_density!(density_sampler,x,1)
    observable_values===nothing||_record_weak_observables!(
        observable_values,observable_ops,observable_blocks,x,plan.offsets,1)
    jump_times=record_jumps ? eltype(ts)[] : nothing
    jump_channels=record_jumps ? Int[] : nothing
    D=length(first(plan.model.basis.sectors).parts)
    jump_records=record_jumps ? WeakPIJumpRecord{D}[] : nothing
    t=ts[1];R=eltype(w.channel_intensities)
    threshold=randexp(rng,R);hazard=zero(R);h=min(options.dt,options.dtmax)
    for output_index in 2:length(ts)
        target=ts[output_index]
        while t<target
            remaining_to_target=target-t
            h,lands_on_target=_trajectory_step_to_target(
                t,target,min(h,options.dtmax))
            minimum_step=min(options.dtmin,remaining_to_target)
            h>=minimum_step||throw(ErrorException(
                "adaptive weak-PI trajectory step fell below dtmin=$(options.dtmin) at t=$t"))
            copyto!(w.start,x)
            increment,error=_weak_conditional_dopri_trial!(
                w,x,t,h,options.parameters,options.abstol,options.reltol)
            if !(isfinite(error)&&isfinite(increment))
                throw(ErrorException(
                    "non-finite adaptive weak-PI trajectory trial at t=$t"))
            end
            if error>one(error)
                h>minimum_step||throw(ErrorException(
                    "adaptive weak-PI trajectory cannot satisfy its error tolerance above dtmin=$(options.dtmin) at t=$t"))
                h=max(minimum_step,h*_adaptive_factor(error))
                continue
            end
            increment>=-R(10)*options.abstol||throw(ErrorException(
                "weak-PI jump hazard decreased by $increment"))
            increment=max(zero(R),increment)
            if hazard+increment<threshold
                copyto!(x,w.trial)
                value=norm(x)
                isfinite(value)&&value>eps(R)||throw(ArgumentError(
                    "conditional weak-PI pseudo-ket acquired zero or nonfinite norm"))
                x./=value
                t=lands_on_target ? target : t+h
                hazard+=increment
                h=min(options.dtmax,
                      max(options.dtmin,h*_adaptive_factor(error)))
                continue
            end

            remaining=threshold-hazard
            _prepare_dopri_dense_output!(w)
            # `eps(t)` already scales with `abs(t)`. Multiplying it by the
            # absolute time a second time would make event localization
            # catastrophically coarse when a simulation uses a large time
            # origin. Use the local ulp at a unit-bounded time scale instead.
            time_tolerance=max(options.event_time_tolerance,
                R(8)*eps(max(abs(t),one(t))))
            event_step=_dopri_dense_root(
                w,h,remaining,increment,time_tolerance)
            if event_step===nothing
                h>minimum_step||throw(ErrorException(
                    "Dormand--Prince dense weak-PI hazard lost its event bracket above dtmin=$(options.dtmin) at t=$t"))
                h=max(minimum_step,h/R(2))
                continue
            end
            theta=event_step/h
            _dopri_dense_state!(x,w.start,w,h,theta)
            value=norm(x)
            isfinite(value)&&value>eps(R)||throw(ArgumentError(
                "conditional weak-PI pseudo-ket acquired zero or nonfinite norm"))
            x./=value;t+=event_step

            total=_weak_channel_intensities!(w,x,t,options.parameters)
            total>zero(total)||throw(ErrorException(
                "weak-PI hazard root has zero channel intensity at t=$t"))
            branch_total=_weak_branch_intensities!(w,x)
            branch_total>zero(branch_total)||throw(ErrorException(
                "weak-PI hazard root has zero Kraus-branch intensity at t=$t"))
            branch_index=_select_jump_channel(w.branch_intensities,
                rand(rng,R)*branch_total)
            branch=_weak_apply_branch!(x,w,branch_index)
            if record_jumps
                push!(jump_times,t);push!(jump_channels,branch.channel)
                push!(jump_records,_weak_jump_record(plan,branch_index))
            end
            threshold=randexp(rng,R);hazard=zero(R)
            h=min(options.dtmax,max(options.dtmin,h-event_step))
        end
        save_states&&
            (states[output_index]=WeakPIPseudoKet(state0.basis,x))
        density_sampler===nothing||
            _record_weak_density!(density_sampler,x,output_index)
        observable_values===nothing||_record_weak_observables!(
            observable_values,observable_ops,observable_blocks,x,
            plan.offsets,output_index)
    end
    save_states ?
        WeakPIQuantumTrajectory(ts,states,jump_times,jump_channels,jump_records) :
        record_jumps ?
            (;jump_times,jump_channels,jump_records) : nothing
end

function _weak_pi_trajectory_prepared(plan,state0,ts,w,rng,options;
        density_sampler=nothing,observable_ops=nothing,
        observable_blocks=nothing,observable_values=nothing,
        save_states::Bool=true,record_jumps::Bool=true)
    save_states&&!record_jumps&&throw(ArgumentError(
        "saved weak-PI trajectories require recorded jump histories"))
    _require_weak_workspace_mode(w,options.algorithm)
    _weak_reset_effective_jump_cache!(w)
    options.algorithm!==:fixed&&return _weak_event_driven_trajectory(
        plan,state0,ts,w,rng,options;density_sampler,observable_ops,
        observable_blocks,observable_values,save_states,record_jumps)
    x=w.current;copyto!(x,state0.data)
    states=save_states ? Vector{typeof(state0)}(undef,length(ts)) : nothing
    save_states&&(states[1]=copy(state0))
    density_sampler===nothing||_record_weak_density!(density_sampler,x,1)
    observable_values===nothing||_record_weak_observables!(
        observable_values,observable_ops,observable_blocks,x,plan.offsets,1)
    jump_times=record_jumps ? eltype(ts)[] : nothing
    jump_channels=record_jumps ? Int[] : nothing
    D=length(first(plan.model.basis.sectors).parts)
    jump_records=record_jumps ? WeakPIJumpRecord{D}[] : nothing
    t=ts[1]
    for output_index in 2:length(ts)
        target_time=ts[output_index]
        while t<target_time
            h,lands_on_target=_trajectory_step_to_target(
                t,target_time,options.dt)
            total=_weak_effective_jump_intensity!(
                w,x,t,options.parameters)
            if total*h>options.max_jump_probability
                h=options.max_jump_probability/total
                lands_on_target=false
            end
            _weak_conditional_rk4!(x,w,t,h,options.parameters)
            t=lands_on_target ? target_time : t+h
            total=_weak_effective_jump_intensity!(
                w,x,t,options.parameters)
            probability=-expm1(-total*h)
            if total>zero(total)&&rand(rng,typeof(probability))<probability
                channel_total=_weak_channel_intensities!(
                    w,x,t,options.parameters)
                tolerance=_intensity_tolerance(typeof(total))*
                    max(one(total),total,channel_total)
                abs(channel_total-total)<=tolerance||throw(ArgumentError(
                    "combined and channel-resolved weak-PI jump intensities disagree at a selected event"))
                branch_total=_weak_branch_intensities!(w,x)
                branch_total>zero(branch_total)||throw(ArgumentError(
                    "selected weak-PI jump has zero branch intensity"))
                branch_index=_select_jump_channel(w.branch_intensities,
                    rand(rng,typeof(branch_total))*branch_total)
                branch=_weak_apply_branch!(x,w,branch_index)
                if record_jumps
                    push!(jump_times,t);push!(jump_channels,branch.channel)
                    push!(jump_records,_weak_jump_record(plan,branch_index))
                end
            end
        end
        save_states&&
            (states[output_index]=WeakPIPseudoKet(state0.basis,x))
        density_sampler===nothing||
            _record_weak_density!(density_sampler,x,output_index)
        observable_values===nothing||_record_weak_observables!(
            observable_values,observable_ops,observable_blocks,x,
            plan.offsets,output_index)
    end
    save_states ?
        WeakPIQuantumTrajectory(ts,states,jump_times,jump_channels,jump_records) :
        record_jumps ?
            (;jump_times,jump_channels,jump_records) : nothing
end

"""
    weak_pi_quantum_trajectory(source, state0, times; dt, rng,
        parameters=nothing, max_jump_probability=0.05,
        algorithm=:fixed, abstol=1e-9, reltol=1e-7,
        dtmin=eps(R), dtmax=dt, event_time_tolerance=1e-10,
        workspace=nothing)

Simulate one fixed-step weak-PI pseudo-ket trajectory.  `source` is a
`PIModel`, `CompiledPIModel`, or reusable [`WeakPITrajectoryPlan`](@ref).
Every local event samples a mathematically complete sector-changing Kraus
branch, while `jump_channels` retains the model-level channel index.

The fixed step is shortened to control the maximum jump probability.
All rate-weighted channel loss blocks are combined before one matrix-vector
action per Schur sector; individual channel and Kraus-branch intensities are
evaluated only when an event must be sampled.
Set `algorithm=:event` (aliases `:adaptive` and `:event_driven`) to integrate
the normalized pseudo-ket and accumulated jump hazard with the embedded
Dormand--Prince 5(4) method.  Events are continuous hazard roots and sample
the same sector-resolved Kraus branches as the fixed-step path. Bisection uses
the quartic continuous extension of the accepted stages and performs no new
RHS trials. `dt` is then
the initial step; `dtmax`, `dtmin`, `abstol`, `reltol`, and
`event_time_tolerance` control the adaptive solve.

This opt-in backend is distinct from [`quantum_trajectory`](@ref), whose
unresolved local event produces a density-valued conditional PI state.
"""
function weak_pi_quantum_trajectory(
        source::Union{PIModel,CompiledPIModel,WeakPITrajectoryPlan},
        state0::WeakPIPseudoKet{R},times;dt::Real,
        rng::AbstractRNG=Random.default_rng(),parameters=nothing,
        max_jump_probability=nothing,workspace=nothing,
        algorithm::Symbol=:fixed,abstol=nothing,reltol=nothing,
        dtmin=nothing,dtmax=nothing,
        event_time_tolerance=nothing,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET) where R<:AbstractFloat
    ts,options=_prepare_trajectory_arguments(times,R;dt,parameters,
        max_jump_probability,algorithm,abstol,reltol,dtmin,dtmax,
        event_time_tolerance)
    _guard_trajectory_history("weak-PI trajectory state history",
        length(state0.data),eltype(state0.data),ts,1,memory_budget;
        guidance="Request fewer saved times for a long path.")
    if workspace===nothing
        plan=_weak_plan_for_state(source,state0)
        mode=algorithm===:fixed ? :fixed : :full
        work=WeakPITrajectoryWorkspace(plan,state0;mode)
    else
        workspace isa WeakPITrajectoryWorkspace||throw(ArgumentError(
            "workspace must be a WeakPITrajectoryWorkspace"))
        work=_check_weak_workspace(workspace,source,state0);plan=work.plan
    end
    _validate_weak_initial_state(plan,state0)
    _weak_pi_trajectory_prepared(plan,state0,ts,work,rng,options)
end

function _weak_ensemble_setup(source,state0,n,threaded,workspace;
                              mode::Symbol=:full)
    n>0||throw(ArgumentError("trajectory count must be positive"))
    if workspace===nothing
        plan=_weak_plan_for_state(source,state0)
        count=threaded ? min(Int(n),Threads.nthreads()) : 1
        batch=WeakPITrajectoryBatchWorkspace(plan,state0;workers=count,mode)
        return plan,batch,nothing
    elseif workspace isa WeakPITrajectoryBatchWorkspace
        batch=_check_weak_batch_workspace(workspace,source,state0)
        return batch.plan,batch,nothing
    elseif workspace isa WeakPITrajectoryWorkspace
        threaded&&throw(ArgumentError(
            "threaded weak-PI ensembles require a WeakPITrajectoryBatchWorkspace"))
        _check_weak_workspace(workspace,source,state0)
        return workspace.plan,nothing,workspace
    end
    throw(ArgumentError(
        "workspace must be a WeakPITrajectoryWorkspace or WeakPITrajectoryBatchWorkspace"))
end

"""
    weak_pi_quantum_trajectories(source, state0, times, n;
                                 seed=0, threaded=false,
                                 workspace=nothing, algorithm=:fixed,
                                 trajectory_keywords...)

Generate an ensemble of weak-PI pseudo-ket trajectories.  A prepared Kraus
plan is shared read-only, worker scratch is task-owned, and random streams are
derived from the trajectory index so serial and threaded runs are reproducible
for a fixed seed. `memory_budget` bounds the predictable pseudo-ket state/time
history before Kraus-plan construction; jump records and worker scratch are
additional data-dependent or reusable storage.
"""
function weak_pi_quantum_trajectories(
        source::Union{PIModel,CompiledPIModel,WeakPITrajectoryPlan},
        state0::WeakPIPseudoKet{R},times,n::Integer;seed::Integer=0,
        threaded::Bool=false,workspace=nothing,dt::Real,
        parameters=nothing,max_jump_probability=nothing,
        algorithm::Symbol=:fixed,abstol=nothing,reltol=nothing,
        dtmin=nothing,dtmax=nothing,
        event_time_tolerance=nothing,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET) where R<:AbstractFloat
    trajectory_count=_trajectory_integer_count(n,"trajectory count",1)
    ts,options=_prepare_trajectory_arguments(times,R;dt,parameters,
        max_jump_probability,algorithm,abstol,reltol,dtmin,dtmax,
        event_time_tolerance)
    _guard_trajectory_history("weak-PI trajectory ensemble state history",
        length(state0.data),eltype(state0.data),ts,trajectory_count,
        memory_budget;guidance=
        "Use adaptive_weak_pi_quantum_trajectories for state-free observable statistics.")
    mode=algorithm===:fixed ? :fixed : :full
    plan,batch,single=_weak_ensemble_setup(
        source,state0,trajectory_count,threaded,workspace;mode)
    _validate_weak_initial_state(plan,state0)
    if batch===nothing
        master=MersenneTwister(seed)
        seeds=rand(master,UInt64,trajectory_count)
        workers=(single,);rngs=(master,);worker_count=1
    else
        master=batch.rngs[1];Random.seed!(master,seed)
        resize!(batch.seeds,trajectory_count);rand!(master,batch.seeds)
        seeds=batch.seeds
        available=length(batch.workers)
        worker_count=threaded ?
            min(trajectory_count,Threads.nthreads(),available) : 1
        workers=batch.workers;rngs=batch.rngs
    end
    TT=eltype(ts);S=typeof(state0);D=length(first(plan.model.basis.sectors).parts)
    Record=WeakPIJumpRecord{D}
    Result=WeakPIQuantumTrajectory{TT,S,Record}
    output=Vector{Result}(undef,trajectory_count)
    if worker_count==1
        worker=workers[1];rng=rngs[1]
        for index in 1:trajectory_count
            Random.seed!(rng,seeds[index])
            output[index]=_weak_pi_trajectory_prepared(
                plan,state0,copy(ts),worker,rng,options)
        end
    else
        chunk_size=max(1,trajectory_count÷(8worker_count))
        next_index=Threads.Atomic{Int}(1)
        @sync for worker_index in 1:worker_count
            let worker=workers[worker_index],rng=rngs[worker_index],
                worker_id=worker_index
                Threads.@spawn begin
                    while true
                        first_index=Threads.atomic_add!(next_index,chunk_size)
                        first_index>trajectory_count&&break
                        final_index=min(trajectory_count,
                                        first_index+chunk_size-1)
                        for index in first_index:final_index
                            Random.seed!(rng,seeds[index])
                            output[index]=_weak_pi_trajectory_prepared(
                                plan,state0,copy(ts),worker,rng,options)
                        end
                    end
                end
            end
        end
    end
    output
end

function _weak_steady_state_path!(sampler,state_accumulator,
        observable_accumulator,observable_buffer,observable_ops,
        plan,state0,times,worker,rng,options,samples_per_trajectory)
    _reset_weak_density_sampler!(sampler)
    _weak_pi_trajectory_prepared(plan,state0,times,worker,rng,options;
        density_sampler=sampler,save_states=false,record_jumps=false)
    sampler.count==samples_per_trajectory||throw(ErrorException(
        "internal weak-PI sampler retained $(sampler.count) densities instead of $samples_per_trajectory"))
    sampler.batch_count==0||throw(ErrorException(
        "internal weak-PI batch-means sampler ended with an incomplete batch"))
    _accumulate_state!(state_accumulator,sampler.mean)
    if observable_accumulator!==nothing
        _record_observables!(observable_buffer,observable_ops,sampler.mean,1)
        _accumulate_observables!(observable_accumulator,observable_buffer)
    end
    nothing
end

"""
    weak_pi_trajectory_steady_state(source, state0;
        trajectories, settling_time, dt,
        samples_per_trajectory=1, sampling_interval=nothing,
        batch_size=nothing, algorithm=:fixed,
        seed=0, threaded=false, workspace=nothing,
        observables=nothing, confidence=0.95, return_info=false,
        parameters=nothing, max_jump_probability=0.05)

Estimate an autonomous stationary PI density operator with weak-PI
pseudo-ket trajectories. For each independent path, the physical
equation-(7) coefficient block is reconstructed at every selected time as

``C_nu(psi)=psi_nu psi_nu^dagger/sqrt(f^nu)``.

Those density contributions are averaged within the path before independent
path means are combined. Pseudo-ket amplitudes are never averaged directly:
the density reconstruction is quadratic and relative phases between Schur
sectors are unphysical. The returned object is consequently a generally
mixed `PIState`, not a [`WeakPIPseudoKet`](@ref).

No pseudo-ket history or jump record is constructed. Sector outer products
are accumulated directly into preallocated PI coefficient vectors, so
retained state and density storage scales with the worker count rather than
with the number of trajectories or post-settling samples. Reproducible random
streams retain one `UInt64` seed per trajectory. Each density sample still
requires ``sum_nu dim(U_nu)^2`` work because the requested result is a full
PI density operator.

At least two trajectories are required. Multiple post-settling samples are
averaged within each path and never treated as independent observations.
Choose the settling time, sampling interval and window, path count, `dt`, and
`max_jump_probability` through separate convergence studies. Strong
symmetries or multiple stationary states can make the estimate depend on
`state0`; the unraveling also affects finite-sample variance.

Set `batch_size` to an integer of at least two that exactly divides
`samples_per_trajectory` to add an autocorrelation-aware batch-means report at
`result.metadata.batch_means`; this requires `return_info=true`. Consecutive
time samples are averaged in each batch before their Hilbert--Schmidt variance
is evaluated. The reported
effective independent sample count is the number of complete batch means and
assumes batches are long enough to be approximately independent.  Refine the
batch length and total window: this diagnostic does not certify burn-in or
finite-window bias.  The primary `standard_error` remains the conservative
independent-path estimate, so enabling batch means never silently changes its
meaning.

The source may be a `PIModel`, `CompiledPIModel`, or reusable
[`WeakPITrajectoryPlan`](@ref), and must be autonomous.  Set
`algorithm=:event` (or `:adaptive`/`:event_driven`) for continuous-hazard
Dormand--Prince evolution. Current weak-PI operator restrictions still apply:
fixed supported jump operators, no `LocalPBodyJump`, and an initial state
representable by one weak-PI pseudo-ket. `threaded=true` uses task-owned workers and
trajectory-indexed random streams. Pass a
[`WeakPITrajectoryBatchWorkspace`](@ref) to reuse them across calls.

By default, return the estimated `PIState`. With `return_info=true`, return a
[`TrajectorySteadyStateResult`](@ref) containing independent-path
Hilbert--Schmidt uncertainty, optional named Hermitian-observable statistics,
the density Liouvillian residual, relative residual, and trace error. These
are diagnostics rather than a steady-state certificate. The averaged state
is never normalized, symmetrized, or positivity-repaired. Requesting
observables without the detailed result is rejected.
"""
function weak_pi_trajectory_steady_state(
        source::Union{PIModel,CompiledPIModel,WeakPITrajectoryPlan},
        state0::WeakPIPseudoKet{R};trajectories::Integer,settling_time,
        dt::Real,samples_per_trajectory::Integer=1,
        sampling_interval=nothing,seed::Integer=0,threaded::Bool=false,
        workspace=nothing,observables=nothing,confidence::Real=0.95,
        return_info::Bool=false,parameters=nothing,
        max_jump_probability=nothing,batch_size=nothing,
        algorithm::Symbol=:fixed,abstol=nothing,reltol=nothing,
        dtmin=nothing,dtmax=nothing,
        event_time_tolerance=nothing) where R<:AbstractFloat
    path_count_int=_trajectory_integer_count(trajectories,"trajectories",2)
    samples_per_path=_trajectory_integer_count(
        samples_per_trajectory,"samples_per_trajectory",1)
    _checked_statistics_count(R,path_count_int,"trajectory")
    _checked_statistics_count(R,samples_per_path,
                              "trajectory time sample")
    samples_per_path<typemax(Int)||throw(ArgumentError(
        "samples_per_trajectory is too large to construct the sampling grid"))
    batch_samples=if batch_size===nothing
        0
    else
        batch_size isa Integer||throw(ArgumentError(
            "batch_size must be an integer or nothing"))
        _trajectory_integer_count(batch_size,"batch_size",2)
    end
    expected_batch_count=if batch_samples>0
        samples_per_path%batch_samples==0||throw(ArgumentError(
            "batch_size must exactly divide samples_per_trajectory"))
        return_info||throw(ArgumentError(
            "batch_size requires return_info=true so its diagnostics are returned"))
        exact=BigInt(path_count_int)*
            BigInt(samples_per_path÷batch_samples)
        exact<=typemax(Int)||throw(ArgumentError(
            "weak-PI batch-mean count exceeds Int indexing"))
        Int(exact)
    else
        0
    end
    0<confidence<1||throw(ArgumentError("confidence must lie in (0,1)"))
    observables!==nothing&&!return_info&&throw(ArgumentError(
        "weak-PI trajectory steady-state observables require return_info=true"))

    mode=algorithm===:fixed ? :fixed : :full
    plan,batch,single=_weak_ensemble_setup(
        source,state0,path_count_int,threaded,workspace;mode)
    _validate_weak_initial_state(plan,state0)
    isautonomous(plan)||throw(ArgumentError(
        "weak_pi_trajectory_steady_state requires an autonomous model; call freeze(...; time=..., parameters=...) only when the stationary state of that instantaneous generator is intended"))

    times,sampling_times,settling,interval=
        _trajectory_steady_sampling_times(R,settling_time,
            samples_per_path,sampling_interval)
    times,options=_prepare_trajectory_arguments(times,R;dt,parameters,
        max_jump_probability,algorithm,abstol,reltol,dtmin,dtmax,
        event_time_tolerance)
    ops=_prepare_streaming_observables(state0.basis,observables;
                                       require_hermitian=true)

    if batch===nothing
        master=MersenneTwister(seed)
        seeds=rand(master,UInt64,path_count_int)
        workers=(single,)
        rngs=(master,)
        worker_count=1
    else
        master=batch.rngs[1]
        Random.seed!(master,seed)
        resize!(batch.seeds,path_count_int)
        rand!(master,batch.seeds)
        seeds=batch.seeds
        available=length(batch.workers)
        worker_count=threaded ?
            min(path_count_int,Threads.nthreads(),available) : 1
        workers=batch.workers
        rngs=batch.rngs
    end

    scales=_weak_pi_density_scales(state0.basis,R)
    samplers=[_WeakPIDensitySampler(plan,state0,scales;
                                    batch_size=batch_samples)
              for _ in 1:worker_count]
    state_accumulators=[_OnlineStateAccumulator(samplers[index].mean)
                        for index in 1:worker_count]
    observable_type=eltype(state0.data)
    for (_,operator) in ops
        observable_type=promote_type(observable_type,eltype(operator.data))
    end
    Rstats=_real_float_type(observable_type)
    observable_buffers=observables===nothing ? nothing :
        [Matrix{Rstats}(undef,length(ops),1) for _ in 1:worker_count]
    observable_accumulators=observables===nothing ? nothing :
        [_OnlineObservableAccumulator(Rstats,length(ops),1)
         for _ in 1:worker_count]

    if worker_count==1
        worker=workers[1]
        rng=rngs[1]
        for trajectory_index in 1:path_count_int
            Random.seed!(rng,seeds[trajectory_index])
            _weak_steady_state_path!(samplers[1],state_accumulators[1],
                observable_accumulators===nothing ? nothing :
                    observable_accumulators[1],
                observable_buffers===nothing ? nothing : observable_buffers[1],
                ops,plan,state0,times,worker,rng,options,samples_per_path)
        end
    else
        chunk_size=max(1,path_count_int÷(8worker_count))
        next_index=Threads.Atomic{Int}(1)
        @sync for worker_index in 1:worker_count
            let worker=workers[worker_index],rng=rngs[worker_index],
                sampler=samplers[worker_index],
                state_accumulator=state_accumulators[worker_index],
                observable_accumulator=observable_accumulators===nothing ?
                    nothing : observable_accumulators[worker_index],
                observable_buffer=observable_buffers===nothing ?
                    nothing : observable_buffers[worker_index],
                counter=next_index
                Threads.@spawn begin
                    while true
                        first_index=Threads.atomic_add!(counter,chunk_size)
                        first_index>path_count_int&&break
                        final_index=min(path_count_int,
                                        first_index+chunk_size-1)
                        for trajectory_index in first_index:final_index
                            Random.seed!(rng,seeds[trajectory_index])
                            _weak_steady_state_path!(sampler,
                                state_accumulator,observable_accumulator,
                                observable_buffer,ops,plan,state0,times,
                                worker,rng,options,samples_per_path)
                        end
                    end
                end
            end
        end
    end

    merged_state=state_accumulators[1]
    for worker_index in 2:worker_count
        _merge_states!(merged_state,state_accumulators[worker_index])
    end
    merged_state.count==path_count_int||throw(ErrorException(
        "internal weak-PI trajectory reduction retained $(merged_state.count) paths instead of $trajectories"))
    variance_denominator=_checked_statistics_count(
        R,path_count_int-1,"trajectory")
    path_count=_checked_statistics_count(R,path_count_int,"trajectory")
    sample_spread=sqrt(merged_state.m2/variance_denominator)
    standard_error=sample_spread/sqrt(path_count)
    state=PIState(state0.basis,merged_state.mean)

    batch_diagnostics=if batch_samples==0
        nothing
    else
        merged_batches=samplers[1].batch_accumulator
        for worker_index in 2:worker_count
            _merge_states!(merged_batches,
                           samplers[worker_index].batch_accumulator)
        end
        expected_batches=expected_batch_count
        merged_batches.count==expected_batches||throw(ErrorException(
            "internal weak-PI batch-means reduction retained $(merged_batches.count) batches instead of $expected_batches"))
        batch_variance_denominator=_checked_statistics_count(
            R,expected_batches-1,"weak-PI batch mean")
        batch_count_R=_checked_statistics_count(
            R,expected_batches,"weak-PI batch mean")
        batch_spread=sqrt(merged_batches.m2/batch_variance_denominator)
        batch_standard_error=batch_spread/sqrt(batch_count_R)
        assumptions=(;approximately_independent_batches=true,
            batch_length_refinement_required=true,
            burn_in_bias_controlled=false,
            finite_window_bias_controlled=false,
            integration_bias_controlled=false)
        WeakPIBatchMeansDiagnostics(batch_samples,expected_batches,
            expected_batches,batch_spread,batch_standard_error,assumptions)
    end

    density_liouvillian=plan.density_plan.liouvillian
    residual_buffer=similar(state.data)
    residual_workspace=LiouvillianWorkspace(density_liouvillian)
    apply!(residual_buffer,density_liouvillian,state.data,
           residual_workspace)
    residual=norm(residual_buffer)
    relative_residual=residual/max(norm(state.data),one(R))
    state_trace_error=R(trace_error(state))

    observable_summary=if observable_accumulators===nothing
        nothing
    else
        merged=observable_accumulators[1]
        for worker_index in 2:worker_count
            _merge_observables!(merged,observable_accumulators[worker_index])
        end
        _steady_observable_statistics(merged,ops,confidence)
    end
    metadata=(;backend=:weak_pi,algorithm=options.algorithm,seed,
        threaded_requested=threaded,threaded=worker_count>1,worker_count,
        dt=options.dt,max_jump_probability=options.max_jump_probability,
        abstol=options.abstol,reltol=options.reltol,dtmin=options.dtmin,
        dtmax=options.dtmax,
        event_time_tolerance=options.event_time_tolerance,
        settling_time=settling,sampling_interval=interval,confidence,
        path_reduction=:post_settling_density_mean,
        state_reconstruction=:sector_outer_products,
        uncertainty_unit=:independent_path_mean,
        effective_independent_samples=path_count_int,
        batch_means=batch_diagnostics,
        finite_window_bias_controlled=false,
        pseudo_ket_dimension=length(state0),
        density_dimension=length(state0.basis))
    result=TrajectorySteadyStateResult(state,path_count_int,
        samples_per_path,copy(sampling_times),sample_spread,
        standard_error,residual,relative_residual,state_trace_error,
        observable_summary,metadata)
    return_info ? result : state
end

"""Average weak-PI pseudo-ket paths into physical `PIState` objects."""
function weak_pi_trajectory_average(
        trajectories::AbstractVector{<:WeakPIQuantumTrajectory})
    times,b=_check_trajectory_ensemble(trajectories)
    count=length(trajectories);T=_real_float_type(
        eltype(trajectories[1].states[1].data))
    result=[PIState(b;T=T) for _ in eachindex(times)]
    offsets=_weak_pi_offsets(b)
    scales=_weak_pi_density_scales(b,T)
    for trajectory in trajectories,time_index in eachindex(times)
        state=trajectory.states[time_index]
        for (sector,partition) in pairs(b.sectors)
            psi=view(state.data,_weak_sector_range(offsets,sector))
            _weak_density_outer!(coefficient_block(result[time_index],partition),
                                 psi,scales[sector];accumulate=true)
        end
    end
    for state in result
        state.data./=count
    end
    result
end

function _weak_jump_statistics(
        trajectories::AbstractVector{<:WeakPIQuantumTrajectory};
        nchannels=nothing)
    times,_=_check_trajectory_ensemble(trajectories)
    n=length(trajectories);duration=times[end]-times[1]
    inferred=maximum((isempty(path.jump_channels) ? 0 :
        maximum(path.jump_channels) for path in trajectories);init=0)
    channels=nchannels===nothing ? inferred : Int(nchannels)
    channels>=inferred||throw(ArgumentError(
        "nchannels is smaller than an observed channel index"))
    channels>=0||throw(ArgumentError("nchannels must be nonnegative"))
    R=_real_float_type(eltype(trajectories[1].states[1].data))
    means=zeros(R,channels);m2=zeros(R,channels);counts=zeros(Int,channels)
    totals=zeros(Int,channels);total_mean=zero(R);total_m2=zero(R)
    nojump=0;waiting=R[]
    transitions=Dict{Tuple{Any,Any},Int}()
    for (sample,path) in pairs(trajectories)
        fill!(counts,0)
        for channel in path.jump_channels
            1<=channel<=channels||throw(ArgumentError(
                "jump channel indices must be positive"))
            counts[channel]+=1
        end
        for channel in 1:channels
            totals[channel]+=counts[channel]
            delta=R(counts[channel])-means[channel]
            means[channel]+=delta/R(sample)
            m2[channel]+=delta*(R(counts[channel])-means[channel])
        end
        njumps=length(path.jump_times);iszero(njumps)&&(nojump+=1)
        delta=R(njumps)-total_mean;total_mean+=delta/R(sample)
        total_m2+=delta*(R(njumps)-total_mean)
        njumps>=2&&append!(waiting,diff(path.jump_times))
        for record in path.jump_records
            key=(record.source_partition,record.target_partition)
            transitions[key]=get(transitions,key,0)+1
        end
    end
    channel_data=[begin
        variance=_sample_variance(m2[channel],n)
        mean=means[channel]
        (channel,total=totals[channel],mean,variance,
         fano=iszero(mean) ? R(NaN) : variance/mean,
         rate=iszero(duration) ? R(NaN) : mean/duration)
    end for channel in 1:channels]
    total_variance=_sample_variance(total_m2,n)
    mean_wait=isempty(waiting) ? R(NaN) : sum(waiting)/length(waiting)
    wait_variance=length(waiting)>1 ?
        sum(value->abs2(value-mean_wait),waiting)/(length(waiting)-1) :
        R(NaN)
    (;trajectories=n,duration,total_jumps=sum(totals),
      mean_count=total_mean,count_variance=total_variance,
      fano=iszero(total_mean) ? R(NaN) : total_variance/total_mean,
      rate=iszero(duration) ? R(NaN) : total_mean/duration,
      no_jump_probability=R(nojump)/R(n),channels=channel_data,
      waiting_times=waiting,mean_waiting_time=mean_wait,
      waiting_time_variance=wait_variance,sector_transitions=transitions)
end

function _prepare_weak_observables(b,observables)
    prepared=_prepare_streaming_observables(b,observables;
                                             require_hermitian=true)
    Tuple(name=>(blocks=_weak_physical_blocks(operator),)
          for (name,operator) in prepared)
end

function _weak_observable_statistics(trajectories,observables;confidence)
    times,b=_check_trajectory_ensemble(trajectories)
    0<confidence<1||throw(ArgumentError("confidence must lie in (0,1)"))
    prepared=_prepare_weak_observables(b,observables)
    n=length(trajectories);nt=length(times)
    R=_real_float_type(eltype(trajectories[1].states[1].data))
    z=R(_normal_quantile((1+confidence)/2));offsets=_weak_pi_offsets(b)
    results=Dict{Any,Any}()
    for (name,observable) in prepared
        means=zeros(R,nt);m2=zeros(R,nt)
        for (sample,path) in pairs(trajectories),time_index in 1:nt
            value=real(_weak_block_expectation(
                path.states[time_index].data,offsets,observable.blocks))
            delta=value-means[time_index]
            means[time_index]+=delta/R(sample)
            m2[time_index]+=delta*(value-means[time_index])
        end
        variances=n>1 ? m2./R(n-1) : zeros(R,nt)
        standard_error=sqrt.(variances./R(n));half=z.*standard_error
        results[name]=(mean=means,variance=variances,standard_error,
            confidence,lower=means.-half,upper=means.+half)
    end
    (;times=copy(times),trajectories=n,observables=results)
end

"""
    weak_pi_trajectory_statistics(trajectories; observables=nothing,
                                  confidence=0.95, nchannels=nothing)

Return physical ensemble-average PI states, channel and Schur-sector-resolved
jump statistics, and optional Hermitian-observable Monte Carlo statistics for
weak-PI pseudo-ket trajectories.
"""
function weak_pi_trajectory_statistics(
        trajectories::AbstractVector{<:WeakPIQuantumTrajectory};
        observables=nothing,confidence::Real=0.95,nchannels=nothing)
    times,_=_check_trajectory_ensemble(trajectories)
    observable_data=observables===nothing ? nothing :
        _weak_observable_statistics(trajectories,observables;confidence)
    (;times=copy(times),average_states=weak_pi_trajectory_average(trajectories),
      jumps=_weak_jump_statistics(trajectories;nchannels),
      observables=observable_data)
end
