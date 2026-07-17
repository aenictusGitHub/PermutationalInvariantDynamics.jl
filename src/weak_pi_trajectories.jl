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
        elseif kernel isa LocalJumpPIKernel&&term isa LocalJump
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
    WeakPITrajectoryWorkspace(plan, state)

Caller-owned integration and Kraus-selection scratch for one weak-PI
trajectory.  Reuse sequentially; concurrent paths require distinct
workspaces.
"""
struct WeakPITrajectoryWorkspace{V,R,P}
    tmp::V
    k1::V
    k2::V
    k3::V
    k4::V
    current::V
    operator_scratch::V
    jump_output::V
    channel_intensities::Vector{R}
    branch_intensities::Vector{R}
    jump_scales::Vector{R}
    plan::P
end

function WeakPITrajectoryWorkspace(plan::WeakPITrajectoryPlan,
                                   state::WeakPIPseudoKet)
    state.basis===plan.model.basis||throw(ArgumentError(
        "pseudo-ket and weak-PI trajectory plan use incompatible bases"))
    eltype(state.data)===plan.density_plan.liouvillian.Ttype||throw(ArgumentError(
        "pseudo-ket scalar type $(eltype(state.data)) does not match prepared weak-PI scalar type $(plan.density_plan.liouvillian.Ttype)"))
    vector=similar(state.data);R=_real_float_type(eltype(vector))
    WeakPITrajectoryWorkspace(similar(vector),similar(vector),similar(vector),
        similar(vector),similar(vector),vector,similar(vector),similar(vector),
        zeros(R,length(plan.jumps)),zeros(R,length(plan.branches)),
        zeros(R,length(plan.jumps)),plan)
end
WeakPITrajectoryWorkspace(model::PIModel,state::WeakPIPseudoKet)=
    WeakPITrajectoryWorkspace(_weak_plan_for_state(model,state),state)
WeakPITrajectoryWorkspace(compiled::CompiledPIModel,state::WeakPIPseudoKet)=
    WeakPITrajectoryWorkspace(_weak_plan_for_state(compiled,state),state)

"""
    WeakPITrajectoryBatchWorkspace(plan, state; workers=Threads.nthreads())

Reusable task-owned workspace/RNG pool for
[`weak_pi_quantum_trajectories`](@ref).  The immutable Kraus plan is shared;
every worker owns all mutable numerical scratch.
"""
struct WeakPITrajectoryBatchWorkspace{P,W,G,S}
    plan::P
    workers::W
    rngs::G
    seeds::S
end

function WeakPITrajectoryBatchWorkspace(plan::WeakPITrajectoryPlan,
        state::WeakPIPseudoKet;workers::Integer=Threads.nthreads())
    workers>0||throw(ArgumentError("worker count must be positive"))
    workspaces=[WeakPITrajectoryWorkspace(plan,state) for _ in 1:Int(workers)]
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

function _check_weak_batch_workspace(batch::WeakPITrajectoryBatchWorkspace,
                                     source,state)
    _weak_source_matches(batch.plan,source)||throw(ArgumentError(
        "weak-PI batch workspace was prepared for a different source"))
    isempty(batch.workers)&&throw(ArgumentError(
        "weak-PI batch workspace has no workers"))
    length(batch.workers)==length(batch.rngs)||throw(ArgumentError(
        "weak-PI batch workspace has inconsistent worker storage"))
    for worker in batch.workers
        worker.plan===batch.plan||throw(ArgumentError(
            "weak-PI batch workers do not share its plan"))
        _check_weak_workspace(worker,batch.plan,state)
    end
    batch
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
    total=_weak_channel_intensities_recursive!(w,x,t,p,destination,
        Val(apply_generator),w.plan.jumps,1)
    isfinite(total)||throw(ArgumentError(
        "total weak-PI jump intensity is nonfinite"))
    total
end

function _weak_conditional_action!(y,x,w,t,p)
    fill!(y,zero(eltype(y)))
    _weak_apply_hamiltonians!(y,x,w,t,p,w.plan.hamiltonians,
                              w.plan.hamiltonian_terms)
    total=_weak_channel_intensities_recursive!(w,x,t,p,y,Val(true),
        w.plan.jumps,1)
    isfinite(total)||throw(ArgumentError(
        "total weak-PI jump intensity is nonfinite"))
    alpha=convert(eltype(y),total/2)
    @inbounds for index in eachindex(y)
        y[index]+=alpha*x[index]
    end
    y
end

function _weak_conditional_rk4!(x,w,t,h,p)
    _weak_conditional_action!(w.k1,x,w,t,p)
    @. w.tmp=x+(h/2)*w.k1
    _weak_conditional_action!(w.k2,w.tmp,w,t+h/2,p)
    @. w.tmp=x+(h/2)*w.k2
    _weak_conditional_action!(w.k3,w.tmp,w,t+h/2,p)
    @. w.tmp=x+h*w.k3
    _weak_conditional_action!(w.k4,w.tmp,w,t+h,p)
    @. x=x+(h/6)*(w.k1+2w.k2+2w.k3+w.k4)
    value=norm(x);R=eltype(w.channel_intensities)
    isfinite(value)&&value>eps(R)||throw(ArgumentError(
        "weak-PI conditional pseudo-ket acquired zero or nonfinite norm"))
    x./=value
    x
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

function _weak_pi_trajectory_prepared(plan,state0,ts,w,rng,options)
    x=w.current;copyto!(x,state0.data)
    states=Vector{typeof(state0)}(undef,length(ts));states[1]=copy(state0)
    jump_times=eltype(ts)[];jump_channels=Int[]
    D=length(first(plan.model.basis.sectors).parts)
    jump_records=WeakPIJumpRecord{D}[]
    t=ts[1]
    for output_index in 2:length(ts)
        target_time=ts[output_index]
        while t<target_time
            h,lands_on_target=_trajectory_step_to_target(
                t,target_time,options.dt)
            total=_weak_channel_intensities!(w,x,t,options.parameters)
            if total*h>options.max_jump_probability
                h=options.max_jump_probability/total
                lands_on_target=false
            end
            _weak_conditional_rk4!(x,w,t,h,options.parameters)
            t=lands_on_target ? target_time : t+h
            total=_weak_channel_intensities!(w,x,t,options.parameters)
            probability=-expm1(-total*h)
            if total>zero(total)&&rand(rng,typeof(probability))<probability
                branch_total=_weak_branch_intensities!(w,x)
                branch_total>zero(branch_total)||throw(ArgumentError(
                    "selected weak-PI jump has zero branch intensity"))
                branch_index=_select_jump_channel(w.branch_intensities,
                    rand(rng,typeof(branch_total))*branch_total)
                branch=_weak_apply_branch!(x,w,branch_index)
                push!(jump_times,t);push!(jump_channels,branch.channel)
                push!(jump_records,_weak_jump_record(plan,branch_index))
            end
        end
        states[output_index]=WeakPIPseudoKet(state0.basis,x)
    end
    WeakPIQuantumTrajectory(ts,states,jump_times,jump_channels,jump_records)
end

"""
    weak_pi_quantum_trajectory(source, state0, times; dt, rng,
                               parameters=nothing,
                               max_jump_probability=0.05,
                               workspace=nothing)

Simulate one fixed-step weak-PI pseudo-ket trajectory.  `source` is a
`PIModel`, `CompiledPIModel`, or reusable [`WeakPITrajectoryPlan`](@ref).
Every local event samples a mathematically complete sector-changing Kraus
branch, while `jump_channels` retains the model-level channel index.

The step is shortened to control the maximum jump probability.  Converge
`dt` and `max_jump_probability` for production calculations.  This opt-in
backend is distinct from [`quantum_trajectory`](@ref), whose unresolved local
event produces a density-valued conditional PI state.
"""
function weak_pi_quantum_trajectory(
        source::Union{PIModel,CompiledPIModel,WeakPITrajectoryPlan},
        state0::WeakPIPseudoKet{R},times;dt::Real,
        rng::AbstractRNG=Random.default_rng(),parameters=nothing,
        max_jump_probability=nothing,workspace=nothing) where R<:AbstractFloat
    if workspace===nothing
        plan=_weak_plan_for_state(source,state0)
        work=WeakPITrajectoryWorkspace(plan,state0)
    else
        workspace isa WeakPITrajectoryWorkspace||throw(ArgumentError(
            "workspace must be a WeakPITrajectoryWorkspace"))
        work=_check_weak_workspace(workspace,source,state0);plan=work.plan
    end
    ts,options=_prepare_trajectory_arguments(times,R;dt,parameters,
        max_jump_probability,algorithm=:fixed)
    _weak_pi_trajectory_prepared(plan,state0,ts,work,rng,options)
end

function _weak_ensemble_setup(source,state0,n,threaded,workspace)
    n>0||throw(ArgumentError("trajectory count must be positive"))
    if workspace===nothing
        plan=_weak_plan_for_state(source,state0)
        count=threaded ? min(Int(n),Threads.nthreads()) : 1
        batch=WeakPITrajectoryBatchWorkspace(plan,state0;workers=count)
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
                                 workspace=nothing, trajectory_keywords...)

Generate an ensemble of weak-PI pseudo-ket trajectories.  A prepared Kraus
plan is shared read-only, worker scratch is task-owned, and random streams are
derived from the trajectory index so serial and threaded runs are reproducible
for a fixed seed.
"""
function weak_pi_quantum_trajectories(
        source::Union{PIModel,CompiledPIModel,WeakPITrajectoryPlan},
        state0::WeakPIPseudoKet{R},times,n::Integer;seed::Integer=0,
        threaded::Bool=false,workspace=nothing,dt::Real,
        parameters=nothing,max_jump_probability=nothing) where R<:AbstractFloat
    plan,batch,single=_weak_ensemble_setup(
        source,state0,n,threaded,workspace)
    ts,options=_prepare_trajectory_arguments(times,R;dt,parameters,
        max_jump_probability,algorithm=:fixed)
    if batch===nothing
        master=MersenneTwister(seed);seeds=rand(master,UInt64,n)
        workers=(single,);rngs=(master,);worker_count=1
    else
        master=batch.rngs[1];Random.seed!(master,seed)
        resize!(batch.seeds,n);rand!(master,batch.seeds);seeds=batch.seeds
        available=length(batch.workers)
        worker_count=threaded ? min(Int(n),Threads.nthreads(),available) : 1
        workers=batch.workers;rngs=batch.rngs
    end
    TT=eltype(ts);S=typeof(state0);D=length(first(plan.model.basis.sectors).parts)
    Record=WeakPIJumpRecord{D}
    Result=WeakPIQuantumTrajectory{TT,S,Record}
    output=Vector{Result}(undef,n)
    if worker_count==1
        worker=workers[1];rng=rngs[1]
        for index in 1:Int(n)
            Random.seed!(rng,seeds[index])
            output[index]=_weak_pi_trajectory_prepared(
                plan,state0,copy(ts),worker,rng,options)
        end
    else
        chunk_size=max(1,Int(n)÷(8worker_count))
        next_index=Threads.Atomic{Int}(1)
        @sync for worker_index in 1:worker_count
            let worker=workers[worker_index],rng=rngs[worker_index],
                worker_id=worker_index
                Threads.@spawn begin
                    while true
                        first_index=Threads.atomic_add!(next_index,chunk_size)
                        first_index>Int(n)&&break
                        final_index=min(Int(n),first_index+chunk_size-1)
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
