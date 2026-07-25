"""
    DiffusiveMonitor(operator; kind=:homodyne, efficiency=1, phase=0,
                     label=:monitor)
    homodyne_monitor(operator; efficiency=1, phase=0, label=:homodyne)
    heterodyne_monitor(operator; efficiency=1, phase=0, label=:heterodyne)

Describe a continuously monitored collective PI channel.  A one-particle
matrix is interpreted as the collective operator `sum_i operator^(i)`; a
`PIOperator` is used directly.  `efficiency` must lie in `[0,1]`, and `phase`
is the local-oscillator phase.  Either scalar may follow the package
`(time, parameters)` schedule convention.

The monitor contributes only the normalized Ito innovation.  Its unconditional
dissipator must already be present in the `PIModel`; this separation avoids
double counting and also permits partially monitored channels.  Homodyne
monitoring produces one real Wiener record.  Heterodyne monitoring uses two
orthogonal real records, each with innovation strength `sqrt(efficiency/2)`.
Particle-resolved local monitoring is not PI and is deliberately unsupported.
"""
struct DiffusiveMonitor{O,E,P,L}
    operator::O
    kind::Symbol
    efficiency::E
    phase::P
    label::L
    function DiffusiveMonitor(operator::O;kind::Symbol=:homodyne,
            efficiency::E=1,phase::P=0,label::L=:monitor) where {O,E,P,L}
        kind in (:homodyne,:heterodyne)||throw(ArgumentError(
            "diffusive monitor kind must be :homodyne or :heterodyne"))
        operator isa Union{AbstractMatrix,PIOperator}||throw(ArgumentError(
            "a diffusive monitor operator must be a one-particle matrix or PIOperator"))
        efficiency isa Number&&_validate_monitor_efficiency(efficiency)
        phase isa Number&&_validate_monitor_phase(phase)
        new{O,E,P,L}(operator,kind,efficiency,phase,label)
    end
end

"""Construct a homodyne [`DiffusiveMonitor`](@ref) with one real record."""
homodyne_monitor(operator;efficiency=1,phase=0,label=:homodyne)=
    DiffusiveMonitor(operator;kind=:homodyne,efficiency,phase,label)
"""Construct a heterodyne [`DiffusiveMonitor`](@ref) with I/Q records."""
heterodyne_monitor(operator;efficiency=1,phase=0,label=:heterodyne)=
    DiffusiveMonitor(operator;kind=:heterodyne,efficiency,phase,label)

function _validate_monitor_efficiency(value)
    value isa Real||throw(ArgumentError("detector efficiency must be real"))
    isfinite(value)||throw(ArgumentError("detector efficiency must be finite"))
    zero(value)<=value<=one(value)||throw(ArgumentError(
        "detector efficiency must lie in [0,1]"))
    value
end

function _validate_monitor_phase(value)
    value isa Real||throw(ArgumentError("local-oscillator phase must be real"))
    isfinite(value)||throw(ArgumentError("local-oscillator phase must be finite"))
    value
end

struct _PreparedDiffusiveMonitor{B,E,P,L}
    blocks::B
    kind::Symbol
    efficiency::E
    phase::P
    label::L
    first_record::Int
end

"""
    DiffusivePlan(model, monitors; T=nothing)
    DiffusivePlan(compiled, monitors; T=nothing)

Prepare an immutable PI stochastic-master-equation plan.  `monitors` may be a
single [`DiffusiveMonitor`](@ref), a tuple/vector, or a named tuple whose keys
replace the monitor labels.  Physical Schur blocks of all fixed monitored
operators and the unconditional matrix-free Liouvillian are prepared once.

`T` only selects the real floating precision of an otherwise empty model; a
nonempty model infers precision from its compiled terms.  A plan is task safe,
but each concurrent realization needs a separate [`DiffusiveWorkspace`](@ref).
"""
struct DiffusivePlan{M,L,P,W,R,S}
    model::M
    liouvillian::L
    monitors::P
    trace_weights::W
    record_labels::Vector{Any}
    record_count::Int
    real_type::Type{R}
    scalar_type::Type{S}
end

function _monitor_collection(monitors)
    monitors isa DiffusiveMonitor&&return (monitors,)
    if monitors isa NamedTuple
        return Tuple(DiffusiveMonitor(m.operator;kind=m.kind,
            efficiency=m.efficiency,phase=m.phase,label=name)
            for (name,m) in pairs(monitors))
    end
    monitors isa Tuple&&return monitors
    monitors isa AbstractVector&&return Tuple(monitors)
    throw(ArgumentError("monitors must be a DiffusiveMonitor or a collection of them"))
end

function _empty_diffusive_liouvillian(model::PIModel,::Type{R}) where R<:AbstractFloat
    isconcretetype(R)||throw(ArgumentError("T must be a concrete AbstractFloat type"))
    CT=Complex{R}
    LiouvillianPlan(model.basis,(),_trace_functional(model.basis,CT),
                    nothing,CT,true)
end

function _prepare_diffusive_operator(b::PIBasis,operator,::Type{R}) where R
    op=if operator isa AbstractMatrix
        size(operator)==(b.d,b.d)||throw(DimensionMismatch(
            "a monitored one-particle operator must be $(b.d)x$(b.d)"))
        collective_operator(b,operator;
            cache=_collective_geometry(b,R,nothing))
    else
        operator.basis===b||throw(ArgumentError(
            "monitored PIOperator belongs to a different basis"))
        operator
    end
    promote_type(Complex{R},eltype(op.data))===Complex{R}||throw(ArgumentError(
        "monitor scalar type $(eltype(op.data)) cannot be represented by plan precision $(Complex{R}) without narrowing"))
    Tuple(Matrix{Complex{R}}(_divide_by_schur_multiplicity_scale(
        Matrix(coefficient_block(op,sector)),R,sector)) for sector in b.sectors)
end

function _prepare_diffusive_monitors(b,monitors,::Type{R}) where R
    prepared=Any[];labels=Any[];row=1
    for monitor in monitors
        monitor isa DiffusiveMonitor||throw(ArgumentError(
            "every monitor must be a DiffusiveMonitor"))
        blocks=_prepare_diffusive_operator(b,monitor.operator,R)
        push!(prepared,_PreparedDiffusiveMonitor(blocks,monitor.kind,
            monitor.efficiency,monitor.phase,monitor.label,row))
        if monitor.kind===:homodyne
            push!(labels,monitor.label);row+=1
        else
            push!(labels,(monitor.label,:I));push!(labels,(monitor.label,:Q));row+=2
        end
    end
    isempty(prepared)&&throw(ArgumentError("at least one diffusive monitor is required"))
    Tuple(prepared),labels,row-1
end

function _diffusive_plan(model::PIModel,liouvillian,monitors)
    collection=_monitor_collection(monitors)
    R=_real_float_type(liouvillian.Ttype)
    prepared,labels,nrecords=_prepare_diffusive_monitors(
        model.basis,collection,R)
    weights=Vector{R}(undef,length(model.basis.sectors))
    for sector in eachindex(model.basis.sectors)
        value=liouvillian.tracevec[model.basis.offsets[sector]]
        iszero(imag(value))||throw(ArgumentError(
            "diffusive trace weights must be real"))
        weights[sector]=real(value)
    end
    DiffusivePlan(model,liouvillian,prepared,weights,labels,nrecords,R,
                  liouvillian.Ttype)
end

function DiffusivePlan(model::PIModel,monitors;T=nothing)
    if isempty(model.terms)&&T!==nothing
        T isa Type&&T<:AbstractFloat||throw(ArgumentError(
            "T must be a concrete AbstractFloat type"))
        return _diffusive_plan(model,_empty_diffusive_liouvillian(model,T),monitors)
    end
    T===nothing||throw(ArgumentError(
        "T only selects precision for an empty diffusive model"))
    _diffusive_plan(model,LiouvillianPlan(model),monitors)
end

function DiffusivePlan(compiled::CompiledPIModel,monitors;T=nothing)
    if T!==nothing
        isempty(compiled.model.terms)||throw(ArgumentError(
            "T only selects precision for an empty diffusive model"))
        return DiffusivePlan(compiled.model,monitors;T)
    end
    _diffusive_plan(compiled.model,compiled.plan,monitors)
end

"""
    DiffusiveWorkspace(plan, rho)

Caller-owned scratch for one diffusive conditional realization.  Reuse it
sequentially; use one workspace per concurrent task.  The Euler--Maruyama hot
step applies the unconditional Liouvillian and all Schur-block innovations
without constructing a global superoperator or allocating integration arrays.
"""
struct DiffusiveWorkspace{V,Q,L,P}
    current::V
    start::V
    drift::V
    quadrature::V
    left::V
    right::V
    record_increment::Q
    innovation_increment::Q
    liouvillian_work::L
    plan::P
end

"""
    DiffusiveBatchPlan(source, rho0, times, monitors=nothing;
                       dt, observables=nothing)

Prepare the immutable setup shared by a batch of diffusive trajectories.  In
addition to the [`DiffusivePlan`](@ref), this stores the validated time grid,
integration step, and concrete tuple of Hermitian observable operators.  A
batch therefore performs no repeated time conversion, collective-operator
construction, or observable validation for individual paths.

The plan is read-only and may be shared across tasks.  Mutable stochastic
scratch belongs to [`DiffusiveBatchWorkspace`](@ref).
"""
struct DiffusiveBatchPlan{P,T,R,O}
    plan::P
    times::Vector{T}
    dt::R
    observables::O
end

function Base.show(io::IO,batch::DiffusiveBatchPlan)
    print(io,"DiffusiveBatchPlan($(length(batch.times)) times, ",
          "$(length(batch.observables)) observables, dt=$(batch.dt))")
end

function DiffusiveBatchPlan(source,rho0::PIState,times,monitors=nothing;
        dt,observables=nothing)
    plan=source isa DiffusivePlan ? source : DiffusivePlan(source,monitors)
    source isa DiffusivePlan&&monitors!==nothing&&throw(ArgumentError(
        "monitors are already stored in the supplied DiffusivePlan"))
    rho0.basis===plan.model.basis||throw(ArgumentError(
        "initial state and diffusive plan use incompatible PI bases"))
    _check_liouvillian_source_precision(plan.liouvillian,eltype(rho0.data),
                                        "diffusive initial state")
    ts,hmax=_prepare_diffusive_times(times,plan.real_type,dt)
    ops=_prepare_streaming_observables(rho0.basis,observables;
                                       require_hermitian=true)
    DiffusiveBatchPlan(plan,ts,hmax,ops)
end

"""
    DiffusiveBatchWorkspace(batch, rho0; workers=1)

Task-owned mutable workspaces and trajectory-indexed RNG storage for a
prepared diffusive batch.  Use at least as many workers as concurrent tasks.
One batch workspace may be reused sequentially, but not by overlapping batch
calls.
"""
struct DiffusiveBatchWorkspace{B,W,G}
    batch::B
    workers::W
    rngs::G
    seeds::Vector{UInt64}
end

function DiffusiveBatchWorkspace(batch::DiffusiveBatchPlan,rho0::PIState;
        workers::Integer=1)
    workers>=1||throw(ArgumentError("workers must be positive"))
    count=Int(workers)
    scratch=[DiffusiveWorkspace(batch.plan,rho0) for _ in 1:count]
    rngs=[MersenneTwister(0) for _ in 1:count]
    DiffusiveBatchWorkspace(batch,scratch,rngs,UInt64[])
end

function _check_diffusive_batch_workspace(work::DiffusiveBatchWorkspace,
        batch::DiffusiveBatchPlan,rho0::PIState)
    work.batch===batch||throw(ArgumentError(
        "diffusive batch workspace belongs to a different batch plan"))
    isempty(work.workers)&&throw(ArgumentError(
        "diffusive batch workspace has no workers"))
    length(work.workers)==length(work.rngs)||throw(ArgumentError(
        "diffusive batch workspace has incompatible RNG storage"))
    for worker in work.workers
        worker.plan===batch.plan||throw(ArgumentError(
            "diffusive batch worker belongs to a different diffusive plan"))
        rho0.basis===worker.plan.model.basis||throw(ArgumentError(
            "state and diffusive batch workspace use incompatible PI bases"))
        eltype(worker.current)===eltype(rho0.data)||throw(ArgumentError(
            "diffusive batch workspace scalar type $(eltype(worker.current)) does not match initial-state scalar type $(eltype(rho0.data)); construct a workspace for this state precision"))
    end
    work
end

function DiffusiveWorkspace(plan::DiffusivePlan,rho::PIState)
    rho.basis===plan.model.basis||throw(ArgumentError(
        "state and diffusive plan use incompatible PI bases"))
    _check_liouvillian_source_precision(plan.liouvillian,eltype(rho.data),
                                        "diffusive state")
    promote_type(eltype(rho.data),plan.scalar_type)===eltype(rho.data)||
        throw(ArgumentError("diffusive state cannot represent plan precision without narrowing"))
    v=similar(rho.data)
    increments=zeros(plan.real_type,plan.record_count)
    DiffusiveWorkspace(v,similar(v),similar(v),similar(v),similar(v),similar(v),
        increments,similar(increments),LiouvillianWorkspace(plan.liouvillian),plan)
end

"""
    DiffusiveTrajectory

One PI stochastic-master-equation realization.  `records` and `innovations`
are cumulative real quadrature records sampled at `times`, with row labels in
`record_labels`.  `states` is `nothing` when `save_states=false`.
"""
struct DiffusiveTrajectory{T,S,R,O}
    times::Vector{T}
    states::S
    records::Matrix{R}
    innovations::Matrix{R}
    record_labels::Vector{Any}
    observables::O
end

Base.length(result::DiffusiveTrajectory)=length(result.times)
Base.firstindex(::DiffusiveTrajectory)=1
Base.lastindex(result::DiffusiveTrajectory)=length(result)
Base.getindex(result::DiffusiveTrajectory,index::Integer)=begin
    result.states===nothing&&throw(ArgumentError(
        "this diffusive trajectory was created with save_states=false"))
    result.states[index]
end
Base.iterate(result::DiffusiveTrajectory,state::Int=1)=
    state>length(result) ? nothing : (result[state],state+1)

function Base.show(io::IO,result::DiffusiveTrajectory)
    storage=result.states===nothing ? "state-free" : "with state history"
    print(io,"DiffusiveTrajectory($(length(result.times)) times, ",
          "$(size(result.records,1)) records, $storage)")
end

function _diffusive_real_input(::Type{R},value,label) where R<:AbstractFloat
    value isa Real||throw(ArgumentError("$label must be real"))
    if value isa Integer
        converted=R(value)
        isfinite(converted)&&BigInt(converted)==BigInt(value)||throw(ArgumentError(
            "$label is not exactly representable in $R"))
        return converted
    end
    promote_type(R,typeof(value))===R||throw(ArgumentError(
        "$label scalar type $(typeof(value)) would narrow in $R precision"))
    converted=R(value);isfinite(converted)||throw(ArgumentError("$label must be finite"))
    converted
end

function _monitor_parameters(monitor,t,p,::Type{R}) where R<:AbstractFloat
    eta=_diffusive_real_input(R,value_at(monitor.efficiency,t,p),
                              "detector efficiency")
    phase=_diffusive_real_input(R,value_at(monitor.phase,t,p),
                                "local-oscillator phase")
    _validate_monitor_efficiency(eta);_validate_monitor_phase(phase)
    eta,phase
end

function _diffusive_monitor_products!(w::_PreparedDiffusiveMonitor,x,b,
                                      left,right)
    fill!(left,zero(eltype(left)));fill!(right,zero(eltype(right)))
    for sector in eachindex(b.sectors)
        n=length(b.patterns[sector]);range=b.offsets[sector]:b.offsets[sector+1]-1
        X=reshape(view(x,range),n,n)
        L=reshape(view(left,range),n,n)
        R=reshape(view(right,range),n,n)
        K=w.blocks[sector]
        mul!(L,K,X);mul!(R,X,adjoint(K))
    end
    nothing
end

function _quadrature_from_products!(destination,x,z,b,weights,left,right)
    mean_value=zero(eltype(weights))
    for sector in eachindex(b.sectors)
        n=length(b.patterns[sector]);range=b.offsets[sector]:b.offsets[sector+1]-1
        D=reshape(view(destination,range),n,n)
        L=reshape(view(left,range),n,n)
        R=reshape(view(right,range),n,n)
        @inbounds for index in eachindex(D,L,R)
            D[index]=z*L[index]+conj(z)*R[index]
        end
        sector_trace=zero(eltype(weights))
        @inbounds for diagonal in 1:n
            sector_trace+=real(D[diagonal,diagonal])
        end
        mean_value+=weights[sector]*sector_trace
    end
    @. destination=destination-mean_value*x
    mean_value
end

@inline _diffusive_monitor_step!(x,start,w,::Tuple{},b,t,p,h,rng,
                                 records,innovations)=nothing
@inline function _diffusive_monitor_step!(x,start,w,
        monitors::Tuple{M,Vararg{Any}},b,t,p,h,rng,records,innovations) where M
    monitor=first(monitors);R=eltype(records)
    eta,phase=_monitor_parameters(monitor,t,p,R)
    z=cis(-phase)
    _diffusive_monitor_products!(monitor,start,b,w.left,w.right)
    root_h=sqrt(h)
    if monitor.kind===:homodyne
        mean_value=_quadrature_from_products!(w.quadrature,start,z,b,
            w.plan.trace_weights,w.left,w.right)
        dW=root_h*randn(rng,R);scale=sqrt(eta)
        @. x=x+scale*dW*w.quadrature
        row=monitor.first_record
        records[row]+=scale*mean_value*h+dW
        innovations[row]+=dW
    else
        scale=sqrt(eta/R(2));row=monitor.first_record
        mean_i=_quadrature_from_products!(w.quadrature,start,z,b,
            w.plan.trace_weights,w.left,w.right)
        dWi=root_h*randn(rng,R)
        @. x=x+scale*dWi*w.quadrature
        records[row]+=scale*mean_i*h+dWi;innovations[row]+=dWi
        mean_q=_quadrature_from_products!(w.quadrature,start,-im*z,b,
            w.plan.trace_weights,w.left,w.right)
        dWq=root_h*randn(rng,R)
        @. x=x+scale*dWq*w.quadrature
        records[row+1]+=scale*mean_q*h+dWq;innovations[row+1]+=dWq
    end
    _diffusive_monitor_step!(x,start,w,Base.tail(monitors),b,t,p,h,rng,
                             records,innovations)
end

function _normalize_diffusive!(x,tracevec,::Type{R}) where R
    value=dot(tracevec,x)
    abs(imag(value))<=R(100)*eps(R)*max(one(R),abs(real(value)))||
        throw(ArgumentError("diffusive state acquired a complex trace $value"))
    normalization=real(value)
    isfinite(normalization)&&abs(normalization)>eps(R)||throw(ArgumentError(
        "diffusive state acquired a zero or nonfinite trace"))
    x./=normalization
    x
end

function _diffusive_step!(w::DiffusiveWorkspace,t,h,p,rng,
                          record_increment,innovation_increment)
    plan=w.plan;b=plan.model.basis;x=w.current
    copyto!(w.start,x)
    apply!(w.drift,plan.liouvillian,w.start,t,p,w.liouvillian_work)
    @. x=w.start+h*w.drift
    fill!(record_increment,zero(eltype(record_increment)))
    fill!(innovation_increment,zero(eltype(innovation_increment)))
    _diffusive_monitor_step!(x,w.start,w,plan.monitors,b,t,p,h,rng,
        record_increment,innovation_increment)
    _normalize_diffusive!(x,plan.liouvillian.tracevec,plan.real_type)
end

function _prepare_diffusive_times(times,::Type{R},dt) where R
    raw=collect(times);length(raw)>=1||throw(ArgumentError("times cannot be empty"))
    ts=Vector{R}(undef,length(raw))
    for i in eachindex(raw);ts[i]=_diffusive_real_input(R,raw[i],"time");end
    issorted(ts)||throw(ArgumentError("times must be nondecreasing"))
    h=_diffusive_real_input(R,dt,"dt");h>0||throw(ArgumentError("dt must be positive"))
    ts,h
end

"""
    diffusive_trajectory(source, rho0, times, monitors; dt,
                         rng=Random.default_rng(), parameters=nothing,
                         workspace=nothing, save_states=true,
                         observables=nothing)

Integrate one normalized PI homodyne/heterodyne stochastic master equation
with preallocated Euler--Maruyama microsteps.  `source` is a `PIModel`, a
`CompiledPIModel`, or a prepared [`DiffusivePlan`](@ref).  The returned
measurement and innovation arrays are cumulative at the requested `times`.

The unconditional `source` must already contain every monitored dissipator.
Decrease `dt` to convergence.  Euler--Maruyama is trace-normalized but is not a
finite-step positivity certificate; use [`validate_state`](@ref) on saved
states when strict physicality auditing is required.
"""
function diffusive_trajectory end

function _diffusive_trajectory_prepared(batch::DiffusiveBatchPlan,
        rho0::PIState,w::DiffusiveWorkspace,rng::AbstractRNG;
        parameters=nothing,save_states::Bool=true,save_records::Bool=true,
        observable_values=nothing,return_result::Bool=true)
    return_result&&!save_records&&throw(ArgumentError(
        "return_result=true requires save_records=true"))
    plan=batch.plan
    rho0.basis===plan.model.basis||throw(ArgumentError(
        "initial state and diffusive plan use incompatible PI bases"))
    _check_liouvillian_source_precision(plan.liouvillian,eltype(rho0.data),
                                        "diffusive initial state")
    w isa DiffusiveWorkspace&&w.plan===plan||throw(ArgumentError(
        "workspace belongs to a different diffusive plan"))
    eltype(w.current)===eltype(rho0.data)||throw(ArgumentError(
        "diffusive workspace scalar type $(eltype(w.current)) does not match initial-state scalar type $(eltype(rho0.data)); construct a workspace for this state precision"))
    length(w.record_increment)==plan.record_count&&
        length(w.innovation_increment)==plan.record_count||throw(
        DimensionMismatch("diffusive workspace has incompatible record scratch"))
    R=plan.real_type;ts=batch.times;hmax=batch.dt
    copyto!(w.current,rho0.data)
    _normalize_diffusive!(w.current,plan.liouvillian.tracevec,R)
    states=save_states ? Vector{typeof(rho0)}(undef,length(ts)) : nothing
    save_states&&(states[1]=PIState(rho0.basis,w.current))
    ops=batch.observables
    values=if isempty(ops)
        observable_values===nothing||throw(ArgumentError(
            "observable_values was supplied for a batch without observables"))
        nothing
    elseif observable_values===nothing
        zeros(R,length(ops),length(ts))
    else
        size(observable_values)==(length(ops),length(ts))||throw(
            DimensionMismatch("observable_values has incompatible dimensions"))
        promote_type(R,eltype(observable_values))===eltype(observable_values)||
            throw(ArgumentError(
                "observable_values cannot represent diffusive observable precision"))
        observable_values
    end
    values===nothing||_record_observables!(values,ops,w.current,1)
    records=save_records ? zeros(R,plan.record_count,length(ts)) : nothing
    innovations=save_records ? zeros(R,plan.record_count,length(ts)) : nothing
    record_increment=w.record_increment
    innovation_increment=w.innovation_increment
    for output_index in 2:length(ts)
        t=ts[output_index-1];target=ts[output_index]
        if save_records
            copyto!(view(records,:,output_index),view(records,:,output_index-1))
            copyto!(view(innovations,:,output_index),view(innovations,:,output_index-1))
        end
        while t<target
            h,lands_on_target=_trajectory_step_to_target(t,target,hmax)
            _diffusive_step!(w,t,h,parameters,rng,record_increment,
                              innovation_increment)
            if save_records
                @views records[:,output_index].+=record_increment
                @views innovations[:,output_index].+=innovation_increment
            end
            t=lands_on_target ? target : t+h
        end
        save_states&&(states[output_index]=PIState(rho0.basis,w.current))
        values===nothing||_record_observables!(
            values,ops,w.current,output_index)
    end
    return_result||return nothing
    observable_result=values===nothing ? nothing :
        (;names=map(first,ops),values)
    DiffusiveTrajectory(copy(ts),states,records,innovations,
                        copy(plan.record_labels),observable_result)
end

function diffusive_trajectory(batch::DiffusiveBatchPlan,rho0::PIState;
        rng=Random.default_rng(),parameters=nothing,workspace=nothing,
        save_states::Bool=true,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    state_entries=save_states ? BigInt(length(rho0.data))*length(batch.times) : 0
    record_entries=2BigInt(batch.plan.record_count)*length(batch.times)
    observable_entries=BigInt(length(batch.observables))*length(batch.times)
    estimate=_performance_entries_bytes(state_entries,eltype(rho0.data))+
        _performance_entries_bytes(record_entries+observable_entries,
                                   batch.plan.real_type)+
        _performance_entries_bytes(length(batch.times),eltype(batch.times))
    _require_performance_budget("diffusive trajectory output",estimate,
        memory_budget;guidance=
        "Set save_states=false and request only required output times.")
    w=workspace===nothing ? DiffusiveWorkspace(batch.plan,rho0) : workspace
    _diffusive_trajectory_prepared(batch,rho0,w,rng;parameters,save_states)
end

function diffusive_trajectory(source,rho0::PIState,times,monitors=nothing;
        dt,rng=Random.default_rng(),parameters=nothing,workspace=nothing,
        save_states::Bool=true,observables=nothing,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    R=_real_float_type(eltype(rho0.data))
    preview_times,_=_prepare_diffusive_times(times,R,dt)
    preview_entries=save_states ?
        BigInt(length(rho0.data))*length(preview_times) : big(0)
    preview=_performance_entries_bytes(preview_entries,eltype(rho0.data))+
        _performance_entries_bytes(length(preview_times),eltype(preview_times))
    _require_performance_budget("diffusive trajectory output",preview,
        memory_budget;guidance=
        "Set save_states=false and request only required output times.")
    batch=DiffusiveBatchPlan(source,rho0,preview_times,monitors;dt,observables)
    diffusive_trajectory(batch,rho0;rng,parameters,workspace,save_states,
                         memory_budget)
end

"""
    diffusive_trajectories(source, rho0, times, monitors, n;
                           seed=0, threaded=false, kwargs...)

Generate `n` reproducible diffusive PI trajectories.  Random streams are
derived from trajectory index, so ordered results are independent of threaded
scheduling.  Each worker owns a reusable [`DiffusiveWorkspace`](@ref).
"""
function diffusive_trajectories(source,rho0::PIState,times,monitors,n::Integer;
        seed::Integer=0,threaded::Bool=false,dt,parameters=nothing,
        workspace=nothing,save_states::Bool=true,observables=nothing,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    n>0||throw(ArgumentError("trajectory count must be positive"))
    R=_real_float_type(eltype(rho0.data))
    preview_times,_=_prepare_diffusive_times(times,R,dt)
    preview_entries=save_states ?
        BigInt(n)*length(rho0.data)*length(preview_times) : big(0)
    preview=_performance_entries_bytes(preview_entries,eltype(rho0.data))+
        _performance_entries_bytes(BigInt(n)*length(preview_times),
                                   eltype(preview_times))
    _require_performance_budget("diffusive trajectory ensemble output",preview,
        memory_budget;guidance=
        "Set save_states=false and request only required output times.")
    batch=DiffusiveBatchPlan(source,rho0,preview_times,monitors;dt,observables)
    if workspace isa DiffusiveWorkspace
        state_entries=save_states ?
            BigInt(n)*length(rho0.data)*length(batch.times) : big(0)
        record_entries=2BigInt(n)*batch.plan.record_count*length(batch.times)
        observable_entries=BigInt(n)*length(batch.observables)*
                           length(batch.times)
        estimate=_performance_entries_bytes(state_entries,
            eltype(rho0.data))+
            _performance_entries_bytes(record_entries+observable_entries,
                                       batch.plan.real_type)+
            _performance_entries_bytes(BigInt(n)*length(batch.times),
                                       eltype(batch.times))
        _require_performance_budget("diffusive trajectory ensemble output",
            estimate,memory_budget;guidance=
            "Set save_states=false and request only required output times.")
        threaded&&throw(ArgumentError(
            "threaded diffusive ensembles require a DiffusiveBatchWorkspace"))
        workspace.plan===batch.plan||throw(ArgumentError(
            "workspace belongs to a different diffusive plan"))
        master=MersenneTwister(seed);seeds=rand(master,UInt64,n)
        results=Vector{DiffusiveTrajectory}(undef,n)
        for index in 1:Int(n)
            Random.seed!(master,seeds[index])
            results[index]=_diffusive_trajectory_prepared(
                batch,rho0,workspace,master;parameters,save_states)
        end
        return results
    end
    diffusive_trajectories(batch,rho0,n;seed,threaded,workspace,
                           parameters,save_states,memory_budget)
end


"""
    diffusive_trajectories(batch, rho0, n; seed=0, threaded=false,
                           workspace=nothing, parameters=nothing,
                           save_states=true)

Generate `n` trajectories from a [`DiffusiveBatchPlan`](@ref).  Prepared times
and observables are reused by every path.  Random streams are derived from the
trajectory index, so serial and threaded results agree for a fixed seed.
`memory_budget` covers saved states, times, fixed-size measurement and
innovation records, and requested observable series; prepared plans and
reusable worker scratch are additional.
"""
function diffusive_trajectories(batch::DiffusiveBatchPlan,rho0::PIState,
        n::Integer;seed::Integer=0,threaded::Bool=false,workspace=nothing,
        parameters=nothing,save_states::Bool=true,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    n>0||throw(ArgumentError("trajectory count must be positive"))
    state_entries=save_states ?
        BigInt(n)*length(rho0.data)*length(batch.times) : big(0)
    record_entries=2BigInt(n)*batch.plan.record_count*length(batch.times)
    observable_entries=BigInt(n)*length(batch.observables)*length(batch.times)
    time_entries=BigInt(n)*length(batch.times)
    estimate=_performance_entries_bytes(state_entries,eltype(rho0.data))+
        _performance_entries_bytes(record_entries+observable_entries,
                                   batch.plan.real_type)+
        _performance_entries_bytes(time_entries,eltype(batch.times))
    _require_performance_budget("diffusive trajectory ensemble output",
        estimate,memory_budget;guidance=
        "Set save_states=false and request only required output times.")
    requested_workers=threaded ? min(Int(n),Threads.nthreads()) : 1
    work=workspace===nothing ?
        DiffusiveBatchWorkspace(batch,rho0;workers=requested_workers) :
        _check_diffusive_batch_workspace(workspace,batch,rho0)
    master=work.rngs[1];Random.seed!(master,seed)
    resize!(work.seeds,n);rand!(master,work.seeds);seeds=work.seeds
    results=Vector{DiffusiveTrajectory}(undef,n)
    available=length(work.workers)
    worker_count=threaded ? min(Int(n),Threads.nthreads(),available) : 1
    if worker_count>1
        chunk_size=max(1,Int(n)÷(8worker_count))
        next=Threads.Atomic{Int}(1)
        @sync for worker_index in 1:worker_count
            let worker=work.workers[worker_index],rng=work.rngs[worker_index],
                counter=next
                Threads.@spawn begin
                    while true
                        first_index=Threads.atomic_add!(counter,chunk_size)
                        first_index>Int(n)&&break
                        final_index=min(Int(n),first_index+chunk_size-1)
                        for index in first_index:final_index
                            Random.seed!(rng,seeds[index])
                            results[index]=_diffusive_trajectory_prepared(
                                batch,rho0,worker,rng;parameters,save_states)
                        end
                    end
                end
            end
        end
    else
        worker=work.workers[1];rng=work.rngs[1]
        for index in 1:n
            Random.seed!(rng,seeds[index])
            results[index]=_diffusive_trajectory_prepared(
                batch,rho0,worker,rng;parameters,save_states)
        end
    end
    results
end

diffusive_trajectories(plan::DiffusivePlan,rho0::PIState,times,n::Integer;
                       kwargs...)=
    diffusive_trajectories(plan,rho0,times,nothing,n;kwargs...)

"""
    diffusive_average(trajectories)

Average saved conditional PI states at every sampling time.  The result is an
unconditional state estimate; all trajectories must share the same basis and
time grid and must have been created with `save_states=true`.
"""
function diffusive_average(trajectories::AbstractVector{<:DiffusiveTrajectory})
    isempty(trajectories)&&throw(ArgumentError("trajectory collection cannot be empty"))
    first_result=first(trajectories);first_result.states===nothing&&throw(ArgumentError(
        "diffusive_average requires saved state histories"))
    ntimes=length(first_result.times);basis=first(first_result.states).basis
    T=eltype(first(first_result.states).data);means=[zeros(T,length(basis)) for _ in 1:ntimes]
    for trajectory in trajectories
        trajectory.states===nothing&&throw(ArgumentError(
            "diffusive_average requires saved state histories"))
        trajectory.times==first_result.times||throw(ArgumentError(
            "all diffusive trajectories must share the same time grid"))
        for index in 1:ntimes
            state=trajectory.states[index];state.basis===basis||throw(ArgumentError(
                "all diffusive trajectories must share the same basis"))
            means[index].+=state.data
        end
    end
    R=_real_float_type(T);scale=one(R)/R(length(trajectories))
    for mean in means
        mean .*= scale
    end
    [PIState(basis,mean) for mean in means]
end
