"""
    RefinementSpec(parameter, levels; atol=0, rtol=nothing,
                   consecutive=2, require_convergence=true)

Immutable numerical-refinement request for [`PIExperiment`](@ref).
`parameter=:steps_per_interval` is supported for deterministic dynamics and
`parameter=:krylov_dimension` for matrix-free GMRES stationary states.

A refinement request never changes the physical representation. In
particular, it does not add or remove Schur sectors, change a pseudomode
cutoff, or replace deterministic dynamics by trajectories.
"""
struct RefinementSpec{L,A,R}
    parameter::Symbol
    levels::L
    atol::A
    rtol::R
    consecutive::Int
    require_convergence::Bool
end

function RefinementSpec(parameter::Symbol,levels;
        atol::Real=0,rtol=nothing,consecutive::Integer=2,
        require_convergence::Bool=true)
    parameter in (:steps_per_interval,:krylov_dimension)||
        throw(ArgumentError(
            "experiment refinement parameter must be :steps_per_interval " *
            "or :krylov_dimension"))
    values=Tuple(levels)
    length(values)>=2||throw(ArgumentError(
        "an experiment refinement requires at least two levels"))
    all(value->value isa Integer&&!(value isa Bool)&&value>0,values)||
        throw(ArgumentError("experiment refinement levels must be positive integers"))
    all(value->BigInt(value)<=typemax(Int),values)||throw(ArgumentError(
        "experiment refinement levels must be representable as Int"))
    all(values[index+1]>values[index] for index in 1:length(values)-1)||
        throw(ArgumentError(
            "experiment refinement levels must be strictly increasing"))
    isfinite(atol)&&atol>=zero(atol)||throw(ArgumentError(
        "experiment refinement atol must be finite and nonnegative"))
    rtol===nothing||rtol isa Real&&isfinite(rtol)&&rtol>=zero(rtol)||
        throw(ArgumentError(
            "experiment refinement rtol must be nothing or finite and nonnegative"))
    consecutive isa Bool&&throw(ArgumentError(
        "experiment refinement consecutive must be an integer, not Bool"))
    consecutive>0||throw(ArgumentError(
        "experiment refinement consecutive must be positive"))
    consecutive<length(values)||throw(ArgumentError(
        "experiment refinement needs at least consecutive + 1 levels"))
    BigInt(consecutive)<=typemax(Int)||throw(ArgumentError(
        "experiment refinement consecutive is not representable as Int"))
    RefinementSpec(parameter,values,atol,rtol,Int(consecutive),
                   require_convergence)
end

"""
    VerificationSpec(; atol=1e-10, rtol=1e-8,
                     positivity_method=:auto, dense_threshold=256,
                     require_physical=true,
                     require_solver_convergence=true,
                     refinement=nothing)

Validation policy applied by [`verified_solve`](@ref). The checks report
physicality and solver/refinement evidence without normalizing, symmetrizing,
clipping, or otherwise repairing a state.
"""
struct VerificationSpec{A,R,F}
    atol::A
    rtol::R
    positivity_method::Symbol
    dense_threshold::Int
    require_physical::Bool
    require_solver_convergence::Bool
    refinement::F
end

function VerificationSpec(;atol::Real=1e-10,rtol::Real=1e-8,
        positivity_method::Symbol=:auto,dense_threshold::Integer=256,
        require_physical::Bool=true,
        require_solver_convergence::Bool=true,refinement=nothing)
    isfinite(atol)&&atol>=zero(atol)||throw(ArgumentError(
        "verification atol must be finite and nonnegative"))
    isfinite(rtol)&&rtol>=zero(rtol)||throw(ArgumentError(
        "verification rtol must be finite and nonnegative"))
    positivity_method in (:auto,:eigen,:cholesky)||throw(ArgumentError(
        "positivity_method must be :auto, :eigen, or :cholesky"))
    dense_threshold isa Bool&&throw(ArgumentError(
        "dense_threshold must be an integer, not Bool"))
    dense_threshold>0||throw(ArgumentError(
        "dense_threshold must be positive"))
    BigInt(dense_threshold)<=typemax(Int)||throw(ArgumentError(
        "dense_threshold is not representable as Int"))
    refinement===nothing||refinement isa RefinementSpec||throw(ArgumentError(
        "refinement must be nothing or RefinementSpec"))
    VerificationSpec(atol,rtol,positivity_method,Int(dense_threshold),
        require_physical,require_solver_convergence,refinement)
end

function _experiment_metadata(metadata)
    values=Pair{String,String}[]
    for (key,value) in pairs(metadata)
        push!(values,string(key)=>string(value))
    end
    sort!(values;by=first)
    length(unique(first.(values)))==length(values)||throw(ArgumentError(
        "experiment metadata keys must be unique after string conversion"))
    Tuple(values)
end

_experiment_snapshot(value)=value
_experiment_snapshot(value::PIState)=copy(value)
_experiment_snapshot(value::PIOperator)=copy(value)
_experiment_snapshot(value::Pair)=
    _experiment_snapshot(first(value))=>_experiment_snapshot(last(value))
_experiment_snapshot(value::Tuple)=map(_experiment_snapshot,value)
function _experiment_snapshot(value::NamedTuple)
    NamedTuple{keys(value)}(map(_experiment_snapshot,values(value)))
end
_experiment_snapshot(value::AbstractRange)=value
function _experiment_snapshot(value::AbstractArray)
    output=copy(value)
    isbitstype(eltype(value))&&return output
    for index in eachindex(value,output)
        output[index]=_experiment_snapshot(value[index])
    end
    output
end
function _experiment_snapshot(value::AbstractDict)
    output=copy(value)
    for key in keys(value)
        output[key]=_experiment_snapshot(value[key])
    end
    output
end

function _experiment_term_operator_bytes(term::_BuiltinPITerm)
    operator=term_operator(term)
    prototype=operator isa InPlaceTimeOperator ?
        operator.prototype : operator
    if prototype isa AbstractPIOperator
        return _performance_entries_bytes(
            length(prototype.data),eltype(prototype.data))
    elseif prototype isa AbstractArray
        return _performance_entries_bytes(
            length(prototype),eltype(prototype))
    end
    big(0)
end

function _experiment_snapshot_term(term::_BuiltinPITerm)
    operator=term_operator(term)
    stored=if operator isa InPlaceTimeOperator
        InPlaceTimeOperator(operator.prototype,operator.update!)
    elseif operator isa Union{AbstractArray,AbstractPIOperator}
        copy(operator)
    else
        operator
    end
    stored===operator ? term :
        rebuild_term(term,stored,term_rate(term))
end
_experiment_snapshot_term(term::AbstractPITerm)=term

function _experiment_snapshot_source(source::PIModel,memory_budget)
    bytes=sum((_experiment_term_operator_bytes(term)
               for term in source.terms if term isa _BuiltinPITerm);
              init=big(0))
    _require_performance_budget(
        "PIExperiment model snapshot",bytes,memory_budget;
        guidance="Compile the model first to pass a prepared immutable source, " *
                 "or raise the explicit experiment memory budget.")
    PIModel(source.basis,map(_experiment_snapshot_term,source.terms))
end
_experiment_snapshot_source(source,memory_budget)=source

function _experiment_basis(source,initial_state)
    basis=_operator_basis(source)
    basis===nothing&&initial_state isa PIState&&(basis=initial_state.basis)
    basis
end

function _experiment_complete_basis(basis::PIBasis)
    _basis_is_complete(basis)
end

"""
    PIExperiment(source; task=:steady_state, initial_state=nothing,
                 algorithm=AutoAlgorithm(), observables=nothing,
                 tspan=nothing, saveat=nothing, steps_per_interval=64,
                 parameters=nothing, save_states=true,
                 memory_budget=512*1024^2,
                 representation=:complete_pi,
                 verification=VerificationSpec(),
                 solver_options=NamedTuple(), metadata=NamedTuple())

Typed immutable description of one reproducible PI calculation. Supported
tasks are `:steady_state` and `:dynamics`.

`representation=:complete_pi` (the default) requires the complete PI Schur
basis. A deliberately restricted basis must be declared explicitly with
`representation=:declared_sectors`; the resulting report marks this as a
user-declared representation restriction rather than silently calling it an
exact full-PI calculation.

`solver_options` is reserved for stationary-state keyword arguments. The
high-level memory guard remains active, and no explicit algorithm is replaced
by another one. `algorithm=:auto` or `AutoAlgorithm()` delegates only to the
package's exact high-level route selection.

Fixed operator matrices in built-in terms of a raw `PIModel` are copied into
the experiment. Prepared immutable sources are retained directly. Callbacks
and custom extension terms remain caller-owned and make the provenance digest
incomplete unless their state is represented by supported immutable fields.
"""
struct PIExperiment{S,I,A,O,TS,SA,P,M,V,K,MD}
    source::S
    initial_state::I
    task::Symbol
    algorithm::A
    observables::O
    tspan::TS
    saveat::SA
    steps_per_interval::Int
    parameters::P
    save_states::Bool
    memory_budget::M
    representation::Symbol
    verification::V
    solver_options::K
    metadata::MD
end

function PIExperiment(source;task::Symbol=:steady_state,
        initial_state=nothing,algorithm=AutoAlgorithm(),observables=nothing,
        tspan=nothing,saveat=nothing,steps_per_interval::Integer=64,
        parameters=nothing,save_states::Bool=true,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,
        representation::Symbol=:complete_pi,
        verification=VerificationSpec(),solver_options=NamedTuple(),
        metadata=NamedTuple())
    task in (:steady_state,:dynamics)||throw(ArgumentError(
        "PIExperiment task must be :steady_state or :dynamics"))
    representation in (:complete_pi,:declared_sectors)||throw(ArgumentError(
        "representation must be :complete_pi or :declared_sectors"))
    verification isa VerificationSpec||throw(ArgumentError(
        "verification must be a VerificationSpec"))
    solver_options isa NamedTuple||throw(ArgumentError(
        "solver_options must be a NamedTuple"))
    reserved=task===:steady_state ?
        (:algorithm,:memory_budget,:return_info) :
        (:algorithm,:memory_budget,:saveat,:steps_per_interval,:parameters,
         :observables,:save_states)
    conflicting=filter(name->haskey(solver_options,name),reserved)
    isempty(conflicting)||throw(ArgumentError(
        "solver_options contains reserved experiment keywords $conflicting"))
    steps_per_interval isa Bool&&throw(ArgumentError(
        "steps_per_interval must be an integer, not Bool"))
    steps_per_interval>0||throw(ArgumentError(
        "steps_per_interval must be positive"))
    BigInt(steps_per_interval)<=typemax(Int)||throw(ArgumentError(
        "steps_per_interval is not representable as Int"))
    _resource_memory_budget(memory_budget)
    source=_experiment_snapshot_source(source,memory_budget)
    basis=_experiment_basis(source,initial_state)
    basis isa PIBasis||throw(ArgumentError(
        "PIExperiment currently requires a PI source with PIBasis metadata"))
    if representation===:complete_pi&&!_experiment_complete_basis(basis)
        throw(ArgumentError(
            "representation=:complete_pi requires every Schur sector; use " *
            "representation=:declared_sectors to acknowledge a restricted basis"))
    end
    if initial_state!==nothing
        initial_state isa PIState||throw(ArgumentError(
            "initial_state must be a PIState or nothing"))
        initial_state.basis===basis||throw(ArgumentError(
            "initial_state belongs to a different PI basis"))
    end
    if task===:steady_state
        _,algorithm_options=_algorithm_options(algorithm)
        algorithm_conflicts=filter(
            name->haskey(solver_options,name),keys(algorithm_options))
        isempty(algorithm_conflicts)||throw(ArgumentError(
            "solver_options duplicates fields already fixed by the algorithm " *
            "object: $algorithm_conflicts"))
        initial_state===nothing||throw(ArgumentError(
            "steady-state initial guesses belong in solver_options=(initial_state=...,)"))
        tspan===nothing||throw(ArgumentError(
            "tspan is only valid for task=:dynamics"))
        saveat===nothing||throw(ArgumentError(
            "saveat is only valid for task=:dynamics"))
        parameters===nothing||throw(ArgumentError(
            "parameters is only valid for task=:dynamics"))
        save_states||throw(ArgumentError(
            "save_states=false is only valid for task=:dynamics"))
    else
        _dynamics_algorithm_options(algorithm)
        initial_state isa PIState||throw(ArgumentError(
            "task=:dynamics requires initial_state=PIState(...)"))
        tspan===nothing&&throw(ArgumentError(
            "task=:dynamics requires tspan=(t0,t1)"))
        length(tspan)==2||throw(ArgumentError(
            "dynamics tspan must contain exactly two endpoints"))
        tspan[2]>=tspan[1]||throw(ArgumentError(
            "dynamics tspan must be ordered"))
        !save_states&&observables===nothing&&throw(ArgumentError(
            "state-free dynamics requires at least one observable"))
        isempty(solver_options)||throw(ArgumentError(
            "solver_options is currently reserved for steady-state experiments"))
    end
    refinement=verification.refinement
    if refinement!==nothing
        expected=task===:steady_state ? :krylov_dimension :
            :steps_per_interval
        refinement.parameter===expected||throw(ArgumentError(
            "$task experiments support refinement parameter :$expected, " *
            "not :$(refinement.parameter)"))
        task===:dynamics&&!save_states&&throw(ArgumentError(
            "dynamics refinement requires save_states=true so successive " *
            "physical final states can be compared"))
        task===:steady_state&&haskey(solver_options,:krylovdim)&&
            throw(ArgumentError(
                "a krylov_dimension refinement owns krylovdim; remove it " *
                "from solver_options"))
    end
    initial_state=_experiment_snapshot(initial_state)
    canonical_observables=if observables===nothing
        nothing
    else
        Pair[name=>copy(operator) for (name,operator) in
            _prepare_streaming_observables(
                basis,observables;require_hermitian=false)]
    end
    canonical_tspan=tspan===nothing ? nothing :
        Tuple(_experiment_snapshot(value) for value in tspan)
    canonical_saveat=_experiment_snapshot(saveat)
    canonical_parameters=_experiment_snapshot(parameters)
    canonical_solver_options=_experiment_snapshot(solver_options)
    PIExperiment(source,initial_state,task,algorithm,canonical_observables,
        canonical_tspan,canonical_saveat,
        Int(steps_per_interval),canonical_parameters,save_states,memory_budget,
        representation,verification,canonical_solver_options,
        _experiment_metadata(metadata))
end

"""
    ExperimentExecutionPlan

Read-only explanation of the route selected for a [`PIExperiment`](@ref).
`recommendation` is the ordinary `recommend_solver` resource report.
Constructing a plan performs validation and resource estimation, but no model
compilation or solve.
"""
struct ExperimentExecutionPlan{A,R,E,N}
    task::Symbol
    requested_algorithm::A
    selected_algorithm::Symbol
    backend::Symbol
    recommendation::R
    exactness::E
    notes::N
end

function Base.show(io::IO,plan::ExperimentExecutionPlan)
    print(io,"ExperimentExecutionPlan(task=$(plan.task), backend=$(plan.backend), ",
        "algorithm=$(plan.selected_algorithm), representation=",
        plan.exactness.representation,", budget_status=",
        plan.recommendation.budget_status,")")
end

function _experiment_observable_count(observables)
    observables===nothing&&return 0
    length(_named_observables(observables))
end

function _experiment_algorithm_request(experiment::PIExperiment)
    if experiment.task===:steady_state
        method,options=_algorithm_options(experiment.algorithm)
        krylovdim=haskey(options,:krylovdim) ? options.krylovdim :
            get(experiment.solver_options,:krylovdim,30)
        return method,krylovdim
    end
    method,options=_dynamics_algorithm_options(experiment.algorithm)
    method,get(options,:krylovdim,30)
end

function _experiment_exactness(experiment::PIExperiment)
    basis=_experiment_basis(experiment.source,experiment.initial_state)
    complete=basis isa PIBasis&&_experiment_complete_basis(basis)
    (;representation=complete ? :complete_pi : experiment.representation,
      declared_representation=experiment.representation,
      complete_pi_basis=complete,
      physical_approximation=complete ? :none : :user_declared_sector_restriction,
      deterministic=true,
      numerical_approximation=experiment.task===:steady_state ?
          :linear_solver_tolerance : :time_propagation_tolerance)
end

"""
    plan_experiment(experiment)
    explain_experiment(experiment)

Return an immutable, explainable resource and route plan without solving the
experiment. `explain_experiment` is an alias intended for interactive use.
"""
function plan_experiment(experiment::PIExperiment)
    method,krylovdim=_experiment_algorithm_request(experiment)
    source=experiment.source
    recommendation=if experiment.task===:steady_state
        recommend_solver(source;task=:steady_state,algorithm=method,
            krylovdim,memory_budget=experiment.memory_budget,
            T=_resource_scalar_type(source))
    else
        times=_saved_times(experiment.tspan,experiment.saveat;
            memory_budget=experiment.memory_budget)
        recommend_solver(source;task=:dynamics,algorithm=method,
            krylovdim,memory_budget=experiment.memory_budget,
            T=_resource_scalar_type(source,experiment.initial_state),
            time_type=eltype(times),samples=length(times),
            saved_states=experiment.save_states ? length(times) : 0,
            observable_series=_experiment_observable_count(
                experiment.observables))
    end
    exactness=_experiment_exactness(experiment)
    notes=String[
        recommendation.reason,
        exactness.complete_pi_basis ?
            "all PI Schur sectors are retained" :
            "the user explicitly declared a restricted Schur-sector model",
        "resource estimates do not include allocator metadata or vendor-library hidden buffers",
    ]
    ExperimentExecutionPlan(experiment.task,experiment.algorithm,
        recommendation.algorithm,recommendation.backend,recommendation,
        exactness,Tuple(notes))
end

"""Interactive alias for [`plan_experiment`](@ref)."""
explain_experiment(experiment::PIExperiment)=plan_experiment(experiment)

"""
    ExperimentProvenance

Immutable environment and model-identity record attached to every
[`ExperimentResult`](@ref). `structural_digest` is a deterministic
non-cryptographic checksum over supported model/specification fields. If
`digest_complete=false`, a callable or opaque user object was represented by
its type only; use experiment metadata to record a source revision or
parameterization identifier. The ambient `BigFloat` precision and rounding
mode are recorded explicitly, and `BigFloat` values contribute their own
stored precision to the digest.
"""
struct ExperimentProvenance{M}
    created_unix::Float64
    package_version::String
    julia_version::String
    kernel::Symbol
    architecture::Symbol
    threads::Int
    bigfloat_precision::Int
    bigfloat_rounding::String
    structural_digest::String
    digest_complete::Bool
    metadata::M
end

const _EXPERIMENT_FNV_OFFSET=UInt64(0xcbf29ce484222325)
const _EXPERIMENT_FNV_PRIME=UInt64(0x00000100000001b3)

@inline function _experiment_hash_byte(hash::UInt64,byte::UInt8)
    (hash⊻UInt64(byte))*_EXPERIMENT_FNV_PRIME
end

function _experiment_hash_text(hash::UInt64,text)
    for byte in codeunits(string(text))
        hash=_experiment_hash_byte(hash,byte)
    end
    _experiment_hash_byte(hash,0xff)
end

function _experiment_digest(hash::UInt64,specification::RefinementSpec)
    _experiment_digest(hash,(
        :RefinementSpec,
        specification.parameter,
        specification.levels,
        specification.atol,
        specification.rtol,
        specification.consecutive,
        specification.require_convergence))
end

function _experiment_digest(hash::UInt64,specification::VerificationSpec)
    _experiment_digest(hash,(
        :VerificationSpec,
        specification.atol,
        specification.rtol,
        specification.positivity_method,
        specification.dense_threshold,
        specification.require_physical,
        specification.require_solver_convergence,
        specification.refinement))
end

function _experiment_digest(hash::UInt64,value)
    if value isa BigFloat
        hash=_experiment_hash_text(hash,typeof(value))
        hash=_experiment_hash_text(hash,precision(value))
        return _experiment_hash_text(hash,string(value)),true
    elseif value isa Complex{BigFloat}
        hash=_experiment_hash_text(hash,typeof(value))
        hash,real_complete=_experiment_digest(hash,real(value))
        hash,imag_complete=_experiment_digest(hash,imag(value))
        return hash,real_complete&&imag_complete
    elseif value===nothing||value===missing||value isa Union{Number,Symbol,
            AbstractString,Bool,VersionNumber}
        return _experiment_hash_text(hash,repr(value)),true
    elseif value isa Partition
        return _experiment_digest(hash,value.parts)
    elseif value isa PIBasis
        hash=_experiment_hash_text(hash,typeof(value))
        hash=_experiment_hash_text(hash,value.N)
        hash=_experiment_hash_text(hash,value.d)
        return _experiment_digest(hash,value.sectors)
    elseif value isa AbstractPIOperator
        hash,complete=_experiment_digest(hash,value.basis)
        next_hash,next_complete=_experiment_digest(hash,value.data)
        return next_hash,complete&&next_complete
    elseif value isa AbstractRange
        hash=_experiment_hash_text(hash,typeof(value))
        hash,complete=_experiment_digest(hash,length(value))
        isempty(value)&&return hash,complete
        hash,first_complete=_experiment_digest(hash,first(value))
        hash,step_complete=_experiment_digest(hash,step(value))
        return hash,complete&&first_complete&&step_complete
    elseif value isa AbstractArray
        hash=_experiment_hash_text(hash,typeof(value))
        hash=_experiment_hash_text(hash,axes(value))
        complete=true
        for entry in value
            hash,entry_complete=_experiment_digest(hash,entry)
            complete&=entry_complete
        end
        return hash,complete
    elseif value isa NamedTuple
        hash=_experiment_hash_text(hash,keys(value))
        return _experiment_digest(hash,values(value))
    elseif value isa Tuple
        hash=_experiment_hash_text(hash,length(value))
        complete=true
        for entry in value
            hash,entry_complete=_experiment_digest(hash,entry)
            complete&=entry_complete
        end
        return hash,complete
    elseif value isa PIModel
        hash,complete=_experiment_digest(hash,value.basis)
        next_hash,next_complete=_experiment_digest(hash,value.terms)
        return next_hash,complete&&next_complete
    elseif value isa AbstractPITerm&&!(value isa _BuiltinPITerm)
        # Extension terms have no general ownership/snapshot contract.  Their
        # type remains useful provenance, but captured mutable state cannot be
        # certified by this structural digest.
        return _experiment_hash_text(hash,typeof(value)),false
    elseif value isa Function
        return _experiment_hash_text(hash,typeof(value)),false
    end
    fields=fieldnames(typeof(value))
    isempty(fields)&&return _experiment_hash_text(hash,typeof(value)),
        isbitstype(typeof(value))
    hash=_experiment_hash_text(hash,typeof(value))
    complete=true
    for name in fields
        hash=_experiment_hash_text(hash,name)
        field=getfield(value,name)
        if field isa Function
            hash=_experiment_hash_text(hash,typeof(field))
            complete=false
        elseif field isa Union{Number,Symbol,AbstractString,Bool,Nothing,
                AbstractArray,Tuple,NamedTuple,PIBasis,AbstractPIOperator}
            hash,field_complete=_experiment_digest(hash,field)
            complete&=field_complete
        else
            hash=_experiment_hash_text(hash,typeof(field))
            complete=false
        end
    end
    hash,complete
end

function _experiment_provenance(experiment::PIExperiment)
    bigfloat_precision=precision(BigFloat)
    bigfloat_rounding=string(rounding(BigFloat))
    hash,complete=_experiment_digest(_EXPERIMENT_FNV_OFFSET,
        (experiment.source,experiment.initial_state,experiment.task,
         experiment.algorithm,experiment.observables,experiment.tspan,
         experiment.saveat,experiment.steps_per_interval,
         experiment.parameters,experiment.save_states,
         experiment.memory_budget,experiment.representation,
         experiment.verification,experiment.solver_options,
         bigfloat_precision,bigfloat_rounding))
    package_version=try
        string(Base.pkgversion(@__MODULE__))
    catch
        "unknown"
    end
    ExperimentProvenance(time(),package_version,string(VERSION),Sys.KERNEL,
        Sys.ARCH,Threads.nthreads(),bigfloat_precision,bigfloat_rounding,
        string(hash;base=16,pad=16),complete,experiment.metadata)
end

"""
    ExperimentReport

Immutable verification summary. `verified=true` means every check requested
by the associated [`VerificationSpec`](@ref) passed. It does not turn a
single-resolution dynamics calculation into a discretization-converged one:
`refinement_converged` remains `missing` unless a refinement was requested.
"""
struct ExperimentReport{E,S,P,C,L}
    task::Symbol
    exactness::E
    solver_converged::S
    physical_valid::P
    refinement_converged::C
    verification_level::Symbol
    verified::Bool
    evidence::L
end

function Base.show(io::IO,report::ExperimentReport)
    print(io,"ExperimentReport(task=$(report.task), verified=$(report.verified), ",
        "level=$(report.verification_level), physical=$(report.physical_valid), ",
        "refinement=$(report.refinement_converged))")
end

"""Typed result of [`verified_solve`](@ref)."""
struct ExperimentResult{S,O,P,R,V}
    solution::S
    observables::O
    plan::P
    report::R
    provenance::V
end

function Base.show(io::IO,result::ExperimentResult)
    print(io,"ExperimentResult(")
    show(io,result.report)
    print(io,", digest=$(result.provenance.structural_digest))")
end

function _experiment_state_diagnostics(state::PIState,
        verification::VerificationSpec)
    state_diagnostics(state;atol=verification.atol,
        rtol=verification.rtol,
        positivity_method=verification.positivity_method,
        dense_threshold=verification.dense_threshold)
end

function _experiment_stationary_observables(state::PIState,observables)
    observables===nothing&&return NamedTuple()
    operations=_prepare_streaming_observables(
        state.basis,observables;require_hermitian=false)
    pairs=Pair{Any,Any}[]
    for (name,operator) in operations
        push!(pairs,name=>dot(operator.data,state.data))
    end
    Dict(pairs)
end

function _experiment_require_checks(verification,solver_converged,
        physical_valid,refinement_converged)
    verification.require_solver_convergence&&solver_converged!==true&&
        throw(ArgumentError(
            "experiment solver convergence was not established"))
    verification.require_physical&&physical_valid!==true&&throw(ArgumentError(
        "experiment output failed physical-state validation"))
    refinement=verification.refinement
    refinement!==nothing&&refinement.require_convergence&&
        refinement_converged!==true&&throw(ArgumentError(
            "experiment refinement convergence was not established"))
end

function _experiment_gmres_algorithm(algorithm,dimension::Int)
    if algorithm isa GMRESAlgorithm
        return GMRESAlgorithm(krylovdim=dimension,
            maxiter=algorithm.maxiter,preconditioner=algorithm.preconditioner)
    elseif algorithm isa Symbol&&
            _canonical_stationary_algorithm(algorithm)===:gmres
        return GMRESAlgorithm(krylovdim=dimension)
    end
    throw(ArgumentError(
        "krylov_dimension refinement requires an explicit GMRESAlgorithm(), " *
        "algorithm=:gmres, or algorithm=:krylov; it never changes a direct " *
        "or automatic request into GMRES silently"))
end

function _experiment_refinement_peak_bytes(experiment::PIExperiment)
    refinement=experiment.verification.refinement
    refinement===nothing&&return nothing
    accumulated_output=big(0)
    peak=big(0)
    reports=NamedTuple[]
    if experiment.task===:steady_state
        for raw_dimension in refinement.levels
            dimension=Int(raw_dimension)
            _experiment_gmres_algorithm(experiment.algorithm,dimension)
            report=recommend_solver(
                experiment.source;task=:steady_state,algorithm=:gmres,
                krylovdim=dimension,memory_budget=Inf,
                T=_resource_scalar_type(experiment.source))
            push!(reports,report)
            peak=max(peak,accumulated_output+report.known_peak_bytes)
            accumulated_output+=report.output_bytes
        end
    else
        method,krylovdim=_experiment_algorithm_request(experiment)
        times=_saved_times(
            experiment.tspan,experiment.saveat;memory_budget=Inf)
        for _ in refinement.levels
            report=recommend_solver(
                experiment.source;task=:dynamics,algorithm=method,
                krylovdim,memory_budget=Inf,
                T=_resource_scalar_type(
                    experiment.source,experiment.initial_state),
                time_type=eltype(times),samples=length(times),
                saved_states=experiment.save_states ? length(times) : 0,
                observable_series=_experiment_observable_count(
                    experiment.observables))
            push!(reports,report)
            peak=max(peak,accumulated_output+report.known_peak_bytes)
            accumulated_output+=report.output_bytes
        end
    end
    (;peak_bytes=peak,retained_output_bytes=accumulated_output,
      levels=refinement.levels,reports=Tuple(reports))
end

function _experiment_refinement_memory_preflight(experiment::PIExperiment)
    estimate=_experiment_refinement_peak_bytes(experiment)
    estimate===nothing&&return nothing
    _require_performance_budget(
        "experiment refinement cumulative retained output and solve peak",
        estimate.peak_bytes,experiment.memory_budget;
        guidance="Reduce saved output or refinement levels, or run an explicitly bounded refinement study.")
    estimate
end

function _experiment_steady_solve(experiment::PIExperiment)
    refinement=experiment.verification.refinement
    if refinement===nothing
        solution=stationary_state(experiment.source;
            algorithm=experiment.algorithm,
            memory_budget=experiment.memory_budget,return_info=true,
            experiment.solver_options...)
        return solution,nothing
    end
    evaluator=function (dimension)
        algorithm=_experiment_gmres_algorithm(
            experiment.algorithm,Int(dimension))
        stationary_state(experiment.source;algorithm,
            memory_budget=experiment.memory_budget,return_info=true,
            experiment.solver_options...)
    end
    study=krylov_dimension_convergence(evaluator,refinement.levels;
        estimate=result->result.state,diagnostics=result->result.info,
        atol=refinement.atol,rtol=refinement.rtol,
        consecutive=refinement.consecutive,
        require_convergence=refinement.require_convergence)
    last(study.results),study
end

function _experiment_dynamics_once(experiment::PIExperiment;
        steps_per_interval=experiment.steps_per_interval)
    solve_dynamics(experiment.source,experiment.initial_state,
        experiment.tspan;algorithm=experiment.algorithm,
        saveat=experiment.saveat,steps_per_interval,
        parameters=experiment.parameters,observables=experiment.observables,
        save_states=experiment.save_states,
        memory_budget=experiment.memory_budget)
end

function _experiment_dynamics_solve(experiment::PIExperiment)
    refinement=experiment.verification.refinement
    refinement===nothing&&return _experiment_dynamics_once(experiment),nothing
    evaluator=level->_experiment_dynamics_once(
        experiment;steps_per_interval=Int(level))
    study=convergence_study(evaluator,refinement.levels;
        parameter=:steps_per_interval,
        estimate=result->last(result.states),
        diagnostics=result->(algorithm=result.algorithm,),
        refinement_scale=level->inv(float(level)),
        atol=refinement.atol,rtol=refinement.rtol,
        consecutive=refinement.consecutive,
        require_convergence=refinement.require_convergence)
    last(study.results),study
end

function _experiment_dynamics_physical(solution,verification)
    solution.states===nothing&&return missing,NamedTuple()
    reports=map(state->_experiment_state_diagnostics(state,verification),
                solution.states)
    all_valid=all(report->report.valid,reports)
    all_valid,(states=reports,)
end

"""
    verified_solve(experiment)

Plan, run, and validate a deterministic `:steady_state` or `:dynamics`
experiment. The result retains the high-level solver output, observables, the
resource-selection plan, task-aware verification evidence, and provenance.

Steady states are checked using solver residual/trace evidence and physical
state diagnostics. Dynamics checks every retained state. A requested
`RefinementSpec(:steps_per_interval, ...)` additionally compares successive
final states. Without that explicit refinement, the report states
`refinement_converged=missing`; it does not claim time-discretization
convergence.
"""
function verified_solve(experiment::PIExperiment)
    plan=plan_experiment(experiment)
    _experiment_refinement_memory_preflight(experiment)
    provenance=_experiment_provenance(experiment)
    verification=experiment.verification
    if experiment.task===:steady_state
        solution,study=_experiment_steady_solve(experiment)
        solution.state isa PIState||throw(ArgumentError(
            "experiment steady-state verification currently supports PIState outputs"))
        physical=_experiment_state_diagnostics(solution.state,verification)
        solver_converged=hasproperty(solution.info,:converged) ?
            solution.info.converged : missing
        refinement_converged=study===nothing ? missing : study.converged
        _experiment_require_checks(verification,solver_converged,
            physical.valid,refinement_converged)
        level=study===nothing ? :solver_and_physical :
            :solver_physical_and_refinement
        refinement=verification.refinement
        verified=(!verification.require_solver_convergence||
            solver_converged===true)&&
            (!verification.require_physical||physical.valid)&&
            (refinement===nothing||!refinement.require_convergence||
             refinement_converged===true)
        evidence=(specification=verification,solver=solution.info,physical,
            refinement=study)
        report=ExperimentReport(:steady_state,plan.exactness,
            solver_converged,physical.valid,refinement_converged,
            level,verified,evidence)
        observables=_experiment_stationary_observables(
            solution.state,experiment.observables)
        return ExperimentResult(solution,observables,plan,report,provenance)
    end
    solution,study=_experiment_dynamics_solve(experiment)
    physical_valid,physical_evidence=
        _experiment_dynamics_physical(solution,verification)
    solver_converged=true
    refinement_converged=study===nothing ? missing : study.converged
    _experiment_require_checks(verification,solver_converged,
        physical_valid,refinement_converged)
    level=study===nothing ? (physical_valid===missing ?
        :single_resolution_observables : :single_resolution_physical) :
        :physical_and_timestep_refinement
    refinement=verification.refinement
    verified=(!verification.require_physical||physical_valid===true)&&
        (refinement===nothing||!refinement.require_convergence||
         refinement_converged===true)
    evidence=(specification=verification,physical=physical_evidence,
        refinement=study,
        algorithm=solution.algorithm)
    report=ExperimentReport(:dynamics,plan.exactness,solver_converged,
        physical_valid,refinement_converged,level,verified,evidence)
    observables=hasproperty(solution,:observables) ?
        solution.observables : NamedTuple()
    ExperimentResult(solution,observables,plan,report,provenance)
end

"""Current directory-archive schema written by [`save_experiment`](@ref)."""
const PI_EXPERIMENT_ARCHIVE_VERSION=UInt16(2)
const _PI_EXPERIMENT_MANIFEST_MAGIC=
    UInt8[0x50,0x49,0x44,0x45,0x58,0x50,0x52,0x31]

"""
    ExperimentArchive

Detached, portable payload returned by [`load_experiment`](@ref). It retains
saved PI states, times, numeric observable series, and string provenance/report
metadata. It intentionally does not deserialize executable models, closures,
workspaces, or solver factorizations.
"""
struct ExperimentArchive{T,S,O,M}
    schema_version::UInt16
    task::Symbol
    times::T
    states::S
    observables::O
    metadata::M
end

function Base.show(io::IO,archive::ExperimentArchive)
    print(io,"ExperimentArchive(task=$(archive.task), states=",
        length(archive.states),", observables=",length(archive.observables),
        ", schema=$(archive.schema_version))")
end

function _experiment_archive_evidence!(metadata,prefix,value,depth::Int=0)
    depth<=16||begin
        metadata[prefix*".truncated"]="maximum evidence nesting depth reached"
        return metadata
    end
    if value===missing||value===nothing||
            value isa Union{Number,Symbol,AbstractString,Bool,VersionNumber}
        metadata[prefix]=repr(value)
    elseif value isa Type||value isa Module
        metadata[prefix]=string(value)
    elseif value isa ConvergenceStudyResult
        metadata[prefix*".type"]="ConvergenceStudyResult"
        for name in (
                :parameter,:refinements,:diagnostics,:pairwise_errors,
                :tolerances,:pairwise_converged,:observed_rates,
                :solver_converged,:converged,:first_passing_index,
                :consecutive_required,:reason,:metadata)
            _experiment_archive_evidence!(
                metadata,prefix*"."*string(name),getproperty(value,name),
                depth+1)
        end
    elseif value isa NamedTuple
        metadata[prefix*".field_count"]=string(length(value))
        for (name,entry) in pairs(value)
            if name in (:state,:resource_preflight)
                metadata[prefix*"."*string(name)*".omitted_type"]=
                    string(typeof(entry))
                continue
            end
            _experiment_archive_evidence!(
                metadata,prefix*"."*string(name),entry,depth+1)
        end
    elseif value isa Tuple||value isa AbstractVector
        metadata[prefix*".count"]=string(length(value))
        for (index,entry) in pairs(value)
            _experiment_archive_evidence!(
                metadata,prefix*"."*string(index),entry,depth+1)
        end
    elseif value isa AbstractDict
        entries=sort!(collect(pairs(value));by=pair->string(first(pair)))
        metadata[prefix*".count"]=string(length(entries))
        for (key,entry) in entries
            _experiment_archive_evidence!(
                metadata,prefix*"."*string(key),entry,depth+1)
        end
    elseif value isa Union{PIState,PIOperator,AbstractArray,Function}
        metadata[prefix*".omitted_type"]=string(typeof(value))
    else
        names=fieldnames(typeof(value))
        metadata[prefix*".type"]=string(typeof(value))
        for name in names
            isdefined(value,name)||begin
                metadata[prefix*"."*string(name)*".undefined"]="true"
                continue
            end
            entry=getfield(value,name)
            entry isa Function&&begin
                metadata[prefix*"."*string(name)*".omitted_type"]=
                    string(typeof(entry))
                continue
            end
            _experiment_archive_evidence!(
                metadata,prefix*"."*string(name),entry,depth+1)
        end
    end
    metadata
end

function _experiment_archive_payload(result::ExperimentResult)
    solution=result.solution
    if result.report.task===:steady_state
        states=(solution.state,)
        times=()
    else
        states=solution.states===nothing ? () : solution.states
        times=solution.times
    end
    metadata=Dict{String,String}(
        "task"=>string(result.report.task),
        "verified"=>string(result.report.verified),
        "verification_level"=>string(result.report.verification_level),
        "representation"=>string(result.report.exactness.representation),
        "physical_approximation"=>
            string(result.report.exactness.physical_approximation),
        "requested_algorithm"=>string(result.plan.requested_algorithm),
        "selected_algorithm"=>string(result.plan.selected_algorithm),
        "backend"=>string(result.plan.backend),
        "budget_status"=>string(result.plan.recommendation.budget_status),
        "created_unix"=>repr(result.provenance.created_unix),
        "package_version"=>result.provenance.package_version,
        "julia_version"=>result.provenance.julia_version,
        "kernel"=>string(result.provenance.kernel),
        "architecture"=>string(result.provenance.architecture),
        "threads"=>string(result.provenance.threads),
        "bigfloat_precision"=>string(result.provenance.bigfloat_precision),
        "bigfloat_rounding"=>result.provenance.bigfloat_rounding,
        "structural_digest"=>result.provenance.structural_digest,
        "digest_complete"=>string(result.provenance.digest_complete),
        "verification.solver_converged"=>
            repr(result.report.solver_converged),
        "verification.physical_valid"=>repr(result.report.physical_valid),
        "verification.refinement_converged"=>
            repr(result.report.refinement_converged),
    )
    _experiment_archive_evidence!(
        metadata,"verification.evidence",result.report.evidence)
    observations=Dict{String,Any}()
    for (name,values) in pairs(result.observables)
        key=string(name)
        if values isa AbstractVector
            observations[key]=values
            metadata["observable."*key*".kind"]="series"
        elseif values isa Number
            observations[key]=[values]
            metadata["observable."*key*".kind"]="scalar"
        end
    end
    for (key,value) in result.provenance.metadata
        metadata["user."*key]=value
    end
    times,states,observations,metadata
end

function _experiment_write_manifest(path,task,state_count,
        observable_names,metadata)
    open(path,"w") do io
        write(io,_PI_EXPERIMENT_MANIFEST_MAGIC)
        write(io,PI_EXPERIMENT_ARCHIVE_VERSION)
        _write_checkpoint_string(io,string(task))
        write(io,Int64(state_count))
        write(io,Int32(length(observable_names)))
        for name in observable_names
            _write_checkpoint_string(io,name)
        end
        write(io,Int32(length(metadata)))
        for (key,value) in sort!(collect(metadata);by=first)
            _write_checkpoint_string(io,key)
            _write_checkpoint_string(io,value)
        end
    end
end

function _experiment_read_manifest(path)
    open(path,"r") do io
        read(io,length(_PI_EXPERIMENT_MANIFEST_MAGIC))==
            _PI_EXPERIMENT_MANIFEST_MAGIC||throw(ArgumentError(
                "file is not a PI experiment manifest"))
        version=read(io,UInt16)
        version in (UInt16(1),PI_EXPERIMENT_ARCHIVE_VERSION)||throw(ArgumentError(
            "unsupported PI experiment archive schema $version"))
        task=Symbol(_read_checkpoint_string(io))
        task in (:steady_state,:dynamics)||throw(ArgumentError(
            "invalid experiment archive task $task"))
        raw_state_count=read(io,Int64)
        0<=raw_state_count<=typemax(Int)||throw(ArgumentError(
            "invalid experiment archive state count"))
        state_count=Int(raw_state_count)
        raw_observable_count=read(io,Int32)
        raw_observable_count>=0||throw(ArgumentError(
            "invalid experiment archive observable count"))
        observable_count=Int(raw_observable_count)
        current=position(io)
        seekend(io);remaining=position(io)-current;seek(io,current)
        observable_count<=max(0,div(remaining-4,4))||throw(ArgumentError(
            "experiment archive observable count exceeds the manifest payload"))
        names=[_read_checkpoint_string(io) for _ in 1:observable_count]
        length(unique(names))==length(names)||throw(ArgumentError(
            "duplicate observable names in experiment archive"))
        raw_metadata_count=read(io,Int32)
        raw_metadata_count>=0||throw(ArgumentError(
            "invalid experiment archive metadata count"))
        metadata_count=Int(raw_metadata_count)
        current=position(io)
        seekend(io);remaining=position(io)-current;seek(io,current)
        metadata_count<=div(remaining,8)||throw(ArgumentError(
            "experiment archive metadata count exceeds the manifest payload"))
        metadata=Dict{String,String}()
        for _ in 1:metadata_count
            key=_read_checkpoint_string(io)
            haskey(metadata,key)&&throw(ArgumentError(
                "duplicate metadata key $key in experiment archive"))
            metadata[key]=_read_checkpoint_string(io)
        end
        eof(io)||throw(ArgumentError(
            "trailing data in PI experiment manifest"))
        (;version,task,state_count,names,metadata)
    end
end

function _experiment_vector_real_type(values)
    isempty(values)&&return Float64
    T=eltype(values)
    R=_real_float_type(T)
    R in (Float16,Float32,Float64,BigFloat)||throw(ArgumentError(
        "portable experiment arrays support Float16, Float32, Float64, or " *
        "BigFloat real components; got $R"))
    R
end

function _experiment_vector_precision(values,::Type{BigFloat})
    precision_value=0
    for value in values
        components=value isa Real ? (real(value),) : (real(value),imag(value))
        for component in components
            current=precision(component)
            if iszero(precision_value)
                precision_value=current
            elseif current!=precision_value
                throw(ArgumentError(
                    "portable experiment arrays require one BigFloat precision"))
            end
        end
    end
    iszero(precision_value) ? precision(BigFloat) : precision_value
end
_experiment_vector_precision(values,::Type)=0

function _experiment_save_vector(path,values;real_only::Bool=false)
    R=_experiment_vector_real_type(values)
    code=_checkpoint_scalar_code(R)
    precision_value=_experiment_vector_precision(values,R)
    open(path,"w") do io
        write(io,UInt8[0x50,0x49,0x44,0x56,0x45,0x43,0x31,0x00])
        write(io,code)
        write(io,UInt8(real_only ? 1 : 0))
        code==128&&write(io,Int32(precision_value))
        write(io,Int64(length(values)))
        for value in values
            _write_checkpoint_real(io,R(real(value)))
            real_only||_write_checkpoint_real(io,R(imag(value)))
        end
    end
end

function _experiment_vector_estimate(path)
    open(path,"r") do io
        read(io,8)==UInt8[0x50,0x49,0x44,0x56,0x45,0x43,0x31,0x00]||
            throw(ArgumentError("file is not a PI experiment vector"))
        code=read(io,UInt8)
        R=_checkpoint_scalar_type(code)
        real_flag=read(io,UInt8)
        real_flag in (0,1)||throw(ArgumentError(
            "invalid PI experiment vector kind"))
        precision_value=code==128 ? Int(read(io,Int32)) : 0
        code==128&&precision_value<=0&&throw(ArgumentError(
            "invalid PI experiment BigFloat precision"))
        raw_count=read(io,Int64)
        0<=raw_count<=typemax(Int)||throw(ArgumentError(
            "invalid PI experiment vector length"))
        count=Int(raw_count)
        components=real_flag==1 ? 1 : 2
        current=position(io)
        seekend(io);remaining=position(io)-current;seek(io,current)
        minimum_component_bytes=code==128 ? 4 : sizeof(R)
        minimum_payload=BigInt(count)*components*minimum_component_bytes
        minimum_payload<=remaining||throw(ArgumentError(
            "PI experiment vector length exceeds the file payload"))
        value_type=real_flag==1 ? R : Complex{R}
        retained_bytes=_performance_entries_bytes(
            count,value_type;
            bigfloat_precision=code==128 ? precision_value :
                precision(BigFloat))
        (;R,code,real_flag,precision_value,count,retained_bytes)
    end
end

function _experiment_load_vector(path;
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    estimate=_experiment_vector_estimate(path)
    _require_performance_budget(
        "PI experiment vector load",estimate.retained_bytes,memory_budget;
        guidance="Use a larger explicit load budget only after validating the archive source.")
    open(path,"r") do io
        read(io,8)==UInt8[0x50,0x49,0x44,0x56,0x45,0x43,0x31,0x00]||
            throw(ArgumentError("file is not a PI experiment vector"))
        code=read(io,UInt8)
        R=_checkpoint_scalar_type(code)
        real_flag=read(io,UInt8)
        real_flag in (0,1)||throw(ArgumentError(
            "invalid PI experiment vector kind"))
        precision_value=code==128 ? Int(read(io,Int32)) : 0
        code==128&&precision_value<=0&&throw(ArgumentError(
            "invalid PI experiment BigFloat precision"))
        raw_count=read(io,Int64)
        0<=raw_count<=typemax(Int)||throw(ArgumentError(
            "invalid PI experiment vector length"))
        count=Int(raw_count)
        (code,real_flag,precision_value,count)==
            (estimate.code,estimate.real_flag,estimate.precision_value,
             estimate.count)||throw(ArgumentError(
            "PI experiment vector changed during validated loading"))
        loader=function ()
            if real_flag==1
                values=Vector{R}(undef,count)
                for index in eachindex(values)
                    values[index]=_read_checkpoint_real(io,R)
                end
            else
                values=Vector{Complex{R}}(undef,count)
                for index in eachindex(values)
                    values[index]=complex(_read_checkpoint_real(io,R),
                        _read_checkpoint_real(io,R))
                end
            end
            eof(io)||throw(ArgumentError(
                "trailing data in PI experiment vector"))
            values
        end
        code==128 ? setprecision(BigFloat,precision_value) do
            loader()
        end : loader()
    end
end

function _experiment_checkpoint_estimate(path)
    open(path,"r") do io
        read(io,length(_PI_CHECKPOINT_MAGIC))==_PI_CHECKPOINT_MAGIC||
            throw(ArgumentError("file is not a PI checkpoint"))
        version=read(io,UInt16)
        version==PI_CHECKPOINT_VERSION||throw(ArgumentError(
            "unsupported PI checkpoint schema version $version"))
        code=read(io,UInt8)
        R=_checkpoint_scalar_type(code)
        precision_value=code==128 ? Int(read(io,Int32)) : 0
        code==128&&precision_value<=0&&throw(ArgumentError(
            "invalid BigFloat checkpoint precision $precision_value"))
        raw_N=read(io,Int64)
        raw_d=read(io,Int32)
        raw_sector_count=read(io,Int32)
        0<=raw_N<=typemax(Int)&&raw_d>=1&&raw_sector_count>=1||
            throw(ArgumentError("invalid PI checkpoint basis metadata"))
        d=Int(raw_d)
        sector_count=Int(raw_sector_count)
        sector_entries=BigInt(d)*sector_count
        current=position(io)
        seekend(io);remaining=position(io)-current;seek(io,current)
        8sector_entries+1<=remaining||throw(ArgumentError(
            "PI checkpoint sector metadata exceeds the file payload"))
        sector_entries<=typemax(Int)||throw(ArgumentError(
            "PI checkpoint sector metadata is too large for this platform"))
        seek(io,current+8Int(sector_entries))
        has_time=read(io,UInt8)
        has_time in (0,1)||throw(ArgumentError(
            "invalid PI checkpoint time flag"))
        if has_time==1
            if code==128
                _read_checkpoint_string(io)
            else
                current=position(io)
                seekend(io);remaining=position(io)-current;seek(io,current)
                sizeof(R)<=remaining||throw(ArgumentError(
                    "PI checkpoint time exceeds the file payload"))
                seek(io,current+sizeof(R))
            end
        end
        raw_coordinate_count=read(io,Int64)
        0<=raw_coordinate_count<=typemax(Int)||throw(ArgumentError(
            "invalid PI checkpoint coefficient length"))
        coordinate_count=Int(raw_coordinate_count)
        current=position(io)
        seekend(io);remaining=position(io)-current;seek(io,current)
        minimum_component_bytes=code==128 ? 4 : sizeof(R)
        minimum_payload=2BigInt(coordinate_count)*minimum_component_bytes
        minimum_payload<=remaining||throw(ArgumentError(
            "PI checkpoint coefficient length exceeds the file payload"))
        coefficient_bytes=_performance_entries_bytes(
            coordinate_count,Complex{R};
            bigfloat_precision=code==128 ? precision_value :
                precision(BigFloat))
        pattern_length=BigInt(d)*(d+1)÷2
        # A retained basis contains at most one stored GT pattern per
        # coordinate.  This deliberately conservative bound covers its tuple
        # entries and sector metadata before `PIBasis` is reconstructed.
        basis_bytes=BigInt(coordinate_count)*pattern_length*sizeof(Int)+
            sector_entries*sizeof(Int)
        (;coefficient_bytes,basis_bytes,
          peak_bytes=coefficient_bytes+basis_bytes)
    end
end

function _experiment_safe_filename(index)
    lpad(string(index),6,'0')
end

"""
    save_experiment(path, result)

Write a dependency-free, versioned experiment archive directory. Existing
paths are never overwritten. States use the portable `.pid` checkpoint
format; time and observable arrays preserve their supported floating
precision. The archive contains result/provenance evidence, not executable
Julia closures or a serialized solver workspace.
"""
function save_experiment(path,result::ExperimentResult)
    target=abspath(String(path))
    ispath(target)&&throw(ArgumentError(
        "experiment archive path already exists: $target"))
    parent=dirname(target)
    isdir(parent)||mkpath(parent)
    temporary=mktempdir(parent;prefix="."*basename(target)*".tmp-")
    success=false
    try
        times,states,observations,metadata=
            _experiment_archive_payload(result)
        _experiment_save_vector(joinpath(temporary,"times.pidvec"),
            times;real_only=true)
        for (index,state) in pairs(states)
            save_checkpoint(joinpath(temporary,
                "state_"*_experiment_safe_filename(index)*".pid"),state;
                metadata=("experiment_index"=>index,))
        end
        names=sort!(collect(keys(observations)))
        for (index,name) in pairs(names)
            _experiment_save_vector(joinpath(temporary,
                "observable_"*_experiment_safe_filename(index)*".pidvec"),
                observations[name])
        end
        _experiment_write_manifest(joinpath(temporary,"manifest.pidexp"),
            result.report.task,length(states),names,metadata)
        mv(temporary,target)
        success=true
        target
    finally
        !success&&isdir(temporary)&&rm(temporary;recursive=true,force=true)
    end
end

"""
    load_experiment(path; memory_budget=512MiB)

Load and validate a [`save_experiment`](@ref) archive into an
[`ExperimentArchive`](@ref). No model or code is executed while loading.
Loaded states remain unmodified; call `validate_state` or inspect the recorded
verification metadata when consuming an untrusted archive.
"""
function load_experiment(path;
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    root=abspath(String(path))
    isdir(root)||throw(ArgumentError(
        "experiment archive directory does not exist: $root"))
    _resource_memory_budget(memory_budget)
    manifest_path=joinpath(root,"manifest.pidexp")
    isfile(manifest_path)||throw(ArgumentError(
        "experiment archive manifest is missing"))
    _require_performance_budget(
        "PI experiment manifest load",filesize(manifest_path),memory_budget;
        guidance="Use a larger explicit load budget only after validating the archive source.")
    manifest=_experiment_read_manifest(manifest_path)
    available_files=Set(readdir(root))
    required_state_files=[
        "state_"*_experiment_safe_filename(index)*".pid"
        for index in 1:min(manifest.state_count,length(available_files)+1)]
    manifest.state_count<=length(available_files)&&
        all(name->name in available_files,required_state_files)||throw(
        ArgumentError("experiment archive is missing one or more state files"))
    required_observable_files=[
        "observable_"*_experiment_safe_filename(index)*".pidvec"
        for index in eachindex(manifest.names)]
    all(name->name in available_files,required_observable_files)||throw(
        ArgumentError("experiment archive is missing one or more observable files"))
    times_path=joinpath(root,"times.pidvec")
    isfile(times_path)||throw(ArgumentError(
        "experiment archive time vector is missing"))

    times_estimate=_experiment_vector_estimate(times_path)
    times_estimate.real_flag==1||throw(ArgumentError(
        "experiment archive times must use a real vector"))
    state_estimates=map(
        index->_experiment_checkpoint_estimate(joinpath(root,
            "state_"*_experiment_safe_filename(index)*".pid")),
        1:manifest.state_count)
    observable_estimates=map(
        index->_experiment_vector_estimate(joinpath(root,
            "observable_"*_experiment_safe_filename(index)*".pidvec")),
        eachindex(manifest.names))
    coefficient_bytes=sum(
        (estimate.coefficient_bytes for estimate in state_estimates);
        init=big(0))
    maximum_basis_bytes=maximum(
        (estimate.basis_bytes for estimate in state_estimates);
        init=big(0))
    vector_bytes=times_estimate.retained_bytes+
        sum((estimate.retained_bytes for estimate in observable_estimates);
            init=big(0))
    # Loaded states share one validated basis.  During reconstruction one
    # additional checkpoint basis can coexist transiently with it.
    estimated_peak=coefficient_bytes+vector_bytes+2maximum_basis_bytes
    _require_performance_budget(
        "PI experiment archive load",estimated_peak,memory_budget;
        guidance="Reduce retained histories or raise the explicit load budget after validating the archive.")

    times=_experiment_load_vector(times_path;memory_budget)
    all(isfinite,times)||throw(ArgumentError(
        "experiment archive times must be finite"))
    issorted(times)||throw(ArgumentError(
        "experiment archive times must be ordered"))
    states=PIState[]
    common_basis=nothing
    for index in 1:manifest.state_count
        checkpoint=load_checkpoint(joinpath(root,
            "state_"*_experiment_safe_filename(index)*".pid"))
        state=checkpoint_state(checkpoint)
        if common_basis===nothing
            common_basis=state.basis
        else
            state.basis.N==common_basis.N&&
                state.basis.d==common_basis.d&&
                state.basis.sectors==common_basis.sectors||throw(ArgumentError(
                "experiment archive states use inconsistent PI bases"))
            state=PIState(common_basis,state.data)
        end
        push!(states,state)
    end
    observations=Dict{String,Any}()
    for (index,name) in pairs(manifest.names)
        values=_experiment_load_vector(joinpath(root,
            "observable_"*_experiment_safe_filename(index)*".pidvec");
            memory_budget)
        default_kind=manifest.task===:steady_state ? "scalar" : "series"
        kind=get(manifest.metadata,"observable."*name*".kind",default_kind)
        kind in ("scalar","series")||throw(ArgumentError(
            "invalid observable kind $kind for $name in experiment archive"))
        if manifest.task===:steady_state
            kind=="scalar"&&length(values)==1||throw(DimensionMismatch(
                "steady-state experiment observable $name must contain one scalar"))
            observations[name]=only(values)
        else
            kind=="series"||throw(ArgumentError(
                "dynamics experiment observable $name must be a series"))
            length(values)==length(times)||throw(DimensionMismatch(
                "dynamics experiment observable $name and time counts differ"))
            observations[name]=values
        end
    end
    if manifest.task===:steady_state
        manifest.state_count==1||throw(DimensionMismatch(
            "steady-state experiment archive must contain exactly one state"))
        isempty(times)||throw(DimensionMismatch(
            "steady-state experiment archive must have an empty time vector"))
    else
        isempty(times)&&throw(DimensionMismatch(
            "dynamics experiment archive must contain at least one time"))
        !isempty(states)&&length(states)!=length(times)&&throw(
            DimensionMismatch(
                "dynamics experiment archive state and time counts differ"))
    end
    recorded_task=get(manifest.metadata,"task",string(manifest.task))
    recorded_task==string(manifest.task)||throw(ArgumentError(
        "experiment archive task metadata disagrees with its manifest"))
    ExperimentArchive(manifest.version,manifest.task,times,states,
        observations,manifest.metadata)
end
