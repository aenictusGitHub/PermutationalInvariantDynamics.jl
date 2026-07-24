"""Common supertype for dispatchable high-level solver choices."""
abstract type AbstractPIAlgorithm end

function _checked_algorithm_int(value::Integer,name::AbstractString;
        minimum::Integer=1)
    value isa Bool&&throw(ArgumentError("$name must be an integer, not a Bool"))
    value>=minimum||throw(ArgumentError(
        "$name must be at least $minimum"))
    BigInt(value)<=typemax(Int)||throw(ArgumentError(
        "$name must be representable as an Int"))
    Int(value)
end

"""Select a conservative algorithm from the problem representation and size."""
struct AutoAlgorithm <: AbstractPIAlgorithm end
"""Trace-bordered direct stationary-state solve."""
struct DirectAlgorithm <: AbstractPIAlgorithm end
"""Dense SVD stationary-manifold solve for diagnostic-sized problems."""
struct SVDAlgorithm <: AbstractPIAlgorithm end
"""Dense eigenvector stationary-state solve."""
struct EigenAlgorithm <: AbstractPIAlgorithm end

"""Sparse shift-invert stationary-state iteration near an explicit or automatic shift."""
struct ShiftInvertAlgorithm{T} <: AbstractPIAlgorithm
    shift::T
    maxiter::Int
end
ShiftInvertAlgorithm(;shift=nothing,maxiter::Integer=200)=
    ShiftInvertAlgorithm(shift,_checked_algorithm_int(maxiter,"maxiter"))

"""Restarted matrix-free GMRES stationary-state algorithm and optional preconditioner."""
struct GMRESAlgorithm{P} <: AbstractPIAlgorithm
    krylovdim::Int
    maxiter::Int
    preconditioner::P
end
GMRESAlgorithm(;krylovdim::Integer=30,maxiter::Integer=500,
               preconditioner=nothing)=
    GMRESAlgorithm(_checked_algorithm_int(krylovdim,"krylovdim"),
        _checked_algorithm_int(maxiter,"maxiter"),preconditioner)

"""
    RecycledGMRESAlgorithm(; krylovdim=30, maxiter=500,
                           recycle_dim=8, preconditioner=nothing)

Matrix-free GCRO-style stationary-state algorithm for continuation through a
sequence of slowly varying Liouvillians. A task-owned
[`RecycledGMRESWorkspace`](@ref) retains up to `recycle_dim` near-zero Ritz
directions between solves; every new operator is reapplied to the retained
space and the full unpreconditioned residual is checked.
"""
struct RecycledGMRESAlgorithm{P} <: AbstractPIAlgorithm
    krylovdim::Int
    maxiter::Int
    recycle_dim::Int
    preconditioner::P
end
function RecycledGMRESAlgorithm(;krylovdim::Integer=30,
        maxiter::Integer=500,recycle_dim::Integer=8,preconditioner=nothing)
    RecycledGMRESAlgorithm(
        _checked_algorithm_int(krylovdim,"krylovdim"),
        _checked_algorithm_int(maxiter,"maxiter"),
        _checked_algorithm_int(recycle_dim,"recycle_dim";minimum=0),
        preconditioner)
end

"""
    ExpvAlgorithm(; krylovdim=30, atol=1e-10, rtol=1e-8,
                  initial_step=nothing, minimum_step=nothing,
                  maximum_step=nothing, max_steps=10_000, safety=0.9)

Adaptive restarted-Arnoldi exponential action for autonomous high-level
dynamics. Each interval between requested output times is propagated as
`exp(Δt * L)ρ` without materializing either `L` or its exponential. The
workspace and task-owned source-action scratch are prepared once and reused
for every interval.

This algorithm deliberately rejects driven generators and non-`nothing`
`parameters`; use the RK4 or SciML paths for explicit time dependence.
"""
struct ExpvAlgorithm{A<:Real,R<:Real,I,M,X,S<:Real} <: AbstractPIAlgorithm
    krylovdim::Int
    atol::A
    rtol::R
    initial_step::I
    minimum_step::M
    maximum_step::X
    max_steps::Int
    safety::S
end
function _checked_expv_algorithm_control(
        value,label::AbstractString;allow_zero::Bool)
    value===nothing&&return nothing
    value isa Real&&!(value isa Bool)&&isfinite(value)||throw(ArgumentError(
        "$label must be nothing or a finite real number"))
    (allow_zero ? value>=0 : value>0)||throw(ArgumentError(
        "$label must be $(allow_zero ? "nonnegative" : "positive")"))
    value
end
function ExpvAlgorithm(;krylovdim::Integer=30,atol::Real=1e-10,
        rtol::Real=1e-8,initial_step=nothing,minimum_step=nothing,
        maximum_step=nothing,max_steps::Integer=10_000,safety::Real=0.9)
    !(atol isa Bool)&&!(rtol isa Bool)&&isfinite(atol)&&isfinite(rtol)&&
        atol>=0&&rtol>=0||throw(ArgumentError(
            "atol and rtol must be finite nonnegative real numbers"))
    !(safety isa Bool)&&isfinite(safety)&&0<safety<1||throw(ArgumentError(
        "safety must lie strictly between zero and one"))
    initial=_checked_expv_algorithm_control(
        initial_step,"initial_step";allow_zero=false)
    minimum=_checked_expv_algorithm_control(
        minimum_step,"minimum_step";allow_zero=true)
    maximum=_checked_expv_algorithm_control(
        maximum_step,"maximum_step";allow_zero=false)
    minimum===nothing||maximum===nothing||minimum<=maximum||
        throw(ArgumentError(
            "minimum_step must not exceed maximum_step"))
    minimum===nothing||initial===nothing||initial>=minimum||
        throw(ArgumentError(
            "initial_step must not be smaller than minimum_step"))
    ExpvAlgorithm(
        _checked_algorithm_int(krylovdim,"krylovdim"),atol,rtol,
        initial,minimum,maximum,
        _checked_algorithm_int(max_steps,"max_steps"),safety)
end

"""Thick-restarted harmonic Arnoldi parameters for modes near zero."""
struct HarmonicArnoldiAlgorithm <: AbstractPIAlgorithm
    nev::Int
    krylovdim::Int
    thickdim::Int
    maxrestarts::Int
end
function HarmonicArnoldiAlgorithm(;nev::Integer=6,krylovdim=nothing,
        thickdim=nothing,maxrestarts::Integer=20)
    requested=_checked_algorithm_int(nev,"nev")
    default_krylov=min(BigInt(typemax(Int)),
        max(BigInt(30),3BigInt(requested)+6))
    default_thick=min(BigInt(typemax(Int)),
        max(BigInt(requested)+2,2BigInt(requested)))
    krylovdim===nothing||krylovdim isa Integer||throw(ArgumentError(
        "krylovdim must be an integer or nothing"))
    thickdim===nothing||thickdim isa Integer||throw(ArgumentError(
        "thickdim must be an integer or nothing"))
    kdim=krylovdim===nothing ? Int(default_krylov) :
        _checked_algorithm_int(krylovdim,"krylovdim")
    thick=thickdim===nothing ? Int(default_thick) :
        _checked_algorithm_int(thickdim,"thickdim")
    restarts=_checked_algorithm_int(maxrestarts,"maxrestarts";minimum=0)
    HarmonicArnoldiAlgorithm(requested,kdim,thick,restarts)
end

"""Typed high-level stationary-state result."""
struct SteadyStateResult{S,I,A}
    state::S
    info::I
    algorithm::A
end

"""High-level dynamics result with collection semantics over saved states."""
struct DynamicsResult{T,S,A}
    times::Vector{T}
    states::S
    algorithm::A
end

"""
    DynamicsStreamResult

Memory-conscious high-level dynamics output returned by
[`solve_dynamics`](@ref) when `observables` are requested or
`save_states=false`. `observables` maps each user-supplied name to its sampled
expectation-value vector. `states` is either the ordinary saved `PIState`
vector or `nothing`.

When `states === nothing`, propagation owns one mutable state and observable
values are evaluated before that state is reused. Thus retained output storage
does not scale with the PI-coordinate dimension. Observable callbacks are not
used in this path, so user code cannot accidentally retain the integrator
state.
"""
struct DynamicsStreamResult{T,S,O,A}
    times::Vector{T}
    states::S
    observables::O
    algorithm::A
end
Base.length(sol::DynamicsStreamResult)=length(sol.times)
Base.firstindex(sol::DynamicsStreamResult)=firstindex(sol.times)
Base.lastindex(sol::DynamicsStreamResult)=lastindex(sol.times)
Base.getindex(sol::DynamicsStreamResult,i::Integer)=state(sol,i)
Base.iterate(sol::DynamicsStreamResult,args...)=begin
    sol.states===nothing&&throw(ArgumentError(
        "this result was created with save_states=false"))
    iterate(sol.states,args...)
end
state(sol::DynamicsStreamResult,i::Integer)=begin
    sol.states===nothing&&throw(ArgumentError(
        "this result was created with save_states=false"))
    sol.states[i]
end
state_at(sol::DynamicsStreamResult,t::Real)=
    state(sol,_saved_time_index(sol.times,t))
state(sol::DynamicsStreamResult,t::Real)=state_at(sol,t)
Base.length(sol::DynamicsResult)=length(sol.states)
Base.getindex(sol::DynamicsResult,i::Integer)=sol.states[i]
Base.firstindex(sol::DynamicsResult)=firstindex(sol.states)
Base.lastindex(sol::DynamicsResult)=lastindex(sol.states)
Base.iterate(sol::DynamicsResult,args...)=iterate(sol.states,args...)
state(sol::DynamicsResult,i::Integer)=sol.states[i]
state_at(sol::DynamicsResult,t::Real)=
    sol.states[_saved_time_index(sol.times,t)]
state(sol::DynamicsResult,t::Real)=state_at(sol,t)

"""Typed selected-spectrum result."""
struct SpectrumResult{V,W,I}
    values::V
    vectors::W
    info::I
end

Base.length(sol::PISolution)=length(sol.raw.u)
Base.getindex(sol::PISolution,i::Integer)=state(sol,i)
Base.firstindex(sol::PISolution)=firstindex(sol.raw.u)
Base.lastindex(sol::PISolution)=lastindex(sol.raw.u)
Base.iterate(sol::PISolution,state_index::Int=1)=
    state_index>length(sol) ? nothing : (state(sol,state_index),state_index+1)

function show(io::IO,rho::PIState)
    print(io,"PIState(N=$(rho.basis.N), d=$(rho.basis.d), dimension=$(length(rho.data)), trace=$(trace(rho)))")
end
function show(io::IO,A::PIOperator)
    print(io,"PIOperator(N=$(A.basis.N), d=$(A.basis.d), dimension=$(length(A.data)))")
end
function show(io::IO,model::PIModel)
    print(io,"PIModel(N=$(model.basis.N), d=$(model.basis.d), dimension=$(length(model.basis)), terms=$(length(model.terms)), autonomous=$(isautonomous(model)))")
end
function show(io::IO,prepared::CompiledPIModel)
    print(io,"CompiledPIModel(N=$(prepared.model.basis.N), d=$(prepared.model.basis.d), dimension=$(size(prepared,1)), backend=$(prepared.backend), autonomous=$(isautonomous(prepared)))")
end
function show(io::IO,family::CompiledPIModelFamily)
    print(io,"CompiledPIModelFamily(N=$(family.model.basis.N), d=$(family.model.basis.d), dimension=$(size(family.plan,1)), varied_rates=$(family.rate_indices))")
end
function show(io::IO,prepared::SpecializedPIModel)
    print(io,"SpecializedPIModel(N=$(prepared.model.basis.N), d=$(prepared.model.basis.d), dimension=$(size(prepared,1)), backend=$(prepared.backend), rates=$(prepared.rates))")
end
function show(io::IO,plan::LiouvillianPlan)
    print(io,"LiouvillianPlan(N=$(plan.basis.N), d=$(plan.basis.d), dimension=$(size(plan,1)), kernels=$(plan.kernels===nothing ? 0 : length(plan.kernels)), autonomous=$(isautonomous(plan)))")
end
function show(io::IO,L::MatrixFreeLiouvillian)
    print(io,"MatrixFreeLiouvillian(dimension=$(size(L,1)), autonomous=$(isautonomous(L)), compiled=$(L.plan!==nothing))")
end
function show(io::IO,result::SteadyStateResult)
    print(io,"SteadyStateResult(method=$(result.info.method), residual=$(result.info.residual), trace_error=$(result.info.trace_error))")
end
function show(io::IO,result::DynamicsResult)
    print(io,"DynamicsResult($(length(result)) states, t=$(first(result.times))…$(last(result.times)), algorithm=$(result.algorithm))")
end
function show(io::IO,result::DynamicsStreamResult)
    stored=result.states===nothing ? "observable-only" : "with states"
    print(io,"DynamicsStreamResult($(length(result.times)) samples, $stored, algorithm=$(result.algorithm))")
end
function show(io::IO,result::SpectrumResult)
    print(io,"SpectrumResult($(length(result.values)) values)")
end

function _algorithm_options(algorithm)
    algorithm isa Symbol && return (
        _canonical_stationary_algorithm(algorithm),NamedTuple())
    algorithm isa AutoAlgorithm && return (:auto,NamedTuple())
    algorithm isa DirectAlgorithm && return (:direct,NamedTuple())
    algorithm isa SVDAlgorithm && return (:svd,NamedTuple())
    algorithm isa EigenAlgorithm && return (:eigen,NamedTuple())
    algorithm isa ShiftInvertAlgorithm && return (:shiftinvert,
        (;shift=algorithm.shift,maxiter=algorithm.maxiter))
    algorithm isa GMRESAlgorithm && return (:gmres,
        (;krylovdim=algorithm.krylovdim,maxiter=algorithm.maxiter,
          preconditioner=algorithm.preconditioner))
    algorithm isa RecycledGMRESAlgorithm && return (:gmres,
        (;krylovdim=algorithm.krylovdim,maxiter=algorithm.maxiter,
          recycle_dim=algorithm.recycle_dim,
          preconditioner=algorithm.preconditioner))
    throw(ArgumentError("unsupported stationary-state algorithm $(typeof(algorithm))"))
end

function _dynamics_algorithm_options(algorithm)
    algorithm isa Symbol&&return (
        _canonical_dynamics_algorithm(algorithm),NamedTuple())
    algorithm isa AutoAlgorithm&&return (:auto,NamedTuple())
    algorithm isa ExpvAlgorithm&&return (:expv,(
        krylovdim=algorithm.krylovdim,
        atol=algorithm.atol,
        rtol=algorithm.rtol,
        initial_step=algorithm.initial_step,
        minimum_step=algorithm.minimum_step,
        maximum_step=algorithm.maximum_step,
        max_steps=algorithm.max_steps,
        safety=algorithm.safety))
    throw(ArgumentError(
        "unsupported dynamics algorithm $(typeof(algorithm)); use a dynamics algorithm symbol, AutoAlgorithm(), or ExpvAlgorithm()"))
end

function _basis_metadata(x,basis)
    basis!==nothing&&return basis
    _operator_basis(x)
end

"""
    stationary_state(x; algorithm=AutoAlgorithm(), basis=nothing,
                     memory_budget=512*1024^2,
                     return_info=false, kwargs...)

High-level stationary-state command. Model and compiled-model inputs return a
`PIState`; `return_info=true` returns a `SteadyStateResult`. The existing
`steady_state` function remains the low-level coordinate-vector interface.
Automatic selection preflights direct and matrix-free peaks. Explicit dense,
direct, SVD, eigen, and shift-invert routes throw before materialization when
their conservative structural peak exceeds `memory_budget`. Pass
`memory_budget=Inf` to opt out explicitly. The returned information includes
the complete `resource_preflight` report.
"""
function stationary_state(x;algorithm=AutoAlgorithm(),basis=nothing,
                          memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,
                          return_info::Bool=false,kwargs...)
    method,options=_algorithm_options(algorithm)
    b=_basis_metadata(x,basis)
    b===nothing&&throw(ArgumentError("stationary_state requires PI basis metadata"))
    requested_method=method
    report_algorithm=method
    report_krylovdim=haskey(options,:krylovdim) ? options.krylovdim :
        get(kwargs,:krylovdim,30)
    report_recycle_dim=haskey(options,:recycle_dim) ? options.recycle_dim :
        get(kwargs,:recycle_dim,0)
    report_shift=haskey(options,:shift) ? options.shift : get(kwargs,:shift,nothing)
    report_preconditioner=haskey(options,:preconditioner) ?
        options.preconditioner : get(kwargs,:preconditioner,nothing)
    report_type=_resource_scalar_type(x,get(kwargs,:initial_state,nothing),
        report_shift,report_preconditioner,get(kwargs,:workspace,nothing))
    preflight=recommend_solver(x;task=:steady_state,
        algorithm=report_algorithm,memory_budget,krylovdim=report_krylovdim,
        recycle_dim=report_recycle_dim,
        T=report_type)
    _enforce_memory_budget(preflight,"stationary_state")
    selected_algorithm=requested_method===:auto ? preflight.algorithm : method
    solver_method=_stationary_solver_method(selected_algorithm)
    info = if x isa Union{PIModel,CompiledPIModel,SpecializedPIModel}
        steady_state(x;method=solver_method,return_info=true,memory_budget,
                     options...,kwargs...)
    else
        steady_state(x;basis=b,method=solver_method,return_info=true,memory_budget,
                     options...,kwargs...)
    end
    info=merge(info,(;resource_preflight=preflight,
        requested_algorithm=requested_method,selected_algorithm))
    rho=PIState(b,info.state)
    result=SteadyStateResult(rho,info,algorithm)
    return_info ? result : rho
end

"""
    stationary_state(model::GlobalPseudomodeModel;
                     algorithm=AutoAlgorithm(),
                     memory_budget=512*1024^2,
                     return_info=false, kwargs...)

High-level stationary-state solve for a PI ensemble coupled to one shared
truncated pseudomode. The result is a [`CompositePIState`](@ref). Automatic
and iterative routes use the factorized matrix-free generator; no global
Kronecker superoperator is assembled. `AutoAlgorithm()` selects matrix-free
GMRES. Explicit choices are limited to [`GMRESAlgorithm`](@ref) and
[`RecycledGMRESAlgorithm`](@ref); dense, direct, and shift-invert algorithms
are rejected.
"""
function stationary_state(
        model::GlobalPseudomodeModel;
        algorithm=AutoAlgorithm(),
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,
        return_info::Bool=false,kwargs...)
    method,options=_algorithm_options(algorithm)
    method in (:auto,:gmres)||throw(ArgumentError(
        "global pseudomode stationary states support AutoAlgorithm or " *
        "GMRESAlgorithm/RecycledGMRESAlgorithm; direct, dense, and " *
        "shift-invert routes would materialize the composite superoperator"))
    haskey(options,:preconditioner)&&
        options.preconditioner===:schur&&throw(ArgumentError(
            "the PI Schur-sector preconditioner is not a preconditioner for " *
            "the PI-system × global-mode composite coordinate; pass a " *
            "compatible composite preconditioner or nothing"))
    _global_pseudomode_with_precision(model) do
        initial=get(kwargs,:initial_state,nothing)
        initial_vector=if initial isa CompositePIState
            initial.basis===model.basis||throw(ArgumentError(
                "initial composite state belongs to a different basis"))
            initial.data
        else
            initial
        end
        operator=global_pseudomode_matrixfree(
            model;memory_budget)
        report_krylovdim=haskey(options,:krylovdim) ?
            options.krylovdim : get(kwargs,:krylovdim,30)
        report_recycle_dim=haskey(options,:recycle_dim) ?
            options.recycle_dim : get(kwargs,:recycle_dim,0)
        preflight=recommend_solver(
            operator;task=:steady_state,algorithm=:gmres,
            memory_budget,krylovdim=report_krylovdim,
            recycle_dim=report_recycle_dim,
            T=_resource_scalar_type(operator,initial_vector),
            bigfloat_precision=model.precision_bits)
        _enforce_memory_budget(
            preflight,"global pseudomode stationary_state")
        selected=:gmres
        solver_method=_stationary_solver_method(selected)
        solve_kwargs=initial isa CompositePIState ?
            merge((;kwargs...),(;initial_state=initial_vector)) :
            (;kwargs...)
        info=steady_state(
            operator;trace_vector=model.trace_vector,
            method=solver_method,return_info=true,memory_budget,
            options...,solve_kwargs...)
        info=merge(info,(;resource_preflight=preflight,
            requested_algorithm=method,selected_algorithm=selected))
        rho=CompositePIState(model.basis,info.state)
        result=SteadyStateResult(rho,info,algorithm)
        return_info ? result : rho
    end
end

function _guard_saved_time_storage(count::Integer,::Type{T},memory_budget) where T
    estimate=_performance_entries_bytes(BigInt(count),T)
    _require_performance_budget("saved time grid",estimate,memory_budget;
        guidance="Request fewer saved times or stream a coarser grid.")
end

function _saved_times(tspan,saveat;
        memory_budget=Inf)
    t0,t1=tspan;t1>=t0||throw(ArgumentError("tspan must be ordered"))
    if saveat===nothing
        T=promote_type(typeof(float(t0)),typeof(float(t1)))
        _guard_saved_time_storage(2,T,memory_budget)
        return T[float(t0),float(t1)]
    end
    if saveat isa Real
        isfinite(saveat)&&saveat>0||throw(ArgumentError(
            "saveat must be finite and positive"))
        grid=float(t0):float(saveat):float(t1)
        append_endpoint=isempty(grid)||last(grid)<t1
        count=BigInt(length(grid))+(append_endpoint ? 1 : 0)
        _guard_saved_time_storage(count,eltype(grid),memory_budget)
        ts=collect(grid)
        (isempty(ts)||ts[end]<t1)&&push!(ts,float(t1))
        return ts
    end
    if applicable(length,saveat)
        count=length(saveat)
        source_type=try eltype(saveat) catch; Any end
        if source_type isa Type&&source_type<:Number&&isconcretetype(source_type)
            saved_type=typeof(float(zero(source_type)))
            _guard_saved_time_storage(count,saved_type,memory_budget)
        end
    end
    ts=float.(collect(saveat));isempty(ts)&&throw(ArgumentError("saveat cannot be empty"))
    first(ts)==t0&&last(ts)==t1||throw(ArgumentError("explicit saveat times must include both endpoints of tspan"))
    all(diff(ts).>=0)||throw(ArgumentError("saveat times must be nondecreasing"))
    ts
end

struct _HighLevelExpvOperator{T,S,W}
    source::S
    workspace::W
end
Base.size(operator::_HighLevelExpvOperator)=size(operator.source)
Base.size(operator::_HighLevelExpvOperator,index::Integer)=
    index in (1,2) ? size(operator.source,index) : 1
Base.eltype(::_HighLevelExpvOperator{T}) where T=T
function LinearAlgebra.mul!(destination::AbstractVector,
        operator::_HighLevelExpvOperator{T},
        input::AbstractVector) where T
    if operator.workspace===nothing
        mul!(destination,operator.source,input)
    else
        apply!(destination,operator.source,input,
            zero(_real_float_type(T)),nothing,operator.workspace)
    end
    destination
end
(operator::_HighLevelExpvOperator)(destination,input)=
    mul!(destination,operator,input)

function _highlevel_expv_setup(source,rho0::PIState,options)
    prepared=_evolution_liouvillian(source)
    source_basis=_operator_basis(prepared)
    source_basis===nothing||source_basis===rho0.basis||throw(ArgumentError(
        "Liouvillian source and initial state use incompatible PI bases"))
    n=length(rho0.data)
    size(prepared)==(n,n)||throw(DimensionMismatch(
        "Liouvillian and initial state dimensions differ"))
    T=_complex_float_type(_resource_scalar_type(prepared,rho0))
    _check_liouvillian_source_precision(
        prepared,T,"exponential-action state")
    current=eltype(rho0.data)===T ? copy(rho0) :
        PIState(rho0.basis,T.(rho0.data))
    action_workspace=_linear_operator_workspace(prepared)
    operator=_HighLevelExpvOperator{T,typeof(prepared),
        typeof(action_workspace)}(prepared,action_workspace)
    workspace=KrylovExpvWorkspace(
        T,n,get(options,:krylovdim,30))
    current,operator,workspace
end

function _highlevel_expv_interval!(current::PIState,operator,
        workspace::KrylovExpvWorkspace,interval,options)
    iszero(interval)&&return current
    interval>zero(interval)||throw(ArgumentError(
        "Krylov exponential output times must be nondecreasing"))
    krylov_expv!(current.data,operator,current.data,interval,workspace;
        atol=get(options,:atol,1e-10),
        rtol=get(options,:rtol,1e-8),
        initial_step=get(options,:initial_step,nothing),
        minimum_step=get(options,:minimum_step,nothing),
        maximum_step=get(options,:maximum_step,nothing),
        max_steps=get(options,:max_steps,10_000),
        safety=get(options,:safety,0.9),
        require_convergence=true)
    current
end

"""
    solve_dynamics(x, rho0, tspan; algorithm=:auto, saveat=nothing,
                   steps_per_interval=64, parameters=nothing,
                   observables=nothing, save_states=true,
                   memory_budget=512*1024^2)

Compile a model once when needed and propagate with the allocation-conscious
fixed-step RK4 path by default. `algorithm=:expv` (or
`:krylov_expv`) selects adaptive restarted-Arnoldi exponential action for an
autonomous generator. Use [`ExpvAlgorithm`](@ref) to set its Krylov dimension,
tolerances, and step controls. The result carries saved times and PI states and
supports indexing and iteration. Use `dynamics_problem` directly for general
adaptive SciML algorithms.

Pass a named tuple, dictionary, pair collection, or one local matrix/
`PIOperator` as `observables`. This returns a [`DynamicsStreamResult`](@ref).
With `save_states=false`, expectation values are accumulated while one mutable
state is propagated, and no sampled state history is constructed. A local
`d`-by-`d` matrix denotes its collective sum. Non-Hermitian observables are
accepted and retain complex expectation values. A state-free call without an
observable is rejected because it would return no dynamics output.

Before compiling a raw model, this command accounts for the matrix-free plan,
the selected RK4 or Krylov exponential workspace, task-owned source-action
scratch, saved states, sampled times, prepared observables, and scalar
observable series. It throws when the known peak exceeds `memory_budget`; use
`save_states=false` to stream output or `memory_budget=Inf` to opt out.
"""
function solve_dynamics(x,rho0::PIState,tspan;saveat=nothing,
                        algorithm=:auto,
                        steps_per_interval::Integer=64,parameters=nothing,
                        observables=nothing,save_states::Bool=true,
                        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    steps_per_interval>0||throw(ArgumentError("steps_per_interval must be positive"))
    requested_algorithm,algorithm_options=
        _dynamics_algorithm_options(algorithm)
    ts=_saved_times(tspan,saveat;memory_budget)
    named_observables=observables===nothing ? Pair[] :
        _named_observables(observables)
    observable_series=length(named_observables)
    state_type=_resource_scalar_type(x,rho0)
    observable_type=state_type
    for (_,observable) in named_observables
        candidate=_resource_value_scalar_type(observable)
        candidate===nothing||
            (observable_type=promote_type(observable_type,candidate))
    end
    report_krylovdim=get(algorithm_options,:krylovdim,30)
    preflight=recommend_solver(x;task=:dynamics,
        algorithm=requested_algorithm,
        krylovdim=report_krylovdim,
        memory_budget,T=state_type,observable_type,
        time_type=eltype(ts),samples=length(ts),
        saved_states=save_states ? length(ts) : 0,
        observable_series)
    _enforce_memory_budget(preflight,"solve_dynamics")
    source = x isa PIModel && isdefined(@__MODULE__,:compile) ?
        getfield(@__MODULE__,:compile)(x;backend=:matrixfree,
            memory_budget=memory_budget) : x
    selected_algorithm=preflight.algorithm
    if selected_algorithm===:expv
        parameters===nothing||throw(ArgumentError(
            "Krylov exponential dynamics requires parameters=nothing because it represents one autonomous generator"))
        applicable(isautonomous,source)||throw(ArgumentError(
            "Krylov exponential dynamics requires a source that explicitly declares whether it is autonomous"))
        isautonomous(source)||throw(ArgumentError(
            "Krylov exponential dynamics requires an autonomous generator; use RK4 or dynamics_problem for driven dynamics"))
        isempty(algorithm_options)&&(algorithm_options=
            last(_dynamics_algorithm_options(ExpvAlgorithm())))
    end
    _solve_dynamics_output(observables,source,rho0,ts;
        steps_per_interval,parameters,save_states,
        algorithm=selected_algorithm,algorithm_options)
end

function _solve_dynamics_output(::Nothing,source,rho0,ts;
                                steps_per_interval,parameters,save_states,
                                algorithm,algorithm_options)
    save_states||throw(ArgumentError(
        "save_states=false requires at least one observable"))
    if algorithm===:rk4
        states=time_evolution(source,rho0,ts;
            steps_per_interval=steps_per_interval,parameters=parameters)
        return DynamicsResult(ts,states,:rk4)
    end
    current,operator,workspace=_highlevel_expv_setup(
        source,rho0,algorithm_options)
    states=Vector{typeof(current)}(undef,length(ts))
    states[1]=copy(current)
    for time_index in 2:length(ts)
        _highlevel_expv_interval!(current,operator,workspace,
            ts[time_index]-ts[time_index-1],algorithm_options)
        states[time_index]=copy(current)
    end
    DynamicsResult(ts,states,:expv)
end

_dynamics_observable_buffers(::Tuple{},current,nsamples)=()
function _dynamics_observable_buffers(ops::Tuple{Any,Vararg},current,nsamples)
    op=last(first(ops));T=typeof(dot(op.data,current.data))
    (Vector{T}(undef,nsamples),
     _dynamics_observable_buffers(Base.tail(ops),current,nsamples)...)
end

_record_dynamics_observables!(::Tuple{},::Tuple{},current,index)=nothing
function _record_dynamics_observables!(buffers::Tuple{Any,Vararg},
                                       ops::Tuple{Any,Vararg},current,index)
    first(buffers)[index]=dot(last(first(ops)).data,current.data)
    _record_dynamics_observables!(Base.tail(buffers),Base.tail(ops),
                                  current,index)
end

function _dynamics_observable_dictionary(ops,buffers)
    values=Dict{Any,Any}()
    for index in eachindex(ops)
        values[first(ops[index])]=buffers[index]
    end
    values
end

function _solve_dynamics_output(observables,source,rho0,ts;
                                steps_per_interval,parameters,save_states,
                                algorithm,algorithm_options)
    ops=_prepare_streaming_observables(rho0.basis,observables;
                                       require_hermitian=false)
    prepared=_evolution_liouvillian(source)
    current,workspace,operator = if algorithm===:rk4
        state=copy(rho0)
        (state,EvolutionWorkspace(prepared,state),nothing)
    else
        state,expv_operator,expv_workspace=_highlevel_expv_setup(
            prepared,rho0,algorithm_options)
        (state,expv_workspace,expv_operator)
    end
    states=save_states ? Vector{typeof(current)}(undef,length(ts)) : nothing
    save_states&&(states[1]=copy(current))
    buffers=_dynamics_observable_buffers(ops,current,length(ts))
    _record_dynamics_observables!(buffers,ops,current,1)
    for time_index in 2:length(ts)
        if algorithm===:rk4
            ts[time_index]==ts[time_index-1]||evolve!(
                current,prepared,current,
                (ts[time_index-1],ts[time_index]);
                steps=steps_per_interval,parameters=parameters,
                workspace=workspace)
        else
            _highlevel_expv_interval!(current,operator,workspace,
                ts[time_index]-ts[time_index-1],algorithm_options)
        end
        save_states&&(states[time_index]=copy(current))
        _record_dynamics_observables!(buffers,ops,current,time_index)
    end
    values=_dynamics_observable_dictionary(ops,buffers)
    S=Union{Nothing,Vector{typeof(current)}}
    DynamicsStreamResult{eltype(ts),S,typeof(values),Symbol}(
        ts,states,values,algorithm)
end

function _spectrum_algorithm(algorithm,target,n,nev)
    if algorithm isa HarmonicArnoldiAlgorithm
        return (:harmonic,(;nev=algorithm.nev,krylovdim=algorithm.krylovdim,
            thickdim=algorithm.thickdim,maxrestarts=algorithm.maxrestarts))
    elseif algorithm isa Symbol && algorithm!==:auto
        return (_canonical_spectrum_algorithm(algorithm),(;nev=Int(nev)))
    elseif algorithm isa AutoAlgorithm || algorithm===:auto
        method=target===:near_zero ? :harmonic : n<=256 ? :dense : :arnoldi
        return (method,(;nev=Int(nev)))
    end
    throw(ArgumentError("unsupported spectrum algorithm $(typeof(algorithm))"))
end

"""
    liouvillian_spectrum(x; target=:largest_real, nev=6,
                         algorithm=:auto, vectors=false,
                         memory_budget=512*1024^2, return_info=false)

Consistent high-level spectral command. `target` is one of `:largest_real`,
`:near_zero`, or `:largest_magnitude`; method-specific `sortby`/`which`
dialects remain available through the lower-level spectral functions.
Automatic selection compares the complete dense-spectrum peak with the budget
before choosing dense or matrix-free Arnoldi. Explicit dense requests exceeding
the structural bound throw before materialization. `memory_budget=Inf` opts out.
An explicit `algorithm=:block_arnoldi` uses the thick-restarted batched solver;
its `block_size` is part of the resource preflight.
With `return_info=true`, solver metadata and `resource_preflight` are returned
without computing right eigenvectors unless `vectors=true`.
"""
function liouvillian_spectrum(x;target=:largest_real,nev::Integer=6,
                              algorithm=:auto,vectors::Bool=false,
                              memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,
                              return_info::Bool=false,kwargs...)
    target in (:largest_real,:near_zero,:largest_magnitude)||
        throw(ArgumentError("target must be :largest_real, :near_zero, or :largest_magnitude"))
    algorithm isa Union{Symbol,AutoAlgorithm,HarmonicArnoldiAlgorithm}||
        throw(ArgumentError(
        "unsupported spectrum algorithm $(typeof(algorithm))"))
    nev isa Bool&&throw(ArgumentError("nev must be an integer, not a Bool"))
    nev>0||throw(ArgumentError("nev must be positive"))
    source=x;n=pi_dimension(source)
    n>0||throw(ArgumentError("the spectral source must have positive dimension"))
    requested_nev=if algorithm isa HarmonicArnoldiAlgorithm
        algorithm.nev<=n||throw(ArgumentError(
            "HarmonicArnoldiAlgorithm.nev exceeds the source dimension"))
        algorithm.nev
    else
        Int(min(BigInt(n),BigInt(nev)))
    end
    auto_requested=algorithm isa AutoAlgorithm||algorithm===:auto
    report_algorithm = auto_requested ?
        (target===:near_zero ? :harmonic : :auto) :
        algorithm isa HarmonicArnoldiAlgorithm ? :harmonic :
        _canonical_spectrum_algorithm(algorithm)
    default_krylovdim=Int(min(BigInt(typemax(Int)),
        max(BigInt(20),2BigInt(requested_nev)+4)))
    report_krylovdim=algorithm isa HarmonicArnoldiAlgorithm ?
        algorithm.krylovdim : get(kwargs,:krylovdim,default_krylovdim)
    report_krylovdim=_checked_algorithm_int(
        report_krylovdim,"krylovdim")
    report_block_size=_checked_algorithm_int(
        get(kwargs,:block_size,min(requested_nev,4)),"block_size")
    report_maxrestarts=_checked_algorithm_int(
        get(kwargs,:maxrestarts,20),"maxrestarts";minimum=0)
    report_type=_resource_scalar_type(source,
        get(kwargs,:initial_vector,nothing),get(kwargs,:initial_subspace,nothing),
        get(kwargs,:operator_scale,nothing),get(kwargs,:shift,nothing))
    preflight=recommend_solver(source;task=:spectrum,
        algorithm=report_algorithm,memory_budget,krylovdim=report_krylovdim,
        nev=requested_nev,block_size=report_block_size,
        maxrestarts=report_maxrestarts,vectors,T=report_type)
    _enforce_memory_budget(preflight,"liouvillian_spectrum")
    selected_algorithm=auto_requested ? preflight.algorithm : report_algorithm
    method,options=_spectrum_algorithm(
        selected_algorithm,target,n,requested_nev)
    method===:jd&&target!==:near_zero&&throw(ArgumentError(
        "Jacobi--Davidson is a near-target solver; use target=:near_zero or call jacobi_davidson_spectrum with a numeric target"))
    sortby=target===:largest_real ? :real : :magnitude
    rev=target!==:near_zero
    default_options=haskey(options,:krylovdim)||haskey(kwargs,:krylovdim) ?
        NamedTuple() : (;krylovdim=report_krylovdim)
    raw=pi_liouvillian_spectrum(source;method=method,sortby=sortby,rev=rev,
                                vectors,return_info,memory_budget,
                                default_options...,options...,kwargs...)
    if !vectors&&!return_info
        return raw[1:min(requested_nev,length(raw))]
    end
    take=1:min(requested_nev,length(raw.values))
    raw_vectors=hasproperty(raw,:vectors) ? raw.vectors : nothing
    vectors&&raw_vectors===nothing&&throw(ArgumentError(
        "the selected spectral solver did not return requested eigenvectors"))
    values=raw.values[take];vecs=vectors ? raw_vectors[:,take] : nothing
    stripped=hasproperty(raw,:vectors) ?
        Base.structdiff(raw,(values=raw.values,vectors=raw_vectors)) :
        Base.structdiff(raw,(values=raw.values,))
    info=merge(stripped,
        (;resource_preflight=preflight,requested_algorithm=algorithm,
          selected_algorithm=method))
    result=SpectrumResult(values,vecs,info)
    return_info ? result : (vectors ? (values=values,vectors=vecs) : values)
end

"""
    liouvillian_spectrum(model::GlobalPseudomodeModel; kwargs...)

Compute selected modes of a shared-pseudomode Liouvillian through its
factorized matrix-free wrapper. Automatic selection uses Arnoldi (or harmonic
Arnoldi for `target=:near_zero`); materializing dense-spectrum algorithms are
rejected by the ordinary resource preflight.
"""
function liouvillian_spectrum(
        model::GlobalPseudomodeModel;
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,kwargs...)
    _global_pseudomode_with_precision(model) do
        operator=global_pseudomode_matrixfree(
            model;memory_budget)
        liouvillian_spectrum(
            operator;memory_budget,kwargs...)
    end
end

"""Return the PI-coordinate dimension of a basis, state, model, or operator."""
pi_dimension(b::PIBasis)=length(b)
pi_dimension(x::AbstractPIOperator)=length(x.data)
pi_dimension(x::PIModel)=length(x.basis)
pi_dimension(x)=size(x,1)

"""
    estimate_state_bytes(x; T=ComplexF64,
                         bigfloat_precision=precision(BigFloat))

Estimated retained bytes occupied by one dense PI coordinate vector. Fixed-size
isbits scalar types use the exact inline `sizeof(T)` value. Heap-backed
`BigFloat` scalars use an explicitly conservative per-element bound at
`bigfloat_precision`; pass the largest intended precision when it differs from
the active process precision. Other heap-backed scalar types use a padded
zero-value sample because their payload may not be bounded by their type; this
route is an estimate, not a worst-case guarantee.
"""
estimate_state_bytes(b::PIBasis;T=ComplexF64,
                     bigfloat_precision::Integer=precision(BigFloat))=
    big(length(b))*_scalar_retained_bytes(T;bigfloat_precision)
estimate_state_bytes(x;T=ComplexF64,
                     bigfloat_precision::Integer=precision(BigFloat))=
    big(pi_dimension(x))*_scalar_retained_bytes(T;bigfloat_precision)

"""Retained Julia heap size of an already constructed basis or plan."""
estimate_basis_bytes(b::PIBasis)=Base.summarysize(b)
"""Return the retained Julia heap size in bytes of an assembled or matrix-free Liouvillian object."""
estimate_liouvillian_bytes(L)=Base.summarysize(L)

"""
    estimate_geometry_bytes(basis; T=Float64,
                            bigfloat_precision=precision(BigFloat))

Return a conservative structural memory estimate for construction and
retention of the shared one-body Schur geometry.  The estimate performs no CG
evaluation and reports exact `BigInt` byte counts.  `setup_bytes` is a peak
live-storage upper bound for the sparse transition constructor, whereas
`retained_bytes` describes the resulting read-only cache.  Allocator metadata
and garbage-collector timing remain platform dependent, so benchmark a
representative basis before a long scan. Fixed-size isbits geometry preserves
the historical byte formula; heap-backed BigFloat tuple entries are inflated
with the shared conservative bound at `bigfloat_precision`.
"""
function estimate_geometry_bytes(b::PIBasis;T=Float64,
                                 bigfloat_precision::Integer=precision(BigFloat))
    R=_real_float_type(T)
    R<:AbstractFloat||throw(ArgumentError(
        "geometry scalar type must promote to an AbstractFloat type"))
    raw=_estimate_onebody_geometry(b,R;bigfloat_precision)
    retained_scalar=_scalar_retained_bytes(R;bigfloat_precision)
    merge(raw,(;scalar_type=R,scalar_retained_bytes=retained_scalar,
               scalar_storage_estimate=_scalar_storage_estimate(R),
               bigfloat_precision_assumption=
                   _scalar_precision_assumption(R,bigfloat_precision),
               estimate=:conservative_structural_upper_bound))
end

function _estimate_diagonal_onebody_geometry(b::PIBasis;T=Float64,
        bigfloat_precision::Integer=precision(BigFloat))
    R=_real_float_type(T)
    R<:AbstractFloat||throw(ArgumentError(
        "geometry scalar type must promote to an AbstractFloat type"))
    raw=_estimate_onebody_geometry(
        b,R;diagonal_only=true,bigfloat_precision)
    retained_scalar=_scalar_retained_bytes(R;bigfloat_precision)
    merge(raw,(;scalar_type=R,scalar_retained_bytes=retained_scalar,
               scalar_storage_estimate=_scalar_storage_estimate(R),
               bigfloat_precision_assumption=
                   _scalar_precision_assumption(R,bigfloat_precision),
               estimate=:conservative_structural_upper_bound))
end

function _sparse_csc_structural_upper_bytes(n::Integer,::Type{T};
        bigfloat_precision::Integer=precision(BigFloat)) where T
    entries=big(n)^2
    entries*_scalar_retained_bytes(T;bigfloat_precision)+
        (entries+big(n)+1)*sizeof(Int)
end

_dense_matrix_structural_bytes(n::Integer,::Type{T};
        bigfloat_precision::Integer=precision(BigFloat)) where T =
    big(n)^2*_scalar_retained_bytes(T;bigfloat_precision)

function _solver_direct_upper_bytes(n::Integer,::Type{T};
        bigfloat_precision::Integer=precision(BigFloat)) where T
    # The bordered system has dimension n+1.  Six dense-pattern CSC arrays
    # conservatively cover the bordered copy, symbolic/numeric factors, and
    # factorization work arrays.  This is deliberately much larger than the
    # common sparse case, but unlike an nnz guess it remains useful before
    # assembly.
    six=6*_sparse_csc_structural_upper_bytes(n+1,T;bigfloat_precision)
    six+4big(n)*_scalar_retained_bytes(T;bigfloat_precision)
end

function _solver_shiftinvert_upper_bytes(n::Integer,::Type{T};
        bigfloat_precision::Integer=precision(BigFloat)) where T
    5*_sparse_csc_structural_upper_bytes(n,T;bigfloat_precision)+
        5big(n)*_scalar_retained_bytes(T;bigfloat_precision)
end

function _solver_dense_upper_bytes(n::Integer,::Type{T},copies::Integer;
        bigfloat_precision::Integer=precision(BigFloat)) where T
    copies*_dense_matrix_structural_bytes(n,T;bigfloat_precision)+
        4big(n)*_scalar_retained_bytes(T;bigfloat_precision)
end

"""
    estimate_solver_bytes(x; algorithm=:gmres, krylovdim=30, recycle_dim=0,
                          block_size=4, T=ComplexF64,
                          bigfloat_precision=precision(BigFloat))

Estimate solver work-array storage, excluding the retained input operator.
Fixed-size isbits values retain exact inline byte accounting; heap-backed
BigFloat values use the same conservative precision-aware bound as
[`estimate_state_bytes`](@ref). GMRES real residual/history storage follows the
real component type of `T` rather than assuming `Float64`. Matrix-free Krylov
and RK4 formulas count their explicit arrays. Direct, shift-invert, dense
eigenvalue, and SVD routes instead return conservative structural upper bounds
for solver-owned matrix/factor arrays; allocator metadata and vendor-library
internal buffers remain platform dependent.
"""
function estimate_solver_bytes(x;algorithm=:gmres,krylovdim::Integer=30,
                               recycle_dim::Integer=0,
                               block_size::Integer=4,
                               T=ComplexF64,
                               bigfloat_precision::Integer=precision(BigFloat))
    krylovdim>0||throw(ArgumentError("krylovdim must be positive"))
    recycle_dim>=0||throw(ArgumentError("recycle_dim must be nonnegative"))
    block_size>0||throw(ArgumentError("block_size must be positive"))
    ni=pi_dimension(x);mi=Int(min(big(ni),big(krylovdim)));n=big(ni)
    k=big(min(recycle_dim,max(ni-1,0)))
    scalar_bytes=_scalar_retained_bytes(T;bigfloat_precision)
    algorithm in (:gmres,:krylov)&&return _performance_gmres_bytes(
        ni,T,mi;recycle_dim=Int(k),bigfloat_precision)
    algorithm in (:arnoldi,:ordinary_arnoldi)&&return _performance_arnoldi_bytes(
        ni,T,mi;mode=:ordinary,bigfloat_precision)
    if algorithm in (:block_arnoldi,:block)
        return _performance_block_arnoldi_bytes(
            ni,T,mi,min(mi,Int(min(BigInt(ni),BigInt(block_size))));
            bigfloat_precision)
    end
    if algorithm in (:harmonic,:iram,:implicit_qr)
        return _performance_arnoldi_bytes(
            ni,T,mi;mode=:full,bigfloat_precision)
    end
    # A conservative JD estimate uses a same-dimension correction GMRES
    # (the allocating solver defaults to at most 20) plus its six explicit
    # full-coordinate correction/locking vectors.
    if algorithm in (:jd,:jacobi_davidson)
        return _performance_arnoldi_bytes(
            ni,T,mi;mode=:full,bigfloat_precision)+
            _performance_gmres_bytes(ni,T,mi;bigfloat_precision)+
            _performance_entries_bytes(6n,T;bigfloat_precision)
    end
    algorithm===:direct&&return _solver_direct_upper_bytes(ni,T;
        bigfloat_precision)
    if algorithm in (:shiftinvert,:shift_invert,:inverse_iteration)
        return _solver_shiftinvert_upper_bytes(ni,T;bigfloat_precision)
    end
    algorithm in (:dense,:eigen)&&return _solver_dense_upper_bytes(ni,T,8;
        bigfloat_precision)
    algorithm===:svd&&return _solver_dense_upper_bytes(ni,T,10;
        bigfloat_precision)
    algorithm in (:expv,:krylov_expv)&&return _performance_krylov_expv_workspace_bytes(
            ni,T,mi;bigfloat_precision)
    algorithm in (:rk4,:dynamics)&&return 3scalar_bytes*n
    throw(ArgumentError("unknown solver-memory algorithm $algorithm"))
end

function _resource_memory_budget(memory_budget)
    memory_budget isa Real&&!(memory_budget isa Bool)||throw(ArgumentError(
        "memory_budget must be a nonnegative number of bytes or Inf"))
    isnan(memory_budget)&&throw(ArgumentError("memory_budget cannot be NaN"))
    memory_budget>=0||throw(ArgumentError("memory_budget must be nonnegative"))
    isinf(memory_budget)&&return (;bytes=nothing,disabled=true)
    (;bytes=floor(BigInt,memory_budget),disabled=false)
end

function _resource_component(bytes,provenance::Symbol;includes=(),excludes=())
    provenance in (:actual,:upper_bound,:estimate,:unknown)||throw(ArgumentError(
        "invalid resource-estimate provenance $provenance"))
    converted=bytes===nothing ? nothing : big(bytes)
    converted===nothing||converted>=0||throw(ArgumentError(
        "resource byte estimates must be nonnegative"))
    (;bytes=converted,provenance,includes=Tuple(includes),excludes=Tuple(excludes))
end


function _resource_peak(setup,retained,solve,output)
    named=((:setup,setup),(:retained,retained),(:solve,solve),(:output,output))
    unknown_components=Symbol[name for (name,component) in named
        if component.provenance in (:estimate,:unknown)]
    missing_bytes=any(component.bytes===nothing for (_,component) in named)
    retained_bytes=retained.bytes===nothing ? big(0) : retained.bytes
    setup_known=retained_bytes+(setup.bytes===nothing ? big(0) : setup.bytes)
    solve_known=retained_bytes+
        (solve.bytes===nothing ? big(0) : solve.bytes)+
        (output.bytes===nothing ? big(0) : output.bytes)
    known_peak=max(setup_known,solve_known)
    provenance = any(component.provenance===:unknown for (_,component) in named) ?
        :unknown : any(component.provenance===:estimate for (_,component) in named) ?
        :estimate : any(component.provenance===:upper_bound for (_,component) in named) ?
        :upper_bound : :actual
    bytes=missing_bytes ? nothing : known_peak
    peak=_resource_component(bytes,provenance;
        includes=(:retained_storage,:setup_or_solve_workspace,:retained_output),
        excludes=(:allocator_metadata,:garbage_collector_timing,
                  :vendor_library_hidden_buffers))
    (;peak,known_peak_bytes=known_peak,unknown_components=Tuple(unknown_components))
end

function _resource_budget_status(peak,known_peak_bytes,budget)
    budget.disabled&&return (:disabled,missing)
    known_peak_bytes>budget.bytes&&return (:exceeds,false)
    peak.provenance in (:actual,:upper_bound)&&peak.bytes!==nothing&&
        return (:fits,isempty(peak.excludes) ? true : missing)
    (:unknown,missing)
end

function _enforce_memory_budget(report,operation::AbstractString)
    report.budget_status===:exceeds||return report
    throw(ArgumentError("$operation preflight estimates a peak of " *
        "$(report.known_peak_bytes) bytes, exceeding memory_budget=" *
        "$(report.memory_budget) bytes. Select a matrix-free/streaming route, " *
        "reduce the Krylov dimension or saved output, raise memory_budget, or " *
        "pass memory_budget=Inf to opt out explicitly."))
end

_resource_source_prepared(x)=x isa Union{
    CompiledPIModel,SpecializedPIModel,LiouvillianPlan,
    MatrixFreeLiouvillian,CompositeSuperoperator,
    GlobalPseudomodeModel,AbstractMatrix}

function _resource_source_has_sparse_operator(x)
    x isa SparseMatrixCSC&&return true
    x isa Union{CompiledPIModel,SpecializedPIModel}&&return x.backend===:sparse
    false
end

_resource_sparse_operator_actual_bytes(::Any)=nothing
_resource_sparse_operator_actual_bytes(matrix::SparseMatrixCSC)=
    big(Base.summarysize(matrix))
function _resource_sparse_operator_actual_bytes(model::CompiledPIModel)
    model.backend===:sparse ? big(Base.summarysize(model.operator)) : nothing
end
function _resource_sparse_operator_actual_bytes(model::SpecializedPIModel)
    model.backend===:sparse ? big(Base.summarysize(model.operator)) : nothing
end

function _resource_base_scalar_type(x)
    if x isa PIModel
        T=Complex{_model_geometry_type(x)}
        for term in x.terms
            rate=try term_rate(term) catch; nothing end
            rate isa Number&&(T=promote_type(T,typeof(rate)))
        end
        return T
    end
    T=try eltype(x) catch; ComplexF64 end
    T===Any ? ComplexF64 : T
end

function _resource_value_scalar_type(value)
    value===nothing&&return nothing
    value isa Symbol&&return nothing
    value isa PIState&&return eltype(value.data)
    value isa Number&&return typeof(value)
    value isa AbstractArray&&return eltype(value)
    if hasproperty(value,:V)
        V=getproperty(value,:V)
        V isa AbstractArray&&return eltype(V)
    end
    T=try eltype(value) catch; Any end
    T===Any ? nothing : T
end

function _resource_scalar_type(x,values...)
    T=_resource_base_scalar_type(x)
    for value in values
        V=_resource_value_scalar_type(value)
        V===nothing||(T=promote_type(T,V))
    end
    T
end

function _recommended_geometry_policy(x,basis)
    basis===nothing&&return (include=false,requirement=:unavailable,
                             source=:no_basis_metadata,kind=:none)
    model=x isa PIModel ? x :
          x isa Union{CompiledPIModel,SpecializedPIModel} ? x.model : nothing
    if model!==nothing
        requirements=_model_onebox_requirements(
            model,_model_geometry_type(model))
        if requirements.needs_full_onebody
            return (include=true,requirement=:required,source=:model_terms,
                    kind=isempty(requirements.pbody_orders) ?
                        :onebody : :onebody_and_pbody)
        elseif requirements.uses_symmetric_collective
            return (include=true,requirement=:required,source=:model_terms,
                    kind=isempty(requirements.pbody_orders) ?
                        :symmetric_collective :
                        :symmetric_collective_and_pbody)
        elseif requirements.uses_diagonal_onebody
            return (include=true,requirement=:required,source=:model_terms,
                    kind=isempty(requirements.pbody_orders) ?
                        :diagonal_onebody : :diagonal_onebody_and_pbody)
        elseif !isempty(requirements.pbody_orders)
            return (include=true,requirement=:required,source=:model_terms,
                    kind=:pbody)
        end
        return (include=false,requirement=:not_required,source=:model_terms,
                kind=:none)
    end
    # A bare basis, state, operator, or lowered plan no longer carries enough
    # term provenance to distinguish local one-body lowering from direct/p-body
    # blocks. Retain the conservative historical geometry allowance and make
    # that assumption explicit in the returned metadata.
    (include=true,requirement=:conservative_unknown,
     source=x isa PIBasis ? :basis_only : :source_without_term_provenance,
     kind=:onebody)
end

_resource_prepared_sparse_plan(::Any)=nothing
_resource_prepared_sparse_plan(plan::LiouvillianPlan)=plan
_resource_prepared_sparse_plan(model::CompiledPIModel)=model.plan
_resource_prepared_sparse_plan(model::SpecializedPIModel)=model.plan
_resource_prepared_sparse_plan(operator::MatrixFreeLiouvillian)=
    operator.plan isa LiouvillianPlan ? operator.plan : nothing

"""
    recommend_solver(x; task=:steady_state, algorithm=:auto,
                     memory_budget=512*1024^2, krylovdim=30, recycle_dim=0,
                     nev=6, block_size=min(nev, 4), vectors=false,
                     maxrestarts=20, samples=1, saved_states=1,
                     observable_series=0, workers=1,
                     bigfloat_precision=precision(BigFloat))

Return an assembly-free solver and resource preflight. The result preserves the
historical flat byte fields and additionally provides
`resources=(setup, retained, solve, output, peak)`. Every component records its
`bytes`, `provenance` (`:actual`, `:upper_bound`, `:estimate`, or `:unknown`),
and explicit inclusions/exclusions. `known_peak_bytes` is the largest known
setup/solve phase total; `budget_status` is `:fits`, `:exceeds`, `:unknown`, or
`:disabled`, and `safe_to_run` is deliberately `missing` unless a measured or
structural upper bound supports a safety claim.

For dynamics, `samples` is the number of retained time points,
`saved_states` is the number of those points whose PI state is retained, and
`observable_series` counts streamed scalar series. `workers` models concurrent
task-owned workspaces with one shared prepared source. `memory_budget=Inf`
explicitly disables budget enforcement. Model and `CompiledPIModel` inputs
include one-body geometry only when their terms require it. Inputs carrying
only basis metadata retain a conservative geometry allowance, identified by
`geometry_requirement=:conservative_unknown`.

Coordinate precision is inferred from `x` unless `T` is supplied. For a
dynamics-only estimate, `observable_type` and `time_type` can describe wider
prepared observables or saved-time vectors without incorrectly widening the
state and selected RK4/Krylov-exponential workspace. Dense compatibility
solvers account for their actual or conservatively bounded storage at no less
than `ComplexF64` precision.
For `task=:spectrum, algorithm=:block_arnoldi`, `block_size` is included in
the complete reusable block-workspace estimate rather than treated as a free
matrix--matrix optimization. Prepared sources also contribute their bounded
per-action materialization transient and any first-use batched Schur buffer;
`operator_action_per_worker_upper_bytes` reports that part separately.
Prepared dynamics additionally includes the mutable propagation state and the
fresh task-owned application workspace constructed by the propagator.
"""
function recommend_solver(x;task=:steady_state,algorithm=:auto,
                          memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,
                          krylovdim::Integer=30,nev::Integer=6,
                          block_size::Integer=min(nev,4),
                          maxrestarts::Integer=20,
                          recycle_dim::Integer=0,
                          vectors::Bool=false,samples::Integer=1,
                          saved_states::Integer=1,
                          observable_series::Integer=0,workers::Integer=1,
                          T=nothing,observable_type=nothing,time_type=nothing,
                          bigfloat_precision::Integer=precision(BigFloat))
    task in (:steady_state,:spectrum,:dynamics)||throw(ArgumentError(
        "task must be :steady_state, :spectrum, or :dynamics"))
    krylovdim>0||throw(ArgumentError("krylovdim must be positive"))
    recycle_dim>=0||throw(ArgumentError("recycle_dim must be nonnegative"))
    nev>0||throw(ArgumentError("nev must be positive"))
    block_size>0||throw(ArgumentError("block_size must be positive"))
    maxrestarts>=0||throw(ArgumentError("maxrestarts must be nonnegative"))
    samples>=0||throw(ArgumentError("samples must be nonnegative"))
    saved_states>=0||throw(ArgumentError("saved_states must be nonnegative"))
    saved_states<=samples||throw(ArgumentError(
        "saved_states cannot exceed samples"))
    observable_series>=0||throw(ArgumentError(
        "observable_series must be nonnegative"))
    workers>0||throw(ArgumentError("workers must be positive"))
    budget=_resource_memory_budget(memory_budget)
    T=T===nothing ? _resource_scalar_type(x) : T
    T isa Type&&T<:Number||throw(ArgumentError(
        "T must be a numeric scalar type or nothing"))
    T=_complex_float_type(T)
    denseT=promote_type(T,ComplexF64)
    observable_type=observable_type===nothing ? T : observable_type
    observable_type isa Type&&observable_type<:Number||throw(ArgumentError(
        "observable_type must be a numeric scalar type or nothing"))
    observableT=_complex_float_type(observable_type)
    time_type=time_type===nothing ? _real_float_type(T) : time_type
    time_type isa Type&&time_type<:Number||throw(ArgumentError(
        "time_type must be a numeric scalar type or nothing"))
    timeT=_real_float_type(time_type)
    n=pi_dimension(x);autonomous=applicable(isautonomous,x) ? isautonomous(x) : true
    scalar_bytes=_scalar_retained_bytes(T;bigfloat_precision)
    observable_scalar_bytes=_scalar_retained_bytes(
        observableT;bigfloat_precision)
    time_scalar_bytes=_scalar_retained_bytes(timeT;bigfloat_precision)
    dense_bytes=_dense_matrix_structural_bytes(n,T;bigfloat_precision)
    dense_sparse_bytes=_sparse_csc_structural_upper_bytes(
        n,T;bigfloat_precision)
    prepared_sparse_plan=_resource_prepared_sparse_plan(x)
    prepared_sparse_bounds=prepared_sparse_plan===nothing ? nothing :
        _performance_sparse_materialization_bounds(
            prepared_sparse_plan;bigfloat_precision)
    actual_sparse_bytes=_resource_sparse_operator_actual_bytes(x)
    sparse_bytes=actual_sparse_bytes!==nothing ? actual_sparse_bytes :
        prepared_sparse_bounds!==nothing ? prepared_sparse_bounds.operator_bytes :
        dense_sparse_bytes
    dense_sparse_entries=BigInt(n)^2
    dense_sparse_assembly=dense_sparse_entries*(scalar_bytes+2sizeof(Int))+
        (BigInt(n)+1)*sizeof(Int)
    sparse_materialization_peak=actual_sparse_bytes!==nothing ? big(0) :
        prepared_sparse_bounds!==nothing ? prepared_sparse_bounds.peak_bytes :
        8dense_sparse_assembly
    sparse_materialization_temporary=max(
        sparse_materialization_peak-sparse_bytes,big(0))
    sparse_structure_supported=actual_sparse_bytes!==nothing||
        (prepared_sparse_bounds!==nothing&&prepared_sparse_bounds.structured)
    gmres_bytes=estimate_solver_bytes(x;algorithm=:gmres,
        krylovdim=krylovdim,recycle_dim,T=T,bigfloat_precision)
    arnoldi_bytes=estimate_solver_bytes(x;algorithm=:arnoldi,
        krylovdim=krylovdim,T=T,bigfloat_precision)
    block_arnoldi_bytes=estimate_solver_bytes(x;algorithm=:block_arnoldi,
        krylovdim=krylovdim,block_size,T=T,bigfloat_precision)
    dynamics_bytes=estimate_solver_bytes(x;algorithm=:rk4,T=T,bigfloat_precision)
    expv_bytes=estimate_solver_bytes(x;algorithm=:expv,
        krylovdim,T=T,bigfloat_precision)
    basis=_basis_metadata(x,nothing)
    geometry_policy=_recommended_geometry_policy(x,basis)
    geometry_model=x isa PIModel ? x :
        x isa Union{CompiledPIModel,SpecializedPIModel} ? x.model : nothing
    geometry = if geometry_model!==nothing
        _estimate_model_geometry(
            geometry_model;bigfloat_precision)
    elseif !geometry_policy.include
        nothing
    elseif geometry_policy.kind===:symmetric_collective
        _estimate_symmetric_collective_geometry(
            basis,_real_float_type(T);bigfloat_precision)
    elseif geometry_policy.kind===:diagonal_onebody
        _estimate_diagonal_onebody_geometry(
            basis;T=_real_float_type(T),bigfloat_precision)
    else
        estimate_geometry_bytes(basis;T=_real_float_type(T),bigfloat_precision)
    end
    geometry_retained=basis===nothing ? nothing :
        geometry===nothing ? big(0) : geometry.retained_bytes
    geometry_setup=basis===nothing ? nothing :
        geometry===nothing ? big(0) : geometry.setup_bytes
    state_bytes=estimate_state_bytes(x;T=T,bigfloat_precision)
    observable_operator_bytes=big(observable_series)*big(n)*
        observable_scalar_bytes
    prepared_source=_resource_source_prepared(x)
    input_retained=big(Base.summarysize(x))
    term_count=x isa PIModel ? length(x.terms) : 0

    algorithm isa Symbol||throw(ArgumentError(
        "recommend_solver algorithm must be a Symbol"))
    normalized = task===:steady_state ?
        _canonical_stationary_algorithm(algorithm) : task===:spectrum ?
        _canonical_spectrum_algorithm(algorithm) :
        _canonical_dynamics_algorithm(algorithm)
    matrixfree_only=_operator_requires_matrixfree(x)
    if matrixfree_only&&normalized!==:auto
        supported=task===:steady_state ? normalized===:gmres :
            task===:spectrum ? normalized in
                (:arnoldi,:block_arnoldi,:harmonic,:iram,:jd) :
            normalized in (:rk4,:expv)
        supported||throw(ArgumentError(
            "this composite source supports only matrix-free $task " *
            "algorithms; full composite materialization is intentionally " *
            "disabled"))
    end

    function output_bytes_for(chosen_algorithm)
        outputT = chosen_algorithm in
            (:direct,:shiftinvert,:svd,:eigen,:dense) ? denseT : T
        output_scalar_bytes=_scalar_retained_bytes(outputT;bigfloat_precision)
        if task===:steady_state
            return big(workers)*big(n)*output_scalar_bytes
        elseif task===:spectrum
            count=min(big(n),big(nev))
            if chosen_algorithm===:block_arnoldi
                return big(workers)*_performance_block_arnoldi_output_bytes(
                    n,T,count,maxrestarts;vectors,
                    bigfloat_precision)
            end
            return big(workers)*(count*output_scalar_bytes+
                (vectors ? count*big(n)*output_scalar_bytes : big(0)))
        end
        big(workers)*(big(samples)*time_scalar_bytes+
            big(saved_states)*state_bytes+
            big(samples)*big(observable_series)*observable_scalar_bytes)
    end

    function resources_for(chosen_backend,chosen_algorithm)
        setup = if prepared_source
            extra=chosen_backend===:sparse&&
                !_resource_source_has_sparse_operator(x) ?
                sparse_materialization_temporary : big(0)
            _resource_component(extra,iszero(extra) ? :actual : :upper_bound;
                includes=(:already_prepared_source,
                          :operator_assembly_temporary))
        elseif geometry_setup===nothing
            _resource_component(chosen_backend===:sparse ?
                sparse_materialization_temporary : nothing,
                :unknown;includes=(:known_operator_assembly_storage,),
                excludes=(:unavailable_geometry_setup,:custom_term_lowering))
        else
            assembly=chosen_backend===:sparse ?
                sparse_materialization_temporary : big(0)
            geometry_temporary=max(big(geometry_setup)-big(geometry_retained),
                                   big(0))
            _resource_component(geometry_temporary+assembly,:estimate;
                includes=(:one_body_geometry_temporary_setup,
                          :pbody_geometry_temporary_setup,
                          :operator_assembly_temporary),
                excludes=(:allocator_metadata,:custom_term_transients))
        end

        retained = if prepared_source
            extra=chosen_backend===:sparse&&!_resource_source_has_sparse_operator(x) ?
                sparse_bytes : big(0)
            provenance=iszero(extra)&&iszero(observable_operator_bytes) ?
                :actual : :upper_bound
            _resource_component(input_retained+extra+observable_operator_bytes,
                provenance;includes=(:prepared_source,:prepared_observables,
                                      :requested_operator_representation),
                excludes=(:allocator_metadata,))
        else
            geometry_known=geometry_retained===nothing ? big(0) : geometry_retained
            # Trace vectors, immutable kernel blocks, and one task workspace are
            # linear in retained PI coordinates for built-in one-body/direct
            # terms.  Geometry has its own stronger structural bound above.
            plan_payload=big(3+2term_count)*state_bytes
            operator_payload=chosen_backend===:sparse ? sparse_bytes : big(0)
            bytes=input_retained+geometry_known+plan_payload+operator_payload+
                observable_operator_bytes
            provenance=geometry_retained===nothing ? :unknown : :estimate
            _resource_component(bytes,provenance;
                includes=(:input_source,:geometry_retention,
                          :compiled_kernel_estimate,:prepared_observables,
                          :requested_operator_representation),
                excludes=(:allocator_metadata,:custom_term_payloads))
        end

        solverT=chosen_algorithm in
            (:direct,:shiftinvert,:svd,:eigen,:dense) ? denseT : T
        solver_workspace_single = if task===:steady_state
            chosen_algorithm===:gmres ? gmres_bytes :
            estimate_solver_bytes(x;algorithm=chosen_algorithm,
                krylovdim=krylovdim,T=solverT,bigfloat_precision)
        elseif task===:spectrum
            chosen_algorithm===:dense ? estimate_solver_bytes(x;
                algorithm=:dense,krylovdim=krylovdim,T=solverT,bigfloat_precision) :
            chosen_algorithm===:block_arnoldi ? block_arnoldi_bytes :
            chosen_algorithm in (:jd,:jacobi_davidson) ? estimate_solver_bytes(x;
                algorithm=:jd,krylovdim=krylovdim,T=T,bigfloat_precision) :
            chosen_algorithm in (:iram,) ? estimate_solver_bytes(x;
                algorithm=:iram,krylovdim=krylovdim,T=T,bigfloat_precision) :
            chosen_algorithm===:harmonic ? estimate_solver_bytes(x;
                algorithm=:harmonic,krylovdim=krylovdim,T=T,bigfloat_precision) :
            arnoldi_bytes
        else
            chosen_algorithm===:expv ? expv_bytes : dynamics_bytes
        end
        # The propagator owns one mutable PI state independently of any
        # returned snapshots. This remains present for observable-only
        # streaming and is not part of either integration workspace.
        task===:dynamics&&(solver_workspace_single+=state_bytes)
        iterative=task===:dynamics||chosen_algorithm in
            (:gmres,:arnoldi,:block_arnoldi,:harmonic,:iram,:jd,
             :jacobi_davidson)
        operator_action_single=if !iterative
            big(0)
        elseif !prepared_source
            task===:dynamics ? _performance_array_bytes(
                n,T,0;linear_arrays=16,bigfloat_precision) : big(0)
        elseif task===:dynamics
            _performance_linear_operator_workspace_bytes(x)+
                _performance_source_action_bytes(x,T)
        else
            fresh_plan_workspace=x isa LiouvillianPlan ?
                _performance_linear_operator_workspace_bytes(x) : big(0)
            batch_growth=chosen_algorithm===:block_arnoldi ?
                _performance_batched_action_growth_bytes(
                    x,min(n,krylovdim,block_size)) : big(0)
            fresh_plan_workspace+
                _performance_source_action_bytes(x,T)+batch_growth
        end
        solver_single=solver_workspace_single+operator_action_single
        solver_provenance = chosen_algorithm in
            (:direct,:shiftinvert,:svd,:eigen,:dense) ? :upper_bound : :actual
        !iszero(operator_action_single)&&(solver_provenance=:upper_bound)
        solve=_resource_component(big(workers)*solver_single,solver_provenance;
            includes=(:task_owned_solver_workspaces,
                      :operator_action_transients,
                      :lazy_batched_operator_workspace),
            excludes=(:preconditioner_payload,:vendor_library_hidden_buffers))
        output_bytes=output_bytes_for(chosen_algorithm)
        output=_resource_component(output_bytes,:upper_bound;
            includes=(:returned_states,:returned_values,:saved_times,
                      :observable_series),excludes=(:container_metadata,))
        peak_info=_resource_peak(setup,retained,solve,output)
        (;setup,retained,solve,output,
          solver_workspace_single,operator_action_single,peak_info...)
    end

    requested_algorithm=normalized
    if normalized===:auto
        if task===:steady_state
            direct_resources=matrixfree_only ? nothing :
                resources_for(:sparse,:direct)
            direct_fits=!matrixfree_only&&(
                budget.disabled||
                direct_resources.known_peak_bytes<=budget.bytes)
            if matrixfree_only
                backend=:matrixfree;selected_algorithm=:gmres
                reason="the prepared composite source intentionally exposes only factorized matrix-free application"
            elseif autonomous&&n<=512&&direct_fits
                backend=:sparse;selected_algorithm=:direct
                reason="autonomous PI dimension and conservative direct-solve peak are below the crossover and budget"
            else
                backend=:matrixfree;selected_algorithm=:gmres
                reason=!autonomous ?
                    "time-dependent generators require explicit-time matrix-free application" :
                    n>512 ? "PI dimension exceeds the conservative direct-solve crossover" :
                    "the conservative direct-solve peak exceeds the requested memory budget"
            end
        elseif task===:spectrum
            dense_resources=matrixfree_only ? nothing :
                resources_for(:sparse,:dense)
            dense_fits=!matrixfree_only&&(
                budget.disabled||
                dense_resources.known_peak_bytes<=budget.bytes)
            if matrixfree_only
                backend=:matrixfree;selected_algorithm=:arnoldi
                reason="the prepared composite source intentionally exposes only factorized matrix-free application"
            elseif autonomous&&n<=256&&dense_fits
                backend=:sparse;selected_algorithm=:dense
                reason="autonomous PI dimension and conservative dense-spectrum peak are below the crossover and budget"
            else
                backend=:matrixfree;selected_algorithm=:arnoldi
                reason=!autonomous ?
                    "stationary spectra require an autonomous generator; freeze the model before solving" :
                    n>256 ? "PI dimension exceeds the conservative dense-spectrum crossover" :
                    "the conservative dense-spectrum peak exceeds the requested memory budget"
            end
        else
            backend=:matrixfree;selected_algorithm=:rk4
            reason="fixed-step high-level dynamics uses preallocated matrix-free RK4 application"
        end
    else
        selected_algorithm=normalized
        backend = task===:dynamics || selected_algorithm in
            (:gmres,:arnoldi,:block_arnoldi,:harmonic,:iram,:jd,
             :jacobi_davidson) ?
            :matrixfree : :sparse
        reason="the explicitly requested $selected_algorithm algorithm determines the $backend backend"
    end

    resources=resources_for(backend,selected_algorithm)
    budget_status,safe_to_run=_resource_budget_status(resources.peak,
        resources.known_peak_bytes,budget)
    budget_status===:exceeds&&(reason *=
        "; the selected route exceeds the requested memory budget")
    selected_solver_bytes=resources.solve.bytes
    estimated_peak=resources.peak.bytes
    fits_memory=budget_status===:fits ? true :
        budget_status===:exceeds ? false : missing
    (;task,dimension=n,autonomous,backend,algorithm=selected_algorithm,
      requested_algorithm,reason,memory_budget,
      state_bytes,dense_upper_bytes=dense_bytes,
      sparse_operator_upper_bytes=sparse_bytes,
      sparse_materialization_peak_upper_bytes=sparse_materialization_peak,
      sparse_structure_supported,
      scalar_retained_bytes=scalar_bytes,
      observable_scalar_type=observableT,time_scalar_type=timeT,
      observable_scalar_retained_bytes=observable_scalar_bytes,
      time_scalar_retained_bytes=time_scalar_bytes,
      scalar_storage_estimate=_scalar_storage_estimate(T),
      bigfloat_precision_assumption=_scalar_precision_assumption(
          T,bigfloat_precision),
      krylov_vector_bytes=task===:spectrum ? arnoldi_bytes : gmres_bytes,
      gmres_vector_bytes=gmres_bytes,arnoldi_vector_bytes=arnoldi_bytes,
      block_arnoldi_vector_bytes=block_arnoldi_bytes,block_size,
      dynamics_workspace_bytes=dynamics_bytes,
      expv_workspace_bytes=expv_bytes,
      selected_solver_bytes,geometry_retained_upper_bytes=geometry_retained,
      geometry_setup_upper_bytes=geometry_setup,
      geometry_requirement=geometry_policy.requirement,
      geometry_assumption_source=geometry_policy.source,
      setup_peak_bytes=resources.setup.bytes,
      retained_bytes=resources.retained.bytes,
      solve_workspace_bytes=resources.solve.bytes,
      operator_action_per_worker_upper_bytes=
          resources.operator_action_single,
      operator_action_upper_bytes=
          big(workers)*resources.operator_action_single,
      output_bytes=resources.output.bytes,
      resources=(setup=resources.setup,retained=resources.retained,
                 solve=resources.solve,output=resources.output,
                 peak=resources.peak),
      known_peak_bytes=resources.known_peak_bytes,
      peak_provenance=resources.peak.provenance,
      unknown_components=resources.unknown_components,
      budget_status,safe_to_run,
      estimated_peak_bytes=estimated_peak,fits_memory,
      heuristic=true)
end

"""Unified diagnostics for PI states, models, and compiled/linear operators."""
diagnostics(rho::PIState;kwargs...)=state_diagnostics(rho;kwargs...)
function diagnostics(model::PIModel;kwargs...)
    merge(model_summary(model),check_generator(model),
          (;autonomous=isautonomous(model),basis_bytes=estimate_basis_bytes(model.basis),
            state_bytes=estimate_state_bytes(model.basis)))
end
function diagnostics(prepared::CompiledPIModel;kwargs...)
    merge(prepared.estimates,(;backend=prepared.backend,
        autonomous=isautonomous(prepared),retained_bytes=Base.summarysize(prepared)))
end
function diagnostics(family::CompiledPIModelFamily;kwargs...)
    merge(family.estimates,(;varied_rates=family.rate_indices,
        retained_bytes=Base.summarysize(family)))
end
function diagnostics(prepared::SpecializedPIModel;kwargs...)
    merge(prepared.estimates,(;backend=prepared.backend,autonomous=true,
        rates=prepared.rates,retained_bytes=Base.summarysize(prepared)))
end
function diagnostics(plan::LiouvillianPlan;kwargs...)
    (;dimension=size(plan,1),scalar_type=eltype(plan),autonomous=isautonomous(plan),
      kernels=plan.kernels===nothing ? 0 : length(plan.kernels),
      retained_bytes=Base.summarysize(plan))
end
function diagnostics(x;kwargs...)
    (;dimension=pi_dimension(x),autonomous=applicable(isautonomous,x) ? isautonomous(x) : missing,
      retained_bytes=Base.summarysize(x))
end
