"""
    ParameterScanPlan(parameters, model_builder; task=:steady_state, ...)

Immutable, shareable description of a parameter scan. `model_builder` is
called as either `model_builder(parameter)` or
`model_builder(parameter, index)` and must return a [`PIModel`](@ref), an
already [`CompiledPIModel`](@ref), or a [`SpecializedPIModel`](@ref). Models are compiled point by point because
their prepared kernels may depend on the scanned parameter; a returned
compiled model is used directly. The default compilation backend is
`:matrixfree`; override `compile_options` explicitly for a materialized scan.

`task=:steady_state` calls [`stationary_state`](@ref), while `task=:spectrum`
calls [`liouvillian_spectrum`](@ref). The corresponding solver is selected by
`algorithm`. `compile_options` and `solver_options` must be named tuples.
Internally managed keywords such as `initial_state`, `workspace`, and
`return_info` are rejected instead of being silently overridden.

`memory_budget` defaults to 512 MiB and is enforced before each solver
workspace is allocated. Its conservative scan peak includes active compiled
operators, solver storage, one live output per worker, every output requested
by `save_outputs=true`, and the final restart seed. The builder and arbitrary
user diagnostics are not size-predictable and are reported as exclusions.
Explicit sparse compilation or family specialization is rejected before
materialization when its own bound exceeds the same budget. Pass
`memory_budget=Inf` only as an explicit opt-out after checking available RAM.

With `continuation=true`, a serial scan uses the preceding successful state or
Ritz vector as the next initial condition and reuses a compatible Krylov
workspace. Plans contain no mutable numerical scratch and may be shared among
tasks; every concurrent caller must own a separate
[`ParameterScanWorkspace`](@ref). The plan owns a copy of the parameter
container and never mutates it.

For spectral scans, `spectrum_target` accepts the same values as
[`liouvillian_spectrum`](@ref), and `nev` is the requested number of modes.
`save_outputs=false` supports streaming scans with bounded state-history
storage. `save_restart=true` still retains one final continuation seed in the
result so a prefix scan can be resumed. `diagnostic` may be a callable of
`(output, parameter, index)`, `(output, parameter)`, or `(output)` and its
return value is stored alongside the solver diagnostics.
"""
struct ParameterScanPlan{P,B,A,C,S,D}
    parameters::P
    model_builder::B
    task::Symbol
    algorithm::A
    compile_options::C
    solver_options::S
    continuation::Bool
    save_outputs::Bool
    save_vectors::Bool
    save_restart::Bool
    spectrum_target::Symbol
    nev::Int
    diagnostic::D
    seed::UInt64
    memory_budget::Int
    budget_disabled::Bool
end

const _SCAN_TASKS=(:steady_state,:spectrum)
const _SCAN_SPECTRUM_TARGETS=(:largest_real,:near_zero,:largest_magnitude)
const _SCAN_EXECUTIONS=(:serial,:threads)
const _SCAN_FAILURE_POLICIES=(:stop,:record,:throw)

function _scan_named_tuple(value,name)
    value isa NamedTuple||throw(ArgumentError("$name must be a NamedTuple"))
    value
end

function _scan_forbid_options(options,forbidden,name)
    present=Symbol[key for key in forbidden if haskey(options,key)]
    isempty(present)||throw(ArgumentError(
        "$name contains internally managed keyword(s): $(join(string.(present), ", "))"))
    options
end

function _scan_seed(seed::Integer)
    seed>=0||throw(ArgumentError("seed must be nonnegative"))
    BigInt(seed)<=BigInt(typemax(UInt64))||throw(ArgumentError(
        "seed must fit in UInt64"))
    UInt64(seed)
end

function ParameterScanPlan(parameters,model_builder;
        task::Symbol=:steady_state,algorithm=task===:steady_state ? GMRESAlgorithm() : :krylov,
        compile_options=(backend=:matrixfree,),solver_options=NamedTuple(),
        continuation::Bool=true,save_outputs::Bool=true,
        save_vectors::Bool=false,save_restart::Bool=true,
        spectrum_target::Symbol=:largest_real,nev::Integer=6,
        diagnostic=nothing,seed::Integer=0,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    task in _SCAN_TASKS||throw(ArgumentError(
        "task must be :steady_state or :spectrum"))
    spectrum_target in _SCAN_SPECTRUM_TARGETS||throw(ArgumentError(
        "spectrum_target must be :largest_real, :near_zero, or :largest_magnitude"))
    nev>0||throw(ArgumentError("nev must be positive"))
    BigInt(nev)<=typemax(Int)||throw(ArgumentError(
        "nev must be representable as an Int"))
    nev_int=Int(nev)
    values=collect(parameters)
    isempty(values)&&throw(ArgumentError("parameters cannot be empty"))
    builder_applicable=applicable(model_builder,values[1])||
                       applicable(model_builder,values[1],1)
    builder_applicable||throw(ArgumentError(
        "model_builder must accept (parameter) or (parameter, index)"))
    coptions=_scan_named_tuple(compile_options,"compile_options")
    soptions=_scan_named_tuple(solver_options,"solver_options")
    haskey(coptions,:memory_budget)&&throw(ArgumentError(
        "use ParameterScanPlan(...; memory_budget=...) instead of "*
        "compile_options.memory_budget so compilation and solver storage "*
        "share one resource limit"))
    if task===:steady_state
        scan_method,_=_algorithm_options(algorithm)
        scan_method in (:auto,:direct,:svd,:eigen,:shiftinvert,
                        :shift_invert,:inverse_iteration,:krylov,:gmres)||
            throw(ArgumentError("unsupported steady-state scan algorithm $scan_method"))
        _scan_forbid_options(soptions,
            (:return_info,:initial_state,:workspace,:memory_budget),
            "solver_options")
        if algorithm isa GMRESAlgorithm
            _scan_forbid_options(soptions,(:krylovdim,),"solver_options")
        elseif algorithm isa RecycledGMRESAlgorithm
            _scan_forbid_options(soptions,(:krylovdim,:recycle_dim),
                                 "solver_options")
        end
    else
        scan_method,_=_spectrum_algorithm(
            algorithm,spectrum_target,max(nev_int,2),nev_int)
        scan_method in (:dense,:arnoldi,:block_arnoldi,:harmonic,:iram,:jd)||
            throw(ArgumentError("unsupported spectral scan algorithm $scan_method"))
        _scan_forbid_options(soptions,
            (:return_info,:initial_vector,:workspace,:rng,:vectors,:nev,
             :subspace_dim,:target,:memory_budget),
            "solver_options")
        if algorithm isa HarmonicArnoldiAlgorithm
            algorithm.nev==nev_int||throw(ArgumentError(
                "nev must match HarmonicArnoldiAlgorithm.nev"))
            _scan_forbid_options(soptions,
                (:krylovdim,:thickdim,:maxrestarts),"solver_options")
        end
    end
    budget_limit=_performance_memory_limit(memory_budget)
    ParameterScanPlan(values,model_builder,task,algorithm,coptions,soptions,
        continuation,save_outputs,save_vectors,save_restart,spectrum_target,
        nev_int,diagnostic,_scan_seed(seed),_memory_budget_bytes(memory_budget),
        budget_limit===nothing)
end

"""
    ParameterScanPlan(parameters, prototype, remaker; kwargs...)

Convenience constructor for a prototype/remaker workflow. `remaker` may accept
`(prototype, parameter)` or `(prototype, parameter, index)` and must return a
fresh `PIModel` or `CompiledPIModel`. A remaker used by concurrent scans must
not mutate shared prototype state.
"""
function ParameterScanPlan(parameters,prototype,remaker;kwargs...)
    builder=function(parameter,index)
        if applicable(remaker,prototype,parameter,index)
            remaker(prototype,parameter,index)
        elseif applicable(remaker,prototype,parameter)
            remaker(prototype,parameter)
        else
            throw(ArgumentError(
                "remaker must accept (prototype, parameter) or (prototype, parameter, index)"))
        end
    end
    ParameterScanPlan(parameters,builder;kwargs...)
end

struct _FamilySpecializationRequest{F,R,O}
    family::F
    rates::R
    options::O
end

"""
    ParameterScanPlan(parameters, family::CompiledPIModelFamily;
                      rate_builder=nothing,
                      specialize_options=(backend=:matrixfree,), kwargs...)

Construct a scan that reuses one family's prepared Schur geometry at every
point. By default each parameter is passed directly to [`specialize`](@ref).
`rate_builder` may instead accept `(parameter)` or `(parameter, index)` and
return the scalar or ordered rate collection to bind. `specialize_options`
controls the per-point backend and must be a named tuple.
"""
function ParameterScanPlan(parameters,family::CompiledPIModelFamily;
        rate_builder=nothing,specialize_options=(backend=:matrixfree,),kwargs...)
    options=_scan_named_tuple(specialize_options,"specialize_options")
    haskey(options,:workspace)&&throw(ArgumentError(
        "family scan workspaces are managed by ParameterScanWorkspace"))
    haskey(options,:memory_budget)&&throw(ArgumentError(
        "use ParameterScanPlan(...; memory_budget=...) instead of "*
        "specialize_options.memory_budget so specialization and solver "*
        "storage share one resource limit"))
    builder=function(parameter,index)
        rates = if rate_builder===nothing
            parameter
        elseif applicable(rate_builder,parameter,index)
            rate_builder(parameter,index)
        elseif applicable(rate_builder,parameter)
            rate_builder(parameter)
        else
            throw(ArgumentError(
                "rate_builder must accept (parameter) or (parameter, index)"))
        end
        _FamilySpecializationRequest(family,rates,options)
    end
    ParameterScanPlan(parameters,builder;kwargs...)
end

"""
    ParameterScanWorkspace()

Task-owned mutable continuation, Liouvillian, and Krylov scratch for
[`parameter_scan`](@ref). Reuse it sequentially across compatible scans, but
never from concurrent tasks. Family scans reuse one `LiouvillianWorkspace`
while the immutable family plan is unchanged. Krylov storage is discarded and
rebuilt whenever the task, PI coordinate dimension, scalar type, or requested
subspace size changes.
"""
mutable struct ParameterScanWorkspace
    continuation_seed::Any
    continuation_signature::Any
    continuation_eltype::Any
    solver_workspace::Any
    solver_signature::Any
    liouvillian_workspace::Any
    liouvillian_plan::Any
end
ParameterScanWorkspace()=ParameterScanWorkspace(
    nothing,nothing,nothing,nothing,nothing,nothing,nothing)

"""Clear all continuation and solver scratch retained by `workspace`."""
function clear_parameter_scan_workspace!(workspace::ParameterScanWorkspace)
    workspace.continuation_seed=nothing
    workspace.continuation_signature=nothing
    workspace.continuation_eltype=nothing
    workspace.solver_workspace=nothing
    workspace.solver_signature=nothing
    workspace.liouvillian_workspace=nothing
    workspace.liouvillian_plan=nothing
    workspace
end

"""
    ParameterScanPoint

One explicit parameter-scan record. `status` is `:success` or `:failed`;
failed records retain a printable error type and message and are never treated
as converged. Residual and trace-error values preserve the solver's scalar
precision. Wall-clock fields are measured in seconds.
"""
struct ParameterScanPoint
    index::Int
    parameter::Any
    status::Symbol
    output::Any
    residual::Any
    trace_error::Any
    converged::Bool
    iterations::Any
    compile_seconds::Float64
    solve_seconds::Float64
    elapsed_seconds::Float64
    warm_started::Bool
    workspace_reused::Bool
    diagnostics::Any
    error_type::Union{Nothing,String}
    message::Union{Nothing,String}
end

"""
    ParameterScanResult

Checkpoint-neutral parameter-scan output. It contains parameter values,
plain point records, optional saved numerical outputs, and at most one final
continuation seed; it never retains the model builder, compiled models, solver
workspaces, locks, random generators, callbacks, or exception backtraces.
Use [`resume_parameter_scan`](@ref) with the original plan to continue a
prefix, and [`merge_parameter_scan_results`](@ref) to join disjoint chunks.
"""
struct ParameterScanResult{P,M}
    task::Symbol
    parameters::P
    points::Vector{ParameterScanPoint}
    restart_index::Int
    restart_seed::Any
    metadata::M
end

Base.length(result::ParameterScanResult)=length(result.points)
Base.firstindex(result::ParameterScanResult)=firstindex(result.points)
Base.lastindex(result::ParameterScanResult)=lastindex(result.points)
Base.getindex(result::ParameterScanResult,index::Integer)=result.points[index]
Base.iterate(result::ParameterScanResult,args...)=iterate(result.points,args...)

function Base.show(io::IO,plan::ParameterScanPlan)
    budget=plan.budget_disabled ? "disabled" : "$(plan.memory_budget) bytes"
    print(io,"ParameterScanPlan($(length(plan.parameters)) points, task=$(plan.task), continuation=$(plan.continuation), memory_budget=$budget)")
end
function Base.show(io::IO,result::ParameterScanResult)
    successes=count(point->point.status===:success,result.points)
    failures=length(result.points)-successes
    print(io,"ParameterScanResult($(length(result.points))/$(length(result.parameters)) points, task=$(result.task), successes=$successes, failures=$failures)")
end

@inline function _scan_model(plan,parameter,index)
    if applicable(plan.model_builder,parameter,index)
        plan.model_builder(parameter,index)
    elseif applicable(plan.model_builder,parameter)
        plan.model_builder(parameter)
    else
        throw(ArgumentError(
            "model_builder is not applicable to parameter at index $index"))
    end
end

function _scan_family_workspace!(workspace::ParameterScanWorkspace,family)
    if workspace.liouvillian_plan===family.plan&&
       workspace.liouvillian_workspace isa LiouvillianWorkspace
        return workspace.liouvillian_workspace,true
    end
    workspace.liouvillian_workspace=LiouvillianWorkspace(family.plan)
    workspace.liouvillian_plan=family.plan
    workspace.liouvillian_workspace,false
end

function _scan_model_scalar_type(model::PIModel)
    T=Complex{_model_geometry_type(model)}
    for term in model.terms
        T=_scale_promoted_type(T,term_rate(term))
    end
    _complex_float_type(T)
end

function _scan_source_shape(source)
    source isa _FamilySpecializationRequest&&return (
        length(source.family.plan.basis),source.family.plan.Ttype)
    source isa PIModel&&return (
        length(source.basis),_scan_model_scalar_type(source))
    source isa Union{CompiledPIModel,SpecializedPIModel}&&return (
        size(source,1),eltype(source))
    throw(ArgumentError("unsupported scan source $(typeof(source))"))
end

function _scan_validate_source(source,index)
    source isa Union{_FamilySpecializationRequest,PIModel,
                     CompiledPIModel,SpecializedPIModel}&&return source
    throw(ArgumentError(
        "model_builder returned $(typeof(source)) at index $index; "*
        "expected PIModel, CompiledPIModel, or SpecializedPIModel"))
end

function _scan_compile(plan,parameter,index,workspace,active_workers::Int=1)
    # Validate the builder contract before resource inspection so an invalid
    # return is reported as such rather than as an internal shape-estimation
    # failure.
    source=_scan_validate_source(_scan_model(plan,parameter,index),index)
    operator_budget=_scan_operator_budget(plan,source,active_workers)
    if source isa _FamilySpecializationRequest
        backend=get(source.options,:backend,:matrixfree)
        matrixfree = backend===:matrixfree||
            (backend===:auto&&!_performance_budget_fits(
                source.family.estimates.sparse_specialization_peak_upper_bound,
                operator_budget))
        if matrixfree
            _require_performance_budget(
                "matrix-free family scan specialization",
                source.family.estimates.matrixfree_specialization_upper_bound,
                operator_budget;guidance=
                "Reduce scan output retention, Krylov size, or worker count.")
            liouvillian_workspace,reused=
                _scan_family_workspace!(workspace,source.family)
            specialized=specialize(source.family,source.rates;
                source.options...,workspace=liouvillian_workspace,
                memory_budget=operator_budget)
            estimates=merge(specialized.estimates,
                (;scan_liouvillian_workspace_reused=reused))
            return SpecializedPIModel(specialized.family,specialized.model,
                specialized.plan,specialized.operator,specialized.rates,
                specialized.backend,estimates)
        end
        specialized=specialize(source.family,source.rates;source.options...,
            memory_budget=operator_budget)
        estimates=merge(specialized.estimates,
            (;scan_liouvillian_workspace_reused=false))
        return SpecializedPIModel(specialized.family,specialized.model,
            specialized.plan,specialized.operator,specialized.rates,
            specialized.backend,estimates)
    elseif source isa Union{CompiledPIModel,SpecializedPIModel}
        source
    elseif source isa PIModel
        compile(source;plan.compile_options...,
            memory_budget=operator_budget)
    end
end

@inline _scan_algorithm_is_auto(algorithm)=
    algorithm===:auto||algorithm isa AutoAlgorithm

function _scan_solver_upper_bytes(plan,n::Int,source_type,
                                  method_override=nothing)
    T=_complex_float_type(source_type)
    if plan.task===:steady_state
        method=method_override===nothing ?
            _scan_steady_method(plan.algorithm) : method_override
        if method===:krylov
            _,algorithm_options=_algorithm_options(plan.algorithm)
            options=merge(algorithm_options,plan.solver_options)
            T=_promote_krylov_scalar_type(
                T,get(options,:operator_scale,nothing))
            T=_promote_krylov_scalar_type(
                T,get(options,:preconditioner_regularization,0))
            preconditioner=get(options,:preconditioner,nothing)
            preconditioner!==nothing&&!(preconditioner isa Symbol)&&
                (T=_promote_krylov_operator_type(T,preconditioner))
            krylovdim=plan.algorithm isa Union{GMRESAlgorithm,
                                                RecycledGMRESAlgorithm} ?
                plan.algorithm.krylovdim : Int(get(options,:krylovdim,30))
            recycle_dim=Int(get(options,:recycle_dim,0))
            bytes=_performance_gmres_bytes(n,T,krylovdim;
                recycle_dim=max(recycle_dim,0))
            preconditioner===nothing||
                (bytes+=_performance_array_bytes(n,T,1;linear_arrays=2))
            return bytes,method
        end
        # Direct, SVD, eigen, shift-invert, and auto may retain a bordered
        # matrix plus factorization/eigensolver scratch. Match the common
        # dense high-level guard rather than probing or materializing L.
        arrays=method in (:svd,:eigen,:auto) ? 8 : 6
        denseT=promote_type(T,ComplexF64)
        return _performance_array_bytes(n,denseT,arrays;linear_arrays=8),method
    end
    method = method_override===nothing ?
        first(_scan_spectrum_method(plan,n)) :
        _canonical_spectrum_algorithm(method_override)
    if method===:dense
        denseT=promote_type(T,ComplexF64)
        # liouvillian_spectrum performs its public high-level preflight before
        # the lower-level 5/7-array eigensolver guard. Match the stronger
        # eight-copy bound so a scan-approved point cannot be rejected by the
        # nested command after aggregate accounting has succeeded.
        return _performance_array_bytes(
            n,denseT,8;linear_arrays=6),method
    elseif method===:block_arnoldi
        dim=Int(get(plan.solver_options,:krylovdim,
            max(30,3plan.nev+2Int(get(plan.solver_options,:block_size,
                                      min(plan.nev,4))))))
        block_size=Int(get(plan.solver_options,:block_size,min(plan.nev,4)))
        return _performance_block_arnoldi_bytes(
            n,T,dim,block_size)+_performance_block_arnoldi_output_bytes(
                n,T,plan.nev,get(plan.solver_options,:maxrestarts,20);
                vectors=true),method
    elseif method===:jd
        T=_promote_krylov_scalar_type(
            T,get(plan.solver_options,:operator_scale,nothing))
        preconditioner=get(plan.solver_options,:preconditioner,nothing)
        preconditioner!==nothing&&
            (T=_promote_krylov_operator_type(T,preconditioner))
        dim=Int(get(plan.solver_options,:krylovdim,
            get(plan.solver_options,:subspace_dim,max(30,3plan.nev+6))))
        correction_dim=Int(get(plan.solver_options,:correction_krylovdim,
            min(n,20)))
        # JacobiDavidsonWorkspace owns a full Arnoldi space, an independent
        # correction GMRES space, and six additional full-coordinate vectors.
        bytes=_performance_arnoldi_bytes(n,T,dim;mode=:full)+
              _performance_gmres_bytes(n,T,correction_dim)+
              _performance_entries_bytes(6BigInt(n),T)
        return bytes,method
    end
    default_dim = plan.algorithm isa HarmonicArnoldiAlgorithm ?
        plan.algorithm.krylovdim : method in (:iram,:implicit_qr,:harmonic) ?
        max(30,3plan.nev+6) : max(20,2plan.nev+4)
    dim=Int(get(plan.solver_options,:krylovdim,default_dim))
    mode=method===:arnoldi ? :ordinary : :full
    bytes=_performance_arnoldi_bytes(n,T,dim;mode)
    if mode===:ordinary
        nb=BigInt(n);mb=min(nb,BigInt(dim))
        nested=_performance_entries_bytes(2nb*mb+nb+3mb*mb,T)
        bytes=max(bytes,nested)
    end
    bytes,method
end

function _scan_output_scalar_type(plan,n_int::Int,source_type,
                                  method_override=nothing)
    T=_complex_float_type(source_type)
    if plan.task===:steady_state
        method=method_override===nothing ?
            _scan_steady_method(plan.algorithm) : method_override
        if method!==:krylov
            # The bordered, SVD, eigen, shift-invert, and feasible :auto
            # routes all form the common ComplexF64-promoted dense problem.
            return promote_type(T,ComplexF64)
        end
        _,algorithm_options=_algorithm_options(plan.algorithm)
        options=merge(algorithm_options,plan.solver_options)
        T=_promote_krylov_scalar_type(T,get(options,:operator_scale,nothing))
        T=_promote_krylov_scalar_type(T,
            get(options,:preconditioner_regularization,0))
        preconditioner=get(options,:preconditioner,nothing)
        preconditioner!==nothing&&!(preconditioner isa Symbol)&&
            (T=_promote_krylov_operator_type(T,preconditioner))
        return T
    end
    method = method_override===nothing ?
        first(_scan_spectrum_method(plan,n_int)) :
        _canonical_spectrum_algorithm(method_override)
    method===:dense&&return promote_type(T,ComplexF64)
    if method===:jd
        T=_promote_krylov_scalar_type(T,
            get(plan.solver_options,:operator_scale,nothing))
        preconditioner=get(plan.solver_options,:preconditioner,nothing)
        preconditioner!==nothing&&
            (T=_promote_krylov_operator_type(T,preconditioner))
    end
    T
end

function _scan_output_upper_bytes(plan,n_int::Int,source_type,
                                  method_override=nothing)
    n=BigInt(n_int);nev=BigInt(plan.nev)
    scalar_bytes=_scalar_retained_bytes(
        _scan_output_scalar_type(plan,n_int,source_type,method_override))
    if plan.task===:steady_state
        per_output=n*scalar_bytes
        restart=plan.continuation&&plan.save_restart ? per_output : BigInt(0)
        # The solver result and defensive PIState construction coexist. A
        # continuation point additionally owns its detached restart copy.
        live=plan.continuation ? 3per_output : 2per_output
        return per_output,restart,live
    end
    per_output=nev*scalar_bytes+
        (plan.save_vectors ? n*nev*scalar_bytes : BigInt(0))
    method = method_override===nothing ?
        first(_scan_spectrum_method(plan,n_int)) :
        _canonical_spectrum_algorithm(method_override)
    iterative=method in (:arnoldi,:block_arnoldi,:harmonic,:iram,:jd)
    restart = if !iterative||!plan.continuation||!plan.save_restart
        BigInt(0)
    elseif method in (:harmonic,:block_arnoldi)
        columns=method===:block_arnoldi ?
            min(nev,BigInt(get(plan.solver_options,:block_size,
                               min(plan.nev,4)))) : nev
        n*columns*scalar_bytes
    else
        n*scalar_bytes
    end
    values=nev*scalar_bytes
    live = if !iterative
        per_output
    elseif method in (:harmonic,:block_arnoldi)
        values+3n*nev*scalar_bytes
    else
        values+n*(nev+2)*scalar_bytes
    end
    per_output,restart,live
end

function _scan_resource_components(plan,n::Int,T,workers::Int,method,
                                   operator_retained_bytes::BigInt,
                                   operator_action_bytes::Integer=big(0))
    operator_action_bytes>=0||throw(ArgumentError(
        "operator action estimate must be nonnegative"))
    method=plan.task===:steady_state ? _stationary_solver_method(method) :
        _canonical_spectrum_algorithm(method)
    solver_bytes,_=_scan_solver_upper_bytes(plan,n,T,method)
    per_output,restart_bytes,live_output=
        _scan_output_upper_bytes(plan,n,T,method)
    retained=plan.save_outputs ? BigInt(length(plan.parameters))*per_output :
        BigInt(0)
    fixed=operator_retained_bytes+BigInt(workers)*live_output+
          retained+restart_bytes
    action_bytes=BigInt(operator_action_bytes)
    worker_peak=solver_bytes+action_bytes
    known_peak=fixed+BigInt(workers)*worker_peak
    (;method,solver_bytes,operator_action_bytes=action_bytes,
      worker_solver_peak_bytes=worker_peak,
      per_output,restart_bytes,live_output,
      retained_output_bytes=retained,fixed_bytes=fixed,known_peak)
end

function _scan_effective_method(plan,n::Int,T,workers::Int,
                                operator_retained_bytes::BigInt,
                                operator_action_bytes::Integer=big(0))
    if !_scan_algorithm_is_auto(plan.algorithm)
        return plan.task===:steady_state ? _scan_steady_method(plan.algorithm) :
            first(_scan_spectrum_method(plan,n))
    end
    dense_method,iterative_method = if plan.task===:steady_state
        n<=512 ? (:direct,:krylov) : (:krylov,:krylov)
    elseif plan.spectrum_target===:near_zero
        (:harmonic,:harmonic)
    else
        n<=256 ? (:dense,:arnoldi) : (:arnoldi,:arnoldi)
    end
    dense_method===iterative_method&&return dense_method
    plan.budget_disabled&&return dense_method
    dense=_scan_resource_components(plan,n,T,workers,dense_method,
                                    operator_retained_bytes,
                                    operator_action_bytes)
    dense.known_peak<=BigInt(plan.memory_budget) ? dense_method :
                                                  iterative_method
end

function _scan_operator_action_upper_bytes(plan,compiled,method,T)
    bytes=_performance_source_action_bytes(compiled,T)
    if plan.task===:spectrum&&
            _canonical_spectrum_algorithm(method)===:block_arnoldi
        n=size(compiled,1)
        block_size=Int(get(plan.solver_options,:block_size,min(plan.nev,4)))
        krylovdim=Int(get(plan.solver_options,:krylovdim,
            max(30,3plan.nev+2block_size)))
        active_block=min(n,krylovdim,block_size)
        bytes+=_performance_batched_action_growth_bytes(
            compiled,active_block)
    end
    bytes
end

function _scan_operator_budget(plan,source,workers::Int)
    plan.budget_disabled&&return Inf
    n,T=_scan_source_shape(source)
    # A family is already retained and shared by local worker tasks; its
    # specialization bound deliberately excludes this common immutable plan.
    shared=source isa _FamilySpecializationRequest ?
        BigInt(Base.summarysize(source.family)) : BigInt(0)
    selection_operator = if source isa _FamilySpecializationRequest
        requested=get(source.options,:backend,:matrixfree)
        per_worker = requested===:sparse ?
            BigInt(source.family.estimates.sparse_operator_upper_bound) :
            source.family.estimates.matrixfree_specialization_upper_bound
        shared+BigInt(workers)*per_worker
    elseif source isa PIModel&&get(plan.compile_options,:backend,:auto)===:sparse
        scalar_bytes=_scalar_retained_bytes(T)
        sparse_operator=BigInt(n)^2*(scalar_bytes+2sizeof(Int))+
                        (BigInt(n)+1)*sizeof(Int)
        BigInt(workers)*sparse_operator
    elseif source isa Union{CompiledPIModel,SpecializedPIModel}
        BigInt(workers)*BigInt(Base.summarysize(source))
    elseif source isa PIModel
        # A raw model has not yet exposed the size of its compiled operator.
        # Reserving zero here can make :auto select a dense solver under a
        # budget which leaves no room even for model preparation.  The
        # assembly-free preparation bound is conservative for this decision
        # and is also the minimum allowance passed to `compile` below.
        BigInt(workers)*_model_preparation_bytes(source)
    else
        shared
    end
    base_action=source isa Union{CompiledPIModel,SpecializedPIModel} ?
        _performance_source_action_bytes(source,T) : big(0)
    method=_scan_effective_method(
        plan,n,T,workers,selection_operator,base_action)
    action=source isa Union{CompiledPIModel,SpecializedPIModel} ?
        _scan_operator_action_upper_bytes(plan,source,method,T) : base_action
    components=_scan_resource_components(
        plan,n,T,workers,method,shared,action)
    available=max(BigInt(plan.memory_budget)-components.known_peak,BigInt(0))
    Int(min(div(available,workers),BigInt(typemax(Int))))
end

function _scan_resource_report(plan,compiled,workers::Int)
    workers>0||throw(ArgumentError("scan worker count must be positive"))
    full_operator_bytes=BigInt(Base.summarysize(compiled))
    shared_operator_bytes = compiled isa SpecializedPIModel ?
        BigInt(Base.summarysize(compiled.family)) : BigInt(0)
    operator_per_worker_bytes=max(
        full_operator_bytes-shared_operator_bytes,BigInt(0))
    operator_retained_bytes=shared_operator_bytes+
        BigInt(workers)*operator_per_worker_bytes
    n=size(compiled,1);T=eltype(compiled)
    base_action=_performance_source_action_bytes(compiled,T)
    method=_scan_effective_method(
        plan,n,T,workers,operator_retained_bytes,base_action)
    operator_action_bytes=_scan_operator_action_upper_bytes(
        plan,compiled,method,T)
    components=_scan_resource_components(
        plan,n,T,workers,method,operator_retained_bytes,
        operator_action_bytes)
    solver_bytes=components.solver_bytes
    restart_bytes=components.restart_bytes
    live_output_bytes=components.live_output
    retained_output_bytes=components.retained_output_bytes
    fixed_bytes=components.fixed_bytes
    known_peak=components.known_peak
    unknown=plan.diagnostic===nothing ? (:builder_allocations,) :
        (:builder_allocations,:diagnostic_payloads)
    if plan.budget_disabled
        status=:disabled;known_fits=true;safe=missing;solver_budget=Inf
    else
        budget=BigInt(plan.memory_budget)
        known_fits=known_peak<=budget
        status=known_fits ? (isempty(unknown) ? :fits : :unknown) : :exceeds
        safe=known_fits ? missing : false
        available=max(budget-fixed_bytes,BigInt(0))
        solver_budget=Int(min(div(available,workers),BigInt(typemax(Int))))
    end
    (;memory_budget=plan.budget_disabled ? Inf : plan.memory_budget,
      budget_status=status,safe_to_run=safe,known_budget_fits=known_fits,
      known_peak_bytes=known_peak,
      active_workers=workers,method,
      operator_retained_bytes,
      shared_operator_bytes,
      operator_per_worker_bytes,
      solver_workspace_upper_bytes=solver_bytes,
      operator_action_per_worker_upper_bytes=operator_action_bytes,
      operator_action_upper_bytes=BigInt(workers)*operator_action_bytes,
      worker_solver_peak_upper_bytes=components.worker_solver_peak_bytes,
      live_output_bytes=BigInt(workers)*live_output_bytes,
      retained_output_upper_bytes=retained_output_bytes,
      restart_upper_bytes=restart_bytes,solver_memory_budget=solver_budget,
      unknown_components=unknown,
      assumptions=(full_point_count_checked_at_each_observed_dimension=true,
                   compiled_size_uses_summarysize=true,
                   family_geometry_counted_once=true,
                   operator_action_counted_per_worker=true))
end

function _enforce_scan_resource_report(report,index)
    report.known_budget_fits&&return report
    throw(ArgumentError(
        "parameter scan point $index has a conservative known peak of "*
        "$(report.known_peak_bytes) bytes, exceeding memory_budget="*
        "$(report.memory_budget); use save_outputs=false, reduce the "*
        "Krylov dimension or worker count, select a matrix-free backend, "*
        "increase memory_budget, or pass memory_budget=Inf as an explicit opt-out"))
end

@inline _scan_basis(compiled::CompiledPIModel)=compiled.plan.basis
@inline _scan_basis(compiled::SpecializedPIModel)=compiled.plan.basis
@inline _scan_basis_signature(basis::PIBasis)=
    (N=basis.N,d=basis.d,sectors=Tuple(basis.sectors),dimension=length(basis))

function _scan_user_diagnostic(plan,output,parameter,index)
    diagnostic=plan.diagnostic
    diagnostic===nothing&&return nothing
    if applicable(diagnostic,output,parameter,index)
        diagnostic(output,parameter,index)
    elseif applicable(diagnostic,output,parameter)
        diagnostic(output,parameter)
    elseif applicable(diagnostic,output)
        diagnostic(output)
    else
        throw(ArgumentError(
            "diagnostic must accept (output, parameter, index), (output, parameter), or (output)"))
    end
end

function _scan_steady_method(algorithm)
    method,_=_algorithm_options(algorithm)
    method===:gmres&&return :krylov
    method in (:shift_invert,:inverse_iteration)&&return :shiftinvert
    method
end

function _scan_steady_workspace!(workspace,plan,compiled,method)
    method===:krylov||return (nothing,false)
    n=size(compiled,1);T=_complex_float_type(eltype(compiled))
    _,algorithm_options=_algorithm_options(plan.algorithm)
    options=merge(algorithm_options,plan.solver_options)
    T=_promote_krylov_scalar_type(T,get(options,:operator_scale,nothing))
    T=_promote_krylov_scalar_type(T,
        get(options,:preconditioner_regularization,0))
    preconditioner=get(options,:preconditioner,nothing)
    if preconditioner!==nothing&&!(preconditioner isa Symbol)
        T=_promote_krylov_operator_type(T,preconditioner)
    end
    krylovdim = if plan.algorithm isa Union{GMRESAlgorithm,
                                            RecycledGMRESAlgorithm}
        plan.algorithm.krylovdim
    else
        Int(get(plan.solver_options,:krylovdim,30))
    end
    recycle_dim=Int(get(options,:recycle_dim,0))
    recycle_dim>=0||throw(ArgumentError("recycle_dim must be nonnegative"))
    if recycle_dim>0
        signature=(:steady_state_recycled,n,T,min(n,krylovdim),
                   min(max(n-1,0),recycle_dim))
        if workspace.solver_signature==signature&&
           workspace.solver_workspace isa RecycledGMRESWorkspace
            return (workspace.solver_workspace,true)
        end
        workspace.solver_workspace=RecycledGMRESWorkspace(
            T,n,krylovdim,recycle_dim)
        workspace.solver_signature=signature
        return (workspace.solver_workspace,false)
    end
    signature=(:steady_state,n,T,min(n,krylovdim))
    if workspace.solver_signature==signature&&
       workspace.solver_workspace isa KrylovWorkspace
        return (workspace.solver_workspace,true)
    end
    workspace.solver_workspace=KrylovWorkspace(T,n,krylovdim)
    workspace.solver_signature=signature
    workspace.solver_workspace,false
end

function _scan_spectrum_method(plan,n)
    _spectrum_algorithm(plan.algorithm,plan.spectrum_target,n,plan.nev)
end

function _scan_spectrum_workspace!(workspace,plan,compiled,method)
    n=size(compiled,1)
    method in (:arnoldi,:block_arnoldi,:harmonic,:iram,:jd)||
        return (nothing,false,method)
    T=_complex_float_type(eltype(compiled))
    if method===:block_arnoldi
        block_size=Int(get(plan.solver_options,:block_size,min(plan.nev,4)))
        krylovdim=Int(get(plan.solver_options,:krylovdim,
            max(30,3plan.nev+2block_size)))
        signature=(:spectrum_block_arnoldi,n,T,min(n,krylovdim),
                   min(n,krylovdim,block_size))
        if workspace.solver_signature==signature&&
           workspace.solver_workspace isa BlockArnoldiWorkspace
            return (workspace.solver_workspace,true,method)
        end
        workspace.solver_workspace=BlockArnoldiWorkspace(
            T,n,krylovdim,block_size)
        workspace.solver_signature=signature
        return (workspace.solver_workspace,false,method)
    elseif method===:jd
        T=_promote_krylov_scalar_type(T,
            get(plan.solver_options,:operator_scale,nothing))
        preconditioner=get(plan.solver_options,:preconditioner,nothing)
        if preconditioner!==nothing
            T=_promote_krylov_operator_type(T,preconditioner)
        end
        subspace_dim=Int(get(plan.solver_options,:krylovdim,
            get(plan.solver_options,:subspace_dim,max(30,3plan.nev+6))))
        correction_dim=Int(get(plan.solver_options,:correction_krylovdim,min(n,20)))
        signature=(:spectrum_jd,n,T,min(n,subspace_dim),min(n,correction_dim))
        if workspace.solver_signature==signature&&
           workspace.solver_workspace isa JacobiDavidsonWorkspace
            return (workspace.solver_workspace,true,method)
        end
        workspace.solver_workspace=JacobiDavidsonWorkspace(
            T,n,subspace_dim,correction_dim)
        workspace.solver_signature=signature
        return (workspace.solver_workspace,false,method)
    end
    default_dim = if plan.algorithm isa HarmonicArnoldiAlgorithm
        plan.algorithm.krylovdim
    elseif method in (:iram,:harmonic)
        max(30,3plan.nev+6)
    else
        max(20,2plan.nev+4)
    end
    krylovdim=Int(get(plan.solver_options,:krylovdim,default_dim))
    workspace_mode=method===:arnoldi ? :ordinary : :full
    signature=(:spectrum_arnoldi,workspace_mode,n,T,min(n,krylovdim))
    if workspace.solver_signature==signature&&
       workspace.solver_workspace isa ArnoldiWorkspace
        return (workspace.solver_workspace,true,method)
    end
    workspace.solver_workspace=ArnoldiWorkspace(
        T,n,krylovdim;mode=workspace_mode)
    workspace.solver_signature=signature
    workspace.solver_workspace,false,method
end

@inline _scan_continuation_signature(plan,compiled)=
    (task=plan.task,basis=_scan_basis_signature(_scan_basis(compiled)))

function _scan_compatible_seed(workspace,plan,compiled,expected_type=
                               _complex_float_type(eltype(compiled)))
    plan.continuation||return nothing
    signature=_scan_continuation_signature(plan,compiled)
    T=_complex_float_type(expected_type)
    workspace.continuation_signature==signature&&
        workspace.continuation_eltype===T ? workspace.continuation_seed : nothing
end

function _scan_store_seed!(workspace,plan,compiled,seed)
    workspace.continuation_seed=seed
    workspace.continuation_signature=_scan_continuation_signature(plan,compiled)
    workspace.continuation_eltype = seed isa PIState ? eltype(seed.data) :
        eltype(seed)
    workspace
end

function _scan_clear_seed!(workspace)
    workspace.continuation_seed=nothing
    workspace.continuation_signature=nothing
    workspace.continuation_eltype=nothing
    workspace.solver_workspace isa RecycledGMRESWorkspace&&
        (workspace.solver_workspace.nrecycle=0)
    workspace
end

# A restart seed crosses an ownership boundary whenever it moves between a
# task-owned workspace and a checkpoint-neutral public result.  Keep those
# arrays detached: users may serialize or modify a result while reusing the
# workspace for another scan.
_copy_scan_seed(::Nothing)=nothing
_copy_scan_seed(seed::PIState)=copy(seed)
_copy_scan_seed(seed::AbstractVector)=copy(seed)
_copy_scan_seed(seed::AbstractMatrix)=copy(seed)
function _copy_scan_seed(seed)
    throw(ArgumentError(
        "unsupported parameter-scan restart seed type $(typeof(seed))"))
end

# Interrupts are control flow, not failed parameter points.  Catch sites that
# convert ordinary model/solver errors into records call this first.
@inline function _scan_rethrow_interrupt(error)
    error isa InterruptException&&throw(error)
    error
end

@inline function _scan_info_value(info,name,default=missing)
    hasproperty(info,name) ? getproperty(info,name) : default
end

function _scan_maximum_residual(info)
    if hasproperty(info,:residuals)
        residuals=getproperty(info,:residuals)
        isempty(residuals) ? missing : maximum(residuals)
    elseif hasproperty(info,:residual)
        getproperty(info,:residual)
    else
        missing
    end
end

function _scan_converged(info)
    hasproperty(info,:converged)||return true
    value=getproperty(info,:converged)
    value isa Bool ? value : all(value)
end

function _scan_point_steady!(plan,compiled,parameter,index,workspace,resources)
    _,algorithm_options=_algorithm_options(plan.algorithm)
    method=resources.method
    iterative=method in (:krylov,:shiftinvert)
    seed_type=_scan_output_scalar_type(
        plan,size(compiled,1),eltype(compiled),method)
    seed=iterative ?
        _scan_compatible_seed(workspace,plan,compiled,seed_type) : nothing
    initial_state = if seed isa PIState
        PIState(_scan_basis(compiled),seed.data)
    elseif seed===nothing
        nothing
    else
        throw(ArgumentError("steady-state continuation seed has an incompatible type"))
    end
    solver_workspace,reused=
        _scan_steady_workspace!(workspace,plan,compiled,method)
    options=merge(algorithm_options,plan.solver_options)
    started=time_ns()
    # Use the specialization's already budgeted operator directly. Calling
    # the compiled wrapper can convert a pre-existing sparse backend to a new
    # matrix-free compatibility operator for Krylov, retaining an additional
    # workspace which is neither needed nor part of this point's preflight.
    info=steady_state(compiled.operator;basis=_scan_basis(compiled),
        trace_vector=compiled.plan.tracevec,method,return_info=true,
        initial_state,workspace=solver_workspace,
        memory_budget=resources.solver_memory_budget,options...)
    solve_seconds=(time_ns()-started)/1e9
    state=PIState(_scan_basis(compiled),info.state)
    solver_info=Base.structdiff(info,(state=info.state,))
    continuation_state=plan.continuation ? copy(state) : nothing
    user=_scan_user_diagnostic(plan,state,parameter,index)
    plan.continuation&&_scan_store_seed!(workspace,plan,compiled,continuation_state)
    diagnostics=(solver=solver_info,user=user,backend=compiled.backend,
                 dimension=size(compiled,1),compile=compiled.estimates)
    state,solver_info,continuation_state,
        initial_state!==nothing,reused,solve_seconds,diagnostics
end

function _scan_point_spectrum!(plan,compiled,parameter,index,workspace,rng,
                               resources)
    seed_type=_scan_output_scalar_type(
        plan,size(compiled,1),eltype(compiled),resources.method)
    seed=_scan_compatible_seed(workspace,plan,compiled,seed_type)
    initial_vector=seed isa AbstractVector ? seed : nothing
    method=resources.method
    solver_workspace,reused,method=
        _scan_spectrum_workspace!(workspace,plan,compiled,method)
    initial_subspace=method in (:harmonic,:block_arnoldi)&&
        seed isa AbstractMatrix ? seed : nothing
    iterative=method in (:arnoldi,:block_arnoldi,:harmonic,:iram,:jd)
    started=time_ns()
    # The high-level spectral command performs its own preflight including
    # Base.summarysize(compiled) and one returned spectrum. Passing only the
    # outer residual solver allowance would count both objects twice. The
    # outer scan report has already enforced the aggregate multi-worker and
    # retained-history peak, so forward the complete point budget here.
    nested_memory_budget=plan.budget_disabled ? Inf : plan.memory_budget
    selected_algorithm=_scan_algorithm_is_auto(plan.algorithm) ?
        method : plan.algorithm
    result = if iterative
        seed_options=method===:block_arnoldi ? (;initial_subspace) :
            initial_subspace===nothing ? (;initial_vector) : (;initial_subspace)
        liouvillian_spectrum(compiled;target=plan.spectrum_target,nev=plan.nev,
            algorithm=selected_algorithm,vectors=true,return_info=true,
            seed_options...,workspace=solver_workspace,rng=rng,
            memory_budget=nested_memory_budget,
            plan.solver_options...)
    elseif plan.save_vectors
        liouvillian_spectrum(compiled;target=plan.spectrum_target,nev=plan.nev,
            algorithm=selected_algorithm,vectors=true,return_info=true,
            memory_budget=nested_memory_budget,
            plan.solver_options...)
    else
        values=liouvillian_spectrum(compiled;target=plan.spectrum_target,
            nev=plan.nev,algorithm=selected_algorithm,vectors=false,
            return_info=false,memory_budget=nested_memory_budget,
            plan.solver_options...)
        SpectrumResult(values,nothing,(method=method,dimension=size(compiled,1)))
    end
    solve_seconds=(time_ns()-started)/1e9
    info=result.info
    continuation_vector = if !iterative||result.vectors===nothing||
                             size(result.vectors,2)==0
        nothing
    elseif method in (:harmonic,:block_arnoldi)
        columns=method===:block_arnoldi ? min(size(result.vectors,2),
            Int(get(plan.solver_options,:block_size,min(plan.nev,4)))) :
            size(result.vectors,2)
        copy(view(result.vectors,:,1:columns))
    else
        # A single exactly converged eigenvector makes an ordinary Arnoldi
        # factorization break down after one step. Seed the next point with a
        # deterministic combination of all selected modes instead. This keeps
        # the continuation information while retaining components along the
        # complete requested slow subspace.
        vector=copy(view(result.vectors,:,1))
        R=_real_float_type(eltype(vector))
        for column in 2:size(result.vectors,2)
            weight=one(R)/R(column)
            @views @. vector=vector+weight*result.vectors[:,column]
        end
        norm_vector=norm(vector)
        iszero(norm_vector) ? copy(view(result.vectors,:,1)) : vector./norm_vector
    end
    plan.continuation&&continuation_vector!==nothing&&
        _scan_store_seed!(workspace,plan,compiled,continuation_vector)
    saved_result = if plan.save_vectors
        result
    else
        SpectrumResult(result.values,nothing,result.info)
    end
    user=_scan_user_diagnostic(plan,saved_result,parameter,index)
    diagnostics=(solver=info,user=user,backend=compiled.backend,
                 dimension=size(compiled,1),method=method,
                 compile=compiled.estimates)
    saved_result,info,plan.continuation ? continuation_vector : nothing,
        iterative&&(initial_vector!==nothing||initial_subspace!==nothing),reused,
        solve_seconds,diagnostics
end

@inline function _scan_point_rng(seed::UInt64,index::Int)
    # Stable index splitting makes serial, threaded, and distributed chunks
    # use identical random starting vectors.
    mixed=xor(seed,UInt64(index)*0x9e3779b97f4a7c15)
    Random.MersenneTwister(mixed)
end

function _scan_success_point!(plan,parameter,index,workspace,
                              active_workers::Int=1)
    total_started=time_ns();compile_started=time_ns()
    compiled=_scan_compile(plan,parameter,index,workspace,active_workers)
    compile_seconds=(time_ns()-compile_started)/1e9
    resources=_enforce_scan_resource_report(
        _scan_resource_report(plan,compiled,active_workers),index)
    result = if plan.task===:steady_state
        _scan_point_steady!(plan,compiled,parameter,index,workspace,resources)
    else
        _scan_point_spectrum!(plan,compiled,parameter,index,workspace,
            _scan_point_rng(plan.seed,index),resources)
    end
    output,info,restart_seed,warm,reused,solve_seconds,diagnostics=result
    diagnostics=merge(diagnostics,(resources=resources,))
    residual=_scan_maximum_residual(info)
    trace_error=_scan_info_value(info,:trace_error,missing)
    converged=_scan_converged(info)
    iterations=_scan_info_value(info,:iterations,missing)
    point=ParameterScanPoint(index,parameter,:success,output,residual,
        trace_error,converged,iterations,compile_seconds,solve_seconds,
        (time_ns()-total_started)/1e9,warm,reused,diagnostics,nothing,nothing)
    point,restart_seed
end

function _scan_failed_point(parameter,index,error,elapsed)
    ParameterScanPoint(index,parameter,:failed,nothing,missing,missing,false,
        missing,0.0,0.0,elapsed,false,false,nothing,
        string(typeof(error)),sprint(showerror,error))
end

function _scan_without_output(point::ParameterScanPoint)
    ParameterScanPoint(point.index,point.parameter,point.status,nothing,
        point.residual,point.trace_error,point.converged,point.iterations,
        point.compile_seconds,point.solve_seconds,point.elapsed_seconds,
        point.warm_started,point.workspace_reused,point.diagnostics,
        point.error_type,point.message)
end

function _scan_indices(plan,indices)
    indices===nothing&&return collect(eachindex(plan.parameters))
    selected=Int[index for index in indices]
    all(index->index in eachindex(plan.parameters),selected)||
        throw(BoundsError(plan.parameters,selected))
    issorted(selected)&&allunique(selected)||throw(ArgumentError(
        "scan indices must be strictly increasing and unique"))
    selected
end

function _scan_max_points(indices,max_points)
    max_points===nothing&&return indices
    max_points isa Integer&&!(max_points isa Bool)&&max_points>=0||throw(ArgumentError(
        "max_points must be a nonnegative integer or nothing"))
    indices[1:min(Int(max_points),length(indices))]
end

function _scan_callback(callback,point)
    callback===nothing&&return false
    applicable(callback,point)||throw(ArgumentError(
        "callback must accept one ParameterScanPoint"))
    value=callback(point)
    value===nothing&&return false
    value===:stop&&return true
    throw(ArgumentError("callback must return nothing or :stop"))
end

function _scan_result(plan,points,restart_index,restart_seed,started;
                      execution,requested_indices,stopped=false,
                      cancelled=false,
                      restart_signature=nothing,restart_eltype=nothing)
    saved_seed=plan.save_restart&&plan.continuation ?
        _copy_scan_seed(restart_seed) : nothing
    if saved_seed===nothing
        restart_index=0;restart_signature=nothing;restart_eltype=nothing
    elseif restart_signature===nothing&&saved_seed isa PIState
        restart_signature=(task=plan.task,
            basis=_scan_basis_signature(saved_seed.basis))
        restart_eltype=eltype(saved_seed.data)
    end
    metadata=(elapsed_seconds=(time_ns()-started)/1e9,execution,
        requested_indices=copy(requested_indices),continuation=plan.continuation,
        save_outputs=plan.save_outputs,save_vectors=plan.save_vectors,
        save_restart=plan.save_restart,memory_budget=plan.memory_budget,
        budget_disabled=plan.budget_disabled,stopped,cancelled,
        successful=count(point->point.status===:success,points),
        failed=count(point->point.status===:failed,points),
        restart_signature,restart_eltype)
    ParameterScanResult(plan.task,copy(plan.parameters),points,restart_index,
                        saved_seed,metadata)
end

function _serial_parameter_scan(plan,indices,workspace,callback,on_error,
                                progress_context)
    points=ParameterScanPoint[];started=time_ns();restart_index=0
    restart_seed=nothing;restart_signature=nothing;restart_eltype=nothing
    stopped=false;cancelled=false;total=length(indices)
    _progress_emit!(progress_context,:started,0,total;
        message="parameter scan started",metadata=(execution=:serial,))
    for index in indices
        if _progress_cancelled(progress_context)
            stopped=true;cancelled=true
            break
        end
        parameter=plan.parameters[index];point_started=time_ns()
        point=nothing;seed=nothing;caught=nothing
        try
            point,seed=_scan_success_point!(plan,parameter,index,workspace,1)
        catch error
            _scan_rethrow_interrupt(error)
            caught=error
            _scan_clear_seed!(workspace)
            point=_scan_failed_point(parameter,index,error,
                (time_ns()-point_started)/1e9)
        end
        push!(points,plan.save_outputs ? point : _scan_without_output(point))
        if point.status===:success
            restart_index=index;restart_seed=seed
            restart_signature=workspace.continuation_signature
            restart_eltype=workspace.continuation_eltype
        end
        callback_stop=_scan_callback(callback,point)
        event_cancelled=_progress_emit!(
            progress_context,:advanced,length(points),total;
            message="completed parameter index $index",
            metadata=(index,parameter,status=point.status,
                      converged=point.converged))
        if caught!==nothing
            on_error===:throw&&throw(caught)
            if on_error===:stop
                stopped=true
                break
            end
        end
        if callback_stop
            stopped=true
            break
        end
        if event_cancelled
            stopped=true;cancelled=true
            break
        end
    end
    if cancelled||_progress_cancelled(progress_context)
        cancelled=true;stopped=true
        _progress_cancel!(progress_context,length(points),total)
    elseif stopped
        _progress_emit!(progress_context,:stopped,length(points),total;
            message="parameter scan stopped")
    else
        _progress_emit!(progress_context,:completed,length(points),total;
            message="parameter scan completed")
    end
    _scan_result(plan,points,restart_index,restart_seed,started;
        execution=:serial,requested_indices=indices,stopped,
        cancelled,restart_signature,restart_eltype)
end

function _threaded_parameter_scan(plan,indices,callback,on_error,
                                  progress_context)
    plan.continuation&&throw(ArgumentError(
        "threaded scans cannot use path-dependent continuation; construct the plan with continuation=false"))
    started=time_ns();total=length(indices)
    _progress_emit!(progress_context,:started,0,total;
        message="parameter scan started",metadata=(execution=:threads,))
    if isempty(indices)||_progress_cancelled(progress_context)
        cancelled=_progress_cancelled(progress_context)
        cancelled ? _progress_cancel!(progress_context,0,total) :
            _progress_emit!(progress_context,:completed,0,total;
                message="parameter scan completed")
        return _scan_result(plan,ParameterScanPoint[],0,nothing,
            started;execution=:threads,requested_indices=indices,
            stopped=cancelled,cancelled)
    end

    # A fixed worker pool bounds live, unsaved outputs by O(nthreads) rather
    # than O(number of scan points). Each worker reuses its own Krylov scratch.
    # A worker waits for an acknowledgement before taking another job, which
    # also bounds the ordered-callback reorder buffer by the worker count.
    workers=min(Threads.nthreads(),length(indices))
    results=Channel{Any}(workers)
    acknowledgements=[Channel{Nothing}(0) for _ in 1:workers]
    stop_requested=Threads.Atomic{Bool}(false)
    tasks=Task[]
    for worker_index in 1:workers
        push!(tasks,Threads.@spawn begin
            try
                workspace=ParameterScanWorkspace()
                for position in worker_index:workers:length(indices)
                    stop_requested[]&&break
                    index=indices[position];parameter=plan.parameters[index]
                    point_started=time_ns();point=nothing
                    try
                        point,_=_scan_success_point!(plan,parameter,index,
                            workspace,workers)
                    catch error
                        _scan_rethrow_interrupt(error)
                        point=_scan_failed_point(parameter,index,error,
                            (time_ns()-point_started)/1e9)
                    end
                    put!(results,(:point,position,point,worker_index))
                    take!(acknowledgements[worker_index])
                end
            catch error
                # Report failures outside the per-point catch (for example a
                # workspace-construction or channel failure) before sending
                # the terminal marker.  The coordinator drains every worker
                # and releases outstanding acknowledgements before throwing.
                put!(results,(:worker_error,worker_index,error))
            finally
                put!(results,(:done,worker_index))
            end
        end)
    end

    pending=Dict{Int,Tuple{ParameterScanPoint,Int}}()
    points=ParameterScanPoint[];next_position=1;finished_workers=0
    accepting=true;stopped=false;cancelled=false;deferred_error=nothing
    while finished_workers<workers
        item=take!(results)
        if first(item)===:done
            finished_workers+=1
            continue
        elseif first(item)===:worker_error
            _,worker_index,error=item
            deferred_error===nothing&&(deferred_error=error)
            if accepting
                accepting=false;stopped=true;stop_requested[]=true
                # Every buffered point has already been published by a
                # worker that is waiting on exactly one acknowledgement.
                for (_,(_,pending_worker)) in pending
                    put!(acknowledgements[pending_worker],nothing)
                end
                empty!(pending)
            end
            continue
        end
        _,position,point,worker_index=item
        if accepting&&_progress_cancelled(progress_context)
            accepting=false;stopped=true;cancelled=true
            stop_requested[]=true
            for (_,(_,pending_worker)) in pending
                put!(acknowledgements[pending_worker],nothing)
            end
            empty!(pending)
        end
        if !accepting
            put!(acknowledgements[worker_index],nothing)
            continue
        end
        pending[position]=(point,worker_index)
        while accepting&&haskey(pending,next_position)
            ordered_point,ordered_worker=pop!(pending,next_position)
            callback_stop=false
            try
                callback_stop=_scan_callback(callback,ordered_point)
            catch error
                deferred_error=error;accepting=false;stopped=true
            end
            push!(points,plan.save_outputs ? ordered_point :
                _scan_without_output(ordered_point))
            event_cancelled=false
            if accepting
                try
                    event_cancelled=_progress_emit!(
                        progress_context,:advanced,length(points),total;
                        message="completed parameter index $(ordered_point.index)",
                        metadata=(index=ordered_point.index,
                                  parameter=ordered_point.parameter,
                                  status=ordered_point.status,
                                  converged=ordered_point.converged))
                catch error
                    deferred_error=error;accepting=false;stopped=true
                end
            end
            if accepting&&ordered_point.status===:failed&&on_error!==:record
                accepting=false;stopped=true
                if on_error===:throw
                    deferred_error=ErrorException(
                        "parameter scan failed at index $(ordered_point.index): $(ordered_point.message)")
                end
            elseif accepting&&callback_stop
                accepting=false;stopped=true
            elseif accepting&&event_cancelled
                accepting=false;stopped=true;cancelled=true
            end
            put!(acknowledgements[ordered_worker],nothing)
            next_position+=1
        end
        if !accepting
            stop_requested[]=true
            # Release at most one buffered output from every other worker.
            # Those workers observe the stop flag immediately after the
            # acknowledgement and do not start another scan point.
            for (_,(_,pending_worker)) in pending
                put!(acknowledgements[pending_worker],nothing)
            end
            empty!(pending)
        end
    end
    foreach(fetch,tasks)
    deferred_error===nothing||throw(deferred_error)
    if cancelled||_progress_cancelled(progress_context)
        cancelled=true;stopped=true
        _progress_cancel!(progress_context,length(points),total)
    elseif stopped
        _progress_emit!(progress_context,:stopped,length(points),total;
            message="parameter scan stopped")
    else
        _progress_emit!(progress_context,:completed,length(points),total;
            message="parameter scan completed")
    end
    _scan_result(plan,points,0,nothing,started;
        execution=:threads,requested_indices=indices,stopped,cancelled)
end

"""
    parameter_scan(plan; workspace=ParameterScanWorkspace(), callback=nothing,
                   indices=nothing, execution=:serial, on_error=:stop,
                   max_points=nothing, progress=false, on_event=nothing,
                   cancellation_token=nothing)

Execute a prepared parameter scan. Serial continuation warm-starts compatible
neighbouring points and reuses Krylov storage. `callback(point)` is invoked in
increasing index order with the complete point output even when
`plan.save_outputs == false`; returning `:stop` creates a resumable prefix.
Callbacks must return `nothing` to continue or `:stop` to stop.

Every public invocation starts a new continuation path: any seed left in the
supplied workspace by an earlier scan is cleared, while compatible Krylov
storage is still reused. Use [`resume_parameter_scan`](@ref) to continue an
existing path from its validated checkpoint.

`on_error=:stop` records the failed point and returns, `:record` records it
and continues without a warm start across the failure, and `:throw` reports
the point to the callback before rethrowing. Thus failures are never silently
skipped. `max_points` intentionally limits this invocation and is useful for
checkpoint intervals.

`execution=:threads` is deterministic by parameter index, uses one workspace
and random stream per point, and requires `continuation=false`. Its builder,
remaker, and diagnostic must themselves be thread safe. Ordered callbacks are
delivered after the parallel calculations. For multi-process execution, run
disjoint `indices` on workers and combine their checkpoint-neutral results
with [`merge_parameter_scan_results`](@ref).

Set `progress=true` for textual updates, pass an `IO` to `progress` to
redirect them, or use `on_event(event)` for structured [`ProgressEvent`](@ref)
records. The event callback may return `:cancel`, and a shared
[`CancellationToken`](@ref) may be cancelled from another task. Cancellation
is observed between parameter points and returns a resumable partial result
with `result.metadata.cancelled == true`.
"""
function _parameter_scan(plan::ParameterScanPlan;
        workspace::ParameterScanWorkspace=ParameterScanWorkspace(),
        callback=nothing,indices=nothing,execution::Symbol=:serial,
        on_error::Symbol=:stop,max_points=nothing,
        progress=false,on_event=nothing,cancellation_token=nothing,
        preserve_continuation_seed::Bool=false)
    execution in _SCAN_EXECUTIONS||throw(ArgumentError(
        "execution must be :serial or :threads"))
    on_error in _SCAN_FAILURE_POLICIES||throw(ArgumentError(
        "on_error must be :stop, :record, or :throw"))
    selected=_scan_max_points(_scan_indices(plan,indices),max_points)
    if plan.continuation&&!isempty(selected)
        all(diff(selected).==1)||throw(ArgumentError(
            "continuation requires a consecutive increasing index range"))
    end
    preserve_continuation_seed||_scan_clear_seed!(workspace)
    progress_context=_prepare_progress(:parameter_scan;
        progress,on_event,cancellation_token)
    execution===:serial ?
        _serial_parameter_scan(plan,selected,workspace,callback,on_error,
            progress_context) :
        _threaded_parameter_scan(plan,selected,callback,on_error,
            progress_context)
end

"""
    parameter_scan(plan; workspace=ParameterScanWorkspace(), callback=nothing,
                   indices=nothing, execution=:serial, on_error=:stop,
                   max_points=nothing, progress=false, on_event=nothing,
                   cancellation_token=nothing)

Execute a prepared parameter scan. Serial execution reuses compatible Krylov
storage and, within this invocation, warm-starts each continuation point from
the preceding success. A fresh public call clears any continuation seed left
in `workspace`; use [`resume_parameter_scan`](@ref) to continue a validated
checkpoint path. Threaded execution requires `continuation=false`.

The callback receives points in increasing index order and may return `:stop`.
Failures follow the explicit `on_error` policy, and `max_points` creates a
bounded resumable invocation. See [`ParameterScanPlan`](@ref) for output and
restart retention controls.
"""
function parameter_scan(plan::ParameterScanPlan;
        workspace::ParameterScanWorkspace=ParameterScanWorkspace(),
        callback=nothing,indices=nothing,execution::Symbol=:serial,
        on_error::Symbol=:stop,max_points=nothing,
        progress=false,on_event=nothing,cancellation_token=nothing)
    _parameter_scan(plan;workspace,callback,indices,execution,on_error,
        max_points,progress,on_event,cancellation_token,
        preserve_continuation_seed=false)
end

function _scan_validate_previous(plan,previous)
    previous.task===plan.task||throw(ArgumentError(
        "previous result task does not match the plan"))
    isequal(previous.parameters,plan.parameters)||throw(ArgumentError(
        "previous result parameters do not match the plan"))
    for name in (:continuation,:save_outputs,:save_vectors,:save_restart,
                 :memory_budget,:budget_disabled)
        hasproperty(previous.metadata,name)||throw(ArgumentError(
            "previous result metadata is missing $name"))
        getproperty(previous.metadata,name)==getproperty(plan,name)||
            throw(ArgumentError(
                "previous result $name setting does not match the plan"))
    end
    seen=Set{Int}()
    for point in previous.points
        point.index in eachindex(plan.parameters)||throw(ArgumentError(
            "previous result contains an out-of-range point index"))
        point.index in seen&&throw(ArgumentError(
            "previous result contains duplicate point index $(point.index)"))
        isequal(point.parameter,plan.parameters[point.index])||
            throw(ArgumentError(
                "previous result parameter differs at index $(point.index)"))
        push!(seen,point.index)
    end
    previous
end

function _scan_prefix(points)
    byindex=Dict(point.index=>point for point in points)
    prefix=0
    while haskey(byindex,prefix+1)&&byindex[prefix+1].status===:success
        prefix+=1
    end
    prefix
end

"""
    resume_parameter_scan(plan, previous; retry_failed=true, kwargs...)

Resume missing points of a checkpoint-neutral [`ParameterScanResult`](@ref).
The parameter sequence, task, indexes, and stored parameter values are checked
strictly before any model is built. The continuation and output/restart
retention settings must also match the checkpoint-producing plan, preventing
mixed-history results. Failed points are retried by default.

Continuation plans require the retained successful records to form a prefix;
the stored final restart seed is used when available. A result created with
`save_restart=false` remains resumable, but the first new point starts cold.
Additional keywords are forwarded to [`parameter_scan`](@ref); execution must
remain serial for continuation.
"""
function resume_parameter_scan(plan::ParameterScanPlan,
        previous::ParameterScanResult;retry_failed::Bool=true,
        workspace::ParameterScanWorkspace=ParameterScanWorkspace(),kwargs...)
    _scan_validate_previous(plan,previous)
    # A resume may reuse solver allocations, but it may never inherit a seed
    # from an unrelated prior invocation of this workspace.
    _scan_clear_seed!(workspace)
    retained=retry_failed ?
        [point for point in previous.points if point.status===:success] :
        copy(previous.points)
    occupied=Set(point.index for point in retained)
    missing=[index for index in eachindex(plan.parameters) if !(index in occupied)]
    prior_restart_index=0;prior_restart_seed=nothing
    prior_restart_signature=nothing;prior_restart_eltype=nothing
    if plan.continuation
        prefix=_scan_prefix(retained)
        all(point->point.index<=prefix,retained)||throw(ArgumentError(
            "continuation can resume only a successful prefix; discard out-of-order chunks or use continuation=false"))
        missing=collect(prefix+1:length(plan.parameters))
        if prefix>0&&previous.restart_index==prefix&&previous.restart_seed!==nothing
            workspace.continuation_seed=_copy_scan_seed(previous.restart_seed)
            workspace.continuation_signature=previous.metadata.restart_signature
            workspace.continuation_eltype=previous.metadata.restart_eltype
            prior_restart_index=prefix
            prior_restart_seed=_copy_scan_seed(previous.restart_seed)
            prior_restart_signature=previous.metadata.restart_signature
            prior_restart_eltype=previous.metadata.restart_eltype
        end
    end
    if isempty(missing)
        completed_points=sort(retained;by=point->point.index)
        completed_metadata=merge(previous.metadata,(
            resumed=true,stopped=false,cancelled=false,
            successful=count(point->point.status===:success,completed_points),
            failed=count(point->point.status===:failed,completed_points)))
        return ParameterScanResult(plan.task,copy(plan.parameters),
            completed_points,prior_restart_index,
            _copy_scan_seed(prior_restart_seed),completed_metadata)
    end
    extension=_parameter_scan(plan;workspace,indices=missing,kwargs...,
        preserve_continuation_seed=true)
    base=ParameterScanResult(plan.task,copy(plan.parameters),retained,
        prior_restart_index,_copy_scan_seed(prior_restart_seed),
        (elapsed_seconds=0.0,execution=:resume_base,requested_indices=Int[],
         continuation=plan.continuation,save_outputs=plan.save_outputs,
         save_vectors=plan.save_vectors,save_restart=plan.save_restart,
         memory_budget=plan.memory_budget,budget_disabled=plan.budget_disabled,
         stopped=false,cancelled=false,
         successful=count(point->point.status===:success,retained),
         failed=count(point->point.status===:failed,retained),
         restart_signature=prior_restart_signature,
         restart_eltype=prior_restart_eltype))
    merged=_merge_parameter_scan_results(plan,(base,extension);
        allow_continuation_chain=true)
    ParameterScanResult(merged.task,merged.parameters,merged.points,
        merged.restart_index,_copy_scan_seed(merged.restart_seed),
        merge(merged.metadata,(resumed=true,)))
end

"""
    merge_parameter_scan_results(plan, results...)

Join nonoverlapping serial, threaded, or distributed scan chunks in parameter
index order. Every result is validated against `plan`; duplicate indexes are
rejected rather than resolved implicitly. The returned object contains no
compiled model or workspace. Multiple independently produced chunks are
accepted only for `continuation=false`: path-dependent continuation chunks do
not prove a common restart chain and must instead be joined through
[`resume_parameter_scan`](@ref).
"""
function _merge_parameter_scan_results(plan::ParameterScanPlan,results;
        allow_continuation_chain::Bool=false)
    isempty(results)&&throw(ArgumentError("at least one result is required"))
    plan.continuation&&length(results)>1&&!allow_continuation_chain&&
        throw(ArgumentError(
            "independent continuation chunks cannot be merged safely; resume from a validated prefix instead"))
    points=ParameterScanPoint[];seen=Set{Int}();elapsed=0.0
    restart_index=0;restart_seed=nothing;restart_signature=nothing
    restart_eltype=nothing
    for result in results
        _scan_validate_previous(plan,result)
        elapsed+=result.metadata.elapsed_seconds
        for point in result.points
            point.index in seen&&throw(ArgumentError(
                "cannot merge duplicate point index $(point.index)"))
            push!(seen,point.index);push!(points,point)
        end
        if result.restart_index>restart_index&&result.restart_seed!==nothing
            restart_index=result.restart_index
            restart_seed=_copy_scan_seed(result.restart_seed)
            restart_signature=result.metadata.restart_signature
            restart_eltype=result.metadata.restart_eltype
        end
    end
    sort!(points;by=point->point.index)
    metadata=(elapsed_seconds=elapsed,execution=:merged,
        requested_indices=[point.index for point in points],
        continuation=plan.continuation,save_outputs=plan.save_outputs,
        save_vectors=plan.save_vectors,save_restart=plan.save_restart,
        memory_budget=plan.memory_budget,budget_disabled=plan.budget_disabled,
        stopped=length(points)<length(plan.parameters),
        cancelled=any(result->hasproperty(result.metadata,:cancelled)&&
                              result.metadata.cancelled,results),
        successful=count(point->point.status===:success,points),
        failed=count(point->point.status===:failed,points),
        restart_signature,restart_eltype)
    ParameterScanResult(plan.task,copy(plan.parameters),points,restart_index,
                        restart_seed,metadata)
end

"""
    merge_parameter_scan_results(plan, results...)

Join nonoverlapping checkpoint-neutral scan chunks in parameter-index order.
Every result's task, parameter sequence, stored indexes, and indexed values
are validated, and duplicate indexes raise. Multiple chunks are accepted only
when `plan.continuation == false`; use [`resume_parameter_scan`](@ref) for a
path-dependent continuation.
"""
function merge_parameter_scan_results(plan::ParameterScanPlan,
        results::ParameterScanResult...)
    _merge_parameter_scan_results(plan,results;
        allow_continuation_chain=false)
end

"""
    distributed_parameter_scan(plan; workers=Distributed.workers(),
                               indices=nothing, callback=nothing,
                               on_error=:stop)

Execute independent parameter points on deterministic, disjoint worker
chunks. This optional method becomes available after loading the Distributed
stdlib. The plan must have `continuation=false`: a path-dependent continuation
cannot be partitioned across workers without changing its initial states.

Workers load `PermutationalInvariantDynamics` in their active project, run
serial chunks with index-derived random streams, and return checkpoint-neutral
records. Failure policy and callbacks are then applied on the master in global
index order, and duplicate or inconsistent records are rejected by
[`merge_parameter_scan_results`](@ref). A master callback requires the live
point outputs to cross the process boundary and is therefore accepted only
when `save_outputs=true`. Prefer a scalar `diagnostic` in the plan when large
states should remain off the master.

Each worker must use an environment containing a compatible package version,
and the builder, remaker, parameter values, algorithm, and diagnostic must be
serializable. Loading Distributed does not add it to the core dependency set.
"""
function distributed_parameter_scan end

function _scan_row(point::ParameterScanPoint,include_output::Bool)
    basic=(index=point.index,parameter=point.parameter,status=point.status,
        residual=point.residual,trace_error=point.trace_error,
        converged=point.converged,iterations=point.iterations,
        compile_seconds=point.compile_seconds,solve_seconds=point.solve_seconds,
        elapsed_seconds=point.elapsed_seconds,warm_started=point.warm_started,
        workspace_reused=point.workspace_reused,error_type=point.error_type,
        message=point.message)
    include_output ? merge(basic,(output=point.output,
                                  diagnostics=point.diagnostics)) : basic
end

"""
    parameter_scan_rows(result; include_output=false)

Return a stable vector of named-tuple rows suitable for display, serialization,
or an optional `Tables.rowtable`-style adapter. Core code does not depend on
Tables.jl. Set `include_output=true` to include potentially large states,
eigenvectors, and nested diagnostics.
"""
parameter_scan_rows(result::ParameterScanResult;include_output::Bool=false)=
    [_scan_row(point,include_output) for point in result.points]

"""
    parameter_scan_columns(result; include_output=false)

Return the same scalar metadata as [`parameter_scan_rows`](@ref) in a named
tuple of columns. This dependency-free representation can be passed directly
to many data and plotting tools. Potentially large numerical outputs are
excluded unless `include_output=true`.
"""
function parameter_scan_columns(result::ParameterScanResult;
                                include_output::Bool=false)
    rows=parameter_scan_rows(result;include_output)
    base_names=(:index,:parameter,:status,:residual,:trace_error,:converged,
        :iterations,:compile_seconds,:solve_seconds,:elapsed_seconds,
        :warm_started,:workspace_reused,:error_type,:message)
    names=isempty(rows) ? (include_output ?
        (base_names...,:output,:diagnostics) : base_names) :
        propertynames(first(rows))
    NamedTuple{names}(Tuple([getproperty(row,name) for row in rows]
                            for name in names))
end
