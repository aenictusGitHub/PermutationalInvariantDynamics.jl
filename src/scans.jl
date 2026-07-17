"""
    ParameterScanPlan(parameters, model_builder; task=:steady_state, ...)

Immutable, shareable description of a parameter scan. `model_builder` is
called as either `model_builder(parameter)` or
`model_builder(parameter, index)` and must return a [`PIModel`](@ref) or an
already [`CompiledPIModel`](@ref). Models are compiled point by point because
their prepared kernels may depend on the scanned parameter; a returned
compiled model is used directly. The default compilation backend is
`:matrixfree`; override `compile_options` explicitly for a materialized scan.

`task=:steady_state` calls [`stationary_state`](@ref), while `task=:spectrum`
calls [`liouvillian_spectrum`](@ref). The corresponding solver is selected by
`algorithm`. `compile_options` and `solver_options` must be named tuples.
Internally managed keywords such as `initial_state`, `workspace`, and
`return_info` are rejected instead of being silently overridden.

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
        diagnostic=nothing,seed::Integer=0)
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
    if task===:steady_state
        scan_method,_=_algorithm_options(algorithm)
        scan_method in (:auto,:direct,:svd,:eigen,:shiftinvert,
                        :shift_invert,:inverse_iteration,:krylov,:gmres)||
            throw(ArgumentError("unsupported steady-state scan algorithm $scan_method"))
        _scan_forbid_options(soptions,
            (:return_info,:initial_state,:workspace),"solver_options")
        if algorithm isa GMRESAlgorithm
            _scan_forbid_options(soptions,(:krylovdim,),"solver_options")
        end
    else
        scan_method,_=_spectrum_algorithm(
            algorithm,spectrum_target,max(nev_int,2),nev_int)
        scan_method in (:dense,:krylov,:arnoldi,:harmonic,:iram,
                        :implicit_qr,:jd,:jacobi_davidson)||
            throw(ArgumentError("unsupported spectral scan algorithm $scan_method"))
        _scan_forbid_options(soptions,
            (:return_info,:initial_vector,:workspace,:rng,:vectors,:nev,
             :subspace_dim,:target),
            "solver_options")
        if algorithm isa HarmonicArnoldiAlgorithm
            algorithm.nev==nev_int||throw(ArgumentError(
                "nev must match HarmonicArnoldiAlgorithm.nev"))
            _scan_forbid_options(soptions,
                (:krylovdim,:thickdim,:maxrestarts),"solver_options")
        end
    end
    ParameterScanPlan(values,model_builder,task,algorithm,coptions,soptions,
        continuation,save_outputs,save_vectors,save_restart,spectrum_target,
        nev_int,diagnostic,_scan_seed(seed))
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

"""
    ParameterScanWorkspace()

Task-owned mutable continuation and Krylov scratch for
[`parameter_scan`](@ref). Reuse it sequentially across compatible scans, but
never from concurrent tasks. Krylov storage is discarded and rebuilt whenever
the task, PI coordinate dimension, scalar type, or requested subspace size
changes.
"""
mutable struct ParameterScanWorkspace
    continuation_seed::Any
    continuation_signature::Any
    continuation_eltype::Any
    solver_workspace::Any
    solver_signature::Any
end
ParameterScanWorkspace()=ParameterScanWorkspace(nothing,nothing,nothing,nothing,nothing)

"""Clear all continuation and solver scratch retained by `workspace`."""
function clear_parameter_scan_workspace!(workspace::ParameterScanWorkspace)
    workspace.continuation_seed=nothing
    workspace.continuation_signature=nothing
    workspace.continuation_eltype=nothing
    workspace.solver_workspace=nothing
    workspace.solver_signature=nothing
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
    print(io,"ParameterScanPlan($(length(plan.parameters)) points, task=$(plan.task), continuation=$(plan.continuation))")
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

function _scan_compile(plan,parameter,index)
    source=_scan_model(plan,parameter,index)
    if source isa CompiledPIModel
        source
    elseif source isa PIModel
        compile(source;plan.compile_options...)
    else
        throw(ArgumentError(
            "model_builder returned $(typeof(source)) at index $index; expected PIModel or CompiledPIModel"))
    end
end

@inline _scan_basis(compiled::CompiledPIModel)=compiled.plan.basis
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

function _scan_steady_workspace!(workspace,plan,compiled)
    method=_scan_steady_method(plan.algorithm)
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
    krylovdim=plan.algorithm isa GMRESAlgorithm ? plan.algorithm.krylovdim :
              Int(get(plan.solver_options,:krylovdim,30))
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

function _scan_spectrum_workspace!(workspace,plan,compiled)
    n=size(compiled,1);method,_=_scan_spectrum_method(plan,n)
    method in (:krylov,:arnoldi,:harmonic,:iram,:implicit_qr,
               :jd,:jacobi_davidson)||return (nothing,false,method)
    T=_complex_float_type(eltype(compiled))
    if method in (:jd,:jacobi_davidson)
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
    elseif method in (:iram,:implicit_qr,:harmonic)
        max(30,3plan.nev+6)
    else
        max(20,2plan.nev+4)
    end
    krylovdim=Int(get(plan.solver_options,:krylovdim,default_dim))
    signature=(:spectrum_arnoldi,n,T,min(n,krylovdim))
    if workspace.solver_signature==signature&&
       workspace.solver_workspace isa ArnoldiWorkspace
        return (workspace.solver_workspace,true,method)
    end
    workspace.solver_workspace=ArnoldiWorkspace(T,n,krylovdim)
    workspace.solver_signature=signature
    workspace.solver_workspace,false,method
end

@inline _scan_continuation_signature(plan,compiled)=
    (task=plan.task,basis=_scan_basis_signature(_scan_basis(compiled)))

function _scan_compatible_seed(workspace,plan,compiled)
    plan.continuation||return nothing
    signature=_scan_continuation_signature(plan,compiled)
    T=_complex_float_type(eltype(compiled))
    workspace.continuation_signature==signature&&
        workspace.continuation_eltype===T ? workspace.continuation_seed : nothing
end

function _scan_store_seed!(workspace,plan,compiled,seed)
    workspace.continuation_seed=seed
    workspace.continuation_signature=_scan_continuation_signature(plan,compiled)
    workspace.continuation_eltype=_complex_float_type(eltype(compiled))
    workspace
end

function _scan_clear_seed!(workspace)
    workspace.continuation_seed=nothing
    workspace.continuation_signature=nothing
    workspace.continuation_eltype=nothing
    workspace
end

# A restart seed crosses an ownership boundary whenever it moves between a
# task-owned workspace and a checkpoint-neutral public result.  Keep those
# arrays detached: users may serialize or modify a result while reusing the
# workspace for another scan.
_copy_scan_seed(::Nothing)=nothing
_copy_scan_seed(seed::PIState)=copy(seed)
_copy_scan_seed(seed::AbstractVector)=copy(seed)
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

function _scan_point_steady!(plan,compiled,parameter,index,workspace)
    _,algorithm_options=_algorithm_options(plan.algorithm)
    method=_scan_steady_method(plan.algorithm)
    iterative=method in (:krylov,:shiftinvert)
    seed=iterative ? _scan_compatible_seed(workspace,plan,compiled) : nothing
    initial_state = if seed isa PIState
        PIState(_scan_basis(compiled),seed.data)
    elseif seed===nothing
        nothing
    else
        throw(ArgumentError("steady-state continuation seed has an incompatible type"))
    end
    solver_workspace,reused=_scan_steady_workspace!(workspace,plan,compiled)
    options=merge(algorithm_options,plan.solver_options)
    started=time_ns()
    info=steady_state(compiled;method,return_info=true,
        initial_state,workspace=solver_workspace,options...)
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

function _scan_point_spectrum!(plan,compiled,parameter,index,workspace,rng)
    seed=_scan_compatible_seed(workspace,plan,compiled)
    initial_vector=seed isa AbstractVector ? seed : nothing
    solver_workspace,reused,method=_scan_spectrum_workspace!(workspace,plan,compiled)
    iterative=method in (:krylov,:arnoldi,:harmonic,:iram,:implicit_qr,
                         :jd,:jacobi_davidson)
    started=time_ns()
    result = if iterative
        liouvillian_spectrum(compiled;target=plan.spectrum_target,nev=plan.nev,
            algorithm=plan.algorithm,vectors=true,return_info=true,
            initial_vector=initial_vector,workspace=solver_workspace,rng=rng,
            plan.solver_options...)
    elseif plan.save_vectors
        liouvillian_spectrum(compiled;target=plan.spectrum_target,nev=plan.nev,
            algorithm=plan.algorithm,vectors=true,return_info=true,
            plan.solver_options...)
    else
        values=liouvillian_spectrum(compiled;target=plan.spectrum_target,
            nev=plan.nev,algorithm=plan.algorithm,vectors=false,
            return_info=false,plan.solver_options...)
        SpectrumResult(values,nothing,(method=method,dimension=size(compiled,1)))
    end
    solve_seconds=(time_ns()-started)/1e9
    info=result.info
    continuation_vector = if !iterative||result.vectors===nothing||
                             size(result.vectors,2)==0
        nothing
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
        iterative&&initial_vector!==nothing,reused,
        solve_seconds,diagnostics
end

@inline function _scan_point_rng(seed::UInt64,index::Int)
    # Stable index splitting makes serial, threaded, and distributed chunks
    # use identical random starting vectors.
    mixed=xor(seed,UInt64(index)*0x9e3779b97f4a7c15)
    Random.MersenneTwister(mixed)
end

function _scan_success_point!(plan,parameter,index,workspace)
    total_started=time_ns();compile_started=time_ns()
    compiled=_scan_compile(plan,parameter,index)
    compile_seconds=(time_ns()-compile_started)/1e9
    result = if plan.task===:steady_state
        _scan_point_steady!(plan,compiled,parameter,index,workspace)
    else
        _scan_point_spectrum!(plan,compiled,parameter,index,workspace,
            _scan_point_rng(plan.seed,index))
    end
    output,info,restart_seed,warm,reused,solve_seconds,diagnostics=result
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
        save_restart=plan.save_restart,stopped,
        successful=count(point->point.status===:success,points),
        failed=count(point->point.status===:failed,points),
        restart_signature,restart_eltype)
    ParameterScanResult(plan.task,copy(plan.parameters),points,restart_index,
                        saved_seed,metadata)
end

function _serial_parameter_scan(plan,indices,workspace,callback,on_error)
    points=ParameterScanPoint[];started=time_ns();restart_index=0
    restart_seed=nothing;restart_signature=nothing;restart_eltype=nothing
    stopped=false
    for index in indices
        parameter=plan.parameters[index];point_started=time_ns()
        point=nothing;seed=nothing;caught=nothing
        try
            point,seed=_scan_success_point!(plan,parameter,index,workspace)
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
    end
    _scan_result(plan,points,restart_index,restart_seed,started;
        execution=:serial,requested_indices=indices,stopped,
        restart_signature,restart_eltype)
end

function _threaded_parameter_scan(plan,indices,callback,on_error)
    plan.continuation&&throw(ArgumentError(
        "threaded scans cannot use path-dependent continuation; construct the plan with continuation=false"))
    started=time_ns()
    isempty(indices)&&return _scan_result(plan,ParameterScanPoint[],0,nothing,
        started;execution=:threads,requested_indices=indices,stopped=false)

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
                        point,_=_scan_success_point!(plan,parameter,index,workspace)
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
    accepting=true;stopped=false;deferred_error=nothing
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
            if accepting&&ordered_point.status===:failed&&on_error!==:record
                accepting=false;stopped=true
                if on_error===:throw
                    deferred_error=ErrorException(
                        "parameter scan failed at index $(ordered_point.index): $(ordered_point.message)")
                end
            elseif accepting&&callback_stop
                accepting=false;stopped=true
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
    _scan_result(plan,points,0,nothing,started;
        execution=:threads,requested_indices=indices,stopped)
end

"""
    parameter_scan(plan; workspace=ParameterScanWorkspace(), callback=nothing,
                   indices=nothing, execution=:serial, on_error=:stop,
                   max_points=nothing)

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
"""
function _parameter_scan(plan::ParameterScanPlan;
        workspace::ParameterScanWorkspace=ParameterScanWorkspace(),
        callback=nothing,indices=nothing,execution::Symbol=:serial,
        on_error::Symbol=:stop,max_points=nothing,
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
    execution===:serial ?
        _serial_parameter_scan(plan,selected,workspace,callback,on_error) :
        _threaded_parameter_scan(plan,selected,callback,on_error)
end

"""
    parameter_scan(plan; workspace=ParameterScanWorkspace(), callback=nothing,
                   indices=nothing, execution=:serial, on_error=:stop,
                   max_points=nothing)

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
        on_error::Symbol=:stop,max_points=nothing)
    _parameter_scan(plan;workspace,callback,indices,execution,on_error,
        max_points,preserve_continuation_seed=false)
end

function _scan_validate_previous(plan,previous)
    previous.task===plan.task||throw(ArgumentError(
        "previous result task does not match the plan"))
    isequal(previous.parameters,plan.parameters)||throw(ArgumentError(
        "previous result parameters do not match the plan"))
    for name in (:continuation,:save_outputs,:save_vectors,:save_restart)
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
    isempty(missing)&&return ParameterScanResult(plan.task,copy(plan.parameters),
        sort(retained;by=point->point.index),prior_restart_index,
        _copy_scan_seed(prior_restart_seed),
        merge(previous.metadata,(resumed=true,)))
    extension=_parameter_scan(plan;workspace,indices=missing,kwargs...,
        preserve_continuation_seed=true)
    base=ParameterScanResult(plan.task,copy(plan.parameters),retained,
        prior_restart_index,_copy_scan_seed(prior_restart_seed),
        (elapsed_seconds=0.0,execution=:resume_base,requested_indices=Int[],
         continuation=plan.continuation,save_outputs=plan.save_outputs,
         save_vectors=plan.save_vectors,save_restart=plan.save_restart,
         stopped=false,successful=count(point->point.status===:success,retained),
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
        stopped=length(points)<length(plan.parameters),
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
