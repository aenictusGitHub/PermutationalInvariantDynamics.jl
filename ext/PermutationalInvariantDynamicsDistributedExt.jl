module PermutationalInvariantDynamicsDistributedExt

using PermutationalInvariantDynamics
using Random
import Distributed
import PermutationalInvariantDynamics: distributed_parameter_scan,
    distributed_quantum_trajectories, distributed_diffusive_trajectories

const PID=PermutationalInvariantDynamics

function _distributed_scan_plan(plan::PID.ParameterScanPlan,save_outputs::Bool)
    PID.ParameterScanPlan(plan.parameters,plan.model_builder;
        task=plan.task,algorithm=plan.algorithm,
        compile_options=plan.compile_options,
        solver_options=plan.solver_options,
        continuation=false,save_outputs,
        save_vectors=plan.save_vectors,save_restart=plan.save_restart,
        spectrum_target=plan.spectrum_target,nev=plan.nev,
        diagnostic=plan.diagnostic,seed=plan.seed)
end

function _distributed_scan_chunk(plan,indices)
    PID.parameter_scan(plan;indices,execution=:serial,on_error=:record)
end

function _distributed_workers(requested,label::AbstractString)
    workers=Int[worker for worker in requested]
    allunique(workers)||throw(ArgumentError("workers must be unique"))
    sort!(workers)
    isempty(workers)&&throw(ArgumentError(
        "$label requires at least one worker process"))
    Distributed.myid() in workers&&throw(ArgumentError(
        "the master process is not a scan worker; call Distributed.addprocs "*
        "and pass the returned worker ids"))
    active=Set(Distributed.workers())
    all(worker->worker in active,workers)||throw(ArgumentError(
        "workers must contain only active Distributed worker process ids"))
    workers
end

_distributed_workers(requested)=
    _distributed_workers(requested,"distributed_parameter_scan")

function _distributed_chunks(indices,nchunks)
    count=length(indices);base,extra=divrem(count,nchunks)
    chunks=Vector{Vector{Int}}(undef,nchunks);first_position=1
    for chunk_index in 1:nchunks
        chunk_length=base+(chunk_index<=extra ? 1 : 0)
        last_position=first_position+chunk_length-1
        chunks[chunk_index]=indices[first_position:last_position]
        first_position=last_position+1
    end
    chunks
end

function _ensure_worker_package(worker)
    expression=:(begin
        using Distributed
        using PermutationalInvariantDynamics
        nothing
    end)
    try
        Distributed.remotecall_fetch(Core.eval,worker,Main,expression)
    catch error
        throw(ArgumentError(
            "worker $worker could not load PermutationalInvariantDynamics "*
            "from its active project: $(sprint(showerror,error))"))
    end
end

function _fetch_distributed_chunks(plan,chunks,workers)
    tasks=Task[]
    for (worker,chunk) in zip(workers,chunks)
        push!(tasks,@async try
            Distributed.remotecall_fetch(
                _distributed_scan_chunk,worker,plan,chunk)
        catch error
            throw(ArgumentError(
                "distributed scan chunk on worker $worker failed before "*
                "returning point records; ensure that the plan and its "*
                "builder, remaker, parameters, algorithm, and diagnostic "*
                "are serializable: $(sprint(showerror,error))"))
        end)
    end
    output=Vector{Any}(undef,length(tasks));failure=nothing
    for index in eachindex(tasks)
        try
            output[index]=fetch(tasks[index])
        catch error
            failure===nothing&&(failure=error)
        end
    end
    failure===nothing||throw(failure)
    output
end

function _distributed_finalize(plan,merged,workers,chunks,callback,on_error)
    points=PID.ParameterScanPoint[];stopped=false
    for point in merged.points
        callback_stop=PID._scan_callback(callback,point)
        push!(points,plan.save_outputs ? point : PID._scan_without_output(point))
        if point.status===:failed&&on_error!==:record
            stopped=true
            on_error===:throw&&throw(ErrorException(
                "parameter scan failed at index $(point.index): $(point.message)"))
            break
        elseif callback_stop
            stopped=true
            break
        end
    end
    metadata=merge(merged.metadata,(
        execution=:distributed,
        workers=copy(workers),
        chunks=map(copy,chunks),
        stopped,
        successful=count(point->point.status===:success,points),
        failed=count(point->point.status===:failed,points),
    ))
    PID.ParameterScanResult(plan.task,copy(plan.parameters),points,0,nothing,
                            metadata)
end

function distributed_parameter_scan(plan::PID.ParameterScanPlan;
        workers=Distributed.workers(),indices=nothing,callback=nothing,
        on_error::Symbol=:stop)
    plan.continuation&&throw(ArgumentError(
        "distributed parameter scans require continuation=false"))
    on_error in (:stop,:record,:throw)||throw(ArgumentError(
        "on_error must be :stop, :record, or :throw"))
    callback!==nothing&&!plan.save_outputs&&throw(ArgumentError(
        "a master-side distributed callback requires save_outputs=true; "*
        "use plan.diagnostic for worker-side scalar streaming without "*
        "transferring and retaining all numerical outputs"))
    selected=PID._scan_indices(plan,indices)
    isempty(selected)&&return PID.parameter_scan(plan;indices=Int[],
        execution=:serial,on_error)
    worker_ids=_distributed_workers(workers)
    active_workers=worker_ids[1:min(length(worker_ids),length(selected))]
    foreach(_ensure_worker_package,active_workers)
    chunks=_distributed_chunks(selected,length(active_workers))

    # Numerical outputs cross the process boundary only after the user has
    # explicitly selected history retention in the plan.
    remote_plan=_distributed_scan_plan(plan,plan.save_outputs)
    chunk_results=_fetch_distributed_chunks(remote_plan,chunks,active_workers)
    merged=PID.merge_parameter_scan_results(plan,chunk_results...)
    _distributed_finalize(plan,merged,active_workers,chunks,callback,on_error)
end

function _distributed_quantum_chunk(model,rho0,times,seeds,keywords)
    plan=PID._trajectory_plan_for_state(model,rho0)
    workspace=PID.TrajectoryWorkspace(plan,rho0)
    R=PID._real_float_type(eltype(rho0.data))
    ts,options=PID._prepare_trajectory_arguments(times,R;
        dt=keywords.dt,parameters=keywords.parameters,
        max_jump_probability=keywords.max_jump_probability,
        algorithm=keywords.algorithm,abstol=keywords.abstol,
        reltol=keywords.reltol,dtmin=keywords.dtmin,dtmax=keywords.dtmax,
        event_time_tolerance=keywords.event_time_tolerance)
    PID._validate_trajectory_initial_state(plan,rho0)
    rng=MersenneTwister(0)
    [begin
         Random.seed!(rng,path_seed)
         PID._quantum_trajectory_prepared(
             plan,rho0,copy(ts),workspace,rng,options)
     end for path_seed in seeds]
end

function _distributed_diffusive_chunk(model,rho0,times,monitors,seeds,keywords)
    batch=PID.DiffusiveBatchPlan(model,rho0,times,monitors;
        dt=keywords.dt,observables=keywords.observables)
    workspace=PID.DiffusiveWorkspace(batch.plan,rho0)
    rng=MersenneTwister(0)
    [begin
         Random.seed!(rng,path_seed)
         PID.diffusive_trajectory(batch,rho0;rng,workspace,
             parameters=keywords.parameters,save_states=keywords.save_states)
     end for path_seed in seeds]
end

function _fetch_distributed_trajectory_chunks(remote_function,model,rho0,
        times,chunks,seeds,workers,label,extra_arguments,keywords)
    tasks=Task[]
    for (worker,chunk) in zip(workers,chunks)
        seed_chunk=seeds[chunk]
        arguments=(model,rho0,times,extra_arguments...,seed_chunk,keywords)
        push!(tasks,@async try
            Distributed.remotecall_fetch(remote_function,worker,arguments...)
        catch error
            throw(ArgumentError(
                "$label chunk on worker $worker failed; ensure that the "*
                "model, state, schedules, parameters, monitors, and "*
                "observables are serializable: $(sprint(showerror,error))"))
        end)
    end
    output=Vector{Any}(undef,length(tasks));failure=nothing
    for index in eachindex(tasks)
        try
            output[index]=fetch(tasks[index])
        catch error
            failure===nothing&&(failure=error)
        end
    end
    failure===nothing||throw(failure)
    merged=copy(first(output))
    for chunk_output in @view output[2:end]
        append!(merged,chunk_output)
    end
    merged
end

function _prepare_distributed_trajectory_run(n,seed,workers,label)
    n isa Integer&&n>0||throw(ArgumentError(
        "trajectory count must be a positive integer"))
    worker_ids=_distributed_workers(workers,label)
    active=worker_ids[1:min(length(worker_ids),Int(n))]
    foreach(_ensure_worker_package,active)
    indices=collect(1:Int(n))
    chunks=_distributed_chunks(indices,length(active))
    master=MersenneTwister(seed)
    seeds=rand(master,UInt64,Int(n))
    active,chunks,seeds
end

function distributed_quantum_trajectories(model::PID.PIModel,
        rho0::PID.PIState,times,n::Integer;
        workers=Distributed.workers(),seed::Integer=0,dt::Real,
        parameters=nothing,max_jump_probability=nothing,
        algorithm::Symbol=:fixed,abstol=nothing,reltol=nothing,
        dtmin=nothing,dtmax=nothing,event_time_tolerance=nothing)
    worker_ids,chunks,seeds=_prepare_distributed_trajectory_run(
        n,seed,workers,"distributed_quantum_trajectories")
    raw_times=collect(times)
    keywords=(;dt,parameters,max_jump_probability,algorithm,abstol,reltol,
              dtmin,dtmax,event_time_tolerance)
    _fetch_distributed_trajectory_chunks(_distributed_quantum_chunk,
        model,rho0,raw_times,chunks,seeds,worker_ids,
        "distributed quantum trajectory",(),keywords)
end

function distributed_diffusive_trajectories(model::PID.PIModel,
        rho0::PID.PIState,times,monitors,n::Integer;
        workers=Distributed.workers(),seed::Integer=0,dt::Real,
        parameters=nothing,observables=nothing,save_states::Bool=true)
    worker_ids,chunks,seeds=_prepare_distributed_trajectory_run(
        n,seed,workers,"distributed_diffusive_trajectories")
    raw_times=collect(times)
    keywords=(;dt,parameters,observables,save_states)
    _fetch_distributed_trajectory_chunks(_distributed_diffusive_chunk,
        model,rho0,raw_times,chunks,seeds,worker_ids,
        "distributed diffusive trajectory",(monitors,),keywords)
end

distributed_diffusive_trajectories(model::PID.PIModel,rho0::PID.PIState,
        times,n::Integer;kwargs...)=
    distributed_diffusive_trajectories(model,rho0,times,nothing,n;kwargs...)

end
