# Confidence-controlled stochastic ensembles.  The stopping decision is made
# only at deterministic batch boundaries.  Every path keeps its
# trajectory-index-derived seed, so serial and threaded execution of the same
# adaptive request produce identical accumulated samples.

"""
    AdaptiveTrajectoryResult

Result of a confidence-controlled trajectory ensemble.  `observables` uses
the same time-resolved mean, variance, standard-error, and confidence-interval
schema as [`TrajectoryEnsembleResult`](@ref).  `convergence_history` records
the worst simultaneous empirical-Bernstein half-width and worst tolerance
ratio at every checked batch boundary. The pointwise normal intervals in
`observables` are descriptive and are not substituted for this stopping
certificate. `converged=false` is returned explicitly when
`max_trajectories` is reached; the routine never labels that result converged.
"""
struct AdaptiveTrajectoryResult{T,O,J,R,H}
    backend::Symbol
    times::Vector{T}
    trajectory_count::Int
    converged::Bool
    stopping_reason::Symbol
    observables::O
    jumps::J
    confidence::R
    atol::R
    rtol::R
    convergence_history::H
end

function Base.show(io::IO,result::AdaptiveTrajectoryResult)
    print(io,"AdaptiveTrajectoryResult(backend=",result.backend,
          ", trajectories=",result.trajectory_count,
          ", converged=",result.converged,
          ", reason=",result.stopping_reason,")")
end

function _adaptive_ensemble_controls(::Type{R};min_trajectories,
        max_trajectories,batch_size,confidence,atol,rtol) where R<:AbstractFloat
    min_trajectories>=2||throw(ArgumentError(
        "min_trajectories must be at least 2"))
    max_trajectories>=min_trajectories||throw(ArgumentError(
        "max_trajectories must be at least min_trajectories"))
    batch_size>=1||throw(ArgumentError("batch_size must be positive"))
    for (label,value) in (("min_trajectories",min_trajectories),
                          ("max_trajectories",max_trajectories),
                          ("batch_size",batch_size))
        BigInt(value)<=typemax(Int)||throw(ArgumentError(
            "$label must be representable as an Int"))
    end
    c=confidence===nothing ? R(0.95) :
        _trajectory_real_input(R,confidence,"adaptive confidence")
    zero(R)<c<one(R)||throw(ArgumentError(
        "confidence must lie in (0,1)"))
    a=atol===nothing ? sqrt(eps(R)) :
        _trajectory_real_input(R,atol,"adaptive absolute tolerance")
    r=rtol===nothing ? one(R)/R(100) :
        _trajectory_real_input(R,rtol,"adaptive relative tolerance")
    a>=zero(R)||throw(ArgumentError("atol must be nonnegative"))
    r>=zero(R)||throw(ArgumentError("rtol must be nonnegative"))
    (!iszero(a)||!iszero(r))||throw(ArgumentError(
        "at least one of atol and rtol must be positive"))
    (min=Int(min_trajectories),max=Int(max_trajectories),
     batch=Int(batch_size),confidence=c,atol=a,rtol=r)
end

function _adaptive_statistics_type(::Type{R},largest_count::Integer) where R<:AbstractFloat
    zero(largest_count)<=largest_count<=typemax(Int)||throw(ArgumentError(
        "adaptive statistical comparison count must be representable as an Int"))
    count=Int(largest_count)
    for S in (R,promote_type(R,Float32),promote_type(R,Float64))
        converted=S(count)
        roundtrips=isfinite(converted)&&try
            Int(converted)==count
        catch
            false
        end
        roundtrips&&return S
    end
    throw(ArgumentError(
        "adaptive trajectory count $count is not exactly representable in supported statistics precision; reduce the ensemble size"))
end

function _adaptive_controls_and_type(::Type{R},nobservables,ntimes;
        min_trajectories,max_trajectories,batch_size,confidence,atol,rtol) where R<:AbstractFloat
    initial=_adaptive_ensemble_controls(R;min_trajectories,max_trajectories,
        batch_size,confidence,atol,rtol)
    checks=cld(initial.max,initial.batch)
    comparisons=BigInt(nobservables)*BigInt(ntimes)*BigInt(checks)
    comparisons<=typemax(Int)||throw(ArgumentError(
        "adaptive simultaneous-comparison count exceeds Int indexing"))
    statistics_type=_adaptive_statistics_type(R,
        max(BigInt(initial.max),comparisons))
    controls=statistics_type===R ? initial :
        _adaptive_ensemble_controls(statistics_type;min_trajectories,
            max_trajectories,batch_size,confidence,atol,rtol)
    controls,statistics_type,checks
end

function _adaptive_observable_bounds(ops,::Type{R}) where R<:AbstractFloat
    lower_bounds=Vector{R}(undef,length(ops))
    upper_bounds=Vector{R}(undef,length(ops))
    ranges=Vector{R}(undef,length(ops))
    C=Complex{R}
    for observable_index in eachindex(ops)
        operator=last(ops[observable_index]);lower=R(Inf);upper=R(-Inf)
        for (sector_index,partition) in pairs(operator.basis.sectors)
            block=Matrix{C}(_divide_by_schur_multiplicity_scale(
                Matrix{C}(coefficient_block(operator,partition)),R,partition))
            dimension=size(block,1)
            @inbounds for row in 1:dimension
                center=real(block[row,row]);radius=zero(R)
                for column in 1:dimension
                    column==row||(radius+=abs(block[row,column]))
                end
                lower=min(lower,center-radius)
                upper=max(upper,center+radius)
            end
        end
        span=upper-lower
        isfinite(span)&&span>=zero(R)||throw(ArgumentError(
            "observable $(first(ops[observable_index])) has no finite spectral-range bound in $R; use wider precision"))
        scale=max(one(R),abs(lower),abs(upper))
        coordinate_count=length(operator.data)
        padding=R(32)*R(coordinate_count)*eps(R)*scale
        padded_lower=lower-padding;padded_upper=upper+padding
        padded_span=padded_upper-padded_lower
        isfinite(padded_lower)&&isfinite(padded_upper)&&
            isfinite(padded_span)&&padded_span>=span||throw(ArgumentError(
            "observable $(first(ops[observable_index])) spectral-range padding overflowed in $R; use wider precision"))
        lower_bounds[observable_index]=padded_lower
        upper_bounds[observable_index]=padded_upper
        ranges[observable_index]=padded_span
    end
    (lower=lower_bounds,upper=upper_bounds,ranges)
end

function _validate_adaptive_observations!(values,bounds,ops)
    size(values,1)==length(bounds.ranges)||throw(DimensionMismatch(
        "adaptive observable buffer has the wrong observable count"))
    @inbounds for time_index in axes(values,2),
                  observable_index in axes(values,1)
        value=values[observable_index,time_index]
        isfinite(value)||throw(ArgumentError(
            "adaptive trajectory produced a nonfinite value for observable $(first(ops[observable_index])) at saved index $time_index"))
        bounds.lower[observable_index]<=value<=bounds.upper[observable_index]||
            throw(ArgumentError(
            "adaptive trajectory produced value $value outside the certified range [$(bounds.lower[observable_index]), $(bounds.upper[observable_index])] for observable $(first(ops[observable_index])) at saved index $time_index; refine the trajectory integrator and validate conditional states"))
    end
    values
end

function _adaptive_observable_convergence(acc::_OnlineObservableAccumulator{R},
        confidence::R,atol::R,rtol::R,ranges::AbstractVector{R},
        maximum_checks::Integer) where R
    n=acc.count
    n>=2||return (false,R(Inf),R(Inf))
    length(ranges)==size(acc.mean,1)||throw(DimensionMismatch(
        "adaptive observable-range vector has the wrong length"))
    maximum_checks>0||throw(ArgumentError(
        "maximum adaptive check count must be positive"))
    comparisons=BigInt(length(acc.mean))*BigInt(maximum_checks)
    comparisons<=typemax(Int)||throw(ArgumentError(
        "adaptive simultaneous-comparison count exceeds Int indexing"))
    comparisons_R=_checked_statistics_count(
        R,Int(comparisons),"adaptive comparison")
    n_R=_checked_statistics_count(R,n,"adaptive trajectory")
    n_minus_one_R=_checked_statistics_count(
        R,n-1,"adaptive trajectory")
    delta=(one(R)-confidence)/comparisons_R
    isfinite(delta)&&delta>zero(R)||throw(ArgumentError(
        "adaptive confidence allocation underflowed in $R; reduce the output/check count or use wider precision"))
    logarithm=log(R(2)/delta)
    worst_ratio=zero(R);worst_half_width=zero(R);converged=true
    @inbounds for time_index in axes(acc.mean,2),
                  observable_index in axes(acc.mean,1)
        variance=max(zero(R),
            acc.m2[observable_index,time_index]/n_minus_one_R)
        half_width=sqrt(R(2)*variance*logarithm/n_R)+
            R(7)*ranges[observable_index]*logarithm/(R(3)*n_minus_one_R)
        tolerance=atol+rtol*abs(acc.mean[observable_index,time_index])
        isfinite(half_width)&&isfinite(tolerance)||throw(ArgumentError(
            "adaptive confidence calculation overflowed; use wider precision"))
        ratio=iszero(tolerance) ? (iszero(half_width) ? zero(R) : R(Inf)) :
            half_width/tolerance
        worst_ratio=max(worst_ratio,ratio)
        worst_half_width=max(worst_half_width,half_width)
        converged&=half_width<=tolerance
    end
    converged,worst_ratio,worst_half_width
end

function _adaptive_summary(acc,ops,times,controls,history,backend,jumps,
                           ranges,maximum_checks)
    converged,_,_=_adaptive_observable_convergence(
        acc,controls.confidence,controls.atol,controls.rtol,ranges,
        maximum_checks)
    observables=_observable_statistics(acc,ops,times,controls.confidence)
    AdaptiveTrajectoryResult(backend,copy(times),acc.count,converged,
        converged ? :confidence_target : :maximum_trajectories,
        observables,jumps,controls.confidence,controls.atol,controls.rtol,
        history)
end

function _reset_adaptive_accumulator!(acc::_OnlineObservableAccumulator)
    acc.count=0;fill!(acc.mean,zero(eltype(acc.mean)))
    fill!(acc.m2,zero(eltype(acc.m2)));acc
end

function _reset_adaptive_accumulator!(acc::_OnlineJumpAccumulator)
    acc.count=0;fill!(acc.totals,0);fill!(acc.channel_mean,zero(eltype(acc.channel_mean)))
    fill!(acc.channel_m2,zero(eltype(acc.channel_m2)))
    acc.total_mean=zero(acc.total_mean);acc.total_m2=zero(acc.total_m2)
    acc.no_jump=0;empty!(acc.waiting_times);fill!(acc.channel_counts,0);acc
end

function _merge_adaptive_batch!(global_observables,global_jumps,
        local_observables,local_jumps)
    for accumulator in local_observables
        _merge_observables!(global_observables,accumulator)
    end
    if global_jumps!==nothing
        for accumulator in local_jumps
            _merge_jumps!(global_jumps,accumulator)
        end
    end
    nothing
end

"""
    adaptive_quantum_trajectories(source, rho0, times;
        observables, dt, min_trajectories=64, max_trajectories=10_000,
        batch_size=32, confidence=0.95, atol=nothing, rtol=nothing,
        seed=0, threaded=false, jump_statistics=false,
        trajectory_keywords...)

Run state-free PI quantum-jump trajectories until every requested observable
at every saved time satisfies

```math
h_{o,t} \\leq \\mathrm{atol}+\\mathrm{rtol}|\\mathrm{mean}_{o,t}|,
```

The convergence decision uses a finite-horizon simultaneous empirical-
Bernstein bound derived from Hermitian-observable spectral ranges.
It is evaluated only after complete batches and never before
`min_trajectories`; zero observed variance therefore does not by itself imply
zero uncertainty for a nonconstant observable. If the target is not reached,
`max_trajectories` samples are returned with `converged=false`.  Model
lowering, time conversion, observable construction, workspaces, and RNG
storage are prepared once.  No sampled state history is retained.
"""
function adaptive_quantum_trajectories(
        source::Union{PIModel,CompiledPIModel,TrajectoryPlan},
        rho0::PIState{R},times;observables,dt::Real,
        min_trajectories::Integer=64,max_trajectories::Integer=10_000,
        batch_size::Integer=32,confidence=nothing,atol=nothing,rtol=nothing,
        seed::Integer=0,threaded::Bool=false,jump_statistics::Bool=false,
        workspace=nothing,parameters=nothing,max_jump_probability=nothing,
        algorithm::Symbol=:fixed,abstol=nothing,reltol=nothing,dtmin=nothing,
        dtmax=nothing,event_time_tolerance=nothing) where {R<:AbstractFloat}
    plan=_plan_for_source(source,rho0)
    _validate_trajectory_initial_state(plan,rho0)
    ts,options=_prepare_trajectory_arguments(times,R;dt,parameters,
        max_jump_probability,algorithm,abstol,reltol,dtmin,dtmax,
        event_time_tolerance)
    ops=_prepare_streaming_observables(rho0.basis,observables;
                                       require_hermitian=true)
    isempty(ops)&&throw(ArgumentError(
        "adaptive trajectory stopping requires at least one observable"))
    Rstats_base=_observable_scalar_type(rho0,ops)
    controls,Rstats,maximum_checks=_adaptive_controls_and_type(
        Rstats_base,length(ops),length(ts);min_trajectories,max_trajectories,
        batch_size,confidence,atol,rtol)
    requested_workers=threaded ? min(controls.batch,Threads.nthreads()) : 1
    batch=if workspace===nothing
        TrajectoryBatchWorkspace(plan,rho0;workers=requested_workers)
    else
        workspace isa TrajectoryBatchWorkspace||throw(ArgumentError(
            "workspace must be a TrajectoryBatchWorkspace"))
        _check_trajectory_batch_workspace(workspace,source,rho0)
    end
    worker_count=threaded ? min(length(batch.workers),Threads.nthreads(),
                                controls.batch) : 1
    bounds=_adaptive_observable_bounds(ops,Rstats)
    ranges=bounds.ranges
    global_observables=_OnlineObservableAccumulator(
        Rstats,length(ops),length(ts))
    global_jumps=jump_statistics ?
        _OnlineJumpAccumulator(Rstats,length(plan.jumps)) : nothing
    buffers=[Matrix{Rstats}(undef,length(ops),length(ts))
             for _ in 1:worker_count]
    master=MersenneTwister(seed)
    local_observables=[_OnlineObservableAccumulator(
        Rstats,length(ops),length(ts)) for _ in 1:worker_count]
    local_jumps=jump_statistics ?
        [_OnlineJumpAccumulator(Rstats,length(plan.jumps))
         for _ in 1:worker_count] : nothing
    history=NamedTuple[];sampled=0
    while sampled<controls.max
        count=min(controls.batch,controls.max-sampled)
        foreach(_reset_adaptive_accumulator!,local_observables)
        local_jumps===nothing||foreach(_reset_adaptive_accumulator!,local_jumps)
        resize!(batch.seeds,count);rand!(master,batch.seeds)
        next=Threads.Atomic{Int}(1)
        function run_worker(worker_index)
            worker=batch.workers[worker_index];rng=batch.rngs[worker_index]
            while true
                local_index=Threads.atomic_add!(next,1)
                local_index>count&&break
                Random.seed!(rng,batch.seeds[local_index])
                result=_quantum_trajectory_prepared(plan,rho0,ts,worker,rng,
                    options;observable_ops=ops,
                    observable_values=buffers[worker_index],
                    save_states=false,record_jumps=jump_statistics)
                _validate_adaptive_observations!(
                    buffers[worker_index],bounds,ops)
                _accumulate_observables!(
                    local_observables[worker_index],buffers[worker_index])
                jump_statistics&&_accumulate_jumps!(
                    local_jumps[worker_index],result.jump_times,
                    result.jump_channels)
            end
        end
        if worker_count==1
            run_worker(1)
        else
            @sync for worker_index in 1:worker_count
                Threads.@spawn run_worker(worker_index)
            end
        end
        _merge_adaptive_batch!(global_observables,global_jumps,
                               local_observables,local_jumps)
        sampled=global_observables.count
        converged,worst_ratio,worst_half_width=
            _adaptive_observable_convergence(global_observables,
                controls.confidence,controls.atol,controls.rtol,ranges,
                maximum_checks)
        push!(history,(trajectories=sampled,worst_ratio,worst_half_width,
            converged=converged&&sampled>=controls.min,
            method=:empirical_bernstein_simultaneous))
        converged&&sampled>=controls.min&&break
    end
    jump_summary=global_jumps===nothing ? nothing :
        _jump_statistics(global_jumps,ts)
    _adaptive_summary(global_observables,ops,ts,controls,history,
                      :quantum_jump,jump_summary,ranges,maximum_checks)
end

"""
    adaptive_diffusive_trajectories(batch, rho0;
        min_trajectories=64, max_trajectories=10_000, batch_size=32,
        confidence=0.95, atol=nothing, rtol=nothing, seed=0,
        threaded=false, workspace=nothing, parameters=nothing)

Run a prepared diffusive ensemble until all observables stored in
`batch.observables` meet the requested sampling-width target.  The batch must
contain at least one observable.  Times, operators, monitor blocks, and task
workspaces are reused for every realization. Because finite-step
Euler--Maruyama is not positivity preserving and has unbounded Gaussian
increments, its empirical-Bernstein interpretation is conditional on a
separate step-size/physicality study; this routine controls sampled Monte
Carlo dispersion, not integrator bias.
"""
function adaptive_diffusive_trajectories(batch::DiffusiveBatchPlan,
        rho0::PIState;min_trajectories::Integer=64,
        max_trajectories::Integer=10_000,batch_size::Integer=32,
        confidence=nothing,atol=nothing,rtol=nothing,seed::Integer=0,
        threaded::Bool=false,workspace=nothing,parameters=nothing)
    R=batch.plan.real_type
    ops=batch.observables;isempty(ops)&&throw(ArgumentError(
        "adaptive diffusive stopping requires observables in DiffusiveBatchPlan"))
    Rstats_base=_observable_scalar_type(rho0,ops)
    controls,Rstats,maximum_checks=_adaptive_controls_and_type(
        Rstats_base,length(ops),length(batch.times);min_trajectories,
        max_trajectories,batch_size,confidence,atol,rtol)
    requested_workers=threaded ? min(controls.batch,Threads.nthreads()) : 1
    work=workspace===nothing ?
        DiffusiveBatchWorkspace(batch,rho0;workers=requested_workers) :
        _check_diffusive_batch_workspace(workspace,batch,rho0)
    worker_count=threaded ? min(length(work.workers),Threads.nthreads(),
                                controls.batch) : 1
    bounds=_adaptive_observable_bounds(ops,Rstats)
    ranges=bounds.ranges
    global_observables=_OnlineObservableAccumulator(
        Rstats,length(ops),length(batch.times))
    master=MersenneTwister(seed)
    buffers=[Matrix{Rstats}(undef,length(ops),length(batch.times))
             for _ in 1:worker_count]
    local_observables=[_OnlineObservableAccumulator(
        Rstats,length(ops),length(batch.times)) for _ in 1:worker_count]
    history=NamedTuple[];sampled=0
    while sampled<controls.max
        count=min(controls.batch,controls.max-sampled)
        foreach(_reset_adaptive_accumulator!,local_observables)
        resize!(work.seeds,count);rand!(master,work.seeds)
        next=Threads.Atomic{Int}(1)
        function run_worker(worker_index)
            worker=work.workers[worker_index];rng=work.rngs[worker_index]
            while true
                local_index=Threads.atomic_add!(next,1)
                local_index>count&&break
                Random.seed!(rng,work.seeds[local_index])
                _diffusive_trajectory_prepared(
                    batch,rho0,worker,rng;parameters,save_states=false,
                    save_records=false,
                    observable_values=buffers[worker_index],
                    return_result=false)
                _validate_adaptive_observations!(
                    buffers[worker_index],bounds,ops)
                _accumulate_observables!(local_observables[worker_index],
                                         buffers[worker_index])
            end
        end
        if worker_count==1
            run_worker(1)
        else
            @sync for worker_index in 1:worker_count
                Threads.@spawn run_worker(worker_index)
            end
        end
        _merge_adaptive_batch!(global_observables,nothing,
                               local_observables,nothing)
        sampled=global_observables.count
        converged,worst_ratio,worst_half_width=
            _adaptive_observable_convergence(global_observables,
                controls.confidence,controls.atol,controls.rtol,ranges,
                maximum_checks)
        push!(history,(trajectories=sampled,worst_ratio,worst_half_width,
            converged=converged&&sampled>=controls.min,
            method=:empirical_bernstein_simultaneous))
        converged&&sampled>=controls.min&&break
    end
    _adaptive_summary(global_observables,ops,batch.times,controls,history,
                      :diffusive,nothing,ranges,maximum_checks)
end

function adaptive_diffusive_trajectories(source,rho0::PIState,times,
        monitors=nothing;dt,observables,kwargs...)
    batch=DiffusiveBatchPlan(source,rho0,times,monitors;dt,observables)
    adaptive_diffusive_trajectories(batch,rho0;kwargs...)
end
