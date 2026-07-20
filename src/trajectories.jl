"""One PI quantum-jump realization and its recorded jump channel indices."""
struct QuantumTrajectory{T,S<:PIState}
    times::Vector{T}
    states::Vector{S}
    jump_times::Vector{T}
    jump_channels::Vector{Int}
end

"""
    TrajectoryEnsembleResult

Memory-conscious result returned by [`quantum_trajectories`](@ref) when
`observables` are requested or `save_states=false`. `trajectories` is either
the ordinary vector of [`QuantumTrajectory`](@ref) objects or `nothing` when
state histories were not saved. `observables` contains online Monte Carlo
statistics and `jumps` contains online channel-resolved jump statistics; either
field is `nothing` when the corresponding output was not requested.
Indexing and iteration return stored trajectories and raise when histories
were disabled.

With `save_states=false`, no sampled `PIState` is constructed. Observable
buffers and Welford accumulators are task-owned and their retained storage is
`O(workers * length(times) * number_of_observables)`. Jump statistics retain
only their summary and the pooled inter-jump waiting times. The pooled waiting
samples have no trajectory-order guarantee under threaded scheduling.
"""
struct TrajectoryEnsembleResult{T,P,O}
    times::Vector{T}
    trajectories::P
    observables::O
    jumps::Union{Nothing,NamedTuple}
    trajectory_count::Int
end

Base.length(result::TrajectoryEnsembleResult)=result.trajectory_count
Base.firstindex(result::TrajectoryEnsembleResult)=1
Base.lastindex(result::TrajectoryEnsembleResult)=result.trajectory_count
Base.getindex(result::TrajectoryEnsembleResult,index::Integer)=begin
    result.trajectories===nothing&&throw(ArgumentError(
        "this result was created with save_states=false"))
    result.trajectories[index]
end
Base.iterate(result::TrajectoryEnsembleResult,args...)=begin
    result.trajectories===nothing&&throw(ArgumentError(
        "this result was created with save_states=false"))
    iterate(result.trajectories,args...)
end

function Base.show(io::IO,result::TrajectoryEnsembleResult)
    stored=result.trajectories===nothing ? "state-free" : "with state histories"
    print(io,"TrajectoryEnsembleResult($(result.trajectory_count) trajectories, $stored, $(length(result.times)) sampling times)")
end

"""
    TrajectorySteadyStateResult

Detailed result returned by [`trajectory_steady_state`](@ref) or
[`weak_pi_trajectory_steady_state`](@ref) when `return_info=true`. `state` is
the unmodified Monte Carlo average of the post-settling path means.
`sample_spread` is the square root of their unbiased sample variance in
Hilbert--Schmidt norm, and `standard_error` is the corresponding
Hilbert--Schmidt standard-error norm.
The PI coefficient basis is orthonormal, so these quantities are evaluated
without reconstructing the full Hilbert space.

`residual`, `relative_residual`, and `trace_error` diagnose the returned
average; they are reports, not convergence certificates. When named
Hermitian `observables` were requested, that field contains statistics across
the same independent path means. Samples taken along one path are averaged
before any uncertainty is estimated and are therefore never counted as
independent trajectories. A weak-PI result requested with `batch_size`
retains an additional [`WeakPIBatchMeansDiagnostics`](@ref) in
`metadata.batch_means`; the primary fields keep their independent-path
meaning.
"""
struct TrajectorySteadyStateResult{T<:AbstractFloat,S<:PIState,O,M}
    state::S
    trajectory_count::Int
    samples_per_trajectory::Int
    sampling_times::Vector{T}
    sample_spread::T
    standard_error::T
    residual::T
    relative_residual::T
    trace_error::T
    observables::O
    metadata::M
end

function Base.show(io::IO,result::TrajectorySteadyStateResult)
    print(io,"TrajectorySteadyStateResult($(result.trajectory_count) trajectories x ",
          "$(result.samples_per_trajectory) samples, residual=$(result.residual), ",
          "HS standard error=$(result.standard_error))")
end

function _named_observables(observables)
    observables isa NamedTuple&&return collect(pairs(observables))
    observables isa AbstractDict&&return collect(pairs(observables))
    observables isa Pair&&return [observables]
    observables isa AbstractVector{<:Pair}&&return collect(observables)
    [(Symbol("observable"),observables)]
end

function _prepare_streaming_observables(b::PIBasis,observables;
                                        require_hermitian::Bool)
    observables===nothing&&return ()
    named=_named_observables(observables)
    isempty(named)&&throw(ArgumentError("observables cannot be empty"))
    prepared=Pair[];seen=Set{Any}()
    for (name,A) in named
        name in seen&&throw(ArgumentError("duplicate observable name $name"))
        push!(seen,name)
        op=A isa AbstractMatrix ? collective_operator(b,A) : A
        op isa PIOperator&&op.basis===b||throw(ArgumentError(
            "observables must be local matrices or compatible PIOperators"))
        require_hermitian&& !ishermitian(op)&&throw(ArgumentError(
            "trajectory observable statistics require Hermitian observables"))
        push!(prepared,name=>op)
    end
    # A tuple retains the concrete operator type of every observable. Keeping
    # these pairs in the abstract `Pair[]` setup buffer would dynamically
    # dispatch and box each scalar `dot` result at every sampled time.
    Tuple(prepared)
end

function _observable_scalar_type(rho,ops)
    T=eltype(rho.data)
    for (_,op) in ops
        T=promote_type(T,eltype(op.data))
    end
    _real_float_type(T)
end

function _record_observables!(values::AbstractMatrix,ops,x,index)
    @inbounds for observable_index in eachindex(ops)
        value=dot(last(ops[observable_index]).data,x)
        values[observable_index,index]=real(value)
    end
    values
end

mutable struct _OnlineObservableAccumulator{R<:AbstractFloat}
    count::Int
    mean::Matrix{R}
    m2::Matrix{R}
end

_OnlineObservableAccumulator(::Type{R},nobservables,ntimes) where R<:AbstractFloat=
    _OnlineObservableAccumulator(0,zeros(R,nobservables,ntimes),
                                 zeros(R,nobservables,ntimes))

@inline function _checked_statistics_count(::Type{R},n::Int,context) where R<:AbstractFloat
    converted=R(n)
    roundtrips=isfinite(converted)&&try
        Int(converted)==n
    catch
        false
    end
    roundtrips||throw(ArgumentError(
        "$context count $n is not exactly representable in statistics precision $R; use wider observable/state precision or a smaller ensemble"))
    converted
end

function _accumulate_observables!(acc::_OnlineObservableAccumulator,values)
    acc.count+=1;n=acc.count
    nR=_checked_statistics_count(eltype(acc.mean),n,"observable")
    @inbounds for index in eachindex(values)
        delta=values[index]-acc.mean[index]
        acc.mean[index]+=delta/nR
        acc.m2[index]+=delta*(values[index]-acc.mean[index])
    end
    acc
end

function _merge_observables!(left::_OnlineObservableAccumulator,
                             right::_OnlineObservableAccumulator)
    right.count==0&&return left
    left.count==0&&(left.count=right.count;copyto!(left.mean,right.mean);
                    copyto!(left.m2,right.m2);return left)
    total=left.count+right.count
    R=eltype(left.mean)
    left_count=_checked_statistics_count(R,left.count,"observable")
    right_count=_checked_statistics_count(R,right.count,"observable")
    total_count=_checked_statistics_count(R,total,"observable")
    @inbounds for index in eachindex(left.mean)
        delta=right.mean[index]-left.mean[index]
        left.m2[index]+=right.m2[index]+delta*delta*
            left_count*right_count/total_count
        left.mean[index]+=delta*right_count/total_count
    end
    left.count=total;left
end

function _observable_statistic(acc::_OnlineObservableAccumulator{R},
                               observable_index,n,z,confidence) where {R}
    means=copy(view(acc.mean,observable_index,:))
    ntimes=size(acc.mean,2)
    vars=Vector{R}(undef,ntimes)
    stderr=Vector{R}(undef,ntimes)
    lower=Vector{R}(undef,ntimes)
    upper=Vector{R}(undef,ntimes)
    denominator=_checked_statistics_count(R,n,"observable")
    variance_denominator=n>1 ?
        _checked_statistics_count(R,n-1,"observable") : one(R)
    @inbounds for time_index in 1:ntimes
        variance=n>1 ?
            acc.m2[observable_index,time_index]/variance_denominator : zero(R)
        standard_error=sqrt(variance/denominator)
        half_width=z*standard_error
        vars[time_index]=variance
        stderr[time_index]=standard_error
        lower[time_index]=means[time_index]-half_width
        upper[time_index]=means[time_index]+half_width
    end
    (mean=means,variance=vars,standard_error=stderr,
     confidence=confidence,lower,upper)
end

function _observable_statistics(acc::_OnlineObservableAccumulator{R},ops,times,
                                confidence::Real) where {R}
    0<confidence<1||throw(ArgumentError("confidence must lie in (0,1)"))
    n=acc.count;n>0||throw(ArgumentError("no observable samples were accumulated"))
    isempty(ops)&&throw(ArgumentError("at least one observable is required"))
    z=R(_normal_quantile((1+confidence)/2))
    results=Dict{Any,Any}()
    for observable_index in eachindex(ops)
        results[first(ops[observable_index])]=_observable_statistic(
            acc,observable_index,n,z,confidence)
    end
    (;times=copy(times),trajectories=n,observables=results)
end

# Streaming state reduction used by the trajectory steady-state estimator.
# The first requested output is the initial state and is deliberately skipped;
# every later output belongs to the post-settling sampling window.  A path
# retains one running mean rather than one PIState per requested time.
mutable struct _TrajectoryStateSampler{V}
    mean::V
    count::Int
    first_output_index::Int
end

function _TrajectoryStateSampler(prototype::AbstractVector;
                                 first_output_index::Integer=2)
    first_output_index>0||throw(ArgumentError(
        "first trajectory sampling index must be positive"))
    _TrajectoryStateSampler(similar(prototype),0,Int(first_output_index))
end

function _reset_trajectory_state_sampler!(sampler::_TrajectoryStateSampler)
    fill!(sampler.mean,zero(eltype(sampler.mean)))
    sampler.count=0
    sampler
end

function _record_trajectory_state!(sampler::_TrajectoryStateSampler,x,
                                   output_index::Integer)
    output_index<sampler.first_output_index&&return sampler
    sampler.count+=1
    R=_real_float_type(eltype(sampler.mean))
    count=_checked_statistics_count(R,sampler.count,"trajectory time sample")
    @inbounds for index in eachindex(sampler.mean,x)
        sampler.mean[index]+=(x[index]-sampler.mean[index])/count
    end
    sampler
end

mutable struct _OnlineStateAccumulator{R<:AbstractFloat,V}
    count::Int
    mean::V
    m2::R
end

function _OnlineStateAccumulator(prototype::AbstractVector)
    R=_real_float_type(eltype(prototype))
    _OnlineStateAccumulator(0,similar(prototype),zero(R))
end

function _accumulate_state!(acc::_OnlineStateAccumulator{R},x) where R
    acc.count+=1
    count=_checked_statistics_count(R,acc.count,"trajectory")
    if acc.count==1
        copyto!(acc.mean,x)
        return acc
    end
    previous=count-one(R)
    factor=previous/count
    distance2=zero(R)
    @inbounds for index in eachindex(acc.mean,x)
        delta=x[index]-acc.mean[index]
        distance2+=abs2(delta)
        acc.mean[index]+=delta/count
    end
    acc.m2+=factor*distance2
    acc
end

function _merge_states!(left::_OnlineStateAccumulator{R},
                        right::_OnlineStateAccumulator{R}) where R
    right.count==0&&return left
    if left.count==0
        left.count=right.count
        copyto!(left.mean,right.mean)
        left.m2=right.m2
        return left
    end
    total=left.count+right.count
    left_count=_checked_statistics_count(R,left.count,"trajectory")
    right_count=_checked_statistics_count(R,right.count,"trajectory")
    total_count=_checked_statistics_count(R,total,"trajectory")
    distance2=zero(R)
    @inbounds for index in eachindex(left.mean,right.mean)
        delta=right.mean[index]-left.mean[index]
        distance2+=abs2(delta)
        left.mean[index]+=delta*right_count/total_count
    end
    left.m2+=right.m2+distance2*left_count*right_count/total_count
    left.count=total
    left
end

function _steady_observable_statistics(acc::_OnlineObservableAccumulator{R},
                                       ops,confidence::Real) where R
    0<confidence<1||throw(ArgumentError("confidence must lie in (0,1)"))
    n=acc.count
    n>1||throw(ArgumentError(
        "at least two independent paths are required for observable uncertainty"))
    z=R(_normal_quantile((1+confidence)/2))
    denominator=_checked_statistics_count(R,n,"observable")
    variance_denominator=_checked_statistics_count(R,n-1,"observable")
    results=Dict{Any,Any}()
    for observable_index in eachindex(ops)
        mean=acc.mean[observable_index,1]
        variance=acc.m2[observable_index,1]/variance_denominator
        standard_error=sqrt(variance/denominator)
        half_width=z*standard_error
        results[first(ops[observable_index])]=(
            ;mean,variance,standard_error,confidence,
            lower=mean-half_width,upper=mean+half_width)
    end
    (;trajectories=n,observables=results)
end

mutable struct _OnlineJumpAccumulator{R<:AbstractFloat}
    count::Int
    totals::Vector{Int}
    channel_mean::Vector{R}
    channel_m2::Vector{R}
    total_mean::R
    total_m2::R
    no_jump::Int
    waiting_times::Vector{R}
    channel_counts::Vector{Int}
end

function _OnlineJumpAccumulator(::Type{R},nchannels::Integer) where R<:AbstractFloat
    nc=Int(nchannels)
    _OnlineJumpAccumulator(0,zeros(Int,nc),zeros(R,nc),zeros(R,nc),
        zero(R),zero(R),0,R[],zeros(Int,nc))
end

function _accumulate_jumps!(acc::_OnlineJumpAccumulator,jump_times,jump_channels)
    acc.count+=1;n=acc.count;fill!(acc.channel_counts,0)
    nR=_checked_statistics_count(eltype(acc.channel_mean),n,"jump")
    for channel in jump_channels
        1<=channel<=length(acc.totals)||throw(ArgumentError(
            "jump channel index $channel is outside the prepared model"))
        acc.channel_counts[channel]+=1
    end
    for channel in eachindex(acc.totals)
        count=acc.channel_counts[channel];acc.totals[channel]+=count
        delta=count-acc.channel_mean[channel]
        acc.channel_mean[channel]+=delta/nR
        acc.channel_m2[channel]+=delta*(count-acc.channel_mean[channel])
    end
    total=length(jump_times);iszero(total)&&(acc.no_jump+=1)
    delta=total-acc.total_mean;acc.total_mean+=delta/nR
    acc.total_m2+=delta*(total-acc.total_mean)
    if total>=2
        @inbounds for index in 2:total
            push!(acc.waiting_times,jump_times[index]-jump_times[index-1])
        end
    end
    acc
end

function _merge_jumps!(left::_OnlineJumpAccumulator,right::_OnlineJumpAccumulator)
    right.count==0&&return left
    if left.count==0
        left.count=right.count;copyto!(left.totals,right.totals)
        copyto!(left.channel_mean,right.channel_mean)
        copyto!(left.channel_m2,right.channel_m2)
        left.total_mean=right.total_mean;left.total_m2=right.total_m2
        left.no_jump=right.no_jump;append!(left.waiting_times,right.waiting_times)
        return left
    end
    total_count=left.count+right.count
    R=eltype(left.channel_mean)
    left_count=_checked_statistics_count(R,left.count,"jump")
    right_count=_checked_statistics_count(R,right.count,"jump")
    total_count_R=_checked_statistics_count(R,total_count,"jump")
    for channel in eachindex(left.totals)
        delta=right.channel_mean[channel]-left.channel_mean[channel]
        left.channel_m2[channel]+=right.channel_m2[channel]+
            delta*delta*left_count*right_count/total_count_R
        left.channel_mean[channel]+=delta*right_count/total_count_R
        left.totals[channel]+=right.totals[channel]
    end
    delta=right.total_mean-left.total_mean
    left.total_m2+=right.total_m2+delta*delta*
        left_count*right_count/total_count_R
    left.total_mean+=delta*right_count/total_count_R
    left.count=total_count;left.no_jump+=right.no_jump
    append!(left.waiting_times,right.waiting_times);left
end

function _jump_statistics(acc::_OnlineJumpAccumulator,times)
    n=acc.count;n>0||throw(ArgumentError("no jump samples were accumulated"))
    duration=times[end]-times[1];duration>=0||throw(ArgumentError(
        "invalid sampling interval"))
    R=eltype(acc.channel_mean);nan=R(NaN)
    count_R=_checked_statistics_count(R,n,"jump")
    channels=[begin
        variance=_sample_variance(acc.channel_m2[channel],n)
        mean=acc.channel_mean[channel]
        (channel=channel,total=acc.totals[channel],mean,
         variance,fano=iszero(mean) ? nan : variance/mean,
         rate=iszero(duration) ? nan : mean/duration)
    end for channel in eachindex(acc.totals)]
    total_variance=_sample_variance(acc.total_m2,n)
    mean_waiting=isempty(acc.waiting_times) ? nan :
        sum(acc.waiting_times)/length(acc.waiting_times)
    waiting_variance=length(acc.waiting_times)>1 ?
        sum(x->abs2(x-mean_waiting),acc.waiting_times)/
            (length(acc.waiting_times)-1) : nan
    (;trajectories=n,duration,total_jumps=sum(acc.totals),
      mean_count=acc.total_mean,count_variance=total_variance,
      fano=iszero(acc.total_mean) ? nan : total_variance/acc.total_mean,
      rate=iszero(duration) ? nan : acc.total_mean/duration,
      no_jump_probability=R(acc.no_jump)/count_R,channels,
      waiting_times=copy(acc.waiting_times),mean_waiting_time=mean_waiting,
      waiting_time_variance=waiting_variance)
end

"""
    TrajectoryPlan(model; T=nothing)
    TrajectoryPlan(compiled; T=nothing)

Immutable prepared geometry for PI quantum trajectories. A plan lowers the
model once, separates Hamiltonian and jump kernels, and retains the
sector-trace weights used to evaluate channel intensities without constructing
gain states. It may be shared by tasks; each concurrent worker must use its
own [`TrajectoryWorkspace`](@ref).

Trajectory plans require fixed jump operators. Scalar rates may still depend
on time and parameters. Rates must evaluate to finite, nonnegative real values
representable in the prepared precision. An empty model has no scalar-bearing
term from which to infer a precision; pass its desired concrete real floating
type as `T` when it is not `Float64`. The same keyword is accepted for an
empty compiled model.
"""
struct TrajectoryPlan{M,L,H,J,W}
    model::M
    liouvillian::L
    hamiltonians::H
    jumps::J
    trace_weights::W
end

isautonomous(plan::TrajectoryPlan)=isautonomous(plan.liouvillian)

function _trajectory_plan(model::PIModel,liouvillian_plan::LiouvillianPlan)
    all(term_has_fixed_operator,model.terms)||throw(ArgumentError(
        "trajectory kernels require fixed operators; scalar rates may depend on time"))
    kernels=liouvillian_plan.kernels
    kernels===nothing&&throw(ArgumentError(
        "trajectory kernels require terms that lower to prepared PI kernels"))
    supported=Union{HamiltonianPIKernel,DissipatorPIKernel,LocalJumpPIKernel,
                    FactorizedLocalJumpPIKernel,
                    FactorizedLocalPBodyJumpPIKernel}
    all(kernel->kernel isa supported,kernels)||throw(ArgumentError(
        "trajectory kernels require Hamiltonian, collective/direct-jump, or local-jump lowerings"))
    # Tuple filtering preserves the statically known kernel count and types;
    # a generator comprehension widens these to an unknown-length Vararg.
    hamiltonians=filter(kernel->kernel isa HamiltonianPIKernel,kernels)
    jumps=filter(kernel->kernel isa Union{DissipatorPIKernel,LocalJumpPIKernel,
                 FactorizedLocalJumpPIKernel,
                 FactorizedLocalPBodyJumpPIKernel},
                 kernels)
    R=_real_float_type(eltype(liouvillian_plan.tracevec))
    weights=Vector{R}(undef,length(model.basis.sectors))
    for sector in eachindex(model.basis.sectors)
        value=liouvillian_plan.tracevec[model.basis.offsets[sector]]
        iszero(imag(value))||throw(ArgumentError(
            "trajectory trace weights must be real"))
        weights[sector]=real(value)
    end
    TrajectoryPlan(model,liouvillian_plan,hamiltonians,jumps,weights)
end

function _empty_trajectory_plan(model::PIModel,::Type{R}) where R<:AbstractFloat
    isconcretetype(R)||throw(ArgumentError(
        "the trajectory scalar type T must be a concrete AbstractFloat type"))
    CT=Complex{R}
    plan=LiouvillianPlan(model.basis,(),_trace_vector(model.basis,CT),
                         nothing,CT,true)
    _trajectory_plan(model,plan)
end

function TrajectoryPlan(model::PIModel;T=nothing)
    if isempty(model.terms)&&T!==nothing
        T isa Type&&T<:AbstractFloat||throw(ArgumentError(
            "the trajectory scalar type T must be a concrete AbstractFloat type"))
        return _empty_trajectory_plan(model,T)
    end
    T===nothing||throw(ArgumentError(
        "T is only used to select the precision of an empty trajectory model; nonempty models infer it from their terms"))
    _trajectory_plan(model,_term_resolved_liouvillian_plan(model))
end
function TrajectoryPlan(compiled::CompiledPIModel;T=nothing)
    if T===nothing
        kernels=compiled.plan.kernels
        resolved=kernels!==nothing&&
            any(kernel->kernel isa FusedStaticPIKernel,kernels) ?
                _term_resolved_liouvillian_plan(compiled.model) : compiled.plan
        return _trajectory_plan(compiled.model,resolved)
    end
    isempty(compiled.model.terms)||throw(ArgumentError(
        "T is only used to select the precision of an empty trajectory model; nonempty models infer it from their terms"))
    T isa Type&&T<:AbstractFloat||throw(ArgumentError(
        "the trajectory scalar type T must be a concrete AbstractFloat type"))
    _empty_trajectory_plan(compiled.model,T)
end

function _trajectory_plan_for_state(model::PIModel,rho::PIState)
    isempty(model.terms) ?
        _empty_trajectory_plan(model,_real_float_type(eltype(rho.data))) :
        TrajectoryPlan(model)
end
function _trajectory_plan_for_state(compiled::CompiledPIModel,rho::PIState)
    isempty(compiled.model.terms) ?
        _empty_trajectory_plan(compiled.model,
                               _real_float_type(eltype(rho.data))) :
        TrajectoryPlan(compiled)
end
_trajectory_plan_for_state(plan::TrajectoryPlan,rho::PIState)=plan

"""
    TrajectoryWorkspace(plan, rho; mode=:full)
    TrajectoryWorkspace(model, rho; mode=:full)
    TrajectoryWorkspace(compiled, rho; mode=:full)

Preallocated mutable stage vectors and Schur-block scratch for one PI quantum
trajectory at a time. Reuse it sequentially. Concurrent paths must have
distinct workspaces, which may all refer to the same immutable
[`TrajectoryPlan`](@ref).

The default `mode=:full` supports both fixed-step RK4 and adaptive
event-driven Dormand--Prince paths. Fixed-step RK4 uses three full-vector
registers; `mode=:fixed` additionally omits `k3`, `k4`, and the six
Dormand--Prince stage, trial, embedded, and event-root vectors. A fixed-only
workspace rejects `algorithm=:event` instead of allocating the omitted
storage lazily.
"""
struct TrajectoryWorkspace{V,R,P,W,E}
    tmp::V;k1::V;k2::V;k3::V;k4::V;k5::V;k6::V;k7::V
    trial::V;embedded::V;start::V
    current::V
    channel_gain::V
    intensities::Vector{R}
    jump_scales::Vector{R}
    hazard_stages::Vector{R}
    dense_hazard::Vector{R}
    plan::P
    liouvillian_work::W
    effective_qblocks::E
    mode::Symbol
end

function TrajectoryWorkspace(plan::TrajectoryPlan,rho::PIState;
                             mode::Symbol=:full)
    mode in (:full,:fixed)||throw(ArgumentError(
        "trajectory workspace mode must be :full or :fixed"))
    rho.basis===plan.model.basis||throw(ArgumentError(
        "state and trajectory plan use incompatible PI bases"))
    _check_liouvillian_source_precision(plan.liouvillian,eltype(rho.data),
                                        "trajectory state")
    promote_type(eltype(rho.data),plan.liouvillian.Ttype)===eltype(rho.data)||
        throw(ArgumentError("trajectory state scalar type $(eltype(rho.data)) cannot represent plan scalar type $(plan.liouvillian.Ttype)"))
    v=similar(rho.data)
    R=_real_float_type(eltype(v));njumps=length(plan.jumps)
    adaptive_vector()=mode===:full ? similar(v) : similar(v,0)
    full_rk4_vector()=mode===:full ? similar(v) : similar(v,0)
    effective_qblocks=isempty(plan.jumps) ? Matrix{eltype(v)}[] :
        [zeros(eltype(v),length(plan.model.basis.patterns[s]),
            length(plan.model.basis.patterns[s]))
         for s in eachindex(plan.model.basis.sectors)]
    TrajectoryWorkspace(similar(v),similar(v),similar(v),full_rk4_vector(),
                        full_rk4_vector(),
                        adaptive_vector(),adaptive_vector(),adaptive_vector(),
                        adaptive_vector(),adaptive_vector(),adaptive_vector(),
                        v,similar(v),zeros(R,njumps),
                        zeros(R,njumps),zeros(R,7),zeros(R,4),plan,
                        LiouvillianWorkspace(plan.liouvillian),
                        effective_qblocks,mode)
end
TrajectoryWorkspace(model::PIModel,rho::PIState;kwargs...)=
    TrajectoryWorkspace(_trajectory_plan_for_state(model,rho),rho;kwargs...)
TrajectoryWorkspace(compiled::CompiledPIModel,rho::PIState;kwargs...)=
    TrajectoryWorkspace(_trajectory_plan_for_state(compiled,rho),rho;kwargs...)

"""
    TrajectoryBatchWorkspace(plan, rho;
                             workers=Threads.nthreads(), mode=:full)
    TrajectoryBatchWorkspace(model, rho;
                             workers=Threads.nthreads(), mode=:full)
    TrajectoryBatchWorkspace(compiled, rho;
                             workers=Threads.nthreads(), mode=:full)

Reusable worker pool for [`quantum_trajectories`](@ref) and
[`trajectory_steady_state`](@ref). The immutable trajectory plan is stored
once, while every worker owns independent integrator scratch and a reusable
random-number generator. A batch workspace may be reused sequentially but
must not be used by concurrent ensemble or estimator calls.
`mode` is forwarded to every worker; use `mode=:fixed` only when all calls
will use the fixed-step algorithm.
"""
struct TrajectoryBatchWorkspace{P,W,R,S}
    plan::P
    workers::W
    rngs::R
    seeds::S
end

function TrajectoryBatchWorkspace(plan::TrajectoryPlan,rho::PIState;
                                  workers::Integer=Threads.nthreads(),
                                  mode::Symbol=:full)
    workers>0||throw(ArgumentError("worker count must be positive"))
    workspaces=[TrajectoryWorkspace(plan,rho;mode) for _ in 1:Int(workers)]
    rngs=[MersenneTwister(0) for _ in 1:Int(workers)]
    TrajectoryBatchWorkspace(plan,workspaces,rngs,UInt64[])
end
TrajectoryBatchWorkspace(model::PIModel,rho::PIState;kwargs...)=
    TrajectoryBatchWorkspace(_trajectory_plan_for_state(model,rho),rho;kwargs...)
TrajectoryBatchWorkspace(compiled::CompiledPIModel,rho::PIState;kwargs...)=
    TrajectoryBatchWorkspace(_trajectory_plan_for_state(compiled,rho),rho;kwargs...)

function _trajectory_real_input(::Type{R},value,label) where R<:AbstractFloat
    value isa Real||throw(ArgumentError("$label must be real"))
    if value isa Integer
        converted=R(value)
        isfinite(converted)&&BigInt(converted)==BigInt(value)||throw(ArgumentError(
            "$label=$value is not exactly representable in trajectory precision $R"))
        return converted
    end
    promote_type(R,typeof(value))===R||throw(ArgumentError(
        "$label scalar type $(typeof(value)) cannot be represented by trajectory precision $R without narrowing; use matching inputs or prepare the model and state at a wider precision"))
    converted=R(value)
    isfinite(converted)||throw(ArgumentError("$label must be finite"))
    converted
end

function _trajectory_jump_scale(kernel,t,p,::Type{R}) where R<:AbstractFloat
    raw=value_at(kernel.scale,t,p)
    scale=_trajectory_real_input(R,raw,"jump rate")
    scale>=zero(R)||throw(ArgumentError(
        "quantum trajectories require nonnegative jump rates"))
    scale
end

function _apply_gain!(y,x,k::DissipatorPIKernel,b,scale,work)
    fill!(y,zero(eltype(y)))
    for s in eachindex(b.sectors)
        n=length(b.patterns[s]);r=b.offsets[s]:b.offsets[s+1]-1
        X=reshape(view(x,r),n,n);Y=reshape(view(y,r),n,n);A=work.blocks[s][1];K=k.blocks[s]
        mul!(A,K,X);mul!(Y,A,adjoint(K));Y .*= scale
    end
    y
end
function _apply_gain!(y,x,k::LocalJumpPIKernel,b,scale,work)
    fill!(y,zero(eltype(y)))
    @inbounds for q in eachindex(k.gain.V);y[k.gain.I[q]]+=scale*k.gain.V[q]*x[k.gain.J[q]];end
    y
end
function _apply_gain!(y,x,k::FactorizedLocalJumpPIKernel,b,scale,work)
    fill!(y,zero(eltype(y)))
    _ensure_batch_capacity!(work.batch,1)
    _apply_factorized_onebody_gain_batch!(reshape(y,:,1),reshape(x,:,1),
        k.branches,k.contractions,b,scale,work.batch,1)
    y
end
function _apply_gain!(y,x,k::FactorizedLocalPBodyJumpPIKernel,b,scale,work)
    fill!(y,zero(eltype(y)))
    _ensure_batch_capacity!(work.batch,1)
    _apply_factorized_pbody_gain_batch!(reshape(y,:,1),reshape(x,:,1),
        k.groups,k.contractions,k.pair_scales,b,scale,work.batch)
    y
end

function _unscaled_channel_intensity(x,k,b,weights)
    R=eltype(weights);value=zero(R)
    for sector in eachindex(b.sectors)
        n=length(b.patterns[sector]);r=b.offsets[sector]:b.offsets[sector+1]-1
        X=reshape(view(x,r),n,n);Q=k.qblocks[sector]
        sector_trace=zero(eltype(x))
        @inbounds for column in 1:n,row in 1:n
            sector_trace+=Q[row,column]*X[column,row]
        end
        # The trace-vector weight supplies sqrt(f^nu), so this is exactly
        # tr(G_k[rho])=sqrt(f^nu)tr(Q_nu C_nu) without constructing the gain
        # state. The explicit contraction does not assume Q is bitwise
        # Hermitian after floating-point setup.
        value+=weights[sector]*real(sector_trace)
    end
    value
end

_intensity_tolerance(::Type{R}) where R=max(R(1e-11),100eps(R))
function _store_intensity!(w,index,value)
    R=eltype(w.intensities);tolerance=_intensity_tolerance(R)
    isfinite(value)||throw(ArgumentError(
        "jump intensity is nonfinite; use a wider scalar type or inspect the state and rate"))
    value>=-tolerance||throw(ArgumentError("jump gain has negative trace $value"))
    w.intensities[index]=max(zero(R),value)
end

@inline _channel_intensities!(w,x,b,t,p,::Tuple{},index)=nothing
@inline function _channel_intensities!(w,x,b,t,p,
        jumps::Tuple{K,Vararg{Any}},index) where K
    kernel=first(jumps);R=eltype(w.intensities)
    scale=_trajectory_jump_scale(kernel,t,p,R);w.jump_scales[index]=scale
    if iszero(scale)
        w.intensities[index]=zero(R)
        return _channel_intensities!(w,x,b,t,p,Base.tail(jumps),index+1)
    end
    value=scale*_unscaled_channel_intensity(x,kernel,b,w.plan.trace_weights)
    _store_intensity!(w,index,value)
    _channel_intensities!(w,x,b,t,p,Base.tail(jumps),index+1)
end

function _channel_intensities!(w::TrajectoryWorkspace,x,b,t,p)
    _channel_intensities!(w,x,b,t,p,w.plan.jumps,1)
    w.intensities
end

function _total_intensity(rates)
    lambda=sum(rates)
    isfinite(lambda)||throw(ArgumentError(
        "total jump intensity is nonfinite; use a wider scalar type or inspect the state and rates"))
    lambda
end

function _select_jump_channel(rates,u)
    cumulative=zero(eltype(rates));last_positive=0
    @inbounds for index in eachindex(rates)
        rate=rates[index]
        rate>zero(rate)&&(last_positive=index)
        cumulative+=rate
        u<cumulative&&return index
    end
    last_positive>0||throw(ErrorException(
        "cannot select a jump channel from zero total intensity"))
    # Guard only against a final-roundoff mismatch between `sum(rates)` and
    # the sequential cumulative sum. A trailing zero channel is never chosen.
    last_positive
end

@inline _apply_trajectory_hamiltonians!(y,x,::Tuple{},b,t,p,work)=nothing
@inline function _apply_trajectory_hamiltonians!(y,x,
        kernels::Tuple{K,Vararg{Any}},b,t,p,work) where K
    _apply_kernel!(y,x,first(kernels),b,t,p,work.blocks)
    _apply_trajectory_hamiltonians!(y,x,Base.tail(kernels),b,t,p,work)
end

function _apply_jump_anticommutator_and_intensity!(y,x,k,b,scale,weights,work)
    R=eltype(weights);unscaled=zero(R)
    for sector in eachindex(b.sectors)
        n=length(b.patterns[sector]);off=b.offsets[sector]
        A,B,X=work.blocks[sector];Q=k.qblocks[sector]
        copyto!(X,1,x,off,n*n)
        mul!(A,Q,X)
        unscaled+=weights[sector]*real(tr(A))
        mul!(B,X,Q)
        @inbounds for index in eachindex(A)
            y[off+index-1]-=(scale/2)*(A[index]+B[index])
        end
    end
    scale*unscaled
end

@inline _apply_conditional_jumps!(y,x,w,b,t,p,::Tuple{},index)=
    zero(eltype(w.intensities))
@inline function _apply_conditional_jumps!(y,x,w,b,t,p,
        jumps::Tuple{K,Vararg{Any}},index) where K
    kernel=first(jumps);R=eltype(w.intensities)
    scale=_trajectory_jump_scale(kernel,t,p,R);w.jump_scales[index]=scale
    if iszero(scale)
        w.intensities[index]=zero(R)
        return _apply_conditional_jumps!(
            y,x,w,b,t,p,Base.tail(jumps),index+1)
    end
    value=_apply_jump_anticommutator_and_intensity!(
        y,x,kernel,b,scale,w.plan.trace_weights,w.liouvillian_work)
    _store_intensity!(w,index,value)
    w.intensities[index]+_apply_conditional_jumps!(
        y,x,w,b,t,p,Base.tail(jumps),index+1)
end


@inline _accumulate_effective_jump_blocks!(w,t,p,::Tuple{},index)=nothing
@inline function _accumulate_effective_jump_blocks!(w,t,p,
        jumps::Tuple{K,Vararg{Any}},index) where K
    kernel=first(jumps);R=eltype(w.intensities)
    scale=_trajectory_jump_scale(kernel,t,p,R)
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
    _accumulate_effective_jump_blocks!(
        w,t,p,Base.tail(jumps),index+1)
end

function _prepare_effective_jump_blocks!(w::TrajectoryWorkspace,t,p)
    for block in w.effective_qblocks
        fill!(block,zero(eltype(block)))
    end
    _accumulate_effective_jump_blocks!(w,t,p,w.plan.jumps,1)
    w.effective_qblocks
end

function _effective_jump_intensity_from_blocks(w,x,b)
    R=eltype(w.intensities);total=zero(R)
    for sector in eachindex(b.sectors)
        n=length(b.patterns[sector]);range=
            b.offsets[sector]:b.offsets[sector+1]-1
        X=reshape(view(x,range),n,n)
        effective=w.effective_qblocks[sector]
        sector_trace=zero(eltype(x))
        @inbounds for column in 1:n,row in 1:n
            sector_trace+=effective[row,column]*X[column,row]
        end
        total+=w.plan.trace_weights[sector]*real(sector_trace)
    end
    tolerance=_intensity_tolerance(R)
    isfinite(total)||throw(ArgumentError(
        "total jump intensity is nonfinite; use a wider scalar type or inspect the state and rates"))
    total>=-tolerance||throw(ArgumentError(
        "combined jump gain has negative trace $total"))
    max(zero(R),total)
end

function _effective_jump_intensity!(w,x,b,t,p)
    isempty(w.plan.jumps)&&return zero(eltype(w.intensities))
    _prepare_effective_jump_blocks!(w,t,p)
    _effective_jump_intensity_from_blocks(w,x,b)
end

function _apply_effective_jump_drift_and_intensity!(y,x,w,b,t,p)
    isempty(w.plan.jumps)&&return zero(eltype(w.intensities))
    _prepare_effective_jump_blocks!(w,t,p)
    R=eltype(w.intensities);total=zero(R)
    for sector in eachindex(b.sectors)
        n=length(b.patterns[sector]);off=b.offsets[sector]
        A,B,X=w.liouvillian_work.blocks[sector]
        copyto!(X,1,x,off,n*n)
        effective=w.effective_qblocks[sector]
        mul!(A,effective,X)
        total+=w.plan.trace_weights[sector]*real(tr(A))
        mul!(B,X,effective)
        @inbounds for index in eachindex(A)
            y[off+index-1]-=(A[index]+B[index])/2
        end
    end
    tolerance=_intensity_tolerance(R)
    isfinite(total)||throw(ArgumentError(
        "total jump intensity is nonfinite; use a wider scalar type or inspect the state and rates"))
    total>=-tolerance||throw(ArgumentError(
        "combined jump gain has negative trace $total"))
    max(zero(R),total)
end


function _conditional_action_and_intensity!(y,x,w,b,t,p,tau)
    fill!(y,zero(eltype(y)))
    _apply_trajectory_hamiltonians!(y,x,w.plan.hamiltonians,b,t,p,
                                    w.liouvillian_work)
    lambda=_apply_effective_jump_drift_and_intensity!(y,x,w,b,t,p)
    @. y=y+lambda*x
    lambda
end

function _conditional_action!(y,x,w,b,t,p,tau)
    _conditional_action_and_intensity!(y,x,w,b,t,p,tau)
    y
end

function _conditional_rk4!(x,w,b,t,h,p,tau)
    _conditional_action!(w.k1,x,w,b,t,p,tau)
    copyto!(w.k2,w.k1)
    @. w.tmp=x+(h/2)*w.k1
    _conditional_action!(w.k1,w.tmp,w,b,t+h/2,p,tau)
    @. w.k2=w.k2+2w.k1
    @. w.tmp=x+(h/2)*w.k1
    _conditional_action!(w.k1,w.tmp,w,b,t+h/2,p,tau)
    @. w.k2=w.k2+2w.k1
    @. w.tmp=x+h*w.k1
    _conditional_action!(w.k1,w.tmp,w,b,t+h,p,tau)
    @. x=x+(h/6)*(w.k2+w.k1)
    z=dot(tau,x);R=_real_float_type(eltype(x))
    abs(z)>eps(R)||throw(ArgumentError("conditional state acquired zero trace"));x./=z;x
end

# Shampine's quartic continuous extension for the Dormand--Prince 5(4)
# stages.  Event-root searches use these retained coefficients instead of
# reintegrating a complete seven-stage trial at every bisection point.
@inline function _dopri_dense_table(::Type{R}) where R<:AbstractFloat
    (R(-8048581381)/R(2820520608),
     R(8663915743)/R(2820520608),
     R(-12715105075)/R(11282082432),
     R(131558114200)/R(32700410799),
     R(-68118460800)/R(10900136933),
     R(87487479700)/R(32700410799),
     R(-1754552775)/R(470086768),
     R(14199869525)/R(1410260304),
     R(-10690763975)/R(1880347072),
     R(127303824393)/R(49829197408),
     R(-318862633887)/R(49829197408),
     R(701980252875)/R(199316789632),
     R(-282668133)/R(205662961),
     R(2019193451)/R(616988883),
     R(-1453857185)/R(822651844),
     R(40617522)/R(29380423),
     R(-110615467)/R(29380423),
     R(69997945)/R(29380423))
end

function _prepare_dopri_dense_output!(w)
    R=eltype(w.hazard_stages)
    c12,c13,c14,c32,c33,c34,c42,c43,c44,c52,c53,c54,
        c62,c63,c64,c72,c73,c74=_dopri_dense_table(R)
    l=w.hazard_stages;q=w.dense_hazard
    q[1]=l[1]
    q[2]=c12*l[1]+c32*l[3]+c42*l[4]+c52*l[5]+c62*l[6]+c72*l[7]
    q[3]=c13*l[1]+c33*l[3]+c43*l[4]+c53*l[5]+c63*l[6]+c73*l[7]
    q[4]=c14*l[1]+c34*l[3]+c44*l[4]+c54*l[5]+c64*l[6]+c74*l[7]
    w
end

@inline function _dopri_dense_hazard(w,h,theta)
    q=w.dense_hazard
    h*theta*(q[1]+theta*(q[2]+theta*(q[3]+theta*q[4])))
end

function _dopri_dense_state!(destination,start,w,h,theta)
    R=eltype(w.hazard_stages)
    c12,c13,c14,c32,c33,c34,c42,c43,c44,c52,c53,c54,
        c62,c63,c64,c72,c73,c74=_dopri_dense_table(R)
    k1=w.k1;k3=w.k3;k4=w.k4;k5=w.k5;k6=w.k6;k7=w.k7
    @inbounds @simd for index in eachindex(destination,start)
        q1=k1[index]
        q2=c12*k1[index]+c32*k3[index]+c42*k4[index]+
            c52*k5[index]+c62*k6[index]+c72*k7[index]
        q3=c13*k1[index]+c33*k3[index]+c43*k4[index]+
            c53*k5[index]+c63*k6[index]+c73*k7[index]
        q4=c14*k1[index]+c34*k3[index]+c44*k4[index]+
            c54*k5[index]+c64*k6[index]+c74*k7[index]
        destination[index]=start[index]+h*theta*(q1+
            theta*(q2+theta*(q3+theta*q4)))
    end
    destination
end

function _dopri_dense_root(w,h,remaining,increment,time_tolerance)
    R=typeof(h)
    endpoint=_dopri_dense_hazard(w,h,one(R))
    scale=max(one(R),abs(increment),abs(endpoint),
        abs(h)*sum(abs,w.hazard_stages))
    tolerance=R(256)*eps(R)*scale
    isfinite(endpoint)&&abs(endpoint-increment)<=tolerance||return nothing
    -tolerance<=remaining<=endpoint+tolerance||return nothing
    lo=zero(R);hi=one(R);hazard_lo=zero(R);hazard_hi=endpoint
    while h*(hi-lo)>time_tolerance
        mid=(lo+hi)/2
        (mid==lo||mid==hi)&&break
        hazard_mid=_dopri_dense_hazard(w,h,mid)
        isfinite(hazard_mid)&&
            hazard_lo-tolerance<=hazard_mid<=hazard_hi+tolerance||
            return nothing
        if hazard_mid>=remaining
            hi=mid;hazard_hi=hazard_mid
        else
            lo=mid;hazard_lo=hazard_mid
        end
    end
    h*hi
end

# One Dormand--Prince 5(4) trial for the normalized conditional state together
# with the accumulated jump hazard.  State and hazard use the same stages, so
# an accepted step controls both errors and no time-grid Bernoulli
# approximation enters the event time.
function _conditional_dopri_trial!(w,x,b,t,h,p,tau,abstol,reltol)
    R=typeof(h)
    l1=_conditional_action_and_intensity!(w.k1,x,w,b,t,p,tau)
    @. w.tmp=x+h*(1//5)*w.k1
    l2=_conditional_action_and_intensity!(w.k2,w.tmp,w,b,t+h*(R(1)/R(5)),p,tau)
    @. w.tmp=x+h*((3//40)*w.k1+(9//40)*w.k2)
    l3=_conditional_action_and_intensity!(w.k3,w.tmp,w,b,t+h*(R(3)/R(10)),p,tau)
    @. w.tmp=x+h*((44//45)*w.k1-(56//15)*w.k2+(32//9)*w.k3)
    l4=_conditional_action_and_intensity!(w.k4,w.tmp,w,b,t+h*(R(4)/R(5)),p,tau)
    @. w.tmp=x+h*((19372//6561)*w.k1-(25360//2187)*w.k2+
                  (64448//6561)*w.k3-(212//729)*w.k4)
    l5=_conditional_action_and_intensity!(w.k5,w.tmp,w,b,t+h*(R(8)/R(9)),p,tau)
    @. w.tmp=x+h*((9017//3168)*w.k1-(355//33)*w.k2+
                  (46732//5247)*w.k3+(49//176)*w.k4-(5103//18656)*w.k5)
    l6=_conditional_action_and_intensity!(w.k6,w.tmp,w,b,t+h,p,tau)
    @. w.trial=x+h*((35//384)*w.k1+(500//1113)*w.k3+
                    (125//192)*w.k4-(2187//6784)*w.k5+(11//84)*w.k6)
    l7=_conditional_action_and_intensity!(w.k7,w.trial,w,b,t+h,p,tau)
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

function _adaptive_factor(error::R) where R<:AbstractFloat
    error==zero(R)&&return R(5)
    clamp((R(9)/R(10))*error^(-one(R)/R(5)),R(1)/R(5),R(5))
end

# Floating addition can leave `t` one ulp below a requested output time after
# an apparently exact number of fixed steps (for example 0:0.01:0.1).  Treat
# a final interval that exceeds the nominal bound only by roundoff as one step
# and snap its accepted endpoint to the caller's target.  This avoids a tiny
# extra stochastic/RK stage without skipping a physically resolvable interval.
@inline function _trajectory_step_to_target(t::R,target::R,
                                            maximum_step::R) where
        R<:AbstractFloat
    remaining=target-t
    # Scale the tolerance with the local ulp, not with an absolute unit scale.
    # The latter would let, for example, a `1e-20` maximum step jump directly
    # across a physically resolvable `1e-16` interval in Float64.
    scale=max(abs(t),abs(target),abs(maximum_step))
    tolerance=R(8)*eps(scale)
    lands=remaining<=maximum_step+tolerance
    lands ? (remaining,true) : (maximum_step,false)
end

function _event_driven_trajectory(plan,rho0,ts,w,rng,parameters,dt,
                                  abstol,reltol,dtmin,dtmax,event_time_tolerance;
                                  observable_ops=nothing,
                                  observable_values=nothing,
                                  state_sampler=nothing,
                                  save_states::Bool=true,
                                  record_jumps::Bool=true)
    save_states&&!record_jumps&&throw(ArgumentError(
        "saved trajectories require recorded jump histories"))
    b=plan.model.basis;tau=plan.liouvillian.tracevec;x=w.current
    copyto!(x,rho0.data)
    states=save_states ? Vector{typeof(rho0)}(undef,length(ts)) : nothing
    save_states&&(states[1]=copy(rho0))
    observable_values===nothing||_record_observables!(
        observable_values,observable_ops,x,1)
    state_sampler===nothing||_record_trajectory_state!(state_sampler,x,1)
    jt=record_jumps ? eltype(ts)[] : nothing
    jc=record_jumps ? Int[] : nothing;t=ts[1]
    R=typeof(dt)
    threshold=randexp(rng,R);hazard=zero(threshold)
    h=min(dt,dtmax)
    for output_index in 2:length(ts)
        target=ts[output_index]
        while t<target
            remaining_to_target=target-t
            h,lands_on_target=_trajectory_step_to_target(
                t,target,min(h,dtmax))
            # As in standard adaptive integrators, landing exactly on a saved
            # output time may require one final step shorter than dtmin.
            minimum_step=min(dtmin,remaining_to_target)
            h>=minimum_step||throw(ErrorException(
                "adaptive trajectory step fell below dtmin=$dtmin at t=$t"))
            copyto!(w.start,x)
            increment,error=_conditional_dopri_trial!(w,x,b,t,h,parameters,tau,abstol,reltol)
            if !(isfinite(error)&&isfinite(increment))
                throw(ErrorException("non-finite adaptive trajectory trial at t=$t"))
            end
            if error>1
                h>minimum_step||throw(ErrorException(
                    "adaptive trajectory cannot satisfy its error tolerance above dtmin=$dtmin at t=$t"))
                h=max(minimum_step,h*_adaptive_factor(error));continue
            end
            increment>=-10abstol||throw(ErrorException("jump hazard decreased by $increment"))
            increment=max(zero(increment),increment)
            if hazard+increment < threshold
                copyto!(x,w.trial);z=dot(tau,x)
                abs(z)>eps(R)||throw(ArgumentError("conditional state acquired zero trace"));x./=z
                t=lands_on_target ? target : t+h;hazard+=increment
                h=min(dtmax,max(dtmin,h*_adaptive_factor(error)))
                continue
            end

            # A continuous event occurred inside the accepted step. Locate the
            # hazard root from the unchanged step-start state, then apply the
            # selected channel at that physical event time.
            remaining=threshold-hazard
            _prepare_dopri_dense_output!(w)
            time_tol=max(event_time_tolerance,
                R(8)*eps(max(abs(t),one(t))))
            event_step=_dopri_dense_root(
                w,h,remaining,increment,time_tol)
            if event_step===nothing
                h>minimum_step||throw(ErrorException(
                    "Dormand--Prince dense hazard lost its event bracket above dtmin=$dtmin at t=$t"))
                h=max(minimum_step,h/2)
                continue
            end
            theta=event_step/h
            _dopri_dense_state!(x,w.start,w,h,theta)
            z=dot(tau,x)
            abs(z)>eps(R)||throw(ArgumentError("conditional state acquired zero trace"));x./=z
            t+=event_step
            rates=_channel_intensities!(w,x,b,t,parameters)
            lambda=_total_intensity(rates)
            lambda>0||throw(ErrorException("hazard root has zero channel intensity at t=$t"))
            u=rand(rng,typeof(lambda))*lambda
            channel=_select_jump_channel(rates,u)
            _apply_gain!(w.channel_gain,x,plan.jumps[channel],b,
                         w.jump_scales[channel],w.liouvillian_work)
            z=dot(tau,w.channel_gain);abs(z)>eps(R)||throw(ArgumentError("selected jump has zero probability"))
            copyto!(x,w.channel_gain);x./=z
            if record_jumps
                push!(jt,t);push!(jc,channel)
            end
            threshold=randexp(rng,typeof(threshold));hazard=zero(threshold)
            h=min(dtmax,max(dtmin,h-event_step))
        end
        save_states&&(states[output_index]=PIState(b,x))
        observable_values===nothing||_record_observables!(
            observable_values,observable_ops,x,output_index)
        state_sampler===nothing||_record_trajectory_state!(
            state_sampler,x,output_index)
    end
    save_states ? QuantumTrajectory(ts,states,jt,jc) :
        record_jumps ? (;jump_times=jt,jump_channels=jc) : nothing
end

function _prepare_trajectory_arguments(times,::Type{R};dt::Real,
        parameters=nothing,max_jump_probability=nothing,
        algorithm::Symbol=:fixed,abstol=nothing,reltol=nothing,
        dtmin=nothing,dtmax=nothing,event_time_tolerance=nothing) where R<:AbstractFloat
    dt=_trajectory_real_input(R,dt,"dt")
    max_jump_probability=max_jump_probability===nothing ? R(0.05) :
        _trajectory_real_input(R,max_jump_probability,"max_jump_probability")
    abstol=abstol===nothing ? R(1e-9) :
        _trajectory_real_input(R,abstol,"abstol")
    reltol=reltol===nothing ? R(1e-7) :
        _trajectory_real_input(R,reltol,"reltol")
    dtmin=dtmin===nothing ? eps(R) : _trajectory_real_input(R,dtmin,"dtmin")
    dtmax=dtmax===nothing ? dt : _trajectory_real_input(R,dtmax,"dtmax")
    event_time_tolerance=event_time_tolerance===nothing ? R(1e-10) :
        _trajectory_real_input(R,event_time_tolerance,"event_time_tolerance")
    dt>0||throw(ArgumentError("dt must be positive"))
    0<max_jump_probability<1||throw(ArgumentError(
        "max_jump_probability must lie in (0,1)"))
    algorithm in (:fixed,:event,:adaptive,:event_driven)||throw(ArgumentError(
        "algorithm must be :fixed or :event"))
    abstol>0||throw(ArgumentError("abstol must be positive"))
    reltol>0||throw(ArgumentError("reltol must be positive"))
    dtmin>0||throw(ArgumentError("dtmin must be positive"))
    dtmax>=dtmin||throw(ArgumentError("dtmax must be at least dtmin"))
    event_time_tolerance>0||throw(ArgumentError(
        "event_time_tolerance must be positive"))
    raw_times=collect(times)
    isempty(raw_times)&&throw(ArgumentError("at least one output time is required"))
    ts=Vector{R}(undef,length(raw_times))
    for index in eachindex(raw_times)
        ts[index]=_trajectory_real_input(R,raw_times[index],
                                         "output time at index $index")
    end
    all(diff(ts).>=0)||throw(ArgumentError("times must be nondecreasing"))
    options=(;dt,parameters,max_jump_probability,algorithm,abstol,reltol,
             dtmin,dtmax,event_time_tolerance)
    ts,options
end

_trajectory_source_matches(plan::TrajectoryPlan,source::PIModel)=
    plan.model===source
_trajectory_source_matches(plan::TrajectoryPlan,source::TrajectoryPlan)=
    plan===source
_trajectory_source_matches(plan::TrajectoryPlan,source::CompiledPIModel)=
    plan.model===source.model&&
    (plan.liouvillian===source.plan||isempty(source.model.terms))

function _check_trajectory_workspace(work::TrajectoryWorkspace,
                                     source,rho::PIState)
    _trajectory_source_matches(work.plan,source)||throw(ArgumentError(
        "trajectory workspace was prepared for a different model or plan"))
    rho.basis===work.plan.model.basis||throw(ArgumentError(
        "state and trajectory workspace use incompatible PI bases"))
    eltype(work.tmp)===eltype(rho.data)||throw(ArgumentError(
        "trajectory workspace has an incompatible scalar type"))
    work
end

function _require_trajectory_workspace_mode(work::TrajectoryWorkspace,
                                            algorithm::Symbol)
    algorithm===:fixed||work.mode===:full||throw(ArgumentError(
        "algorithm=:event requires TrajectoryWorkspace(mode=:full); " *
        "the supplied fixed-only workspace omits adaptive stages"))
    work
end

function _check_trajectory_batch_workspace(work::TrajectoryBatchWorkspace,
                                           source,rho::PIState)
    _trajectory_source_matches(work.plan,source)||throw(ArgumentError(
        "trajectory batch workspace was prepared for a different model or plan"))
    isempty(work.workers)&&throw(ArgumentError(
        "trajectory batch workspace has no workers"))
    length(work.workers)==length(work.rngs)||throw(ArgumentError(
        "trajectory batch workspace has inconsistent worker storage"))
    work.seeds isa Vector{UInt64}||throw(ArgumentError(
        "trajectory batch workspace has incompatible seed storage"))
    all(rng->rng isa AbstractRNG,work.rngs)||throw(ArgumentError(
        "trajectory batch workspace has incompatible RNG storage"))
    for worker in work.workers
        worker.plan===work.plan||throw(ArgumentError(
            "trajectory batch workers do not share its plan"))
        rho.basis===worker.plan.model.basis||throw(ArgumentError(
            "state and trajectory batch workspace use incompatible PI bases"))
        eltype(worker.tmp)===eltype(rho.data)||throw(ArgumentError(
            "trajectory batch workspace has an incompatible scalar type"))
    end
    work
end

_plan_for_source(source,rho)=_trajectory_plan_for_state(source,rho)

function _validate_trajectory_initial_state(plan,rho0)
    rho0.basis===plan.model.basis||throw(ArgumentError(
        "state and trajectory plan use incompatible PI bases"))
    R=_real_float_type(eltype(rho0.data))
    tolerance=max(R(1e-10),R(100)*eps(R))
    abs(trace(rho0)-one(R))<=tolerance||throw(ArgumentError(
        "initial state must have unit trace"))
    nothing
end

function _trajectory_history_bytes(state_dimension::Integer,::Type{T},
        time_count::Integer,path_count::Integer,::Type{RT}) where {T,RT}
    entries=BigInt(state_dimension)*BigInt(time_count)*BigInt(path_count)
    time_entries=BigInt(time_count)*BigInt(path_count)
    _performance_entries_bytes(entries,T)+
        _performance_entries_bytes(time_entries,RT)
end

function _guard_trajectory_history(label,state_dimension,::Type{T},times,
        path_count,memory_budget;guidance) where T
    estimate=_trajectory_history_bytes(state_dimension,T,length(times),
        path_count,eltype(times))
    _require_performance_budget(label,estimate,memory_budget;guidance)
end

function _trajectory_observable_statistics_bytes(nobservables::Integer,
        ntimes::Integer,worker_count::Integer,::Type{R}) where R
    # Per worker: one path buffer and Welford mean/M2. Final reporting retains
    # mean, variance, standard error, and two confidence limits while the
    # merged accumulator is still live.
    entries=BigInt(nobservables)*BigInt(ntimes)*(3BigInt(worker_count)+5)
    _performance_entries_bytes(entries,R)
end

function _quantum_trajectory_prepared(plan,rho0,ts,w,rng,options;
                                      observable_ops=nothing,
                                      observable_values=nothing,
                                      state_sampler=nothing,
                                      save_states::Bool=true,
                                      record_jumps::Bool=true)
    save_states&&!record_jumps&&throw(ArgumentError(
        "saved trajectories require recorded jump histories"))
    _require_trajectory_workspace_mode(w,options.algorithm)
    b=plan.model.basis;tau=plan.liouvillian.tracevec
    R=eltype(w.intensities)
    options.algorithm!==:fixed&&return _event_driven_trajectory(
        plan,rho0,ts,w,rng,options.parameters,options.dt,options.abstol,
        options.reltol,options.dtmin,options.dtmax,
        options.event_time_tolerance;observable_ops,observable_values,
        state_sampler,save_states,record_jumps)
    x=w.current;copyto!(x,rho0.data)
    states=save_states ? Vector{typeof(rho0)}(undef,length(ts)) : nothing
    save_states&&(states[1]=copy(rho0))
    observable_values===nothing||_record_observables!(
        observable_values,observable_ops,x,1)
    state_sampler===nothing||_record_trajectory_state!(state_sampler,x,1)
    jt=record_jumps ? eltype(ts)[] : nothing
    jc=record_jumps ? Int[] : nothing;t=ts[1]
    for output_index in 2:length(ts)
        target=ts[output_index]
        while t<target
            h,lands_on_target=_trajectory_step_to_target(
                t,target,options.dt)
            lambda=_effective_jump_intensity!(
                w,x,b,t,options.parameters)
            if lambda*h>options.max_jump_probability
                h=options.max_jump_probability/lambda
                lands_on_target=false
            end
            _conditional_rk4!(x,w,b,t,h,options.parameters,tau)
            t=lands_on_target ? target : t+h
            lambda=_effective_jump_intensity!(
                w,x,b,t,options.parameters)
            jump_probability=-expm1(-lambda*h)
            if lambda>0&&rand(rng,typeof(jump_probability))<jump_probability
                rates=_channel_intensities!(w,x,b,t,options.parameters)
                selected_total=_total_intensity(rates)
                tolerance=_intensity_tolerance(typeof(lambda))*
                    max(one(lambda),lambda,selected_total)
                abs(selected_total-lambda)<=tolerance||throw(ArgumentError(
                    "combined and channel-resolved jump intensities disagree at a selected event"))
                u=rand(rng,typeof(selected_total))*selected_total
                channel=_select_jump_channel(rates,u)
                _apply_gain!(w.channel_gain,x,plan.jumps[channel],b,
                             w.jump_scales[channel],w.liouvillian_work)
                z=dot(tau,w.channel_gain)
                abs(z)>eps(R)||throw(ArgumentError(
                    "selected jump has zero probability"))
                copyto!(x,w.channel_gain);x./=z
                if record_jumps
                    push!(jt,t);push!(jc,channel)
                end
            end
        end
        save_states&&(states[output_index]=PIState(b,x))
        observable_values===nothing||_record_observables!(
            observable_values,observable_ops,x,output_index)
        state_sampler===nothing||_record_trajectory_state!(
            state_sampler,x,output_index)
    end
    save_states ? QuantumTrajectory(ts,states,jt,jc) :
        record_jumps ? (;jump_times=jt,jump_channels=jc) : nothing
end

"""
    quantum_trajectory(source, rho0, times; dt, algorithm=:fixed, rng,
                       parameters=nothing, max_jump_probability=0.05,
                       abstol=1e-9, reltol=1e-7,
                       dtmin=eps(R), dtmax=dt,
                       event_time_tolerance=1e-10, workspace=nothing)

Simulate one PI quantum-jump trajectory from a `PIModel`, `CompiledPIModel`,
or reusable [`TrajectoryPlan`](@ref). Local jump channels are unresolved
over particle labels and therefore generally produce mixed conditional PI
states. The fixed step is automatically shortened so the total jump
probability remains below `max_jump_probability`.

All rate-weighted channel loss blocks are combined into one effective Schur
operator per integration stage. The normalized no-jump drift therefore does
not apply one matrix product per channel; individual channel intensities are
evaluated only when an event must select its channel.

Set `algorithm=:event` (aliases `:adaptive` and `:event_driven`) to integrate
the normalized no-jump equation and its accumulated hazard with an embedded
Dormand--Prince 5(4) method. Jump times are then continuous hazard roots,
rather than endpoints of Bernoulli time steps. Root bisection evaluates the
quartic continuous extension of the accepted stages, so it does not rerun a
seven-stage trial at every candidate time. `dt` is the initial adaptive
step and `dtmax` its upper bound; `abstol`, `reltol`, `dtmin`, and
`event_time_tolerance` control the adaptive solve. A final step may be shorter
than `dtmin` solely to land on a requested output time. Here `R` is the real
floating precision of the prepared trajectory. Time grids and explicitly
supplied controls must be representable in `R` without narrowing; default
controls are constructed directly in `R`.

`memory_budget` preflights the predictable saved state/time history before
trajectory-plan or workspace construction. It is an output lower-bound guard;
data-dependent jump records and reusable worker scratch remain additional.
"""
function quantum_trajectory(source::Union{PIModel,CompiledPIModel,TrajectoryPlan},
                            rho0::PIState{R},times;
                            dt::Real, rng::AbstractRNG=Random.default_rng(),parameters=nothing,
                            max_jump_probability=nothing,workspace=nothing,
                            algorithm::Symbol=:fixed,abstol=nothing,
                            reltol=nothing,dtmin=nothing,
                            dtmax=nothing,event_time_tolerance=nothing,
                            memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET) where {R<:AbstractFloat}
    ts,options=_prepare_trajectory_arguments(times,R;dt,parameters,
        max_jump_probability,algorithm,abstol,reltol,dtmin,dtmax,
        event_time_tolerance)
    _guard_trajectory_history("quantum-trajectory state history",
        length(rho0.data),eltype(rho0.data),ts,1,memory_budget;
        guidance="Request fewer saved times for a long path.")
    if workspace===nothing
        plan=_plan_for_source(source,rho0)
        mode=algorithm===:fixed ? :fixed : :full
        w=TrajectoryWorkspace(plan,rho0;mode)
    else
        workspace isa TrajectoryWorkspace||throw(ArgumentError(
            "workspace must be a TrajectoryWorkspace"))
        w=_check_trajectory_workspace(workspace,source,rho0);plan=w.plan
    end
    _validate_trajectory_initial_state(plan,rho0)
    _quantum_trajectory_prepared(plan,rho0,ts,w,rng,options)
end

"""
    quantum_trajectories(source, rho0, times, n; seed=0, threaded=false,
                         workspace=nothing, observables=nothing,
                         save_states=true, jump_statistics=true,
                         confidence=0.95, trajectory_keywords...)

Generate independent PI trajectories. Model geometry is lowered once per
batch and shared read-only. Serial execution reuses one worker; threaded
execution uses dynamically scheduled task-owned workers and small work chunks,
avoiding scratch races while amortizing atomic scheduling and retaining load
balance for paths with different jump counts. Random streams are seeded by
trajectory index, so a fixed seed gives the same ordered results independently
of scheduling and thread count.

Pass a reusable [`TrajectoryBatchWorkspace`](@ref) to amortize setup across
several ensembles. A single [`TrajectoryWorkspace`](@ref) is accepted only for
serial execution. Returned trajectories retain independent time and state
storage. With `threaded=true`, callable scalar rates and objects supplied via
`parameters` must themselves be safe for concurrent read/evaluation. Because
every requested state is returned, output storage scales as
`n * length(times) * length(rho0.data)`; request only the sampling times needed
for analysis when state-history memory is limiting.

Pass named `observables` and set `save_states=false` for memory-light online
ensemble statistics. This returns a [`TrajectoryEnsembleResult`](@ref) and
never constructs sampled `PIState` objects. One observable buffer and one
Welford accumulator are retained per active worker. Channel-resolved jump
statistics are accumulated online by default; set `jump_statistics=false` to
omit them. A state-free call requires at least one observable because the
no-observable route preserves the legacy trajectory-vector return type.
The default `memory_budget` rejects predictable saved state/time histories and
retained observable-statistics arrays above 512 MiB before their allocation.
It does not bound the data-dependent number of jump records; disable
`jump_statistics` when those records are unnecessary.
"""
function quantum_trajectories(source::Union{PIModel,CompiledPIModel,TrajectoryPlan},
        rho0::PIState{R},times,n::Integer;seed::Integer=0,
        threaded::Bool=false,workspace=nothing,dt::Real,
        parameters=nothing,max_jump_probability=nothing,
        algorithm::Symbol=:fixed,abstol=nothing,reltol=nothing,
        dtmin=nothing,dtmax=nothing,event_time_tolerance=nothing,
        observables=nothing,save_states::Bool=true,
        jump_statistics::Bool=true,confidence::Real=0.95,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET) where {R<:AbstractFloat}
    _quantum_trajectories_dispatch(observables,
        source,rho0,times,n;
        seed,threaded,workspace,dt,parameters,max_jump_probability,algorithm,
        abstol,reltol,dtmin,dtmax,event_time_tolerance,save_states,
        jump_statistics,confidence,memory_budget)
end

function _quantum_trajectories_dispatch(::Nothing,
        source,rho0,times,n;
        seed,threaded,workspace,dt,parameters,max_jump_probability,algorithm,
        abstol,reltol,dtmin,dtmax,event_time_tolerance,save_states,
        jump_statistics,confidence,memory_budget)
    save_states||throw(ArgumentError(
        "save_states=false requires at least one observable"))
    _quantum_trajectories_legacy(source,rho0,times,n;seed,threaded,workspace,
        dt,parameters,max_jump_probability,algorithm,abstol,reltol,dtmin,
        dtmax,event_time_tolerance,memory_budget)
end

function _quantum_trajectories_dispatch(observables,
        source,rho0,times,n;seed,threaded,workspace,dt,parameters,
        max_jump_probability,algorithm,abstol,reltol,dtmin,dtmax,
        event_time_tolerance,save_states,jump_statistics,confidence,
        memory_budget)
    _quantum_trajectories_streaming(source,rho0,times,n;
        seed,threaded,workspace,dt,parameters,max_jump_probability,algorithm,
        abstol,reltol,dtmin,dtmax,event_time_tolerance,observables,
        save_states,jump_statistics,confidence,memory_budget)
end

function _quantum_trajectories_legacy(
        source::Union{PIModel,CompiledPIModel,TrajectoryPlan},
        rho0::PIState{R},times,n::Integer;seed::Integer,threaded::Bool,
        workspace,dt::Real,parameters,max_jump_probability,algorithm::Symbol,
        abstol,reltol,dtmin,dtmax,event_time_tolerance,
        memory_budget) where {R<:AbstractFloat}
    n>0||throw(ArgumentError("trajectory count must be positive"))
    ts,options=_prepare_trajectory_arguments(times,R;dt,parameters,
        max_jump_probability,algorithm,abstol,reltol,dtmin,dtmax,
        event_time_tolerance)
    _guard_trajectory_history("quantum-trajectory ensemble state history",
        length(rho0.data),eltype(rho0.data),ts,n,memory_budget;
        guidance=
        "Pass observables=... and save_states=false for online statistics.")
    if workspace===nothing
        plan=_plan_for_source(source,rho0)
        worker_count=threaded ? min(Int(n),Threads.nthreads()) : 1
        mode=algorithm===:fixed ? :fixed : :full
        batch=TrajectoryBatchWorkspace(plan,rho0;workers=worker_count,mode)
    elseif workspace isa TrajectoryBatchWorkspace
        batch=_check_trajectory_batch_workspace(workspace,source,rho0)
        plan=batch.plan
    elseif workspace isa TrajectoryWorkspace
        threaded&&throw(ArgumentError(
            "threaded ensembles require a TrajectoryBatchWorkspace with independent worker scratch"))
        _check_trajectory_workspace(workspace,source,rho0)
        plan=workspace.plan
        batch=nothing
    else
        throw(ArgumentError(
            "workspace must be a TrajectoryWorkspace or TrajectoryBatchWorkspace"))
    end
    _validate_trajectory_initial_state(plan,rho0)
    if batch===nothing
        master=MersenneTwister(seed);seeds=rand(master,UInt64,n)
    else
        master=batch.rngs[1]
        Random.seed!(master,seed)
        resize!(batch.seeds,n);rand!(master,batch.seeds)
        seeds=batch.seeds
    end
    TT=eltype(ts)
    out=Vector{QuantumTrajectory{TT,typeof(rho0)}}(undef,n)
    if batch===nothing
        rng=master
        for i in 1:n
            Random.seed!(rng,seeds[i])
            out[i]=_quantum_trajectory_prepared(plan,rho0,copy(ts),workspace,
                                                rng,options)
        end
    else
        available=length(batch.workers)
        worker_count=threaded ? min(Int(n),Threads.nthreads(),available) : 1
        if worker_count==1
            worker=batch.workers[1];rng=batch.rngs[1]
            for i in 1:n
                Random.seed!(rng,seeds[i])
                out[i]=_quantum_trajectory_prepared(plan,rho0,copy(ts),worker,
                                                    rng,options)
            end
        else
            # Fetch small chunks rather than one index at a time. This keeps
            # enough chunks for load balance while amortizing atomic traffic
            # when thousands of short paths are requested.
            chunk_size=max(1,Int(n)÷(8worker_count))
            next_index=Threads.Atomic{Int}(1)
            @sync for worker_index in 1:worker_count
                # Bind task-owned resources outside the spawned closure. A
                # captured loop index is otherwise boxed and can make several
                # tasks select the same mutable workspace/RNG.
                let worker=batch.workers[worker_index],
                    rng=batch.rngs[worker_index],
                    trajectory_plan=plan,
                    trajectory_seeds=seeds,
                    trajectory_times=ts,
                    trajectory_options=options,
                    initial_state=rho0,
                    results=out,
                    path_count=Int(n),
                    counter=next_index,
                    chunk=chunk_size
                    Threads.@spawn begin
                        while true
                            first_index=Threads.atomic_add!(counter,chunk)
                            first_index>path_count&&break
                            final_index=min(path_count,first_index+chunk-1)
                            for i in first_index:final_index
                                Random.seed!(rng,trajectory_seeds[i])
                                results[i]=_quantum_trajectory_prepared(
                                    trajectory_plan,initial_state,
                                    copy(trajectory_times),worker,rng,
                                    trajectory_options)
                            end
                        end
                    end
                end
            end
        end
    end
    out
end

function _stream_trajectory_path!(out,observable_accumulators,jump_accumulators,
        observable_buffers,observable_ops,plan,rho0,ts,worker,rng,options,
        trajectory_index,worker_index,save_states,record_jumps)
    values=observable_buffers===nothing ? nothing :
        observable_buffers[worker_index]
    result=_quantum_trajectory_prepared(plan,rho0,
        save_states ? copy(ts) : ts,worker,rng,options;
        observable_ops,observable_values=values,save_states,record_jumps)
    save_states&&(out[trajectory_index]=result)
    observable_accumulators===nothing||_accumulate_observables!(
        observable_accumulators[worker_index],values)
    if jump_accumulators!==nothing
        _accumulate_jumps!(jump_accumulators[worker_index],
            result.jump_times,result.jump_channels)
    end
    nothing
end

function _quantum_trajectories_streaming(
        source::Union{PIModel,CompiledPIModel,TrajectoryPlan},
        rho0::PIState{R},times,n::Integer;seed::Integer,threaded::Bool,
        workspace,dt::Real,parameters,max_jump_probability,algorithm::Symbol,
        abstol,reltol,dtmin,dtmax,event_time_tolerance,observables,
        save_states::Bool,jump_statistics::Bool,confidence::Real,
        memory_budget) where {R<:AbstractFloat}
    n>0||throw(ArgumentError("trajectory count must be positive"))
    !save_states&&observables===nothing&&throw(ArgumentError(
        "save_states=false requires at least one observable"))
    ts,options=_prepare_trajectory_arguments(times,R;dt,parameters,
        max_jump_probability,algorithm,abstol,reltol,dtmin,dtmax,
        event_time_tolerance)
    if save_states
        _guard_trajectory_history("quantum-trajectory ensemble state history",
            length(rho0.data),eltype(rho0.data),ts,n,memory_budget;
            guidance=
            "Set save_states=false to retain only online observable statistics.")
    else
        _performance_memory_limit(memory_budget)
    end
    if workspace===nothing
        plan=_plan_for_source(source,rho0)
        requested_workers=threaded ? min(Int(n),Threads.nthreads()) : 1
        mode=algorithm===:fixed ? :fixed : :full
        batch=TrajectoryBatchWorkspace(plan,rho0;workers=requested_workers,mode)
    elseif workspace isa TrajectoryBatchWorkspace
        batch=_check_trajectory_batch_workspace(workspace,source,rho0)
        plan=batch.plan
    elseif workspace isa TrajectoryWorkspace
        threaded&&throw(ArgumentError(
            "threaded ensembles require a TrajectoryBatchWorkspace with independent worker scratch"))
        _check_trajectory_workspace(workspace,source,rho0)
        plan=workspace.plan;batch=nothing
    else
        throw(ArgumentError(
            "workspace must be a TrajectoryWorkspace or TrajectoryBatchWorkspace"))
    end
    _validate_trajectory_initial_state(plan,rho0)
    ops=_prepare_streaming_observables(rho0.basis,observables;
                                       require_hermitian=true)
    0<confidence<1||throw(ArgumentError(
        "confidence must lie in (0,1)"))

    if batch===nothing
        master=MersenneTwister(seed);seeds=rand(master,UInt64,n)
        workers=(workspace,);rngs=(master,);worker_count=1
    else
        master=batch.rngs[1];Random.seed!(master,seed)
        resize!(batch.seeds,n);rand!(master,batch.seeds);seeds=batch.seeds
        available=length(batch.workers)
        worker_count=threaded ? min(Int(n),Threads.nthreads(),available) : 1
        workers=batch.workers;rngs=batch.rngs
    end
    TT=eltype(ts)
    out=save_states ? Vector{QuantumTrajectory{TT,typeof(rho0)}}(undef,n) : nothing
    Rstats=_observable_scalar_type(rho0,ops)
    statistics_estimate=_trajectory_observable_statistics_bytes(
        length(ops),length(ts),worker_count,Rstats)
    history_estimate=save_states ? _trajectory_history_bytes(
        length(rho0.data),eltype(rho0.data),length(ts),n,eltype(ts)) : big(0)
    _require_performance_budget("trajectory ensemble retained output",
        history_estimate+statistics_estimate,memory_budget;guidance=
        "Set save_states=false, request fewer times, or reduce the observable set.")
    observable_buffers=
        [Matrix{Rstats}(undef,length(ops),length(ts)) for _ in 1:worker_count]
    observable_accumulators=
        [_OnlineObservableAccumulator(Rstats,length(ops),length(ts))
         for _ in 1:worker_count]
    jump_accumulators=jump_statistics ?
        [_OnlineJumpAccumulator(R,length(plan.jumps)) for _ in 1:worker_count] :
        nothing
    record_jumps=save_states||jump_statistics

    if worker_count==1
        worker=workers[1];rng=rngs[1]
        for trajectory_index in 1:Int(n)
            Random.seed!(rng,seeds[trajectory_index])
            _stream_trajectory_path!(out,observable_accumulators,
                jump_accumulators,observable_buffers,ops,plan,rho0,ts,worker,
                rng,options,trajectory_index,1,save_states,record_jumps)
        end
    else
        chunk_size=max(1,Int(n)÷(8worker_count))
        next_index=Threads.Atomic{Int}(1)
        @sync for worker_index in 1:worker_count
            let worker=workers[worker_index],rng=rngs[worker_index],
                worker_id=worker_index,counter=next_index
                Threads.@spawn begin
                    while true
                        first_index=Threads.atomic_add!(counter,chunk_size)
                        first_index>Int(n)&&break
                        final_index=min(Int(n),first_index+chunk_size-1)
                        for trajectory_index in first_index:final_index
                            Random.seed!(rng,seeds[trajectory_index])
                            _stream_trajectory_path!(out,
                                observable_accumulators,jump_accumulators,
                                observable_buffers,ops,plan,rho0,ts,worker,rng,
                                options,trajectory_index,worker_id,save_states,
                                record_jumps)
                        end
                    end
                end
            end
        end
    end

    merged_observables=observable_accumulators[1]
    for worker_index in 2:worker_count
        _merge_observables!(merged_observables,
                            observable_accumulators[worker_index])
    end
    observable_summary=_observable_statistics(
        merged_observables,ops,ts,confidence)
    jump_summary=if jump_accumulators===nothing
        nothing
    else
        merged=jump_accumulators[1]
        for worker_index in 2:worker_count
            _merge_jumps!(merged,jump_accumulators[worker_index])
        end
        _jump_statistics(merged,ts)
    end
    P=Union{Nothing,Vector{QuantumTrajectory{TT,typeof(rho0)}}}
    TrajectoryEnsembleResult{TT,P,typeof(observable_summary)}(
        copy(ts),out,observable_summary,jump_summary,Int(n))
end

function _trajectory_integer_count(value::Integer,label,minimum::Int)
    converted=try
        Int(value)
    catch
        throw(ArgumentError("$label=$value cannot be represented as an Int"))
    end
    converted>=minimum||throw(ArgumentError(
        "$label must be at least $minimum"))
    converted
end

function _trajectory_steady_sampling_times(::Type{R},settling_time,
        samples_per_trajectory::Int,sampling_interval) where R<:AbstractFloat
    settling=_trajectory_real_input(R,settling_time,"settling_time")
    settling>zero(R)||throw(ArgumentError("settling_time must be positive"))
    interval=if sampling_interval===nothing
        samples_per_trajectory==1||throw(ArgumentError(
            "sampling_interval is required when samples_per_trajectory is greater than one"))
        nothing
    else
        value=_trajectory_real_input(R,sampling_interval,"sampling_interval")
        value>zero(R)||throw(ArgumentError(
            "sampling_interval must be positive"))
        value
    end
    sample_times=Vector{R}(undef,samples_per_trajectory)
    sample_times[1]=settling
    if samples_per_trajectory>1
        for index in 2:samples_per_trajectory
            offset=_checked_statistics_count(R,index-1,
                "trajectory time-sample offset")
            value=settling+offset*interval
            isfinite(value)||throw(ArgumentError(
                "post-settling sampling times overflow trajectory precision $R"))
            value>sample_times[index-1]||throw(ArgumentError(
                "sampling_interval is too small to produce distinct sampling times in trajectory precision $R"))
            sample_times[index]=value
        end
    end
    times=Vector{R}(undef,length(sample_times)+1)
    times[1]=zero(R)
    copyto!(times,2,sample_times,1,length(sample_times))
    times,sample_times,settling,interval
end

function _steady_state_path!(sampler,state_accumulator,
        observable_accumulator,observable_buffer,observable_ops,
        plan,rho0,times,worker,rng,options,samples_per_trajectory)
    _reset_trajectory_state_sampler!(sampler)
    _quantum_trajectory_prepared(plan,rho0,times,worker,rng,options;
        state_sampler=sampler,save_states=false,record_jumps=false)
    sampler.count==samples_per_trajectory||throw(ErrorException(
        "internal trajectory sampler retained $(sampler.count) states instead of $samples_per_trajectory"))
    _accumulate_state!(state_accumulator,sampler.mean)
    if observable_accumulator!==nothing
        _record_observables!(observable_buffer,observable_ops,sampler.mean,1)
        _accumulate_observables!(observable_accumulator,observable_buffer)
    end
    nothing
end

"""
    trajectory_steady_state(source, rho0;
        trajectories, settling_time, dt,
        samples_per_trajectory=1, sampling_interval=nothing,
        seed=0, threaded=false, workspace=nothing,
        observables=nothing, confidence=0.95, return_info=false,
        parameters=nothing, max_jump_probability=0.05,
        algorithm=:fixed, abstol=1e-9, reltol=1e-7,
        dtmin=eps(R), dtmax=dt, event_time_tolerance=1e-10)

Estimate an autonomous stationary PI density operator with quantum-jump
trajectories. Each independent path is evolved from `rho0` to
`settling_time`. One or more states are then sampled at equal
`sampling_interval`s and averaged within that path. The returned density
operator is the average of those independent path means. No state history or
jump record is constructed, so retained state storage is
`O(workers * length(rho0.data))` rather than proportional to the number of
paths or post-settling samples.

At least two trajectories are required. Multiple samples can reduce the
variance of an ergodic path average, but samples from the same path may be
correlated. They are therefore averaged before the path-to-path uncertainty
is evaluated and never counted as independent observations. Choose the
settling time, sampling interval, sampling-window length, path count, and
integration controls through explicit convergence studies. Strong symmetries
or multiple stationary states can make the result depend on `rho0`.

The source may be a `PIModel`, `CompiledPIModel`, or reusable
[`TrajectoryPlan`](@ref), but it must be autonomous and use the fixed jump
operators supported by PI trajectories. For a periodically driven stationary
regime, use Floquet analysis. Call `freeze` only when the stationary state of
one explicitly selected instantaneous generator is the intended question;
freezing does not solve the driven problem. `threaded=true` uses task-owned
workspaces and trajectory-indexed random streams. Pass a
[`TrajectoryBatchWorkspace`](@ref) to reuse those buffers across calls.

By default, return the estimated `PIState`. Set `return_info=true` to receive
a [`TrajectorySteadyStateResult`](@ref), including the Hilbert--Schmidt sample
spread and standard-error norm, the Liouvillian residual, relative residual,
and trace error. These diagnostics are not a steady-state certificate and the
state is never normalized, symmetrized, or positivity-repaired. With
`return_info=true`, named Hermitian `observables` add means, unbiased
variances, standard errors, and normal confidence intervals across the
independent path means; requesting observables without the detailed result is
rejected rather than computing and discarding their statistics.
"""
function trajectory_steady_state(
        source::Union{PIModel,CompiledPIModel,TrajectoryPlan},
        rho0::PIState{R};trajectories::Integer,settling_time,dt::Real,
        samples_per_trajectory::Integer=1,sampling_interval=nothing,
        seed::Integer=0,threaded::Bool=false,workspace=nothing,
        observables=nothing,confidence::Real=0.95,return_info::Bool=false,
        parameters=nothing,max_jump_probability=nothing,
        algorithm::Symbol=:fixed,abstol=nothing,reltol=nothing,
        dtmin=nothing,dtmax=nothing,event_time_tolerance=nothing) where
        {R<:AbstractFloat}
    path_count_int=_trajectory_integer_count(trajectories,"trajectories",2)
    samples_per_path=_trajectory_integer_count(
        samples_per_trajectory,"samples_per_trajectory",1)
    _checked_statistics_count(R,path_count_int,"trajectory")
    _checked_statistics_count(R,samples_per_path,"trajectory time sample")
    samples_per_path<typemax(Int)||throw(ArgumentError(
        "samples_per_trajectory is too large to construct the sampling grid"))
    0<confidence<1||throw(ArgumentError("confidence must lie in (0,1)"))
    observables!==nothing&&!return_info&&throw(ArgumentError(
        "trajectory steady-state observables require return_info=true"))

    if workspace===nothing
        plan=_plan_for_source(source,rho0)
        requested_workers=threaded ? min(path_count_int,Threads.nthreads()) : 1
        mode=algorithm===:fixed ? :fixed : :full
        batch=TrajectoryBatchWorkspace(plan,rho0;workers=requested_workers,mode)
    elseif workspace isa TrajectoryBatchWorkspace
        batch=_check_trajectory_batch_workspace(workspace,source,rho0)
        plan=batch.plan
    elseif workspace isa TrajectoryWorkspace
        threaded&&throw(ArgumentError(
            "threaded steady-state trajectories require a TrajectoryBatchWorkspace with independent worker scratch"))
        _check_trajectory_workspace(workspace,source,rho0)
        plan=workspace.plan
        batch=nothing
    else
        throw(ArgumentError(
            "workspace must be a TrajectoryWorkspace or TrajectoryBatchWorkspace"))
    end
    _validate_trajectory_initial_state(plan,rho0)
    isautonomous(plan)||throw(ArgumentError(
        "trajectory_steady_state requires an autonomous model; call freeze(...; time=..., parameters=...) before estimating a stationary state of a driven source"))

    times,sampling_times,settling,interval=
        _trajectory_steady_sampling_times(R,settling_time,
            samples_per_path,sampling_interval)
    times,options=_prepare_trajectory_arguments(times,R;dt,parameters,
        max_jump_probability,algorithm,abstol,reltol,dtmin,dtmax,
        event_time_tolerance)
    ops=_prepare_streaming_observables(rho0.basis,observables;
                                       require_hermitian=true)

    if batch===nothing
        master=MersenneTwister(seed)
        seeds=rand(master,UInt64,path_count_int)
        workers=(workspace,)
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

    samplers=[_TrajectoryStateSampler(rho0.data) for _ in 1:worker_count]
    state_accumulators=[_OnlineStateAccumulator(rho0.data)
                        for _ in 1:worker_count]
    Rstats=_observable_scalar_type(rho0,ops)
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
            _steady_state_path!(samplers[1],state_accumulators[1],
                observable_accumulators===nothing ? nothing :
                    observable_accumulators[1],
                observable_buffers===nothing ? nothing : observable_buffers[1],
                ops,plan,rho0,times,worker,rng,options,
                samples_per_path)
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
                            _steady_state_path!(sampler,state_accumulator,
                                observable_accumulator,observable_buffer,ops,
                                plan,rho0,times,worker,rng,options,
                                samples_per_path)
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
        "internal trajectory reduction retained $(merged_state.count) paths instead of $trajectories"))
    path_variance_denominator=_checked_statistics_count(
        R,path_count_int-1,"trajectory")
    path_count=_checked_statistics_count(R,path_count_int,"trajectory")
    sample_spread=sqrt(merged_state.m2/path_variance_denominator)
    standard_error=sample_spread/sqrt(path_count)
    state=PIState(rho0.basis,merged_state.mean)

    residual_buffer=workers[1].tmp
    apply!(residual_buffer,plan.liouvillian,state.data,
           workers[1].liouvillian_work)
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
    metadata=(;algorithm=options.algorithm,seed,
        threaded_requested=threaded,threaded=worker_count>1,worker_count,
        dt=options.dt,max_jump_probability=options.max_jump_probability,
        abstol=options.abstol,reltol=options.reltol,dtmin=options.dtmin,
        dtmax=options.dtmax,
        event_time_tolerance=options.event_time_tolerance,
        settling_time=settling,sampling_interval=interval,confidence,
        path_reduction=:post_settling_mean,
        uncertainty_unit=:independent_path_mean)
    result=TrajectorySteadyStateResult(state,path_count_int,
        samples_per_path,copy(sampling_times),sample_spread,
        standard_error,residual,relative_residual,state_trace_error,
        observable_summary,metadata)
    return_info ? result : state
end

"""Average equally sampled quantum trajectories into PI density matrices."""
function trajectory_average(trajs::AbstractVector{<:QuantumTrajectory})
    isempty(trajs)&&throw(ArgumentError("at least one trajectory is required"));times=trajs[1].times;m=length(times);b=trajs[1].states[1].basis
    all(q->q.times==times&&q.states[1].basis===b,trajs)||throw(ArgumentError("trajectories must share times and basis"))
    out=[PIState(b;T=real(eltype(trajs[1].states[1].data))) for _ in 1:m]
    for q in trajs,i in 1:m;out[i].data .+= q.states[i].data;end
    for x in out;x.data./=length(trajs);end
    out
end

function _check_trajectory_ensemble(trajs)
    isempty(trajs)&&throw(ArgumentError("at least one trajectory is required"))
    times=trajs[1].times;b=trajs[1].states[1].basis
    all(q->q.times==times&&length(q.states)==length(times)&&q.states[1].basis===b,trajs)||throw(ArgumentError("trajectories must share sampling times and basis"))
    times,b
end

_sample_variance(m2,n)=n>1 ? m2/(n-1) : zero(m2)

"""
    jump_statistics(trajectories; nchannels=nothing)

Return total and channel-resolved jump-count statistics, empirical rates,
Fano factors, no-jump probability, and pooled inter-jump waiting times.
Unbiased sample variances are used when at least two trajectories are present.
"""
function jump_statistics(trajs::AbstractVector{<:QuantumTrajectory};nchannels=nothing)
    times,_=_check_trajectory_ensemble(trajs);n=length(trajs);duration=times[end]-times[1]
    duration>=0||throw(ArgumentError("invalid sampling interval"))
    inferred=maximum((isempty(q.jump_channels) ? 0 : maximum(q.jump_channels) for q in trajs);init=0)
    nc=nchannels===nothing ? inferred : Int(nchannels);nc>=inferred||throw(ArgumentError("nchannels is smaller than an observed channel index"));nc>=0||throw(ArgumentError("nchannels must be nonnegative"))
    means=zeros(Float64,nc);m2=zeros(Float64,nc);totmean=0.0;totm2=0.0;nojump=0;waiting=Float64[]
    totals=zeros(Int,nc);counts=zeros(Int,nc)
    for (r,q) in pairs(trajs)
        fill!(counts,0)
        for c in q.jump_channels;1<=c<=nc||throw(ArgumentError("jump channel indices must be positive"));counts[c]+=1;end
        for c in 1:nc
            totals[c]+=counts[c];delta=counts[c]-means[c];means[c]+=delta/r;m2[c]+=delta*(counts[c]-means[c])
        end
        nt=length(q.jump_times);nt==0&&(nojump+=1);delta=nt-totmean;totmean+=delta/r;totm2+=delta*(nt-totmean)
        nt>=2&&append!(waiting,diff(q.jump_times))
    end
    vars=[_sample_variance(m2[c],n) for c in 1:nc]
    channels=[(channel=c,total=totals[c],mean=means[c],variance=vars[c],
               fano=iszero(means[c]) ? NaN : vars[c]/means[c],
               rate=iszero(duration) ? NaN : means[c]/duration) for c in 1:nc]
    totalvar=_sample_variance(totm2,n)
    meanwait=isempty(waiting) ? NaN : sum(waiting)/length(waiting)
    waitvar=length(waiting)>1 ? sum(x->abs2(x-meanwait),waiting)/(length(waiting)-1) : NaN
    (;trajectories=n,duration,total_jumps=sum(totals),mean_count=totmean,
      count_variance=totalvar,fano=iszero(totmean) ? NaN : totalvar/totmean,
      rate=iszero(duration) ? NaN : totmean/duration,
      no_jump_probability=nojump/n,channels,waiting_times=waiting,
      mean_waiting_time=meanwait,waiting_time_variance=waitvar)
end

# Acklam's rational approximation for the standard-normal quantile.
function _normal_quantile(p::Real)
    0<p<1||throw(ArgumentError("probability must lie in (0,1)"))
    a=(-39.69683028665376,220.9460984245205,-275.9285104469687,138.3577518672690,-30.66479806614716,2.506628277459239)
    b=(-54.47609879822406,161.5858368580409,-155.6989798598866,66.80131188771972,-13.28068155288572)
    c=(-0.007784894002430293,-0.3223964580411365,-2.400758277161838,-2.549732539343734,4.374664141464968,2.938163982698783)
    d=(0.007784695709041462,0.3224671290700398,2.445134137142996,3.754408661907416)
    pl=0.02425
    if p<pl
        q=sqrt(-2log(p));return (((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6])/((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1)
    elseif p>1-pl
        q=sqrt(-2log(1-p));return -(((((c[1]*q+c[2])*q+c[3])*q+c[4])*q+c[5])*q+c[6])/((((d[1]*q+d[2])*q+d[3])*q+d[4])*q+1)
    end
    q=p-0.5;r=q*q
    (((((a[1]*r+a[2])*r+a[3])*r+a[4])*r+a[5])*r+a[6])*q/(((((b[1]*r+b[2])*r+b[3])*r+b[4])*r+b[5])*r+1)
end

"""
    trajectory_observable_statistics(trajectories, observables; confidence=0.95)

Compute time-resolved Monte Carlo means, unbiased variances, standard errors,
and normal confidence intervals. `observables` may be a named tuple, dictionary,
pair collection, or a single Hermitian local matrix/`PIOperator`.
"""
function trajectory_observable_statistics(trajs::AbstractVector{<:QuantumTrajectory},observables;confidence::Real=0.95)
    times,b=_check_trajectory_ensemble(trajs);0<confidence<1||throw(ArgumentError("confidence must lie in (0,1)"));n=length(trajs);nt=length(times)
    ops=_prepare_streaming_observables(b,observables;
                                       require_hermitian=true)
    z=_normal_quantile((1+confidence)/2);results=Dict{Any,Any}()
    for (name,A) in ops
        means=zeros(Float64,nt);m2=zeros(Float64,nt)
        for (r,q) in pairs(trajs),i in 1:nt
            value=real(expectation(q.states[i],A));delta=value-means[i];means[i]+=delta/r;m2[i]+=delta*(value-means[i])
        end
        vars=n>1 ? m2./(n-1) : zeros(nt);stderr=sqrt.(vars./n);half=z.*stderr
        results[name]=(mean=means,variance=vars,standard_error=stderr,
                       confidence=confidence,lower=means.-half,upper=means.+half)
    end
    (;times=copy(times),trajectories=n,observables=results)
end

"""Return averaged states together with jump and optional observable statistics."""
function trajectory_statistics(trajs::AbstractVector{<:QuantumTrajectory};observables=nothing,confidence::Real=0.95,nchannels=nothing)
    times,_=_check_trajectory_ensemble(trajs)
    obs=observables===nothing ? nothing : trajectory_observable_statistics(trajs,observables;confidence=confidence)
    (;times=copy(times),average_states=trajectory_average(trajs),jumps=jump_statistics(trajs;nchannels=nchannels),observables=obs)
end
