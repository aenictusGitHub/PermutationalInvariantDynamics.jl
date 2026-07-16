"""One PI quantum-jump realization and its recorded jump channel indices."""
struct QuantumTrajectory{T,S<:PIState}
    times::Vector{T}
    states::Vector{S}
    jump_times::Vector{T}
    jump_channels::Vector{Int}
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

function _trajectory_plan(model::PIModel,liouvillian_plan::LiouvillianPlan)
    all(term_has_fixed_operator,model.terms)||throw(ArgumentError(
        "trajectory kernels require fixed operators; scalar rates may depend on time"))
    kernels=liouvillian_plan.kernels
    kernels===nothing&&throw(ArgumentError(
        "trajectory kernels require terms that lower to prepared PI kernels"))
    supported=Union{HamiltonianPIKernel,DissipatorPIKernel,LocalJumpPIKernel}
    all(kernel->kernel isa supported,kernels)||throw(ArgumentError(
        "trajectory kernels require Hamiltonian, collective/direct-jump, or local-jump lowerings"))
    # Tuple filtering preserves the statically known kernel count and types;
    # a generator comprehension widens these to an unknown-length Vararg.
    hamiltonians=filter(kernel->kernel isa HamiltonianPIKernel,kernels)
    jumps=filter(kernel->kernel isa Union{DissipatorPIKernel,LocalJumpPIKernel},
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
    _trajectory_plan(model,LiouvillianPlan(model))
end
function TrajectoryPlan(compiled::CompiledPIModel;T=nothing)
    T===nothing&&return _trajectory_plan(compiled.model,compiled.plan)
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
    TrajectoryWorkspace(plan, rho)
    TrajectoryWorkspace(model, rho)
    TrajectoryWorkspace(compiled, rho)

Preallocated mutable stage vectors and Schur-block scratch for one PI quantum
trajectory at a time. Reuse it sequentially. Concurrent paths must have
distinct workspaces, which may all refer to the same immutable
[`TrajectoryPlan`](@ref).
"""
struct TrajectoryWorkspace{V,R,P,W}
    tmp::V;k1::V;k2::V;k3::V;k4::V;k5::V;k6::V;k7::V
    trial::V;embedded::V;start::V
    current::V
    channel_gain::V
    intensities::Vector{R}
    jump_scales::Vector{R}
    plan::P
    liouvillian_work::W
end

function TrajectoryWorkspace(plan::TrajectoryPlan,rho::PIState)
    rho.basis===plan.model.basis||throw(ArgumentError(
        "state and trajectory plan use incompatible PI bases"))
    _check_liouvillian_source_precision(plan.liouvillian,eltype(rho.data),
                                        "trajectory state")
    promote_type(eltype(rho.data),plan.liouvillian.Ttype)===eltype(rho.data)||
        throw(ArgumentError("trajectory state scalar type $(eltype(rho.data)) cannot represent plan scalar type $(plan.liouvillian.Ttype)"))
    v=similar(rho.data)
    R=_real_float_type(eltype(v));njumps=length(plan.jumps)
    TrajectoryWorkspace(similar(v),similar(v),similar(v),similar(v),similar(v),
                        similar(v),similar(v),similar(v),similar(v),similar(v),
                        similar(v),v,similar(v),zeros(R,njumps),
                        zeros(R,njumps),plan,
                        LiouvillianWorkspace(plan.liouvillian))
end
TrajectoryWorkspace(model::PIModel,rho::PIState)=
    TrajectoryWorkspace(_trajectory_plan_for_state(model,rho),rho)
TrajectoryWorkspace(compiled::CompiledPIModel,rho::PIState)=
    TrajectoryWorkspace(_trajectory_plan_for_state(compiled,rho),rho)

"""
    TrajectoryBatchWorkspace(plan, rho; workers=Threads.nthreads())
    TrajectoryBatchWorkspace(model, rho; workers=Threads.nthreads())
    TrajectoryBatchWorkspace(compiled, rho; workers=Threads.nthreads())

Reusable worker pool for [`quantum_trajectories`](@ref). The immutable
trajectory plan is stored once, while every worker owns independent integrator
scratch and a reusable random-number generator. A batch workspace may be
reused sequentially but must not be used by concurrent ensemble calls.
"""
struct TrajectoryBatchWorkspace{P,W,R,S}
    plan::P
    workers::W
    rngs::R
    seeds::S
end

function TrajectoryBatchWorkspace(plan::TrajectoryPlan,rho::PIState;
                                  workers::Integer=Threads.nthreads())
    workers>0||throw(ArgumentError("worker count must be positive"))
    workspaces=[TrajectoryWorkspace(plan,rho) for _ in 1:Int(workers)]
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


function _conditional_action_and_intensity!(y,x,w,b,t,p,tau)
    fill!(y,zero(eltype(y)))
    _apply_trajectory_hamiltonians!(y,x,w.plan.hamiltonians,b,t,p,
                                    w.liouvillian_work)
    lambda=_apply_conditional_jumps!(y,x,w,b,t,p,w.plan.jumps,1)
    isfinite(lambda)||throw(ArgumentError(
        "total jump intensity is nonfinite; use a wider scalar type or inspect the state and rates"))
    @. y=y+lambda*x
    lambda
end

function _conditional_action!(y,x,w,b,t,p,tau)
    _conditional_action_and_intensity!(y,x,w,b,t,p,tau)
    y
end

function _conditional_rk4!(x,w,b,t,h,p,tau)
    _conditional_action!(w.k1,x,w,b,t,p,tau)
    @. w.tmp=x+(h/2)*w.k1;_conditional_action!(w.k2,w.tmp,w,b,t+h/2,p,tau)
    @. w.tmp=x+(h/2)*w.k2;_conditional_action!(w.k3,w.tmp,w,b,t+h/2,p,tau)
    @. w.tmp=x+h*w.k3;_conditional_action!(w.k4,w.tmp,w,b,t+h,p,tau)
    @. x=x+(h/6)*(w.k1+2w.k2+2w.k3+w.k4)
    z=dot(tau,x);R=_real_float_type(eltype(x))
    abs(z)>eps(R)||throw(ArgumentError("conditional state acquired zero trace"));x./=z;x
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

function _event_driven_trajectory(plan,rho0,ts,w,rng,parameters,dt,
                                  abstol,reltol,dtmin,dtmax,event_time_tolerance)
    b=plan.model.basis;tau=plan.liouvillian.tracevec;x=w.current
    copyto!(x,rho0.data)
    states=Vector{typeof(rho0)}(undef,length(ts));states[1]=copy(rho0)
    jt=eltype(ts)[];jc=Int[];t=ts[1]
    R=typeof(dt)
    threshold=randexp(rng,R);hazard=zero(threshold)
    h=min(dt,dtmax)
    for output_index in 2:length(ts)
        target=ts[output_index]
        while t<target
            remaining_to_target=target-t
            h=min(h,remaining_to_target,dtmax)
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
                t+=h;hazard+=increment
                h=min(dtmax,max(dtmin,h*_adaptive_factor(error)))
                continue
            end

            # A continuous event occurred inside the accepted step. Locate the
            # hazard root from the unchanged step-start state, then apply the
            # selected channel at that physical event time.
            remaining=threshold-hazard;lo=zero(h);hi=h
            time_tol=max(event_time_tolerance,R(8)*eps(t)*max(abs(t),one(t)))
            for _ in 1:60
                hi-lo<=time_tol&&break
                mid=(lo+hi)/2
                mid_increment,_=_conditional_dopri_trial!(w,w.start,b,t,mid,parameters,tau,abstol,reltol)
                if mid_increment>=remaining;hi=mid;else;lo=mid;end
            end
            event_step=hi
            _conditional_dopri_trial!(w,w.start,b,t,event_step,parameters,tau,abstol,reltol)
            copyto!(x,w.trial);z=dot(tau,x)
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
            copyto!(x,w.channel_gain);x./=z;push!(jt,t);push!(jc,channel)
            threshold=randexp(rng,typeof(threshold));hazard=zero(threshold)
            h=min(dtmax,max(dtmin,h-event_step))
        end
        states[output_index]=PIState(b,x)
    end
    QuantumTrajectory(ts,states,jt,jc)
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

function _quantum_trajectory_prepared(plan,rho0,ts,w,rng,options)
    b=plan.model.basis;tau=plan.liouvillian.tracevec
    R=eltype(w.intensities)
    options.algorithm!==:fixed&&return _event_driven_trajectory(
        plan,rho0,ts,w,rng,options.parameters,options.dt,options.abstol,
        options.reltol,options.dtmin,options.dtmax,
        options.event_time_tolerance)
    x=w.current;copyto!(x,rho0.data)
    states=Vector{typeof(rho0)}(undef,length(ts));states[1]=copy(rho0)
    jt=eltype(ts)[];jc=Int[];t=ts[1]
    for output_index in 2:length(ts)
        target=ts[output_index]
        while t<target
            h=min(options.dt,target-t)
            rates=_channel_intensities!(w,x,b,t,options.parameters)
            lambda=_total_intensity(rates)
            lambda*h>options.max_jump_probability&&
                (h=options.max_jump_probability/lambda)
            _conditional_rk4!(x,w,b,t,h,options.parameters,tau);t+=h
            rates=_channel_intensities!(w,x,b,t,options.parameters)
            lambda=_total_intensity(rates)
            jump_probability=-expm1(-lambda*h)
            if lambda>0&&rand(rng,typeof(jump_probability))<jump_probability
                u=rand(rng,typeof(lambda))*lambda
                channel=_select_jump_channel(rates,u)
                _apply_gain!(w.channel_gain,x,plan.jumps[channel],b,
                             w.jump_scales[channel],w.liouvillian_work)
                z=dot(tau,w.channel_gain)
                abs(z)>eps(R)||throw(ArgumentError(
                    "selected jump has zero probability"))
                copyto!(x,w.channel_gain);x./=z
                push!(jt,t);push!(jc,channel)
            end
        end
        states[output_index]=PIState(b,x)
    end
    QuantumTrajectory(ts,states,jt,jc)
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

Set `algorithm=:event` (aliases `:adaptive` and `:event_driven`) to integrate
the normalized no-jump equation and its accumulated hazard with an embedded
Dormand--Prince 5(4) method. Jump times are then continuous hazard roots,
rather than endpoints of Bernoulli time steps. `dt` is the initial adaptive
step and `dtmax` its upper bound; `abstol`, `reltol`, `dtmin`, and
`event_time_tolerance` control the adaptive solve. A final step may be shorter
than `dtmin` solely to land on a requested output time. Here `R` is the real
floating precision of the prepared trajectory. Time grids and explicitly
supplied controls must be representable in `R` without narrowing; default
controls are constructed directly in `R`.
"""
function quantum_trajectory(source::Union{PIModel,CompiledPIModel,TrajectoryPlan},
                            rho0::PIState{R},times;
                            dt::Real, rng::AbstractRNG=Random.default_rng(),parameters=nothing,
                            max_jump_probability=nothing,workspace=nothing,
                            algorithm::Symbol=:fixed,abstol=nothing,
                            reltol=nothing,dtmin=nothing,
                            dtmax=nothing,event_time_tolerance=nothing) where {R<:AbstractFloat}
    if workspace===nothing
        plan=_plan_for_source(source,rho0);w=TrajectoryWorkspace(plan,rho0)
    else
        workspace isa TrajectoryWorkspace||throw(ArgumentError(
            "workspace must be a TrajectoryWorkspace"))
        w=_check_trajectory_workspace(workspace,source,rho0);plan=w.plan
    end
    _validate_trajectory_initial_state(plan,rho0)
    ts,options=_prepare_trajectory_arguments(times,R;dt,parameters,
        max_jump_probability,algorithm,abstol,reltol,dtmin,dtmax,
        event_time_tolerance)
    _quantum_trajectory_prepared(plan,rho0,ts,w,rng,options)
end

"""
    quantum_trajectories(source, rho0, times, n; seed=0, threaded=false,
                         workspace=nothing, trajectory_keywords...)

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
"""
function quantum_trajectories(source::Union{PIModel,CompiledPIModel,TrajectoryPlan},
        rho0::PIState{R},times,n::Integer;seed::Integer=0,
        threaded::Bool=false,workspace=nothing,dt::Real,
        parameters=nothing,max_jump_probability=nothing,
        algorithm::Symbol=:fixed,abstol=nothing,reltol=nothing,
        dtmin=nothing,dtmax=nothing,event_time_tolerance=nothing) where {R<:AbstractFloat}
    n>0||throw(ArgumentError("trajectory count must be positive"))
    if workspace===nothing
        plan=_plan_for_source(source,rho0)
        worker_count=threaded ? min(Int(n),Threads.nthreads()) : 1
        batch=TrajectoryBatchWorkspace(plan,rho0;workers=worker_count)
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
    ts,options=_prepare_trajectory_arguments(times,R;dt,parameters,
        max_jump_probability,algorithm,abstol,reltol,dtmin,dtmax,
        event_time_tolerance)
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

function _named_observables(observables)
    observables isa NamedTuple&&return collect(pairs(observables))
    observables isa AbstractDict&&return collect(pairs(observables))
    observables isa Pair&&return [observables]
    observables isa AbstractVector{<:Pair}&&return collect(observables)
    [(Symbol("observable"),observables)]
end

"""
    trajectory_observable_statistics(trajectories, observables; confidence=0.95)

Compute time-resolved Monte Carlo means, unbiased variances, standard errors,
and normal confidence intervals. `observables` may be a named tuple, dictionary,
pair collection, or a single Hermitian local matrix/`PIOperator`.
"""
function trajectory_observable_statistics(trajs::AbstractVector{<:QuantumTrajectory},observables;confidence::Real=0.95)
    times,b=_check_trajectory_ensemble(trajs);0<confidence<1||throw(ArgumentError("confidence must lie in (0,1)"));n=length(trajs);nt=length(times)
    named=_named_observables(observables);ops=Pair[]
    for (name,A) in named
        op=A isa AbstractMatrix ? collective_operator(b,A) : A
        op isa PIOperator&&op.basis===b||throw(ArgumentError("observables must be local matrices or compatible PIOperators"))
        ishermitian(op)||throw(ArgumentError("trajectory statistics require Hermitian observables"));push!(ops,name=>op)
    end
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
