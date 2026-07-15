"""One PI quantum-jump realization and its recorded jump channel indices."""
struct QuantumTrajectory{T,S<:PIState}
    times::Vector{T}
    states::Vector{S}
    jump_times::Vector{T}
    jump_channels::Vector{Int}
end

"""Preallocated vectors and channel kernels for PI quantum trajectories."""
struct TrajectoryWorkspace{V,K,L,W}
    tmp::V;k1::V;k2::V;k3::V;k4::V;k5::V;k6::V;k7::V
    trial::V;embedded::V;start::V
    gain::V;channel_gain::V
    intensities::Vector{Float64}
    kernels::K
    liouvillian::L
    liouvillian_work::W
end

function TrajectoryWorkspace(model::PIModel,rho::PIState)
    rho.basis===model.basis||throw(ArgumentError("state and model use incompatible PI bases"))
    all(t->!(t.operator isa Function),model.terms)||throw(ArgumentError("trajectory kernels require fixed operators; scalar rates may depend on time"))
    L=liouvillian(model;representation=:matrixfree)
    kernels=L.plan.kernels;jumps=Tuple(k for k in kernels if k isa Union{DissipatorPIKernel,LocalJumpPIKernel})
    v=similar(rho.data)
    TrajectoryWorkspace(similar(v),similar(v),similar(v),similar(v),similar(v),
                        similar(v),similar(v),similar(v),similar(v),similar(v),similar(v),
                        similar(v),similar(v),zeros(length(jumps)),
                        (all=kernels,jumps=jumps),L,LiouvillianWorkspace(L))
end

function _apply_gain!(y,x,k::DissipatorPIKernel,b,t,p,work)
    raw=value_at(k.scale,t,p);iszero(imag(complex(raw)))||throw(ArgumentError(
        "quantum trajectories require real jump rates"))
    scale=Float64(real(raw));scale>=0||throw(ArgumentError("quantum trajectories require nonnegative jump rates"))
    fill!(y,zero(eltype(y)))
    for (s,part) in pairs(b.sectors)
        n=length(b.patterns[s]);r=b.offsets[s]:b.offsets[s+1]-1
        X=reshape(view(x,r),n,n);Y=reshape(view(y,r),n,n);A=work.blocks[s][1];K=k.blocks[s]
        mul!(A,K,X);mul!(Y,A,adjoint(K));Y .*= scale
    end
    y
end
function _apply_gain!(y,x,k::LocalJumpPIKernel,b,t,p,work)
    raw=value_at(k.scale,t,p);iszero(imag(complex(raw)))||throw(ArgumentError(
        "quantum trajectories require real jump rates"))
    scale=Float64(real(raw));scale>=0||throw(ArgumentError("quantum trajectories require nonnegative jump rates"))
    fill!(y,zero(eltype(y)))
    @inbounds for q in eachindex(k.gain.V);y[k.gain.I[q]]+=scale*k.gain.V[q]*x[k.gain.J[q]];end
    y
end

function _channel_intensities!(w::TrajectoryWorkspace,x,b,t,p,tau)
    for (i,k) in pairs(w.kernels.jumps)
        _apply_gain!(w.channel_gain,x,k,b,t,p,w.liouvillian_work)
        z=real(dot(tau,w.channel_gain));z>=-1e-11||throw(ArgumentError("jump gain has negative trace $z"))
        w.intensities[i]=max(0.0,z)
    end
    w.intensities
end

function _conditional_action_and_intensity!(y,x,w,b,t,p,tau)
    apply!(y,w.liouvillian,x,t,p,w.liouvillian_work);fill!(w.gain,zero(eltype(w.gain)));lambda=0.0
    for k in w.kernels.jumps
        _apply_gain!(w.channel_gain,x,k,b,t,p,w.liouvillian_work);axpy!(1,w.channel_gain,w.gain)
        lambda+=max(0.0,real(dot(tau,w.channel_gain)))
    end
    @. y=y-w.gain+lambda*x
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
    z=dot(tau,x);abs(z)>eps()||throw(ArgumentError("conditional state acquired zero trace"));x./=z;x
end

# One Dormand--Prince 5(4) trial for the normalized conditional state together
# with the accumulated jump hazard.  State and hazard use the same stages, so
# an accepted step controls both errors and no time-grid Bernoulli
# approximation enters the event time.
function _conditional_dopri_trial!(w,x,b,t,h,p,tau,abstol,reltol)
    l1=_conditional_action_and_intensity!(w.k1,x,w,b,t,p,tau)
    @. w.tmp=x+h*(1/5)*w.k1
    l2=_conditional_action_and_intensity!(w.k2,w.tmp,w,b,t+h*(1/5),p,tau)
    @. w.tmp=x+h*((3/40)*w.k1+(9/40)*w.k2)
    l3=_conditional_action_and_intensity!(w.k3,w.tmp,w,b,t+h*(3/10),p,tau)
    @. w.tmp=x+h*((44/45)*w.k1-(56/15)*w.k2+(32/9)*w.k3)
    l4=_conditional_action_and_intensity!(w.k4,w.tmp,w,b,t+h*(4/5),p,tau)
    @. w.tmp=x+h*((19372/6561)*w.k1-(25360/2187)*w.k2+
                  (64448/6561)*w.k3-(212/729)*w.k4)
    l5=_conditional_action_and_intensity!(w.k5,w.tmp,w,b,t+h*(8/9),p,tau)
    @. w.tmp=x+h*((9017/3168)*w.k1-(355/33)*w.k2+
                  (46732/5247)*w.k3+(49/176)*w.k4-(5103/18656)*w.k5)
    l6=_conditional_action_and_intensity!(w.k6,w.tmp,w,b,t+h,p,tau)
    @. w.trial=x+h*((35/384)*w.k1+(500/1113)*w.k3+
                    (125/192)*w.k4-(2187/6784)*w.k5+(11/84)*w.k6)
    l7=_conditional_action_and_intensity!(w.k7,w.trial,w,b,t+h,p,tau)
    @. w.embedded=x+h*((5179/57600)*w.k1+(7571/16695)*w.k3+
                       (393/640)*w.k4-(92097/339200)*w.k5+
                       (187/2100)*w.k6+(1/40)*w.k7)

    hazard5=h*((35/384)*l1+(500/1113)*l3+(125/192)*l4-
               (2187/6784)*l5+(11/84)*l6)
    hazard4=h*((5179/57600)*l1+(7571/16695)*l3+(393/640)*l4-
               (92097/339200)*l5+(187/2100)*l6+(1/40)*l7)
    state_scale=abstol+reltol*max(norm(x),norm(w.trial),1)
    @. w.tmp=w.trial-w.embedded
    state_error=norm(w.tmp)/(sqrt(length(x))*state_scale)
    hazard_scale=abstol+reltol*max(abs(hazard5),one(hazard5))
    error=max(state_error,abs(hazard5-hazard4)/hazard_scale)
    hazard5,error
end

_adaptive_factor(error)=error==0 ? 5.0 : clamp(0.9*error^(-1/5),0.2,5.0)

function _event_driven_trajectory(model,rho0,ts,w,rng,parameters,dt,
                                  abstol,reltol,dtmin,dtmax,event_time_tolerance)
    b=model.basis;tau=w.liouvillian.tracevec;x=copy(rho0)
    states=typeof(x)[copy(x)];jt=eltype(ts)[];jc=Int[];t=ts[1]
    threshold=-log(rand(rng));hazard=0.0;h=min(float(dt),float(dtmax))
    for target in ts[2:end]
        while t<target
            remaining_to_target=target-t
            h=min(h,remaining_to_target,float(dtmax))
            # As in standard adaptive integrators, landing exactly on a saved
            # output time may require one final step shorter than dtmin.
            minimum_step=min(float(dtmin),remaining_to_target)
            h>=minimum_step||throw(ErrorException(
                "adaptive trajectory step fell below dtmin=$dtmin at t=$t"))
            copyto!(w.start,x.data)
            increment,error=_conditional_dopri_trial!(w,x.data,b,t,h,parameters,tau,abstol,reltol)
            if !(isfinite(error)&&isfinite(increment))
                throw(ErrorException("non-finite adaptive trajectory trial at t=$t"))
            end
            if error>1
                h>minimum_step||throw(ErrorException(
                    "adaptive trajectory cannot satisfy its error tolerance above dtmin=$dtmin at t=$t"))
                h=max(minimum_step,h*_adaptive_factor(error));continue
            end
            increment>=-10abstol||throw(ErrorException("jump hazard decreased by $increment"))
            increment=max(0.0,increment)
            if hazard+increment < threshold
                copyto!(x.data,w.trial);z=dot(tau,x.data)
                abs(z)>eps()||throw(ArgumentError("conditional state acquired zero trace"));x.data./=z
                t+=h;hazard+=increment
                h=min(float(dtmax),max(float(dtmin),h*_adaptive_factor(error)))
                continue
            end

            # A continuous event occurred inside the accepted step. Locate the
            # hazard root from the unchanged step-start state, then apply the
            # selected channel at that physical event time.
            remaining=threshold-hazard;lo=0.0;hi=h
            time_tol=max(float(event_time_tolerance),8eps(float(t))*max(abs(float(t)),1.0))
            for _ in 1:60
                hi-lo<=time_tol&&break
                mid=(lo+hi)/2
                mid_increment,_=_conditional_dopri_trial!(w,w.start,b,t,mid,parameters,tau,abstol,reltol)
                if mid_increment>=remaining;hi=mid;else;lo=mid;end
            end
            event_step=hi
            _conditional_dopri_trial!(w,w.start,b,t,event_step,parameters,tau,abstol,reltol)
            copyto!(x.data,w.trial);z=dot(tau,x.data)
            abs(z)>eps()||throw(ArgumentError("conditional state acquired zero trace"));x.data./=z
            t+=event_step
            rates=_channel_intensities!(w,x.data,b,t,parameters,tau);lambda=sum(rates)
            lambda>0||throw(ErrorException("hazard root has zero channel intensity at t=$t"))
            u=rand(rng)*lambda;s=0.0;channel=lastindex(rates)
            for i in eachindex(rates);s+=rates[i];if u<=s;channel=i;break;end;end
            _apply_gain!(w.channel_gain,x.data,w.kernels.jumps[channel],b,t,parameters,w.liouvillian_work)
            z=dot(tau,w.channel_gain);abs(z)>eps()||throw(ArgumentError("selected jump has zero probability"))
            copyto!(x.data,w.channel_gain);x.data./=z;push!(jt,t);push!(jc,channel)
            threshold=-log(rand(rng));hazard=0.0
            h=min(float(dtmax),max(float(dtmin),h-event_step))
        end
        push!(states,copy(x))
    end
    QuantumTrajectory(ts,states,jt,jc)
end

"""
    quantum_trajectory(model, rho0, times; dt, algorithm=:fixed, rng,
                       parameters=nothing, max_jump_probability=0.05,
                       abstol=1e-9, reltol=1e-7,
                       dtmin=eps(Float64), dtmax=dt,
                       event_time_tolerance=1e-10, workspace=nothing)

Simulate one PI quantum-jump trajectory. Local jump channels are unresolved
over particle labels and therefore generally produce mixed conditional PI
states. The fixed step is automatically shortened so the total jump
probability remains below `max_jump_probability`.

Set `algorithm=:event` (aliases `:adaptive` and `:event_driven`) to integrate
the normalized no-jump equation and its accumulated hazard with an embedded
Dormand--Prince 5(4) method. Jump times are then continuous hazard roots,
rather than endpoints of Bernoulli time steps. `dt` is the initial adaptive
step and `dtmax` its upper bound; `abstol`, `reltol`, `dtmin`, and
`event_time_tolerance` control the adaptive solve. A final step may be shorter
than `dtmin` solely to land on a requested output time.
"""
function quantum_trajectory(model::PIModel,rho0::PIState,times;
                            dt::Real, rng::AbstractRNG=Random.default_rng(),parameters=nothing,
                            max_jump_probability::Real=0.05,workspace=nothing,
                            algorithm::Symbol=:fixed,abstol::Real=1e-9,
                            reltol::Real=1e-7,dtmin::Real=eps(Float64),
                            dtmax::Real=dt,event_time_tolerance::Real=1e-10)
    dt>0||throw(ArgumentError("dt must be positive"));0<max_jump_probability<1||throw(ArgumentError("max_jump_probability must lie in (0,1)"))
    algorithm in (:fixed,:event,:adaptive,:event_driven)||throw(ArgumentError("algorithm must be :fixed or :event"))
    abstol>0||throw(ArgumentError("abstol must be positive"));reltol>0||throw(ArgumentError("reltol must be positive"))
    dtmin>0||throw(ArgumentError("dtmin must be positive"));dtmax>=dtmin||throw(ArgumentError("dtmax must be at least dtmin"))
    event_time_tolerance>0||throw(ArgumentError("event_time_tolerance must be positive"))
    ts=float.(collect(times));isempty(ts)&&throw(ArgumentError("at least one output time is required"));all(diff(ts).>=0)||throw(ArgumentError("times must be nondecreasing"))
    w=workspace===nothing ? TrajectoryWorkspace(model,rho0) : workspace;b=model.basis;tau=w.liouvillian.tracevec
    x=copy(rho0);abs(trace(x)-1)<=1e-10||throw(ArgumentError("initial state must have unit trace"))
    algorithm!==:fixed&&return _event_driven_trajectory(model,rho0,ts,w,rng,parameters,
        dt,abstol,reltol,dtmin,dtmax,event_time_tolerance)
    states=typeof(x)[copy(x)];jt=eltype(ts)[];jc=Int[];t=ts[1]
    for target in ts[2:end]
        while t<target
            h=min(float(dt),target-t);rates=_channel_intensities!(w,x.data,b,t,parameters,tau);lambda=sum(rates)
            lambda*h>max_jump_probability&&(h=max_jump_probability/lambda)
            _conditional_rk4!(x.data,w,b,t,h,parameters,tau);t+=h
            rates=_channel_intensities!(w,x.data,b,t,parameters,tau);lambda=sum(rates)
            if lambda>0 && rand(rng)<1-exp(-lambda*h)
                u=rand(rng)*lambda;s=0.0;channel=lastindex(rates)
                for i in eachindex(rates);s+=rates[i];if u<=s;channel=i;break;end;end
                _apply_gain!(w.channel_gain,x.data,w.kernels.jumps[channel],b,t,parameters,w.liouvillian_work)
                z=dot(tau,w.channel_gain);abs(z)>eps()||throw(ArgumentError("selected jump has zero probability"))
                copyto!(x.data,w.channel_gain);x.data./=z;push!(jt,t);push!(jc,channel)
            end
        end
        push!(states,copy(x))
    end
    QuantumTrajectory(ts,states,jt,jc)
end

"""Generate independent PI trajectories with reusable sequential or thread-local workspaces."""
function quantum_trajectories(model::PIModel,rho0::PIState,times,n::Integer;
                              seed::Integer=0,threaded::Bool=false,kwargs...)
    n>0||throw(ArgumentError("trajectory count must be positive"));master=MersenneTwister(seed)
    seeds=rand(master,UInt64,n);TT=eltype(float.(collect(times)))
    out=Vector{QuantumTrajectory{TT,typeof(rho0)}}(undef,n)
    if threaded && Threads.nthreads()>1
        workspaces=[TrajectoryWorkspace(model,rho0) for _ in 1:Threads.nthreads()]
        Threads.@threads for i in 1:n
            out[i]=quantum_trajectory(model,rho0,times;rng=MersenneTwister(seeds[i]),workspace=workspaces[Threads.threadid()],kwargs...)
        end
    else
        workspace=TrajectoryWorkspace(model,rho0)
        for i in 1:n
            out[i]=quantum_trajectory(model,rho0,times;rng=MersenneTwister(seeds[i]),workspace=workspace,kwargs...)
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
