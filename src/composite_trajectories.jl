"""
    CompositeJumpChannel(basis, factor=>operator...; rate=1, label=:jump)

A monitored tensor-product jump channel for a [`CompositePIBasis`](@ref).
Each named PI factor requires a compatible fixed [`PIOperator`](@ref), and
each finite factor requires a fixed square matrix. Unnamed factors carry the
identity. The channel owns copied, factorized gain and anticommutator maps;
it never constructs the global Kronecker matrix.

`rate` may be a fixed real number or a scalar `value_at`-compatible schedule.
Quantum trajectories require every evaluated rate to be finite and
nonnegative. The channel order and `label` identify the chosen unraveling.
"""
struct CompositeJumpChannel{B,G,L,R,C}
    basis::B
    gain::G
    loss_left::L
    loss_right::R
    rate::C
    label::Symbol
end

function _validated_composite_stochastic_rate(rate)
    resolved=_validated_composite_real_rate(rate,"stochastic jump")
    if resolved isa Number
        isfinite(resolved)||throw(ArgumentError(
            "a stochastic jump rate must be finite"))
        resolved>=zero(resolved)||throw(ArgumentError(
            "a stochastic jump rate must be nonnegative"))
        return resolved
    end
    (t,p)->begin
        value=resolved(t,p)
        isfinite(value)||throw(ArgumentError(
            "a stochastic jump rate must evaluate to a finite number"))
        value>=zero(value)||throw(ArgumentError(
            "a stochastic jump rate must evaluate to a nonnegative number"))
        value
    end
end

function CompositeJumpChannel(b::CompositePIBasis,pairs::Pair...;
                              rate=1,label::Symbol=:jump)
    operators=_factor_operator_pairs(b,pairs)
    isempty(operators)&&throw(ArgumentError(
        "a composite jump channel needs at least one nonidentity factor"))
    resolved_rate=_validated_composite_stochastic_rate(rate)
    gain_pairs=map(pair->first(pair)=>factor_sandwich_superoperator(
        b.factors[first(pair)],last(pair)),operators)
    products=map(pair->first(pair)=>_factor_product(last(pair)),operators)
    left_pairs=map(pair->first(pair)=>factor_left_superoperator(
        b.factors[first(pair)],last(pair)),products)
    right_pairs=map(pair->first(pair)=>factor_right_superoperator(
        b.factors[first(pair)],last(pair)),products)
    gain=factorized_superoperator_term(b,gain_pairs...)
    loss_left=factorized_superoperator_term(b,left_pairs...)
    loss_right=factorized_superoperator_term(b,right_pairs...)
    CompositeJumpChannel{typeof(b),typeof(gain),typeof(loss_left),
        typeof(loss_right),typeof(resolved_rate)}(
        b,gain,loss_left,loss_right,resolved_rate,label)
end

isautonomous(channel::CompositeJumpChannel)=channel.rate isa Number

struct _PreparedCompositeJumpChannel{G,L,R,C}
    gain::G
    loss_left::L
    loss_right::R
    rate::C
    label::Symbol
end

isautonomous(channel::_PreparedCompositeJumpChannel)=channel.rate isa Number

function _composite_trajectory_action(action::SparseMatrixCSC,
                                      ::Type{T}) where T
    promote_type(T,eltype(action))===T||throw(ArgumentError(
        "composite jump maps of type $(eltype(action)) cannot be represented in trajectory scalar type $T"))
    I,J,V=findnz(action)
    sparse(I,J,T.(V),size(action,1),size(action,2))
end

function _composite_trajectory_action(action::AbstractMatrix,
                                      ::Type{T}) where T
    promote_type(T,eltype(action))===T||throw(ArgumentError(
        "composite jump maps of type $(eltype(action)) cannot be represented in trajectory scalar type $T"))
    Matrix{T}(action)
end

_composite_trajectory_action(::Nothing,::Type)=nothing

function _prepare_composite_jump_term(term,::Type{T}) where T
    actions=map(action->_composite_trajectory_action(action,T),term.actions)
    CompositeSuperoperatorTerm(term.basis,actions)
end

function _prepare_composite_jump_channel(channel,::Type{T}) where T
    rate=if channel.rate isa Number
        R=_real_float_type(T)
        converted=_checked_composite_coefficient(R,channel.rate)
        converted>=zero(R)||throw(ArgumentError(
            "composite quantum trajectories require nonnegative jump rates"))
        converted
    else
        channel.rate
    end
    _PreparedCompositeJumpChannel(
        _prepare_composite_jump_term(channel.gain,T),
        _prepare_composite_jump_term(channel.loss_left,T),
        _prepare_composite_jump_term(channel.loss_right,T),
        rate,channel.label)
end

function _composite_jump_required_type(channel::CompositeJumpChannel)
    types=Type[]
    channel.rate isa Number&&push!(types,typeof(channel.rate))
    for term in (channel.gain,channel.loss_left,channel.loss_right),
        action in term.actions
        action===nothing||push!(types,eltype(action))
    end
    isempty(types) ? nothing : foldl(promote_type,types)
end

struct _CompositeTracePlan{I,O,S}
    dimension::Int
    indices::I
    offsets::O
    scales::S
end

function _append_composite_diagonal_indices!(indices,combo,strides,
                                             factor::Int,index::Int)
    if factor>length(combo)
        push!(indices,index)
        return indices
    end
    @inbounds for coordinate in combo[factor].diagonal
        _append_composite_diagonal_indices!(indices,combo,strides,factor+1,
            index+(coordinate-1)*strides[factor])
    end
    indices
end

function _prepare_composite_trace(b::CompositePIBasis,
                                  ::Type{R}) where R<:AbstractFloat
    groups=map(_composite_trace_groups,b.factors)
    strides=ntuple(i->i==1 ? 1 : prod(b.dimensions[1:i-1]),
                   length(b.factors))
    indices=Int[]
    offsets=Int[1]
    scales=_PreparedExactScale{R,true}[]
    for combo in Iterators.product(groups...)
        _append_composite_diagonal_indices!(indices,combo,strides,1,1)
        push!(offsets,length(indices)+1)
        multiplicity=prod(group->group.multiplicity,combo;init=big(1))
        push!(scales,_prepare_exact_scale(R,multiplicity,one(BigInt),Val(true);
            context="composite trajectory trace weight"))
    end
    _CompositeTracePlan(length(b),indices,offsets,scales)
end

@inline function _prepared_composite_trace(plan::_CompositeTracePlan,x)
    length(x)==plan.dimension||throw(DimensionMismatch(
        "composite trace source has the wrong length"))
    total=zero(eltype(x))
    correction=zero(eltype(x))
    @inbounds for block in eachindex(plan.scales)
        block_trace=zero(eltype(x))
        for position in plan.offsets[block]:plan.offsets[block+1]-1
            block_trace+=x[plan.indices[position]]
        end
        contribution=_apply_prepared_exact_scale(
            block_trace,plan.scales[block];
            context="composite trajectory trace contribution")
        updated=total+contribution
        correction+=abs(total)>=abs(contribution) ?
            (total-updated)+contribution : (contribution-updated)+total
        total=updated
    end
    total+correction
end

function _checked_composite_trace_value(value,::Type{R},context;
                                        real_required::Bool=false) where
        R<:AbstractFloat
    isfinite(real(value))&&isfinite(imag(value))||throw(ArgumentError(
        "$context is nonfinite; use a wider scalar type or inspect the state"))
    if real_required
        tolerance=_intensity_tolerance(R)*max(one(R),abs(real(value)))
        abs(imag(value))<=tolerance||throw(ArgumentError(
            "$context has a non-negligible imaginary part $(imag(value))"))
        return real(value)
    end
    value
end

function _composite_trace_zero_check(value,scale,::Type{R},context) where
        R<:AbstractFloat
    tolerance=_intensity_tolerance(R)*max(one(R),R(scale))
    abs(value)<=tolerance||throw(ArgumentError(
        "$context is not trace preserving: trace derivative $value exceeds tolerance $tolerance"))
    nothing
end

"""
    CompositeTrajectoryPlan(basis, channels...; background=nothing, T=nothing)
    CompositeTrajectoryPlan(background, channels...; T=nothing)

Immutable prepared model for density-valued composite quantum-jump
trajectories. `background` is a trace-preserving
[`CompositeSuperoperator`](@ref) containing coherent and unmonitored physics;
it must *exclude* the monitored `channels`. The plan assembles the complete
unconditional generator as the background plus one Lindblad dissipator per
channel. Use [`composite_master_superoperator`](@ref) to access it.

The plan retains factorized gain/loss maps and multiplicity-aware trace
metadata. It may be shared by tasks. Each task needs its own
[`CompositeTrajectoryWorkspace`](@ref). Fixed jump operators and scalar
time-dependent rates are supported; arbitrary gain superoperators are not
silently interpreted as an unraveling.
"""
struct CompositeTrajectoryPlan{B,S,C,G,TP}
    basis::B
    background::S
    channels::C
    generator::G
    trace_plan::TP
end

function _composite_trajectory_scalar_type(b,background,channels,T)
    required=Type[]
    background===nothing||push!(required,eltype(background))
    for channel in channels
        channel_type=_composite_jump_required_type(channel)
        channel_type===nothing||push!(required,channel_type)
    end
    if T===nothing
        isempty(required)&&throw(ArgumentError(
            "cannot infer an empty composite trajectory precision; pass T"))
        raw=foldl(promote_type,required)
        return _composite_coordinate_type(_real_float_type(raw))
    end
    requested=_composite_coordinate_type(T)
    if !isempty(required)
        needed=_composite_coordinate_type(_real_float_type(
            foldl(promote_type,required)))
        promote_type(requested,needed)===requested||throw(ArgumentError(
            "explicit composite trajectory type $requested would narrow fixed data of type $needed"))
    end
    requested
end

function _composite_channel_generator_terms(b,channel)
    gain=CompositeSuperoperatorTerm(b,channel.gain.actions;
        coefficient=channel.rate)
    left=CompositeSuperoperatorTerm(b,channel.loss_left.actions;
        coefficient=_scaled_composite_coefficient(channel.rate,-1//2))
    right=CompositeSuperoperatorTerm(b,channel.loss_right.actions;
        coefficient=_scaled_composite_coefficient(channel.rate,-1//2))
    (gain,left,right)
end

function CompositeTrajectoryPlan(b::CompositePIBasis,
        channels::CompositeJumpChannel...;background=nothing,T=nothing)
    all(channel->channel.basis===b,channels)||throw(ArgumentError(
        "every composite jump channel must use the exact CompositePIBasis object"))
    background===nothing||background isa CompositeSuperoperator||throw(
        ArgumentError("background must be a CompositeSuperoperator or nothing"))
    background===nothing||background.basis===b||throw(ArgumentError(
        "background and trajectory channels use incompatible composite bases"))
    scalar=_composite_trajectory_scalar_type(b,background,channels,T)
    if background!==nothing&&
       _real_float_type(eltype(background))!==_real_float_type(scalar)
        throw(ArgumentError(
            "background precision $(eltype(background)) differs from prepared trajectory precision $scalar; rebuild the background at the common precision"))
    end
    base=background===nothing ? CompositeSuperoperator(b;T=scalar) : background
    prepared=map(channel->_prepare_composite_jump_channel(channel,scalar),
                 channels)
    terms=base.terms
    for channel in prepared
        terms=(terms...,_composite_channel_generator_terms(b,channel)...)
    end
    generator=CompositeSuperoperator(b,terms...;T=scalar)
    trace_plan=_prepare_composite_trace(b,_real_float_type(scalar))
    CompositeTrajectoryPlan(b,base,prepared,generator,trace_plan)
end

CompositeTrajectoryPlan(background::CompositeSuperoperator,
        channels::CompositeJumpChannel...;T=nothing)=
    CompositeTrajectoryPlan(background.basis,channels...;background,T)

eltype(plan::CompositeTrajectoryPlan)=eltype(plan.generator)
size(plan::CompositeTrajectoryPlan)=size(plan.generator)
size(plan::CompositeTrajectoryPlan,index::Integer)=size(plan.generator,index)
isautonomous(plan::CompositeTrajectoryPlan)=
    isautonomous(plan.background)&&all(isautonomous,plan.channels)

function show(io::IO,plan::CompositeTrajectoryPlan)
    print(io,"CompositeTrajectoryPlan($(length(plan.channels)) monitored channels, dimension=$(length(plan.basis)))")
end

"""
    composite_master_superoperator(plan)

Return the complete unconditional matrix-free generator prepared by a
[`CompositeTrajectoryPlan`](@ref). It is the supplied background plus the
Lindblad dissipators of every monitored channel and is useful for direct
ensemble comparisons.
"""
composite_master_superoperator(plan::CompositeTrajectoryPlan)=plan.generator

struct _CompositeJumpTermWorkspace{G,L,R}
    gain::G
    loss_left::L
    loss_right::R
end

function _composite_trajectory_term_workspace(term,::Type{T}) where T
    factors=map((action,n)->_factor_workspace(action,n,T),
                term.actions,term.basis.dimensions)
    _CompositeTermWorkspace(factors)
end

function _composite_jump_workspace(channel,::Type{T}) where T
    _CompositeJumpTermWorkspace(
        _composite_trajectory_term_workspace(channel.gain,T),
        _composite_trajectory_term_workspace(channel.loss_left,T),
        _composite_trajectory_term_workspace(channel.loss_right,T))
end

"""
    CompositeTrajectoryWorkspace(plan, rho)

Preallocated stages, shared tensor-mode buffers, exact-trace scratch, and
factor-fibre workspaces for one density-valued composite trajectory. Full
composite buffers are shared sequentially across all jump channels, so their
count does not grow with the number of channels. Reuse one workspace
sequentially; concurrent paths require distinct workspaces.
"""
struct CompositeTrajectoryWorkspace{V,R,P,BW,CW}
    tmp::V
    k1::V
    k2::V
    k3::V
    k4::V
    current::V
    channel_gain::V
    tensor_buffer1::V
    tensor_buffer2::V
    intensities::Vector{R}
    integrated_intensities::Vector{R}
    jump_scales::Vector{R}
    gain_traces::Vector{R}
    plan::P
    background_work::BW
    channel_work::CW
end

function CompositeTrajectoryWorkspace(plan::CompositeTrajectoryPlan,
                                      rho::CompositePIState)
    rho.basis===plan.basis||throw(ArgumentError(
        "state and composite trajectory plan use incompatible bases"))
    eltype(rho.data)===eltype(plan)||throw(ArgumentError(
        "composite trajectory state type $(eltype(rho.data)) must exactly match prepared type $(eltype(plan))"))
    vector=similar(rho.data)
    R=_real_float_type(eltype(vector))
    channel_work=map(channel->_composite_jump_workspace(
        channel,eltype(vector)),plan.channels)
    CompositeTrajectoryWorkspace(
        similar(vector),similar(vector),similar(vector),similar(vector),
        similar(vector),vector,similar(vector),similar(vector),similar(vector),
        zeros(R,length(plan.channels)),zeros(R,length(plan.channels)),
        zeros(R,length(plan.channels)),zeros(R,length(plan.channels)),plan,
        CompositeSuperoperatorWorkspace(plan.background;T=eltype(vector)),
        channel_work)
end

"""
    CompositeTrajectoryBatchWorkspace(plan, rho; workers=Threads.nthreads())

Reusable task-local workspace and RNG pool for batched composite quantum
trajectories. The immutable plan is shared, while every active worker owns
all mutable numerical scratch. Reuse a batch workspace sequentially only.
"""
struct CompositeTrajectoryBatchWorkspace{P,W,R,S}
    plan::P
    workers::W
    rngs::R
    seeds::S
end


function CompositeTrajectoryBatchWorkspace(plan::CompositeTrajectoryPlan,
        rho::CompositePIState;workers::Integer=Threads.nthreads())
    workers>0||throw(ArgumentError("worker count must be positive"))
    workspaces=[CompositeTrajectoryWorkspace(plan,rho)
                for _ in 1:Int(workers)]
    rngs=[MersenneTwister(0) for _ in 1:Int(workers)]
    CompositeTrajectoryBatchWorkspace(plan,workspaces,rngs,UInt64[])
end

@inline function _apply_composite_jump_actions!(x,term,termwork,plan,w,t,p)
    _apply_composite_actions!(x,w.tensor_buffer1,w.tensor_buffer2,
        term.actions,termwork.factors,plan.basis.dimensions,1,false,t,p)
end

@inline function _add_composite_source!(y,source,scale)
    @inbounds @simd for index in eachindex(y,source)
        y[index]+=scale*source[index]
    end
    y
end

function _composite_jump_rate(channel,t,p,::Type{R}) where R<:AbstractFloat
    raw=value_at(channel.rate,t,p)
    raw isa Real||throw(ArgumentError("composite jump rate must be real"))
    value=_checked_composite_coefficient(R,raw)
    value>=zero(R)||throw(ArgumentError(
        "composite quantum trajectories require nonnegative jump rates"))
    value
end

function _checked_composite_gain_trace(value,::Type{R},context) where
        R<:AbstractFloat
    real_value=_checked_composite_trace_value(
        value,R,context;real_required=true)
    tolerance=_intensity_tolerance(R)*max(one(R),abs(real_value))
    real_value>=-tolerance||throw(ArgumentError(
        "$context is negative ($real_value); inspect state positivity and channel data"))
    max(zero(R),real_value)
end

function _checked_scaled_composite_intensity(scale::R,value::R) where
        R<:AbstractFloat
    (iszero(scale)||iszero(value))&&return zero(R)
    result=scale*value
    isfinite(result)&&!iszero(result)||throw(ArgumentError(
        "composite jump intensity is outside the nonzero finite range of $R; use a wider scalar type"))
    result
end

@inline _composite_channel_intensities!(w,x,t,p,::Tuple{},::Tuple{},index)=
    nothing

@inline function _composite_channel_intensities!(w,x,t,p,
        channels::Tuple{C,Vararg{Any}},works::Tuple{W,Vararg{Any}},
        index) where {C,W}
    channel=first(channels)
    rate=_composite_jump_rate(channel,t,p,eltype(w.intensities))
    w.jump_scales[index]=rate
    if iszero(rate)
        w.gain_traces[index]=zero(rate)
        w.intensities[index]=zero(rate)
    else
        source=_apply_composite_jump_actions!(x,channel.loss_left,
            first(works).loss_left,w.plan,w,t,p)
        unscaled=_checked_composite_gain_trace(
            _prepared_composite_trace(w.plan.trace_plan,source),
            eltype(w.intensities),"composite jump intensity")
        w.gain_traces[index]=unscaled
        w.intensities[index]=_checked_scaled_composite_intensity(rate,unscaled)
    end
    _composite_channel_intensities!(w,x,t,p,Base.tail(channels),
                                    Base.tail(works),index+1)
end

function _composite_channel_intensities!(w::CompositeTrajectoryWorkspace,
                                         x,t,p)
    _composite_channel_intensities!(w,x,t,p,w.plan.channels,
                                    w.channel_work,1)
    w.intensities
end

@inline _apply_composite_conditional_channels!(y,x,w,t,p,::Tuple{},
                                               ::Tuple{},index)=
    zero(eltype(w.intensities))

@inline function _apply_composite_conditional_channels!(y,x,w,t,p,
        channels::Tuple{C,Vararg{Any}},works::Tuple{W,Vararg{Any}},
        index) where {C,W}
    channel=first(channels)
    channel_work=first(works)
    rate=_composite_jump_rate(channel,t,p,eltype(w.intensities))
    w.jump_scales[index]=rate
    intensity=zero(rate)
    if iszero(rate)
        w.gain_traces[index]=zero(rate)
        w.intensities[index]=zero(rate)
    else
        left_source=_apply_composite_jump_actions!(x,channel.loss_left,
            channel_work.loss_left,w.plan,w,t,p)
        unscaled=_checked_composite_gain_trace(
            _prepared_composite_trace(w.plan.trace_plan,left_source),
            eltype(w.intensities),"composite jump intensity")
        intensity=_checked_scaled_composite_intensity(rate,unscaled)
        w.gain_traces[index]=unscaled
        w.intensities[index]=intensity
        _add_composite_source!(y,left_source,-rate/2)
        right_source=_apply_composite_jump_actions!(x,channel.loss_right,
            channel_work.loss_right,w.plan,w,t,p)
        _add_composite_source!(y,right_source,-rate/2)
    end
    intensity+_apply_composite_conditional_channels!(
        y,x,w,t,p,Base.tail(channels),Base.tail(works),index+1)
end

function _composite_conditional_action!(y,x,w,t,p)
    apply!(y,w.plan.background,x,t,p,w.background_work)
    R=eltype(w.intensities)
    background_trace=_checked_composite_trace_value(
        _prepared_composite_trace(w.plan.trace_plan,y),R,
        "composite background trace derivative")
    _composite_trace_zero_check(background_trace,norm(y),R,
        "composite trajectory background")
    total=_apply_composite_conditional_channels!(
        y,x,w,t,p,w.plan.channels,w.channel_work,1)
    isfinite(total)||throw(ArgumentError(
        "total composite jump intensity is nonfinite"))
    @inbounds @simd for index in eachindex(y,x)
        y[index]+=total*x[index]
    end
    drift_trace=_checked_composite_trace_value(
        _prepared_composite_trace(w.plan.trace_plan,y),R,
        "composite conditional trace derivative")
    _composite_trace_zero_check(drift_trace,max(norm(y),total),R,
        "composite conditional evolution")
    y
end

@inline function _initialize_composite_hazards!(w)
    copyto!(w.integrated_intensities,w.intensities)
end

@inline function _accumulate_composite_hazards!(w,weight)
    @inbounds @simd for index in eachindex(w.integrated_intensities,
                                           w.intensities)
        w.integrated_intensities[index]+=weight*w.intensities[index]
    end
end

@inline function _finish_composite_hazards!(w,h)
    scale=h/6
    @inbounds @simd for index in eachindex(w.integrated_intensities,
                                           w.intensities)
        w.integrated_intensities[index]=scale*(
            w.integrated_intensities[index]+w.intensities[index])
    end
    _total_intensity(w.integrated_intensities)
end

function _composite_conditional_rk4_trial!(x,w,t,h,p,hazard_limit)
    _composite_conditional_action!(w.k1,x,w,t,p)
    _initialize_composite_hazards!(w)
    @. w.tmp=x+(h/2)*w.k1
    _composite_conditional_action!(w.k2,w.tmp,w,t+h/2,p)
    _accumulate_composite_hazards!(w,2)
    @. w.tmp=x+(h/2)*w.k2
    _composite_conditional_action!(w.k3,w.tmp,w,t+h/2,p)
    _accumulate_composite_hazards!(w,2)
    @. w.tmp=x+h*w.k3
    _composite_conditional_action!(w.k4,w.tmp,w,t+h,p)
    hazard=_finish_composite_hazards!(w,h)
    hazard<=hazard_limit||return false,hazard
    @. x=x+(h/6)*(w.k1+2w.k2+2w.k3+w.k4)
    trace_value=_checked_composite_trace_value(
        _prepared_composite_trace(w.plan.trace_plan,x),
        eltype(w.intensities),"conditional composite state trace")
    abs(trace_value)>eps(eltype(w.intensities))*max(one(eltype(w.intensities)),norm(x))||
        throw(ArgumentError("conditional composite state acquired zero trace"))
    x./=trace_value
    true,hazard
end

function _composite_capped_conditional_step!(x,w,t,h,p,
        max_jump_probability,hazard_limit)
    R=eltype(w.intensities)
    while true
        h>zero(R)&&t+h>t||throw(ErrorException(
            "composite trajectory step cannot advance time at t=$t; use a wider precision or a smaller rate"))
        accepted,hazard=_composite_conditional_rk4_trial!(
            x,w,t,h,p,hazard_limit)
        if accepted
            probability=-expm1(-hazard)
            if probability>max_jump_probability
                # `hazard_limit` is obtained from the cap with `log1p`.
                # Its inverse `expm1` can round one ulp above the original
                # floating-point cap even though the accepted hazard is not
                # larger than that limit.  Repair only that round trip; a
                # larger excess still signals a broken cap invariant.
                probability<=nextfloat(max_jump_probability)||
                    throw(ErrorException(
                        "accepted composite jump probability exceeds its configured cap"))
                probability=max_jump_probability
            end
            return h,hazard,probability
        end
        reduction=R(0.9)*hazard_limit/hazard
        zero(R)<reduction<one(R)||throw(ErrorException(
            "cannot reduce a composite trajectory step with hazard $hazard"))
        new_h=h*reduction
        new_h>zero(R)&&t+new_h>t||throw(ErrorException(
            "composite trajectory step cannot advance time at t=$t; use a wider precision or a smaller rate"))
        h=new_h
    end
end

@inline function _apply_selected_composite_gain!(w,x,target,t,p,
        channels::Tuple{C,Vararg{Any}},works::Tuple{W,Vararg{Any}},
        index) where {C,W}
    if index!=target
        return _apply_selected_composite_gain!(
            w,x,target,t,p,Base.tail(channels),Base.tail(works),index+1)
    end
    channel=first(channels)
    channel_work=first(works)
    source=_apply_composite_jump_actions!(x,channel.gain,
        channel_work.gain,w.plan,w,t,p)
    copyto!(w.channel_gain,source)
    gain_trace=_checked_composite_gain_trace(
        _prepared_composite_trace(w.plan.trace_plan,w.channel_gain),
        eltype(w.intensities),"selected composite jump gain")
    gain_trace>zero(gain_trace)||throw(ArgumentError(
        "selected composite jump has zero gain trace"))
    w.channel_gain./=gain_trace
    copyto!(x,w.channel_gain)
    x
end

@inline _apply_selected_composite_gain!(w,x,target,t,p,::Tuple{},::Tuple{},
                                        index)=throw(BoundsError(
    w.plan.channels,target))

function _apply_selected_composite_gain!(w,x,channel_index,t,p)
    _apply_selected_composite_gain!(w,x,channel_index,t,p,
        w.plan.channels,w.channel_work,1)
end

"""One density-valued composite quantum-jump realization."""
struct CompositeQuantumTrajectory{T,S<:CompositePIState}
    times::Vector{T}
    states::Vector{S}
    jump_times::Vector{T}
    jump_channels::Vector{Int}
end

function _prepare_composite_streaming_observables(b::CompositePIBasis,
                                                  observables)
    observables===nothing&&return ()
    named=_named_observables(observables)
    isempty(named)&&throw(ArgumentError("observables cannot be empty"))
    prepared=Pair[]
    seen=Set{Any}()
    for (name,observable) in named
        name in seen&&throw(ArgumentError("duplicate observable name $name"))
        push!(seen,name)
        observable isa CompositePIOperator&&observable.basis===b||
            throw(ArgumentError(
                "composite trajectory observables must be compatible CompositePIOperators"))
        ishermitian(observable)||throw(ArgumentError(
            "composite trajectory observable statistics require Hermitian observables"))
        push!(prepared,name=>observable)
    end
    Tuple(prepared)
end

function _validate_composite_trajectory_initial_state(plan,rho)
    rho.basis===plan.basis||throw(ArgumentError(
        "state and composite trajectory plan use incompatible bases"))
    all(value->isfinite(real(value))&&isfinite(imag(value)),rho.data)||
        throw(ArgumentError("initial composite trajectory state must be finite"))
    R=_real_float_type(eltype(rho.data))
    value=_checked_composite_trace_value(
        _prepared_composite_trace(plan.trace_plan,rho.data),R,
        "initial composite trajectory trace")
    tolerance=max(R(1e-10),R(100)*eps(R))
    abs(value-one(R))<=tolerance||throw(ArgumentError(
        "initial composite trajectory state must have unit trace"))
    nothing
end

function _check_composite_trajectory_workspace(work,plan,rho)
    work isa CompositeTrajectoryWorkspace||throw(ArgumentError(
        "workspace must be a CompositeTrajectoryWorkspace"))
    work.plan===plan||throw(ArgumentError(
        "composite trajectory workspace belongs to a different plan"))
    rho.basis===plan.basis||throw(ArgumentError(
        "state and composite trajectory workspace use incompatible bases"))
    eltype(work.current)===eltype(rho.data)||throw(ArgumentError(
        "composite trajectory workspace has an incompatible scalar type"))
    work
end

function _check_composite_batch_workspace(batch,plan,rho)
    batch isa CompositeTrajectoryBatchWorkspace||throw(ArgumentError(
        "workspace must be a CompositeTrajectoryWorkspace or CompositeTrajectoryBatchWorkspace"))
    batch.plan===plan||throw(ArgumentError(
        "composite trajectory batch workspace belongs to a different plan"))
    isempty(batch.workers)&&throw(ArgumentError(
        "composite trajectory batch workspace has no workers"))
    length(batch.workers)==length(batch.rngs)||throw(ArgumentError(
        "composite trajectory batch workspace has inconsistent storage"))
    batch.seeds isa Vector{UInt64}||throw(ArgumentError(
        "composite trajectory batch workspace has incompatible seed storage"))
    all(rng->rng isa AbstractRNG,batch.rngs)||throw(ArgumentError(
        "composite trajectory batch workspace has incompatible RNG storage"))
    for worker in batch.workers
        _check_composite_trajectory_workspace(worker,plan,rho)
    end
    batch
end

function _composite_quantum_trajectory_prepared(plan,rho0,ts,w,rng,options;
        observable_ops=nothing,observable_values=nothing,
        save_states::Bool=true,record_jumps::Bool=true)
    save_states&&!record_jumps&&throw(ArgumentError(
        "saved trajectories require recorded jump histories"))
    x=w.current
    copyto!(x,rho0.data)
    states=save_states ? Vector{typeof(rho0)}(undef,length(ts)) : nothing
    save_states&&(states[1]=copy(rho0))
    observable_values===nothing||_record_observables!(
        observable_values,observable_ops,x,1)
    jump_times=record_jumps ? eltype(ts)[] : nothing
    jump_channels=record_jumps ? Int[] : nothing
    t=ts[1]
    R=eltype(w.intensities)
    hazard_limit=-log1p(-options.max_jump_probability)
    for output_index in 2:length(ts)
        target=ts[output_index]
        while t<target
            h,lands_on_target=_trajectory_step_to_target(t,target,options.dt)
            proposed_h=h
            h,total_hazard,jump_probability=
                _composite_capped_conditional_step!(
                    x,w,t,h,options.parameters,
                    options.max_jump_probability,hazard_limit)
            h==proposed_h||(lands_on_target=false)
            t=lands_on_target ? target : t+h
            if total_hazard>zero(total_hazard)&&
               rand(rng,R)<jump_probability
                channel=_select_jump_channel(
                    w.integrated_intensities,rand(rng,R)*total_hazard)
                _apply_selected_composite_gain!(
                    w,x,channel,t,options.parameters)
                if record_jumps
                    push!(jump_times,t)
                    push!(jump_channels,channel)
                end
            end
        end
        save_states&&(states[output_index]=CompositePIState(plan.basis,x))
        observable_values===nothing||_record_observables!(
            observable_values,observable_ops,x,output_index)
    end
    save_states ? CompositeQuantumTrajectory(
        ts,states,jump_times,jump_channels) :
        (;jump_times,jump_channels)
end

"""
    quantum_trajectory(plan::CompositeTrajectoryPlan, rho0, times;
                       dt, rng=Random.default_rng(), parameters=nothing,
                       max_jump_probability=0.05, workspace=nothing)

Simulate one density-valued composite quantum-jump trajectory with
preallocated RK4 conditional evolution. Channel hazards are integrated with
the same four stages, and a trial step is shortened and retried until its
total jump probability is at most `max_jump_probability`. Rates, times, and
`dt` must be representable in the prepared real precision.

Composite event-driven trajectories are not currently implemented; refine
`dt` explicitly to establish time-discretization convergence.
"""
function quantum_trajectory(plan::CompositeTrajectoryPlan,
        rho0::CompositePIState{R},times;dt::Real,
        rng::AbstractRNG=Random.default_rng(),parameters=nothing,
        max_jump_probability=nothing,workspace=nothing,
        algorithm::Symbol=:fixed,kwargs...) where R<:AbstractFloat
    isempty(kwargs)||throw(ArgumentError(
        "adaptive trajectory controls are unavailable for composite fixed-step trajectories"))
    algorithm===:fixed||throw(ArgumentError(
        "composite trajectories currently support only algorithm=:fixed"))
    w=workspace===nothing ? CompositeTrajectoryWorkspace(plan,rho0) :
        _check_composite_trajectory_workspace(workspace,plan,rho0)
    _validate_composite_trajectory_initial_state(plan,rho0)
    ts,options=_prepare_trajectory_arguments(times,R;dt,parameters,
        max_jump_probability,algorithm=:fixed)
    _composite_quantum_trajectory_prepared(plan,rho0,ts,w,rng,options)
end

function _composite_ensemble_resources(plan,rho,n,threaded,workspace)
    if workspace===nothing
        count=threaded ? min(Int(n),Threads.nthreads()) : 1
        batch=CompositeTrajectoryBatchWorkspace(plan,rho;workers=count)
        return batch,batch.workers,batch.rngs
    elseif workspace isa CompositeTrajectoryWorkspace
        threaded&&throw(ArgumentError(
            "threaded composite ensembles require a CompositeTrajectoryBatchWorkspace"))
        worker=_check_composite_trajectory_workspace(workspace,plan,rho)
        return nothing,(worker,),(MersenneTwister(0),)
    end
    batch=_check_composite_batch_workspace(workspace,plan,rho)
    batch,batch.workers,batch.rngs
end

function _composite_ensemble_seeds!(batch,rngs,n,seed)
    master=rngs[1]
    Random.seed!(master,seed)
    if batch===nothing
        return rand(master,UInt64,n)
    end
    resize!(batch.seeds,n)
    rand!(master,batch.seeds)
    batch.seeds
end

function _run_composite_path!(out,observable_accumulators,jump_accumulators,
        observable_buffers,observable_ops,plan,rho0,ts,worker,rng,options,
        trajectory_index,worker_index,save_states,record_jumps)
    values=observable_buffers===nothing ? nothing :
        observable_buffers[worker_index]
    result=_composite_quantum_trajectory_prepared(
        plan,rho0,save_states ? copy(ts) : ts,worker,rng,options;
        observable_ops,observable_values=values,save_states,record_jumps)
    save_states&&(out[trajectory_index]=result)
    observable_accumulators===nothing||_accumulate_observables!(
        observable_accumulators[worker_index],values)
    jump_accumulators===nothing||_accumulate_jumps!(
        jump_accumulators[worker_index],result.jump_times,
        result.jump_channels)
    nothing
end

"""
    quantum_trajectories(plan::CompositeTrajectoryPlan, rho0, times, n;
                         seed=0, threaded=false, workspace=nothing,
                         observables=nothing, save_states=true,
                         jump_statistics=true, confidence=0.95, dt, ...)

Generate independent density-valued composite trajectories. The immutable
plan is shared, each worker owns its workspace and RNG, and random streams are
derived from the global trajectory index. Serial and threaded calls therefore
sample the same ordered paths for a fixed seed.

With named Hermitian `CompositePIOperator` observables and
`save_states=false`, return a [`TrajectoryEnsembleResult`](@ref) containing
online means, variances, confidence intervals, and optional jump statistics
without constructing sampled state histories.
"""
function quantum_trajectories(plan::CompositeTrajectoryPlan,
        rho0::CompositePIState{R},times,n::Integer;seed::Integer=0,
        threaded::Bool=false,workspace=nothing,dt::Real,
        parameters=nothing,max_jump_probability=nothing,
        algorithm::Symbol=:fixed,observables=nothing,
        save_states::Bool=true,jump_statistics::Bool=true,
        confidence::Real=0.95,kwargs...) where R<:AbstractFloat
    n>0||throw(ArgumentError("trajectory count must be positive"))
    isempty(kwargs)||throw(ArgumentError(
        "adaptive trajectory controls are unavailable for composite fixed-step trajectories"))
    algorithm===:fixed||throw(ArgumentError(
        "composite trajectories currently support only algorithm=:fixed"))
    !save_states&&observables===nothing&&throw(ArgumentError(
        "save_states=false requires at least one composite observable"))
    _validate_composite_trajectory_initial_state(plan,rho0)
    ts,options=_prepare_trajectory_arguments(times,R;dt,parameters,
        max_jump_probability,algorithm=:fixed)
    batch,workers,rngs=_composite_ensemble_resources(
        plan,rho0,n,threaded,workspace)
    seeds=_composite_ensemble_seeds!(batch,rngs,Int(n),seed)
    available=length(workers)
    worker_count=threaded ? min(Int(n),Threads.nthreads(),available) : 1
    trajectory_type=CompositeQuantumTrajectory{eltype(ts),typeof(rho0)}
    out=save_states ? Vector{trajectory_type}(undef,Int(n)) : nothing
    ops=_prepare_composite_streaming_observables(plan.basis,observables)
    return_legacy=observables===nothing&&save_states
    0<confidence<1||throw(ArgumentError("confidence must lie in (0,1)"))
    Rstats=_observable_scalar_type(rho0,ops)
    observable_buffers=isempty(ops) ? nothing :
        [Matrix{Rstats}(undef,length(ops),length(ts))
         for _ in 1:worker_count]
    observable_accumulators=isempty(ops) ? nothing :
        [_OnlineObservableAccumulator(Rstats,length(ops),length(ts))
         for _ in 1:worker_count]
    collect_jumps=!return_legacy&&jump_statistics
    jump_accumulators=collect_jumps ?
        [_OnlineJumpAccumulator(R,length(plan.channels))
         for _ in 1:worker_count] : nothing
    record_jumps=save_states||collect_jumps

    if worker_count==1
        worker=workers[1]
        rng=rngs[1]
        for trajectory_index in 1:Int(n)
            Random.seed!(rng,seeds[trajectory_index])
            _run_composite_path!(out,observable_accumulators,
                jump_accumulators,observable_buffers,ops,plan,rho0,ts,
                worker,rng,options,trajectory_index,1,save_states,
                record_jumps)
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
                            _run_composite_path!(out,observable_accumulators,
                                jump_accumulators,observable_buffers,ops,plan,
                                rho0,ts,worker,rng,options,trajectory_index,
                                worker_id,save_states,record_jumps)
                        end
                    end
                end
            end
        end
    end
    return_legacy&&return out

    observable_summary=if observable_accumulators===nothing
        nothing
    else
        merged=observable_accumulators[1]
        for worker_index in 2:worker_count
            _merge_observables!(merged,observable_accumulators[worker_index])
        end
        _observable_statistics(merged,ops,ts,confidence)
    end
    jump_summary=if jump_accumulators===nothing
        nothing
    else
        merged=jump_accumulators[1]
        for worker_index in 2:worker_count
            _merge_jumps!(merged,jump_accumulators[worker_index])
        end
        _jump_statistics(merged,ts)
    end
    P=Union{Nothing,Vector{trajectory_type}}
    TrajectoryEnsembleResult{eltype(ts),P,typeof(observable_summary)}(
        copy(ts),out,observable_summary,jump_summary,Int(n))
end

"""Average equally sampled composite quantum trajectories."""
function trajectory_average(
        trajectories::AbstractVector{<:CompositeQuantumTrajectory})
    times,basis=_check_composite_trajectory_ensemble(trajectories)
    R=_real_float_type(eltype(trajectories[1].states[1].data))
    output=[CompositePIState(basis;T=R) for _ in eachindex(times)]
    for path in trajectories,index in eachindex(times)
        output[index].data .+= path.states[index].data
    end
    denominator=_checked_statistics_count(
        R,length(trajectories),"composite trajectory average")
    for state in output
        state.data./=denominator
    end
    output
end

function _check_composite_trajectory_ensemble(trajectories)
    isempty(trajectories)&&throw(ArgumentError(
        "at least one composite trajectory is required"))
    times=trajectories[1].times
    isempty(times)&&throw(ArgumentError(
        "composite trajectories require at least one sampling time"))
    all(isfinite,times)&&issorted(times)||throw(ArgumentError(
        "composite trajectory sampling times must be finite and nondecreasing"))
    length(trajectories[1].states)==length(times)||throw(ArgumentError(
        "composite trajectory state and sampling-time counts differ"))
    basis=trajectories[1].states[1].basis
    state_type=eltype(trajectories[1].states[1].data)
    for path in trajectories
        path.times==times&&length(path.states)==length(times)||
            throw(ArgumentError(
                "composite trajectories must share sampling times"))
        all(state->state.basis===basis&&eltype(state.data)===state_type,
            path.states)||throw(ArgumentError(
                "every saved composite state must share the same basis and scalar type"))
        length(path.jump_times)==length(path.jump_channels)||
            throw(ArgumentError(
                "composite jump-time and channel histories have different lengths"))
        issorted(path.jump_times)&&
            all(time->times[1]<=time<=times[end],path.jump_times)||
            throw(ArgumentError(
                "composite jump times must be ordered inside the sampling interval"))
        all(channel->channel>0,path.jump_channels)||throw(ArgumentError(
            "composite jump channel indices must be positive"))
    end
    times,basis
end

function jump_statistics(
        trajectories::AbstractVector{<:CompositeQuantumTrajectory};
        nchannels=nothing)
    times,_=_check_composite_trajectory_ensemble(trajectories)
    inferred=maximum((isempty(path.jump_channels) ? 0 :
        maximum(path.jump_channels) for path in trajectories);init=0)
    channel_count=nchannels===nothing ? inferred : Int(nchannels)
    channel_count>=inferred||throw(ArgumentError(
        "nchannels is smaller than an observed channel index"))
    channel_count>=0||throw(ArgumentError(
        "nchannels must be nonnegative"))
    accumulator=_OnlineJumpAccumulator(
        _real_float_type(eltype(times)),channel_count)
    for path in trajectories
        _accumulate_jumps!(accumulator,path.jump_times,path.jump_channels)
    end
    _jump_statistics(accumulator,times)
end

function trajectory_observable_statistics(
        trajectories::AbstractVector{<:CompositeQuantumTrajectory},
        observables;confidence::Real=0.95)
    times,basis=_check_composite_trajectory_ensemble(trajectories)
    0<confidence<1||throw(ArgumentError("confidence must lie in (0,1)"))
    ops=_prepare_composite_streaming_observables(basis,observables)
    R=_observable_scalar_type(trajectories[1].states[1],ops)
    accumulator=_OnlineObservableAccumulator(R,length(ops),length(times))
    values=Matrix{R}(undef,length(ops),length(times))
    for path in trajectories
        for index in eachindex(times)
            _record_observables!(values,ops,path.states[index].data,index)
        end
        _accumulate_observables!(accumulator,values)
    end
    _observable_statistics(accumulator,ops,times,confidence)
end

function trajectory_statistics(
        trajectories::AbstractVector{<:CompositeQuantumTrajectory};
        observables=nothing,confidence::Real=0.95,nchannels=nothing)
    times,_=_check_composite_trajectory_ensemble(trajectories)
    observable_summary=observables===nothing ? nothing :
        trajectory_observable_statistics(
            trajectories,observables;confidence)
    (;times=copy(times),average_states=trajectory_average(trajectories),
      jumps=jump_statistics(trajectories;nchannels),
      observables=observable_summary)
end
