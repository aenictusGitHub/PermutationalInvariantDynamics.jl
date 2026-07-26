"""
    BatchedConditionalPlan(source[, rho])

Read-only plan for propagating several density-matrix PI quantum trajectories
through the same conditional (no-jump) time interval. `source` may be a
[`TrajectoryPlan`](@ref), `PIModel`, or `CompiledPIModel`. The optional `rho`
argument is needed only to infer the scalar type of an empty model.

This is deliberately a low-level plan. Different stochastic paths generally
acquire different intensity-limited step sizes after their first jump. A
high-level routine that forced all paths onto the smallest such step would no
longer reproduce [`quantum_trajectories`](@ref). This plan therefore batches
only cohorts for which the caller has already selected the same physical
`time` and `step`; it never changes either value.
"""
struct BatchedConditionalPlan{P}
    trajectory::P
end

BatchedConditionalPlan(plan::TrajectoryPlan)=BatchedConditionalPlan{typeof(plan)}(plan)
BatchedConditionalPlan(model::PIModel)=BatchedConditionalPlan(TrajectoryPlan(model))
BatchedConditionalPlan(compiled::CompiledPIModel)=
    BatchedConditionalPlan(TrajectoryPlan(compiled))
BatchedConditionalPlan(model::PIModel,rho::PIState)=
    BatchedConditionalPlan(_trajectory_plan_for_state(model,rho))
BatchedConditionalPlan(compiled::CompiledPIModel,rho::PIState)=
    BatchedConditionalPlan(_trajectory_plan_for_state(compiled,rho))

Base.size(plan::BatchedConditionalPlan)=size(plan.trajectory.liouvillian)
Base.size(plan::BatchedConditionalPlan,index::Integer)=
    size(plan.trajectory.liouvillian,index)
Base.eltype(plan::BatchedConditionalPlan)=eltype(plan.trajectory.liouvillian)
isautonomous(plan::BatchedConditionalPlan)=isautonomous(plan.trajectory)

"""
    BatchedConditionalWorkspace(plan, rho, capacity; memory_budget=512MiB)

Fixed-capacity, task-owned scratch for [`BatchedConditionalPlan`](@ref).
The five `length(rho) × capacity` stage/gather matrices and the prepared
Liouvillian batch buffers are allocated by the constructor. Subsequent calls
reject a wider matrix instead of growing hidden storage.

One workspace may be reused sequentially. Concurrent calls need distinct
workspaces referring to the same immutable plan.
"""
struct BatchedConditionalWorkspace{P,V,R,L,E,I,C}
    plan::P
    capacity::Int
    tmp::V
    k1::V
    k2::V
    gather::V
    gain::V
    total_intensities::Vector{R}
    channel_intensities::Matrix{R}
    jump_scales::Vector{R}
    liouvillian_work::L
    effective_qblocks::E
    effective_cache::C
    selected_indices::I
end

function _batched_conditional_capacity(capacity::Integer)
    capacity isa Bool&&throw(ArgumentError(
        "batch capacity must be an integer, not Bool"))
    capacity>0||throw(ArgumentError("batch capacity must be positive"))
    BigInt(capacity)<=typemax(Int)||throw(ArgumentError(
        "batch capacity exceeds the addressable Int range"))
    Int(capacity)
end

function _batched_conditional_workspace_bytes(plan::BatchedConditionalPlan,
        ::Type{T},capacity::Integer;
        bigfloat_precision::Integer=precision(BigFloat)) where T
    columns=_batched_conditional_capacity(capacity)
    trajectory=plan.trajectory
    n=BigInt(length(trajectory.model.basis))
    njumps=BigInt(length(trajectory.jumps))
    R=_real_float_type(T)
    matrix_bytes=_performance_entries_bytes(
        5n*BigInt(columns),T;bigfloat_precision)
    rate_bytes=_performance_entries_bytes(
        (njumps+1)*BigInt(columns)+njumps,R;bigfloat_precision)
    qentries=isempty(trajectory.jumps) ? big(0) :
        sum(block->BigInt(length(block)),
            first(trajectory.jumps).qblocks;init=big(0))
    effective_bytes=_performance_entries_bytes(
        qentries,T;bigfloat_precision)
    index_bytes=BigInt(columns)*sizeof(Int)
    liouvillian_bytes=_performance_liouvillian_workspace_bytes(
        trajectory.liouvillian;batch_columns=columns,bigfloat_precision)
    matrix_bytes+rate_bytes+effective_bytes+index_bytes+liouvillian_bytes
end

function BatchedConditionalWorkspace(plan::BatchedConditionalPlan,
        rho::PIState,capacity::Integer;
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    columns=_batched_conditional_capacity(capacity)
    trajectory=plan.trajectory
    rho.basis===trajectory.model.basis||throw(ArgumentError(
        "state and batched conditional plan use incompatible PI bases"))
    _check_liouvillian_source_precision(
        trajectory.liouvillian,eltype(rho.data),"batched trajectory state")
    promote_type(eltype(rho.data),trajectory.liouvillian.Ttype)===
        eltype(rho.data)||throw(ArgumentError(
        "batched trajectory state scalar type $(eltype(rho.data)) cannot represent plan scalar type $(trajectory.liouvillian.Ttype)"))
    estimate=_batched_conditional_workspace_bytes(
        plan,eltype(rho.data),columns)
    _require_performance_budget("batched conditional trajectory workspace",
        estimate,memory_budget;guidance=
        "Reduce capacity, use fewer concurrent cohorts, or pass memory_budget=Inf to opt out explicitly.")

    n=length(rho.data)
    # Allocate exactly the five retained matrices counted by the preflight.
    # A temporary prototype plus five `similar` calls would make setup peak
    # one full n-by-capacity matrix larger than the guarded estimate.
    tmp=Matrix{eltype(rho.data)}(undef,n,columns)
    k1=Matrix{eltype(rho.data)}(undef,n,columns)
    k2=Matrix{eltype(rho.data)}(undef,n,columns)
    gather=Matrix{eltype(rho.data)}(undef,n,columns)
    gain=Matrix{eltype(rho.data)}(undef,n,columns)
    R=_real_float_type(eltype(rho.data))
    work=LiouvillianWorkspace(trajectory.liouvillian)
    # Allocate the promised capacity once. Public batch methods below reject
    # larger inputs before reaching any lazily growing Liouvillian helper.
    _ensure_batch_capacity!(work.batch,columns)
    effective=isempty(trajectory.jumps) ? Matrix{eltype(rho.data)}[] :
        [zeros(eltype(rho.data),length(rho.basis.patterns[sector]),
                                  length(rho.basis.patterns[sector]))
         for sector in eachindex(rho.basis.sectors)]
    BatchedConditionalWorkspace(
        plan,columns,tmp,k1,k2,gather,gain,
        zeros(R,columns),
        zeros(R,length(trajectory.jumps),columns),
        zeros(R,length(trajectory.jumps)),work,effective,
        _EffectiveJumpNodeCache(zero(R),false),
        Vector{Int}(undef,columns))
end

BatchedConditionalWorkspace(plan::TrajectoryPlan,rho::PIState,
        capacity::Integer;kwargs...)=
    BatchedConditionalWorkspace(BatchedConditionalPlan(plan),rho,capacity;
                                kwargs...)
BatchedConditionalWorkspace(model::PIModel,rho::PIState,
        capacity::Integer;kwargs...)=
    BatchedConditionalWorkspace(BatchedConditionalPlan(model,rho),rho,capacity;
                                kwargs...)
BatchedConditionalWorkspace(compiled::CompiledPIModel,rho::PIState,
        capacity::Integer;kwargs...)=
    BatchedConditionalWorkspace(BatchedConditionalPlan(compiled,rho),rho,
                                capacity;kwargs...)

function _check_batched_conditional_workspace(
        work::BatchedConditionalWorkspace,plan::BatchedConditionalPlan)
    work.plan===plan||throw(ArgumentError(
        "batched conditional workspace belongs to a different plan"))
    work.liouvillian_work.basis===plan.trajectory.model.basis||
        throw(ArgumentError(
        "batched conditional workspace has an incompatible PI basis"))
    work
end

function _check_batched_trajectory_matrix(X,plan::BatchedConditionalPlan,
        work::BatchedConditionalWorkspace,label)
    size(X,1)==length(plan.trajectory.model.basis)||throw(DimensionMismatch(
        "$label has the wrong leading dimension"))
    size(X,2)<=work.capacity||throw(ArgumentError(
        "$label has $(size(X,2)) columns, exceeding the fixed workspace capacity $(work.capacity)"))
    eltype(X)===eltype(work.tmp)||throw(ArgumentError(
        "$label has scalar type $(eltype(X)); the workspace requires $(eltype(work.tmp))"))
    X
end

function _check_batched_destination(Y,X,plan,work)
    _check_batched_trajectory_matrix(X,plan,work,"batched trajectory source")
    size(Y)==size(X)||throw(DimensionMismatch(
        "batched trajectory destination has the wrong dimensions"))
    eltype(Y)===eltype(X)||throw(ArgumentError(
        "batched trajectory source and destination scalar types differ"))
    Base.mightalias(Y,X)&&throw(ArgumentError(
        "batched trajectory source and destination must not alias"))
    Y
end

@inline _apply_batched_trajectory_hamiltonians!(
    Y,X,::Tuple{},b,t,p,scratch)=nothing
@inline function _apply_batched_trajectory_hamiltonians!(
        Y,X,kernels::Tuple{K,Vararg{Any}},b,t,p,scratch) where K
    _apply_kernel_batch!(Y,X,first(kernels),b,t,p,scratch)
    _apply_batched_trajectory_hamiltonians!(
        Y,X,Base.tail(kernels),b,t,p,scratch)
end

function _invalidate_batched_effective_cache!(work)
    # Public calls may reuse a workspace at the same time with different
    # parameter objects. Invalidate driven rates at that boundary; internal
    # RK stages still reuse the duplicated midpoint node.
    _trajectory_jump_rates_autonomous(work.plan.trajectory.jumps)||
        (work.effective_cache.valid=false)
    work
end

@inline _accumulate_batched_effective_jump_blocks!(
    work,t,p,::Tuple{},index)=nothing
@inline function _accumulate_batched_effective_jump_blocks!(
        work,t,p,jumps::Tuple{K,Vararg{Any}},index) where K
    kernel=first(jumps)
    R=eltype(work.total_intensities)
    scale=_trajectory_jump_scale(kernel,t,p,R)
    work.jump_scales[index]=scale
    if !iszero(scale)
        @inbounds for sector in eachindex(work.effective_qblocks)
            effective=work.effective_qblocks[sector]
            qblock=kernel.qblocks[sector]
            for block_index in eachindex(effective,qblock)
                effective[block_index]+=scale*qblock[block_index]
            end
        end
    end
    _accumulate_batched_effective_jump_blocks!(
        work,t,p,Base.tail(jumps),index+1)
end

function _prepare_batched_effective_jump_blocks!(work,t,p)
    jumps=work.plan.trajectory.jumps
    if work.effective_cache.valid&&
       (_trajectory_jump_rates_autonomous(jumps)||
        isequal(work.effective_cache.time,t))
        return work.effective_qblocks
    end
    for block in work.effective_qblocks
        fill!(block,zero(eltype(block)))
    end
    _accumulate_batched_effective_jump_blocks!(work,t,p,jumps,1)
    work.effective_cache.time=t
    work.effective_cache.valid=true
    work.effective_qblocks
end

function _batched_effective_intensities!(destination,X,work,b)
    columns=size(X,2)
    @inbounds for rhs in 1:columns
        total=zero(eltype(destination))
        for sector in eachindex(b.sectors)
            dimension=length(b.patterns[sector])
            offset=b.offsets[sector]
            effective=work.effective_qblocks[sector]
            sector_trace=zero(eltype(X))
            for column in 1:dimension,row in 1:dimension
                coordinate=offset+column-1+(row-1)*dimension
                sector_trace+=effective[row,column]*X[coordinate,rhs]
            end
            total+=work.plan.trajectory.trace_weights[sector]*
                real(sector_trace)
        end
        tolerance=_intensity_tolerance(eltype(destination))
        isfinite(total)||throw(ArgumentError(
            "total jump intensity is nonfinite in batch column $rhs"))
        total>=-tolerance||throw(ArgumentError(
            "combined jump gain has negative trace $total in batch column $rhs"))
        destination[rhs]=max(zero(total),total)
    end
    destination
end

function _batched_conditional_action_and_intensity!(
        Y,lambdas,plan::BatchedConditionalPlan,X,t,p,
        work::BatchedConditionalWorkspace)
    columns=size(X,2)
    length(lambdas)==columns||throw(DimensionMismatch(
        "the intensity destination must have one entry per batch column"))
    eltype(lambdas)===eltype(work.total_intensities)||throw(ArgumentError(
        "the intensity destination has scalar type $(eltype(lambdas)); the workspace requires $(eltype(work.total_intensities))"))
    (Base.mightalias(lambdas,X)||Base.mightalias(lambdas,Y))&&
        throw(ArgumentError(
        "the intensity destination must not alias a trajectory matrix"))
    fill!(Y,zero(eltype(Y)))
    trajectory=plan.trajectory
    b=trajectory.model.basis
    _prepare_batched_effective_jump_blocks!(work,t,p)
    _apply_batched_trajectory_hamiltonians!(
        Y,X,trajectory.hamiltonians,b,t,p,work.liouvillian_work.batch)
    if !isempty(trajectory.jumps)
        for sector in eachindex(b.sectors)
            dimension=length(b.patterns[sector])
            offset=b.offsets[sector]
            effective=work.effective_qblocks[sector]
            _batch_add_left_right!(Y,X,offset,dimension,
                effective,effective,-one(eltype(Y))/2,
                -one(eltype(Y))/2,work.liouvillian_work.batch)
        end
        _batched_effective_intensities!(lambdas,X,work,b)
        @inbounds for rhs in 1:columns,index in axes(X,1)
            Y[index,rhs]+=lambdas[rhs]*X[index,rhs]
        end
    else
        fill!(lambdas,zero(eltype(lambdas)))
    end
    Y
end

"""
    batched_conditional_action!(Y, intensities, plan, X, time,
                                parameters, workspace)

Apply the normalized conditional trajectory drift to all columns of `X` with
one matrix-RHS Schur-kernel pass. `intensities[j]` receives the total jump
intensity of column `j`. `Y` and `X` must not alias and the column count may
not exceed the workspace capacity.

Driven jump-rate caches are invalidated at every public call, so reusing a
workspace with a new parameter object at the same time cannot expose stale
loss blocks.
"""
function batched_conditional_action!(Y::AbstractMatrix,
        intensities::AbstractVector,plan::BatchedConditionalPlan,
        X::AbstractMatrix,t,p,work::BatchedConditionalWorkspace)
    _check_batched_conditional_workspace(work,plan)
    _check_batched_destination(Y,X,plan,work)
    _invalidate_batched_effective_cache!(work)
    _batched_conditional_action_and_intensity!(
        Y,intensities,plan,X,t,p,work)
end

function batched_conditional_action!(Y::AbstractMatrix,
        intensities::AbstractVector,plan::BatchedConditionalPlan,
        X::AbstractMatrix,work::BatchedConditionalWorkspace)
    _require_autonomous(plan,"batched_conditional_action!")
    batched_conditional_action!(Y,intensities,plan,X,0.0,nothing,work)
end

"""
    batched_conditional_rk4!(X, plan, time, step, parameters, workspace)

Advance every column of `X` by the same requested fixed RK4 step under the
normalized conditional trajectory equation. Each column is trace-normalized
after the step, matching the scalar fixed-step path. The function neither
caps nor changes `step`; callers must use the same intensity-cap decision that
they would use for each scalar path and batch only columns sharing that step.
"""
function batched_conditional_rk4!(X::AbstractMatrix,
        plan::BatchedConditionalPlan,t,h,p,
        work::BatchedConditionalWorkspace)
    _check_batched_conditional_workspace(work,plan)
    _check_batched_trajectory_matrix(X,plan,work,"batched trajectory state")
    columns=size(X,2)
    R=eltype(work.total_intensities)
    tR=_trajectory_real_input(R,t,"batched trajectory time")
    hR=_trajectory_real_input(R,h,"batched trajectory step")
    hR>zero(R)||throw(ArgumentError(
        "batched trajectory step must be positive"))
    _invalidate_batched_effective_cache!(work)
    tmp=@view work.tmp[:,1:columns]
    k1=@view work.k1[:,1:columns]
    k2=@view work.k2[:,1:columns]
    lambdas=@view work.total_intensities[1:columns]

    _batched_conditional_action_and_intensity!(
        k1,lambdas,plan,X,tR,p,work)
    copyto!(k2,k1)
    @inbounds for index in eachindex(tmp,X,k1)
        tmp[index]=X[index]+(hR/2)*k1[index]
    end
    _batched_conditional_action_and_intensity!(
        k1,lambdas,plan,tmp,tR+hR/2,p,work)
    @inbounds for index in eachindex(k2,k1)
        k2[index]+=2k1[index]
    end
    @inbounds for index in eachindex(tmp,X,k1)
        tmp[index]=X[index]+(hR/2)*k1[index]
    end
    _batched_conditional_action_and_intensity!(
        k1,lambdas,plan,tmp,tR+hR/2,p,work)
    @inbounds for index in eachindex(k2,k1)
        k2[index]+=2k1[index]
    end
    @inbounds for index in eachindex(tmp,X,k1)
        tmp[index]=X[index]+hR*k1[index]
    end
    _batched_conditional_action_and_intensity!(
        k1,lambdas,plan,tmp,tR+hR,p,work)
    @inbounds for index in eachindex(X,k2,k1)
        X[index]+=(hR/6)*(k2[index]+k1[index])
    end

    tau=plan.trajectory.liouvillian.tracevec
    @inbounds for rhs in 1:columns
        z=zero(eltype(X))
        for index in axes(X,1)
            z+=conj(tau[index])*X[index,rhs]
        end
        abs(z)>eps(R)||throw(ArgumentError(
            "conditional state in batch column $rhs acquired zero trace"))
        for index in axes(X,1)
            X[index,rhs]/=z
        end
    end
    X
end

function batched_conditional_rk4!(X::AbstractMatrix,
        plan::BatchedConditionalPlan,t,h,
        work::BatchedConditionalWorkspace)
    _require_autonomous(plan,"batched_conditional_rk4!")
    batched_conditional_rk4!(X,plan,t,h,nothing,work)
end

function _batched_unscaled_channel_intensities!(
        destination,X,kernel,b,weights)
    columns=size(X,2)
    @inbounds for rhs in 1:columns
        value=zero(eltype(destination))
        for sector in eachindex(b.sectors)
            dimension=length(b.patterns[sector])
            offset=b.offsets[sector]
            qblock=kernel.qblocks[sector]
            sector_trace=zero(eltype(X))
            for column in 1:dimension,row in 1:dimension
                coordinate=offset+column-1+(row-1)*dimension
                sector_trace+=qblock[row,column]*X[coordinate,rhs]
            end
            value+=weights[sector]*real(sector_trace)
        end
        destination[rhs]=value
    end
    destination
end

@inline _batched_channel_intensities!(
    rates,X,work,b,::Tuple{},index)=nothing
@inline function _batched_channel_intensities!(
        rates,X,work,b,jumps::Tuple{K,Vararg{Any}},index) where K
    kernel=first(jumps)
    scale=work.jump_scales[index]
    row=@view rates[index,:]
    if iszero(scale)
        fill!(row,zero(eltype(row)))
    else
        _batched_unscaled_channel_intensities!(
            row,X,kernel,b,work.plan.trajectory.trace_weights)
        tolerance=_intensity_tolerance(eltype(rates))
        @inbounds for rhs in eachindex(row)
            value=scale*row[rhs]
            isfinite(value)||throw(ArgumentError(
                "jump intensity is nonfinite in batch column $rhs, channel $index"))
            value>=-tolerance||throw(ArgumentError(
                "jump gain has negative trace $value in batch column $rhs, channel $index"))
            row[rhs]=max(zero(value),value)
        end
    end
    _batched_channel_intensities!(
        rates,X,work,b,Base.tail(jumps),index+1)
end

"""
    batched_channel_intensities!(rates, plan, X, time, parameters, workspace)

Evaluate every channel intensity for every column. `rates` must have shape
`(number_of_channels, size(X,2))`. Scalar schedules are evaluated once per
channel and shared by the batch.
"""
function batched_channel_intensities!(rates::AbstractMatrix,
        plan::BatchedConditionalPlan,X::AbstractMatrix,t,p,
        work::BatchedConditionalWorkspace)
    _check_batched_conditional_workspace(work,plan)
    _check_batched_trajectory_matrix(X,plan,work,"batched trajectory state")
    expected=(length(plan.trajectory.jumps),size(X,2))
    size(rates)==expected||throw(DimensionMismatch(
        "channel intensity destination must have dimensions $expected"))
    eltype(rates)===eltype(work.channel_intensities)||throw(ArgumentError(
        "channel intensity destination has an incompatible scalar type"))
    _invalidate_batched_effective_cache!(work)
    _prepare_batched_effective_jump_blocks!(work,t,p)
    _batched_channel_intensities!(
        rates,X,work,plan.trajectory.model.basis,
        plan.trajectory.jumps,1)
    rates
end

function batched_channel_intensities!(rates::AbstractMatrix,
        plan::BatchedConditionalPlan,X::AbstractMatrix,
        work::BatchedConditionalWorkspace)
    _require_autonomous(plan,"batched_channel_intensities!")
    batched_channel_intensities!(rates,plan,X,0.0,nothing,work)
end

function _apply_gain_batch!(Y,X,k::DissipatorPIKernel,b,scale,work)
    fill!(Y,zero(eltype(Y)))
    for sector in eachindex(b.sectors)
        dimension=length(b.patterns[sector])
        _batch_add_sandwich!(Y,X,b.offsets[sector],dimension,
            k.blocks[sector],adjoint(k.blocks[sector]),scale,work.batch)
    end
    Y
end

function _apply_gain_batch!(Y,X,k::LocalJumpPIKernel,b,scale,work)
    fill!(Y,zero(eltype(Y)))
    @inbounds for rhs in axes(X,2),index in eachindex(k.gain.V)
        Y[k.gain.I[index],rhs]+=scale*k.gain.V[index]*
            X[k.gain.J[index],rhs]
    end
    Y
end

function _apply_gain_batch!(Y,X,k::FactorizedLocalJumpPIKernel,b,scale,work)
    fill!(Y,zero(eltype(Y)))
    _apply_factorized_onebody_gain_batch!(
        Y,X,k.branches,k.contractions,b,scale,work.batch,1)
    Y
end

function _apply_gain_batch!(
        Y,X,k::FactorizedLocalPBodyJumpPIKernel,b,scale,work)
    fill!(Y,zero(eltype(Y)))
    _apply_factorized_pbody_gain_batch!(
        Y,X,k.groups,k.contractions,k.pair_scales,b,scale,work.batch)
    Y
end

@inline _batched_apply_jumps!(
    X,channels,work,b,tau,::Tuple{},channel)=X

@inline function _batched_apply_jumps!(
        X,channels,work,b,tau,jumps::Tuple{K,Vararg{Any}},
        channel) where K
    columns=size(X,2)
    count=0
    @inbounds for rhs in 1:columns
        if channels[rhs]==channel
            count+=1
            work.selected_indices[count]=rhs
            for coordinate in axes(X,1)
                work.gather[coordinate,count]=X[coordinate,rhs]
            end
        end
    end
    if !iszero(count)
        gathered=@view work.gather[:,1:count]
        gained=@view work.gain[:,1:count]
        _apply_gain_batch!(gained,gathered,first(jumps),b,
            work.jump_scales[channel],work.liouvillian_work)
        R=eltype(work.total_intensities)
        @inbounds for local_index in 1:count
            z=zero(eltype(X))
            for coordinate in axes(X,1)
                z+=conj(tau[coordinate])*gained[coordinate,local_index]
            end
            abs(z)>eps(R)||throw(ArgumentError(
                "selected jump $channel has zero probability in batch column $(work.selected_indices[local_index])"))
            rhs=work.selected_indices[local_index]
            for coordinate in axes(X,1)
                X[coordinate,rhs]=gained[coordinate,local_index]/z
            end
        end
    end
    _batched_apply_jumps!(
        X,channels,work,b,tau,Base.tail(jumps),channel+1)
end

"""
    batched_apply_jumps!(X, channels, plan, time, parameters, workspace)

Apply selected jump gains in-place to columns of `X`. `channels[j] == 0`
means no jump for column `j`; positive values identify a prepared jump
channel. Columns selecting the same channel are gathered, propagated through
one matrix-RHS gain kernel, normalized, and scattered back.

This function does not draw random numbers. Pair it with
[`batched_sample_jumps!`](@ref) so each trajectory index retains its own RNG
stream.
"""
function batched_apply_jumps!(X::AbstractMatrix,
        channels::AbstractVector{<:Integer},plan::BatchedConditionalPlan,
        t,p,work::BatchedConditionalWorkspace)
    _check_batched_conditional_workspace(work,plan)
    _check_batched_trajectory_matrix(X,plan,work,"batched trajectory state")
    columns=size(X,2)
    length(channels)==columns||throw(DimensionMismatch(
        "channels must contain one entry per batch column"))
    eltype(channels)<:Bool&&throw(ArgumentError(
        "selected jump channels must be integer indices, not Bool"))
    njumps=length(plan.trajectory.jumps)
    all(channel->0<=channel<=njumps,channels)||throw(ArgumentError(
        "selected jump channel lies outside 0:$njumps"))
    _invalidate_batched_effective_cache!(work)
    _prepare_batched_effective_jump_blocks!(work,t,p)
    b=plan.trajectory.model.basis
    tau=plan.trajectory.liouvillian.tracevec
    _batched_apply_jumps!(
        X,channels,work,b,tau,plan.trajectory.jumps,1)
end

function batched_apply_jumps!(X::AbstractMatrix,
        channels::AbstractVector{<:Integer},plan::BatchedConditionalPlan,
        work::BatchedConditionalWorkspace)
    _require_autonomous(plan,"batched_apply_jumps!")
    batched_apply_jumps!(X,channels,plan,0.0,nothing,work)
end

"""
    batched_trajectory_rngs(seed, count)

Construct one independent `MersenneTwister` stream per trajectory index using
the same master-seed expansion as [`quantum_trajectories`](@ref). Keep each
stream attached to its trajectory when regrouping columns into common-step
cohorts.
"""
function batched_trajectory_rngs(seed::Integer,count::Integer)
    count isa Bool&&throw(ArgumentError(
        "trajectory RNG count must be an integer, not Bool"))
    count>0||throw(ArgumentError("trajectory RNG count must be positive"))
    BigInt(count)<=typemax(Int)||throw(ArgumentError(
        "trajectory RNG count exceeds the addressable Int range"))
    master=MersenneTwister(seed)
    seeds=rand(master,UInt64,Int(count))
    [MersenneTwister(value) for value in seeds]
end

"""
    batched_sample_jumps!(channels, rates, step, rngs)

Draw one fixed-step jump decision per column from channel intensities `rates`.
`rngs[j]` is used only for trajectory `j`; a no-jump decision consumes one
draw and a jump consumes the same second channel-selection draw as the scalar
algorithm. `channels[j]` is set to zero for no jump.

The supplied `step` is used exactly as given and is never capped. Callers must
first enforce their requested `max_jump_probability` using the pre-step total
intensity, exactly as in the scalar path.
"""
function batched_sample_jumps!(channels::AbstractVector{<:Integer},
        rates::AbstractMatrix,step,rngs::AbstractVector{<:AbstractRNG})
    columns=size(rates,2)
    length(channels)==columns||throw(DimensionMismatch(
        "channels must contain one entry per rate column"))
    eltype(channels)<:Bool&&throw(ArgumentError(
        "jump-decision storage must contain integer indices, not Bool"))
    length(rngs)>=columns||throw(DimensionMismatch(
        "one RNG stream is required per rate column"))
    R=eltype(rates)
    R<:AbstractFloat||throw(ArgumentError(
        "batched jump intensities must use a real floating scalar type"))
    step isa Bool&&throw(ArgumentError(
        "batched trajectory step must be a real number, not Bool"))
    h=_trajectory_real_input(R,step,"batched trajectory step")
    h>zero(R)||throw(ArgumentError("batched trajectory step must be positive"))
    @inbounds for rhs in 1:columns
        column=@view rates[:,rhs]
        for rate in column
            isfinite(rate)||throw(ArgumentError(
                "jump intensity is nonfinite in batch column $rhs"))
            rate>=zero(R)||throw(ArgumentError(
                "jump intensity is negative in batch column $rhs"))
        end
        lambda=_total_intensity(column)
        isfinite(lambda)||throw(ArgumentError(
            "total jump intensity overflows in batch column $rhs; use a " *
            "wider trajectory precision or rescale the channels"))
        probability=-expm1(-lambda*h)
        if lambda>zero(R)&&rand(rngs[rhs],R)<probability
            channels[rhs]=_select_jump_channel(
                column,rand(rngs[rhs],R)*lambda)
        else
            channels[rhs]=0
        end
    end
    channels
end
