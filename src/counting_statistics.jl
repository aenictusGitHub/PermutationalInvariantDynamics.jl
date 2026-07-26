@doc raw"""
    TiltedLiouvillianPlan(source; channels=:all, increments=1)

Prepare the jump-gain maps needed for full counting statistics. `source` may
be an autonomous [`PIModel`](@ref), [`CompiledPIModel`](@ref), or
[`TrajectoryPlan`](@ref). Channel numbers are the jump-channel numbers of the
term-resolved trajectory plan; a local channel counts the sum of identical
local jumps and does not resolve the particle label.

For the moment-generating convention used here, a trajectory with channel
counts `K_a` is weighted by

```math
\exp\!\left(s\sum_a q_a K_a\right),
```

where `q_a` is the corresponding entry of `increments`. The tilted generator
is therefore

```math
\mathcal L_s=\mathcal L+
\sum_a\left(e^{s q_a}-1\right)\mathcal J_a.
```

Only the gain map `J_a[rho] = rate_a * L_a*rho*L_a'` is tilted; its
anticommutator loss is unchanged. Signed real increments are supported.
Time-dependent models must first be frozen at an explicit time. All jump
rates are checked to be finite and nonnegative, as required for a counting
interpretation.

The plan is immutable and shareable. Each concurrent application needs its
own [`TiltedLiouvillianWorkspace`](@ref).
"""
struct TiltedLiouvillianPlan{P,J,C,Q,S}
    trajectory::P
    jumps::J
    channels::C
    increments::Q
    scales::S
end

function _counting_channel_indices(channel_specification,jump_count::Int)
    jump_count>0||throw(ArgumentError(
        "full counting statistics requires at least one jump channel"))
    raw=if channel_specification===:all
        collect(1:jump_count)
    elseif channel_specification isa Integer &&
           !(channel_specification isa Bool)
        [channel_specification]
    else
        try
            collect(channel_specification)
        catch
            throw(ArgumentError(
                "channels must be :all, an integer, or an iterable of integers"))
        end
    end
    isempty(raw)&&throw(ArgumentError(
        "at least one counted jump channel must be selected"))
    indices=Int[]
    sizehint!(indices,length(raw))
    for (position,value) in pairs(raw)
        value isa Integer&&!(value isa Bool)||throw(ArgumentError(
            "counted channel at position $position must be an integer"))
        1<=value<=jump_count||throw(ArgumentError(
            "counted channel $value lies outside 1:$jump_count"))
        index=Int(value)
        push!(indices,index)
    end
    length(unique(indices))==length(indices)||throw(ArgumentError(
        "counted jump channels must be unique"))
    indices
end

function _counting_increments(values,count::Int,::Type{R}) where
        R<:AbstractFloat
    raw=values isa Number ? fill(values,count) : try
        collect(values)
    catch
        throw(ArgumentError(
            "increments must be a real scalar or an iterable of real scalars"))
    end
    length(raw)==count||throw(DimensionMismatch(
        "increments must contain one value per selected jump channel"))
    Tuple(_trajectory_real_input(
        R,value,"count increment at position $position")
        for (position,value) in pairs(raw))
end

@inline _validate_counting_jump_rates(::Tuple{},::Type)=nothing
@inline function _validate_counting_jump_rates(
        jumps::Tuple{K,Vararg{Any}},::Type{R}) where {K,R<:AbstractFloat}
    _trajectory_jump_scale(first(jumps),zero(R),nothing,R)
    _validate_counting_jump_rates(Base.tail(jumps),R)
end

function TiltedLiouvillianPlan(source;
        channels=:all,increments=1)
    trajectory=source isa TrajectoryPlan ? source : TrajectoryPlan(source)
    isautonomous(trajectory)||throw(ArgumentError(
        "full counting statistics currently requires an autonomous model; " *
        "freeze the model at an explicit time and parameters first"))
    R=_real_float_type(eltype(trajectory.liouvillian))
    _validate_counting_jump_rates(trajectory.jumps,R)
    indices=_counting_channel_indices(
        channels,length(trajectory.jumps))
    charges=_counting_increments(increments,length(indices),R)
    selected=Tuple(trajectory.jumps[index] for index in indices)
    scales=Tuple(_trajectory_jump_scale(
        trajectory.jumps[index],zero(R),nothing,R) for index in indices)
    TiltedLiouvillianPlan(
        trajectory,selected,Tuple(indices),charges,scales)
end

Base.size(plan::TiltedLiouvillianPlan)=size(plan.trajectory.liouvillian)
Base.size(plan::TiltedLiouvillianPlan,index::Integer)=
    size(plan.trajectory.liouvillian,index)
Base.eltype(plan::TiltedLiouvillianPlan)=eltype(plan.trajectory.liouvillian)
isautonomous(::TiltedLiouvillianPlan)=true

"""
    TiltedLiouvillianWorkspace(plan)

Task-owned mutable scratch for a [`TiltedLiouvillianPlan`](@ref). It retains
one ordinary Liouvillian workspace and one PI-coordinate gain vector. Reuse it
sequentially; never share one workspace between concurrent tasks.
"""
struct TiltedLiouvillianWorkspace{P,W,V,S}
    plan::P
    liouvillian::W
    gain::V
    gain_scratch::S
end

function TiltedLiouvillianWorkspace(plan::TiltedLiouvillianPlan)
    T=eltype(plan)
    largest=maximum(
        length,plan.trajectory.model.basis.patterns;init=1)
    TiltedLiouvillianWorkspace(
        plan,LiouvillianWorkspace(plan.trajectory.liouvillian),
        zeros(T,size(plan,1)),zeros(T,largest,largest))
end

function _check_tilted_workspace(work::TiltedLiouvillianWorkspace,
                                 plan::TiltedLiouvillianPlan)
    work.plan===plan||throw(ArgumentError(
        "tilted-Liouvillian workspace belongs to a different plan"))
    work
end

function _counting_tilt_factor(field,increment,::Type{T}) where T
    field isa Number||throw(ArgumentError(
        "counting field must be a finite number"))
    isfinite(real(field))&&isfinite(imag(field))||throw(ArgumentError(
        "counting field must be finite"))
    checked_component(component)=begin
        product=component*increment
        isfinite(product)||throw(ArgumentError(
            "the counting-field product is nonfinite; reduce the field or " *
            "increment magnitude, or use wider precision"))
        !iszero(component)&&!iszero(increment)&&iszero(product)&&
            throw(ArgumentError(
                "the counting-field product underflows before exponentiation; " *
                "prepare the model and counting data at wider precision"))
        product
    end
    product=field isa Real ? checked_component(field) :
        complex(checked_component(real(field)),
                checked_component(imag(field)))
    raw=expm1(product)
    isfinite(real(raw))&&isfinite(imag(raw))||throw(ArgumentError(
        "the exponential counting weight is nonfinite; reduce the field " *
        "magnitude or use a wider prepared scalar type"))
    promote_type(T,typeof(raw))===T||throw(ArgumentError(
        "counting field type $(typeof(field)) would widen the prepared " *
        "tilted-Liouvillian scalar type $T"))
    converted=try
        convert(T,raw)
    catch error
        error isa InexactError||error isa OverflowError||rethrow()
        throw(ArgumentError(
            "the exponential counting weight is not representable in $T"))
    end
    !iszero(raw)&&iszero(converted)&&throw(ArgumentError(
        "the exponential counting weight underflows in $T; prepare the " *
        "model at wider precision"))
    converted
end

function _counting_plan_with_budget(source;
        channels=:all,increments=1,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,
        operation::AbstractString="counting-statistics plan preparation")
    source isa TiltedLiouvillianPlan&&return (
        plan=source,hidden_retained_bytes=big(0))
    created_liouvillian=false
    if source isa PIModel
        _require_model_preparation_budget(
            source,memory_budget;operation)
        created_liouvillian=true
    elseif source isa CompiledPIModel
        kernels=source.plan.kernels
        created_liouvillian=kernels!==nothing&&
            any(kernel->kernel isa FusedStaticPIKernel,kernels)
        created_liouvillian&&
            _require_model_preparation_budget(
                source.model,memory_budget;operation)
    end
    plan=TiltedLiouvillianPlan(source;channels,increments)
    hidden=BigInt(Base.summarysize(plan.trajectory.trace_weights))+
        BigInt(Base.summarysize(plan.channels))+
        BigInt(Base.summarysize(plan.increments))+
        BigInt(Base.summarysize(plan.scales))+512
    created_liouvillian&&
        (hidden+=BigInt(Base.summarysize(plan.trajectory.liouvillian)))
    (plan,hidden_retained_bytes=hidden)
end

function _counting_nested_budget(
        memory_budget,hidden_retained_bytes::Integer,
        operation::AbstractString)
    _require_performance_budget(
        operation,hidden_retained_bytes,memory_budget;
        guidance="Prepare and retain TiltedLiouvillianPlan explicitly, " *
                 "or raise memory_budget.")
    limit=_performance_memory_limit(memory_budget)
    limit===nothing ? Inf : limit-BigInt(hidden_retained_bytes)
end

@inline _counting_tilt_factors(::Tuple{},field,::Type)=()
@inline function _counting_tilt_factors(
        increments::Tuple{Q,Vararg{Any}},field,::Type{T}) where {Q,T}
    (_counting_tilt_factor(field,first(increments),T),
     _counting_tilt_factors(Base.tail(increments),field,T)...)
end

@inline _add_tilted_gains!(y,x,::Tuple{},::Tuple{},::Tuple{},
                           b,work)=nothing

_apply_counting_gain!(destination,source,kernel,basis,scale,work)=
    _apply_gain!(
        destination,source,kernel,basis,scale,work.liouvillian)

function _apply_counting_gain!(destination,source,
        kernel::FactorizedLocalJumpPIKernel,basis,scale,work)
    fill!(destination,zero(eltype(destination)))
    # The ordinary Liouvillian action immediately preceding this call packed
    # every immutable source block once. Reuse those blocks for all counted
    # gains instead of repacking per selected channel.
    _apply_factorized_onebody_gain!(
        destination,kernel.branches,kernel.contractions,
        work.gain_scratch,basis,scale,work.liouvillian.blocks,1)
    destination
end

function _apply_counting_gain!(destination,source,
        kernel::FactorizedLocalPBodyJumpPIKernel,basis,scale,work)
    fill!(destination,zero(eltype(destination)))
    _apply_factorized_pbody_gain!(
        destination,kernel.groups,kernel.contractions,kernel.pair_scales,
        work.gain_scratch,basis,scale,work.liouvillian.blocks)
    destination
end

@inline function _add_tilted_gains!(y,x,
        jumps::Tuple{K,Vararg{Any}},
        scales::Tuple{S,Vararg{Any}},
        factors::Tuple{F,Vararg{Any}},b,work) where {K,S,F}
    factor=first(factors)
    if !iszero(factor)
        _apply_counting_gain!(
            work.gain,x,first(jumps),b,first(scales),work)
        @inbounds @simd for index in eachindex(y,work.gain)
            y[index]+=factor*work.gain[index]
        end
    end
    _add_tilted_gains!(
        y,x,Base.tail(jumps),Base.tail(scales),Base.tail(factors),b,work)
end

function _add_counting_gain_batch!(destination,source,
        kernel::DissipatorPIKernel,basis,scale,scratch;adjoint::Bool=false)
    for sector in eachindex(basis.sectors)
        dimension=length(basis.patterns[sector])
        offset=basis.offsets[sector]
        block=adjoint ? LinearAlgebra.adjoint(kernel.blocks[sector]) :
                        kernel.blocks[sector]
        right=adjoint ? kernel.blocks[sector] :
                        LinearAlgebra.adjoint(kernel.blocks[sector])
        _batch_add_sandwich!(
            destination,source,offset,dimension,block,right,scale,scratch)
    end
    destination
end

function _add_counting_gain_batch!(destination,source,
        kernel::LocalJumpPIKernel,basis,scale,scratch;adjoint::Bool=false)
    if adjoint
        @inbounds for rhs in axes(source,2),index in eachindex(kernel.gain.V)
            destination[kernel.gain.J[index],rhs]+=
                scale*conj(kernel.gain.V[index])*
                source[kernel.gain.I[index],rhs]
        end
    else
        @inbounds for rhs in axes(source,2),index in eachindex(kernel.gain.V)
            destination[kernel.gain.I[index],rhs]+=
                scale*kernel.gain.V[index]*
                source[kernel.gain.J[index],rhs]
        end
    end
    destination
end

function _add_counting_gain_batch!(destination,source,
        kernel::FactorizedLocalJumpPIKernel,basis,scale,scratch;
        adjoint::Bool=false)
    _apply_factorized_onebody_gain_batch!(
        destination,source,kernel.branches,kernel.contractions,basis,scale,
        scratch,1;adjoint)
    destination
end

function _add_counting_gain_batch!(destination,source,
        kernel::FactorizedLocalPBodyJumpPIKernel,basis,scale,scratch;
        adjoint::Bool=false)
    _apply_factorized_pbody_gain_batch!(
        destination,source,kernel.groups,kernel.contractions,
        kernel.pair_scales,basis,scale,scratch;adjoint)
    destination
end

@inline _add_tilted_gains_batch!(
    y,x,::Tuple{},::Tuple{},::Tuple{},b,work;adjoint::Bool=false)=nothing
@inline function _add_tilted_gains_batch!(y,x,
        jumps::Tuple{K,Vararg{Any}},
        scales::Tuple{S,Vararg{Any}},
        factors::Tuple{F,Vararg{Any}},b,work;
        adjoint::Bool=false) where {K,S,F}
    factor=adjoint ? conj(first(factors)) : first(factors)
    if !iszero(factor)
        _add_counting_gain_batch!(
            y,x,first(jumps),b,factor*first(scales),
            work.liouvillian.batch;adjoint)
    end
    _add_tilted_gains_batch!(
        y,x,Base.tail(jumps),Base.tail(scales),Base.tail(factors),b,work;
        adjoint)
end

"""
    apply_tilted!(destination, plan, source, field, workspace)

Apply the tilted generator in PI coordinates without materializing it. Source
and destination must not alias. For repeated applications at one field, use
[`tilted_liouvillian`](@ref), which prepares the exponential channel factors
once.
"""
function apply_tilted!(destination::AbstractVector,
        plan::TiltedLiouvillianPlan,source::AbstractVector,field,
        work::TiltedLiouvillianWorkspace)
    _check_tilted_workspace(work,plan)
    factors=_counting_tilt_factors(plan.increments,field,eltype(plan))
    apply!(destination,plan.trajectory.liouvillian,source,
           work.liouvillian)
    _add_tilted_gains!(
        destination,source,plan.jumps,plan.scales,factors,
        plan.trajectory.model.basis,work)
    destination
end

function apply_tilted!(destination::AbstractMatrix,
        plan::TiltedLiouvillianPlan,source::AbstractMatrix,field,
        work::TiltedLiouvillianWorkspace)
    _check_tilted_workspace(work,plan)
    factors=_counting_tilt_factors(plan.increments,field,eltype(plan))
    apply!(destination,plan.trajectory.liouvillian,source,
           work.liouvillian)
    _add_tilted_gains_batch!(
        destination,source,plan.jumps,plan.scales,factors,
        plan.trajectory.model.basis,work)
    destination
end

function _apply_gain_adjoint!(destination,source,
        kernel::DissipatorPIKernel,basis,scale,work)
    fill!(destination,zero(eltype(destination)))
    for sector in eachindex(basis.sectors)
        dimension=length(basis.patterns[sector])
        range=basis.offsets[sector]:basis.offsets[sector+1]-1
        input=reshape(view(source,range),dimension,dimension)
        output=reshape(view(destination,range),dimension,dimension)
        scratch=work.liouvillian.blocks[sector][1]
        block=kernel.blocks[sector]
        mul!(scratch,adjoint(block),input)
        mul!(output,scratch,block)
        output .*= scale
    end
    destination
end

function _apply_gain_adjoint!(destination,source,
        kernel::LocalJumpPIKernel,basis,scale,work)
    fill!(destination,zero(eltype(destination)))
    @inbounds for index in eachindex(kernel.gain.V)
        destination[kernel.gain.J[index]]+=
            scale*conj(kernel.gain.V[index])*source[kernel.gain.I[index]]
    end
    destination
end

function _apply_gain_adjoint!(destination,source,
        kernel::FactorizedLocalJumpPIKernel,basis,scale,work)
    fill!(destination,zero(eltype(destination)))
    _apply_factorized_onebody_gain!(
        destination,kernel.branches,kernel.contractions,
        work.gain_scratch,basis,scale,work.liouvillian.blocks,1;
        adjoint=true)
    destination
end

function _apply_gain_adjoint!(destination,source,
        kernel::FactorizedLocalPBodyJumpPIKernel,basis,scale,work)
    fill!(destination,zero(eltype(destination)))
    _apply_adjoint_factorized_pbody_gain!(
        destination,kernel.groups,kernel.contractions,kernel.pair_scales,
        work.gain_scratch,basis,scale,work.liouvillian.blocks)
    destination
end

@inline _add_tilted_adjoint_gains!(
    y,x,::Tuple{},::Tuple{},::Tuple{},b,work)=nothing
@inline function _add_tilted_adjoint_gains!(y,x,
        jumps::Tuple{K,Vararg{Any}},
        scales::Tuple{S,Vararg{Any}},
        factors::Tuple{F,Vararg{Any}},b,work) where {K,S,F}
    factor=first(factors)
    if !iszero(factor)
        _apply_gain_adjoint!(
            work.gain,x,first(jumps),b,first(scales),work)
        conjugate_factor=conj(factor)
        @inbounds @simd for index in eachindex(y,work.gain)
            y[index]+=conjugate_factor*work.gain[index]
        end
    end
    _add_tilted_adjoint_gains!(
        y,x,Base.tail(jumps),Base.tail(scales),Base.tail(factors),b,work)
end

"""
    apply_tilted_adjoint!(destination, plan, source, field, workspace)

Apply the Frobenius adjoint of a tilted PI generator with caller-owned
scratch.
"""
function apply_tilted_adjoint!(destination::AbstractVector,
        plan::TiltedLiouvillianPlan,source::AbstractVector,field,
        work::TiltedLiouvillianWorkspace)
    _check_tilted_workspace(work,plan)
    factors=_counting_tilt_factors(plan.increments,field,eltype(plan))
    apply_adjoint!(
        destination,plan.trajectory.liouvillian,source,work.liouvillian)
    _add_tilted_adjoint_gains!(
        destination,source,plan.jumps,plan.scales,factors,
        plan.trajectory.model.basis,work)
    destination
end

function apply_tilted_adjoint!(destination::AbstractMatrix,
        plan::TiltedLiouvillianPlan,source::AbstractMatrix,field,
        work::TiltedLiouvillianWorkspace)
    _check_tilted_workspace(work,plan)
    factors=_counting_tilt_factors(plan.increments,field,eltype(plan))
    apply_adjoint!(
        destination,plan.trajectory.liouvillian,source,work.liouvillian)
    _add_tilted_gains_batch!(
        destination,source,plan.jumps,plan.scales,factors,
        plan.trajectory.model.basis,work;adjoint=true)
    destination
end

"""
    TiltedLiouvillian

Autonomous matrix-free tilted-generator adapter. Construct it with
[`tilted_liouvillian`](@ref). The adapter owns and synchronizes one workspace;
parallel hot loops should share its immutable plan and use separate
[`TiltedLiouvillianWorkspace`](@ref) objects instead.
"""
struct TiltedLiouvillian{P,F,C,W,K}
    plan::P
    field::F
    factors::C
    workspace::W
    lock::K
end

"""
    tilted_liouvillian(plan, field; workspace=nothing)

Return an autonomous matrix-free operator for the counting-field generator
`L_field`. The compatibility operator owns and synchronizes one workspace.
Parallel Krylov or custom hot loops should instead call
[`apply_tilted!`](@ref) with one explicit workspace per task.
"""
function tilted_liouvillian(plan::TiltedLiouvillianPlan,field;
                            workspace=nothing)
    factors=_counting_tilt_factors(plan.increments,field,eltype(plan))
    work=workspace===nothing ? TiltedLiouvillianWorkspace(plan) :
        _check_tilted_workspace(workspace,plan)
    TiltedLiouvillian(plan,field,factors,work,ReentrantLock())
end

Base.size(operator::TiltedLiouvillian)=size(operator.plan)
Base.size(operator::TiltedLiouvillian,index::Integer)=
    size(operator.plan,index)
Base.eltype(operator::TiltedLiouvillian)=eltype(operator.plan)
isautonomous(::TiltedLiouvillian)=true

function LinearAlgebra.mul!(destination::AbstractVector,
        operator::TiltedLiouvillian,source::AbstractVector)
    lock(operator.lock)
    try
        apply!(
            destination,operator.plan.trajectory.liouvillian,source,
            operator.workspace.liouvillian)
        _add_tilted_gains!(
            destination,source,operator.plan.jumps,operator.plan.scales,
            operator.factors,operator.plan.trajectory.model.basis,
            operator.workspace)
    finally
        unlock(operator.lock)
    end
    destination
end

function LinearAlgebra.mul!(destination::AbstractMatrix,
        operator::TiltedLiouvillian,source::AbstractMatrix)
    size(source,1)==size(operator,2)||throw(DimensionMismatch(
        "tilted-Liouvillian matrix input has the wrong leading dimension"))
    size(destination)==(size(operator,1),size(source,2))||
        throw(DimensionMismatch(
            "tilted-Liouvillian matrix output has the wrong dimensions"))
    Base.mightalias(destination,source)&&throw(ArgumentError(
        "tilted-Liouvillian source and destination must not alias"))
    lock(operator.lock)
    try
        apply!(
            destination,operator.plan.trajectory.liouvillian,source,
            operator.workspace.liouvillian)
        _add_tilted_gains_batch!(
            destination,source,operator.plan.jumps,operator.plan.scales,
            operator.factors,operator.plan.trajectory.model.basis,
            operator.workspace)
    finally
        unlock(operator.lock)
    end
    destination
end

Base.:*(operator::TiltedLiouvillian,source::AbstractVector)=
    mul!(_product_destination(
        operator,source,size(operator,1)),operator,source)
Base.:*(operator::TiltedLiouvillian,source::AbstractMatrix)=
    mul!(_product_destination(
        operator,source,size(operator,1),size(source,2)),operator,source)

function apply_adjoint!(destination::AbstractVector,
        operator::TiltedLiouvillian,source::AbstractVector)
    lock(operator.lock)
    try
        apply_adjoint!(
            destination,operator.plan.trajectory.liouvillian,source,
            operator.workspace.liouvillian)
        _add_tilted_adjoint_gains!(
            destination,source,operator.plan.jumps,operator.plan.scales,
            operator.factors,operator.plan.trajectory.model.basis,
            operator.workspace)
    finally
        unlock(operator.lock)
    end
    destination
end

function apply_adjoint!(destination::AbstractMatrix,
        operator::TiltedLiouvillian,source::AbstractMatrix)
    lock(operator.lock)
    try
        apply_adjoint!(
            destination,operator.plan.trajectory.liouvillian,source,
            operator.workspace.liouvillian)
        _add_tilted_gains_batch!(
            destination,source,operator.plan.jumps,operator.plan.scales,
            operator.factors,operator.plan.trajectory.model.basis,
            operator.workspace;adjoint=true)
    finally
        unlock(operator.lock)
    end
    destination
end

struct AdjointTiltedLiouvillian{L}
    parent::L
end
Base.size(operator::AdjointTiltedLiouvillian)=reverse(size(operator.parent))
Base.size(operator::AdjointTiltedLiouvillian,index::Integer)=
    size(operator.parent,3-index)
Base.eltype(operator::AdjointTiltedLiouvillian)=eltype(operator.parent)
isautonomous(::AdjointTiltedLiouvillian)=true
Base.adjoint(operator::TiltedLiouvillian)=
    AdjointTiltedLiouvillian(operator)
Base.adjoint(operator::AdjointTiltedLiouvillian)=operator.parent
LinearAlgebra.mul!(destination::AbstractVector,
        operator::AdjointTiltedLiouvillian,source::AbstractVector)=
    apply_adjoint!(destination,operator.parent,source)
LinearAlgebra.mul!(destination::AbstractMatrix,
        operator::AdjointTiltedLiouvillian,source::AbstractMatrix)=
    apply_adjoint!(destination,operator.parent,source)
Base.:*(operator::AdjointTiltedLiouvillian,source::AbstractVector)=
    mul!(_product_destination(
        operator,source,size(operator,1)),operator,source)
Base.:*(operator::AdjointTiltedLiouvillian,source::AbstractMatrix)=
    mul!(_product_destination(
        operator,source,size(operator,1),size(source,2)),operator,source)

# Source-protocol extensions. A tilted generator is not trace preserving away
# from zero field, but this is still its physical trace readout for finite-time
# moment-generating functions.
_operator_basis(plan::TiltedLiouvillianPlan)=
    plan.trajectory.model.basis
_operator_basis(operator::TiltedLiouvillian)=
    _operator_basis(operator.plan)
_operator_trace_functional(plan::TiltedLiouvillianPlan)=
    plan.trajectory.liouvillian.tracevec
_operator_trace_functional(operator::TiltedLiouvillian)=
    _operator_trace_functional(operator.plan)
_operator_has_adjoint(::TiltedLiouvillian)=true
_operator_requires_matrixfree(::Union{
    TiltedLiouvillianPlan,TiltedLiouvillian})=true
_linear_operator_workspace(plan::TiltedLiouvillianPlan)=
    TiltedLiouvillianWorkspace(plan)
_linear_operator_workspace(operator::TiltedLiouvillian)=
    TiltedLiouvillianWorkspace(operator.plan)
function _linear_operator_batch_workspace(
        plan::TiltedLiouvillianPlan,columns::Integer,::Type{T}) where T
    columns>=0||throw(ArgumentError(
        "batch column count must be nonnegative"))
    promote_type(eltype(plan),T)===eltype(plan)||throw(ArgumentError(
        "batch scalar type $T would widen tilted-Liouvillian precision " *
        "$(eltype(plan))"))
    work=TiltedLiouvillianWorkspace(plan)
    _ensure_batch_capacity!(work.liouvillian.batch,columns)
    work
end
_linear_operator_batch_workspace(
    operator::TiltedLiouvillian,columns::Integer,::Type{T}) where T=
    _linear_operator_batch_workspace(operator.plan,columns,T)

function _performance_tilted_workspace_bytes(
        plan::TiltedLiouvillianPlan;
        batch_columns::Integer=0,
        bigfloat_precision::Integer=precision(BigFloat))
    batch_columns>=0||throw(ArgumentError(
        "batch_columns must be nonnegative"))
    largest=BigInt(maximum(
        length,plan.trajectory.model.basis.patterns;init=1))
    _performance_liouvillian_workspace_bytes(
        plan.trajectory.liouvillian;batch_columns,bigfloat_precision)+
    _performance_entries_bytes(
        BigInt(size(plan,1))+largest^2,eltype(plan);bigfloat_precision)
end

_performance_linear_operator_workspace_bytes(
    plan::TiltedLiouvillianPlan;batch_columns::Integer=0)=begin
    _performance_tilted_workspace_bytes(plan;batch_columns)
end
_performance_linear_operator_workspace_bytes(
    operator::TiltedLiouvillian;batch_columns::Integer=0)=
    _performance_linear_operator_workspace_bytes(
        operator.plan;batch_columns)
_performance_source_action_bytes(
    plan::TiltedLiouvillianPlan,::Type{T}) where T=
    _performance_tilted_workspace_bytes(plan)
_performance_source_action_bytes(
    operator::TiltedLiouvillian,::Type{T}) where T=
    _performance_tilted_workspace_bytes(operator.plan)
function _performance_batched_workspace_growth_bytes(
        work::TiltedLiouvillianWorkspace,batch_columns::Integer)
    _performance_batched_workspace_growth_bytes(
        work.liouvillian,batch_columns)
end
_performance_batched_action_growth_bytes(
    operator::TiltedLiouvillian,batch_columns::Integer)=
    _performance_batched_workspace_growth_bytes(
        operator.workspace,batch_columns)

function _counting_real_field(field,::Type{R}) where R<:AbstractFloat
    field isa Real&&!(field isa Bool)||throw(ArgumentError(
        "the scaled cumulant-generating function requires a real counting field"))
    _trajectory_real_input(R,field,"counting field")
end

function _counting_tolerance(value,label,::Type{R}) where R<:AbstractFloat
    value isa Real&&!(value isa Bool)||throw(ArgumentError(
        "$label must be a real number"))
    converted=_trajectory_real_input(R,value,label)
    converted>=zero(R)||throw(ArgumentError(
        "$label must be nonnegative"))
    converted
end

"""
    counting_scgf(plan, field; krylovdim=40, return_info=false, ...)

Return the scaled cumulant-generating function, the eigenvalue of the tilted
generator with largest real part. The counting field must be real.
`return_info=true` returns the raw complex Ritz value, residual, convergence
flags, and channel convention. No PI-coordinate matrix is materialized.
"""
function counting_scgf(plan::TiltedLiouvillianPlan,field;
        krylovdim::Integer=min(size(plan,1),40),
        atol::Real=100eps(_real_float_type(eltype(plan))),
        rtol::Real=100eps(_real_float_type(eltype(plan))),
        require_convergence::Bool=true,
        return_info::Bool=false,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,
        rng=Random.MersenneTwister(0),kwargs...)
    R=_real_float_type(eltype(plan))
    counting_field=_counting_real_field(field,R)
    absolute_tolerance=_counting_tolerance(atol,"atol",R)
    relative_tolerance=_counting_tolerance(rtol,"rtol",R)
    if iszero(counting_field)
        value=zero(eltype(plan))
        info=(value,field=counting_field,residual=zero(R),converged=true,
              exact_zero=true,operator_applications=0,
              channels=plan.channels,increments=plan.increments,
              convention=:moment_generating)
        return return_info ? info : value
    end
    # Validate the field and guard the Arnoldi plus tilted-application
    # workspaces before constructing the compatibility operator, which owns
    # the dominant application scratch.
    _counting_tilt_factors(
        plan.increments,counting_field,eltype(plan))
    _guard_selected_spectrum_workspace(
        plan,:arnoldi,krylovdim,1,false,memory_budget;kwargs...)
    operator=tilted_liouvillian(plan,counting_field)
    spectrum=krylov_liouvillian_spectrum(
        operator;nev=1,krylovdim,which=:LR,
        atol=absolute_tolerance,rtol=relative_tolerance,
        require_convergence,rng,kwargs...)
    value=only(spectrum.values)
    info=(
        value,field=counting_field,residual=only(spectrum.residuals),
        converged=only(spectrum.converged),exact_zero=false,
        operator_applications=spectrum.iterations,
        channels=plan.channels,increments=plan.increments,
        convention=:moment_generating,spectrum)
    return_info ? info : value
end

function counting_scgf(source,field;channels=:all,increments=1,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,kwargs...)
    prepared=_counting_plan_with_budget(
        source;channels,increments,memory_budget,
        operation="counting-statistics SCGF model preparation")
    solve_budget=_counting_nested_budget(
        memory_budget,prepared.hidden_retained_bytes,
        "counting-statistics hidden plan retention")
    counting_scgf(prepared.plan,field;memory_budget=solve_budget,kwargs...)
end

function _counting_real_scgf(info,atol,rtol)
    value=info.value
    R=_real_float_type(typeof(value))
    tolerance=R(atol)+R(rtol)*max(abs(real(value)),floatmin(R))+
              R(10)*R(info.residual)
    abs(imag(value))<=tolerance||throw(ArgumentError(
        "the dominant tilted eigenvalue has imaginary part $(imag(value)), " *
        "larger than the spectral reliability scale $tolerance; increase " *
        "krylovdim or tighten the spectral tolerances"))
    real(value),abs(imag(value))
end

"""
    counting_cumulants(plan; step=nothing, ...)

Estimate the stationary current and zero-frequency noise (the first two
cumulants per unit time) from Richardson-extrapolated centered derivatives of
the SCGF at zero. Results include the coarse/fine discrepancy as a
finite-difference error estimate and all four nonzero-field spectral reports.

The convention is `theta(s) = lim(t->Inf) log(E[exp(s*K_t)])/t`, so
`current = theta'(0)` and `noise = theta''(0)`.
"""
function counting_cumulants(plan::TiltedLiouvillianPlan;
        step=nothing,krylovdim::Integer=min(size(plan,1),40),
        atol::Real=100eps(_real_float_type(eltype(plan))),
        rtol::Real=100eps(_real_float_type(eltype(plan))),
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,
        rng=Random.MersenneTwister(0),kwargs...)
    R=_real_float_type(eltype(plan))
    absolute_tolerance=_counting_tolerance(atol,"atol",R)
    relative_tolerance=_counting_tolerance(rtol,"rtol",R)
    h=step===nothing ? max(R(1e-3),sqrt(sqrt(eps(R)))) :
        _trajectory_real_input(R,step,"finite-difference step")
    h>zero(R)||throw(ArgumentError(
        "finite-difference step must be positive"))
    fields=(h,-h,h/2,-h/2)
    estimate=_selected_spectrum_workspace_bytes(
        plan,:arnoldi,krylovdim,1;kwargs...)+
        _counting_curve_output_bytes(length(fields),eltype(plan))
    _require_performance_budget(
        "counting-cumulant solver and retained reports",estimate,
        memory_budget;guidance="Reduce krylovdim or raise memory_budget.")
    reports=map(fields) do field
        counting_scgf(
            plan,field;krylovdim,atol=absolute_tolerance,
            rtol=relative_tolerance,require_convergence=true,
            return_info=true,memory_budget,rng,kwargs...)
    end
    real_values=map(report->_counting_real_scgf(
        report,absolute_tolerance,relative_tolerance),reports)
    vp,vm,vph,vmh=first.(real_values)
    first_coarse=(vp-vm)/(2h)
    first_fine=(vph-vmh)/h
    second_coarse=(vp+vm)/(h*h)
    second_fine=4(vph+vmh)/(h*h)
    current=first_fine+(first_fine-first_coarse)/3
    noise=second_fine+(second_fine-second_coarse)/3
    current_error=abs(current-first_fine)
    noise_error=abs(noise-second_fine)
    fano=iszero(current) ? nothing : noise/current
    (current,noise,fano,step=h,current_error,noise_error,
     maximum_imaginary_leakage=maximum(last,real_values),
     coarse=(current=first_coarse,noise=second_coarse),
     fine=(current=first_fine,noise=second_fine),
     fields,reports,channels=plan.channels,increments=plan.increments,
     convention=:cumulants_per_unit_time)
end

function counting_cumulants(source;channels=:all,increments=1,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,kwargs...)
    prepared=_counting_plan_with_budget(
        source;channels,increments,memory_budget,
        operation="counting-cumulant model preparation")
    solve_budget=_counting_nested_budget(
        memory_budget,prepared.hidden_retained_bytes,
        "counting-cumulant hidden plan retention")
    counting_cumulants(
        prepared.plan;memory_budget=solve_budget,kwargs...)
end

function _counting_known_length(values)
    iterator_size=Base.IteratorSize(typeof(values))
    iterator_size isa Union{Base.HasLength,Base.HasShape} ?
        length(values) : nothing
end

function _counting_curve_output_bytes(count::Integer,::Type{T}) where T
    count>=0||throw(ArgumentError(
        "SCGF field-grid length must be nonnegative"))
    # Retained fields, values, residuals, the report vector, and the small
    # one-Ritz-value arrays nested in each report. The deliberately
    # conservative factor also covers object/reference headers without relying
    # on allocator-specific sizes.
    _performance_entries_bytes(64BigInt(count),T)
end

function _counting_curve_peak_bytes(
        plan::TiltedLiouvillianPlan,count::Integer,krylovdim::Integer;
        kwargs...)
    _selected_spectrum_workspace_bytes(
        plan,:arnoldi,krylovdim,1;kwargs...)+
        _counting_curve_output_bytes(count,eltype(plan))
end

function _counting_collect_curve_fields(
        fields,plan::TiltedLiouvillianPlan,krylovdim::Integer,
        memory_budget;kwargs...)
    known=_counting_known_length(fields)
    if known!==nothing
        known>0||throw(ArgumentError(
            "the SCGF field grid must be nonempty"))
        estimate=_counting_curve_peak_bytes(
            plan,known,krylovdim;kwargs...)
        _require_performance_budget(
            "SCGF curve solver and retained output",estimate,memory_budget;
            guidance="Reduce the field count or krylovdim, or raise memory_budget.")
        return collect(fields)
    end
    raw=Any[]
    for value in fields
        count=length(raw)+1
        estimate=_counting_curve_peak_bytes(
            plan,count,krylovdim;kwargs...)
        _require_performance_budget(
            "SCGF curve solver and retained output",estimate,memory_budget;
            guidance="Reduce the field count or krylovdim, or raise memory_budget.")
        push!(raw,value)
    end
    isempty(raw)&&throw(ArgumentError(
        "the SCGF field grid must be nonempty"))
    raw
end

"""
    counting_scgf_curve(plan, fields; ...)

Evaluate the dominant tilted eigenvalue on a strictly increasing real field
grid. The returned values remain complex so numerical branch leakage is never
silently discarded.
"""
function counting_scgf_curve(plan::TiltedLiouvillianPlan,fields;
        krylovdim::Integer=min(size(plan,1),40),
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,
        kwargs...)
    raw=_counting_collect_curve_fields(
        fields,plan,krylovdim,memory_budget;kwargs...)
    R=_real_float_type(eltype(plan))
    prepared=R[_counting_real_field(value,R) for value in raw]
    all(diff(prepared).>zero(R))||throw(ArgumentError(
        "SCGF fields must be strictly increasing"))
    reports=[counting_scgf(
        plan,value;krylovdim,return_info=true,memory_budget,kwargs...)
             for value in prepared]
    (fields=prepared,values=getproperty.(reports,:value),
     residuals=getproperty.(reports,:residual),reports,
     channels=plan.channels,increments=plan.increments,
     convention=:moment_generating)
end

function counting_scgf_curve(source,fields;
        channels=:all,increments=1,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,kwargs...)
    prepared=_counting_plan_with_budget(
        source;channels,increments,memory_budget,
        operation="SCGF curve model preparation")
    solve_budget=_counting_nested_budget(
        memory_budget,prepared.hidden_retained_bytes,
        "SCGF curve hidden plan retention")
    counting_scgf_curve(
        prepared.plan,fields;memory_budget=solve_budget,kwargs...)
end

"""
    large_deviation_rate_function(curve, currents; atol=..., rtol=...)

Compute the discrete Legendre--Fenchel estimate
`I(j) = maximum_s (s*j - theta(s))` from a result of
[`counting_scgf_curve`](@ref). A maximum at either field-grid boundary is
reported in `boundary_maxima`; such a point is not certified as a resolved
continuous-field supremum.
"""
function large_deviation_rate_function(curve::NamedTuple,currents;
        atol::Real=1e-12,rtol::Real=1e-9)
    haskey(curve,:fields)&&haskey(curve,:values)||throw(ArgumentError(
        "curve must contain fields and values"))
    fields=curve.fields
    values=curve.values
    length(fields)==length(values)>0||throw(DimensionMismatch(
        "SCGF fields and values must have equal nonzero length"))
    R=promote_type(eltype(fields),_real_float_type(eltype(values)))
    absolute_tolerance=_counting_tolerance(atol,"atol",R)
    relative_tolerance=_counting_tolerance(rtol,"rtol",R)
    theta=Vector{R}(undef,length(values))
    for index in eachindex(values)
        value=values[index]
        residual=haskey(curve,:residuals) ? curve.residuals[index] : zero(R)
        tolerance=absolute_tolerance+
                  relative_tolerance*max(abs(real(value)),floatmin(R))+
                  R(10)*R(residual)
        abs(imag(value))<=tolerance||throw(ArgumentError(
            "SCGF value $index has unresolved imaginary part $(imag(value))"))
        theta[index]=real(value)
    end
    requested=R[_trajectory_real_input(
        R,value,"current at position $index")
        for (index,value) in pairs(collect(currents))]
    rates=Vector{R}(undef,length(requested))
    maximizing_fields=similar(rates)
    boundary=BitVector(undef,length(requested))
    for index in eachindex(requested)
        current=requested[index]
        best=firstindex(fields)
        best_value=fields[best]*current-theta[best]
        for field_index in (firstindex(fields)+1):lastindex(fields)
            candidate=fields[field_index]*current-theta[field_index]
            if candidate>best_value
                best=field_index
                best_value=candidate
            end
        end
        rates[index]=best_value
        maximizing_fields[index]=fields[best]
        boundary[index]=best==firstindex(fields)||best==lastindex(fields)
    end
    (currents=requested,rates,maximizing_fields,
     boundary_maxima=boundary,grid_resolved=.!boundary,
     fields=copy(fields),scgf=theta,
     convention=:discrete_legendre_fenchel)
end

function _counting_check_expv_workspace(
        work,n::Integer,::Type{T}) where T
    work isa KrylovExpvWorkspace||throw(ArgumentError(
        "expv_workspace must be a KrylovExpvWorkspace"))
    m=size(work.H,2)
    0<m<=n||throw(DimensionMismatch(
        "exponential-action workspace has an invalid Krylov dimension"))
    size(work.V)==(n,m+1)||throw(DimensionMismatch(
        "exponential-action workspace V has incompatible dimensions"))
    size(work.H)==(m+1,m)||throw(DimensionMismatch(
        "exponential-action workspace H has incompatible dimensions"))
    size(work.small)==(m+1,m+1)||throw(DimensionMismatch(
        "exponential-action workspace small matrix has incompatible dimensions"))
    all(length(vector)==n for vector in
        (work.w,work.current,work.trial))||throw(DimensionMismatch(
        "exponential-action workspace vectors have incompatible lengths"))
    S=eltype(work.V)
    promote_type(S,T)===S||throw(ArgumentError(
        "exponential-action workspace scalar type $S cannot represent $T"))
    m
end

function _counting_tilted_workspace_peak_bytes(
        plan::TiltedLiouvillianPlan,work)
    if work===nothing
        return _performance_tilted_workspace_bytes(plan)
    end
    _check_tilted_workspace(work,plan)
    _performance_tilted_workspace_bytes(
        plan;batch_columns=work.liouvillian.batch.capacity)
end

function _counting_finite_time_mgf_peak_bytes(
        plan::TiltedLiouvillianPlan,work,n::Integer,::Type{T},
        krylovdim::Integer,return_info::Bool) where T
    _performance_krylov_expv_workspace_bytes(n,T,krylovdim)+
        _counting_tilted_workspace_peak_bytes(plan,work)+
        _performance_entries_bytes(
            BigInt(n)*(return_info ? 2 : 1),T)
end

"""
    finite_time_mgf(plan, rho0, time, field; ...)

Compute `tr(exp(time*L_field)*rho0)` by an adaptive matrix-free Krylov
exponential action. The returned scalar is the finite-time moment-generating
function. Set `return_info=true` to also retain the unnormalized tilted state,
its logarithm, and Krylov convergence diagnostics.
"""
function finite_time_mgf(plan::TiltedLiouvillianPlan,rho0::PIState,
        time::Real,field;krylovdim::Integer=min(size(plan,1),30),
        workspace=nothing,expv_workspace=nothing,
        check::Bool=true,return_info::Bool=false,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,kwargs...)
    rho0.basis===plan.trajectory.model.basis||throw(ArgumentError(
        "initial state and tilted-Liouvillian plan use different PI bases"))
    check&&validate_state(rho0)
    _check_liouvillian_source_precision(
        plan.trajectory.liouvillian,eltype(rho0.data),
        "finite-time counting state")
    R=_real_float_type(eltype(plan))
    prepared_time=_trajectory_real_input(R,time,"counting time")
    prepared_time>=zero(R)||throw(ArgumentError(
        "counting time must be nonnegative"))
    T=promote_type(eltype(plan),eltype(rho0.data))
    n=size(plan,1)
    factors=_counting_tilt_factors(plan.increments,field,eltype(plan))
    checked_work=workspace===nothing ? nothing :
        _check_tilted_workspace(workspace,plan)
    krylov_type=T
    effective_krylovdim=krylovdim
    if expv_workspace!==nothing
        effective_krylovdim=_counting_check_expv_workspace(
            expv_workspace,n,T)
        krylov_type=eltype(expv_workspace.V)
    end
    estimate=_counting_finite_time_mgf_peak_bytes(
        plan,checked_work,n,krylov_type,effective_krylovdim,return_info)
    _require_performance_budget(
        "finite-time counting-statistics Krylov action",estimate,
        memory_budget;guidance="Reduce krylovdim or raise memory_budget.")
    work=checked_work===nothing ? TiltedLiouvillianWorkspace(plan) :
        checked_work
    operator=TiltedLiouvillian(
        plan,field,factors,work,ReentrantLock())
    krylov=expv_workspace===nothing ?
        KrylovExpvWorkspace(T,n,krylovdim) : expv_workspace
    exponential=krylov_expv(
        operator,rho0.data,prepared_time;workspace=krylov,kwargs...)
    mgf=dot(plan.trajectory.liouvillian.tracevec,exponential.value)
    isfinite(real(mgf))&&isfinite(imag(mgf))||throw(ArgumentError(
        "finite-time moment-generating function is nonfinite; reduce the " *
        "field/time or use wider precision"))
    return_info||return mgf
    info=(mgf,log_mgf=iszero(mgf) ? nothing : log(mgf),
          field,time=prepared_time,
          state=PIState(rho0.basis,exponential.value),
          channels=plan.channels,increments=plan.increments,
          convention=:moment_generating,krylov=exponential)
    info
end

function finite_time_mgf(source,rho0::PIState,time::Real,field;
        channels=:all,increments=1,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,kwargs...)
    prepared=_counting_plan_with_budget(
        source;channels,increments,memory_budget,
        operation="finite-time counting-statistics model preparation")
    solve_budget=_counting_nested_budget(
        memory_budget,prepared.hidden_retained_bytes,
        "finite-time counting-statistics hidden plan retention")
    finite_time_mgf(
        prepared.plan,rho0,time,field;memory_budget=solve_budget,kwargs...)
end
