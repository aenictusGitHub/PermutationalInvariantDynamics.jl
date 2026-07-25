function _apply_columns!(Y,L,X,t,p,work=nothing)
    if work!==nothing
        apply!(Y,L,X,t,p,work)
    else
        # MatrixFreeLiouvillian performs its own safe column fallback when a
        # custom operator supplies no batched callback. Calling the matrix
        # route here lets an explicit callback run once for the complete
        # block instead of being hidden behind an unconditional column loop.
        apply!(Y,L,X,t,p)
    end
    Y
end

"""
    FloquetWorkspace(map)

Caller-owned scratch for one-period applications of a [`FloquetMap`](@ref).
One workspace may be reused sequentially for forward and adjoint actions, but
must not be shared by concurrent tasks.  The adjoint action reverses the
actual discrete RK4 graph, so it is the numerical adjoint of the finite-step
map rather than a separate integration of the continuous adjoint equation.
"""
struct FloquetWorkspace{S,V,W}
    source::S
    current::V
    temporary::V
    k1::V
    k2::V
    k3::V
    k4::V
    stage_adjoint::V
    source_workspace::W
end

"""
    FloquetBatchWorkspace(map, capacity; mode=:forward,
                          memory_budget=512*1024^2)

Fixed-capacity caller-owned scratch for applying a [`FloquetMap`](@ref) to
several vectors at once. `mode=:forward` stores only three `n`-by-`capacity`
arrays and uses a low-storage classical RK4 schedule. `mode=:full` stores the
four stage adjoints required by the exact reverse graph and additionally
supports [`apply_adjoint!`](@ref). Adjoint application through a forward-only
workspace raises; it is never approximated or silently disabled.

The capacity is immutable. Calls with more columns raise before propagation
instead of growing hidden storage. A prepared PI source evaluates every
driven schedule once per RK stage and applies its sectorwise matrix--matrix
kernels to the whole block. A custom `MatrixFreeLiouvillian` uses its batched
callback when present and otherwise retains the established safe column
fallback.

Because the caller's destination is also the low-storage RK4 accumulator, its
element type must exactly match the map element type. Inputs may be narrower
when they are representable and are copied into the map-precision `current`
buffer before propagation.

One workspace may be reused sequentially, never concurrently. Construction
checks the predictable stage and source-action storage against
`memory_budget`; `memory_budget=Inf` is the explicit opt-out.
"""
struct FloquetBatchWorkspace{S,M,K,W}
    source::S
    capacity::Int
    current::M
    temporary::M
    k1::M
    k2::K
    k3::K
    k4::K
    stage_adjoint::K
    source_workspace::W
    mode::Symbol
end

function _block_operator_workspace_bytes(work::FloquetBatchWorkspace)
    T=eltype(work.current)
    stage_entries=sum(BigInt(length(array)) for array in
        (work.current,work.temporary,work.k1,work.k2,work.k3,work.k4,
         work.stage_adjoint) if array!==nothing)
    source_batch_entries=if work.source_workspace isa LiouvillianWorkspace
        batch=work.source_workspace.batch
        BigInt(length(batch.input))+BigInt(length(batch.left))+
            BigInt(length(batch.right))
    else
        big(0)
    end
    _performance_entries_bytes(stage_entries+source_batch_entries,T)+
        _performance_linear_operator_workspace_bytes(
            work.source;batch_columns=0)+
        _performance_source_action_bytes(work.source,T)+
        (work.source_workspace===nothing ?
            _performance_batched_action_growth_bytes(
                work.source,work.capacity) : big(0))
end

function _floquet_batch_source_bytes(source,capacity::Int,::Type{T}) where T
    plan=_matrixfree_pi_plan(source)
    plan===nothing&&return big(0)
    maximum_block=maximum(length,plan.basis.patterns;init=1)
    # LiouvillianWorkspace grows three maximum-sector matrix buffers, each
    # packed as maximum_block by maximum_block*capacity.
    _performance_entries_bytes(
        3BigInt(maximum_block)^2*BigInt(capacity),T)
end

function _floquet_batch_workspace_bytes(map,capacity::Int,mode::Symbol)
    n=size(map,1);T=eltype(map);stage_arrays=mode===:full ? 7 : 3
    _performance_entries_bytes(
        BigInt(stage_arrays)*BigInt(n)*BigInt(capacity),T)+
        _performance_batched_operator_workspace_bytes(
            map.source,capacity)+
        _performance_source_action_bytes(map.source,T)
end

function FloquetBatchWorkspace(map,capacity::Integer;
        mode::Symbol=:forward,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    map isa FloquetMap||throw(ArgumentError(
        "FloquetBatchWorkspace requires a FloquetMap"))
    capacity>0||throw(ArgumentError(
        "Floquet batch capacity must be positive"))
    BigInt(capacity)<=typemax(Int)||throw(ArgumentError(
        "Floquet batch capacity must be representable as an Int"))
    mode in (:forward,:full)||throw(ArgumentError(
        "Floquet batch workspace mode must be :forward or :full"))
    columns=Int(capacity);n=size(map,1);T=eltype(map)
    estimate=_floquet_batch_workspace_bytes(map,columns,mode)
    _require_performance_budget("batched Floquet workspace",estimate,
        memory_budget;guidance=
        "Reduce capacity, use mode=:forward, or increase the budget.")
    current=zeros(T,n,columns);temporary=similar(current);k1=similar(current)
    k2=mode===:full ? similar(current) : nothing
    k3=mode===:full ? similar(current) : nothing
    k4=mode===:full ? similar(current) : nothing
    stage_adjoint=mode===:full ? similar(current) : nothing
    source_workspace=_linear_operator_workspace(map.source)
    source_workspace isa LiouvillianWorkspace&&
        _ensure_batch_capacity!(source_workspace.batch,columns)
    FloquetBatchWorkspace(map.source,columns,current,temporary,k1,k2,k3,k4,
        stage_adjoint,source_workspace,mode)
end

"""
    FloquetMap

Reusable matrix-free one-period map.  Construct it with [`floquet_map`](@ref)
or `FloquetMap(source, period; ...)`.  The map stores immutable integration
controls and a synchronized compatibility workspace; hot or parallel code
should call [`apply!`](@ref) or [`apply_adjoint!`](@ref) with one explicit
[`FloquetWorkspace`](@ref) per task.
"""
struct FloquetMap{T,S,R,P,B,V,W,K}
    source::S
    period::R
    t0::R
    steps::Int
    parameters::P
    basis::B
    tracevec::V
    Ttype::Type{T}
    compatibility_workspace::W
    lock::K
end

Base.size(map::FloquetMap)=(size(map.source,1),size(map.source,2))
Base.size(map::FloquetMap,index::Integer)=
    index in (1,2) ? size(map)[index] : 1
Base.eltype(map::FloquetMap)=map.Ttype
isautonomous(::FloquetMap)=true

function _performance_source_action_bytes(map::FloquetMap,::Type{T}) where T
    _performance_source_action_bytes(map.source,T)
end

function _performance_linear_operator_workspace_bytes(map::FloquetMap;
        batch_columns::Integer=0)
    _performance_array_bytes(size(map,1),eltype(map),0;linear_arrays=7)+
        _performance_linear_operator_workspace_bytes(map.source)
end

function Base.show(io::IO,map::FloquetMap)
    print(io,"FloquetMap($(size(map,1)) coordinates, period=$(map.period), ",
          "steps=$(map.steps))")
end

function _floquet_source(x)
    x isa PIModel ? compile(x;backend=:matrixfree) : x
end
_floquet_source(x,memory_budget)=x isa PIModel ?
    compile(x;backend=:matrixfree,memory_budget) : x
_floquet_source(x::AbstractMatrix)=copy(x)
_floquet_source(x::AbstractMatrix,memory_budget)=copy(x)

function _floquet_source_basis(x,source,basis)
    inferred=_operator_basis(x)
    inferred===nothing&&(inferred=_operator_basis(source))
    if basis!==nothing
        basis isa PIBasis||throw(ArgumentError("basis must be a PIBasis"))
        inferred===nothing||inferred===basis||throw(ArgumentError(
            "the supplied Floquet basis differs from the source basis"))
        return basis
    end
    inferred
end

function _floquet_source_trace(source,basis,trace_vector,::Type{T}) where T
    inferred=_operator_trace_functional(source)
    inferred===nothing&&basis!==nothing&&
        (inferred=_trace_functional(basis,T))
    values=trace_vector===nothing ? inferred : trace_vector
    values===nothing&&return nothing
    length(values)==size(source,1)||throw(DimensionMismatch(
        "Floquet trace vector has the wrong length"))
    _convert_trace_functional(
        values,T;context="Floquet trace-vector")
end

function _floquet_workspace(source,::Type{T},dimension::Integer) where T
    vector=zeros(T,dimension)
    FloquetWorkspace(source,vector,similar(vector),similar(vector),
        similar(vector),similar(vector),similar(vector),similar(vector),
        _linear_operator_workspace(source))
end

function _check_exact_floquet_integer(::Type{R},value::Integer,label) where R
    converted=R(value)
    isfinite(converted)&&BigInt(converted)==BigInt(value)||throw(ArgumentError(
        "$label=$value is not exactly representable in the Floquet working precision $R; pass a smaller integer or prepare the model at wider precision"))
    converted
end

function _validated_floquet_grid(::Type{R},period,t0,steps::Integer) where R
    step_count=_check_exact_floquet_integer(R,steps,"steps")
    h=R(period)/step_count
    isfinite(h)&&h>zero(R)||throw(ArgumentError(
        "Floquet integration step underflows or is not finite in $R; use fewer steps or wider precision"))
    first_advance=R(t0)+h
    isfinite(first_advance)||throw(ArgumentError(
        "Floquet stage time overflows in $R; shift the time origin or use wider precision"))
    first_advance!=R(t0)||throw(ArgumentError(
        "Floquet integration step does not advance time from t0 in $R; use fewer steps, shift the time origin, or use wider precision"))
    isfinite(R(t0)+R(period))||throw(ArgumentError(
        "Floquet period endpoint overflows in $R; shift the time origin or use wider precision"))
    h
end

"""
    floquet_map(source, period; steps=256, t0=0, parameters=nothing,
                basis=nothing, trace_vector=nothing,
                memory_budget=512*1024^2)

Prepare a matrix-free one-period propagator without constructing an
`n_PI`-by-`n_PI` matrix.  One application integrates a single PI coefficient
vector with the same fixed-step RK4 rule as [`floquet_propagator`](@ref).
`basis` or `trace_vector` supplies the physical trace functional for a raw
operator; PI models and compiled PI sources provide it automatically.

The returned map is a fixed linear operator even when its underlying
Liouvillian is periodically time dependent: `t0`, `period`, `parameters`, and
the RK grid are part of the prepared object.  Converge `steps` independently
of Krylov spectral or fixed-point tolerances.
"""
function floquet_map(x,period::Real;steps::Integer=256,t0::Real=0,
        parameters=nothing,basis=nothing,trace_vector=nothing,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    isfinite(period)&&period>0||throw(ArgumentError(
        "period must be finite and positive"))
    isfinite(t0)||throw(ArgumentError("t0 must be finite"))
    steps>0||throw(ArgumentError("steps must be positive"))
    BigInt(steps)<=typemax(Int)||throw(ArgumentError(
        "steps must be representable as an Int"))
    x isa PIModel&&_require_model_preparation_budget(x,memory_budget;
        operation="matrix-free Floquet model preparation")
    if x isa AbstractMatrix
        n=size(x,1);size(x,2)==n||throw(DimensionMismatch(
            "Floquet Liouvillian must be square"))
        Tpreview=_complex_float_type(eltype(x))
        preview=_performance_array_bytes(n,Tpreview,1;linear_arrays=7)
        _require_performance_budget("matrix-free Floquet matrix preparation",
            preview,memory_budget;guidance=
            "Pass a prepared matrix-free source for large problems.")
    end
    source=_floquet_source(x,memory_budget)
    source isa Union{MatrixFreeLiouvillian,LiouvillianPlan,CompiledPIModel,
                     SpecializedPIModel,RestrictedLiouvillian,
                     AbstractMatrix}||throw(ArgumentError(
        "unsupported Floquet Liouvillian representation"))
    size(source,1)==size(source,2)||throw(DimensionMismatch(
        "Floquet Liouvillian must be square"))
    T=_complex_float_type(eltype(source))
    period isa Integer||(T=promote_type(T,_complex_float_type(typeof(period))))
    t0 isa Integer||(T=promote_type(T,_complex_float_type(typeof(t0))))
    R=_real_float_type(T)
    period isa Integer&&_check_exact_floquet_integer(R,period,"period")
    t0 isa Integer&&_check_exact_floquet_integer(R,t0,"t0")
    _validated_floquet_grid(R,period,t0,steps)
    _check_liouvillian_source_precision(source,T,"Floquet map")
    prepared_basis=_floquet_source_basis(x,source,basis)
    tracevec=_floquet_source_trace(source,prepared_basis,trace_vector,T)
    estimate=_performance_array_bytes(size(source,1),T,0;linear_arrays=7)+
        _performance_linear_operator_workspace_bytes(source)+
        _performance_source_action_bytes(source,T)
    _require_performance_budget("matrix-free Floquet map workspace",estimate,
        memory_budget;guidance="Reduce the retained PI basis/model size.")
    compatibility=_floquet_workspace(source,T,size(source,1))
    FloquetMap(source,R(period),R(t0),Int(steps),parameters,prepared_basis,
               tracevec,T,compatibility,ReentrantLock())
end

FloquetMap(source,period::Real;kwargs...)=floquet_map(source,period;kwargs...)
FloquetWorkspace(map::FloquetMap)=
    _floquet_workspace(map.source,eltype(map),size(map,1))

function _check_floquet_workspace(work::FloquetWorkspace,map::FloquetMap)
    work.source===map.source||throw(ArgumentError(
        "Floquet workspace belongs to a different source"))
    n=size(map,1);T=eltype(map)
    all(vector->length(vector)==n,
        (work.current,work.temporary,work.k1,work.k2,work.k3,work.k4,
         work.stage_adjoint))||throw(DimensionMismatch(
        "Floquet workspace has the wrong dimension"))
    all(vector->eltype(vector)===T,
        (work.current,work.temporary,work.k1,work.k2,work.k3,work.k4,
         work.stage_adjoint))||throw(ArgumentError(
        "Floquet workspace has an incompatible scalar type"))
    work
end

function _check_floquet_apply(destination,map,input,work)
    length(input)==size(map,2)&&length(destination)==size(map,1)||
        throw(DimensionMismatch("Floquet vector has the wrong length"))
    T=eltype(map)
    promote_type(T,eltype(input))===T||throw(ArgumentError(
        "Floquet input cannot be represented in map precision $T"))
    promote_type(T,eltype(destination))===eltype(destination)||
        throw(ArgumentError(
        "Floquet destination cannot represent map precision $T"))
    _check_floquet_workspace(work,map)
end

function _floquet_source_action!(destination,map::FloquetMap,input,time,work)
    source=map.source
    if work.source_workspace===nothing
        source isa AbstractMatrix ? mul!(destination,source,input) :
            apply!(destination,source,input,time,map.parameters)
    else
        apply!(destination,source,input,time,map.parameters,
               work.source_workspace)
    end
end

function _floquet_source_adjoint!(destination,map::FloquetMap,input,time,work)
    source=map.source
    if source isa AbstractMatrix
        mul!(destination,adjoint(source),input)
    elseif work.source_workspace===nothing
        apply_adjoint!(destination,source,input,time,map.parameters)
    else
        apply_adjoint!(destination,source,input,time,map.parameters,
                       work.source_workspace)
    end
end

"""Apply a prepared one-period map with caller-owned scratch."""
function apply!(destination::AbstractVector,map::FloquetMap,
                input::AbstractVector,work::FloquetWorkspace)
    _check_floquet_apply(destination,map,input,work)
    copyto!(work.current,input)
    h=map.period/_real_float_type(eltype(map))(map.steps)
    for step in 1:map.steps
        time=map.t0+(step-1)*h
        _floquet_source_action!(work.k1,map,work.current,time,work)
        @. work.temporary=work.current+(h/2)*work.k1
        _floquet_source_action!(work.k2,map,work.temporary,time+h/2,work)
        @. work.temporary=work.current+(h/2)*work.k2
        _floquet_source_action!(work.k3,map,work.temporary,time+h/2,work)
        @. work.temporary=work.current+h*work.k3
        _floquet_source_action!(work.k4,map,work.temporary,time+h,work)
        @. work.current=work.current+
            (h/6)*(work.k1+2work.k2+2work.k3+work.k4)
    end
    copyto!(destination,work.current)
end

function _check_floquet_batch_workspace(work::FloquetBatchWorkspace,
        map::FloquetMap,columns::Integer;adjoint_action::Bool=false)
    work.source===map.source||throw(ArgumentError(
        "Floquet batch workspace belongs to a different source"))
    0<=columns<=work.capacity||throw(ArgumentError(
        "Floquet batch contains $columns columns, exceeding workspace capacity $(work.capacity)"))
    adjoint_action&&work.mode!==:full&&throw(ArgumentError(
        "adjoint Floquet batch application requires mode=:full"))
    n=size(map,1);T=eltype(map)
    for array in (work.current,work.temporary,work.k1)
        size(array)==(n,work.capacity)||throw(DimensionMismatch(
            "Floquet batch workspace has incompatible stage dimensions"))
        eltype(array)===T||throw(ArgumentError(
            "Floquet batch workspace has an incompatible scalar type"))
    end
    if work.mode===:full
        for array in (work.k2,work.k3,work.k4,work.stage_adjoint)
            array isa AbstractMatrix&&size(array)==(n,work.capacity)||
                throw(DimensionMismatch(
                    "full Floquet batch workspace has incompatible adjoint stages"))
            eltype(array)===T||throw(ArgumentError(
                "full Floquet batch workspace has an incompatible scalar type"))
        end
    else
        all(isnothing,(work.k2,work.k3,work.k4,work.stage_adjoint))||
            throw(ArgumentError(
                "forward Floquet batch workspace unexpectedly retains adjoint stages"))
    end
    work
end

function _check_floquet_batch_apply(destination,map,input,work;
                                    adjoint_action::Bool=false)
    size(input,1)==size(map,2)&&
        size(destination)==(size(map,1),size(input,2))||
        throw(DimensionMismatch("Floquet batch has the wrong dimensions"))
    T=eltype(map)
    promote_type(T,eltype(input))===T||throw(ArgumentError(
        "Floquet batch input cannot be represented in map precision $T"))
    eltype(destination)===T||throw(ArgumentError(
        "Floquet batch destination must have the map scalar type $T because it is the low-storage RK4 accumulator"))
    _check_floquet_batch_workspace(work,map,size(input,2);adjoint_action)
end

function _floquet_batch_source_action!(destination,map::FloquetMap,input,time,
                                       work::FloquetBatchWorkspace)
    source=map.source
    if work.source_workspace===nothing
        source isa AbstractMatrix ? mul!(destination,source,input) :
            apply!(destination,source,input,time,map.parameters)
    else
        apply!(destination,source,input,time,map.parameters,
               work.source_workspace)
    end
end

function _floquet_batch_source_adjoint!(destination,map::FloquetMap,input,time,
                                        work::FloquetBatchWorkspace)
    source=map.source
    if source isa AbstractMatrix
        mul!(destination,adjoint(source),input)
    elseif work.source_workspace===nothing
        apply_adjoint!(destination,source,input,time,map.parameters)
    else
        apply_adjoint!(destination,source,input,time,map.parameters,
                       work.source_workspace)
    end
end

"""Apply a prepared one-period map to a fixed-capacity block in one batched RK4 traversal."""
function apply!(destination::AbstractMatrix,map::FloquetMap,
                input::AbstractMatrix,work::FloquetBatchWorkspace)
    _check_floquet_batch_apply(destination,map,input,work)
    columns=size(input,2);iszero(columns)&&return destination
    current=view(work.current,:,1:columns)
    temporary=view(work.temporary,:,1:columns)
    derivative=view(work.k1,:,1:columns)
    copyto!(current,input)
    h=map.period/_real_float_type(eltype(map))(map.steps)
    for step in 1:map.steps
        time=map.t0+(step-1)*h
        # `destination` is the RK accumulator. The caller's input may alias
        # it because the complete step origin is already detached in current.
        copyto!(destination,current)
        _floquet_batch_source_action!(derivative,map,current,time,work)
        @. destination=destination+(h/6)*derivative
        @. temporary=current+(h/2)*derivative
        _floquet_batch_source_action!(derivative,map,temporary,time+h/2,work)
        @. destination=destination+(h/3)*derivative
        @. temporary=current+(h/2)*derivative
        _floquet_batch_source_action!(derivative,map,temporary,time+h/2,work)
        @. destination=destination+(h/3)*derivative
        @. temporary=current+h*derivative
        _floquet_batch_source_action!(derivative,map,temporary,time+h,work)
        @. destination=destination+(h/6)*derivative
        copyto!(current,destination)
    end
    destination
end

function apply!(destination::AbstractMatrix,map::FloquetMap,
                input::AbstractMatrix,work::FloquetWorkspace)
    size(input,1)==size(map,2)&&
        size(destination)==(size(map,1),size(input,2))||
        throw(DimensionMismatch("Floquet batch has the wrong dimensions"))
    safe_input=Base.mightalias(destination,input)&&destination!==input ?
        copy(input) : input
    for column in axes(safe_input,2)
        apply!(view(destination,:,column),map,view(safe_input,:,column),work)
    end
    destination
end

"""
    apply_adjoint!(destination, map::FloquetMap, input, workspace)

Apply the exact adjoint of the *discrete* RK4 period map.  Reverse accumulation
uses the same stage times and coefficients as [`apply!`](@ref), so the dot-
product identity holds to roundoff for the chosen finite step count.
"""
function apply_adjoint!(destination::AbstractVector,map::FloquetMap,
                        input::AbstractVector,work::FloquetWorkspace)
    _check_floquet_apply(destination,map,input,work)
    _operator_has_adjoint(map.source)||throw(ArgumentError(
        "the Floquet source does not provide an explicit adjoint action"))
    copyto!(work.current,input)
    h=map.period/_real_float_type(eltype(map))(map.steps)
    for step in map.steps:-1:1
        time=map.t0+(step-1)*h
        copyto!(work.temporary,work.current)
        @. work.k1=(h/6)*work.current
        @. work.k2=(h/3)*work.current
        @. work.k3=(h/3)*work.current
        @. work.k4=(h/6)*work.current

        _floquet_source_adjoint!(work.stage_adjoint,map,work.k4,time+h,work)
        @. work.temporary=work.temporary+work.stage_adjoint
        @. work.k3=work.k3+h*work.stage_adjoint

        _floquet_source_adjoint!(work.stage_adjoint,map,work.k3,time+h/2,work)
        @. work.temporary=work.temporary+work.stage_adjoint
        @. work.k2=work.k2+(h/2)*work.stage_adjoint

        _floquet_source_adjoint!(work.stage_adjoint,map,work.k2,time+h/2,work)
        @. work.temporary=work.temporary+work.stage_adjoint
        @. work.k1=work.k1+(h/2)*work.stage_adjoint

        _floquet_source_adjoint!(work.stage_adjoint,map,work.k1,time,work)
        @. work.current=work.temporary+work.stage_adjoint
    end
    copyto!(destination,work.current)
end

function apply_adjoint!(destination::AbstractMatrix,map::FloquetMap,
                        input::AbstractMatrix,
                        work::FloquetBatchWorkspace)
    _check_floquet_batch_apply(destination,map,input,work;
                               adjoint_action=true)
    _operator_has_adjoint(map.source)||throw(ArgumentError(
        "the Floquet source does not provide an explicit adjoint action"))
    columns=size(input,2);iszero(columns)&&return destination
    current=view(work.current,:,1:columns)
    temporary=view(work.temporary,:,1:columns)
    k1=view(work.k1,:,1:columns);k2=view(work.k2,:,1:columns)
    k3=view(work.k3,:,1:columns);k4=view(work.k4,:,1:columns)
    stage_adjoint=view(work.stage_adjoint,:,1:columns)
    copyto!(current,input)
    h=map.period/_real_float_type(eltype(map))(map.steps)
    for step in map.steps:-1:1
        time=map.t0+(step-1)*h
        copyto!(temporary,current)
        @. k1=(h/6)*current
        @. k2=(h/3)*current
        @. k3=(h/3)*current
        @. k4=(h/6)*current

        _floquet_batch_source_adjoint!(stage_adjoint,map,k4,time+h,work)
        @. temporary=temporary+stage_adjoint
        @. k3=k3+h*stage_adjoint

        _floquet_batch_source_adjoint!(stage_adjoint,map,k3,time+h/2,work)
        @. temporary=temporary+stage_adjoint
        @. k2=k2+(h/2)*stage_adjoint

        _floquet_batch_source_adjoint!(stage_adjoint,map,k2,time+h/2,work)
        @. temporary=temporary+stage_adjoint
        @. k1=k1+(h/2)*stage_adjoint

        _floquet_batch_source_adjoint!(stage_adjoint,map,k1,time,work)
        @. current=temporary+stage_adjoint
    end
    copyto!(destination,current)
    destination
end

function apply_adjoint!(destination::AbstractMatrix,map::FloquetMap,
                        input::AbstractMatrix,work::FloquetWorkspace)
    size(input,1)==size(map,1)&&
        size(destination)==(size(map,2),size(input,2))||
        throw(DimensionMismatch("adjoint Floquet batch has the wrong dimensions"))
    safe_input=Base.mightalias(destination,input)&&destination!==input ?
        copy(input) : input
    for column in axes(safe_input,2)
        apply_adjoint!(view(destination,:,column),map,
                       view(safe_input,:,column),work)
    end
    destination
end

# Compatibility signatures let Floquet maps participate in the generic
# restriction layer, which passes an otherwise irrelevant time/parameter pair.
apply!(destination,map::FloquetMap,input,time,parameters,
       work::FloquetWorkspace)=apply!(destination,map,input,work)
apply_adjoint!(destination,map::FloquetMap,input,time,parameters,
               work::FloquetWorkspace)=apply_adjoint!(destination,map,input,work)

function LinearAlgebra.mul!(destination::AbstractVecOrMat,map::FloquetMap,input)
    lock(map.lock)
    try
        apply!(destination,map,input,map.compatibility_workspace)
    finally
        unlock(map.lock)
    end
end

Base.:*(map::FloquetMap,input::AbstractVector)=
    mul!(_product_destination(map,input,size(map,1)),map,input)
Base.:*(map::FloquetMap,input::AbstractMatrix)=
    mul!(_product_destination(map,input,size(map,1),size(input,2)),map,input)

apply!(destination,map::FloquetMap,input)=mul!(destination,map,input)
function apply_adjoint!(destination,map::FloquetMap,input)
    mul!(destination,adjoint(map),input)
end

struct _AdjointFloquetMap{T,F}
    parent::F
end
Base.size(map::_AdjointFloquetMap)=reverse(size(map.parent))
Base.size(map::_AdjointFloquetMap,index::Integer)=
    index in (1,2) ? size(map)[index] : 1
Base.eltype(map::_AdjointFloquetMap{T}) where T=T
isautonomous(::_AdjointFloquetMap)=true
adjoint(map::FloquetMap)=_AdjointFloquetMap{eltype(map),typeof(map)}(map)
adjoint(map::_AdjointFloquetMap)=map.parent

function LinearAlgebra.mul!(destination::AbstractVecOrMat,
                            wrapper::_AdjointFloquetMap,input)
    map=wrapper.parent
    lock(map.lock)
    try
        apply_adjoint!(destination,map,input,map.compatibility_workspace)
    finally
        unlock(map.lock)
    end
end
Base.:*(map::_AdjointFloquetMap,input::AbstractVector)=
    mul!(_product_destination(map,input,size(map,1)),map,input)
Base.:*(map::_AdjointFloquetMap,input::AbstractMatrix)=
    mul!(_product_destination(map,input,size(map,1),size(input,2)),map,input)

# Certified coordinate restrictions use their embedded matrix-free backend for a
# Floquet map, retaining reduced Krylov vectors and caller-owned ambient
# period-action scratch.
_restricted_source_workspace(map::FloquetMap)=FloquetWorkspace(map)
_operator_basis(map::FloquetMap)=map.basis
_operator_trace_functional(map::FloquetMap)=map.tracevec
_operator_has_adjoint(map::FloquetMap)=_operator_has_adjoint(map.source)

"""
    restricted_floquet_map(map, restriction; atol=0, rtol=nothing)

Exhaustively certify and prepare a symmetry-coordinate restriction of
a matrix-free period map.  The returned [`RestrictedLiouvillian`](@ref) uses
compressed solver vectors and an embedded caller-owned Floquet action; it does
not materialize the one-period matrix.
"""
restricted_floquet_map(map::FloquetMap,
        restriction::SymmetryCoordinateRestriction;kwargs...)=
    RestrictedLiouvillian(map,restriction;backend=:embedded,kwargs...)

"""
    floquet_propagator(model_or_L, period; steps=256, t0=0,
                       parameters=nothing, memory_budget=512*1024^2)

Integrate one period of a time-dependent PI Liouvillian with a fixed-step RK4
scheme. All stage matrices and matrix-free sector workspaces are preallocated.
Fixed operator terms with time-dependent scalar `rate=(t,p)->...` never
assemble an instantaneous sparse Liouvillian. An `InPlaceTimeOperator` is
evaluated once per RK stage and its dynamic blocks are reused for every
propagator column.

This convenience necessarily retains several dense `n_PI`-by-`n_PI` RK
arrays. Their conservative peak estimate is checked before allocation. For
large problems use [`floquet_map`](@ref), which applies one period to selected
vectors matrix free. `memory_budget=Inf` is an explicit opt-in to the dense
route.
"""
function floquet_propagator(x,period::Real;steps::Integer=256,t0::Real=0,
                            parameters=nothing,
                            memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    isfinite(period)&&period>0||throw(ArgumentError("period must be finite and positive"))
    isfinite(t0)||throw(ArgumentError("t0 must be finite"))
    steps>0||throw(ArgumentError("steps must be positive"))
    BigInt(steps)<=typemax(Int)||throw(ArgumentError(
        "steps must be representable as an Int"))
    x isa PIModel&&_require_model_preparation_budget(x,memory_budget;
        operation="dense Floquet model preparation")
    L=x isa PIModel ? compile(x;backend=:matrixfree,memory_budget) : x
    L isa Union{MatrixFreeLiouvillian,LiouvillianPlan,CompiledPIModel,
                SpecializedPIModel,AbstractMatrix}||
        throw(ArgumentError("unsupported Floquet Liouvillian representation"))
    n=size(L,1);T=_complex_float_type(eltype(L))
    period isa Integer||(T=promote_type(T,_complex_float_type(typeof(period))))
    t0 isa Integer||(T=promote_type(T,_complex_float_type(typeof(t0))))
    R=_real_float_type(T)
    period isa Integer&&_check_exact_floquet_integer(R,period,"period")
    t0 isa Integer&&_check_exact_floquet_integer(R,t0,"t0")
    int_steps=Int(steps)
    h=_validated_floquet_grid(R,period,t0,int_steps)
    _check_liouvillian_source_precision(L,T,"Floquet propagator")
    # Four global matrices are explicit: the returned propagator plus three
    # low-storage RK4 arrays. Batched Liouvillian application can
    # additionally grow sectorwise matrix--matrix scratch up to the column
    # count, so use a conservative twelve-array peak.
    estimate=_performance_array_bytes(n,T,12;linear_arrays=4)
    _require_performance_budget("dense Floquet propagator construction",
        estimate,memory_budget;guidance=
        "Use floquet_map for matrix-free period applications.")
    work=_linear_operator_workspace(L)
    U=Matrix{T}(I,n,n);origin=similar(U);stage=similar(U);derivative=similar(U)
    for step in 1:int_steps
        t=R(t0)+(step-1)*h
        copyto!(origin,U)
        _apply_columns!(derivative,L,origin,t,parameters,work)
        @. U=origin+(h/6)*derivative
        @. stage=origin+(h/2)*derivative
        _apply_columns!(derivative,L,stage,t+h/2,parameters,work)
        @. U=U+(h/3)*derivative
        @. stage=origin+(h/2)*derivative
        _apply_columns!(derivative,L,stage,t+h/2,parameters,work)
        @. U=U+(h/3)*derivative
        @. stage=origin+h*derivative
        _apply_columns!(derivative,L,stage,t+h,parameters,work)
        @. U=U+(h/6)*derivative
    end
    U
end

_floquet_period(map::FloquetMap)=map.period
_floquet_period(operator::RestrictedLiouvillian)=
    _floquet_period(operator.source)
_floquet_period(::Any)=nothing

function _require_floquet_operator(operator)
    _floquet_period(operator)===nothing&&throw(ArgumentError(
        "selected Floquet analysis requires a FloquetMap or a restriction of one"))
    operator
end

"""
    selected_floquet_multipliers(map; nev=min(6, size(map,1)),
        method=:iram, which=:LM, vectors=false, block_size=min(nev,4),
        operator_workspace=nothing, kwargs...)

Compute selected Floquet multipliers without constructing the one-period
matrix. `method=:arnoldi` uses ordinary Arnoldi, `:iram` uses exact-shift
implicitly restarted Arnoldi, and `:jd` uses Jacobi--Davidson near `target`
(one by default). `method=:block_arnoldi` uses the explicitly named
thick-restarted block method. For a `FloquetMap`, it automatically prepares a
forward-only [`FloquetBatchWorkspace`](@ref) of `block_size`; pass a compatible
`operator_workspace` to reuse that bounded period-action storage.

The returned named tuple always contains Ritz `residuals`,
`converged`, the selected `values`, the full operator `dimension`, and
`partial_scope`; request `vectors=true` to retain right Floquet modes.

Largest-modulus Arnoldi/IRAM selections are appropriate for slow decays.
Jacobi--Davidson is a near-target calculation and is never labeled a global
spectral-radius search. Converge the period map's RK `steps` separately from
the eigensolver tolerances.
"""
function selected_floquet_multipliers(operator;
        nev::Integer=min(6,size(operator,1)),method::Symbol=:iram,
        which::Symbol=:LM,target=nothing,vectors::Bool=false,
        block_size::Integer=min(nev,4),operator_workspace=nothing,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,kwargs...)
    _require_floquet_operator(operator)
    n=size(operator,1);size(operator,2)==n||throw(DimensionMismatch(
        "Floquet operator must be square"))
    0<nev<=n||throw(ArgumentError(
        "nev must lie between 1 and the Floquet operator dimension"))
    method=_canonical_spectrum_algorithm(method)
    method in (:arnoldi,:block_arnoldi,:iram,:jd)||throw(ArgumentError(
        "Floquet multiplier method must be :arnoldi, :block_arnoldi, :iram, or :jd"))
    which in (:LR,:LM,:SM)||throw(ArgumentError(
        "which must be :LR, :LM, or :SM"))
    method===:block_arnoldi||operator_workspace===nothing||
        throw(ArgumentError(
            "operator_workspace is only used by method=:block_arnoldi"))
    spectral_method=method
    if spectral_method===:block_arnoldi
        block_size>0||throw(ArgumentError("block_size must be positive"))
        BigInt(block_size)<=typemax(Int)||throw(ArgumentError(
            "block_size must be representable as an Int"))
    end
    solver_krylovdim=Int(method===:jd ?
        get(kwargs,:subspace_dim,max(30,3Int(nev)+6)) :
        get(kwargs,:krylovdim,spectral_method===:block_arnoldi ?
            max(30,3Int(nev)+2Int(block_size)) : method===:iram ?
            max(30,3Int(nev)+6) : max(20,2Int(nev)+4)))
    local_operator_workspace=if spectral_method===:block_arnoldi&&
            operator_workspace===nothing&&operator isa FloquetMap
        nothing
    else
        operator_workspace
    end
    estimate=_selected_spectrum_workspace_bytes(operator,spectral_method,
        solver_krylovdim,nev;vectors,target,block_size,
        operator_workspace=local_operator_workspace,kwargs...)
    if spectral_method===:block_arnoldi&&local_operator_workspace===nothing&&
            operator isa FloquetMap
        estimate+=_floquet_batch_workspace_bytes(operator,
            min(Int(block_size),n),:forward)
    end
    _require_performance_budget("selected Floquet spectral workspace",
        estimate,memory_budget;guidance=
        "Reduce nev/Krylov dimensions or increase the budget.")
    result=if method===:arnoldi
        target===nothing||throw(ArgumentError(
            "ordinary Arnoldi does not accept target; use method=:jd"))
        krylov_liouvillian_spectrum(operator;nev,which,vectors,kwargs...)
    elseif spectral_method===:block_arnoldi
        block_work=local_operator_workspace===nothing&&operator isa FloquetMap ?
            FloquetBatchWorkspace(operator,min(Int(block_size),n);
                mode=:forward,memory_budget) : local_operator_workspace
        block_arnoldi_spectrum(operator;nev,which,target,vectors,block_size,
            operator_workspace=block_work,memory_budget,kwargs...)
    elseif method===:iram
        implicitly_restarted_arnoldi_spectrum(operator;nev,which,target,
                                               vectors,kwargs...)
    else
        requested_target=target===nothing ? one(eltype(operator)) : target
        jacobi_davidson_spectrum(operator;nev,target=requested_target,
                                 vectors,kwargs...)
    end
    merge(result,(method=spectral_method,period=_floquet_period(operator),
        selection=method===:jd||target!==nothing ? :near_target : which,
        partial_scope=length(result.values)<n,
        representation=:multipliers))
end

"""Floquet multipliers, sorted by decreasing modulus for an explicit map."""
function floquet_multipliers(F::AbstractMatrix;
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    n=size(F,1);size(F,2)==n||throw(DimensionMismatch(
        "Floquet propagator must be square"))
    T=promote_type(_complex_float_type(eltype(F)),ComplexF64)
    estimate=_performance_array_bytes(n,T,5;linear_arrays=4)
    _require_performance_budget("complete Floquet eigendecomposition",estimate,
        memory_budget;guidance=
        "Use selected_floquet_multipliers on a FloquetMap for selected roots.")
    sort(eigvals(Matrix(F));by=abs,rev=true)
end

function floquet_multipliers(map::FloquetMap;return_info::Bool=false,kwargs...)
    result=selected_floquet_multipliers(map;kwargs...)
    return_info ? result : result.values
end

function floquet_multipliers(map::FloquetMap,period::Real;kwargs...)
    period==map.period||throw(ArgumentError(
        "the supplied period differs from the prepared FloquetMap period"))
    floquet_multipliers(map;kwargs...)
end

function floquet_multipliers(x,period::Real;
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,kwargs...)
    F=floquet_propagator(x,period;memory_budget,kwargs...)
    floquet_multipliers(F;memory_budget)
end

"""Principal-branch Floquet exponents `log(lambda)/period`."""
floquet_exponents(F::AbstractMatrix,period::Real;
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)=
    log.(complex.(floquet_multipliers(F;memory_budget)))./period
function floquet_exponents(x,period::Real;
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,kwargs...)
    F=floquet_propagator(x,period;memory_budget,kwargs...)
    floquet_exponents(F,period;memory_budget)
end

function floquet_exponents(map::FloquetMap;return_info::Bool=false,kwargs...)
    result=selected_floquet_multipliers(map;kwargs...)
    values=log.(complex.(result.values))./map.period
    return_info ? merge(result,(multipliers=result.values,values,
        representation=:exponents,residual_representation=:multipliers,
        branch=:principal)) : values
end


function floquet_exponents(map::FloquetMap,period::Real;
                           return_info::Bool=false,kwargs...)
    period==map.period||throw(ArgumentError(
        "the supplied period differs from the prepared FloquetMap period"))
    floquet_exponents(map;return_info,kwargs...)
end

function _explicit_floquet_residuals(operator,values,vectors)
    size(vectors,1)==size(operator,1)&&size(vectors,2)==length(values)||
        throw(DimensionMismatch("Floquet Ritz vectors have incompatible dimensions"))
    R=_real_float_type(eltype(vectors));residuals=Vector{R}(undef,length(values))
    image=similar(view(vectors,:,1))
    for index in eachindex(values)
        vector=view(vectors,:,index)
        mul!(image,operator,vector)
        @. image=image-values[index]*vector
        residuals[index]=norm(image)
    end
    residuals
end

function _selected_floquet_gap(operator,period;return_info::Bool=false,
        nev::Integer=min(6,size(operator,1)),method::Symbol=:iram,
        atol::Real=1e-10,rtol::Real=1e-8,
        fixed_atol::Real=atol,fixed_rtol::Real=rtol,
        check_stability::Bool=true,kwargs...)
    isfinite(period)&&period>0||throw(ArgumentError(
        "period must be finite and positive"))
    for (label,value) in (("atol",atol),("rtol",rtol),
                          ("fixed_atol",fixed_atol),
                          ("fixed_rtol",fixed_rtol))
        value isa Real&&isfinite(value)&&value>=0||throw(ArgumentError(
            "$label must be finite and nonnegative"))
    end
    selected=selected_floquet_multipliers(operator;nev,method,vectors=true,
                                           atol,rtol,kwargs...)
    values=selected.values;vectors=selected.vectors
    residuals=_explicit_floquet_residuals(operator,values,vectors)
    R=promote_type(_real_float_type(eltype(values)),typeof(float(period)),
                   typeof(float(atol)),typeof(float(rtol)))
    residual_tolerances=Vector{R}(undef,length(values))
    residual_converged=BitVector(undef,length(values))
    for index in eachindex(values)
        vector=view(vectors,:,index);scale=max(norm(vector),
            abs(values[index])*norm(vector),one(R))
        residual_tolerances[index]=R(atol)+R(rtol)*scale
        residual_converged[index]=residuals[index]<=residual_tolerances[index]
    end
    fixed_tolerance=R(fixed_atol)+R(fixed_rtol)
    fixed=findall(index->abs(values[index]-one(values[index]))<=fixed_tolerance&&
        residual_converged[index],eachindex(values))
    isempty(fixed)&&throw(ArgumentError(
        "selected Floquet modes contain no residual-certified unit multiplier; increase nev or eigensolver accuracy"))
    nonstationary=setdiff(collect(eachindex(values)),fixed)
    complete=length(values)==size(operator,1)
    rate=if isempty(nonstationary)
        complete&&size(operator,1)>1 ? zero(R) : R(NaN)
    else
        radius=maximum(abs(values[index]) for index in nonstationary)
        check_stability&&radius>one(R)+fixed_tolerance&&throw(ArgumentError(
            "selected Floquet spectrum is unstable: subleading spectral radius=$radius"))
        max(zero(R),-log(R(radius))/R(period))
    end
    controlling=isempty(nonstationary) ? nothing :
        nonstationary[argmax(abs(values[index]) for index in nonstationary)]
    residual_certified=all(residual_converged)
    global_gap_certified=complete&&residual_certified&&size(operator,1)>1
    info=(gap=rate,
        subleading_multiplier=controlling===nothing ? nothing : values[controlling],
        fixed_multipliers=values[fixed],stationary_multiplicity=length(fixed),
        values,residuals,residual_tolerances,
        converged=residual_converged,residual_certified,
        global_gap_certified,
        stability_certified=complete&&residual_certified&&check_stability,
        dimension=size(operator,1),selected_count=length(values),
        partial_scope=!complete,period=R(period),method,
        solver_result=selected)
    return_info&&return info
    complete&&size(operator,1)==1&&return rate
    global_gap_certified||throw(ArgumentError(
        "selected multipliers have certified residuals but do not span the full Floquet dimension; set return_info=true and inspect global_gap_certified, or request nev=$(size(operator,1))"))
    rate
end

"""
    floquet_gap(F, period; atol=1e-10)

Return the nonnegative asymptotic decay rate from the subleading Floquet
multiplier. The map must have a fixed multiplier within `atol` of one and no
remaining multiplier with modulus larger than `1 + atol`; otherwise an
`ArgumentError` is thrown. A one-dimensional map has no subleading mode and
returns a precision-matched `NaN`.
"""
function floquet_gap(F::AbstractMatrix,period::Real;atol::Real=1e-10,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    isfinite(period)&&period>0||throw(ArgumentError(
        "period must be finite and positive"))
    isfinite(atol)&&atol>=0||throw(ArgumentError(
        "atol must be finite and nonnegative"))
    vals=floquet_multipliers(F;memory_budget);isempty(vals)&&throw(ArgumentError(
        "Floquet gap requires a nonempty propagator"))
    R=_real_float_type(eltype(vals))
    distances=abs.(vals.-one(eltype(vals)));i=argmin(distances)
    distances[i]<=atol||throw(ArgumentError(
        "Floquet map has no unit multiplier within atol=$atol; closest distance=$(distances[i])"))
    deleteat!(vals,i)
    rate_zero=zero(R)/period
    isempty(vals)&&return oftype(rate_zero,NaN)
    radius=maximum(abs,vals)
    radius<=one(radius)+atol||throw(ArgumentError(
        "Floquet map is unstable within atol=$atol: subleading spectral radius=$radius"))
    rate=-log(radius)/period
    max(zero(rate),rate)
end
function floquet_gap(x,period::Real;atol::Real=1e-10,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,kwargs...)
    F=floquet_propagator(x,period;memory_budget,kwargs...)
    floquet_gap(F,period;atol,memory_budget)
end

function floquet_gap(map::FloquetMap;kwargs...)
    _selected_floquet_gap(map,map.period;kwargs...)
end

function floquet_gap(map::FloquetMap,period::Real;kwargs...)
    period==map.period||throw(ArgumentError(
        "the supplied period differs from the prepared FloquetMap period"))
    _selected_floquet_gap(map,period;kwargs...)
end

function floquet_gap(operator::RestrictedLiouvillian,period::Real;kwargs...)
    _require_floquet_operator(operator)
    period==_floquet_period(operator)||throw(ArgumentError(
        "the supplied period differs from the restricted Floquet map period"))
    _selected_floquet_gap(operator,period;kwargs...)
end

struct _FloquetFixedPointOperator{T,F,W}
    map::F
    workspace::W
end
Base.size(operator::_FloquetFixedPointOperator)=size(operator.map)
Base.size(operator::_FloquetFixedPointOperator,index::Integer)=
    index in (1,2) ? size(operator.map,index) : 1
Base.eltype(::_FloquetFixedPointOperator{T}) where T=T

_floquet_application_workspace(map::FloquetMap)=FloquetWorkspace(map)
_floquet_application_workspace(operator::RestrictedLiouvillian)=
    RestrictedLiouvillianWorkspace(operator.source,operator.restriction)

function _floquet_apply_with_workspace!(destination,map::FloquetMap,input,work)
    apply!(destination,map,input,work)
end
function _floquet_apply_with_workspace!(destination,
        operator::RestrictedLiouvillian,input,work)
    apply!(destination,operator,input,
        zero(_real_float_type(eltype(operator))),nothing,work)
end

function _floquet_fixed_point_operator(map,map_workspace=nothing)
    work=map_workspace===nothing ? _floquet_application_workspace(map) :
        map_workspace
    _FloquetFixedPointOperator{eltype(map),typeof(map),typeof(work)}(map,work)
end

function LinearAlgebra.mul!(destination::AbstractVector,
        operator::_FloquetFixedPointOperator,input::AbstractVector)
    length(input)==size(operator,2)&&length(destination)==size(operator,1)||
        throw(DimensionMismatch("Floquet fixed-point vector has the wrong length"))
    safe_input=Base.mightalias(destination,input) ? copy(input) : input
    _floquet_apply_with_workspace!(destination,operator.map,safe_input,
                                   operator.workspace)
    @. destination=destination-safe_input
    destination
end

_floquet_trace(map::FloquetMap)=map.tracevec
_floquet_trace(operator::RestrictedLiouvillian)=operator.tracevec
_floquet_basis(map::FloquetMap)=map.basis
_floquet_basis(operator::RestrictedLiouvillian)=operator.restriction.basis

function _floquet_primary_application_bytes(operator,map_workspace,::Type{T}) where T
    if map_workspace isa FloquetWorkspace
        vectors=(map_workspace.current,map_workspace.temporary,
            map_workspace.k1,map_workspace.k2,map_workspace.k3,
            map_workspace.k4,map_workspace.stage_adjoint)
        S=foldl(promote_type,(eltype(vector) for vector in vectors);
                init=T)
        nested=map_workspace.source_workspace
        nested_bytes=nested===nothing ? big(0) : BigInt(Base.summarysize(nested))
        retained=operator isa FloquetMap ?
            _performance_source_action_bytes(operator,T) : big(0)
        return _performance_entries_bytes(
            sum(BigInt(length(vector)) for vector in vectors),S)+
            nested_bytes+retained
    elseif map_workspace isa RestrictedLiouvillianWorkspace
        entries=BigInt(length(map_workspace.ambient_input))+
                BigInt(length(map_workspace.ambient_output))
        S=promote_type(T,eltype(map_workspace.ambient_input),
                      eltype(map_workspace.ambient_output))
        nested=map_workspace.source_workspace
        if nested isa FloquetWorkspace
            vectors=(nested.current,nested.temporary,nested.k1,nested.k2,
                nested.k3,nested.k4,nested.stage_adjoint)
            entries+=sum(BigInt(length(vector)) for vector in vectors)
            for vector in vectors
                S=promote_type(S,eltype(vector))
            end
            nested.source_workspace===nothing||
                (entries+=cld(BigInt(Base.summarysize(
                    nested.source_workspace)),
                    _scalar_retained_bytes(S)))
        end
        retained=operator isa RestrictedLiouvillian ?
            _performance_source_action_bytes(operator.source,T) : big(0)
        return _performance_entries_bytes(entries,S)+retained
    end
    # `_floquet_fixed_point_operator` constructs one fresh task-owned period
    # workspace when none is supplied.  The common source protocol includes
    # all seven Floquet vectors, any nested PI/HEOM workspace, and the two
    # ambient vectors of an embedded restriction. Per-action fallback
    # materialization remains a separate transient.
    _performance_linear_operator_workspace_bytes(operator)+
        _performance_source_action_bytes(operator,T)
end

function _floquet_initial_coordinates(operator,initial_state)
    initial_state isa PIState||return initial_state
    basis=_floquet_basis(operator)
    basis===initial_state.basis||throw(ArgumentError(
        "Floquet initial state and map use incompatible PI bases"))
    if operator isa RestrictedLiouvillian
        reduced=zeros(promote_type(eltype(operator),eltype(initial_state.data)),
                      size(operator,1))
        restrict!(reduced,operator.restriction,initial_state.data)
        return reduced
    end
    initial_state.data
end

function _floquet_output_state(operator,coordinates)
    basis=_floquet_basis(operator)
    basis===nothing&&return coordinates
    if operator isa RestrictedLiouvillian
        full=zeros(eltype(coordinates),length(basis))
        embed!(full,operator.restriction,coordinates)
        return PIState(basis,full)
    end
    PIState(basis,coordinates)
end

"""
    floquet_steady_state(map::FloquetMap; method=:krylov,
                         return_info=false, ...)

Solve the periodic fixed-point equation `(F-I)rho=0` with the physical trace
constraint. `method=:krylov` (the default) uses restarted matrix-free GMRES
and never constructs the period map. `method=:dense` materializes the map and
retains the legacy SVD route for manageable PI dimensions.

An explicitly restricted Floquet map is also accepted. Its result is embedded
back into the ambient `PIState`, and `return_info=true` reports the full-space
period residual and restriction leakage. `map_workspace` owns one-period
application scratch; `workspace` is the independent GMRES workspace.
For the Krylov route both are included in the `memory_budget` preflight. The
check uses the actual capacity and scalar type of supplied reusable
workspaces.
"""
function floquet_steady_state(operator::Union{FloquetMap,RestrictedLiouvillian};
        method::Symbol=:krylov,return_info::Bool=false,initial_state=nothing,
        krylovdim::Integer=30,maxiter::Integer=500,
        atol::Real=1e-10,rtol::Real=1e-8,workspace=nothing,
        map_workspace=nothing,preconditioner=nothing,operator_scale=nothing,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    _require_floquet_operator(operator)
    method===:gmres&&(method=:krylov)
    method in (:krylov,:dense)||throw(ArgumentError(
        "Floquet steady-state method must be :krylov or :dense"))
    tracevec=_floquet_trace(operator)
    tracevec===nothing&&throw(ArgumentError(
        "the physical trace is ambiguous; construct floquet_map with basis=... or trace_vector=..."))
    norm(tracevec)>zero(_real_float_type(eltype(tracevec)))||throw(ArgumentError(
        "the selected Floquet coordinate block has zero physical trace"))
    initial=_floquet_initial_coordinates(operator,initial_state)
    source_type=_complex_float_type(eltype(operator))
    initial_type=initial===nothing ? eltype(tracevec) : eltype(initial)
    solver_type=promote_type(source_type,eltype(tracevec),initial_type)
    if method===:krylov
        solver_type=_promote_krylov_scalar_type(solver_type,operator_scale)
        preconditioner===nothing||
            (solver_type=_promote_krylov_operator_type(solver_type,
                                                       preconditioner))
    end
    if method===:krylov&&workspace!==nothing
        workspace isa Union{KrylovWorkspace,RecycledGMRESWorkspace}||
            throw(ArgumentError(
                "workspace must be a KrylovWorkspace or RecycledGMRESWorkspace"))
        solver_type=promote_type(solver_type,eltype(workspace.V))
    end
    dense_matrix=if method===:dense
        n=size(operator,1);T=promote_type(solver_type,ComplexF64)
        estimate=_performance_array_bytes(n,T,12;linear_arrays=8)
        _require_performance_budget("dense Floquet fixed-point solve",estimate,
            memory_budget;guidance=
            "Use the default method=:krylov for a matrix-free solve.")
        _materialize(operator)
    else
        n=size(operator,1)
        effective_krylovdim=workspace===nothing ? krylovdim :
            size(workspace.H,2)
        effective_recycle_dim=workspace isa RecycledGMRESWorkspace ?
            size(workspace.U,2) : 0
        estimate=_performance_gmres_bytes(n,solver_type,
            effective_krylovdim;recycle_dim=effective_recycle_dim)+
            _floquet_primary_application_bytes(operator,map_workspace,
                                                solver_type)
        _require_performance_budget("matrix-free Floquet fixed-point workspace",
            estimate,memory_budget;guidance=
            "Reduce krylovdim or reuse a smaller compatible workspace.")
        nothing
    end
    fixed=method===:krylov ?
        _floquet_fixed_point_operator(operator,map_workspace) : nothing
    result=if method===:krylov
        krylov_steady_state(fixed;trace_vector=tracevec,
            initial_state=initial,krylovdim,maxiter,atol,rtol,workspace,
            preconditioner,operator_scale,return_info=true)
    else
        steady_state(dense_matrix-I;trace_vector=tracevec,method=:svd,
            initial_state=initial,atol,rtol,return_info=true,memory_budget)
    end
    coordinates=result.state
    image=similar(coordinates);mul!(image,operator,coordinates)
    @. image=image-coordinates
    residual=norm(image)
    relative_residual=residual/max(norm(coordinates),
        one(_real_float_type(eltype(coordinates))))
    state=_floquet_output_state(operator,coordinates)
    full_report=operator isa RestrictedLiouvillian ?
        restriction_full_residual(operator,coordinates;eigenvalue=1) : nothing
    info=merge(result,(state,reduced_state=coordinates,
        residual,relative_residual,trace_error=abs(dot(tracevec,coordinates)-1),
        method,period=_floquet_period(operator),map=operator,
        propagator=dense_matrix,
        full_residual=full_report===nothing ? residual : full_report.residual,
        leakage_residual=full_report===nothing ?
            zero(_real_float_type(eltype(coordinates))) :
            full_report.leakage_residual))
    return_info ? info : state
end

function floquet_steady_state(model::PIModel,period::Real;
        steps::Integer=256,t0::Real=0,parameters=nothing,
        method::Symbol=:krylov,kwargs...)
    memory_budget=get(kwargs,:memory_budget,_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    map=floquet_map(model,period;steps,t0,parameters,memory_budget)
    floquet_steady_state(map;method,kwargs...)
end

"""Return PI states after `0:nperiods` applications of a Floquet propagator."""
function stroboscopic_evolution(rho::PIState,F,nperiods::Integer;
        include_initial::Bool=true,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    nperiods>=0||throw(ArgumentError("nperiods must be nonnegative"))
    saved_count=BigInt(nperiods)+(include_initial ? 1 : 0)
    saved_count<=typemax(Int)||throw(ArgumentError(
        "the requested stroboscopic state count must be representable as an Int"))
    n=length(rho.data);size(F)==(n,n)||throw(DimensionMismatch())
    saved=Int(saved_count)
    T=promote_type(eltype(F),eltype(rho.data))
    estimate=_performance_array_bytes(n,T,0;
                                       linear_arrays=saved_count+2)
    _require_performance_budget("stroboscopic state history",estimate,
        memory_budget;guidance=
        "Use floquet_evolve when only the final state is required.")
    x=_product_destination(F,rho.data,n);copyto!(x,rho.data);y=similar(x)
    R=_real_float_type(eltype(x));out=PIState{R,typeof(rho.basis)}[]
    # `PIState` already makes one defensive copy, so passing `x` directly
    # keeps every saved state detached without an immediately discarded copy.
    include_initial&&push!(out,PIState(rho.basis,x))
    for _ in 1:nperiods
        mul!(y,F,x);x,y=y,x;push!(out,PIState(rho.basis,x))
    end
    out
end

"""State after an integer number of periods."""
function floquet_evolve(rho::PIState,F,nperiods::Integer)
    n=length(rho.data);size(F)==(n,n)||throw(DimensionMismatch())
    # Preserve the established inverse-map semantics for negative periods.
    # The allocation-saving repeated `mul!` path applies to forward periods;
    # an inverse power may fail exactly as ordinary matrix powering did when
    # the supplied map is singular.
    nperiods<0&&(F isa AbstractMatrix||throw(ArgumentError(
        "negative stroboscopic powers require an explicit invertible Floquet matrix")))
    nperiods<0&&return PIState(rho.basis,F^nperiods*rho.data)
    x=_product_destination(F,rho.data,n)
    if iszero(nperiods)
        copyto!(x,rho.data)
        return PIState(rho.basis,x)
    end
    mul!(x,F,rho.data)
    if nperiods>1
        y=similar(x)
        for _ in 2:nperiods
            mul!(y,F,x);x,y=y,x
        end
    end
    PIState(rho.basis,x)
end
