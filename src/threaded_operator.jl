"""
    ThreadedMatrixFreeLiouvillian

Prepared matrix-free wrapper which applies one PI Liouvillian action with
deterministic Schur-sector task parallelism. Construct it with
[`threaded_matrixfree`](@ref). The immutable source and sector assignment may
be shared for read access, but the wrapper owns one mutable
[`ThreadedLiouvillianWorkspace`](@ref) and therefore serializes public
applications. Krylov solvers call it sequentially and each application uses
the configured task team internally.

This wrapper is intended for large mixed-sector actions. It does not change
BLAS thread settings and is not automatically preferable to the ordinary
matrix-free route for small blocks.
"""
struct ThreadedMatrixFreeLiouvillian{S,W,V,T,L}
    source::S
    workspace::W
    tracevec::V
    Ttype::Type{T}
    lock::L
end

Base.size(operator::ThreadedMatrixFreeLiouvillian)=
    size(operator.workspace.plan)
Base.size(operator::ThreadedMatrixFreeLiouvillian,index::Integer)=
    size(operator.workspace.plan,index)
Base.eltype(operator::ThreadedMatrixFreeLiouvillian)=operator.Ttype
isautonomous(operator::ThreadedMatrixFreeLiouvillian)=
    isautonomous(operator.source)

function Base.show(io::IO,operator::ThreadedMatrixFreeLiouvillian)
    print(io,"ThreadedMatrixFreeLiouvillian(dimension=$(size(operator,1)), ",
          "tasks=$(length(operator.workspace.assignments)), ",
          "autonomous=$(isautonomous(operator)))")
end

_threaded_operator_plan(plan::LiouvillianPlan)=plan
_threaded_operator_plan(model::CompiledPIModel)=model.plan
_threaded_operator_plan(model::SpecializedPIModel)=model.plan

function _threaded_operator_apply!(destination,source::LiouvillianPlan,input,
        time,parameters,workspace;adjoint::Bool=false)
    adjoint ?
        threaded_apply_adjoint!(
            destination,source,input,time,parameters,workspace) :
        threaded_apply!(destination,source,input,time,parameters,workspace)
end
function _threaded_operator_apply!(destination,source::CompiledPIModel,input,
        time,parameters,workspace;adjoint::Bool=false)
    adjoint ?
        threaded_apply_adjoint!(
            destination,source,input,time,parameters,workspace) :
        threaded_apply!(destination,source,input,time,parameters,workspace)
end
function _threaded_operator_apply!(destination,source::SpecializedPIModel,input,
        time,parameters,workspace;adjoint::Bool=false)
    plan=source.plan
    adjoint ?
        threaded_apply_adjoint!(
            destination,plan,input,time,source.rates,workspace) :
        threaded_apply!(
            destination,plan,input,time,source.rates,workspace)
end

"""
    threaded_matrixfree(source; tasks=Threads.nthreads())

Wrap an autonomous or driven prepared `LiouvillianPlan`, `CompiledPIModel`, or
`SpecializedPIModel` in a matrix-free operator whose vector actions use
Schur-sector task parallelism. A driven wrapper supports explicit
[`apply!`](@ref) calls; autonomous-only Krylov operations such as `mul!`,
`stationary_state`, and `liouvillian_spectrum` retain their ordinary
autonomy checks.

The returned operator owns fixed-capacity worker scratch and is safe to share
between callers only through its compatibility lock. Concurrent hot loops
should construct one wrapper per task.
"""
function threaded_matrixfree(source::Union{
        LiouvillianPlan,CompiledPIModel,SpecializedPIModel};
        tasks::Integer=Threads.nthreads())
    plan=_threaded_operator_plan(source)
    workspace=ThreadedLiouvillianWorkspace(plan;tasks)
    ThreadedMatrixFreeLiouvillian(
        source,workspace,plan.tracevec,plan.Ttype,ReentrantLock())
end

function apply!(destination,operator::ThreadedMatrixFreeLiouvillian,input,
        time,parameters)
    lock(operator.lock)
    try
        _threaded_operator_apply!(
            destination,operator.source,input,time,parameters,
            operator.workspace;adjoint=false)
    finally
        unlock(operator.lock)
    end
end
function apply_adjoint!(destination,
        operator::ThreadedMatrixFreeLiouvillian,input,time,parameters)
    lock(operator.lock)
    try
        _threaded_operator_apply!(
            destination,operator.source,input,time,parameters,
            operator.workspace;adjoint=true)
    finally
        unlock(operator.lock)
    end
end

function LinearAlgebra.mul!(destination,
        operator::ThreadedMatrixFreeLiouvillian,input)
    _require_autonomous(operator,"mul!")
    R=_real_float_type(operator.Ttype)
    apply!(destination,operator,input,zero(R),nothing)
end

function apply!(destination,operator::ThreadedMatrixFreeLiouvillian,input)
    mul!(destination,operator,input)
end
function apply_adjoint!(destination,
        operator::ThreadedMatrixFreeLiouvillian,input)
    _require_autonomous(operator,"apply_adjoint!")
    R=_real_float_type(operator.Ttype)
    apply_adjoint!(destination,operator,input,zero(R),nothing)
end

function Base.:*(operator::ThreadedMatrixFreeLiouvillian,
        input::AbstractVector)
    destination=similar(input,promote_type(
        eltype(input),eltype(operator)),size(operator,1))
    mul!(destination,operator,input)
end

# Shared prepared-source protocol. The wrapper already owns its complete
# threaded action workspace; iterative consumers must not allocate or charge a
# second ordinary LiouvillianWorkspace.
_operator_basis(operator::ThreadedMatrixFreeLiouvillian)=
    operator.workspace.plan.basis
_operator_trace_functional(operator::ThreadedMatrixFreeLiouvillian)=
    operator.tracevec
_operator_has_adjoint(::ThreadedMatrixFreeLiouvillian)=true
_operator_requires_matrixfree(::ThreadedMatrixFreeLiouvillian)=true
_linear_operator_workspace(::ThreadedMatrixFreeLiouvillian)=nothing
_fixed_liouvillian_scalar_type(
    operator::ThreadedMatrixFreeLiouvillian)=operator.Ttype
_performance_linear_operator_workspace_bytes(
    ::ThreadedMatrixFreeLiouvillian;batch_columns::Integer=0)=begin
    batch_columns>=0||throw(ArgumentError(
        "batch_columns must be nonnegative"))
    big(0)
end
_performance_source_action_bytes(
    ::ThreadedMatrixFreeLiouvillian,::Type{T}) where T=big(0)
_resource_source_prepared(::ThreadedMatrixFreeLiouvillian)=true
