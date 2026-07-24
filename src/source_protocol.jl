# Internal traits shared by deterministic consumers of linear operators.
#
# Keep this protocol deliberately private: public source types retain their
# existing APIs, while downstream algorithms avoid duplicating type switches.

_operator_basis(::Any)=nothing
_operator_basis(basis::PIBasis)=basis
_operator_basis(operator::AbstractPIOperator)=operator.basis
_operator_basis(model::PIModel)=model.basis
_operator_basis(plan::LiouvillianPlan)=plan.basis
_operator_basis(model::CompiledPIModel)=model.plan.basis
_operator_basis(model::SpecializedPIModel)=model.plan.basis
_operator_basis(operator::MatrixFreeLiouvillian)=
    _operator_basis(operator.plan)

_operator_trace_vector(::Any)=nothing
_operator_trace_vector(plan::LiouvillianPlan)=plan.tracevec
_operator_trace_vector(model::CompiledPIModel)=model.plan.tracevec
_operator_trace_vector(model::SpecializedPIModel)=model.plan.tracevec
_operator_trace_vector(operator::MatrixFreeLiouvillian)=operator.tracevec

# Return a PI plan only when application of the source actually uses that
# plan. Sparse compiled sources therefore do not allocate unused Schur scratch.
_matrixfree_pi_plan(::Any)=nothing
_matrixfree_pi_plan(plan::LiouvillianPlan)=plan
_matrixfree_pi_plan(operator::MatrixFreeLiouvillian)=
    operator.plan isa LiouvillianPlan ? operator.plan : nothing
_matrixfree_pi_plan(model::CompiledPIModel)=
    model.backend===:matrixfree ? model.plan : nothing
_matrixfree_pi_plan(model::SpecializedPIModel)=
    model.backend===:matrixfree ? model.plan : nothing

_linear_operator_workspace(source)=begin
    plan=_matrixfree_pi_plan(source)
    plan===nothing ? nothing : LiouvillianWorkspace(plan)
end

# Consumers which know their matrix-right-hand-side width can request all
# retained batch scratch up front. Extensions for composite and hierarchy
# operators return their dedicated fixed-capacity workspaces.
function _linear_operator_batch_workspace(source,columns::Integer,::Type{T}) where T
    columns>=0||throw(ArgumentError(
        "batch column count must be nonnegative"))
    work=_linear_operator_workspace(source)
    work isa LiouvillianWorkspace&&
        _ensure_batch_capacity!(work.batch,columns)
    work
end

# Compatibility hook for internal extensions written against the original
# evolution-owned name. New code should use `_linear_operator_workspace`.
_liouvillian_workspace(source)=_linear_operator_workspace(source)

# Structural payload of a fresh task-owned application workspace. Keep this
# distinct from per-application transients: a compiled compatibility operator
# already retains one workspace, while Floquet/response/HEOM deliberately
# allocate another through `_linear_operator_workspace`.
_performance_linear_operator_workspace_bytes(::Any;
    batch_columns::Integer=0)=big(0)
_performance_linear_operator_workspace_bytes(::AbstractMatrix;
    batch_columns::Integer=0)=big(0)
_performance_linear_operator_workspace_bytes(plan::LiouvillianPlan;
    batch_columns::Integer=0)=
    _performance_liouvillian_workspace_bytes(plan;batch_columns)
_performance_linear_operator_workspace_bytes(model::CompiledPIModel;
    batch_columns::Integer=0)=model.backend===:matrixfree ?
    _performance_liouvillian_workspace_bytes(
        model.plan;batch_columns) : big(0)
_performance_linear_operator_workspace_bytes(model::SpecializedPIModel;
    batch_columns::Integer=0)=model.backend===:matrixfree ?
    _performance_liouvillian_workspace_bytes(
        model.plan;batch_columns) : big(0)
_performance_linear_operator_workspace_bytes(source::MatrixFreeLiouvillian;
    batch_columns::Integer=0)=source.plan isa LiouvillianPlan ?
    _performance_liouvillian_workspace_bytes(
        source.plan;batch_columns) : big(0)

# A batched matrix-free application may lazily replace the three largest-sector
# buffers in a retained or caller-owned LiouvillianWorkspace.  Count the full
# target allocation, rather than only the capacity delta: the previous arrays
# can remain live until the allocator/GC releases them.
_performance_batched_workspace_growth_bytes(::Any,
    batch_columns::Integer)=begin
    batch_columns>=0||throw(ArgumentError(
        "batch_columns must be nonnegative"))
    big(0)
end
function _performance_batched_workspace_growth_bytes(
        work::LiouvillianWorkspace,batch_columns::Integer)
    batch_columns>=0||throw(ArgumentError(
        "batch_columns must be nonnegative"))
    batch_columns<=work.batch.capacity&&return big(0)
    _performance_liouvillian_batch_payload_bytes(
        work.basis,work.Ttype,batch_columns)
end

_performance_batched_action_growth_bytes(::Any,
    batch_columns::Integer)=begin
    batch_columns>=0||throw(ArgumentError(
        "batch_columns must be nonnegative"))
    big(0)
end
function _performance_batched_action_growth_bytes(
        source::MatrixFreeLiouvillian,batch_columns::Integer)
    source.workspace isa LiouvillianWorkspace ?
        _performance_batched_workspace_growth_bytes(
            source.workspace,batch_columns) : big(0)
end
function _performance_batched_action_growth_bytes(
        model::CompiledPIModel,batch_columns::Integer)
    model.backend===:matrixfree ?
        _performance_batched_action_growth_bytes(
            model.operator,batch_columns) : big(0)
end
function _performance_batched_action_growth_bytes(
        model::SpecializedPIModel,batch_columns::Integer)
    model.backend===:matrixfree ?
        _performance_batched_action_growth_bytes(
            model.operator,batch_columns) : big(0)
end

# Consumers that perform a batched action use a fresh explicit workspace when
# the source protocol can construct one. A plan-less prepared callback (for
# example the bare operator of a scalar-rate specialization) instead grows its
# retained compatibility workspace. Compose those mutually exclusive cases
# here so Floquet and HEOM do not duplicate the distinction.
function _performance_batched_operator_workspace_bytes(source,
        batch_columns::Integer)
    fresh=_performance_linear_operator_workspace_bytes(
        source;batch_columns)
    fresh+(iszero(fresh) ? _performance_batched_action_growth_bytes(
        source,batch_columns) : big(0))
end

_operator_has_adjoint(::Any)=false
_operator_has_adjoint(::AbstractMatrix)=true
_operator_has_adjoint(::LiouvillianPlan)=true
_operator_has_adjoint(::CompiledPIModel)=true
_operator_has_adjoint(::SpecializedPIModel)=true
_operator_has_adjoint(source::MatrixFreeLiouvillian)=
    source.plan isa LiouvillianPlan||
    getfield(source,:adjoint_action!)!==nothing||
    getfield(source,:batched_adjoint_action!)!==nothing

# Prepared sources which deliberately do not expose a sparse/dense
# materialization route extend this trait. Solver recommendation uses it to
# keep automatic and explicit choices consistent with that representation.
_operator_requires_matrixfree(::Any)=false
