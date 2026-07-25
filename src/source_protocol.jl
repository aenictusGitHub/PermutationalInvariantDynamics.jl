# Internal traits shared by deterministic consumers of linear operators.
#
# Keep this protocol deliberately private: public source types retain their
# existing APIs, while downstream algorithms avoid duplicating type switches.

# Physical traces are sparse in PI coefficient coordinates: only the diagonal
# of each Schur block contributes.  Keep that structure through prepared
# matrix-free workflows instead of retaining a coordinate-sized dense vector.
# `SparseVector` already provides the required read-only `AbstractVector`
# compatibility (indexing, `dot`, and `norm`); plans own their freshly built
# arrays and wrappers copy only its nonzero support.
_trace_nonzero_indices(values::SparseVector)=
    SparseArrays.nonzeroinds(values)

function _trace_functional(basis::PIBasis,::Type{T}=ComplexF64) where T
    T<:Number&&isconcretetype(T)||throw(ArgumentError(
        "trace-functional scalar type must be a concrete Number type"))
    indices=Int[]
    values=T[]
    sizehint!(indices,sum(length,basis.patterns;init=0))
    sizehint!(values,sum(length,basis.patterns;init=0))
    R=_real_float_type(T)
    for (sector,partition) in pairs(basis.sectors)
        block_dimension=length(basis.patterns[sector])
        offset=basis.offsets[sector]-1
        scale=_schur_multiplicity_scale(R,partition)
        converted=convert(T,scale)
        for diagonal in 1:block_dimension
            push!(indices,offset+diagonal+(diagonal-1)*block_dimension)
            push!(values,converted)
        end
    end
    sparsevec(indices,values,length(basis))
end

function _checked_trace_functional_value(value,::Type{T},index,
        context::AbstractString) where T
    value isa Number&&isfinite(real(value))&&isfinite(imag(value))||
        throw(ArgumentError(
            "$context entry $index must be a finite number"))
    integer_components=real(value) isa Integer&&imag(value) isa Integer
    integer_components||promote_type(T,typeof(value))===T||
        throw(ArgumentError(
            "$context entry $index of type $(typeof(value)) would narrow " *
            "in working precision $T"))
    converted=try
        convert(T,value)
    catch error
        error isa InexactError||error isa OverflowError||rethrow()
        throw(ArgumentError(
            "$context entry $index is not representable in working " *
            "precision $T"))
    end
    isfinite(real(converted))&&isfinite(imag(converted))||
        throw(ArgumentError(
            "$context entry $index is not finite in working precision $T"))
    for (component,converted_component,label) in
            ((real(value),real(converted),"real"),
             (imag(value),imag(converted),"imaginary"))
        if component isa Integer
            BigInt(converted_component)==BigInt(component)||
                throw(ArgumentError(
                    "the $label component of $context entry $index is not " *
                    "exactly representable in working precision $T"))
        elseif !iszero(component)&&iszero(converted_component)
            throw(ArgumentError(
                "the $label component of $context entry $index underflows " *
                "in working precision $T"))
        end
    end
    converted
end

function _convert_trace_functional(
        values::SparseVector,::Type{T};
        context::AbstractString="trace-functional") where T
    converted=Vector{T}(undef,nnz(values))
    @inbounds for position in eachindex(nonzeros(values))
        coordinate=_trace_nonzero_indices(values)[position]
        converted[position]=_checked_trace_functional_value(
            nonzeros(values)[position],T,coordinate,context)
    end
    SparseVector(length(values),copy(_trace_nonzero_indices(values)),converted)
end

function _convert_trace_functional(
        values::AbstractVector,::Type{T};
        context::AbstractString="trace-functional") where T
    converted=Vector{T}(undef,length(values))
    @inbounds for index in eachindex(values)
        converted[index]=_checked_trace_functional_value(
            values[index],T,index,context)
    end
    converted
end

function _resize_trace_functional(values::SparseVector,dimension::Integer)
    dimension>=0||throw(ArgumentError(
        "trace-functional dimension must be nonnegative"))
    isempty(_trace_nonzero_indices(values))||
        last(_trace_nonzero_indices(values))<=dimension||throw(DimensionMismatch(
            "the resized trace functional would discard nonzero entries"))
    SparseVector(Int(dimension),copy(_trace_nonzero_indices(values)),
                 copy(nonzeros(values)))
end

function _normalized_trace_functional(values::AbstractVector)
    denominator=dot(values,values)
    isfinite(real(denominator))&&isfinite(imag(denominator))&&
        !iszero(denominator)||throw(ArgumentError(
            "the physical trace functional must have finite nonzero norm"))
    values/denominator
end

function _trace_axpy!(destination::AbstractVector,coefficient,
                      values::SparseVector)
    @inbounds for position in eachindex(nonzeros(values))
        index=_trace_nonzero_indices(values)[position]
        destination[index]+=coefficient*nonzeros(values)[position]
    end
    destination
end

function _trace_axpy!(destination::AbstractVector,coefficient,
                      values::AbstractVector)
    @inbounds @simd for index in eachindex(destination,values)
        destination[index]+=coefficient*values[index]
    end
    destination
end

function _trace_dense_copy(values::SparseVector,::Type{T}=eltype(values)) where T
    destination=zeros(T,length(values))
    _trace_axpy!(destination,one(T),values)
end
_trace_dense_copy(values::AbstractVector,::Type{T}=eltype(values)) where T=
    T.(values)

function _performance_trace_functional_bytes(
        values::SparseVector;
        bigfloat_precision::Integer=precision(BigFloat))
    payload=_performance_entries_bytes(
        nnz(values),eltype(values);bigfloat_precision)+
        BigInt(nnz(values))*sizeof(Int)+3*BigInt(sizeof(Int))
    max(BigInt(Base.summarysize(values)),payload)
end
function _performance_trace_functional_bytes(
        values::AbstractVector;
        bigfloat_precision::Integer=precision(BigFloat))
    max(BigInt(Base.summarysize(values)),_performance_entries_bytes(
        length(values),eltype(values);bigfloat_precision))
end

function _add_trace_border_block!(block::AbstractMatrix,
        left::SparseVector,right::SparseVector,offset::Integer)
    dimension=size(block,1)
    size(block,2)==dimension||throw(DimensionMismatch(
        "trace-border block must be square"))
    first_coordinate=offset+1
    last_coordinate=offset+dimension
    left_first=searchsortedfirst(
        _trace_nonzero_indices(left),first_coordinate)
    left_last=searchsortedlast(
        _trace_nonzero_indices(left),last_coordinate)
    right_first=searchsortedfirst(
        _trace_nonzero_indices(right),first_coordinate)
    right_last=searchsortedlast(
        _trace_nonzero_indices(right),last_coordinate)
    @inbounds for left_position in left_first:left_last
        row=_trace_nonzero_indices(left)[left_position]-offset
        left_value=nonzeros(left)[left_position]
        for right_position in right_first:right_last
            column=_trace_nonzero_indices(right)[right_position]-offset
            block[row,column]+=
                left_value*conj(nonzeros(right)[right_position])
        end
    end
    block
end

function _add_trace_border_block!(block::AbstractMatrix,
        left::AbstractVector,right::AbstractVector,offset::Integer)
    dimension=size(block,1)
    size(block,2)==dimension||throw(DimensionMismatch(
        "trace-border block must be square"))
    range=offset+1:offset+dimension
    @views block .+=left[range]*adjoint(right[range])
    block
end

_operator_basis(::Any)=nothing
_operator_basis(basis::PIBasis)=basis
_operator_basis(operator::AbstractPIOperator)=operator.basis
_operator_basis(model::PIModel)=model.basis
_operator_basis(plan::LiouvillianPlan)=plan.basis
_operator_basis(model::CompiledPIModel)=model.plan.basis
_operator_basis(model::SpecializedPIModel)=model.plan.basis
_operator_basis(operator::MatrixFreeLiouvillian)=
    _operator_basis(operator.plan)

_operator_trace_functional(::Any)=nothing
_operator_trace_functional(plan::LiouvillianPlan)=plan.tracevec
_operator_trace_functional(model::CompiledPIModel)=model.plan.tracevec
_operator_trace_functional(model::SpecializedPIModel)=model.plan.tracevec
_operator_trace_functional(operator::MatrixFreeLiouvillian)=operator.tracevec

# Compatibility alias for internal integrations written before structured
# functionals were introduced.  The returned value remains an AbstractVector,
# but callers must not assume dense storage.
_operator_trace_vector(source)=_operator_trace_functional(source)

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
