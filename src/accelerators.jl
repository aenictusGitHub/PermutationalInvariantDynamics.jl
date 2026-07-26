"""
    AcceleratorCapability

Immutable description of one optional accelerator backend. Query it with
[`accelerator_capability`](@ref) before requesting an upload.

`transfer_policy=:explicit_once` means that a backend may upload a prepared
operator and caller-owned right-hand sides, but must not perform an implicit
host/device transfer during each multiplication. A backend reporting
`supports_matrixfree_pi=false` accepts only an explicitly materialized sparse
PI Liouvillian, not the general Schur-block matrix-free kernels.
"""
struct AcceleratorCapability{R,S}
    backend::Symbol
    extension_loaded::Bool
    functional::Bool
    supports_sparse_pi::Bool
    supports_matrixfree_pi::Bool
    rhs_kinds::R
    scalar_types::S
    index_type::DataType
    transfer_policy::Symbol
    reason::Symbol
    message::String
end

function _unavailable_accelerator_capability(::Val{backend}) where backend
    AcceleratorCapability(
        backend,
        false,
        false,
        false,
        false,
        (),
        (),
        Int,
        :unsupported,
        :unknown_backend,
        "accelerator backend :$backend is not registered",
    )
end

_accelerator_extension_capability(::Val)=nothing

function _core_accelerator_capability(::Val{:cuda})
    AcceleratorCapability(
        :cuda,
        false,
        false,
        true,
        false,
        (:vector,:matrix),
        (Float32,Float64,ComplexF32,ComplexF64),
        Int32,
        :explicit_once,
        :extension_not_loaded,
        "the CUDA accelerator extension is not loaded; the core package " *
        "does not claim CUDA functionality without a tested CUDA extension",
    )
end

_core_accelerator_capability(backend::Val)=
    _unavailable_accelerator_capability(backend)

function _accelerator_capability(backend::Val)
    extension=_accelerator_extension_capability(backend)
    extension===nothing ?
        _core_accelerator_capability(backend) : extension
end

"""
    accelerator_capability(backend=:cuda)

Report whether an optional accelerator is loaded and functional. This query
never initializes a device and never treats package availability as proof that
hardware works. In particular, `functional=true` is reserved for an extension
which has completed its own runtime/device checks.

The core package currently provides a conservative CUDA contract for
preflighting only. It reports `functional=false` unless a separately tested
extension implements the backend.
"""
function accelerator_capability(backend::Symbol=:cuda)
    _accelerator_capability(Val(backend))
end

"""
    AcceleratorPreflight

Read-only resource and compatibility report returned by
[`accelerator_preflight`](@ref). `basis` is the exact prepared [`PIBasis`](@ref)
owned by the source, and `scalar_type` is the prepared Liouvillian precision.

`combined_peak_bytes` accounts conservatively for the already retained source,
the simultaneous host sparse-materialization peak when needed, the uploaded
sparse operator, and both input and output device right-hand sides. The device
subset is reported separately in `device_peak_bytes`.
"""
struct AcceleratorPreflight{B,C,I}
    backend::Symbol
    basis::B
    scalar_type::DataType
    bigfloat_precision_assumption::Union{Nothing,Int}
    dimension::Int
    sectors::Int
    rhs_columns::Int
    rhs_kind::Symbol
    source_backend::Symbol
    autonomous::Bool
    sparse_materialization_supported::Bool
    materialization_required::Bool
    exact_nnz::Union{Nothing,Int}
    retained_nnz_upper_bound::BigInt
    source_retained_bytes::BigInt
    host_sparse_operator_bytes::BigInt
    host_materialization_peak_bytes::BigInt
    device_sparse_operator_bytes::BigInt
    device_vector_bytes::BigInt
    device_peak_bytes::BigInt
    combined_peak_bytes::BigInt
    memory_budget::Union{Nothing,BigInt}
    device_memory_budget::Union{Nothing,BigInt}
    fits_memory_budget::Bool
    fits_device_memory_budget::Bool
    capability::C
    issues::I
    ready::Bool
end

_accelerator_plan(source::LiouvillianPlan)=source
_accelerator_plan(source::CompiledPIModel)=source.plan
_accelerator_plan(source::SpecializedPIModel)=source.plan

_accelerator_source_backend(::LiouvillianPlan)=:plan
_accelerator_source_backend(source::CompiledPIModel)=source.backend
_accelerator_source_backend(source::SpecializedPIModel)=source.backend

_accelerator_existing_sparse(::LiouvillianPlan)=nothing
_accelerator_existing_sparse(source::CompiledPIModel)=
    source.backend===:sparse && source.operator isa SparseMatrixCSC ?
        source.operator : nothing
_accelerator_existing_sparse(source::SpecializedPIModel)=
    source.backend===:sparse && source.operator isa SparseMatrixCSC ?
        source.operator : nothing

function _accelerator_budget(value,name::AbstractString)
    value isa Real&&!(value isa Bool)||throw(ArgumentError(
        "$name must be a nonnegative real number of bytes or Inf"))
    isnan(value)&&throw(ArgumentError("$name cannot be NaN"))
    value>=0||throw(ArgumentError("$name must be nonnegative"))
    isfinite(value) ? floor(BigInt,value) : nothing
end

_accelerator_fits(bytes::Integer,budget)=
    budget===nothing||BigInt(bytes)<=budget

function _accelerator_push_issue(issues::Tuple,condition::Bool,issue::Symbol)
    condition ? issues : (issues...,issue)
end

function _accelerator_sparse_indices_supported(
        index_type::Type,dimension::Integer,retained_nnz::Integer)
    index_type<:Integer||return false
    dimension>=0&&retained_nnz>=0||return false
    limit=BigInt(typemax(index_type))
    # Julia CSC pointers are one based, so the final column pointer is
    # `nnz + 1`. The nonzero count itself fitting in the device index type is
    # not sufficient at the upper boundary.
    BigInt(dimension)<=limit&&BigInt(retained_nnz)+1<=limit
end

function _accelerator_rhs_kind(
        rhs_columns::Integer,requested::Symbol=:auto)
    requested in (:auto,:vector,:matrix)||throw(ArgumentError(
        "rhs_kind must be :auto, :vector, or :matrix"))
    if requested===:vector
        rhs_columns==1||throw(ArgumentError(
            "rhs_kind=:vector requires rhs_columns=1"))
        return :vector
    end
    requested===:matrix&&return :matrix
    rhs_columns==1 ? :vector : :matrix
end

_accelerator_rhs_kind_supported(capability,rhs_kind::Symbol)=
    rhs_kind in capability.rhs_kinds

"""
    accelerator_preflight(source; backend=:cuda, rhs_columns=1,
                          rhs_kind=:auto,
                          memory_budget=512*1024^2,
                          device_memory_budget=512*1024^2,
                          bigfloat_precision=precision(BigFloat))

Estimate the simultaneous host and accelerator memory required to upload and
apply an exact-support sparse PI Liouvillian. `source` must be an already
prepared [`LiouvillianPlan`](@ref), [`CompiledPIModel`](@ref), or
[`SpecializedPIModel`](@ref). Requiring prepared input keeps this query from
silently constructing geometry.

The report uses the actual sparse nonzero count when `source` already owns a
sparse operator. Otherwise it uses the same exact-support upper bound as the
guarded sparse PI materializer. No matrix is materialized, no device is
initialized, and no memory is allocated in proportion to the PI dimension.

The intended CUDA contract is narrow: autonomous sources, `Float32`,
`Float64`, `ComplexF32`, or `ComplexF64`, 32-bit sparse indices, and vector or
matrix right-hand sides already resident on the device. General matrix-free
Schur kernels and implicit per-action transfers are not supported.
"""
function accelerator_preflight(
        source::Union{LiouvillianPlan,CompiledPIModel,SpecializedPIModel};
        backend::Symbol=:cuda,
        rhs_columns::Integer=1,
        rhs_kind::Symbol=:auto,
        memory_budget=512*1024^2,
        device_memory_budget=512*1024^2,
        bigfloat_precision::Integer=precision(BigFloat))
    rhs_columns isa Bool&&throw(ArgumentError(
        "rhs_columns must be a positive integer, not a Bool"))
    rhs_columns>0||throw(ArgumentError(
        "rhs_columns must be a positive integer"))
    rhs_columns<=typemax(Int)||throw(ArgumentError(
        "rhs_columns exceeds the addressable Int range"))
    selected_rhs_kind=_accelerator_rhs_kind(rhs_columns,rhs_kind)
    2<=bigfloat_precision<=typemax(Int)||throw(ArgumentError(
        "bigfloat_precision must lie between 2 and typemax(Int)"))

    combined_budget=_accelerator_budget(memory_budget,"memory_budget")
    device_budget=_accelerator_budget(
        device_memory_budget,"device_memory_budget")
    capability=accelerator_capability(backend)
    plan=_accelerator_plan(source)
    basis=plan.basis
    dimension=length(basis)
    scalar_type=plan.Ttype
    precision_assumption=_scalar_precision_assumption(
        scalar_type,bigfloat_precision)
    existing=_accelerator_existing_sparse(source)
    materialization_required=existing===nothing
    sparse_materialization_supported=
        !materialization_required||plan.kernels!==nothing

    bounds=_performance_sparse_materialization_bounds(
        plan;bigfloat_precision)
    exact_nnz=existing===nothing ? nothing : nnz(existing)
    retained_nnz=exact_nnz===nothing ?
        bounds.retained_nnz_upper_bound : BigInt(exact_nnz)
    scalar_bytes=_scalar_retained_bytes(
        scalar_type;bigfloat_precision)
    host_operator_bytes=exact_nnz===nothing ? bounds.operator_bytes :
        BigInt(Base.summarysize(existing))
    source_bytes=BigInt(Base.summarysize(source))
    host_materialization_peak=materialization_required ?
        source_bytes+bounds.peak_bytes : source_bytes

    index_bytes=BigInt(sizeof(capability.index_type))
    device_operator_bytes=retained_nnz*(scalar_bytes+index_bytes)+
        (BigInt(dimension)+1)*index_bytes
    vector_entries=2BigInt(dimension)*BigInt(rhs_columns)
    device_vector_bytes=vector_entries*scalar_bytes
    device_peak=device_operator_bytes+device_vector_bytes
    combined_peak=host_materialization_peak+device_peak
    fits_combined=_accelerator_fits(combined_peak,combined_budget)
    fits_device=_accelerator_fits(device_peak,device_budget)

    autonomous=isautonomous(source)
    scalar_supported=scalar_type in capability.scalar_types
    # A CSC backend with integer indices must represent every pointer and row
    # coordinate. Check the conservative support count before any conversion.
    index_supported=_accelerator_sparse_indices_supported(
        capability.index_type,dimension,retained_nnz)
    rhs_supported=_accelerator_rhs_kind_supported(
        capability,selected_rhs_kind)
    transfer_supported=capability.transfer_policy===:explicit_once

    issues=()
    issues=_accelerator_push_issue(
        issues,capability.functional,:backend_unavailable)
    issues=_accelerator_push_issue(
        issues,capability.supports_sparse_pi,:sparse_pi_unsupported)
    issues=_accelerator_push_issue(
        issues,sparse_materialization_supported,
        :sparse_materialization_unsupported)
    issues=_accelerator_push_issue(
        issues,autonomous,:nonautonomous_source)
    issues=_accelerator_push_issue(
        issues,scalar_supported,:unsupported_scalar_type)
    issues=_accelerator_push_issue(
        issues,index_supported,:sparse_index_overflow)
    issues=_accelerator_push_issue(
        issues,rhs_supported,:rhs_kind_unsupported)
    issues=_accelerator_push_issue(
        issues,transfer_supported,:unsupported_transfer_policy)
    issues=_accelerator_push_issue(
        issues,fits_combined,:memory_budget_exceeded)
    issues=_accelerator_push_issue(
        issues,fits_device,:device_memory_budget_exceeded)
    ready=isempty(issues)

    AcceleratorPreflight(
        backend,
        basis,
        scalar_type,
        precision_assumption,
        dimension,
        length(basis.sectors),
        Int(rhs_columns),
        selected_rhs_kind,
        _accelerator_source_backend(source),
        autonomous,
        sparse_materialization_supported,
        materialization_required,
        exact_nnz,
        retained_nnz,
        source_bytes,
        host_operator_bytes,
        host_materialization_peak,
        device_operator_bytes,
        device_vector_bytes,
        device_peak,
        combined_peak,
        combined_budget,
        device_budget,
        fits_combined,
        fits_device,
        capability,
        issues,
        ready,
    )
end

function accelerator_preflight(source;kwargs...)
    throw(ArgumentError(
        "accelerator_preflight requires an already prepared LiouvillianPlan, " *
        "CompiledPIModel, or SpecializedPIModel; call compile first"))
end

function _accelerator_unavailable_error(
        backend::Symbol,report::AcceleratorPreflight)
    detail=isempty(report.issues) ? "no implementation is registered" :
        join(string.(report.issues),", ")
    throw(ArgumentError(
        "accelerator backend :$backend is not ready ($detail). " *
        report.capability.message))
end

function _accelerate(::Val{backend},source,report;kwargs...) where backend
    _accelerator_unavailable_error(backend,report)
end

"""
    accelerate(source; backend=:cuda, rhs_columns=1, rhs_kind=:auto, ...)

Prepare an optional accelerated sparse PI Liouvillian. The core method always
runs [`accelerator_preflight`](@ref) first and rejects unavailable,
nonfunctional, incompatible, or over-budget backends before materialization.
An optional backend extension must then perform one explicit sparse upload and
return an operator whose multiplication accepts device-resident vector and
matrix right-hand sides without hidden transfers.

No CUDA implementation is claimed by the core package. This is intentional:
loading CUDA.jl without a supported functional device and GPU test coverage is
not enough to establish a reliable numerical backend.
"""
function accelerate(
        source;
        backend::Symbol=:cuda,
        rhs_columns::Integer=1,
        rhs_kind::Symbol=:auto,
        memory_budget=512*1024^2,
        device_memory_budget=512*1024^2,
        bigfloat_precision::Integer=precision(BigFloat),
        kwargs...)
    report=accelerator_preflight(
        source;
        backend,
        rhs_columns,
        rhs_kind,
        memory_budget,
        device_memory_budget,
        bigfloat_precision,
    )
    report.ready||_accelerator_unavailable_error(backend,report)
    _accelerate(Val(backend),source,report;kwargs...)
end
