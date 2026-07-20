"""
    CompiledPIModelFamily

Reusable compilation of a fixed-operator [`PIModel`](@ref) whose selected
scalar term rates will be varied. The family owns one read-only
[`LiouvillianPlan`](@ref); [`specialize`](@ref) binds numerical rates without
rebuilding Schur geometry. Construct families with [`compile_family`](@ref).

A family is not itself a Liouvillian. Share it freely between tasks, then give
each task its own specialization or an explicit [`LiouvillianWorkspace`](@ref).
"""
struct CompiledPIModelFamily{M,P,I,R,E}
    model::M
    plan::P
    rate_indices::I
    default_rates::R
    estimates::E
end

"""
    SpecializedPIModel

An autonomous PI model obtained by binding numerical rates to a
[`CompiledPIModelFamily`](@ref). It behaves like a compiled model in
`apply!`, `steady_state`, spectral routines, dynamics, and parameter scans,
while retaining the family's prepared Schur geometry.
"""
struct SpecializedPIModel{F,M,P,O,R,E}
    family::F
    model::M
    plan::P
    operator::O
    rates::R
    backend::Symbol
    estimates::E
end

function _performance_source_action_bytes(source::SpecializedPIModel,
        ::Type{T}) where T
    source.backend===:sparse ? big(0) :
        _performance_liouvillian_fallback_bytes(source.plan)
end

struct _FamilyRate{I} <: Function end
@inline (::_FamilyRate{I})(time,parameters) where I=parameters[I]

function _family_rate_indices(model::PIModel,rate_indices)
    nterms=length(model.terms)
    indices=rate_indices===nothing ? collect(1:nterms) : collect(rate_indices)
    isempty(indices)&&throw(ArgumentError("rate_indices cannot be empty"))
    all(index->index isa Integer&&!(index isa Bool),indices)||
        throw(ArgumentError("rate_indices must contain integers"))
    converted=Int[index for index in indices]
    all(index->1<=index<=nterms,converted)||throw(BoundsError(model.terms,converted))
    allunique(converted)||throw(ArgumentError("rate_indices must be unique"))
    sort!(converted)
    Tuple(converted)
end

function _family_rebuild_rate(term::AbstractPITerm,rate)
    rebuild_term(term,term_operator(term),rate)
end
function _family_rebuild_rate(term::CorrelatedLocalJumps,rate)
    CorrelatedLocalJumps(term.operators,term.kossakowski,term.factor,rate,
                         term.atol,term.rtol)
end
function _family_rebuild_rate(term::CorrelatedCollectiveJumps,rate)
    CorrelatedCollectiveJumps(term.operators,term.kossakowski,term.factor,rate,
                              term.atol,term.rtol)
end

function _family_replace_rates(model::PIModel,indices,rates)
    terms=ntuple(length(model.terms)) do index
        position=findfirst(==(index),indices)
        position===nothing ? model.terms[index] :
            _family_rebuild_rate(model.terms[index],rates[position])
    end
    PIModel(model.basis,terms)
end

function _family_parameterized_model(model::PIModel,indices)
    rates=ntuple(position->_FamilyRate{position}(),length(indices))
    _family_replace_rates(model,indices,rates)
end

function _family_estimates(plan::LiouvillianPlan,indices,bigfloat_precision)
    n=length(plan.basis);T=plan.Ttype
    scalar_bytes=_scalar_retained_bytes(T;bigfloat_precision)
    plan_bytes=Base.summarysize(plan)
    # A synchronized matrix-free specialization owns the complete prepared
    # kernel workspace plus one copied trace vector. Sparse specialization can
    # use the exact-support contribution bound of the already prepared standard
    # kernels; unknown kernels retain that helper's dense-coordinate fallback.
    workspace_bytes=_performance_liouvillian_workspace_bytes(
        plan;bigfloat_precision)
    matrixfree_bytes=workspace_bytes+BigInt(n)*scalar_bytes
    sparse_bounds=_performance_sparse_materialization_bounds(
        plan;bigfloat_precision)
    sparse_operator_bytes=sparse_bounds.operator_bytes
    sparse_peak_bytes=sparse_bounds.peak_bytes
    (;scalar_type=T,dimension=n,plan_bytes,
      shared_plan_bytes=plan_bytes,rate_indices=indices,
      scalar_retained_bytes=scalar_bytes,
      scalar_storage_estimate=_scalar_storage_estimate(T),
      bigfloat_precision_assumption=
          _scalar_precision_assumption(T,bigfloat_precision),
      matrixfree_workspace_upper_bound=workspace_bytes,
      matrixfree_specialization_estimate=Int(min(matrixfree_bytes,
          BigInt(typemax(Int)))),
      matrixfree_specialization_upper_bound=matrixfree_bytes,
      sparse_operator_upper_bound=Int(min(sparse_operator_bytes,
          BigInt(typemax(Int)))),
      sparse_structure_supported=sparse_bounds.structured,
      sparse_contribution_upper_bound=Int(min(
          sparse_bounds.contribution_upper_bound,BigInt(typemax(Int)))),
      sparse_retained_nnz_upper_bound=Int(min(
          sparse_bounds.retained_nnz_upper_bound,BigInt(typemax(Int)))),
      sparse_assembly_upper_bound=Int(min(
          sparse_bounds.assembly_bytes,BigInt(typemax(Int)))),
      sparse_specialization_peak_upper_bound=sparse_peak_bytes,
      geometry_reused=true)
end

"""
    compile_family(model; rate_indices=nothing,
                   bigfloat_precision=precision(BigFloat),
                   coefficient_cache=nothing)

Compile fixed operator geometry once for a family of autonomous models that
differ only in selected scalar term rates. `rate_indices=nothing` varies every
term rate; otherwise it is a nonempty collection of one-based term indices.
Call [`specialize`](@ref) for each numerical parameter point.

The input model must be autonomous and every selected rate must be numeric.
Operators, body order, Kossakowski matrices, and the retained PI basis remain
fixed. This strict contract is what makes specialization independent of Schur
geometry construction. Negative time-local rates are preserved.

Pass a compatible [`OneBoxCGCache`](@ref) to reuse one-box
Clebsch--Gordan coefficients during the single shared geometry preparation.
When omitted, the same bounded small-model automatic sharing used by
[`compile`](@ref) applies.
"""
function compile_family(model::PIModel;rate_indices=nothing,
        bigfloat_precision::Integer=precision(BigFloat),
        coefficient_cache=nothing)
    isautonomous(model)||throw(ArgumentError(
        "compile_family requires an autonomous fixed-operator prototype"))
    indices=_family_rate_indices(model,rate_indices)
    defaults=ntuple(position->term_rate(model.terms[indices[position]]),
                     length(indices))
    all(rate->rate isa Number,defaults)||throw(ArgumentError(
        "every selected family rate must be numeric in the prototype"))
    parameterized=_family_parameterized_model(model,indices)
    plan=LiouvillianPlan(parameterized;coefficient_cache)
    plan.kernels===nothing&&throw(ArgumentError(
        "compile_family supports only fixed-operator terms that lower to prepared kernels"))
    estimates=_family_estimates(plan,indices,bigfloat_precision)
    CompiledPIModelFamily(model,plan,indices,defaults,estimates)
end

function _family_rates(family::CompiledPIModelFamily,rates)
    count=length(family.rate_indices)
    values = if rates isa Number
        count==1||throw(ArgumentError(
            "a scalar rate is valid only for a family with one selected term"))
        (rates,)
    else
        Tuple(rates)
    end
    length(values)==count||throw(DimensionMismatch(
        "expected $count family rates, got $(length(values))"))
    for (position,value) in pairs(values)
        value isa Number||throw(ArgumentError(
            "family rate $position must be numeric"))
        isfinite(real(value))&&isfinite(imag(value))||throw(ArgumentError(
            "family rate $position must be finite"))
    end
    values
end

function _family_matrixfree_operator(plan,rates,workspace=nothing)
    work=workspace===nothing ? LiouvillianWorkspace(plan) : workspace
    _check_liouvillian_workspace(work,plan)
    action! = (y,x,t,p)->apply!(y,plan,x,zero(t),rates,work)
    adjoint_action! = (y,x,t,p)->apply_adjoint!(y,plan,x,zero(t),rates,work)
    batched_action! = (Y,X,t,p)->apply!(Y,plan,X,zero(t),rates,work)
    batched_adjoint_action! =
        (Y,X,t,p)->apply_adjoint!(Y,plan,X,zero(t),rates,work)
    MatrixFreeLiouvillian(length(plan.basis),action!,plan.Ttype,
        copy(plan.tracevec);autonomous=true,workspace=work,
        adjoint_action!,batched_action!,batched_adjoint_action!)
end

_family_kernel_scale(kernel::HamiltonianPIKernel,rates)=
    value_at(kernel.scale,0.0,rates)
_family_kernel_scale(kernel::Union{DissipatorPIKernel,LocalJumpPIKernel,
        FactorizedLocalJumpPIKernel,FactorizedLocalPBodyJumpPIKernel},rates)=
    _evaluated_dissipative_rate(kernel.scale,0.0,rates)
_family_kernel_scale(::FusedStaticPIKernel,rates)=1

_family_add_kernel_matrices(matrix,plan,::Tuple{},rates)=matrix
function _family_add_kernel_matrices(matrix,plan,
        kernels::Tuple{K,Vararg{Any}},rates) where K
    kernel=first(kernels)
    scale=convert(plan.Ttype,_family_kernel_scale(kernel,rates))
    _family_add_kernel_matrices(
        matrix+_kernel_matrix(plan,kernel,scale),plan,Base.tail(kernels),rates)
end

function _family_matrix_from_plan(plan::LiouvillianPlan,rates)
    plan.kernels===nothing&&throw(ArgumentError(
        "a family plan without prepared kernels cannot be materialized"))
    n=length(plan.basis)
    _family_add_kernel_matrices(spzeros(plan.Ttype,n,n),plan,plan.kernels,rates)
end

"""
    specialize(family[, rates]; backend=:matrixfree, workspace=nothing,
               memory_budget=512*1024^2)

Bind numerical rates to a [`CompiledPIModelFamily`](@ref) without rebuilding
Schur geometry. `rates` follows `family.rate_indices` order; a scalar is
accepted when exactly one rate is selected. Omitting `rates` uses the
prototype values.

`backend=:matrixfree` creates only a synchronized compatibility workspace;
hot or parallel loops should pass an explicit `LiouvillianWorkspace` to
`apply!`. Repeated sequential specializations may pass the same task-owned
`workspace`; this is how family parameter scans avoid allocating Schur-block
scratch at every point. Never share that mutable workspace concurrently.
`backend=:sparse` materializes the bound linear combination from the already
prepared kernels. `backend=:auto` selects sparse only when its conservative
live-materialization bound fits `memory_budget`; otherwise it selects the
matrix-free route. An explicit sparse request that exceeds the budget raises
before allocating the sparse operator. The matrix-free bound includes its
compatibility workspace and copied trace vector but excludes the already
retained shared family plan. Pass `memory_budget=Inf` only as an explicit
opt-out. Both backends are autonomous.
"""
function specialize(family::CompiledPIModelFamily,
        rates=family.default_rates;backend::Symbol=:matrixfree,workspace=nothing,
        memory_budget=512*1024^2)
    backend in (:auto,:matrixfree,:sparse)||throw(ArgumentError(
        "backend must be :auto, :matrixfree, or :sparse"))
    budget=_memory_budget_bytes(memory_budget)
    sparse_peak=family.estimates.sparse_specialization_peak_upper_bound
    matrixfree_peak=family.estimates.matrixfree_specialization_upper_bound
    chosen = if backend===:auto
        workspace===nothing&&_performance_budget_fits(
            sparse_peak,memory_budget) ? :sparse : :matrixfree
    else
        backend
    end
    chosen===:sparse&&workspace!==nothing&&throw(ArgumentError(
        "workspace is supported only with backend=:matrixfree"))
    estimated_peak=chosen===:sparse ? sparse_peak : matrixfree_peak
    _require_performance_budget("specialize(...; backend=$chosen)",
        estimated_peak,memory_budget;guidance=
        "Use backend=:matrixfree or increase the family specialization budget.")
    bound=_family_rates(family,rates)
    model=_family_replace_rates(family.model,family.rate_indices,bound)
    compatibility_workspace = chosen===:matrixfree ?
        (workspace===nothing ? LiouvillianWorkspace(family.plan) : workspace) :
        nothing
    operator=chosen===:matrixfree ? _family_matrixfree_operator(
        family.plan,bound,compatibility_workspace) :
        _family_matrix_from_plan(family.plan,bound)
    owned_workspace_bytes=chosen===:matrixfree&&workspace===nothing ?
        Base.summarysize(compatibility_workspace) : 0
    estimates=merge(family.estimates,
        (;requested_backend=backend,chosen_backend=chosen,geometry_reused=true,
          specialization=:scalar_rate_binding,
          compatibility_workspace_reused=workspace!==nothing,
          specialization_workspace_bytes=owned_workspace_bytes,
          specialization_rate_bytes=Base.summarysize(bound),
          memory_budget=budget,estimated_additional_peak_bytes=estimated_peak,
          budget_status=_performance_memory_limit(memory_budget)===nothing ?
              :disabled : :fits))
    SpecializedPIModel(family,model,family.plan,operator,bound,chosen,estimates)
end

size(model::SpecializedPIModel)=size(model.operator)
size(model::SpecializedPIModel,index::Integer)=size(model.operator,index)
eltype(model::SpecializedPIModel)=eltype(model.operator)
isautonomous(::SpecializedPIModel)=true

LiouvillianWorkspace(model::SpecializedPIModel)=LiouvillianWorkspace(model.plan)

function apply!(y,model::SpecializedPIModel,x,t,p,
                work::LiouvillianWorkspace)
    apply!(y,model.plan,x,t,model.rates,work)
end
function apply!(Y::AbstractMatrix,model::SpecializedPIModel,X::AbstractMatrix,
                t,p,work::LiouvillianWorkspace)
    apply!(Y,model.plan,X,t,model.rates,work)
end
function apply!(y,model::SpecializedPIModel,x,t,p)
    model.backend===:matrixfree ? apply!(y,model.operator,x,t,p) :
        mul!(y,model.operator,x)
end
apply!(y,model::SpecializedPIModel,x)=mul!(y,model,x)

function apply_adjoint!(y,model::SpecializedPIModel,x,t,p,
                        work::LiouvillianWorkspace)
    apply_adjoint!(y,model.plan,x,t,model.rates,work)
end
function apply_adjoint!(Y::AbstractMatrix,model::SpecializedPIModel,
                        X::AbstractMatrix,t,p,work::LiouvillianWorkspace)
    apply_adjoint!(Y,model.plan,X,t,model.rates,work)
end
function apply_adjoint!(y,model::SpecializedPIModel,x,t,p)
    model.backend===:matrixfree ?
        apply_adjoint!(y,model.operator,x,t,p) :
        mul!(y,adjoint(model.operator),x)
end
apply_adjoint!(y,model::SpecializedPIModel,x)=
    apply_adjoint!(y,model,x,0.0,nothing)

function mul!(y,model::SpecializedPIModel,x)
    mul!(y,model.operator,x)
end
function mul!(Y::AbstractMatrix,model::SpecializedPIModel,X::AbstractMatrix)
    mul!(Y,model.operator,X)
end
*(model::SpecializedPIModel,x::AbstractVector)=
    mul!(_product_destination(model,x,size(model,1)),model,x)
*(model::SpecializedPIModel,X::AbstractMatrix)=
    mul!(_product_destination(model,X,size(model,1),size(X,2)),model,X)

_liouvillian_action!(y,model::SpecializedPIModel,x,t,p)=
    apply!(y,model,x,t,p)
_fixed_liouvillian_scalar_type(model::SpecializedPIModel)=model.plan.Ttype
_materialize(model::SpecializedPIModel)=
    model.backend===:sparse ? model.operator :
        _family_matrix_from_plan(model.plan,model.rates)

function liouvillian(model::SpecializedPIModel;
                     representation::Symbol=model.backend,
                     memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    representation in (:matrixfree,:sparse)||throw(ArgumentError(
        "representation must be :matrixfree or :sparse"))
    representation===model.backend&&return model.operator
    if representation===:sparse
        _require_performance_budget(
            "specialized-family sparse materialization",
            model.family.estimates.sparse_specialization_peak_upper_bound,
            memory_budget;guidance="Keep representation=:matrixfree.")
    else
        _require_performance_budget(
            "specialized-family matrix-free workspace",
            model.family.estimates.matrixfree_specialization_upper_bound,
            memory_budget;guidance="Increase the explicit budget.")
    end
    representation===:matrixfree ?
        _family_matrixfree_operator(model.plan,model.rates) :
        _family_matrix_from_plan(model.plan,model.rates)
end

function steady_state(model::SpecializedPIModel;method=:auto,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,kwargs...)
    method isa Symbol||throw(ArgumentError("method must be a Symbol"))
    method=_stationary_solver_method(method)
    representation=method===:krylov ? :matrixfree : model.backend
    source=liouvillian(model;representation,memory_budget)
    steady_state(source;basis=model.plan.basis,trace_vector=model.plan.tracevec,
                 method,memory_budget,kwargs...)
end

function freeze(model::SpecializedPIModel;time=0.0,parameters=nothing,
                representation::Symbol=model.backend,
                memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    liouvillian(model;representation,memory_budget)
end

function pi_liouvillian_spectrum(model::SpecializedPIModel;method=:dense,
        basis=model.plan.basis,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,kwargs...)
    method isa Symbol||throw(ArgumentError("method must be a Symbol"))
    method=_canonical_spectrum_algorithm(method)
    representation=method in (:arnoldi,:block_arnoldi,:harmonic,:iram,:jd) ?
        :matrixfree : model.backend
    pi_liouvillian_spectrum(
        liouvillian(model;representation,memory_budget);
        method,basis,memory_budget,kwargs...)
end

function pi_liouvillian_gap(model::SpecializedPIModel;method=:dense,
        basis=model.plan.basis,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,kwargs...)
    method isa Symbol||throw(ArgumentError("method must be a Symbol"))
    method=_canonical_spectrum_algorithm(method)
    representation=method in (:arnoldi,:block_arnoldi,:harmonic,:iram,:jd) ?
        :matrixfree : model.backend
    pi_liouvillian_gap(liouvillian(model;representation,memory_budget);
                       method,basis,memory_budget,kwargs...)
end
