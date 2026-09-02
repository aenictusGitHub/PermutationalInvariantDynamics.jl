# No-jump-resolvent iterative methods ----------------------------------------

# The algorithms in this file implement the jump/no-jump splitting described
# by Beugnot, Gregory, Robin, and Tilloy without leaving PI coordinates.  A PI
# Hilbert-space operator is block diagonal in the Schur basis, so the effective
# no-jump generator G has one physical block per retained sector.  Inverting
# lambda-S, with S(X)=G*X+X*G', therefore needs only independent sector-sized
# Lyapunov solves rather than a PI-coordinate or d^N superoperator.

abstract type _AbstractNoJumpSectorFactorization end

struct _NoJumpSchurSector{M,V} <: _AbstractNoJumpSectorFactorization
    triangular::M
    vectors::M
    values::V
end

struct _NoJumpEigenSector{M,V,R} <: _AbstractNoJumpSectorFactorization
    vectors::M
    inverse_vectors::M
    values::V
    condition_number::R
end

"""
    NoJumpResolventPlan(source; backend=:schur, memory_budget=512*1024^2,
                         condition_limit=nothing)

Immutable sectorwise factorization of the effective no-jump generator
``G=-im*H-sum_j L_j'*L_j/2`` of a fixed PI Lindblad generator. `source` may
be a [`PIModel`](@ref), [`CompiledPIModel`](@ref), [`LiouvillianPlan`](@ref),
[`SpecializedPIModel`](@ref), or [`NoJumpIterativePlan`](@ref). A specialization uses
its bound rates with the family's already prepared geometry.

`backend=:schur` uses a unitary Schur factorization and a preallocated
Bartels--Stewart recurrence. It is the robust default and also supports
defective `G` blocks. `backend=:eigen` implements the four-GEMM
eigendecomposition route used in the no-jump-resolvent iterative algorithm; it rejects singular or
overly ill-conditioned eigenvector matrices instead of returning an
uncertified inverse. Set `condition_limit=Inf` only to opt out of the finite
conditioning threshold explicitly.

Only fixed GKSL generators with Hermitian Hamiltonians and finite nonnegative
jump rates are accepted. For a driven `PIModel`, pass an explicit finite
`time` and optional `parameters`; the constructor freezes every physical term
before lowering so the jump/no-jump split is retained. Evaluation keywords are
rejected for autonomous or already prepared sources. The plan never forms a
`d^N` object or a PI-coordinate inverse.
"""
struct NoJumpResolventPlan{B,G,F,T,M}
    basis::B
    generator_blocks::G
    factors::F
    Ttype::Type{T}
    metadata::M
end

size(plan::NoJumpResolventPlan)=(length(plan.basis),length(plan.basis))
size(plan::NoJumpResolventPlan,index::Integer)=
    index in (1,2) ? length(plan.basis) : 1
eltype(plan::NoJumpResolventPlan)=plan.Ttype

"""
    NoJumpResolventWorkspace(plan; memory_budget=512*1024^2)

Task-owned matrix scratch for [`no_jump_resolvent!`](@ref). It retains two
matrices per Schur sector and is tied to the exact plan basis and scalar type.
"""
struct NoJumpResolventWorkspace{B,W,T}
    basis::B
    blocks::W
    Ttype::Type{T}
    accounted_peak_bytes::BigInt
end

function _no_jump_iterative_add_scaled_matrix!(destination,source,scale)
    @inbounds for column in axes(source,2),row in axes(source,1)
        destination[row,column]+=scale*source[row,column]
    end
    destination
end

function _no_jump_iterative_add_scaled_matrix!(destination,source::SparseMatrixCSC,scale)
    @inbounds for column in axes(source,2)
        for pointer in nzrange(source,column)
            destination[source.rowval[pointer],column]+=
                scale*source.nzval[pointer]
        end
    end
    destination
end

function _no_jump_iterative_jump_scale(kernel,parameters)
    scale=_family_kernel_scale(kernel,parameters)
    scale>=0||throw(ArgumentError(
        "no-jump-resolvent iterative CPTP and contraction methods require nonnegative jump rates; " *
        "freeze or rebuild the model with a nonnegative GKSL rate"))
    scale
end

function _no_jump_iterative_check_tolerances(atol,rtol,::Type{R}) where R
    converted_atol,converted_rtol=
        _advanced_check_tolerances(atol,rtol,R)
    !iszero(atol)&&iszero(converted_atol)&&throw(ArgumentError(
        "absolute tolerance is nonzero but underflows in prepared " *
        "precision $R; use a wider scalar type or a representable tolerance"))
    !iszero(rtol)&&iszero(converted_rtol)&&throw(ArgumentError(
        "relative tolerance is nonzero but underflows in prepared " *
        "precision $R; use a wider scalar type or a representable tolerance"))
    converted_atol,converted_rtol
end

function _no_jump_iterative_require_hermitian_blocks(blocks,description)
    for (sector,block) in pairs(blocks)
        ishermitian(block)||throw(ArgumentError(
            "$description contains a non-Hermitian Hamiltonian block in " *
            "Schur sector $sector. no-jump-resolvent iterative CPTP/contraction guarantees " *
            "require a Hermitian Hamiltonian, including when term " *
            "constructor checks were disabled."))
    end
    blocks
end

_no_jump_iterative_accumulate_hamiltonian!(blocks,kernel,parameters)=blocks

function _no_jump_iterative_accumulate_hamiltonian!(blocks,
        kernel::HamiltonianPIKernel,parameters)
    scale=_family_kernel_scale(kernel,parameters)
    @inbounds for sector in eachindex(blocks)
        _no_jump_iterative_add_scaled_matrix!(blocks[sector],kernel.blocks[sector],scale)
    end
    blocks
end

function _no_jump_iterative_accumulate_hamiltonian!(blocks,
        kernel::FusedStaticPIKernel,parameters)
    kernel.hamiltonian_blocks===nothing&&return blocks
    @inbounds for sector in eachindex(blocks)
        _no_jump_iterative_add_scaled_matrix!(
            blocks[sector],kernel.hamiltonian_blocks[sector],one(eltype(
                kernel.hamiltonian_blocks[sector])))
    end
    blocks
end

function _no_jump_iterative_accumulate_generator!(blocks,kernel::HamiltonianPIKernel,
        parameters)
    scale=_family_kernel_scale(kernel,parameters)
    @inbounds for sector in eachindex(blocks)
        _no_jump_iterative_add_scaled_matrix!(blocks[sector],kernel.blocks[sector],
                                   -1im*scale)
    end
    0
end

function _no_jump_iterative_accumulate_loss!(blocks,qblocks,scale)
    @inbounds for sector in eachindex(blocks)
        _no_jump_iterative_add_scaled_matrix!(blocks[sector],qblocks[sector],-scale/2)
    end
    blocks
end

function _no_jump_iterative_accumulate_generator!(blocks,kernel::DissipatorPIKernel,
        parameters)
    scale=_no_jump_iterative_jump_scale(kernel,parameters)
    _no_jump_iterative_accumulate_loss!(blocks,kernel.qblocks,scale)
    1
end

function _no_jump_iterative_accumulate_generator!(blocks,kernel::Union{
        LocalJumpPIKernel,FactorizedLocalJumpPIKernel,
        FactorizedLocalPBodyJumpPIKernel},parameters)
    scale=_no_jump_iterative_jump_scale(kernel,parameters)
    _no_jump_iterative_accumulate_loss!(blocks,kernel.qblocks,scale)
    1
end

function _no_jump_iterative_check_fused_gain_scale(scale)
    scale isa Number&&isfinite(real(scale))&&isfinite(imag(scale))&&
        iszero(imag(scale))&&real(scale)>=0||throw(ArgumentError(
        "no-jump-resolvent iterative CPTP and contraction methods require finite nonnegative " *
        "jump rates in every fused gain channel"))
    real(scale)
end

function _no_jump_iterative_accumulate_generator!(blocks,kernel::FusedStaticPIKernel,
        parameters)
    if kernel.hamiltonian_blocks!==nothing
        @inbounds for sector in eachindex(blocks)
            _no_jump_iterative_add_scaled_matrix!(blocks[sector],
                kernel.hamiltonian_blocks[sector],-1im)
        end
    end
    if kernel.loss_blocks!==nothing
        @inbounds for sector in eachindex(blocks)
            _no_jump_iterative_add_scaled_matrix!(blocks[sector],
                kernel.loss_blocks[sector],-1/2)
        end
    end
    for gain in kernel.collective_gains
        _no_jump_iterative_check_fused_gain_scale(gain.scale)
    end
    for gain in kernel.onebody_gains
        _no_jump_iterative_check_fused_gain_scale(gain.scale)
    end
    for gain in kernel.pbody_gains
        _no_jump_iterative_check_fused_gain_scale(gain.scale)
    end
    length(kernel.collective_gains)+length(kernel.onebody_gains)+
        length(kernel.pbody_gains)
end

function _no_jump_iterative_accumulate_generator!(blocks,kernel,parameters)
    throw(ArgumentError(
        "compiled kernel $(typeof(kernel)) does not expose the fixed GKSL " *
        "jump/no-jump split required by the no-jump-resolvent iterative method; custom terms must " *
        "delegate compile_term to a supported built-in term"))
end

_no_jump_iterative_supported_kernel(::Union{
    HamiltonianPIKernel,DissipatorPIKernel,LocalJumpPIKernel,
    FactorizedLocalJumpPIKernel,FactorizedLocalPBodyJumpPIKernel,
    FusedStaticPIKernel})=true
_no_jump_iterative_supported_kernel(::Any)=false

function _no_jump_iterative_check_supported_kernels(kernels)
    for kernel in kernels
        _no_jump_iterative_supported_kernel(kernel)||throw(ArgumentError(
            "compiled kernel $(typeof(kernel)) does not expose the fixed " *
            "GKSL jump/no-jump split required by the no-jump-resolvent iterative method; custom " *
            "terms must delegate compile_term to a supported built-in term"))
    end
    nothing
end

function _no_jump_iterative_generator_blocks(plan::LiouvillianPlan,parameters=nothing)
    T=plan.Ttype
    blocks=[zeros(T,length(plan.basis.patterns[sector]),
                    length(plan.basis.patterns[sector]))
            for sector in eachindex(plan.basis.sectors)]
    hamiltonian_blocks=[zeros(T,size(block)) for block in blocks]
    channels=0
    for kernel in plan.kernels
        _no_jump_iterative_accumulate_hamiltonian!(
            hamiltonian_blocks,kernel,parameters)
        channels+=_no_jump_iterative_accumulate_generator!(blocks,kernel,parameters)
    end
    _no_jump_iterative_require_hermitian_blocks(
        hamiltonian_blocks,"the total prepared PI generator")
    blocks,channels
end

function _no_jump_iterative_factorization_error(backend,T)
    ArgumentError(
        "the $backend no-jump factorization is unavailable for scalar type " *
        "$T with Julia's active LinearAlgebra backend; use Float32/Float64 " *
        "data or load a compatible generic factorization backend")
end

function _no_jump_iterative_check_factorization_precision(::Type{T}) where T
    R=_real_float_type(T)
    R===BigFloat&&throw(ArgumentError(
        "no-jump-resolvent iterative no-jump plans do not currently support BigFloat: Julia's " *
        "core LinearAlgebra does not provide the required general complex " *
        "Schur/eigen factorization with a package-controlled precision and " *
        "rounding context. Use Float32/Float64, or an ordinary solver that " *
        "supports BigFloat."))
    nothing
end

function _no_jump_iterative_schur_factor(block)
    decomposition=try
        schur(block)
    catch error
        error isa MethodError||rethrow()
        throw(_no_jump_iterative_factorization_error(:schur,eltype(block)))
    end
    triangular=Matrix(decomposition.T)
    vectors=Matrix(decomposition.Z)
    values=Vector(diag(triangular))
    _NoJumpSchurSector(triangular,vectors,values)
end

function _no_jump_iterative_eigen_factor(block,condition_limit)
    decomposition=try
        eigen(block)
    catch error
        error isa MethodError||rethrow()
        throw(_no_jump_iterative_factorization_error(:eigen,eltype(block)))
    end
    vectors=Matrix(decomposition.vectors)
    condition_number=cond(vectors)
    isfinite(condition_number)||throw(ArgumentError(
        "a no-jump generator block has a singular eigenvector matrix; use " *
        "backend=:schur for this defective or numerically defective block"))
    condition_number<=condition_limit||throw(ArgumentError(
        "a no-jump generator eigenvector matrix has condition number " *
        "$condition_number, above condition_limit=$condition_limit; use " *
        "backend=:schur or explicitly raise condition_limit"))
    inverse_vectors=try
        inv(vectors)
    catch error
        error isa SingularException||rethrow()
        throw(ArgumentError(
            "a no-jump generator block is not diagonalizable; use " *
            "backend=:schur"))
    end
    _NoJumpEigenSector(vectors,inverse_vectors,
                       Vector(decomposition.values),condition_number)
end

function _no_jump_iterative_plan_storage_estimate(basis,::Type{T}) where T
    n=BigInt(length(basis))
    largest=BigInt(maximum(length,basis.patterns;init=1))
    # Three retained block families plus conservative Schur/eigen setup
    # temporaries for the largest sector and one trace-one deflation vector.
    _performance_entries_bytes(4n+12largest^2,T)
end

function _no_jump_iterative_workspace_estimate(plan::NoJumpResolventPlan)
    _performance_entries_bytes(2BigInt(length(plan.basis)),plan.Ttype)
end

function _prepare_nojump_resolvent_plan(plan::LiouvillianPlan,parameters;
        require_autonomous::Bool,backend::Symbol=:schur,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,
        condition_limit=nothing,stability_atol::Real=0,
        stability_rtol=nothing)
    backend in (:schur,:eigen)||throw(ArgumentError(
        "no-jump backend must be :schur or :eigen"))
    require_autonomous&& !isautonomous(plan)&&throw(ArgumentError(
        "the no-jump-resolvent iterative no-jump split requires an autonomous fixed model; freeze " *
        "time-dependent rates and operators first"))
    plan.kernels===nothing&&throw(ArgumentError(
        "the no-jump-resolvent iterative no-jump split requires prepared built-in kernels"))
    _no_jump_iterative_check_supported_kernels(plan.kernels)
    T=plan.Ttype
    R=_real_float_type(T)
    # A prepared algorithm may not depend on the ambient BigFloat context.
    # General complex Schur/eigen factorization is not provided by Julia's
    # core LinearAlgebra for BigFloat, and accepting an opportunistically
    # loaded method here would leave all retained factors and mutable scratch
    # without a durable precision/rounding contract.  Reject this unsupported
    # route explicitly until the plan can own a generic factorization backend
    # and its arithmetic context end to end.
    _no_jump_iterative_check_factorization_precision(T)
    atol=_checked_prepared_real(stability_atol,R,"stability_atol")
    atol>=zero(R)||throw(ArgumentError(
        "stability_atol must be nonnegative"))
    rtol=stability_rtol===nothing ? R(64)*eps(R) : begin
        stability_rtol isa Real&&stability_rtol>=0||
            throw(ArgumentError(
                "stability_rtol must be finite and nonnegative"))
        _checked_prepared_real(stability_rtol,R,"stability_rtol")
    end
    limit=condition_limit===nothing ? inv(sqrt(eps(R))) : begin
        condition_limit isa Real&&condition_limit>0&&
            (isfinite(condition_limit)||condition_limit==Inf)||
            throw(ArgumentError(
                "condition_limit must be positive and finite or Inf"))
        condition_limit==Inf ? R(Inf) : _checked_prepared_real(
            condition_limit,R,"condition_limit";nonzero=true)
    end
    estimate=BigInt(Base.summarysize(plan))+
        _no_jump_iterative_plan_storage_estimate(plan.basis,T)
    _require_performance_budget("no-jump-resolvent iterative no-jump factorization",estimate,
        memory_budget;guidance="Reduce the retained basis or increase the budget.")
    blocks,jump_channels=_no_jump_iterative_generator_blocks(plan,parameters)
    factors=backend===:schur ? map(_no_jump_iterative_schur_factor,blocks) :
        map(block->_no_jump_iterative_eigen_factor(block,limit),blocks)
    spectral_abscissae=R[
        maximum(real,factor.values;init=-R(Inf)) for factor in factors]
    block_scales=R[max(norm(block,Inf),floatmin(R)) for block in blocks]
    stability_tolerances=R[
        atol+rtol*scale for scale in block_scales]
    strictly_stable=all(index->
        spectral_abscissae[index]<-stability_tolerances[index],
        eachindex(spectral_abscissae))
    condition_numbers=backend===:eigen ?
        R[factor.condition_number for factor in factors] : missing
    metadata=(backend,jump_channels,strictly_stable,spectral_abscissae,
        stability_tolerances,condition_numbers,condition_limit=limit,
        retained_coefficients=sum(length,blocks;init=0),
        largest_schur_dimension=maximum(size.(blocks,1);init=0),
        scaling=:sum_of_sector_cubes,
        generator_mode=:autonomous,
        unique_steady_state=:not_applicable)
    NoJumpResolventPlan(plan.basis,blocks,factors,T,metadata)
end

function _no_jump_iterative_reject_evaluation_arguments(source,time,parameters)
    time===nothing&&parameters===nothing&&return nothing
    throw(ArgumentError(
        "instantaneous `time`/`parameters` are accepted only with the " *
        "original PIModel, where every physical term can be frozen without " *
        "losing the jump/no-jump split; got $(typeof(source))"))
end

function _no_jump_iterative_instantaneous_model(model::PIModel,time,parameters)
    if isautonomous(model)
        time===nothing||throw(ArgumentError(
            "`time` is not used for an autonomous PIModel; omit it"))
        parameters===nothing||throw(ArgumentError(
            "`parameters` are not used for an autonomous PIModel; omit them"))
        return model,nothing
    end
    time===nothing&&throw(ArgumentError(
        "a driven PIModel requires an explicit finite real `time` for " *
        "instantaneous no-jump-resolvent iterative preparation"))
    time isa Real&&isfinite(time)||throw(ArgumentError(
        "instantaneous no-jump-resolvent iterative preparation time must be finite and real"))
    frozen=PIModel(model.basis,
        (freeze_term(term,time,parameters) for term in model.terms))
    isautonomous(frozen)||throw(ArgumentError(
        "freezing the driven PIModel did not produce fixed physical terms; " *
        "custom driven terms must implement rebuild_term/freeze_term so the " *
        "jump/no-jump split remains explicit"))
    metadata=(instantaneous=true,instantaneous_time=time,
              generator_mode=:frozen_instantaneous,
              instantaneous_parameters_supplied=parameters!==nothing)
    frozen,metadata
end

function _no_jump_iterative_with_metadata(plan::NoJumpResolventPlan,metadata)
    metadata===nothing&&return plan
    NoJumpResolventPlan(plan.basis,plan.generator_blocks,plan.factors,
        plan.Ttype,merge(plan.metadata,metadata))
end

function NoJumpResolventPlan(plan::LiouvillianPlan;time=nothing,
        parameters=nothing,kwargs...)
    _no_jump_iterative_reject_evaluation_arguments(plan,time,parameters)
    _prepare_nojump_resolvent_plan(
        plan,nothing;require_autonomous=true,kwargs...)
end

function NoJumpResolventPlan(model::PIModel;
        coefficient_cache=nothing,time=nothing,parameters=nothing,kwargs...)
    budget=get(kwargs,:memory_budget,_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    _require_model_preparation_budget(model,budget;
        operation="no-jump-resolvent iterative model preparation",coefficient_cache)
    frozen,metadata=_no_jump_iterative_instantaneous_model(model,time,parameters)
    plan=NoJumpResolventPlan(
        LiouvillianPlan(frozen;coefficient_cache);kwargs...)
    _no_jump_iterative_with_metadata(plan,metadata)
end

function NoJumpResolventPlan(compiled::CompiledPIModel;time=nothing,
        parameters=nothing,kwargs...)
    _no_jump_iterative_reject_evaluation_arguments(compiled,time,parameters)
    NoJumpResolventPlan(compiled.plan;kwargs...)
end

function NoJumpResolventPlan(specialized::SpecializedPIModel;time=nothing,
        parameters=nothing,kwargs...)
    _no_jump_iterative_reject_evaluation_arguments(specialized,time,parameters)
    _prepare_nojump_resolvent_plan(specialized.plan,specialized.rates;
        require_autonomous=false,kwargs...)
end

function NoJumpResolventWorkspace(plan::NoJumpResolventPlan;
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    estimate=BigInt(Base.summarysize(plan))+
        _no_jump_iterative_workspace_estimate(plan)
    _require_performance_budget("no-jump resolvent workspace",estimate,
        memory_budget;guidance="Reduce the retained basis or increase the budget.")
    blocks=[(zeros(plan.Ttype,size(block)),zeros(plan.Ttype,size(block)))
            for block in plan.generator_blocks]
    NoJumpResolventWorkspace(plan.basis,blocks,plan.Ttype,BigInt(estimate))
end

isautonomous(::NoJumpResolventPlan)=true

function _check_nojump_workspace(work::NoJumpResolventWorkspace,
                                 plan::NoJumpResolventPlan)
    work.basis===plan.basis||throw(ArgumentError(
        "no-jump workspace belongs to a different PI basis"))
    work.Ttype===plan.Ttype||throw(ArgumentError(
        "no-jump workspace has an incompatible scalar type"))
    work
end


function _nojump_external_vector_bytes(work::NoJumpResolventWorkspace,array)
    any(pair->Base.mightalias(array,pair[1])||
              Base.mightalias(array,pair[2]),work.blocks)&&return big(0)
    _performance_entries_bytes(BigInt(length(array)),eltype(array))
end

function _no_jump_iterative_shift(plan::NoJumpResolventPlan,shift)
    shift isa Real&&isfinite(shift)&&shift>=0||throw(ArgumentError(
        "no-jump resolvent shift must be finite, real, and nonnegative"))
    R=_real_float_type(plan.Ttype)
    converted=_checked_prepared_real(shift,R,"no-jump resolvent shift")
    iszero(converted)&&!plan.metadata.strictly_stable&&throw(ArgumentError(
        "the zero-shift no-jump resolvent is singular or cannot be certified: " *
        "the effective generator has an eigenvalue on the imaginary axis. " *
        "This is the dark-state branch of the no-jump-resolvent iterative theorem; use a positive " *
        "shift or a dark-state/ordinary steady-state solver."))
    converted
end

function _no_jump_iterative_complex_shift(plan::NoJumpResolventPlan,shift,
        name::AbstractString="complex no-jump resolvent shift")
    shift isa Number&&isfinite(real(shift))&&isfinite(imag(shift))||
        throw(ArgumentError("$name must be a finite number"))
    T=plan.Ttype
    !_exact_number(shift)&&promote_type(T,typeof(shift))!==T&&
        throw(ArgumentError(
            "$name of type $(typeof(shift)) would narrow in prepared " *
            "precision $T; convert it explicitly or rebuild the plan in " *
            "wider precision"))
    converted=try
        convert(T,shift)
    catch error
        error isa InexactError||error isa OverflowError||rethrow()
        throw(ArgumentError("$name is not representable in prepared precision $T"))
    end
    isfinite(real(converted))&&isfinite(imag(converted))||
        throw(ArgumentError("$name is not finite in prepared precision $T"))
    (!iszero(real(shift))&&iszero(real(converted))||
     !iszero(imag(shift))&&iszero(imag(converted)))&&throw(ArgumentError(
        "$name has a nonzero component that underflows in prepared " *
        "precision $T; use a wider scalar type"))
    iszero(converted)&&!plan.metadata.strictly_stable&&throw(ArgumentError(
        "the zero-shift no-jump resolvent is singular or cannot be " *
        "certified; choose a nonzero complex shift or use a dark-state/" *
        "ordinary steady-state solver"))
    converted
end

function _check_nojump_arrays(destination,source,plan)
    length(destination)==length(plan.basis)&&
        length(source)==length(plan.basis)||throw(DimensionMismatch(
            "no-jump resolvent vectors have the wrong length"))
    Base.mightalias(destination,source)&&throw(ArgumentError(
        "no-jump resolvent source and destination must not alias"))
    source_type=eltype(source);destination_type=eltype(destination)
    promote_type(plan.Ttype,source_type)===plan.Ttype||throw(ArgumentError(
        "no-jump source scalar type $source_type is wider than plan scalar " *
        "type $(plan.Ttype)"))
    promote_type(destination_type,plan.Ttype)===destination_type||
        throw(ArgumentError(
            "no-jump destination scalar type $destination_type cannot " *
            "represent plan scalar type $(plan.Ttype)"))
    nothing
end

@inline function _no_jump_iterative_checked_resolvent_division(
        value,denominator,backend::Symbol)
    isfinite(real(denominator))&&isfinite(imag(denominator))||
        throw(ArgumentError(
            "the requested $backend no-jump resolvent has a nonfinite " *
            "denominator; use a wider scalar type or a better-separated shift"))
    iszero(denominator)&&throw(ArgumentError(
        "the requested no-jump resolvent has a zero $backend denominator"))
    quotient=value/denominator
    isfinite(real(quotient))&&isfinite(imag(quotient))||
        throw(ArgumentError(
            "the requested $backend no-jump resolvent division overflowed " *
            "or produced a nonfinite value; use a wider scalar type or a " *
            "larger shift"))
    quotient
end

@inline function _no_jump_iterative_checked_resolvent_accumulate(
        value,left,right,backend::Symbol)
    contribution=left*right
    isfinite(real(contribution))&&isfinite(imag(contribution))||
        throw(ArgumentError(
            "the $backend no-jump recurrence overflowed while forming an " *
            "intermediate product; rescale the problem or use an ordinary " *
            "solver with a wider scalar type"))
    accumulated=value+contribution
    isfinite(real(accumulated))&&isfinite(imag(accumulated))||
        throw(ArgumentError(
            "the $backend no-jump recurrence overflowed while accumulating " *
            "an intermediate sum; rescale the problem or use an ordinary " *
            "solver with a wider scalar type"))
    accumulated
end

function _no_jump_iterative_check_resolvent_output(destination,backend::Symbol)
    @inbounds for value in destination
        isfinite(real(value))&&isfinite(imag(value))||throw(ArgumentError(
            "the $backend no-jump resolvent produced a nonfinite output; " *
            "use a wider scalar type or a larger shift"))
    end
    destination
end

function _solve_nojump_sector!(transformed,triangular,shift)
    n=size(triangular,1)
    @inbounds for column in n:-1:1,row in n:-1:1
        value=transformed[row,column]
        isfinite(real(value))&&isfinite(imag(value))||throw(ArgumentError(
            "the Schur no-jump recurrence received a nonfinite transformed " *
            "right-hand side"))
        for index in row+1:n
            value=_no_jump_iterative_checked_resolvent_accumulate(value,
                triangular[row,index],transformed[index,column],:Schur)
        end
        for index in column+1:n
            value=_no_jump_iterative_checked_resolvent_accumulate(value,
                transformed[row,index],conj(triangular[column,index]),:Schur)
        end
        denominator=shift-triangular[row,row]-conj(triangular[column,column])
        transformed[row,column]=_no_jump_iterative_checked_resolvent_division(
            value,denominator,:Schur)
    end
    transformed
end

function _apply_nojump_sector!(destination,source,
        factor::_NoJumpSchurSector,shift,left,right)
    mul!(left,adjoint(factor.vectors),source)
    mul!(right,left,factor.vectors)
    _solve_nojump_sector!(right,factor.triangular,shift)
    mul!(left,factor.vectors,right)
    mul!(destination,left,adjoint(factor.vectors))
    _no_jump_iterative_check_resolvent_output(destination,:Schur)
end

function _apply_nojump_sector!(destination,source,
        factor::_NoJumpEigenSector,shift,left,right)
    mul!(left,factor.inverse_vectors,source)
    mul!(right,left,adjoint(factor.inverse_vectors))
    @inbounds for column in axes(right,2),row in axes(right,1)
        denominator=shift-factor.values[row]-conj(factor.values[column])
        right[row,column]=_no_jump_iterative_checked_resolvent_division(
            right[row,column],denominator,:eigen)
    end
    mul!(left,factor.vectors,right)
    mul!(destination,left,adjoint(factor.vectors))
    _no_jump_iterative_check_resolvent_output(destination,:eigen)
end

function _solve_nojump_adjoint_sector!(transformed,triangular,shift)
    n=size(triangular,1)
    @inbounds for column in 1:n,row in 1:n
        value=transformed[row,column]
        isfinite(real(value))&&isfinite(imag(value))||throw(ArgumentError(
            "the adjoint Schur no-jump recurrence received a nonfinite " *
            "transformed right-hand side"))
        for index in 1:row-1
            value=_no_jump_iterative_checked_resolvent_accumulate(value,
                conj(triangular[index,row]),transformed[index,column],
                :adjoint_Schur)
        end
        for index in 1:column-1
            value=_no_jump_iterative_checked_resolvent_accumulate(value,
                transformed[row,index],triangular[index,column],
                :adjoint_Schur)
        end
        denominator=shift-conj(triangular[row,row])-triangular[column,column]
        transformed[row,column]=_no_jump_iterative_checked_resolvent_division(
            value,denominator,:adjoint_Schur)
    end
    transformed
end

function _apply_nojump_adjoint_sector!(destination,source,
        factor::_NoJumpSchurSector,shift,left,right)
    mul!(left,adjoint(factor.vectors),source)
    mul!(right,left,factor.vectors)
    _solve_nojump_adjoint_sector!(right,factor.triangular,shift)
    mul!(left,factor.vectors,right)
    mul!(destination,left,adjoint(factor.vectors))
    _no_jump_iterative_check_resolvent_output(destination,:adjoint_Schur)
end

function _apply_nojump_adjoint_sector!(destination,source,
        factor::_NoJumpEigenSector,shift,left,right)
    mul!(left,adjoint(factor.vectors),source)
    mul!(right,left,factor.vectors)
    @inbounds for column in axes(right,2),row in axes(right,1)
        denominator=shift-conj(factor.values[row])-factor.values[column]
        right[row,column]=_no_jump_iterative_checked_resolvent_division(
            right[row,column],denominator,:adjoint_eigen)
    end
    mul!(left,adjoint(factor.inverse_vectors),right)
    mul!(destination,left,factor.inverse_vectors)
    _no_jump_iterative_check_resolvent_output(destination,:adjoint_eigen)
end

function _no_jump_resolvent_complex!(destination::AbstractVector,
        plan::NoJumpResolventPlan,source::AbstractVector,shift,
        work::NoJumpResolventWorkspace;adjoint_action::Bool=false)
    _check_nojump_workspace(work,plan)
    _check_nojump_arrays(destination,source,plan)
    lambda=_no_jump_iterative_complex_shift(plan,shift)
    for sector in eachindex(plan.basis.sectors)
        dimension=length(plan.basis.patterns[sector])
        range=plan.basis.offsets[sector]:plan.basis.offsets[sector+1]-1
        input=reshape(view(source,range),dimension,dimension)
        output=reshape(view(destination,range),dimension,dimension)
        left,right=work.blocks[sector]
        if adjoint_action
            _apply_nojump_adjoint_sector!(output,input,plan.factors[sector],
                lambda,left,right)
        else
            _apply_nojump_sector!(output,input,plan.factors[sector],lambda,
                left,right)
        end
    end
    destination
end

"""
    no_jump_resolvent!(destination, plan, source, shift, workspace)

Apply ``R_shift^S=(shift-S)^(-1)`` to a PI coefficient vector using the
prepared sectorwise Lyapunov solver. `shift` must be a finite nonnegative real
number. The zero-shift route requires a strictly stable no-jump generator;
positive shifts remain valid in the dark-state branch.
"""
function no_jump_resolvent!(destination::AbstractVector,
        plan::NoJumpResolventPlan,source::AbstractVector,shift::Real,
        work::NoJumpResolventWorkspace)
    _check_nojump_workspace(work,plan)
    _check_nojump_arrays(destination,source,plan)
    lambda=_no_jump_iterative_shift(plan,shift)
    for sector in eachindex(plan.basis.sectors)
        dimension=length(plan.basis.patterns[sector])
        range=plan.basis.offsets[sector]:plan.basis.offsets[sector+1]-1
        input=reshape(view(source,range),dimension,dimension)
        output=reshape(view(destination,range),dimension,dimension)
        left,right=work.blocks[sector]
        _apply_nojump_sector!(output,input,plan.factors[sector],lambda,
                              left,right)
    end
    destination
end

no_jump_resolvent!(destination,plan::NoJumpResolventPlan,source,shift)=
    no_jump_resolvent!(destination,plan,source,shift,
                       NoJumpResolventWorkspace(plan))

"""
    no_jump_resolvent(plan, source; shift=0, workspace=nothing,
                       memory_budget=512*1024^2)

Allocating wrapper for [`no_jump_resolvent!`](@ref). A vector input returns a
coefficient vector; a [`PIState`](@ref) input returns a `PIState` on the exact
same basis. Reuse `workspace` for repeated applications. The allocating route
guards the prepared workspace, input, and returned output together.
"""
function no_jump_resolvent(plan::NoJumpResolventPlan,
        source::AbstractVector;shift::Real=0,workspace=nothing,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    work=workspace===nothing ? NoJumpResolventWorkspace(
        plan;memory_budget) : _check_nojump_workspace(workspace,plan)
    output_bytes=_performance_entries_bytes(
        BigInt(length(plan.basis)),plan.Ttype)
    estimate=work.accounted_peak_bytes+
        _nojump_external_vector_bytes(work,source)+output_bytes
    _require_performance_budget("allocating no-jump resolvent",estimate,
        memory_budget;guidance=
            "Reuse caller-owned output with no_jump_resolvent! or increase " *
            "the budget.")
    destination=zeros(plan.Ttype,length(plan.basis))
    no_jump_resolvent!(destination,plan,source,shift,work)
end

function no_jump_resolvent(plan::NoJumpResolventPlan,source::PIState;
        shift::Real=0,workspace=nothing,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    source.basis===plan.basis||throw(ArgumentError(
        "state and no-jump plan use different PI bases"))
    work=workspace===nothing ? NoJumpResolventWorkspace(
        plan;memory_budget) : _check_nojump_workspace(workspace,plan)
    output_bytes=_performance_entries_bytes(
        BigInt(length(plan.basis)),plan.Ttype)
    estimate=work.accounted_peak_bytes+
        _nojump_external_vector_bytes(work,source.data)+output_bytes
    _require_performance_budget("allocating no-jump state resolvent",estimate,
        memory_budget;guidance=
            "Use no_jump_resolvent! with caller-owned storage or increase " *
            "the budget.")
    result=PIState(plan.basis;T=_real_float_type(plan.Ttype))
    no_jump_resolvent!(result.data,plan,source.data,shift,work)
    result
end

"""
    NoJumpIterativePlan(source; backend=:schur, memory_budget=512*1024^2, ...)

Prepare the exact PI jump/no-jump splitting used by the no-jump-resolvent iterative fixed-point,
right-preconditioned GMRES, low-mode shift-invert, and implicit-Euler methods.
The plan owns the full compiled PI Liouvillian, a [`NoJumpResolventPlan`](@ref),
the physical trace functional, and a trace-one maximally mixed deflation
direction. Construction certifies trace preservation by one matrix-free
adjoint application to that functional. Consequently, a restricted Schur
basis is accepted only when it is invariant under the complete generator;
`trace_atol` and `trace_rtol` control this setup certificate.

The implementation uses normalized trace deflation
``L_delta=L+delta*|I/D><trace|``. Thus the stationary eigenvalue moves to
`delta`, independent of the retained Hilbert dimension `D`; this is exactly
the paper's unnormalized-identity convention after rescaling its deflation
coefficient by `D`.

A driven `PIModel` may be prepared at one explicitly selected instant with
`NoJumpIterativePlan(model; time=t, parameters=p)`. This is the stationary/spectral
problem for that frozen generator, not the long-time state of driven dynamics.
"""
struct NoJumpIterativePlan{B,L,N,V,D,T,M}
    basis::B
    liouvillian::L
    no_jump::N
    tracevec::V
    deflation_vector::D
    Ttype::Type{T}
    metadata::M
end

size(plan::NoJumpIterativePlan)=size(plan.no_jump)
size(plan::NoJumpIterativePlan,index::Integer)=size(plan.no_jump,index)
eltype(plan::NoJumpIterativePlan)=plan.Ttype

function _no_jump_iterative_source_adjoint!(destination,source::LiouvillianPlan,
        tracevec,workspace)
    apply_adjoint!(destination,source,tracevec,workspace)
end

function _no_jump_iterative_source_adjoint!(destination,source::SpecializedPIModel,
        tracevec,workspace)
    # A family's prototype plan contains callable rate schedules.  Supplying
    # the specialization here is essential: its bound rates, rather than the
    # prototype schedules or their defaults, define the physical generator.
    apply_adjoint!(destination,source,tracevec,0.0,nothing,workspace)
end

function _no_jump_iterative_trace_preservation_certificate(source,
        no_jump::NoJumpResolventPlan;trace_atol::Real=0,
        trace_rtol=nothing,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    underlying=source isa SpecializedPIModel ? source.plan : source
    T=underlying.Ttype;R=_real_float_type(T)
    largest_block=maximum(length,underlying.basis.patterns;init=1)
    requested_rtol=trace_rtol===nothing ?
        R(128)*eps(R)*sqrt(R(max(largest_block,1))) : trace_rtol
    atol,rtol=_no_jump_iterative_check_tolerances(trace_atol,requested_rtol,R)

    # The no-jump blocks bound the natural Hamiltonian/loss scale.  Gain and
    # loss have the same physical channel scale, so the factor two also
    # covers the corresponding gain contribution without materializing it.
    generator_scale=max(one(R),
        maximum(block->R(norm(block,Inf)),no_jump.generator_blocks;
                init=zero(R)))
    trace_scale=R(norm(underlying.tracevec,Inf))
    scale=max(one(R),R(2)*generator_scale*trace_scale)
    isfinite(scale)||throw(ArgumentError(
        "no-jump-resolvent iterative trace-preservation certification scale is nonfinite; " *
        "use a wider prepared scalar type"))
    tolerance=atol+rtol*scale
    isfinite(tolerance)||throw(ArgumentError(
        "no-jump-resolvent iterative trace-preservation tolerance overflowed; use a wider " *
        "prepared scalar type"))

    temporary_bytes=
        _performance_liouvillian_workspace_bytes(underlying)+
        _performance_entries_bytes(BigInt(length(underlying.basis)),T)
    estimate=BigInt(Base.summarysize(source))+
        BigInt(Base.summarysize(no_jump))+temporary_bytes
    _require_performance_budget(
        "no-jump-resolvent iterative trace-preservation certification",estimate,memory_budget;
        guidance="Reduce the retained basis or increase the budget.")

    workspace=LiouvillianWorkspace(underlying)
    residual=zeros(T,length(underlying.basis))
    _no_jump_iterative_source_adjoint!(
        residual,source,underlying.tracevec,workspace)
    residual_inf=R(norm(residual,Inf))
    isfinite(residual_inf)||throw(ArgumentError(
        "no-jump-resolvent iterative trace-preservation certification produced a nonfinite " *
        "adjoint residual"))
    residual_inf<=tolerance||throw(ArgumentError(
        "the prepared generator is not trace preserving on the retained PI " *
        "basis: norm(L' * tracevec, Inf)=$residual_inf exceeds " *
        "tolerance=$tolerance. A restricted basis must be invariant under " *
        "every retained channel; restore the leaked Schur sectors or use an " *
        "invariant collective-only restriction."))
    (trace_preserving=true,trace_preservation_residual=residual_inf,
     trace_preservation_tolerance=tolerance,
     trace_preservation_scale=scale,trace_atol=atol,trace_rtol=rtol,
     trace_certification=:matrix_free_adjoint)
end

function NoJumpIterativePlan(plan::LiouvillianPlan;trace_atol::Real=0,
        trace_rtol=nothing,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,time=nothing,
        parameters=nothing,kwargs...)
    _no_jump_iterative_reject_evaluation_arguments(plan,time,parameters)
    no_jump=NoJumpResolventPlan(plan;memory_budget,kwargs...)
    trace_certificate=_no_jump_iterative_trace_preservation_certificate(
        plan,no_jump;trace_atol,trace_rtol,memory_budget)
    R=_real_float_type(plan.Ttype)
    deflation=plan.Ttype.(maximally_mixed_state(plan.basis;T=R).data)
    guarantees=no_jump.metadata.strictly_stable ?
        (:cptp_fixed_point,:positive_shift_contraction) :
        (:positive_shift_contraction,)
    metadata=merge(no_jump.metadata,trace_certificate,(
        deflation_normalization=:trace_one_identity,
        source=:prepared_pi_liouvillian,
        fixed_point_action=:direct_gain,
        zero_shift_fixed_point_available=no_jump.metadata.strictly_stable,
        guarantees,
        unique_steady_state=:assumed_not_certified,
        unsupported_sources=(:composite,:global_pseudomode,:heom,:hops)))
    NoJumpIterativePlan(plan.basis,plan,no_jump,plan.tracevec,deflation,
               plan.Ttype,metadata)
end

function _no_jump_iterative_with_metadata(plan::NoJumpIterativePlan,metadata)
    metadata===nothing&&return plan
    no_jump=_no_jump_iterative_with_metadata(plan.no_jump,metadata)
    NoJumpIterativePlan(plan.basis,plan.liouvillian,no_jump,plan.tracevec,
        plan.deflation_vector,plan.Ttype,merge(plan.metadata,metadata))
end

function NoJumpIterativePlan(model::PIModel;coefficient_cache=nothing,time=nothing,
        parameters=nothing,kwargs...)
    budget=get(kwargs,:memory_budget,_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    _require_model_preparation_budget(model,budget;
        operation="no-jump-resolvent iterative model preparation",coefficient_cache)
    frozen,metadata=_no_jump_iterative_instantaneous_model(model,time,parameters)
    plan=NoJumpIterativePlan(LiouvillianPlan(frozen;coefficient_cache);kwargs...)
    _no_jump_iterative_with_metadata(plan,metadata)
end

function NoJumpIterativePlan(compiled::CompiledPIModel;time=nothing,
        parameters=nothing,kwargs...)
    _no_jump_iterative_reject_evaluation_arguments(compiled,time,parameters)
    NoJumpIterativePlan(compiled.plan;kwargs...)
end

function NoJumpIterativePlan(specialized::SpecializedPIModel;trace_atol::Real=0,
        trace_rtol=nothing,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,time=nothing,
        parameters=nothing,kwargs...)
    _no_jump_iterative_reject_evaluation_arguments(specialized,time,parameters)
    no_jump=NoJumpResolventPlan(specialized;memory_budget,kwargs...)
    trace_certificate=_no_jump_iterative_trace_preservation_certificate(
        specialized,no_jump;trace_atol,trace_rtol,memory_budget)
    R=_real_float_type(specialized.plan.Ttype)
    deflation=specialized.plan.Ttype.(
        maximally_mixed_state(specialized.plan.basis;T=R).data)
    guarantees=no_jump.metadata.strictly_stable ?
        (:cptp_fixed_point,:positive_shift_contraction) :
        (:positive_shift_contraction,)
    metadata=merge(no_jump.metadata,trace_certificate,(
        deflation_normalization=:trace_one_identity,
        source=:specialized_pi_liouvillian,
        bound_rates=specialized.rates,
        fixed_point_action=:direct_gain,
        zero_shift_fixed_point_available=no_jump.metadata.strictly_stable,
        guarantees,
        unique_steady_state=:assumed_not_certified,
        unsupported_sources=(:composite,:global_pseudomode,:heom,:hops)))
    NoJumpIterativePlan(specialized.plan.basis,specialized,no_jump,
        specialized.plan.tracevec,deflation,specialized.plan.Ttype,metadata)
end

isautonomous(::NoJumpIterativePlan)=true

function NoJumpResolventPlan(plan::NoJumpIterativePlan;kwargs...)
    isempty(kwargs)||throw(ArgumentError(
        "a NoJumpIterativePlan already owns a prepared no-jump resolvent; constructor " *
        "keywords cannot rebuild it implicitly. Construct a new NoJumpIterativePlan " *
        "with the requested options instead."))
    plan.no_jump
end

_no_jump_iterative_underlying_plan(plan::NoJumpIterativePlan)=
    plan.liouvillian isa SpecializedPIModel ?
        plan.liouvillian.plan : plan.liouvillian

function _apply_no_jump_iterative_liouvillian!(destination,plan::NoJumpIterativePlan,source,
        workspace::LiouvillianWorkspace)
    if plan.liouvillian isa SpecializedPIModel
        apply!(destination,plan.liouvillian,source,0.0,nothing,workspace)
    else
        apply!(destination,plan.liouvillian,source,workspace)
    end
end

_no_jump_iterative_gain_parameters(plan::NoJumpIterativePlan)=
    plan.liouvillian isa SpecializedPIModel ? plan.liouvillian.rates : nothing

function _apply_no_jump_iterative_gain_kernel!(y,x,::HamiltonianPIKernel,prepared,b,
        parameters,work)
    nothing
end

function _apply_no_jump_iterative_gain_kernel!(y,x,kernel::DissipatorPIKernel,
        prepared,b,parameters,work)
    scale=convert(eltype(work[1][1]),
        _no_jump_iterative_jump_scale(kernel,parameters))
    for sector in eachindex(b.sectors)
        offset=b.offsets[sector]
        left,right,input=work[sector]
        block=kernel.blocks[sector]
        mul!(left,block,input)
        mul!(right,left,adjoint(block))
        @inbounds for index in eachindex(right)
            y[offset+index-1]+=scale*right[index]
        end
    end
    nothing
end

function _apply_no_jump_iterative_gain_kernel!(y,x,kernel::LocalJumpPIKernel,
        prepared,b,parameters,work)
    scale=convert(eltype(work[1][1]),
        _no_jump_iterative_jump_scale(kernel,parameters))
    @inbounds for index in eachindex(kernel.gain.V)
        y[kernel.gain.I[index]]+=scale*kernel.gain.V[index]*
            x[kernel.gain.J[index]]
    end
    nothing
end

function _apply_no_jump_iterative_gain_kernel!(y,x,
        kernel::FactorizedLocalJumpPIKernel,
        prepared::StaticFactorizedGainKernelWorkspace,b,parameters,work)
    scale=convert(eltype(work[1][1]),
        _no_jump_iterative_jump_scale(kernel,parameters))
    _apply_factorized_onebody_gain!(y,kernel.branches,kernel.contractions,
        prepared.gain_scratch,b,scale,work,1)
    nothing
end

function _apply_no_jump_iterative_gain_kernel!(y,x,
        kernel::FactorizedLocalPBodyJumpPIKernel,
        prepared::StaticFactorizedGainKernelWorkspace,b,parameters,work)
    scale=convert(eltype(work[1][1]),
        _no_jump_iterative_jump_scale(kernel,parameters))
    _apply_factorized_pbody_gain!(y,kernel.groups,kernel.contractions,
        kernel.pair_scales,prepared.gain_scratch,b,scale,work)
    nothing
end

function _apply_no_jump_iterative_gain_kernel!(y,x,kernel::FusedStaticPIKernel,
        prepared::StaticFactorizedGainKernelWorkspace,b,parameters,work)
    for gain in kernel.collective_gains
        scale=convert(eltype(work[1][1]),
            _no_jump_iterative_check_fused_gain_scale(gain.scale))
        for sector in eachindex(b.sectors)
            offset=b.offsets[sector]
            left,right,input=work[sector]
            block=gain.blocks[sector]
            mul!(left,block,input)
            mul!(right,left,adjoint(block))
            @inbounds for index in eachindex(right)
                y[offset+index-1]+=scale*right[index]
            end
        end
    end
    for gain in kernel.onebody_gains
        scale=convert(eltype(work[1][1]),
            _no_jump_iterative_check_fused_gain_scale(gain.scale))
        _apply_factorized_onebody_gain!(y,gain.branches,gain.contractions,
            prepared.gain_scratch,b,scale,work,1)
    end
    for gain in kernel.pbody_gains
        scale=convert(eltype(work[1][1]),
            _no_jump_iterative_check_fused_gain_scale(gain.scale))
        _apply_factorized_pbody_gain!(y,gain.groups,gain.contractions,
            gain.pair_scales,prepared.gain_scratch,b,scale,work)
    end
    nothing
end

function _apply_no_jump_iterative_gain_kernel!(y,x,kernel,prepared,b,parameters,work)
    throw(ArgumentError(
        "compiled kernel $(typeof(kernel)) does not expose the fixed gain " *
        "map required by the no-jump-resolvent iterative method"))
end

@inline _apply_no_jump_iterative_gain_kernels!(y,x,::Tuple{},::Tuple{},b,parameters,
    work)=nothing
@inline function _apply_no_jump_iterative_gain_kernels!(y,x,
        kernels::Tuple{K,Vararg{Any}},
        prepared::Tuple{W,Vararg{Any}},b,parameters,work) where {K,W}
    _apply_no_jump_iterative_gain_kernel!(
        y,x,first(kernels),first(prepared),b,parameters,work)
    _apply_no_jump_iterative_gain_kernels!(y,x,Base.tail(kernels),
        Base.tail(prepared),b,parameters,work)
end

function _apply_no_jump_iterative_gain!(destination,plan::NoJumpIterativePlan,source,
        workspace::LiouvillianWorkspace)
    underlying=_no_jump_iterative_underlying_plan(plan)
    underlying.kernels===nothing&&throw(ArgumentError(
        "the no-jump-resolvent iterative gain map requires prepared physical kernels"))
    Base.mightalias(destination,source)&&throw(ArgumentError(
        "no-jump-resolvent iterative gain-map source and destination must not alias"))
    length(destination)==length(plan.basis)&&
        length(source)==length(plan.basis)||throw(DimensionMismatch(
            "no-jump-resolvent iterative gain-map vectors have the wrong length"))
    _check_liouvillian_workspace(workspace,underlying)
    _check_liouvillian_apply_types(destination,source,underlying)
    parameters=_no_jump_iterative_gain_parameters(plan)
    _prepare_kernels!(underlying.kernels,workspace.kernel_workspaces,
        plan.basis,0.0,parameters)
    fill!(destination,zero(eltype(destination)))
    _copy_input_blocks!(workspace.blocks,source,plan.basis)
    _apply_no_jump_iterative_gain_kernels!(destination,source,underlying.kernels,
        workspace.kernel_workspaces,plan.basis,parameters,workspace.blocks)
    destination
end

"""
    NoJumpIterativeWorkspace(plan; krylovdim=30, recycle_dim=8,
                     memory_budget=512*1024^2)

Task-owned no-jump, Liouvillian, and recycled-GMRES scratch for no-jump-resolvent iterative linear
solves. Capacity is fixed at construction. Reuse one workspace across nearby
resolvent solves or implicit-Euler steps to retain warm starts and recycled
Krylov directions; do not share it concurrently between tasks.
"""
mutable struct NoJumpIterativeWorkspace{P,N,L,G,T,R}
    plan::P
    no_jump::N
    liouvillian::L
    gmres::G
    preconditioned::Vector{T}
    image::Vector{T}
    transformed_solution::Vector{T}
    physical_solution::Vector{T}
    residual::Vector{T}
    rhs::Vector{T}
    identity_resolvent::Vector{T}
    cached_shift::T
    cached_deflation::R
    cached_adjoint_action::Bool
    cached_adjoint_functional::Vector{T}
    denominator::T
    cache_valid::Bool
    warm_start_valid::Bool
    accounted_peak_bytes::BigInt
end

function _no_jump_iterative_workspace_estimate(plan::NoJumpIterativePlan,krylovdim,recycle_dim)
    n=length(plan.basis);T=plan.Ttype
    m=min(BigInt(n),BigInt(krylovdim))
    # Recycle extraction forms a projected matrix and its dense eigensystem.
    # Harmonic extraction additionally solves one projected adjoint system.
    # These arrays are transient, but live on top of all retained task scratch
    # and therefore belong in the guarded peak.
    projected_transient=_performance_entries_bytes(4m^2+6m,T)+
        _performance_entries_bytes(2m,_real_float_type(T))
    _no_jump_iterative_workspace_estimate(plan.no_jump)+
        _performance_liouvillian_workspace_bytes(
            _no_jump_iterative_underlying_plan(plan))+
        _performance_gmres_bytes(n,T,krylovdim;recycle_dim)+
        _performance_entries_bytes(8BigInt(n),T)+projected_transient
end

function NoJumpIterativeWorkspace(plan::NoJumpIterativePlan;krylovdim::Integer=30,
        recycle_dim::Integer=8,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    !(krylovdim isa Bool)&&krylovdim>0||throw(ArgumentError(
        "krylovdim must be a positive integer"))
    !(recycle_dim isa Bool)&&recycle_dim>=0||throw(ArgumentError(
        "recycle_dim must be a nonnegative integer"))
    estimate=BigInt(Base.summarysize(plan))+
        _no_jump_iterative_workspace_estimate(plan,krylovdim,recycle_dim)
    _require_performance_budget("no-jump-resolvent iterative workspace",estimate,
        memory_budget;guidance=
            "Reduce krylovdim/recycle_dim or increase the budget.")
    n=length(plan.basis);T=plan.Ttype;R=_real_float_type(T)
    vectors=ntuple(_->zeros(T,n),7)
    NoJumpIterativeWorkspace(plan,NoJumpResolventWorkspace(plan.no_jump;
            memory_budget=Inf),LiouvillianWorkspace(
                _no_jump_iterative_underlying_plan(plan)),
        RecycledGMRESWorkspace(T,n,krylovdim,recycle_dim),vectors...,
        T(complex(R(NaN),R(NaN))),R(NaN),false,zeros(T,n),one(T),false,false,
        BigInt(estimate))
end

function _check_no_jump_iterative_workspace(work::NoJumpIterativeWorkspace,plan::NoJumpIterativePlan)
    work.plan===plan||throw(ArgumentError(
        "no-jump-resolvent iterative workspace belongs to a different prepared plan"))
    _check_nojump_workspace(work.no_jump,plan.no_jump)
    _check_liouvillian_workspace(
        work.liouvillian,_no_jump_iterative_underlying_plan(plan))
    work
end

function _no_jump_iterative_workspace_vectors(work::NoJumpIterativeWorkspace)
    (work.preconditioned,work.image,work.transformed_solution,
     work.physical_solution,work.residual,work.rhs,
     work.identity_resolvent,work.cached_adjoint_functional)
end

function _check_no_jump_iterative_workspace_aliases(destination,right_hand_side,
        work::NoJumpIterativeWorkspace;adjoint_deflation_functional=nothing)
    # `work.rhs` and `work.physical_solution` are the two deliberate internal
    # call sites used by stationary solving.  Every other alias with retained
    # linear scratch can be overwritten before it is consumed, can invalidate
    # a cached Sherman--Morrison correction, or can corrupt the final true
    # residual.  Reject partial aliases as well as identical arrays.
    for owner in _no_jump_iterative_workspace_vectors(work)
        if Base.mightalias(right_hand_side,owner)&&
                !(right_hand_side===work.rhs&&owner===work.rhs)
            throw(ArgumentError(
                "no-jump-resolvent iterative right-hand side must not alias " *
                "task-owned workspace scratch (except the workspace rhs buffer)"))
        end
        if Base.mightalias(destination,owner)&&
                !(destination===work.physical_solution&&
                  owner===work.physical_solution)
            throw(ArgumentError(
                "no-jump-resolvent iterative destination must not alias " *
                "task-owned workspace scratch (except the workspace physical-solution buffer)"))
        end
        if adjoint_deflation_functional!==nothing&&
                Base.mightalias(adjoint_deflation_functional,owner)
            throw(ArgumentError(
                "adjoint deflation functional must not alias task-owned " *
                "workspace scratch"))
        end
    end
    for owner in (work.gmres.V,work.gmres.H,work.gmres.U,work.gmres.C,
            work.gmres.candidate,work.gmres.AU,work.gmres.coupling,
            work.gmres.smallR,work.gmres.w,work.gmres.image,
            work.gmres.residual,work.gmres.projected_residual,
            work.gmres.correction)
        (Base.mightalias(right_hand_side,owner)||
         Base.mightalias(destination,owner)||
         (adjoint_deflation_functional!==nothing&&
          Base.mightalias(adjoint_deflation_functional,owner)))&&
            throw(ArgumentError(
                "no-jump-resolvent iterative arrays must not alias retained " *
                "GMRES workspace storage"))
    end
    nothing
end

function _no_jump_iterative_external_vector_bytes(work::NoJumpIterativeWorkspace,array)
    any(owner->Base.mightalias(array,owner),
        _no_jump_iterative_workspace_vectors(work))&&return big(0)
    _performance_entries_bytes(BigInt(length(array)),eltype(array))
end

function _no_jump_iterative_state_validation_bytes(plan::NoJumpIterativePlan)
    largest=BigInt(maximum(length,plan.basis.patterns;init=1))
    # Hermiticity differencing, physical/coefficient block copies, shifted
    # Cholesky or Hermitian eigensolver input, and conservative dense-driver
    # scratch for the largest live Schur block.
    _performance_entries_bytes(8largest^2+4largest,plan.Ttype)
end

function _no_jump_iterative_stationary_output_bytes(plan::NoJumpIterativePlan)
    _performance_entries_bytes(BigInt(length(plan.basis)),plan.Ttype)+
        _no_jump_iterative_state_validation_bytes(plan)
end

function _no_jump_iterative_outer_arnoldi_bytes(plan::NoJumpIterativePlan,nev,krylovdim,
        maxrestarts;vectors::Bool)
    n=length(plan.basis)
    _performance_block_arnoldi_bytes(
        n,plan.Ttype,krylovdim,1)+
        _performance_block_arnoldi_output_bytes(
            n,plan.Ttype,nev,maxrestarts;vectors)
end

function _no_jump_iterative_solver_scalar(plan::NoJumpIterativePlan,value,name)
    value isa Real&&isfinite(value)&&value>=0||throw(ArgumentError(
        "$name must be finite, real, and nonnegative"))
    _checked_prepared_real(value,_real_float_type(plan.Ttype),name)
end

function _no_jump_iterative_spectral_shift(plan::NoJumpIterativePlan,value,name)
    _no_jump_iterative_complex_shift(plan.no_jump,value,name)
end

function _prepare_no_jump_iterative_preconditioner!(work::NoJumpIterativeWorkspace,
        plan::NoJumpIterativePlan,shift,deflation;singularity_atol::Real=0,
        singularity_rtol=nothing,adjoint_action::Bool=false,
        adjoint_deflation_functional=nothing)
    lambda=_no_jump_iterative_spectral_shift(
        plan,shift,"no-jump-resolvent iterative shift")
    delta=_no_jump_iterative_solver_scalar(plan,deflation,"no-jump-resolvent iterative deflation")
    adjoint_functional=adjoint_action ?
        (adjoint_deflation_functional===nothing ? plan.deflation_vector :
            adjoint_deflation_functional) : nothing
    if adjoint_action
        length(adjoint_functional)==length(plan.basis)||throw(DimensionMismatch(
            "adjoint deflation functional has the wrong length"))
        promote_type(plan.Ttype,eltype(adjoint_functional))===plan.Ttype||
            throw(ArgumentError(
                "adjoint deflation functional is wider than prepared precision"))
        all(value->isfinite(real(value))&&isfinite(imag(value)),
            adjoint_functional)||throw(ArgumentError(
                "adjoint deflation functional must contain only finite values"))
    end
    adjoint_functional_matches=!adjoint_action||
        (work.cache_valid&&work.cached_adjoint_action&&
         (adjoint_functional===work.cached_adjoint_functional||
          all(index->isequal(work.cached_adjoint_functional[index],
              adjoint_functional[index]),eachindex(adjoint_functional))))
    if !(work.cache_valid&&lambda==work.cached_shift&&
            delta==work.cached_deflation&&
            adjoint_action==work.cached_adjoint_action&&
            adjoint_functional_matches)
        # Snapshot before applying a resolvent because a deliberately exposed
        # low-level caller could otherwise point the functional into scratch
        # that the factorization action overwrites.
        adjoint_action&&copyto!(
            work.cached_adjoint_functional,adjoint_functional)
        effective_adjoint_functional=adjoint_action ?
            work.cached_adjoint_functional : nothing
        if iszero(delta)
            fill!(work.identity_resolvent,zero(eltype(work.identity_resolvent)))
            work.denominator=one(eltype(work.identity_resolvent))
        elseif adjoint_action
            _no_jump_resolvent_complex!(work.identity_resolvent,plan.no_jump,
                plan.tracevec,lambda,work.no_jump;adjoint_action=true)
            work.denominator=one(eltype(work.identity_resolvent))-
                delta*dot(effective_adjoint_functional,
                    work.identity_resolvent)
        else
            _no_jump_resolvent_complex!(work.identity_resolvent,plan.no_jump,
                plan.deflation_vector,lambda,work.no_jump)
            work.denominator=one(eltype(work.identity_resolvent))-
                delta*dot(plan.tracevec,work.identity_resolvent)
        end
        work.cached_shift=lambda
        work.cached_deflation=delta
        work.cached_adjoint_action=adjoint_action
        work.cache_valid=true
    end
    R=_real_float_type(plan.Ttype)
    requested_rtol=singularity_rtol===nothing ? sqrt(eps(R)) :
        singularity_rtol
    atol,rtol=_no_jump_iterative_check_tolerances(
        singularity_atol,requested_rtol,R)
    correction=one(plan.Ttype)-work.denominator
    threshold=atol+rtol*max(abs(correction),one(R))
    abs(work.denominator)>threshold||throw(ArgumentError(
        "the deflated no-jump Sherman--Morrison denominator is zero or " *
        "numerically singular; choose a different deflation"))
    lambda,delta
end

function _apply_no_jump_iterative_preconditioner!(destination,plan::NoJumpIterativePlan,source,
        work::NoJumpIterativeWorkspace,shift,deflation;
        adjoint_action::Bool=false,adjoint_deflation_functional=nothing)
    _no_jump_resolvent_complex!(destination,plan.no_jump,source,shift,
        work.no_jump;adjoint_action)
    if !iszero(deflation)
        coefficient=deflation*(adjoint_action ?
            dot(adjoint_deflation_functional===nothing ?
                plan.deflation_vector : adjoint_deflation_functional,
                destination) :
            dot(plan.tracevec,destination))/work.denominator
        @inbounds @simd for index in eachindex(destination)
            destination[index]+=coefficient*work.identity_resolvent[index]
        end
    end
    destination
end

function _apply_no_jump_iterative_shifted_deflated!(destination,plan::NoJumpIterativePlan,
        source,work::NoJumpIterativeWorkspace,shift,deflation;
        adjoint_action::Bool=false,adjoint_deflation_functional=nothing)
    if adjoint_action
        _no_jump_iterative_source_adjoint!(
            work.image,plan.liouvillian,source,work.liouvillian)
        functional=adjoint_deflation_functional===nothing ?
            plan.deflation_vector : adjoint_deflation_functional
        stationary_component=dot(functional,source)
        @inbounds @simd for index in eachindex(destination)
            destination[index]=shift*source[index]-work.image[index]-
                deflation*stationary_component*plan.tracevec[index]
        end
    else
        _apply_no_jump_iterative_liouvillian!(
            work.image,plan,source,work.liouvillian)
        trace_component=dot(plan.tracevec,source)
        @inbounds @simd for index in eachindex(destination)
            destination[index]=shift*source[index]-work.image[index]-
                deflation*trace_component*plan.deflation_vector[index]
        end
    end
    destination
end

function _no_jump_iterative_physical_maximum(basis,data)
    R=_real_float_type(eltype(data));maximum_value=zero(R)
    for (sector,partition) in pairs(basis.sectors)
        range=basis.offsets[sector]:basis.offsets[sector+1]-1
        @inbounds for coordinate in range
            physical=_divide_by_schur_multiplicity_scale(
                data[coordinate],R,partition)
            maximum_value=max(maximum_value,abs(physical))
        end
    end
    maximum_value
end

function _no_jump_iterative_reset_linear_workspace!(work::NoJumpIterativeWorkspace)
    fill!(work.transformed_solution,zero(eltype(work.transformed_solution)))
    work.gmres.nrecycle=0
    work.warm_start_valid=false
    work
end

function _check_no_jump_iterative_linear_arrays(destination,right_hand_side,
        plan::NoJumpIterativePlan)
    source_type=eltype(right_hand_side)
    destination_type=eltype(destination)
    source_type<:Number&&isconcretetype(source_type)||throw(ArgumentError(
        "no-jump-resolvent iterative right-hand sides must expose a concrete numeric eltype"))
    destination_type<:Number&&isconcretetype(destination_type)||
        throw(ArgumentError(
            "no-jump-resolvent iterative destinations must expose a concrete numeric eltype"))
    promote_type(plan.Ttype,source_type)===plan.Ttype||throw(ArgumentError(
        "no-jump-resolvent iterative right-hand-side scalar type $source_type is wider than " *
        "prepared precision $(plan.Ttype)"))
    promote_type(destination_type,plan.Ttype)===destination_type||
        throw(ArgumentError(
            "no-jump-resolvent iterative destination scalar type $destination_type cannot " *
            "represent prepared precision $(plan.Ttype)"))
    nothing
end

"""
    no_jump_iterative_resolvent!(destination, plan, right_hand_side, workspace;
                       shift, deflation=0, reuse=false, ...)

Solve
``(shift-L-deflation*|I/D><trace|)*destination=right_hand_side`` by
right-preconditioned recycled GMRES. The exact sectorwise no-jump resolvent,
including the rank-one deflation through Sherman--Morrison, is embedded on the
right of the Krylov operator. Full unpreconditioned and physical-Schur-block
residuals are recomputed before return.

Set `reuse=true` only when intentionally retaining the previous transformed
warm start and recycled subspace, as in nearby resolvents or nested
shift-invert solves.
"""
function no_jump_iterative_resolvent!(destination::AbstractVector,plan::NoJumpIterativePlan,
        right_hand_side::AbstractVector,work::NoJumpIterativeWorkspace;
        shift::Number,deflation::Real=0,reuse::Bool=false,
        adjoint_action::Bool=false,
        adjoint_deflation_functional=nothing,
        maxiter::Integer=500,atol::Real=1e-10,rtol::Real=1e-8,
        singularity_atol::Real=0,singularity_rtol=nothing,
        require_convergence::Bool=true,recycle_extraction::Symbol=:harmonic,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    _check_no_jump_iterative_workspace(work,plan)
    length(destination)==length(plan.basis)&&
        length(right_hand_side)==length(plan.basis)||throw(DimensionMismatch(
            "no-jump-resolvent iterative resolvent vectors have the wrong length"))
    Base.mightalias(destination,right_hand_side)&&throw(ArgumentError(
        "no-jump-resolvent iterative resolvent source and destination must not alias"))
    _check_no_jump_iterative_linear_arrays(destination,right_hand_side,plan)
    !adjoint_action&&adjoint_deflation_functional!==nothing&&
        throw(ArgumentError(
            "adjoint_deflation_functional is valid only with adjoint_action=true"))
    _check_no_jump_iterative_workspace_aliases(destination,right_hand_side,work;
        adjoint_deflation_functional=adjoint_action ?
            adjoint_deflation_functional : nothing)
    !(maxiter isa Bool)&&maxiter>0||throw(ArgumentError(
        "maxiter must be a positive integer"))
    R=_real_float_type(plan.Ttype)
    atolT,rtolT=_no_jump_iterative_check_tolerances(atol,rtol,R)
    estimate=work.accounted_peak_bytes+
        _no_jump_iterative_external_vector_bytes(work,destination)+
        _no_jump_iterative_external_vector_bytes(work,right_hand_side)
    if adjoint_action&&adjoint_deflation_functional!==nothing&&
            !Base.mightalias(adjoint_deflation_functional,destination)&&
            !Base.mightalias(adjoint_deflation_functional,right_hand_side)
        estimate+=_no_jump_iterative_external_vector_bytes(
            work,adjoint_deflation_functional)
    end
    _require_performance_budget("no-jump-resolvent iterative resolvent solve",
        estimate,
        memory_budget;guidance=
            "Use a smaller prepared workspace or increase the budget.")
    # Preserve the physical right-hand side before any factorized
    # preconditioner scratch is touched.
    copyto!(work.rhs,right_hand_side)
    lambda,delta=_prepare_no_jump_iterative_preconditioner!(work,plan,shift,
        deflation;singularity_atol,singularity_rtol,adjoint_action,
        adjoint_deflation_functional)
    # Hold an immutable-for-this-solve snapshot.  Besides insulating the
    # nested GMRES action from caller mutation, this keeps the final true
    # residual valid when the caller deliberately aliases its custom
    # functional with `destination`.
    effective_adjoint_functional=adjoint_action ?
        work.cached_adjoint_functional : nothing
    reuse&&work.warm_start_valid||_no_jump_iterative_reset_linear_workspace!(work)
    function right_preconditioned_action!(output,input)
        _apply_no_jump_iterative_preconditioner!(work.preconditioned,plan,input,work,
                                      lambda,delta;adjoint_action,
                                      adjoint_deflation_functional=
                                          effective_adjoint_functional)
        _apply_no_jump_iterative_shifted_deflated!(output,plan,work.preconditioned,
                                        work,lambda,delta;adjoint_action,
                                        adjoint_deflation_functional=
                                            effective_adjoint_functional)
    end
    inner=recycled_gmres!(work.transformed_solution,
        right_preconditioned_action!,work.rhs,work.gmres;
        atol=atolT,rtol=rtolT,maxiter,target=zero(plan.Ttype),
        preconditioner=nothing,
        require_convergence=false,recycle_extraction)
    work.warm_start_valid=true
    _apply_no_jump_iterative_preconditioner!(work.physical_solution,plan,
        work.transformed_solution,work,lambda,delta;adjoint_action,
        adjoint_deflation_functional=effective_adjoint_functional)
    _apply_no_jump_iterative_shifted_deflated!(work.residual,plan,
                                    work.physical_solution,work,
                                    lambda,delta;adjoint_action,
                                    adjoint_deflation_functional=
                                        effective_adjoint_functional)
    @. work.residual=work.residual-work.rhs
    residual=norm(work.residual)
    residual_inf=norm(work.residual,Inf)
    physical_residual_inf=_no_jump_iterative_physical_maximum(plan.basis,work.residual)
    image_scale=zero(_real_float_type(plan.Ttype))
    @inbounds for index in eachindex(work.residual,work.rhs)
        image_scale=max(image_scale,abs(work.residual[index]+work.rhs[index]))
    end
    scale=max(norm(work.rhs,Inf),image_scale,floatmin(typeof(image_scale)))
    tolerance=atolT+rtolT*scale
    converged=inner.converged&&residual_inf<=tolerance
    if require_convergence&&!converged
        throw(ArgumentError(
            "no-jump-resolvent iterative right-preconditioned GMRES did not converge in " *
            "$(inner.iterations) iterations; true residual=$residual_inf, " *
            "tolerance=$tolerance"))
    end
    # Copy only after the independent true-residual calculation.  The two
    # intentional internal destination/RHS buffers therefore remain safe.
    copyto!(destination,work.physical_solution)
    (solution=destination,converged,residual,residual_inf,
     physical_residual_inf,residual_tolerance=tolerance,
     shift=lambda,deflation=delta,iterations=inner.iterations,
     restarts=inner.restarts,operator_applications=inner.operator_applications,
     recycle_dimension=inner.recycle_dimension,
     recycled_initially=inner.recycled_initially,
     recycle_extraction=inner.recycle_extraction,
     recycle_extraction_used=inner.recycle_extraction_used,
     transformed_residual=inner.residual,
     transformed_projected_residual=inner.projected_residual,
     preconditioner_denominator=work.denominator,
     right_preconditioned=true,adjoint_action,workspace_reused=reuse)
end

"""
    no_jump_iterative_resolvent(plan, right_hand_side; shift, workspace=nothing, ...)

Allocating wrapper for [`no_jump_iterative_resolvent!`](@ref). The returned named tuple
contains the solution vector and true residual diagnostics.
"""
function no_jump_iterative_resolvent(plan::NoJumpIterativePlan,
        right_hand_side::AbstractVector;shift::Number,workspace=nothing,
        krylovdim::Integer=30,recycle_dim::Integer=8,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,kwargs...)
    work=workspace===nothing ? NoJumpIterativeWorkspace(plan;krylovdim,recycle_dim,
        memory_budget) : _check_no_jump_iterative_workspace(workspace,plan)
    output_bytes=_performance_entries_bytes(
        BigInt(length(plan.basis)),plan.Ttype)
    estimate=work.accounted_peak_bytes+
        _no_jump_iterative_external_vector_bytes(work,right_hand_side)+output_bytes
    _require_performance_budget("allocating no-jump-resolvent iterative resolvent",estimate,
        memory_budget;guidance=
            "Use no_jump_iterative_resolvent! with caller-owned storage or increase " *
            "the budget.")
    destination=zeros(plan.Ttype,length(plan.basis))
    no_jump_iterative_resolvent!(destination,plan,right_hand_side,work;
        shift,memory_budget,kwargs...)
end

struct _NoJumpIterativeSizedOperator{F,T,W}
    n::Int
    action!::F
    Ttype::Type{T}
    action_workspace::W
end
size(operator::_NoJumpIterativeSizedOperator)=(operator.n,operator.n)
size(operator::_NoJumpIterativeSizedOperator,index::Integer)=
    index in (1,2) ? operator.n : 1
eltype(operator::_NoJumpIterativeSizedOperator)=operator.Ttype
function mul!(destination::AbstractVector,operator::_NoJumpIterativeSizedOperator,
              source::AbstractVector)
    operator.action!(destination,source)
    destination
end
# The internal sized adapter applies into its already-retained, explicitly
# task-owned no-jump-resolvent iterative workspace. It has no additional per-application vector
# allowance. High-level no-jump-resolvent iterative consumers compose that retained workspace with
# their outer Krylov storage explicitly before allocating it.
_performance_source_action_bytes(::_NoJumpIterativeSizedOperator,::Type)=big(0)

function _no_jump_iterative_stationary_diagnostics(plan,state,work,atol,rtol)
    _apply_no_jump_iterative_liouvillian!(
        work.residual,plan,state.data,work.liouvillian)
    residual=norm(work.residual)
    residual_inf=norm(work.residual,Inf)
    physical_residual_inf=_no_jump_iterative_physical_maximum(plan.basis,work.residual)
    trace_error=abs(dot(plan.tracevec,state.data)-one(plan.Ttype))
    diagnostics=state_diagnostics(state;atol,rtol)
    (;residual,residual_inf,physical_residual_inf,trace_error,
      state_diagnostics=diagnostics)
end

function _require_no_jump_iterative_stationary_result(diagnostics,atol,rtol)
    R=typeof(diagnostics.physical_residual_inf)
    tolerance=R(atol)+R(rtol)
    diagnostics.physical_residual_inf<=tolerance||throw(ArgumentError(
        "no-jump-resolvent iterative stationary candidate fails the true Liouvillian residual " *
        "check: physical max residual=$(diagnostics.physical_residual_inf), " *
        "tolerance=$tolerance"))
    diagnostics.state_diagnostics.valid||throw(ArgumentError(
        "no-jump-resolvent iterative stationary candidate is not a valid density state within " *
        "the requested tolerances; diagnostics=$(diagnostics.state_diagnostics)"))
    nothing
end

function _no_jump_iterative_stationary_linear_tolerances(plan::NoJumpIterativePlan,deflation,
        atol,rtol)
    R=_real_float_type(plan.Ttype)
    state_atol,state_rtol=_no_jump_iterative_check_tolerances(atol,rtol,R)
    state_trace_tolerance=state_atol+state_rtol
    trace_norm=R(norm(plan.tracevec))
    rhs_norm=abs(deflation)*R(norm(plan.deflation_vector))
    trace_norm>zero(R)&&isfinite(trace_norm)||throw(ArgumentError(
        "no-jump-resolvent iterative trace functional has zero or nonfinite norm"))
    rhs_norm>zero(R)&&isfinite(rhs_norm)||throw(ArgumentError(
        "no-jump-resolvent iterative trace-deflated stationary right-hand side has zero or " *
        "nonfinite norm"))

    # If r=A*x-b for A=-L-delta*|I/D><trace|, trace preservation and
    # <trace|I/D>=1 imply
    #
    #     delta*(1-tr(x)) = <trace|r>.
    #
    # Therefore an absolute coordinate residual that looks excellent can
    # still give a poor trace when ||trace|| grows as sqrt(d^N).  Budget half
    # of a fourfold safety margin to each absolute/relative GMRES component.
    target=(abs(deflation)/trace_norm)*state_trace_tolerance/R(4)
    isfinite(target)||throw(ArgumentError(
        "no-jump-resolvent iterative trace-aware stationary residual target is nonfinite"))
    iszero(target)&&!iszero(deflation)&&!iszero(state_trace_tolerance)&&
        throw(ArgumentError(
            "no-jump-resolvent iterative trace-aware stationary residual target underflowed in " *
            "the prepared precision; use a wider scalar type"))
    inner_atol=min(state_atol,target/R(2))
    inner_rtol=min(state_rtol,target/(R(2)*rhs_norm))
    (;state_atol,state_rtol,state_trace_tolerance,trace_norm,rhs_norm,
      residual_target=target,inner_atol,inner_rtol)
end

"""
    no_jump_iterative_steady_state(source; method=:gmres, ...)

Compute a PI stationary state with the no-jump-resolvent iterative jump/no-jump split. `source` may
be a [`PIModel`](@ref), [`CompiledPIModel`](@ref),
[`SpecializedPIModel`](@ref), or prepared [`NoJumpIterativePlan`](@ref).

`method=:gmres` solves the trace-deflated system with the exact no-jump right
preconditioner. `method=:fixed_point` applies thick-restarted Arnoldi to the
CPTP map ``Phi=K*R_0^S=I+L*R_0^S`` and selects the Ritz value nearest one;
it never assumes that power iteration converges. Both routes validate the raw,
unnormalized Liouvillian residual and the returned state's trace,
Hermiticity, and positivity. They do not clip eigenvalues or symmetrize an
invalid result. Uniqueness is an assumption of the method, not a certificate;
use [`evans_uniqueness`](@ref) separately when applicable.

For `method=:gmres`, the internal residual tolerance is tightened from the
requested state tolerance and the norm of the physical trace functional. This
prevents a small coefficient-space absolute residual from being amplified
into an unacceptable trace error at larger `N`; the solution is not normalized
or otherwise repaired after the solve.
"""
function no_jump_iterative_steady_state(plan::NoJumpIterativePlan;method::Symbol=:gmres,
        workspace=nothing,krylovdim::Integer=30,recycle_dim::Integer=8,
        deflation::Real=1,maxiter::Integer=500,maxrestarts::Integer=20,
        atol::Real=1e-10,rtol::Real=1e-8,return_info::Bool=false,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,
        rng=Random.default_rng())
    method in (:gmres,:fixed_point)||throw(ArgumentError(
        "no-jump-resolvent iterative steady-state method must be :gmres or :fixed_point"))
    R=_real_float_type(plan.Ttype)
    atolT,rtolT=_no_jump_iterative_check_tolerances(atol,rtol,R)
    # The fixed-point route uses outer Arnoldi but not the embedded GMRES
    # basis. Keep its task workspace mode-specific instead of retaining a
    # second, unused n-by-krylovdim basis.
    work=workspace===nothing ?
        (method===:gmres ? NoJumpIterativeWorkspace(plan;krylovdim,recycle_dim,
            memory_budget) : NoJumpIterativeWorkspace(plan;krylovdim=1,
            recycle_dim=0,memory_budget)) :
        _check_no_jump_iterative_workspace(workspace,plan)
    if method===:gmres
        _require_performance_budget("no-jump-resolvent iterative stationary-state solve",
            work.accounted_peak_bytes+
                _no_jump_iterative_stationary_output_bytes(plan),
            memory_budget;guidance=
                "Reduce Krylov capacity or increase the budget.")
        delta=_no_jump_iterative_solver_scalar(plan,deflation,"no-jump-resolvent iterative deflation")
        iszero(delta)&&throw(ArgumentError(
            "steady-state trace deflation must be positive"))
        @. work.rhs=-delta*plan.deflation_vector
        trace_control=_no_jump_iterative_stationary_linear_tolerances(
            plan,delta,atolT,rtolT)
        linear=no_jump_iterative_resolvent!(work.physical_solution,plan,work.rhs,work;
            shift=0,deflation=delta,reuse=false,maxiter,
            atol=trace_control.inner_atol,rtol=trace_control.inner_rtol,
            memory_budget)
        state=PIState(plan.basis,work.physical_solution)
        stationary=_no_jump_iterative_stationary_diagnostics(
            plan,state,work,atolT,rtolT)
        _require_no_jump_iterative_stationary_result(stationary,atolT,rtolT)
        # The in-place linear result points at task-owned workspace storage.
        # Retain the detached state-owned copy in high-level diagnostics so a
        # later solve with the same workspace cannot mutate an older result.
        detached_linear=merge(linear,(solution=state.data,))
        info=(state,method=:no_jump_iterative_gmres,converged=true,
              residual=stationary.residual,
              residual_inf=stationary.residual_inf,
              physical_residual_inf=stationary.physical_residual_inf,
              trace_error=stationary.trace_error,
              state_diagnostics=stationary.state_diagnostics,
              linear_solver=detached_linear,
              trace_control,
              backend=plan.no_jump.metadata.backend,
              generator_mode=plan.metadata.generator_mode,
              unique_steady_state=:assumed_not_certified)
        return return_info ? info : state
    end

    _no_jump_iterative_shift(plan.no_jump,0)
    !(krylovdim isa Bool)&&krylovdim>0||throw(ArgumentError(
        "krylovdim must be a positive integer"))
    !(maxrestarts isa Bool)&&maxrestarts>=0||throw(ArgumentError(
        "maxrestarts must be a nonnegative integer"))
    fixed_point_peak=work.accounted_peak_bytes+
        _no_jump_iterative_outer_arnoldi_bytes(
            plan,1,krylovdim,maxrestarts;vectors=true)+
        _performance_entries_bytes(BigInt(length(plan.basis)),plan.Ttype)+
        _no_jump_iterative_stationary_output_bytes(plan)
    _require_performance_budget("no-jump-resolvent iterative fixed-point stationary solve",
        fixed_point_peak,memory_budget;guidance=
            "Reduce krylovdim/maxrestarts or increase the budget.")
    function fixed_point_action!(destination,source)
        no_jump_resolvent!(work.preconditioned,plan.no_jump,source,0,
                           work.no_jump)
        _apply_no_jump_iterative_gain!(
            destination,plan,work.preconditioned,work.liouvillian)
    end
    operator=_NoJumpIterativeSizedOperator(length(plan.basis),fixed_point_action!,
                                  plan.Ttype,work)
    initial=reshape(copy(plan.deflation_vector),:,1)
    arnoldi=block_arnoldi_spectrum(operator;nev=1,block_size=1,
        krylovdim,retained_dimension=1,maxrestarts,which=:LM,target=1,
        initial_subspace=initial,atol=atolT,rtol=rtolT,vectors=true,rng,
        require_convergence=false,memory_budget=Inf)
    xi=view(arnoldi.vectors,:,1)
    no_jump_resolvent!(work.physical_solution,plan.no_jump,xi,0,
                       work.no_jump)
    normalization=dot(plan.tracevec,work.physical_solution)
    isfinite(real(normalization))&&isfinite(imag(normalization))&&
        !iszero(normalization)||throw(ArgumentError(
            "no-jump-resolvent iterative fixed-point Ritz vector reconstructs a state with zero " *
            "or nonfinite trace"))
    work.physical_solution ./= normalization
    state=PIState(plan.basis,work.physical_solution)
    stationary=_no_jump_iterative_stationary_diagnostics(
        plan,state,work,atolT,rtolT)
    _require_no_jump_iterative_stationary_result(stationary,atolT,rtolT)
    info=(state,method=:no_jump_iterative_fixed_point,converged=true,
          residual=stationary.residual,residual_inf=stationary.residual_inf,
          physical_residual_inf=stationary.physical_residual_inf,
          trace_error=stationary.trace_error,
          state_diagnostics=stationary.state_diagnostics,
          fixed_point_value=arnoldi.values[1],
          fixed_point_residual=arnoldi.residuals[1],
          fixed_point_action=:direct_gain,
          arnoldi,backend=plan.no_jump.metadata.backend,
          generator_mode=plan.metadata.generator_mode,
          unique_steady_state=:assumed_not_certified)
    return_info ? info : state
end

function _no_jump_iterative_plan_from_source(source;backend,memory_budget,
        coefficient_cache,time,parameters,kwargs...)
    if source isa PIModel
        return NoJumpIterativePlan(source;backend,memory_budget,coefficient_cache,
                          time,parameters,kwargs...)
    end
    coefficient_cache===nothing||throw(ArgumentError(
        "`coefficient_cache` is used only while lowering a PIModel; it is " *
        "irrelevant for an already prepared $(typeof(source))"))
    _no_jump_iterative_reject_evaluation_arguments(source,time,parameters)
    NoJumpIterativePlan(source;backend,memory_budget,kwargs...)
end

function no_jump_iterative_steady_state(
        source::Union{PIModel,CompiledPIModel,SpecializedPIModel};
        backend::Symbol=:schur,memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,
        coefficient_cache=nothing,time=nothing,parameters=nothing,kwargs...)
    plan=_no_jump_iterative_plan_from_source(source;backend,memory_budget,
        coefficient_cache,time,parameters)
    no_jump_iterative_steady_state(plan;memory_budget,kwargs...)
end

"""
    no_jump_iterative_liouvillian_spectrum(source; nev=3, shift=0, deflation=1,
                                candidate_oversampling=nothing, ...)

Compute selected nonzero low-lying Liouvillian modes with thick-restarted
shift-invert Arnoldi. Every inverse action is a no-jump-resolvent iterative right-preconditioned,
recycled GMRES solve of the trace-deflated generator. Transformed Ritz values
`nu` are mapped back as ``lambda=shift-1/nu``. Returned modes are selected and
validated with fresh residuals of the original, undeflated Liouvillian; inner
or transformed convergence is never reported as a physical eigenpair
certificate.

The default candidate window adds `max(2, nev)` transformed Ritz values to
allow for the deflated stationary direction and rejected numerical modes.
Increase `candidate_oversampling` together with `krylovdim` when a nonunique
generator or a crowded target region needs a larger filtering window.
"""
function no_jump_iterative_liouvillian_spectrum(plan::NoJumpIterativePlan;nev::Integer=3,
        shift::Real=0,deflation::Real=1,workspace=nothing,
        krylovdim::Integer=max(30,3nev+6),retained_dimension=nothing,
        candidate_oversampling=nothing,
        maxrestarts::Integer=20,inner_krylovdim::Integer=30,
        inner_recycle_dim::Integer=8,inner_maxiter::Integer=500,
        inner_atol::Real=1e-11,inner_rtol::Real=1e-9,
        atol::Real=1e-9,rtol::Real=1e-7,vectors::Bool=false,
        rng=Random.default_rng(),require_convergence::Bool=true,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    n=length(plan.basis);!(nev isa Bool)&&0<nev<n||throw(ArgumentError(
        "nev must lie between 1 and dimension-1 for nonzero slow modes"))
    R=_real_float_type(plan.Ttype)
    atolT,rtolT=_no_jump_iterative_check_tolerances(atol,rtol,R)
    inner_atolT,inner_rtolT=
        _no_jump_iterative_check_tolerances(inner_atol,inner_rtol,R)
    lambda=_no_jump_iterative_solver_scalar(plan,shift,"no-jump-resolvent iterative spectral shift")
    delta=_no_jump_iterative_solver_scalar(plan,deflation,"no-jump-resolvent iterative deflation")
    iszero(delta)&&throw(ArgumentError(
        "low-mode shift-invert requires positive trace deflation"))
    !(krylovdim isa Bool)&&krylovdim>0||throw(ArgumentError(
        "krylovdim must be a positive integer"))
    !(maxrestarts isa Bool)&&maxrestarts>=0||throw(ArgumentError(
        "maxrestarts must be a nonnegative integer"))
    oversampling=candidate_oversampling===nothing ? max(2,Int(nev)) : begin
        candidate_oversampling isa Integer&&
            !(candidate_oversampling isa Bool)&&candidate_oversampling>=0||
            throw(ArgumentError(
                "candidate_oversampling must be a nonnegative integer or nothing"))
        BigInt(candidate_oversampling)<=typemax(Int)||throw(ArgumentError(
            "candidate_oversampling must be representable as an Int"))
        Int(candidate_oversampling)
    end
    candidate_total=min(BigInt(n),BigInt(nev)+BigInt(oversampling))
    candidates=Int(candidate_total)
    candidates<=min(n,krylovdim)||throw(ArgumentError(
        "krylovdim=$krylovdim is too small for nev=$nev plus " *
        "candidate_oversampling=$oversampling; enlarge krylovdim or reduce " *
        "candidate_oversampling"))
    work=workspace===nothing ? NoJumpIterativeWorkspace(plan;
        krylovdim=inner_krylovdim,recycle_dim=inner_recycle_dim,
        memory_budget) : _check_no_jump_iterative_workspace(workspace,plan)
    _no_jump_iterative_reset_linear_workspace!(work)
    inner_solves=Ref(0);inner_iterations=Ref(0)
    maximum_inner_residual=Ref(zero(R))
    function inverse_action!(destination,source)
        info=no_jump_iterative_resolvent!(destination,plan,source,work;
            shift=lambda,deflation=delta,reuse=inner_solves[]>0,
            maxiter=inner_maxiter,atol=inner_atolT,rtol=inner_rtolT,
            require_convergence=true,memory_budget=Inf)
        inner_solves[]+=1
        inner_iterations[]+=info.iterations
        maximum_inner_residual[]=max(maximum_inner_residual[],
                                     info.residual_inf)
        destination
    end
    operator=_NoJumpIterativeSizedOperator(n,inverse_action!,plan.Ttype,work)
    retained=retained_dimension===nothing ?
        min(max(2candidates,candidates),max(krylovdim-1,candidates)) :
        retained_dimension
    selection_bytes=_performance_entries_bytes(
        16BigInt(candidates),plan.Ttype)+
        (vectors ? _performance_entries_bytes(
            BigInt(n)*BigInt(nev),plan.Ttype) : big(0))
    spectrum_peak=work.accounted_peak_bytes+
        _no_jump_iterative_outer_arnoldi_bytes(
            plan,candidates,krylovdim,maxrestarts;vectors=true)+
        selection_bytes
    _require_performance_budget("no-jump-resolvent iterative shift-invert spectrum",
        spectrum_peak,memory_budget;guidance=
            "Reduce krylovdim/nev/candidate_oversampling, request " *
            "vectors=false, or increase the budget.")
    outer=block_arnoldi_spectrum(operator;nev=candidates,block_size=1,
        krylovdim,retained_dimension=retained,maxrestarts,which=:LM,
        atol=inner_atolT,rtol=inner_rtolT,vectors=true,rng,
        require_convergence=false,memory_budget=Inf)
    values=Complex{R}[]
    residuals=R[]
    physical_residuals=R[]
    selected_columns=Int[]
    trace_errors=R[]
    zero_exclusion_tolerance=atolT+rtolT*max(one(R),abs(lambda))
    for column in axes(outer.vectors,2)
        nu=outer.values[column]
        iszero(nu)&&continue
        value=lambda-inv(nu)
        isfinite(real(value))&&isfinite(imag(value))||continue
        vector=view(outer.vectors,:,column)
        _apply_no_jump_iterative_liouvillian!(
            work.image,plan,vector,work.liouvillian)
        @. work.residual=work.image-value*vector
        residual=norm(work.residual)
        physical=_no_jump_iterative_physical_maximum(plan.basis,work.residual)
        trace_error=abs(dot(plan.tracevec,vector))
        scale=max(_no_jump_iterative_physical_maximum(plan.basis,work.image),
                  abs(value)*_no_jump_iterative_physical_maximum(plan.basis,vector),
                  floatmin(R))
        tolerance=atolT+rtolT*scale
        # The deflated stationary direction is not traceless and also fails
        # the original-L eigenpair residual at its shifted value.  A second,
        # explicit zero test is required for nonunique generators: another
        # traceless stationary direction can otherwise pass both certificates.
        if abs(value)>zero_exclusion_tolerance&&physical<=tolerance&&
                trace_error<=atolT+rtolT
            push!(values,value);push!(residuals,residual)
            push!(physical_residuals,physical);push!(trace_errors,trace_error)
            push!(selected_columns,column)
        end
    end
    order=sortperm(values;by=value->abs(value-lambda))
    if length(order)<nev
        message="no-jump-resolvent iterative shift-invert produced only $(length(order)) of $nev " *
            "requested true-residual-certified nonzero modes; tighten inner " *
            "tolerances, enlarge krylovdim/maxrestarts, or increase " *
            "candidate_oversampling when stationary or rejected candidates " *
            "occupy the requested window"
        require_convergence&&throw(ArgumentError(message))
    end
    keep=order[1:min(nev,length(order))]
    kept_values=values[keep]
    kept_residuals=residuals[keep]
    kept_physical=physical_residuals[keep]
    kept_traces=trace_errors[keep]
    kept_vectors=vectors ?
        (isempty(keep) ? zeros(plan.Ttype,n,0) :
            Matrix(view(outer.vectors,:,selected_columns[keep]))) : nothing
    converged=length(keep)==nev
    outer_diagnostics=Base.structdiff(outer,(vectors=nothing,))
    result=(values=kept_values,residuals=kept_residuals,
        physical_residuals=kept_physical,trace_errors=kept_traces,
        converged,method=:no_jump_iterative_shift_invert,shift=lambda,
        deflation=delta,outer=outer_diagnostics,inner_solves=inner_solves[],
        inner_iterations=inner_iterations[],
        maximum_inner_residual=maximum_inner_residual[],
        zero_exclusion_tolerance,
        candidate_count=candidates,candidate_oversampling=oversampling,
        backend=plan.no_jump.metadata.backend,
        generator_mode=plan.metadata.generator_mode,
        unique_steady_state=:assumed_not_certified)
    vectors ? merge(result,(vectors=kept_vectors,)) : result
end

function no_jump_iterative_liouvillian_spectrum(
        source::Union{PIModel,CompiledPIModel,SpecializedPIModel};
        backend::Symbol=:schur,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,
        coefficient_cache=nothing,time=nothing,parameters=nothing,kwargs...)
    plan=_no_jump_iterative_plan_from_source(source;backend,memory_budget,
        coefficient_cache,time,parameters)
    no_jump_iterative_liouvillian_spectrum(plan;memory_budget,kwargs...)
end

"""
    no_jump_iterative_implicit_euler_step!(destination, plan, source, dt, workspace; ...)

Advance one autonomous master-equation step with
``destination=(I-dt*L)^(-1)*source``. The solve is written at the positive
shift `1/dt`, where the no-jump-resolvent iterative no-jump error map is a strict trace-norm
contraction. The source is copied before solving, so `destination === source`
is supported. No trace normalization or positivity repair is performed.
"""
function no_jump_iterative_implicit_euler_step!(destination::PIState,plan::NoJumpIterativePlan,
        source::PIState,dt::Real,work::NoJumpIterativeWorkspace;
        reuse::Bool=true,kwargs...)
    destination.basis===plan.basis&&source.basis===plan.basis||
        throw(ArgumentError(
            "implicit-Euler states and no-jump-resolvent iterative plan use different PI bases"))
    _check_no_jump_iterative_workspace(work,plan)
    _check_no_jump_iterative_linear_arrays(destination.data,source.data,plan)
    dt>0&&isfinite(dt)||throw(ArgumentError(
        "implicit-Euler step must be finite and positive"))
    R=_real_float_type(plan.Ttype)
    step=_checked_prepared_real(
        dt,R,"implicit-Euler step";nonzero=true)
    shift=_no_jump_iterative_solver_scalar(plan,inv(step),"implicit-Euler shift")
    @inbounds @simd for index in eachindex(work.rhs,source.data)
        work.rhs[index]=shift*source.data[index]
    end
    info=no_jump_iterative_resolvent!(destination.data,plan,work.rhs,work;
        shift,deflation=zero(R),reuse,kwargs...)
    merge(info,(step=step,method=:no_jump_iterative_implicit_euler,))
end

"""
    no_jump_iterative_implicit_euler(source, initial_state, times; ...)

Evolve an autonomous PI model on a strictly increasing saved-time grid with
first-order implicit Euler and the no-jump-resolvent iterative positive-shift resolvent. Returns a
named tuple containing detached `PIState`s and per-step linear-solver
diagnostics. This is a stiff, low-order integration option; perform time-step
convergence before interpreting physical results. If a driven `PIModel` was
prepared with `time` and `parameters`, every step uses that one frozen
instantaneous generator; `generator_mode` records this explicitly.
"""
function no_jump_iterative_implicit_euler(plan::NoJumpIterativePlan,initial_state::PIState,times;
        workspace=nothing,krylovdim::Integer=30,recycle_dim::Integer=8,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,check::Bool=true,
        atol::Real=1e-10,rtol::Real=1e-8,kwargs...)
    initial_state.basis===plan.basis||throw(ArgumentError(
        "initial state and no-jump-resolvent iterative plan use different PI bases"))
    promote_type(plan.Ttype,eltype(initial_state.data))===plan.Ttype||
        throw(ArgumentError(
            "initial-state scalar type $(eltype(initial_state.data)) is " *
            "wider than prepared precision $(plan.Ttype)"))
    !(krylovdim isa Bool)&&krylovdim>0||throw(ArgumentError(
        "krylovdim must be a positive integer"))
    !(recycle_dim isa Bool)&&recycle_dim>=0||throw(ArgumentError(
        "recycle_dim must be a nonnegative integer"))
    applicable(length,times)||throw(ArgumentError(
        "times must have a known finite length so saved-output memory can " *
        "be checked before allocation"))
    saved_count=length(times)
    saved_count>0||throw(ArgumentError("times cannot be empty"))
    prepared_workspace=workspace===nothing ? nothing :
        _check_no_jump_iterative_workspace(workspace,plan)
    effective_krylovdim=prepared_workspace===nothing ? krylovdim :
        size(prepared_workspace.gmres.H,2)
    effective_recycle_dim=prepared_workspace===nothing ? recycle_dim :
        size(prepared_workspace.gmres.U,2)
    workspace_peak=prepared_workspace===nothing ?
        BigInt(Base.summarysize(plan))+
            _no_jump_iterative_workspace_estimate(
                plan,effective_krylovdim,effective_recycle_dim) :
        prepared_workspace.accounted_peak_bytes
    retained_entries=(BigInt(saved_count)+2)*BigInt(length(plan.basis))
    diagnostic_entries=48*max(BigInt(saved_count)-1,big(0))
    R=_real_float_type(plan.Ttype)
    output_bytes=_performance_entries_bytes(
        retained_entries,plan.Ttype)+
        _performance_entries_bytes(BigInt(saved_count),R)+
        _performance_entries_bytes(diagnostic_entries,plan.Ttype)
    estimate=workspace_peak+output_bytes+
        _performance_entries_bytes(
            BigInt(length(initial_state.data)),eltype(initial_state.data))+
        _no_jump_iterative_state_validation_bytes(plan)
    _require_performance_budget("no-jump-resolvent iterative implicit-Euler saved history",
        estimate,memory_budget;guidance=
            "Save fewer times, reduce Krylov capacity, or increase the budget.")
    saved=Vector{R}(undef,saved_count)
    yielded=0
    for time in times
        yielded+=1
        yielded<=saved_count||throw(ArgumentError(
            "times yielded more values than its reported length"))
        saved[yielded]=_checked_prepared_real(
            time,R,"saved time $yielded")
    end
    yielded==saved_count||throw(ArgumentError(
        "times yielded fewer values than its reported length"))
    all(index->saved[index]>saved[index-1],2:length(saved))||
        throw(ArgumentError("times must be strictly increasing"))
    check&&validate_state(initial_state;atol,rtol)
    work=prepared_workspace===nothing ? NoJumpIterativeWorkspace(
        plan;krylovdim,recycle_dim,memory_budget) : prepared_workspace
    initial=PIState(plan.basis;T=_real_float_type(plan.Ttype))
    copyto!(initial.data,initial_state.data)
    states=Vector{typeof(initial)}(undef,length(saved))
    states[1]=initial
    diagnostics=NamedTuple[]
    current=copy(states[1]);next=PIState(plan.basis;T=_real_float_type(plan.Ttype))
    for index in 2:length(saved)
        info=no_jump_iterative_implicit_euler_step!(next,plan,current,
            saved[index]-saved[index-1],work;reuse=index>2,atol,rtol,
            memory_budget,kwargs...)
        check&&validate_state(next;atol,rtol)
        states[index]=copy(next)
        # Redirect the in-place solver's solution reference to this detached
        # saved state. Otherwise diagnostics from alternate steps would alias
        # the two rotating integration buffers and change retroactively.
        info=merge(info,(solution=states[index].data,))
        current,next=next,current
        push!(diagnostics,info)
    end
    (times=saved,states,diagnostics,converged=all(info->info.converged,
        diagnostics),method=:no_jump_iterative_implicit_euler,order=1,
        backend=plan.no_jump.metadata.backend,
        generator_mode=plan.metadata.generator_mode,
        unique_steady_state=:not_applicable)
end

function no_jump_iterative_implicit_euler(
        source::Union{PIModel,CompiledPIModel,SpecializedPIModel},
        initial_state::PIState,times;backend::Symbol=:schur,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,
        coefficient_cache=nothing,time=nothing,parameters=nothing,kwargs...)
    plan=_no_jump_iterative_plan_from_source(source;backend,memory_budget,
        coefficient_cache,time,parameters)
    no_jump_iterative_implicit_euler(plan,initial_state,times;memory_budget,kwargs...)
end
