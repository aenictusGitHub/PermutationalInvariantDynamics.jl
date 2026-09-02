# Complex trace-deflated inexact shift-invert spectra -----------------------

"""
    TraceDeflatedShiftInvertPlan(source; shift=0, deflation=1, ...)

Immutable complex-shift spectral plan for an autonomous, trace-preserving PI
GKSL generator.  The transformed operator is
``(shift-L-deflation*|I/D><trace|)^(-1)`` and every inverse action is solved
inexactly by right-preconditioned GMRES with the prepared sectorwise no-jump
resolvent.  `shift` may be complex.  `deflation` must be finite and strictly
positive.  The rank-one update leaves every traceless nonzero mode unchanged
and displaces the traceful zero branch; the maximally mixed vector need not
itself be the physical stationary state.

This is an advanced spectral plan.  Complex shifts do not carry the
positive-real contraction or CPTP claims of [`no_jump_resolvent`](@ref).
Returned physical modes are always certified against the original undeflated
Liouvillian.
"""
struct TraceDeflatedShiftInvertPlan{P,T,R,M}
    no_jump_iterative::P
    shift::T
    deflation::R
    metadata::M
end

size(plan::TraceDeflatedShiftInvertPlan)=size(plan.no_jump_iterative)
size(plan::TraceDeflatedShiftInvertPlan,index::Integer)=
    size(plan.no_jump_iterative,index)
eltype(plan::TraceDeflatedShiftInvertPlan)=
    eltype(plan.no_jump_iterative)
isautonomous(::TraceDeflatedShiftInvertPlan)=true

function TraceDeflatedShiftInvertPlan(plan::NoJumpIterativePlan;
        shift::Number=0,deflation::Real=1)
    lambda=_no_jump_iterative_spectral_shift(
        plan,shift,"trace-deflated shift-invert shift")
    delta=_no_jump_iterative_solver_scalar(
        plan,deflation,"trace-deflated shift-invert deflation")
    delta>zero(delta)||throw(ArgumentError(
        "trace-deflated shift-invert requires a strictly positive deflation"))
    lambda==convert(plan.Ttype,delta)&&throw(ArgumentError(
        "shift equals the deliberately displaced stationary eigenvalue; " *
        "choose a different shift or deflation"))
    metadata=(algorithm=:inexact_shiftinvert,
        outer_restart=:implicit_qr_arnoldi,
        preconditioner=:sectorwise_no_jump_resolvent,
        shift=lambda,deflation=delta,
        trace_deflated=true,original_residual_certification=true,
        complex_shift=!iszero(imag(lambda)),
        positive_real_contraction_guarantee=false,
        unique_steady_state=:assumed_not_certified)
    TraceDeflatedShiftInvertPlan(plan,lambda,delta,metadata)
end

function TraceDeflatedShiftInvertPlan(
        source::Union{PIModel,CompiledPIModel,SpecializedPIModel};
        shift::Number=0,deflation::Real=1,backend::Symbol=:schur,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,
        coefficient_cache=nothing,time=nothing,parameters=nothing,kwargs...)
    plan=_no_jump_iterative_plan_from_source(source;backend,memory_budget,
        coefficient_cache,time,parameters,kwargs...)
    TraceDeflatedShiftInvertPlan(plan;shift,deflation)
end

"""
    TraceDeflatedShiftInvertWorkspace(plan; outer_krylovdim=40,
        inner_krylovdim=30, inner_recycle_dim=0, memory_budget=512*1024^2)

Fixed-capacity, task-owned storage for
[`trace_deflated_shiftinvert_spectrum`](@ref).  It owns one full implicit-QR
Arnoldi workspace and one no-jump-preconditioned GMRES workspace.  The default
does not recycle between unrelated Arnoldi right-hand sides; nonzero
`inner_recycle_dim` is an explicit inexact-solver heuristic and never weakens
the final original-Liouvillian residual checks.
"""
struct TraceDeflatedShiftInvertWorkspace{P,L,A}
    plan::P
    linear::L
    outer::A
    accounted_peak_bytes::BigInt
end

function _trace_deflated_checked_int(value::Integer,label::AbstractString;
        minimum::Integer=0)
    !(value isa Bool)&&value>=minimum||throw(ArgumentError(
        "$label must be an integer greater than or equal to $minimum"))
    BigInt(value)<=typemax(Int)||throw(ArgumentError(
        "$label must be representable as an Int"))
    Int(value)
end

function _trace_deflated_shiftinvert_workspace_bytes(plan,
        outer_krylovdim,inner_krylovdim,inner_recycle_dim)
    n=size(plan,1);T=eltype(plan)
    BigInt(Base.summarysize(plan))+
        _no_jump_iterative_workspace_estimate(plan.no_jump_iterative,
            inner_krylovdim,inner_recycle_dim)+
        _performance_arnoldi_bytes(n,T,outer_krylovdim;mode=:full)
end

function TraceDeflatedShiftInvertWorkspace(plan::TraceDeflatedShiftInvertPlan;
        outer_krylovdim::Integer=40,inner_krylovdim::Integer=30,
        inner_recycle_dim::Integer=0,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    outer_dimension=_trace_deflated_checked_int(
        outer_krylovdim,"outer_krylovdim";minimum=2)
    inner_dimension=_trace_deflated_checked_int(
        inner_krylovdim,"inner_krylovdim";minimum=1)
    recycle_dimension=_trace_deflated_checked_int(
        inner_recycle_dim,"inner_recycle_dim";minimum=0)
    estimate=_trace_deflated_shiftinvert_workspace_bytes(plan,
        outer_dimension,inner_dimension,recycle_dimension)
    _require_performance_budget("trace-deflated shift-invert workspace",
        estimate,memory_budget;guidance=
        "Reduce outer/inner Krylov dimensions or increase the budget.")
    n=size(plan,1);T=eltype(plan)
    TraceDeflatedShiftInvertWorkspace(plan,
        NoJumpIterativeWorkspace(plan.no_jump_iterative;
            krylovdim=inner_dimension,recycle_dim=recycle_dimension,
            memory_budget=Inf),
        ArnoldiWorkspace(T,n,outer_dimension;mode=:full),BigInt(estimate))
end

function _check_trace_deflated_shiftinvert_workspace(
        work::TraceDeflatedShiftInvertWorkspace,
        plan::TraceDeflatedShiftInvertPlan;outer_krylovdim=nothing)
    work.plan===plan||throw(ArgumentError(
        "trace-deflated shift-invert workspace belongs to a different plan"))
    _check_no_jump_iterative_workspace(work.linear,plan.no_jump_iterative)
    if outer_krylovdim!==nothing
        required=min(size(plan,1),Int(outer_krylovdim))
        _check_arnoldi_workspace(work.outer,size(plan,1),required;mode=:full)
    end
    work
end

mutable struct _InexactShiftInvertController{R}
    adaptive::Bool
    minimum_atol::R
    minimum_rtol::R
    current_atol::R
    current_rtol::R
    initial_atol::R
    initial_rtol::R
    safety::R
    decay::R
    history::Vector{NamedTuple}
end

mutable struct _NoJumpInexactShiftInvertOperator{P,W,C,T,R,F}
    plan::P
    workspace::W
    controller::C
    operator_shift::T
    adjoint_action::Bool
    adjoint_deflation_functional::F
    reuse_inner::Bool
    inner_maxiter::Int
    required_modes::Int
    physical_atol::R
    physical_rtol::R
    zero_tolerance::R
    inner_solves::Int
    inner_iterations::Int
    inner_restarts::Int
    maximum_inner_residual::R
    maximum_inner_residual_ratio::R
    cycle_start_solves::Int
    cycle_start_iterations::Int
    cycle_maximum_inner_residual::R
    cycle_maximum_inner_residual_ratio::R
end

size(operator::_NoJumpInexactShiftInvertOperator)=
    size(operator.plan)
size(operator::_NoJumpInexactShiftInvertOperator,index::Integer)=
    size(operator.plan,index)
eltype(operator::_NoJumpInexactShiftInvertOperator)=eltype(operator.plan)

function LinearAlgebra.mul!(destination::AbstractVector,
        operator::_NoJumpInexactShiftInvertOperator,
        source::AbstractVector)
    controller=operator.controller
    info=no_jump_iterative_resolvent!(destination,
        operator.plan.no_jump_iterative,source,operator.workspace.linear;
        shift=operator.operator_shift,
        deflation=operator.plan.deflation,
        adjoint_action=operator.adjoint_action,
        adjoint_deflation_functional=
            operator.adjoint_deflation_functional,
        reuse=operator.reuse_inner&&operator.inner_solves>0,
        maxiter=operator.inner_maxiter,
        atol=controller.current_atol,rtol=controller.current_rtol,
        require_convergence=true,memory_budget=Inf)
    operator.inner_solves+=1
    operator.inner_iterations+=info.iterations
    operator.inner_restarts+=info.restarts
    operator.maximum_inner_residual=max(
        operator.maximum_inner_residual,info.residual_inf)
    residual_ratio=info.residual_inf/max(
        info.residual_tolerance,floatmin(typeof(info.residual_tolerance)))
    operator.maximum_inner_residual_ratio=max(
        operator.maximum_inner_residual_ratio,residual_ratio)
    operator.cycle_maximum_inner_residual=max(
        operator.cycle_maximum_inner_residual,info.residual_inf)
    operator.cycle_maximum_inner_residual_ratio=max(
        operator.cycle_maximum_inner_residual_ratio,residual_ratio)
    destination
end

function _shiftinvert_original_action!(destination,operator,source)
    plan=operator.plan.no_jump_iterative
    if operator.adjoint_action
        _no_jump_iterative_source_adjoint!(
            destination,plan.liouvillian,source,
            operator.workspace.linear.liouvillian)
    else
        _apply_no_jump_iterative_liouvillian!(
            destination,plan,source,operator.workspace.linear.liouvillian)
    end
    destination
end

_shiftinvert_stationary_functional(operator)=operator.adjoint_action ?
    operator.adjoint_deflation_functional :
    operator.plan.no_jump_iterative.tracevec

function _shiftinvert_candidate_diagnostics!(operator,V,Y,transformed_values)
    work=operator.workspace.linear
    R=_real_float_type(eltype(operator));T=eltype(operator)
    values=Complex{R}[];residuals=R[];relative_residuals=R[]
    physical_residuals=R[];physical_scales=R[]
    overlaps=R[];normalized_overlaps=R[]
    finite_candidates=Int[]
    functional=_shiftinvert_stationary_functional(operator)
    functional_norm=R(norm(functional))
    for column in eachindex(transformed_values)
        nu=transformed_values[column]
        iszero(nu)&&continue
        value=operator.operator_shift-inv(nu)
        isfinite(real(value))&&isfinite(imag(value))||continue
        mul!(work.rhs,V,view(Y,:,column))
        vector_norm=R(norm(work.rhs))
        iszero(vector_norm)&&continue
        work.rhs ./= vector_norm
        _shiftinvert_original_action!(work.image,operator,work.rhs)
        @. work.residual=work.image-value*work.rhs
        residual=R(norm(work.residual))
        physical=_no_jump_iterative_physical_maximum(
            operator.plan.no_jump_iterative.basis,work.residual)
        vector_scale=_no_jump_iterative_physical_maximum(
            operator.plan.no_jump_iterative.basis,work.rhs)
        image_scale=_no_jump_iterative_physical_maximum(
            operator.plan.no_jump_iterative.basis,work.image)
        scale=max(image_scale,abs(value)*vector_scale,floatmin(R))
        overlap=R(abs(dot(functional,work.rhs)))
        normalized=functional_norm>zero(R) ?
            overlap/(functional_norm*R(norm(work.rhs))) : R(Inf)
        push!(values,Complex{R}(value));push!(residuals,residual)
        push!(relative_residuals,physical/scale)
        push!(physical_residuals,physical);push!(physical_scales,scale)
        push!(overlaps,overlap)
        push!(normalized_overlaps,normalized);push!(finite_candidates,column)
    end
    (;values,residuals,relative_residuals,physical_residuals,
      physical_scales,overlaps,normalized_overlaps,finite_candidates)
end

function _implicit_arnoldi_restart_feedback!(
        operator::_NoJumpInexactShiftInvertOperator,V,Y,values,residuals,
        cycle,tolerance)
    diagnostics=_shiftinvert_candidate_diagnostics!(operator,V,Y,values)
    order=sortperm(diagnostics.values;
        by=value->abs(value-operator.operator_shift))
    accepted=Int[]
    overlap_tolerance=operator.physical_atol+operator.physical_rtol
    for index in order
        value=diagnostics.values[index]
        scale=diagnostics.physical_scales[index]
        physical_tolerance=operator.physical_atol+
            operator.physical_rtol*scale
        if abs(value)>operator.zero_tolerance&&
                diagnostics.normalized_overlaps[index]<=overlap_tolerance&&
                diagnostics.physical_residuals[index]<=physical_tolerance
            push!(accepted,index)
        end
    end
    metric=if isempty(order)
        one(operator.physical_rtol)
    else
        relevant=order[1:min(operator.required_modes,length(order))]
        maximum(diagnostics.relative_residuals[relevant];init=
            zero(operator.physical_rtol))
    end
    controller=operator.controller
    previous_atol=controller.current_atol
    previous_rtol=controller.current_rtol
    if controller.adaptive
        controller.current_rtol=max(controller.minimum_rtol,min(
            controller.current_rtol*controller.decay,
            controller.safety*metric))
        controller.current_atol=max(controller.minimum_atol,min(
            controller.current_atol*controller.decay,
            controller.safety*metric))
    end
    cycle_solves=operator.inner_solves-operator.cycle_start_solves
    cycle_iterations=operator.inner_iterations-operator.cycle_start_iterations
    feedback=(accept=length(accepted)>=operator.required_modes,
        certified_modes=length(accepted),candidate_modes=length(order),
        physical_relative_residual=metric,
        inner_atol=previous_atol,inner_rtol=previous_rtol,
        next_inner_atol=controller.current_atol,
        next_inner_rtol=controller.current_rtol,
        inner_solves=cycle_solves,inner_iterations=cycle_iterations,
        maximum_inner_residual=operator.cycle_maximum_inner_residual,
        maximum_inner_residual_ratio=
            operator.cycle_maximum_inner_residual_ratio,
        transformed_tolerance=tolerance)
    push!(controller.history,feedback)
    operator.cycle_start_solves=operator.inner_solves
    operator.cycle_start_iterations=operator.inner_iterations
    operator.cycle_maximum_inner_residual=zero(
        operator.cycle_maximum_inner_residual)
    operator.cycle_maximum_inner_residual_ratio=zero(
        operator.cycle_maximum_inner_residual_ratio)
    feedback
end

function _shiftinvert_tolerance_pair(plan,atol,rtol,label)
    R=_real_float_type(eltype(plan))
    checked=_no_jump_iterative_check_tolerances(atol,rtol,R)
    checked[1],checked[2]
end

function _shiftinvert_initial_tolerance(value,minimum,::Type{R},label) where R
    value===nothing&&return minimum
    checked=_no_jump_iterative_check_tolerances(value,zero(R),R)[1]
    checked>=minimum||throw(ArgumentError(
        "$label must not be smaller than the requested final tolerance"))
    checked
end

function _shiftinvert_unit_interval(value,::Type{R},label;
        allow_one::Bool=false) where R
    value isa Real&&!(value isa Bool)&&isfinite(value)||throw(ArgumentError(
        "$label must be a finite real number"))
    converted=_no_jump_iterative_check_tolerances(
        value,zero(R),R)[1]
    (zero(R)<converted&&(allow_one ? converted<=one(R) : converted<one(R)))||
        throw(ArgumentError("$label must lie in (0, $(allow_one ? "1]" : "1"))"))
    converted
end

function _run_trace_deflated_shiftinvert_side(plan,work;
        nev,candidates,krylovdim,retained_dimension,maxrestarts,
        inner_maxiter,inner_atol,inner_rtol,inner_initial_atol,
        inner_initial_rtol,adaptive_inner,inner_safety,inner_decay,
        reuse_inner,atol,rtol,transformed_atol,transformed_rtol,
        initial_vector,rng,adjoint_action,
        adjoint_deflation_functional=nothing)
    R=_real_float_type(eltype(plan))
    current_atol=_shiftinvert_initial_tolerance(
        inner_initial_atol,inner_atol,R,"inner_initial_atol")
    default_initial_rtol=min(R(1e-3),max(inner_rtol,sqrt(inner_rtol)))
    current_rtol=_shiftinvert_initial_tolerance(
        inner_initial_rtol===nothing ? default_initial_rtol :
            inner_initial_rtol,inner_rtol,R,"inner_initial_rtol")
    adaptive_inner||(current_atol=inner_atol;current_rtol=inner_rtol)
    controller=_InexactShiftInvertController(adaptive_inner,
        inner_atol,inner_rtol,current_atol,current_rtol,current_atol,
        current_rtol,inner_safety,inner_decay,NamedTuple[])
    sizehint!(controller.history,maxrestarts+1)
    operator_shift=adjoint_action ? conj(plan.shift) : plan.shift
    stationary_functional=adjoint_action ? begin
        adjoint_deflation_functional===nothing&&throw(ArgumentError(
            "the adjoint shift-invert solve requires a stationary-state " *
            "deflation functional"))
        adjoint_deflation_functional
    end : nothing
    zero_tolerance=atol+rtol*max(one(R),abs(operator_shift))
    operator=_NoJumpInexactShiftInvertOperator(plan,work,controller,
        operator_shift,adjoint_action,stationary_functional,reuse_inner,
        Int(inner_maxiter),Int(nev),
        atol,rtol,zero_tolerance,0,0,0,zero(R),zero(R),0,0,
        zero(R),zero(R))
    _no_jump_iterative_reset_linear_workspace!(work.linear)
    outer=implicitly_restarted_arnoldi_spectrum(operator;nev=candidates,
        krylovdim,retained_dimension,maxrestarts,which=:LM,target=nothing,
        initial_vector,atol=transformed_atol,rtol=transformed_rtol,
        vectors=true,rng,require_convergence=false,workspace=work.outer)
    # The outer result is only a candidate generator.  Re-evaluate every
    # mapped pair against the original, undeflated Liouvillian.
    identity_coefficients=Matrix{eltype(plan)}(I,
        length(outer.values),length(outer.values))
    diagnostics=_shiftinvert_candidate_diagnostics!(operator,
        outer.vectors,identity_coefficients,outer.values)
    eligible=Int[]
    overlap_tolerance=atol+rtol
    for index in eachindex(diagnostics.values)
        scale=diagnostics.physical_scales[index]
        tolerance=atol+rtol*scale
        if abs(diagnostics.values[index])>zero_tolerance&&
                diagnostics.normalized_overlaps[index]<=overlap_tolerance&&
                diagnostics.physical_residuals[index]<=tolerance
            push!(eligible,index)
        end
    end
    sort!(eligible;by=index->abs(
        diagnostics.values[index]-operator_shift))
    keep=eligible[1:min(Int(nev),length(eligible))]
    columns=diagnostics.finite_candidates[keep]
    vectors=isempty(columns) ? zeros(eltype(plan),size(plan,1),0) :
        Matrix(view(outer.vectors,:,columns))
    for column in axes(vectors,2)
        nrm=norm(view(vectors,:,column));iszero(nrm)||
            (view(vectors,:,column)./=nrm)
    end
    result=(values=diagnostics.values[keep],vectors,
        residuals=diagnostics.residuals[keep],
        physical_residuals=diagnostics.physical_residuals[keep],
        relative_residuals=diagnostics.relative_residuals[keep],
        stationary_overlap_errors=diagnostics.overlaps[keep],
        normalized_stationary_overlap_errors=
            diagnostics.normalized_overlaps[keep],
        zero_exclusion_tolerance=zero_tolerance,
        converged=length(keep)==nev,
        outer=outer,inner_solves=operator.inner_solves,
        inner_iterations=operator.inner_iterations,
        inner_restarts=operator.inner_restarts,
        maximum_inner_residual=operator.maximum_inner_residual,
        maximum_inner_residual_ratio=
            operator.maximum_inner_residual_ratio,
        adaptive_inner,inner_tolerance_history=copy(controller.history),
        final_inner_atol=controller.current_atol,
        final_inner_rtol=controller.current_rtol,
        adjoint_action)
    result
end

function _minimum_cost_assignment(cost::AbstractMatrix{R}) where R<:Real
    n,m=size(cost);n==m||throw(DimensionMismatch(
        "mode-assignment cost matrix must be square"))
    all(isfinite,cost)||throw(ArgumentError(
        "mode-assignment costs must be finite"))
    u=zeros(R,n+1);v=zeros(R,m+1);p=zeros(Int,m+1);way=zeros(Int,m+1)
    for row in 1:n
        p[1]=row;j0=1;minv=fill(R(Inf),m+1);used=falses(m+1)
        while true
            used[j0]=true;i0=p[j0];delta=R(Inf);j1=0
            for j in 2:m+1
                used[j]&&continue
                current=cost[i0,j-1]-u[i0+1]-v[j]
                if current<minv[j]
                    minv[j]=current;way[j]=j0
                end
                if minv[j]<delta
                    delta=minv[j];j1=j
                end
            end
            isfinite(delta)&&j1>0||throw(ArgumentError(
                "mode assignment is numerically singular"))
            for j in 1:m+1
                if used[j]
                    u[p[j]+1]+=delta;v[j]-=delta
                else
                    minv[j]-=delta
                end
            end
            j0=j1;p[j0]==0&&break
        end
        while true
            j1=way[j0];p[j0]=p[j1];j0=j1;j0==1&&break
        end
    end
    assignment=zeros(Int,n)
    for j in 2:m+1
        assignment[p[j]]=j-1
    end
    assignment
end

function _biorthogonal_mode_diagnostics_bytes(n,count,::Type{T}) where T
    R=_real_float_type(T)
    _performance_entries_bytes(
        10BigInt(n)*BigInt(count)+12BigInt(count)^2,T)+
        _performance_entries_bytes(8BigInt(count)^2+8BigInt(count),R)
end

function _require_shiftinvert_spectral_scalar(::Type{T},label) where T
    R=_real_float_type(T)
    R in (Float32,Float64)||throw(ArgumentError(
        "$label currently requires a Float32 or Float64 real component " *
        "because its projected eigenvalue/SVD backend is LAPACK; convert " *
        "the inputs explicitly or use a supported wider-precision backend"))
    nothing
end

function _check_biorthogonal_input(array,label)
    S=eltype(array)
    S<:Number&&isconcretetype(S)||throw(ArgumentError(
        "$label must expose a concrete numeric eltype"))
    all(value->isfinite(real(value))&&isfinite(imag(value)),array)||
        throw(ArgumentError("$label must contain only finite values"))
    nothing
end


function _normalize_biorthogonal_column!(column,label,::Type{R}) where R
    scale=zero(R)
    @inbounds for value in column
        scale=max(scale,abs(real(value)),abs(imag(value)))
    end
    isfinite(scale)&&scale>zero(R)||throw(ArgumentError(
        "$label must have a finite, nonzero component scale; rescale the " *
        "mode or use a wider supported scalar type"))
    column ./= scale
    nrm=R(norm(column))
    isfinite(nrm)&&nrm>zero(R)||throw(ArgumentError(
        "$label could not be normalized finitely; rescale the mode or use " *
        "a wider supported scalar type"))
    column ./= nrm
    all(value->isfinite(real(value))&&isfinite(imag(value)),column)||
        throw(ArgumentError("$label normalization produced nonfinite values"))
    column
end

"""
    biorthogonal_mode_diagnostics(values, right_vectors,
        adjoint_values, left_vectors; ...)

Globally pair right eigenmodes of `L` with left eigenmodes represented as
right eigenvectors of `L'`.  Pairing minimizes
`abs(conj(values[i])-adjoint_values[j])`.  For a simple mode the function
reports the scale-invariant eigenvalue condition number
``norm(l)*norm(r)/abs(dot(l,r))`` and returns copies normalized so that
`norm(r)==1` and, when resolvable, `dot(l,r)==1`.

Numerically clustered modes are reported as clusters with singular values of
their left/right overlap matrix.  They are not silently mixed or repaired;
selected iterative modes cannot by themselves certify defectiveness.  The
projected SVD backend currently supports Float32 and Float64 real components;
unsupported wider scalar types are rejected before factorization.
"""
function biorthogonal_mode_diagnostics(values::AbstractVector,
        right_vectors::AbstractMatrix,adjoint_values::AbstractVector,
        left_vectors::AbstractMatrix;pairing_atol::Real=1e-10,
        pairing_rtol::Real=1e-8,cluster_atol=nothing,cluster_rtol=nothing,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    count=length(values);length(adjoint_values)==count||throw(DimensionMismatch(
        "right and adjoint spectra must contain the same number of modes"))
    size(right_vectors,2)==count&&size(left_vectors,2)==count||
        throw(DimensionMismatch("mode-vector column count is inconsistent"))
    size(right_vectors,1)==size(left_vectors,1)||throw(DimensionMismatch(
        "left and right mode vectors have different dimensions"))
    count<=size(right_vectors,1)||throw(DimensionMismatch(
        "the number of selected modes cannot exceed the vector dimension"))
    _check_biorthogonal_input(values,"right eigenvalues")
    _check_biorthogonal_input(adjoint_values,"adjoint eigenvalues")
    _check_biorthogonal_input(right_vectors,"right mode vectors")
    _check_biorthogonal_input(left_vectors,"left mode vectors")
    T=promote_type(_complex_float_type(eltype(right_vectors)),
        _complex_float_type(eltype(left_vectors)),
        _complex_float_type(eltype(values)),
        _complex_float_type(eltype(adjoint_values)))
    _require_shiftinvert_spectral_scalar(T,"biorthogonal mode diagnostics")
    R=_real_float_type(T)
    atol,rtol=_no_jump_iterative_check_tolerances(pairing_atol,pairing_rtol,R)
    catol=cluster_atol===nothing ? atol :
        _no_jump_iterative_check_tolerances(cluster_atol,zero(R),R)[1]
    crtol=cluster_rtol===nothing ? sqrt(eps(R)) :
        _no_jump_iterative_check_tolerances(zero(R),cluster_rtol,R)[2]
    n=size(right_vectors,1)
    peak=_biorthogonal_mode_diagnostics_bytes(n,count,T)
    _require_performance_budget("biorthogonal mode diagnostics",peak,
        memory_budget;guidance=
        "Reduce the selected mode count or increase the budget.")
    costs=Matrix{R}(undef,count,count)
    @inbounds for column in 1:count,row in 1:count
        costs[row,column]=abs(conj(T(values[row]))-T(adjoint_values[column]))
    end
    assignment=_minimum_cost_assignment(costs)
    Rvectors=Matrix{T}(right_vectors)
    Lvectors=Matrix{T}(left_vectors[:,assignment])
    paired_adjoint=T.(adjoint_values[assignment])
    pairing_errors=R[costs[index,assignment[index]] for index in 1:count]
    condition_numbers=Vector{R}(undef,count)
    reciprocal_condition_numbers=Vector{R}(undef,count)
    overlaps=Vector{T}(undef,count)
    pairing_matched=BitVector(undef,count)
    normalized=BitVector(undef,count)
    statuses=Vector{Symbol}(undef,count)
    threshold=sqrt(eps(R))
    for index in 1:count
        right=view(Rvectors,:,index);left=view(Lvectors,:,index)
        _normalize_biorthogonal_column!(right,"right mode vector $index",R)
        _normalize_biorthogonal_column!(left,"left mode vector $index",R)
        overlap=dot(left,right);overlaps[index]=overlap
        reciprocal=R(abs(overlap))
        reciprocal_condition_numbers[index]=reciprocal
        condition_numbers[index]=iszero(reciprocal) ? R(Inf) : inv(reciprocal)
        scale=max(one(R),abs(T(values[index])))
        paired=pairing_errors[index]<=atol+rtol*scale
        pairing_matched[index]=paired
        if reciprocal>threshold
            left./=conj(overlap);normalized[index]=true
            statuses[index]=paired ? :ok : :pairing_mismatch
        else
            normalized[index]=false
            statuses[index]=paired ? :defective_or_unresolved :
                :pairing_mismatch_and_defective_or_unresolved
        end
    end
    overlap_matrix=adjoint(Lvectors)*Rvectors
    identity_matrix=Matrix{T}(I,count,count)
    biorthogonality_error=R(norm(overlap_matrix-identity_matrix,Inf))
    cluster_ids=zeros(Int,count);clusters=NamedTuple[];next_cluster=0
    for seed in 1:count
        cluster_ids[seed]!=0&&continue
        next_cluster+=1;members=Int[seed];cluster_ids[seed]=next_cluster
        changed=true
        while changed
            changed=false
            for candidate in 1:count
                cluster_ids[candidate]!=0&&continue
                if any(member->abs(T(values[candidate])-T(values[member]))<=
                        catol+crtol*max(one(R),abs(T(values[candidate])),
                            abs(T(values[member]))),members)
                    push!(members,candidate);cluster_ids[candidate]=next_cluster
                    changed=true
                end
            end
        end
        raw_left=Matrix{T}(left_vectors[:,assignment[members]])
        raw_right=Matrix{T}(right_vectors[:,members])
        for column in axes(raw_left,2)
            _normalize_biorthogonal_column!(view(raw_left,:,column),
                "cluster left mode vector $(members[column])",R)
            _normalize_biorthogonal_column!(view(raw_right,:,column),
                "cluster right mode vector $(members[column])",R)
        end
        left_singular_values=R.(svdvals(raw_left))
        right_singular_values=R.(svdvals(raw_right))
        rank_scale=eps(R)*R(max(size(raw_left)...))
        left_rank=Base.count(value->value>rank_scale*
            maximum(left_singular_values;init=zero(R)),left_singular_values)
        right_rank=Base.count(value->value>rank_scale*
            maximum(right_singular_values;init=zero(R)),right_singular_values)
        full_rank=left_rank==length(members)&&right_rank==length(members)
        singular_values=if full_rank
            qleft=Matrix(qr(raw_left).Q[:,1:length(members)])
            qright=Matrix(qr(raw_right).Q[:,1:length(members)])
            R.(svdvals(adjoint(qleft)*qright))
        else
            zeros(R,length(members))
        end
        minimum_singular=minimum(singular_values)
        projector_condition=full_rank&&minimum_singular>threshold ?
            inv(minimum_singular) : R(Inf)
        push!(clusters,(indices=Tuple(members),size=length(members),
            singular_values,left_rank,right_rank,
            left_basis_singular_values=left_singular_values,
            right_basis_singular_values=right_singular_values,
            projector_condition,
            status=isfinite(projector_condition) ?
                (length(members)==1 ? :simple : :cluster_unmixed) :
                :defective_or_unresolved))
    end
    pairing_converged=all(pairing_matched)
    clusters_resolved=all(cluster->
        cluster.status!==:defective_or_unresolved,clusters)
    diagnostics_complete=pairing_converged&&all(normalized)&&
        clusters_resolved
    (;values=T.(values),adjoint_values=paired_adjoint,
      right_vectors=Rvectors,left_vectors=Lvectors,assignment,
      pairing_errors,pairing_matched,overlaps_before_normalization=overlaps,
      normalized,condition_numbers,reciprocal_condition_numbers,statuses,
      overlap_matrix,biorthogonality_error,cluster_ids,clusters,
      pairing_converged,clusters_resolved,diagnostics_complete,
      defectiveness=:not_certified)
end

function _integrated_mode_residuals!(plan,work,values,right_vectors,
        left_vectors,stationary_state)
    count=length(values);R=_real_float_type(eltype(plan))
    right_residuals=zeros(R,count);left_residuals=zeros(R,count)
    right_relative=zeros(R,count);left_relative=zeros(R,count)
    trace_overlaps=zeros(R,count);stationary_overlaps=zeros(R,count)
    source=plan.no_jump_iterative
    for column in 1:count
        right=view(right_vectors,:,column);left=view(left_vectors,:,column)
        _apply_no_jump_iterative_liouvillian!(
            work.linear.image,source,right,work.linear.liouvillian)
        @. work.linear.residual=work.linear.image-values[column]*right
        right_residuals[column]=norm(work.linear.residual)
        right_scale=max(norm(work.linear.image),
            abs(values[column])*norm(right),floatmin(R))
        right_relative[column]=right_residuals[column]/right_scale
        _no_jump_iterative_source_adjoint!(work.linear.image,
            source.liouvillian,left,work.linear.liouvillian)
        @. work.linear.residual=work.linear.image-conj(values[column])*left
        left_residuals[column]=norm(work.linear.residual)
        left_scale=max(norm(work.linear.image),
            abs(values[column])*norm(left),floatmin(R))
        left_relative[column]=left_residuals[column]/left_scale
        trace_overlaps[column]=abs(dot(source.tracevec,right))/
            max(norm(source.tracevec)*norm(right),floatmin(R))
        stationary_overlaps[column]=abs(dot(stationary_state,left))/
            max(norm(stationary_state)*norm(left),floatmin(R))
    end
    (;right_residuals,left_residuals,right_relative_residuals=right_relative,
      left_relative_residuals=left_relative,trace_overlaps,
      stationary_overlaps)
end

"""
    trace_deflated_shiftinvert_spectrum(plan; nev=3, ...)

Compute nonzero Liouvillian modes nearest a finite real or complex `shift`
with an inexact, trace-deflated shift-invert implicit-QR Arnoldi method.  Inner
GMRES starts at `inner_initial_rtol` and tightens monotonically at each outer
restart toward `inner_rtol`, using the fresh original-Liouvillian residuals as
its forcing signal.  `inner_tolerance_history` records the complete schedule.

Set `mode_diagnostics=true` to solve the adjoint problem as well and report
globally paired left modes, biorthogonality, eigenvalue condition numbers, and
unmixed cluster diagnostics.  Left and right vectors are returned only when
`vectors=true`, but their temporary storage is always included in the memory
preflight.  A mode is accepted only after an original, undeflated
Liouvillian residual and the appropriate stationary-direction overlap check;
neither inner-GMRES nor transformed Ritz convergence is a certificate.
"""
function trace_deflated_shiftinvert_spectrum(
        plan::TraceDeflatedShiftInvertPlan;nev::Integer=3,
        workspace=nothing,krylovdim::Integer=max(30,3nev+8),
        retained_dimension::Integer=max(nev+2,min(2nev+2,krylovdim-1)),
        candidate_oversampling=nothing,maxrestarts::Integer=20,
        outer_restart::Symbol=:iram,
        inner_krylovdim=nothing,inner_recycle_dim=nothing,
        inner_maxiter::Integer=500,inner_atol::Real=1e-11,
        inner_rtol::Real=1e-9,inner_initial_atol=nothing,
        inner_initial_rtol=nothing,adaptive_inner::Bool=true,
        inner_safety::Real=0.1,inner_decay::Real=0.25,
        reuse_inner::Bool=false,transformed_atol::Real=1e-10,
        transformed_rtol::Real=1e-8,atol::Real=1e-9,
        rtol::Real=1e-7,vectors::Bool=false,
        mode_diagnostics::Bool=false,initial_vector=nothing,
        rng=Random.default_rng(),require_convergence::Bool=true,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    n=size(plan,1)
    _require_shiftinvert_spectral_scalar(eltype(plan),
        "trace-deflated shift-invert IRAM")
    outer_restart===:iram||throw(ArgumentError(
        "outer_restart must be :iram; a genuine Krylov--Schur restart is " *
        "not silently aliased to implicit-QR Arnoldi"))
    nev_int=_trace_deflated_checked_int(nev,"nev";minimum=1)
    nev_int<n||throw(ArgumentError(
        "nev must lie between 1 and dimension-1 for nonzero modes"))
    krylovdim_int=_trace_deflated_checked_int(
        krylovdim,"krylovdim";minimum=2)
    retained_int=_trace_deflated_checked_int(
        retained_dimension,"retained_dimension";minimum=1)
    maxrestarts_int=_trace_deflated_checked_int(
        maxrestarts,"maxrestarts";minimum=0)
    inner_maxiter_int=_trace_deflated_checked_int(
        inner_maxiter,"inner_maxiter";minimum=1)
    oversampling=candidate_oversampling===nothing ? max(2,nev_int) : begin
        candidate_oversampling isa Integer&&
            !(candidate_oversampling isa Bool)&&candidate_oversampling>=0||
            throw(ArgumentError(
                "candidate_oversampling must be a nonnegative integer or nothing"))
        _trace_deflated_checked_int(candidate_oversampling,
            "candidate_oversampling";minimum=0)
    end
    candidates=Int(min(BigInt(n),BigInt(nev_int)+BigInt(oversampling)))
    required_outer=min(n,krylovdim_int)
    candidates<=required_outer||throw(ArgumentError(
        "krylovdim is too small for nev plus candidate_oversampling"))
    candidates==n||required_outer>=candidates+1||throw(ArgumentError(
        "implicit-QR Arnoldi needs one expansion direction beyond the " *
        "candidate window; increase krylovdim"))
    retained_int<required_outer||n==required_outer||throw(ArgumentError(
        "retained_dimension must be smaller than krylovdim"))
    if workspace===nothing
        inner_krylovdim_int=inner_krylovdim===nothing ? 30 :
            (inner_krylovdim isa Integer ? _trace_deflated_checked_int(
                inner_krylovdim,"inner_krylovdim";minimum=1) :
             throw(ArgumentError("inner_krylovdim must be an integer or nothing")))
        inner_recycle_dim_int=inner_recycle_dim===nothing ? 0 :
            (inner_recycle_dim isa Integer ? _trace_deflated_checked_int(
                inner_recycle_dim,"inner_recycle_dim";minimum=0) :
             throw(ArgumentError("inner_recycle_dim must be an integer or nothing")))
    else
        _check_trace_deflated_shiftinvert_workspace(
            workspace,plan;outer_krylovdim=required_outer)
        inner_capacity=size(workspace.linear.gmres.H,2)
        recycle_capacity=size(workspace.linear.gmres.U,2)
        requested_inner=inner_krylovdim===nothing ? inner_capacity :
            (inner_krylovdim isa Integer ? _trace_deflated_checked_int(
                inner_krylovdim,"inner_krylovdim";minimum=1) :
             throw(ArgumentError("inner_krylovdim must be an integer or nothing")))
        requested_recycle=inner_recycle_dim===nothing ? recycle_capacity :
            (inner_recycle_dim isa Integer ? _trace_deflated_checked_int(
                inner_recycle_dim,"inner_recycle_dim";minimum=0) :
             throw(ArgumentError("inner_recycle_dim must be an integer or nothing")))
        effective_inner=min(n,requested_inner)
        effective_recycle=min(requested_recycle,max(n-1,0))
        effective_inner==inner_capacity||throw(ArgumentError(
            "inner_krylovdim=$requested_inner has effective capacity " *
            "$effective_inner, which does not match the supplied workspace " *
            "capacity $inner_capacity"))
        effective_recycle==recycle_capacity||throw(ArgumentError(
            "inner_recycle_dim=$requested_recycle has effective capacity " *
            "$effective_recycle, which does not match the supplied workspace " *
            "capacity $recycle_capacity"))
        inner_krylovdim_int=inner_capacity
        inner_recycle_dim_int=recycle_capacity
    end
    reuse_inner&&iszero(inner_recycle_dim_int)&&throw(ArgumentError(
        "reuse_inner=true requires a positive inner_recycle_dim or a " *
        "workspace with positive recycle capacity"))
    R=_real_float_type(eltype(plan))
    physical_atol,physical_rtol=_shiftinvert_tolerance_pair(
        plan,atol,rtol,"physical")
    final_inner_atol,final_inner_rtol=_shiftinvert_tolerance_pair(
        plan,inner_atol,inner_rtol,"inner")
    transformed_atolT,transformed_rtolT=_shiftinvert_tolerance_pair(
        plan,transformed_atol,transformed_rtol,"transformed")
    safety=_shiftinvert_unit_interval(inner_safety,R,"inner_safety";
        allow_one=true)
    decay=_shiftinvert_unit_interval(inner_decay,R,"inner_decay")
    initial_bytes=big(0)
    if initial_vector!==nothing
        length(initial_vector)==n||throw(DimensionMismatch(
            "initial_vector has the wrong length"))
        S=eltype(initial_vector)
        S<:Number&&isconcretetype(S)||throw(ArgumentError(
            "initial_vector must expose a concrete numeric eltype"))
        promote_type(eltype(plan),S)===eltype(plan)||throw(ArgumentError(
            "initial_vector is wider than the prepared shift-invert precision"))
        all(value->isfinite(real(value))&&isfinite(imag(value)),
            initial_vector)||throw(ArgumentError(
                "initial_vector must contain only finite values"))
        any(!iszero,initial_vector)||throw(ArgumentError(
            "initial_vector must be nonzero"))
        initial_bytes=_performance_entries_bytes(BigInt(n),S)
    end
    # Each side retains the detached outer candidate matrix and its filtered
    # vectors until the complete result has been assembled.  The diagnostics
    # path additionally owns normalized right/left copies.  Count these peak
    # temporaries even when vectors are omitted from the returned value.
    temporary_columns=mode_diagnostics ?
        2BigInt(candidates)+2BigInt(nev_int) :
        BigInt(candidates)+BigInt(nev_int)
    history_sides=mode_diagnostics ? big(2) : big(1)
    output_bytes=_performance_entries_bytes(
        BigInt(n)*temporary_columns+12BigInt(candidates)+
        8BigInt(nev_int)^2+
        32history_sides*(BigInt(maxrestarts_int)+1),
        eltype(plan))
    # The implicit-QR cycle forms projected eigensystems, accumulated QR
    # factors, and dense Hessenberg products on top of the retained Arnoldi
    # workspace.  Candidate certification also forms one small identity
    # coefficient matrix.  These are simultaneous transient allocations and
    # must remain guarded even though none is returned.
    outer_transient_bytes=_performance_entries_bytes(
        12BigInt(required_outer)^2+
        BigInt(required_outer)*BigInt(candidates)+BigInt(candidates)^2,
        eltype(plan))
    stationary_bytes=mode_diagnostics ?
        _no_jump_iterative_stationary_output_bytes(
            plan.no_jump_iterative) : big(0)
    diagnostics_bytes=mode_diagnostics ?
        _biorthogonal_mode_diagnostics_bytes(n,nev_int,eltype(plan)) : big(0)
    workspace_bytes=workspace===nothing ?
        _trace_deflated_shiftinvert_workspace_bytes(plan,required_outer,
            inner_krylovdim_int,inner_recycle_dim_int) :
        workspace.accounted_peak_bytes
    peak=workspace_bytes+output_bytes+outer_transient_bytes+initial_bytes+
        stationary_bytes+diagnostics_bytes
    _require_performance_budget("trace-deflated inexact shift-invert spectrum",
        peak,memory_budget;guidance=
        "Reduce krylovdim/nev/candidate_oversampling, disable mode " *
        "diagnostics, or increase the budget.")
    work=workspace===nothing ? TraceDeflatedShiftInvertWorkspace(plan;
        outer_krylovdim=required_outer,
        inner_krylovdim=inner_krylovdim_int,
        inner_recycle_dim=inner_recycle_dim_int,memory_budget=Inf) : workspace
    right=_run_trace_deflated_shiftinvert_side(plan,work;
        nev=nev_int,candidates,krylovdim=required_outer,
        retained_dimension=retained_int,maxrestarts=maxrestarts_int,
        inner_maxiter=inner_maxiter_int,inner_atol=final_inner_atol,
        inner_rtol=final_inner_rtol,inner_initial_atol,inner_initial_rtol,
        adaptive_inner,inner_safety=safety,inner_decay=decay,reuse_inner,
        atol=physical_atol,rtol=physical_rtol,
        transformed_atol=transformed_atolT,
        transformed_rtol=transformed_rtolT,initial_vector,rng,
        adjoint_action=false)
    if require_convergence&&!right.converged
        throw(ArgumentError(
            "inexact shift-invert IRAM produced only $(length(right.values)) " *
            "of $nev_int requested original-Liouvillian-certified modes; " *
            "increase krylovdim/maxrestarts or tighten inner tolerances"))
    end
    base=(values=right.values,residuals=right.residuals,
        physical_residuals=right.physical_residuals,
        relative_residuals=right.relative_residuals,
        trace_errors=right.stationary_overlap_errors,
        normalized_trace_errors=right.normalized_stationary_overlap_errors,
        converged=right.converged,
        method=:trace_deflated_inexact_shiftinvert_iram,
        outer_restart,
        shift=plan.shift,deflation=plan.deflation,
        outer=Base.structdiff(right.outer,(vectors=right.outer.vectors,)),
        inner_solves=right.inner_solves,
        inner_iterations=right.inner_iterations,
        inner_restarts=right.inner_restarts,
        maximum_inner_residual=right.maximum_inner_residual,
        maximum_inner_residual_ratio=right.maximum_inner_residual_ratio,
        adaptive_inner=right.adaptive_inner,
        inner_tolerance_history=right.inner_tolerance_history,
        final_inner_atol=right.final_inner_atol,
        final_inner_rtol=right.final_inner_rtol,
        zero_exclusion_tolerance=right.zero_exclusion_tolerance,
        candidate_count=candidates,candidate_oversampling=oversampling,
        backend=plan.no_jump_iterative.no_jump.metadata.backend,
        generator_mode=plan.no_jump_iterative.metadata.generator_mode,
        trace_deflated=true,original_residual_certification=true,
        unique_steady_state=:assumed_not_certified)
    if !mode_diagnostics
        return vectors ? merge(base,(vectors=right.vectors,)) : base
    end
    stationary=no_jump_iterative_steady_state(plan.no_jump_iterative;
        method=:gmres,workspace=work.linear,deflation=plan.deflation,
        maxiter=inner_maxiter_int,atol=physical_atol,rtol=physical_rtol,
        return_info=true,memory_budget=Inf)
    stationary_state=stationary.state.data
    left=_run_trace_deflated_shiftinvert_side(plan,work;
        nev=length(right.values),candidates,krylovdim=required_outer,
        retained_dimension=retained_int,maxrestarts=maxrestarts_int,
        inner_maxiter=inner_maxiter_int,inner_atol=final_inner_atol,
        inner_rtol=final_inner_rtol,inner_initial_atol,inner_initial_rtol,
        adaptive_inner,inner_safety=safety,inner_decay=decay,reuse_inner,
        atol=physical_atol,rtol=physical_rtol,
        transformed_atol=transformed_atolT,
        transformed_rtol=transformed_rtolT,initial_vector=nothing,rng,
        adjoint_action=true,
        adjoint_deflation_functional=stationary_state)
    if require_convergence&&!left.converged
        throw(ArgumentError(
            "adjoint inexact shift-invert IRAM produced only " *
            "$(length(left.values)) of $(length(right.values)) requested " *
            "certified left modes"))
    end
    count=min(length(right.values),length(left.values))
    right_values=right.values[1:count]
    right_vectors=right.vectors[:,1:count]
    diagnostics=biorthogonal_mode_diagnostics(right_values,right_vectors,
        left.values[1:count],left.vectors[:,1:count];
        pairing_atol=physical_atol,pairing_rtol=physical_rtol,
        memory_budget=Inf)
    residuals=_integrated_mode_residuals!(plan,work,right_values,
        diagnostics.right_vectors,diagnostics.left_vectors,
        stationary_state)
    diagnostic_summary=Base.structdiff(diagnostics,(
        right_vectors=diagnostics.right_vectors,
        left_vectors=diagnostics.left_vectors))
    mode_info=merge(diagnostic_summary,residuals,(
        right_solver=(inner_solves=right.inner_solves,
            inner_iterations=right.inner_iterations,
            maximum_inner_residual=right.maximum_inner_residual,
            maximum_inner_residual_ratio=
                right.maximum_inner_residual_ratio,
            tolerance_history=right.inner_tolerance_history),
        left_solver=(inner_solves=left.inner_solves,
            inner_iterations=left.inner_iterations,
            maximum_inner_residual=left.maximum_inner_residual,
            maximum_inner_residual_ratio=
                left.maximum_inner_residual_ratio,
            tolerance_history=left.inner_tolerance_history),
        stationary_state_solver=(method=stationary.method,
            residual=stationary.residual,
            residual_inf=stationary.residual_inf,
            physical_residual_inf=stationary.physical_residual_inf,
            trace_error=stationary.trace_error),
        original_liouvillian_certified=true))
    pairing_converged=diagnostics.pairing_converged
    clusters_resolved=diagnostics.clusters_resolved
    diagnostics_complete=diagnostics.diagnostics_complete
    require_convergence&&!pairing_converged&&throw(ArgumentError(
        "adjoint modes could not be paired with the certified right modes " *
        "within the requested physical tolerances; enlarge the candidate " *
        "window or tighten both eigensolves"))
    result=merge(base,(values=right_values,
        residuals=residuals.right_residuals,
        physical_residuals=right.physical_residuals[1:count],
        relative_residuals=right.relative_residuals[1:count],
        trace_errors=right.stationary_overlap_errors[1:count],
        normalized_trace_errors=
            right.normalized_stationary_overlap_errors[1:count],
        euclidean_right_relative_residuals=
            residuals.right_relative_residuals,
        mode_diagnostics=mode_info,
        condition_numbers=diagnostics.condition_numbers,
        reciprocal_condition_numbers=diagnostics.reciprocal_condition_numbers,
        left_values=diagnostics.adjoint_values,
        left_converged=left.converged,
        pairing_converged,clusters_resolved,diagnostics_complete,
        converged=right.converged&&left.converged&&count==nev_int&&
            pairing_converged))
    vectors ? merge(result,(vectors=diagnostics.right_vectors,
        left_vectors=diagnostics.left_vectors)) : result
end

function trace_deflated_shiftinvert_spectrum(source::NoJumpIterativePlan;
        shift::Number=0,deflation::Real=1,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,kwargs...)
    preparation_keywords=(:backend,:coefficient_cache,:time,:parameters)
    supplied=filter(key->key in preparation_keywords,keys(kwargs))
    isempty(supplied)||throw(ArgumentError(
        "a NoJumpIterativePlan is already prepared; keywords " *
        "$(Tuple(supplied)) cannot rebuild it implicitly"))
    plan=TraceDeflatedShiftInvertPlan(source;shift,deflation)
    trace_deflated_shiftinvert_spectrum(plan;memory_budget,kwargs...)
end

function trace_deflated_shiftinvert_spectrum(
        source::Union{PIModel,CompiledPIModel,SpecializedPIModel};
        shift::Number=0,
        deflation::Real=1,backend::Symbol=:schur,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,
        coefficient_cache=nothing,time=nothing,parameters=nothing,kwargs...)
    prepared=_no_jump_iterative_plan_from_source(source;backend,memory_budget,
        coefficient_cache,time,parameters)
    plan=TraceDeflatedShiftInvertPlan(prepared;shift,deflation)
    trace_deflated_shiftinvert_spectrum(plan;memory_budget,kwargs...)
end
