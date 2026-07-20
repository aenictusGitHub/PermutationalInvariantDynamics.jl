function _research_linear_operator(operator)
    operator isa PIModel&&return compile(operator;backend=:matrixfree)
    operator isa LiouvillianPlan&&return _matrixfree_liouvillian(operator)
    operator
end

function _research_apply!(destination,operator,source,work)
    if work===nothing
        operator isa CompiledPIModel ? apply!(destination,operator,source,0.0,nothing) :
        operator isa MatrixFreeLiouvillian ? apply!(destination,operator,source,0.0,nothing) :
        mul!(destination,operator,source)
    else
        apply!(destination,operator,source,0.0,nothing,work)
    end
    destination
end

function _research_apply_adjoint!(destination,operator,source,work)
    if operator isa AbstractMatrix
        mul!(destination,adjoint(operator),source)
    elseif work===nothing
        apply_adjoint!(destination,operator,source,0.0,nothing)
    else
        apply_adjoint!(destination,operator,source,0.0,nothing,work)
    end
    destination
end

"""
    SteadyStateGradientPlan(source, stationary_state; operator_scale=nothing,
                            atol=1e-10, rtol=1e-8)

Prepare the trace-fixed matrix-free linear system used for implicit stationary
state derivatives.  `stationary_state` is validated for trace and residual but
is never normalized or repaired.  The immutable plan may be shared; every
concurrent solve requires a separate [`SteadyStateGradientWorkspace`](@ref).
"""
struct SteadyStateGradientPlan{L,S,V,R}
    operator::L
    state::S
    trace_vector::V
    normalizer::V
    operator_scale::R
end

function SteadyStateGradientPlan(source,state::PIState;operator_scale=nothing,
        atol::Real=1e-10,rtol::Real=1e-8)
    operator=_research_linear_operator(source)
    size(operator)==(length(state.basis),length(state.basis))||
        throw(DimensionMismatch("stationary state and generator dimensions differ"))
    _require_autonomous(operator,"implicit steady-state gradients")
    T=promote_type(_complex_float_type(eltype(operator)),eltype(state.data))
    _check_liouvillian_source_precision(operator,T,"steady-state gradient state")
    trace_vector=T.(_trace_vector(state.basis,T));normalizer=trace_vector/dot(trace_vector,trace_vector)
    trace_error=abs(dot(trace_vector,state.data)-one(T))
    R=_real_float_type(T);tolerance=R(atol)+R(rtol)
    trace_error<=tolerance||throw(ArgumentError(
        "implicit-gradient reference state is not trace one: error=$trace_error"))
    probe=zeros(T,length(state.data));image=similar(probe)
    work=_linear_operator_workspace(operator)
    _research_apply!(image,operator,state.data,work)
    residual=norm(image)
    residual<=R(atol)+R(rtol)*max(norm(state.data),one(R))||throw(ArgumentError(
        "implicit-gradient reference state is not stationary: residual=$residual"))
    scale=_validated_operator_scale(operator_scale===nothing ?
        _estimated_operator_scale!(operator,probe,image) : operator_scale)
    SteadyStateGradientPlan(operator,copy(state),trace_vector,normalizer,scale)
end

"""Reusable GMRES and application scratch for implicit steady-state gradients."""
struct SteadyStateGradientWorkspace{V,K,W,P}
    rhs::V
    solution::V
    temporary::V
    residual::V
    krylov::K
    operator_work::W
    plan::P
end

function SteadyStateGradientWorkspace(plan::SteadyStateGradientPlan;
                                      krylovdim::Integer=30)
    T=eltype(plan.trace_vector);n=length(plan.trace_vector);v=zeros(T,n)
    SteadyStateGradientWorkspace(v,similar(v),similar(v),similar(v),
        KrylovWorkspace(T,n,krylovdim),_linear_operator_workspace(plan.operator),plan)
end

"""Implicit tangent states and optional observable gradients."""
struct SteadyStateGradientResult{S,G,I}
    tangents::Vector{S}
    observable_gradients::G
    solver_info::Vector{I}
end

"""
    implicit_steady_state_gradient(plan, derivatives; observables=nothing, ...)
    implicit_steady_state_gradient(source, state, derivatives; kwargs...)

Solve

```math
\\mathcal L\\,\\partial_\\mu\\rho_{ss}=-(\\partial_\\mu\\mathcal L)\\rho_{ss},
\\qquad \\mathrm{tr}(\\partial_\\mu\\rho_{ss})=0
```

with restarted matrix-free GMRES on a rank-one trace-fixed operator.  Neither
the Liouvillian nor its inverse is materialized.  `derivatives` may contain
matrices, compiled PI generators, or matrix-free PI generators.  Optional
`observables` returns `real(tr(A' * partial_mu rho))` in rows.
"""
function implicit_steady_state_gradient(plan::SteadyStateGradientPlan,
        derivatives;observables=nothing,krylovdim::Integer=30,
        maxiter::Integer=500,atol::Real=1e-10,rtol::Real=1e-8,
        workspace=nothing,preconditioner=nothing)
    prepared=map(_research_linear_operator,collect(derivatives))
    isempty(prepared)&&throw(ArgumentError(
        "at least one generator derivative is required"))
    w=workspace===nothing ? SteadyStateGradientWorkspace(plan;krylovdim) : workspace
    w.plan===plan||throw(ArgumentError(
        "steady-state gradient workspace belongs to a different plan"))
    size(w.krylov.H,2)>=min(krylovdim,length(plan.trace_vector))||
        throw(DimensionMismatch("steady-state gradient Krylov workspace is too small"))
    T=eltype(plan.trace_vector);R=_real_float_type(T);invscale=inv(plan.operator_scale)
    absolute=R(atol);relative=R(rtol);tangents=PIState[];infos=NamedTuple[]
    for derivative in prepared
        size(derivative)==size(plan.operator)||throw(DimensionMismatch(
            "generator derivative has the wrong dimensions"))
        derivative_work=_linear_operator_workspace(derivative)
        _research_apply!(w.rhs,derivative,plan.state.data,derivative_work)
        trace_forcing=abs(dot(plan.trace_vector,w.rhs))
        trace_forcing<=absolute+relative*max(norm(w.rhs),one(R))||throw(ArgumentError(
            "generator derivative is not trace preserving on the stationary state: trace forcing=$trace_forcing"))
        w.rhs.*=-invscale;fill!(w.solution,zero(T))
        function fixed_apply!(destination,source)
            _research_apply!(destination,plan.operator,source,w.operator_work)
            alpha=dot(plan.trace_vector,source)
            @inbounds @simd for index in eachindex(destination)
                destination[index]=invscale*destination[index]+plan.normalizer[index]*alpha
            end
            destination
        end
        info=_gmres!(w.solution,fixed_apply!,w.rhs,w.krylov;
            atol=absolute,rtol=relative,maxiter,preconditioner)
        _research_apply!(w.residual,plan.operator,w.solution,w.operator_work)
        _research_apply!(w.temporary,derivative,plan.state.data,derivative_work)
        @. w.residual=w.residual+w.temporary
        physical_residual=norm(w.residual)
        trace_error=abs(dot(plan.trace_vector,w.solution))
        tolerance=absolute+relative*max(norm(w.solution),one(R))
        converged=info.converged&&physical_residual/plan.operator_scale<=tolerance&&
                  trace_error<=absolute+relative
        converged||throw(ArgumentError(
            "implicit steady-state gradient GMRES did not converge: residual=$physical_residual, trace_error=$trace_error"))
        push!(tangents,PIState(plan.state.basis,w.solution))
        push!(infos,merge(info,(physical_residual,trace_error,converged=true,
                                operator_scale=plan.operator_scale)))
    end
    gradients=if observables===nothing
        nothing
    else
        ops=collect(observables)
        all(A->A isa PIOperator&&A.basis===plan.state.basis,ops)||throw(ArgumentError(
            "gradient observables must be PIOperators on the stationary-state basis"))
        [real(expectation(tangent,observable)) for observable in ops,
                                             tangent in tangents]
    end
    SteadyStateGradientResult(tangents,gradients,infos)
end

function implicit_steady_state_gradient(source,state::PIState,derivatives;kwargs...)
    plan=SteadyStateGradientPlan(source,state)
    implicit_steady_state_gradient(plan,derivatives;kwargs...)
end

"""
    AdjointControlResult

Terminal objective, piecewise-control gradient, final state, initial costate,
and checkpoint/recomputation metadata from [`checkpointed_adjoint_gradient`](@ref).
"""
struct AdjointControlResult{R,S,V,M}
    objective::R
    gradient::Matrix{R}
    final_state::S
    initial_adjoint::V
    metadata::M
end

function _control_action!(destination,source,base,derivatives,controls,
                          works,temporary;adjoint_action::Bool=false)
    if adjoint_action
        _research_apply_adjoint!(destination,base,source,works[1])
    else
        _research_apply!(destination,base,source,works[1])
    end
    for index in eachindex(derivatives)
        if adjoint_action
            _research_apply_adjoint!(temporary,derivatives[index],source,works[index+1])
        else
            _research_apply!(temporary,derivatives[index],source,works[index+1])
        end
        coefficient=controls[index]
        @inbounds @simd for coordinate in eachindex(destination)
            destination[coordinate]+=coefficient*temporary[coordinate]
        end
    end
    destination
end

function _control_rk4!(state,h,base,derivatives,controls,works,
        temporary,k1,k2,k3,k4,stage;adjoint_action::Bool=false)
    _control_action!(k1,state,base,derivatives,controls,works,temporary;
                     adjoint_action)
    @. stage=state+(h/2)*k1
    _control_action!(k2,stage,base,derivatives,controls,works,temporary;
                     adjoint_action)
    @. stage=state+(h/2)*k2
    _control_action!(k3,stage,base,derivatives,controls,works,temporary;
                     adjoint_action)
    @. stage=state+h*k3
    _control_action!(k4,stage,base,derivatives,controls,works,temporary;
                     adjoint_action)
    @. state=state+(h/6)*(k1+2k2+2k3+k4)
    state
end

"""
    checkpointed_adjoint_gradient(base, derivatives, rho0, objective,
                                  times, controls; checkpoint_stride=16)

Compute the terminal-objective gradient for piecewise-constant controls in
`L(t)=base+sum_mu controls[mu,i]*derivatives[mu]`.  `controls` has one column
per interval of `times`; `objective` is a Hermitian PI operator and the scalar
target is `real(tr(objective*rho(T)))`.

Forward states are retained only every `checkpoint_stride` intervals.  Each
segment is recomputed once during the backward continuous-adjoint RK4 sweep,
reducing retained state memory from `O(n_PI*n_times)` to
`O(n_PI*(n_times/checkpoint_stride + checkpoint_stride))`.  The reported
gradient uses endpoint trapezoidal quadrature; refine the control grid to
converge both propagation and gradient.  No dense Liouvillian is required.
"""
function checkpointed_adjoint_gradient(base,derivatives,rho0::PIState,
        objective::PIOperator,times,controls::AbstractMatrix;
        checkpoint_stride::Integer=16)
    objective.basis===rho0.basis||throw(ArgumentError(
        "control objective and initial state use different bases"))
    ishermitian(objective)||throw(ArgumentError(
        "control objective must be Hermitian"))
    checkpoint_stride>0||throw(ArgumentError("checkpoint_stride must be positive"))
    prepared_base=_research_linear_operator(base)
    prepared_derivatives=map(_research_linear_operator,collect(derivatives))
    m=length(prepared_derivatives);m>0||throw(ArgumentError(
        "at least one control derivative is required"))
    size(prepared_base)==(length(rho0.basis),length(rho0.basis))||
        throw(DimensionMismatch("base control generator has wrong dimensions"))
    all(D->size(D)==size(prepared_base),prepared_derivatives)||throw(DimensionMismatch(
        "a control derivative has the wrong dimensions"))
    _require_autonomous(prepared_base,"checkpointed adjoint control")
    all(isautonomous,prepared_derivatives)||throw(ArgumentError(
        "checkpointed adjoint control requires autonomous derivative generators"))
    raw_times=collect(times);length(raw_times)>=2||throw(ArgumentError(
        "control time grid needs at least two points"))
    intervals=length(raw_times)-1
    size(controls)==(m,intervals)||throw(DimensionMismatch(
        "controls must have size (number of derivatives, length(times)-1)"))
    T=foldl(promote_type,(eltype(operator) for operator in
        (prepared_base,prepared_derivatives...));init=eltype(rho0.data))
    T=promote_type(_complex_float_type(T),eltype(controls));R=_real_float_type(T)
    ts=R.(raw_times);all(isfinite,ts)&&all(diff(ts).>zero(R))||throw(ArgumentError(
        "control times must be finite and strictly increasing"))
    control_values=Matrix{R}(controls)
    works=(_linear_operator_workspace(prepared_base),
           map(_linear_operator_workspace,prepared_derivatives)...)
    n=length(rho0.data);current=T.(rho0.data)
    temporary=zeros(T,n);k1=similar(temporary);k2=similar(temporary)
    k3=similar(temporary);k4=similar(temporary);stage=similar(temporary)
    checkpoint_indices=unique(vcat(1,collect(1+checkpoint_stride:checkpoint_stride:intervals+1),
                                   intervals+1))
    checkpoints=Dict{Int,Vector{T}}(1=>copy(current))
    next_checkpoint=2
    for interval in 1:intervals
        h=ts[interval+1]-ts[interval]
        _control_rk4!(current,h,prepared_base,prepared_derivatives,
            view(control_values,:,interval),works,temporary,k1,k2,k3,k4,stage)
        if next_checkpoint<=length(checkpoint_indices)&&
           interval+1==checkpoint_indices[next_checkpoint]
            checkpoints[interval+1]=copy(current);next_checkpoint+=1
        end
    end
    final_state=PIState(rho0.basis,current)
    objective_value=R(real(dot(objective.data,current)))
    costate=T.(objective.data);gradient=zeros(R,m,intervals)
    segment_states=Vector{Vector{T}}();recomputed=0
    derivative_image=zeros(T,n)
    for segment in length(checkpoint_indices)-1:-1:1
        first_index=checkpoint_indices[segment]
        last_index=checkpoint_indices[segment+1]
        empty!(segment_states);push!(segment_states,copy(checkpoints[first_index]))
        segment_current=copy(checkpoints[first_index])
        for interval in first_index:last_index-1
            h=ts[interval+1]-ts[interval]
            _control_rk4!(segment_current,h,prepared_base,prepared_derivatives,
                view(control_values,:,interval),works,temporary,k1,k2,k3,k4,stage)
            push!(segment_states,copy(segment_current));recomputed+=1
        end
        for interval in last_index-1:-1:first_index
            local_index=interval-first_index+1
            rho_start=segment_states[local_index]
            rho_end=segment_states[local_index+1]
            lambda_end=copy(costate)
            h=ts[interval+1]-ts[interval]
            # Backward continuous adjoint: stepping from t+h to t turns
            # dλ/dt=-L'λ into a positive-h application of L'.
            _control_rk4!(costate,h,prepared_base,prepared_derivatives,
                view(control_values,:,interval),works,temporary,k1,k2,k3,k4,stage;
                adjoint_action=true)
            for parameter in 1:m
                work=works[parameter+1]
                _research_apply!(derivative_image,prepared_derivatives[parameter],
                                 rho_start,work)
                start_value=real(dot(costate,derivative_image))
                _research_apply!(derivative_image,prepared_derivatives[parameter],
                                 rho_end,work)
                end_value=real(dot(lambda_end,derivative_image))
                gradient[parameter,interval]=R(h*(start_value+end_value)/2)
            end
        end
    end
    metadata=(checkpoint_stride=Int(checkpoint_stride),
              retained_checkpoints=length(checkpoints),
              largest_recomputed_segment=min(Int(checkpoint_stride),intervals),
              recomputed_intervals=recomputed,intervals,
              storage_state_vectors=length(checkpoints)+
                  min(Int(checkpoint_stride),intervals)+1,
              quadrature=:endpoint_trapezoidal,
              propagation=:rk4)
    AdjointControlResult(objective_value,gradient,final_state,costate,metadata)
end
