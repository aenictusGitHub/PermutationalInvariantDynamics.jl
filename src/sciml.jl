"""
    dynamics_problem(source, rho0, tspan; parameters=nothing)

Construct an in-place `SciMLBase.ODEProblem` for PI coefficient dynamics.
`source` may be a `PIModel`, compiled model, Liouvillian plan, or compatible
matrix/matrix-free operator. The initial coefficients are copied, and prepared
sources reuse one problem-owned Liouvillian workspace. Coefficients are
promoted to the generator's precision when supported. Endpoints must be finite
real numbers; backward integration is allowed. Choose an adaptive or
stiff solver from the SciML ecosystem separately.
"""
function dynamics_problem(x,rho0::PIState,tspan;parameters=nothing)
    _checked_evolution_tspan(tspan)
    _check_evolution_basis(x,rho0)
    L=x isa PIModel ? compile(x) : x
    current=_prepare_evolution_state(L,rho0)
    work=_linear_operator_workspace(L)
    f! = work===nothing ? ((du,u,p,t)->_liouvillian_action!(du,L,u,t,p)) :
                         ((du,u,p,t)->apply!(du,L,u,t,p,work))
    SciMLBase.ODEProblem(f!,current.data,tspan,parameters)
end
"""
    PISolution(raw, basis)

Attach a `PIBasis` to a SciML solution whose state vectors are PI
coefficients. Use `state` to reconstruct `PIState` objects at saved indices or
interpolated times.
`result_times` borrows `raw.t`; `result_states` returns this solution as a
lazy saved-state history. `result_final_state` wraps the last saved vector,
or returns `nothing` for an empty history. These accessors never interpolate.
"""
struct PISolution{S,B<:PIBasis};raw::S;basis::B;end

"""
    state(solution::PISolution, index)
    state(solution::PISolution, time)

Return a `PIState` from a saved solution index. Use [`state_at`](@ref) for the
raw solution's continuous interpolation at a physical time, including an
integer-valued time; `state(solution, integer)` always selects a saved index.
"""
state(sol::PISolution,i::Integer)=PIState(sol.basis,sol.raw.u[i])
state_at(sol::PISolution,t::Real)=PIState(sol.basis,sol.raw(t))
state(sol::PISolution,t::Real)=state_at(sol,t)
coefficient_block(sol::PISolution,p::Partition,i::Integer)=coefficient_block(state(sol,i),p)
physical_block(sol::PISolution,p::Partition,i::Integer)=physical_block(state(sol,i),p)
expectation(sol::PISolution,A::PIOperator)=[expectation(state(sol,i),A) for i in eachindex(sol.raw.u)]
sector_populations(sol::PISolution)=[sector_populations(state(sol,i)) for i in eachindex(sol.raw.u)]
collective_expectation(sol::PISolution,X::AbstractMatrix)=[collective_expectation(state(sol,i),X) for i in eachindex(sol.raw.u)]
collective_variance(sol::PISolution,X::AbstractMatrix;kwargs...)=[collective_variance(state(sol,i),X;kwargs...) for i in eachindex(sol.raw.u)]
quantum_fisher_information_matrix(sol::PISolution,generators;kwargs...)=[qfim(state(sol,i),generators;kwargs...) for i in eachindex(sol.raw.u)]
qfim(sol::PISolution,generators;kwargs...)=quantum_fisher_information_matrix(sol,generators;kwargs...)
