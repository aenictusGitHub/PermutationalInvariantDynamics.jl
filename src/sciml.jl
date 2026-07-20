"""
    dynamics_problem(source, rho0, tspan; parameters=nothing)

Construct an in-place `SciMLBase.ODEProblem` for PI coefficient dynamics.
`source` may be a `PIModel`, compiled model, Liouvillian plan, or compatible
matrix/matrix-free operator. The initial coefficients are copied, and prepared
sources reuse one problem-owned Liouvillian workspace. Choose an adaptive or
stiff solver from the SciML ecosystem separately.
"""
function dynamics_problem(x,rho0::PIState,tspan;parameters=nothing)
    source_basis=_operator_basis(x)
    source_basis===nothing||source_basis===rho0.basis||throw(ArgumentError(
        "Liouvillian source and initial state use incompatible PI bases"))
    L=x isa PIModel ? compile(x) : x
    size(L)==(length(rho0.data),length(rho0.data)) ||
        throw(DimensionMismatch("Liouvillian and initial state dimensions differ"))
    work=_linear_operator_workspace(L)
    f! = work===nothing ? ((du,u,p,t)->_liouvillian_action!(du,L,u,t,p)) :
                         ((du,u,p,t)->apply!(du,L,u,t,p,work))
    SciMLBase.ODEProblem(f!,copy(rho0.data),tspan,parameters)
end
"""
    PISolution(raw, basis)

Attach a `PIBasis` to a SciML solution whose state vectors are PI
coefficients. Use `state` to reconstruct `PIState` objects at saved indices or
interpolated times.
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
