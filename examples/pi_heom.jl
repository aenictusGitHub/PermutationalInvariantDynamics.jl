using PermutationalInvariantDynamics

include(joinpath(@__DIR__, "utils", "makie_support.jl"))
using .ExampleMakie

# A collective exponentially correlated dephasing bath. The physical model is
# H_int = Jz tensor B and C(t) = coefficient * exp(-frequency*t). Every ADO is
# represented directly in the PI operator space; no 2^N Hilbert-space matrix
# is constructed.
N = 8
coefficient = 0.30
frequency = 1.20
final_time = 1.50
depths = (2, 4, 6)
times = range(0.0, final_time; length=61)

basis = PIBasis(N, 2)
spin = spin_matrices()
Jz = collective_operator(basis, spin.jz)
Jx = collective_operator(basis, spin.jx)
system = PIModel(basis, ())
bath = HEOMBath(Jz, coefficient, frequency)
rho0 = iid_pure_state(basis, ComplexF64[1, 1] / sqrt(2))

# For this commuting, real-correlation benchmark, the exact normalized
# transverse coherence is exp[-g(t)] with the Gaussian dephasing line shape
# below. It is independent of N after division by N/2.
line_shape(t) = coefficient / frequency^2 *
                (frequency*t - 1 + exp(-frequency*t))
exact = exp.(-line_shape.(times))

curves = NamedTuple[]
for depth in depths
    # Store an exact similarity-scaled hierarchy. This balances auxiliary
    # tiers without changing the physical root density operator.
    plan = HEOMPlan(system, bath; max_depth=depth, scaling=:scaled)
    hierarchy = heom_time_evolution(
        plan, rho0, times; steps_per_interval=4)
    normalized_Jx = [
        2real(expectation(heom_reduced_state(state), Jx)) / N
        for state in hierarchy]
    error = maximum(abs.(normalized_Jx .- exact))
    trace_error = maximum(
        abs(trace(heom_reduced_state(state)) - 1) for state in hierarchy)
    push!(curves, (; depth, plan, normalized_Jx, error, trace_error))
    println("depth=$depth: ADOs=$(heom_number_ados(plan)), " *
            "PI coordinates=$(length(basis)), HEOM coordinates=$(size(plan, 1)), " *
            "max signal error=$error, max trace error=$trace_error")
end

@assert all(curves[index+1].error < curves[index].error
            for index in 1:length(curves)-1)
@assert last(curves).error < 5e-10
@assert last(curves).trace_error < 5e-12

# Produce a compact automated truncation report at the final time. The helper
# compiles the system once, shares the prepared coupling blocks across depth
# prefixes, and retains only the reduced state at intermediate depths.
depth_report = heom_depth_convergence(
    system, bath, rho0, (first(times), last(times));
    depths, steps=4(length(times)-1), scaling=:scaled,
    atol=1e-7, rtol=0)
# The displayed Jx signal has converged at depth 6, but the stronger full-root
# Hilbert--Schmidt test has not. This distinction is intentional and prevents
# an observable-specific agreement from being presented as state convergence.
@assert isequal(depth_report.pairwise_converged,
                Union{Missing,Bool}[missing, false, false])
@assert !depth_report.converged
println("successive-depth Hilbert--Schmidt differences = ",
        collect(skipmissing(depth_report.pairwise_errors)))

# The low-level matrix-free adapter can be passed to existing Krylov spectral
# tools. This example only inspects its size; it never materializes the map.
Lheom = heom_liouvillian(last(curves).plan)
@assert size(Lheom) == (size(last(curves).plan, 1),
                       size(last(curves).plan, 1))

# The same matrix-free RHS is available to a user-selected adaptive or stiff
# SciML integrator. Constructing the problem adds no solver dependency.
problem = heom_problem(
    last(curves).plan, rho0, (first(times), last(times)))
@assert length(problem.u0) == size(last(curves).plan, 1)

# Physical spectral-density helpers keep the pole approximation and its
# white-noise residue explicit. This common-bath plan is only a setup smoke
# test; the analytic benchmark above intentionally uses one exact exponential.
physical_bath = drude_lorentz_bath(
    Jz, 0.20, 0.90, 2.0;
    matsubara_terms=2, decomposition=:pade)
physical_plan = HEOMPlan(
    system, physical_bath; max_depth=2, terminator=:residue,
    importance_cutoff=1e-3)
physical_metadata = heom_hierarchy_metadata(physical_plan)
@assert physical_metadata.retained_ados <= physical_metadata.full_ados
println("Drude Padé residue = ", heom_bath_residue(physical_bath),
        "; retained/full ADOs = ", physical_metadata.retained_ados,
        "/", physical_metadata.full_ados)

# Independent identical local baths are different from the collective Jz
# bath. A positive damped pole can instead be represented by one finite
# pseudomode in every PI supersite. The mode cutoff must be converged.
local_embedding = independent_local_pseudomode_model(
    2, zeros(2, 2), ComplexF64[0 1; 0 0];
    nmax=1, frequency=1.0, coupling_strength=0.2, damping=0.4)
@assert local_embedding.basis.d == 4
@assert local_embedding.metadata.cutoff_approximation

# A tiny uniquely damped hierarchy demonstrates the reusable ADO-diagonal
# preconditioner. The collective-dephasing benchmark above has a nonunique
# stationary manifold and is deliberately not sent to a unique-state solve.
small_basis = PIBasis(1, 2)
small_spin = spin_matrices()
small_Q = collective_operator(small_basis, small_spin.jz)
small_system = PIModel(
    small_basis, (LocalJump(ComplexF64[0 1; 0 0]),))
small_bath = HEOMBath(small_Q, 0.08, 1.2)
small_plan = HEOMPlan(
    small_system, small_bath; max_depth=2, scaling=:scaled)
small_preconditioner = heom_block_preconditioner(
    small_plan; expected_reuses=2, warn_unamortized=false)
small_stationary = heom_steady_state(
    small_plan; preconditioner=small_preconditioner,
    return_info=true, krylovdim=12, maxiter=200,
    atol=1e-11, rtol=1e-9)
small_ground = iid_pure_state(small_basis, ComplexF64[1, 0])
@assert heom_reduced_state(small_stationary.state).data ≈
        small_ground.data atol=2e-9
@assert small_stationary.residual < 2e-9
println("small stationary HEOM residual = ", small_stationary.residual,
        "; block-preconditioner setup applications = ",
        preconditioner_cost(small_preconditioner).setup_liouvillian_applications)

if makie_available()
    M = makie_module()
    figure = M.Figure(size=(1120, 440), fontsize=17)
    signal_axis = M.Axis(
        figure[1, 1]; xlabel="νt", ylabel="2⟨Jx⟩ / N",
        title="PI–HEOM collective dephasing (N=$N)")
    error_axis = M.Axis(
        figure[1, 2]; xlabel="νt", ylabel="absolute error",
        yscale=log10, title="Hierarchy-depth convergence")

    scaled_times = frequency .* collect(times)
    M.lines!(signal_axis, scaled_times, exact;
             color=:black, linewidth=3, label="analytic")
    colors = (:firebrick, :royalblue, :seagreen)
    for (curve, color) in zip(curves, colors)
        label = "depth $(curve.depth)"
        M.lines!(signal_axis, scaled_times, curve.normalized_Jx;
                 color, linewidth=2, linestyle=:dash, label)
        pointwise_error = abs.(curve.normalized_Jx .- exact)
        # A zero at t=0 cannot be displayed on a logarithmic axis.
        shown_error = max.(pointwise_error, eps(Float64))
        M.lines!(error_axis, scaled_times, shown_error;
                 color, linewidth=2.3, label)
    end
    M.axislegend(signal_axis; position=:lb, labelsize=12)
    M.axislegend(error_axis; position=:rb, labelsize=12)
    save_example_figure(figure, "pi_heom")
end
