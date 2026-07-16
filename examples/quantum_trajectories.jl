using PermutationalInvariantDynamics
using LinearAlgebra

include(joinpath(@__DIR__, "utils", "makie_support.jl"))
using .ExampleMakie

# A directly checkable Monte Carlo wave-function benchmark in the setting of
# Dalibard--Castin--Mølmer (1992) and Mølmer--Castin--Dalibard (1993): N
# initially excited, independently decaying two-level atoms.
N = 6
gamma = 1.0
ntrajectories = 500
basis = PIBasis(N, 2)
sm = ComplexF64[0 1; 0 0]
model = PIModel(basis, (LocalJump(sm; rate=gamma),))
rho0 = iid_pure_state(basis, ComplexF64[0, 1])
times = collect(0.0:0.1:1.0)
prepared = compile(model; backend=:matrixfree)
trajectory_plan = TrajectoryPlan(prepared)
trajectory_batch = TrajectoryBatchWorkspace(trajectory_plan, rho0)
excited_population = CollectiveObservablePlan(basis, adjoint(sm) * sm)

# Continuous event times avoid a time-grid Bernoulli approximation. Geometry
# is compiled once and reused by all realizations. Start Julia with multiple
# threads to use dynamically scheduled task-owned workspaces. The same batch
# workspace can be reused sequentially for later ensembles.
trajectories = quantum_trajectories(
    trajectory_plan, rho0, times, ntrajectories;
    algorithm=:event, dt=0.1, dtmax=0.2,
    abstol=1e-10, reltol=1e-8, event_time_tolerance=1e-9,
    seed=2025, threaded=Threads.nthreads() > 1,
    workspace=trajectory_batch,
)
statistics = trajectory_statistics(
    trajectories;
    observables=(excitation=adjoint(sm) * sm,), nchannels=1,
)
average = statistics.average_states

# Independent emitters give a tensor-power density matrix with
# p_e(t)=exp(-gamma*t). This is an exact PI-space reference, not a full 2^N
# construction. The deterministic RK4 solution supplies a second route.
excited_probability = exp.(-gamma .* times)
exact_states = [iid_state(
    basis, ComplexF64[1-p 0; 0 p]) for p in excited_probability]
deterministic = solve_dynamics(
    prepared, rho0, (first(times), last(times));
    saveat=times, steps_per_interval=64,
)
deterministic_errors = [
    norm(deterministic[index].data - exact_states[index].data)
    for index in eachindex(times)
]
ensemble_errors = [
    norm(average[index].data - exact_states[index].data)
    for index in eachindex(times)
]

# At every time the exact excitation count is Binomial(N,p_e). Test the Monte
# Carlo means in units of their analytical standard error. At the final time,
# the number of emitted photons is Binomial(N,1-p_e), and the no-jump
# probability is exp(-N*gamma*T).
observable = statistics.observables.observables[:excitation]
exact_excitation_mean = N .* excited_probability
exact_excitation_sem = sqrt.(
    N .* excited_probability .* (1 .- excited_probability) ./ ntrajectories)
standardized_excitation_error = maximum(
    abs(observable.mean[index] - exact_excitation_mean[index]) /
    max(exact_excitation_sem[index], 100eps(Float64))
    for index in eachindex(times)
)

emission_probability = 1 - excited_probability[end]
exact_count_mean = N * emission_probability
exact_count_variance = N * emission_probability * (1-emission_probability)
exact_no_jump_probability = exp(-N * gamma * times[end])
count_mean_tolerance = 6sqrt(exact_count_variance / ntrajectories)
no_jump_tolerance = 6sqrt(
    exact_no_jump_probability * (1-exact_no_jump_probability) /
    ntrajectories) + inv(ntrajectories)

ensemble_excitation = real(collective_expectation(
    average[end], excited_population)) / N
exact_excitation = excited_probability[end]
report = diagnostics(average[end])

println("reference: Dalibard--Castin--Mølmer / Mølmer--Castin--Dalibard")
println("prepared backend: ", diagnostics(prepared).backend)
println("trajectories: ", length(trajectories))
println("maximum deterministic/exact state error: ", maximum(deterministic_errors))
println("maximum ensemble/exact state error: ", maximum(ensemble_errors))
println("maximum excitation-mean standardized error: ",
        standardized_excitation_error)
println("final excited fraction (ensemble, exact): ",
        (ensemble_excitation, exact_excitation))
println("final jump-count mean (sample, exact): ",
        (statistics.jumps.mean_count, exact_count_mean))
println("final jump-count variance (sample, exact): ",
        (statistics.jumps.count_variance, exact_count_variance))
println("no-jump probability (sample, exact): ",
        (statistics.jumps.no_jump_probability, exact_no_jump_probability))

@assert maximum(deterministic_errors) < 2e-10
@assert standardized_excitation_error < 6
@assert abs(statistics.jumps.mean_count-exact_count_mean) < count_mean_tolerance
@assert abs(statistics.jumps.no_jump_probability-exact_no_jump_probability) <
        no_jump_tolerance
@assert report.valid

if makie_available()
    M = makie_module()
    figure = M.Figure(size=(1100, 450), fontsize=17)
    excitation_axis = M.Axis(
        figure[1, 1]; xlabel="γt", ylabel="excited fraction",
        title="Independent-emitter trajectories")
    state_axis = M.Axis(
        figure[1, 2]; xlabel="γt", ylabel="PI-state 2-norm error",
        yscale=log10, title="Ensemble and integration errors")

    excitation_mean = observable.mean ./ N
    excitation_sem = observable.standard_error ./ N
    M.band!(excitation_axis, times,
            excitation_mean .- excitation_sem,
            excitation_mean .+ excitation_sem;
            color=(:dodgerblue, 0.25), label="trajectory ±1 SE")
    M.scatter!(excitation_axis, times, excitation_mean;
               color=:dodgerblue, markersize=9, label="trajectory mean")
    M.lines!(excitation_axis, times, excited_probability;
             color=:black, linewidth=2.5, label="exact exp(-γt)")
    M.axislegend(excitation_axis; position=:rt)

    floor_error = eps(Float64)
    M.lines!(state_axis, times, max.(ensemble_errors, floor_error);
             color=:darkorange, linewidth=2.5, label="trajectory average")
    M.lines!(state_axis, times, max.(deterministic_errors, floor_error);
             color=:seagreen, linewidth=2.5, linestyle=:dash,
             label="deterministic RK4")
    M.axislegend(state_axis; position=:rb)
    save_example_figure(figure, "quantum_trajectories_molmer")
end
