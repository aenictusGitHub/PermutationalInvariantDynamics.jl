using PermutationalInvariantDynamics
using LinearAlgebra

include(joinpath(@__DIR__, "utils", "makie_support.jl"))
using .ExampleMakie

# A directly checkable Monte Carlo wave-function benchmark in the setting of
# Dalibard--Castin--Mølmer (1992) and Mølmer--Castin--Dalibard (1993): N
# initially excited, independently decaying two-level atoms.
quick_example = get(ENV, "PID_EXAMPLE_QUICK", "0") == "1"
N = quick_example ? 6 : 10
gamma = 1.0
ntrajectories = 500
basis = PIBasis(N, 2)
sm = ComplexF64[0 1; 0 0]
model = PIModel(basis, (LocalJump(sm; rate=gamma),))
rho0 = iid_pure_state(basis, ComplexF64[0, 1])
times = collect(range(0.0, 1.0; length=quick_example ? 11 : 31))
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
    saveat=times, steps_per_interval=quick_example ? 64 : 24,
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

# Show the full discrete photon-count law, including rarely occupied tails.
photon_counts = [length(path.jump_times) for path in trajectories]
count_grid = collect(0:N)
sample_count_probability = [count(==(k), photon_counts) / ntrajectories for k in count_grid]
exact_count_probability = [Float64(exact_binomial(N, k)) *
    emission_probability^k * (1-emission_probability)^(N-k) for k in count_grid]
@assert all(k -> 0 <= k <= N, photon_counts)
@assert sum(sample_count_probability) ≈ 1
@assert sum(exact_count_probability) ≈ 1

if makie_available()
    M = makie_module()
    figure = example_figure(size=(1150, 830))
    M.Label(figure[0, 1:2], ExampleMakie.latex("Independent-emitter trajectories  •  \$N=$N\$, $ntrajectories paths, seed=2025");
            fontsize=22, font=:bold, halign=:left)
    excitation_axis = M.Axis(
        figure[1, 1]; xlabel=ExampleMakie.latex(raw"$\gamma t$"), ylabel=ExampleMakie.latex("excited fraction"),
        title=ExampleMakie.latex("(a) Excitation and sampling uncertainty"))
    state_axis = M.Axis(
        figure[1, 2]; xlabel=ExampleMakie.latex(raw"$\gamma t$"), ylabel=ExampleMakie.latex("PI-state 2-norm error"),
        yscale=log10, yticks=(10.0 .^ [-12, -9, -6, -3],
                             ExampleMakie.latex.([raw"$10^{-12}$", raw"$10^{-9}$", raw"$10^{-6}$", raw"$10^{-3}$"])),
        title=ExampleMakie.latex("(b) Ensemble and integration errors"))
    difference_axis = M.Axis(figure[2, 1]; xlabel=ExampleMakie.latex(raw"$\gamma t$"),
        ylabel=ExampleMakie.latex("excitation fraction − exact"), title=ExampleMakie.latex("(c) Difference from the exact mean"))
    count_axis = M.Axis(figure[2, 2]; xlabel=ExampleMakie.latex("emitted photons at \$\\gamma T=$(gamma*last(times))\$"),
        ylabel=ExampleMakie.latex("probability"), xticks=0:2:N, title=ExampleMakie.latex("(d) Photon-count distribution"))

    excitation_mean = observable.mean ./ N
    excitation_sem = observable.standard_error ./ N
    scaled_times = gamma .* times
    M.band!(excitation_axis, scaled_times,
            excitation_mean .- excitation_sem,
            excitation_mean .+ excitation_sem;
            color=(example_colors.blue, 0.22), label=ExampleMakie.latex("trajectory ±1 SE"))
    M.scatter!(excitation_axis, scaled_times, excitation_mean;
               color=example_colors.blue, markersize=5, label=ExampleMakie.latex("trajectory mean"))
    M.lines!(excitation_axis, scaled_times, excited_probability;
             color=:black, linewidth=2.5, label=ExampleMakie.latex(raw"exact $\exp(-\gamma t)$"))
    M.axislegend(excitation_axis; position=:rt)

    M.lines!(state_axis, scaled_times, [iszero(e) ? NaN : e for e in ensemble_errors];
             color=example_colors.orange, linewidth=2.5, label=ExampleMakie.latex("trajectory average"))
    M.lines!(state_axis, scaled_times, [iszero(e) ? NaN : e for e in deterministic_errors];
             color=example_colors.green, linewidth=2.5, linestyle=:dash,
             label=ExampleMakie.latex("deterministic RK4"))
    M.axislegend(state_axis; position=:rc)
    difference = excitation_mean .- excited_probability
    M.band!(difference_axis, scaled_times, difference .- excitation_sem,
            difference .+ excitation_sem; color=(example_colors.blue, 0.22))
    M.lines!(difference_axis, scaled_times, difference; color=example_colors.blue)
    M.hlines!(difference_axis, [0]; color=:black, linestyle=:dash)
    M.barplot!(count_axis, count_grid, sample_count_probability;
               color=(example_colors.blue, 0.6), width=0.75, label=ExampleMakie.latex("sample frequency"))
    M.scatter!(count_axis, count_grid, exact_count_probability;
               color=:black, marker=:diamond, markersize=8, label=ExampleMakie.latex("exact binomial law"))
    M.axislegend(count_axis; position=:lt, labelsize=12)
    M.Label(figure[3, 1:2], ExampleMakie.latex("Bands: pointwise ±1 standard error, not simultaneous confidence bounds.  •  Exact zeros omitted only from the log-error panel.");
            fontsize=13, color=example_colors.gray)
    save_example_figure(figure, "independent_emitter_quantum_trajectories")
    save_example_data("independent_emitter_quantum_trajectories", (;
        time=times, gamma_t=scaled_times, excited_fraction=excitation_mean,
        sample_standard_error=excitation_sem, exact_fraction=excited_probability,
        exact_standard_error=exact_excitation_sem ./ N,
        deterministic_state_error=deterministic_errors, ensemble_state_error=ensemble_errors);
        metadata=(; N,gamma,ntrajectories,seed=2025,algorithm="event",
                  abstol=1e-10,reltol=1e-8,event_time_tolerance=1e-9,
                  dt=0.1,dtmax=0.2,rk4_steps_per_interval=quick_example ? 64 : 24))
    save_example_data("independent_emitter_photon_counts", (;
        photon_count=count_grid, sample_probability=sample_count_probability,
        exact_probability=exact_count_probability);
        metadata=(; N,gamma,time=last(times),ntrajectories,seed=2025))
end
