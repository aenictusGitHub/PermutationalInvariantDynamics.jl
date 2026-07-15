using PermutationalInvariantDynamics
using LinearAlgebra

N = 6
basis = PIBasis(N, 2)
sm = ComplexF64[0 1; 0 0]
model = PIModel(basis, [LocalJump(sm; rate=1.0)])
rho0 = iid_pure_state(basis, ComplexF64[0, 1])
times = collect(0.0:0.1:1.0)
prepared = compile(model; backend=:matrixfree)
excited_population = CollectiveObservablePlan(basis, ComplexF64[0 0; 0 1])

# Geometry is compiled once and reused by every realization. Start Julia with
# multiple threads and pass threaded=true for thread-local parallel workspaces.
trajectories = quantum_trajectories(model, rho0, times, 500;
                                    dt=0.005, seed=2025)
average = trajectory_average(trajectories)
statistics = trajectory_statistics(trajectories;
    observables=(excited_population=ComplexF64[0 0; 0 1],), nchannels=1)

# The deterministic reference uses the same compiled model through the typed
# high-level dynamics result rather than rebuilding a Liouvillian.
deterministic = solve_dynamics(prepared, rho0, (first(times), last(times));
                               saveat=times, steps_per_interval=20)
errors = [norm(average[i].data - deterministic[i].data) for i in eachindex(times)]
ensemble_excitation = real(collective_expectation(average[end], excited_population)) / N
deterministic_excitation = real(collective_expectation(deterministic[end], excited_population)) / N
report = diagnostics(average[end])

println("prepared backend: ", diagnostics(prepared).backend)
println("trajectories: ", length(trajectories))
println("recorded jumps: ", sum(length(q.jump_times) for q in trajectories))
println("maximum ensemble error: ", maximum(errors))
println("final trace: ", trace(average[end]))
println("final excited fraction (ensemble, deterministic): ",
        (ensemble_excitation, deterministic_excitation))
println("mean jumps per trajectory: ", statistics.jumps.mean_count)
println("jump-count Fano factor: ", statistics.jumps.fano)
println("final excited-population standard error: ",
        statistics.observables.observables[:excited_population].standard_error[end])

@assert report.valid
