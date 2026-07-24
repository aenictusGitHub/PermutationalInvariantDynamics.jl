using PermutationalInvariantDynamics
include(joinpath(@__DIR__, "utils", "makie_support.jl"))
using .ExampleMakie

# Independent spontaneous emission has an exact excitation-count law and is a
# compact benchmark for memory-light output. Nothing below constructs a full
# 2^N Hilbert-space state.
N = 4
gamma = 0.8
ntrajectories = 128
basis = PIBasis(N, 2)
sm = ComplexF64[0 1; 0 0]
number = adjoint(sm) * sm
rho0 = iid_pure_state(basis, ComplexF64[0, 1])
model = PIModel(basis, (LocalJump(sm; rate=gamma),))
prepared = compile(model; backend=:matrixfree)
times = collect(range(0.0, 0.6; length=7))

# Deterministic observable-only propagation retains one complex number per
# output time, rather than one PI coordinate vector per time.
deterministic = solve_dynamics(
    prepared, rho0, (first(times), last(times));
    saveat=times, steps_per_interval=32,
    observables=(excitations=number,), save_states=false,
)
exact_excitations = N .* exp.(-gamma .* times)
deterministic_excitations = real.(
    deterministic.observables[:excitations])

@assert deterministic.states === nothing
@assert maximum(abs.(deterministic_excitations .- exact_excitations)) < 2e-10

# The ensemble owns one observable buffer and one online Welford accumulator
# per active worker. It returns observable and jump statistics, but no sampled
# PIState histories. Index-derived seeds keep the statistical experiment
# independent of task scheduling.
trajectory_plan = TrajectoryPlan(prepared)
batch = TrajectoryBatchWorkspace(trajectory_plan, rho0)
ensemble = quantum_trajectories(
    trajectory_plan, rho0, times, ntrajectories;
    seed=2026, dt=0.005,
    threaded=Threads.nthreads() > 1, workspace=batch,
    observables=(excitations=number,), save_states=false,
)
sample = ensemble.observables.observables[:excitations]

@assert ensemble.trajectories === nothing
@assert ensemble.observables.trajectories == ntrajectories
@assert ensemble.jumps.trajectories == ntrajectories
for index in eachindex(times)
    tolerance = 6sample.standard_error[index] + 2e-12
    @assert abs(sample.mean[index] - exact_excitations[index]) < tolerance
end

final_emission_probability = 1 - exp(-gamma * times[end])
exact_jump_mean = N * final_emission_probability
exact_jump_variance = N * final_emission_probability *
                      (1-final_emission_probability)
count_tolerance = 6sqrt(exact_jump_variance / ntrajectories)
@assert abs(ensemble.jumps.mean_count - exact_jump_mean) < count_tolerance

history_bytes = ntrajectories * length(times) * estimate_state_bytes(basis)
println("deterministic excitation error: ",
        maximum(abs.(deterministic_excitations .- exact_excitations)))
println("trajectory final excitation (sample, exact): ",
        (sample.mean[end], exact_excitations[end]))
println("jump-count mean (sample, exact): ",
        (ensemble.jumps.mean_count, exact_jump_mean))
println("state histories retained: ", ensemble.trajectories !== nothing)
println("PI-coordinate bytes avoided for sampled trajectory states: ",
        history_bytes)

if makie_available()
    M = makie_module()
    figure = M.Figure(size=(880, 460), fontsize=17)
    axis = M.Axis(
        figure[1, 1];
        xlabel="time",
        ylabel="mean excitation count",
        title="Memory-light deterministic and trajectory output",
    )
    M.band!(
        axis, times,
        sample.mean .- sample.standard_error,
        sample.mean .+ sample.standard_error;
        color=(:darkorange, 0.22))
    M.lines!(
        axis, times, exact_excitations;
        color=:black, linewidth=2.7, label="analytic")
    M.scatter!(
        axis, times, deterministic_excitations;
        color=:royalblue, markersize=9,
        label="deterministic stream")
    M.scatter!(
        axis, times, sample.mean;
        color=:darkorange, marker=:diamond, markersize=9,
        label="trajectory stream ±1 SE")
    M.axislegend(axis; position=:rt, labelsize=12)
    save_example_figure(figure, "streaming_output")
end
