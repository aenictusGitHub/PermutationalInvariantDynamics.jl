using PermutationalInvariantDynamics
using LinearAlgebra

render_plots = get(ENV, "PI_EXAMPLE_PLOTS", "1") != "0"
render_plots && include(joinpath(@__DIR__, "utils", "makie_support.jl"))

# Decay-only specialization of Zhang, Zhang, and Mølmer,
# New J. Phys. 20, 112001 (2018). Their efficient Dicke pseudo-state method
# motivates resolving an unresolved local event into total-spin-sector
# branches. Here those branches come from the exact PI gain-map Kraus
# factorization and work for every retained Schur irrep.
#
# Related weak-symmetry trajectory work:
# E. W. Lloyd, A. A. Ziolkowska, and J. Keeling,
# "Permutation-symmetric quantum trajectories," arXiv:2605.11103 (2026).
# Their construction treats emitters coupled to a common system, including a
# shared cavity. This executable remains the emitter-only Zhang--Mølmer decay
# benchmark and does not claim to reproduce the shared-cavity calculations.
N = 6
GammaC = 1.0
gammaL = 1.0
nweak = 400
ndensity = 400

basis = PIBasis(N, 2)
sm = ComplexF64[0 1; 0 0]
number = adjoint(sm) * sm
model = PIModel(basis, (
    CollectiveJump(sm; rate=GammaC),
    LocalJump(sm; rate=gammaL),
))
rho0 = iid_pure_state(basis, ComplexF64[0, 1])
psi0 = weak_pi_pseudoket(rho0)
times = collect(range(0.0, 1.0; length=11))

prepared = compile(model; backend=:matrixfree)
weak_plan = WeakPITrajectoryPlan(prepared)
density_plan = TrajectoryPlan(prepared)
weak_batch = WeakPITrajectoryBatchWorkspace(weak_plan, psi0)
density_batch = TrajectoryBatchWorkspace(density_plan, rho0)
use_threads = Threads.nthreads() > 1

Jminus = collective_operator(basis, sm)
cavity_flux = GammaC * (adjoint(Jminus) * Jminus)
free_flux = gammaL * collective_operator(basis, number)
observables = (cavity=cavity_flux, free_space=free_flux)

# Compile both prepared ensemble paths once.  The measured batches below use
# the same fixed-step controls and include state-history allocation, but not
# plan construction, JIT compilation, statistics, or plotting.
weak_pi_quantum_trajectories(
    weak_plan, psi0, times, 1;
    dt=0.005, max_jump_probability=0.025, seed=12026,
    threaded=false, workspace=weak_batch,
)
quantum_trajectories(
    density_plan, rho0, times, 1;
    algorithm=:fixed, dt=0.005, max_jump_probability=0.025,
    seed=12027, threaded=false, workspace=density_batch,
)

GC.gc()
weak_elapsed = @elapsed weak_paths = weak_pi_quantum_trajectories(
    weak_plan, psi0, times, nweak;
    dt=0.005, max_jump_probability=0.025, seed=2026,
    threaded=use_threads, workspace=weak_batch,
)
weak_statistics = weak_pi_trajectory_statistics(
    weak_paths; observables, nchannels=2)
weak_average = weak_statistics.average_states

# The established density-valued PI unraveling combines every local Kraus
# outcome into one mixed conditional state. It is a different trajectory
# record but has the same ensemble master equation.  Fixed-step controls are
# matched to the weak-PI run so the descriptive per-path timing is meaningful.
GC.gc()
density_elapsed = @elapsed density_paths = quantum_trajectories(
    density_plan, rho0, times, ndensity;
    algorithm=:fixed, dt=0.005, max_jump_probability=0.025,
    seed=2027, threaded=use_threads, workspace=density_batch,
)
density_statistics = trajectory_statistics(
    density_paths; observables, nchannels=2)
density_average = density_statistics.average_states

# The same Schur-Kraus plan supports continuous-hazard event times. This
# small state-free batch demonstrates confidence-controlled stopping; its
# deliberately loose cap is not used for the quantitative curves below.
adaptive_weak = adaptive_weak_pi_quantum_trajectories(
    weak_plan, psi0, times;
    observables, algorithm=:event, dt=0.02, dtmax=0.05,
    abstol=1e-9, reltol=1e-7,
    min_trajectories=16, max_trajectories=32, batch_size=8,
    atol=0.75, rtol=0, seed=3026, threaded=use_threads,
)

# A history-free stationary estimate averages density reconstructions inside
# each independent path. Complete time batches add an autocorrelation-aware
# diagnostic without replacing the primary across-path standard error.
weak_stationary = weak_pi_trajectory_steady_state(
    weak_plan, psi0;
    trajectories=8, settling_time=5.0,
    samples_per_trajectory=8, sampling_interval=0.1, batch_size=4,
    algorithm=:event, dt=0.02, dtmax=0.05,
    abstol=1e-9, reltol=1e-7,
    seed=4026, threaded=use_threads, return_info=true,
)

# Decay from the excited state remains diagonal in the Schur/GT basis.  The
# certified population backend is therefore an independent reduced reference.
population_plan = PopulationPlan(model)
@assert population_plan.invariance.invariant === true
population_solution = solve_populations(
    population_plan, rho0, (first(times), last(times));
    saveat=times, steps_per_interval=100,
)
population_states = [state(population_solution, index)
                     for index in eachindex(times)]

# General matrix-free PI master evolution is checked separately.  The black
# reference curves below use the reduced population solution, while all state
# errors use this full PI solution as their common target.
deterministic = solve_dynamics(
    prepared, rho0, (first(times), last(times));
    saveat=times, steps_per_interval=100,
)
exact_cavity = real.([expectation(rho, cavity_flux)
                      for rho in population_states])
exact_free = real.([expectation(rho, free_flux)
                    for rho in population_states])

weak_cavity = weak_statistics.observables.observables[:cavity]
weak_free = weak_statistics.observables.observables[:free_space]
density_cavity = density_statistics.observables.observables[:cavity]
density_free = density_statistics.observables.observables[:free_space]

function standardized_error(sample, reference; numerical_floor=0.025)
    maximum(abs(sample.mean[index] - reference[index]) /
            max(6sample.standard_error[index] + numerical_floor, eps(Float64))
            for index in eachindex(reference))
end

weak_cavity_error = standardized_error(weak_cavity, exact_cavity)
weak_free_error = standardized_error(weak_free, exact_free)
density_cavity_error = standardized_error(density_cavity, exact_cavity)
density_free_error = standardized_error(density_free, exact_free)
weak_state_errors = [norm(weak_average[index].data - deterministic[index].data)
                     for index in eachindex(times)]
density_state_errors = [
    norm(density_average[index].data - deterministic[index].data)
    for index in eachindex(times)
]
population_state_errors = [
    norm(population_states[index].data - deterministic[index].data)
    for index in eachindex(times)
]
weak_state_error = maximum(weak_state_errors)
density_state_error = maximum(density_state_errors)
population_state_error = maximum(population_state_errors)

sector_changes = sum(count for ((source, target), count) in
    weak_statistics.jumps.sector_transitions if source != target)
sector_spins = sort(Float64[
    (partition.parts[1] - partition.parts[2]) / 2
    for partition in basis.sectors
])
spin_index = Dict(spin => index for (index, spin) in pairs(sector_spins))
transition_counts = zeros(Int, length(sector_spins), length(sector_spins))
for ((source, target), count) in weak_statistics.jumps.sector_transitions
    source == target && continue
    source_spin = (source.parts[1] - source.parts[2]) / 2
    target_spin = (target.parts[1] - target.parts[2]) / 2
    transition_counts[spin_index[source_spin], spin_index[target_spin]] += count
end
transition_rates = transition_counts ./ nweak

# Exact representation sizes for a complete qubit basis.  These arrays are
# theoretical snapshot-coordinate counts; no full-Hilbert object is built.
scaling_sizes = collect(1:50)
weak_dimensions = Float64[
    (fld(n, 2) + 1) * (cld(n, 2) + 1) for n in scaling_sizes
]
pi_density_dimensions = Float64[binomial(n + 3, 3) for n in scaling_sizes]
full_ket_dimensions = exp2.(Float64.(scaling_sizes))
full_density_dimensions = exp2.(2 .* Float64.(scaling_sizes))
weak_time_per_path = weak_elapsed / nweak
density_time_per_path = density_elapsed / ndensity
weak_history_bytes = Base.summarysize(weak_paths)
density_history_bytes = Base.summarysize(density_paths)

println("reference: Zhang--Zhang--Mølmer, NJP 20, 112001 (2018)")
println("related reference: Lloyd--Ziolkowska--Keeling, ",
        "arXiv:2605.11103 (2026)")
println("full Hilbert dimension: ", 2^N)
println("PI density dimension: ",length(basis))
println("weak-PI pseudo-ket dimension: ",weak_pi_dimension(basis))
println("certified population dimension: ",population_dimension(basis))
println("weak/density trajectory counts: ",(nweak,ndensity))
println("sampled sector-changing local events: ",sector_changes)
println("six-SE normalized intensity errors (weak cavity/free): ",
        (weak_cavity_error,weak_free_error))
println("six-SE normalized intensity errors (density cavity/free): ",
        (density_cavity_error,density_free_error))
println("maximum PI-state errors (weak,density): ",
        (weak_state_error,density_state_error))
println("population/full-PI state agreement: ", population_state_error)
println("prepared batch time per path (weak,density) [s]: ",
        (weak_time_per_path, density_time_per_path))
println("timing ratio density/weak: ",
        density_time_per_path / weak_time_per_path)
println("retained trajectory-history bytes (weak,density): ",
        (weak_history_bytes, density_history_bytes))
println("history-size ratio density/weak: ",
        density_history_bytes / weak_history_bytes)
println("N=50 coordinate counts (weak PI, PI density, full ket): ",
        (Int(weak_dimensions[50]), Int(pi_density_dimensions[50]), 2^BigInt(50)))
println("adaptive event-driven weak-PI paths: ", adaptive_weak.trajectory_count,
        "; converged sampling target: ", adaptive_weak.converged)
println("weak-PI stationary batch means: ",
        weak_stationary.metadata.batch_means)

@assert weak_pi_dimension(basis)<length(basis)
@assert weak_pi_dimension(basis)<2^N
@assert population_dimension(basis)==weak_pi_dimension(basis)
@assert weak_dimensions[N]==weak_pi_dimension(basis)
@assert pi_density_dimensions[N]==length(basis)
@assert sector_changes>0
@assert sum(transition_counts)==sector_changes
@assert all(record.channel != 1 ||
            record.source_partition == record.target_partition
            for path in weak_paths for record in path.jump_records)
@assert all(record.source_partition == record.target_partition ||
            record.channel == 2
            for path in weak_paths for record in path.jump_records)
@assert abs(exact_cavity[1] - N * GammaC)<2e-11
@assert abs(exact_free[1] - N * gammaL)<2e-11
@assert maximum(abs(sum(populations) - 1)
                for populations in population_solution)<2e-11
@assert weak_cavity_error<1
@assert weak_free_error<1
@assert density_cavity_error<1
@assert density_free_error<1
@assert weak_state_error<0.12
@assert density_state_error<0.12
@assert population_state_error<2e-11
@assert all(state->abs(trace(state)-1)<5e-12,weak_average)
@assert adaptive_weak.backend == :weak_pi
@assert adaptive_weak.metadata.algorithm == :event
@assert weak_stationary.metadata.algorithm == :event
@assert weak_stationary.metadata.batch_means.batch_count == 16
@assert abs(trace(weak_stationary.state)-1) < 5e-11

if render_plots && ExampleMakie.makie_available()
    M = ExampleMakie.makie_module()
    figure = M.Figure(size=(1250, 850), fontsize=17)
    cavity_axis = M.Axis(
        figure[1, 1]; xlabel="time", ylabel="Γc ⟨J₊J₋⟩",
        title="(a) Collective cavity channel")
    free_axis = M.Axis(
        figure[1, 2]; xlabel="time", ylabel="γl ⟨Ne⟩",
        title="(b) Sector-changing local channel")
    error_axis = M.Axis(
        figure[2, 1]; xlabel="time", ylabel="PI-state 2-norm error",
        yscale=log10, title="(c) Ensemble convergence")
    transition_axis = M.Axis(
        figure[2, 2]; xlabel="source total spin J",
        ylabel="target total spin J′",
        title="(d) Sampled sector changes")

    for (axis, reference, weak, density) in
        ((cavity_axis,exact_cavity,weak_cavity,density_cavity),
         (free_axis,exact_free,weak_free,density_free))
        M.band!(axis, times, weak.lower, weak.upper;
                color=(:dodgerblue, 0.16))
        M.lines!(axis, times, weak.mean; color=:dodgerblue, linewidth=1.4)
        M.scatter!(axis, times, weak.mean; color=:dodgerblue, markersize=8,
                   label="weak-PI pseudo-kets (95% CI)")
        M.band!(axis, times, density.lower, density.upper;
                color=(:darkorange, 0.14))
        M.lines!(axis, times, density.mean; color=:darkorange, linewidth=1.4)
        M.scatter!(axis, times, density.mean; color=:darkorange,
                   marker=:diamond, markersize=8,
                   label="density-valued PI paths (95% CI)")
        M.lines!(axis, times, reference; color=:black, linewidth=2.7,
                 label="population master equation")
    end
    M.axislegend(cavity_axis; position=:rt, labelsize=11)

    error_floor = eps(Float64)
    M.lines!(error_axis, times, max.(weak_state_errors, error_floor);
             color=:dodgerblue, linewidth=2.4, label="weak-PI average")
    M.lines!(error_axis, times, max.(density_state_errors, error_floor);
             color=:darkorange, linewidth=2.4, linestyle=:dash,
             label="density-valued average")
    M.lines!(error_axis, times, max.(population_state_errors, error_floor);
             color=:black, linewidth=1.8, linestyle=:dot,
             label="population vs full PI")
    M.axislegend(error_axis; position=:rb, labelsize=11)

    heatmap = M.heatmap!(
        transition_axis, sector_spins, sector_spins, transition_rates;
        colormap=:magma)
    M.Colorbar(figure[2, 3], heatmap;
               label="sector-changing events / trajectory")
    ExampleMakie.save_example_figure(
        figure, "weak_pi_decay_trajectory_comparison")

    comparison = M.Figure(size=(1650, 480), fontsize=17)
    scaling_axis = M.Axis(
        comparison[1, 1]; xlabel="number of qubits N",
        ylabel="coordinates in one stored state", yscale=log10,
        title="(a) Representation scaling")
    M.lines!(scaling_axis, scaling_sizes, full_density_dimensions;
             color=:gray35, linewidth=2.3, linestyle=:dot,
             label="full density matrix 4ᴺ")
    M.lines!(scaling_axis, scaling_sizes, full_ket_dimensions;
             color=:seagreen4, linewidth=2.3, linestyle=:dash,
             label="full labeled ket 2ᴺ")
    M.lines!(scaling_axis, scaling_sizes, pi_density_dimensions;
             color=:darkorange, linewidth=2.7,
             label="PI density ∑ν dim(Uν)²")
    M.lines!(scaling_axis, scaling_sizes, weak_dimensions;
             color=:dodgerblue, linewidth=2.7,
             label="weak-PI ket ∑ν dim(Uν)")
    M.vlines!(scaling_axis, [N]; color=(:black, 0.35), linewidth=1.5)
    M.axislegend(scaling_axis; position=:lt, labelsize=11)

    timing_axis = M.Axis(
        comparison[1, 2]; xlabel="prepared trajectory backend",
        ylabel="wall time per trajectory [ms]",
        xticks=([1, 2], ["weak-PI pseudo-ket", "density-valued PI"]),
        title="(b) Same fixed-step controls, this run")
    timing_values = 1e3 .* [weak_time_per_path, density_time_per_path]
    M.barplot!(timing_axis, [1, 2], timing_values;
               color=[:dodgerblue, :darkorange])
    M.ylims!(timing_axis, 0, 1.15 * maximum(timing_values))

    memory_axis = M.Axis(
        comparison[1, 3]; xlabel="prepared trajectory backend",
        ylabel="retained history [MiB]",
        xticks=([1, 2], ["weak-PI pseudo-ket", "density-valued PI"]),
        title="(c) Equal paths and saved times")
    history_values = [weak_history_bytes, density_history_bytes] ./ 2.0^20
    M.barplot!(memory_axis, [1, 2], history_values;
               color=[:dodgerblue, :darkorange])
    M.ylims!(memory_axis, 0, 1.15 * maximum(history_values))
    ExampleMakie.save_example_figure(
        comparison, "weak_pi_trajectories_method_comparison")
end
