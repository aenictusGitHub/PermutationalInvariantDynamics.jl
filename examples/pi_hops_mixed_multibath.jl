using PermutationalInvariantDynamics
using LinearAlgebra

include(joinpath(@__DIR__, "utils", "makie_support.jl"))
using .ExampleMakie

# A genuinely mixed product state under two independent shared dephasing
# baths. Every noise realization remains PI because both baths multiply the
# same collective operator Jz.
N = 3
initial_coherence = 0.55
coefficients = (0.12, 0.05)
frequencies = (0.80, 2.00)
final_time = 1.5
times = collect(range(0.0, final_time; length=61))
dt = 0.01
trajectories = 384

basis = PIBasis(N, 2)
spin = spin_matrices()
Jz = collective_operator(basis, spin.jz)
Jx = collective_operator(basis, spin.jx)
H = PIOperator(basis; T=Float64)

local_density = ComplexF64[
    0.5 initial_coherence / 2
    initial_coherence / 2 0.5
]
rho0 = iid_state(basis, local_density)

slow_bath = HOPSBath(
    Jz, coefficients[1], frequencies[1]; label=:slow_bath)
fast_bath = HOPSBath(
    Jz, coefficients[2], frequencies[2]; label=:fast_bath)
baths = (slow_bath, fast_bath)
plan = HOPSPlan(H, baths; max_depth=4, scaling=:scaled)

# The Schur spectral ensemble is prepared once from the mixed PI state. It
# samples only positive eigencomponents of multiplicity-weighted Schur blocks;
# no multiplicity tableaux or labeled-particle vectors are introduced.
initial = hops_initial_ensemble(plan, rho0)

# Reuse one task-owned hierarchy workspace per active worker. A tiny direct
# PIState call first exercises the convenience overload, then the main run
# reuses the same batch storage with the explicitly prepared initial ensemble.
worker_count = min(Threads.nthreads(), 4)
threaded = worker_count > 1
batch = HOPSBatchWorkspace(plan; workers=worker_count)
probe = hops_average(
    plan, rho0, [0.0], max(2, worker_count);
    dt, seed=1026, threaded, workspace=batch, return_info=true)
result = hops_average(
    plan, initial, times, trajectories;
    dt, seed=2026, threaded, workspace=batch, return_info=true)

hops_signal = [
    2real(expectation(rho, Jx)) / N for rho in result
]
line_shape(t) = sum(
    coefficients[index] / frequencies[index]^2 *
    (frequencies[index] * t - 1 +
     exp(-frequencies[index] * t))
    for index in eachindex(coefficients))
analytic_signal =
    initial_coherence .* exp.(-line_shape.(times))

# HOPSEnsembleResult reports a Hilbert--Schmidt state standard error. The
# Cauchy--Schwarz contraction below turns it into a conservative error bar for
# the normalized Jx signal; it is not a hierarchy- or time-step-error estimate.
observable_norm = 2norm(Jx.data) / N
signal_standard_error_bound =
    observable_norm .* result.standard_error
signal_error = abs.(hops_signal .- analytic_signal)
six_sigma_violation = maximum(
    signal_error .- 6signal_standard_error_bound)
trace_error = maximum(abs(real(trace(rho)) - 1) for rho in result)

metadata = hops_hierarchy_metadata(plan)
multiindices = hops_multiindices(plan)
importances = hops_auxiliary_importances(plan)
coordinate_scales = [
    hops_coordinate_scale(plan, index)
    for index in eachindex(multiindices)
]

# Importance pruning is demonstrated as setup metadata only. The retained
# downward-closed hierarchy is an approximation and this score is not an
# accuracy certificate.
pruned_plan = HOPSPlan(
    H, baths; max_depth=10, scaling=:scaled,
    importance_cutoff=0.01)
pruned_metadata = hops_hierarchy_metadata(pruned_plan)
pruned_importances = hops_auxiliary_importances(pruned_plan)

println("PI--HOPS mixed two-bath dephasing")
println("initial spectral components = ", length(initial.weights),
        "; occupied Schur sectors = ", sort(unique(initial.sectors)))
println("complete hierarchy metadata = ", metadata)
println("first hierarchy labels = ",
        multiindices[1:min(end, 6)])
println("first coordinate scales = ",
        coordinate_scales[1:min(end, 6)])
println("pruned/full auxiliaries = ",
        pruned_metadata.retained_auxiliaries, "/",
        pruned_metadata.full_auxiliaries)
println("maximum signal error = ", maximum(signal_error))
println("maximum six-standard-error violation = ", six_sigma_violation)
println("maximum mean trace error = ", trace_error)

@assert abs(trace(rho0) - 1) < 3e-14
@assert length(unique(initial.sectors)) > 1
@assert abs(sum(initial.weights) - 1) < 3e-14
@assert real(trace(only(probe))) ≈ 1 atol=3e-14
@assert result.trajectory_count == trajectories
@assert metadata.baths == 2
@assert metadata.poles == 2
@assert metadata.retained_auxiliaries == binomial(2 + 4, 4)
@assert length(multiindices) == metadata.retained_auxiliaries
@assert length(importances) == metadata.retained_auxiliaries
@assert length(coordinate_scales) == metadata.retained_auxiliaries
@assert pruned_metadata.pruning_approximation
@assert pruned_metadata.retained_auxiliaries <
        pruned_metadata.full_auxiliaries
@assert minimum(pruned_importances) >=
        pruned_metadata.importance_cutoff
@assert six_sigma_violation < 0.01
@assert trace_error < 0.12

if makie_available()
    M = makie_module()
    figure = M.Figure(size=(1120, 440), fontsize=17)
    signal_axis = M.Axis(
        figure[1, 1];
        xlabel="time", ylabel="2⟨Jx⟩ / N",
        title="Mixed PI state, two shared baths")
    hierarchy_axis = M.Axis(
        figure[1, 2];
        xlabel="hierarchy node", ylabel="importance score",
        yscale=log10, title="Pruning diagnostic (not an error bound)")

    lower = hops_signal .- 2signal_standard_error_bound
    upper = hops_signal .+ 2signal_standard_error_bound
    M.band!(
        signal_axis, times, lower, upper;
        color=(:dodgerblue3, 0.22),
        label="2× state-HS standard-error bound")
    M.lines!(
        signal_axis, times, analytic_signal;
        color=:black, linewidth=3, label="analytic")
    M.scatterlines!(
        signal_axis, times, hops_signal;
        color=:dodgerblue3, markersize=5, linewidth=1.5,
        label="$trajectories HOPS paths")

    M.scatter!(
        hierarchy_axis, eachindex(importances), importances;
        color=:gray45, markersize=7, label="complete depth 4")
    M.hlines!(
        hierarchy_axis, [pruned_metadata.importance_cutoff];
        color=:firebrick3, linewidth=2, linestyle=:dash,
        label="depth-10 cutoff")
    M.scatter!(
        hierarchy_axis,
        1:length(pruned_importances), pruned_importances;
        color=:darkorange2, markersize=8,
        label="retained depth 10")
    M.axislegend(signal_axis; position=:lb)
    M.axislegend(hierarchy_axis; position=:rt)
    save_example_figure(figure, "pi_hops_mixed_multibath")
end
