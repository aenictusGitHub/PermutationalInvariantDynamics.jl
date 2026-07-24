using PermutationalInvariantDynamics
using LinearAlgebra
using Random

include(joinpath(@__DIR__, "utils", "makie_support.jl"))
using .ExampleMakie

# A small collective-dephasing benchmark. HOPS propagates pure Schur-irrep
# amplitudes, while HEOM below propagates PI density operators. Neither route
# constructs a 2^N state.
N = 3
coefficient = 0.10
frequency = 1.50
final_time = 1.0
times = collect(range(0.0, final_time; length=41))
dt = 0.005
trajectories = 512

basis = PIBasis(N, 2)
spin = spin_matrices()
Jz = collective_operator(basis, spin.jz)
Jx = collective_operator(basis, spin.jx)
H = PIOperator(basis; T=Float64)

rho0 = iid_pure_state(basis, ComplexF64[1, 1] / sqrt(2))
psi0 = weak_pi_pseudoket(rho0)

bath = HOPSBath(Jz, coefficient, frequency)
plan = HOPSPlan(H, bath; max_depth=4, scaling=:scaled)

# One path illustrates the unnormalized linear estimator. Its trace need not
# equal one: only the ensemble average of root outer products is physical.
workspace = HOPSWorkspace(plan)
path = hops_trajectory(
    plan, psi0, times;
    dt, rng=Random.Xoshiro(2026), workspace)
one_path_final = hops_density(path, lastindex(times))
println("one-path final root weight = ", real(trace(one_path_final)))

# Seeds are attached to trajectory indices, so this serial reference remains
# reproducible. Set threaded=true for a larger production ensemble.
hops_states = hops_average(
    plan, psi0, times, trajectories;
    dt, seed=12026, threaded=false)
hops_signal = [
    2real(expectation(rho, Jx)) / N for rho in hops_states
]

# Deterministic PI--HEOM comparison using the same correlation and a hierarchy
# deep enough for this weak, short benchmark.
system = PIModel(basis, ())
heom_bath = HEOMBath(Jz, coefficient, frequency)
heom_plan = HEOMPlan(
    system, heom_bath; max_depth=6, scaling=:scaled)
heom_states = [
    heom_reduced_state(state) for state in
    heom_time_evolution(heom_plan, rho0, times; steps_per_interval=8)
]
heom_signal = [
    2real(expectation(rho, Jx)) / N for rho in heom_states
]

# For commuting collective dephasing with a real exponential correlation, the
# normalized transverse coherence has this Gaussian line shape.
line_shape(t) = coefficient / frequency^2 *
                (frequency*t - 1 + exp(-frequency*t))
analytic_signal = exp.(-line_shape.(times))

hops_heom_error = maximum(abs.(hops_signal .- heom_signal))
heom_analytic_error = maximum(abs.(heom_signal .- analytic_signal))
trace_error = maximum(abs(real(trace(rho)) - 1) for rho in hops_states)

println("HOPS paths = ", trajectories)
println("maximum HOPS--HEOM signal difference = ", hops_heom_error)
println("maximum HEOM--analytic signal difference = ", heom_analytic_error)
println("maximum HOPS mean trace error = ", trace_error)

# The HOPS bounds include Monte Carlo error and are deliberately loose enough
# to be robust for the fixed finite ensemble. Production work should report
# standard errors and repeat with more paths, a smaller dt, and greater depth.
@assert heom_analytic_error < 1e-8
@assert hops_heom_error < 0.10
@assert trace_error < 0.10

if makie_available()
    M = makie_module()
    figure = M.Figure(size=(1120, 440), fontsize=17)
    signal_axis = M.Axis(
        figure[1, 1];
        xlabel="time", ylabel="2⟨Jx⟩ / N",
        title="PI--HOPS collective dephasing (N=$N)")
    error_axis = M.Axis(
        figure[1, 2];
        xlabel="time", ylabel="absolute error",
        yscale=log10, title="Stochastic and hierarchy errors")

    M.lines!(
        signal_axis, times, analytic_signal;
        color=:black, linewidth=3, label="analytic")
    M.lines!(
        signal_axis, times, heom_signal;
        color=:darkorange2, linewidth=2.2, label="PI--HEOM")
    M.scatterlines!(
        signal_axis, times, hops_signal;
        color=:dodgerblue3, markersize=5, linewidth=1.4,
        label="$trajectories PI--HOPS paths")
    M.lines!(
        error_axis, times,
        max.(abs.(hops_signal .- analytic_signal), eps(Float64));
        color=:dodgerblue3, linewidth=2, label="HOPS")
    M.lines!(
        error_axis, times,
        max.(abs.(heom_signal .- analytic_signal), eps(Float64));
        color=:darkorange2, linewidth=2, label="HEOM")
    M.axislegend(signal_axis; position=:lb)
    M.axislegend(error_axis; position=:lt)
    save_example_figure(figure, "pi_hops")
end
