using LinearAlgebra
using Random
using Statistics
using PermutationalInvariantDynamics

include(joinpath(@__DIR__, "utils", "makie_support.jl"))
using .ExampleMakie

# Collective fluorescence monitored by inefficient homodyne detection.  This
# is the PI stochastic-master-equation convention of Wiseman and Milburn.  The
# example is deliberately modest; increase ntrajectories after converging dt.
N = 8
gamma = 0.3 / N
efficiency = 0.75
ntrajectories = 256
times = collect(range(0.0, 2.0; length=81))
dt = 0.0025

basis = PIBasis(N, 2)
sm = ComplexF64[0 1; 0 0]
model = PIModel(basis, (CollectiveJump(sm; rate=gamma),))
rho0 = computational_product_state(basis, 2)
Jz = collective_spin(basis, :z)

# CollectiveJump(sm; rate=gamma) is D[sqrt(gamma) J_-].  The unconditional
# channel is in model; the monitor supplies its conditional innovation only.
monitor = homodyne_monitor(sqrt(gamma) * sm;
    efficiency, phase=0.0, label=:fluorescence)
plan = DiffusivePlan(model, monitor)

conditional = diffusive_trajectory(
    plan, rho0, times; dt, rng=MersenneTwister(1985),
    observables=(magnetization=Jz,),
)
paths = diffusive_trajectories(
    plan, rho0, times, ntrajectories; dt, seed=1985, threaded=true,
)
averaged = diffusive_average(paths)
master = time_evolution(model, rho0, times; steps_per_interval=4)

conditional_mz = vec(conditional.observables.values)
ensemble_mz = [mean(real(expectation(path[index], Jz)) for path in paths)
               for index in eachindex(times)]
ensemble_se = [std(real(expectation(path[index], Jz)) for path in paths) /
               sqrt(ntrajectories) for index in eachindex(times)]
master_mz = real.([expectation(state, Jz) for state in master])
average_mz = real.([expectation(state, Jz) for state in averaged])

@assert maximum(abs.(average_mz .- ensemble_mz)) < 5e-12
@assert all(abs.(ensemble_mz .- master_mz) .<= 6 .* ensemble_se .+ 0.04)
@assert maximum(abs(trace(state) - 1) for state in conditional.states) < 2e-11

println("PI coordinates = ", length(basis), ", full Hilbert dimension = 2^", N)
println("trajectories = ", ntrajectories, ", efficiency = ", efficiency)
println("max ensemble/master |Δ<Jz>| = ", maximum(abs.(ensemble_mz .- master_mz)))

if makie_available()
    M = makie_module()
    figure = M.Figure(size=(1050, 760), fontsize=17)
    axis1 = M.Axis(figure[1, 1]; xlabel="time", ylabel="<Jz>",
                   title="Conditional and unconditional PI dynamics")
    M.lines!(axis1, times, conditional_mz; color=:royalblue,
             label="one homodyne record")
    M.band!(axis1, times, ensemble_mz .- ensemble_se,
            ensemble_mz .+ ensemble_se; color=(:firebrick, 0.20))
    M.lines!(axis1, times, ensemble_mz; color=:firebrick,
             label="trajectory mean")
    M.lines!(axis1, times, master_mz; color=:black, linestyle=:dash,
             label="master equation")
    M.axislegend(axis1; position=:rb)

    axis2 = M.Axis(figure[2, 1]; xlabel="time", ylabel="integrated Y(t)",
                   title="Cumulative homodyne measurement record")
    M.lines!(axis2, times, vec(conditional.records); color=:darkgreen)
    save_example_figure(figure, "homodyne_pi_trajectories")
end
