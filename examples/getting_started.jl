using PermutationalInvariantDynamics

include(joinpath(@__DIR__, "utils", "makie_support.jl"))
using .ExampleMakie

# 1. Complete PI basis for N identical qubits.
N = 8
basis = PIBasis(N, 2)

# 2. Local basis: |g> = [1,0], |e> = [0,1].
spin = spin_matrices()
sm = spin.jm
sp = spin.jp
number = sp * sm

# 3. Declarative physical terms and model.
model = PIModel(basis, (
    LocalHamiltonian(spin.jx; rate=0.7),
    LocalJump(sm; rate=0.12),
    LocalJump(sp; rate=0.02),
))

# 4. Fully excited identical product state.
rho0 = computational_product_state(basis, 2)
validate_state(rho0)

# 5. Lower Schur geometry and Liouvillian kernels once.
prepared = compile(model; backend=:auto)

# 6. Evolve and sample the collective excitation.
times = collect(0.0:0.1:4.0)
solution = solve_dynamics(
    prepared, rho0, (first(times), last(times));
    saveat=times,
    steps_per_interval=16,
    observables=(excited=number,),
    save_states=true,
)
excited_fraction = real.(solution.observables[:excited]) ./ N

# 7. Solve the autonomous stationary problem and retain its residual metadata.
steady = stationary_state(prepared; return_info=true)

# 8. Compare two time-step resolutions for the reported observable.
coarse = solve_dynamics(
    prepared, rho0, (first(times), last(times));
    saveat=times,
    steps_per_interval=8,
    observables=(excited=number,),
    save_states=false,
)
coarse_excited_fraction = real.(coarse.observables[:excited]) ./ N
pointwise_step_error = abs.(
    coarse_excited_fraction .- excited_fraction)
step_error = maximum(pointwise_step_error)
steady_excited_fraction = real(
    collective_expectation(steady.state, number)) / N

@assert diagnostics(solution[end]).valid
@assert steady.info.converged
@assert diagnostics(steady.state).valid
@assert step_error < 1e-6

println("PI coordinates: ", pi_dimension(basis))
println("selected backend: ", diagnostics(prepared).backend)
println("final excitation fraction: ", last(excited_fraction))
println("stationary residual: ", steady.info.residual)
println("8-versus-16-step excitation difference: ", step_error)

# Plot only arrays and the stationary state already computed above. The
# package-root environment has no CairoMakie dependency, so numerical checks
# still run there and this optional block is skipped.
if makie_available()
    M = makie_module()
    figure = M.Figure(size=(1080, 430), fontsize=17)
    dynamics_axis = M.Axis(
        figure[1, 1]; xlabel="time", ylabel="excited fraction",
        title="Prepared PI dynamics and stationary value")
    convergence_axis = M.Axis(
        figure[1, 2]; xlabel="time",
        ylabel="|8-step − 16-step result|",
        yscale=log10, title="RK4 output-grid refinement")

    M.lines!(dynamics_axis, times, excited_fraction;
             color=:royalblue, linewidth=2.7, label="16 steps / interval")
    M.scatter!(dynamics_axis, times, coarse_excited_fraction;
               color=:darkorange, markersize=6, label="8 steps / interval")
    M.hlines!(dynamics_axis, [steady_excited_fraction];
              color=:black, linewidth=2, linestyle=:dash,
              label="stationary state")
    M.axislegend(dynamics_axis; position=:rt, labelsize=11)

    shown_error = max.(pointwise_step_error, eps(Float64))
    M.lines!(convergence_axis, times, shown_error;
             color=:firebrick, linewidth=2.4)
    M.scatter!(convergence_axis, times, shown_error;
               color=:firebrick, markersize=6)
    save_example_figure(figure, "getting_started")
end
