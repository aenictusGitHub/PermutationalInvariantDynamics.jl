using PermutationalInvariantDynamics

include(joinpath(@__DIR__, "utils", "makie_support.jl"))
using .ExampleMakie

basis = PIBasis(20, 2)
sx = ComplexF64[0 1; 1 0]
sm = ComplexF64[0 1; 0 0]
model = PIModel(basis, [LocalHamiltonian(0.5sx), LocalJump(sm; rate=0.1)])
rho0 = iid_pure_state(basis, ComplexF64[1, 0])

# Compile the reusable matrix-free kernels once. The collective observable
# and one-body marginal share one prepared one-box geometry.
prepared = compile(model; backend=:matrixfree)
one_body_geometry = OneBodyGeometry(basis)
excited_population = CollectiveObservablePlan(
    basis, ComplexF64[0 0; 0 1]; cache=one_body_geometry)
one_body_workspace = OneBodyRDMWorkspace(one_body_geometry, rho0)
rho1 = zeros(ComplexF64, basis.d, basis.d)

solution = solve_dynamics(prepared, rho0, (0.0, 1.0);
                          saveat=0.25, steps_per_interval=32)
excited_fractions = [real(collective_expectation(rho, excited_population)) / basis.N
                     for rho in solution]
one_body_rdm!(rho1, solution[end], one_body_workspace)
report = diagnostics(solution[end])

println("PI dimension: ", pi_dimension(prepared),
        "; backend: ", diagnostics(prepared).backend)
println("excited fraction at t = ", solution.times, ": ", excited_fractions)
println("final one-body state:\n", rho1)
println("final trace error: ", report.trace_error,
        "; minimum eigenvalue: ", report.minimum_eigenvalue)

@assert diagnostics(rho0).valid
@assert report.valid
@assert abs(sum(rho1[index, index] for index in axes(rho1, 1)) - 1) < 1e-10

# Rendering consumes only the sampled excitation values and final one-body
# state already used by the numerical checks.
if makie_available()
    M = makie_module()
    figure = M.Figure(size=(1050, 430), fontsize=17)
    dynamics_axis = M.Axis(
        figure[1, 1]; xlabel="time", ylabel="excited fraction",
        title="Driven qubits with independent decay")
    density_axis = M.Axis(
        figure[1, 2]; xlabel="column state", ylabel="row state",
        xticks=([1, 2], ["|g⟩", "|e⟩"]),
        yticks=([1, 2], ["|g⟩", "|e⟩"]),
        title="Final one-qubit |ρ₁|")

    M.lines!(dynamics_axis, solution.times, excited_fractions;
             color=:royalblue, linewidth=2.7)
    M.scatter!(dynamics_axis, solution.times, excited_fractions;
               color=:royalblue, markersize=8)
    density_plot = M.heatmap!(
        density_axis, 1:2, 1:2, permutedims(abs.(rho1));
        colormap=:viridis, colorrange=(0.0, 1.0))
    M.Colorbar(figure[1, 3], density_plot; label="absolute matrix element")
    save_example_figure(figure, "driven_qubits")
end
