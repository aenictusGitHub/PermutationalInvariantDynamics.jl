using PermutationalInvariantDynamics
using LinearAlgebra

include(joinpath(@__DIR__, "utils", "makie_support.jl"))
using .ExampleMakie

basis = PIBasis(20, 2)
omega = 1.0
gamma = 0.1
sx = ComplexF64[0 1; 1 0]
sm = ComplexF64[0 1; 0 0]
model = PIModel(basis, [LocalHamiltonian(omega * sx / 2), LocalJump(sm; rate=gamma)])
rho0 = iid_pure_state(basis, ComplexF64[1, 0])

# Compile the reusable matrix-free kernels once. The collective observable
# and one-body marginal share one prepared one-box geometry.
prepared = compile(model; backend=:matrixfree)
one_body_geometry = OneBodyGeometry(basis)
excited_population = CollectiveObservablePlan(
    basis, ComplexF64[0 0; 0 1]; cache=one_body_geometry)
one_body_workspace = OneBodyRDMWorkspace(one_body_geometry, rho0)
rho1 = zeros(ComplexF64, basis.d, basis.d)

times = collect(range(0.0, 12.0; length=121))
solution = solve_dynamics(prepared, rho0, (first(times), last(times));
                          saveat=times, steps_per_interval=32)
excited_fractions = [real(collective_expectation(rho, excited_population)) / basis.N
                     for rho in solution]
one_body_rdm!(rho1, solution[end], one_body_workspace)
report = diagnostics(solution[end])

# Independent sites obey the same one-qubit optical Bloch equation. Evolve
# [p_e, Im(rho_ge), 1] with a constant 3-by-3 affine generator; its size is
# independent of N. This reference also checks the sign of the coherence.
bloch_generator = [-gamma omega 0; -omega -gamma/2 omega/2; 0 0 0]
bloch = [exp(t * bloch_generator) * [0.0, 0.0, 1.0] for t in times]
exact_fractions = first.(bloch)
fraction_errors = abs.(excited_fractions .- exact_fractions)
p, coherence = bloch[end][1:2]
exact_rho1 = ComplexF64[1-p im*coherence; -im*coherence p]
steady_fraction = omega^2 / (gamma^2 + 2omega^2)

println("PI dimension: ", pi_dimension(prepared),
        "; backend: ", diagnostics(prepared).backend)
println("sampled ", length(times), " times on Ωt ∈ [0, 12]")
println("maximum excitation error against optical Bloch solution: ", maximum(fraction_errors))
println("final one-body state:\n", rho1)
println("final trace error: ", report.trace_error,
        "; minimum eigenvalue: ", report.minimum_eigenvalue)

@assert diagnostics(rho0).valid
@assert report.valid
@assert abs(sum(rho1[index, index] for index in axes(rho1, 1)) - 1) < 1e-10
@assert maximum(fraction_errors) < 1e-9
@assert norm(rho1 - exact_rho1) < 1e-9

# Rendering consumes only the sampled excitation values and final one-body
# state already used by the numerical checks.
if makie_available()
    M = makie_module()
    figure = example_figure(size=(1140, 780))
    M.Label(figure[0, 1:3], "Driven independent qubits  •  N=$(basis.N), γ/Ω=$(gamma/omega)";
            fontsize=22, font=:bold, halign=:left)
    dynamics_axis = M.Axis(
        figure[1, 1]; xlabel="Ωt", ylabel="excited fraction",
        title="(a) Resolved Rabi oscillations")
    error_axis = M.Axis(figure[1, 2]; xlabel="Ωt", ylabel="absolute fraction error",
                        title="(b) PI versus optical Bloch solution")
    M.lines!(dynamics_axis, omega .* times, exact_fractions;
             color=:black, label="optical Bloch reference")
    shown = 1:4:length(times)
    M.scatter!(dynamics_axis, omega .* times[shown], excited_fractions[shown];
               color=example_colors.blue, markersize=7, label="PI samples (every fourth)")
    M.hlines!(dynamics_axis, [steady_fraction]; color=example_colors.gray,
              linestyle=:dash, label="stationary fraction")
    M.ylims!(dynamics_axis, 0, 1.1)
    M.axislegend(dynamics_axis; position=:rt, labelsize=12)
    M.lines!(error_axis, omega .* times, fraction_errors; color=example_colors.red)

    # Signed components expose coherence phase; magnitude alone discards it.
    for (column, component, label) in ((1, real, "(c) Re ρ₁ at Ωt = 12"),
                                      (2, imag, "(d) Im ρ₁ at Ωt = 12"))
        axis = M.Axis(figure[2, column]; xlabel="column state", ylabel="row state",
            xticks=(1:2, ["g", "e"]), yticks=(1:2, ["g", "e"]),
            yreversed=true, aspect=M.DataAspect(), title=label,
            xgridvisible=false, ygridvisible=false)
        values = component.(rho1)
        plot = M.heatmap!(axis, 1:2, 1:2, permutedims(values);
                         colormap=:RdBu, colorrange=(-1, 1))
        for row in 1:2, col in 1:2
            M.text!(axis, col, row; text=string(round(values[row, col]; digits=4)),
                    align=(:center, :center), fontsize=18,
                    color=abs(values[row, col]) > 0.6 ? :white : :black)
        end
        column == 2 && M.Colorbar(figure[2, 3], plot; label="signed matrix element")
    end
    M.Label(figure[3, 1:3], "RK4: 32 steps per saved interval  •  All 121 samples are exported; errors are unmodified.";
            fontsize=13, color=example_colors.gray)
    save_example_figure(figure, "driven_qubits")
    save_example_data("driven_qubits", (;
        time=times, omega_t=omega .* times, excited_fraction=excited_fractions,
        exact_fraction=exact_fractions, absolute_error=fraction_errors);
        metadata=(; N=basis.N, omega, gamma, steps_per_interval=32))
    save_example_data("driven_qubits_final_state", (;
        row=[1, 2, 1, 2], column=[1, 1, 2, 2],
        real=vec(real.(rho1)), imaginary=vec(imag.(rho1)));
        metadata=(; time=last(times), basis_order="g, e", N=basis.N))
end
