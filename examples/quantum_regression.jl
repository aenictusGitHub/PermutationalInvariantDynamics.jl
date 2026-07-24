using LinearAlgebra
using PermutationalInvariantDynamics

include(joinpath(@__DIR__, "utils", "makie_support.jl"))
using .ExampleMakie

# A pumped, decaying two-level system has closed-form stationary optical
# correlations, while still testing non-Hermitian QRT insertions.
basis = PIBasis(1, 2)
spin = spin_matrices()
c = collective_spin(basis, :minus)
cdag = adjoint(c)

omega0 = 1.3
gamma_down = 0.8
gamma_up = 0.2
Gamma = gamma_down + gamma_up
excited_population = gamma_up / Gamma

model = PIModel(basis, (
    LocalHamiltonian(spin.jz; rate=omega0),
    LocalJump(spin.jm; rate=gamma_down),
    LocalJump(spin.jp; rate=gamma_up),
))
prepared = compile(model; backend=:matrixfree)
rho_ss = iid_state(basis, ComplexF64[
    1 - excited_population  0
    0                       excited_population
])

# C(tau) = Tr[c' exp(L*tau)(c*rho_ss)].  The plan copies the Schur insertion
# blocks once; all propagation scratch belongs to the reusable workspace.
plan = CorrelationPlan(prepared, cdag, c)
workspace = CorrelationWorkspace(plan; krylovdim=8)
delays = collect(range(0.0, 4.0; length=41))
correlation = two_time_correlation(
    plan, rho_ss, delays; steps_per_interval=16, workspace=workspace)
correlation_exact = excited_population .* exp.(
    (-Gamma / 2 + im * omega0) .* delays)
correlation_error = maximum(abs.(correlation .- correlation_exact))

# A single two-level emitter antibunches: g2(tau)=1-exp(-Gamma*tau).
g2 = delayed_second_order_correlation(
    prepared, rho_ss, c, delays; steps_per_interval=16)
g2_exact = 1 .- exp.(-Gamma .* delays)
g2_error = maximum(abs.(g2 .- g2_exact))

# The stationary spectrum is obtained without materializing the Liouvillian:
# shifted GMRES evaluates the one-sided exp(-i*omega*tau) transform.
frequencies = collect(range(omega0 - 2, omega0 + 2; length=41))
spectrum = optical_spectrum(
    plan, rho_ss, frequencies;
    workspace=workspace, atol=1e-12, rtol=1e-10)
spectrum_exact = [
    excited_population / (Gamma / 2 + im * (frequency - omega0))
    for frequency in frequencies
]
spectrum_error = maximum(abs.(spectrum.values .- spectrum_exact))

# The same sampled correlation can be transformed over a finite window by the
# dependency-free radix-two FFT route.  This is a finite-window approximation,
# whereas `optical_spectrum` above is the infinite-time resolvent result.
fft_spectrum = correlation_spectrum_fft(
    plan, rho_ss, delays; workspace=workspace,
    steps_per_interval=16, nfft=64)

println("compiled backend: ", prepared.backend)
println("maximum time-correlation error: ", correlation_error)
println("maximum delayed-g2 error: ", g2_error)
println("maximum matrix-free spectrum error: ", spectrum_error)
println("g2(0): ", first(g2), "; resonant one-sided spectrum: ",
        spectrum.values[argmin(abs.(frequencies .- omega0))])
println("finite-window FFT bins: ", length(fft_spectrum.frequencies),
        "; convention: ", fft_spectrum.convention)

@assert prepared.backend === :matrixfree
@assert correlation_error < 2e-8
@assert g2_error < 2e-8
@assert abs(first(g2)) < 1e-13
@assert spectrum_error < 2e-10

# The figure reuses the numerical and analytical arrays that enter the error
# assertions; plotting performs no further propagation or shifted solve.
if makie_available()
    M = makie_module()
    figure = M.Figure(size=(1480, 440), fontsize=17)
    correlation_axis = M.Axis(
        figure[1, 1]; xlabel="delay τ", ylabel="C(τ)",
        title="First-order quantum regression")
    antibunching_axis = M.Axis(
        figure[1, 2]; xlabel="delay τ", ylabel="g²(τ)",
        title="Single-emitter antibunching")
    spectrum_axis = M.Axis(
        figure[1, 3]; xlabel="frequency ω", ylabel="S(ω)",
        title="One-sided matrix-free spectrum")

    M.lines!(correlation_axis, delays, real.(correlation_exact);
             color=:black, linewidth=2.5, label="Re analytic")
    M.lines!(correlation_axis, delays, imag.(correlation_exact);
             color=:gray45, linewidth=2.3, linestyle=:dash,
             label="Im analytic")
    M.scatter!(correlation_axis, delays, real.(correlation);
               color=:royalblue, markersize=5, label="Re PI")
    M.scatter!(correlation_axis, delays, imag.(correlation);
               color=:darkorange, markersize=5, marker=:utriangle,
               label="Im PI")
    M.axislegend(correlation_axis; position=:rt, labelsize=10)

    M.lines!(antibunching_axis, delays, g2_exact;
             color=:black, linewidth=2.6, label="analytic")
    M.scatter!(antibunching_axis, delays, real.(g2);
               color=:firebrick, markersize=6, label="PI")
    M.axislegend(antibunching_axis; position=:rb, labelsize=11)

    M.lines!(spectrum_axis, frequencies, real.(spectrum_exact);
             color=:black, linewidth=2.5, label="Re analytic")
    M.lines!(spectrum_axis, frequencies, imag.(spectrum_exact);
             color=:gray45, linewidth=2.3, linestyle=:dash,
             label="Im analytic")
    M.scatter!(spectrum_axis, frequencies, real.(spectrum.values);
               color=:seagreen, markersize=5, label="Re GMRES")
    M.scatter!(spectrum_axis, frequencies, imag.(spectrum.values);
               color=:purple, markersize=5, marker=:diamond,
               label="Im GMRES")
    M.axislegend(spectrum_axis; position=:rt, labelsize=10)
    save_example_figure(figure, "quantum_regression")
end
