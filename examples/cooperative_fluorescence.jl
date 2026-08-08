using LinearAlgebra
using PermutationalInvariantDynamics
include("paper_models.jl")
using .PaperModels
include(joinpath(@__DIR__, "utils", "makie_support.jl"))
using .ExampleMakie

# S. Morrison and A. S. Parkins, PRA 77, 043810 (2008), Eqs. (1)-(5).
# The normal example uses a research-quality finite-size curve.  Executable
# example CI can request the original three-point validation through its
# stable PID_EXAMPLE_QUICK marker.
quick_example = get(ENV, "PID_EXAMPLE_QUICK", "0") == "1"
N = quick_example ? 20 : 40
Omega = 0.2
gamma_values = quick_example ? [0.1, 0.2, 0.35] :
    collect(range(0.05, 0.60; length=31))
results = NamedTuple[]

# Only scalar rates change along this scan.  Compile the common Schur geometry
# once, then bind the physical Hamiltonian and dissipative rates at each point.
prototype = cooperative_fluorescence_model(
    N; Omega=Omega, gamma=first(gamma_values))
family = compile_family(prototype)
basis = prototype.basis
geometry = OneBodyGeometry(basis)
sy = -ComplexF64[0 -im; im 0] / 2
sz = -ComplexF64[1 0; 0 -1] / 2
Jy = CollectiveObservablePlan(basis, sy; cache=geometry)
Jz = CollectiveObservablePlan(basis, sz; cache=geometry)

for gamma in gamma_values
    prepared = specialize(family, (Omega, 2gamma / N); backend=:sparse)
    stationary = stationary_state(
        prepared; algorithm=DirectAlgorithm(), return_info=true)
    numeric = stationary.state
    exact = cooperative_fluorescence_exact_state(
        basis; Omega=Omega, gamma=gamma)
    # The package basis is |g>,|e>; the paper's spin axes take |e> as m=+1/2.
    Y = real(collective_expectation(numeric, Jy)) / (N / 2)
    Z = real(collective_expectation(numeric, Jz)) / (N / 2)
    exact_Y = real(collective_expectation(exact, Jy)) / (N / 2)
    exact_Z = real(collective_expectation(exact, Jz)) / (N / 2)
    error = norm(numeric.data - exact.data)
    push!(results, (; gamma, Y, Z, exact_Y, exact_Z, error))
    @assert error < 2e-9
end

println("cooperative-fluorescence scan: N=$N, points=$(length(results)), " *
        "maximum exact-state error=$(maximum(result.error for result in results))")

if makie_available()
    M = makie_module()
    controls = [result.gamma / Omega for result in results]
    figure = M.Figure(size=(1000, 430), fontsize=17)
    spin_axis = M.Axis(
        figure[1, 1]; xlabel="γ / Ω", ylabel="collective spin / j",
        title="Finite-size stationary polarization, N=$N")
    error_axis = M.Axis(
        figure[1, 2]; xlabel="γ / Ω", ylabel="‖ρPI − ρexact‖₂",
        yscale=log10, title="Exact-state validation")

    M.lines!(spin_axis, controls, [result.exact_Y for result in results];
             color=:royalblue, linewidth=2.7, label="⟨Jy⟩/j exact")
    M.scatter!(spin_axis, controls, [result.Y for result in results];
               color=:royalblue, marker=:circle, markersize=6,
               label="⟨Jy⟩/j PI solve")
    M.lines!(spin_axis, controls, [result.exact_Z for result in results];
             color=:firebrick, linewidth=2.7, linestyle=:dash,
             label="⟨Jz⟩/j exact")
    M.scatter!(spin_axis, controls, [result.Z for result in results];
               color=:firebrick, marker=:rect, markersize=6,
               label="⟨Jz⟩/j PI solve")
    M.axislegend(spin_axis; position=:rb, labelsize=12)

    error_floor = eps(Float64)
    M.lines!(error_axis, controls,
             [max(result.error, error_floor) for result in results];
             color=:black, linewidth=2.5)
    M.scatter!(error_axis, controls,
               [max(result.error, error_floor) for result in results];
               color=:black, markersize=6)

    save_example_figure(figure, "cooperative_fluorescence")
end
