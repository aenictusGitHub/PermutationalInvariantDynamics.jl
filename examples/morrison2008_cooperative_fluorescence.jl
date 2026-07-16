using LinearAlgebra
using PermutationalInvariantDynamics
include("paper_models.jl")
using .PaperModels
include(joinpath(@__DIR__, "utils", "makie_support.jl"))
using .ExampleMakie

# S. Morrison and A. S. Parkins, PRA 77, 043810 (2008), Eqs. (1)-(5).
N = 20
Omega = 0.2
results = NamedTuple[]

for gamma in (0.1, 0.2, 0.35)
    model = morrison2008_model(N; Omega=Omega, gamma=gamma)
    prepared = compile(model)
    stationary = stationary_state(
        prepared; algorithm=DirectAlgorithm(), return_info=true)
    numeric = stationary.state
    exact = morrison2008_exact_state(
        model.basis; Omega=Omega, gamma=gamma)
    # The package basis is |g>,|e>; the paper's spin axes take |e> as m=+1/2.
    sy = -ComplexF64[0 -im; im 0] / 2
    sz = -ComplexF64[1 0; 0 -1] / 2
    geometry = OneBodyGeometry(model.basis)
    Jy = CollectiveObservablePlan(model.basis, sy; cache=geometry)
    Jz = CollectiveObservablePlan(model.basis, sz; cache=geometry)
    Y = real(collective_expectation(numeric, Jy)) / (N / 2)
    Z = real(collective_expectation(numeric, Jz)) / (N / 2)
    exact_Y = real(collective_expectation(exact, Jy)) / (N / 2)
    exact_Z = real(collective_expectation(exact, Jz)) / (N / 2)
    error = norm(numeric.data - exact.data)
    push!(results, (; gamma, Y, Z, exact_Y, exact_Z, error))
    println("gamma=$gamma: exact-state error=$error, " *
            "<Jy>/j=$Y, <Jz>/j=$Z")
    @assert error < 2e-9
end

if makie_available()
    M = makie_module()
    controls = [result.gamma / Omega for result in results]
    figure = M.Figure(size=(1000, 430), fontsize=17)
    spin_axis = M.Axis(
        figure[1, 1]; xlabel="γ / Ω", ylabel="collective spin / j",
        title="Finite-size stationary polarization")
    error_axis = M.Axis(
        figure[1, 2]; xlabel="γ / Ω", ylabel="‖ρPI − ρexact‖₂",
        yscale=log10, title="Exact-state validation")

    M.lines!(spin_axis, controls, [result.exact_Y for result in results];
             color=:royalblue, linewidth=2.7, label="⟨Jy⟩/j exact")
    M.scatter!(spin_axis, controls, [result.Y for result in results];
               color=:royalblue, marker=:circle, markersize=11,
               label="⟨Jy⟩/j PI solve")
    M.lines!(spin_axis, controls, [result.exact_Z for result in results];
             color=:firebrick, linewidth=2.7, linestyle=:dash,
             label="⟨Jz⟩/j exact")
    M.scatter!(spin_axis, controls, [result.Z for result in results];
               color=:firebrick, marker=:rect, markersize=11,
               label="⟨Jz⟩/j PI solve")
    M.axislegend(spin_axis; position=:rb, labelsize=12)

    error_floor = eps(Float64)
    M.lines!(error_axis, controls,
             [max(result.error, error_floor) for result in results];
             color=:black, linewidth=2.5)
    M.scatter!(error_axis, controls,
               [max(result.error, error_floor) for result in results];
               color=:black, markersize=11)

    save_example_figure(figure, "morrison2008_cooperative_fluorescence")
end
