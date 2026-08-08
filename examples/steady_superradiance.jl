using LinearAlgebra
using PermutationalInvariantDynamics
include("paper_models.jl")
using .PaperModels
include(joinpath(@__DIR__, "utils", "makie_support.jl"))
using .ExampleMakie

# D. Meiser and M. J. Holland, PRA 81, 033847 (2010), Eqs. (1),(2),(8)-(10).
quick_example = get(ENV, "PID_EXAMPLE_QUICK", "0") == "1"
N = quick_example ? 10 : 20
GammaC = 1.0
sm = ComplexF64[0 1; 0 0]
excited = ComplexF64[0 0; 0 1]
results = NamedTuple[]

# Resolve the broad weak-pump, superradiant, and saturated regimes on a log
# grid. Include the two analytically important N-scaled rates exactly.
pump_values = if quick_example
    [0.1, 1.0, N * GammaC / 2, N * GammaC, 30.0]
else
    sort!(unique!(vcat(
        exp.(range(log(0.05GammaC), log(4N * GammaC); length=41)),
        [N * GammaC / 2, N * GammaC])))
end
prototype = steady_superradiance_model(
    N; GammaC=GammaC, pump=first(pump_values))
family = compile_family(prototype)
basis = prototype.basis
geometry = OneBodyGeometry(basis)
Jm = collective_operator(basis, sm; cache=geometry)
Neplan = CollectiveObservablePlan(basis, excited; cache=geometry)

for pump in pump_values
    prepared = specialize(family, (GammaC, pump); backend=:sparse)
    rho = stationary_state(prepared; algorithm=DirectAlgorithm())
    intensity = GammaC * real(expectation(rho, adjoint(Jm) * Jm))
    Ne = real(collective_expectation(rho, Neplan))
    enhancement = intensity / (max(Ne, eps()) * GammaC)
    push!(results, (; pump, intensity, Ne, enhancement))
end
large_N_maximum = N^2 / 8
peak = results[argmax(result.intensity for result in results)]
println("steady-superradiance scan: N=$N, points=$(length(results)), " *
        "maximum I/GammaC=$(peak.intensity / GammaC) at " *
        "w/GammaC=$(peak.pump / GammaC)")
println("large-N prediction at w=N GammaC/2: Imax/GammaC = ",
        large_N_maximum)

if makie_available()
    M = makie_module()
    pump_ratios = [result.pump / GammaC for result in results]
    figure = M.Figure(size=(1350, 430), fontsize=17)
    intensity_axis = M.Axis(
        figure[1, 1]; xlabel="w / Γc", ylabel="I / Γc",
        xscale=log10, title="Steady radiated intensity")
    excitation_axis = M.Axis(
        figure[1, 2]; xlabel="w / Γc", ylabel="Ne / N",
        xscale=log10, title="Excited-state fraction")
    enhancement_axis = M.Axis(
        figure[1, 3]; xlabel="w / Γc", ylabel="I / (Ne Γc)",
        xscale=log10, title="Collective enhancement")

    M.lines!(intensity_axis, pump_ratios,
             [result.intensity / GammaC for result in results];
             color=:firebrick, linewidth=2.7)
    M.scatter!(intensity_axis, pump_ratios,
               [result.intensity / GammaC for result in results];
               color=:firebrick, markersize=5, label="finite N = $N")
    M.hlines!(intensity_axis, [large_N_maximum];
              color=:gray45, linestyle=:dash,
              label="large-N peak N²/8")
    M.axislegend(intensity_axis; position=:lt, labelsize=12)

    M.lines!(excitation_axis, pump_ratios,
             [result.Ne / N for result in results];
             color=:royalblue, linewidth=2.7)
    M.scatter!(excitation_axis, pump_ratios,
               [result.Ne / N for result in results];
               color=:royalblue, markersize=5)

    M.lines!(enhancement_axis, pump_ratios,
             [result.enhancement for result in results];
             color=:seagreen, linewidth=2.7)
    M.scatter!(enhancement_axis, pump_ratios,
               [result.enhancement for result in results];
               color=:seagreen, markersize=5, label="collective result")
    M.hlines!(enhancement_axis, [1.0];
              color=:gray45, linestyle=:dash,
              label="independent emission")
    M.axislegend(enhancement_axis; position=:rt, labelsize=12)

    save_example_figure(figure, "steady_superradiance")
end
