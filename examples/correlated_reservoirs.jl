using PermutationalInvariantDynamics
using LinearAlgebra

include(joinpath(@__DIR__, "utils", "makie_support.jl"))
using .ExampleMakie

const N = 8
const BASIS = PIBasis(N, 2)
const SIGMA_MINUS = ComplexF64[0 1; 0 0]
const SIGMA_Z = ComplexF64[1 0; 0 -1]

# Writing Gamma = C*C' makes complete positivity manifest while retaining a
# complex off-diagonal cross correlation between emission and dephasing.
const MIXING = ComplexF64[1.0 0.20im; 0.30 0.50]
const GAMMA = MIXING * adjoint(MIXING)
const RATE = 0.08

rho0 = iid_pure_state(BASIS, ComplexF64[0, 1])
model = PIModel(BASIS, (
    CorrelatedLocalJumps((SIGMA_MINUS, SIGMA_Z), GAMMA; rate=RATE),
))
prepared = compile(model; backend=:matrixfree)

times = range(0.0, 4.0; length=41)
excitation = ComplexF64[0 0; 0 1]
excitation_operator = collective_operator(BASIS, excitation)
dynamics = solve_dynamics(
    prepared, rho0, (first(times), last(times));
    saveat=collect(times), steps_per_interval=8,
    observables=(excitation=excitation_operator,), save_states=false)
population = real.(dynamics.observables[:excitation]) ./ N
@assert dynamics.states === nothing

# A fixed positive-semidefinite Gamma is factorized once. The resulting
# effective independent jumps provide a useful exact regression.
effective = ntuple(channel ->
    MIXING[1,channel]*SIGMA_MINUS + MIXING[2,channel]*SIGMA_Z, 2)
reference = PIModel(BASIS,
    map(operator -> LocalJump(operator; rate=RATE), effective))
L_correlated = liouvillian(model; representation=:sparse)
L_reference = liouvillian(reference; representation=:sparse)
@assert Matrix(L_correlated) ≈ Matrix(L_reference) atol=5e-12 rtol=5e-12

# For a driven reservoir, InPlaceTimeOperator makes Gamma(t) task-local and
# preallocated. This schedule keeps Gamma(t) PSD by multiplying it by 1+t.
schedule = InPlaceTimeOperator(GAMMA, (destination, t, parameters) -> begin
    scale = one(t) + parameters.ramp*t
    @inbounds for index in eachindex(destination)
        destination[index] *= scale
    end
    nothing
end)
driven = PIModel(BASIS, (
    CorrelatedCollectiveJumps((SIGMA_MINUS, SIGMA_Z), schedule;
                              rate=RATE),
))
plan = LiouvillianPlan(driven)
workspace = LiouvillianWorkspace(plan)
source = copy(rho0.data)
destination = similar(source)
apply!(destination, plan, source, 0.4, (ramp=0.2,), workspace)
instantaneous = freeze(driven; time=0.4, parameters=(ramp=0.2,),
                       representation=:sparse)
@assert destination ≈ instantaneous*source atol=5e-12 rtol=5e-12

println("Correlated local reservoir final excitation fraction: ",
        population[end])
println("PI dimension: ", length(BASIS),
        "; full Hilbert dimension avoided: ", big(2)^N)

if makie_available()
    M = makie_module()
    figure = M.Figure(size=(1040, 420), fontsize=17)
    population_axis = M.Axis(
        figure[1, 1];
        xlabel="time", ylabel="excited fraction",
        title="Correlated local-reservoir dynamics")
    kossakowski_axis = M.Axis(
        figure[1, 2];
        xlabel="input channel", ylabel="output channel",
        xticks=(1:2, ["σ₋", "σz"]),
        yticks=(1:2, ["σ₋", "σz"]),
        title="|Γab| (positive-semidefinite bath)")

    M.lines!(
        population_axis, collect(times), population;
        color=:dodgerblue3, linewidth=2.7)
    M.scatter!(
        population_axis, collect(times), population;
        color=:dodgerblue3, markersize=6)
    gamma_plot = M.heatmap!(
        kossakowski_axis, 1:2, 1:2, abs.(GAMMA);
        colormap=:viridis, colorrange=(0, maximum(abs, GAMMA)))
    M.Colorbar(figure[1, 3], gamma_plot; label="magnitude")
    save_example_figure(figure, "correlated_reservoirs")
end
