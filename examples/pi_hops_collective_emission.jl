using PermutationalInvariantDynamics
using LinearAlgebra
using Random

include(joinpath(@__DIR__, "utils", "makie_support.jl"))
using .ExampleMakie

# One symmetric excitation coupled through J_- to a shared exponential bath.
# Collective operators preserve total spin, so the fully symmetric Schur
# sector is an exact restricted basis for this model.
N = 20
coefficient = 0.01
frequency = 1.20
final_time = 2.0
times = collect(range(0.0, final_time; length=81))
dt = 0.0025
depths = (0, 1, 2)

basis = PIBasis(N, 2; sectors=[(N, 0)])
spin = spin_matrices()
Jminus = collective_operator(basis, spin.jm)
excitation = collective_operator(basis, spin.jp * spin.jm)
H = PIOperator(basis; T=Float64)

rho0 = dicke_state(basis, N / 2, 1 - N / 2)
psi0 = weak_pi_pseudoket(rho0)
bath = HOPSBath(
    Jminus, coefficient, frequency; label=:lorentzian_emission)

# In the one-excitation manifold the symmetric survival amplitude G obeys
#
#     G'' + frequency*G' + N*coefficient*G = 0,
#     G(0)=1, G'(0)=0.
#
# The complex square root keeps the same formula valid on both sides of the
# overdamped/underdamped crossover.
delta = sqrt(complex(frequency^2 - 4N * coefficient))
survival_amplitude(t) =
    exp(-frequency * t / 2) *
    (cosh(delta * t / 2) + (frequency / delta) * sinh(delta * t / 2))
exact_excitation = abs2.(survival_amplitude.(times))

plans = [
    HOPSPlan(H, bath; max_depth=depth, scaling=:scaled)
    for depth in depths
]
paths = [
    hops_trajectory(
        plan, psi0, times;
        dt, rng=Random.Xoshiro(2026), workspace=HOPSWorkspace(plan))
    for plan in plans
]
path_excitation = [
    [real(expectation(hops_density(path, index), excitation))
     for index in eachindex(times)]
    for path in paths
]
path_errors = [
    maximum(abs.(signal .- exact_excitation))
    for signal in path_excitation
]

# Depth one is exact for this initial manifold: J_- maps the W state to the
# ground state, and applying J_- once more gives zero. A small ensemble still
# demonstrates the streaming average and uncertainty result. The excitation
# curve itself is path-independent even though the unnormalized root norm is
# stochastic.
exact_plan = plans[2]
ensemble = hops_average(
    exact_plan, psi0, times, 8;
    dt, seed=3026, threaded=false,
    workspace=HOPSWorkspace(exact_plan), return_info=true)
ensemble_excitation = [
    real(expectation(rho, excitation)) for rho in ensemble
]
ensemble_error = maximum(abs.(ensemble_excitation .- exact_excitation))

# A deterministic external provider is useful for conditioned-path debugging.
# Zero noise is not a stochastic realization of the stated covariance and is
# therefore not used as a physical ensemble estimate.
zero_noise = (_time::Real, _bath::Integer) -> 0.0 + 0.0im
conditioned_workspace = HOPSWorkspace(exact_plan)
conditioned_path = hops_trajectory(
    exact_plan, psi0, times;
    dt, noise=zero_noise, workspace=conditioned_workspace)
conditioned_excitation = [
    real(expectation(hops_density(conditioned_path, index), excitation))
    for index in eachindex(times)
]
conditioned_error =
    maximum(abs.(conditioned_excitation .- exact_excitation))

# The low-level conditioned hierarchy action is deterministic and
# allocation-ready once a task-owned workspace has been prepared.
nweak = weak_pi_dimension(basis)
source = zeros(ComplexF64, size(exact_plan, 1))
copyto!(view(source, 1:nweak), psi0.data)
rhs = similar(source)
rhs_repeated = similar(source)
hops_rhs!(
    rhs, exact_plan, source, ComplexF64[0], conditioned_workspace)
hops_rhs!(
    rhs_repeated, exact_plan, source, ComplexF64[0],
    conditioned_workspace)

metadata = hops_hierarchy_metadata(exact_plan)
multiindices = hops_multiindices(exact_plan)
importances = hops_auxiliary_importances(exact_plan)
first_auxiliary_scale = hops_coordinate_scale(exact_plan, [1])

final_path_density = hops_density(paths[2], lastindex(times))
final_root_weight = norm(paths[2].states[end].data)^2

println("PI--HOPS collective emission")
println("N=$N; weak-PI root coordinates=$nweak; " *
        "PI density coordinates=$(length(basis))")
println("depth errors ", Dict(depths[index] => path_errors[index]
                              for index in eachindex(depths)))
println("depth-one ensemble excitation error = ", ensemble_error)
println("conditioned zero-noise excitation error = ", conditioned_error)
println("depth-one hierarchy metadata = ", metadata)
println("multi-indices = ", multiindices)
println("heuristic auxiliary importances = ", importances)
println("scale of auxiliary [1] = ", first_auxiliary_scale)
println("one-path final root weight = ", final_root_weight)

@assert metadata.retained_auxiliaries == 2
@assert metadata.weak_pi_dimension == N + 1
@assert metadata.coordinate_dimension == 2(N + 1)
@assert multiindices == [[0], [1]]
@assert length(importances) == metadata.retained_auxiliaries
@assert first_auxiliary_scale ≈ sqrt(coefficient) atol=2e-15
@assert path_errors[1] > 0.25
@assert path_errors[2] < 2e-9
@assert path_errors[3] < 2e-9
@assert ensemble_error < 2e-9
@assert conditioned_error < 2e-9
@assert rhs_repeated == rhs
@assert real(trace(final_path_density)) ≈
        final_root_weight atol=3e-13 rtol=3e-13

if makie_available()
    M = makie_module()
    figure = M.Figure(size=(1120, 440), fontsize=17)
    signal_axis = M.Axis(
        figure[1, 1];
        xlabel="time", ylabel="collective excitation",
        title="One-excitation collective emission (N=$N)")
    error_axis = M.Axis(
        figure[1, 2];
        xlabel="time", ylabel="absolute error",
        yscale=log10, title="Hard hierarchy boundary")

    M.lines!(
        signal_axis, times, exact_excitation;
        color=:black, linewidth=3, label="analytic")
    colors = (:gray45, :dodgerblue3, :darkorange2)
    for (index, depth) in pairs(depths)
        M.lines!(
            signal_axis, times, path_excitation[index];
            color=colors[index], linewidth=2,
            linestyle=depth == 0 ? :dash : :solid,
            label="HOPS depth $depth")
        M.lines!(
            error_axis, times,
            max.(abs.(path_excitation[index] .- exact_excitation),
                 eps(Float64));
            color=colors[index], linewidth=2, label="depth $depth")
    end
    M.axislegend(signal_axis; position=:lb)
    M.axislegend(error_axis; position=:rt)
    save_example_figure(figure, "pi_hops_collective_emission")
end
