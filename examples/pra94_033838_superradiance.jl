using LinearAlgebra
using PermutationalInvariantDynamics

include("paper_models.jl")
using .PaperModels
include(joinpath(@__DIR__, "utils", "makie_support.jl"))
using .ExampleMakie

# Reproduce Fig. 6 and its analytical Eqs. (41)--(43) for two atoms.
small_times = range(0.0, 3.0; length=121)
gamma0 = 1.0
small_curves = NamedTuple[]
for gamma in (gamma0, 0.75gamma0, 0.0)
    model = correlated_superradiance_model(2; gamma0, gamma)
    basis = model.basis
    rho0 = iid_pure_state(basis, ComplexF64[0, 1])
    prepared = compile(model; backend=:sparse)

    # Dense exponentiation is intentional here: it is the independent
    # small-N validation route used for the pointwise analytical comparison.
    L = Matrix(liouvillian(prepared))
    intensity_operator = correlated_superradiance_intensity_operator(basis; gamma0, gamma)
    numeric = [real(expectation(
        PIState(basis, exp(t * L) * rho0.data), intensity_operator))
        for t in small_times]
    exact = [two_qubit_correlated_superradiance_intensity_exact(t; gamma0, gamma)
             for t in small_times]
    error = maximum(abs.(numeric - exact))
    @assert error < 2e-11
    push!(small_curves, (; gamma, numeric, exact, error))
    println("N=2, gamma/gamma0=$(gamma / gamma0): " *
            "max |numeric - Eqs. (41)--(43)| = $error")
end

# Compute the altered-superradiance pulse at the largest system size studied
# in the paper's Figs. 7--10 (and the fixed size of Fig. 8).  The fully excited
# state remains diagonal in the Schur/GT basis, so the certified population
# backend evolves only 256 physical probabilities instead of all 5,456 PI
# operator coordinates.
N = 30
delta_gamma = 0.4gamma0
gamma = gamma0 - delta_gamma
model = correlated_superradiance_model(N; gamma0, gamma)
basis = model.basis
rho0 = iid_pure_state(basis, ComplexF64[0, 1])
population_plan = PopulationPlan(model)
@assert population_plan.invariance.invariant === true

large_times = range(0.0, 1.0; length=201)
population_solution = solve_populations(
    population_plan, rho0, (first(large_times), last(large_times));
    saveat=large_times, steps_per_interval=16)

# In the population coordinates p_(nu,W)=f^nu (rho_nu)_(W,W), the expectation
# of a Schur-diagonal observable is a single dot product with the diagonal of
# each physical Schur block.  Prepare these weights once for the whole pulse.
intensity_operator = correlated_superradiance_intensity_operator(basis; gamma0, gamma)
intensity_diagonal = reduce(vcat,
    (diag(block) for (_, block) in each_schur_block(intensity_operator)))
@assert maximum(abs, imag.(intensity_diagonal)) < 2e-12
intensity_weights = real.(intensity_diagonal)
intensity = [real(dot(intensity_weights, populations))
             for populations in population_solution]
normalized_intensity = intensity ./ (N * gamma0)
pulse = (times=collect(large_times), intensity, normalized_intensity)

# Convergence-check the complete pulse after halving the RK4 step size.  The
# immutable population plan and its prepared sparse kernels are reused.
refined_solution = solve_populations(
    population_plan, rho0, (first(large_times), last(large_times));
    saveat=large_times, steps_per_interval=32)
refined_intensity = [real(dot(intensity_weights, populations))
                     for populations in refined_solution]
integration_error = maximum(abs.(intensity - refined_intensity))
@assert integration_error < 1e-8

peak_index = argmax(intensity)
peak_time = large_times[peak_index]
peak_intensity = intensity[peak_index]
peak_height = peak_intensity - N * gamma0
rho_peak = state(population_solution, peak_index)

normalization_error = maximum(
    abs(sum(populations) - 1) for populations in population_solution)
direct_intensity_error = abs(
    real(expectation(rho_peak, intensity_operator)) - peak_intensity)
@assert population_dimension(basis) == 256
@assert length(basis) == 5456
@assert abs(first(intensity) - N * gamma0) < 2e-11
@assert normalization_error < 2e-11
@assert direct_intensity_error < 2e-11
@assert diagnostics(rho_peak).valid

# Visualize rho at the pulse maximum in terms of its physical Schur-sector
# populations.  A partially correlated decay is used deliberately: the local
# component transfers weight out of the fully symmetric (N,0) sector.
peak_structure = schur_block_structure(
    rho_peak; metric=:population, threshold=1e-13)
peak_figure = visualize_schur_blocks(
    peak_structure;
    title="Damanet 2016: N=$N, Δγ/γ₀=$(delta_gamma / gamma0), " *
          "peak γ₀t=$(gamma0 * peak_time)",
    scale=:log, normalize=:global, show_values=false,
    show_young_diagrams=true, width=1200, height=1000)
peak_sector_populations = diag(peak_structure.weights)

@assert isapprox(sum(peak_sector_populations), 1; atol=2e-11)
@assert count(>(1e-13), peak_sector_populations) == length(basis.sectors)

# Notebook/VS Code displays render the dependency-free SVG directly.  Also
# exercise the file writer in a temporary directory so the example leaves no
# generated artifact in the repository.
display(peak_figure)
mktempdir() do directory
    path = joinpath(directory, "correlated_superradiance_N30_peak_irrep_blocks.svg")
    @assert save_schur_block_visualization(path, peak_figure) == path
    @assert occursin("<svg", read(path, String))
end

println("N=$N, Delta gamma/gamma0=$(delta_gamma / gamma0):")
println("  population coordinates / PI coordinates = ",
        population_dimension(basis), " / ", length(basis))
println("  pulse maximum I/(N gamma0) = ",
        peak_intensity / (N * gamma0), " at gamma0*t = ", gamma0 * peak_time)
println("  pulse height [Imax-I(0)]/(N gamma0) = ",
        peak_height / (N * gamma0))
println("  population normalization error = ", normalization_error)
println("  maximum 16-versus-32-substep intensity difference = ",
        integration_error)
println("  direct PI expectation error at the maximum = ", direct_intensity_error)
println("  peak Schur-sector populations = ", peak_sector_populations)

if makie_available()
    M = makie_module()
    figure = M.Figure(size=(1450, 450), fontsize=17)
    small_axis = M.Axis(
        figure[1, 1]; xlabel="γ₀t", ylabel="I / γ₀",
        title="Two-atom analytical benchmark")
    pulse_axis = M.Axis(
        figure[1, 2]; xlabel="γ₀t", ylabel="I / (Nγ₀)",
        title="Altered-superradiance pulse, N=30")
    sector_axis = M.Axis(
        figure[1, 3]; xlabel="total spin j", ylabel="sector population",
        yscale=log10, title="Peak-state Schur populations")

    colors = (:firebrick, :royalblue, :seagreen)
    for (curve, color) in zip(small_curves, colors)
        label = "γ/γ₀ = $(curve.gamma / gamma0)"
        M.lines!(small_axis, small_times, curve.exact;
                 color, linewidth=2.7, label)
        marker_indices = 1:8:length(small_times)
        M.scatter!(small_axis, small_times[marker_indices],
                   curve.numeric[marker_indices];
                   color, markersize=5)
    end
    M.axislegend(small_axis; position=:rt, labelsize=12)

    M.lines!(pulse_axis, pulse.times, pulse.normalized_intensity;
             color=:black, linewidth=2.7, label="population dynamics")
    M.hlines!(pulse_axis, [1.0]; color=:gray50, linestyle=:dash,
              label="independent-emitter level")
    M.scatter!(pulse_axis, [peak_time], [peak_intensity/(N*gamma0)];
               color=:firebrick, markersize=12, label="pulse maximum")
    M.axislegend(pulse_axis; position=:rt, labelsize=12)

    sector_spins = [
        (partition.parts[1]-partition.parts[2])/2 for partition in basis.sectors]
    M.barplot!(sector_axis, sector_spins, peak_sector_populations;
               color=:mediumpurple, strokecolor=:black, strokewidth=0.5)
    save_example_figure(figure, "pra94_033838_superradiance")
end
