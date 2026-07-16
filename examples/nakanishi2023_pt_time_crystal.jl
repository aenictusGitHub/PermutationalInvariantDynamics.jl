using LinearAlgebra
using PermutationalInvariantDynamics
include("paper_models.jl")
using .PaperModels
include(joinpath(@__DIR__, "utils", "makie_support.jl"))
using .ExampleMakie

# Y. Nakanishi and T. Sasamoto, PRA 107, L010201 (2023),
# balanced one-spin PT model, Eqs. (13)-(14).

# Match two spectra as multisets.  A bipartite augmenting-path match is robust
# to arbitrary eigensolver ordering and to the exact degeneracies in Eq. (14).
function spectrum_multiset_error(numerical, exact; atol=1e-11, rtol=1e-10)
    length(numerical) == length(exact) ||
        throw(DimensionMismatch("the numerical and exact spectra have different sizes"))
    owner = zeros(Int, length(exact))

    function augment(i, seen)
        # Trying the closest candidates first is not required for correctness,
        # but makes the matching deterministic when several values coincide.
        for j in sortperm(abs.(exact .- numerical[i]))
            tolerance = atol + rtol * max(abs(numerical[i]), abs(exact[j]), 1.0)
            (seen[j] || abs(numerical[i] - exact[j]) > tolerance) && continue
            seen[j] = true
            if owner[j] == 0 || augment(owner[j], seen)
                owner[j] = i
                return true
            end
        end
        false
    end

    for i in eachindex(numerical)
        augment(i, falses(length(exact))) ||
            error("the numerical spectrum does not match Eq. (14) within tolerance")
    end
    maximum(abs(numerical[owner[j]] - exact[j]) for j in eachindex(exact))
end

function main()
    # Complete-spectrum validation is deliberately kept small and dense.
    N = 6
    g = 1.3
    kappa = 0.4
    model = nakanishi2023_pt_model(N; g=g, kappa=kappa, p=0.0)
    prepared = compile(model; backend=:sparse)
    numerical = liouvillian_spectrum(
        prepared; target=:largest_real, nev=pi_dimension(prepared),
        algorithm=:dense)
    exact = nakanishi2023_pt_spectrum(N; g=g, kappa=kappa)
    spectrum_error = spectrum_multiset_error(numerical, exact)

    gap = pi_liouvillian_gap(prepared; method=:dense)
    exact_gap = 4kappa / N

    # Balanced gain and loss make the maximally mixed state in the spin-S
    # irrep stationary.  The restricted sector has multiplicity one.
    steady = stationary_state(
        prepared; algorithm=DirectAlgorithm(), return_info=true)
    symmetric_sector = only(model.basis.sectors)
    uniform = sector_density_matrix(
        model.basis, symmetric_sector,
        Matrix{ComplexF64}(I, N + 1, N + 1) / (N + 1))
    steady_error = norm(steady.state.data - uniform.data)

    println("N=$N complete-spectrum maximum matching error: $spectrum_error")
    println("Liouvillian gap: $gap (Eq. (14): $exact_gap)")
    println("uniform symmetric-sector steady-state error: $steady_error")

    @assert spectrum_error < 2e-10
    @assert isapprox(gap, exact_gap; atol=2e-10, rtol=2e-10)
    @assert steady.info.converged
    @assert steady_error < 2e-10

    # A larger finite-size propagation uses only matrix-free kernels.  The
    # q=+-1, l=0 modes of Eq. (14) give this damped collective oscillation.
    Ndyn = 24
    dynamic_model = nakanishi2023_pt_model(Ndyn; g=g, kappa=kappa, p=0.0)
    dynamic = compile(dynamic_model; backend=:matrixfree)
    basis = dynamic_model.basis
    rho0 = iid_pure_state(basis, ComplexF64[1, 0])
    sz = ComplexF64[1 0; 0 -1] / 2
    Sz = CollectiveObservablePlan(basis, sz)
    times = range(0.0, 6.0; length=13)
    solution = solve_dynamics(
        dynamic, rho0, (first(times), last(times));
        saveat=times, steps_per_interval=32)
    magnetization = [real(collective_expectation(rho, Sz)) / (Ndyn / 2)
                     for rho in solution]
    exact_magnetization = exp.(-4kappa .* times ./ Ndyn) .* cos.(g .* times)
    dynamics_error = maximum(abs.(magnetization .- exact_magnetization))

    println("N=$Ndyn matrix-free max |<Sz>/S - exact|: $dynamics_error")
    println("sample <Sz>/S values: ", magnetization[1:3:end])

    @assert dynamics_error < 2e-7
    @assert diagnostics(last(solution); atol=2e-9, rtol=2e-9).valid

    if makie_available()
        M = makie_module()
        figure = M.Figure(size=(1150, 470), fontsize=17)
        spectrum_axis = M.Axis(
            figure[1, 1]; xlabel="Re(λ)", ylabel="Im(λ)",
            title="Balanced finite-N spectrum, N=$N")
        dynamics_axis = M.Axis(
            figure[1, 2]; xlabel="time", ylabel="⟨Sz⟩ / S",
            title="Damped collective oscillation, N=$Ndyn")

        M.vlines!(spectrum_axis, [0.0]; color=:gray65, linestyle=:dash)
        M.scatter!(spectrum_axis, real.(exact), imag.(exact);
                   marker=:circle, color=(:white, 0.0),
                   strokecolor=:firebrick, strokewidth=1.7,
                   markersize=11, label="Eq. (14)")
        M.scatter!(spectrum_axis, real.(numerical), imag.(numerical);
                   marker=:cross, color=:black, markersize=8,
                   label="PI diagonalization")
        M.axislegend(spectrum_axis; position=:lt, labelsize=12)

        M.lines!(dynamics_axis, times, exact_magnetization;
                 color=:black, linewidth=2.7,
                 label="exp(-4κt/N) cos(gt)")
        M.scatter!(dynamics_axis, times, magnetization;
                   color=:royalblue, markersize=9,
                   label="matrix-free PI dynamics")
        M.hlines!(dynamics_axis, [0.0]; color=:gray70, linewidth=1)
        M.axislegend(dynamics_axis; position=:rt, labelsize=12)

        save_example_figure(figure, "nakanishi2023_pt_time_crystal")
    end
end

main()
