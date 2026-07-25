using LinearAlgebra
using PermutationalInvariantDynamics

include("paper_models.jl")
using .PaperModels
include(joinpath(@__DIR__, "utils", "makie_support.jl"))
using .ExampleMakie

# Y. Nakanishi and T. Sasamoto, Phys. Rev. A 107, L010201
# (2023), balanced one-spin PT model.  The example compares three distinct
# statements:
#
#   1. the finite-N product closure implemented by MeanFieldPlan;
#   2. the exact finite-N one-body formula from the paper; and
#   3. the leading thermodynamic closure, whose oscillation is undamped.

function main()
    N = 16
    g = 1.3
    kappa = 0.4
    model = balanced_gain_loss_time_crystal_model(N; g=g, kappa=kappa, p=0.0)

    finite = MeanFieldPlan(model; limit=:finite)
    thermodynamic = MeanFieldPlan(model; limit=:thermodynamic)

    # The first package basis vector is polarized along the observable below.
    sigma0 = ComplexF64[1 0; 0 0]
    sigma_z = ComplexF64[1 0; 0 -1]
    times = range(0.0, 6.0; length=25)

    finite_solution = solve_meanfield(
        finite, sigma0, (first(times), last(times));
        saveat=times, steps_per_interval=48)
    thermodynamic_solution = solve_meanfield(
        thermodynamic, sigma0, (first(times), last(times));
        saveat=times, steps_per_interval=48)

    finite_magnetization = [
        real(meanfield_expectation(sigma, sigma_z))
        for sigma in finite_solution
    ]
    thermodynamic_magnetization = [
        real(meanfield_expectation(sigma, sigma_z))
        for sigma in thermodynamic_solution
    ]

    exact_finite = exp.(-4kappa .* times ./ N) .* cos.(g .* times)
    exact_thermodynamic = cos.(g .* times)
    finite_error = maximum(abs.(finite_magnetization .- exact_finite))
    thermodynamic_error = maximum(
        abs.(thermodynamic_magnetization .- exact_thermodynamic))

    # The explicit workspace is the allocation-conscious route for repeated
    # right-hand-side evaluations, parameter scans, and user-written solvers.
    workspace = MeanFieldWorkspace(finite, sigma0)
    derivative = similar(sigma0)
    meanfield_rhs!(derivative, finite, sigma0, 0.0, nothing, workspace)
    trace_derivative = abs(tr(derivative))

    # A modest exact PI calculation supplies an independent comparison.  It
    # is matrix-free, but still evolves (N+1)^2 symmetric-sector coordinates;
    # each mean-field state above contains only d^2=4 entries.
    prepared = compile(model; backend=:matrixfree)
    rho0 = iid_pure_state(model.basis, ComplexF64[1, 0])
    pi_solution = solve_dynamics(
        prepared, rho0, (first(times), last(times));
        saveat=times, steps_per_interval=48)
    observable = CollectiveObservablePlan(model.basis, sigma_z)
    pi_magnetization = [
        real(collective_expectation(rho, observable)) / N
        for rho in pi_solution
    ]
    pi_formula_error = maximum(abs.(pi_magnetization .- exact_finite))
    pi_closure_error = maximum(
        abs.(pi_magnetization .- finite_magnetization))

    final_moments = meanfield_collective_moments(
        finite, last(finite_solution), sigma_z)

    println("Balanced gain/loss time crystal")
    println("finite product-closure error: ", finite_error)
    println("thermodynamic undamped-curve error: ", thermodynamic_error)
    println("exact finite-N PI formula error: ", pi_formula_error)
    println("PI versus finite product closure: ", pi_closure_error)
    println("trace of the prepared mean-field RHS: ", trace_derivative)
    println("final product-state collective moments: ", final_moments)
    println("sample finite/thermodynamic magnetizations: ",
            collect(zip(times[1:6:end],
                        finite_magnetization[1:6:end],
                        thermodynamic_magnetization[1:6:end])))

    @assert finite_error < 3e-8
    @assert thermodynamic_error < 3e-8
    @assert pi_formula_error < 3e-7
    @assert pi_closure_error < 3e-7
    @assert trace_derivative < 2e-13

    if makie_available()
        M = makie_module()
        error_floor = eps(Float64)
        finite_curve_error = abs.(finite_magnetization .- exact_finite)
        thermodynamic_curve_error = abs.(
            thermodynamic_magnetization .- exact_thermodynamic)
        pi_curve_error = abs.(pi_magnetization .- exact_finite)

        figure = M.Figure(size=(1180, 470), fontsize=17)
        magnetization_axis = M.Axis(
            figure[1, 1]; xlabel="time", ylabel="longitudinal magnetization",
            title="Finite and thermodynamic predictions")
        error_axis = M.Axis(
            figure[1, 2]; xlabel="time", ylabel="absolute curve error",
            yscale=log10, title="Analytical validation")

        M.lines!(magnetization_axis, times, exact_finite;
                 color=:black, linewidth=2.8, label="exact finite-N")
        M.scatter!(magnetization_axis, times, pi_magnetization;
                   color=:royalblue, markersize=7, label="exact PI, N=$N")
        M.lines!(magnetization_axis, times, finite_magnetization;
                 color=:darkorange, linewidth=2.1, linestyle=:dash,
                 label="finite product closure")
        M.lines!(magnetization_axis, times, exact_thermodynamic;
                 color=:seagreen, linewidth=2.5,
                 label="thermodynamic cos(gt)")
        M.scatter!(magnetization_axis, times, thermodynamic_magnetization;
                   color=:seagreen, marker=:diamond, markersize=6,
                   label="thermodynamic closure")
        M.hlines!(magnetization_axis, [0.0]; color=:gray75, linewidth=1)
        M.axislegend(magnetization_axis; position=:lb, labelsize=11)

        M.lines!(error_axis, times, max.(finite_curve_error, error_floor);
                 color=:darkorange, linewidth=2.3,
                 label="finite product vs formula")
        M.lines!(error_axis, times, max.(pi_curve_error, error_floor);
                 color=:royalblue, linewidth=2.3,
                 label="exact PI vs formula")
        M.lines!(error_axis, times,
                 max.(thermodynamic_curve_error, error_floor);
                 color=:seagreen, linewidth=2.3,
                 label="thermodynamic vs cos(gt)")
        M.axislegend(error_axis; position=:rt, labelsize=11)

        M.Label(
            figure[2, 1:2],
            "The undamped curve is a thermodynamic closure; both finite-N curves decay.";
            fontsize=14, color=:gray35, tellwidth=false)
        save_example_figure(figure, "meanfield_time_crystal")
    end
end

main()
