using LinearAlgebra
using PermutationalInvariantDynamics

function main()
    N = 6
    spin = spin_matrices(2)

    # Every term preserves diagonality in the Dicke/GT basis. The diagonal
    # Hamiltonian changes coherence phases but has no effect on populations.
    model = qubit_ensemble_model(N;
        hamiltonian=0.23spin.jz,
        emission=0.35,
        dephasing=0.08,
        pumping=0.12,
        collective_emission=0.025,
        collective_dephasing=0.015,
        collective_pumping=0.010,
    )
    basis = model.basis

    plan = PopulationPlan(model)
    invariance = plan.invariance
    @assert invariance.invariant === true
    println("population-invariance report: ", invariance)
    println("population coordinates: ", population_dimension(basis),
            "; full PI coordinates: ", length(basis))

    # The central symmetric Dicke state is Schur diagonal, so it can initialize
    # both the reduced population solver and the general PI solver exactly.
    rho0 = dicke_state(basis, N / 2, 0)
    p0 = diagonal_populations(rho0)
    times = range(0.0, 2.0; length=9)

    population_solution = solve_populations(plan, p0, (0.0, 2.0);
        saveat=times, steps_per_interval=64)

    prepared = compile(model; backend=:sparse)
    full_solution = solve_dynamics(prepared, rho0, (0.0, 2.0);
        saveat=times, steps_per_interval=64)

    evolution_error = maximum(
        norm(population_solution[index] -
             diagonal_populations(full_solution[index]))
        for index in eachindex(times)
    )
    normalization_error = maximum(
        abs(sum(populations) - 1) for populations in population_solution)
    println("maximum population/full-PI discrepancy: ", evolution_error)
    println("maximum population normalization error: ", normalization_error)

    @assert evolution_error < 2e-10
    @assert normalization_error < 2e-10

    # The stationary problem uses the same certified reduced generator. Its
    # trace functional is simply the sum of physical populations.
    generator = population_generator(plan; representation=:sparse)
    stationary = stationary_populations(plan; method=:direct)
    stationary_residual = norm(generator * stationary)
    rho_stationary = state_from_populations(basis, stationary; validate=true)

    full_stationary = stationary_state(prepared;
        algorithm=DirectAlgorithm(), return_info=true)
    stationary_error = norm(rho_stationary.data - full_stationary.state.data)

    println("stationary population residual: ", stationary_residual)
    println("stationary population/full-PI discrepancy: ", stationary_error)
    println("stationary trace: ", real(sum(stationary)))

    @assert stationary_residual < 2e-9
    @assert abs(sum(stationary) - 1) < 2e-10
    @assert maximum(abs, imag.(stationary)) < 2e-10
    @assert minimum(real.(stationary)) > -2e-10
    @assert stationary_error < 2e-8
    @assert full_stationary.info.converged
    @assert diagnostics(rho_stationary).valid
end

main()
