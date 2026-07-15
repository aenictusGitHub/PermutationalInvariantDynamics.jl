using PermutationalInvariantDynamics

basis = PIBasis(20, 2)
sx = ComplexF64[0 1; 1 0]
sm = ComplexF64[0 1; 0 0]
model = PIModel(basis, [LocalHamiltonian(0.5sx), LocalJump(sm; rate=0.1)])
rho0 = iid_pure_state(basis, ComplexF64[1, 0])

# Compile the reusable matrix-free kernels once. Prepared observable and
# reduction plans likewise amortize geometry work over all saved states.
prepared = compile(model; backend=:matrixfree)
excited_population = CollectiveObservablePlan(basis, ComplexF64[0 0; 0 1])
one_body = ReductionPlan(basis, 1)

solution = solve_dynamics(prepared, rho0, (0.0, 1.0);
                          saveat=0.25, steps_per_interval=32)
excited_fractions = [real(collective_expectation(rho, excited_population)) / basis.N
                     for rho in solution]
rho1 = reduced_state(solution[end], 1; plan=one_body)
report = diagnostics(solution[end])

println("PI dimension: ", pi_dimension(prepared),
        "; backend: ", diagnostics(prepared).backend)
println("excited fraction at t = ", solution.times, ": ", excited_fractions)
println("final one-body state:\n", Matrix(physical_block(rho1, rho1.basis.sectors[1])))
println("final trace error: ", report.trace_error,
        "; minimum eigenvalue: ", report.minimum_eigenvalue)

@assert diagnostics(rho0).valid
@assert report.valid
@assert abs(trace(rho1) - 1) < 1e-10
