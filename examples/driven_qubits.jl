using PermutationalInvariantDynamics

basis = PIBasis(20, 2)
sx = ComplexF64[0 1; 1 0]
sm = ComplexF64[0 1; 0 0]
model = PIModel(basis, [LocalHamiltonian(0.5sx), LocalJump(sm; rate=0.1)])
rho0 = iid_pure_state(basis, ComplexF64[1, 0])

# Compile the reusable matrix-free kernels once. The collective observable
# and one-body marginal share one prepared one-box geometry.
prepared = compile(model; backend=:matrixfree)
one_body_geometry = OneBodyGeometry(basis)
excited_population = CollectiveObservablePlan(
    basis, ComplexF64[0 0; 0 1]; cache=one_body_geometry)
one_body_workspace = OneBodyRDMWorkspace(one_body_geometry, rho0)
rho1 = zeros(ComplexF64, basis.d, basis.d)

solution = solve_dynamics(prepared, rho0, (0.0, 1.0);
                          saveat=0.25, steps_per_interval=32)
excited_fractions = [real(collective_expectation(rho, excited_population)) / basis.N
                     for rho in solution]
one_body_rdm!(rho1, solution[end], one_body_workspace)
report = diagnostics(solution[end])

println("PI dimension: ", pi_dimension(prepared),
        "; backend: ", diagnostics(prepared).backend)
println("excited fraction at t = ", solution.times, ": ", excited_fractions)
println("final one-body state:\n", rho1)
println("final trace error: ", report.trace_error,
        "; minimum eigenvalue: ", report.minimum_eigenvalue)

@assert diagnostics(rho0).valid
@assert report.valid
@assert abs(sum(rho1[index, index] for index in axes(rho1, 1)) - 1) < 1e-10
