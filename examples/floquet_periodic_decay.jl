using LinearAlgebra
using PermutationalInvariantDynamics

N = 4
period = 2.0
basis = PIBasis(N, 2)
sm = ComplexF64[0 1; 0 0]
rate = (t, p) -> 0.4 * (1 + 0.7cos(2pi * t / period))
model = PIModel(basis, [LocalJump(sm; rate=rate)])

# Time-dependent models use reusable matrix-free kernels. The constant model
# is compiled sparsely only to construct the analytical reference channel.
prepared = compile(model; backend=:matrixfree)
constant_model = PIModel(basis, [LocalJump(sm; rate=0.4)])
constant_prepared = compile(constant_model; backend=:sparse)
Laverage = Matrix(liouvillian(constant_prepared))
exact = exp(period * Laverage) # cosine modulation integrates to zero over a period

for steps in (40, 80, 160)
    approximation = floquet_propagator(prepared, period; steps=steps)
    println("RK4 steps=$steps: one-period error = ", norm(approximation - exact))
end

F = floquet_propagator(prepared, period; steps=160)
steady = stationary_state(F - I; basis=basis,
                          algorithm=SVDAlgorithm(), return_info=true)
rhoF = steady.state
excited = iid_pure_state(basis, ComplexF64[0, 1])
trajectory = stroboscopic_evolution(excited, F, 4)
excited_population = CollectiveObservablePlan(basis, ComplexF64[0 0; 0 1])
populations = [real(collective_expectation(rho, excited_population)) / N
               for rho in trajectory]
report = diagnostics(rhoF)

println("prepared backend: ", diagnostics(prepared).backend)
println("leading Floquet multipliers: ", floquet_multipliers(F)[1:4])
println("Floquet gap: ", floquet_gap(F, period),
        "; steady-state residual: ", steady.info.residual)
println("excited fraction at periods 0:4: ", populations)
println("steady-state trace error: ", report.trace_error,
        "; minimum eigenvalue: ", report.minimum_eigenvalue)

@assert norm(F - exact) < 2e-8
@assert steady.info.residual < 1e-10
@assert report.valid
