using LinearAlgebra
using PermutationalInvariantDynamics

N = 4
period = 2.0
basis = PIBasis(N, 2)
sm = ComplexF64[0 1; 0 0]
rate = (t, p) -> 0.4 * (1 + 0.7cos(2pi * t / period))
model = PIModel(basis, [LocalJump(sm; rate=rate)])

# Time-dependent rates are evaluated inside reusable matrix-free kernels.  A
# constant sparse model is used only for this small analytical validation.
prepared = compile(model; backend=:matrixfree)
constant_model = PIModel(basis, [LocalJump(sm; rate=0.4)])
constant_prepared = compile(constant_model; backend=:sparse)
Laverage = Matrix(liouvillian(constant_prepared))
exact = exp(period * Laverage) # the cosine modulation integrates to zero

for steps in (40, 80, 160)
    approximation = floquet_propagator(prepared, period; steps)
    println("RK4 steps=$steps: one-period error = ",
            norm(approximation - exact))
end

# Production calculations use this reusable linear operator.  Applying it to
# a vector integrates one period and never constructs an n_PI-by-n_PI map.
period_map = floquet_map(prepared, period; steps=160)
map_workspace = FloquetWorkspace(period_map)
excited = iid_pure_state(basis, ComplexF64[0, 1])
mapped = similar(excited.data)
apply!(mapped, period_map, excited.data, map_workspace)

# Keep one explicit map only as an independent small-N oracle for the example.
F = floquet_propagator(prepared, period; steps=160)
@assert mapped ≈ F * excited.data atol=2e-12

# Selected slow multipliers and their residuals are obtained directly from
# period-map products.  The partial result deliberately does not claim a
# globally certified gap unless the complete operator dimension is spanned.
selected = selected_floquet_multipliers(
    period_map; nev=4, method=:arnoldi,
    krylovdim=length(basis), atol=1e-10, rtol=1e-9,
)
gap_report = floquet_gap(
    period_map; return_info=true, nev=4, method=:arnoldi,
    krylovdim=length(basis), atol=1e-10, rtol=1e-9,
)

# Solve (F-I)rho=0 with a physical-trace row and restarted GMRES.  The period
# map remains matrix free throughout the solve.
steady = floquet_steady_state(
    period_map; return_info=true, krylovdim=20, maxiter=400,
    atol=1e-11, rtol=1e-9,
)
rhoF = steady.state

# Local decay has the weak parity symmetry rho -> P rho P'.  The + Liouville
# charge contains the (+,+) and (-,-) Hilbert-charge coordinate blocks.  The
# constructor below certifies the selected union by exhaustive period-map
# probes before any reduced Krylov calculation is allowed.
parity = Diagonal(ComplexF64[1, -1])
positive = diagonal_symmetry_restriction(basis, parity; charge=1)
negative = diagonal_symmetry_restriction(basis, parity; charge=-1)
even_restriction = SymmetryCoordinateRestriction(
    basis,
    vcat(retained_indices(positive), retained_indices(negative));
    label=:even_parity_conjugation,
)
restricted_map = restricted_floquet_map(
    period_map, even_restriction; atol=2e-11, rtol=2e-10)
restricted_steady = floquet_steady_state(
    restricted_map; return_info=true,
    krylovdim=min(20, size(restricted_map, 1)), maxiter=400,
    atol=1e-11, rtol=1e-9,
)

trajectory = stroboscopic_evolution(excited, period_map, 4)
excited_population = CollectiveObservablePlan(
    basis, ComplexF64[0 0; 0 1])
populations = [real(collective_expectation(rho, excited_population)) / N
               for rho in trajectory]
report = diagnostics(rhoF)
dense_values = floquet_multipliers(F)
selected_error = maximum(
    minimum(abs(value - reference) for reference in dense_values)
    for value in selected.values)

println("prepared backend: ", diagnostics(prepared).backend)
println("Floquet operator: ", period_map)
println("selected multipliers: ", selected.values)
println("maximum selected Ritz residual: ", maximum(selected.residuals))
println("selected multiplier agreement with dense oracle: ", selected_error)
println("partial multiplier gap estimate: ", gap_report.gap,
        "; global certificate: ", gap_report.global_gap_certified)
println("matrix-free steady residual: ", steady.residual)
println("parity restriction: ", size(restricted_map, 1), " / ",
        size(period_map, 1), " coordinates; leakage residual = ",
        restricted_steady.leakage_residual)
println("excited fraction at periods 0:4: ", populations)
println("steady-state trace error: ", report.trace_error,
        "; minimum eigenvalue: ", report.minimum_eigenvalue)

ground = iid_pure_state(basis, ComplexF64[1, 0])
@assert norm(F - exact) < 2e-8
@assert all(selected.converged)
@assert selected_error < 2e-8
@assert gap_report.residual_certified
@assert gap_report.gap ≈ 0.2 atol=2e-8
@assert steady.residual < 2e-9
@assert rhoF.data ≈ ground.data atol=2e-9
@assert restricted_steady.state.data ≈ rhoF.data atol=2e-9
@assert restricted_steady.full_residual < 2e-9
@assert report.valid
