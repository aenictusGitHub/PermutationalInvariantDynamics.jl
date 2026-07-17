using PermutationalInvariantDynamics

# 1. Complete PI basis for N identical qubits.
N = 8
basis = PIBasis(N, 2)

# 2. Local basis: |g> = [1,0], |e> = [0,1].
spin = spin_matrices()
sm = spin.jm
sp = spin.jp
number = sp * sm

# 3. Declarative physical terms and model.
model = PIModel(basis, (
    LocalHamiltonian(spin.jx; rate=0.7),
    LocalJump(sm; rate=0.12),
    LocalJump(sp; rate=0.02),
))

# 4. Fully excited identical product state.
rho0 = computational_product_state(basis, 2)
validate_state(rho0)

# 5. Lower Schur geometry and Liouvillian kernels once.
prepared = compile(model; backend=:auto)

# 6. Evolve and sample the collective excitation.
times = collect(0.0:0.1:4.0)
solution = solve_dynamics(
    prepared, rho0, (first(times), last(times));
    saveat=times,
    steps_per_interval=16,
    observables=(excited=number,),
    save_states=true,
)
excited_fraction = real.(solution.observables[:excited]) ./ N

# 7. Solve the autonomous stationary problem and retain its residual metadata.
steady = stationary_state(prepared; return_info=true)

# 8. Compare two time-step resolutions for the reported observable.
coarse = solve_dynamics(
    prepared, rho0, (first(times), last(times));
    saveat=times,
    steps_per_interval=8,
    observables=(excited=number,),
    save_states=false,
)
step_error = maximum(abs.(
    coarse.observables[:excited] .- solution.observables[:excited])) / N

@assert diagnostics(solution[end]).valid
@assert steady.info.converged
@assert diagnostics(steady.state).valid
@assert step_error < 1e-6

println("PI coordinates: ", pi_dimension(basis))
println("selected backend: ", diagnostics(prepared).backend)
println("final excitation fraction: ", last(excited_fraction))
println("stationary residual: ", steady.info.residual)
println("8-versus-16-step excitation difference: ", step_error)
