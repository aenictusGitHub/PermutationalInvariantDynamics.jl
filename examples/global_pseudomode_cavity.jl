using LinearAlgebra
using PermutationalInvariantDynamics

# Resonant Tavis--Cummings dynamics in a rotating frame. The oscillator is one
# cavity shared by the whole ensemble, not one replicated mode per emitter.
N = 3
nmax = N + 1
system_basis = PIBasis(N, 2; sectors=[(N, 0)])
spin = spin_matrices()

system_model = PIModel(system_basis, ())
mode = BosonicPseudomode(
    nmax;
    frequency=0.0,
    damping=0.35,
    thermal_occupation=0.0,
    label=:cavity,
)
coupling = PseudomodeCoupling(
    spin.jm;
    mode=:cavity,
    strength=0.55,
)
model = global_pseudomode_model(
    system_model, mode; couplings=coupling)

# All emitters excited and the shared cavity in its default vacuum.
system_initial = computational_product_state(system_basis, 2)
rho0 = pseudomode_product_state(model, system_initial)
@assert isapprox(trace(rho0), 1; atol=2e-14)

# Tensor-factor observables remain factorized. No composite Kronecker
# superoperator is materialized.
system_identity = identity_operator(system_basis)
mode_identity = model.mode_operators.identity
atomic_excitation = collective_operator(
    system_basis, spin.jp * spin.jm)

atom_number = composite_tensor_operator(
    model.basis, atomic_excitation, mode_identity)
cavity_number = composite_tensor_operator(
    model.basis, system_identity,
    model.mode_operators.number_operator)
cutoff_projector = composite_tensor_operator(
    model.basis, system_identity,
    model.mode_operators.top_projector)

times = collect(range(0.0, 4.0; length=41))
states = time_evolution(
    model.generator, rho0, times; steps_per_interval=8)

atom_population = [
    real(expectation(rho, atom_number)) for rho in states]
cavity_population = [
    real(expectation(rho, cavity_number)) for rho in states]
top_population = [
    real(expectation(rho, cutoff_projector)) for rho in states]
radiated_flux = model.metadata.damping .* cavity_population

# The specialized and generic factor reductions agree. They contract
# composite coordinates directly and do not reconstruct the 2^N system.
rho_system = trace_pseudomodes(last(states), model)
rho_cavity = global_pseudomode_state(last(states), model)
rho_cavity_generic = composite_reduced_state(last(states), 2)

@assert isapprox(rho_cavity, rho_cavity_generic; atol=2e-12, rtol=2e-12)
@assert isapprox(trace(rho_system), 1; atol=2e-10)
@assert isapprox(LinearAlgebra.tr(rho_cavity), 1; atol=2e-10)
@assert maximum(abs(real(trace(rho)) - 1) for rho in states) < 2e-10

# Under the rotating-wave interaction the initial state contains only N
# excitations, so level nmax=N+1 is unreachable. A nonzero value beyond
# roundoff would expose an implementation or time-stepping problem here.
@assert maximum(abs, top_population) < 2e-10

println("Shared-cavity coordinate dimension: ", length(model.basis))
println("PI system coordinates: ", length(system_basis))
println("Mode operator coordinates: ", mode.levels^2)
println("Initial/final atomic excitation: ",
        first(atom_population), " / ", last(atom_population))
println("Peak cavity population: ", maximum(cavity_population))
println("Peak radiated flux kappa*<n>: ", maximum(radiated_flux))
println("Maximum top-level population: ", maximum(abs, top_population))
println("Final reduced traces (system, cavity): ",
        real(trace(rho_system)), ", ", real(LinearAlgebra.tr(rho_cavity)))
