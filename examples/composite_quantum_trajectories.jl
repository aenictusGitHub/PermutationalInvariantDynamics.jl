using LinearAlgebra
using Random
using PermutationalInvariantDynamics

# One compressed ensemble coupled to an explicitly truncated auxiliary qubit.
atoms=PIBasis(4,2)
auxiliary=FiniteOperatorBasis(2;label=:ancilla)
basis=CompositePIBasis(atoms,auxiliary)

sm=ComplexF64[0 1;0 0]
sx=ComplexF64[0 1;1 0]
excited=ComplexF64[0 0;0 1]

rho_atoms=iid_state(atoms,excited)
rho0=composite_tensor_state(basis,rho_atoms,excited)

# Unmonitored trace-preserving background physics. Monitored channels must not
# already be present here: CompositeTrajectoryPlan adds their dissipators.
Jx=collective_operator(atoms,sx)
background=composite_hamiltonian_superoperator(
    basis,1=>Jx,2=>sx;rate=0.08)

# One measured cross-factor channel J_- tensor sigma_-. Its gain, Q-left, and
# Q-right maps remain factorized throughout trajectory propagation.
Jm=collective_operator(atoms,sm)
loss=CompositeJumpChannel(
    basis,1=>Jm,2=>sm;rate=0.35,label=:joint_emission)
plan=CompositeTrajectoryPlan(background,loss)

times=collect(0.0:0.25:2.0)
dt=0.01
npaths=1024
batch_workspace=CompositeTrajectoryBatchWorkspace(
    plan,rho0;workers=Threads.nthreads())
serial_workspace=CompositeTrajectoryWorkspace(plan,rho0)

paths=quantum_trajectories(
    plan,rho0,times,npaths;dt,seed=2026,threaded=true,
    workspace=batch_workspace)
stochastic=trajectory_average(paths)

# The plan exposes the independently propagated unconditional generator.
master=composite_master_superoperator(plan)
deterministic=time_evolution(
    master,rho0,times;steps_per_interval=100)

final_error=norm(stochastic[end].data-deterministic[end].data)
@assert final_error<0.12
@assert all(state->isapprox(trace(state),1;atol=2e-11),stochastic)
@assert all(state->isapprox(trace(state),1;atol=2e-11),deterministic)

# State-free online statistics avoid storing another trajectory ensemble.
identity_atoms=identity_operator(atoms)
identity_aux=Matrix{ComplexF64}(I,2,2)
Jz=collective_operator(atoms,ComplexF64[1 0;0 -1]/2)
atom_signal=composite_tensor_operator(basis,Jz,identity_aux)
auxiliary_excitation=composite_tensor_operator(
    basis,identity_atoms,excited)
summary=quantum_trajectories(
    plan,rho0,times,512;dt,seed=17,threaded=true,
    workspace=batch_workspace,
    observables=(atom_z=atom_signal,auxiliary_excitation=auxiliary_excitation),
    save_states=false,jump_statistics=true)

# Global trajectory-index seeding makes serial and threaded scheduling sample
# the same ordered paths.
serial_check=quantum_trajectories(
    plan,rho0,[0.0,0.5],16;dt,seed=91,threaded=false,
    workspace=serial_workspace)
threaded_check=quantum_trajectories(
    plan,rho0,[0.0,0.5],16;dt,seed=91,threaded=true,
    workspace=batch_workspace)
@assert map(path->path.jump_times,serial_check)==
        map(path->path.jump_times,threaded_check)

println("Composite coordinate dimension: ",length(basis))
println("Full density-matrix coordinate formula: ",
        "(2^4 * 2)^2 = ",(2^4*2)^2)
println("Final stochastic/master coefficient error: ",final_error)
println("Mean jumps per trajectory: ",summary.jumps.mean_count)
println("Final <Jz>: ",summary.observables.observables[:atom_z].mean[end])
