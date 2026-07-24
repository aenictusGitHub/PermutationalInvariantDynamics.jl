using LinearAlgebra
using Random
using PermutationalInvariantDynamics

include(joinpath(@__DIR__, "utils", "makie_support.jl"))
using .ExampleMakie

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

if makie_available()
    M=makie_module()
    atom_statistics=summary.observables.observables[:atom_z]
    auxiliary_statistics=
        summary.observables.observables[:auxiliary_excitation]
    stochastic_atom=[
        real(expectation(state,atom_signal)) for state in stochastic]
    deterministic_atom=[
        real(expectation(state,atom_signal)) for state in deterministic]
    stochastic_auxiliary=[
        real(expectation(state,auxiliary_excitation)) for state in stochastic]
    deterministic_auxiliary=[
        real(expectation(state,auxiliary_excitation)) for state in deterministic]

    figure=M.Figure(size=(1120,440),fontsize=17)
    atom_axis=M.Axis(
        figure[1,1];xlabel="time",ylabel="⟨Jz⟩",
        title="Compressed ensemble")
    auxiliary_axis=M.Axis(
        figure[1,2];xlabel="time",ylabel="ancilla excitation",
        title="Finite auxiliary factor")

    for (axis,deterministic_values,stochastic_values,statistics) in (
        (atom_axis,deterministic_atom,stochastic_atom,atom_statistics),
        (auxiliary_axis,deterministic_auxiliary,stochastic_auxiliary,
         auxiliary_statistics),
    )
        M.band!(
            axis,times,statistics.lower,statistics.upper;
            color=(:dodgerblue3,0.20),
            label="512-path 95% normal interval")
        M.lines!(
            axis,times,deterministic_values;
            color=:black,linewidth=2.8,label="master equation")
        M.scatterlines!(
            axis,times,stochastic_values;
            color=:darkorange2,linewidth=1.5,markersize=6,
            label="1,024-path state average")
        M.lines!(
            axis,times,statistics.mean;
            color=:dodgerblue3,linewidth=2,linestyle=:dash,
            label="512-path online mean")
    end
    M.axislegend(atom_axis;position=:rb,labelsize=11)
    M.axislegend(auxiliary_axis;position=:rt,labelsize=11)
    save_example_figure(figure,"composite_quantum_trajectories")
end
