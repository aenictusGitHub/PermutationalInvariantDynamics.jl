# Representative compilation workload for an optional PackageCompiler
# sysimage. It intentionally uses small PI objects and never writes output.
using PermutationalInvariantDynamics
using LinearAlgebra
using Random

let
    basis=PIBasis(4,2)
    spin=spin_matrices()
    model=PIModel(basis,(
        LocalHamiltonian(spin.jx;rate=0.2),
        LocalJump(spin.jm;rate=0.3),
        CollectiveJump(spin.jm;rate=0.05),
    ))
    rho0=computational_product_state(basis,2)

    compiled=compile(model;backend=:matrixfree,memory_budget=Inf)
    work=LiouvillianWorkspace(compiled)
    output=similar(rho0.data)
    apply!(output,compiled,rho0.data,0.0,nothing,work)
    apply_adjoint!(output,compiled,rho0.data,0.0,nothing,work)

    solve_dynamics(
        compiled,rho0,(0.0,0.01);
        saveat=[0.0,0.01],steps_per_interval=1,memory_budget=Inf)
    stationary_state(
        Models.local_pump_decay(2);
        algorithm=DirectAlgorithm(),memory_budget=Inf)

    observable=CollectiveObservablePlan(basis,spin.jz)
    collective_expectation(rho0,observable)
    one_body_rdm(rho0)
    reduction=ReductionPlan(basis,2)
    reduced_purity(rho0,2;plan=reduction)

    trajectory=TrajectoryPlan(model)
    trajectory_work=TrajectoryWorkspace(trajectory,rho0;mode=:fixed)
    quantum_trajectory(
        trajectory,rho0,[0.0,0.01];
        dt=0.01,rng=MersenneTwister(1),workspace=trajectory_work)
end
