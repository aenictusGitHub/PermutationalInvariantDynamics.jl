using Aqua
using JET
using LinearAlgebra
using PermutationalInvariantDynamics
using Test

const PID = PermutationalInvariantDynamics

@testset "Aqua package quality" begin
    Aqua.test_all(PID)
end

# JET's compiler integration is Julia-version specific, so this focused gate
# runs in the dedicated Julia 1.12 quality environment. Concrete public calls
# give JET substantially more useful type information than a whole-package
# scan of deliberately generic method signatures.
@testset "JET public hot paths" begin
    basis = PIBasis(3, 2)
    rho = iid_pure_state(basis, ComplexF64[1, 1] ./ sqrt(2))
    sx = ComplexF64[0 1; 1 0]
    sm = ComplexF64[0 1; 0 0]
    model = PIModel(basis, [
        LocalHamiltonian(sx; rate=0.2),
        LocalJump(sm; rate=0.1),
    ])
    prepared = compile(model; backend=:matrixfree)
    workspace = LiouvillianWorkspace(prepared)
    destination = similar(rho.data)
    trajectory_plan = TrajectoryPlan(model)
    trajectory_workspace = TrajectoryBatchWorkspace(
        trajectory_plan, rho; workers=1)
    weak_state = weak_pi_pseudoket(rho)
    weak_plan = WeakPITrajectoryPlan(model)
    weak_workspace = WeakPITrajectoryBatchWorkspace(
        weak_plan, weak_state; workers=1)

    JET.@test_call target_modules=(PID,) purity(rho)
    JET.@test_call target_modules=(PID,) collective_expectation(rho, sx)
    JET.@test_call target_modules=(PID,) apply!(destination, prepared,
                                                rho.data, 0.0, nothing,
                                                workspace)
    JET.@test_call target_modules=(PID,) trajectory_steady_state(
        trajectory_plan, rho;
        trajectories=2, settling_time=0.01, dt=0.01,
        workspace=trajectory_workspace, return_info=true)
    JET.@test_call target_modules=(PID,) weak_pi_trajectory_steady_state(
        weak_plan, weak_state;
        trajectories=2, settling_time=0.01, dt=0.01,
        workspace=weak_workspace, return_info=true)
end
