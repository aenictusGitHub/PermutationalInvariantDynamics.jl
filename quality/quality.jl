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
    symmetric_basis = PIBasis(3, 2; sectors=[(3, 0)])
    symmetric_state = symmetric_occupation_ket(symmetric_basis, (2, 1))
    symmetric_plan = SymmetricKetHamiltonianPlan(
        symmetric_basis, Diagonal(ComplexF64[0.5, -0.5]))
    symmetric_destination = similar(symmetric_state.data)
    symmetric_density = symmetric_ket_density(symmetric_state)
    entropy_plan = HilbertBlockEntropyPlan(
        symmetric_basis, Diagonal(ComplexF64[1, -1]))
    entropy_workspace = HilbertBlockEntropyWorkspace(entropy_plan, Float64)
    moment_workspace = DensityPowerWorkspace(rho)
    reduction_plan = ReductionPlan(basis, 1)
    reduction_workspace = ReductionWorkspace(
        reduction_plan, rho; mode=:reduction)
    reduced_moment_workspace = DensityPowerWorkspace(reduction_workspace)

    JET.@test_call target_modules=(PID,) purity(rho)
    JET.@test_call target_modules=(PID,) trace_power(
        rho, 3; workspace=moment_workspace, check=false,
        memory_budget=Inf)
    JET.@test_call target_modules=(PID,) reduced_trace_power(
        rho, 1, 3; plan=reduction_plan, workspace=reduction_workspace,
        power_workspace=reduced_moment_workspace, check=false,
        memory_budget=Inf)
    JET.@test_call target_modules=(PID,) reduced_trace_powers(
        rho, 3; ks=0:2, check=false, memory_budget=Inf)
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
    JET.@test_call target_modules=(PID,) apply_symmetric_hamiltonian!(
        symmetric_destination, symmetric_plan, symmetric_state.data,
        0.0, nothing)
    JET.@test_call target_modules=(PID,) block_von_neumann_entropy(
        symmetric_density, entropy_plan; workspace=entropy_workspace)
end
