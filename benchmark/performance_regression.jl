using PermutationalInvariantDynamics
using LinearAlgebra
using Random

# Stable regression gates deliberately avoid wall-clock thresholds. They guard
# allocation behavior, backend equivalence, and shared-operator correctness;
# benchmark/performance_audit.jl remains the human-readable timing report.
b=PIBasis(6,2)
sm=ComplexF64[0 1;0 0]
sx=ComplexF64[0 1;1 0]
model=PIModel(b,[LocalJump(sm),CollectiveHamiltonian(sx;rate=0.2)])
sparse_model=compile(model;backend=:sparse)
matrixfree=compile(model;backend=:matrixfree)
work=LiouvillianWorkspace(matrixfree)
x=iid_pure_state(b,ComplexF64[0,1]).data
y=similar(x);reference=similar(x)

mul!(reference,sparse_model,x)
apply!(y,matrixfree,x,0.0,nothing,work)
@assert isapprox(y,reference;atol=1e-11,rtol=1e-11)

apply!(y,matrixfree,x,0.0,nothing,work)
allocated=@allocated apply!(y,matrixfree,x,0.0,nothing,work)
@assert allocated<=2048 "explicit-workspace Liouvillian apply allocated $allocated bytes"

X=hcat(x,0.3x,complex.(reverse(x)))
Y=similar(X);apply!(Y,matrixfree.plan,X,0.0,nothing,work)
@assert isapprox(Y,sparse_model.operator*X;atol=1e-11,rtol=1e-11)
batch_alloc=@allocated apply!(Y,matrixfree.plan,X,0.0,nothing,work)
@assert batch_alloc<=2048 "explicit-workspace batched Liouvillian apply allocated $batch_alloc bytes"
apply_adjoint!(Y,matrixfree.plan,X,0.0,nothing,work)
batch_adjoint_alloc=@allocated apply_adjoint!(Y,matrixfree.plan,X,0.0,nothing,work)
@assert isapprox(Y,adjoint(sparse_model.operator)*X;atol=1e-11,rtol=1e-11)
@assert batch_adjoint_alloc<=2048 "explicit-workspace batched adjoint apply allocated $batch_adjoint_alloc bytes"

if Threads.nthreads()>1
    rng=MersenneTwister(71)
    inputs=[randn(rng,ComplexF64,length(b)) for _ in 1:8Threads.nthreads()]
    outputs=[similar(x) for _ in inputs]
    Threads.@threads for i in eachindex(inputs)
        mul!(outputs[i],matrixfree,inputs[i])
    end
    @assert all(isapprox(outputs[i],sparse_model.operator*inputs[i];
                         atol=2e-11,rtol=2e-11) for i in eachindex(inputs))
end

trajectory_state=PIState(b,copy(x))
trajectory_plan=TrajectoryPlan(model)
trajectory_batch=TrajectoryBatchWorkspace(trajectory_plan,trajectory_state;workers=1)
quantum_trajectories(trajectory_plan,trajectory_state,[0.0,0.05],16;
    dt=0.01,seed=81,workspace=trajectory_batch)
trajectory_batch_alloc=@allocated quantum_trajectories(
    trajectory_plan,trajectory_state,[0.0,0.05],16;
    dt=0.01,seed=81,workspace=trajectory_batch)
@assert trajectory_batch_alloc<=256*1024 "reused trajectory batch allocated $trajectory_batch_alloc bytes"
@assert trajectory_batch.workers[1].plan===trajectory_plan

population_model=qubit_ensemble_model(b;
    hamiltonian=spin_matrices().jz,
    emission=0.4,dephasing=0.1,pumping=0.07,
    collective_emission=0.02)
population_plan=PopulationPlan(population_model)
@assert population_plan.invariance.reason===:certified
@assert !(:coordinate_map in fieldnames(typeof(population_plan)))
population_source=diagonal_populations(iid_pure_state(b,ComplexF64[0,1]))
population_output=similar(population_source)
population_work=PopulationWorkspace(population_plan,population_source)
apply!(population_output,population_plan,population_source,
       0.0,nothing,population_work)
population_apply_alloc=@allocated apply!(
    population_output,population_plan,population_source,
    0.0,nothing,population_work)
@assert population_apply_alloc<=2048 "explicit-workspace population apply allocated $population_apply_alloc bytes"
evolve_populations!(population_output,population_plan,population_source,
    (0.0,0.01);steps=2,workspace=population_work)
population_evolve_alloc=@allocated evolve_populations!(
    population_output,population_plan,population_source,
    (0.0,0.01);steps=2,workspace=population_work)
@assert population_evolve_alloc<=2048 "explicit-workspace population evolution allocated $population_evolve_alloc bytes"

geometry=OneBodyGeometry(b)
observable=CollectiveObservablePlan(b,sx;cache=geometry)
rho=PIState(b,copy(x))
collective_moments(rho,observable)
planned=@allocated collective_moments(rho,observable)
@assert planned<=64*1024 "prepared collective moments allocated $planned bytes"

reduction=ReductionPlan(b,2)
planned_reduced=reduced_state(rho,2;plan=reduction)
reduction_alloc=@allocated reduced_state(rho,2;plan=reduction)
@assert reduction_alloc<=2*1024^2 "prepared reduction allocated $reduction_alloc bytes"
reduction_work=ReductionWorkspace(reduction,rho)
reduction_out=PIState(reduction.output_basis)
reduced_state!(reduction_out,rho,reduction,reduction_work)
reduction_inplace_alloc=@allocated reduced_state!(
    reduction_out,rho,reduction,reduction_work)
@assert reduction_out.data≈planned_reduced.data atol=2e-11
# Full state validation deliberately retains LAPACK scratch.  This gate guards
# the optimal caller-owned contraction/output path without weakening it.
@assert reduction_inplace_alloc<=64*1024 "in-place prepared reduction allocated $reduction_inplace_alloc bytes"

sigma=ComplexF64[0.62 0.08-0.03im;0.08+0.03im 0.38]
meanfield=MeanFieldPlan(10^6,2,[CollectiveJump(sm;rate=0.4/10^6),
    PBodyHamiltonian(kron(sx,sx),2;rate=0.2/10^6)])
meanfield_work=MeanFieldWorkspace(meanfield,sigma);meanfield_out=similar(sigma)
meanfield_rhs!(meanfield_out,meanfield,sigma,0.0,nothing,meanfield_work)
meanfield_alloc=@allocated meanfield_rhs!(meanfield_out,meanfield,sigma,0.0,nothing,meanfield_work)
@assert meanfield_alloc<=256 "explicit-workspace mean-field RHS allocated $meanfield_alloc bytes"
@assert abs(tr(meanfield_out))<=1e-12 "mean-field RHS did not preserve trace"

println("Performance regression gates passed (threads=$(Threads.nthreads()), apply_alloc=$allocated, batch_alloc=$batch_alloc, batch_adjoint_alloc=$batch_adjoint_alloc, trajectory_batch_alloc=$trajectory_batch_alloc, population_apply_alloc=$population_apply_alloc, population_evolve_alloc=$population_evolve_alloc, observable_alloc=$planned, reduction_alloc=$reduction_alloc, reduction_inplace_alloc=$reduction_inplace_alloc, meanfield_alloc=$meanfield_alloc)")
