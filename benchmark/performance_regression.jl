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

z=ComplexF64[1 0;0 -1]
restricted_model=PIModel(b,(
    LocalHamiltonian(z;rate=0.2),LocalJump(z;rate=0.05)))
restricted_source=compile(restricted_model;backend=:matrixfree)
restricted_selection=diagonal_symmetry_restriction(
    b,Diagonal(ComplexF64[1,-1]);charge=1)
restricted=RestrictedLiouvillian(restricted_source,restricted_selection)
@assert restricted.backend===:lowered
restricted_work=RestrictedLiouvillianWorkspace(restricted)
@assert !hasproperty(restricted_work,:ambient_input)
restricted_input=randn(MersenneTwister(70),ComplexF64,
    length(restricted_selection))
restricted_output=similar(restricted_input)
apply!(restricted_output,restricted,restricted_input,restricted_work)
restricted_alloc=@allocated apply!(
    restricted_output,restricted,restricted_input,restricted_work)
restricted_indices=retained_indices(restricted_selection)
restricted_reference=liouvillian(
    restricted_model;representation=:sparse)[restricted_indices,restricted_indices]
@assert isapprox(restricted_output,restricted_reference*restricted_input;
                 atol=1e-11,rtol=1e-11)
@assert restricted_alloc<=2048 "lowered restricted apply allocated $restricted_alloc bytes"
restricted_response=ResponseWorkspace(restricted;krylovdim=8,mode=:linear)
@assert !hasproperty(restricted_response.action_workspace,:ambient_input)

threaded_alloc=0
if Threads.nthreads()>1
    rng=MersenneTwister(71)
    inputs=[randn(rng,ComplexF64,length(b)) for _ in 1:8Threads.nthreads()]
    outputs=[similar(x) for _ in inputs]
    Threads.@threads for i in eachindex(inputs)
        mul!(outputs[i],matrixfree,inputs[i])
    end
    @assert all(isapprox(outputs[i],sparse_model.operator*inputs[i];
                         atol=2e-11,rtol=2e-11) for i in eachindex(inputs))

    # One target task owns every coordinate of each assigned Schur sector.
    # Repeated forward and adjoint actions must therefore be bit-identical to
    # the serial prepared plan, independent of scheduling, without growing
    # caller-owned block scratch after warm-up.
    threaded_work=ThreadedLiouvillianWorkspace(
        matrixfree.plan;tasks=Threads.nthreads())
    threaded_output=similar(x)
    serial_output=similar(x)
    apply!(serial_output,matrixfree.plan,x,0.0,nothing,work)
    threaded_apply!(threaded_output,matrixfree.plan,x,threaded_work)
    @assert threaded_output==serial_output
    repeated_threaded=copy(threaded_output)
    threaded_apply!(threaded_output,matrixfree.plan,x,threaded_work)
    @assert threaded_output==repeated_threaded
    threaded_alloc=@allocated threaded_apply!(
        threaded_output,matrixfree.plan,x,threaded_work)
    @assert threaded_alloc<=512*1024 "target-sector threaded apply allocated $threaded_alloc bytes"
    apply_adjoint!(serial_output,matrixfree.plan,x,0.0,nothing,work)
    threaded_apply_adjoint!(threaded_output,matrixfree.plan,x,threaded_work)
    @assert threaded_output==serial_output

    coefficient_basis=PIBasis(4,3)
    coefficient_cache=OneBoxCGCache(
        coefficient_basis;max_depth=2,T=Float64)
    coefficient_queries=Tuple[]
    for ((lower,upper),table) in coefficient_cache.transitions
        for lower_index in eachindex(coefficient_cache.patterns[lower])
            for term_index in (table.offsets[lower_index]:
                    (table.offsets[lower_index+1]-1))
                upper_index,label,_=table.terms[term_index]
                push!(coefficient_queries,(
                    coefficient_cache.patterns[lower][lower_index],label-1,
                    coefficient_cache.patterns[upper][upper_index]))
            end
        end
    end
    coefficient_signature=(length(coefficient_cache.transitions),
        coefficient_cache.candidate_count,
        coefficient_cache.coefficient_count,
        Base.summarysize(coefficient_cache))
    coefficient_references=[cgc(mu,label,parent;T=Float64)
        for (mu,label,parent) in coefficient_queries]
    repeated_indices=repeat(eachindex(coefficient_queries);
        outer=max(2,Threads.nthreads()))
    coefficient_outputs=zeros(Float64,length(repeated_indices))
    Threads.@threads for index in eachindex(repeated_indices)
        query_index=repeated_indices[index]
        mu,label,parent=coefficient_queries[query_index]
        coefficient_outputs[index]=cgc(
            mu,label,parent;cache=coefficient_cache)
    end
    @assert all(coefficient_outputs[index]===
        coefficient_references[repeated_indices[index]]
        for index in eachindex(repeated_indices))

    # Geometry constructors own all mutable staging. The shared primitive
    # table is read-only and therefore safe across task-local preparations.
    constructed=Vector{Any}(undef,2Threads.nthreads())
    Threads.@threads for index in eachindex(constructed)
        constructed[index]=isodd(index) ? OneBodyGeometry(
            coefficient_basis;coefficient_cache=coefficient_cache) :
            PBodyGeometry(coefficient_basis,2;
                coefficient_cache=coefficient_cache)
    end
    @assert all(item isa Union{OneBodyGeometry,PBodyGeometry}
                for item in constructed)
    @assert coefficient_signature==(
        length(coefficient_cache.transitions),
        coefficient_cache.candidate_count,
        coefficient_cache.coefficient_count,
        Base.summarysize(coefficient_cache))
end

trajectory_state=PIState(b,copy(x))
trajectory_plan=TrajectoryPlan(model)
trajectory_batch=TrajectoryBatchWorkspace(
    trajectory_plan,trajectory_state;workers=1,mode=:fixed)
trajectory_work=trajectory_batch.workers[1]
@assert sum(length,(trajectory_work.tmp,trajectory_work.k1,
                    trajectory_work.k2))==3length(trajectory_state.data)
@assert all(isempty,(trajectory_work.k3,trajectory_work.k4,
                     trajectory_work.k5,trajectory_work.k6,
                     trajectory_work.k7,trajectory_work.trial,
                     trajectory_work.embedded,trajectory_work.start))
quantum_trajectories(trajectory_plan,trajectory_state,[0.0,0.05],16;
    dt=0.01,seed=81,workspace=trajectory_batch)
trajectory_batch_alloc=@allocated quantum_trajectories(
    trajectory_plan,trajectory_state,[0.0,0.05],16;
    dt=0.01,seed=81,workspace=trajectory_batch)
@assert trajectory_batch_alloc<=256*1024 "reused trajectory batch allocated $trajectory_batch_alloc bytes"
@assert trajectory_batch.workers[1].plan===trajectory_plan

composite_factor=FiniteOperatorBasis(2;label=:trajectory_auxiliary)
composite_basis=CompositePIBasis(b,composite_factor)
composite_state=composite_tensor_state(
    composite_basis,trajectory_state,ComplexF64[0 0;0 1])
composite_jump=CompositeJumpChannel(
    composite_basis,1=>collective_operator(b,sm),2=>sm;rate=0.05)
composite_trajectory_plan=CompositeTrajectoryPlan(
    composite_basis,composite_jump)
composite_trajectory_work=CompositeTrajectoryWorkspace(
    composite_trajectory_plan,composite_state)
@assert sum(length,(composite_trajectory_work.tmp,
                    composite_trajectory_work.k1,
                    composite_trajectory_work.k2))==
        3length(composite_state.data)
@assert !hasproperty(composite_trajectory_work,:k3)
@assert !hasproperty(composite_trajectory_work,:k4)
composite_rhs=similar(composite_state.data)
PermutationalInvariantDynamics._composite_conditional_action!(
    composite_rhs,composite_state.data,composite_trajectory_work,0.0,nothing)
composite_trajectory_apply_alloc=@allocated(
    PermutationalInvariantDynamics._composite_conditional_action!(
        composite_rhs,composite_state.data,composite_trajectory_work,
        0.0,nothing))
@assert composite_trajectory_apply_alloc<=4096 "prepared composite trajectory RHS allocated $composite_trajectory_apply_alloc bytes"
composite_step_state=copy(composite_state.data)
composite_hazard_limit=-log1p(-0.05)
PermutationalInvariantDynamics._composite_capped_conditional_step!(
    composite_step_state,composite_trajectory_work,0.0,0.005,nothing,
    0.05,composite_hazard_limit)
copyto!(composite_step_state,composite_state.data)
composite_trajectory_step_alloc=@allocated(
    PermutationalInvariantDynamics._composite_capped_conditional_step!(
        composite_step_state,composite_trajectory_work,0.0,0.005,nothing,
        0.05,composite_hazard_limit))
@assert composite_trajectory_step_alloc<=4096 "prepared composite trajectory RK4/hazard step allocated $composite_trajectory_step_alloc bytes"
composite_batch=CompositeTrajectoryBatchWorkspace(
    composite_trajectory_plan,composite_state;workers=1)
quantum_trajectories(
    composite_trajectory_plan,composite_state,[0.0,0.02],8;
    dt=0.005,seed=83,workspace=composite_batch)
composite_trajectory_batch_alloc=@allocated quantum_trajectories(
    composite_trajectory_plan,composite_state,[0.0,0.02],8;
    dt=0.005,seed=83,workspace=composite_batch)
@assert composite_trajectory_batch_alloc<=256*1024 "reused composite trajectory batch allocated $composite_trajectory_batch_alloc bytes"

weak_state=weak_pi_pseudoket(trajectory_state)
weak_plan=WeakPITrajectoryPlan(model)
weak_batch=WeakPITrajectoryBatchWorkspace(
    weak_plan,weak_state;workers=1,mode=:fixed)
weak_work=weak_batch.workers[1]
@assert sum(length,(weak_work.tmp,weak_work.k1,weak_work.k2))==
        3length(weak_state.data)
@assert all(isempty,(weak_work.k3,weak_work.k4,weak_work.k5,
                     weak_work.k6,weak_work.k7,weak_work.trial,
                     weak_work.embedded,weak_work.start))
weak_paths=weak_pi_quantum_trajectories(
    weak_plan,weak_state,[0.0,0.05],16;
    dt=0.01,seed=82,workspace=weak_batch)
weak_pi_trajectory_average(weak_paths)
weak_average_alloc=@allocated weak_pi_trajectory_average(weak_paths)
weak_average=weak_pi_trajectory_average(weak_paths)
@assert weak_average_alloc<=256*1024 "weak-PI trajectory averaging allocated $weak_average_alloc bytes"
@assert all(abs(trace(state)-1)<=2e-12 for state in weak_average)

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
@assert sum(length,(population_work.stage,population_work.k1,
                    population_work.k2,population_work.k3,
                    population_work.k4))==3length(population_source)
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

# Compare allocations rather than wall time so this setup regression remains
# stable across Julia versions and CI hosts.  The uncached route is retained
# privately as a bit-identical oracle for the plan-local factorial cache.
reduction_setup_basis=PIBasis(16,2)
ReductionPlan(reduction_setup_basis,8)
PermutationalInvariantDynamics._qubit_reduction_couplings(
    reduction_setup_basis,8,nothing)
reduction_setup_alloc=@allocated ReductionPlan(reduction_setup_basis,8)
reduction_uncached_alloc=@allocated(
    PermutationalInvariantDynamics._qubit_reduction_couplings(
        reduction_setup_basis,8,nothing))
@assert 3*reduction_setup_alloc<2*reduction_uncached_alloc "cached qubit reduction setup allocated $reduction_setup_alloc bytes versus $reduction_uncached_alloc uncached bytes"

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
@assert sum(length,(meanfield_work.stage,meanfield_work.k1,
                    meanfield_work.k2,meanfield_work.k3,
                    meanfield_work.k4))==3length(sigma)
meanfield_rhs!(meanfield_out,meanfield,sigma,0.0,nothing,meanfield_work)
meanfield_alloc=@allocated meanfield_rhs!(meanfield_out,meanfield,sigma,0.0,nothing,meanfield_work)
@assert meanfield_alloc<=256 "explicit-workspace mean-field RHS allocated $meanfield_alloc bytes"
@assert abs(tr(meanfield_out))<=1e-12 "mean-field RHS did not preserve trace"

println("Performance regression gates passed (threads=$(Threads.nthreads()), apply_alloc=$allocated, batch_alloc=$batch_alloc, batch_adjoint_alloc=$batch_adjoint_alloc, threaded_alloc=$threaded_alloc, restricted_alloc=$restricted_alloc, trajectory_batch_alloc=$trajectory_batch_alloc, composite_trajectory_apply_alloc=$composite_trajectory_apply_alloc, composite_trajectory_step_alloc=$composite_trajectory_step_alloc, composite_trajectory_batch_alloc=$composite_trajectory_batch_alloc, weak_average_alloc=$weak_average_alloc, population_apply_alloc=$population_apply_alloc, population_evolve_alloc=$population_evolve_alloc, observable_alloc=$planned, reduction_alloc=$reduction_alloc, reduction_setup_alloc=$reduction_setup_alloc, reduction_uncached_alloc=$reduction_uncached_alloc, reduction_inplace_alloc=$reduction_inplace_alloc, meanfield_alloc=$meanfield_alloc)")
