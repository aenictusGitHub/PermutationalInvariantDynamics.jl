using PermutationalInvariantDynamics
using LinearAlgebra
using Random
using SparseArrays

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

# The fully symmetric collective route must retain occupation-number geometry
# and exact sparse support at sizes where a dense PI-coordinate temporary is
# already expensive.  These are structural, allocation, and equivalence gates;
# timings remain the responsibility of performance_audit.jl.
collective_basis=PIBasis(64,2;sectors=[(64,0)])
collective_spin=spin_matrices()
collective_model=PIModel(collective_basis,(
    CollectiveHamiltonian(collective_spin.jx;rate=0.11),
    CollectiveJump(collective_spin.jm;rate=0.29)))
collective_context=PermutationalInvariantDynamics.TermCompileContext(
    collective_model)
@assert collective_context.onebody isa
    PermutationalInvariantDynamics._SymmetricCollectiveGeometry
collective_plan=LiouvillianPlan(collective_model)
collective_bounds=
    PermutationalInvariantDynamics._performance_sparse_materialization_bounds(
        collective_plan)
@assert collective_bounds.structured
@assert collective_bounds.contribution_upper_bound<
    big(length(collective_basis))^2

# This budget is intentionally below the corresponding dense coordinate
# matrix but comfortably above the support-aware sparse estimate.  :auto must
# therefore use the structured estimate instead of rejecting the model or
# selecting a needlessly matrix-free solve.
collective_auto=compile(
    collective_model;backend=:auto,memory_budget=128*1024^2)
@assert collective_auto.backend===:sparse
@assert collective_auto.estimates.sparse_structure_supported
@assert nnz(collective_auto.operator)<=
    collective_bounds.retained_nnz_upper_bound

collective_matrixfree=compile(
    collective_model;backend=:matrixfree,memory_budget=Inf)
collective_work=LiouvillianWorkspace(collective_matrixfree)
collective_input=randn(MersenneTwister(72),ComplexF64,
    length(collective_basis))
collective_output=similar(collective_input)
collective_reference=collective_auto.operator*collective_input
apply!(collective_output,collective_matrixfree,collective_input,
       0.0,nothing,collective_work)
@assert isapprox(collective_output,collective_reference;
                 atol=2e-11,rtol=2e-11)
collective_apply_alloc=@allocated apply!(
    collective_output,collective_matrixfree,collective_input,
    0.0,nothing,collective_work)
@assert collective_apply_alloc<=2048 "symmetric collective apply allocated $collective_apply_alloc bytes"
apply_adjoint!(collective_output,collective_matrixfree.plan,
               collective_input,0.0,nothing,collective_work)
@assert isapprox(collective_output,
                 adjoint(collective_auto.operator)*collective_input;
                 atol=2e-11,rtol=2e-11)

# Repeated materialization includes the returned sparse matrix but must not
# allocate even one dense PI-coordinate matrix.  The relative gate remains
# stable across Julia versions and integer index widths.
PermutationalInvariantDynamics._matrix_from_plan(collective_plan)
GC.gc()
collective_materialization_alloc=@allocated(
    PermutationalInvariantDynamics._matrix_from_plan(collective_plan))
collective_dense_temporary_bytes=
    big(length(collective_basis))^2*sizeof(ComplexF64)
@assert collective_materialization_alloc<collective_dense_temporary_bytes "symmetric sparse materialization allocated $collective_materialization_alloc bytes, consistent with a dense PI-coordinate temporary"

# A collective-only mixed-sector model must not retain rectangular one-box
# contractions that only local gain channels can consume.
mixed_collective_model=PIModel(
    b,(CollectiveHamiltonian(sx;rate=0.2),
       CollectiveJump(sm;rate=0.1)))
mixed_collective_context=
    PermutationalInvariantDynamics.TermCompileContext(
        mixed_collective_model)
full_onebody_geometry=OneBodyGeometry(b)
@assert all(key->key[1]==key[3],
    keys(mixed_collective_context.onebody.contractions))
@assert length(mixed_collective_context.onebody.contractions)<
    length(full_onebody_geometry.contractions)
@assert Base.summarysize(mixed_collective_context.onebody)<
    Base.summarysize(full_onebody_geometry)
mixed_geometry_report=recommend_solver(
    mixed_collective_model;memory_budget=Inf)
@assert mixed_geometry_report.geometry_setup_upper_bytes==
    PermutationalInvariantDynamics._estimate_diagonal_onebody_geometry(
        b).setup_bytes
@assert mixed_geometry_report.geometry_setup_upper_bytes<
    estimate_geometry_bytes(b).setup_bytes

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

# Composite block methods batch equal tensor fibers through every factor and
# keep capacity immutable.
composite_batch_generator=CompositeSuperoperator(
    composite_basis,
    local_superoperator_term(composite_basis,1,matrixfree))
composite_batch_source=hcat(
    composite_state.data,0.7composite_state.data,
    complex.(reverse(composite_state.data)))
composite_batch_output=similar(composite_batch_source)
composite_batch_work=CompositeSuperoperatorBatchWorkspace(
    composite_batch_generator;capacity=3)
apply!(composite_batch_output,composite_batch_generator,
       composite_batch_source,0.0,nothing,composite_batch_work)
composite_batch_reference=hcat((
    composite_batch_generator*view(composite_batch_source,:,column)
    for column in axes(composite_batch_source,2))...)
@assert isapprox(composite_batch_output,composite_batch_reference;
                 atol=2e-11,rtol=2e-11)
composite_batch_alloc=@allocated apply!(
    composite_batch_output,composite_batch_generator,
    composite_batch_source,0.0,nothing,composite_batch_work)
@assert composite_batch_alloc<=4096 "prepared composite matrix RHS allocated $composite_batch_alloc bytes"
apply_adjoint!(composite_batch_output,composite_batch_generator,
               composite_batch_source,0.0,nothing,composite_batch_work)
composite_batch_adjoint_alloc=@allocated apply_adjoint!(
    composite_batch_output,composite_batch_generator,
    composite_batch_source,0.0,nothing,composite_batch_work)
@assert composite_batch_adjoint_alloc<=4096 "prepared composite adjoint matrix RHS allocated $composite_batch_adjoint_alloc bytes"
@assert PermutationalInvariantDynamics._performance_linear_operator_workspace_bytes(
    composite_batch_generator;batch_columns=3)>
    PermutationalInvariantDynamics._performance_linear_operator_workspace_bytes(
        composite_batch_generator)

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

# Local pseudomodes enlarge one identical supersite but remain an ordinary PI
# model.  Exercise public construction, both generator directions, and the
# prepared spin-factor trace without reconstructing the full Hilbert space.
pseudomode_mode=BosonicPseudomode(
    1;frequency=0.73,damping=0.31,label=:regression_local)
pseudomode_coupling=PseudomodeCoupling(
    sm;mode=:regression_local,strength=0.16)
pseudomode_embedding=pseudomode_model(
    2,zeros(ComplexF64,2,2),pseudomode_mode;
    couplings=pseudomode_coupling)
pseudomode_sparse=compile(
    pseudomode_embedding.model;backend=:sparse,memory_budget=Inf)
pseudomode_prepared=compile(
    pseudomode_embedding.model;backend=:matrixfree,memory_budget=Inf)
pseudomode_state=pseudomode_product_state(
    pseudomode_embedding.supersite,ComplexF64[0,1])
pseudomode_output=similar(pseudomode_state.data)
pseudomode_reference=similar(pseudomode_state.data)
pseudomode_work=LiouvillianWorkspace(pseudomode_prepared)
mul!(
    pseudomode_reference,pseudomode_sparse,pseudomode_state.data)
apply!(
    pseudomode_output,pseudomode_prepared,pseudomode_state.data,
    0.0,nothing,pseudomode_work)
@assert isapprox(
    pseudomode_output,pseudomode_reference;
    atol=5e-11,rtol=5e-11)
@assert abs(trace(PIState(
    pseudomode_embedding.basis,pseudomode_output)))<=5e-11
pseudomode_apply_alloc=@allocated apply!(
    pseudomode_output,pseudomode_prepared,pseudomode_state.data,
    0.0,nothing,pseudomode_work)
@assert pseudomode_apply_alloc<=4096 "local pseudomode apply allocated $pseudomode_apply_alloc bytes"
pseudomode_adjoint_source=randn(
    MersenneTwister(0x50534d41),ComplexF64,
    length(pseudomode_embedding.basis))
pseudomode_adjoint_output=similar(pseudomode_adjoint_source)
pseudomode_adjoint_reference=similar(pseudomode_adjoint_source)
apply_adjoint!(
    pseudomode_adjoint_reference,pseudomode_sparse,
    pseudomode_adjoint_source)
apply_adjoint!(
    pseudomode_adjoint_output,pseudomode_prepared,
    pseudomode_adjoint_source,0.0,nothing,pseudomode_work)
@assert isapprox(
    pseudomode_adjoint_output,pseudomode_adjoint_reference;
    atol=5e-11,rtol=5e-11)
pseudomode_trace_prepared=pseudomode_trace_plan(
    pseudomode_embedding.supersite)
pseudomode_trace_work=LocalFactorTraceWorkspace(
    pseudomode_trace_prepared)
pseudomode_trace_output=PIState(
    pseudomode_trace_prepared.output_basis)
pseudomode_trace_expected=iid_pure_state(
    pseudomode_trace_prepared.output_basis,ComplexF64[0,1])
trace_pseudomodes!(
    pseudomode_trace_output,pseudomode_state,
    pseudomode_embedding.supersite,pseudomode_trace_prepared,
    pseudomode_trace_work;check=false)
@assert isapprox(
    pseudomode_trace_output.data,pseudomode_trace_expected.data;
    atol=5e-11,rtol=5e-11)
pseudomode_trace_alloc=@allocated trace_pseudomodes!(
    pseudomode_trace_output,pseudomode_state,
    pseudomode_embedding.supersite,pseudomode_trace_prepared,
    pseudomode_trace_work;check=false)
@assert pseudomode_trace_alloc<=4096 "local pseudomode trace allocated $pseudomode_trace_alloc bytes"

# A single shared mode is kept as a finite composite factor.  Its action and
# two prepared reductions must remain allocation-free after warm-up.  Adjoint
# duality supplies an independent matrix-free correctness check.
global_mode=BosonicPseudomode(
    1;frequency=0.73,damping=0.31,label=:regression_shared)
global_coupling=PseudomodeCoupling(
    sm;mode=:regression_shared,strength=0.16)
global_system=PIModel(PIBasis(2,2),(
    LocalJump(sm;rate=0.04),))
global_embedding=global_pseudomode_model(
    global_system,global_mode;couplings=global_coupling)
global_state=pseudomode_product_state(
    global_embedding,ComplexF64[0,1])
global_work=global_pseudomode_workspace(global_embedding)
global_output=similar(global_state.data)
global_adjoint_source=randn(
    MersenneTwister(0x47504d41),ComplexF64,
    length(global_embedding.basis))
global_adjoint_output=similar(global_adjoint_source)
apply!(global_output,global_embedding,global_state.data,global_work)
apply_adjoint!(
    global_adjoint_output,global_embedding,
    global_adjoint_source,global_work)
@assert abs(dot(global_embedding.trace_vector,global_output))<=5e-11
@assert isapprox(
    dot(global_adjoint_source,global_output),
    dot(global_adjoint_output,global_state.data);
    atol=5e-11,rtol=5e-11)
global_apply_alloc=@allocated apply!(
    global_output,global_embedding,global_state.data,global_work)
@assert global_apply_alloc<=4096 "global pseudomode apply allocated $global_apply_alloc bytes"
global_system_output=PIState(global_embedding.system_basis)
global_mode_output=zeros(
    ComplexF64,global_mode.levels,global_mode.levels)
trace_pseudomodes!(
    global_system_output,global_state,global_embedding)
global_pseudomode_state!(
    global_mode_output,global_state,global_embedding)
global_system_expected=iid_pure_state(
    global_embedding.system_basis,ComplexF64[0,1])
global_mode_expected=
    global_mode.vacuum*adjoint(global_mode.vacuum)
@assert isapprox(
    global_system_output.data,global_system_expected.data;
    atol=5e-11,rtol=5e-11)
@assert isapprox(
    global_mode_output,global_mode_expected;
    atol=5e-11,rtol=5e-11)
global_system_trace_alloc=@allocated trace_pseudomodes!(
    global_system_output,global_state,global_embedding)
global_mode_trace_alloc=@allocated global_pseudomode_state!(
    global_mode_output,global_state,global_embedding)
@assert global_system_trace_alloc<=1024 "global pseudomode system trace allocated $global_system_trace_alloc bytes"
@assert global_mode_trace_alloc<=1024 "global pseudomode mode trace allocated $global_mode_trace_alloc bytes"

# PI-HEOM stores PI density operators at hierarchy nodes; PI-HOPS stores one
# direct-sum Schur pseudo-ket per node.  Both conditioned hot kernels are
# preallocated, and HEOM additionally supports exact forward/adjoint batches.
hierarchy_basis=PIBasis(2,2)
hierarchy_spin=spin_matrices()
hierarchy_model=qubit_ensemble_model(
    hierarchy_basis;emission=0.04)
hierarchy_coupling=collective_operator(
    hierarchy_basis,hierarchy_spin.jz)
hierarchy_bath=HEOMBath(
    hierarchy_coupling,[0.18,0.05],[1.1,1.7];
    right_coefficients=[0.18,0.05])
heom_plan=HEOMPlan(
    hierarchy_model,hierarchy_bath;
    max_depth=2,scaling=:scaled)
heom_work=HEOMWorkspace(heom_plan;batch_columns=3)
heom_source=randn(
    MersenneTwister(0x48454f4d),ComplexF64,size(heom_plan,1))
heom_output=similar(heom_source)
heom_adjoint_source=randn(
    MersenneTwister(0x48454f4e),ComplexF64,size(heom_plan,1))
heom_adjoint_output=similar(heom_source)
apply!(heom_output,heom_plan,heom_source,heom_work)
apply_adjoint!(
    heom_adjoint_output,heom_plan,heom_adjoint_source,heom_work)
@assert abs(dot(heom_plan.tracevec,heom_output))<=5e-11
@assert isapprox(
    dot(heom_adjoint_source,heom_output),
    dot(heom_adjoint_output,heom_source);
    atol=5e-11,rtol=5e-11)
heom_apply_alloc=@allocated apply!(
    heom_output,heom_plan,heom_source,heom_work)
@assert heom_apply_alloc<=4096 "PI-HEOM apply allocated $heom_apply_alloc bytes"
heom_batch_source=hcat(
    heom_source,0.5heom_source,complex.(reverse(heom_source)))
heom_batch_output=similar(heom_batch_source)
apply!(
    heom_batch_output,heom_plan,heom_batch_source,heom_work)
heom_batch_reference=hcat((
    begin
        column=similar(heom_source)
        apply!(
            column,heom_plan,view(heom_batch_source,:,index),
            HEOMWorkspace(heom_plan))
        column
    end for index in axes(heom_batch_source,2))...)
@assert isapprox(
    heom_batch_output,heom_batch_reference;atol=5e-11,rtol=5e-11)
heom_batch_alloc=@allocated apply!(
    heom_batch_output,heom_plan,heom_batch_source,heom_work)
@assert heom_batch_alloc<=4096 "PI-HEOM batch allocated $heom_batch_alloc bytes"

hops_hamiltonian=collective_operator(
    hierarchy_basis,0.11 .* hierarchy_spin.jx)
hops_bath=HOPSBath(
    hierarchy_coupling,[0.18,0.05],[1.1,1.7])
hops_plan=HOPSPlan(
    hops_hamiltonian,hops_bath;
    max_depth=2,scaling=:scaled)
hops_work=HOPSWorkspace(hops_plan)
hops_source=randn(
    MersenneTwister(0x484f5053),ComplexF64,size(hops_plan,1))
hops_output=similar(hops_source)
hops_noise=ComplexF64[0.03-0.01im]
hops_rhs!(
    hops_output,hops_plan,hops_source,hops_noise,hops_work)
hops_repeated=similar(hops_output)
hops_rhs!(
    hops_repeated,hops_plan,hops_source,hops_noise,hops_work)
@assert hops_repeated==hops_output
@assert all(
    value->isfinite(real(value))&&isfinite(imag(value)),hops_output)
hops_scaled_source=0.37hops_source
hops_scaled_output=similar(hops_output)
hops_rhs!(
    hops_scaled_output,hops_plan,hops_scaled_source,hops_noise,hops_work)
@assert isapprox(
    hops_scaled_output,0.37hops_output;
    atol=5e-12,rtol=5e-12)
hops_apply_alloc=@allocated hops_rhs!(
    hops_repeated,hops_plan,hops_source,hops_noise,hops_work)
@assert hops_apply_alloc<=4096 "PI-HOPS conditioned RHS allocated $hops_apply_alloc bytes"
hops_initial=weak_pi_pseudoket(iid_pure_state(
    hierarchy_basis,ComplexF64[1,1]/sqrt(2)))
hops_batch=HOPSBatchWorkspace(hops_plan;workers=1)
hops_times=[0.0,0.01]
hops_average_reference=hops_average(
    hops_plan,hops_initial,hops_times,4;
    dt=0.005,seed=0x484f5053,threaded=false,
    workspace=hops_batch)
hops_average_repeated=hops_average(
    hops_plan,hops_initial,hops_times,4;
    dt=0.005,seed=0x484f5053,threaded=false,
    workspace=hops_batch)
@assert all(
    reference.data==repeated.data
    for (reference,repeated) in
    zip(hops_average_reference,hops_average_repeated))
@assert all(
    all(value->isfinite(real(value))&&isfinite(imag(value)),state.data)
    for state in hops_average_reference)
hops_batch_alloc=@allocated hops_average(
    hops_plan,hops_initial,hops_times,4;
    dt=0.005,seed=0x484f5053,threaded=false,
    workspace=hops_batch)
@assert hops_batch_alloc<=512*1024 "reused PI-HOPS ensemble allocated $hops_batch_alloc bytes"

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

# Repeated one-body reductions reuse the largest Schur block instead of
# allocating one multiplicity-weighted block per sector.  The independent
# particle-reduction route is the correctness oracle.
onebody_reduction=ReductionPlan(b,1)
onebody_reference=one_body_rdm(rho;plan=onebody_reduction)
onebody_work=OneBodyRDMWorkspace(geometry,rho)
onebody_output=zeros(ComplexF64,b.d,b.d)
one_body_rdm!(onebody_output,rho,onebody_work;check=false)
@assert isapprox(onebody_output,onebody_reference;atol=2e-11,rtol=2e-11)
onebody_alloc=@allocated one_body_rdm!(
    onebody_output,rho,onebody_work;check=false)
@assert onebody_alloc<=2048 "validate-once one-body RDM contraction allocated $onebody_alloc bytes"
onebody_max_block=maximum(length.(b.patterns))
onebody_block_payload=
    onebody_max_block^2*sizeof(eltype(onebody_work.weighted_block))
@assert length(onebody_work.weighted_block)==onebody_max_block^2
@assert length(onebody_work.multiplicity_scales)==length(b.sectors)
@assert Base.summarysize(onebody_work.weighted_block)<
    onebody_block_payload+1024

# The local-factor partial trace retains exact-support CSC transforms.  This
# representative supersite has a sparse plan more than an order of magnitude
# smaller than the corresponding dense rectangular transforms.
local_factor_basis=PIBasis(3,4)
local_factor_ket=ComplexF64[
    sqrt(0.6),0,0,exp(0.3im)*sqrt(0.4)]
local_factor_source=iid_pure_state(
    local_factor_basis,local_factor_ket)
local_factor_plan=LocalFactorTracePlan(
    local_factor_source,(2,2);traced_factor=2)
local_factor_work=LocalFactorTraceWorkspace(local_factor_plan)
local_factor_output=PIState(local_factor_plan.output_basis)
local_factor_expected=iid_state(
    local_factor_plan.output_basis,ComplexF64[0.6 0;0 0.4])
local_factor_trace!(
    local_factor_output,local_factor_source,
    local_factor_plan,local_factor_work;check=false)
@assert isapprox(
    local_factor_output.data,local_factor_expected.data;
    atol=3e-11,rtol=3e-11)
local_factor_alloc=@allocated local_factor_trace!(
    local_factor_output,local_factor_source,
    local_factor_plan,local_factor_work;check=false)
@assert local_factor_alloc<=2048 "prepared local-factor trace allocated $local_factor_alloc bytes"
@assert local_factor_plan.lifted_columns isa SparseMatrixCSC
@assert local_factor_plan.output_columns isa SparseMatrixCSC
@assert local_factor_plan.estimates.storage===:exact_support_sparse_csc
@assert local_factor_plan.estimates.gram_validation===
    :streamed_sparse_columns
@assert local_factor_plan.estimates.gram_workspace_bytes<
    local_factor_plan.estimates.output_dimension^2*
    sizeof(eltype(local_factor_plan.output_columns))
@assert local_factor_plan.estimates.retained_entries==
    nnz(local_factor_plan.lifted_columns)+
    nnz(local_factor_plan.output_columns)
@assert local_factor_plan.estimates.retained_bytes>=
    Base.summarysize(local_factor_plan.lifted_columns)+
    Base.summarysize(local_factor_plan.output_columns)
@assert 10local_factor_plan.estimates.retained_entries<
    local_factor_plan.estimates.dense_entries

# Composite reductions retain only packed joint-diagonal offsets and exact
# multiplicity groups.  They must not cache the full composite trace vector.
composite_reduction_factor=FiniteOperatorBasis(
    8;label=:reduction_auxiliary)
composite_reduction_basis=CompositePIBasis(
    b,composite_reduction_factor)
composite_reduction_system=maximally_mixed_state(b)
composite_reduction_auxiliary=
    Matrix{ComplexF64}(I,8,8)/8
composite_reduction_source=composite_tensor_state(
    composite_reduction_basis,composite_reduction_system,
    composite_reduction_auxiliary)
composite_system_plan=CompositeReductionPlan(
    composite_reduction_source,1)
composite_auxiliary_plan=CompositeReductionPlan(
    composite_reduction_source,2)
composite_system_output=PIState(b)
composite_auxiliary_output=zeros(ComplexF64,8,8)
composite_reduced_state!(
    composite_system_output,composite_reduction_source,
    composite_system_plan)
composite_reduced_state!(
    composite_auxiliary_output,composite_reduction_source,
    composite_auxiliary_plan)
@assert isapprox(
    composite_system_output.data,composite_reduction_system.data;
    atol=2e-12,rtol=2e-12)
@assert isapprox(
    composite_auxiliary_output,composite_reduction_auxiliary;
    atol=2e-12,rtol=2e-12)
composite_system_alloc=@allocated composite_reduced_state!(
    composite_system_output,composite_reduction_source,
    composite_system_plan)
composite_auxiliary_alloc=@allocated composite_reduced_state!(
    composite_auxiliary_output,composite_reduction_source,
    composite_auxiliary_plan)
@assert composite_system_alloc<=1024 "prepared composite-to-PI reduction allocated $composite_system_alloc bytes"
@assert composite_auxiliary_alloc<=1024 "prepared composite-to-finite reduction allocated $composite_auxiliary_alloc bytes"
composite_trace_payload=
    length(composite_reduction_basis)*sizeof(ComplexF64)
for plan in (composite_system_plan,composite_auxiliary_plan)
    @assert !(:trace_vector in fieldnames(typeof(plan)))
    @assert plan.estimates.retained_bytes<composite_trace_payload
    packed_bytes=
        Base.summarysize(plan.traced_offsets)+
        Base.summarysize(plan.group_boundaries)+
        Base.summarysize(plan.exact_multiplicities)+
        Base.summarysize(plan.scales)+
        Base.summarysize(plan.prepared_scales)
    @assert packed_bytes<composite_trace_payload
end

reduction=ReductionPlan(b,2)
planned_reduced=reduced_state(rho,2;plan=reduction)
reduction_alloc=@allocated reduced_state(rho,2;plan=reduction)
@assert reduction_alloc<=2*1024^2 "prepared reduction allocated $reduction_alloc bytes"

# Appendix-D geometry and qudit LR plans retain exact support instead of dense
# zero-heavy path tensors or product-weight intertwiners.  Exercise the packed
# two-body geometry through both production Liouvillian backends, rather than
# testing its storage metadata in isolation.
qutrit_kernel_basis=PIBasis(3,3)
packed_pbody=PBodyGeometry(qutrit_kernel_basis,2)
@assert packed_pbody.estimates.storage===:exact_support_sparse_csc
@assert packed_pbody.estimates.retained_entries<
        packed_pbody.estimates.dense_entries
@assert packed_pbody.estimates.retained_bytes<
        packed_pbody.estimates.dense_payload_bytes

qutrit_seed=ComplexF64[
    1.0 0.10im 0.05
    0.20 0.80 0.08im
    0.10im 0.15 0.60]
qutrit_local_density=qutrit_seed*qutrit_seed'
qutrit_local_density./=tr(qutrit_local_density)
qutrit_kernel_state=iid_state(qutrit_kernel_basis,qutrit_local_density)
qutrit_level=ComplexF64[1 0 0;0 0 0;0 0 -1]
qutrit_lowering=ComplexF64[0 1 0;0 0 sqrt(2);0 0 0]
qutrit_pbody_model=PIModel(qutrit_kernel_basis,(
    PBodyHamiltonian(kron(qutrit_level,qutrit_level),2;rate=0.07),
    LocalPBodyJump(kron(qutrit_lowering,qutrit_lowering),2;rate=0.03)))
qutrit_pbody_sparse=compile(qutrit_pbody_model;backend=:sparse)
qutrit_pbody_matrixfree=compile(qutrit_pbody_model;backend=:matrixfree)
qutrit_pbody_work=LiouvillianWorkspace(qutrit_pbody_matrixfree)
qutrit_pbody_out=similar(qutrit_kernel_state.data)
qutrit_pbody_reference=similar(qutrit_kernel_state.data)
mul!(qutrit_pbody_reference,qutrit_pbody_sparse,qutrit_kernel_state.data)
apply!(
    qutrit_pbody_out,qutrit_pbody_matrixfree,qutrit_kernel_state.data,
    0.0,nothing,qutrit_pbody_work)
@assert isapprox(
    qutrit_pbody_out,qutrit_pbody_reference;atol=5e-11,rtol=5e-11)
@assert abs(trace(PIState(qutrit_kernel_basis,qutrit_pbody_out)))<=5e-11

qutrit_pbody_adjoint_source=randn(
    MersenneTwister(93),ComplexF64,length(qutrit_kernel_basis))
qutrit_pbody_adjoint_out=similar(qutrit_pbody_adjoint_source)
qutrit_pbody_adjoint_reference=similar(qutrit_pbody_adjoint_source)
apply_adjoint!(
    qutrit_pbody_adjoint_reference,qutrit_pbody_sparse,
    qutrit_pbody_adjoint_source)
apply_adjoint!(
    qutrit_pbody_adjoint_out,qutrit_pbody_matrixfree,
    qutrit_pbody_adjoint_source,0.0,nothing,qutrit_pbody_work)
@assert isapprox(
    qutrit_pbody_adjoint_out,qutrit_pbody_adjoint_reference;
    atol=5e-11,rtol=5e-11)

# The allocation gate is measured only after both directions have initialized
# their task-owned scratch.
apply!(
    qutrit_pbody_out,qutrit_pbody_matrixfree,qutrit_kernel_state.data,
    0.0,nothing,qutrit_pbody_work)
qutrit_pbody_alloc=@allocated apply!(
    qutrit_pbody_out,qutrit_pbody_matrixfree,qutrit_kernel_state.data,
    0.0,nothing,qutrit_pbody_work)
@assert qutrit_pbody_alloc<=16*1024 "packed qutrit p-body apply allocated $qutrit_pbody_alloc bytes"
qutrit_pbody_adjoint_alloc=@allocated apply_adjoint!(
    qutrit_pbody_adjoint_out,qutrit_pbody_matrixfree,
    qutrit_pbody_adjoint_source,0.0,nothing,qutrit_pbody_work)
@assert qutrit_pbody_adjoint_alloc<=16*1024 "packed qutrit p-body adjoint apply allocated $qutrit_pbody_adjoint_alloc bytes"

# Keep two of three qutrits and compare with the analytic product-state
# marginal.  This covers the packed LR intertwiners with nonzero coherences and
# a nontrivial discarded particle.
packed_reduction=ReductionPlan(qutrit_kernel_basis,2)
@assert packed_reduction.estimates.storage===:weight_block_sparse_csc
@assert packed_reduction.estimates.retained_entries<
        packed_reduction.estimates.dense_entries
@assert packed_reduction.estimates.retained_bytes<
        packed_reduction.estimates.dense_payload_bytes
packed_reduction_source=qutrit_kernel_state
packed_reduction_expected=iid_state(
    packed_reduction.output_basis,qutrit_local_density)
packed_reduction_work=ReductionWorkspace(
    packed_reduction,packed_reduction_source;mode=:reduction)
@assert isempty(packed_reduction_work.product_block)
packed_reduction_out=PIState(packed_reduction.output_basis)
reduced_state!(
    packed_reduction_out,packed_reduction_source,
    packed_reduction,packed_reduction_work;check=false)
@assert isapprox(
    packed_reduction_out.data,packed_reduction_expected.data;
    atol=5e-11,rtol=5e-11)
packed_reduction_alloc=@allocated reduced_state!(
    packed_reduction_out,packed_reduction_source,
    packed_reduction,packed_reduction_work;check=false)
@assert packed_reduction_alloc<=32*1024 "packed qudit reduction allocated $packed_reduction_alloc bytes"

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
reduction_only_work=ReductionWorkspace(reduction,rho;mode=:reduction)
@assert isempty(reduction_only_work.product_block)
@assert size(reduction_only_work.product_tmp,1)==
    maximum(c.da for c in reduction.couplings)
reduced_state!(
    reduction_out,rho,reduction,reduction_only_work;check=false)
reduction_unchecked_alloc=@allocated reduced_state!(
    reduction_out,rho,reduction,reduction_only_work;check=false)
@assert reduction_out.data≈planned_reduced.data atol=2e-11
@assert reduction_unchecked_alloc<=8*1024 "validate-once in-place reduction allocated $reduction_unchecked_alloc bytes"

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

# Adaptive exponential-action storage is bounded by the one Arnoldi basis,
# projected matrices, and three full-coordinate vectors. Rejected projected
# slices must not repeat full-space operator applications.
expv_n=6;expv_krylovdim=4
expv_expected_entries=
    expv_n*(expv_krylovdim+4)+
    (expv_krylovdim+1)*expv_krylovdim+
    (expv_krylovdim+1)^2
expv_workspace=KrylovExpvWorkspace(
    ComplexF64,expv_n,expv_krylovdim)
@assert sum(sizeof,(
    expv_workspace.V,expv_workspace.H,expv_workspace.small,
    expv_workspace.w,expv_workspace.current,expv_workspace.trial))==
    expv_expected_entries*sizeof(ComplexF64)
expv_operator=diagm(0=>ComplexF64.(range(-1,1;length=expv_n)))
expv_source=ones(ComplexF64,expv_n)
expv_result=krylov_expv(
    expv_operator,expv_source,0.1;
    krylovdim=expv_krylovdim,initial_step=0.1,
    atol=1e-8,rtol=1e-7)
@assert expv_result.rejected_steps>0
@assert expv_result.arnoldi_factorizations==expv_result.accepted_steps
@assert expv_result.operator_applications==
    expv_krylovdim*expv_result.arnoldi_factorizations
@assert PermutationalInvariantDynamics._performance_krylov_expv_workspace_bytes(
    expv_n,ComplexF64,expv_krylovdim)==
    expv_expected_entries*sizeof(ComplexF64)

println("Performance regression gates passed (threads=$(Threads.nthreads()), apply_alloc=$allocated, batch_alloc=$batch_alloc, batch_adjoint_alloc=$batch_adjoint_alloc, collective_apply_alloc=$collective_apply_alloc, collective_materialization_alloc=$collective_materialization_alloc, threaded_alloc=$threaded_alloc, restricted_alloc=$restricted_alloc, trajectory_batch_alloc=$trajectory_batch_alloc, composite_trajectory_apply_alloc=$composite_trajectory_apply_alloc, composite_trajectory_step_alloc=$composite_trajectory_step_alloc, composite_trajectory_batch_alloc=$composite_trajectory_batch_alloc, weak_average_alloc=$weak_average_alloc, pseudomode_apply_alloc=$pseudomode_apply_alloc, pseudomode_trace_alloc=$pseudomode_trace_alloc, global_apply_alloc=$global_apply_alloc, global_system_trace_alloc=$global_system_trace_alloc, global_mode_trace_alloc=$global_mode_trace_alloc, heom_apply_alloc=$heom_apply_alloc, heom_batch_alloc=$heom_batch_alloc, hops_apply_alloc=$hops_apply_alloc, hops_batch_alloc=$hops_batch_alloc, population_apply_alloc=$population_apply_alloc, population_evolve_alloc=$population_evolve_alloc, observable_alloc=$planned, onebody_alloc=$onebody_alloc, local_factor_alloc=$local_factor_alloc, composite_system_alloc=$composite_system_alloc, composite_auxiliary_alloc=$composite_auxiliary_alloc, reduction_alloc=$reduction_alloc, qutrit_pbody_alloc=$qutrit_pbody_alloc, qutrit_pbody_adjoint_alloc=$qutrit_pbody_adjoint_alloc, packed_reduction_alloc=$packed_reduction_alloc, reduction_setup_alloc=$reduction_setup_alloc, reduction_uncached_alloc=$reduction_uncached_alloc, reduction_inplace_alloc=$reduction_inplace_alloc, reduction_unchecked_alloc=$reduction_unchecked_alloc, meanfield_alloc=$meanfield_alloc)")
