using PermutationalInvariantDynamics
using LinearAlgebra
using Printf
using Random

function measure(label,f; samples=3)
    f() # compile and warm caches owned by the object
    runs=[@timed f() for _ in 1:samples]
    best=argmin(getfield.(runs,:time));r=runs[best]
    @printf("%-34s %10.3f ms %12.3f MiB\n",label,1e3r.time,r.bytes/2.0^20)
    r.value
end

println("PermutationalInvariantDynamics performance audit")
println("operation                                time          allocated")
println("----------------------------------------------------------------")

b=measure("PIBasis(N=10,d=2)",()->PIBasis(10,2))
b3=measure("PIBasis(N=8,d=3)",()->PIBasis(8,3);samples=1)
g3=first(first(b3.patterns));measure("GT content tuple",()->content(g3))
geometry3=measure("qudit one-body geometry",()->OneBodyGeometry(b3);samples=1)
legacy_contractions=Dict(key=>
    [collect(table[row,column]) for row in axes(table,1),column in axes(table,2)]
    for (key,table) in geometry3.contractions)
@printf("  One-body contractions: packed %.3f MiB, legacy cells %.3f MiB\n",
        (Base.summarysize(geometry3)-Base.summarysize(b3))/2.0^20,
        Base.summarysize(legacy_contractions)/2.0^20)
sm=ComplexF64[0 1;0 0];sx=ComplexF64[0 1;1 0]
model=PIModel(b,[LocalJump(sm),CollectiveHamiltonian(sx;rate=0.15)])
Ls=measure("sparse Liouvillian assembly",()->liouvillian(model;representation=:sparse);samples=1)
Lm=measure("matrix-free construction",()->liouvillian(model;representation=:matrixfree);samples=1)
prepared=measure("compiled matrix-free model",()->compile(model;backend=:matrixfree);samples=1)
rho=iid_pure_state(b,ComplexF64[0,1]);y=similar(rho.data)
measure("sparse application",()->mul!(y,Ls,rho.data))
measure("matrix-free application",()->mul!(y,Lm,rho.data))
liouvillian_work=LiouvillianWorkspace(prepared)
measure("explicit-workspace application",()->apply!(y,prepared,rho.data,0.0,nothing,liouvillian_work))
batch=hcat(rho.data,0.5rho.data,rho.data);batch_out=similar(batch)
measure("batched matrix-free application",()->apply!(batch_out,prepared.plan,batch,0.0,nothing,liouvillian_work))
measure("batched adjoint application",()->apply_adjoint!(batch_out,prepared.plan,batch,0.0,nothing,liouvillian_work))

# Driven local gain maps use common-child rectangular contractions rather
# than a quartic PI-coordinate I/J/value table.  Report retained storage next
# to the size of that removed coarse representation.
driven_basis=PIBasis(20,2)
driven_operator=ComplexF64[0.2 1.0;0.4im -0.1]
driven_schedule=InPlaceTimeOperator(
    driven_operator,(destination,time,parameters)->nothing)
driven_plan=measure("driven local-jump plan (N=20)",()->LiouvillianPlan(
    PIModel(driven_basis,[LocalJump(driven_schedule;rate=0.2)]));samples=1)
driven_work=LiouvillianWorkspace(driven_plan)
driven_kernel=only(driven_plan.kernels)
driven_pairs=Set((branch.output_sector,branch.input_sector)
    for branch in driven_kernel.branches.entries)
coarse_coordinates=sum(length(driven_basis.patterns[li])^2*
                       length(driven_basis.patterns[ni])^2
                       for (li,ni) in driven_pairs)
coarse_bytes=coarse_coordinates*(2sizeof(Int)+sizeof(eltype(driven_plan)))
@printf("  Driven local gain: retained %.3f MiB, removed coarse table %.3f MiB\n",
        (Base.summarysize(driven_plan)+Base.summarysize(driven_work))/2.0^20,
        coarse_bytes/2.0^20)
driven_input=randn(MersenneTwister(100),ComplexF64,length(driven_basis))
driven_output=similar(driven_input)
measure("driven local-jump application",()->apply!(
    driven_output,driven_plan,driven_input,0.2,nothing,driven_work))
driven_inputs=hcat(driven_input,0.5driven_input,reverse(driven_input))
driven_outputs=similar(driven_inputs)
measure("driven local-jump batch",()->apply!(
    driven_outputs,driven_plan,driven_inputs,0.2,nothing,driven_work))
w=EvolutionWorkspace(Lm,rho)
@printf("  Low-storage RK4 scratch: %.3f MiB (3 full vectors)\n",
        sum(Base.summarysize,(w.tmp,w.k1,w.k2))/2.0^20)
measure("low-storage RK4 step",()->evolve!(y,Lm,rho.data,(0.0,1e-3);steps=1,workspace=w))

# Stochastic backends separate reusable propagation scratch from the returned
# histories.  Report both path generation and ensemble reconstruction because
# their memory scaling differs materially.
trajectory_plan=measure("density trajectory plan",()->TrajectoryPlan(model);samples=1)
trajectory_batch=TrajectoryBatchWorkspace(
    trajectory_plan,rho;workers=1,mode=:fixed)
trajectory_full_batch=TrajectoryBatchWorkspace(
    trajectory_plan,rho;workers=1,mode=:full)
@printf("  Trajectory worker: fixed %.3f MiB, full %.3f MiB\n",
        Base.summarysize(trajectory_batch)/2.0^20,
        Base.summarysize(trajectory_full_batch)/2.0^20)
density_paths=measure("density trajectory batch",()->quantum_trajectories(
    trajectory_plan,rho,[0.0,0.02],16;dt=0.005,seed=101,
    workspace=trajectory_batch);samples=1)
measure("density trajectory average",()->trajectory_average(density_paths))

weak_state=weak_pi_pseudoket(rho)
weak_plan=measure("weak-PI trajectory plan",()->WeakPITrajectoryPlan(model);samples=1)
weak_batch=WeakPITrajectoryBatchWorkspace(weak_plan,weak_state;workers=1)
weak_paths=measure("weak-PI trajectory batch",()->weak_pi_quantum_trajectories(
    weak_plan,weak_state,[0.0,0.02],16;dt=0.005,seed=102,
    workspace=weak_batch);samples=1)
measure("weak-PI trajectory average",()->weak_pi_trajectory_average(weak_paths))

diffusive_model=PIModel(b,(CollectiveJump(sm;rate=0.05),))
diffusive_plan=measure("diffusive trajectory plan",()->DiffusivePlan(
    diffusive_model,homodyne_monitor(sqrt(0.05)*sm;efficiency=0.8));samples=1)
diffusive_work=DiffusiveWorkspace(diffusive_plan,rho)
measure("prepared diffusive path",()->diffusive_trajectory(
    diffusive_plan,rho,[0.0,0.002];dt=0.001,rng=MersenneTwister(103),
    workspace=diffusive_work,save_states=false);samples=1)

# Composite application is genuinely tensor-mode matrix-free: this audit uses
# a nontrivial finite auxiliary factor but never forms its global Kronecker
# superoperator.
composite_basis=CompositePIBasis(b,FiniteOperatorBasis(2;label=:auxiliary))
lifted=local_superoperator_term(composite_basis,1,prepared)
composite_map=measure("composite matrix-free setup",()->
    CompositeSuperoperator(composite_basis,lifted);samples=1)
composite_x=randn(MersenneTwister(104),ComplexF64,length(composite_basis))
composite_y=similar(composite_x)
composite_work=CompositeSuperoperatorWorkspace(composite_map,composite_x)
measure("composite workspace application",()->apply!(
    composite_y,composite_map,composite_x,0.0,nothing,composite_work))

composite_state=composite_tensor_state(
    composite_basis,rho,ComplexF64[0 0;0 1])
composite_jump=CompositeJumpChannel(
    composite_basis,1=>collective_operator(b,sm),2=>sm;rate=0.05)
composite_trajectory_plan=measure("composite trajectory plan",()->
    CompositeTrajectoryPlan(composite_basis,composite_jump);samples=1)
composite_trajectory_work=CompositeTrajectoryWorkspace(
    composite_trajectory_plan,composite_state)
composite_rhs=similar(composite_state.data)
measure("composite conditional RHS",()->
    PermutationalInvariantDynamics._composite_conditional_action!(
        composite_rhs,composite_state.data,composite_trajectory_work,
        0.0,nothing))
composite_step_state=copy(composite_state.data)
composite_hazard_limit=-log1p(-0.05)
measure("composite RK4/hazard step",()->begin
    copyto!(composite_step_state,composite_state.data)
    PermutationalInvariantDynamics._composite_capped_conditional_step!(
        composite_step_state,composite_trajectory_work,0.0,0.005,nothing,
        0.05,composite_hazard_limit)
end)
composite_batch=CompositeTrajectoryBatchWorkspace(
    composite_trajectory_plan,composite_state;workers=1)
measure("composite trajectory batch",()->quantum_trajectories(
    composite_trajectory_plan,composite_state,[0.0,0.02],8;
    dt=0.005,seed=105,workspace=composite_batch);samples=1)

identity_map=identity_channel(b);channel_output=PIState(b)
measure("in-place PI channel",()->apply_channel!(
    channel_output,identity_map,rho))

lowering=collective_operator(b,sm)
correlation_plan=measure("quantum-regression plan",()->CorrelationPlan(
    prepared,adjoint(lowering),lowering);samples=1)
correlation_work=CorrelationWorkspace(
    correlation_plan;krylovdim=30,mode=:time)
correlation_full_work=CorrelationWorkspace(correlation_plan;krylovdim=30)
@printf("  Correlation workspace: time-only %.3f MiB, full %.3f MiB\n",
        Base.summarysize(correlation_work)/2.0^20,
        Base.summarysize(correlation_full_work)/2.0^20)
correlation_delays=[0.0,0.002,0.004]
correlation_output=zeros(ComplexF64,length(correlation_delays))
measure("prepared two-time correlation",()->two_time_correlation!(
    correlation_output,correlation_plan,rho,correlation_delays;
    steps_per_interval=2,workspace=correlation_work))

floquet_map=Matrix{ComplexF64}(I,length(b),length(b))
floquet_map[1,2]=0.001
measure("Floquet repeated-vector evolution",()->floquet_evolve(
    rho,floquet_map,8))
measure("stroboscopic saved evolution",()->stroboscopic_evolution(
    rho,floquet_map,8))
spin=spin_matrices()
population_model=qubit_ensemble_model(b;
    hamiltonian=spin.jz,emission=0.4,dephasing=0.1,pumping=0.07,
    collective_emission=0.02)
population_plan=measure("certified population plan",()->PopulationPlan(
    population_model);samples=1)
population_source=diagonal_populations(rho)
population_output=similar(population_source)
population_work=PopulationWorkspace(population_plan,population_source)
measure("population application",()->apply!(population_output,
    population_plan,population_source,0.0,nothing,population_work))
measure("low-storage population RK4",()->evolve_populations!(
    population_output,population_plan,population_source,(0.0,1e-3);
    steps=1,workspace=population_work))
coherent=spin_coherent_state(b,0.8,0.3)
measure("spin Husimi-Q grid",()->spin_husimi_q(
    coherent;ntheta=31,nphi=60);samples=1)
measure("spin Wigner grid",()->spin_wigner(
    coherent;ntheta=31,nphi=60);samples=1)
sigma=ComplexF64[0.62 0.08-0.03im;0.08+0.03im 0.38]
meanfield_plan=measure("large-N mean-field plan",()->MeanFieldPlan(10^8,2,
    [CollectiveJump(sm;rate=0.4/10^8),
     PBodyHamiltonian(kron(sx,sx),2;rate=0.2/10^8)]);samples=1)
meanfield_work=MeanFieldWorkspace(meanfield_plan,sigma);meanfield_out=similar(sigma)
measure("preallocated mean-field RHS",()->meanfield_rhs!(meanfield_out,
    meanfield_plan,sigma,0.0,nothing,meanfield_work))
measure("collective moments",()->collective_moments(rho,sx))
geometry=measure("reusable observable geometry",()->OneBodyGeometry(b);samples=1)
measure("collective moments (cached)",()->collective_moments(rho,sx;cache=geometry))
observable_plan=measure("prepared collective observable",()->CollectiveObservablePlan(b,sx;cache=geometry);samples=1)
measure("collective moments (prepared)",()->collective_moments(rho,observable_plan))
measure("collective covariance matrix",()->collective_covariance_matrix(rho,[sx,ComplexF64[0 -im;im 0],ComplexF64[1 0;0 -1]]))
measure("covariance matrix (cached)",()->collective_covariance_matrix(rho,[sx,ComplexF64[0 -im;im 0],ComplexF64[1 0;0 -1]];cache=geometry))
measure("quantum Fisher information",()->qfi(rho,sx))
measure("QFI (cached)",()->qfi(rho,sx;cache=geometry))
measure("von Neumann entropy",()->von_neumann_entropy(rho))
measure("five-particle reduced state",()->reduced_state(rho,5);samples=1)
reduction_plan=measure("five-particle reduction plan",()->ReductionPlan(b,5);samples=1)
reduction_setup_basis=PIBasis(16,2)
measure("qubit reduction plan (N=16)",()->ReductionPlan(
    reduction_setup_basis,8);samples=1)
measure("reduced state (prepared)",()->reduced_state(rho,5;plan=reduction_plan))
reduction_work=ReductionWorkspace(reduction_plan,rho)
reduction_only_work=ReductionWorkspace(reduction_plan,rho;mode=:reduction)
negativity_only_work=ReductionWorkspace(reduction_plan,rho;mode=:negativity)
@printf("  Reduction workspace: both %.3f MiB, reduction %.3f MiB, negativity %.3f MiB\n",
        Base.summarysize(reduction_work)/2.0^20,
        Base.summarysize(reduction_only_work)/2.0^20,
        Base.summarysize(negativity_only_work)/2.0^20)
reduction_out=PIState(reduction_plan.output_basis)
measure("reduced state (in-place workspace)",()->reduced_state!(
    reduction_out,rho,reduction_plan,reduction_work))
measure("bipartite negativity",()->negativity(rho,5);samples=1)
measure("negativity (prepared)",()->negativity(rho,5;plan=reduction_plan))

krylov_workspace=KrylovWorkspace(prepared,30)
measure("matrix-free GMRES steady state",()->steady_state(prepared;method=:krylov,
    workspace=krylov_workspace,krylovdim=30,maxiter=300,atol=1e-10,rtol=1e-8))
arnoldi_workspace=ArnoldiWorkspace(
    prepared,min(length(b),30);mode=:ordinary)
arnoldi_full_workspace=ArnoldiWorkspace(prepared,min(length(b),30))
@printf("  Arnoldi workspace: ordinary %.3f MiB, full %.3f MiB\n",
        Base.summarysize(arnoldi_workspace)/2.0^20,
        Base.summarysize(arnoldi_full_workspace)/2.0^20)
measure("matrix-free Arnoldi spectrum",()->krylov_liouvillian_spectrum(prepared;
    nev=4,krylovdim=min(length(b),30),workspace=arnoldi_workspace,
    require_convergence=false))
seed=randn(MersenneTwister(91),ComplexF64,length(b))
measure("matrix-free harmonic Arnoldi",()->harmonic_arnoldi_spectrum(prepared;
    nev=3,krylovdim=min(length(b),30),maxrestarts=2,
    initial_vector=seed,workspace=arnoldi_full_workspace,
    require_convergence=false))
schur=measure("Schur preconditioner setup",()->schur_sector_preconditioner(
    prepared,b;expected_reuses=20,warn_unamortized=false);samples=1)
@printf("  Schur setup: %d L applications, %d coefficients, suggested reuse >= %d\n",
    schur.metadata.setup_liouvillian_applications,schur.metadata.stored_coefficients,
    schur.metadata.recommended_minimum_reuses)
preconditioner_rhs=randn(MersenneTwister(92),ComplexF64,length(b));preconditioner_out=similar(preconditioner_rhs)
measure("Schur preconditioner apply",()->ldiv!(preconditioner_out,schur,preconditioner_rhs))
sz=ComplexF64[1 0;0 -1];projector=matrixfree_symmetry_projector(b,sz)
projector_work=SymmetryProjectorWorkspace(projector)
measure("symmetry projection (workspace)",()->apply!(y,projector,rho.data,projector_work))
strong_model=PIModel(b,(LocalHamiltonian(sz;rate=0.2),
                        LocalJump(sz;rate=0.05)))
strong_source=liouvillian(strong_model;representation=:sparse)
strong_restriction=diagonal_symmetry_restriction(
    b,sz;charge=1,label=:global_z_parity)
strong_operator=measure("compressed symmetry setup",()->
    RestrictedLiouvillian(strong_source,strong_restriction);samples=1)
strong_input=randn(MersenneTwister(93),ComplexF64,length(strong_restriction))
strong_output=similar(strong_input)
measure("compressed symmetry application",()->mul!(
    strong_output,strong_operator,strong_input))
@printf("  Strong-symmetry coordinates: %d / %d; ambient scratch retained: %s\n",
        length(strong_restriction),length(b),
        string(strong_operator.compatibility_workspace!==nothing))

strong_prepared=compile(strong_model;backend=:matrixfree)
strong_lowered=measure("lowered symmetry setup",()->
    RestrictedLiouvillian(strong_prepared,strong_restriction);samples=1)
strong_lowered_work=RestrictedLiouvillianWorkspace(strong_lowered)
measure("lowered symmetry application",()->apply!(
    strong_output,strong_lowered,strong_input,strong_lowered_work))
@printf("  Lowered symmetry scratch: %.3f MiB; ambient vectors: %s\n",
        Base.summarysize(strong_lowered_work)/2.0^20,
        string(hasproperty(strong_lowered_work,:ambient_input)))

bp=PIBasis(6,2);pair=kron(sm,sm);mp=PIModel(bp,[LocalPBodyJump(pair,2)])
Lmp=measure("p-body matrix-free construction",()->liouvillian(mp;representation=:matrixfree);samples=1)
rhop=iid_pure_state(bp,ComplexF64[0,1]);yp=similar(rhop.data)
measure("p-body matrix-free application",()->mul!(yp,Lmp,rhop.data))
pbody_setup_basis=PIBasis(6,3)
coefficient_cache=measure("one-box CG cache d=3 depth=3",()->
    OneBoxCGCache(pbody_setup_basis;max_depth=3,T=Float64);samples=1)
@printf("  One-box coefficients: %d / %d candidates, estimated %.3f MiB\n",
        coefficient_cache.coefficient_count,coefficient_cache.candidate_count,
        Float64(coefficient_cache.estimated_bytes)/2.0^20)
onebody_uncached=measure("one-body geometry d=3 uncached",()->
    OneBodyGeometry(pbody_setup_basis;T=Float64);samples=1)
onebody_cached=measure("one-body geometry d=3 cached",()->
    OneBodyGeometry(pbody_setup_basis;T=Float64,
        coefficient_cache=coefficient_cache);samples=1)
pbody_uncached=measure("p-body geometry d=3 p=3 uncached",()->
    PBodyGeometry(pbody_setup_basis,3;T=Float64);samples=1)
pbody_cached=measure("p-body geometry d=3 p=3 cached",()->
    PBodyGeometry(pbody_setup_basis,3;T=Float64,
        coefficient_cache=coefficient_cache);samples=1)
@printf("  Cached p-body geometry retained: %.3f MiB\n",
        (Base.summarysize(pbody_cached)-Base.summarysize(pbody_setup_basis))/2.0^20)

# Benchmark comparisons remain guarded by numerical equivalence; the cache is
# a setup optimization and must not alter a coefficient or Appendix-D path.
qutrit_probe=ComplexF64[0 1 0;1 0 0;0 0 0.5]
onebody_reference=collective_operator(
    pbody_setup_basis,qutrit_probe;cache=onebody_uncached)
onebody_reused=collective_operator(
    pbody_setup_basis,qutrit_probe;cache=onebody_cached)
isapprox(onebody_reference.data,onebody_reused.data;atol=0,rtol=0)||error(
    "precision guard failed: cached and uncached one-body geometry differ")
keys(pbody_uncached.isometries)==keys(pbody_cached.isometries)&&
    all(key->pbody_uncached.isometries[key]==pbody_cached.isometries[key],
        keys(pbody_uncached.isometries))||error(
    "precision guard failed: cached and uncached p-body geometry differ")

ys=Ls*rho.data;ym=Lm*rho.data
isapprox(ys,ym;atol=1e-11,rtol=1e-11)||error("precision guard failed: sparse and matrix-free actions differ")
apply!(y,prepared,rho.data,0.0,nothing,liouvillian_work)
isapprox(ys,y;atol=1e-11,rtol=1e-11)||error("precision guard failed: compiled-plan action differs")
population_full=liouvillian(population_model;representation=:sparse)*rho.data
population_reduced=population_plan*population_source
isapprox(population_reduced,diagonal_populations(PIState(b,population_full));
    atol=1e-11,rtol=1e-11)||error(
    "precision guard failed: reduced and full population actions differ")
strong_indices=retained_indices(strong_restriction)
isapprox(strong_output,
    strong_source[strong_indices,strong_indices]*strong_input;
    atol=1e-11,rtol=1e-11)||error(
    "precision guard failed: compressed strong-symmetry action differs")
isapprox(strong_lowered*strong_input,
    strong_source[strong_indices,strong_indices]*strong_input;
    atol=1e-11,rtol=1e-11)||error(
    "precision guard failed: lowered strong-symmetry action differs")
rs=time_evolve(Ls,rho,(0.0,0.02);steps=20)
rm=time_evolve(Lm,rho,(0.0,0.02);steps=20)
isapprox(rs.data,rm.data;atol=1e-11,rtol=1e-11)||error("precision guard failed: evolution paths differ")
println("Precision guards: passed")
