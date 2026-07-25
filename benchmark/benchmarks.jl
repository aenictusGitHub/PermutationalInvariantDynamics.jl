using PermutationalInvariantDynamics
using BenchmarkTools
using LinearAlgebra
using Random

SUITE = BenchmarkGroup()
SUITE["basis N=50 d=2"] = @benchmarkable PIBasis(50, 2)
b = PIBasis(20, 2); sm = ComplexF64[0 1; 0 0]
m = PIModel(b, [LocalJump(sm)])
SUITE["sparse assembly N=20 d=2"] = @benchmarkable liouvillian($m; representation=:sparse)
L = liouvillian(m; representation=:matrixfree)
rho = iid_pure_state(b, ComplexF64[0,1]); y = similar(rho.data)
w = EvolutionWorkspace(rho)
sx = ComplexF64[0 1;1 0]
SUITE["compatibility matrix-free mul! N=20 d=2"] =
    @benchmarkable mul!($y,$L,$rho.data)
prepared_model=compile(m;backend=:matrixfree)
prepared_work=LiouvillianWorkspace(prepared_model)
SUITE["prepared matrix-free apply N=20 d=2"] = @benchmarkable apply!(
    $y,$prepared_model,$rho.data,0.0,nothing,$prepared_work)

# The sole `(N,0)` sector has a dedicated occupation-number collective lift.
# Keep its reusable plan setup, sparse-first assembly, fixed sparse/matrix-free
# actions, driven preallocated action, and observable preparation visible as
# separate measurements.  N=64 is large enough to expose setup asymptotics
# while remaining a short local benchmark.
collective_fast_basis=PIBasis(64,2;sectors=[(64,0)])
collective_fast_model=PIModel(collective_fast_basis,(
    CollectiveHamiltonian(sx;rate=0.13),
    CollectiveJump(sm;rate=0.27)))
SUITE["symmetric collective plan setup N=64 d=2"] = @benchmarkable LiouvillianPlan(
    $collective_fast_model)
collective_fast_plan=LiouvillianPlan(collective_fast_model)
SUITE["symmetric sparse-first assembly N=64 d=2"] = @benchmarkable PermutationalInvariantDynamics._matrix_from_plan(
    $collective_fast_plan)
SUITE["symmetric auto compilation N=64 d=2"] = @benchmarkable compile(
    $collective_fast_model;backend=:auto,memory_budget=512*1024^2)
collective_fast_sparse=PermutationalInvariantDynamics._matrix_from_plan(
    collective_fast_plan)
collective_fast_compiled=compile(
    collective_fast_model;backend=:matrixfree,memory_budget=Inf)
collective_fast_work=LiouvillianWorkspace(collective_fast_compiled)
collective_fast_input=randn(
    MersenneTwister(0xc011ec71),ComplexF64,length(collective_fast_basis))
collective_fast_input./=norm(collective_fast_input)
collective_fast_output=similar(collective_fast_input)
SUITE["symmetric matrix-free apply N=64 d=2"] = @benchmarkable apply!(
    $collective_fast_output,$collective_fast_compiled,$collective_fast_input,
    0.0,nothing,$collective_fast_work)
SUITE["symmetric sparse apply N=64 d=2"] = @benchmarkable mul!(
    $collective_fast_output,$collective_fast_sparse,$collective_fast_input)

collective_fast_h_schedule=let h0=copy(sx),h1=ComplexF64[1 0;0 -1]
    InPlaceTimeOperator(h0,(destination,time,parameters)->begin
        @. destination=cos(time)*h0+parameters.mix*sin(time)*h1
        nothing
    end)
end
collective_fast_l_schedule=let l0=copy(sm),l1=copy(adjoint(sm))
    InPlaceTimeOperator(l0,(destination,time,parameters)->begin
        @. destination=l0+parameters.mix*time*l1
        nothing
    end)
end
collective_fast_driven_model=PIModel(collective_fast_basis,(
    CollectiveHamiltonian(collective_fast_h_schedule;rate=0.13),
    CollectiveJump(collective_fast_l_schedule;rate=0.27)))
SUITE["driven symmetric collective plan N=64 d=2"] = @benchmarkable LiouvillianPlan(
    $collective_fast_driven_model)
collective_fast_driven_plan=LiouvillianPlan(collective_fast_driven_model)
collective_fast_driven_work=LiouvillianWorkspace(
    collective_fast_driven_plan)
collective_fast_parameters=(mix=0.31,)
SUITE["driven symmetric collective apply N=64 d=2"] = @benchmarkable apply!(
    $collective_fast_output,$collective_fast_driven_plan,
    $collective_fast_input,0.23,$collective_fast_parameters,
    $collective_fast_driven_work)

SUITE["symmetric collective observable setup N=64 d=2"] = @benchmarkable CollectiveObservablePlan(
    $collective_fast_basis,$sx)
collective_fast_state=iid_pure_state(
    collective_fast_basis,ComplexF64[0,1])
collective_fast_observable=CollectiveObservablePlan(
    collective_fast_basis,sx)
SUITE["symmetric collective moments prepared N=64 d=2"] = @benchmarkable collective_moments(
    $collective_fast_state,$collective_fast_observable)

# Fixed local gains now retain rectangular Schur contractions.  The explicit
# four-index triplet plan below is the former implementation and remains a
# useful setup/storage/application oracle for performance audits.
factor_basis=PIBasis(8,2)
factor_model=PIModel(factor_basis,(LocalJump(sm;rate=0.2),))
factor_plan=LiouvillianPlan(factor_model)
factor_geometry=OneBodyGeometry(factor_basis)
factor_q=[collective_block(factor_basis,adjoint(sm)*sm,sector;
                           cache=factor_geometry)
          for sector in factor_basis.sectors]
factor_triplets=PermutationalInvariantDynamics._local_kernel_triplets(
    factor_basis,factor_geometry,sm,sm)
factor_reference_kernel=PermutationalInvariantDynamics.LocalJumpPIKernel(
    factor_q,factor_triplets,0.2)
factor_reference_plan=LiouvillianPlan(factor_basis,(factor_reference_kernel,),
    copy(factor_plan.tracevec),nothing,factor_plan.Ttype,true)
factor_input=randn(MersenneTwister(0xfa01),ComplexF64,length(factor_basis))
factor_output=similar(factor_input)
factor_work=LiouvillianWorkspace(factor_plan)
factor_reference_work=LiouvillianWorkspace(factor_reference_plan)
SUITE["factorized fixed local gain N=8 d=2"] = @benchmarkable apply!(
    $factor_output,$factor_plan,$factor_input,$factor_work)
SUITE["triplet fixed local gain oracle N=8 d=2"] = @benchmarkable apply!(
    $factor_output,$factor_reference_plan,$factor_input,$factor_reference_work)

fusion_model=PIModel(factor_basis,(
    LocalHamiltonian(sx;rate=0.2),
    CollectiveHamiltonian(ComplexF64[1 0;0 -1];rate=0.1),
    LocalJump(sm;rate=0.2),CollectiveJump(sm;rate=0.05)))
fusion_plan=LiouvillianPlan(fusion_model)
fusion_reference_plan=LiouvillianPlan(fusion_model;fuse_static=false)
fusion_work=LiouvillianWorkspace(fusion_plan)
fusion_reference_work=LiouvillianWorkspace(fusion_reference_plan)
SUITE["fused fixed kernels N=8 d=2"] = @benchmarkable apply!(
    $factor_output,$fusion_plan,$factor_input,$fusion_work)
SUITE["termwise fixed kernels oracle N=8 d=2"] = @benchmarkable apply!(
    $factor_output,$fusion_reference_plan,$factor_input,$fusion_reference_work)
threaded_liouvillian_work=ThreadedLiouvillianWorkspace(
    L.plan;tasks=Threads.nthreads())
SUITE["target-sector threaded apply N=20 d=2"] = @benchmarkable threaded_apply!(
    $y,$L.plan,$rho.data,$threaded_liouvillian_work)
SUITE["target-sector threaded adjoint N=20 d=2"] = @benchmarkable threaded_apply_adjoint!(
    $y,$L.plan,$rho.data,$threaded_liouvillian_work)
krylov_work=KrylovWorkspace(L,40)
arnoldi_work=ArnoldiWorkspace(L,40;mode=:ordinary)
harmonic_arnoldi_work=ArnoldiWorkspace(L,40)
arnoldi_seed=randn(MersenneTwister(71),ComplexF64,length(b))
SUITE["reused GMRES N=20 d=2"] = @benchmarkable krylov_steady_state($L;
    basis=$b,workspace=$krylov_work,krylovdim=40,maxiter=500)
SUITE["reused Arnoldi N=20 d=2"] = @benchmarkable krylov_liouvillian_spectrum($L;
    nev=6,krylovdim=40,initial_vector=$arnoldi_seed,workspace=$arnoldi_work,
    require_convergence=false)
SUITE["reused harmonic Arnoldi N=20 d=2"] = @benchmarkable harmonic_arnoldi_spectrum($L;
    nev=4,krylovdim=40,maxrestarts=3,initial_vector=$arnoldi_seed,
    workspace=$harmonic_arnoldi_work,require_convergence=false)
SUITE["Schur preconditioner setup N=20 d=2"] = @benchmarkable schur_sector_preconditioner(
    $L,$b;expected_reuses=20,warn_unamortized=false)
schur=schur_sector_preconditioner(L,b;expected_reuses=20,warn_unamortized=false)
schur_rhs=randn(MersenneTwister(72),ComplexF64,length(b));schur_out=similar(schur_rhs)
SUITE["Schur preconditioner apply N=20 d=2"] = @benchmarkable ldiv!(
    $schur_out,$schur,$schur_rhs)
sz=ComplexF64[1 0;0 -1];projector=matrixfree_symmetry_projector(b,sz)
projector_work=SymmetryProjectorWorkspace(projector)
SUITE["symmetry projector workspace N=20 d=2"] = @benchmarkable apply!(
    $y,$projector,$rho.data,$projector_work)
SUITE["low-storage RK4 step N=20 d=2"] = @benchmarkable evolve!($y,$L,$rho.data,(0.0,0.01);steps=1,workspace=$w)
floquet_period=0.2
floquet_h_rate=let period=floquet_period
    (time,parameters)->0.07sin(2pi*time/period)
end
floquet_jump_rate=let period=floquet_period
    (time,parameters)->0.2+0.03cos(2pi*time/period)
end
floquet_model=PIModel(b,(
    LocalHamiltonian(sx;rate=floquet_h_rate),
    LocalJump(sm;rate=floquet_jump_rate)))
SUITE["matrix-free FloquetMap setup N=20 steps=16"] =
    @benchmarkable floquet_map(
        $floquet_model,$floquet_period;steps=16)
prepared_floquet=floquet_map(
    floquet_model,floquet_period;steps=16)
floquet_work=FloquetWorkspace(prepared_floquet)
floquet_input=copy(rho.data)
floquet_output=similar(floquet_input)
SUITE["preallocated Floquet period N=20 steps=16"] =
    @benchmarkable apply!(
        $floquet_output,$prepared_floquet,$floquet_input,$floquet_work)
floquet_batch_input=hcat(
    floquet_input,0.5floquet_input,complex.(reverse(floquet_input)))
floquet_batch_output=similar(floquet_batch_input)
floquet_batch_work=FloquetBatchWorkspace(
    prepared_floquet,3;mode=:forward)
SUITE["preallocated Floquet batch width=3 N=20 steps=16"] =
    @benchmarkable apply!(
        $floquet_batch_output,$prepared_floquet,
        $floquet_batch_input,$floquet_batch_work)
SUITE["collective moments N=20 d=2"] = @benchmarkable collective_moments($rho,$sx)
SUITE["QFI N=20 d=2"] = @benchmarkable qfi($rho,$sx)
SUITE["entropy N=20 d=2"] = @benchmarkable von_neumann_entropy($rho)
analysis_observable=CollectiveObservablePlan(b,sx)
SUITE["prepared collective moments N=20 d=2"] =
    @benchmarkable collective_moments($rho,$analysis_observable)
SUITE["prepared QFI N=20 d=2"] =
    @benchmarkable qfi($rho,$analysis_observable)
negativity_basis=PIBasis(10,2)
negativity_state=maximally_mixed_state(negativity_basis)
negativity_plan=ReductionPlan(negativity_basis,5)
negativity_work=ReductionWorkspace(
    negativity_plan,negativity_state;mode=:negativity)
SUITE["public prepared qubit negativity N=10 k=5"] =
    @benchmarkable negativity(
        $negativity_state,5;
        plan=$negativity_plan,workspace=$negativity_work)
SUITE["internal cached qubit negativity oracle N=10 k=5"] = @benchmarkable PermutationalInvariantDynamics._qubit_negativity(
    $negativity_state,5)
SUITE["internal uncached qubit negativity oracle N=10 k=5"] = @benchmarkable PermutationalInvariantDynamics._qubit_negativity(
    $negativity_state,5,nothing)
bt = PIBasis(6,2); rt = iid_pure_state(bt,ComplexF64[0,1]); mt = PIModel(bt,[LocalJump(sm)])
SUITE["100 PI trajectories N=6 d=2"] = @benchmarkable quantum_trajectories($mt,$rt,[0.0,0.2],100;dt=0.01,seed=1)
trajectory_plan=TrajectoryPlan(mt)
trajectory_batch=TrajectoryBatchWorkspace(trajectory_plan,rt;workers=1)
SUITE["100 reused-plan PI trajectories N=6 d=2"] = @benchmarkable quantum_trajectories(
    $trajectory_plan,$rt,[0.0,0.2],100;dt=0.01,seed=1,workspace=$trajectory_batch)

# Composite deterministic and stochastic paths keep every factor action
# matrix-free.  Explicit matrix-RHS and trajectory workspaces expose the
# reusable hot paths separately from their setup.
composite_factor=FiniteOperatorBasis(3;label=:benchmark_auxiliary)
composite_basis=CompositePIBasis(bt,composite_factor)
composite_system=compile(mt;backend=:matrixfree)
composite_term=local_superoperator_term(
    composite_basis,1,composite_system)
SUITE["composite superoperator setup N=6 auxiliary=3"] =
    @benchmarkable CompositeSuperoperator(
        $composite_basis,$composite_term)
composite_operator=CompositeSuperoperator(
    composite_basis,composite_term)
composite_source=randn(
    MersenneTwister(0x434f4d50),ComplexF64,length(composite_basis))
composite_output=similar(composite_source)
composite_work=CompositeSuperoperatorWorkspace(
    composite_operator,composite_source)
SUITE["composite prepared apply N=6 auxiliary=3"] =
    @benchmarkable apply!(
        $composite_output,$composite_operator,$composite_source,
        0.0,nothing,$composite_work)
composite_batch_source=hcat(
    composite_source,0.5composite_source,
    complex.(reverse(composite_source)))
composite_batch_output=similar(composite_batch_source)
composite_batch_work=CompositeSuperoperatorBatchWorkspace(
    composite_operator;capacity=3)
SUITE["composite prepared batch width=3 N=6 auxiliary=3"] =
    @benchmarkable apply!(
        $composite_batch_output,$composite_operator,
        $composite_batch_source,0.0,nothing,$composite_batch_work)
SUITE["composite prepared adjoint batch width=3 N=6 auxiliary=3"] =
    @benchmarkable apply_adjoint!(
        $composite_batch_output,$composite_operator,
        $composite_batch_source,0.0,nothing,$composite_batch_work)
composite_mode_state=zeros(ComplexF64,3,3)
composite_mode_state[1,1]=1
composite_state=composite_tensor_state(
    composite_basis,rt,composite_mode_state)
composite_mode_lowering=ComplexF64[
    0 1 0
    0 0 sqrt(2)
    0 0 0
]
composite_jump=CompositeJumpChannel(
    composite_basis,
    1=>collective_operator(bt,sm),
    2=>composite_mode_lowering;
    rate=0.03)
SUITE["composite trajectory plan N=6 auxiliary=3"] =
    @benchmarkable CompositeTrajectoryPlan(
        $composite_basis,$composite_jump)
composite_trajectory_plan=CompositeTrajectoryPlan(
    composite_basis,composite_jump)
composite_trajectory_batch=CompositeTrajectoryBatchWorkspace(
    composite_trajectory_plan,composite_state;workers=1)
SUITE["16 reused composite trajectories N=6 auxiliary=3"] =
    @benchmarkable quantum_trajectories(
        $composite_trajectory_plan,$composite_state,[0.0,0.02],16;
        dt=0.005,seed=0x434f4d50,
        workspace=$composite_trajectory_batch)

bq = PIBasis(8,3); rhoq = maximally_mixed_state(bq)
SUITE["setup-inclusive qudit reduced state N=8 d=3"] =
    @benchmarkable reduced_state($rhoq,4)
prepared_reduction_basis=PIBasis(3,3)
prepared_reduction_local=ComplexF64[
    0.50 0.06+0.02im 0.03-0.01im
    0.06-0.02im 0.30 0.04+0.01im
    0.03+0.01im 0.04-0.01im 0.20
]
prepared_reduction_state=iid_state(
    prepared_reduction_basis,prepared_reduction_local)
SUITE["packed qudit reduction plan N=3 d=3 k=2"] =
    @benchmarkable ReductionPlan($prepared_reduction_basis,2)
prepared_reduction_plan=ReductionPlan(prepared_reduction_basis,2)
prepared_reduction_work=ReductionWorkspace(
    prepared_reduction_plan,prepared_reduction_state;mode=:reduction)
prepared_reduction_output=PIState(
    prepared_reduction_plan.output_basis)
SUITE["packed qudit reduced_state! N=3 d=3 k=2"] =
    @benchmarkable reduced_state!(
        $prepared_reduction_output,$prepared_reduction_state,
        $prepared_reduction_plan,$prepared_reduction_work;check=false)
local_factor_basis=PIBasis(3,4)
local_factor_ket=ComplexF64[
    sqrt(0.6),0,0,exp(0.3im)*sqrt(0.4)]
local_factor_state=iid_pure_state(
    local_factor_basis,local_factor_ket)
SUITE["local-factor trace plan N=3 dims=(2,2)"] =
    @benchmarkable LocalFactorTracePlan(
        $local_factor_state,(2,2);traced_factor=2)
local_factor_plan=LocalFactorTracePlan(
    local_factor_state,(2,2);traced_factor=2)
local_factor_work=LocalFactorTraceWorkspace(local_factor_plan)
local_factor_output=PIState(local_factor_plan.output_basis)
SUITE["local-factor prepared trace N=3 dims=(2,2)"] =
    @benchmarkable local_factor_trace!(
        $local_factor_output,$local_factor_state,
        $local_factor_plan,$local_factor_work;check=false)

# A basis-owned coefficient cache amortizes exact-rational one-box CG setup
# across several one-body and Appendix-D geometries.  The uncached entries
# retain the call-local construction path for a direct comparison.
coefficient_basis=PIBasis(6,3)
SUITE["one-box CG cache N=6 d=3 depth=3"] = @benchmarkable OneBoxCGCache(
    $coefficient_basis;max_depth=3,T=Float64)
coefficient_cache=OneBoxCGCache(
    coefficient_basis;max_depth=3,T=Float64)
SUITE["one-body geometry uncached N=6 d=3"] = @benchmarkable OneBodyGeometry(
    $coefficient_basis;T=Float64)
SUITE["one-body geometry cached N=6 d=3"] = @benchmarkable OneBodyGeometry(
    $coefficient_basis;T=Float64,coefficient_cache=$coefficient_cache)
SUITE["p-body geometry uncached N=6 d=3 p=3"] = @benchmarkable PBodyGeometry(
    $coefficient_basis,3;T=Float64)
SUITE["p-body geometry cached N=6 d=3 p=3"] = @benchmarkable PBodyGeometry(
    $coefficient_basis,3;T=Float64,coefficient_cache=$coefficient_cache)

# A nontrivial qutrit Appendix-D model keeps packed geometry in the actual
# compiled kernels.  Setup and warmed forward/adjoint actions are separate
# from the primitive geometry benchmarks above.
qutrit_pbody_basis=PIBasis(3,3)
qutrit_lowering=ComplexF64[0 1 0;0 0 1;0 0 0]
qutrit_x=qutrit_lowering+adjoint(qutrit_lowering)
qutrit_pair_hamiltonian=kron(qutrit_x,qutrit_x)
qutrit_pair_jump=kron(qutrit_lowering,qutrit_lowering)
qutrit_pbody_model=PIModel(qutrit_pbody_basis,(
    PBodyHamiltonian(qutrit_pair_hamiltonian,2;rate=0.08),
    LocalPBodyJump(qutrit_pair_jump,2;rate=0.05)))
SUITE["packed qutrit p-body plan N=3 d=3 p=2"] =
    @benchmarkable compile(
        $qutrit_pbody_model;backend=:matrixfree,memory_budget=Inf)
qutrit_pbody_prepared=compile(
    qutrit_pbody_model;backend=:matrixfree,memory_budget=Inf)
qutrit_pbody_work=LiouvillianWorkspace(qutrit_pbody_prepared)
qutrit_pbody_input=randn(
    MersenneTwister(0x5032),ComplexF64,length(qutrit_pbody_basis))
qutrit_pbody_output=similar(qutrit_pbody_input)
SUITE["packed qutrit p-body apply N=3 d=3 p=2"] =
    @benchmarkable apply!(
        $qutrit_pbody_output,$qutrit_pbody_prepared,$qutrit_pbody_input,
        0.0,nothing,$qutrit_pbody_work)
SUITE["packed qutrit p-body adjoint N=3 d=3 p=2"] =
    @benchmarkable apply_adjoint!(
        $qutrit_pbody_output,$qutrit_pbody_prepared,
        $qutrit_pbody_input,0.0,nothing,$qutrit_pbody_work)

# A certified Cartesian parity block lowers the common fixed Schur kernels
# once. Hot response/Krylov applications then use only reduced scratch.
restricted_basis=PIBasis(10,2)
restricted_parity=Diagonal(ComplexF64[1,-1])
restricted_model=PIModel(restricted_basis,(
    LocalHamiltonian(ComplexF64[1 0;0 -1];rate=0.2),
    LocalJump(ComplexF64[1 0;0 -1];rate=0.05)))
restricted_source=compile(restricted_model;backend=:matrixfree)
restricted_coordinates=diagonal_symmetry_restriction(
    restricted_basis,restricted_parity;charge=1)
SUITE["lowered strong-symmetry setup N=10 d=2"] = @benchmarkable RestrictedLiouvillian(
    $restricted_source,$restricted_coordinates)
restricted_operator=RestrictedLiouvillian(
    restricted_source,restricted_coordinates)
restricted_work=RestrictedLiouvillianWorkspace(restricted_operator)
restricted_input=randn(MersenneTwister(0x5e71),ComplexF64,
    length(restricted_coordinates))
restricted_output=similar(restricted_input)
SUITE["lowered strong-symmetry apply N=10 d=2"] = @benchmarkable apply!(
    $restricted_output,$restricted_operator,$restricted_input,$restricted_work)

# Packed hierarchy topology, source-ADO/bath coupling reuse, and shifted-block
# preconditioning. Equal pole frequencies deliberately exercise exact LU reuse.
heom_basis=PIBasis(6,2);heom_spin=spin_matrices()
heom_model=qubit_ensemble_model(heom_basis;emission=0.2)
heom_coupling=collective_operator(heom_basis,heom_spin.jz)
heom_bath=HEOMBath(heom_coupling,[0.18,0.07],[1.1,1.1];
    right_coefficients=[0.18,0.07])
heom_plan=HEOMPlan(heom_model,heom_bath;max_depth=3,scaling=:scaled)
heom_rng=MersenneTwister(0x6e10)
heom_source=randn(heom_rng,ComplexF64,size(heom_plan,1))
heom_destination=similar(heom_source)
heom_ados=heom_number_ados(heom_plan)
SUITE["PI-HEOM plan setup N=6 poles=2 depth=3"] =
    @benchmarkable HEOMPlan(
        $heom_model,$heom_bath;max_depth=3,scaling=:scaled)
SUITE["PI-HEOM workspace setup N=6 poles=2 depth=3"] =
    @benchmarkable HEOMWorkspace($heom_plan;batch_columns=3)
heom_work=HEOMWorkspace(heom_plan;batch_columns=3)
apply!(heom_destination,heom_plan,heom_source,heom_work)
SUITE["packed PI-HEOM apply N=6 poles=2 depth=3 ados=$heom_ados"] = @benchmarkable apply!(
    $heom_destination,$heom_plan,$heom_source,$heom_work)
SUITE["packed PI-HEOM adjoint N=6 poles=2 depth=3 ados=$heom_ados"] =
    @benchmarkable apply_adjoint!(
        $heom_destination,$heom_plan,$heom_source,$heom_work)
heom_batch_source=hcat(
    heom_source,0.5heom_source,complex.(reverse(heom_source)))
heom_batch_destination=similar(heom_batch_source)
SUITE["packed PI-HEOM batch width=3 N=6 poles=2 depth=3"] =
    @benchmarkable apply!(
        $heom_batch_destination,$heom_plan,$heom_batch_source,$heom_work)
SUITE["guarded Schur-shift HEOM preconditioner setup N=6 poles=2 depth=3"] =
    @benchmarkable heom_block_preconditioner($heom_plan;
        operator_scale=1.0,expected_reuses=20,warn_unamortized=false)
heom_preconditioner=heom_block_preconditioner(heom_plan;
    operator_scale=1.0,expected_reuses=20,warn_unamortized=false)
heom_preconditioned=similar(heom_source)
SUITE["guarded Schur-shift HEOM preconditioner apply N=6 poles=2 depth=3"] =
    @benchmarkable ldiv!(
        $heom_preconditioned,$heom_preconditioner,$heom_source)
heom_lu_preconditioner=heom_block_preconditioner(heom_plan;
    operator_scale=1.0,shift_backend=:lu,expected_reuses=20,
    warn_unamortized=false)
SUITE["duplicate-aware LU HEOM preconditioner apply N=6 poles=2 depth=3"] =
    @benchmarkable ldiv!(
        $heom_preconditioned,$heom_lu_preconditioner,$heom_source)

# Local and shared pseudomodes use distinct PI representations.  Keep model
# construction, prepared generator application, and factor trace separate so
# cutoff/setup costs are not attributed to repeated dynamics or observables.
local_mode=BosonicPseudomode(
    1;frequency=0.73,damping=0.31,label=:local_benchmark)
local_coupling=PseudomodeCoupling(
    sm;mode=:local_benchmark,strength=0.16)
SUITE["local pseudomode supersite N=3 nmax=1"] =
    @benchmarkable pseudomode_supersite(3,2,$local_mode)
local_site=pseudomode_supersite(3,2,local_mode)
local_zero_hamiltonian=zeros(ComplexF64,2,2)
SUITE["local pseudomode model N=3 nmax=1"] =
    @benchmarkable pseudomode_model(
        $local_site,$local_zero_hamiltonian;couplings=$local_coupling)
local_embedding=pseudomode_model(
    local_site,local_zero_hamiltonian;couplings=local_coupling)
SUITE["local pseudomode matrix-free compile N=3 nmax=1"] =
    @benchmarkable compile(
        $local_embedding.model;backend=:matrixfree,memory_budget=Inf)
local_prepared=compile(
    local_embedding.model;backend=:matrixfree,memory_budget=Inf)
local_state=pseudomode_product_state(
    local_embedding.supersite,ComplexF64[0,1])
local_output=similar(local_state.data)
local_work=LiouvillianWorkspace(local_prepared)
SUITE["local pseudomode prepared apply N=3 nmax=1"] =
    @benchmarkable apply!(
        $local_output,$local_prepared,$local_state.data,
        0.0,nothing,$local_work)
SUITE["local pseudomode trace plan N=3 nmax=1"] =
    @benchmarkable pseudomode_trace_plan($local_embedding.supersite)
local_trace_plan=pseudomode_trace_plan(local_embedding.supersite)
local_trace_work=LocalFactorTraceWorkspace(local_trace_plan)
local_trace_output=PIState(local_trace_plan.output_basis)
SUITE["local pseudomode prepared trace N=3 nmax=1"] =
    @benchmarkable trace_pseudomodes!(
        $local_trace_output,$local_state,$local_embedding.supersite,
        $local_trace_plan,$local_trace_work;check=false)

global_mode=BosonicPseudomode(
    2;frequency=0.73,damping=0.31,label=:shared_benchmark)
global_coupling=PseudomodeCoupling(
    sm;mode=:shared_benchmark,strength=0.16)
global_system=PIModel(PIBasis(6,2),(
    LocalJump(sm;rate=0.04),))
SUITE["global pseudomode model N=6 nmax=2"] =
    @benchmarkable global_pseudomode_model(
        $global_system,$global_mode;couplings=$global_coupling)
global_embedding=global_pseudomode_model(
    global_system,global_mode;couplings=global_coupling)
global_state=pseudomode_product_state(
    global_embedding,ComplexF64[0,1])
global_output=similar(global_state.data)
global_work=global_pseudomode_workspace(global_embedding)
SUITE["global pseudomode prepared apply N=6 nmax=2"] =
    @benchmarkable apply!(
        $global_output,$global_embedding,$global_state.data,$global_work)
SUITE["global pseudomode prepared adjoint N=6 nmax=2"] =
    @benchmarkable apply_adjoint!(
        $global_output,$global_embedding,$global_state.data,$global_work)
global_system_output=PIState(global_embedding.system_basis)
global_mode_output=zeros(
    ComplexF64,global_mode.levels,global_mode.levels)
SUITE["global pseudomode system trace N=6 nmax=2"] =
    @benchmarkable trace_pseudomodes!(
        $global_system_output,$global_state,$global_embedding)
SUITE["global pseudomode mode trace N=6 nmax=2"] =
    @benchmarkable global_pseudomode_state!(
        $global_mode_output,$global_state,$global_embedding)

# HOPS propagates one direct-sum Schur pseudo-ket per hierarchy node.  The
# conditioned RHS is the hot deterministic kernel; the short ensemble entry
# additionally exposes path/workspace reuse without claiming convergence.
hops_hamiltonian=collective_operator(heom_basis,0.11 .* heom_spin.jx)
hops_coupling=collective_operator(heom_basis,heom_spin.jz)
hops_bath=HOPSBath(
    hops_coupling,[0.18,0.05],[1.1,1.7])
SUITE["PI-HOPS plan setup N=6 poles=2 depth=3"] =
    @benchmarkable HOPSPlan(
        $hops_hamiltonian,$hops_bath;max_depth=3,scaling=:scaled)
hops_plan=HOPSPlan(
    hops_hamiltonian,hops_bath;max_depth=3,scaling=:scaled)
hops_work=HOPSWorkspace(hops_plan)
hops_source=randn(
    MersenneTwister(0x484f5053),ComplexF64,size(hops_plan,1))
hops_destination=similar(hops_source)
hops_noise=ComplexF64[0.03-0.01im]
SUITE["PI-HOPS conditioned RHS N=6 poles=2 depth=3 auxiliaries=$(hops_number_auxiliaries(hops_plan))"] =
    @benchmarkable hops_rhs!(
        $hops_destination,$hops_plan,$hops_source,$hops_noise,$hops_work)
hops_initial=weak_pi_pseudoket(
    iid_pure_state(heom_basis,ComplexF64[1,1]/sqrt(2)))
hops_batch=HOPSBatchWorkspace(hops_plan;workers=1)
hops_times=[0.0,0.01,0.02]
SUITE["PI-HOPS reused 32-path ensemble N=6 depth=3"] =
    @benchmarkable hops_average(
        $hops_plan,$hops_initial,$hops_times,32;
        dt=0.005,seed=0x484f5053,threaded=false,workspace=$hops_batch)

# Parameter-family specialization measures the intended scan setup path:
# fixed Schur geometry is prepared once and only scalar rates are rebound.
family_model=PIModel(b,(
    LocalJump(sm;rate=1.0),CollectiveHamiltonian(sx;rate=0.15)))
compiled_family=compile_family(family_model)
family_specialization_work=LiouvillianWorkspace(compiled_family.plan)
SUITE["compiled-family specialization N=20 d=2"] = @benchmarkable specialize(
    $compiled_family,(0.9,0.17);workspace=$family_specialization_work)
family_point=specialize(compiled_family,(0.9,0.17))
recycled_work=RecycledGMRESWorkspace(ComplexF64,length(b),40,8)
recycled_seed=krylov_steady_state(family_point;basis=b,workspace=recycled_work,
    krylovdim=40,maxiter=500,return_info=true).state
next_family_point=specialize(compiled_family,(0.91,0.171))
SUITE["recycled steady continuation N=20 d=2"] = @benchmarkable krylov_steady_state(
    $next_family_point;basis=$b,initial_state=$recycled_seed,
    workspace=$recycled_work,krylovdim=40,maxiter=500)

# Shared-Arnoldi frequency batches are compared with the historical
# sequential shifted-GMRES route on the same prepared regression plan.
correlation_basis=PIBasis(1,2);correlation_spin=spin_matrices()
correlation_lowering=collective_spin(correlation_basis,:minus)
correlation_model=PIModel(correlation_basis,(
    LocalHamiltonian(correlation_spin.jz;rate=1.3),
    LocalJump(correlation_spin.jm;rate=0.8),
    LocalJump(correlation_spin.jp;rate=0.2)))
correlation_state=iid_state(correlation_basis,ComplexF64[0.8 0;0 0.2])
frequency_plan=CorrelationPlan(compile(correlation_model;backend=:matrixfree),
    adjoint(correlation_lowering),correlation_lowering)
frequency_work=CorrelationWorkspace(frequency_plan;krylovdim=4)
frequency_grid=collect(range(0.0,2.6;length=16))
SUITE["shared multi-shift correlation grid"] = @benchmarkable stationary_correlation_spectrum(
    $frequency_plan,$correlation_state,$frequency_grid;
    workspace=$frequency_work,solver=:multishift)
SUITE["sequential correlation grid"] = @benchmarkable stationary_correlation_spectrum(
    $frequency_plan,$correlation_state,$frequency_grid;
    workspace=$frequency_work,solver=:sequential)
