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
w=EvolutionWorkspace(Lm,rho)
measure("preallocated RK4 step",()->evolve!(y,Lm,rho.data,(0.0,1e-3);steps=1,workspace=w))
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
measure("preallocated population RK4",()->evolve_populations!(
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
measure("reduced state (prepared)",()->reduced_state(rho,5;plan=reduction_plan))
reduction_work=ReductionWorkspace(reduction_plan,rho)
reduction_out=PIState(reduction_plan.output_basis)
measure("reduced state (in-place workspace)",()->reduced_state!(
    reduction_out,rho,reduction_plan,reduction_work))
measure("bipartite negativity",()->negativity(rho,5);samples=1)
measure("negativity (prepared)",()->negativity(rho,5;plan=reduction_plan))

krylov_workspace=KrylovWorkspace(prepared,30)
measure("matrix-free GMRES steady state",()->steady_state(prepared;method=:krylov,
    workspace=krylov_workspace,krylovdim=30,maxiter=300,atol=1e-10,rtol=1e-8))
arnoldi_workspace=ArnoldiWorkspace(prepared,min(length(b),30))
measure("matrix-free Arnoldi spectrum",()->krylov_liouvillian_spectrum(prepared;
    nev=4,krylovdim=min(length(b),30),workspace=arnoldi_workspace,
    require_convergence=false))
seed=randn(MersenneTwister(91),ComplexF64,length(b))
measure("matrix-free harmonic Arnoldi",()->harmonic_arnoldi_spectrum(prepared;
    nev=3,krylovdim=min(length(b),30),maxrestarts=2,
    initial_vector=seed,workspace=arnoldi_workspace,require_convergence=false))
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

bp=PIBasis(6,2);pair=kron(sm,sm);mp=PIModel(bp,[LocalPBodyJump(pair,2)])
Lmp=measure("p-body matrix-free construction",()->liouvillian(mp;representation=:matrixfree);samples=1)
rhop=iid_pure_state(bp,ComplexF64[0,1]);yp=similar(rhop.data)
measure("p-body matrix-free application",()->mul!(yp,Lmp,rhop.data))

ys=Ls*rho.data;ym=Lm*rho.data
isapprox(ys,ym;atol=1e-11,rtol=1e-11)||error("precision guard failed: sparse and matrix-free actions differ")
apply!(y,prepared,rho.data,0.0,nothing,liouvillian_work)
isapprox(ys,y;atol=1e-11,rtol=1e-11)||error("precision guard failed: compiled-plan action differs")
population_full=liouvillian(population_model;representation=:sparse)*rho.data
population_reduced=population_plan*population_source
isapprox(population_reduced,diagonal_populations(PIState(b,population_full));
    atol=1e-11,rtol=1e-11)||error(
    "precision guard failed: reduced and full population actions differ")
rs=time_evolve(Ls,rho,(0.0,0.02);steps=20)
rm=time_evolve(Lm,rho,(0.0,0.02);steps=20)
isapprox(rs.data,rm.data;atol=1e-11,rtol=1e-11)||error("precision guard failed: evolution paths differ")
println("Precision guards: passed")
