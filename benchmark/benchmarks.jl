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
SUITE["matrix-free apply N=20 d=2"] = @benchmarkable mul!($y,$L,$rho.data)
krylov_work=KrylovWorkspace(L,40)
arnoldi_work=ArnoldiWorkspace(L,40)
arnoldi_seed=randn(MersenneTwister(71),ComplexF64,length(b))
SUITE["reused GMRES N=20 d=2"] = @benchmarkable krylov_steady_state($L;
    basis=$b,workspace=$krylov_work,krylovdim=40,maxiter=500)
SUITE["reused Arnoldi N=20 d=2"] = @benchmarkable krylov_liouvillian_spectrum($L;
    nev=6,krylovdim=40,initial_vector=$arnoldi_seed,workspace=$arnoldi_work,
    require_convergence=false)
SUITE["reused harmonic Arnoldi N=20 d=2"] = @benchmarkable harmonic_arnoldi_spectrum($L;
    nev=4,krylovdim=40,maxrestarts=3,initial_vector=$arnoldi_seed,
    workspace=$arnoldi_work,require_convergence=false)
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
SUITE["preallocated RK4 step N=20 d=2"] = @benchmarkable evolve!($y,$L,$rho.data,(0.0,0.01);steps=1,workspace=$w)
SUITE["collective moments N=20 d=2"] = @benchmarkable collective_moments($rho,$sx)
SUITE["QFI N=20 d=2"] = @benchmarkable qfi($rho,$sx)
SUITE["entropy N=20 d=2"] = @benchmarkable von_neumann_entropy($rho)
bt = PIBasis(6,2); rt = iid_pure_state(bt,ComplexF64[0,1]); mt = PIModel(bt,[LocalJump(sm)])
SUITE["100 PI trajectories N=6 d=2"] = @benchmarkable quantum_trajectories($mt,$rt,[0.0,0.2],100;dt=0.01,seed=1)

bq = PIBasis(8,3); rhoq = maximally_mixed_state(bq)
SUITE["qudit reduced state N=8 d=3"] = @benchmarkable reduced_state($rhoq,4)
