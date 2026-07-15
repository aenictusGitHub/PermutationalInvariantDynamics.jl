using PermutationalInvariantDynamics
include("paper_models.jl")
using .PaperModels

N=8;chi=0.3;sx=ComplexF64[0 1;1 0]
model=kitagawa1993_oat_model(N;chi=chi);b=model.basis
rho0=iid_pure_state(b,ComplexF64[1,1]/sqrt(2))
times=range(0,1.5;length=31)
prepared=compile(model;backend=:matrixfree)
solution=solve_dynamics(prepared,rho0,(first(times),last(times));
                        saveat=times,steps_per_interval=64)
geometry=OneBodyGeometry(b)
Jx=CollectiveObservablePlan(b,sx/2;cache=geometry)
numeric=[collective_expectation(rho,Jx) for rho in solution]
exact=[kitagawa1993_mean_spin_exact(N,t;chi=chi) for t in times]
@assert diagnostics(last(solution);atol=1e-10,rtol=1e-10).valid
println("Kitagawa--Ueda OAT maximum |<Jx>-exact| = ",maximum(abs.(numeric.-exact)))

# Product-Schur data for the fixed one-particle marginal are also reusable.
one_body=ReductionPlan(b,1)
println("final one-particle purity = ",reduced_purity(last(solution),1;plan=one_body))
