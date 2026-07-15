using PermutationalInvariantDynamics
include("paper_models.jl")
using .PaperModels

N=12; gamma=0.4; sx=ComplexF64[0 1;1 0]
model=huelga1997_dephasing_model(N;gamma=gamma); b=model.basis
rho0=iid_pure_state(b,ComplexF64[1,1]/sqrt(2))
times=range(0,4;length=25)
prepared=compile(model;backend=:matrixfree)
solution=solve_dynamics(prepared,rho0,(first(times),last(times));
                        saveat=times,steps_per_interval=64)
Jx=CollectiveObservablePlan(b,sx/2)
numeric=[collective_expectation(rho,Jx) for rho in solution]
exact=[huelga1997_ramsey_exact(N,t;gamma=gamma) for t in times]
@assert diagnostics(last(solution)).valid
println("Huelga Ramsey-dephasing maximum error = ",maximum(abs.(numeric.-exact)))
