using LinearAlgebra
using PermutationalInvariantDynamics
include("paper_models.jl")
using .PaperModels

# Finite-size version of the qubit cut through Figs. 5-6.
N=8; d=2; V=1.0; gammaC=0.2
for gammaI in (1.0,1.8,2.6)
    model=pausch2024_model(N,d;V=V,gammaI=gammaI,gammaC=gammaC)
    prepared=compile(model;backend=:sparse)
    steady=stationary_state(prepared;algorithm=DirectAlgorithm(),return_info=true)
    gap=pi_liouvillian_gap(prepared;method=:dense)
    s=spin_matrices(d)
    Jz=CollectiveObservablePlan(model.basis,s.jz)
    Z=real(collective_expectation(steady.state,Jz))/(N*s.j)
    Zmf=gammaI+gammaC<2abs(V) ? -gammaI/(2abs(V)-gammaC) : -1.0 # Eq. (11)
    println("(gammaI+gammaC)/|V|=$((gammaI+gammaC)/abs(V)), gap=$gap, Z=$Z, mean-field Z=$Zmf")
end
