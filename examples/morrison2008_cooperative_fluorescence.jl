using LinearAlgebra
using PermutationalInvariantDynamics
include("paper_models.jl")
using .PaperModels

# S. Morrison and A. S. Parkins, PRA 77, 043810 (2008), Eqs. (1)-(5).
N=20;Omega=0.2
for gamma in (0.1,0.2,0.35)
    model=morrison2008_model(N;Omega=Omega,gamma=gamma)
    prepared=compile(model)
    stationary=stationary_state(prepared;algorithm=DirectAlgorithm(),return_info=true)
    numeric=stationary.state
    exact=morrison2008_exact_state(model.basis;Omega=Omega,gamma=gamma)
    # The package basis is |g>,|e>; the paper's spin axes take |e> as m=+1/2.
    sy=-ComplexF64[0 -im;im 0]/2;sz=-ComplexF64[1 0;0 -1]/2
    geometry=OneBodyGeometry(model.basis)
    Jy=CollectiveObservablePlan(model.basis,sy;cache=geometry)
    Jz=CollectiveObservablePlan(model.basis,sz;cache=geometry)
    Y=real(collective_expectation(numeric,Jy))/(N/2)
    Z=real(collective_expectation(numeric,Jz))/(N/2)
    println("gamma=$gamma: exact-state error=",norm(numeric.data-exact.data),
            ", <Jy>/j=$Y, <Jz>/j=$Z")
    @assert norm(numeric.data-exact.data)<2e-9
end
