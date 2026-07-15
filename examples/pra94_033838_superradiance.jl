using LinearAlgebra
import Pkg
Pkg.add(url="https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl")
using PermutationalInvariantDynamics
include("paper_models.jl")
using .PaperModels

# Reproduce Fig. 6 and its analytical Eqs. (41)-(43).
times=range(0,3;length=121); gamma0=1.0
for gamma in (gamma0,0.75gamma0,0.0)
    model=damanet2016_model(2;gamma0=gamma0,gamma=gamma)
    b=model.basis; rho0=iid_pure_state(b,ComplexF64[0,1])
    prepared=compile(model;backend=:sparse)
    # Dense exponentiation is intentional here: it is the independent
    # small-N validation route used for the pointwise analytical comparison.
    L=Matrix(liouvillian(prepared))
    Iop=damanet2016_intensity_operator(b;gamma0=gamma0,gamma=gamma)
    numeric=[real(expectation(PIState(b,exp(t*L)*rho0.data),Iop)) for t in times]
    exact=[damanet2016_intensity_exact(t;gamma0=gamma0,gamma=gamma) for t in times]
    println("gamma/gamma0=$(gamma/gamma0): max |numeric - Eq.(41)| = ",maximum(abs.(numeric-exact)))
end
