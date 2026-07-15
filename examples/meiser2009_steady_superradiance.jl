using LinearAlgebra
using PermutationalInvariantDynamics
include("paper_models.jl")
using .PaperModels

# D. Meiser and M. J. Holland, PRA 81, 033847 (2010), Eqs. (1),(2),(8)-(10).
N=10;GammaC=1.0;sm=ComplexF64[0 1;0 0];excited=ComplexF64[0 0;0 1]
for pump in (0.1,1.0,N*GammaC/2,N*GammaC,30.0)
    model=meiser2009_superradiance_model(N;GammaC=GammaC,pump=pump)
    prepared=compile(model)
    rho=stationary_state(prepared;algorithm=DirectAlgorithm())
    Jm=collective_operator(model.basis,sm)
    Neplan=CollectiveObservablePlan(model.basis,excited)
    intensity=GammaC*real(expectation(rho,adjoint(Jm)*Jm))
    Ne=real(collective_expectation(rho,Neplan))
    enhancement=intensity/(max(Ne,eps())*GammaC)
    println("w/GammaC=$(pump/GammaC): I/GammaC=$intensity, Ne=$Ne, I/(Ne GammaC)=$enhancement")
end
println("large-N prediction at w=N GammaC/2: Imax/GammaC = ",N^2/8)
