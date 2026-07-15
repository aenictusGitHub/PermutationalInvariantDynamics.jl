using LinearAlgebra
using PermutationalInvariantDynamics
include("paper_models.jl")
using .PaperModels

# F. Iemini et al., PRL 121, 035301 (2018), Eq. (2) and Figs. 2-4.
kappa=1.0
for ratio in (0.5,1.5)
    gaps=Float64[];modes=ComplexF64[]
    for N in (8,12,16)
        model=iemini2018_btc_model(N;omega0=ratio*kappa,kappa=kappa)
        prepared=compile(model;backend=:sparse)
        # The complete small-sector spectrum is needed to identify the slow
        # oscillatory branch, so dense spectral validation is intentional.
        vals=liouvillian_spectrum(prepared;target=:largest_real,
                                  nev=pi_dimension(prepared),algorithm=:dense)
        nonzero=sort(filter(z->abs(z)>1e-9,vals);by=z->real(z),rev=true)
        oscillatory=filter(z->abs(imag(z))>1e-7,nonzero)
        slowosc=isempty(oscillatory) ? NaN+0im : oscillatory[argmax(real.(oscillatory))]
        push!(gaps,-real(nonzero[1]));push!(modes,slowosc)
        println("omega0/kappa=$ratio, N=$N: gap=",gaps[end],", slow oscillatory mode=$slowosc")
    end
    ratio>1&&@assert gaps[end]<gaps[1]
end
