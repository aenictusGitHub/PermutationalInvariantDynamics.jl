using LinearAlgebra
using PermutationalInvariantDynamics
include("paper_models.jl")
using .PaperModels

N=10; down=1.0; up=0.3; model=shammah2018_thermal_model(N;down=down,up=up)
prepared=compile(model)
result=stationary_state(prepared;return_info=true)
numeric=result.state
exact=shammah2018_thermal_state(model.basis;down=down,up=up)
@assert result.info.converged
@assert diagnostics(numeric).valid
println("Shammah local pump/emission steady-state error = ",norm(numeric.data-exact.data))
