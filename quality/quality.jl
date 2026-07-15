using Aqua
using JET
using LinearAlgebra
using PermutationalInvariantDynamics
using Test

const PID = PermutationalInvariantDynamics

@testset "Aqua package quality" begin
    Aqua.test_all(PID)
end

# JET's compiler integration is Julia-version specific, so this focused gate
# runs in the dedicated Julia 1.12 quality environment. Concrete public calls
# give JET substantially more useful type information than a whole-package
# scan of deliberately generic method signatures.
@testset "JET public hot paths" begin
    basis = PIBasis(3, 2)
    rho = iid_pure_state(basis, ComplexF64[1, 1] ./ sqrt(2))
    sx = ComplexF64[0 1; 1 0]
    sm = ComplexF64[0 1; 0 0]
    model = PIModel(basis, [
        LocalHamiltonian(sx; rate=0.2),
        LocalJump(sm; rate=0.1),
    ])
    prepared = compile(model; backend=:matrixfree)
    workspace = LiouvillianWorkspace(prepared)
    destination = similar(rho.data)

    JET.@test_call target_modules=(PID,) purity(rho)
    JET.@test_call target_modules=(PID,) collective_expectation(rho, sx)
    JET.@test_call target_modules=(PID,) apply!(destination, prepared,
                                                rho.data, 0.0, nothing,
                                                workspace)
end
