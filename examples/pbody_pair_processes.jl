using LinearAlgebra
using PermutationalInvariantDynamics

N = 6
basis = PIBasis(N, 2)
sm = ComplexF64[0 1; 0 0]
sz = ComplexF64[1 0; 0 -1]
pair_loss = kron(sm, sm)
pair_interaction = kron(sz, sz)

# Appendix D represents the unordered-pair sum directly. Verify the familiar
# identity sum_{i<j} sz_i sz_j = ((sum_i sz_i)^2-N I)/2.
pair_sum = pbody_collective_operator(basis, pair_interaction, 2)
Jz = collective_operator(basis, sz)
pair_sum_reference = (Jz * Jz - N * identity_operator(basis)) * (1 / 2)
identity_error = norm(pair_sum.data - pair_sum_reference.data)

model = PIModel(basis, [PBodyHamiltonian(pair_interaction, 2; rate=0.05),
                        LocalPBodyJump(pair_loss, 2; rate=0.02),
                        CollectivePBodyJump(pair_loss, 2; rate=0.001)])
rho0 = iid_pure_state(basis, ComplexF64[0, 1])

# Explicit backend access is intentional here: the example validates the
# preallocated matrix-free Appendix-D kernel against sparse assembly.
prepared = compile(model; backend=:matrixfree)
workspace = LiouvillianWorkspace(prepared)
Ls = liouvillian(prepared; representation=:sparse)
y = similar(rho0.data)
apply!(y, prepared, rho0.data, 0.0, nothing, workspace)
derivative = PIState(basis, copy(y))
action_error = norm(y - Ls * rho0.data)
trace_derivative = abs(trace(derivative))
compiled_report = diagnostics(prepared)

println("N=$N pair-process PI dimension: ", length(basis),
        "; full density-matrix entries: ", 2^(2N))
println("prepared backend: ", compiled_report.backend,
        "; retained bytes: ", compiled_report.retained_bytes)
println("pair-sum identity error: ",identity_error)
println("matrix-free/sparse action error: ", action_error)
println("initial trace derivative: ", trace_derivative)

@assert identity_error < 1e-10
@assert action_error < 1e-10
@assert trace_derivative < 1e-10
