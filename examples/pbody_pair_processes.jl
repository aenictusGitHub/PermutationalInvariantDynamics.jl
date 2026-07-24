using LinearAlgebra
using PermutationalInvariantDynamics

include(joinpath(@__DIR__, "utils", "makie_support.jl"))
using .ExampleMakie

N = 6
basis = PIBasis(N, 2)
sm = ComplexF64[0 1; 0 0]
sz = ComplexF64[1 0; 0 -1]
pair_loss = kron(sm, sm)
pair_interaction = kron(sz, sz)

# Appendix D represents the unordered-pair sum directly. Verify the familiar
# identity sum_{i<j} sz_i sz_j = ((sum_i sz_i)^2-N I)/2.
pair_geometry = PBodyGeometry(basis, 2)
pair_sum = pbody_collective_operator(
    basis, pair_interaction, 2; cache=pair_geometry)
Jz = collective_operator(basis, sz)
pair_sum_reference = (Jz * Jz - N * identity_operator(basis)) * (1 / 2)
identity_error = norm(pair_sum.data - pair_sum_reference.data)
packing = pair_geometry.estimates

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
println("Appendix-D path entries (packed/dense): ",
        packing.retained_entries, " / ", packing.dense_entries)
println("pair-sum identity error: ",identity_error)
println("matrix-free/sparse action error: ", action_error)
println("initial trace derivative: ", trace_derivative)

@assert packing.storage === :exact_support_sparse_csc
@assert packing.retained_entries < packing.dense_entries
@assert identity_error < 1e-10
@assert action_error < 1e-10
@assert trace_derivative < 1e-10

if makie_available()
    M=makie_module()
    figure=M.Figure(size=(1050,430),fontsize=17)
    storage_axis=M.Axis(
        figure[1,1];xlabel="Appendix-D representation",
        ylabel="stored path entries",
        xticks=([1,2],["exact-support CSC","dense reference"]),
        title="Prepared p-body geometry")
    validation_axis=M.Axis(
        figure[1,2];xlabel="validation quantity",ylabel="absolute error",
        yscale=log10,
        xticks=(
            1:3,
            ["operator","action","trace"],
        ),
        title="Backend agreement (display floor ε)")

    M.barplot!(
        storage_axis,[1,2],
        Float64[packing.retained_entries,packing.dense_entries];
        color=[:dodgerblue3,:gray55],
        strokecolor=:black,strokewidth=0.6)
    M.scatterlines!(
        validation_axis,1:3,
        max.([identity_error,action_error,trace_derivative],eps(Float64));
        color=:firebrick3,linewidth=2,markersize=10)
    save_example_figure(figure,"pbody_pair_processes")
end
