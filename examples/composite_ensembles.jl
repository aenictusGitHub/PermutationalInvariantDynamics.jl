using LinearAlgebra
using PermutationalInvariantDynamics

include(joinpath(@__DIR__, "utils", "makie_support.jl"))
using .ExampleMakie

# Two independently PI ensembles and one finite two-level auxiliary system.
# Ensemble A already spans the full N=2 PI space, including its j=1 and j=0
# Schur sectors, while the complete example remains intentionally small.
ensemble_a = PIBasis(2, 2)
ensemble_b = PIBasis(1, 2)
auxiliary = FiniteOperatorBasis(2; label=:ancilla)
basis = CompositePIBasis(ensemble_a, ensemble_b, auxiliary)

sm = ComplexF64[0 1; 0 0]
sx = ComplexF64[0 1; 1 0]
sz = ComplexF64[1 0; 0 -1]

rho_a = iid_state(ensemble_a, ComplexF64[0 0; 0 1])
rho_b = iid_state(ensemble_b, ComplexF64[1 0; 0 0])
rho_aux = ComplexF64[1 0; 0 0]
rho0 = composite_tensor_state(basis, rho_a, rho_b, rho_aux)

# The first declared factor is fastest. This assertion is a small, explicit
# statement of the coordinate convention used by all composite kernels.
@assert rho0.data == kron(vec(rho_aux), kron(rho_b.data, rho_a.data))
@assert isapprox(trace(rho0), 1; atol=2e-14)
@assert length(ensemble_a.sectors) == 2

# Compile each ensemble's autonomous local physics exactly as for an ordinary
# single-ensemble calculation, then lift the actions to their factors.
local_a = compile(PIModel(ensemble_a, (LocalJump(sm; rate=0.18),));
                  backend=:matrixfree)
local_b = compile(PIModel(ensemble_b, (LocalJump(sm; rate=0.11),));
                  backend=:matrixfree)
local_terms = CompositeSuperoperator(
    basis,
    local_superoperator_term(basis, 1, local_a),
    local_superoperator_term(basis, 2, local_b),
)

# Coherent coupling H = g Jx^(a) tensor sigma_x^(aux). The second ensemble is
# an identity factor and is therefore absent from the pair list.
Jx_a = collective_operator(ensemble_a, sx)
coherent_coupling = composite_hamiltonian_superoperator(
    basis, 1 => Jx_a, 3 => sx; rate=0.07)

# A correlated jump J_-^(b) tensor sigma_-^(aux). This is one physical
# Lindblad channel, not two independently applied jumps.
Jm_b = collective_operator(ensemble_b, sm)
correlated_loss = composite_dissipator_superoperator(
    basis, 2 => Jm_b, 3 => sm; rate=0.04)

generator = local_terms + coherent_coupling + correlated_loss

# Explicit task-owned scratch gives an allocation-free warmed application.
derivative = similar(rho0.data)
apply_workspace = CompositeSuperoperatorWorkspace(generator, rho0.data)
apply!(derivative, generator, rho0.data, 0.0, nothing, apply_workspace)
trace_vector = composite_trace_vector(basis)
@assert abs(dot(trace_vector, derivative)) < 2e-13

# Block Krylov and sensitivity calculations apply the same generator to
# several right-hand sides. Fixed-capacity scratch batches equal tensor fibres
# across all columns without constructing a global Kronecker matrix.
batch_source = hcat(rho0.data, derivative)
batch_forward = similar(batch_source)
batch_adjoint = similar(batch_source)
batch_workspace = CompositeSuperoperatorBatchWorkspace(
    generator; capacity=size(batch_source, 2))
apply!(
    batch_forward, generator, batch_source, 0.0, nothing, batch_workspace)
apply_adjoint!(
    batch_adjoint, generator, batch_source, 0.0, nothing, batch_workspace)

forward_reference = similar(batch_source)
adjoint_reference = similar(batch_source)
for column in axes(batch_source, 2)
    apply!(
        view(forward_reference, :, column), generator,
        view(batch_source, :, column), 0.0, nothing, apply_workspace)
    apply_adjoint!(
        view(adjoint_reference, :, column), generator,
        view(batch_source, :, column), 0.0, nothing, apply_workspace)
end
batch_forward_error = norm(batch_forward - forward_reference)
batch_adjoint_error = norm(batch_adjoint - adjoint_reference)
@assert batch_forward_error < 2e-13
@assert batch_adjoint_error < 2e-13

# The generic preallocated RK4 evolution discovers a composite workspace and
# reuses the nested workspaces of both local compiled PI actions.
final_data = copy(rho0.data)
evolution_workspace = EvolutionWorkspace(generator, rho0.data)
evolve!(final_data, generator, rho0.data, (0.0, 1.5);
        steps=192, workspace=evolution_workspace)
rho_final = CompositePIState(basis, final_data)
@assert isapprox(trace(rho_final), 1; atol=3e-11)

# A factorized observable: Jz on ensemble A and identities elsewhere.
Jz_a = collective_operator(ensemble_a, sz)
identity_b = identity_operator(ensemble_b)
identity_aux = Matrix{ComplexF64}(I, 2, 2)
observable = composite_tensor_operator(
    basis, Jz_a, identity_b, identity_aux)
initial_signal = real(expectation(rho0, observable))
final_signal = real(expectation(rho_final, observable))

println("Composite coordinate dimension: ", length(basis))
println("Ensemble-A Schur sectors: ", length(ensemble_a.sectors))
println("Initial <Jz_a>: ", initial_signal)
println("Final   <Jz_a>: ", final_signal)
println("Final trace: ", real(trace(rho_final)))
println("Batched forward/adjoint errors: ",
        batch_forward_error, " / ", batch_adjoint_error)

if makie_available()
    M = makie_module()
    figure = M.Figure(size=(1050, 430), fontsize=17)
    signal_axis = M.Axis(
        figure[1, 1];
        xlabel="state", ylabel="⟨Σᵢ σz⁽ᴬ⁾⟩",
        xticks=([1, 2], ["initial", "final"]),
        title="Composite evolution")
    validation_axis = M.Axis(
        figure[1, 2];
        xlabel="validation quantity", ylabel="absolute error",
        yscale=log10,
        xticks=(
            1:4,
            ["tr(ℒρ₀)", "batch", "adjoint", "tr(ρf)−1"],
        ),
        title="Prepared-kernel checks (display floor ε)")

    M.barplot!(
        signal_axis, [1, 2], [initial_signal, final_signal];
        color=[:gray50, :dodgerblue3],
        strokecolor=:black, strokewidth=0.6)
    M.hlines!(signal_axis, [0.0]; color=:gray65, linestyle=:dash)

    validation_errors = [
        abs(dot(trace_vector, derivative)),
        batch_forward_error,
        batch_adjoint_error,
        abs(trace(rho_final) - 1),
    ]
    M.scatterlines!(
        validation_axis, 1:4,
        max.(validation_errors, eps(Float64));
        color=:firebrick3, linewidth=2, markersize=10)
    save_example_figure(figure, "composite_ensembles")
end
