using LinearAlgebra
using PermutationalInvariantDynamics

include(joinpath(@__DIR__, "utils", "makie_support.jl"))
using .ExampleMakie

# A qutrit product state sampled along a one-parameter family of local
# Hermitian generators. The plan is prepared once and reused for two states.
N = 3
basis = PIBasis(N, 3)
sigma = ComplexF64[0.62 0.08 0.02im;
                   0.08 0.28 0.03;
                  -0.02im 0.03 0.10]
rho = iid_state(basis, sigma)
mixed = maximally_mixed_state(basis)

lambda3 = ComplexF64[1 0 0; 0 -1 0; 0 0 0]
lambda1 = ComplexF64[0 1 0; 1 0 0; 0 0 0]
angles = collect(range(0.0, 2pi; length=121))
generators = [cos(angle) * lambda3 + sin(angle) * lambda1
              for angle in angles]
plan = QuditHusimiPlan(basis, generators; representation=:generator)
q = qudit_husimi_q(rho, plan; resolved=true)
q_mixed = qudit_husimi_q(mixed, plan)

@assert diagnostics(rho).valid
@assert minimum(q.values) > -2e-13
@assert maximum(abs.(q.values .- vec(sum(q.sector_values; dims=1)))) < 2e-14
@assert abs(sum(q.populations) - 1) < 2e-13
@assert maximum(abs.(q_mixed.values .- 1)) < 3e-13

# Qubit sanity check: normalized SU(2)-Haar Q equals 4pi times the sphere
# density returned by spin_husimi_q at the corresponding coherent direction.
qubit_basis = PIBasis(1, 2)
qubit = computational_product_state(qubit_basis, 2)
theta = 0.73
phi = 0.0
generator = theta * spin_matrices(2).jy
qudit_point = qudit_husimi_q(
    qubit, [generator]; representation=:generator)
spin_point = spin_husimi_q(qubit, [theta], [phi])
@assert isapprox(qudit_point.values[1],4pi * spin_point.values[1, 1];
                 atol=3e-13,rtol=0)

println("qutrit sectors: ", q.sectors)
println("sector populations: ", q.populations)
println("generalized Q range: ", extrema(q.values))
println("qubit Haar/sphere sanity error: ",
        abs(qudit_point.values[1] - 4pi * spin_point.values[1, 1]))

if makie_available()
    M = makie_module()
    figure = M.Figure(size=(900, 470), fontsize=16)
    axis = M.Axis(figure[1, 1]; xlabel="generator angle",
        ylabel="Haar-normalized Q", title="Generalized qutrit Husimi data")
    M.lines!(axis, angles, q.values; linewidth=2.7, color=:black,
             label="aggregate")
    for (sector_index, sector) in pairs(q.sectors)
        M.lines!(axis, angles, view(q.sector_values, sector_index, :);
                 linewidth=1.5, label=string(sector.parts))
    end
    M.axislegend(axis; position=:rt, labelsize=11)
    save_example_figure(figure, "qudit_husimi")
end
