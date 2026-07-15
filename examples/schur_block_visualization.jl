using LinearAlgebra
using PermutationalInvariantDynamics

# The complete N=4 qubit PI basis has the three Schur sectors
# (4,0), (3,1), and (2,2), with irrep dimensions 5, 3, and 1.
N = 4
basis = PIBasis(N, 2)

# Independent pumping and emission have an exact full-rank iid stationary
# state. Solve for it rather than constructing the displayed state by hand,
# then retain the analytical product state as an independent oracle.
sm = ComplexF64[0 1; 0 0]
sp = ComplexF64[0 0; 1 0]
down = 0.4
up = 0.15
steady_model = PIModel(basis, [
    LocalJump(sm; rate=down),
    LocalJump(sp; rate=up),
])
steady_prepared = compile(steady_model; backend=:sparse)
steady = stationary_state(
    steady_prepared; algorithm=DirectAlgorithm(), return_info=true)
rho = steady.state
sigma = ComplexF64[down / (down + up) 0; 0 up / (down + up)]
exact = iid_state(basis, sigma)

@assert steady.info.converged
@assert diagnostics(rho).valid
@assert norm(rho.data - exact.data) < 2e-12

# Population mode reports the physical trace carried by every Schur sector.
# The density spectrum is kept multiplicity-compressed: one eigenvalue per
# irrep eigenvector plus its exact symmetric-group degeneracy.
state_structure = schur_block_structure(
    rho; metric=:population, threshold=1e-13)
density_spectrum = pi_density_spectrum(rho)

# A collective jump acts independently inside every Schur sector. The
# unresolved sum of local jumps is still PI, but it transfers weight between
# sectors. Probe both through their matrix-free compiled kernels.
collective_model = PIModel(basis, [CollectiveJump(sm; rate=0.2)])
local_model = PIModel(basis, [LocalJump(sm; rate=0.2)])
collective = compile(collective_model; backend=:matrixfree)
local_decay = compile(local_model; backend=:matrixfree)

collective_structure = schur_block_structure(
    collective.operator, basis; metric=:frobenius, threshold=1e-12)
local_structure = schur_block_structure(
    local_decay.operator, basis; metric=:frobenius, threshold=1e-12)

function has_active_offdiagonal(structure)
    n = length(structure.sectors)
    any(structure.active[i, j] for i in 1:n for j in 1:n if i != j)
end

function largest_offdiagonal_weight(structure)
    n = length(structure.sectors)
    maximum((structure.weights[i, j]
             for i in 1:n for j in 1:n if i != j); init=0.0)
end

@assert state_structure.kind == :state
@assert state_structure.block_dimensions == (5, 3, 1)
@assert state_structure.coordinate_dimensions == (25, 9, 1)
@assert all(state_structure.weights[i, i] > 0 for i in eachindex(basis.sectors))
@assert sum(diag(state_structure.weights)) ≈ 1 atol=2e-12
@assert !has_active_offdiagonal(state_structure)
@assert length(density_spectrum.values) == sum(state_structure.block_dimensions)
@assert density_spectrum.total_dimension == big(2)^N
@assert abs(density_spectrum.trace - 1) < big"2e-12"
@assert density_spectrum.minimum > -2e-12
@assert !has_active_offdiagonal(collective_structure)
@assert has_active_offdiagonal(local_structure)

state_figure = visualize_schur_blocks(
    state_structure; title="Thermal steady-state sector populations",
    scale=:linear, show_values=true, show_young_diagrams=true)
density_figure = visualize_density_spectrum(
    density_spectrum; title="Thermal steady-state density spectrum",
    show_degeneracies=true)
collective_figure = visualize_schur_blocks(
    collective_structure; title="Collective decay: sector diagonal",
    scale=:log, show_young_diagrams=true)
local_figure = visualize_schur_blocks(
    local_structure; title="Local decay: Schur-sector coupling", scale=:log,
    show_young_diagrams=true)

# Keep generated artifacts out of the repository while still exercising the
# complete SVG-writing path.
mktempdir() do directory
    schur_figures = (
        "state.svg" => state_figure,
        "collective_liouvillian.svg" => collective_figure,
        "local_liouvillian.svg" => local_figure,
    )
    for (filename, figure) in schur_figures
        path = joinpath(directory, filename)
        save_schur_block_visualization(path, figure)
        @assert isfile(path)
        @assert occursin("<svg", read(path, String))
    end

    density_path = joinpath(directory, "steady_state_density_spectrum.svg")
    @assert save_density_spectrum_visualization(
        density_path, density_figure) == density_path
    @assert isfile(density_path)
    @assert occursin("<svg", read(density_path, String))

    println("rendered four temporary SVG files (directory removed on exit)")
end

println("Schur sectors: ", state_structure.sectors)
println("irrep dimensions: ", state_structure.block_dimensions)
println("PI-coordinate dimensions: ", state_structure.coordinate_dimensions)
println("steady-state sector populations: ", diag(state_structure.weights))
println("compressed density eigenvalues: ", density_spectrum.values)
println("exact density-eigenvalue degeneracies: ", density_spectrum.degeneracies)
println("collective off-diagonal block norm: ",
        largest_offdiagonal_weight(collective_structure))
println("local off-diagonal block norm: ",
        largest_offdiagonal_weight(local_structure))
