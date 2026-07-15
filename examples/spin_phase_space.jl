using PermutationalInvariantDynamics

# A genuinely multi-sector PI state: a coherent state in j=N/2 and a Dicke
# state uniformly repeated over the multiplicity copies of j=N/2-1.
N = 4
basis = PIBasis(N, 2)
coherent_weight = 0.68
theta0 = 0.74
phi0 = -0.36
coherent = spin_coherent_state(basis, theta0, phi0)
lower_spin = dicke_state(basis, N / 2 - 1, 0)
rho = PIState(
    basis,
    coherent_weight .* coherent.data .+
    (1 - coherent_weight) .* lower_spin.data,
)

@assert diagnostics(rho).valid

# The default regular grid omits the duplicate phi=2pi endpoint. Individual
# sector matrices are retained because this example visualizes them below.
q = spin_husimi_q(rho; ntheta=81, nphi=160, resolved=true)
w = spin_wigner(rho; ntheta=81, nphi=160, resolved=true)

@assert q.sectors == basis.sectors
@assert q.multiplicities == BigInt[1, 3, 2]
@assert maximum(abs.(q.populations .-
    [coherent_weight, 1 - coherent_weight, 0])) < 2e-13
@assert maximum(abs.(w.populations .- q.populations)) < 2e-13
@assert maximum(abs.(q.values .- sum(q.sector_values))) < 2e-15
@assert maximum(abs.(w.values .- sum(w.sector_values))) < 2e-15
@assert minimum(w.values) < 0

# At the coherent direction, the normalized symmetric-sector Q density is
# p*(N+1)/(4pi). This independently checks the coherent-state convention and
# the sector population prefactor.
peak = spin_husimi_q(
    rho, [theta0], [phi0]; sectors=Partition((N, 0)))
@assert peak.values[1, 1] ≈ coherent_weight * (N + 1) / (4pi) atol=2e-13

q_figure = visualize_spin_phase_space(
    q; title="Multi-sector Husimi-Q marginal")
w_figure = visualize_spin_phase_space(
    w; title="Multi-sector spin-Wigner marginal")
symmetric_wigner_figure = visualize_spin_phase_space(
    w; sector=Partition((N, 0)), title="Symmetric-sector spin Wigner")

# Exercise the dependency-free writers without leaving generated repository
# artifacts behind.
mktempdir() do directory
    for (filename, figure) in (
        "husimi_q.svg" => q_figure,
        "wigner_marginal.svg" => w_figure,
        "wigner_symmetric_sector.svg" => symmetric_wigner_figure,
    )
        path = joinpath(directory, filename)
        save_spin_phase_space_visualization(path, figure)
        @assert isfile(path)
        @assert occursin("<svg", read(path, String))
    end
    println("rendered three temporary spin phase-space SVG files")
end

println("Schur sectors: ", q.sectors)
println("twice-spin labels: ", q.twice_spins)
println("exact multiplicities: ", q.multiplicities)
println("sector populations: ", q.populations)
println("Husimi-Q range: ", extrema(q.values))
println("spin-Wigner range: ", extrema(w.values))
