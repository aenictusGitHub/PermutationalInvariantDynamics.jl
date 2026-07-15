using LinearAlgebra
using PermutationalInvariantDynamics

# A complete N=3 qubit PI spectrum has dimension 20, so dense diagonalization
# is inexpensive and gives every mode rather than a selected Krylov window.
N = 3
basis = PIBasis(N, 2)
sx = ComplexF64[0 1; 1 0]
sm = ComplexF64[0 1; 0 0]

autonomous_model = PIModel(basis, [
    CollectiveHamiltonian(sx / 2; rate=0.8),
    LocalJump(sm; rate=0.35),
])
autonomous = compile(autonomous_model; backend=:sparse)
liouvillian_data = liouvillian_spectrum_data(
    autonomous;
    target=:largest_real,
    nev=pi_dimension(autonomous),
    algorithm=:dense,
)

liouvillian_values = liouvillian_data.values
stationary_index = argmin(abs.(liouvillian_values))
nonstationary = liouvillian_values[eachindex(liouvillian_values) .!= stationary_index]

@assert liouvillian_data.kind == :liouvillian
@assert liouvillian_data.representation == :eigenvalues
@assert liouvillian_data.complete
@assert abs(liouvillian_values[stationary_index]) < 2e-12
@assert maximum(real.(liouvillian_values)) < 2e-12
@assert maximum(real.(nonstationary)) < -1e-4

liouvillian_figure = visualize_spectrum(
    liouvillian_data;
    title="N=3 driven-decaying Liouvillian spectrum",
    show_indices=true,
)

# A fixed collective operator with a scalar periodic rate exercises the
# preallocated time-dependent kernel. The one-period map is convergence-tested
# before its spectrum is interpreted.
period = 2.0
parameters = (omega=0.9, modulation=0.45)
drive_rate = (t, p) -> p.omega *
    (1 + p.modulation * cos(2pi * t / period))
periodic_model = PIModel(basis, [
    CollectiveHamiltonian(sx / 2; rate=drive_rate),
    LocalJump(sm; rate=0.35),
])
periodic = compile(periodic_model; backend=:matrixfree)

Fcoarse = floquet_propagator(
    periodic, period; steps=128, parameters=parameters)
F = floquet_propagator(
    periodic, period; steps=256, parameters=parameters)
propagator_error = norm(F - Fcoarse)
@assert propagator_error < 1.5e-7

# Diagonalize the converged map once. Reuse the stored multipliers through the
# principal logarithm; rendering either data object performs no further
# eigensolve or propagation.
multiplier_data = floquet_spectrum_data(
    F;
    period=period,
    representation=:multipliers,
)
multipliers = multiplier_data.values
exponent_data = floquet_spectrum_data(
    multipliers;
    input=:multipliers,
    period=period,
    representation=:exponents,
)

fixed_multiplier = multipliers[argmin(abs.(multipliers .- 1))]
principal_exponents = log.(multipliers) ./ period
principal_log_error = maximum(
    minimum(abs(exponent - reference) for reference in principal_exponents)
    for exponent in exponent_data.values)

@assert multiplier_data.kind == :floquet
@assert multiplier_data.representation == :multipliers
@assert exponent_data.representation == :exponents
@assert multiplier_data.complete
@assert abs(fixed_multiplier - 1) < 2e-10
@assert maximum(abs.(multipliers)) < 1 + 2e-10
@assert minimum(abs.(exponent_data.values)) < 2e-10
@assert maximum(real.(exponent_data.values)) < 2e-10
@assert maximum(abs.(imag.(exponent_data.values))) <= pi / period + 2e-12
@assert principal_log_error < 2e-12

multiplier_figure = visualize_spectrum(
    multiplier_data;
    title="Floquet multipliers (unit disk)",
)
exponent_figure = visualize_spectrum(
    exponent_data;
    title="Floquet exponents (principal branch)",
)

# Generated figures are deliberately temporary: running an example must not
# leave derived artifacts in the source tree.
mktempdir() do directory
    figures = (
        "liouvillian_spectrum.svg" => liouvillian_figure,
        "floquet_multipliers.svg" => multiplier_figure,
        "floquet_exponents.svg" => exponent_figure,
    )
    for (filename, figure) in figures
        path = joinpath(directory, filename)
        @assert save_spectrum_visualization(path, figure) == path
        svg = read(path, String)
        @assert startswith(lstrip(svg), "<svg")
        @assert occursin("</svg>", svg)
    end
    println("rendered three temporary spectral SVGs (directory removed on exit)")
end

println("PI dimension: ", pi_dimension(autonomous))
println("Liouvillian stationary eigenvalue: ",
        liouvillian_values[stationary_index])
println("slowest nonstationary real part: ", maximum(real.(nonstationary)))
println("Floquet step-doubling error: ", propagator_error)
println("Floquet fixed multiplier: ", fixed_multiplier)
println("principal-log exponent matching error: ", principal_log_error)
