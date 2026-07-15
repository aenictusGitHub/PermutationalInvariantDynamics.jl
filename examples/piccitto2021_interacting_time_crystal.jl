using LinearAlgebra
using PermutationalInvariantDynamics
include("paper_models.jl")
using .PaperModels

# A. Piccitto et al., Phys. Rev. B 104, 014307 (2021),
# p = 2, q = 1 model and the oscillatory regime of Figs. 6 and 8.
omega_z = 1.0
omega_x = 3omega_z
Gamma_up = 0.2omega_z
Gamma_down = 0.0

function slow_oscillatory_mode(values; stationary_tol=1e-9, frequency_tol=1e-7)
    modes = filter(z -> abs(z) > stationary_tol && abs(imag(z)) > frequency_tol,
                   values)
    isempty(modes) && error("no oscillatory Liouvillian mode was resolved")
    modes[argmax(real.(modes))]
end

sizes = (8, 12, 16)
decay_rates = Float64[]
frequencies = Float64[]

for N in sizes
    model = piccitto2021_interacting_btc_model(
        N; omega_z=omega_z, omega_x=omega_x,
        Gamma_up=Gamma_up, Gamma_down=Gamma_down)
    prepared = compile(model; backend=:sparse)

    # The complete spectrum is affordable in the fully symmetric sector at
    # these sizes and lets us identify the slow complex-conjugate branch
    # without assuming that it is the first nonstationary mode.
    values = liouvillian_spectrum(
        prepared; target=:largest_real, nev=pi_dimension(prepared),
        algorithm=:dense)
    mode = slow_oscillatory_mode(values)
    decay = -real(mode)
    frequency = abs(imag(mode))
    push!(decay_rates, decay)
    push!(frequencies, frequency)

    conjugate_error = minimum(abs.(values .- conj(mode)))
    @assert conjugate_error < 1e-8
    println("N=$N: slow oscillatory mode=$mode, decay=$decay, ",
            "frequency=$frequency")
end

# This is only a modest-size finite-N check, not a fit of the asymptotic
# exponent reported in the paper. It nevertheless resolves the expected
# movement of the oscillatory branch toward the imaginary axis.
@assert all(isfinite, decay_rates) && all(>(0), decay_rates)
@assert decay_rates[end] < decay_rates[1]
println("decay ratio N=$(sizes[end]) / N=$(sizes[1]) = ",
        decay_rates[end] / decay_rates[1])
println("resolved frequencies = ", frequencies)
