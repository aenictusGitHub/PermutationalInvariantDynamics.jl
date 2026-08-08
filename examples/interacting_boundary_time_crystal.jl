using LinearAlgebra
using PermutationalInvariantDynamics
include("paper_models.jl")
using .PaperModels
include(joinpath(@__DIR__, "utils", "makie_support.jl"))
using .ExampleMakie

# A. Piccitto et al., Phys. Rev. B 104, 014307 (2021),
# p = 2, q = 1 model and the oscillatory regime of Figs. 6 and 8.
omega_z = 1.0
omega_x = 3omega_z
Gamma_up = 0.2omega_z
Gamma_down = 0.0
quick_example = get(ENV, "PID_EXAMPLE_QUICK", "0") == "1"

function slow_oscillatory_mode(values; stationary_tol=1e-9, frequency_tol=1e-7)
    modes = filter(z -> abs(z) > stationary_tol && abs(imag(z)) > frequency_tol,
                   values)
    isempty(modes) && error("no oscillatory Liouvillian mode was resolved")
    modes[argmax(real.(modes))]
end

# The normal figure uses nine even sizes and extends the checked curve to
# N=24.  The original three-size calculation remains available as a quick
# smoke run; it keeps the same complete-spectrum branch identification.
sizes = quick_example ? (8, 12, 16) : Tuple(8:2:24)
decay_rates = Float64[]
frequencies = Float64[]

for N in sizes
    model = interacting_boundary_time_crystal_model(
        N; omega_z=omega_z, omega_x=omega_x,
        Gamma_up=Gamma_up, Gamma_down=Gamma_down)
    prepared = compile(model; backend=:sparse)

    # The complete spectrum remains affordable for this bounded research grid
    # and lets us identify the slow complex-conjugate branch without assuming
    # that it is the first nonstationary mode.
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

# This is a finite-N check, not a fit of the asymptotic exponent reported in
# the paper. It nevertheless resolves the expected movement of the
# oscillatory branch toward the imaginary axis over nine default sizes.
@assert all(isfinite, decay_rates) && all(>(0), decay_rates)
@assert decay_rates[end] < decay_rates[1]
println("decay ratio N=$(sizes[end]) / N=$(sizes[1]) = ",
        decay_rates[end] / decay_rates[1])
println("resolved frequencies = ", frequencies)

if makie_available()
    M = makie_module()
    plotted_sizes = collect(sizes)
    figure = M.Figure(size=(1050, 450), fontsize=17)
    decay_axis = M.Axis(
        figure[1, 1]; xlabel="particle number N", ylabel="decay rate -Re(λ)",
        title="Slow oscillatory-mode decay")
    frequency_axis = M.Axis(
        figure[1, 2]; xlabel="particle number N", ylabel="frequency |Im(λ)|",
        title="Slow oscillatory-mode frequency")

    M.lines!(decay_axis, plotted_sizes, decay_rates;
             color=:firebrick, linewidth=2.7)
    M.scatter!(decay_axis, plotted_sizes, decay_rates;
               color=:firebrick, markersize=8, label="finite-N PI spectrum")
    M.axislegend(decay_axis; position=:rt, labelsize=12)

    M.lines!(frequency_axis, plotted_sizes, frequencies;
             color=:royalblue, linewidth=2.7)
    M.scatter!(frequency_axis, plotted_sizes, frequencies;
               color=:royalblue, markersize=8, label="finite-N PI spectrum")
    M.axislegend(frequency_axis; position=:rb, labelsize=12)

    M.Label(
        figure[2, 1:2],
        "Nine default sizes through N=24 resolve the finite-size trend; " *
        "no asymptotic exponent is fitted.";
        fontsize=14, color=:gray35, tellwidth=false)
    save_example_figure(figure, "interacting_boundary_time_crystal")
end
