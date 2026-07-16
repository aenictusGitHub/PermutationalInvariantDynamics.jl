using LinearAlgebra
using PermutationalInvariantDynamics
include("paper_models.jl")
using .PaperModels
include(joinpath(@__DIR__, "utils", "makie_support.jl"))
using .ExampleMakie

# F. Iemini et al., PRL 121, 035301 (2018), Eq. (2) and Figs. 2-4.
kappa = 1.0
sizes = (8, 12, 16)
cases = NamedTuple[]

for ratio in (0.5, 1.5)
    gaps = Float64[]
    modes = ComplexF64[]
    for N in sizes
        model = iemini2018_btc_model(
            N; omega0=ratio * kappa, kappa=kappa)
        prepared = compile(model; backend=:sparse)
        # The complete small-sector spectrum is needed to identify the slow
        # oscillatory branch, so dense spectral validation is intentional.
        vals = liouvillian_spectrum(
            prepared; target=:largest_real,
            nev=pi_dimension(prepared), algorithm=:dense)
        nonzero = sort(filter(z -> abs(z) > 1e-9, vals);
                       by=z -> real(z), rev=true)
        oscillatory = filter(z -> abs(imag(z)) > 1e-7, nonzero)
        slowosc = isempty(oscillatory) ? NaN + 0im :
            oscillatory[argmax(real.(oscillatory))]
        push!(gaps, -real(nonzero[1]))
        push!(modes, slowosc)
        println("omega0/kappa=$ratio, N=$N: gap=", gaps[end],
                ", slow oscillatory mode=$slowosc")
    end
    ratio > 1 && @assert gaps[end] < gaps[1]
    push!(cases, (; ratio, sizes, gaps, modes))
end

if makie_available()
    M = makie_module()
    figure = M.Figure(size=(1350, 430), fontsize=17)
    gap_axis = M.Axis(
        figure[1, 1]; xlabel="1 / N", ylabel="gap / κ",
        title="Liouvillian gap")
    damping_axis = M.Axis(
        figure[1, 2]; xlabel="1 / N", ylabel="−Re λosc / κ",
        title="Oscillatory-mode damping")
    frequency_axis = M.Axis(
        figure[1, 3]; xlabel="1 / N", ylabel="|Im λosc| / κ",
        title="Oscillation frequency")

    colors = (:royalblue, :firebrick)
    markers = (:circle, :rect)
    for (case, color, marker) in zip(cases, colors, markers)
        inverse_sizes = 1.0 ./ collect(case.sizes)
        label = "ω₀ / κ = $(case.ratio)"
        damping = [-real(mode) / kappa for mode in case.modes]
        frequency = [abs(imag(mode)) / kappa for mode in case.modes]
        M.lines!(gap_axis, inverse_sizes, case.gaps ./ kappa;
                 color, linewidth=2.7)
        M.scatter!(gap_axis, inverse_sizes, case.gaps ./ kappa;
                   color, marker, markersize=11, label)
        M.lines!(damping_axis, inverse_sizes, damping;
                 color, linewidth=2.7)
        M.scatter!(damping_axis, inverse_sizes, damping;
                   color, marker, markersize=11)
        M.lines!(frequency_axis, inverse_sizes, frequency;
                 color, linewidth=2.7)
        M.scatter!(frequency_axis, inverse_sizes, frequency;
                   color, marker, markersize=11)
    end
    M.axislegend(gap_axis; position=:lt, labelsize=12)

    save_example_figure(figure, "iemini2018_boundary_time_crystal")
end
