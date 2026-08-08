using LinearAlgebra
using PermutationalInvariantDynamics
include("paper_models.jl")
using .PaperModels
include(joinpath(@__DIR__, "utils", "makie_support.jl"))
using .ExampleMakie

# F. Iemini et al., PRL 121, 035301 (2018), Eq. (2) and Figs. 2-4.
kappa = 1.0
quick_example = lowercase(strip(get(ENV, "PID_EXAMPLE_QUICK", "0"))) in
    ("1", "true", "yes", "on")

# The normal literature run resolves the finite-size curves at 17 sizes and
# reaches N=40.  CI and local smoke checks can retain the original three
# modest sizes without changing the physical assertions or solver tolerance.
sizes = quick_example ? (8, 12, 16) : Tuple(8:2:40)
cases = NamedTuple[]
selected_residual_tol = 2e-5
stationary_tol = 1e-5

println(quick_example ?
    "Boundary-time-crystal smoke grid: N = $(collect(sizes))" :
    "Boundary-time-crystal research grid: N = 8:2:40 (17 sizes)")

for ratio in (0.5, 1.5)
    gaps = Float64[]
    modes = ComplexF64[]
    residuals = Float64[]
    conjugate_errors = Float64[]
    for N in sizes
        model = boundary_time_crystal_model(
            N; omega0=ratio * kappa, kappa=kappa)
        dense_oracle = quick_example || N <= 16
        prepared = compile(
            model; backend=dense_oracle ? :sparse : :matrixfree)

        # Small sizes use the complete spectrum as an oracle.  Larger sizes
        # use matrix-free Jacobi--Davidson at zero for the steady/gap modes.
        # In the time-crystalline regime two additional solves target the
        # positive/negative thermodynamic frequency scale.  This avoids both
        # a dense Liouvillian and a costly broad largest-real search.
        vals, max_residual = if dense_oracle
            complete = liouvillian_spectrum(
                prepared; target=:largest_real,
                nev=pi_dimension(prepared), algorithm=:dense)
            complete, 0.0
        else
            gap_window = pi_liouvillian_spectrum(
                prepared; method=:jd, target=0, nev=2,
                krylovdim=48, maxiter=240, atol=1e-9, rtol=1e-7,
                return_info=true)
            if ratio > 1
                frequency_seed = sqrt((ratio * kappa)^2 - kappa^2)
                positive = pi_liouvillian_spectrum(
                    prepared; method=:jd, target=im * frequency_seed, nev=1,
                    krylovdim=48, maxiter=180, atol=1e-9, rtol=1e-7,
                    return_info=true)
                negative = pi_liouvillian_spectrum(
                    prepared; method=:jd, target=-im * frequency_seed, nev=1,
                    krylovdim=48, maxiter=180, atol=1e-9, rtol=1e-7,
                    return_info=true)
                vcat(gap_window.values, positive.values, negative.values),
                    maximum(vcat(gap_window.residuals,
                                 positive.residuals, negative.residuals))
            else
                gap_window.values, maximum(gap_window.residuals)
            end
        end
        max_residual <= selected_residual_tol || error(
            "unconverged selected spectrum at omega0/kappa=$ratio, N=$N: " *
            "maximum Ritz residual $max_residual")

        nonzero = sort(filter(z -> abs(z) > stationary_tol, vals);
                       by=z -> real(z), rev=true)
        slowosc = if ratio > 1
            oscillatory = filter(z -> abs(imag(z)) > 1e-7, nonzero)
            isempty(oscillatory) && error(
                "no oscillatory slow mode at omega0/kappa=$ratio, N=$N")
            oscillatory[argmax(real.(oscillatory))]
        else
            NaN + NaN * im
        end
        conjugate_error = ratio > 1 ?
            minimum(abs.(vals .- conj(slowosc))) : 0.0
        conjugate_error <= 1e-5 || error(
            "unresolved conjugate partner at omega0/kappa=$ratio, N=$N: " *
            "pairing error $conjugate_error")

        push!(gaps, -real(nonzero[1]))
        push!(modes, slowosc)
        push!(residuals, max_residual)
        push!(conjugate_errors, conjugate_error)
        println("omega0/kappa=$ratio, N=$N: gap=", gaps[end],
                ratio > 1 ? ", slow oscillatory mode=$slowosc" : "",
                ", Ritz residual=$max_residual",
                ratio > 1 ? ", conjugate error=$conjugate_error" : "")
    end
    ratio > 1 && @assert gaps[end] < gaps[1]
    push!(cases, (; ratio, sizes, gaps, modes, residuals, conjugate_errors))
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
                 color, linewidth=2.4)
        M.scatter!(gap_axis, inverse_sizes, case.gaps ./ kappa;
                   color, marker, markersize=7, label)
        if all(isfinite, damping) && all(isfinite, frequency)
            M.lines!(damping_axis, inverse_sizes, damping;
                     color, linewidth=2.4)
            M.scatter!(damping_axis, inverse_sizes, damping;
                       color, marker, markersize=7)
            M.lines!(frequency_axis, inverse_sizes, frequency;
                     color, linewidth=2.4)
            M.scatter!(frequency_axis, inverse_sizes, frequency;
                       color, marker, markersize=7)
        end
    end
    M.axislegend(gap_axis; position=:lt, labelsize=12)

    save_example_figure(figure, "boundary_time_crystal")
end
