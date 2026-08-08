using LinearAlgebra
using PermutationalInvariantDynamics

include("paper_models.jl")
using .PaperModels
include(joinpath(@__DIR__, "utils", "makie_support.jl"))
using .ExampleMakie

# Finite-size version of the qubit cut through Figs. 5-6, supplemented by the
# finite-N product closure and the thermodynamic mean-field equations (10).

function article_fixed_point(V, gammaI, gammaC)
    gammaI > 0 || throw(ArgumentError(
        "this branch formula assumes gammaI > 0; gammaI = 0 is the singular case of Eq. (11)"))
    if gammaI + gammaC < 2abs(V)
        denominator = 2abs(V) - gammaC
        X = sqrt(gammaI * (2abs(V) - gammaI - gammaC)) / denominator
        Y = sign(V) * X
        Z = -gammaI / denominator
        return (; X, Y, Z, phase=:symmetry_broken)
    end
    (; X=0.0, Y=0.0, Z=-1.0, phase=:polarized)
end

function one_site_state(point, spin)
    Matrix{ComplexF64}(I, 2, 2) / 2 +
        point.X * spin.jx + point.Y * spin.jy + point.Z * spin.jz
end

function normalized_spin_coordinates(sigma, spin)
    (; X=real(meanfield_expectation(sigma, spin.jx)) / spin.j,
       Y=real(meanfield_expectation(sigma, spin.jy)) / spin.j,
       Z=real(meanfield_expectation(sigma, spin.jz)) / spin.j)
end

function transverse_pair_order(rho, Jx, Jy, N, j)
    x2 = collective_moments(rho, Jx).second_moment
    y2 = collective_moments(rho, Jy).second_moment
    numerator = x2 + y2 - 2N * j^2
    @assert abs(imag(numerator)) < 1e-9
    real(numerator) / (2N * (N - 1) * j^2)
end

function main()
    quick_example = get(ENV, "PID_EXAMPLE_QUICK", "0") == "1"
    N = quick_example ? 8 : 10
    d = 2
    V = 1.0
    gammaC = 0.2
    spin = spin_matrices(d)
    identity_state = Matrix{ComplexF64}(I, d, d) / d
    results = NamedTuple[]
    gammaI_values = quick_example ? [1.0, 1.8, 2.6] :
        sort!(unique!(vcat(
            collect(range(0.2, 3.0; length=20)), [1.8])))

    # Only the independent decay rate changes. Share the costly one-/two-body
    # Schur geometry across the complete finite-size scan.
    prototype = dissipative_collective_spin_pairing_model(
        N, d; V=V, gammaI=first(gammaI_values), gammaC=gammaC)
    family = compile_family(prototype; rate_indices=(3,))

    for gammaI in gammaI_values
        # dissipative_collective_spin_pairing_model retains the exact one-/two-body decomposition of
        # Jx^2-Jy^2.  The same model therefore drives both exact PI and
        # product-state mean-field calculations without changing conventions.
        prepared = specialize(family, gammaI / spin.j; backend=:sparse)
        model = prepared.model
        steady = stationary_state(
            prepared; algorithm=DirectAlgorithm(), return_info=true)
        gap = pi_liouvillian_gap(prepared; method=:dense)

        Jx = CollectiveObservablePlan(model.basis, spin.jx)
        Jy = CollectiveObservablePlan(model.basis, spin.jy)
        Jz = CollectiveObservablePlan(model.basis, spin.jz)
        Xpi = real(collective_expectation(steady.state, Jx)) / (N * spin.j)
        Ypi = real(collective_expectation(steady.state, Jy)) / (N * spin.j)
        Zpi = real(collective_expectation(steady.state, Jz)) / (N * spin.j)
        Cperp_pi = transverse_pair_order(
            steady.state, Jx, Jy, N, spin.j)

        article = article_fixed_point(V, gammaI, gammaC)
        article_state = one_site_state(article, spin)
        # Relax from inside the basin of the positive broken-symmetry branch.
        # A small identity admixture makes this a numerical fixed-point check
        # rather than returning the analytical seed unchanged.
        seed = 0.98article_state + 0.02identity_state
        finite_plan = MeanFieldPlan(model; limit=:finite)
        thermodynamic_plan = MeanFieldPlan(model; limit=:thermodynamic)
        finite = meanfield_stationary_state(
            finite_plan, seed; dt=0.02, max_steps=100_000,
            tol=1e-10, return_info=true)
        thermodynamic = meanfield_stationary_state(
            thermodynamic_plan, seed; dt=0.02, max_steps=100_000,
            tol=1e-10, return_info=true)
        finite_spin = normalized_spin_coordinates(finite.state, spin)
        thermodynamic_spin = normalized_spin_coordinates(
            thermodynamic.state, spin)
        article_residual = norm(meanfield_rhs(
            thermodynamic_plan, article_state))
        thermodynamic_error = maximum(abs, (
            thermodynamic_spin.X - article.X,
            thermodynamic_spin.Y - article.Y,
            thermodynamic_spin.Z - article.Z))
        stability = meanfield_stability(
            thermodynamic_plan, thermodynamic.state)

        Cperp_finite = (finite_spin.X^2 + finite_spin.Y^2) / 2
        Cperp_thermodynamic =
            (thermodynamic_spin.X^2 + thermodynamic_spin.Y^2) / 2
        Cperp_article = (article.X^2 + article.Y^2) / 2

        @assert steady.info.converged
        @assert finite.converged
        @assert thermodynamic.converged
        @assert article_residual < 2e-12
        @assert thermodynamic_error < 2e-8
        # The unique finite-N steady state preserves the Z2 symmetry.  Its
        # transverse order is instead visible in the parity-even pair moment.
        @assert max(abs(Xpi), abs(Ypi)) < 2e-9

        result = (;
            control=(gammaI + gammaC) / abs(V), gammaI, gap,
            finite_pi=(; X=Xpi, Y=Ypi, Z=Zpi, Cperp=Cperp_pi),
            finite_product=(; finite_spin..., Cperp=Cperp_finite,
                             residual=finite.residual),
            thermodynamic=(; thermodynamic_spin...,
                            Cperp=Cperp_thermodynamic,
                            residual=thermodynamic.residual),
            article=(; article..., Cperp=Cperp_article),
            article_residual,
            meanfield_spectral_abscissa=stability.spectral_abscissa)
        push!(results, result)

        if quick_example || gammaI in (first(gammaI_values), 1.8,
                                       last(gammaI_values))
            println("(gammaI+gammaC)/|V| = ", result.control,
                    ", finite-N gap = ", gap)
            println("  Z: exact PI = ", Zpi,
                    ", finite product = ", finite_spin.Z,
                    ", thermodynamic = ", thermodynamic_spin.Z,
                    ", article = ", article.Z)
            println("  Cperp: exact PI = ", Cperp_pi,
                    ", finite product = ", Cperp_finite,
                    ", thermodynamic = ", Cperp_thermodynamic,
                    ", article = ", Cperp_article)
            println("  selected thermodynamic branch (X,Y) = ",
                    (thermodynamic_spin.X, thermodynamic_spin.Y),
                    ", fixed-point residual = ", thermodynamic.residual,
                    ", stability abscissa = ", stability.spectral_abscissa)
        end
    end
    println("collective-spin-pairing scan: N=$N, points=$(length(results))")
    results
end

results = main()
plot_N = get(ENV, "PID_EXAMPLE_QUICK", "0") == "1" ? 8 : 10

if makie_available()
    M = makie_module()
    controls = [result.control for result in results]
    figure = M.Figure(size=(1350, 430), fontsize=17)
    gap_axis = M.Axis(
        figure[1, 1]; ylabel="Liouvillian gap",
        title="Finite-N relaxation")
    z_axis = M.Axis(
        figure[1, 2]; xlabel="(γᵢ + γc) / |V|", ylabel="Z",
        title="Longitudinal polarization")
    order_axis = M.Axis(
        figure[1, 3]; ylabel="C⊥",
        title="Parity-even transverse order")

    M.lines!(gap_axis, controls, [result.gap for result in results];
             color=:black, linewidth=2.5)
    M.scatter!(gap_axis, controls, [result.gap for result in results];
               color=:black, markersize=5, label="exact PI, N=$plot_N")
    M.vlines!(gap_axis, [2.0]; color=:gray50, linestyle=:dash,
              label="mean-field transition")
    M.axislegend(gap_axis; position=:lt, labelsize=12)

    predictions = (
        ("exact PI", :black, :circle, :finite_pi),
        ("finite product", :darkorange, :rect, :finite_product),
        ("thermodynamic", :seagreen, :utriangle, :thermodynamic),
        ("article", :royalblue, :diamond, :article),
    )
    for (label, color, marker, field) in predictions
        values = [getfield(result, field) for result in results]
        M.lines!(z_axis, controls, [value.Z for value in values];
                 color, linewidth=2.0)
        M.scatter!(z_axis, controls, [value.Z for value in values];
                   color, marker, markersize=5, label)
        M.lines!(order_axis, controls, [value.Cperp for value in values];
                 color, linewidth=2.0)
        M.scatter!(order_axis, controls, [value.Cperp for value in values];
                   color, marker, markersize=5, label)
    end
    M.vlines!(z_axis, [2.0]; color=:gray50, linestyle=:dash)
    M.vlines!(order_axis, [2.0]; color=:gray50, linestyle=:dash)
    M.axislegend(order_axis; position=:rt, labelsize=12)
    save_example_figure(figure, "dissipative_collective_spin_pairing_meanfield")
end
