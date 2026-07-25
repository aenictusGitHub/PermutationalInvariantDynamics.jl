using LinearAlgebra
using PermutationalInvariantDynamics
include(joinpath(@__DIR__, "utils", "makie_support.jl"))
using .ExampleMakie

# F. M. Gambetta et al., Phys. Rev. Lett. 122, 015701 (2019).
# Energies and times are expressed in units with Gamma = 1.
const GAMMA = 1.0
const OMEGA_X_0 = 0.7
const DELTA_0 = -3.5
const V0 = 6.0
const PERIOD = 2.0
const ROTATION_TIME = 0.01

# The paper prescribes a pi rotation around the normalized bisector of its two
# stable mean-field fixed points. Solving its stationary equations with the
# literal ordered-pair interaction (effective V = 2V0 = 12) gives
# M1 = ( 0.4228756113, 0.0733088525, -0.8973676066) and
# M2 = (-0.4683299751, 0.4753156586, -0.3345580780).
# Normalizing M1 and M2 separately and then their sum gives this axis.
const ROTATION_AXIS = (-0.13142675641321774,
                        0.46158758421366874,
                       -0.8773049126720367)

# The package basis is ordered as [ground, Rydberg]. This swaps the paper's
# up/down basis, hence sigma_y has the sign below while n and sigma^- take the
# familiar excitation-projector and decay forms.
const SIGMA_X = ComplexF64[0 1; 1 0]
const SIGMA_Y = ComplexF64[0 im; -im 0]
const NUMBER = ComplexF64[0 0; 0 1]
const SIGMA_MINUS = ComplexF64[0 1; 0 0]

"""Construct one autonomous segment of the dissipative periodic cycle."""
function segment_model(basis; omega_x, omega_y, detuning)
    N = basis.N
    terms = [
        LocalHamiltonian(SIGMA_X; rate=omega_x),
        LocalHamiltonian(SIGMA_Y; rate=omega_y),
        LocalHamiltonian(NUMBER; rate=detuning),

        # The article writes (V0/N) sum_{i != j} n_i n_j, an ordered-pair
        # sum. PBodyHamiltonian sums i < j, so its rate is twice as large.
        PBodyHamiltonian(kron(NUMBER, NUMBER), 2; rate=2V0 / N),
        LocalJump(SIGMA_MINUS; rate=GAMMA),
    ]
    PIModel(basis, terms)
end

"""Compose the rotation segment first and the relaxing segment second."""
function period_map(relaxing, rotation;
                    relaxing_steps::Integer, rotation_steps::Integer)
    FR = floquet_propagator(rotation, ROTATION_TIME; steps=rotation_steps)
    F0 = floquet_propagator(relaxing, PERIOD - ROTATION_TIME;
                            steps=relaxing_steps)
    F0 * FR
end

function period_doubling_multiplier(F)
    multipliers = floquet_multipliers(F)
    multipliers[argmin(abs.(multipliers .+ 1))]
end

function main()
    N = 4
    basis = PIBasis(N, 2)
    dx, dy, dz = ROTATION_AXIS
    @assert abs(sqrt(dx^2 + dy^2 + dz^2) - 1) < 1e-14

    relaxing_model = segment_model(
        basis; omega_x=OMEGA_X_0, omega_y=0.0, detuning=DELTA_0)
    rotation_model = segment_model(
        basis;
        omega_x=pi * dx / (2 * ROTATION_TIME),
        omega_y=pi * dy / (2 * ROTATION_TIME),
        detuning=pi * dz / ROTATION_TIME,
    )
    relaxing = compile(relaxing_model; backend=:matrixfree)
    rotation = compile(rotation_model; backend=:matrixfree)

    # Treat the narrow pulse and long relaxation interval as separate
    # autonomous segments. Step doubling tests both integrations without
    # forcing a uniform grid to resolve the discontinuity at t_R.
    Fcoarse = period_map(relaxing, rotation;
                         relaxing_steps=120, rotation_steps=48)
    Ffine = period_map(relaxing, rotation;
                       relaxing_steps=240, rotation_steps=96)
    epsilon_coarse = period_doubling_multiplier(Fcoarse)
    epsilon_fine = period_doubling_multiplier(Ffine)
    map_step_error = norm(Ffine - Fcoarse)
    multiplier_step_error = abs(epsilon_fine - epsilon_coarse)

    # Independent full-Hilbert-space exponentiation of these same N=4
    # segment generators gives this value. It is a regression reference, not
    # a number quoted by the article.
    epsilon_reference = -0.32402271
    lifetime = -PERIOD / log(abs(epsilon_fine))

    steady = stationary_state(Ffine - I; basis=basis,
                              algorithm=SVDAlgorithm(), return_info=true)
    rhoF = steady.state
    stationary_residual = norm(Ffine * rhoF.data - rhoF.data)
    report = diagnostics(rhoF)

    ground = iid_pure_state(basis, ComplexF64[1, 0])
    sx_plan = CollectiveObservablePlan(basis, SIGMA_X)
    stroboscopic = stroboscopic_evolution(ground, Ffine, 8)
    sx = [real(collective_expectation(rho, sx_plan)) / N
          for rho in stroboscopic]
    sx_stationary = real(collective_expectation(rhoF, sx_plan)) / N
    late_ratio = (sx[end] - sx_stationary) / (sx[end-1] - sx_stationary)

    println("Dissipative discrete-time-crystal finite-N PI precursor")
    println("N=$N; PI dimension=$(length(basis)); backends=",
            (diagnostics(relaxing).backend, diagnostics(rotation).backend))
    println("rotation axis: ", ROTATION_AXIS)
    println("rotation rates (Omega_x, Omega_y, Delta): ",
            (pi * dx / (2 * ROTATION_TIME), pi * dy / (2 * ROTATION_TIME),
             pi * dz / ROTATION_TIME))
    println("period-doubling multiplier (coarse, fine, reference): ",
            (epsilon_coarse, epsilon_fine, epsilon_reference))
    println("step-doubling errors (map, multiplier): ",
            (map_step_error, multiplier_step_error))
    println("finite-N decay time from multiplier: ", lifetime)
    println("stationary residual: ", stationary_residual,
            "; trace error: ", report.trace_error)
    println("Sx/N at periods 0:8: ", sx)
    println("stationary Sx/N: ", sx_stationary)
    println("last two deviations ratio (Floquet prediction): ",
            (late_ratio, real(epsilon_fine)))

    @assert map_step_error < 1e-4
    @assert multiplier_step_error < 1e-6
    @assert abs(epsilon_fine - epsilon_reference) < 5e-7
    @assert real(epsilon_fine) < 0
    @assert abs(imag(epsilon_fine)) < 1e-8
    @assert isapprox(late_ratio, real(epsilon_fine); atol=2e-3, rtol=2e-3)
    @assert stationary_residual < 1e-10
    @assert report.valid

    if makie_available()
        M = makie_module()
        multipliers = floquet_multipliers(Ffine)
        stationary_index = argmin(abs.(multipliers .- 1))
        angles = range(0.0, 2pi; length=361)
        periods = 0:(length(stroboscopic)-1)
        plotted_periods = periods[2:end]
        deviations = sx[2:end] .- sx_stationary

        figure = M.Figure(size=(1100, 470), fontsize=17)
        spectrum_axis = M.Axis(
            figure[1, 1]; xlabel="Re(ε)", ylabel="Im(ε)",
            aspect=M.DataAspect(), title="Finite-N Floquet multipliers, N=$N")
        signal_axis = M.Axis(
            figure[1, 2]; xlabel="period number n",
            ylabel="⟨Sx⟩ / N - stationary value",
            title="Decaying subharmonic response")

        M.lines!(spectrum_axis, cos.(angles), sin.(angles);
                 color=:gray60, linestyle=:dash, linewidth=1.5,
                 label="unit circle")
        M.scatter!(spectrum_axis, real.(multipliers), imag.(multipliers);
                   color=(:royalblue, 0.65), markersize=8,
                   label="Floquet spectrum")
        M.scatter!(spectrum_axis,
                   [real(multipliers[stationary_index])],
                   [imag(multipliers[stationary_index])];
                   color=:seagreen, marker=:diamond, markersize=13,
                   label="stationary ε₀")
        M.scatter!(spectrum_axis, [real(epsilon_fine)], [imag(epsilon_fine)];
                   color=:firebrick, marker=:star5, markersize=16,
                   label="subharmonic ε₋")
        M.axislegend(spectrum_axis; position=:lb, labelsize=11)

        M.hlines!(signal_axis, [0.0];
                  color=:gray50, linestyle=:dash,
                  label="Floquet stationary reference")
        M.lines!(signal_axis, plotted_periods, deviations;
                 color=:black, linewidth=2.2)
        M.scatter!(signal_axis, plotted_periods, deviations;
                   color=[isodd(n) ? :firebrick : :royalblue
                          for n in plotted_periods],
                   markersize=10, label="stroboscopic PI dynamics")
        M.axislegend(signal_axis; position=:rb, labelsize=11)

        M.Label(
            figure[2, 1:2],
            "At N=4, |ε₋| < 1 and the alternating signal decays: this is a finite-size precursor.";
            fontsize=14, color=:gray35, tellwidth=false)
        save_example_figure(
            figure, "dissipative_discrete_time_crystal")
    end
end

main()
