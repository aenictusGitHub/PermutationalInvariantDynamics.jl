using PermutationalInvariantDynamics
using LinearAlgebra

include(joinpath(@__DIR__, "utils", "makie_support.jl"))
using .ExampleMakie

# This example follows Colin Read, "Studying pure dephasing and dynamical
# decoupling: comparison of the HOPS method with exact analytical solutions"
# (September 2023).  The paper studies one qubit, so N=1 is intentional.  For
# N>1 the same PI coupling would describe one shared collective bath, not
# independent local environments.

"""
Return the ideal CPMG pulse times for `cycles` periods of length `period`.

One period is

    U(period/4) X U(period/2) X U(period/4),

so its two pulses occur at one and three quarters of the period.
"""
function cpmg_pulse_times(period::Real, cycles::Integer)
    period > 0 || throw(ArgumentError("the CPMG period must be positive"))
    cycles >= 0 || throw(ArgumentError("the CPMG cycle count must be nonnegative"))
    [cycle * period + fraction * period
     for cycle in 0:cycles-1 for fraction in (1 / 4, 3 / 4)]
end

"""
Return the ideal UDD pulse times for `cycles` periods.

Within each period, an order-`order` UDD sequence places pulse `j` at
`period*sinpi(j/(2order+2))^2`.
"""
function udd_pulse_times(period::Real, cycles::Integer, order::Integer)
    period > 0 || throw(ArgumentError("the UDD period must be positive"))
    cycles >= 0 || throw(ArgumentError("the UDD cycle count must be nonnegative"))
    order > 0 || throw(ArgumentError("the UDD order must be positive"))
    [cycle * period + period * sinpi(j / (2order + 2))^2
     for cycle in 0:cycles-1 for j in 1:order]
end

# For Q=sigma_z and a zero-temperature exponential bath correlation
#
#     C(t) = coefficient * exp(-pole*t),
#
# the exact ideal-pulse coherence is exp(-Gamma).  Writing the toggling
# function as y(t)=+1,-1,+1,... gives
#
#     z'(t) = y(t) - pole*z(t),
#     Gamma = 4*real(coefficient * integral(y(t)*z(t), t)).
#
# The update below integrates both quantities analytically on every interval.
# It is equivalent to the full-frequency Lorentzian filter-function integral,
# but avoids a frequency cutoff and the removable omega=0 singularity.
function full_line_dephasing_exponent(
        time::Real, pulse_times, coefficient, pole)
    time >= 0 || throw(ArgumentError("the evaluation time must be nonnegative"))
    real(pole) > 0 || throw(ArgumentError("the bath pole must decay"))
    all(isfinite, pulse_times) ||
        throw(ArgumentError("pulse times must be finite"))
    issorted(pulse_times) ||
        throw(ArgumentError("pulse times must be sorted"))

    value_type = promote_type(typeof(complex(coefficient)), typeof(pole))
    z = zero(value_type)
    integral = zero(z)
    start = zero(float(time))
    sign = one(float(time))
    last_pulse = searchsortedlast(pulse_times, time)

    for pulse_index in 1:last_pulse+1
        stop = pulse_index <= last_pulse ? pulse_times[pulse_index] : time
        stop >= start || throw(ArgumentError("pulse times must be nondecreasing"))
        interval = stop - start
        decay = exp(-pole * interval)
        response = -expm1(-pole * interval) / pole
        integral += sign * z * response + interval / pole - response / pole
        z = z * decay + sign * response
        sign = -sign
        start = stop
    end
    4real(coefficient * integral)
end

full_line_fidelity(time, pulse_times, coefficient, pole) =
    (1 + exp(-full_line_dephasing_exponent(
        time, pulse_times, coefficient, pole))) / 2

"""
Prepare Gauss--Legendre nodes and weights on the positive half-line.

The rational map `omega=scale*(1+x)/(1-x)` includes the infinite tail without
an arbitrary frequency cutoff. The returned rule is used only for the
analytical positive-frequency comparison, not by either hierarchy solver.
"""
function positive_frequency_quadrature(order::Integer, scale::Real)
    order > 1 || throw(ArgumentError("the quadrature order must exceed one"))
    scale > 0 || throw(ArgumentError("the quadrature scale must be positive"))
    R = typeof(float(scale))
    off_diagonal = R[
        index / sqrt(4index^2 - 1) for index in 1:order-1
    ]
    decomposition = eigen(SymTridiagonal(
        zeros(R, order), off_diagonal))
    nodes = decomposition.values
    base_weights = 2 .* abs2.(decomposition.vectors[1, :])
    frequencies = scale .* (1 .+ nodes) ./ (1 .- nodes)
    weights = base_weights .* (2scale) ./ (1 .- nodes).^2
    (; frequencies, weights)
end

# Stable Fourier transform of the piecewise-constant toggling function. Julia's
# sinc is normalized as sin(pi*x)/(pi*x), hence the factor 2pi below.
function toggling_integral(
        frequency::Real, time::Real, pulse_times)
    value = zero(Complex{typeof(float(frequency + time))})
    start = zero(float(time))
    sign = one(float(time))
    last_pulse = searchsortedlast(pulse_times, time)
    for pulse_index in 1:last_pulse+1
        stop = pulse_index <= last_pulse ? pulse_times[pulse_index] : time
        interval = stop - start
        midpoint = (start + stop) / 2
        value += sign * interval * cis(frequency * midpoint) *
                 sinc(frequency * interval / (2pi))
        sign = -sign
        start = stop
    end
    value
end

lorentzian_spectral_density(frequency, g, kappa, omega_c) =
    g * kappa^2 / (2((frequency - omega_c)^2 + kappa^2))

"""
Evaluate the physical, positive-frequency zero-temperature exponent.

The normalization

    Gamma = (2/pi) integral_0^infinity J(omega)*abs2(q(omega)) d omega

reduces without pulses to the manuscript's
`4/pi * integral J(omega)*(1-cos(omega*t))/omega^2 d omega`.
"""
function positive_frequency_dephasing_exponent(
        time, pulse_times, g, kappa, omega_c, quadrature)
    integral = zero(eltype(quadrature.weights))
    for index in eachindex(quadrature.frequencies)
        frequency = quadrature.frequencies[index]
        filter_value = toggling_integral(frequency, time, pulse_times)
        integral += quadrature.weights[index] *
                    lorentzian_spectral_density(
                        frequency, g, kappa, omega_c) *
                    abs2(filter_value)
    end
    2 * integral / pi
end

"""
Check the algebraic zeroth-order average-Hamiltonian contract of one cycle.

The cumulative control before event `k` is one toggling frame.  A complete
Eulerian Platonic word traverses every directed Cayley-graph edge once and
therefore samples every vertex uniformly (twice for the 24-edge tetrahedral
word), so averaging `G' * coupling * G` over those frames must give the
scalar part of the coupling.  The final cumulative control closes to the
identity up to a global phase, which is physically immaterial after forming
either an HEOM root density or a HOPS density estimator.

This compact check is intentionally restricted to the one-particle,
one-sector model used in this literature example.
"""
function zeroth_order_decoupling_check(
        sequence::HierarchyPulseSequence,
        coupling::PIOperator)
    sequence.basis === coupling.basis ||
        throw(ArgumentError("the pulse sequence and coupling use different bases"))
    basis = coupling.basis
    length(basis.sectors) == 1 ||
        throw(ArgumentError("this example check expects one retained Schur sector"))
    isempty(sequence.pulses) &&
        throw(ArgumentError("the pulse sequence must contain at least one event"))

    block = Matrix(physical_block(coupling, only(basis.sectors)))
    dimension = size(block, 1)
    scalar_type = promote_type(eltype(block), eltype(first(sequence.pulses)))
    frame = Matrix{scalar_type}(I, dimension, dimension)
    average = zeros(scalar_type, dimension, dimension)

    for event in sequence.pulses
        average .+= frame' * block * frame
        frame = event.blocks[1] * frame
    end
    average ./= length(sequence.pulses)

    identity_block = Matrix{scalar_type}(I, dimension, dimension)
    target = (tr(block) / dimension) .* identity_block
    phase = tr(frame) / dimension
    (twirl_residual=norm(average - target, Inf),
     closure_residual=norm(frame - phase .* identity_block, Inf),
     phase_modulus_residual=abs(abs(phase) - 1))
end

# Figure-8 parameters.  The single exponential is exactly the Fourier
# transform of the Lorentzian extended over the whole real-frequency axis:
#
# J(omega) = g*kappa^2 / (2*((omega-omega_c)^2+kappa^2)),
# C(t)     = (g*kappa/2) * exp(-(kappa-im*omega_c)*t).
#
# Consequently both HEOM and HOPS below must approach the full-line reference.
# A positive-frequency-only bath requires a controlled multi-exponential fit;
# increasing hierarchy depth cannot remove that separate modeling error.
N = 1
omega_c = 10.0
kappa = 1.5
g = 1.6 * omega_c^2 / kappa
coefficient = g * kappa / 2
pole = kappa - im * omega_c
Omega = g * kappa^2 / omega_c^2

cpmg_period = 2 / (7omega_c)
comparison_period = 2cpmg_period
comparison_cycles = 4
final_time = comparison_cycles * comparison_period
times = collect(range(0.0, final_time; length=65))
dt = cpmg_period / 48
depth = 6
trajectories = 512

# Two orders make the deterministic quadrature error visible rather than
# silently trusting one frequency grid.
positive_quadrature = positive_frequency_quadrature(
    768, hypot(omega_c, kappa))
positive_quadrature_check = positive_frequency_quadrature(
    384, hypot(omega_c, kappa))

# Two CPMG periods contain four pulses, as does one UDD4 period.  Comparing
# them over `comparison_period` therefore fixes the mean pulse rate.
cpmg_times = cpmg_pulse_times(cpmg_period, 2comparison_cycles)
udd4_times = udd_pulse_times(comparison_period, comparison_cycles, 4)
@assert length(cpmg_times) == length(udd4_times) == 4comparison_cycles
@assert all(diff(cpmg_times) .> 0) && all(diff(udd4_times) .> 0)
@assert first(cpmg_times) > 0 && last(cpmg_times) < final_time
@assert first(udd4_times) > 0 && last(udd4_times) < final_time

basis = PIBasis(N, 2)
spin = spin_matrices()
Jx = collective_operator(basis, spin.jx)
Jz = collective_operator(basis, spin.jz)
H0 = PIOperator(basis; T=Float64)
Q = 2Jz
rho0 = iid_pure_state(basis, ComplexF64[1, 1] / sqrt(2))
psi0 = weak_pi_pseudoket(rho0)

# exp(-im*pi*sigma_x/2) is an ideal pi pulse. PIUnitaryPulse lifts the local
# unitary to every retained Schur block without constructing a 2^N matrix.
pulse = PIUnitaryPulse(basis, exp(-im * pi * spin.jx))
cpmg_sequence = HierarchyPulseSequence(cpmg_times, pulse)
udd4_sequence = HierarchyPulseSequence(udd4_times, pulse)

# One published tetrahedral Eulerian DD (TEDD) cycle has 24 noncommuting
# rotations.  Give every edge the same free interval so that one complete
# cycle spans `comparison_period`.  Unlike CPMG/UDD, TEDD is not described by
# the scalar sign filter y(t)=+/-1 used above; it is checked independently
# through its toggling-frame group average below.
tedd_interval = comparison_period / 24
tedd_sequence = tetrahedral_pulse_sequence(basis, tedd_interval)
tedd_check = zeroth_order_decoupling_check(tedd_sequence, Q)

# HEOM uses the conjugate-right completion of the complex pole. HOPS needs
# only the physical left correlation. Both hierarchy preparations are shared
# between the two protocols; only the immutable pulse schedule changes.
system = PIModel(basis, ())
heom_bath = HEOMBath(
    Q, coefficient, pole;
    metadata=(model=:full_line_lorentzian,
              spectral_center=omega_c,
              spectral_width=kappa))
heom_plan = HEOMPlan(
    system, heom_bath; max_depth=depth, scaling=:scaled)

hops_bath = HOPSBath(Q, coefficient, pole; label=:dephasing_bath)
hops_plan = HOPSPlan(
    H0, hops_bath; max_depth=depth, scaling=:scaled)
hops_workspace = HOPSBatchWorkspace(hops_plan; workers=1)

# Exercise the same TEDD event data through both hierarchy drivers in a
# zero-generator round trip. The final pulse closes the cycle projectively,
# so the density reconstructed from either hierarchy must return to `rho0`.
# This is an event-handling check, separate from the finite-bath simulation.
tedd_final_time = last(tedd_sequence.times)
tedd_heom_plan = HEOMPlan(
    system, HEOMBath(Q, 0.0, 1.0); max_depth=0)
tedd_heom_path = heom_time_evolution(
    tedd_heom_plan, rho0, [0.0, tedd_final_time];
    steps_per_interval=2, pulses=tedd_sequence)
tedd_heom_state = heom_reduced_state(last(tedd_heom_path))

tedd_hops_plan = HOPSPlan(
    H0, HOPSBath(Q, 0.0, 1.0); max_depth=0)
tedd_hops_path = hops_trajectory(
    tedd_hops_plan, psi0, [0.0, tedd_final_time];
    dt=tedd_interval / 2,
    noise=(time, bath) -> 0.0 + 0.0im,
    pulses=tedd_sequence)
tedd_hops_state = hops_density(tedd_hops_path, 2)
tedd_heom_roundtrip_error =
    norm(tedd_heom_state.data - rho0.data, Inf)
tedd_hops_roundtrip_error =
    norm(tedd_hops_state.data - rho0.data, Inf)

# Contract the unnormalized HOPS estimator with |+x><+x| directly. Replacing
# `trace(rho)/2` by 1/2 would bias a finite ensemble toward unit trace.
fidelity_x(rho) = 0.5 * real(trace(rho)) + real(expectation(rho, Jx))

function solve_protocol(label, pulse_times, sequence, seed)
    exact_exponents = [
        full_line_dephasing_exponent(
            time, pulse_times, coefficient, pole)
        for time in times
    ]
    exact = (1 .+ exp.(-exact_exponents)) ./ 2
    positive_exponents = [
        positive_frequency_dephasing_exponent(
            time, pulse_times, g, kappa, omega_c,
            positive_quadrature)
        for time in times
    ]
    positive = (1 .+ exp.(-positive_exponents)) ./ 2
    positive_check = [
        (1 + exp(-positive_frequency_dephasing_exponent(
            time, pulse_times, g, kappa, omega_c,
            positive_quadrature_check))) / 2
        for time in times
    ]
    quadrature_error = maximum(abs.(positive .- positive_check))
    frequency_domain_separation = maximum(abs.(positive .- exact))

    heom_hierarchy = heom_time_evolution(
        heom_plan, rho0, times;
        steps_per_interval=6, pulses=sequence)
    heom_states = heom_reduced_state.(heom_hierarchy)
    heom_fidelity = fidelity_x.(heom_states)

    hops_result = hops_average(
        hops_plan, psi0, times, trajectories;
        dt, seed, threaded=false, workspace=hops_workspace,
        pulses=sequence, return_info=true)
    hops_fidelity = fidelity_x.(hops_result.states)
    # For the rank-one |+x><+x| projector the Hilbert--Schmidt norm is one.
    # The reported state-norm standard error is therefore a conservative
    # Cauchy--Schwarz bound on the fidelity standard error, not an observable-
    # specific sample variance.
    hops_fidelity_error_bound = hops_result.standard_error

    heom_error = maximum(abs.(heom_fidelity .- exact))
    hops_error = maximum(abs.(hops_fidelity .- exact))
    heom_trace_error = maximum(abs(trace(rho) - 1) for rho in heom_states)
    hops_trace_error = maximum(
        abs(real(trace(rho)) - 1) for rho in hops_result.states)
    maximum_standard_error = maximum(hops_result.standard_error)

    println("$label: HEOM/exact error=$heom_error, " *
            "HOPS/exact error=$hops_error, " *
            "HOPS max fidelity-error bound=$maximum_standard_error, " *
            "positive/full-line separation=$frequency_domain_separation")

    (; label, pulse_times, exact_exponents, exact,
       positive_exponents, positive, quadrature_error,
       frequency_domain_separation, heom_fidelity,
       hops_fidelity, hops_fidelity_error_bound,
       heom_error, hops_error, heom_trace_error, hops_trace_error,
       maximum_standard_error)
end

println("Non-Markovian dynamical decoupling with PI--HEOM and PI--HOPS")
println("N=$N, omega_c=$omega_c, kappa=$kappa, g=$g")
println("coefficient=$coefficient, pole=$pole, hierarchy depth=$depth")
println("CPMG and UDD4 use $(length(cpmg_times)) pulses over $final_time")
println("TEDD: $(length(tedd_sequence.times)) pulses over $tedd_final_time, " *
        "twirl residual=$(tedd_check.twirl_residual), " *
        "closure residual=$(tedd_check.closure_residual), " *
        "HEOM/HOPS round-trip errors=" *
        "$tedd_heom_roundtrip_error/$tedd_hops_roundtrip_error")

@assert length(tedd_sequence.times) == 24
@assert all(isapprox.(diff(tedd_sequence.times), tedd_interval;
                     atol=8eps(tedd_final_time), rtol=8eps(Float64)))
@assert isapprox(tedd_final_time, comparison_period;
                 atol=8eps(comparison_period), rtol=8eps(Float64))
@assert tedd_check.twirl_residual < 3e-12
@assert tedd_check.closure_residual < 3e-12
@assert tedd_check.phase_modulus_residual < 3e-12
@assert tedd_heom_roundtrip_error < 5e-12
@assert tedd_hops_roundtrip_error < 5e-12

cpmg = solve_protocol(
    "CPMG", cpmg_times, cpmg_sequence, 0x43504d47)
udd4 = solve_protocol(
    "UDD4", udd4_times, udd4_sequence, 0x55444434)

# The deterministic bounds test hierarchy and RK4 convergence for this fixed
# example. The stochastic bounds intentionally remain wider: a production
# result must independently increase depth and paths and decrease dt.
for result in (cpmg, udd4)
    @assert isapprox(first(result.exact), 1; atol=5e-15, rtol=0)
    @assert minimum(result.exact_exponents) > -2e-12
    @assert minimum(result.positive_exponents) >= 0
    @assert result.quadrature_error < 5e-5
    @assert result.frequency_domain_separation >
            10 * result.quadrature_error
    @assert all(isfinite, result.heom_fidelity)
    @assert all(isfinite, result.hops_fidelity)
    @assert result.heom_trace_error < 2e-9
    @assert result.heom_error < 1e-7
    @assert result.heom_error < result.frequency_domain_separation / 100
    @assert result.hops_trace_error < 0.05
    @assert result.hops_error < 0.03
end

if makie_available()
    M = makie_module()
    figure = M.Figure(size=(1160, 470), fontsize=17)
    scaled_times = Omega .* times

    for (column, result) in enumerate((cpmg, udd4))
        axis = M.Axis(
            figure[1, column];
            xlabel="Omega t", ylabel="fidelity",
            title="$(result.label), $(length(result.pulse_times)) pulses")
        M.vlines!(
            axis, Omega .* result.pulse_times;
            color=(:gray45, 0.18), linewidth=0.8)
        M.lines!(
            axis, scaled_times, result.exact;
            color=:black, linewidth=3, linestyle=:dash,
            label="full-line analytic")
        M.lines!(
            axis, scaled_times, result.positive;
            color=:firebrick, linewidth=2.5,
            label="positive-frequency analytic")
        M.lines!(
            axis, scaled_times, result.heom_fidelity;
            color=:darkorange2, linewidth=2.2, linestyle=:dash,
            label="PI--HEOM")
        shown = 1:4:length(times)
        M.errorbars!(
            axis, scaled_times[shown], result.hops_fidelity[shown],
            result.hops_fidelity_error_bound[shown];
            color=:dodgerblue3, whiskerwidth=7)
        M.scatter!(
            axis, scaled_times[shown], result.hops_fidelity[shown];
            color=:dodgerblue3, markersize=7,
            label="$trajectories PI--HOPS paths")
        M.axislegend(axis; position=:lb, labelsize=12)
    end
    save_example_figure(figure, "nonmarkovian_dynamical_decoupling")
end
