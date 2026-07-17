using PermutationalInvariantDynamics

include(joinpath(@__DIR__, "utils", "makie_support.jl"))
using .ExampleMakie

# A collective exponentially correlated dephasing bath. The physical model is
# H_int = Jz tensor B and C(t) = coefficient * exp(-frequency*t). Every ADO is
# represented directly in the PI operator space; no 2^N Hilbert-space matrix
# is constructed.
N = 8
coefficient = 0.30
frequency = 1.20
final_time = 1.50
depths = (2, 4, 6)
times = range(0.0, final_time; length=61)

basis = PIBasis(N, 2)
spin = spin_matrices()
Jz = collective_operator(basis, spin.jz)
Jx = collective_operator(basis, spin.jx)
system = PIModel(basis, ())
bath = HEOMBath(Jz, coefficient, frequency)
rho0 = iid_pure_state(basis, ComplexF64[1, 1] / sqrt(2))

# For this commuting, real-correlation benchmark, the exact normalized
# transverse coherence is exp[-g(t)] with the Gaussian dephasing line shape
# below. It is independent of N after division by N/2.
line_shape(t) = coefficient / frequency^2 *
                (frequency*t - 1 + exp(-frequency*t))
exact = exp.(-line_shape.(times))

curves = NamedTuple[]
for depth in depths
    plan = HEOMPlan(system, bath; max_depth=depth)
    hierarchy = heom_time_evolution(
        plan, rho0, times; steps_per_interval=4)
    normalized_Jx = [
        2real(expectation(heom_reduced_state(state), Jx)) / N
        for state in hierarchy]
    error = maximum(abs.(normalized_Jx .- exact))
    trace_error = maximum(
        abs(trace(heom_reduced_state(state)) - 1) for state in hierarchy)
    push!(curves, (; depth, plan, normalized_Jx, error, trace_error))
    println("depth=$depth: ADOs=$(heom_number_ados(plan)), " *
            "PI coordinates=$(length(basis)), HEOM coordinates=$(size(plan, 1)), " *
            "max signal error=$error, max trace error=$trace_error")
end

@assert all(curves[index+1].error < curves[index].error
            for index in 1:length(curves)-1)
@assert last(curves).error < 5e-10
@assert last(curves).trace_error < 5e-12

# Produce a compact automated truncation report at the final time. The helper
# compiles the system once, shares the prepared coupling blocks across depth
# prefixes, and retains only the reduced state at intermediate depths.
depth_report = heom_depth_convergence(
    system, bath, rho0, (first(times), last(times));
    depths, steps=4(length(times)-1), atol=1e-7, rtol=0)
# The displayed Jx signal has converged at depth 6, but the stronger full-root
# Hilbert--Schmidt test has not. This distinction is intentional and prevents
# an observable-specific agreement from being presented as state convergence.
@assert isequal(depth_report.pairwise_converged,
                Union{Missing,Bool}[missing, false, false])
@assert !depth_report.converged
println("successive-depth Hilbert--Schmidt differences = ",
        collect(skipmissing(depth_report.pairwise_errors)))

# The low-level matrix-free adapter can be passed to existing Krylov spectral
# tools. This example only inspects its size; it never materializes the map.
Lheom = heom_liouvillian(last(curves).plan)
@assert size(Lheom) == (size(last(curves).plan, 1),
                       size(last(curves).plan, 1))

if makie_available()
    M = makie_module()
    figure = M.Figure(size=(1120, 440), fontsize=17)
    signal_axis = M.Axis(
        figure[1, 1]; xlabel="νt", ylabel="2⟨Jx⟩ / N",
        title="PI–HEOM collective dephasing (N=$N)")
    error_axis = M.Axis(
        figure[1, 2]; xlabel="νt", ylabel="absolute error",
        yscale=log10, title="Hierarchy-depth convergence")

    scaled_times = frequency .* collect(times)
    M.lines!(signal_axis, scaled_times, exact;
             color=:black, linewidth=3, label="analytic")
    colors = (:firebrick, :royalblue, :seagreen)
    for (curve, color) in zip(curves, colors)
        label = "depth $(curve.depth)"
        M.lines!(signal_axis, scaled_times, curve.normalized_Jx;
                 color, linewidth=2, linestyle=:dash, label)
        pointwise_error = abs.(curve.normalized_Jx .- exact)
        # A zero at t=0 cannot be displayed on a logarithmic axis.
        shown_error = max.(pointwise_error, eps(Float64))
        M.lines!(error_axis, scaled_times, shown_error;
                 color, linewidth=2.3, label)
    end
    M.axislegend(signal_axis; position=:lb, labelsize=12)
    M.axislegend(error_axis; position=:rb, labelsize=12)
    save_example_figure(figure, "pi_heom")
end
