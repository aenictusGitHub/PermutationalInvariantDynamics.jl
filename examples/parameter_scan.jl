using PermutationalInvariantDynamics

include(joinpath(@__DIR__, "utils", "makie_support.jl"))
using .ExampleMakie

# A finite-temperature local reservoir. Only scalar rates vary, so one
# CompiledPIModelFamily shares all fixed Schur geometry across the scan.
basis=PIBasis(12,2)
sm=ComplexF64[0 1;0 0]
sp=Matrix(adjoint(sm))
excited=ComplexF64[0 0;0 1]

function thermal_model(pump)
    PIModel(basis,(
        LocalJump(sm;rate=1.0),
        LocalJump(sp;rate=pump),
    ))
end

quick_example=get(ENV,"PID_EXAMPLE_QUICK","0")=="1"
pump_rates=collect(range(0.05,0.50;length=quick_example ? 7 : 25))
family=compile_family(thermal_model(first(pump_rates)))
excited_fraction=(rho,pump,index)->(
    value=real(collective_expectation(rho,excited))/basis.N,
    pump=pump,index=index)

plan=ParameterScanPlan(pump_rates,family;
    rate_builder=pump->(1.0,pump),
    algorithm=RecycledGMRESAlgorithm(
        krylovdim=24,maxiter=300,recycle_dim=6),
    specialize_options=(backend=:matrixfree,),
    solver_options=(atol=1e-11,rtol=1e-8),
    continuation=true,save_outputs=false,save_restart=true,
    diagnostic=excited_fraction)

# The callback receives the live output even though state histories are not
# retained. Returning :stop would produce another resumable prefix.
streamed=NamedTuple[]
function record_point(point)
    point.status===:success&&push!(streamed,(
        pump=point.parameter,
        excited_fraction=point.diagnostics.user.value,
        residual=point.residual,
    ))
    nothing
end

# Emulate a checkpoint boundary and restart in a fresh solver workspace.
prefix=parameter_scan(plan;max_points=3,callback=record_point)
result=resume_parameter_scan(plan,prefix;
    workspace=ParameterScanWorkspace(),callback=record_point)

println("completed ",length(result)," scan points")
for row in streamed
    println("pump = ",row.pump,
            ", excited fraction = ",row.excited_fraction,
            ", residual = ",row.residual)
end

@assert all(point->point.status===:success,result)
@assert result[4].warm_started
@assert result[5].workspace_reused
@assert result[4].diagnostics.compile.geometry_reused
@assert all(point->point.output===nothing,result)
@assert maximum(abs(row.excited_fraction-row.pump/(1+row.pump))
                for row in streamed)<2e-8

# Dependency-free column output can be handed to Tables-compatible analysis
# or plotting code without making Tables.jl a core dependency.
columns=parameter_scan_columns(result)
@assert columns.parameter==pump_rates

# A related dynamic sensitivity keeps [rho, d rho/d pump] as matrix columns.
# The specialized family source applies both columns in one prepared
# matrix-RHS call; the derivative generator owns separate task-local scratch.
dynamic_model=specialize(family,(1.0,first(pump_rates)))
rho_dynamic=iid_pure_state(basis,ComplexF64[1,0])
pump_derivative=compile(
    PIModel(basis,(LocalJump(sp;rate=1.0),));backend=:matrixfree)
sensitivity=sensitivity_problem(
    dynamic_model,rho_dynamic,(0.0,0.1),(pump_derivative,))
sensitivity_rhs=similar(sensitivity.u0)
sensitivity.f(
    sensitivity_rhs,sensitivity.u0,sensitivity.p,first(sensitivity.tspan))
@assert size(sensitivity_rhs)==(length(basis),2)
@assert all(isfinite,sensitivity_rhs)
@assert sensitivity_rhs[:,1]≈dynamic_model*rho_dynamic.data
@assert sensitivity_rhs[:,2]≈pump_derivative*rho_dynamic.data

if makie_available()
    M = makie_module()
    streamed_pumps = [row.pump for row in streamed]
    streamed_fractions = [row.excited_fraction for row in streamed]
    streamed_residuals = [row.residual for row in streamed]
    exact_fractions = streamed_pumps ./ (1 .+ streamed_pumps)
    fraction_errors = abs.(streamed_fractions .- exact_fractions)

    figure = example_figure(size=(1200, 510))
    M.Label(figure[0, 1:3], "Thermal stationary scan  •  N=$(basis.N), $(length(pump_rates)) pump rates";
            fontsize=22, font=:bold, halign=:left)
    fraction_axis = M.Axis(
        figure[1, 1];
        xlabel="pump / decay rate r", ylabel="stationary excited fraction",
        title="(a) Prepared continuation")
    residual_axis = M.Axis(
        figure[1, 2];
        xlabel="pump / decay rate r", ylabel="stationary residual",
        yscale=log10, title="(b) Solver residual")
    error_axis = M.Axis(figure[1, 3]; xlabel="pump / decay rate r",
        ylabel="absolute fraction error", title="(c) Observable accuracy")

    M.lines!(
        fraction_axis, streamed_pumps, exact_fractions;
        color=:black, linewidth=2.7, label="exact r / (1 + r)")
    M.scatter!(
        fraction_axis, streamed_pumps, streamed_fractions;
        color=example_colors.blue, markersize=8, label="PI scan")
    M.lines!(
        residual_axis, streamed_pumps, [iszero(r) ? NaN : r for r in streamed_residuals];
        color=example_colors.orange)
    M.scatter!(
        residual_axis, streamed_pumps, [iszero(r) ? NaN : r for r in streamed_residuals];
        color=example_colors.orange, markersize=7)
    M.lines!(error_axis, streamed_pumps, fraction_errors; color=example_colors.red)
    M.scatter!(error_axis, streamed_pumps, fraction_errors; color=example_colors.red, markersize=6)
    M.axislegend(fraction_axis; position=:lt)
    M.Label(figure[2, 1:3], "Recycled GMRES: atol=10⁻¹¹, rtol=10⁻⁸  •  $(count(iszero, streamed_residuals)) zero residuals omitted only on the log axis; raw values are exported.";
            fontsize=13, color=example_colors.gray)
    save_example_figure(figure, "parameter_scan")
    save_example_data("parameter_scan", (;
        pump_over_decay=streamed_pumps, excited_fraction=streamed_fractions,
        exact_fraction=exact_fractions, fraction_error=fraction_errors,
        stationary_residual=streamed_residuals,
        warm_started=[point.warm_started for point in result],
        workspace_reused=[point.workspace_reused for point in result]);
        metadata=(; N=basis.N, decay_rate=1.0, atol=1e-11, rtol=1e-8,
                  continuation=true, restart_after=3))
end
