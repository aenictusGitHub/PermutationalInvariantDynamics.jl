using LinearAlgebra
using PermutationalInvariantDynamics
include("paper_models.jl")
using .PaperModels
include(joinpath(@__DIR__, "utils", "makie_support.jl"))
using .ExampleMakie

# Local pumping and emission have an exact tensor-power thermal stationary
# state. This makes a useful comparison of every steady-state algorithm.
function main()
    N = 6
    down = 1.0
    up = 0.3
    model = shammah2018_thermal_model(N; down=down, up=up)
    exact = shammah2018_thermal_state(model.basis; down=down, up=up)

    # The small-system methods share one sparse compilation. Algorithms are
    # values, so solver selection is explicit and checked by dispatch.
    prepared = compile(model; backend=:sparse)
    settings = [
        ("direct", DirectAlgorithm(), NamedTuple()),
        ("svd", SVDAlgorithm(), NamedTuple()),
        ("eigen", EigenAlgorithm(), NamedTuple()),
        ("shift-invert", ShiftInvertAlgorithm(shift=-1e-3, maxiter=100),
         (atol=1e-12, rtol=1e-10)),
        ("GMRES", GMRESAlgorithm(krylovdim=20, maxiter=200),
         (atol=1e-12, rtol=1e-10)),
    ]

    reference = nothing
    solver_labels = String[]
    residuals = Float64[]
    exact_errors = Float64[]
    iterations = Int[]
    for (label, algorithm, options) in settings
        elapsed = @elapsed result = stationary_state(prepared;
            algorithm=algorithm, return_info=true, options...)
        info = result.info
        error = norm(result.state.data - exact.data)
        push!(solver_labels, label)
        push!(residuals, info.residual)
        push!(exact_errors, error)
        push!(iterations, info.iterations)
        reference === nothing && (reference = result.state)
        println(rpad(label, 13),
                " residual=", info.residual,
                " trace error=", info.trace_error,
                " exact-state error=", error,
                " iterations=", info.iterations,
                " elapsed=", elapsed, " s")
        @assert info.converged
        @assert error < 2e-8
        @assert diagnostics(result.state).valid
    end

    # The shift may be tuned near zero. A typed state from a previous solve is
    # a valid warm start when scanning nearby model parameters.
    warm = stationary_state(prepared;
        algorithm=ShiftInvertAlgorithm(shift=-1e-2, maxiter=200),
        initial_state=reference, atol=1e-12, rtol=1e-10, return_info=true)
    println("warm-start shift-invert iterations: ", warm.info.iterations)

    # This final solve exposes the reusable backend internals deliberately:
    # matrix-free kernels, Krylov storage, and a Schur-sector preconditioner.
    matrixfree_prepared = compile(model; backend=:matrixfree)
    workspace = KrylovWorkspace(matrixfree_prepared, 20)
    preconditioner = schur_sector_preconditioner(
        matrixfree_prepared, model.basis;
        expected_reuses=10, warn_unamortized=false)
    matrixfree = stationary_state(matrixfree_prepared;
        algorithm=GMRESAlgorithm(krylovdim=20, maxiter=200,
                                 preconditioner=preconditioner),
        workspace=workspace, initial_state=reference,
        atol=1e-12, rtol=1e-10, return_info=true)
    cost = preconditioner_cost(preconditioner)
    println("warm-start preconditioned matrix-free GMRES iterations: ",
            matrixfree.info.iterations)
    println("Schur preconditioner cost metadata: ", cost)

    # A compatible compiled PI source lowers diagonal sector blocks directly
    # from the immutable term plan. Only the operator-scale probes remain.
    @assert cost.block_construction === :prepared_kernels
    @assert cost.setup_block_applications == 0
    @assert matrixfree.info.converged
    matrixfree_error = norm(matrixfree.state.data - exact.data)
    @assert matrixfree_error < 2e-8

    push!(solver_labels, "precond. GMRES")
    push!(residuals, matrixfree.info.residual)
    push!(exact_errors, matrixfree_error)
    push!(iterations, matrixfree.info.iterations)

    if makie_available()
        M = makie_module()
        positions = collect(eachindex(solver_labels))
        figure = M.Figure(size=(1220, 470), fontsize=17)
        accuracy_axis = M.Axis(
            figure[1, 1];
            xlabel="stationary-state algorithm",
            ylabel="error",
            yscale=log10,
            xticks=(positions, solver_labels),
            xticklabelrotation=pi / 6,
            title="Raw residual and exact-state distance",
        )
        iteration_axis = M.Axis(
            figure[1, 2];
            xlabel="stationary-state algorithm",
            ylabel="reported iterations",
            xticks=(positions, solver_labels),
            xticklabelrotation=pi / 6,
            title="Iterative work reported by each solver",
        )
        floor_value = eps(Float64)
        M.lines!(
            accuracy_axis, positions, max.(residuals, floor_value);
            color=:firebrick, linewidth=2.2)
        M.scatter!(
            accuracy_axis, positions, max.(residuals, floor_value);
            color=:firebrick, markersize=10, label="Liouvillian residual")
        M.lines!(
            accuracy_axis, positions, max.(exact_errors, floor_value);
            color=:royalblue, linewidth=2.2, linestyle=:dash)
        M.scatter!(
            accuracy_axis, positions, max.(exact_errors, floor_value);
            color=:royalblue, marker=:diamond, markersize=10,
            label="distance to exact state")
        M.axislegend(accuracy_axis; position=:lb, labelsize=12)
        M.barplot!(
            iteration_axis, positions, iterations;
            color=:slateblue, width=0.65)
        save_example_figure(figure, "steady_state_methods")
    end
end

main()
