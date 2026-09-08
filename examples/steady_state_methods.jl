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
    model = local_pump_decay_model(N; down=down, up=up)
    exact = local_pump_decay_steady_state(model.basis; down=down, up=up)

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

    push!(solver_labels, "precond. GMRES (warm)")
    push!(residuals, matrixfree.info.residual)
    push!(exact_errors, matrixfree_error)
    push!(iterations, matrixfree.info.iterations)

    if makie_available()
        M = makie_module()
        positions = collect(eachindex(solver_labels))
        figure = example_figure(size=(1260, 550))
        M.Label(figure[0, 1:2], ExampleMakie.latex("Stationary-state solvers  •  \$N=$N\$, pump/decay=$(up/down)");
                fontsize=22, font=:bold, halign=:left)
        accuracy_axis = M.Axis(figure[1, 1];
            xlabel=ExampleMakie.latex("raw error / residual"), xscale=log10, xticks=10.0 .^ (-16:2:-10),
            yticks=(positions, solver_labels), yreversed=true,
            title=ExampleMakie.latex("(a) Accuracy against the exact thermal state"))
        iteration_axis = M.Axis(figure[1, 2];
            xlabel=ExampleMakie.latex("reported iterations"),
            yticks=(positions, solver_labels), yreversed=true,
            title=ExampleMakie.latex("(b) Solver diagnostics"))
        # Categories have no continuous interpolation. Retain sub-epsilon
        # values and omit exact zeros only from the logarithmic axis.
        for (values, offset, color, marker, label) in (
                (residuals, -0.13, example_colors.red, :circle, ExampleMakie.latex("Liouvillian residual")),
                (exact_errors, 0.13, example_colors.blue, :diamond, ExampleMakie.latex("PI-state distance")))
            positive = findall(>(0), values)
            M.scatter!(accuracy_axis, values[positive], positions[positive] .+ offset;
                       color, marker, markersize=10, label)
        end
        M.axislegend(accuracy_axis; position=:rb, labelsize=12)
        for index in positions
            if index <= 3
                M.text!(iteration_axis, 0.2, index; text=ExampleMakie.latex("factorization / decomposition"),
                        align=(:left, :center), fontsize=13, color=example_colors.gray)
            else
                M.barplot!(iteration_axis, [index], [iterations[index]];
                           direction=:x, color=index == last(positions) ?
                           example_colors.orange : example_colors.blue, width=0.55)
                        M.text!(iteration_axis, iterations[index] + 0.2, index;
                        text=ExampleMakie.latex(string(iterations[index])), align=(:left, :center), fontsize=14)
            end
        end
        M.ylims!(accuracy_axis, length(positions)+0.5, 0.5)
        M.ylims!(iteration_axis, length(positions)+0.5, 0.5)
        M.xlims!(iteration_axis, 0, max(10, maximum(iterations)+2))
        M.Label(figure[2, 1:2],
            ExampleMakie.latex("Warm preconditioned GMRES starts from the direct solution.  •  Iterations do not measure speed; no categorical lines or error floors.");
            fontsize=13, color=example_colors.gray)
        save_example_figure(figure, "steady_state_methods")
        save_example_data("steady_state_methods", (
            solver=solver_labels, residual=residuals, state_error=exact_errors,
            iterations=iterations, initial_guess=[fill("default", 5); "direct solution"],
            iteration_count_applicable=[false, false, false, true, true, true]);
            metadata=(; N, down, up, iterative_atol=1e-12, iterative_rtol=1e-10,
                      note="Reported iterations have algorithm-specific meanings; last solve is warm."))
    end
end

main()
