module PIDTimeToSolutionBenchmark

using LinearAlgebra
using Random
using PermutationalInvariantDynamics

include(joinpath(@__DIR__, "harness.jl"))
using .PIBenchmarkHarness

const RESULT_COLUMNS = (
    :schema_version,
    :implementation,
    :suite,
    :mode,
    :workflow,
    :phase,
    :sample,
    :N,
    :d,
    :pi_dimension,
    :work_units,
    :memory_budget_bytes,
    :elapsed_time_ns,
    :gc_time_ns,
    :allocated_bytes,
    :validation_passed,
    :validation_metric,
    :validation_value,
    :validation_tolerance,
    :trace_error,
    :solver_residual,
)

function usage(io=stdout)
    println(io, """
Usage:
  julia --startup-file=no --project=benchmark benchmark/time_to_solution.jl [options]

Options:
  --mode quick|full          Case sizes and measurement effort (default: quick)
  --samples INTEGER         Measured repetitions of each workflow
  --warmups INTEGER         Discarded complete workflow repetitions
  --memory-budget-mib FLOAT Explicit per-operation memory budget (default: 512)
  --output PATH             Result TSV path
  --dry-run                 Parse options and write empty TSV plus metadata
  -h, --help                Show this message

Each measured repetition reports setup, solve, validation, and their total
separately. Julia/package loading and JIT warm-up are excluded; use
`benchmark/cold_start.jl` for fresh-process latency.
""")
end

function _option_value(args, index, name)
    index < length(args) || throw(ArgumentError("$name needs a value"))
    args[index + 1], index + 1
end

function parse_options(args)
    mode = :quick
    samples = nothing
    warmups = nothing
    memory_budget_mib = 512.0
    output = nothing
    dry_run = false
    index = 1
    while index <= length(args)
        argument = args[index]
        if argument in ("-h", "--help")
            usage()
            return nothing
        elseif startswith(argument, "--mode=")
            mode = Symbol(split(argument, '='; limit=2)[2])
        elseif argument == "--mode"
            value, index = _option_value(args, index, argument)
            mode = Symbol(value)
        elseif startswith(argument, "--samples=")
            samples = parse(Int, split(argument, '='; limit=2)[2])
        elseif argument == "--samples"
            value, index = _option_value(args, index, argument)
            samples = parse(Int, value)
        elseif startswith(argument, "--warmups=")
            warmups = parse(Int, split(argument, '='; limit=2)[2])
        elseif argument == "--warmups"
            value, index = _option_value(args, index, argument)
            warmups = parse(Int, value)
        elseif startswith(argument, "--memory-budget-mib=")
            memory_budget_mib =
                parse(Float64, split(argument, '='; limit=2)[2])
        elseif argument == "--memory-budget-mib"
            value, index = _option_value(args, index, argument)
            memory_budget_mib = parse(Float64, value)
        elseif startswith(argument, "--output=")
            output = split(argument, '='; limit=2)[2]
        elseif argument == "--output"
            output, index = _option_value(args, index, argument)
        elseif argument == "--dry-run"
            dry_run = true
        else
            throw(ArgumentError("unknown option $argument"))
        end
        index += 1
    end

    mode in (:quick, :full) ||
        throw(ArgumentError("mode must be quick or full"))
    samples = something(samples, mode === :quick ? 2 : 7)
    warmups = something(warmups, 1)
    samples > 0 || throw(ArgumentError("samples must be positive"))
    warmups >= 0 || throw(ArgumentError("warmups must be nonnegative"))
    isfinite(memory_budget_mib) && memory_budget_mib > 0 ||
        throw(ArgumentError("memory-budget-mib must be finite and positive"))
    bytes_float = memory_budget_mib * 1024^2
    bytes_float <= typemax(Int) ||
        throw(ArgumentError("memory budget exceeds Int range"))
    memory_budget = floor(Int, bytes_float)
    output = something(output, joinpath(
        @__DIR__, "results", "time_to_solution_$(mode).tsv"))
    (; mode, samples, warmups, memory_budget_mib, memory_budget,
       output, dry_run)
end

function _timed(f)
    measured = @timed f()
    (;
        value=measured.value,
        elapsed_time_ns=round(Int, measured.time * 1e9),
        gc_time_ns=round(Int, measured.gctime * 1e9),
        allocated_bytes=Int(measured.bytes),
    )
end

function _guard_known_bytes(required::Integer, budget::Integer, label)
    required <= budget || throw(ArgumentError(
        "$label estimates $required bytes, exceeding the explicit " *
        "memory budget of $budget bytes"))
    nothing
end

function _thermal_steady_setup(options)
    N = options.mode === :quick ? 4 : 8
    basis = PIBasis(N, 2)
    _guard_known_bytes(
        8 * estimate_basis_bytes(basis) +
        16 * estimate_state_bytes(basis),
        options.memory_budget, "thermal steady-state benchmark setup")
    spin = spin_matrices()
    down = 0.7
    up = 0.2
    model = PIModel(basis, (
        LocalJump(spin.jm; rate=down),
        LocalJump(spin.jp; rate=up),
    ))
    prepared = compile(
        model; backend=:auto, memory_budget=options.memory_budget)
    excited_probability = up / (up + down)
    local_state = ComplexF64[
        1 - excited_probability 0
        0 excited_probability
    ]
    exact = iid_state(basis, local_state)
    (; N, basis, prepared, exact, work_units=1)
end

function _thermal_steady_solve(context, options)
    stationary_state(
        context.prepared;
        algorithm=AutoAlgorithm(),
        memory_budget=options.memory_budget,
        return_info=true,
    )
end

function _thermal_steady_validate(context, result)
    diagnostics_report = diagnostics(result.state)
    error = norm(result.state.data - context.exact.data) /
            max(norm(context.exact.data), eps(Float64))
    tolerance = 2e-9
    trace_error = abs(trace(result.state) - 1)
    residual = result.info.residual
    passed = result.info.converged && diagnostics_report.valid &&
             error <= tolerance && trace_error <= 2e-10
    (;
        passed,
        metric="relative_state_error",
        value=error,
        tolerance,
        trace_error,
        solver_residual=residual,
    )
end

function _dynamics_setup(options)
    N = options.mode === :quick ? 6 : 12
    basis = PIBasis(N, 2)
    _guard_known_bytes(
        8 * estimate_basis_bytes(basis) +
        24 * estimate_state_bytes(basis),
        options.memory_budget, "dynamics benchmark setup")
    spin = spin_matrices()
    gamma = 0.8
    final_time = 0.6
    model = PIModel(basis, (LocalJump(spin.jm; rate=gamma),))
    prepared = compile(
        model; backend=:matrixfree, memory_budget=options.memory_budget)
    rho0 = iid_pure_state(basis, ComplexF64[0, 1])
    exact_probability = exp(-gamma * final_time)
    exact = iid_state(basis, ComplexF64[
        1 - exact_probability 0
        0 exact_probability
    ])
    steps = options.mode === :quick ? 64 : 128
    (; N, basis, prepared, rho0, exact, final_time, steps,
       work_units=steps)
end

function _dynamics_solve(context, options)
    solve_dynamics(
        context.prepared, context.rho0, (0.0, context.final_time);
        saveat=[0.0, context.final_time],
        algorithm=:rk4,
        steps_per_interval=context.steps,
        memory_budget=options.memory_budget,
    )
end

function _dynamics_validate(context, result)
    final_state = result[end]
    diagnostics_report = diagnostics(final_state)
    error = norm(final_state.data - context.exact.data) /
            max(norm(context.exact.data), eps(Float64))
    # This is an end-to-end RK4 workload rather than a solver-convergence
    # study. The tolerance remains well below the plotted/physical scale while
    # accommodating the deliberately finite quick-mode step count.
    tolerance = 2e-7
    trace_error = abs(trace(final_state) - 1)
    passed = diagnostics_report.valid && error <= tolerance &&
             trace_error <= 2e-10
    (;
        passed,
        metric="relative_final_state_error",
        value=error,
        tolerance,
        trace_error,
        solver_residual=missing,
    )
end

function _trajectory_setup(options)
    N = options.mode === :quick ? 4 : 6
    trajectories = options.mode === :quick ? 48 : 256
    basis = PIBasis(N, 2)
    _guard_known_bytes(
        10 * estimate_basis_bytes(basis) +
        32 * estimate_state_bytes(basis),
        options.memory_budget, "trajectory benchmark setup")
    spin = spin_matrices()
    gamma = 0.8
    final_time = 0.6
    model = PIModel(basis, (LocalJump(spin.jm; rate=gamma),))
    prepared = compile(
        model; backend=:matrixfree, memory_budget=options.memory_budget)
    rho0 = iid_pure_state(basis, ComplexF64[0, 1])
    plan = TrajectoryPlan(prepared)
    workspace =
        TrajectoryBatchWorkspace(plan, rho0; workers=1, mode=:fixed)
    number = adjoint(spin.jm) * spin.jm
    dt = options.mode === :quick ? 0.01 : 0.005
    (; N, basis, rho0, plan, workspace, number, gamma, final_time, dt,
       trajectories, work_units=trajectories)
end

function _trajectory_solve(context, options)
    quantum_trajectories(
        context.plan, context.rho0, [0.0, context.final_time],
        context.trajectories;
        seed=0x51a7,
        threaded=false,
        workspace=context.workspace,
        dt=context.dt,
        algorithm=:fixed,
        observables=(excitations=context.number,),
        save_states=false,
        jump_statistics=true,
        memory_budget=options.memory_budget,
    )
end

function _trajectory_validate(context, result)
    sample = result.observables.observables[:excitations]
    exact_probability = exp(-context.gamma * context.final_time)
    exact_excitation = context.N * exact_probability
    excitation_error = abs(sample.mean[end] - exact_excitation)
    excitation_tolerance = 8 * sample.standard_error[end] + 2e-12

    emission_probability = 1 - exact_probability
    exact_jump_mean = context.N * emission_probability
    exact_jump_variance =
        context.N * emission_probability * (1 - emission_probability)
    jump_error = abs(result.jumps.mean_count - exact_jump_mean)
    jump_tolerance =
        8 * sqrt(exact_jump_variance / context.trajectories) +
        inv(context.trajectories)

    normalized_error = max(
        excitation_error / max(excitation_tolerance, eps(Float64)),
        jump_error / max(jump_tolerance, eps(Float64)),
    )
    passed = result.trajectories === nothing &&
             result.observables.trajectories == context.trajectories &&
             result.jumps.trajectories == context.trajectories &&
             normalized_error <= 1
    (;
        passed,
        metric="maximum_normalized_sampling_error",
        value=normalized_error,
        tolerance=1.0,
        trace_error=missing,
        solver_residual=missing,
    )
end

function _reduction_setup(options)
    N = options.mode === :quick ? 6 : 10
    k = N ÷ 2
    basis = PIBasis(N, 2)
    retained_bound =
        BigInt(estimate_basis_bytes(basis)) +
        16 * BigInt(pi_dimension(basis))^2
    retained_bound <= options.memory_budget || throw(ArgumentError(
        "reduction benchmark setup bound $retained_bound bytes exceeds " *
        "the explicit memory budget $(options.memory_budget) bytes"))
    rho = ghz_state(basis)
    validate_state(rho)
    plan = ReductionPlan(basis, k)
    workspace = ReductionWorkspace(plan, rho; mode=:reduction)
    (; N, k, basis, rho, plan, workspace, work_units=k)
end

function _reduction_solve(context, options)
    reduced_state(
        context.rho, context.k;
        plan=context.plan,
        workspace=context.workspace,
        check=false,
    )
end

function _reduction_validate(context, result)
    diagnostics_report = diagnostics(result)
    expected_purity = 0.5
    error = abs(purity(result) - expected_purity)
    tolerance = 2e-10
    trace_error = abs(trace(result) - 1)
    passed = diagnostics_report.valid && error <= tolerance &&
             trace_error <= 2e-10
    (;
        passed,
        metric="ghz_reduced_purity_error",
        value=error,
        tolerance,
        trace_error,
        solver_residual=missing,
    )
end

const WORKFLOWS = (
    (
        name=:steady_state,
        setup=_thermal_steady_setup,
        solve=_thermal_steady_solve,
        validate=_thermal_steady_validate,
    ),
    (
        name=:dynamics,
        setup=_dynamics_setup,
        solve=_dynamics_solve,
        validate=_dynamics_validate,
    ),
    (
        name=:trajectory,
        setup=_trajectory_setup,
        solve=_trajectory_solve,
        validate=_trajectory_validate,
    ),
    (
        name=:reduction,
        setup=_reduction_setup,
        solve=_reduction_solve,
        validate=_reduction_validate,
    ),
)

function _run_workflow(workflow, options)
    setup = _timed(() -> workflow.setup(options))
    solve = _timed(() -> workflow.solve(setup.value, options))
    validation = _timed(() -> workflow.validate(setup.value, solve.value))
    validation.value.passed || throw(ErrorException(
        "$(workflow.name) validation failed: " *
        "$(validation.value.metric)=$(validation.value.value), tolerance=" *
        "$(validation.value.tolerance)"))
    setup, solve, validation
end

function _phase_row(options, workflow, sample, context, phase, timing;
                    validation=nothing)
    (;
        schema_version=1,
        implementation="PermutationalInvariantDynamics",
        suite="end_to_end_time_to_solution",
        mode=options.mode,
        workflow=workflow.name,
        phase,
        sample,
        N=context.N,
        d=context.basis.d,
        pi_dimension=pi_dimension(context.basis),
        work_units=context.work_units,
        memory_budget_bytes=options.memory_budget,
        elapsed_time_ns=timing.elapsed_time_ns,
        gc_time_ns=timing.gc_time_ns,
        allocated_bytes=timing.allocated_bytes,
        validation_passed=
            validation === nothing ? missing : validation.passed,
        validation_metric=
            validation === nothing ? missing : validation.metric,
        validation_value=
            validation === nothing ? missing : validation.value,
        validation_tolerance=
            validation === nothing ? missing : validation.tolerance,
        trace_error=
            validation === nothing ? missing : validation.trace_error,
        solver_residual=
            validation === nothing ? missing : validation.solver_residual,
    )
end

function _rows(options)
    for workflow in WORKFLOWS, _ in 1:options.warmups
        _run_workflow(workflow, options)
    end

    rows = NamedTuple[]
    for workflow in WORKFLOWS
        for sample in 1:options.samples
            GC.gc()
            setup, solve, validation = _run_workflow(workflow, options)
            context = setup.value
            push!(rows, _phase_row(
                options, workflow, sample, context, :setup, setup))
            push!(rows, _phase_row(
                options, workflow, sample, context, :solve, solve))
            push!(rows, _phase_row(
                options, workflow, sample, context, :validation, validation;
                validation=validation.value))
            total = (;
                elapsed_time_ns=setup.elapsed_time_ns +
                    solve.elapsed_time_ns + validation.elapsed_time_ns,
                gc_time_ns=setup.gc_time_ns +
                    solve.gc_time_ns + validation.gc_time_ns,
                allocated_bytes=setup.allocated_bytes +
                    solve.allocated_bytes + validation.allocated_bytes,
            )
            push!(rows, _phase_row(
                options, workflow, sample, context, :total, total;
                validation=validation.value))
        end
    end
    rows
end

function main(args=ARGS)
    options = parse_options(args)
    options === nothing && return nothing
    BLAS.set_num_threads(1)
    rows = options.dry_run ? NamedTuple[] : _rows(options)
    write_tsv(options.output, rows, RESULT_COLUMNS)
    metadata = benchmark_metadata(
        mode=options.mode,
        requested_samples=options.samples,
        seconds=0,
        warmups=options.warmups,
        output=options.output,
        repository_root=normpath(joinpath(@__DIR__, "..")),
        extra=[
            "result_schema_version" => "1",
            "benchmark_kind" => "warmed_end_to_end_time_to_solution",
            "workflows" => join(string.(getproperty.(WORKFLOWS, :name)), ","),
            "phase_policy" => "setup,solve,validation,total",
            "memory_budget_bytes" => string(options.memory_budget),
            "blas_threads_forced" => "1",
            "trajectory_threaded" => "false",
            "trajectory_output_policy" => "streaming_observables",
            "dry_run" => string(options.dry_run),
        ],
    )
    write_metadata(metadata_path(options.output), metadata)
    println("wrote ", options.output)
    println("wrote ", metadata_path(options.output))
    rows
end

end # module PIDTimeToSolutionBenchmark

if abspath(PROGRAM_FILE) == @__FILE__
    PIDTimeToSolutionBenchmark.main()
end
