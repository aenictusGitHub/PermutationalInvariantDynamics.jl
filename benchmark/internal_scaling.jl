using PermutationalInvariantDynamics
using LinearAlgebra
using Random
using SparseArrays

include(joinpath(@__DIR__, "harness.jl"))
using .PIBenchmarkHarness

const RESULT_COLUMNS = (
    :schema_version, :implementation, :suite, :case, :mode, :workload,
    :sector_policy, :N, :d, :sector_count, :pi_dimension,
    :full_hilbert_dimension, :full_liouville_dimension,
    :basis_minimum_time_ns, :basis_median_time_ns,
    :basis_minimum_gc_time_ns, :basis_allocated_bytes, :basis_allocations,
    :basis_samples,
    :setup_minimum_time_ns, :setup_median_time_ns,
    :setup_minimum_gc_time_ns, :setup_allocated_bytes, :setup_allocations,
    :setup_samples,
    :geometry_kind, :geometry_retained_upper_bytes,
    :geometry_setup_upper_bytes,
    :apply_minimum_time_ns, :apply_median_time_ns,
    :apply_minimum_gc_time_ns, :apply_allocated_bytes, :apply_allocations,
    :apply_samples,
    :sparse_setup_minimum_time_ns, :sparse_setup_median_time_ns,
    :sparse_setup_minimum_gc_time_ns, :sparse_setup_allocated_bytes,
    :sparse_setup_allocations, :sparse_setup_samples,
    :sparse_apply_minimum_time_ns, :sparse_apply_median_time_ns,
    :sparse_apply_minimum_gc_time_ns, :sparse_apply_allocated_bytes,
    :sparse_apply_allocations, :sparse_apply_samples,
    :sparse_nnz, :sparse_standalone_bytes,
    :sparse_structure_supported, :sparse_contribution_upper_bound,
    :sparse_retained_nnz_upper_bound, :sparse_operator_upper_bound_bytes,
    :sparse_assembly_upper_bound_bytes,
    :sparse_materialization_peak_upper_bound_bytes,
    :auto_memory_budget_bytes, :auto_chosen_backend,
    :auto_compiled_standalone_bytes, :auto_relative_error,
    :driven_setup_minimum_time_ns, :driven_setup_median_time_ns,
    :driven_setup_minimum_gc_time_ns, :driven_setup_allocated_bytes,
    :driven_setup_allocations, :driven_setup_samples,
    :driven_apply_minimum_time_ns, :driven_apply_median_time_ns,
    :driven_apply_minimum_gc_time_ns, :driven_apply_allocated_bytes,
    :driven_apply_allocations, :driven_apply_samples,
    :driven_plan_standalone_bytes, :driven_compiled_standalone_bytes,
    :driven_workspace_standalone_bytes,
    :basis_standalone_bytes, :model_standalone_bytes,
    :plan_standalone_bytes, :compiled_standalone_bytes,
    :workspace_standalone_bytes, :vectors_standalone_bytes,
    :hot_retained_total_bytes,
    :validation_oracle, :sparse_relative_error,
    :trace_relative_error, :adjoint_relative_error,
    :auto_validation_passed, :driven_validation_oracle,
    :driven_sparse_relative_error, :driven_trace_relative_error,
    :driven_adjoint_relative_error, :driven_validation_time_ns,
    :driven_validation_passed,
    :validation_tolerance, :validation_time_ns, :validation_passed,
)

const AUTO_MEMORY_BUDGET_BYTES = 512 * 1024^2

const MISSING_BENCHMARK_STATS = (
    minimum_time_ns=missing,
    median_time_ns=missing,
    minimum_gc_time_ns=missing,
    allocated_bytes=missing,
    allocations=missing,
    samples=missing,
)

function usage(io=stdout)
    println(io, """
Usage:
  julia --project=benchmark benchmark/internal_scaling.jl [options]

Options:
  --mode quick|full                  Scaling matrix (default: quick)
  --samples INTEGER                 Maximum samples per operation
  --seconds FLOAT                   Time budget per operation
  --warmups INTEGER                 Explicit warm-up evaluations
  --validation-dimension-limit INT  Largest sparse-oracle PI dimension
  --output PATH                     Result TSV path
  -h, --help                        Show this message
""")
end

function option_value(args, index, name)
    index < length(args) || throw(ArgumentError("$name needs a value"))
    args[index + 1], index + 1
end

function parse_options(args)
    mode = :quick
    samples = nothing
    seconds = nothing
    warmups = 2
    validation_dimension_limit = 5_000
    output = nothing
    index = 1
    while index <= length(args)
        argument = args[index]
        if argument in ("-h", "--help")
            usage()
            exit(0)
        elseif startswith(argument, "--mode=")
            mode = Symbol(split(argument, '='; limit=2)[2])
        elseif argument == "--mode"
            value, index = option_value(args, index, argument)
            mode = Symbol(value)
        elseif startswith(argument, "--samples=")
            samples = parse(Int, split(argument, '='; limit=2)[2])
        elseif argument == "--samples"
            value, index = option_value(args, index, argument)
            samples = parse(Int, value)
        elseif startswith(argument, "--seconds=")
            seconds = parse(Float64, split(argument, '='; limit=2)[2])
        elseif argument == "--seconds"
            value, index = option_value(args, index, argument)
            seconds = parse(Float64, value)
        elseif startswith(argument, "--warmups=")
            warmups = parse(Int, split(argument, '='; limit=2)[2])
        elseif argument == "--warmups"
            value, index = option_value(args, index, argument)
            warmups = parse(Int, value)
        elseif startswith(argument, "--validation-dimension-limit=")
            validation_dimension_limit = parse(Int,
                split(argument, '='; limit=2)[2])
        elseif argument == "--validation-dimension-limit"
            value, index = option_value(args, index, argument)
            validation_dimension_limit = parse(Int, value)
        elseif startswith(argument, "--output=")
            output = split(argument, '='; limit=2)[2]
        elseif argument == "--output"
            output, index = option_value(args, index, argument)
        else
            throw(ArgumentError("unknown option $argument"))
        end
        index += 1
    end

    mode in (:quick, :full) || throw(ArgumentError(
        "mode must be quick or full"))
    samples = something(samples, mode === :quick ? 5 : 12)
    seconds = something(seconds, mode === :quick ? 0.5 : 2.0)
    samples > 0 || throw(ArgumentError("samples must be positive"))
    seconds > 0 || throw(ArgumentError("seconds must be positive"))
    warmups > 0 || throw(ArgumentError("warmups must be positive"))
    validation_dimension_limit >= 0 || throw(ArgumentError(
        "validation dimension limit must be nonnegative"))
    output = something(output, joinpath(@__DIR__, "results",
        "internal_scaling_$(mode).tsv"))
    (; mode, samples, seconds, warmups, validation_dimension_limit, output)
end

function benchmark_cases(mode::Symbol)
    qubits = mode === :quick ? (2, 4, 8) : (2, 4, 8, 12, 16, 20, 24)
    qutrits = mode === :quick ? (2, 3, 4) : (2, 3, 4, 5, 6)
    symmetric = mode === :quick ? (4, 8, 16) : (4, 8, 16, 32, 48, 64, 96)
    symmetric_qutrits = mode === :quick ? (2, 4) : (2, 4, 8, 12)
    cases = NamedTuple[]
    append!(cases, ((suite=:all_sector_qubits, N=N, d=2,
                     sector_policy=:all, workload=:local_collective)
                    for N in qubits))
    append!(cases, ((suite=:all_sector_qutrits, N=N, d=3,
                     sector_policy=:all, workload=:local_collective)
                    for N in qutrits))
    append!(cases, ((suite=:symmetric_sector_qubits, N=N, d=2,
                     sector_policy=:symmetric,
                     workload=:collective_only)
                    for N in symmetric))
    append!(cases, ((suite=:symmetric_sector_qutrits, N=N, d=3,
                     sector_policy=:symmetric,
                     workload=:collective_only)
                    for N in symmetric_qutrits))
    cases
end

function make_basis(case)
    if case.sector_policy === :all
        PIBasis(case.N, case.d)
    else
        sector = ntuple(index -> index == 1 ? case.N : 0, case.d)
        PIBasis(case.N, case.d; sectors=[sector])
    end
end

function local_operators(d::Integer)
    lowering = zeros(ComplexF64, d, d)
    for level in 1:(d - 1)
        lowering[level, level + 1] = sqrt(level)
    end
    drive = lowering + adjoint(lowering)
    lowering, drive
end

function make_model(basis, case)
    lowering, drive = local_operators(case.d)
    terms = if case.workload === :local_collective
        (LocalHamiltonian(drive; rate=0.13),
         LocalJump(lowering; rate=0.27),
         CollectiveJump(lowering; rate=0.03))
    else
        (CollectiveHamiltonian(drive; rate=0.13),
         CollectiveJump(lowering; rate=0.27))
    end
    PIModel(basis, terms)
end

function make_driven_model(basis, case)
    case.sector_policy === :symmetric || throw(ArgumentError(
        "driven collective benchmarks require a symmetric-sector case"))
    lowering, drive = local_operators(case.d)
    diagonal = zeros(ComplexF64, case.d, case.d)
    center = (case.d + 1) / 2
    for level in 1:case.d
        diagonal[level, level] = level - center
    end
    raising = Matrix(adjoint(lowering))
    hamiltonian_prototype = copy(drive)
    jump_prototype = copy(lowering)
    hamiltonian_schedule = InPlaceTimeOperator(
        hamiltonian_prototype,
        let drive=copy(drive), diagonal=diagonal
            function (destination, time, parameters)
                c = cos(time)
                s = parameters.mix * sin(time)
                @inbounds for index in eachindex(destination, drive, diagonal)
                    destination[index] = c * drive[index] +
                                         s * diagonal[index]
                end
                nothing
            end
        end)
    jump_schedule = InPlaceTimeOperator(
        jump_prototype,
        let lowering=copy(lowering), raising=raising
            function (destination, time, parameters)
                s = parameters.mix * sin(time)
                @inbounds for index in eachindex(destination, lowering, raising)
                    destination[index] = lowering[index] + s * raising[index]
                end
                nothing
            end
        end)
    PIModel(basis, (
        CollectiveHamiltonian(hamiltonian_schedule; rate=0.13),
        CollectiveJump(jump_schedule; rate=0.27),
    ))
end

relative_error(left, right) = norm(left - right) /
    max(norm(left), norm(right), eps(Float64))

function validate_application(prepared, model, input, output, workspace,
                              dimension_limit, tolerance, rng;
                              sparse_oracle=nothing)
    started = time_ns()
    trace_vector = prepared.plan.tracevec
    trace_error = abs(dot(trace_vector, output)) /
        max(norm(trace_vector) * norm(output), eps(Float64))

    probe = randn(rng, ComplexF64, length(input))
    probe ./= norm(probe)
    adjoint_output = similar(input)
    apply_adjoint!(adjoint_output, prepared.plan, probe, 0.0, nothing,
                   workspace)
    left = dot(probe, output)
    right = dot(adjoint_output, input)
    adjoint_error = abs(left - right) /
        max(abs(left), abs(right), eps(Float64))

    sparse_error = missing
    oracle = :trace_adjoint
    if sparse_oracle !== nothing || length(input) <= dimension_limit
        sparse = sparse_oracle === nothing ?
            liouvillian(model; representation=:sparse,
                        memory_budget=Inf) : sparse_oracle
        reference = sparse * input
        sparse_error = relative_error(output, reference)
        oracle = :sparse_trace_adjoint
    end
    elapsed = time_ns() - started
    passed = isfinite(trace_error) && trace_error <= tolerance &&
        isfinite(adjoint_error) && adjoint_error <= tolerance &&
        (sparse_error === missing ||
         (isfinite(sparse_error) && sparse_error <= tolerance))
    (; oracle, sparse_error, trace_error, adjoint_error,
       elapsed, passed)
end


function validate_driven_application(prepared, model, input, output,
                                     workspace, time, parameters,
                                     dimension_limit, tolerance, rng)
    started = time_ns()
    trace_vector = prepared.plan.tracevec
    trace_error = abs(dot(trace_vector, output)) /
        max(norm(trace_vector) * norm(output), eps(Float64))

    probe = randn(rng, ComplexF64, length(input))
    probe ./= norm(probe)
    adjoint_output = similar(input)
    apply_adjoint!(adjoint_output, prepared.plan, probe, time, parameters,
                   workspace)
    left = dot(probe, output)
    right = dot(adjoint_output, input)
    adjoint_error = abs(left - right) /
        max(abs(left), abs(right), eps(Float64))

    sparse_error = missing
    oracle = :trace_adjoint
    if length(input) <= dimension_limit
        frozen = freeze(model; time, parameters, representation=:sparse)
        sparse_error = relative_error(output, frozen * input)
        oracle = :frozen_sparse_trace_adjoint
    end
    elapsed = time_ns() - started
    passed = isfinite(trace_error) && trace_error <= tolerance &&
        isfinite(adjoint_error) && adjoint_error <= tolerance &&
        (sparse_error === missing ||
         (isfinite(sparse_error) && sparse_error <= tolerance))
    (; oracle, sparse_error, trace_error, adjoint_error, elapsed, passed)
end

function run_case(case, options, case_index)
    label = "$(case.suite)_N$(case.N)"
    println("  ", label)
    basis_builder = () -> make_basis(case)
    basis_stats = benchmark_call(basis_builder; samples=options.samples,
        seconds=options.seconds, warmups=options.warmups)
    basis = basis_builder()
    model = make_model(basis, case)
    setup_builder = () -> compile(model; backend=:matrixfree,
                                  memory_budget=Inf)
    setup_stats = benchmark_call(setup_builder; samples=options.samples,
        seconds=options.seconds, warmups=options.warmups)
    prepared = setup_builder()
    workspace = LiouvillianWorkspace(prepared)
    recommendation = recommend_solver(model; memory_budget=Inf)

    seed = Int(0x5049_0000) + 100 * case.d + case.N + case_index
    rng = MersenneTwister(seed)
    input = randn(rng, ComplexF64, length(basis))
    input ./= norm(input)
    output = similar(input)
    apply_call = () -> apply!(output, prepared, input, 0.0, nothing,
                              workspace)
    apply_stats = benchmark_call(apply_call; samples=options.samples,
        seconds=options.seconds, warmups=options.warmups)
    apply_call()

    collective_fastpath = case.sector_policy === :symmetric
    sparse_setup_stats = MISSING_BENCHMARK_STATS
    sparse_apply_stats = MISSING_BENCHMARK_STATS
    sparse_prepared = nothing
    sparse_output = nothing
    auto_prepared = nothing
    auto_error = missing
    auto_passed = missing
    if collective_fastpath
        sparse_builder = () -> compile(
            model; backend=:sparse, memory_budget=Inf)
        sparse_setup_stats = benchmark_call(
            sparse_builder; samples=options.samples, seconds=options.seconds,
            warmups=options.warmups)
        sparse_prepared = sparse_builder()
        sparse_output = similar(input)
        sparse_apply_call = let destination=sparse_output,
                                operator=sparse_prepared.operator,
                                source=input
            () -> mul!(destination, operator, source)
        end
        sparse_apply_stats = benchmark_call(
            sparse_apply_call; samples=options.samples,
            seconds=options.seconds, warmups=options.warmups)
        sparse_apply_call()

        auto_prepared = compile(
            model; backend=:auto,
            memory_budget=AUTO_MEMORY_BUDGET_BYTES)
        auto_output = auto_prepared * input
        auto_error = relative_error(output, auto_output)
        auto_passed = isfinite(auto_error) && auto_error <= 1.0e-10
    end

    driven_setup_stats = MISSING_BENCHMARK_STATS
    driven_apply_stats = MISSING_BENCHMARK_STATS
    driven_prepared = nothing
    driven_workspace = nothing
    driven_validation = nothing
    if collective_fastpath
        driven_model = make_driven_model(basis, case)
        driven_builder = () -> compile(
            driven_model; backend=:matrixfree, memory_budget=Inf)
        driven_setup_stats = benchmark_call(
            driven_builder; samples=options.samples, seconds=options.seconds,
            warmups=options.warmups)
        driven_prepared = driven_builder()
        driven_workspace = LiouvillianWorkspace(driven_prepared)
        driven_output = similar(input)
        driven_time = 0.31
        driven_parameters = (mix=0.37,)
        driven_apply_call = let destination=driven_output,
                                source=driven_prepared,
                                input=input,
                                time=driven_time,
                                parameters=driven_parameters,
                                workspace=driven_workspace
            () -> apply!(destination, source, input, time, parameters,
                         workspace)
        end
        driven_apply_stats = benchmark_call(
            driven_apply_call; samples=options.samples,
            seconds=options.seconds, warmups=options.warmups)
        driven_apply_call()
        driven_validation = validate_driven_application(
            driven_prepared, driven_model, input, driven_output,
            driven_workspace, driven_time, driven_parameters,
            options.validation_dimension_limit, 1.0e-10, rng)
    end

    tolerance = 1.0e-10
    validation = validate_application(prepared, model, input, output,
        workspace, options.validation_dimension_limit, tolerance, rng;
        sparse_oracle=sparse_prepared === nothing ? nothing :
            sparse_prepared.operator)
    validation_passed = validation.passed &&
        (auto_passed === missing || auto_passed) &&
        (driven_validation === nothing || driven_validation.passed)

    vectors = (input, output)
    retained_total = Base.summarysize((prepared, workspace, vectors))
    estimates = prepared.estimates
    (; schema_version=2,
       implementation="PermutationalInvariantDynamics.jl",
       suite=case.suite,
       case=label,
       mode=options.mode,
       workload=case.workload,
       sector_policy=case.sector_policy,
       N=case.N,
       d=case.d,
       sector_count=length(basis.sectors),
       pi_dimension=length(basis),
       full_hilbert_dimension=big(case.d)^case.N,
       full_liouville_dimension=big(case.d)^(2 * case.N),
       basis_minimum_time_ns=basis_stats.minimum_time_ns,
       basis_median_time_ns=basis_stats.median_time_ns,
       basis_minimum_gc_time_ns=basis_stats.minimum_gc_time_ns,
       basis_allocated_bytes=basis_stats.allocated_bytes,
       basis_allocations=basis_stats.allocations,
       basis_samples=basis_stats.samples,
       setup_minimum_time_ns=setup_stats.minimum_time_ns,
       setup_median_time_ns=setup_stats.median_time_ns,
       setup_minimum_gc_time_ns=setup_stats.minimum_gc_time_ns,
       setup_allocated_bytes=setup_stats.allocated_bytes,
       setup_allocations=setup_stats.allocations,
       setup_samples=setup_stats.samples,
       geometry_kind=collective_fastpath ? :symmetric_collective : :onebody,
       geometry_retained_upper_bytes=recommendation.geometry_retained_upper_bytes,
       geometry_setup_upper_bytes=recommendation.geometry_setup_upper_bytes,
       apply_minimum_time_ns=apply_stats.minimum_time_ns,
       apply_median_time_ns=apply_stats.median_time_ns,
       apply_minimum_gc_time_ns=apply_stats.minimum_gc_time_ns,
       apply_allocated_bytes=apply_stats.allocated_bytes,
       apply_allocations=apply_stats.allocations,
       apply_samples=apply_stats.samples,
       sparse_setup_minimum_time_ns=sparse_setup_stats.minimum_time_ns,
       sparse_setup_median_time_ns=sparse_setup_stats.median_time_ns,
       sparse_setup_minimum_gc_time_ns=sparse_setup_stats.minimum_gc_time_ns,
       sparse_setup_allocated_bytes=sparse_setup_stats.allocated_bytes,
       sparse_setup_allocations=sparse_setup_stats.allocations,
       sparse_setup_samples=sparse_setup_stats.samples,
       sparse_apply_minimum_time_ns=sparse_apply_stats.minimum_time_ns,
       sparse_apply_median_time_ns=sparse_apply_stats.median_time_ns,
       sparse_apply_minimum_gc_time_ns=sparse_apply_stats.minimum_gc_time_ns,
       sparse_apply_allocated_bytes=sparse_apply_stats.allocated_bytes,
       sparse_apply_allocations=sparse_apply_stats.allocations,
       sparse_apply_samples=sparse_apply_stats.samples,
       sparse_nnz=sparse_prepared === nothing ? missing :
           nnz(sparse_prepared.operator),
       sparse_standalone_bytes=sparse_prepared === nothing ? missing :
           Base.summarysize(sparse_prepared.operator),
       sparse_structure_supported=estimates.sparse_structure_supported,
       sparse_contribution_upper_bound=estimates.sparse_contribution_upper_bound,
       sparse_retained_nnz_upper_bound=estimates.sparse_retained_nnz_upper_bound,
       sparse_operator_upper_bound_bytes=estimates.sparse_operator_upper_bound,
       sparse_assembly_upper_bound_bytes=estimates.sparse_assembly_upper_bound,
       sparse_materialization_peak_upper_bound_bytes=
           estimates.sparse_materialization_peak_upper_bound,
       auto_memory_budget_bytes=collective_fastpath ?
           AUTO_MEMORY_BUDGET_BYTES : missing,
       auto_chosen_backend=auto_prepared === nothing ? missing :
           auto_prepared.backend,
       auto_compiled_standalone_bytes=auto_prepared === nothing ? missing :
           Base.summarysize(auto_prepared),
       auto_relative_error=auto_error,
       driven_setup_minimum_time_ns=driven_setup_stats.minimum_time_ns,
       driven_setup_median_time_ns=driven_setup_stats.median_time_ns,
       driven_setup_minimum_gc_time_ns=driven_setup_stats.minimum_gc_time_ns,
       driven_setup_allocated_bytes=driven_setup_stats.allocated_bytes,
       driven_setup_allocations=driven_setup_stats.allocations,
       driven_setup_samples=driven_setup_stats.samples,
       driven_apply_minimum_time_ns=driven_apply_stats.minimum_time_ns,
       driven_apply_median_time_ns=driven_apply_stats.median_time_ns,
       driven_apply_minimum_gc_time_ns=driven_apply_stats.minimum_gc_time_ns,
       driven_apply_allocated_bytes=driven_apply_stats.allocated_bytes,
       driven_apply_allocations=driven_apply_stats.allocations,
       driven_apply_samples=driven_apply_stats.samples,
       driven_plan_standalone_bytes=driven_prepared === nothing ? missing :
           Base.summarysize(driven_prepared.plan),
       driven_compiled_standalone_bytes=driven_prepared === nothing ? missing :
           Base.summarysize(driven_prepared),
       driven_workspace_standalone_bytes=driven_workspace === nothing ? missing :
           Base.summarysize(driven_workspace),
       basis_standalone_bytes=Base.summarysize(basis),
       model_standalone_bytes=Base.summarysize(model),
       plan_standalone_bytes=Base.summarysize(prepared.plan),
       compiled_standalone_bytes=Base.summarysize(prepared),
       workspace_standalone_bytes=Base.summarysize(workspace),
       vectors_standalone_bytes=Base.summarysize(vectors),
       hot_retained_total_bytes=retained_total,
       validation_oracle=validation.oracle,
       sparse_relative_error=validation.sparse_error,
       trace_relative_error=validation.trace_error,
       adjoint_relative_error=validation.adjoint_error,
       auto_validation_passed=auto_passed,
       driven_validation_oracle=driven_validation === nothing ? missing :
           driven_validation.oracle,
       driven_sparse_relative_error=driven_validation === nothing ? missing :
           driven_validation.sparse_error,
       driven_trace_relative_error=driven_validation === nothing ? missing :
           driven_validation.trace_error,
       driven_adjoint_relative_error=driven_validation === nothing ? missing :
           driven_validation.adjoint_error,
       driven_validation_time_ns=driven_validation === nothing ? missing :
           driven_validation.elapsed,
       driven_validation_passed=driven_validation === nothing ? missing :
           driven_validation.passed,
       validation_tolerance=tolerance,
       validation_time_ns=validation.elapsed,
       validation_passed=validation_passed)
end

function main(args=ARGS)
    options = parse_options(args)
    BLAS.set_num_threads(1)
    println("PermutationalInvariantDynamics internal scaling benchmark")
    println("mode=$(options.mode), samples=$(options.samples), " *
            "seconds=$(options.seconds), BLAS threads=$(BLAS.get_num_threads())")
    rows = [run_case(case, options, index)
            for (index, case) in enumerate(benchmark_cases(options.mode))]
    all(row.validation_passed for row in rows) || error(
        "at least one benchmark validation failed; results were not written")

    write_tsv(options.output, rows, RESULT_COLUMNS)
    metadata = benchmark_metadata(
        mode=options.mode,
        requested_samples=options.samples,
        seconds=options.seconds,
        warmups=options.warmups,
        output=options.output,
        repository_root=normpath(joinpath(@__DIR__, "..")),
        extra=Pair{String,String}[
            "benchmark_suite" => "internal_scaling",
            "result_schema_version" => "2",
            "package_version" => string(Base.pkgversion(
                PermutationalInvariantDynamics)),
            "validation_dimension_limit" =>
                string(options.validation_dimension_limit),
            "validation_tolerance" => "1.0e-10",
            "rng_seed_policy" => "0x50490000 + 100*d + N + case_index",
            "matrixfree_memory_budget" => "Inf (explicit benchmark opt-out)",
            "sparse_phase_policy" =>
                "fully symmetric collective-only cases",
            "driven_phase_policy" =>
                "fully symmetric collective-only cases",
            "auto_memory_budget_bytes" =>
                string(AUTO_MEMORY_BUDGET_BYTES),
        ])
    metadata_output = metadata_path(options.output)
    write_metadata(metadata_output, metadata)
    println("results:  ", abspath(options.output))
    println("metadata: ", abspath(metadata_output))
    nothing
end

main()
