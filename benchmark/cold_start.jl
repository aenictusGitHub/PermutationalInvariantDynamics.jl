module PIDColdStartBenchmark

using LinearAlgebra

include(joinpath(@__DIR__, "harness.jl"))
using .PIBenchmarkHarness

const RESULT_COLUMNS = (
    :schema_version,
    :implementation,
    :suite,
    :mode,
    :probe,
    :sample,
    :child_threads,
    :total_time_ns,
    :inner_load_time_ns,
    :pi_dimension,
    :validation_passed,
)

function usage(io=stdout)
    println(io, """
Usage:
  julia --startup-file=no --project=benchmark benchmark/cold_start.jl [options]

Options:
  --mode quick|full    Measurement effort (default: quick)
  --samples INTEGER   Fresh Julia processes per probe
  --warmups INTEGER   Discarded fresh processes per probe
  --threads INTEGER   Threads in every child Julia process (default: 1)
  --output PATH       Result TSV path
  --dry-run           Parse options and write empty TSV plus metadata
  -h, --help          Show this message

The parent process is not timed. Each row is a new Julia process with startup
and history files disabled. The package-load probe reports both the external
process wall time and the child's time spent executing `using
PermutationalInvariantDynamics`.
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
    child_threads = 1
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
        elseif startswith(argument, "--threads=")
            child_threads = parse(Int, split(argument, '='; limit=2)[2])
        elseif argument == "--threads"
            value, index = _option_value(args, index, argument)
            child_threads = parse(Int, value)
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
    child_threads > 0 || throw(ArgumentError("threads must be positive"))
    output = something(output, joinpath(
        @__DIR__, "results", "cold_start_$(mode).tsv"))
    (; mode, samples, warmups, child_threads, output, dry_run)
end

const _STARTUP_CODE = """
println("PID_COLD_START_OK")
"""

const _LOAD_CODE = """
started = time_ns()
using PermutationalInvariantDynamics
loaded = time_ns()
basis = PermutationalInvariantDynamics.PIBasis(2, 2)
println("PID_PACKAGE_LOAD_OK\\t", loaded - started, "\\t",
        PermutationalInvariantDynamics.pi_dimension(basis))
"""

function _child_command(code, child_threads)
    project_file = Base.active_project()
    project_file === nothing && throw(ArgumentError(
        "cold-start benchmark requires an active benchmark project"))
    project = dirname(abspath(project_file))
    julia = Base.julia_cmd()
    `$julia --startup-file=no --history-file=no --threads=$child_threads --project=$project -e $code`
end

function _run_child(probe::Symbol, child_threads::Int)
    code = probe === :julia_startup ? _STARTUP_CODE :
           probe === :package_load ? _LOAD_CODE :
           throw(ArgumentError("unknown cold-start probe $probe"))
    stdout = IOBuffer()
    stderr = IOBuffer()
    started = time_ns()
    process = run(pipeline(ignorestatus(_child_command(code, child_threads));
                           stdout, stderr))
    elapsed = time_ns() - started
    output = String(take!(stdout))
    error_output = String(take!(stderr))
    success(process) || throw(ErrorException(
        "child Julia process failed for $probe with exit code " *
        "$(process.exitcode): $(strip(error_output))"))

    if probe === :julia_startup
        strip(output) == "PID_COLD_START_OK" || throw(ErrorException(
            "startup probe returned unexpected output: $(repr(output))"))
        return (; total_time_ns=elapsed, inner_load_time_ns=missing,
                pi_dimension=missing, validation_passed=true)
    end

    line = only(filter(line -> startswith(line, "PID_PACKAGE_LOAD_OK\t"),
                       split(chomp(output), '\n')))
    fields = split(line, '\t')
    length(fields) == 3 || throw(ErrorException(
        "package-load probe returned malformed output: $(repr(line))"))
    inner = parse(UInt64, fields[2])
    dimension = parse(Int, fields[3])
    inner > 0 || throw(ErrorException("reported package-load time is zero"))
    dimension > 0 || throw(ErrorException("PI smoke-test dimension is invalid"))
    (; total_time_ns=elapsed, inner_load_time_ns=inner,
       pi_dimension=dimension, validation_passed=true)
end

function _rows(options)
    for probe in (:julia_startup, :package_load), _ in 1:options.warmups
        _run_child(probe, options.child_threads)
    end

    rows = NamedTuple[]
    for probe in (:julia_startup, :package_load)
        for sample in 1:options.samples
            measured = _run_child(probe, options.child_threads)
            push!(rows, (;
                schema_version=1,
                implementation="PermutationalInvariantDynamics",
                suite="cold_process_latency",
                mode=options.mode,
                probe,
                sample,
                child_threads=options.child_threads,
                measured...,
            ))
        end
    end
    rows
end

function main(args=ARGS)
    options = parse_options(args)
    options === nothing && return nothing
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
            "benchmark_kind" => "fresh_process_startup_and_load",
            "child_threads" => string(options.child_threads),
            "child_startup_file_disabled" => "true",
            "child_history_file_disabled" => "true",
            "measurement_policy" =>
                "external process wall time plus child package-load interval",
            "dry_run" => string(options.dry_run),
        ],
    )
    write_metadata(metadata_path(options.output), metadata)
    println("wrote ", options.output)
    println("wrote ", metadata_path(options.output))
    rows
end

end # module PIDColdStartBenchmark

if abspath(PROGRAM_FILE) == @__FILE__
    PIDColdStartBenchmark.main()
end
