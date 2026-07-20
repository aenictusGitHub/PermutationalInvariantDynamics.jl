const COMPARISON_ROOT = @__DIR__

function parse_arguments(arguments)
    output = joinpath(COMPARISON_ROOT, "results")
    samples = 20
    seconds = 1.0
    threads = 1
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        index == length(arguments) &&
            throw(ArgumentError("missing value after $argument"))
        value = arguments[index + 1]
        if argument == "--output"
            output = abspath(value)
        elseif argument == "--samples"
            samples = parse(Int, value)
            samples > 0 || throw(ArgumentError("--samples must be positive"))
        elseif argument == "--seconds"
            seconds = parse(Float64, value)
            isfinite(seconds) && seconds > 0 ||
                throw(ArgumentError("--seconds must be finite and positive"))
        elseif argument == "--threads"
            threads = parse(Int, value)
            threads > 0 || throw(ArgumentError("--threads must be positive"))
        else
            throw(ArgumentError("unknown argument $argument"))
        end
        index += 2
    end
    return (; output, samples, seconds, threads)
end

function combine_tables(destination, sources)
    header = nothing
    rows = String[]
    for source in sources
        lines = readlines(source)
        isempty(lines) && error("empty comparison table: $source")
        if isnothing(header)
            header = first(lines)
        elseif first(lines) != header
            error("inconsistent TSV schema in $source")
        end
        append!(rows, @view(lines[2:end]))
    end
    open(destination, "w") do io
        println(io, header)
        foreach(row -> println(io, row), rows)
    end
    return destination
end

function main(arguments=ARGS)
    options = parse_arguments(arguments)
    mkpath(options.output)
    outputs = String[]
    for name in ("pid", "quantumoptics", "quantumtoolbox")
        environment = joinpath(COMPARISON_ROOT, name)
        script = joinpath(environment, "run.jl")
        output = joinpath(options.output, "$name.tsv")
        command = `$(Base.julia_cmd()) --startup-file=no --threads=$(options.threads) --project=$environment $script --output $output --samples $(options.samples) --seconds $(options.seconds)`
        println("\n==> Running ", name)
        run(command)
        push!(outputs, output)
    end
    combined = combine_tables(joinpath(options.output, "comparison.tsv"), outputs)
    println("\nCombined comparison: ", combined)
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
