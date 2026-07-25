const REPOSITORY_ROOT = normpath(joinpath(@__DIR__, ".."))
const EXAMPLE_DIRECTORY = joinpath(REPOSITORY_ROOT, "examples")

# Keep this list deliberately small and cross-cutting. Each entry runs its
# default numerical assertions; "quick" refers to suite selection, not weaker
# physics or solver tolerances.
const QUICK_EXAMPLES = (
    "getting_started",
    "pbody_pair_processes",
    "composite_ensembles",
    "qubit_population_dynamics",
    "global_pseudomode_cavity",
    "pi_heom",
)

function _parse_shard(value::AbstractString)
    match_result = match(r"^([1-9][0-9]*)/([1-9][0-9]*)$", value)
    match_result === nothing &&
        error("shard must have the form INDEX/TOTAL, received $value")
    index, total = parse.(Int, match_result.captures)
    index <= total ||
        error("shard index $index exceeds shard count $total")
    index, total
end

function _selected_examples(arguments::Vector{String})
    if arguments == ["--list"]
        foreach(println, QUICK_EXAMPLES)
        return nothing
    elseif isempty(arguments)
        return collect(QUICK_EXAMPLES)
    elseif length(arguments) == 2 && arguments[1] == "--shard"
        index, total = _parse_shard(arguments[2])
        return [name for (position, name) in enumerate(QUICK_EXAMPLES)
                if mod1(position, total) == index]
    end
    error("usage: julia --project=. test/run_quick_examples.jl " *
          "[--list | --shard INDEX/TOTAL]")
end

function _run_example(name::AbstractString)
    path = joinpath(EXAMPLE_DIRECTORY, "$name.jl")
    isfile(path) || error("quick example does not exist: $path")
    println("\n", repeat("=", 72))
    println("quick example: ", name)
    println(repeat("=", 72))
    flush(stdout)
    command = `$(Base.julia_cmd()) --startup-file=no --threads=1 --project=$REPOSITORY_ROOT $path`
    run(addenv(
        command,
        "PID_EXAMPLE_QUICK" => "1",
        "PID_EXAMPLE_RENDER" => "0",
    ))
    nothing
end

function main(arguments::Vector{String}=ARGS)
    selected = _selected_examples(arguments)
    selected === nothing && return nothing
    isempty(selected) && error("selected shard contains no examples")

    # The root environment intentionally lacks CairoMakie. The explicit switch
    # also prevents rendering if a future dependency happens to load it.
    ENV["PID_EXAMPLE_QUICK"] = "1"
    ENV["PID_EXAMPLE_RENDER"] = "0"
    mktempdir() do output_directory
        ENV["PI_EXAMPLE_FIGURE_DIR"] = output_directory
        for name in selected
            _run_example(name)
        end
        isempty(readdir(output_directory)) ||
            error("quick examples generated files despite disabled rendering")
    end
    println("\nquick-example shard passed: ", join(selected, ", "))
    nothing
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
