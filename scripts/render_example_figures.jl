# Regenerate checked default example figures; never replace documentation
# snapshots automatically. Uses the separate, optional examples environment.
using TOML

const FIGURE_REPOSITORY_ROOT = normpath(joinpath(@__DIR__, ".."))

function render_example_figures(arguments=ARGS)
    catalog = TOML.parsefile(joinpath(FIGURE_REPOSITORY_ROOT,
                                    "examples", "catalog.toml"))["example"]
    available = Dict(splitext(basename(entry["script"]))[1] => entry
        for entry in catalog if "CairoMakie" in entry["optional_dependencies"] &&
        entry["id"] != "paper-models") # Shared constructors, not a rendering script.
    if arguments == ["--list"]
        foreach(println, sort!(collect(keys(available))))
        return
    end
    selected = isempty(arguments) ? sort!(collect(keys(available))) : arguments
    for name in selected
        haskey(available, name) || throw(ArgumentError(
            "unknown figure example $name; use --list for available names"))
    end
    project = Base.active_project()
    project !== nothing && isfile(project) &&
        haskey(get(TOML.parsefile(project), "deps", Dict()), "CairoMakie") ||
        error("Use julia --project=examples scripts/render_example_figures.jl")

    failures = String[]
    output = abspath(get(ENV, "PI_EXAMPLE_FIGURE_DIR",
                        joinpath(FIGURE_REPOSITORY_ROOT, "examples", "figures")))
    mkpath(output)
    for (index, name) in enumerate(selected)
        println("Rendering checked example $index/$(length(selected)): $name")
        flush(stdout)
        script = joinpath(FIGURE_REPOSITORY_ROOT, available[name]["script"])
        # Isolate globals, RNG state and solver/figure storage in a fresh
        # process. Never reduce the example's grid or numerical tolerances.
        command = `$(Base.julia_cmd()) --startup-file=no --project=$(dirname(project)) $script`
        try
            # Require fresh outputs, so a skipped renderer cannot be mistaken
            # for success just because an old PNG/PDF already exists.
            mktempdir(output; prefix=".render-$name-") do staging
                run(addenv(command, "PID_EXAMPLE_RENDER" => "1",
                    "PID_EXAMPLE_QUICK" => "0", "PI_EXAMPLE_FIGURE_DIR" => staging))
                expected = filter(path -> endswith(path, ".png"),
                                  available[name]["expected_outputs"])
                isempty(expected) && error("No PNG output declared for $name")
                for path in expected, format in ("png", "pdf")
                    generated = joinpath(staging, splitext(basename(path))[1] * ".$format")
                    isfile(generated) && filesize(generated) > 0 ||
                        error("Example $name did not render $generated")
                end
                for file in readdir(staging)
                    mv(joinpath(staging, file), joinpath(output, file); force=true)
                end
            end
        catch error
            error isa ProcessFailedException || rethrow()
            push!(failures, name)
        end
    end
    isempty(failures) || error("Figure examples failed: " * join(failures, ", "))
    println("Refreshed $(length(selected)) checked example scripts.")
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && render_example_figures()
