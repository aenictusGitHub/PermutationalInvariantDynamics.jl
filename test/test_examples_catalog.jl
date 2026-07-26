using Test
using TOML
using PermutationalInvariantDynamics

@testset "curated examples catalog" begin
    root = normpath(joinpath(@__DIR__, ".."))
    catalog_path = joinpath(root, "examples", "catalog.toml")
    catalog = TOML.parsefile(catalog_path)
    model_catalog=Models.catalog()

    @test catalog["schema_version"] == 1
    entries = catalog["example"]
    @test length(entries) >= 12

    allowed_difficulties = Set(("beginner", "intermediate", "advanced"))
    allowed_runtime_classes = Set(("instant", "short", "medium", "long"))
    slug = r"^[a-z0-9]+(?:-[a-z0-9]+)*$"
    dependency_name = r"^[A-Za-z][A-Za-z0-9_]*$"

    ids = String[]
    scripts = String[]
    guides = String[]
    covered_difficulties = Set{String}()
    covered_runtime_classes = Set{String}()

    for entry in entries
        for field in (
            "id", "title", "script", "guide", "difficulty", "tasks",
            "runtime_class", "stochastic", "optional_dependencies",
            "citation", "expected_outputs",
        )
            @test haskey(entry, field)
        end

        id = entry["id"]
        script = entry["script"]
        guide = entry["guide"]
        difficulty = entry["difficulty"]
        runtime_class = entry["runtime_class"]

        @test id isa String && occursin(slug, id)
        @test entry["title"] isa String && !isempty(strip(entry["title"]))
        @test script isa String && startswith(script, "examples/")
        @test guide isa String && startswith(guide, "examples/")
        @test splitext(script)[2] == ".jl"
        @test splitext(guide)[2] == ".md"
        @test splitext(basename(script))[1] == splitext(basename(guide))[1]
        @test difficulty in allowed_difficulties
        @test runtime_class in allowed_runtime_classes
        @test entry["stochastic"] isa Bool
        @test entry["citation"] isa String &&
              !isempty(strip(entry["citation"]))

        tasks = entry["tasks"]
        @test tasks isa Vector && !isempty(tasks)
        @test all(task -> task isa String && occursin(slug, task), tasks)
        @test length(tasks) == length(unique(tasks))

        optional_dependencies = entry["optional_dependencies"]
        @test optional_dependencies isa Vector
        @test all(dep -> dep isa String &&
                         occursin(dependency_name, dep),
                  optional_dependencies)
        @test length(optional_dependencies) ==
              length(unique(optional_dependencies))

        script_path = joinpath(root, script)
        guide_path = joinpath(root, guide)
        @test isfile(script_path)
        @test isfile(guide_path)

        guide_text = read(guide_path, String)
        @test occursin(basename(script), guide_text)
        @test occursin("## Expected output", guide_text)

        outputs = entry["expected_outputs"]
        @test outputs isa Vector && !isempty(outputs)
        @test length(outputs) == length(unique(outputs))
        for output in outputs
            @test output isa String
            @test startswith(output, "docs/src/assets/example_figures/")
            @test splitext(output)[2] in (".png", ".svg")
            @test isfile(joinpath(root, output))
            @test occursin(output, guide_text)
        end

        if haskey(entry, "model_recipe")
            recipe=entry["model_recipe"]
            @test occursin(r"^Models\.[a-z][a-z0-9_]*$",recipe)
            recipe_name=Symbol(split(recipe,'.';limit=2)[2])
            @test isdefined(Models,recipe_name)
            @test getfield(Models,recipe_name) isa Function
            @test hasproperty(model_catalog,recipe_name)
        end

        push!(ids, id)
        push!(scripts, script)
        push!(guides, guide)
        push!(covered_difficulties, difficulty)
        push!(covered_runtime_classes, runtime_class)
    end

    @test length(ids) == length(unique(ids))
    @test length(scripts) == length(unique(scripts))
    @test length(guides) == length(unique(guides))
    @test covered_difficulties == allowed_difficulties
    @test Set(("short", "medium", "long")) ⊆ covered_runtime_classes
    @test any(entry -> entry["stochastic"], entries)
    @test any(entry -> !entry["stochastic"], entries)
    @test any(entry -> haskey(entry, "model_recipe"), entries)
end
