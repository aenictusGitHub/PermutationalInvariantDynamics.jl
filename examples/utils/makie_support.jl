module ExampleMakie

import TOML

export makie_available, makie_module, save_example_figure,
       figure_output_directory

const _load_error = Ref{Any}(nothing)
const _declared_in_active_project = let project=Base.active_project()
    project!==nothing&&isfile(project)&&
        haskey(get(TOML.parsefile(project),"deps",Dict{String,Any}()),
               "CairoMakie")
end
const _available = if !_declared_in_active_project
    _load_error[]=ArgumentError(
        "CairoMakie is not a direct dependency of the active project")
    false
else
    try
        @eval import CairoMakie
        CairoMakie.activate!(type="png")
        true
    catch error
        _load_error[] = error
        false
    end
end

"""Return whether CairoMakie is available in the active Julia environment."""
function makie_available()
    _available || @info(
        "Makie figure skipped. Prepare the examples environment with " *
        "`julia --project=examples -e 'using Pkg; " *
        "Pkg.develop(path=\".\"); Pkg.instantiate()'`.",
        reason=sprint(showerror, _load_error[]))
    _available
end

"""Return the loaded CairoMakie module, or throw when it is unavailable."""
makie_module() = _available ? CairoMakie : throw(ArgumentError(
    "CairoMakie is not available; call makie_available() before plotting"))

"""Directory used for generated example figures."""
figure_output_directory() = get(
    ENV, "PI_EXAMPLE_FIGURE_DIR",
    normpath(joinpath(@__DIR__, "..", "figures")))

"""
Display a Makie figure and save vector PDF and raster PNG copies.

Set `PI_EXAMPLE_FIGURE_DIR` to redirect output outside the default ignored
`examples/figures` directory.
"""
function save_example_figure(figure, stem::AbstractString;
                             formats=("pdf", "png"))
    _available || throw(ArgumentError("CairoMakie is not available"))
    occursin(r"^[A-Za-z0-9_.-]+$", stem) || throw(ArgumentError(
        "figure stem may contain only letters, digits, dot, underscore, and hyphen"))
    directory = figure_output_directory()
    mkpath(directory)
    paths = String[]
    for format in formats
        format in ("pdf", "png") || throw(ArgumentError(
            "supported example figure formats are pdf and png"))
        path = joinpath(directory, "$stem.$format")
        CairoMakie.save(path, figure)
        push!(paths, path)
    end
    display(figure)
    println("Makie figure written to ", join(paths, " and "))
    paths
end

end
