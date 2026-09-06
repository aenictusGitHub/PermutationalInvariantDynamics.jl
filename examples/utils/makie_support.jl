module ExampleMakie

import TOML

export makie_available, makie_module, save_example_figure,
       figure_output_directory, example_figure, example_colors,
       save_example_data

# Distinct luminance and hue, with markers/dashes used for overlapping series.
const example_colors = (blue="#0072B2", orange="#E69F00", green="#009E73",
                        red="#D55E00", purple="#CC79A7", gray="#667085")

const _load_error = Ref{Any}(nothing)
const _render_enabled = lowercase(strip(get(
    ENV, "PID_EXAMPLE_RENDER", "1"))) ∉ ("0", "false", "no", "off")
const _declared_in_active_project = let project=Base.active_project()
    project!==nothing&&isfile(project)&&
        haskey(get(TOML.parsefile(project),"deps",Dict{String,Any}()),
               "CairoMakie")
end
const _available = if !_render_enabled
    _load_error[]=ArgumentError(
        "example rendering was disabled by PID_EXAMPLE_RENDER")
    false
elseif !_declared_in_active_project
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
    _available || !_render_enabled || @info(
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

"""Construct a consistently styled figure without changing Makie's global theme."""
function example_figure(; kwargs...)
    M = makie_module()
    M.Figure(;
        figure_padding=24, fontsize=18, backgroundcolor=:white,
        Axis=(titlealign=:left, titlesize=18, titlegap=12,
              xlabelsize=17, ylabelsize=17, xticklabelsize=14, yticklabelsize=14,
              topspinevisible=false, rightspinevisible=false,
              xgridcolor=(:black, 0.06), ygridcolor=(:black, 0.06),
              spinewidth=1, xticksize=4, yticksize=4),
        Legend=(labelsize=14, framevisible=false, backgroundcolor=(:white, 0.92),
                patchsize=(26, 12), rowgap=5),
        Colorbar=(labelsize=16, ticklabelsize=14, spinewidth=0, width=14),
        Lines=(linewidth=2.5,), Scatter=(markersize=8,),
        palette=(color=collect(values(example_colors)),), kwargs...)
end

function _validate_stem(stem)
    occursin(r"^[A-Za-z0-9_.-]+$", stem) && stem ∉ (".", "..") ||
        throw(ArgumentError(
            "output stem must be a filename using letters, digits, dot, underscore, or hyphen"))
end

"""
    save_example_data(stem, columns; metadata=(;), directory=figure_output_directory())

Write checked, equal-length named vector columns to UTF-8 TSV. Comment lines
record controls and Julia version; values retain their original precision.
Only finite real numbers and single-line strings are accepted. No plotting
dependency is needed. Examples call this inside their optional output block,
so `PID_EXAMPLE_RENDER=0` still produces no files.
"""
function save_example_data(stem::AbstractString, columns::NamedTuple;
                           metadata::NamedTuple=(;),
                           directory=figure_output_directory())
    _validate_stem(stem)
    isempty(columns) && throw(ArgumentError("data needs at least one column"))
    all(column -> column isa AbstractVector, values(columns)) ||
        throw(ArgumentError("data columns must be vectors"))
    rows = length(first(columns))
    all(column -> length(column) == rows, values(columns)) ||
        throw(DimensionMismatch("data columns must have equal lengths"))
    safe_text(value) = !any(c -> c in ('\t', '\r', '\n'), string(value))
    all(safe_text, keys(columns)) || throw(ArgumentError("invalid column name"))
    for column in values(columns), value in column
        (value isa Real && isfinite(value)) ||
        (value isa AbstractString && safe_text(value)) || throw(ArgumentError(
            "data cells must be finite real numbers or single-line strings"))
    end
    all(safe_text, keys(metadata)) || throw(ArgumentError("invalid metadata name"))
    mkpath(directory)
    path = joinpath(directory, "$stem.tsv")
    temporary, io = mktemp(directory)
    try
        println(io, "# PermutationalInvariantDynamics.jl example data; format_version = 1")
        println(io, "# julia_version = ", VERSION)
        for (key, value) in pairs(metadata)
            # Escaped newlines keep metadata from injecting table rows.
            rendered = replace(repr(value), '\n'=>"\\n", '\r'=>"\\r", '\t'=>"\\t")
            println(io, "# ", key, " = ", rendered)
        end
        println(io, join(string.(keys(columns)), '\t'))
        for row in 1:rows
            println(io, join((column[firstindex(column) + row - 1]
                              for column in values(columns)), '\t'))
        end
        close(io)
        Base.Filesystem.rename(temporary, path)
    finally
        isopen(io) && close(io)
        isfile(temporary) && rm(temporary)
    end
    println("Example data written to ", path)
    path
end

"""
Save vector PDF and high-resolution PNG copies; display in interactive Julia.

Set `PI_EXAMPLE_FIGURE_DIR` to redirect output outside the default ignored
`examples/figures` directory.
"""
function save_example_figure(figure, stem::AbstractString;
                             formats=("pdf", "png"), px_per_unit=2,
                             display_figure=isinteractive())
    _available || throw(ArgumentError("CairoMakie is not available"))
    _validate_stem(stem)
    isfinite(px_per_unit) && px_per_unit > 0 || throw(ArgumentError(
        "px_per_unit must be finite and positive"))
    all(format -> format in ("pdf", "png"), formats) || throw(ArgumentError(
        "supported example figure formats are pdf and png"))
    directory = figure_output_directory()
    mkpath(directory)
    paths = String[]
    for format in formats
        path = joinpath(directory, "$stem.$format")
        if format == "png"
            CairoMakie.save(path, figure; px_per_unit)
        else
            CairoMakie.save(path, figure; pt_per_unit=1)
        end
        push!(paths, path)
    end
    display_figure && display(figure)
    println("Makie figure written to ", join(paths, " and "))
    paths
end

end
