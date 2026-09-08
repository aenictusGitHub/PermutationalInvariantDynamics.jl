# Standalone syntax gate: no package, plotting backend, or macro is loaded.
module ExamplePlotLabelLintTests

using Test

const LABEL_FIELDS = (:label, :xlabel, :ylabel, :title, :text)
const Hit = NamedTuple{(:file, :line, :keyword, :snippet),
                       Tuple{String, Int, Symbol, String}}

function terminal_name(node)
    node isa Symbol && return node
    node isa QuoteNode && return terminal_name(node.value)
    node isa GlobalRef && return node.name
    if node isa Expr
        node.head === :. && return terminal_name(last(node.args))
        # Observable/property updates, e.g. ax.title[] = "...".
        node.head === :ref && return terminal_name(first(node.args))
    end
    return nothing
end

function formatted_string(node)
    node isa Expr || return false
    node.head === :macrocall &&
        return terminal_name(first(node.args)) === Symbol("@L_str")
    return node.head === :call && terminal_name(first(node.args)) in
        (:latex, :latex_text, :latexstring, :LaTeXString)
end

function plain_string(node)
    node isa String && return true
    node isa Expr || return false
    node.head === :string && return true  # Includes interpolation.
    if node.head === :macrocall
        return terminal_name(first(node.args)) === Symbol("@raw_str")
    elseif node.head === :call
        name = terminal_name(first(node.args))
        name in (:string, :join) && return true
        name === :String && return any(plain_string, node.args[2:end])
        # Even concatenating two LaTeXStrings loses the formatted type.
        name === :* && return any(
            arg -> plain_string(arg) || formatted_string(arg), node.args[2:end])
    elseif node.head === :if
        return any(plain_string, node.args[2:end])
    elseif node.head in (:vect, :tuple)
        return any(plain_string, node.args)
    end
    # Explicit LaTeX wrappers are accepted without inspecting their payload.
    # This is a syntax lint, not inference of arbitrary variables/call results.
    return false
end

function walk!(hits, node, file, line)
    node isa LineNumberNode && return Int(node.line)
    node isa Expr || return line
    if node.head in (:error, :incomplete)
        push!(hits, Hit((file, line, :parse, "invalid Julia syntax")))
        return line
    end
    if node.head in (:kw, :(=), :(.=)) && length(node.args) == 2
        field = terminal_name(node.args[1])
        if field in LABEL_FIELDS && plain_string(node.args[2])
            push!(hits, Hit((file, line, field, repr(node.args[2]))))
        end
    elseif node.head === :call && terminal_name(first(node.args)) in (:text, :text!)
        # Makie also accepts positional text, including vectors of strings.
        for arg in node.args[2:end]
            plain_string(arg) && push!(hits, Hit((file, line, :text, repr(arg))))
        end
    end
    for child in node.args
        line = walk!(hits, child, file, line)
    end
    return line
end

function violations(source::AbstractString; file="<fixture>")
    parsed = Meta.parseall(source; filename=file)
    hits = Hit[]
    walk!(hits, parsed, String(file), 1)
    return hits
end

# Reconstruct literal label templates only, never evaluate example code.
# Interpolated values use a harmless number for TeX syntax/layout coverage;
# labels obtained from arbitrary variables or callbacks still need review.
function label_template(node)
    node isa String && return node
    node isa Expr || return nothing
    if node.head === :string
        return join(arg isa String ? arg : "4" for arg in node.args)
    elseif node.head === :macrocall && terminal_name(first(node.args)) === Symbol("@raw_str")
        return last(node.args)
    elseif node.head === :call && terminal_name(first(node.args)) in (:*, :string)
        parts = label_template.(node.args[2:end])
        any(!isnothing, parts) || return nothing
        return join(isnothing(part) ? "4" : part for part in parts)
    end
    nothing
end

function label_templates!(labels, node, line=1)
    node isa LineNumberNode && return Int(node.line)
    node isa Expr || return line
    wrapper = isempty(node.args) ? nothing : terminal_name(first(node.args))
    if node.head === :. && length(node.args) == 2 &&
            node.args[2] isa Expr && node.args[2].head === :tuple &&
            wrapper in (:latex, :latex_text, :LaTeXString)
        # Broadcasted explicit tick labels, e.g. latex.([raw"$E_0$", ...]).
        for args in node.args[2].args
            if args isa Expr && args.head in (:vect, :tuple)
                for arg in args.args
                    template = label_template(arg)
                    isnothing(template) || push!(labels, (; line, text=template))
                end
            end
        end
    elseif node.head === :call && wrapper in (:latex, :latex_text, :LaTeXString)
        # SVG deliberately expects Unicode, not TeX delimiters.
        svg = any(node.args) do arg
            arg isa Expr && arg.head === :parameters && any(arg.args) do kw
                kw isa Expr && kw.head === :kw && kw.args[1] === :renderer &&
                    kw.args[2] == QuoteNode(:svg)
            end
        end
        if !svg
            for arg in node.args[2:end]
                template = label_template(arg)
                isnothing(template) || push!(labels, (; line, text=template))
            end
        end
    end
    for child in node.args
        line = label_templates!(labels, child, line)
    end
    line
end

function label_templates(source::AbstractString)
    labels = NamedTuple{(:line, :text), Tuple{Int, String}}[]
    label_templates!(labels, Meta.parseall(source))
    labels
end

function math_label_problem(text)
    # Remove math spans while respecting escaped dollars and backslashes.
    prose = IOBuffer()
    in_math = false
    escaped = false
    for char in text
        if escaped
            in_math || print(prose, '\\', char)
            escaped = false
        elseif char === '\\'
            escaped = true
        elseif char === '$'
            in_math = !in_math
            print(prose, ' ')
        elseif !in_math
            print(prose, char)
        end
    end
    in_math && return "unclosed math delimiter"
    outside = String(take!(prose))
    # Deliberately bounded: flag common scientific syntax, not arbitrary
    # English words. Whitelisting the wrapper's type alone misses this bug.
    occursin(r"[α-ωΑ-Ωϵϑϕ⟨⟩ℒ₀-₉⁰¹²³⁴⁵⁶⁷⁸⁹_^]|\\[A-Za-z]+|\b1\s*/\s*N\b|\b[JSC][xyz]{1,2}\b|\bN\s*=", outside) &&
        return "formula outside math delimiters"
    occursin(r"[JS]\\_[xyz]", text) &&
        return "escaped underscore prints literally; use a mathematical subscript"
    occursin(r"\\mathrm\{[^{}]*(?:1\s*/\s*N|[JS]_[xyz]|\\langle)", text) &&
        return "variables/formulas in mathrm; reserve it for operator names"
    nothing
end

function math_violations(source::AbstractString; file="<fixture>")
    hits = Hit[]
    for label in label_templates(source)
        problem = math_label_problem(label.text)
        isnothing(problem) || push!(hits,
            Hit((file, label.line, :math, "$problem: $(repr(label.text))")))
    end
    hits
end

@testset "Mixed text/math labels delimit scientific formulas" begin
    for payload in ("1 / N", "⟨Jz⟩", "γ / Ω", raw"\langle J_z\rangle", "ρ₁",
                    raw"$1/N", raw"$\langle J\_z\rangle$",
                    raw"$\mathrm{1/N}$", raw"$\mathrm{\langle J_z\rangle}$")
        source = "Axis(f; ylabel=ExampleMakie.latex($(repr(payload))))"
        @test length(math_violations(source)) == 1
    end
    for payload in (raw"$1/N$", raw"$\langle J_z\rangle/N$", "PI solution",
                    raw"polarization $\langle J_z\rangle$",
                    raw"literal price \$4, $N=4$", raw"$\mathrm{Re}\,\lambda$")
        source = "Axis(f; ylabel=ExampleMakie.latex($(repr(payload))))"
        @test isempty(math_violations(source))
    end
    @test isempty(math_violations(raw"""title=latex("N=$N, γ"; renderer=:svg)"""))
    @test isempty(math_violations(raw"""title=latex("finite \$N=$N\$")"""))
    @test length(math_violations(raw"""title=latex("N=" * string(N))""")) == 1
    @test only(label_templates(raw"""title=latex("finite \$N=$N\$")""")).text == raw"finite $N=4$"
    @test isempty(label_templates("latex(builder())")) # Never execute callbacks.
    @test length(label_templates(raw"""latex.([raw"$E_0$", raw"$E_1$"])""")) == 2
    @test length(math_violations(raw"""latex.(["σ₋", "σz"])""")) == 2
end

@testset "Plot-label lint detects regressions without executing examples" begin
    for field in LABEL_FIELDS
        for source in (
            "plot(x; $field=\"plain\")",
            "plot(x, $field=\"plain\")",
            "($field=\"plain\",)",
            "(; $field=\"plain\")",
            "$field = \"plain\"",
            "ax.$field = \"plain\"",
            "ax.$field[] = \"plain\"",
            "ax.$field .= \"plain\"",
            "plot(x; $field=\"N=\$N\")",
        )
            hits = violations(source)
            @test length(hits) == 1
            @test only(hits).keyword === field
        end
    end
    for source in (
        raw"Axis(fig; title=raw\"\rho\")",
        raw"Axis(fig; title=string(N))",
        raw"Axis(fig; title=Base.string(N))",
        raw"Axis(fig; title=String(\"plain\"))",
        raw"Axis(fig; title=join(parts, \", \"))",
        raw"Axis(fig; title=\"N=\" * string(N))",
        raw"Axis(fig; title=L\"N\" * \" extra\")",
        raw"Axis(fig; title=L\"N\" * L\"t\")",
        raw"Axis(fig; title=latexstring(\"N\") * latexstring(\"t\"))",
        raw"Axis(fig; title=ExampleMakie.latex(\"N\") * \" extra\")",
        raw"Axis(fig; title=flag ? L\"N\" : \"plain\")",
        raw"M.text!(ax, \"plain\")",
        raw"Makie.text(ax, [\"first\", \"second\"])",
        raw"M.text!(ax; text=[\"first\", \"second\"])",
    )
        hits = violations(source)
        @test length(hits) == 1
        @test only(hits).keyword in (:title, :text)
    end
    for source in (
        raw"Axis(fig; title=L\"N\", xlabel=LaTeXStrings.L\"t\")",
        raw"Axis(fig; title=ExampleMakie.latex(\"N=$N\"))",
        raw"Axis(fig; title=ExampleMakie.latex(\"N=\" * string(N)))",
        raw"Axis(fig; title=LaTeXStrings.latexstring(\"N=\", N))",
        raw"Axis(fig; title=latexstring(\"N=\", N))",
        raw"(; title=ExampleMakie.latex_text(\"caption\"))",
        raw"ax.title[] = L\"N\"",
        raw"M.text!(ax, [L\"N\", L\"t\"])",
        raw"Axis(fig; title=prepared_label, xlabel=nothing, ylabel=automatic)",
        raw"HOPSBath(Q, c, nu; label=:bath)",
        raw"(; label=String(label), x, y)", # Non-plot contour-export metadata.
        raw"println(\"title=plain\")",
        "# Axis(fig; title=\"plain\")\nx = 1",
        "\"\"\"Use title=\\\"plain\\\" in this documented snippet.\"\"\"\nf() = nothing",
        # Would throw if the gate evaluated source or expanded macros.
        "error(\"do not execute\")\n@undefined_macro Axis(fig; title=L\"N\")",
    )
        @test isempty(violations(source))
    end
    hits = violations("# first line\nax.title = \"bad\"\nax.xlabel = \"bad\"";
                      file="nested/example.jl")
    @test getproperty.(hits, :line) == [2, 3]
    @test all(hit -> hit.file == "nested/example.jl", hits)
    # Julia records the enclosing expression's line for multiline keywords.
    hits = violations("\nAxis(fig;\n    title=\"bad\",\n    xlabel=\"bad\")")
    @test getproperty.(hits, :keyword) == [:title, :xlabel]
    @test all(hit -> hit.line == 2, hits)
    @test any(hit -> hit.keyword === :parse, violations("Axis(; title="))
end

@testset "All example plot labels use LaTeX formatting" begin
    root = dirname(@__DIR__)
    files = sort!([joinpath(dir, name)
                   for (dir, _, names) in walkdir(joinpath(root, "examples"))
                   for name in names if endswith(name, ".jl")])
    @test !isempty(files)
    for file in files
        source = read(file, String)
        hits = vcat(violations(source; file=relpath(file, root)),
                    math_violations(source; file=relpath(file, root)))
        for hit in hits
            reason = hit.keyword === :parse ? "invalid Julia syntax" :
                hit.keyword === :math ? hit.snippet :
                "plain field $(hit.keyword): $(hit.snippet); wrap the complete value " *
                "in ExampleMakie.latex(...) or use L\"...\""
            println(stderr, "$(hit.file):$(hit.line): $reason")
        end
        @test isempty(hits)
    end
end

end # module ExamplePlotLabelLintTests
