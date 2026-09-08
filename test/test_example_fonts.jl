# Rendering smoke test: julia --project=examples test/test_example_fonts.jl
# Deliberately separate from the dependency-free numerical example checks.
using Test
include(joinpath(@__DIR__, "..", "examples", "utils", "makie_support.jl"))
using .ExampleMakie
include(joinpath(@__DIR__, "test_example_plot_label_lint.jl"))

@testset "Bundled LaTeX fonts for example text and numeric ticks" begin
    @test makie_available()
    M = makie_module()
    original_theme = deepcopy(M.Makie.current_default_theme())
    try
        # Simulate another plot resetting Makie's process-wide theme. The
        # shared figure constructor must still own all four LaTeX font faces.
        M.set_theme!()
        figure = example_figure(size=(600, 400))
        fonts = figure.scene.theme.fonts
        for face in (:regular, :bold, :italic, :bolditalic)
            @test startswith(fonts[face][].family_name, "NewComputerModern")
        end
        axis = M.Axis(figure[1, 1];
            xlabel=latex(raw"$1/N$"), ylabel=latex(raw"$\langle J_z\rangle/N$"),
            title=latex("Computer Modern: labels, ticks and legend"))
        M.lines!(axis, [0.0, 1.0], [0.2, 0.8]; label=latex("PI solution"))
        M.axislegend(axis)
        @test axis.xticklabelfont[] === :regular
        @test axis.yticklabelfont[] === :regular
        @test axis.title[] isa M.Makie.LaTeXStrings.LaTeXString
        @test String(latex("ρ₁², 10⁻¹²")) == "ρ_{1}^{2}, 10^{-12}"
        @test latex("ρ₁²"; renderer=:svg) == "ρ₁²"
        # A Computer Modern font alone is not proof of math typesetting.
        # Check the actual glyph roles and script positions produced by Makie.
        E = M.Makie.MathTeXEngine
        glyphs(label) = filter(item -> item[1] isa E.TeXChar,
                               E.generate_tex_elements(label))
        char_entry(entries, char) = only(filter(item -> item[1].represented_char == char, entries))
        xglyphs = glyphs(axis.xlabel[])
        yglyphs = glyphs(axis.ylabel[])
        n = char_entry(xglyphs, 'N')
        j = char_entry(yglyphs, 'J')
        z = char_entry(yglyphs, 'z')
        @test E.is_slanted(n[1])
        @test E.is_slanted(j[1])
        @test E.is_slanted(z[1])
        @test z[2][2] < j[2][2] # True subscript, not "Jz" on the baseline.
        @test z[3] < j[3]
        @test all(char -> char_entry(yglyphs, char)[1].glyph_id != 0, ('⟨', '⟩'))
        @test all(item -> !E.is_slanted(item[1]), glyphs(latex("PI solution")))
        # The same strings without delimiters reproduce the historical bug.
        @test !E.is_slanted(char_entry(glyphs(latex("1/N")), 'N')[1])
        @test String(axis.ylabel[]) == raw"$\langle J_z\rangle/N$"
        mktempdir() do directory
            withenv("PI_EXAMPLE_FIGURE_DIR" => directory) do
                paths = save_example_figure(figure, "font_smoke";
                    formats=("png",), px_per_unit=1, display_figure=false)
                @test length(paths) == 1
                @test filesize(only(paths)) > 1000
            end
        end
    finally
        M.set_theme!(original_theme)
    end
end

@testset "Example label templates use supported TeX and real glyphs" begin
    M = makie_module()
    E = M.Makie.MathTeXEngine
    examples = joinpath(@__DIR__, "..", "examples")
    for file in sort(filter(path -> endswith(path, ".jl"), readdir(examples; join=true)))
        labels = ExamplePlotLabelLintTests.label_templates(read(file, String))
        for label in labels
            @testset "$(basename(file)):$(label.line): $(label.text)" begin
                elements = E.generate_tex_elements(latex(label.text))
                @test all(elements) do (element, _, _)
                    !(element isa E.TeXChar) ||
                        isspace(element.represented_char) || element.glyph_id != 0
                end
            end
        end
    end
end
