module ExampleOutputTestSupport
    # The output format is usable without importing a rendering backend.
    withenv("PID_EXAMPLE_RENDER" => "0") do
        include(joinpath(@__DIR__, "..", "examples", "utils", "makie_support.jl"))
    end
end

@testset "example data preserves values and validates before writing" begin
    support = ExampleOutputTestSupport.ExampleMakie
    @test !support.makie_available()
    @test !isdefined(support, :CairoMakie)
    @test !isdefined(support, :LaTeXStrings)
    @test support.latex("N=4") == "N=4"
    @test support.latex(raw"$\kappa t$") == raw"$\kappa t$"
    @test support.latex("ρ₁²") == "ρ₁²" # Preserve dependency-free SVG text.
    @test support.latex("ρ₁²"; renderer=:svg) == "ρ₁²"
    @test_throws ArgumentError support.latex("ρ₁²"; renderer=:unknown)
    @test support._latex_unicode_scripts("ρ₁²") == "ρ_{1}^{2}"
    @test support._latex_unicode_scripts("10⁻¹², J₊J₋, ω₀") == "10^{-12}, J_{+}J_{-}, ω_{0}"
    @test support._latex_unicode_scripts(raw"$\rho_1^2$") == raw"$\rho_1^2$"
    @test length(support._subscript_pairs) == length(support._subscript_characters)
    @test length(support._superscript_pairs) == length(support._superscript_characters)
    mktempdir() do directory
        setprecision(BigFloat, 192) do
            values = [BigFloat("1.000000000000000000000000000000000000001"), BigFloat(0)]
            path = support.save_example_data("checked", (
                time=[0, 1], value=values, method=["RK4", "exact"]);
                metadata=(N=8, note="first\nsecond"), directory)
            lines = readlines(path)
            @test any(line -> occursin("# N = 8", line), lines)
            @test any(line -> occursin(raw"first\nsecond", line), lines)
            table = filter(line -> !startswith(line, "#"), lines)
            @test table[1] == "time\tvalue\tmethod"
            @test length(table) == 3
            @test parse(BigFloat, split(table[2], '\t')[2]) == values[1]
            @test parse(BigFloat, split(table[3], '\t')[2]) == 0
            original = read(path)
            @test_throws DimensionMismatch support.save_example_data(
                "checked", (x=[1], y=[1, 2]); directory)
            @test_throws ArgumentError support.save_example_data(
                "checked", (x=[NaN],); directory)
            @test_throws ArgumentError support.save_example_data(
                "checked", (x=[1+im],); directory)
            @test_throws ArgumentError support.save_example_data(
                "checked", (x=["hidden\tcolumn"],); directory)
            @test_throws ArgumentError support.save_example_data(
                "../escape", (x=[1],); directory)
            @test read(path) == original
            @test readdir(directory) == ["checked.tsv"]
        end
    end
end
