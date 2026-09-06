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
