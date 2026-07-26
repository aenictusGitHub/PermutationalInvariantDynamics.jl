using Test

include(joinpath(@__DIR__, "cold_start.jl"))
include(joinpath(@__DIR__, "time_to_solution.jl"))

@testset "productization benchmark CLI and dry-run metadata" begin
    batched_source=read(
        joinpath(@__DIR__,"batched_trajectories.jl"),String)
    @test Meta.parseall(batched_source) isa Expr
    @test occursin("BatchedConditionalWorkspace",batched_source)
    @test occursin("scalar_conditional_action!",batched_source)

    @test PIDColdStartBenchmark.parse_options([
        "--mode=quick", "--samples=3", "--warmups=0", "--threads=2",
        "--output=/tmp/cold.tsv",
    ]).child_threads == 2
    @test_throws ArgumentError PIDColdStartBenchmark.parse_options([
        "--mode=invalid",
    ])

    @test PIDTimeToSolutionBenchmark.parse_options([
        "--mode", "full", "--samples", "2", "--warmups", "0",
        "--memory-budget-mib", "64", "--output", "/tmp/tts.tsv",
    ]).memory_budget == 64 * 1024^2
    @test_throws ArgumentError PIDTimeToSolutionBenchmark.parse_options([
        "--memory-budget-mib=0",
    ])

    mktempdir() do directory
        cold = joinpath(directory, "cold.tsv")
        cold_rows = PIDColdStartBenchmark.main([
            "--mode=quick", "--samples=1", "--warmups=0",
            "--output=$cold", "--dry-run",
        ])
        @test isempty(cold_rows)
        @test isfile(cold)
        cold_metadata =
            PIDColdStartBenchmark.PIBenchmarkHarness.metadata_path(cold)
        @test isfile(cold_metadata)
        @test occursin("dry_run\ttrue", read(cold_metadata, String))

        tts = joinpath(directory, "tts.tsv")
        tts_rows = PIDTimeToSolutionBenchmark.main([
            "--mode=quick", "--samples=1", "--warmups=0",
            "--output=$tts", "--dry-run",
        ])
        @test isempty(tts_rows)
        @test isfile(tts)
        metadata =
            PIDTimeToSolutionBenchmark.PIBenchmarkHarness.metadata_path(tts)
        @test isfile(metadata)
        @test occursin("phase_policy\tsetup,solve,validation,total",
                       read(metadata, String))
    end
end
