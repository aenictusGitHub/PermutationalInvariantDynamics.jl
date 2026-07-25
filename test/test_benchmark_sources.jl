@testset "benchmark source syntax" begin
    function parse_diagnostics!(diagnostics,node)
        node isa Expr||return diagnostics
        node.head in (:error,:incomplete)&&push!(diagnostics,node)
        for argument in node.args
            parse_diagnostics!(diagnostics,argument)
        end
        diagnostics
    end

    @test !isempty(parse_diagnostics!(
        Expr[],Meta.parseall("incomplete_assignment =")))

    repository_root=normpath(joinpath(@__DIR__,".."))
    benchmark_root=joinpath(repository_root,"benchmark")
    benchmark_sources=String[]
    for (directory,_,files) in walkdir(benchmark_root)
        append!(
            benchmark_sources,
            joinpath(directory,file) for file in files
            if endswith(file,".jl"))
    end
    sort!(benchmark_sources)
    @test !isempty(benchmark_sources)
    for path in benchmark_sources
        parsed=Meta.parseall(read(path,String);filename=path)
        @test parsed isa Expr
        @test isempty(parse_diagnostics!(Expr[],parsed))
    end
end
