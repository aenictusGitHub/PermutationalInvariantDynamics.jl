function _example_has_parse_error(value)
    value isa Expr||return false
    value.head in (:error,:incomplete)||any(_example_has_parse_error,value.args)
end

@testset "example inventory, syntax, and GitHub Markdown" begin
    @test _example_has_parse_error(Meta.parseall("x="))
    @test !_example_has_parse_error(Meta.parseall("x=1"))

    example_directory=normpath(joinpath(@__DIR__,"..","examples"))
    entries=readdir(example_directory)
    scripts=sort(filter(name->endswith(name,".jl"),entries))
    guides=sort(filter(name->endswith(name,".md")&&name!="README.md",entries))
    script_stems=Set(first(splitext(name)) for name in scripts)
    guide_stems=Set(first(splitext(name)) for name in guides)

    @test script_stems==guide_stems

    readme=read(joinpath(example_directory,"README.md"),String)
    for script in scripts
        source=read(joinpath(example_directory,script),String)
        parsed=Meta.parseall(
            source;filename=joinpath(example_directory,script),lineno=1)
        @test !_example_has_parse_error(parsed)
        @test occursin("`$script`",readme)
    end

    # GitHub's math renderer rejects this macro. Keep the paired guides on
    # portable primitives such as \mathrm and \mathop.
    for guide in guides
        markdown=read(joinpath(example_directory,guide),String)
        @test !occursin(raw"\operatorname",markdown)
    end
end
