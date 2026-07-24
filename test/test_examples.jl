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
    figure_directory=normpath(joinpath(
        example_directory,"..","docs","src","assets","example_figures"))
    referenced_figures=Set{String}()
    for guide in guides
        markdown=read(joinpath(example_directory,guide),String)
        @test !occursin(raw"\operatorname",markdown)
        @test occursin("## Expected output",markdown)

        matches=collect(eachmatch(
            r"!\[[^\]]+\]\(\.\./docs/src/assets/example_figures/([^)[:space:]]+)\)",
            markdown))
        @test !isempty(matches)
        for match in matches
            filename=only(match.captures)
            push!(referenced_figures,filename)
            @test isfile(joinpath(figure_directory,filename))
        end
    end

    # Curated figures are static documentation assets. Numerical assertions in
    # the source examples remain the scientific regression; CI checks only
    # that each reviewed raster/vector asset is structurally valid, bounded,
    # embedded by a guide, and not orphaned.
    figure_files=Set(filter(
        name->endswith(name,".png")||endswith(name,".svg"),
        readdir(figure_directory)))
    @test figure_files==referenced_figures
    @test !any(endswith(name,".pdf") for name in readdir(figure_directory))

    png_signature=UInt8[0x89,0x50,0x4e,0x47,0x0d,0x0a,0x1a,0x0a]
    for filename in sort!(collect(figure_files))
        path=joinpath(figure_directory,filename)
        @test filesize(path)<=2^21
        if endswith(filename,".png")
            bytes=read(path)
            @test length(bytes)>=24
            @test bytes[1:8]==png_signature
            @test String(bytes[13:16])=="IHDR"
            width=foldl(
                (value,byte)->(value<<8)|UInt32(byte),
                bytes[17:20];init=UInt32(0))
            height=foldl(
                (value,byte)->(value<<8)|UInt32(byte),
                bytes[21:24];init=UInt32(0))
            @test width>=320
            @test height>=240
        else
            svg=read(path,String)
            @test startswith(lstrip(svg),"<svg")
            @test occursin("</svg>",svg)
        end
    end
end
