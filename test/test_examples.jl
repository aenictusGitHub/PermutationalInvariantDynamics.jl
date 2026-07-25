function _example_has_parse_error(value)
    value isa Expr||return false
    value.head in (:error,:incomplete)||any(_example_has_parse_error,value.args)
end

function _exact_julia_fences(markdown::AbstractString)
    lines=split(markdown,'\n';keepempty=true)
    blocks=Tuple{Int,String}[]
    line=1
    while line<=length(lines)
        if lines[line]=="```julia"
            opening_line=line
            line+=1
            first_source_line=line
            while line<=length(lines)&&lines[line]!="```"
                line+=1
            end
            line<=length(lines)||error(
                "unclosed exact Julia fence opened on line $opening_line")
            push!(blocks,(
                opening_line,
                join(lines[first_source_line:line-1],'\n')))
        end
        line+=1
    end
    blocks
end

function _markdown_fence_marker(line::AbstractString)
    bytes=codeunits(lstrip(line))
    isempty(bytes)&&return nothing
    marker=bytes[1]
    marker in (UInt8('`'),UInt8('~'))||return nothing
    run=1
    while run<length(bytes)&&bytes[run+1]==marker
        run+=1
    end
    run>=3 ? (marker,run) : nothing
end

function _matching_backtick_end(bytes,start,count)
    index=start
    while index<=length(bytes)
        if bytes[index]==UInt8('`')
            run=1
            while index+run<=length(bytes)&&
                    bytes[index+run]==UInt8('`')
                run+=1
            end
            run==count&&return index+run
            index+=run
        else
            index+=1
        end
    end
    nothing
end

function _prose_math_delimiter_violations(
        markdown::AbstractString;reject_single_dollar::Bool=true)
    violations=NamedTuple{(:line,:token),Tuple{Int,String}}[]
    fence=nothing
    for (line_number,line) in
            enumerate(split(markdown,'\n';keepempty=true))
        marker=_markdown_fence_marker(line)
        if fence===nothing
            if marker!==nothing
                fence=marker
                continue
            end
        else
            if marker!==nothing&&marker[1]==fence[1]&&marker[2]>=fence[2]
                fence=nothing
            end
            continue
        end

        bytes=codeunits(line)
        index=1
        while index<=length(bytes)
            byte=bytes[index]
            if byte==UInt8('`')
                run=1
                while index+run<=length(bytes)&&
                        bytes[index+run]==UInt8('`')
                    run+=1
                end
                closing=_matching_backtick_end(bytes,index+run,run)
                index=closing===nothing ? index+run : closing
            elseif byte==UInt8('\\')&&index<length(bytes)&&
                    bytes[index+1] in
                        (UInt8('('),UInt8(')'),UInt8('['),UInt8(']'))
                token=String(UInt8[byte,bytes[index+1]])
                push!(violations,(line=line_number,token=token))
                index+=2
            elseif reject_single_dollar&&byte==UInt8('$')
                preceding_backslashes=0
                previous=index-1
                while previous>=1&&bytes[previous]==UInt8('\\')
                    preceding_backslashes+=1
                    previous-=1
                end
                iseven(preceding_backslashes)&&
                    push!(violations,(line=line_number,token="\$"))
                index+=1
            else
                index+=1
            end
        end
    end
    violations
end

@testset "example inventory, syntax, and GitHub Markdown" begin
    @test _example_has_parse_error(Meta.parseall("x="))
    @test !_example_has_parse_error(Meta.parseall("x=1"))

    invalid_math=raw"""
legacy \(x\), \[y\], and $z$
"""
    @test _prose_math_delimiter_violations(invalid_math)==[
        (line=1,token=raw"\("),
        (line=1,token=raw"\)"),
        (line=1,token=raw"\["),
        (line=1,token=raw"\]"),
        (line=1,token="\$"),
        (line=1,token="\$"),
    ]
    valid_math=raw"""
`inline code \(x\) and $x$`
``inline Documenter math containing \(x\) and $x$``
```math
\[x\] + $y$
```
```julia
source = raw"\(x\) and $y$"
```
"""
    @test isempty(_prose_math_delimiter_violations(valid_math))

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
        @test isempty(_prose_math_delimiter_violations(
            markdown;reject_single_dollar=false))
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

    # Exact Julia fences in the Documenter sources are executable syntax, not
    # schematic pseudocode. This catches malformed snippets without evaluating
    # them or requiring their surrounding tutorial setup.
    docs_directory=normpath(joinpath(example_directory,"..","docs","src"))
    julia_fence_count=0
    markdown_count=0
    for (root,_,files) in walkdir(docs_directory)
        for filename in sort!(filter(name->endswith(name,".md"),files))
            path=joinpath(root,filename)
            markdown=read(path,String)
            markdown_count+=1
            violations=_prose_math_delimiter_violations(markdown)
            @test isempty(violations)
            for (opening_line,source) in _exact_julia_fences(markdown)
                julia_fence_count+=1
                parsed=Meta.parseall(
                    source;filename=path,lineno=opening_line+1)
                @test !_example_has_parse_error(parsed)
            end
        end
    end
    @test markdown_count>0
    @test julia_fence_count>0

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
