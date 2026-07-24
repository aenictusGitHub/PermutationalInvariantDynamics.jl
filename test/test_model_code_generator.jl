@testset "static PI model code generator" begin
    has_parse_error=nothing
    has_parse_error=value->value isa Expr&&(
        value.head in (:error,:incomplete)||
        any(has_parse_error,value.args))
    root=normpath(joinpath(@__DIR__,".."))
    core=joinpath(root,"docs","src","assets","model_code_generator_core.js")
    ui=joinpath(root,"docs","src","assets","model_code_generator_ui.js")
    stylesheet=joinpath(
        root,"docs","src","assets","model_code_generator.css")
    page=joinpath(root,"docs","src","model_code_generator.md")
    javascript_tests=joinpath(
        root,"docs","test","model_code_generator_tests.js")
    fixture=joinpath(
        root,"docs","test","model_code_generator_fixture.js")

    for path in (core,ui,stylesheet,page,javascript_tests,fixture)
        @test isfile(path)
    end

    core_source=read(core,String)
    ui_source=read(ui,String)
    page_source=read(page,String)
    make_source=read(joinpath(root,"docs","make.jl"),String)
    @test occursin("PIDModelCodeGenerator",core_source)
    @test occursin("id=\"pid-code-generator\"",page_source)
    @test occursin("\"Model code generator\"=>\"model_code_generator.md\"",
                   make_source)
    @test occursin("assets/model_code_generator_core.js",make_source)
    @test occursin("assets/model_code_generator_ui.js",make_source)
    @test !occursin(r"\beval\s*\(",core_source)
    @test !occursin("new Function",core_source)
    @test !occursin(".innerHTML",ui_source)

    node=Sys.which("node")
    mac_jsc="/System/Library/Frameworks/JavaScriptCore.framework/"*
            "Versions/A/Helpers/jsc"
    runner=node===nothing&&isfile(mac_jsc) ? mac_jsc : node
    if runner===nothing
        @test_skip false
    else
        test_command=node===nothing ?
            `$runner $core $javascript_tests` :
            `$runner $javascript_tests`
        test_output=read(test_command,String)
        @test occursin("model code generator tests:",test_output)

        ui_command=node===nothing ?
            `$runner $core $ui` :
            `$runner --check $ui`
        @test success(ui_command)

        fixture_command=node===nothing ?
            `$runner $core $fixture` :
            `$runner $fixture`
        generated=read(fixture_command,String)
        parsed=Meta.parseall(generated;filename="generated_pi_steady_state.jl")
        @test !has_parse_error(parsed)
        @test occursin("DirectPIHamiltonian",generated)
        @test occursin("LocalJump",generated)
        @test occursin("CollectiveJump",generated)
        @test occursin("CollectiveObservablePlan",generated)
        @test !occursin("kron(",generated)
    end
end
