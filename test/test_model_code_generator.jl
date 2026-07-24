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
    local_pseudomode_fixture=joinpath(
        root,"docs","test",
        "model_code_generator_local_pseudomode_fixture.js")
    global_pseudomode_fixture=joinpath(
        root,"docs","test",
        "model_code_generator_global_pseudomode_fixture.js")

    for path in (
            core,ui,stylesheet,page,javascript_tests,fixture,
            local_pseudomode_fixture,global_pseudomode_fixture)
        @test isfile(path)
    end

    core_source=read(core,String)
    ui_source=read(ui,String)
    stylesheet_source=read(stylesheet,String)
    page_source=read(page,String)
    make_source=read(joinpath(root,"docs","make.jl"),String)
    @test occursin("PIDModelCodeGenerator",core_source)
    @test occursin("\"local-pseudomode\"",core_source)
    @test occursin("\"global-pseudomode\"",core_source)
    @test occursin("pseudomode_supersite",core_source)
    @test occursin("global_pseudomode_model",core_source)
    @test occursin("id=\"pid-code-generator\"",page_source)
    @test occursin("id=\"pid-architecture\"",page_source)
    @test occursin("id=\"pid-pseudomode-section\"",page_source)
    @test occursin(
        "id=\"pid-pseudomode-coupling-operator\"",page_source)
    @test occursin("value=\"localPseudomode\"",page_source)
    @test occursin("value=\"globalPseudomode\"",page_source)
    @test occursin("\"Model code generator\"=>\"model_code_generator.md\"",
                   make_source)
    @test occursin("assets/model_code_generator_core.js",make_source)
    @test occursin("assets/model_code_generator_ui.js",make_source)
    @test occursin("pid-generator-page",ui_source)
    @test occursin("localPseudomode",ui_source)
    @test occursin("globalPseudomode",ui_source)
    @test occursin("updateArchitectureVisibility",ui_source)
    @test occursin(
        "html.pid-generator-page #documenter .docs-main",
        stylesheet_source)
    @test occursin("@media (max-width: 1320px)",stylesheet_source)
    @test occursin("container-type: inline-size",stylesheet_source)
    @test occursin(
        "@container pid-generator (min-width: 68rem)",
        stylesheet_source)
    @test occursin("minmax(0, 1.55fr)",stylesheet_source)
    @test occursin(".pid-pseudomode-grid",stylesheet_source)
    @test occursin("#pid-code-generator [hidden]",stylesheet_source)
    @test occursin(
        "class=\"pid-code-shell\" role=\"region\" tabindex=\"0\"",
        page_source)
    @test !occursin(r"\beval\s*\(",core_source)
    @test !occursin("new Function",core_source)
    @test !occursin(".innerHTML",ui_source)
    @test !occursin("\\operatorname",page_source)

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

        for (pseudomode_fixture,topology) in (
                (local_pseudomode_fixture,:local),
                (global_pseudomode_fixture,:global))
            pseudomode_command=node===nothing ?
                `$runner $core $pseudomode_fixture` :
                `$runner $pseudomode_fixture`
            pseudomode_generated=read(pseudomode_command,String)
            pseudomode_parsed=Meta.parseall(
                pseudomode_generated;
                filename="generated_$(topology)_pseudomode.jl")
            @test !has_parse_error(pseudomode_parsed)
            @test occursin("BosonicPseudomode",pseudomode_generated)
            @test occursin("PseudomodeCoupling",pseudomode_generated)
            @test occursin("mode_top_population",pseudomode_generated)
            @test !occursin("kron(",pseudomode_generated)
            if topology===:local
                @test occursin(
                    "pseudomode_supersite",pseudomode_generated)
                @test occursin(
                    "pseudomode_model(",pseudomode_generated)
                @test occursin(
                    "lift_system_operator",pseudomode_generated)
                @test occursin("backend=:auto",pseudomode_generated)
                @test !occursin(
                    "collective_spin(basis",pseudomode_generated)
            else
                @test occursin(
                    "global_pseudomode_model",pseudomode_generated)
                @test occursin(
                    "trace_pseudomodes",pseudomode_generated)
                @test occursin(
                    "global_pseudomode_state",pseudomode_generated)
                @test occursin(
                    "LinearAlgebra.ishermitian",pseudomode_generated)
                @test occursin(
                    "GMRESAlgorithm",pseudomode_generated)
                @test !occursin(
                    "compile(\n    embedding",pseudomode_generated)
            end
        end
    end
end
