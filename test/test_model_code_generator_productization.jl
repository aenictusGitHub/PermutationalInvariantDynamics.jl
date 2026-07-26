@testset "model generator experiment bundles" begin
    root=normpath(joinpath(@__DIR__,".."))
    core=joinpath(
        root,"docs","src","assets","model_code_generator_core.js")
    ui=joinpath(
        root,"docs","src","assets","model_code_generator_ui.js")
    page=joinpath(root,"docs","src","model_code_generator.md")
    javascript_tests=joinpath(
        root,"docs","test","model_code_generator_productization_tests.js")
    verified_fixture=joinpath(
        root,"docs","test","model_code_generator_verified_fixture.js")
    verified_stationary_fixture=joinpath(
        root,"docs","test",
        "model_code_generator_verified_stationary_fixture.js")

    for path in (
            core,ui,page,javascript_tests,verified_fixture,
            verified_stationary_fixture)
        @test isfile(path)
    end
    core_source=read(core,String)
    ui_source=read(ui,String)
    page_source=read(page,String)
    @test occursin("verified-experiment",core_source)
    @test occursin("manifestFor",core_source)
    @test occursin("boundedBinomial",core_source)
    @test occursin("id=\"pid-workflow\"",page_source)
    @test occursin("id=\"pid-download-bundle\"",page_source)
    @test occursin("function downloadBundle()",ui_source)
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
        output=read(test_command,String)
        @test occursin(
            "model code generator productization tests: 22 passed",output)

        fixture_command=node===nothing ?
            `$runner $core $verified_fixture` :
            `$runner $verified_fixture`
        generated=read(fixture_command,String)
        parsed=Meta.parseall(generated;filename="generated_verified_experiment.jl")
        has_parse_error=nothing
        has_parse_error=value->value isa Expr&&(
            value.head in (:error,:incomplete)||
            any(has_parse_error,value.args))
        @test !has_parse_error(parsed)
        @test occursin("PIExperiment(",generated)
        @test occursin("explain_experiment(",generated)
        @test occursin("verified_solve(",generated)
        @test occursin("save_states=true",generated)

        stationary_command=node===nothing ?
            `$runner $core $verified_stationary_fixture` :
            `$runner $verified_stationary_fixture`
        stationary=read(stationary_command,String)
        stationary_parsed=Meta.parseall(
            stationary;filename="generated_verified_stationary.jl")
        @test !has_parse_error(stationary_parsed)
        @test occursin("task=:steady_state",stationary)
        @test occursin("solver_options=(atol=STEADY_ATOL",stationary)
        @test occursin("steady = experiment_result.solution",stationary)

        function execute_verified(source,filename)
            generated_module=Module(gensym(:GeneratedVerifiedPIModel))
            Core.eval(generated_module,:(using Base))
            redirect_stdout(devnull) do
                Base.invokelatest(
                    Base.include_string,generated_module,source,filename)
            end
            generated_module
        end

        dynamics_module=execute_verified(
            generated,"runtime_verified_dynamics.jl")
        dynamics_result=Core.eval(dynamics_module,:experiment_result)
        @test dynamics_result.report.verified
        @test length(dynamics_result.solution.times)==3

        stationary_module=execute_verified(
            stationary,"runtime_verified_stationary.jl")
        stationary_result=Core.eval(stationary_module,:experiment_result)
        @test stationary_result.report.verified
        @test validate_state(stationary_result.solution.state)===
            stationary_result.solution.state
    end
end
