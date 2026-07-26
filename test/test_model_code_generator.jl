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
    trajectory_fixture=joinpath(
        root,"docs","test",
        "model_code_generator_trajectory_fixture.js")
    local_pseudomode_trajectory_fixture=joinpath(
        root,"docs","test",
        "model_code_generator_local_pseudomode_trajectory_fixture.js")
    dynamics_fixture=joinpath(
        root,"docs","test","model_code_generator_dynamics_fixture.js")
    local_pseudomode_dynamics_fixture=joinpath(
        root,"docs","test",
        "model_code_generator_local_pseudomode_dynamics_fixture.js")
    trajectory_dynamics_fixture=joinpath(
        root,"docs","test",
        "model_code_generator_trajectory_dynamics_fixture.js")
    local_pseudomode_trajectory_dynamics_fixture=joinpath(
        root,"docs","test",
        "model_code_generator_local_pseudomode_trajectory_dynamics_fixture.js")
    global_pseudomode_trajectory_dynamics_fixture=joinpath(
        root,"docs","test",
        "model_code_generator_global_pseudomode_trajectory_dynamics_fixture.js")
    spectrum_fixture=joinpath(
        root,"docs","test","model_code_generator_spectrum_fixture.js")
    global_pseudomode_spectrum_fixture=joinpath(
        root,"docs","test",
        "model_code_generator_global_pseudomode_spectrum_fixture.js")
    gap_fixture=joinpath(
        root,"docs","test","model_code_generator_gap_fixture.js")
    global_pseudomode_gap_fixture=joinpath(
        root,"docs","test",
        "model_code_generator_global_pseudomode_gap_fixture.js")
    analysis_fixture=joinpath(
        root,"docs","test","model_code_generator_analysis_fixture.js")
    local_pseudomode_analysis_fixture=joinpath(
        root,"docs","test",
        "model_code_generator_local_pseudomode_analysis_fixture.js")
    global_pseudomode_analysis_fixture=joinpath(
        root,"docs","test",
        "model_code_generator_global_pseudomode_analysis_fixture.js")

    for path in (
            core,ui,stylesheet,page,javascript_tests,fixture,
            local_pseudomode_fixture,global_pseudomode_fixture,
            trajectory_fixture,local_pseudomode_trajectory_fixture,
            dynamics_fixture,local_pseudomode_dynamics_fixture,
            trajectory_dynamics_fixture,
            local_pseudomode_trajectory_dynamics_fixture,
            global_pseudomode_trajectory_dynamics_fixture,
            spectrum_fixture,global_pseudomode_spectrum_fixture,
            gap_fixture,global_pseudomode_gap_fixture,
            analysis_fixture,local_pseudomode_analysis_fixture,
            global_pseudomode_analysis_fixture)
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
    @test occursin("id=\"pid-calculation\"",page_source)
    @test occursin("id=\"pid-steady-method\"",page_source)
    @test occursin("id=\"pid-trajectory-section\"",page_source)
    @test occursin("id=\"pid-trajectory-count\"",page_source)
    @test occursin("id=\"pid-initial-level\"",page_source)
    @test occursin("id=\"pid-trajectory-settling-time\"",page_source)
    @test occursin("id=\"pid-trajectory-dt\"",page_source)
    @test occursin("id=\"pid-trajectory-samples\"",page_source)
    @test occursin(
        "id=\"pid-trajectory-sampling-interval\"",page_source)
    @test occursin(
        "id=\"pid-trajectory-max-jump-probability\"",page_source)
    @test occursin("id=\"pid-trajectory-seed\"",page_source)
    for id in (
            "pid-dynamics-section","pid-dynamics-start-time",
            "pid-dynamics-final-time","pid-dynamics-samples",
            "pid-dynamics-steps","pid-spectrum-section",
            "pid-spectrum-target","pid-spectrum-nev","pid-spectrum-seed",
            "pid-gap-section","pid-gap-nev","pid-gap-krylovdim",
            "pid-analysis-section","pid-analysis-purity",
            "pid-analysis-entropy","pid-analysis-one-body-rdm",
            "pid-analysis-qfi-axis","pid-memory-budget")
        @test occursin("id=\"$id\"",page_source)
    end
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
    @test occursin("steadyMethod",ui_source)
    @test occursin("initialState",ui_source)
    @test occursin("maxJumpProbability",ui_source)
    @test occursin("function updateVisibility()",ui_source)
    @test occursin("function readConfiguration()",ui_source)
    page_ids=[
        match.captures[1]
        for match in eachmatch(r"id=\"([^\"]+)\"",page_source)
    ]
    @test length(page_ids)==length(unique(page_ids))
    ui_id_selectors=unique(
        match.match[2:end]
        for match in eachmatch(r"#pid-[A-Za-z0-9_-]+",ui_source))
    @test all(selector->selector in page_ids,ui_id_selectors)
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

        trajectory_command=node===nothing ?
            `$runner $core $trajectory_fixture` :
            `$runner $trajectory_fixture`
        trajectory_generated=read(trajectory_command,String)
        trajectory_parsed=Meta.parseall(
            trajectory_generated;
            filename="generated_pi_trajectory_steady_state.jl")
        @test !has_parse_error(trajectory_parsed)
        @test occursin(
            "rho0 = computational_product_state(basis, INITIAL_LEVEL)",
            trajectory_generated)
        @test occursin("TrajectoryPlan(model)",trajectory_generated)
        @test occursin(
            "TrajectoryBatchWorkspace(",trajectory_generated)
        @test occursin("trajectory_steady_state(",trajectory_generated)
        @test occursin("return_info=true",trajectory_generated)
        @test occursin("const TRAJECTORIES = 4",trajectory_generated)
        @test occursin("const INITIAL_LEVEL = 1",trajectory_generated)
        @test occursin(
            "const MAX_JUMP_PROBABILITY = 0.02",
            trajectory_generated)
        @test !occursin("stationary_state(",trajectory_generated)
        @test !occursin("steady.info.",trajectory_generated)

        local_trajectory_command=node===nothing ?
            `$runner $core $local_pseudomode_trajectory_fixture` :
            `$runner $local_pseudomode_trajectory_fixture`
        local_trajectory_generated=read(local_trajectory_command,String)
        local_trajectory_parsed=Meta.parseall(
            local_trajectory_generated;
            filename="generated_local_pseudomode_trajectory_steady_state.jl")
        @test !has_parse_error(local_trajectory_parsed)
        @test occursin(
            "rho0 = pseudomode_product_state(",
            local_trajectory_generated)
        @test occursin(
            "system_initial[INITIAL_LEVEL] = 1",
            local_trajectory_generated)
        @test occursin(
            "trajectory_plan = TrajectoryPlan(model)",
            local_trajectory_generated)
        @test occursin(
            "trajectory_steady_state(",local_trajectory_generated)
        @test !occursin("stationary_state(",local_trajectory_generated)
        @test !occursin("prepared = compile(",local_trajectory_generated)

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

        expanded_fixture_specs=(
            (
                key=:dynamics,
                path=dynamics_fixture,
                filename="generated_pi_dynamics.jl",
                fragments=(
                    "const INITIAL_LEVEL = 2",
                    "backend=:matrixfree",
                    "dynamics = solve_dynamics(",
                    "save_states=false",
                ),
            ),
            (
                key=:local_dynamics,
                path=local_pseudomode_dynamics_fixture,
                filename="generated_local_pseudomode_dynamics.jl",
                fragments=(
                    "rho0 = pseudomode_product_state(",
                    "lift_system_operator(site, spin.jz",
                    "dynamics = solve_dynamics(",
                ),
            ),
            (
                key=:trajectory_dynamics,
                path=trajectory_dynamics_fixture,
                filename="generated_pi_trajectory_dynamics.jl",
                fragments=(
                    "TrajectoryPlan(model)",
                    "TrajectoryBatchWorkspace(",
                    "dynamics = quantum_trajectories(",
                    "streaming_observable = observable",
                    "observable_standard_error",
                ),
            ),
            (
                key=:local_trajectory_dynamics,
                path=local_pseudomode_trajectory_dynamics_fixture,
                filename="generated_local_pseudomode_trajectory_dynamics.jl",
                fragments=(
                    "rho0 = pseudomode_product_state(",
                    "TrajectoryBatchWorkspace(",
                    "dynamics = quantum_trajectories(",
                ),
            ),
            (
                key=:global_trajectory_dynamics,
                path=global_pseudomode_trajectory_dynamics_fixture,
                filename="generated_global_pseudomode_trajectory_dynamics.jl",
                fragments=(
                    "CompositeTrajectoryPlan(",
                    "CompositeTrajectoryBatchWorkspace(",
                    "streaming_observable = observable",
                    "dynamics = quantum_trajectories(",
                ),
            ),
            (
                key=:spectrum,
                path=spectrum_fixture,
                filename="generated_pi_spectrum.jl",
                fragments=(
                    "spectrum = liouvillian_spectrum(",
                    "target=:largest_real",
                    "Random.MersenneTwister(SPECTRUM_SEED)",
                    "return_info=true",
                ),
            ),
            (
                key=:global_spectrum,
                path=global_pseudomode_spectrum_fixture,
                filename="generated_global_pseudomode_spectrum.jl",
                fragments=(
                    "spectrum = liouvillian_spectrum(\n    embedding;",
                    "target=:near_zero",
                    "return_info=true",
                ),
            ),
            (
                key=:gap,
                path=gap_fixture,
                filename="generated_pi_gap.jl",
                fragments=(
                    "gap_source = model",
                    "gap_result = pi_liouvillian_gap(",
                    "nev=GAP_NEV, krylovdim=GAP_KRYLOVDIM",
                    "gap_result.gap_certified",
                ),
            ),
            (
                key=:global_gap,
                path=global_pseudomode_gap_fixture,
                filename="generated_global_pseudomode_gap.jl",
                fragments=(
                    "gap_source = global_pseudomode_matrixfree(",
                    "gap_result = pi_liouvillian_gap(",
                    "gap_result.gap_certified",
                ),
            ),
            (
                key=:analysis,
                path=analysis_fixture,
                filename="generated_pi_analysis.jl",
                fragments=(
                    "analysis_state = rho_ss",
                    "system_purity = purity(analysis_state)",
                    "system_entropy = von_neumann_entropy(",
                    "one_body_density_matrix = one_body_rdm(",
                    "qfi_value = qfi(",
                    "analysis_state, qfi_plan; atol=STATE_VALIDATION_TOL)",
                ),
            ),
            (
                key=:local_analysis,
                path=local_pseudomode_analysis_fixture,
                filename="generated_local_pseudomode_analysis.jl",
                fragments=(
                    "analysis_trace_plan = pseudomode_trace_plan(",
                    "analysis_state = trace_pseudomodes(",
                    "analysis_basis = analysis_trace_plan.output_basis",
                ),
            ),
            (
                key=:global_analysis,
                path=global_pseudomode_analysis_fixture,
                filename="generated_global_pseudomode_analysis.jl",
                fragments=(
                    "analysis_state = rho_system",
                    "analysis_basis = system_basis",
                    "spin.jy; cache=analysis_geometry",
                ),
            ),
        )
        expanded_generated=Dict{Symbol,String}()
        for spec in expanded_fixture_specs
            command=node===nothing ?
                `$runner $core $(spec.path)` :
                `$runner $(spec.path)`
            source=read(command,String)
            parsed=Meta.parseall(source;filename=spec.filename)
            @test !has_parse_error(parsed)
            @test !occursin("kron(",source)
            for fragment in spec.fragments
                @test occursin(fragment,source)
            end
            expanded_generated[spec.key]=source
        end

        function execute_generated(source,filename)
            generated_module=Module(gensym(:GeneratedPIModel))
            Core.eval(generated_module,:(using Base))
            redirect_stdout(devnull) do
                Base.invokelatest(
                    Base.include_string,generated_module,source,filename)
            end
            return generated_module
        end

        dynamics_module=execute_generated(
            expanded_generated[:dynamics],"runtime_pi_dynamics.jl")
        dynamics_values=Core.eval(dynamics_module,:observable_values)
        @test length(dynamics_values)==3
        @test all(isfinite,dynamics_values)

        trajectory_dynamics_module=execute_generated(
            expanded_generated[:trajectory_dynamics],
            "runtime_pi_trajectory_dynamics.jl")
        trajectory_values=Core.eval(
            trajectory_dynamics_module,:observable_values)
        trajectory_errors=Core.eval(
            trajectory_dynamics_module,:observable_standard_error)
        @test length(trajectory_values)==3
        @test all(isfinite,trajectory_values)
        @test length(trajectory_errors)==3
        @test all(error->isfinite(error)&&error>=0,trajectory_errors)

        spectrum_module=execute_generated(
            expanded_generated[:spectrum],"runtime_pi_spectrum.jl")
        spectrum_values=Core.eval(spectrum_module,:spectrum_values)
        @test length(spectrum_values)>=2
        @test all(isfinite,spectrum_values)

        analysis_module=execute_generated(
            expanded_generated[:analysis],"runtime_pi_analysis.jl")
        @test isfinite(Core.eval(analysis_module,:system_purity))
        @test isfinite(Core.eval(analysis_module,:system_entropy))
        @test size(Core.eval(
            analysis_module,:one_body_density_matrix))==(2,2)
        @test isfinite(Core.eval(analysis_module,:qfi_value))
    end
end
