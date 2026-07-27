using Test
import SciMLBase

@testset "guided PI studies and structured diagnostics" begin
    basis=PIBasis(1,2)
    spin=spin_matrices(2)
    model=PIModel(
        basis,
        (LocalJump(spin.jm;rate=1.0),
         LocalJump(spin.jp;rate=0.2)))

    study=PIStudy(
        model;task=:steady_state,algorithm=:direct,
        observables=(magnetization=spin.jz,))
    @test study.task===:steady_state
    @test occursin("PIStudy",sprint(show,study))
    report=check(study)
    @test report isa PIStudyReport
    @test report.dimension==length(basis)
    @test report.autonomous
    @test report.recommendation.algorithm===:direct
    @test !any(issue->issue.severity===:error,report.issues)
    @test explain(study).recommendation.algorithm===:direct
    @test occursin("PI study explanation",
        sprint(show,MIME"text/plain"(),report))
    @test check(model;task=:steady_state,algorithm=:direct) isa
        PIStudyReport

    result=solve(study)
    @test result isa PIStudyResult
    @test result.task===:steady_state
    @test result.raw isa SteadyStateResult
    @test result.state===result.raw.state
    @test result.states===nothing
    @test result.values===nothing
    @test result.converged===true
    @test result.selected_algorithm===:direct
    @test result.residual<=1e-10
    @test result.observables[:magnetization]≈-1/3 atol=1e-12
    @test result.diagnostics.physical.valid
    @test result_state(result)===result.state
    @test result_final_state(result)===result.state
    @test result_state(result.raw)===result.state
    @test result_times(result)===nothing
    @test result_states(result)===nothing
    @test result_values(result)===nothing
    @test result_observables(result)===result.observables
    @test result_converged(result.raw)===true
    @test result_residual(result.raw)==result.residual
    @test result_selected_algorithm(result.raw)===:direct
    @test result_stats(result.raw)===result.raw.info
    @test result_diagnostics(result.raw)===result.raw.info
    @test PermutationalInvariantDynamics.solve===SciMLBase.solve
    @test Workflow.PIStudy===PIStudy
    @test Workflow.solve===solve
    @test Workflow.result_state===result_state

    initial=computational_product_state(basis,1)
    dynamics_study=PIStudy(
        model;task=:dynamics,initial_state=initial,
        tspan=(0.0,0.1),saveat=[0.0,0.1],
        observables=(magnetization=spin.jz,),
        steps_per_interval=2)
    dynamics_report=check(dynamics_study)
    @test !any(issue->issue.severity===:error,dynamics_report.issues)
    @test dynamics_report.recommendation.task===:dynamics
    for invalid_saveat in (
            [0.0,0.08,0.02,0.1],
            [0.01,0.1],
            [0.0,Inf,0.1],
            -0.1)
        invalid_report=check(PIStudy(
            model;task=:dynamics,initial_state=initial,
            tspan=(0.0,0.1),saveat=invalid_saveat))
        @test invalid_report.runnable===false
        @test Symbol("PID-E-INVALID-SAVEAT") in
            Set(issue.code for issue in invalid_report.issues)
        @test invalid_report.recommendation===nothing
    end
    dynamics_result=solve(dynamics_study)
    @test dynamics_result.task===:dynamics
    @test dynamics_result.raw isa DynamicsStreamResult
    @test length(dynamics_result.states)==2
    @test dynamics_result.state===last(dynamics_result.states)
    @test result_final_state(dynamics_result)===last(dynamics_result.states)
    @test result_times(dynamics_result)===dynamics_result.raw.times
    @test result_states(dynamics_result)===dynamics_result.states
    @test result_values(dynamics_result)===nothing
    @test length(dynamics_result.observables[:magnetization])==2
    @test dynamics_result.converged===missing
    @test dynamics_result.selected_algorithm===:rk4
    @test dynamics_result.stats.samples==2
    @test dynamics_result.stats.saved_states==2
    @test dynamics_result.stats.observable_count==1
    @test dynamics_result.diagnostics.physical.valid

    spectrum_study=PIStudy(model;task=:spectrum,nev=2)
    spectrum_result=solve(spectrum_study)
    @test spectrum_result.task===:spectrum
    @test spectrum_result.raw isa SpectrumResult
    @test length(spectrum_result.values)==2
    @test result_values(spectrum_result)===spectrum_result.values
    @test result_times(spectrum_result)===nothing
    @test result_states(spectrum_result)===nothing
    @test spectrum_result.state===nothing
    @test spectrum_result.observables==NamedTuple()
    @test spectrum_result.selected_algorithm===:dense

    incomplete=PIStudy(model;task=:dynamics)
    incomplete_report=check(incomplete)
    @test incomplete_report.runnable===false
    incomplete_codes=Set(issue.code for issue in incomplete_report.issues)
    @test Symbol("PID-E-MISSING-INITIAL-STATE") in incomplete_codes
    @test Symbol("PID-E-MISSING-TSPAN") in incomplete_codes
    @test_throws ArgumentError solve(incomplete)

    other_basis=PIBasis(1,2)
    mismatched=PIStudy(
        model;task=:dynamics,
        initial_state=computational_product_state(other_basis,1),
        tspan=(0.0,1.0))
    @test Symbol("PID-E-BASIS-MISMATCH") in
        Set(issue.code for issue in check(mismatched).issues)

    driven=PIModel(
        basis,(LocalJump(spin.jm;rate=t->one(t)),))
    driven_report=check(PIStudy(driven;task=:steady_state))
    @test Symbol("PID-E-TIME-DEPENDENT-STATIONARY") in
        Set(issue.code for issue in driven_report.issues)

    negative=PIModel(
        basis,(LocalJump(spin.jm;rate=-0.1),))
    negative_report=check(PIStudy(
        negative;task=:dynamics,initial_state=initial,tspan=(0.0,0.1)))
    @test Symbol("PID-W-NEGATIVE-DETERMINISTIC-RATE") in
        Set(issue.code for issue in negative_report.issues)

    tiny_budget=check(PIStudy(
        model;task=:steady_state,algorithm=:direct,memory_budget=1))
    @test tiny_budget.runnable===false
    @test Symbol("PID-E-MEMORY-BUDGET") in
        Set(issue.code for issue in tiny_budget.issues)

    memory_issue=explain_failure(ArgumentError(
        "stationary_state exceeds memory_budget"))
    @test memory_issue.code===Symbol("PID-E-MEMORY-BUDGET")
    @test memory_issue.severity===:error
    dimension_issue=explain_failure(DimensionMismatch("wrong basis size"))
    @test dimension_issue.code===Symbol("PID-E-DIMENSION-MISMATCH")

    environment_only=doctor(smoke_test=false)
    @test environment_only isa PIDoctorReport
    @test environment_only.healthy
    @test environment_only.smoke_test.ran===false
    @test environment_only.smoke_test.passed===missing
    @test environment_only.environment.julia_version==VERSION
    @test hasproperty(environment_only.extensions,:Makie)
    @test occursin("PIDoctorReport",sprint(show,environment_only))

    smoke=doctor()
    @test smoke.healthy
    @test smoke.smoke_test.ran
    @test smoke.smoke_test.passed===true
    @test isempty(smoke.issues)
    @test occursin("PermutationalInvariantDynamics doctor",
        sprint(show,MIME"text/plain"(),smoke))
end
