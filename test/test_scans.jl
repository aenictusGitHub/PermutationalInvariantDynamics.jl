@testset "prepared parameter scans and continuation" begin
    basis=PIBasis(2,2)
    sm=ComplexF32[0 1;0 0]
    sp=Matrix(adjoint(sm))
    number=ComplexF32[0 0;0 1]

    function thermal_model(rate)
        PIModel(basis,(LocalJump(sm;rate=1.0f0),
                       LocalJump(sp;rate=rate)))
    end

    @test_throws ArgumentError ParameterScanPlan(Float32[],thermal_model)
    bad_builder=ParameterScanPlan([0.1f0],identity)
    bad_builder_result=parameter_scan(bad_builder)
    @test bad_builder_result[1].status===:failed
    @test occursin("expected PIModel",bad_builder_result[1].message)
    @test_throws ArgumentError ParameterScanPlan([0.1f0],thermal_model;
        solver_options=(workspace=nothing,))
    @test_throws ArgumentError ParameterScanPlan([0.1f0],thermal_model;
        algorithm=GMRESAlgorithm(krylovdim=8),
        solver_options=(krylovdim=4,))
    @test_throws ArgumentError ParameterScanPlan([0.1f0],thermal_model;
        task=:other)

    diagnostic=(rho,rate,index)->(
        excited=real(collective_expectation(rho,number))/basis.N,
        rate=rate,index=index)
    plan=ParameterScanPlan(Float32[0.1,0.2,0.4],thermal_model;
        algorithm=GMRESAlgorithm(krylovdim=12,maxiter=200),
        compile_options=(backend=:matrixfree,),
        solver_options=(atol=1.0f-6,rtol=1.0f-5),
        diagnostic=diagnostic)
    workspace=ParameterScanWorkspace()
    result=parameter_scan(plan;workspace)

    @test length(result)==3
    @test all(point->point.status===:success,result)
    @test all(point->point.converged,result)
    @test !result[1].warm_started
    @test result[2].warm_started
    @test !result[1].workspace_reused
    @test result[2].workspace_reused
    @test result.restart_index==3
    @test result.restart_seed isa PIState
    @test result.restart_seed!==workspace.continuation_seed
    @test result.restart_seed.data!==workspace.continuation_seed.data
    workspace_seed=copy(workspace.continuation_seed.data)
    result.restart_seed.data[1]+=one(eltype(result.restart_seed.data))
    @test workspace.continuation_seed.data==workspace_seed
    result.restart_seed.data.=workspace_seed
    @test eltype(result[1].output.data)==ComplexF32
    for point in result
        expected=point.parameter/(1+point.parameter)
        @test point.diagnostics.user.excited≈expected atol=3.0f-4
        @test abs(trace(point.output)-1)<3.0f-5
        @test point.compile_seconds>=0
        @test point.solve_seconds>=0
    end
    @test occursin("ParameterScanPlan",sprint(show,plan))
    @test occursin("successes=3",sprint(show,result))

    # Reusing allocations for a fresh invocation must not reuse the previous
    # path's final continuation state.
    fresh_prefix=parameter_scan(plan;workspace,max_points=1)
    @test !fresh_prefix[1].warm_started
    @test fresh_prefix[1].workspace_reused

    complete_workspace=ParameterScanWorkspace()
    complete_resume=resume_parameter_scan(plan,result;
        workspace=complete_workspace)
    @test complete_workspace.continuation_seed!==result.restart_seed
    @test complete_workspace.continuation_seed.data!==result.restart_seed.data
    @test complete_resume.restart_seed!==result.restart_seed
    @test complete_resume.restart_seed!==complete_workspace.continuation_seed

    rows=parameter_scan_rows(result)
    columns=parameter_scan_columns(result)
    @test rows[2].index==2
    @test columns.parameter==Float32[0.1,0.2,0.4]
    @test columns.status==fill(:success,3)
    @test !hasproperty(rows[1],:output)
    @test hasproperty(parameter_scan_rows(result;include_output=true)[1],:output)

    streamed=Any[]
    stream_plan=ParameterScanPlan(Float32[0.1,0.2,0.4],thermal_model;
        algorithm=GMRESAlgorithm(krylovdim=12,maxiter=200),
        compile_options=(backend=:matrixfree,),
        solver_options=(atol=1.0f-6,rtol=1.0f-5),
        save_outputs=false,save_restart=true)
    prefix=parameter_scan(stream_plan;max_points=1,
        callback=point->begin
            push!(streamed,point.output)
            fill!(point.output.data,0)
            nothing
        end)
    @test length(prefix)==1
    @test prefix[1].output===nothing
    @test !hasproperty(prefix[1].diagnostics.solver,:state)
    @test only(streamed) isa PIState
    @test prefix.restart_seed isa PIState
    @test abs(trace(prefix.restart_seed)-1)<3.0f-5
    merged_prefix=merge_parameter_scan_results(stream_plan,prefix)
    @test merged_prefix.restart_seed!==prefix.restart_seed
    @test merged_prefix.restart_seed.data!==prefix.restart_seed.data
    resumed=resume_parameter_scan(stream_plan,prefix)
    @test length(resumed)==3
    @test resumed[2].warm_started
    @test all(point->point.output===nothing,resumed)
    @test resumed.metadata.resumed
    paused=resume_parameter_scan(stream_plan,prefix;max_points=0)
    @test paused.restart_index==prefix.restart_index
    @test paused.restart_seed!==nothing
    @test paused.restart_seed!==prefix.restart_seed

    # A checkpoint without a restart vector resumes cold even if the supplied
    # workspace happens to contain a compatible seed from another scan.
    cold_plan=ParameterScanPlan(Float32[0.1,0.2,0.4],thermal_model;
        algorithm=GMRESAlgorithm(krylovdim=12,maxiter=200),
        compile_options=(backend=:matrixfree,),
        solver_options=(atol=1.0f-6,rtol=1.0f-5),
        save_restart=false)
    cold_prefix=parameter_scan(cold_plan;max_points=1)
    @test cold_prefix.restart_seed===nothing
    @test cold_prefix.restart_index==0
    cold_resume=resume_parameter_scan(cold_plan,cold_prefix;
        workspace,max_points=1)
    @test !cold_resume[2].warm_started
    mismatched_retention=ParameterScanPlan(
        Float32[0.1,0.2,0.4],thermal_model;
        save_outputs=true,save_restart=true)
    @test_throws ArgumentError resume_parameter_scan(
        mismatched_retention,prefix)
    @test_throws ArgumentError resume_parameter_scan(
        ParameterScanPlan(Float32[0.1,0.3,0.4],thermal_model),prefix)

    stopped=parameter_scan(stream_plan;
        callback=point->point.index==2 ? :stop : nothing)
    @test length(stopped)==2
    @test stopped.metadata.stopped

    failing_builder=(rate,index)->index==2 ?
        throw(ArgumentError("deliberate scan failure")) : thermal_model(rate)
    failure_plan=ParameterScanPlan(Float32[0.1,0.2,0.3],failing_builder;
        algorithm=GMRESAlgorithm(krylovdim=12,maxiter=200),
        compile_options=(backend=:matrixfree,),
        solver_options=(atol=1.0f-6,rtol=1.0f-5))
    recorded=parameter_scan(failure_plan;on_error=:record)
    @test [point.status for point in recorded]==[:success,:failed,:success]
    @test occursin("deliberate scan failure",recorded[2].message)
    @test !recorded[3].warm_started
    stopped_failure=parameter_scan(failure_plan;on_error=:stop)
    @test length(stopped_failure)==2
    @test stopped_failure[2].status===:failed
    failure_prefix=parameter_scan(failure_plan;max_points=1)
    failed_resume=resume_parameter_scan(failure_plan,failure_prefix;
        on_error=:stop)
    @test failed_resume.restart_index==1
    @test failed_resume.restart_seed!==nothing
    @test_throws ArgumentError parameter_scan(failure_plan;on_error=:throw)

    independent=ParameterScanPlan(Float32[0.1,0.2,0.3,0.4],thermal_model;
        algorithm=GMRESAlgorithm(krylovdim=12,maxiter=200),
        compile_options=(backend=:matrixfree,),
        solver_options=(atol=1.0f-6,rtol=1.0f-5),continuation=false)
    threaded=parameter_scan(independent;execution=:threads)
    @test [point.index for point in threaded]==collect(1:4)
    @test all(point->point.status===:success,threaded)
    threaded_callbacks=Int[]
    threaded_prefix=parameter_scan(independent;execution=:threads,
        callback=point->begin
            push!(threaded_callbacks,point.index)
            point.index==2 ? :stop : nothing
        end)
    @test [point.index for point in threaded_prefix]==[1,2]
    @test threaded_callbacks==[1,2]
    @test threaded_prefix.metadata.stopped
    @test_throws ArgumentError parameter_scan(independent;
        execution=:threads,callback=point->false)
    @test_throws ArgumentError parameter_scan(independent;
        callback=point->false)
    threaded_failure_plan=ParameterScanPlan(Float32[0.1,0.2,0.3],
        failing_builder;algorithm=GMRESAlgorithm(krylovdim=12,maxiter=200),
        compile_options=(backend=:matrixfree,),
        solver_options=(atol=1.0f-6,rtol=1.0f-5),continuation=false)
    threaded_failure=parameter_scan(threaded_failure_plan;
        execution=:threads,on_error=:stop)
    @test [point.status for point in threaded_failure]==[:success,:failed]
    @test_throws ErrorException parameter_scan(threaded_failure_plan;
        execution=:threads,on_error=:throw)
    chunk1=parameter_scan(independent;indices=1:2)
    chunk2=parameter_scan(independent;indices=3:4)
    merged=merge_parameter_scan_results(independent,chunk2,chunk1)
    @test [point.index for point in merged]==collect(1:4)
    @test_throws ArgumentError merge_parameter_scan_results(
        independent,chunk1,chunk1)

    continuation_chunk1=parameter_scan(stream_plan;indices=1:1)
    continuation_chunk2=parameter_scan(stream_plan;indices=2:2)
    @test_throws ArgumentError merge_parameter_scan_results(
        stream_plan,continuation_chunk1,continuation_chunk2)

    spectral_basis=PIBasis(1,2)
    sx=ComplexF64[0 1;1 0]
    sm64=ComplexF64[0 1;0 0]
    spectral_builder=drive->PIModel(spectral_basis,(
        LocalHamiltonian((drive/2)*sx),LocalJump(sm64;rate=1.0)))
    @test_throws ArgumentError ParameterScanPlan(
        [0.2],spectral_builder;task=:spectrum,
        nev=big(typemax(Int))+1)
    spectral_plan=ParameterScanPlan([0.2,0.3],spectral_builder;
        task=:spectrum,algorithm=:krylov,spectrum_target=:largest_real,
        nev=2,compile_options=(backend=:matrixfree,),
        solver_options=(krylovdim=4,atol=1e-10,rtol=1e-8))
    spectral=parameter_scan(spectral_plan)
    @test all(point->point.status===:success,spectral)
    @test spectral[1].output isa SpectrumResult
    @test length(spectral[1].output.values)==2
    @test spectral[1].output.vectors===nothing
    @test spectral[2].warm_started
    @test spectral[2].workspace_reused

    empty_chunk=parameter_scan(independent;indices=Int[])
    empty_columns=parameter_scan_columns(empty_chunk;include_output=true)
    @test isempty(empty_columns.index)
    @test hasproperty(empty_columns,:output)

    clear_parameter_scan_workspace!(workspace)
    @test workspace.continuation_seed===nothing
    @test workspace.solver_workspace===nothing
end
