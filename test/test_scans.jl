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
        compile_options=(backend=:matrixfree,memory_budget=1024,))
    @test_throws ArgumentError ParameterScanPlan([0.1f0],thermal_model;
        solver_options=(memory_budget=1024,))
    @test_throws ArgumentError ParameterScanPlan([0.1f0],thermal_model;
        memory_budget=-1)
    @test_throws ArgumentError ParameterScanPlan([0.1f0],thermal_model;
        memory_budget=true)
    @test_throws ArgumentError ParameterScanPlan([0.1f0],thermal_model;
        memory_budget=NaN)
    @test_throws ArgumentError ParameterScanPlan([0.1f0],thermal_model;
        algorithm=GMRESAlgorithm(krylovdim=8),
        solver_options=(krylovdim=4,))
    @test_throws ArgumentError ParameterScanPlan([0.1f0],thermal_model;
        task=:other)

    # Float32 iterative outputs retain 8-byte ComplexF32 coordinates, whereas
    # factorizing steady-state and dense spectral routes return 16-byte
    # ComplexF64 data. Exercise both the component accounting and its exact
    # one-byte budget boundary.
    compiled32=compile(thermal_model(0.1f0);backend=:matrixfree)
    npi=length(basis)
    direct_probe=ParameterScanPlan(Float32[0.1],thermal_model;
        algorithm=DirectAlgorithm(),continuation=false,memory_budget=Inf)
    direct_per,_,direct_live=
        PermutationalInvariantDynamics._scan_output_upper_bytes(
            direct_probe,npi,ComplexF32,:direct)
    @test direct_per==BigInt(npi)*sizeof(ComplexF64)
    @test direct_live==2direct_per
    krylov_probe=ParameterScanPlan(Float32[0.1],thermal_model;
        algorithm=GMRESAlgorithm(krylovdim=12),continuation=false,
        memory_budget=Inf)
    krylov_per,_,_=PermutationalInvariantDynamics._scan_output_upper_bytes(
        krylov_probe,npi,ComplexF32,:krylov)
    @test krylov_per==BigInt(npi)*sizeof(ComplexF32)

    direct_report=PermutationalInvariantDynamics._scan_resource_report(
        direct_probe,compiled32,1)
    direct_exact=ParameterScanPlan(Float32[0.1],thermal_model;
        algorithm=DirectAlgorithm(),continuation=false,
        memory_budget=Int(direct_report.known_peak_bytes))
    @test PermutationalInvariantDynamics._scan_resource_report(
        direct_exact,compiled32,1).known_budget_fits
    direct_tight=ParameterScanPlan(Float32[0.1],thermal_model;
        algorithm=DirectAlgorithm(),continuation=false,
        memory_budget=Int(direct_report.known_peak_bytes-1))
    direct_tight_report=PermutationalInvariantDynamics._scan_resource_report(
        direct_tight,compiled32,1)
    @test !direct_tight_report.known_budget_fits
    @test_throws ArgumentError PermutationalInvariantDynamics._enforce_scan_resource_report(
        direct_tight_report,1)

    dense_probe=ParameterScanPlan(Float32[0.1],thermal_model;
        task=:spectrum,algorithm=:dense,nev=2,continuation=false,
        memory_budget=Inf)
    dense_per,_,_=PermutationalInvariantDynamics._scan_output_upper_bytes(
        dense_probe,npi,ComplexF32,:dense)
    @test dense_per==2BigInt(sizeof(ComplexF64))
    arnoldi_probe=ParameterScanPlan(Float32[0.1],thermal_model;
        task=:spectrum,algorithm=:krylov,nev=2,continuation=false,
        memory_budget=Inf)
    arnoldi_per,_,_=PermutationalInvariantDynamics._scan_output_upper_bytes(
        arnoldi_probe,npi,ComplexF32,:krylov)
    @test arnoldi_per==2BigInt(sizeof(ComplexF32))
    dense_report=PermutationalInvariantDynamics._scan_resource_report(
        dense_probe,compiled32,1)
    dense_exact=ParameterScanPlan(Float32[0.1],thermal_model;
        task=:spectrum,algorithm=:dense,nev=2,continuation=false,
        memory_budget=Int(dense_report.known_peak_bytes))
    @test PermutationalInvariantDynamics._scan_resource_report(
        dense_exact,compiled32,1).known_budget_fits
    dense_tight=ParameterScanPlan(Float32[0.1],thermal_model;
        task=:spectrum,algorithm=:dense,nev=2,continuation=false,
        memory_budget=Int(dense_report.known_peak_bytes-1))
    @test !PermutationalInvariantDynamics._scan_resource_report(
        dense_tight,compiled32,1).known_budget_fits

    block_probe=ParameterScanPlan(Float32[0.1],thermal_model;
        task=:spectrum,algorithm=:block_arnoldi,nev=2,
        continuation=false,save_outputs=false,
        solver_options=(krylovdim=4,block_size=2,maxrestarts=0),
        memory_budget=Inf)
    block_growth=
        PermutationalInvariantDynamics._performance_batched_action_growth_bytes(
            compiled32,2)
    @test block_growth>0
    block_report=PermutationalInvariantDynamics._scan_resource_report(
        block_probe,compiled32,2)
    @test block_report.operator_action_per_worker_upper_bytes==block_growth
    @test block_report.operator_action_upper_bytes==2block_growth
    block_without_action=
        PermutationalInvariantDynamics._scan_resource_components(
            block_probe,npi,ComplexF32,2,:block_arnoldi,
            block_report.operator_retained_bytes)
    @test block_report.known_peak_bytes==
        block_without_action.known_peak+2block_growth

    # Budget-aware :auto must fall back to GMRES when the aggregate direct
    # scan peak does not fit, while the same budget remains an error for an
    # explicitly requested direct solve.
    auto_probe=ParameterScanPlan(Float32[0.1],thermal_model;
        algorithm=AutoAlgorithm(),continuation=false,save_outputs=false,
        solver_options=(atol=1.0f-6,rtol=1.0f-5),memory_budget=Inf)
    operator_bytes=BigInt(Base.summarysize(compiled32))
    direct_components=PermutationalInvariantDynamics._scan_resource_components(
        auto_probe,npi,ComplexF32,1,:direct,operator_bytes)
    gmres_components=PermutationalInvariantDynamics._scan_resource_components(
        auto_probe,npi,ComplexF32,1,:krylov,operator_bytes)
    @test gmres_components.known_peak<direct_components.known_peak
    # A raw-model scan must also be able to prepare the compiled operator while
    # the reusable Krylov workspace is retained.  Include that phase in the
    # tight auto budget rather than deriving a bound from an already compiled
    # input alone.
    preparation_bytes=PermutationalInvariantDynamics._model_preparation_bytes(
        thermal_model(0.1f0))
    gmres_without_operator=
        PermutationalInvariantDynamics._scan_resource_components(
            auto_probe,npi,ComplexF32,1,:krylov,big(0)).known_peak
    auto_budget=Int(max(gmres_components.known_peak,
                        preparation_bytes+gmres_without_operator))
    auto_low=ParameterScanPlan(Float32[0.1],thermal_model;
        algorithm=AutoAlgorithm(),continuation=false,save_outputs=false,
        solver_options=(atol=1.0f-6,rtol=1.0f-5),memory_budget=auto_budget)
    auto_result=parameter_scan(auto_low;on_error=:throw)
    @test only(auto_result).diagnostics.resources.method===:krylov
    explicit_low=ParameterScanPlan(Float32[0.1],thermal_model;
        algorithm=DirectAlgorithm(),continuation=false,save_outputs=false,
        memory_budget=auto_budget)
    explicit_result=parameter_scan(explicit_low)
    @test only(explicit_result).status===:failed
    @test occursin("memory_budget",only(explicit_result).message)

    auto_spectrum_probe=ParameterScanPlan(Float32[0.1],thermal_model;
        task=:spectrum,algorithm=AutoAlgorithm(),nev=2,
        continuation=false,save_outputs=false,
        solver_options=(atol=1.0f-6,rtol=1.0f-5),memory_budget=Inf)
    dense_components=PermutationalInvariantDynamics._scan_resource_components(
        auto_spectrum_probe,npi,ComplexF32,1,:dense,operator_bytes)
    arnoldi_components=PermutationalInvariantDynamics._scan_resource_components(
        auto_spectrum_probe,npi,ComplexF32,1,:krylov,operator_bytes)
    @test arnoldi_components.known_peak<dense_components.known_peak
    arnoldi_without_operator=
        PermutationalInvariantDynamics._scan_resource_components(
            auto_spectrum_probe,npi,ComplexF32,1,:krylov,big(0)).known_peak
    spectrum_budget=Int(max(arnoldi_components.known_peak,
        preparation_bytes+arnoldi_without_operator))
    auto_spectrum=ParameterScanPlan(Float32[0.1],thermal_model;
        task=:spectrum,algorithm=AutoAlgorithm(),nev=2,
        continuation=false,save_outputs=false,
        solver_options=(atol=1.0f-6,rtol=1.0f-5),
        memory_budget=spectrum_budget)
    auto_spectrum_result=parameter_scan(auto_spectrum;on_error=:throw)
    @test only(auto_spectrum_result).diagnostics.resources.method===:arnoldi
    explicit_dense=ParameterScanPlan(Float32[0.1],thermal_model;
        task=:spectrum,algorithm=:dense,nev=2,
        continuation=false,save_outputs=false,memory_budget=spectrum_budget)
    explicit_dense_result=parameter_scan(explicit_dense)
    @test only(explicit_dense_result).status===:failed
    @test occursin("memory_budget",only(explicit_dense_result).message)

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
    @test plan.memory_budget==512*1024^2
    @test !plan.budget_disabled
    @test result.metadata.memory_budget==plan.memory_budget
    @test result[1].diagnostics.resources.known_budget_fits
    @test result[1].diagnostics.resources.budget_status===:unknown
    @test ismissing(result[1].diagnostics.resources.safe_to_run)
    @test :diagnostic_payloads in
          result[1].diagnostics.resources.unknown_components
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

    # The resource preflight happens after model lowering but before Krylov
    # workspace construction. An intentionally impossible budget therefore
    # produces an explicit failed point without allocating solver storage.
    budget_workspace=ParameterScanWorkspace()
    impossible=ParameterScanPlan(Float32[0.1],thermal_model;
        algorithm=GMRESAlgorithm(krylovdim=12,maxiter=200),memory_budget=1)
    budget_result=parameter_scan(impossible;workspace=budget_workspace)
    @test only(budget_result).status===:failed
    @test occursin("memory_budget",only(budget_result).message)
    @test budget_workspace.solver_workspace===nothing

    disabled_budget=ParameterScanPlan(Float32[0.1],thermal_model;
        algorithm=GMRESAlgorithm(krylovdim=12,maxiter=200),
        solver_options=(atol=1.0f-6,rtol=1.0f-5),memory_budget=Inf)
    disabled_result=parameter_scan(disabled_budget;on_error=:throw)
    @test disabled_budget.budget_disabled
    @test only(disabled_result).diagnostics.resources.budget_status===:disabled
    @test only(disabled_result).diagnostics.resources.known_budget_fits
    @test ismissing(only(disabled_result).diagnostics.resources.safe_to_run)

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

    block_plan=ParameterScanPlan([0.2,0.3],spectral_builder;
        task=:spectrum,algorithm=:block,spectrum_target=:largest_real,
        nev=2,compile_options=(backend=:matrixfree,),
        solver_options=(krylovdim=4,block_size=2,maxrestarts=0,
                        require_convergence=false))
    block_result=parameter_scan(block_plan;on_error=:throw)
    @test all(point->point.status===:success,block_result)
    @test block_result[1].diagnostics.resources.method===:block_arnoldi
    @test block_result[2].warm_started
    @test block_result[2].workspace_reused
    @test block_result.restart_seed isa Matrix
    @test size(block_result.restart_seed,2)<=2
    for point in block_result
        dense_values=liouvillian_spectrum(
            spectral_builder(point.parameter);algorithm=:dense,nev=2)
        @test point.output.values≈dense_values atol=2e-8 rtol=2e-8
    end

    empty_chunk=parameter_scan(independent;indices=Int[])
    empty_columns=parameter_scan_columns(empty_chunk;include_output=true)
    @test isempty(empty_columns.index)
    @test hasproperty(empty_columns,:output)

    clear_parameter_scan_workspace!(workspace)
    @test workspace.continuation_seed===nothing
    @test workspace.solver_workspace===nothing
end
