@testset "uniform result summaries and exports" begin
    @testset "dependency-free result table" begin
        table=ResultTable((time=[0.0,0.5],value=ComplexF64[1,2im]);
            metadata=(source=:test,))
        @test length(table)==2
        @test first(table)==(time=0.0,value=1.0+0.0im)
        @test last(table)==(time=0.5,value=0.0+2.0im)
        @test collect(table)==[
            (time=0.0,value=1.0+0.0im),
            (time=0.5,value=0.0+2.0im)]
        @test table.metadata.source===:test
        @test result_table(table)===table
        @test_throws DimensionMismatch ResultTable((x=[1],y=[1,2]))
        @test_throws ArgumentError ResultTable((x=1,))
        @test_throws ArgumentError result_table((value=1,))
        @test Workflow.result_table===result_table
        @test Workflow.summarize===summarize
        @test Workflow.save_result===save_result
        @test Workflow.save_checkpoint===save_checkpoint
        @test Workflow.load_checkpoint===load_checkpoint
    end

    @testset "high-level deterministic results" begin
        basis=PIBasis(1,2)
        lowering=ComplexF64[0 1;0 0]
        number=ComplexF64[0 0;0 1]
        model=PIModel(basis,(LocalJump(lowering;rate=1.0),))
        rho0=computational_product_state(basis,2)

        state_summary=summarize(rho0)
        @test state_summary.result_type=="PIState"
        @test state_summary.N==1
        @test state_summary.d==2
        @test state_summary.trace≈1
        @test result_table(rho0;include_output=true)[1].state===rho0

        stationary=stationary_state(
            model;algorithm=DirectAlgorithm(),return_info=true)
        stationary_summary=summarize(stationary)
        @test stationary_summary.converged
        @test stationary_summary.algorithm===:DirectAlgorithm
        @test stationary_summary.residual≤1e-12
        @test result_table(stationary)[1].method===:direct
        @test !hasproperty(first(result_table(stationary)),:state)
        @test result_table(stationary;include_output=true)[1].state===
            stationary.state

        guided=solve(PIStudy(model;task=:steady_state,
            algorithm=DirectAlgorithm(),observables=(excited=number,)))
        guided_summary=summarize(guided)
        @test guided_summary.task===:steady_state
        @test guided_summary.converged===true
        @test guided_summary.observable_count==1
        @test result_table(guided).columns.excited==
            [guided.observables[:excited]]
        @test result_table(guided;include_output=true)[1].state===
            guided.state

        dynamics=solve_dynamics(model,rho0,(0.0,0.1);
            saveat=[0.0,0.05,0.1],algorithm=:rk4,
            steps_per_interval=2)
        dynamics_table=result_table(dynamics)
        @test dynamics_table.columns.time===dynamics.times
        @test propertynames(dynamics_table.columns)==(:time,)
        @test propertynames(
            result_table(dynamics;include_output=true).columns)==
            (:time,:state)

        streaming=solve_dynamics(model,rho0,(0.0,0.1);
            saveat=[0.0,0.05,0.1],algorithm=:rk4,
            steps_per_interval=2,observables=(excited=number,),
            save_states=false)
        streaming_table=result_table(streaming)
        @test propertynames(streaming_table.columns)==(:time,:excited)
        @test length(streaming_table.columns.excited)==3
        @test summarize(streaming).saved_states==0
        @test summarize(streaming).observable_count==1
        @test_throws ArgumentError result_table(DynamicsStreamResult(
            [0.0],nothing,(time=[1.0],),:rk4))
        @test_throws ArgumentError result_table(DynamicsStreamResult(
            [0.0],[rho0],(state=[1.0],),:rk4);
            include_output=true)
    end

    @testset "spectra, scans, convergence, and inference" begin
        spectrum=liouvillian_spectrum_data(
            ComplexF64[0,-0.5+0.2im];residuals=[0.0,1e-10],
            converged=[true,false],dimension=2,complete=true)
        table=result_table(spectrum)
        @test table.columns.value===spectrum.values
        @test table.columns.classification===spectrum.classifications
        @test summarize(spectrum).mode_count==2
        @test summarize(spectrum).converged===false

        compact=SpectrumResult(
            ComplexF64[0,-1],nothing,
            (residuals=[0.0,1e-12],converged=[true,true]))
        compact_table=result_table(compact)
        @test compact_table.columns.residual==[0.0,1e-12]
        @test all(compact_table.columns.converged)
        @test summarize(compact).vectors_saved===false

        point=ParameterScanPoint(1,(rate=0.2,),:success,nothing,
            1e-12,0.0,true,4,0.01,0.02,0.03,false,true,
            (user=(excited=0.1,),),nothing,nothing)
        metadata=(stopped=false,)
        scan=ParameterScanResult(:steady_state,[(rate=0.2,)],
            [point],0,nothing,metadata)
        scan_table=result_table(scan)
        @test scan_table.columns.parameter==[(rate=0.2,)]
        @test summarize(scan).successful==1
        @test summarize(scan).failed==0
        @test summarize(scan).cancelled===missing

        convergence=convergence_study(
            level->inv(float(level)),[1,2,4];consecutive=1,rtol=1)
        convergence_table=result_table(convergence)
        @test convergence_table.columns.estimate===
            convergence.estimates
        @test summarize(convergence).levels==3

        inference=ParameterInferenceResult(
            [0.2,0.4],[1.0,2.0],[0.1,-0.1],[1.0,-1.0],
            0.02,3,true,:gradient_tolerance,:finite_difference,
            (identifiable=true,),NamedTuple[],NamedTuple())
        inference_table=result_table(inference)
        @test inference_table.columns.observation==1:2
        @test inference_table.columns.standardized_residual==[1.0,-1.0]
        @test summarize(inference).parameter_count==2
    end

    @testset "portable text and pidrun output" begin
        table=ResultTable((
            label=["comma,value","ordinary"],
            value=[1.0,2.0]))
        mktempdir() do directory
            csv=joinpath(directory,"result.csv")
            @test save_result(csv,table)==csv
            csv_text=read(csv,String)
            @test occursin("\"comma,value\"",csv_text)
            @test startswith(csv_text,"label,value\n")

            tsv=joinpath(directory,"result.tsv")
            @test save_result(tsv,table)==tsv
            @test startswith(read(tsv,String),"label\tvalue\n")

            basis=PIBasis(1,2)
            state=computational_product_state(basis,1)
            archive=joinpath(directory,"state.pidrun")
            @test save_result(archive,state;
                metadata=Dict("purpose"=>"round-trip smoke"))==archive
            @test isfile(joinpath(archive,"metadata.tsv"))
            @test isfile(joinpath(archive,"table.tsv"))
            state_path=joinpath(
                archive,"states","state_000001.pid")
            @test isfile(state_path)
            checkpoint=load_checkpoint(state_path)
            @test checkpoint.state.data==state.data
            @test checkpoint.metadata["purpose"]=="round-trip smoke"
            @test checkpoint.metadata["result_type"]==string(typeof(state))
            @test_throws ArgumentError save_result(archive,state)
            reserved=joinpath(directory,"reserved.pidrun")
            @test_throws ArgumentError save_result(
                reserved,state;metadata=Dict("schema_version"=>"invalid"))
            @test !ispath(reserved)

            @test_throws ArgumentError save_result(
                joinpath(directory,"unknown.extension"),table)
            @test_throws ArgumentError save_result(
                joinpath(directory,"missing.jld2"),table)
            @test_throws ArgumentError save_result(
                joinpath(directory,"missing.h5"),table)
        end
    end
end
