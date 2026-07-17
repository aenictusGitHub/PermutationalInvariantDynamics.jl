using Tables
using Distributed

@testset "optional parameter-scan extensions" begin
    @testset "Tables interface" begin
        point=ParameterScanPoint(1,0.25,:success,nothing,1.0e-12,2.0e-14,
            true,7,0.01,0.02,0.03,true,true,
            (solver=(method=:krylov,),user=(value=0.2,)),nothing,nothing)
        failure=ParameterScanPoint(2,0.5,:failed,nothing,missing,missing,
            false,missing,0.01,0.0,0.01,false,false,nothing,
            "ArgumentError","deliberate table failure")
        metadata=(elapsed_seconds=0.03,execution=:serial,
            requested_indices=[1,2],continuation=false,save_outputs=false,
            save_vectors=false,save_restart=false,stopped=false,
            successful=1,failed=1,restart_signature=nothing,
            restart_eltype=nothing)
        result=ParameterScanResult(:steady_state,[0.25,0.5],[point,failure],
                                   0,nothing,metadata)

        @test Tables.istable(typeof(result))
        @test Tables.rowaccess(typeof(result))
        rows=Tables.rows(result)
        @test length(rows)==2
        schema=Tables.schema(rows)
        @test schema.names==propertynames(first(parameter_scan_rows(result)))
        @test :output ∉ schema.names
        collected=collect(rows)
        @test isequal(collected,parameter_scan_rows(result))
        columns=Tables.columntable(result)
        @test columns.index==[1,2]
        @test columns.parameter==[0.25,0.5]
        @test columns.status==[:success,:failed]
        @test ismissing(columns.residual[2])
        @test columns.message[2]=="deliberate table failure"
        @test !hasproperty(columns,:output)

        empty_result=ParameterScanResult(:steady_state,Float64[],
            ParameterScanPoint[],0,nothing,
            merge(metadata,(requested_indices=Int[],successful=0,)))
        @test isempty(collect(Tables.rows(empty_result)))
        @test Tables.schema(Tables.rows(empty_result)).names==schema.names

        spectrum=liouvillian_spectrum_data(
            ComplexF64[0,-0.4+0.2im];residuals=[1e-14,2e-12],
            converged=[true,false],dimension=2,complete=true)
        spectrum_columns=Tables.columns(spectrum)
        @test propertynames(spectrum_columns)==
              (:index,:value,:classification,:residual,:converged)
        @test spectrum_columns.index==1:2
        @test spectrum_columns.value===spectrum.values
        @test spectrum_columns.classification===spectrum.classifications
        @test spectrum_columns.residual===spectrum.residuals
        @test spectrum_columns.converged===spectrum.converged
        @test Tables.columntable(spectrum).value===spectrum.values

        spectrum_without_diagnostics=liouvillian_spectrum_data(
            ComplexF64[0,-1];dimension=2,complete=true)
        missing_columns=Tables.columns(spectrum_without_diagnostics)
        @test missing_columns.residual isa AbstractVector{Missing}
        @test missing_columns.converged isa AbstractVector{Missing}
        @test length(missing_columns.residual)==2
        @test all(ismissing,missing_columns.residual)
        @test all(ismissing,missing_columns.converged)

        qudit_basis=PIBasis(1,3)
        qudit_state=computational_product_state(qudit_basis,3)
        qudit_plan=QuditHusimiPlan(
            qudit_basis,zeros(ComplexF64,3,3);
            representation=:generator)
        husimi=qudit_husimi_q(qudit_state,qudit_plan)
        husimi_columns=Tables.columns(husimi)
        @test propertynames(husimi_columns)==(:point,:value)
        @test husimi_columns.point==1:length(husimi.values)
        @test husimi_columns.value===husimi.values
        @test Tables.columntable(husimi).value===husimi.values

        convergence=convergence_study(
            level->inv(float(level)),[1,2,4];consecutive=1,rtol=1)
        convergence_columns=Tables.columns(convergence)
        @test propertynames(convergence_columns)==(
            :level,:refinement,:estimate,:pairwise_error,:tolerance,
            :pairwise_converged,:observed_rate,:solver_converged)
        @test convergence_columns.level==1:3
        @test convergence_columns.refinement===convergence.refinements
        @test convergence_columns.estimate===convergence.estimates
        @test convergence_columns.pairwise_error===convergence.pairwise_errors
        @test convergence_columns.tolerance===convergence.tolerances
        @test convergence_columns.pairwise_converged===
              convergence.pairwise_converged
        @test convergence_columns.observed_rate===convergence.observed_rates
        @test convergence_columns.solver_converged===convergence.solver_converged
        @test Tables.columntable(convergence).estimate===convergence.estimates
    end

    @testset "Distributed execution" begin
        project=dirname(Base.active_project())
        added=Distributed.addprocs(2;exeflags="--project=$project")
        try
            basis=PIBasis(1,2)
            sm=ComplexF64[0 1;0 0]
            sp=Matrix(adjoint(sm))
            number=ComplexF64[0 0;0 1]
            builder=rate->PIModel(basis,(
                LocalJump(sm;rate=1.0),LocalJump(sp;rate=rate)))
            diagnostic=(rho,rate,index)->(
                excited=real(collective_expectation(rho,number)),
                rate=rate,index=index)
            plan=ParameterScanPlan([0.1,0.2,0.4,0.8],builder;
                algorithm=DirectAlgorithm(),continuation=false,
                save_outputs=false,diagnostic=diagnostic)

            local_result=parameter_scan(plan)
            distributed_result=distributed_parameter_scan(plan;workers=added)
            @test [point.index for point in distributed_result]==collect(1:4)
            @test all(point->point.status===:success,distributed_result)
            @test all(point->point.output===nothing,distributed_result)
            @test distributed_result.metadata.execution===:distributed
            @test distributed_result.metadata.workers==sort(added)
            @test distributed_result.metadata.chunks==[[1,2],[3,4]]
            distributed_excitation=
                [point.diagnostics.user.excited for point in distributed_result]
            local_excitation=
                [point.diagnostics.user.excited for point in local_result]
            @test distributed_excitation≈local_excitation

            callback_indices=Int[]
            callback_types=DataType[]
            callback_plan=ParameterScanPlan([0.1,0.2,0.4,0.8],builder;
                algorithm=DirectAlgorithm(),continuation=false,
                save_outputs=true,diagnostic=diagnostic)
            callback_result=distributed_parameter_scan(
                callback_plan;workers=added,
                callback=point->begin
                    push!(callback_indices,point.index)
                    push!(callback_types,typeof(point.output))
                    nothing
                end)
            @test callback_indices==collect(1:4)
            @test all(type->type<:PIState,callback_types)
            @test all(point->point.output isa PIState,callback_result)
            @test_throws ArgumentError distributed_parameter_scan(
                plan;workers=added,callback=identity)

            empty_distributed=distributed_parameter_scan(
                plan;workers=Int[],indices=Int[])
            @test isempty(empty_distributed)

            sliced=distributed_parameter_scan(plan;workers=added,indices=2:3)
            @test [point.index for point in sliced]==[2,3]

            failing_builder=(rate,index)->index==2 ?
                throw(ArgumentError("remote deliberate failure")) : builder(rate)
            failing_plan=ParameterScanPlan([0.1,0.2,0.4],failing_builder;
                algorithm=DirectAlgorithm(),continuation=false,
                save_outputs=false)
            recorded=distributed_parameter_scan(failing_plan;workers=added,
                on_error=:record)
            @test [point.status for point in recorded]==
                [:success,:failed,:success]
            @test occursin("remote deliberate failure",recorded[2].message)
            stopped=distributed_parameter_scan(failing_plan;workers=added,
                on_error=:stop)
            @test length(stopped)==2
            @test stopped[2].status===:failed
            @test_throws ErrorException distributed_parameter_scan(
                failing_plan;workers=added,on_error=:throw)

            continuation_plan=ParameterScanPlan([0.1,0.2],builder;
                algorithm=DirectAlgorithm(),continuation=true)
            @test_throws ArgumentError distributed_parameter_scan(
                continuation_plan;workers=added)
            @test_throws ArgumentError distributed_parameter_scan(
                plan;workers=[typemax(Int)])
            @test_throws ArgumentError distributed_parameter_scan(
                plan;workers=[Distributed.myid()])
            @test_throws ArgumentError distributed_parameter_scan(
                plan;workers=[first(added),first(added)])

            rho0=computational_product_state(basis,2)
            trajectory_model=builder(0.2)
            trajectory_times=[0.0,0.02,0.05]
            serial_jumps=quantum_trajectories(
                trajectory_model,rho0,trajectory_times,5;seed=314,dt=0.005)
            distributed_jumps=distributed_quantum_trajectories(
                trajectory_model,rho0,trajectory_times,5;
                workers=added,seed=314,dt=0.005)
            @test map(path->path.jump_times,distributed_jumps)==
                map(path->path.jump_times,serial_jumps)
            @test map(path->path.jump_channels,distributed_jumps)==
                map(path->path.jump_channels,serial_jumps)
            @test all(distributed_jumps[path].states[time].data==
                      serial_jumps[path].states[time].data
                      for path in eachindex(serial_jumps)
                      for time in eachindex(trajectory_times))

            monitored_model=PIModel(basis,(
                CollectiveJump(sm;rate=0.3),))
            monitor=homodyne_monitor(sqrt(0.3)*sm;
                efficiency=0.7,phase=0.2,label=:emission)
            serial_diffusive=diffusive_trajectories(
                monitored_model,rho0,trajectory_times,monitor,5;
                seed=2718,dt=0.002,observables=(population=number,))
            distributed_diffusive=distributed_diffusive_trajectories(
                monitored_model,rho0,trajectory_times,monitor,5;
                workers=added,seed=2718,dt=0.002,
                observables=(population=number,))
            @test map(path->path.records,distributed_diffusive)==
                map(path->path.records,serial_diffusive)
            @test map(path->path.innovations,distributed_diffusive)==
                map(path->path.innovations,serial_diffusive)
            @test map(path->path.observables.values,distributed_diffusive)==
                map(path->path.observables.values,serial_diffusive)
            @test all(distributed_diffusive[path][time].data==
                      serial_diffusive[path][time].data
                      for path in eachindex(serial_diffusive)
                      for time in eachindex(trajectory_times))
        finally
            Distributed.rmprocs(added)
        end
    end
end
