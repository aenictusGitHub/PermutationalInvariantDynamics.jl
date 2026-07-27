@testset "progress events and cooperative cancellation" begin
    @test Workflow.resume_parameter_scan===resume_parameter_scan
    token=CancellationToken()
    @test !iscancelled(token)
    @test cancel!(token)===token
    @test iscancelled(token)
    @test reset_cancellation!(token)===token
    @test !iscancelled(token)

    basis=PIBasis(1,2)
    lowering=ComplexF64[0 1;0 0]
    model=PIModel(basis,(LocalJump(lowering;rate=1.0),))
    initial=iid_pure_state(basis,ComplexF64[0,1])

    evolution_events=ProgressEvent[]
    evolved=copy(initial)
    evolution_error=try
        evolve!(evolved,model,initial,(0.0,0.5);steps=5,
            on_event=event->begin
                push!(evolution_events,event)
                event.stage===:advanced&&event.completed==2 ?
                    :cancel : nothing
            end)
        nothing
    catch error
        error
    end
    @test evolution_error isa OperationCancelled
    @test evolution_error.operation===:evolve
    @test evolution_error.completed==2
    @test first(evolution_events).stage===:started
    @test last(evolution_events).stage===:cancelled

    reference=copy(initial)
    evolve!(reference,model,initial,(0.0,0.2);steps=2)
    @test evolved.data≈reference.data atol=2e-14

    text=IOBuffer()
    completed=evolve!(copy(initial),model,initial,(0.0,0.1);
        steps=2,progress=text)
    @test completed isa PIState
    progress_text=String(take!(text))
    @test occursin("[evolve] started 0/2",progress_text)
    @test occursin("[evolve] completed 2/2",progress_text)

    scan_builder=rate->PIModel(
        basis,(LocalJump(lowering;rate=rate),))
    scan_plan=ParameterScanPlan([0.5,1.0,1.5],scan_builder;
        algorithm=DirectAlgorithm(),continuation=false,memory_budget=Inf)
    scan_events=ProgressEvent[]
    scan_token=CancellationToken()
    partial=parameter_scan(scan_plan;cancellation_token=scan_token,
        on_event=event->begin
            push!(scan_events,event)
            event.stage===:advanced ? :cancel : nothing
        end)
    @test length(partial)==1
    @test partial.metadata.stopped
    @test partial.metadata.cancelled
    @test summarize(partial).cancelled
    @test iscancelled(scan_token)
    @test first(scan_events).stage===:started
    @test last(scan_events).stage===:cancelled

    threaded_partial=parameter_scan(scan_plan;execution=:threads,
        on_event=event->event.stage===:advanced ? :cancel : nothing)
    @test !isempty(threaded_partial)
    @test threaded_partial.metadata.stopped
    @test threaded_partial.metadata.cancelled

    entered=Channel{Nothing}(1)
    release=Channel{Nothing}(1)
    external_token=CancellationToken()
    gated_builder=rate->begin
        put!(entered,nothing)
        take!(release)
        scan_builder(rate)
    end
    gated_plan=ParameterScanPlan([0.5],gated_builder;
        algorithm=DirectAlgorithm(),continuation=false,memory_budget=Inf)
    gated_task=@async parameter_scan(gated_plan;execution=:threads,
        cancellation_token=external_token)
    take!(entered)
    cancel!(external_token)
    put!(release,nothing)
    externally_cancelled=fetch(gated_task)
    @test isempty(externally_cancelled)
    @test externally_cancelled.metadata.cancelled

    reset_cancellation!(scan_token)
    resumed=resume_parameter_scan(scan_plan,partial;
        cancellation_token=scan_token)
    @test length(resumed)==3
    @test all(point->point.status===:success,resumed)
    @test !resumed.metadata.stopped
    @test !resumed.metadata.cancelled

    single_plan=ParameterScanPlan([0.5],scan_builder;
        algorithm=DirectAlgorithm(),continuation=false,memory_budget=Inf)
    completed_then_cancelled=parameter_scan(single_plan;
        on_event=event->event.stage===:advanced ? :cancel : nothing)
    @test completed_then_cancelled.metadata.stopped
    @test completed_then_cancelled.metadata.cancelled
    completed_resume=resume_parameter_scan(
        single_plan,completed_then_cancelled)
    @test length(completed_resume)==1
    @test completed_resume.metadata.resumed
    @test !completed_resume.metadata.stopped
    @test !completed_resume.metadata.cancelled

    dynamics_events=ProgressEvent[]
    dynamics_error=try
        solve_dynamics(model,initial,(0.0,0.2);
            saveat=[0.0,0.1,0.2],steps_per_interval=2,
            on_event=event->begin
                push!(dynamics_events,event)
                event.stage===:advanced ? :cancel : nothing
            end,memory_budget=Inf)
        nothing
    catch error
        error
    end
    @test dynamics_error isa OperationCancelled
    @test dynamics_error.operation===:solve_dynamics
    @test dynamics_error.completed==1
    @test last(dynamics_events).stage===:cancelled

    spin=spin_matrices()
    coupling=collective_operator(basis,spin.jz)
    bath=HEOMBath(coupling,0.05,1.0)
    heom_plan=HEOMPlan(model,bath;max_depth=1)
    hierarchy=heom_initial_state(heom_plan,initial)
    heom_events=ProgressEvent[]
    heom_error=try
        heom_evolve!(copy(hierarchy),heom_plan,hierarchy,(0.0,0.1);
            steps=3,on_event=event->begin
                push!(heom_events,event)
                event.stage===:advanced ? :cancel : nothing
            end)
        nothing
    catch error
        error
    end
    @test heom_error isa OperationCancelled
    @test heom_error.operation===:heom_evolve
    @test heom_error.completed==1
    @test last(heom_events).stage===:cancelled

    hamiltonian=PIOperator(basis;T=Float64)
    hops_plan=HOPSPlan(hamiltonian,HOPSBath(coupling,0.05,1.0);
        max_depth=1)
    root=weak_pi_pseudoket(initial)
    hops_events=ProgressEvent[]
    hops_error=try
        hops_trajectory(hops_plan,root,[0.0,0.05,0.1];
            dt=0.01,noise=(time,bath_index)->0.0+0.0im,
            on_event=event->begin
                push!(hops_events,event)
                event.stage===:advanced ? :cancel : nothing
            end)
        nothing
    catch error
        error
    end
    @test hops_error isa OperationCancelled
    @test hops_error.operation===:hops_trajectory
    @test hops_error.completed==1
    @test last(hops_events).stage===:cancelled

    @test_throws ArgumentError evolve!(
        copy(initial),model,initial,(0.0,0.1);steps=1,progress=:yes)
    @test_throws ArgumentError parameter_scan(
        scan_plan;on_event=event->:invalid)
end
