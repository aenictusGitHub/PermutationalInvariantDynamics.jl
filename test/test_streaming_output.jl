_streamed_observables(result)=result.observables
_stream_record_allocations(buffers,ops,rho)=@allocated(
    PermutationalInvariantDynamics._record_dynamics_observables!(
        buffers,ops,rho,2))

@testset "memory-light streaming output" begin
    PIDStreaming=PermutationalInvariantDynamics
    b=PIBasis(2,2)
    sm=ComplexF64[0 1;0 0]
    number=ComplexF64[0 0;0 1]
    rho0=iid_pure_state(b,ComplexF64[0,1])
    model=PIModel(b,[LocalJump(sm;rate=0.7)])
    prepared=compile(model;backend=:matrixfree)
    observable=collective_operator(b,number)
    times=[0.0,0.05,0.1,0.2]
    prepared_observables=PIDStreaming._prepare_streaming_observables(
        b,(excitation=observable,);require_hermitian=true)
    @test prepared_observables isa Tuple
    sampling_buffers=PIDStreaming._dynamics_observable_buffers(
        prepared_observables,rho0,2)
    PIDStreaming._record_dynamics_observables!(
        sampling_buffers,prepared_observables,rho0,1)
    _stream_record_allocations(sampling_buffers,prepared_observables,rho0)
    @test _stream_record_allocations(
        sampling_buffers,prepared_observables,rho0)==0

    @testset "deterministic observable-only propagation" begin
        reference=@inferred solve_dynamics(
            prepared,rho0,(first(times),last(times));
            saveat=times,steps_per_interval=24)
        streamed=@inferred solve_dynamics(
            prepared,rho0,(first(times),last(times));
            saveat=times,steps_per_interval=24,
            observables=(excitation=observable,),save_states=false)
        @test streamed isa PIDStreaming.DynamicsStreamResult
        @test (@inferred _streamed_observables(streamed))===streamed.observables
        @test streamed.states===nothing
        @test length(streamed)==length(times)
        @test streamed.times==times
        @test streamed.observables[:excitation]≈
              [expectation(state,observable) for state in reference.states]
        @test_throws ArgumentError state(streamed,1)
        @test_throws ArgumentError streamed[1]
        @test_throws ArgumentError collect(streamed)
        @test occursin("observable-only",sprint(show,streamed))

        with_states=solve_dynamics(prepared,rho0,(first(times),last(times));
            saveat=times,steps_per_interval=24,
            observables=(excitation=observable,),save_states=true)
        @test length(with_states.states)==length(times)
        @test all(with_states.states[index].data==reference.states[index].data
                  for index in eachindex(times))
        @test collect(with_states)==with_states.states
        @test with_states[2]===with_states.states[2]
        @test state(with_states,0.1)===with_states.states[3]

        @test_throws ArgumentError solve_dynamics(prepared,rho0,(0.0,0.1);
            save_states=false)
        @test_throws ArgumentError solve_dynamics(prepared,rho0,(0.0,0.1);
            observables=Pair[])
        @test_throws ArgumentError solve_dynamics(prepared,rho0,(0.0,0.1);
            observables=[:x=>observable,:x=>observable])
    end

    @testset "online trajectory observables and jumps" begin
        npaths=32
        trajectories=@inferred quantum_trajectories(
            model,rho0,times,npaths;
            seed=381,dt=0.005)
        streamed=@inferred quantum_trajectories(
            model,rho0,times,npaths;
            seed=381,dt=0.005,observables=(excitation=observable,),
            save_states=false)
        @test streamed isa PIDStreaming.TrajectoryEnsembleResult
        @test (@inferred _streamed_observables(streamed))===streamed.observables
        @test streamed.trajectories===nothing
        @test length(streamed)==npaths
        @test streamed.trajectory_count==npaths
        @test streamed.times==times
        @test occursin("state-free",sprint(show,streamed))
        @test_throws ArgumentError streamed[1]
        @test_throws ArgumentError collect(streamed)

        reference_observables=trajectory_observable_statistics(
            trajectories,(excitation=observable,))
        streamed_observable=streamed.observables.observables[:excitation]
        reference_observable=reference_observables.observables[:excitation]
        @test streamed_observable.mean==reference_observable.mean
        @test streamed_observable.variance==reference_observable.variance
        @test streamed_observable.standard_error==
              reference_observable.standard_error

        reference_jumps=jump_statistics(trajectories;nchannels=1)
        @test streamed.jumps.total_jumps==reference_jumps.total_jumps
        @test streamed.jumps.mean_count==reference_jumps.mean_count
        @test streamed.jumps.count_variance==reference_jumps.count_variance
        @test streamed.jumps.no_jump_probability==
              reference_jumps.no_jump_probability
        @test streamed.jumps.channels==reference_jumps.channels
        @test streamed.jumps.waiting_times==reference_jumps.waiting_times
        @test Base.summarysize(streamed)<Base.summarysize(trajectories)

        # Requesting observables need not force histories to be discarded.
        with_states=quantum_trajectories(model,rho0,times,5;
            seed=92,dt=0.005,observables=(excitation=observable,))
        plain=quantum_trajectories(model,rho0,times,5;
            seed=92,dt=0.005)
        @test with_states.trajectories isa Vector
        @test collect(with_states)==with_states.trajectories
        @test with_states[2]===with_states.trajectories[2]
        @test all(with_states.trajectories[path].jump_times==
                  plain[path].jump_times for path in eachindex(plain))
        @test all(with_states.trajectories[path].states[index].data==
                  plain[path].states[index].data
                  for path in eachindex(plain),index in eachindex(times))

        observable_only=quantum_trajectories(model,rho0,times,4;
            seed=7,dt=0.005,save_states=false,jump_statistics=false,
            observables=(excitation=observable,))
        @test observable_only.jumps===nothing
        @test_throws ArgumentError quantum_trajectories(model,rho0,times,4;
            seed=7,dt=0.005,save_states=false)
        @test_throws ArgumentError quantum_trajectories(model,rho0,times,4;
            seed=7,dt=0.005,save_states=false,observables=(bad=sm,))

        if Threads.nthreads()>1
            plan=TrajectoryPlan(model)
            batch=TrajectoryBatchWorkspace(plan,rho0;
                workers=min(Threads.nthreads(),4))
            serial=quantum_trajectories(plan,rho0,times,24;
                seed=930,dt=0.005,workspace=batch,
                observables=(excitation=observable,),save_states=false)
            threaded=quantum_trajectories(plan,rho0,times,24;
                seed=930,dt=0.005,workspace=batch,threaded=true,
                observables=(excitation=observable,),save_states=false)
            @test threaded.jumps.total_jumps==serial.jumps.total_jumps
            @test threaded.observables.observables[:excitation].mean≈
                  serial.observables.observables[:excitation].mean atol=5e-15
            @test threaded.observables.observables[:excitation].variance≈
                  serial.observables.observables[:excitation].variance atol=5e-15
        end
    end

    @testset "precision and adaptive state-free paths" begin
        b32=PIBasis(1,2);sm32=ComplexF32[0 1;0 0]
        number32=ComplexF32[0 0;0 1]
        rho32=iid_pure_state(b32,ComplexF32[0,1])
        model32=PIModel(b32,[LocalJump(sm32;rate=1f0)])
        # The CG geometry convention is Float64; this explicit conversion
        # selects a Float32 observable and therefore a Float32 accumulator.
        obs32=PIOperator(b32,ComplexF32.(
            collective_operator(b32,number32).data))
        times32=Float32[0,0.1,0.2]
        result32=quantum_trajectories(model32,rho32,times32,8;
            seed=12,dt=0.02f0,algorithm=:event,dtmax=0.02f0,
            abstol=1f-6,reltol=1f-5,event_time_tolerance=1f-6,
            observables=(excitation=obs32,),save_states=false)
        @test eltype(result32.times)===Float32
        @test eltype(result32.observables.observables[:excitation].mean)===
              Float32
        @test result32.trajectories===nothing

        deterministic32=solve_dynamics(model32,rho32,(0f0,0.2f0);
            saveat=times32,steps_per_interval=8,
            observables=(lowering=PIOperator(b32,ComplexF32.(
                collective_operator(b32,sm32).data)),),
            save_states=false)
        @test eltype(deterministic32.times)===Float32
        @test eltype(deterministic32.observables[:lowering])===ComplexF32
    end
end
