@testset "PI quantum-jump trajectories" begin
    sm=ComplexF64[0 1;0 0];sx=ComplexF64[0 1;1 0];b=PIBasis(2,2)
    excited=iid_pure_state(b,ComplexF64[0,1]);times=collect(range(0,0.5;length=6))

    # With no jump channels the stochastic path reduces to deterministic RK4.
    hm=PIModel(b,[LocalHamiltonian(sx;rate=0.2)])
    qh=quantum_trajectory(hm,excited,times;dt=0.002,rng=MersenneTwister(3))
    exact=time_evolve(liouvillian(hm;representation=:matrixfree),excited,(0.0,0.5);steps=250)
    @test isempty(qh.jump_times)
    @test qh.states[end].data≈exact.data atol=2e-11

    model=PIModel(b,[LocalJump(sm)])
    qs=quantum_trajectories(model,excited,times,500;seed=42,dt=0.01,max_jump_probability=0.03)
    avg=trajectory_average(qs)
    exact_decay=time_evolve(liouvillian(model;representation=:matrixfree),excited,(0.0,0.5);steps=500)
    @test norm(avg[end].data-exact_decay.data)<0.065
    @test abs(trace(avg[end])-1)<2e-12
    @test all(q->all(s->abs(trace(s)-1)<2e-10,q.states),qs)
    @test sum(length(q.jump_times) for q in qs)>0

    stats=trajectory_statistics(qs;observables=(excitation=ComplexF64[0 0;0 1],),nchannels=1)
    @test stats.jumps.total_jumps==sum(length(q.jump_times) for q in qs)
    @test stats.jumps.channels[1].total==stats.jumps.total_jumps
    @test 0<=stats.jumps.no_jump_probability<=1
    @test stats.observables.observables[:excitation].mean[end]≈collective_expectation(avg[end],ComplexF64[0 0;0 1]) atol=2e-12
    @test all(stats.observables.observables[:excitation].standard_error.>=0)
    @test all(stats.observables.observables[:excitation].lower.<=stats.observables.observables[:excitation].upper)

    # Exact count and waiting-time statistics on a small hand-built ensemble.
    synthetic=[QuantumTrajectory([0.0,1.0],PIState[excited,excited],[0.2,0.7],[1,2]),
               QuantumTrajectory([0.0,1.0],PIState[excited,excited],Float64[],Int[])]
    js=jump_statistics(synthetic;nchannels=2)
    @test js.total_jumps==2
    @test js.mean_count==1
    @test js.count_variance==2
    @test js.no_jump_probability==0.5
    @test js.mean_waiting_time≈0.5

    q1=quantum_trajectories(model,excited,times,3;seed=17,dt=0.02)
    q2=quantum_trajectories(model,excited,times,3;seed=17,dt=0.02)
    @test [q.jump_times for q in q1]==[q.jump_times for q in q2]

    @testset "prepared and threaded batches" begin
        plan=@inferred TrajectoryPlan(model)
        batch=TrajectoryBatchWorkspace(plan,excited;workers=max(2,Threads.nthreads()))
        @test all(worker->worker.plan===plan,batch.workers)
        @test batch.workers[1].tmp!==batch.workers[2].tmp
        @test batch.workers[1].current!==batch.workers[2].current
        @test batch.workers[1].liouvillian_work!==batch.workers[2].liouvillian_work
        @test batch.workers[1].plan.liouvillian===batch.workers[2].plan.liouvillian

        prepared1=quantum_trajectories(plan,excited,times,7;seed=913,dt=0.02,
            workspace=batch)
        snapshot=[(copy(q.times),[copy(s.data) for s in q.states],
                   copy(q.jump_times),copy(q.jump_channels)) for q in prepared1]
        prepared2=quantum_trajectories(plan,excited,times,7;seed=913,dt=0.02,
            workspace=batch)
        threaded=quantum_trajectories(plan,excited,times,7;seed=913,dt=0.02,
            threaded=true,workspace=batch)
        for i in eachindex(prepared1)
            @test prepared1[i].times==prepared2[i].times==threaded[i].times
            @test prepared1[i].jump_times==prepared2[i].jump_times==threaded[i].jump_times
            @test prepared1[i].jump_channels==prepared2[i].jump_channels==threaded[i].jump_channels
            @test all(prepared1[i].states[j].data==prepared2[i].states[j].data==
                      threaded[i].states[j].data for j in eachindex(times))
            @test snapshot[i][1]==prepared1[i].times
            @test snapshot[i][2]==[s.data for s in prepared1[i].states]
            @test snapshot[i][3]==prepared1[i].jump_times
            @test snapshot[i][4]==prepared1[i].jump_channels
        end
        @test prepared1[1].times!==prepared1[2].times
        @test prepared1[1].states[1].data!==prepared1[2].states[1].data
        @test prepared1[1].jump_times!==prepared1[2].jump_times
        @test all(prepared1[1].states[i].data!==prepared1[1].states[j].data
                  for i in eachindex(times),j in eachindex(times) if i!=j)
        @test all(state.data!==worker.current for q in prepared1
                  for state in q.states for worker in batch.workers)

        # The batch consumes a one-shot iterator once, rather than once per
        # realization after an initial type-inference pass.
        one_shot=(t for t in (0.0,0.05,0.1))
        generated=quantum_trajectories(plan,excited,one_shot,3;seed=4,dt=0.01,
            workspace=batch)
        @test all(q->q.times==[0.0,0.05,0.1],generated)

        compiled=compile(model;backend=:matrixfree)
        compiled_plan=TrajectoryPlan(compiled)
        @test compiled_plan.liouvillian===compiled.plan
        single_work=TrajectoryWorkspace(plan,excited)
        @test length(quantum_trajectories(plan,excited,[0.0,0.02],2;
            seed=1,dt=0.01,workspace=single_work))==2
        @test_throws ArgumentError quantum_trajectories(plan,excited,
            [0.0,0.02],2;seed=1,dt=0.01,threaded=true,workspace=single_work)
        other_model=PIModel(b,[LocalJump(sm;rate=0.5)])
        @test_throws ArgumentError quantum_trajectories(other_model,excited,
            times,2;seed=1,dt=0.02,workspace=batch)

        # Exercise the chunked atomic scheduler (chunk_size > 1 on the
        # four-thread CI job) and retain index-stable random streams.
        chunked_serial=quantum_trajectories(plan,excited,[0.0,0.02],65;
            seed=191,dt=0.01,workspace=batch)
        chunked_threaded=quantum_trajectories(plan,excited,[0.0,0.02],65;
            seed=191,dt=0.01,threaded=true,workspace=batch)
        @test all(chunked_serial[i].jump_times==chunked_threaded[i].jump_times&&
                  chunked_serial[i].jump_channels==chunked_threaded[i].jump_channels&&
                  all(chunked_serial[i].states[j].data==
                      chunked_threaded[i].states[j].data
                      for j in eachindex(chunked_serial[i].states))
                  for i in eachindex(chunked_serial))
    end

    @testset "adaptive event timing" begin
        b1=PIBasis(1,2);excited1=iid_pure_state(b1,ComplexF64[0,1])
        decay=PIModel(b1,[LocalJump(sm)])
        seed=91;rng_reference=MersenneTwister(seed)
        expected=randexp(rng_reference)
        event=quantum_trajectory(decay,excited1,[0.0,5.0];dt=0.7,
            dtmax=0.7,algorithm=:event,rng=MersenneTwister(seed),
            abstol=1e-11,reltol=1e-10,event_time_tolerance=1e-11)
        @test length(event.jump_times)==1
        @test event.jump_times[1]≈expected atol=2e-9
        @test abs(trace(event.states[end])-1)<2e-10

        # For gamma(t)=1+t the survival hazard is t+t^2/2, which gives an
        # analytic event time for the same exponential threshold.
        driven_decay=PIModel(b1,[LocalJump(sm;rate=(t,p)->1+t)])
        expected_driven=-1+sqrt(1+2expected)
        driven_event=quantum_trajectory(driven_decay,excited1,[0.0,5.0];dt=0.6,
            dtmax=0.6,algorithm=:adaptive,rng=MersenneTwister(seed),
            abstol=1e-11,reltol=1e-10,event_time_tolerance=1e-11)
        @test driven_event.jump_times[1]≈expected_driven atol=3e-8

        nojump=quantum_trajectory(hm,excited,[0.0,0.5];dt=0.2,dtmax=0.2,
            algorithm=:event,rng=MersenneTwister(4),abstol=1e-11,reltol=1e-10)
        @test isempty(nojump.jump_times)
        @test nojump.states[end].data≈exact.data atol=2e-9

        # Saved output times may require an endpoint step below dtmin; this is
        # distinct from an error-driven step-size underflow.
        short_endpoint=quantum_trajectory(hm,excited,[0.0,0.05];dt=0.2,
            dtmin=0.1,dtmax=0.2,algorithm=:event,rng=MersenneTwister(5),
            abstol=1e-9,reltol=1e-8)
        @test isempty(short_endpoint.jump_times)
        @test abs(trace(short_endpoint.states[end])-1)<2e-10

        # An accepted step with error close to one proposes a smaller next
        # step; the controller must clamp that proposal to dtmin rather than
        # failing before the next valid retry.
        near_floor_model=PIModel(b,[LocalHamiltonian(sx;rate=2.0)])
        near_floor=quantum_trajectory(near_floor_model,excited,[0.0,0.2];
            dt=0.1,dtmin=0.1,dtmax=0.1,algorithm=:event,
            rng=MersenneTwister(6),abstol=4e-7,reltol=4e-5)
        @test isempty(near_floor.jump_times)
        @test abs(trace(near_floor.states[end])-1)<2e-10

        # Pump and decay alternate exactly for one qubit, providing a
        # deterministic multiple-event/channel regression.
        pump_decay=PIModel(b1,[LocalJump(sm;rate=3.0),
                                LocalJump(adjoint(sm);rate=3.0)])
        multiple=quantum_trajectory(pump_decay,excited1,[0.0,5.0];dt=0.4,
            dtmax=0.4,algorithm=:event,rng=MersenneTwister(717),
            abstol=1e-10,reltol=1e-9)
        @test length(multiple.jump_times)>2
        @test issorted(multiple.jump_times)
        @test all(diff(multiple.jump_times).>0)
        @test multiple.jump_channels==[isodd(index) ? 1 : 2
                                        for index in eachindex(multiple.jump_channels)]

        # Appendix-D gain maps use the same continuous hazard integration. A
        # two-particle pair decay has unit intensity until its sole jump.
        pair_basis=PIBasis(2,2)
        pair_excited=iid_pure_state(pair_basis,ComplexF64[0,1])
        pair_model=PIModel(pair_basis,[LocalPBodyJump(kron(sm,sm),2)])
        pair_seed=319;pair_rng=MersenneTwister(pair_seed)
        pair_expected=randexp(pair_rng)
        pair_event=quantum_trajectory(pair_model,pair_excited,[0.0,5.0];
            dt=0.5,dtmax=0.5,algorithm=:event,rng=MersenneTwister(pair_seed),
            abstol=1e-11,reltol=1e-10,event_time_tolerance=1e-11)
        @test pair_event.jump_channels==[1]
        @test pair_event.jump_times[1]≈pair_expected atol=2e-9

        # An event-driven ensemble recovers the analytical symmetric
        # pump/decay population p_e(t)=(1+exp(-2t))/2 within Monte Carlo error.
        ensemble_model=PIModel(b1,[LocalJump(sm),LocalJump(adjoint(sm))])
        ensemble=quantum_trajectories(ensemble_model,excited1,[0.0,1.5],200;
            seed=22,dt=0.3,dtmax=0.3,algorithm=:event,
            abstol=1e-9,reltol=1e-8)
        ensemble_state=trajectory_average(ensemble)[end]
        excitation=ComplexF64[0 0;0 1]
        expected_population=(1+exp(-3))/2
        @test real(collective_expectation(ensemble_state,excitation))≈
              expected_population atol=0.1

        # Index-derived RNG streams make adaptive paths independent of task
        # scheduling as well as fixed-step paths.
        event_plan=TrajectoryPlan(ensemble_model)
        event_batch=TrajectoryBatchWorkspace(event_plan,excited1;
            workers=max(2,Threads.nthreads()))
        event_serial=quantum_trajectories(event_plan,excited1,[0.0,0.4],5;
            seed=812,dt=0.2,dtmax=0.2,algorithm=:event,
            abstol=1e-9,reltol=1e-8,workspace=event_batch)
        event_threaded=quantum_trajectories(event_plan,excited1,[0.0,0.4],5;
            seed=812,dt=0.2,dtmax=0.2,algorithm=:event,
            abstol=1e-9,reltol=1e-8,threaded=true,workspace=event_batch)
        @test all(event_serial[i].jump_times==event_threaded[i].jump_times&&
                  event_serial[i].jump_channels==event_threaded[i].jump_channels&&
                  all(event_serial[i].states[j].data==event_threaded[i].states[j].data
                      for j in eachindex(event_serial[i].states))
                  for i in eachindex(event_serial))
        @test_throws ArgumentError quantum_trajectory(decay,excited1,[0.0,1.0];
            dt=0.1,algorithm=:unknown)
    end

    # Every supported one-body/direct and Appendix-D jump representation uses
    # a channel-resolved gain kernel and remains trace normalized.
    pair=kron(sm,sm);J=collective_operator(b,sm)
    jump_models=(PIModel(b,[CollectiveJump(sm)]),PIModel(b,[DirectPIJump(J)]),
                 PIModel(b,[LocalPBodyJump(pair,2)]),PIModel(b,[CollectivePBodyJump(pair,2)]),
                 PIModel(b,[LocalJump(sm;rate=(t,p)->1+0.1t)]))
    for (i,m) in pairs(jump_models)
        plan=TrajectoryPlan(m);work=TrajectoryWorkspace(plan,excited);t=0.025
        rates=PermutationalInvariantDynamics._channel_intensities!(
            work,excited.data,b,t,nothing)
        PermutationalInvariantDynamics._apply_gain!(work.channel_gain,
            excited.data,plan.jumps[1],b,work.jump_scales[1],
            work.liouvillian_work)
        reference_intensity=real(dot(plan.liouvillian.tracevec,
                                     work.channel_gain))
        @test rates[1]≈reference_intensity atol=2e-12 rtol=2e-12
        q=quantum_trajectory(m,excited,[0.0,0.05];dt=0.005,rng=MersenneTwister(i))
        @test abs(trace(q.states[end])-1)<2e-10
    end

    negative=PIModel(b,[LocalJump(sm;rate=-1)])
    @test_throws ArgumentError quantum_trajectory(negative,excited,times;dt=0.01)
    complex_rate=PIModel(b,[LocalJump(sm;rate=1+0.1im)])
    @test_throws ArgumentError quantum_trajectory(complex_rate,excited,times;dt=0.01)
    complex_real_rate=PIModel(b,[LocalJump(sm;rate=1+0im)])
    @test_throws ArgumentError quantum_trajectory(complex_real_rate,excited,times;dt=0.01)
    nonfinite_rate=PIModel(b,[LocalJump(sm;rate=Inf)])
    @test_throws ArgumentError quantum_trajectory(nonfinite_rate,excited,times;dt=0.01)
    b32=PIBasis(1,2);sm32=ComplexF32[0 1;0 0]
    excited32=iid_pure_state(b32,ComplexF32[0,1])
    model32=PIModel(b32,[LocalJump(sm32;rate=1f0)])
    plan32=TrajectoryPlan(model32)
    work32=TrajectoryWorkspace(plan32,excited32)
    q32=quantum_trajectory(plan32,excited32,Float32[0,0.05];dt=0.005f0,
        rng=MersenneTwister(8),workspace=work32)
    @test eltype(work32.intensities)===Float32
    @test eltype(q32.times)===Float32
    @test eltype(q32.states[end].data)===ComplexF32
    inferred_batch=@inferred quantum_trajectories(model32,excited32,
        Float32[0,0.02],9;seed=72,dt=0.01f0,threaded=true)
    @test length(inferred_batch)==9

    # Empty models inherit the initial-state precision through the convenience
    # API, or accept it explicitly when a reusable plan is constructed.
    empty32=PIModel(b32,())
    empty_plan32=TrajectoryPlan(empty32;T=Float32)
    @test empty_plan32.liouvillian.Ttype===ComplexF32
    @test eltype(empty_plan32.trace_weights)===Float32
    @test_throws ArgumentError TrajectoryPlan(empty32;T=AbstractFloat)
    @test_throws ArgumentError TrajectoryPlan(empty32;T=Union{Float32,Float64})
    empty_compiled=compile(empty32;backend=:matrixfree)
    compiled_empty_plan32=TrajectoryPlan(empty_compiled;T=Float32)
    @test compiled_empty_plan32.liouvillian.Ttype===ComplexF32
    empty_path=quantum_trajectory(empty_compiled,excited32,Float32[0,0.05];
        dt=0.005f0,rng=MersenneTwister(18))
    @test empty_path.times==Float32[0,0.05]
    @test empty_path.states[end].data==excited32.data

    # Adaptive stages, controller values, and callback times stay in the
    # prepared precision; no hidden Float64 stage may narrow into Float32.
    callback_time_types=DataType[]
    driven32=PIModel(b32,[LocalJump(sm32;rate=(t,p)->begin
        push!(callback_time_types,typeof(t));one(t)
    end)])
    seed32=91
    expected32=randexp(MersenneTwister(seed32),Float32)
    event32=quantum_trajectory(driven32,excited32,Float32[0,5];
        dt=0.5f0,dtmax=0.5f0,algorithm=:event,
        abstol=1f-6,reltol=1f-5,event_time_tolerance=1f-6,
        rng=MersenneTwister(seed32))
    @test all(==(Float32),callback_time_types)
    @test eltype(event32.times)===Float32
    @test eltype(event32.jump_times)===Float32
    @test eltype(event32.states[end].data)===ComplexF32
    @test only(event32.jump_times)≈expected32 atol=3f-5

    # Inputs wider than a prepared Float32 trajectory must be explicit rather
    # than silently rounded in integrator scratch.
    @test_throws ArgumentError quantum_trajectory(plan32,excited32,
        [0.0,0.05];dt=0.005f0,rng=MersenneTwister(8))
    @test_throws ArgumentError quantum_trajectory(plan32,excited32,
        Float32[0,0.05];dt=0.005,rng=MersenneTwister(8))
    wider_driven=TrajectoryPlan(PIModel(b32,[LocalJump(sm32;rate=(t,p)->1.0)]))
    @test_throws ArgumentError quantum_trajectory(wider_driven,excited32,
        Float32[0,0.05];dt=0.005f0,rng=MersenneTwister(8))
    @test_throws ArgumentError quantum_trajectory(model,excited,times;dt=0)
    @test_throws ArgumentError quantum_trajectory(model,excited,[NaN];dt=0.01)
    @test_throws ArgumentError quantum_trajectory(model,excited,[0.0,Inf];dt=0.01)
    @test PermutationalInvariantDynamics._select_jump_channel([0.0,1.0,0.0],0.0)==2
    @test_throws ArgumentError PermutationalInvariantDynamics._total_intensity(
        [floatmax(Float64),floatmax(Float64)])
    @test_throws ArgumentError trajectory_observable_statistics(qs,(bad=sm,))
end
