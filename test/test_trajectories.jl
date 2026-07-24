@testset "PI quantum-jump trajectories" begin
    sm=ComplexF64[0 1;0 0];sx=ComplexF64[0 1;1 0];b=PIBasis(2,2)
    excited=iid_pure_state(b,ComplexF64[0,1]);times=collect(range(0,0.5;length=6))

    # Decimal fixed steps must not leave a one-ulp remainder that triggers an
    # eleventh stochastic/RK evaluation on the nominal ten-step interval.
    t=0.0;target=0.1;steps=0
    while t<target
        h,lands=PermutationalInvariantDynamics._trajectory_step_to_target(
            t,target,0.01)
        t=lands ? target : t+h
        steps+=1
    end
    @test t==target
    @test steps==10

    # Roundoff snapping must remain relative to the time scale; it may not
    # turn a genuinely small maximum step into one much larger physical step.
    tiny_h,tiny_lands=PermutationalInvariantDynamics.
        _trajectory_step_to_target(0.0,1e-16,1e-20)
    @test tiny_h==1e-20
    @test !tiny_lands

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
        fixed_work=TrajectoryWorkspace(plan,excited;mode=:fixed)
        full_work=TrajectoryWorkspace(plan,excited)
        @test fixed_work.mode===:fixed
        @test all(isempty,(fixed_work.k3,fixed_work.k4,
                           fixed_work.k5,fixed_work.k6,fixed_work.k7,
                           fixed_work.trial,fixed_work.embedded,fixed_work.start))
        @test sum(length,(fixed_work.tmp,fixed_work.k1,fixed_work.k2))==
              3length(excited.data)
        @test Base.summarysize(fixed_work)<Base.summarysize(full_work)
        @test quantum_trajectory(plan,excited,[0.0,0.02];dt=0.01,
            rng=MersenneTwister(912),workspace=fixed_work) isa QuantumTrajectory
        @test_throws ArgumentError quantum_trajectory(
            plan,excited,[0.0,0.02];algorithm=:event,dt=0.01,
            rng=MersenneTwister(912),workspace=fixed_work)
        @test_throws ArgumentError TrajectoryWorkspace(plan,excited;mode=:invalid)
        fixed_batch=TrajectoryBatchWorkspace(plan,excited;workers=2,mode=:fixed)
        @test all(worker->worker.mode===:fixed,fixed_batch.workers)
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

    @testset "streamed trajectory steady state" begin
        steady_basis=PIBasis(1,2)
        steady_initial=iid_pure_state(
            steady_basis,ComplexF64[0,1])
        steady_model=PIModel(steady_basis,
            [LocalJump(sm;rate=3.0),
             LocalJump(adjoint(sm);rate=2.0)])
        steady_plan=TrajectoryPlan(steady_model)
        steady_batch=TrajectoryBatchWorkspace(
            steady_plan,steady_initial;workers=max(2,Threads.nthreads()))
        steady_times=[0.0,0.1,0.2,0.3]
        path_count=8
        explicit_paths=quantum_trajectories(
            steady_plan,steady_initial,steady_times,path_count;
            seed=184,dt=0.01,workspace=steady_batch)
        path_means=[reduce(+,
            (path.states[index].data for index in 2:length(steady_times)))/3
            for path in explicit_paths]
        reference_mean=reduce(+,path_means)/path_count
        reference_m2=sum(norm(path_mean-reference_mean)^2
                         for path_mean in path_means)
        reference_spread=sqrt(reference_m2/(path_count-1))
        reference_standard_error=reference_spread/sqrt(path_count)
        excitation=ComplexF64[0 0;0 1]
        excitation_operator=collective_operator(steady_basis,excitation)
        path_excitations=[real(dot(excitation_operator.data,path_mean))
                          for path_mean in path_means]
        mean_excitation=sum(path_excitations)/path_count
        variance_excitation=sum(abs2(value-mean_excitation)
            for value in path_excitations)/(path_count-1)

        estimate=trajectory_steady_state(
            steady_plan,steady_initial;
            trajectories=path_count,settling_time=0.1,
            samples_per_trajectory=3,sampling_interval=0.1,
            seed=184,dt=0.01,workspace=steady_batch,
            observables=(excitation=excitation,),return_info=true)
        @test estimate isa TrajectorySteadyStateResult
        @test estimate.state.data≈reference_mean atol=2e-15
        @test estimate.trajectory_count==path_count
        @test estimate.samples_per_trajectory==3
        @test estimate.sampling_times≈steady_times[2:end] atol=eps(Float64)
        @test estimate.sample_spread≈reference_spread atol=2e-15
        @test estimate.standard_error≈reference_standard_error atol=2e-15
        @test estimate.trace_error≈abs(trace(estimate.state)-1) atol=1e-15
        sparse_steady=liouvillian(steady_model;representation=:sparse)
        @test estimate.residual≈norm(sparse_steady*estimate.state.data) atol=2e-14
        @test estimate.relative_residual≈
            estimate.residual/max(norm(estimate.state.data),1) atol=2e-14
        observable_estimate=estimate.observables.observables[:excitation]
        @test observable_estimate.mean≈mean_excitation atol=2e-15
        @test observable_estimate.variance≈variance_excitation atol=2e-15
        @test observable_estimate.standard_error≈
            sqrt(variance_excitation/path_count) atol=2e-15
        @test observable_estimate.lower<=observable_estimate.mean<=
              observable_estimate.upper
        @test estimate.metadata.worker_count==1
        @test estimate.metadata.sampling_interval==0.1
        @test Base.summarysize(estimate)<Base.summarysize(explicit_paths)
        @test occursin("HS standard error",sprint(show,estimate))

        state_only=trajectory_steady_state(
            steady_plan,steady_initial;
            trajectories=path_count,settling_time=0.1,
            samples_per_trajectory=3,sampling_interval=0.1,
            seed=184,dt=0.01,workspace=steady_batch)
        @test state_only isa PIState
        @test state_only.data≈estimate.state.data atol=2e-15

        event_estimate=trajectory_steady_state(
            steady_plan,steady_initial;
            trajectories=2,settling_time=0.05,dt=0.05,dtmax=0.05,
            algorithm=:event,seed=713,return_info=true)
        @test event_estimate.sampling_times==[0.05]
        @test abs(trace(event_estimate.state)-1)<2e-10

        if Threads.nthreads()>1
            threaded_estimate=trajectory_steady_state(
                steady_plan,steady_initial;
                trajectories=path_count,settling_time=0.1,
                samples_per_trajectory=3,sampling_interval=0.1,
                seed=184,dt=0.01,threaded=true,workspace=steady_batch,
                observables=(excitation=excitation,),return_info=true)
            @test threaded_estimate.state.data≈estimate.state.data atol=2e-14
            @test threaded_estimate.sample_spread≈
                  estimate.sample_spread atol=2e-14
            @test threaded_estimate.observables.observables[:excitation].mean≈
                  observable_estimate.mean atol=2e-14
            @test threaded_estimate.metadata.threaded
        end

        driven_steady=PIModel(steady_basis,
            [LocalJump(sm;rate=(t,p)->one(t))])
        @test_throws ArgumentError trajectory_steady_state(
            driven_steady,steady_initial;
            trajectories=2,settling_time=0.1,dt=0.01)
        @test_throws ArgumentError trajectory_steady_state(
            steady_plan,steady_initial;
            trajectories=1,settling_time=0.1,dt=0.01)
        @test_throws ArgumentError trajectory_steady_state(
            steady_plan,steady_initial;
            trajectories=big(typemax(Int))+1,settling_time=0.1,dt=0.01)
        @test_throws ArgumentError trajectory_steady_state(
            steady_plan,steady_initial;
            trajectories=2,settling_time=0.0,dt=0.01)
        @test_throws ArgumentError trajectory_steady_state(
            steady_plan,steady_initial;
            trajectories=2,settling_time=0.1,
            samples_per_trajectory=2,dt=0.01)
        @test_throws ArgumentError trajectory_steady_state(
            steady_plan,steady_initial;
            trajectories=2,settling_time=0.1,
            samples_per_trajectory=2,sampling_interval=0.0,dt=0.01)
        @test_throws ArgumentError trajectory_steady_state(
            steady_plan,steady_initial;
            trajectories=2,settling_time=1.0,
            samples_per_trajectory=2,
            sampling_interval=eps(Float64)/4,dt=0.01)
        @test_throws ArgumentError trajectory_steady_state(
            steady_plan,steady_initial;
            trajectories=2,settling_time=0.1,dt=0.01,
            observables=(bad=sm,),return_info=true)
        @test_throws ArgumentError trajectory_steady_state(
            steady_plan,steady_initial;
            trajectories=2,settling_time=0.1,dt=0.01,
            observables=(excitation=excitation,))
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

        # One accepted DOPRI trial evaluates every scalar channel rate once
        # per distinct physical node. Its sixth and seventh stages share the
        # endpoint, while dense event localization and root-state
        # reconstruction perform no additional RHS evaluation.
        rate_calls=Ref(0)
        counted_decay=PIModel(b1,[LocalJump(sm;rate=(t,p)->begin
            rate_calls[]+=1
            1+t
        end)])
        counted_plan=TrajectoryPlan(counted_decay)
        counted_work=TrajectoryWorkspace(counted_plan,excited1)
        hazard,error=PermutationalInvariantDynamics._conditional_dopri_trial!(
            counted_work,excited1.data,b1,0.0,0.4,nothing,
            counted_plan.liouvillian.tracevec,1e-11,1e-10)
        @test isfinite(hazard)&&isfinite(error)
        @test rate_calls[]==6
        PermutationalInvariantDynamics._prepare_dopri_dense_output!(
            counted_work)
        root=PermutationalInvariantDynamics._dopri_dense_root(
            counted_work,0.4,hazard/2,hazard,1e-12)
        @test root!==nothing
        PermutationalInvariantDynamics._dopri_dense_state!(
            counted_work.tmp,excited1.data,counted_work,0.4,root/0.4)
        @test rate_calls[]==6

        # Fixed RK4 similarly prepares only t, t+h/2, and t+h. Channel
        # resolution at the endpoint reuses the already evaluated scales.
        rate_calls[]=0
        copyto!(counted_work.current,excited1.data)
        PermutationalInvariantDynamics._reset_effective_jump_cache!(
            counted_work)
        PermutationalInvariantDynamics._conditional_rk4!(
            counted_work.current,counted_work,b1,0.0,0.1,nothing,
            counted_plan.liouvillian.tracevec)
        @test rate_calls[]==3
        PermutationalInvariantDynamics._channel_intensities!(
            counted_work,counted_work.current,b1,0.1,nothing)
        @test rate_calls[]==3

        # Reusing a workspace for a new driven trajectory invalidates the
        # node cache before applying a different parameter set.
        parameter_calls=Ref(0)
        parameter_model=PIModel(b1,[LocalJump(sm;
            rate=(t,p)->begin
                parameter_calls[]+=1
                p.gamma*(1+t)
            end)])
        parameter_plan=TrajectoryPlan(parameter_model)
        parameter_work=TrajectoryWorkspace(parameter_plan,excited1)
        quantum_trajectory(parameter_plan,excited1,[0.0,0.02];
            dt=0.01,parameters=(gamma=0.1,),rng=MersenneTwister(1701),
            workspace=parameter_work)
        @test parameter_calls[]==5
        @test parameter_work.jump_scales[1]≈0.102
        quantum_trajectory(parameter_plan,excited1,[0.0,0.02];
            dt=0.01,parameters=(gamma=0.2,),rng=MersenneTwister(1701),
            workspace=parameter_work)
        @test parameter_calls[]==10
        @test parameter_work.jump_scales[1]≈0.204

        # Combining Q=sum_c gamma_c K_c'K_c gives the same normalized
        # no-jump RHS as the channel-by-channel reference contraction.
        combined_model=PIModel(b1,[LocalJump(sm;rate=0.7),
            LocalJump(adjoint(sm);rate=0.2)])
        combined_plan=TrajectoryPlan(combined_model)
        combined_work=TrajectoryWorkspace(combined_plan,excited1)
        combined_rhs=similar(excited1.data)
        combined_total=PermutationalInvariantDynamics.
            _conditional_action_and_intensity!(combined_rhs,excited1.data,
                combined_work,b1,0.0,nothing,
                combined_plan.liouvillian.tracevec)
        reference_rhs=zeros(eltype(excited1.data),length(excited1.data))
        reference_total=PermutationalInvariantDynamics.
            _apply_conditional_jumps!(reference_rhs,excited1.data,
                combined_work,b1,0.0,nothing,combined_plan.jumps,1)
        @. reference_rhs=reference_rhs+reference_total*excited1.data
        @test combined_total≈reference_total atol=2e-14
        @test combined_rhs≈reference_rhs atol=3e-14 rtol=3e-14

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
    steady32=trajectory_steady_state(plan32,excited32;
        trajectories=2,settling_time=0.02f0,dt=0.01f0,return_info=true)
    @test eltype(steady32.state.data)===ComplexF32
    @test eltype(steady32.sampling_times)===Float32
    @test steady32.sample_spread isa Float32
    @test steady32.standard_error isa Float32
    @test steady32.residual isa Float32

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
