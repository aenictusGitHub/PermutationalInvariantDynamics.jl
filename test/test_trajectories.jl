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

    @testset "adaptive event timing" begin
        b1=PIBasis(1,2);excited1=iid_pure_state(b1,ComplexF64[0,1])
        decay=PIModel(b1,[LocalJump(sm)])
        seed=91;rng_reference=MersenneTwister(seed)
        expected=-log(rand(rng_reference))
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
        pair_expected=-log(rand(pair_rng))
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
        q=quantum_trajectory(m,excited,[0.0,0.05];dt=0.005,rng=MersenneTwister(i))
        @test abs(trace(q.states[end])-1)<2e-10
    end

    negative=PIModel(b,[LocalJump(sm;rate=-1)])
    @test_throws ArgumentError quantum_trajectory(negative,excited,times;dt=0.01)
    complex_rate=PIModel(b,[LocalJump(sm;rate=1+0.1im)])
    @test_throws ArgumentError quantum_trajectory(complex_rate,excited,times;dt=0.01)
    @test_throws ArgumentError quantum_trajectory(model,excited,times;dt=0)
    @test_throws ArgumentError trajectory_observable_statistics(qs,(bad=sm,))
end
