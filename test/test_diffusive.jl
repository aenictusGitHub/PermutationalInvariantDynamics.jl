@testset "Diffusive PI conditional dynamics" begin
    b=PIBasis(1,2)
    sm=ComplexF64[0 1;0 0]
    rho0=computational_product_state(b,2)

    @test_throws ArgumentError DiffusiveMonitor(sm;kind=:counting)
    @test_throws ArgumentError homodyne_monitor(sm;efficiency=-0.1)
    @test_throws ArgumentError homodyne_monitor(sm;efficiency=1.1)
    @test_throws ArgumentError homodyne_monitor(sm;phase=1+im)
    @test_throws DimensionMismatch DiffusivePlan(
        PIModel(b,()),homodyne_monitor(zeros(3,3));T=Float64)
    @test_throws ArgumentError DiffusivePlan(PIModel(b,()),();T=Float64)

    gamma=0.4
    model=PIModel(b,(CollectiveJump(sm;rate=gamma),))
    monitor=homodyne_monitor(sqrt(gamma)*sm;
        efficiency=0.8,phase=0.2,label=:emission)
    plan=DiffusivePlan(model,monitor)
    work=DiffusiveWorkspace(plan,rho0)
    times=collect(0.0:0.05:0.2)
    trajectory=diffusive_trajectory(plan,rho0,times;dt=0.002,
        rng=MersenneTwister(11),workspace=work,
        observables=(population=ComplexF64[0 0;0 1],))
    @test length(trajectory)==length(times)
    @test size(trajectory.records)==(1,length(times))
    @test size(trajectory.innovations)==size(trajectory.records)
    @test trajectory.record_labels==Any[:emission]
    @test trajectory.observables.names==(:population,)
    @test size(trajectory.observables.values)==(1,length(times))
    @test all(state->isapprox(trace(state),1;atol=2e-12),trajectory)
    @test all(state->hermiticity_error(state)<2e-11,trajectory)

    state_free=diffusive_trajectory(plan,rho0,times;dt=0.002,
        rng=MersenneTwister(11),save_states=false)
    @test state_free.states===nothing
    @test state_free.records==trajectory.records
    @test_throws ArgumentError state_free[1]
    @test_throws ArgumentError diffusive_trajectory(plan,rho0,[0.0];dt=0.002,
        observables=(lowering=sm,))
    incompatible=iid_state(PIBasis(1,2),ComplexF64[1 0;0 0])
    @test_throws ArgumentError diffusive_trajectory(
        plan,incompatible,[0.0];dt=0.002,workspace=work)

    heterodyne=heterodyne_monitor(sqrt(gamma)*sm;
        efficiency=0.7,label=:field)
    heterodyne_result=diffusive_trajectory(
        DiffusivePlan(model,heterodyne),rho0,times;dt=0.002,
        rng=MersenneTwister(5))
    @test size(heterodyne_result.records)==(2,length(times))
    @test heterodyne_result.record_labels==Any[(:field,:I),(:field,:Q)]
    @test all(state->isapprox(trace(state),1;atol=2e-12),
              heterodyne_result.states)

    # Zero efficiency leaves only the deterministic Euler drift while the
    # detector record is vacuum noise.  Reusing the same seed therefore gives
    # identical states for unrelated monitor phases.
    dark_a=DiffusivePlan(model,homodyne_monitor(sm;efficiency=0,phase=0))
    dark_b=DiffusivePlan(model,homodyne_monitor(sm;efficiency=0,phase=1.3))
    result_a=diffusive_trajectory(dark_a,rho0,times;dt=0.001,
                                  rng=MersenneTwister(8))
    result_b=diffusive_trajectory(dark_b,rho0,times;dt=0.001,
                                  rng=MersenneTwister(8))
    @test all(i->isapprox(result_a[i].data,result_b[i].data;atol=2e-13),
              eachindex(times))
    @test result_a.records==result_a.innovations

    # The stochastic ensemble recovers the unconditional master equation.
    ensemble=diffusive_trajectories(plan,rho0,[0.0,0.15],600;
        dt=0.0015,seed=91,save_states=true)
    averaged=diffusive_average(ensemble)
    deterministic=time_evolution(model,rho0,[0.0,0.15];steps_per_interval=300)
    @test norm(averaged[end].data-deterministic[end].data)<0.065

    serial=diffusive_trajectories(plan,rho0,[0.0,0.02],4;
        dt=0.002,seed=123)
    threaded=diffusive_trajectories(plan,rho0,[0.0,0.02],4;
        dt=0.002,seed=123,threaded=true)
    @test map(x->x.records,serial)==map(x->x.records,threaded)
    @test map(x->last(x).data,serial)==map(x->last(x).data,threaded)

    batch=DiffusiveBatchPlan(plan,rho0,[0.0,0.02];dt=0.002,
        observables=(population=ComplexF64[0 0;0 1],))
    batch_work=DiffusiveBatchWorkspace(batch,rho0;workers=2)
    prepared_serial=diffusive_trajectories(batch,rho0,4;
        seed=123,workspace=batch_work)
    prepared_threaded=diffusive_trajectories(batch,rho0,4;
        seed=123,threaded=true,workspace=batch_work)
    @test map(x->x.records,prepared_serial)==map(x->x.records,serial)
    @test map(x->x.records,prepared_serial)==map(x->x.records,prepared_threaded)
    @test all(result->result.observables.names==(:population,),prepared_serial)
    @test occursin("DiffusiveBatchPlan",sprint(show,batch))
    @test_throws ArgumentError DiffusiveBatchWorkspace(batch,rho0;workers=0)
    other_batch=DiffusiveBatchPlan(plan,rho0,[0.0,0.01];dt=0.001)
    @test_throws ArgumentError diffusive_trajectories(
        other_batch,rho0,2;workspace=batch_work)
    same_basis_rho32=computational_product_state(b,2;T=Float32)
    @test_throws ArgumentError diffusive_trajectories(
        batch,same_basis_rho32,2;workspace=batch_work)

    scheduled=DiffusivePlan(model,homodyne_monitor(sm;
        efficiency=(t,p)->p.eta,phase=(t,p)->p.phase))
    @test length(diffusive_trajectory(scheduled,rho0,[0.0,0.01];dt=0.002,
        parameters=(eta=0.5,phase=0.1),rng=MersenneTwister(2)))==2
    @test_throws ArgumentError diffusive_trajectory(scheduled,rho0,[0.0,0.01];
        dt=0.002,parameters=(eta=1.5,phase=0.1),rng=MersenneTwister(2))

    b32=PIBasis(1,2)
    sm32=ComplexF32[0 1;0 0]
    rho32=computational_product_state(b32,2;T=Float32)
    model32=PIModel(b32,(CollectiveJump(sm32;rate=Float32(0.2)),))
    plan32=DiffusivePlan(model32,homodyne_monitor(sm32;
        efficiency=Float32(0.5),phase=Float32(0)))
    result32=diffusive_trajectory(plan32,rho32,Float32[0,0.01];
        dt=Float32(0.002),rng=MersenneTwister(3))
    @test eltype(result32.records)===Float32
    @test eltype(last(result32).data)===ComplexF32
end
