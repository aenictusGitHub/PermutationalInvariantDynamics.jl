@testset "confidence-controlled trajectory ensembles" begin
    basis=PIBasis(1,2)
    rho0=computational_product_state(basis,2)
    sz=ComplexF64[-1 0;0 1]
    identity_observable=ComplexF64[1 0;0 1]
    empty_model=PIModel(basis,())

    exact=adaptive_quantum_trajectories(empty_model,rho0,[0.0,0.1];
        observables=(identity=identity_observable,),dt=0.01,min_trajectories=4,
        max_trajectories=12,batch_size=2,atol=1e-12,rtol=0,
        seed=12,threaded=false)
    exact_threaded=adaptive_quantum_trajectories(empty_model,rho0,[0.0,0.1];
        observables=(identity=identity_observable,),dt=0.01,min_trajectories=4,
        max_trajectories=12,batch_size=2,atol=1e-12,rtol=0,
        seed=12,threaded=true)
    @test exact.converged
    @test exact.stopping_reason===:confidence_target
    @test exact.trajectory_count==4
    @test exact.observables.observables[:identity].mean==[1.0,1.0]
    @test exact.observables.observables[:identity].variance==[0.0,0.0]
    @test exact_threaded.observables==exact.observables
    @test exact_threaded.convergence_history==exact.convergence_history
    @test occursin("AdaptiveTrajectoryResult",sprint(show,exact))
    @test_throws ArgumentError adaptive_quantum_trajectories(
        empty_model,rho0,[0.0,0.1];observables=(z=sz,),dt=0.01,
        min_trajectories=1)
    @test_throws ArgumentError adaptive_quantum_trajectories(
        empty_model,rho0,[0.0,0.1];observables=(z=sz,),dt=0.01,
        min_trajectories=4,max_trajectories=3)
    @test_throws ArgumentError adaptive_quantum_trajectories(
        empty_model,rho0,[0.0,0.1];observables=(z=sz,),dt=0.01,
        min_trajectories=2,max_trajectories=2,atol=0,rtol=0)
    @test_throws ArgumentError adaptive_quantum_trajectories(
        empty_model,rho0,[0.0,0.1];observables=(z=sz,),dt=0.01,
        min_trajectories=big(typemax(Int))+1,
        max_trajectories=big(typemax(Int))+1)
    @test PermutationalInvariantDynamics._adaptive_statistics_type(
        Float16,70_000)===Float32

    monitor=homodyne_monitor(zeros(ComplexF64,2,2);efficiency=0)
    diffusive_plan=DiffusivePlan(empty_model,monitor;T=Float64)
    batch=DiffusiveBatchPlan(diffusive_plan,rho0,[0.0,0.1];dt=0.01,
        observables=(identity=identity_observable,))
    batch_workspace=DiffusiveBatchWorkspace(batch,rho0;workers=2)
    diffusive=adaptive_diffusive_trajectories(batch,rho0;
        min_trajectories=4,max_trajectories=12,batch_size=2,
        atol=1e-12,rtol=0,seed=9,threaded=true,
        workspace=batch_workspace)
    @test diffusive.backend===:diffusive
    @test diffusive.converged
    @test diffusive.trajectory_count==4
    @test diffusive.observables.observables[:identity].mean==[1.0,1.0]

    # A rare jump may be absent from an early batch.  Zero empirical variance
    # must not certify a nonconstant observable at zero uncertainty.
    emission=PIModel(basis,(LocalJump(ComplexF64[0 1;0 0];rate=0.01),))
    rare=adaptive_quantum_trajectories(emission,rho0,[0.0,1.0];
        observables=(excited=ComplexF64[0 0;0 1],),dt=0.01,
        min_trajectories=64,max_trajectories=64,batch_size=64,
        confidence=0.95,atol=1e-3,rtol=0,seed=2)
    @test !rare.converged
    @test rare.stopping_reason===:maximum_trajectories
    @test rare.convergence_history[end].worst_half_width>0

    # Statistical storage follows the promoted observable precision, while
    # the prepared trajectory integrator remains Float32.
    rho32=computational_product_state(basis,2;T=Float32)
    model32=PIModel(basis,(
        LocalHamiltonian(zeros(ComplexF32,2,2);rate=0.0f0),))
    promoted_statistics=adaptive_quantum_trajectories(
        model32,rho32,Float32[0,0.01];
        observables=(identity=identity_observable,),dt=0.01f0,
        min_trajectories=2,max_trajectories=2,batch_size=2,
        confidence=0.95,atol=1e-10,rtol=0.0)
    @test promoted_statistics.confidence isa Float64
    @test promoted_statistics.converged

    no_observable_batch=DiffusiveBatchPlan(
        diffusive_plan,rho0,[0.0,0.1];dt=0.01)
    @test_throws ArgumentError adaptive_diffusive_trajectories(
        no_observable_batch,rho0;min_trajectories=2,max_trajectories=2)
end
