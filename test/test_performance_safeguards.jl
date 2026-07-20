@testset "high-level performance safeguards" begin
    basis=PIBasis(1,2)
    lowering=ComplexF64[0 1;0 0]
    model=PIModel(basis,[LocalJump(lowering;rate=0.3)])
    rho0=iid_pure_state(basis,ComplexF64[0,1])
    L=liouvillian(model;representation=:sparse)

    @test_throws ArgumentError PermutationalInvariantDynamics._memory_budget_bytes(true)
    @test_throws ArgumentError compile(model;memory_budget=true)
    @test_throws ArgumentError liouvillian(
        model;representation=:matrixfree,memory_budget=1)

    # Explicit dense and Krylov requests both respect the same budget. Auto
    # may change representation, but it may not allocate a solver which also
    # exceeds the declared limit.
    @test_throws ArgumentError steady_state(L;basis,method=:direct,
                                             memory_budget=1)
    @test_throws ArgumentError steady_state(L;basis,method=:krylov,
                                             krylovdim=4,memory_budget=1)
    @test_throws ArgumentError steady_state(L;basis,method=:krylov,
        krylovdim=4,preconditioner=:schur,memory_budget=1)
    @test_throws ArgumentError steady_state(L;basis,method=:auto,
                                             memory_budget=1)
    stationary=steady_state(L;basis,method=:direct,memory_budget=Inf)
    @test abs(dot(PermutationalInvariantDynamics._trace_vector(basis),
                  stationary)-1)<1e-11

    @test_throws ArgumentError pi_liouvillian_spectrum(
        L;method=:dense,memory_budget=1)
    @test_throws ArgumentError pi_liouvillian_spectrum(
        L;method=:krylov,nev=2,krylovdim=4,memory_budget=1)
    compiled_sparse=compile(model;backend=:sparse,memory_budget=Inf)
    @test_throws ArgumentError pi_liouvillian_spectrum(
        compiled_sparse;method=:krylov,nev=2,krylovdim=4,memory_budget=1)
    selected=pi_liouvillian_spectrum(L;method=:krylov,nev=2,krylovdim=4,
        return_info=true,require_convergence=false,memory_budget=Inf)
    @test selected.vectors===nothing
    @test length(selected.values)==2
    @test_throws ArgumentError liouvillian_eigenvalues(
        L,2;memory_budget=1)

    @test_throws ArgumentError floquet_propagator(
        model,0.1;steps=2,memory_budget=1)
    @test_throws ArgumentError floquet_map(
        model,0.1;steps=2,memory_budget=1)
    F=floquet_propagator(model,0.1;steps=2,memory_budget=Inf)
    @test_throws ArgumentError floquet_multipliers(F;memory_budget=1)
    map=floquet_map(model,0.1;steps=2)
    @test_throws ArgumentError floquet_steady_state(
        map;krylovdim=4,memory_budget=1)
    @test_throws ArgumentError floquet_steady_state(
        model,0.1;steps=2,krylovdim=4,memory_budget=1)
    @test_throws ArgumentError stroboscopic_evolution(
        rho0,F,2;memory_budget=1)
    @test_throws ArgumentError stroboscopic_evolution(
        rho0,F,typemax(Int);include_initial=true,memory_budget=Inf)
    @test length(stroboscopic_evolution(
        rho0,F,2;memory_budget=Inf))==3

    @test_throws ArgumentError liouvillian_modes(
        L;k=2,method=:dense,memory_budget=1)
    @test_throws ArgumentError liouvillian_modes(
        L;k=2,method=:arnoldi,memory_budget=1)
    @test_throws ArgumentError resolvent_norm(
        L,1.0;method=:dense,memory_budget=1)
    @test_throws ArgumentError resolvent_norm(
        model,1.0;method=:krylov,memory_budget=1)
    @test_throws ArgumentError ResponseWorkspace(
        model;krylovdim=4,memory_budget=1)
    @test_throws ArgumentError adjoint_evolve(
        L,collective_operator(basis,ComplexF64[1 0;0 -1]),0.1;
        method=:dense,memory_budget=1)
    @test_throws ArgumentError pseudospectral_abscissa(
        L,1.0;real_grid=1:400,imag_grid=1:400,max_grid_points=100_000)
    @test_throws ArgumentError pseudospectral_abscissa(
        L,1.0;real_grid=0:0,imag_grid=0:0,max_grid_points=true)

    times=[0.0,0.1]
    @test_throws ArgumentError quantum_trajectory(
        model,rho0,times;dt=0.01,memory_budget=1,
        rng=MersenneTwister(1))
    @test_throws ArgumentError quantum_trajectories(
        model,rho0,times,2;dt=0.01,memory_budget=1,seed=1)

    diffusive=DiffusivePlan(model,homodyne_monitor(lowering))
    @test_throws ArgumentError diffusive_trajectory(
        diffusive,rho0,times;dt=0.01,save_states=false,
        observables=(z=ComplexF64[1 0;0 -1],),memory_budget=55)
    diffusive_batch=DiffusiveBatchPlan(diffusive,rho0,times;dt=0.01,
        observables=(z=ComplexF64[1 0;0 -1],))
    @test_throws ArgumentError diffusive_trajectories(
        diffusive_batch,rho0,2;save_states=false,memory_budget=110)
end
