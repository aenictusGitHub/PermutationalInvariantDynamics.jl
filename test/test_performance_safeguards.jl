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

@testset "exact support counts and reusable resource preflight" begin
    PID=PermutationalInvariantDynamics
    dense=ComplexF64[0 -0.0 1e-300; -2im 0 3]
    stored=sparse(dense)
    stored[1,1]=1
    stored[1,1]=0 # a deliberately retained explicit CSC zero
    for (matrix,expected) in ((dense,3),(stored,3),
                              (view(dense,:,2:3),2),(transpose(dense),3))
        count=PID._performance_matrix_nonzeros(matrix)
        @test count isa BigInt
        @test count==expected
    end
    @test PID._performance_matrix_nonzeros(zeros(0,3))==0
    huge=big(typemax(Int))+1
    bounds=PID._performance_sparse_materialization_bounds(
        typemax(Int),ComplexF64,huge)
    @test bounds.retained_nnz_upper_bound==huge
    @test bounds.operator_bytes>typemax(Int)

    basis=PIBasis(6,2)
    spin=spin_matrices()
    model=PIModel(basis,(LocalJump(spin.jm;rate=0.4),
                         CollectiveHamiltonian(spin.jx;rate=0.1)))
    family=compile_family(model)
    for source in (compile(model;backend=:matrixfree),
                   specialize(family,(0.4,0.1)))
        @test PID._prepared_resource_metadata(source)!==nothing
        for precision_bits in (128,512)
            @test PID._performance_prepared_sparse_bounds(
                source;bigfloat_precision=precision_bits)==
                PID._performance_sparse_materialization_bounds(
                    source.plan;bigfloat_precision=precision_bits)
            report=recommend_solver(source;task=:dynamics,
                bigfloat_precision=precision_bits)
            geometry=PID._estimate_model_geometry(source.model;
                bigfloat_precision=precision_bits)
            @test report.geometry_setup_upper_bytes==geometry.setup_bytes
            @test report.geometry_retained_upper_bytes==geometry.retained_bytes
        end
        before=recommend_solver(source;task=:spectrum,
            algorithm=:block_arnoldi,nev=4,block_size=4,krylovdim=8)
        before_actual=Base.summarysize(source)
        @test before.retained_bytes>=before_actual
        @test before.resources.retained.provenance===:upper_bound
        input=ones(ComplexF64,length(basis),4)
        output=similar(input)
        mul!(output,source,input)
        after=recommend_solver(source;task=:spectrum,
            algorithm=:block_arnoldi,nev=4,block_size=4,krylovdim=8)
        after_actual=Base.summarysize(source)
        @test after_actual>before_actual
        @test after.retained_bytes>=after_actual
        # The cached inline-record allowance can already cover new wrapper
        # records. Require live growth for all three numerical batch buffers;
        # the total upper-bound checks above cover their container storage.
        batch_payload=3maximum(length,basis.patterns)^2*
            size(input,2)*sizeof(eltype(input))
        @test after.retained_bytes-before.retained_bytes>=
              batch_payload
        @test after.operator_action_per_worker_upper_bytes<
              before.operator_action_per_worker_upper_bytes
        @test recommend_solver(source;task=:dynamics,
            memory_budget=after_actual-1).budget_status===:exceeds
        # A conservative cached record must not reject a budget that fits
        # the measured source plus the requested workspace and output.
        measured_peak=after.known_peak_bytes-after.retained_bytes+after_actual
        tight=recommend_solver(source;task=:spectrum,
            algorithm=:block_arnoldi,nev=4,block_size=4,krylovdim=8,
            memory_budget=measured_peak)
        @test tight.budget_status===:fits
        @test tight.retained_bytes==after_actual
        one_worker=recommend_solver(source;task=:dynamics,samples=2,saved_states=0)
        two_workers=recommend_solver(source;task=:dynamics,
            samples=2,saved_states=0,workers=2)
        @test two_workers.retained_bytes==one_worker.retained_bytes
        @test two_workers.solve_workspace_bytes==2one_worker.solve_workspace_bytes
    end

    # Sparse storage and mutable callback captures must be inspected live.
    source=compile(model;backend=:sparse)
    before=recommend_solver(source)
    for column in 1:8,row in 1:8
        source.operator[row,column]=1
    end
    after=recommend_solver(source)
    @test after.retained_bytes>before.retained_bytes
    @test after.retained_bytes>=Base.summarysize(source)
    captured=ones(1)
    rate=let values=captured
        (time,parameters)->values[1]
    end
    driven=compile(PIModel(basis,(LocalJump(spin.jm;rate),));backend=:matrixfree)
    @test PID._prepared_resource_metadata(driven)===nothing
    before=recommend_solver(driven;task=:dynamics)
    resize!(captured,4096)
    after=recommend_solver(driven;task=:dynamics)
    @test after.retained_bytes>before.retained_bytes
    @test after.retained_bytes==Base.summarysize(driven)

    wide=compile(PIModel(PIBasis(1,2),(
        LocalJump(Complex{BigFloat}.(spin.jm);rate=big"0.4"),));backend=:matrixfree)
    @test PID._prepared_resource_metadata(wide)===nothing
    @test recommend_solver(wide;task=:dynamics,bigfloat_precision=512).
        retained_bytes==Base.summarysize(wide)
end
