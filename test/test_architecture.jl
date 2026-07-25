@testset "shared source, result, and solver protocols" begin
    PID=PermutationalInvariantDynamics
    basis=PIBasis(1,2)
    spin=spin_matrices()
    prototype=PIModel(basis,(
        LocalHamiltonian(spin.jx;rate=0.2),
        LocalJump(spin.jm;rate=0.7),
    ))
    matrixfree=compile(prototype;backend=:matrixfree)
    sparse_source=compile(prototype;backend=:sparse)
    family=compile_family(prototype)
    specialized=specialize(family,(0.2,0.7);backend=:matrixfree)

    for source in (prototype,matrixfree,sparse_source,specialized,
                   matrixfree.plan,matrixfree.operator)
        @test PID._operator_basis(source)===basis
    end
    @test matrixfree.plan.tracevec isa SparseVector
    @test nnz(matrixfree.plan.tracevec)==
        sum(length,matrixfree.plan.basis.patterns;init=0)
    @test collect(matrixfree.plan.tracevec)==
        PID._trace_vector(basis,eltype(matrixfree))
    @test PID._operator_trace_functional(matrixfree)===
        matrixfree.plan.tracevec
    @test PID._operator_trace_vector(matrixfree)===matrixfree.plan.tracevec
    @test PID._operator_trace_vector(specialized)===specialized.plan.tracevec
    @test PID._linear_operator_workspace(matrixfree) isa LiouvillianWorkspace
    @test PID._linear_operator_workspace(specialized) isa LiouvillianWorkspace
    @test PID._linear_operator_workspace(sparse_source)===nothing
    @test PID._operator_has_adjoint(matrixfree)
    @test PID._operator_has_adjoint(specialized)
    @test iszero(PID._performance_source_action_bytes(
        matrixfree.plan,eltype(matrixfree)))
    @test PID._performance_linear_operator_workspace_bytes(matrixfree.plan)>0
    opaque=MatrixFreeLiouvillian(3,(y,x,t,p)->copyto!(y,x),ComplexF64,
        ComplexF64[1,0,0];workspace=Ref(:opaque))
    @test PID._performance_source_action_bytes(opaque,ComplexF64)==
        PID._performance_array_bytes(3,ComplexF64,0;linear_arrays=16)
    @test PID._canonical_stationary_algorithm(:shift_invert)===:shiftinvert
    @test PID._canonical_stationary_algorithm(:inverse_iteration)===:shiftinvert
    @test PID._canonical_stationary_algorithm(:krylov)===:gmres
    @test PID._canonical_spectrum_algorithm(:ordinary_arnoldi)===:arnoldi
    @test PID._canonical_spectrum_algorithm(:block)===:block_arnoldi
    @test PID._canonical_spectrum_algorithm(:implicit_qr)===:iram
    @test PID._canonical_spectrum_algorithm(:jacobi_davidson)===:jd

    rho0=iid_pure_state(basis,ComplexF64[1,1]/sqrt(2))
    problem=dynamics_problem(specialized,rho0,(0.0,0.05))
    derivative=similar(rho0.data)
    problem.f(derivative,rho0.data,nothing,0.0)
    reference=similar(derivative)
    apply!(reference,specialized,rho0.data,0.0,nothing,
           LiouvillianWorkspace(specialized))
    @test derivative≈reference atol=2e-13 rtol=2e-13

    period_map=floquet_map(specialized,0.05;steps=2)
    @test period_map.basis===basis
    @test period_map.tracevec≈specialized.plan.tracevec
    @test PID._operator_has_adjoint(period_map)

    bath=HEOMBath(collective_operator(basis,spin.jz),0.03,1.2)
    hierarchy=HEOMPlan(specialized,bath;max_depth=1)
    hierarchy_work=HEOMWorkspace(hierarchy)
    @test hierarchy_work.system isa LiouvillianWorkspace
    @test hierarchy_work.system.batch.capacity==heom_number_ados(hierarchy)
    expected_heom_bytes=3hierarchy.npi*sizeof(eltype(hierarchy))+
        PID._performance_liouvillian_workspace_bytes(
            specialized.plan;batch_columns=heom_number_ados(hierarchy))
    @test PID._performance_heom_workspace_bytes(hierarchy)==
          expected_heom_bytes
    hierarchy_input=zeros(eltype(hierarchy),size(hierarchy,1))
    hierarchy_input[1]=1
    hierarchy_output=similar(hierarchy_input)
    apply!(hierarchy_output,hierarchy,hierarchy_input,hierarchy_work)
    @test all(isfinite,hierarchy_output)

    states=[copy(rho0),copy(rho0),copy(rho0)]
    result=DynamicsResult([0.0,1.0,2.0],states,:rk4)
    @test state(result,1)===states[1]
    @test state_at(result,1)===states[2]
    @test state_at(result,2.0)===states[3]
    @test_throws ArgumentError state_at(result,1.25)

    recommendation=recommend_solver(prototype;task=:spectrum,
        algorithm=:ordinary_arnoldi,nev=2,krylovdim=4,memory_budget=Inf)
    @test recommendation.algorithm===:arnoldi
    alias_values=liouvillian_spectrum(prototype;nev=2,
        algorithm=:ordinary_arnoldi,krylovdim=length(basis),
        require_convergence=false)
    canonical_values=liouvillian_spectrum(prototype;nev=2,
        algorithm=:arnoldi,krylovdim=length(basis),
        require_convergence=false)
    @test alias_values≈canonical_values atol=2e-12 rtol=2e-12
    compiled_alias=pi_liouvillian_spectrum(sparse_source;
        method=:ordinary_arnoldi,nev=2,krylovdim=length(basis),
        return_info=true,require_convergence=false)
    @test compiled_alias.method===:arnoldi
    @test_throws ArgumentError liouvillian_spectrum(
        prototype;algorithm=DirectAlgorithm(),nev=2)

    steady=stationary_state(prototype;algorithm=:krylov,
        return_info=true,krylovdim=4,maxiter=200)
    @test steady.info.selected_algorithm===:gmres
end
