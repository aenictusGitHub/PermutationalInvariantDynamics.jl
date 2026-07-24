@testset "collective covariance, squeezing, and correlations" begin
    sx=ComplexF64[0 1;1 0];sy=ComplexF64[0 -im;im 0];sz=ComplexF64[1 0;0 -1]
    for N in 2:7
        b=PIBasis(N,2);rho=iid_pure_state(b,ComplexF64[1,0]);C=collective_covariance_matrix(rho,[sx/2,sy/2,sz/2])
        @test C≈Diagonal([N/4,N/4,0]) atol=3e-10
        @test kitagawa_ueda_squeezing(rho)≈1 atol=3e-10
        @test wineland_squeezing(rho)≈1 atol=3e-10
        @test two_body_expectation(rho,sz,sz)≈1 atol=3e-10
        @test connected_two_body_correlation(rho,sz,sz)≈0 atol=3e-10
        @test trace(two_body_rdm(rho))≈1 atol=2e-10
    end
end

@testset "Matrix-free response workspaces" begin
    b=PIBasis(2,2);sm=ComplexF64[0 1;0 0];sx=ComplexF64[0 1;1 0]
    damping=PIModel(b,[LocalJump(sm)])
    matrixfree=liouvillian(damping;representation=:matrixfree)
    dense=Matrix(liouvillian(damping;representation=:sparse))
    rho=iid_pure_state(b,ComplexF64[1,0])
    observable=collective_operator(b,sx)
    n=length(b);work=ResponseWorkspace(matrixfree;krylovdim=n)

    modes=liouvillian_modes(matrixfree;k=3,method=:arnoldi,
        krylovdim=n,rng=MersenneTwister(501))
    dense_modes=liouvillian_modes(dense;k=3)
    @test modes.values≈dense_modes.values atol=2e-12
    @test maximum(modes.residuals)<1e-11
    @test modes.partial_scope
    small_basis=PIBasis(1,2)
    small_dense=Matrix(liouvillian(PIModel(small_basis,[LocalJump(sm)]);
        representation=:sparse))
    @test length(liouvillian_modes(small_dense).values)==length(small_basis)

    z=2+0.3im
    estimate=resolvent_norm(matrixfree,z;workspace=work,
        rng=MersenneTwister(502),max_power_iterations=100,return_info=true)
    @test estimate.value≈resolvent_norm(dense,z) rtol=2e-8
    @test_throws ArgumentError resolvent_norm(dense,Inf)
    @test estimate.converged&&estimate.forward_solve.converged&&
        estimate.adjoint_solve.converged
    @test estimate.workspace_reused
    singular_failed=try
        resolvent_norm(matrixfree,0.0;workspace=work,
            rng=MersenneTwister(503),max_power_iterations=4)
        false
    catch
        true
    end
    @test singular_failed

    evolved=adjoint_evolve(matrixfree,observable,0.2;
        workspace=work,return_info=true)
    @test evolved.value.data≈adjoint(exp(0.2dense))*observable.data atol=2e-11
    @test evolved.converged&&evolved.workspace_reused
    correlation=integrated_correlation_time(matrixfree,rho,observable;
        workspace=work,return_info=true)
    @test correlation.value≈integrated_correlation_time(
        dense,rho,observable) atol=3e-10
    @test correlation.residual<2e-10&&correlation.trace_error<2e-10

    perturbation=PIModel(b,[LocalHamiltonian(sx)])
    derivative_matrixfree=liouvillian(perturbation;representation=:matrixfree)
    derivative_dense=Matrix(liouvillian(perturbation;representation=:sparse))
    tangent=steady_state_susceptibility(matrixfree,rho,
        derivative_matrixfree;workspace=work,return_info=true)
    tangent_dense=steady_state_susceptibility(dense,rho,derivative_dense)
    @test tangent.state.data≈tangent_dense.data atol=3e-10
    @test tangent.residual<2e-10&&tangent.trace_error<2e-10

    evolution_only=ResponseWorkspace(matrixfree;mode=:evolution)
    linear_only=ResponseWorkspace(matrixfree;mode=:linear,krylovdim=n)
    @test_throws ArgumentError resolvent_norm(matrixfree,z;
        workspace=evolution_only)
    @test_throws ArgumentError adjoint_evolve(matrixfree,observable,0.1;
        workspace=linear_only)
    different=liouvillian(PIModel(b,[LocalJump(sm;rate=2)]);
        representation=:matrixfree)
    @test_throws ArgumentError resolvent_norm(different,z;workspace=work)
    broken_forward=KrylovWorkspace(ComplexF64,n+1,3)
    broken=ResponseWorkspace(work.source,broken_forward,work.adjoint,
        work.exponential,work.x,work.y,work.z,work.rhs,
        work.action_workspace,work.mode)
    @test_throws DimensionMismatch resolvent_norm(matrixfree,z;
        workspace=broken)

    b32=PIBasis(1,2);sm32=ComplexF32.(sm)
    source32=liouvillian(PIModel(b32,[LocalJump(sm32)]);
        representation=:matrixfree)
    work32=ResponseWorkspace(source32;krylovdim=length(b32),mode=:linear)
    estimate32=resolvent_norm(source32,ComplexF32(2+0.3im);
        workspace=work32,rng=MersenneTwister(504),atol=1f-5,rtol=1f-4,
        power_atol=1f-5,power_rtol=1f-4,max_power_iterations=100)
    @test estimate32 isa Float32
    @test_throws ArgumentError resolvent_norm(source32,ComplexF32(2+0.3im);
        workspace=work32,atol=1e-5)
end

@testset "SLD compatibility and PPT diagnostics" begin
    sx=ComplexF64[0 1;1 0]/2;sy=ComplexF64[0 -im;im 0]/2
    b=PIBasis(3,2);rho=iid_pure_state(b,ComplexF64[1,0]);L=symmetric_logarithmic_derivatives(rho,[sx,sy]);F=qfim(rho,[sx,sy])
    for i in 1:2,j in 1:2
        z=sum(Float64(symmetric_group_dimension(p))*LinearAlgebra.tr(physical_block(rho,p)*physical_block(L[i],p)*physical_block(L[j],p)) for p in b.sectors)
        @test real(z)≈F[i,j] atol=3e-10
    end
    C=sld_commutator_matrix(rho,[sx,sy]);@test C≈-C' atol=2e-12;@test !multiparameter_compatible(rho,[sx,sy])
    @test mean_uhlmann_curvature(rho,[sx,sy])≈C/2
    @test qfi_entanglement_depth(rho,sx)==1
    sym=Partition((3,0));gs=b.patterns[b.index[sym]];v=zeros(ComplexF64,4);v[1]=v[end]=inv(sqrt(2));ghz=sector_density_matrix(b,sym,v*v')
    spec=partial_transpose_spectrum(ghz,1)
    neg=sum(Float64(x.multiplicity)*sum(y->max(0,-y),x.eigenvalues) for x in spec)
    @test neg≈negativity(ghz,1) atol=3e-10
    @test minimum_partial_transpose_eigenvalue(ghz,1)<0
    @test bipartition_negativities(ghz)≈fill(0.5,1) atol=3e-10
    @test qfi_entanglement_depth(ghz,ComplexF64[1 0;0 -1]/2)==3
    tangents=PIState[]
    for G in (sx,sy)
        A=collective_operator(b,G);d=PIState(b;T=Float64)
        for p in b.sectors
            R=Matrix(physical_block(rho,p));H=Matrix(physical_block(A,p));coefficient_block(d,p).=sqrt(Float64(symmetric_group_dimension(p)))*(-im*(H*R-R*H))
        end
        push!(tangents,d)
    end
    @test qfim_from_derivatives(rho,tangents)≈qfim(rho,[sx,sy]) atol=3e-10
end

@testset "Liouvillian response and sensitivities" begin
    b=PIBasis(2,2);sx=ComplexF64[0 1;1 0];rho=iid_pure_state(b,ComplexF64[1,0]);A=collective_operator(b,sx)
    L=liouvillian(PIModel(b,[LocalHamiltonian(sx)]);representation=:sparse)
    modes=liouvillian_modes(L;k=3);@test length(modes.values)==3
    @test length(observable_decay_modes(L,A;k=3).overlaps)==3
    @test resolvent_norm(L,10im)>0
    @test pseudospectral_abscissa(L,0.1;real_grid=-2:1:0,imag_grid=[0.3])<=0
    At=adjoint_evolve(L,A,0.2);rt=PIState(b,exp(0.2Matrix(L))*rho.data)
    @test expectation(rho,At)≈expectation(rt,A) atol=2e-10
    D=Matrix(L);prob=sensitivity_problem(L,rho,(0.0,1.0),[D]);du=similar(prob.u0);prob.f(du,prob.u0,nothing,0.0)
    @test du[:,1]≈L*rho.data
    @test du[:,2]≈D*rho.data

    # The augmented state and every tangent column share one genuine
    # matrix-RHS generator call. A vector fallback would increment
    # `vector_calls` once for every column instead.
    vector_calls=Ref(0);batch_calls=Ref(0)
    action! = (destination,source,time,parameters)->begin
        vector_calls[]+=1
        mul!(destination,L,source)
    end
    batch_action! = (destination,source,time,parameters)->begin
        batch_calls[]+=1
        mul!(destination,L,source)
    end
    counted=MatrixFreeLiouvillian(
        length(b),action!,ComplexF64,identity_operator(b).data;
        batched_action! = batch_action!)
    batch_problem=sensitivity_problem(
        counted,rho,(0.0,1.0),(D,2D))
    batch_du=similar(batch_problem.u0)
    batch_problem.f(
        batch_du,batch_problem.u0,batch_problem.p,0.0)
    @test batch_calls[]==1
    @test vector_calls[]==0
    @test batch_du[:,1]≈L*rho.data
    @test batch_du[:,2]≈D*rho.data
    @test batch_du[:,3]≈2D*rho.data

    id=identity_operator(b);zero_tangent=PIState(b,zeros(ComplexF64,length(b)))
    @test classical_fisher_information(rho,[zero_tangent],[id])≈zeros(1,1)
    sm=ComplexF64[0 1;0 0];Ld=liouvillian(PIModel(b,[LocalJump(sm)]);representation=:sparse)
    @test integrated_correlation_time(Ld,rho,A)≈2 atol=3e-10
    tangent=steady_state_susceptibility(Ld,rho,zero(Matrix(Ld)));@test abs(trace(tangent))<2e-10
end
