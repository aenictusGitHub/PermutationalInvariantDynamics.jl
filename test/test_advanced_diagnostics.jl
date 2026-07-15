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
    id=identity_operator(b);zero_tangent=PIState(b,zeros(ComplexF64,length(b)))
    @test classical_fisher_information(rho,[zero_tangent],[id])≈zeros(1,1)
    sm=ComplexF64[0 1;0 0];Ld=liouvillian(PIModel(b,[LocalJump(sm)]);representation=:sparse)
    @test integrated_correlation_time(Ld,rho,A)≈2 atol=3e-10
    tangent=steady_state_susceptibility(Ld,rho,zero(Matrix(Ld)));@test abs(trace(tangent))<2e-10
end
