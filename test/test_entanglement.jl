@testset "symmetric-sector bipartite negativity" begin
    for N in 2:7
        b=PIBasis(N,2); sym=Partition((N,0)); gs=b.patterns[b.index[sym]]
        # Product state has zero negativity.
        product=iid_pure_state(b,ComplexF64[1,0])
        @test negativity(product,1)<2e-14
        @test negativity(product,0)==0
        # GHZ has two equal Schmidt coefficients across every nontrivial cut.
        v=zeros(ComplexF64,length(gs))
        v[findfirst(g->content(g)==(N,0),gs)]=inv(sqrt(2))
        v[findfirst(g->content(g)==(0,N),gs)]=inv(sqrt(2))
        ghz=sector_density_matrix(b,sym,v*v')
        for k in 1:N-1
            @test negativity(ghz,k) ≈ 0.5 atol=5e-13
            @test logarithmic_negativity(ghz,k) ≈ 1 atol=5e-13
        end
        # W_N Schmidt coefficients are sqrt(k/N), sqrt((N-k)/N).
        w=zeros(ComplexF64,length(gs))
        w[findfirst(g->content(g)==(N-1,1),gs)]=1
        wrho=sector_density_matrix(b,sym,w*w')
        for k in 1:N-1
            @test negativity(wrho,k) ≈ sqrt(k*(N-k))/N atol=5e-13
        end
    end
    # General PI sectors use the SU(2) subduction path rather than the
    # fully-symmetric occupation shortcut.
    b=PIBasis(2,2); mixed=maximally_mixed_state(b)
    @test negativity(mixed,1) ≈ 0 atol=2e-14
    singlet=basis_state(b,Partition((1,1)),b.patterns[b.index[Partition((1,1))]][1])
    @test negativity(singlet,1) ≈ 0.5 atol=5e-13
    @test_throws ArgumentError negativity(iid_pure_state(b,ComplexF64[1,0]),3)
end

@testset "large-spin SU(2) recoupling stability" begin
    PID=PermutationalInvariantDynamics

    # The hybrid boundary retains the allocation-light Racah sum through
    # doubled spin 32 and switches immediately afterwards.  Check complete
    # coupled-column orthogonality on both sides, not only isolated values.
    for J in (32,34)
        total_spins=(J-2,J,J+2)
        columns=hcat(([PID._su2_cgc(J,m,J,-m,total_j,0)
                       for m in -J:2:J] for total_j in total_spins)...)
        @test adjoint(columns)*columns≈Matrix{Float64}(I,3,3) atol=2e-14
    end
    @test PID._su2_cgc(32,0,32,0,32,0)==
          PID._su2_cgc_small(32,0,32,0,32,0)
    @test PID._su2_cgc(34,0,34,0,34,0)==
          PID._su2_cgc_exact(34,0,34,0,34,0)

    # Coupling two equal extremal spins to the singlet has the closed form
    # <J,J;J,-J|0,0> = 1/sqrt(J+1), with all labels doubled here.  The former
    # factorial-to-Float64 path overflowed to Inf at J=99.
    for J in (65,99,100,200,500,1000)
        value=PID._su2_cgc(J,J,J,-J,0,0)
        @test value isa Float64
        @test isfinite(value)
        @test value≈inv(sqrt(J+1)) atol=2eps(Float64)
    end

    # A nontrivial complete coupled column remains normalized after an
    # alternating Racah sum with large factorial arguments.
    J=120;total_j=120;total_m=0
    column=[PID._su2_cgc(J,m1,J,total_m-m1,total_j,total_m)
            for m1 in -J:2:J]
    @test sum(abs2,column)≈1 atol=2e-14
    @test PID._su2_cgc(J,J,J,J,2J,2J)≈1 atol=2eps(Float64)

    # In the N=1100 singlet sector the physical parent eigenvalue 1/f is below
    # Float64's nonzero range, while the one-body reduced state and the
    # multiplicity-weighted partial-transpose norm are both O(1).  Fusing the
    # exact subsystem/parent multiplicities must preserve those bounded
    # answers without reconstructing the multiplicity space.
    N=1100;singlet=Partition((N÷2,N÷2))
    basis=PIBasis(N,2;sectors=[singlet.parts])
    state=basis_state(basis,singlet,only(only(basis.patterns)))
    plan=ReductionPlan(basis,1)
    @test length(plan.couplings)==1
    reduced=reduced_state(state,1;plan=plan)
    @test Matrix(physical_block(reduced,first(reduced.basis.sectors)))≈
          Matrix{ComplexF64}(I,2,2)/2 atol=3e-14
    @test reduced_purity(state,1;plan=plan)≈0.5 atol=3e-14
    @test negativity(state,1;plan=plan)≈0.5 atol=3e-14
    @test negativity(state,1)≈0.5 atol=3e-14
    # The unweighted per-copy spectrum itself is genuinely unrepresentable in
    # Float64 and therefore raises instead of returning a silent zero.
    @test_throws ArgumentError partial_transpose_spectrum(state,1;plan=plan)
end

@testset "plan-local SU(2) factorial cache" begin
    PID=PermutationalInvariantDynamics
    cache=PID._SU2FactorialCache(401)

    # Cached factorial lookup changes setup work only: both the ordinary
    # Float64 Racah route and the exact-cancellation route remain bit-identical
    # to standalone coefficient evaluation.
    samples=((32,0,32,0,32,0),
             (34,0,34,0,34,0),
             (120,-44,120,44,120,0),
             (200,200,200,-200,0,0))
    for labels in samples
        @test PID._su2_cgc(cache,labels...)===PID._su2_cgc(labels...)
    end

    # Check complete retained plan data, including zero entries and every
    # coupled parent sector, against the uncached construction oracle.
    basis=PIBasis(12,2)
    cached=PID._qubit_reduction_couplings(basis,6)
    uncached=PID._qubit_reduction_couplings(basis,6,nothing)
    @test [(c.alpha,c.beta,c.da,c.db,c.alpha_multiplicity,
            c.beta_multiplicity,c.product_multiplicity) for c in cached]==
          [(c.alpha,c.beta,c.da,c.db,c.alpha_multiplicity,
            c.beta_multiplicity,c.product_multiplicity) for c in uncached]
    cached_connections=[(sector,U) for c in cached
                        for (sector,intertwiners) in c.intertwiners
                        for U in intertwiners]
    uncached_connections=[(sector,U) for c in uncached
                          for (sector,intertwiners) in c.intertwiners
                          for U in intertwiners]
    @test cached_connections==uncached_connections
end

@testset "reduced density-matrix purity" begin
    for N in 2:6
        b=PIBasis(N,2);sym=Partition((N,0));gs=b.patterns[b.index[sym]]
        product=iid_pure_state(b,ComplexF64[1,0])
        @test reduced_purities(product) ≈ ones(N+1) atol=3e-12
        v=zeros(ComplexF64,length(gs))
        v[findfirst(g->content(g)==(N,0),gs)]=inv(sqrt(2))
        v[findfirst(g->content(g)==(0,N),gs)]=inv(sqrt(2))
        ghz=sector_density_matrix(b,sym,v*v')
        @test reduced_purity(ghz,0)≈1
        @test reduced_purity(ghz,N)≈1
        for k in 1:N-1;@test reduced_purity(ghz,k)≈0.5 atol=3e-12;end
        w=zeros(ComplexF64,length(gs));w[findfirst(g->content(g)==(N-1,1),gs)]=1
        wrho=sector_density_matrix(b,sym,w*w')
        for k in 1:N-1
            @test reduced_purity(wrho,k)≈(k/N)^2+((N-k)/N)^2 atol=3e-12
        end
        mm=maximally_mixed_state(b)
        for k in 0:N;@test reduced_purity(mm,k)≈2.0^(-k) atol=3e-11;end
    end
    b=PIBasis(2,2);singlet=basis_state(b,Partition((1,1)),b.patterns[b.index[Partition((1,1))]][1])
    @test reduced_purity(singlet,1)≈0.5 atol=3e-12
    b3=PIBasis(2,3);anti=Partition((1,1,0));rho=basis_state(b3,anti,b3.patterns[b3.index[anti]][1])
    @test reduced_purity(rho,1)≈0.5 atol=2e-10
    @test reduced_purity(maximally_mixed_state(PIBasis(3,3)),1)≈1/3 atol=5e-9
    @test_throws ArgumentError reduced_purity(rho,3)
end

@testset "reduced-purity range guards" begin
    PID=PermutationalInvariantDynamics
    @test PID._reduced_blocks_purity([zeros(ComplexF64,1,1)],Float64)==0
    @test PID._reduced_blocks_purity([fill(ComplexF64(3e-100),1,1)],Float64)≈
          9e-200 rtol=2eps(Float64)
    @test_throws ArgumentError PID._reduced_blocks_purity(
        [fill(ComplexF64(1e-200),1,1)],Float64)
    @test_throws ArgumentError PID._reduced_blocks_purity(
        [fill(ComplexF64(1e200),1,1)],Float64)
end

@testset "reduced PI states" begin
    for d in (2,3),N in 2:4
        rho=maximally_mixed_state(PIBasis(N,d))
        for k in 0:N
            rk=reduced_state(rho,k)
            @test rk.basis.N==k
            @test rk.basis.d==d
            @test trace(rk)≈1 atol=5e-10
            @test isphysical(rk;atol=5e-10,rtol=5e-10)
            @test purity(rk)≈float(d)^(-k) atol=5e-9
    end

end
    b=PIBasis(5,2);psi=ComplexF64[1,2im];psi./=norm(psi)
    rho=iid_pure_state(b,psi)
    for k in 1:4
        rk=reduced_state(rho,k)
        @test purity(rk)≈1 atol=5e-11
        @test one_body_rdm(rk)≈psi*psi' atol=5e-11
    end
    # Partial traces compose.
    b3=PIBasis(3,3);rho3=maximally_mixed_state(b3)
    @test reduced_state(reduced_state(rho3,2),1).data ≈ reduced_state(rho3,1).data atol=5e-9
end

@testset "general-qudit LR partial transpose" begin
    one=Partition((1,0,0))
    @test littlewood_richardson_coefficient(one,one,Partition((2,0,0)))==1
    @test littlewood_richardson_coefficient(one,one,Partition((1,1,0)))==1
    @test littlewood_richardson_coefficient(one,one,Partition((3,0,0)))==0

    # Sparse equal-weight constraints reproduce the dense reference projector,
    # including a genuine multiplicity-two SU(3) product.
    adjoint_irrep=Partition((2,1,0));shifted_adjoint=Partition((3,2,1))
    @test littlewood_richardson_coefficient(adjoint_irrep,adjoint_irrep,
                                             shifted_adjoint)==2
    sparse_intertwiners=subduction_intertwiners(adjoint_irrep,adjoint_irrep,
                                                 shifted_adjoint;algorithm=:sparse)
    dense_intertwiners=subduction_intertwiners(adjoint_irrep,adjoint_irrep,
                                                shifted_adjoint;algorithm=:dense)
    sparse_projector=sum(T*T' for T in sparse_intertwiners)
    dense_projector=sum(T*T' for T in dense_intertwiners)
    @test sparse_projector≈dense_projector atol=2e-10 rtol=2e-10
    @test all(T->isapprox(T'*T,I;atol=2e-10),sparse_intertwiners)
    @test norm(sparse_intertwiners[1]'*sparse_intertwiners[2])<2e-10
    # For two qutrits, every vector in the antisymmetric irrep has Schmidt
    # coefficients (1/sqrt(2),1/sqrt(2)).
    b=PIBasis(2,3); anti=Partition((1,1,0))
    for g in b.patterns[b.index[anti]]
        rho=basis_state(b,anti,g)
        @test negativity(rho,1) ≈ 0.5 atol=2e-10
    end
    anti_mixed=sector_density_matrix(b,anti,Matrix{ComplexF64}(I,3,3)/3)
    @test negativity(anti_mixed,1) ≈ 1/3 atol=2e-10
    @test negativity(maximally_mixed_state(b),1) ≈ 0 atol=2e-10
    @test negativity(maximally_mixed_state(PIBasis(3,3)),1) ≈ 0 atol=5e-9

    # Sanity check: the general LR engine reproduces the specialized SU(2)
    # implementation for a state spanning all total-spin sectors.
    b2=PIBasis(4,2);rho=PIState(b2;T=Float64)
    for (s,p) in pairs(b2.sectors)
        n=length(b2.patterns[s]);A=reshape(ComplexF64.(1:n^2),n,n)
        coefficient_block(rho,p).=A*A'
    end
    PermutationalInvariantDynamics.normalize!(rho)
    @test PermutationalInvariantDynamics._qudit_negativity(rho,2) ≈ negativity(rho,2) atol=2e-10
end
