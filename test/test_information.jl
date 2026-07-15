@testset "PI entropy and state distinguishability" begin
    for d in (2,3),N in 1:4
        b=PIBasis(N,d);mm=maximally_mixed_state(b);pure=iid_pure_state(b,ComplexF64[1;zeros(d-1)])
        @test von_neumann_entropy(mm)≈N*log2(d) atol=2e-10
        @test renyi_entropy(mm,2)≈N*log2(d) atol=2e-10
        @test renyi_entropy(mm,Inf)≈N*log2(d) atol=2e-10
        @test von_neumann_entropy(pure)≈0 atol=2e-11
        @test reduced_entropy(mm,min(1,N))≈log2(d) atol=3e-9
        @test mutual_information(mm,fld(N,2))≈0 atol=5e-9
        @test trace_distance(mm,mm)≈0 atol=2e-12
        @test fidelity(mm,mm)≈1 atol=2e-10
        @test bures_distance(mm,mm)≈0 atol=2e-8
        @test quantum_relative_entropy(mm,mm)≈0 atol=2e-10
        @test hilbert_schmidt_distance(mm,mm)==0
    end
    b=PIBasis(3,2);up=iid_pure_state(b,ComplexF64[1,0]);down=iid_pure_state(b,ComplexF64[0,1])
    @test trace_distance(up,down)≈1 atol=2e-12
    @test fidelity(up,down)≈0 atol=2e-12
    @test bures_distance(up,down)≈sqrt(2) atol=2e-12
    @test quantum_relative_entropy(up,down)==Inf
    sym=Partition((3,0));gs=b.patterns[b.index[sym]];v=zeros(ComplexF64,4)
    v[findfirst(g->content(g)==(3,0),gs)]=inv(sqrt(2));v[findfirst(g->content(g)==(0,3),gs)]=inv(sqrt(2))
    ghz=sector_density_matrix(b,sym,v*v')
    @test reduced_entropy(ghz,1)≈1 atol=2e-10
    @test mutual_information(ghz,1)≈2 atol=2e-10
    @test conditional_entropy(ghz,1)≈-1 atol=2e-10
end

@testset "rank-deficient fidelity and relative-entropy support" begin
    basis=PIBasis(1,3)
    sector=only(basis.sectors)
    u=ComplexF64[1,1,0]/sqrt(2)
    v=ComplexF64[1,-1,0]/sqrt(2)
    sigma=sector_density_matrix(basis,sector,u*u')

    # This weight lies below the former per-eigenvalue/per-overlap cutoff but
    # above the sector-level projector roundoff floor.  Rotating the support
    # makes the test independent of the computational basis.
    delta=1.5e-14
    psi=sqrt(1-delta)*u+sqrt(delta)*v
    rotated_pure=sector_density_matrix(basis,sector,psi*psi')
    rotated_mixed=sector_density_matrix(
        basis,sector,(1-delta)*(u*u')+delta*(v*v'))
    @test quantum_relative_entropy(rotated_pure,sigma)==Inf
    @test quantum_relative_entropy(rotated_mixed,sigma)==Inf
    @test relative_entropy_decomposition(rotated_pure,sigma).total==Inf
    @test relative_entropy_decomposition(rotated_mixed,sigma).total==Inf

    # Conversely, a small but numerically resolved sigma eigenvalue is true
    # support and must not be replaced by a numerical zero.
    epsilon=1e-14
    tiny_support=sector_density_matrix(
        basis,sector,(1-epsilon)*(u*u')+epsilon*(v*v'))
    on_tiny_support=sector_density_matrix(basis,sector,v*v')
    expected=-log2(epsilon)
    @test quantum_relative_entropy(on_tiny_support,tiny_support)≈expected rtol=2e-3
    @test relative_entropy_decomposition(
        on_tiny_support,tiny_support).total≈expected rtol=2e-3

    @test fidelity(tiny_support,tiny_support)≈1 atol=2e-12
    @test bures_distance(tiny_support,tiny_support)≈0 atol=2e-7
    PID=PermutationalInvariantDynamics
    @test PID._unit_interval_roundoff(
        1+eps(Float64),0.0,0.0;context="test fidelity")==1.0
    @test_throws ArgumentError PID._unit_interval_roundoff(
        1+1e-8,0.0,0.0;context="test fidelity")
end


@testset "symmetry- and sector-resolved information" begin
    sx=ComplexF64[0 1;1 0]/2;sz=ComplexF64[1 0;0 -1]/2
    for d in (2,3),N in 2:4
        b=PIBasis(N,d);rho=maximally_mixed_state(b);dec=entropy_decomposition(rho)
        @test sum(x->x.probability,dec.sectors)≈1 atol=2e-11
        @test dec.total≈von_neumann_entropy(rho) atol=2e-10
        @test dec.classical+dec.intra_sector+dec.multiplicity≈dec.total
        @test relative_entropy_of_coherence(rho)≈0 atol=2e-11
        rd=relative_entropy_decomposition(rho,rho)
        @test rd.total≈quantum_relative_entropy(rho,rho) atol=2e-10
    end
    b=PIBasis(4,2);plus=iid_pure_state(b,ComplexF64[1,1]/sqrt(2));up=iid_pure_state(b,ComplexF64[1,0])
    sr=sector_resolved_qfi(plus,sz)
    @test sum(x->x.contribution,sr)≈qfi(plus,sz) atol=3e-10
    @test relative_entropy_of_asymmetry(plus,sz)≈von_neumann_entropy(symmetry_twirl(plus,sz)) atol=3e-10
    @test relative_entropy_of_symmetry(plus,sz)≈relative_entropy_of_asymmetry(plus,sz)
    @test relative_entropy_of_asymmetry(up,sz)≈0 atol=2e-11
    @test wigner_yanase_asymmetry(plus,sz)≈collective_variance(plus,sz) atol=3e-10
    @test relative_entropy_of_coherence(plus)≈relative_entropy_of_asymmetry(plus,sz) atol=3e-10
    zero_tangent=PIState(b,zeros(ComplexF64,length(b)))
    fd=qfim_sector_decomposition(plus,[zero_tangent])
    @test fd.total≈fd.classical+fd.intra_sector
    bm=PIBasis(2,2);mm=maximally_mixed_state(bm);dm=PIState(bm;T=Float64)
    for (p,dp) in zip(bm.sectors,(0.1,-0.1))
        n=length(bm.patterns[bm.index[p]]);f=Float64(symmetric_group_dimension(p))
        coefficient_block(dm,p).=sqrt(f)*(dp/(f*n))*Matrix{ComplexF64}(I,n,n)
    end
    fdm=qfim_sector_decomposition(mm,[dm])
    @test fdm.intra_sector≈zeros(1,1) atol=2e-12
    @test fdm.classical[1,1]≈0.1^2/(3/4)+0.1^2/(1/4) atol=2e-12

    b2=PIBasis(2,2);p=Partition((2,0));gs=b2.patterns[b2.index[p]]
    v=zeros(ComplexF64,3);v[findfirst(g->content(g)==(1,1),gs)]=1
    dicke=sector_density_matrix(b2,p,v*v')
    nr=number_resolved_negativity(dicke,1)
    @test sum(x->x.negativity,nr)≈negativity(dicke,1) atol=3e-10
    @test sum(x->x.weight,nr)≈1 atol=3e-10
    ghzv=zeros(ComplexF64,3);ghzv[1]=ghzv[end]=inv(sqrt(2));ghz=sector_density_matrix(b2,p,ghzv*ghzv')
    @test_throws ArgumentError number_resolved_negativity(ghz,1)

    b3=PIBasis(2,3);p3=Partition((2,0,0));g3=b3.patterns[b3.index[p3]]
    v3=zeros(ComplexF64,length(g3));v3[findfirst(g->content(g)==(1,1,0),g3)]=1
    qdicke=sector_density_matrix(b3,p3,v3*v3');q=Diagonal([0,1,2])
    qr=charge_resolved_negativity(qdicke,1,q)
    @test sum(x->x.negativity,qr)≈negativity(qdicke,1) atol=2e-9
end
