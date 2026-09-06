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

@testset "entropy spectra retain state validation" begin
    PID=PermutationalInvariantDynamics
    for T in (Float32,Float64)
        basis=PIBasis(8,2)
        rho=iid_state(basis,T[0.75 0;0 0.25])
        expected=-8*(T(0.75)*log2(T(0.75))+T(0.25)*log2(T(0.25)))
        @test von_neumann_entropy(rho)≈expected rtol=20eps(T)
        @test von_neumann_entropy(rho) isa T
        original=copy(rho.data)
        @test renyi_entropy(rho,1)===von_neumann_entropy(rho)
        @test rho.data==original
        rho.data.*=2
        @test_throws ArgumentError von_neumann_entropy(rho)
        @test_throws ArgumentError renyi_entropy(rho,2)
        @test_throws ArgumentError renyi_entropy(rho,Inf)
    end

    basis=PIBasis(8,2)
    rho=PIState(basis)
    coefficient_block(rho,first(basis.sectors))[1,1]=1
    sector=Partition((5,3))
    block=coefficient_block(rho,sector)
    atol=1e-10
    # This negative block passes the coefficient-space absolute tolerance,
    # but must fail the stricter multiplicity-weighted entropy check.
    block[1,1]=-atol/2
    block[2,2]=atol/2
    @test validate_state(rho;atol,rtol=0)===rho
    original=copy(rho.data)
    @test_throws ArgumentError von_neumann_entropy(rho;atol,rtol=0)
    @test_throws ArgumentError renyi_entropy(rho,2;atol,rtol=0)
    @test_throws ArgumentError renyi_entropy(rho,Inf;atol,rtol=0)
    @test rho.data==original
    fill!(block,0)
    @test von_neumann_entropy(rho;atol,rtol=0)≈0 atol=1e-12
    block[1,2]=1e-3
    @test_throws ArgumentError von_neumann_entropy(rho;atol,rtol=0)
    block[1,2]=NaN
    @test_throws ArgumentError renyi_entropy(rho,2;atol,rtol=0)

    # Preserve the original two-stage validation decision near roundoff,
    # including strict zero tolerances, where rescaling can change eigvals.
    v=ComplexF64[1,2im,1+im]
    for scale in (1e-3,1e-12),tolerance in (0.0,1e-20,1e-14)
        fill!(rho.data,0)
        block.=scale*(v*v')
        coefficient_block(rho,first(basis.sectors))[1,1]=
            1-sqrt(Float64(symmetric_group_dimension(sector)))*real(tr(block))
        accepted=try
            validate_state(rho;atol=tolerance,rtol=0)
            for p in basis.sectors
                PID._weighted_sector_eigvals(rho,p;atol=tolerance,rtol=0)
            end
            true
        catch error
            error isa ArgumentError||rethrow()
            false
        end
        for entropy in (von_neumann_entropy,
                        state->renyi_entropy(state,2;atol=tolerance,rtol=0),
                        state->renyi_entropy(state,Inf;atol=tolerance,rtol=0))
            current=try
                entropy===von_neumann_entropy ?
                    entropy(rho;atol=tolerance,rtol=0) : entropy(rho)
                true
            catch error
                error isa ArgumentError||rethrow()
                false
            end
            @test current==accepted
        end
    end

    # The automatic large-block Cholesky check uses entry-norm scaling.
    # Its negative direction must not be accepted by substituting the looser
    # spectral relative tolerance. This is a bounded, 257-dimensional Schur
    # block, not a reconstruction of the full 2^256 Hilbert space.
    n=257
    basis=PIBasis(n-1,2;sectors=[(n-1,0)])
    u=fill(inv(sqrt(n)),n)
    v=zeros(n);v[1]=inv(sqrt(2));v[2]=-v[1]
    delta=1e-6
    density=(1+delta)*(u*u')-delta*(v*v')
    rho=PIState(basis,ComplexF64.(vec(density)))
    @test PID._automatic_positivity_method(rho)===:cholesky
    @test minimum(PID._weighted_sector_eigvals(
        rho,only(basis.sectors);atol=1e-12,rtol=1e-4).values)<0
    @test_throws ArgumentError validate_state(rho;atol=1e-12,rtol=1e-4)
    @test_throws ArgumentError von_neumann_entropy(rho;atol=1e-12,rtol=1e-4)
    @test_throws ArgumentError renyi_entropy(rho,2;atol=1e-12,rtol=1e-4)
    @test_throws ArgumentError renyi_entropy(rho,Inf;atol=1e-12,rtol=1e-4)
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

@testset "entropy controls remain in-domain at state precision" begin
    rho=iid_state(PIBasis(1,2),Float32[0.7 0;0 0.3])
    for alpha in (1+1e-8,1-1e-8,1e-50,1e50)
        @test_throws ArgumentError renyi_entropy(rho,alpha)
    end
    for base in (1+1e-8,1-1e-8,1e-50,1e50,Inf,NaN)
        @test_throws ArgumentError von_neumann_entropy(rho;base)
        @test_throws ArgumentError renyi_entropy(rho,2;base)
        @test_throws ArgumentError renyi_entropy(rho,Inf;base)
    end
    @test renyi_entropy(rho,1)===von_neumann_entropy(rho)
    @test renyi_entropy(rho,2) isa Float32
    @test renyi_entropy(rho,Inf) isa Float32
    # A wider, exactly normalized state supports the original near-one request.
    wide=iid_state(PIBasis(1,2),[0.7 0;0 0.3])
    entropy=-(0.7log(0.7)+0.3log(0.3))
    @test renyi_entropy(wide,1+1e-8)≈entropy/log(2) rtol=1e-7
    @test von_neumann_entropy(wide;base=1+1e-8)≈entropy/log(1+1e-8)
end
