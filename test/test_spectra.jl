@testset "PI Liouvillian and density-operator spectra" begin
    b=PIBasis(3,2);rho=maximally_mixed_state(b)
    spec=pi_density_spectrum(rho)
    @test spec.total_dimension==2^3
    @test sum(spec.degeneracies)==8
    @test spec.trace≈1 atol=2e-15
    @test all(x->x≈1/8,spec.values)
    @test pi_density_operator_spectrum(rho).values==spec.values
    @test density_operator_spectrum(rho;expanded=true)≈fill(1/8,8)
    @test_throws ArgumentError pi_density_spectrum(rho;expanded=true,max_expanded_dimension=7)

    pure=iid_pure_state(b,ComplexF64[1,0]);ps=pi_density_spectrum(pure)
    @test count(>(1-1e-12),ps.values)==1
    @test sum(ps.degeneracies[i] for i in eachindex(ps.values) if abs(ps.values[i])<1e-12)==7

    sm=ComplexF64[0 1;0 0];model=PIModel(b,[LocalJump(sm)])
    L=liouvillian(model;representation=:sparse)
    vals=pi_liouvillian_spectrum(L)
    @test length(vals)==length(b)
    @test minimum(abs,vals)<2e-12
    @test vals≈liouvillian_eigenvalues(L,length(b)) atol=2e-12
    full=pi_liouvillian_spectrum(model;vectors=true)
    @test full.values≈vals atol=2e-12
    @test size(full.vectors)==(length(b),length(b))
    @test norm(Matrix(L)*full.vectors-full.vectors*Diagonal(full.values))<2e-10
    Lm=liouvillian(model;representation=:matrixfree)
    @test pi_liouvillian_spectrum(Lm)≈vals atol=2e-12

    gap=pi_liouvillian_gap(model;return_info=true)
    @test gap.gap>0
    @test gap.gap≈liouvillian_gap(L) atol=2e-12
    @test gap.stationary_multiplicity==1
    @test gap.unique_stationary_mode
    @test gap.stable
    @test gap.decay_eigenvalue!==nothing

    sz=ComplexF64[1 0;0 -1]
    symmetric_gap=pi_liouvillian_gap(model;symmetry=sz,return_info=true)
    @test symmetric_gap.symmetry_used
    @test symmetric_gap.gap≈gap.gap atol=2e-11
    @test length(symmetric_gap.symmetry_sectors)>1
    @test sum(s.dimension for s in symmetric_gap.symmetry_sectors)==length(b)
    @test sum(s.stationary_multiplicity for s in symmetric_gap.symmetry_sectors)==gap.stationary_multiplicity
    automatic=pi_liouvillian_gap(model;symmetry=:auto,return_info=true)
    @test automatic.symmetry_used
    @test automatic.gap≈gap.gap atol=2e-11
    @test pi_liouvillian_gap(L;symmetry=sz,basis=b)≈gap.gap atol=2e-11
    @test_throws ArgumentError pi_liouvillian_gap(PIModel(b,[LocalHamiltonian(ComplexF64[0 1;1 0])]);symmetry=sz)
    @test_throws ArgumentError pi_liouvillian_gap(model;symmetry=sz,symmetry_kind=:antiunitary)

    degenerate=Diagonal(ComplexF64[0,0,-0.25,-1])
    dg=pi_liouvillian_gap(degenerate;return_info=true)
    @test dg.gap≈0.25
    @test dg.stationary_multiplicity==2
    @test !dg.unique_stationary_mode
    @test pi_liouvillian_gap(zeros(ComplexF64,1,1))==Inf
    @test_throws ArgumentError pi_liouvillian_gap(Diagonal(ComplexF64[0,0.1]))

    driven=PIModel(b,[LocalJump(sm;rate=(t,p)->1+t)])
    @test_throws ArgumentError pi_liouvillian_spectrum(driven)
    @test_throws ArgumentError pi_liouvillian_gap(driven)
    frozen=PermutationalInvariantDynamics.freeze(driven;time=0.25)
    @test minimum(abs,pi_liouvillian_spectrum(frozen))<2e-12
end
