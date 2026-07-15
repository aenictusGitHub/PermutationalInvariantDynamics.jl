@testset "collective-observable quantum Fisher information" begin
    sx=ComplexF64[0 1;1 0];sz=ComplexF64[1 0;0 -1]
    for N in 1:8
        b=PIBasis(N,2);rho=iid_pure_state(b,ComplexF64[1,0])
        @test qfi(rho,0.5sx)≈N atol=2e-10
        @test qfi(rho,0.5sz)≈0 atol=2e-11
        @test qfi(maximally_mixed_state(b),0.5sx)≈0 atol=2e-11

        sym=Partition((N,0));gs=b.patterns[b.index[sym]]
        v=zeros(ComplexF64,length(gs))
        v[findfirst(g->content(g)==(N,0),gs)]=inv(sqrt(2))
        v[findfirst(g->content(g)==(0,N),gs)]=inv(sqrt(2))
        ghz=sector_density_matrix(b,sym,v*v')
        @test quantum_fisher_information(ghz,0.5sz)≈N^2 atol=5e-9
        @test qfi(ghz,collective_operator(b,0.5sz))≈N^2 atol=5e-9
    end
    # Collective spin annihilates the two-qubit singlet.
    b=PIBasis(2,2);p=Partition((1,1));singlet=basis_state(b,p,b.patterns[b.index[p]][1])
    @test qfi(singlet,0.5sx)≈0 atol=2e-11
    @test qfi(singlet,0.5sx;cache=OneBodyGeometry(b))≈qfi(singlet,0.5sx) atol=2e-11

    # For pure qutrit tensor powers QFI equals four times the collective variance.
    b3=PIBasis(4,3);psi=normalize(ComplexF64[1,1im,2]);rho=iid_pure_state(b3,psi)
    h=ComplexF64[1 1 0;1 0 2im;0 -2im -1]
    H=collective_operator(b3,h)
    @test qfi(rho,H)≈4variance(rho,H) atol=2e-9
    @test_throws ArgumentError qfi(rho,ComplexF64[0 1 0;0 0 0;0 0 0])
end
