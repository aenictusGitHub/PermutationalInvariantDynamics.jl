@testset "multiparameter quantum Fisher information matrix" begin
    sx=ComplexF64[0 1;1 0];sy=ComplexF64[0 -im;im 0];sz=ComplexF64[1 0;0 -1]
    for N in 1:8
        b=PIBasis(N,2);rho=iid_pure_state(b,ComplexF64[1,0])
        generators=[0.5sx,0.5sy,0.5sz]
        F=qfim(rho,generators)
        @test F≈Diagonal([N,N,0]) atol=3e-10
        @test F≈F' atol=2e-12
        @test minimum(eigvals(Symmetric(F)))>=-2e-10
        for mu in eachindex(generators)
            @test F[mu,mu]≈qfi(rho,generators[mu]) atol=3e-10
        end
        # Mixed local matrices and already assembled PI operators.
        mixed=[generators[1],collective_operator(b,generators[2])]
        @test qfim(rho,mixed)≈F[1:2,1:2] atol=3e-10
    end
    # Covariance under a linear reparameterization of generators.
    b=PIBasis(4,3);psi=normalize(ComplexF64[1,1im,2]);rho=iid_pure_state(b,psi)
    G1=ComplexF64[1 1 0;1 0 0;0 0 -1]
    G2=ComplexF64[0 -im 0;im 2 1;0 1 0]
    F=qfim(rho,[G1,G2]);A=[2.0 -1.0;0.5 3.0]
    transformed=[A[i,1]*G1+A[i,2]*G2 for i in 1:2]
    @test qfim(rho,transformed)≈A*F*A' atol=2e-8
    @test qfim(maximally_mixed_state(b),[G1,G2])≈zeros(2,2) atol=2e-11
    @test_throws ArgumentError qfim(rho,Matrix{ComplexF64}[])
    @test_throws ArgumentError qfim(rho,[ComplexF64[0 1 0;0 0 0;0 0 0]])
end
