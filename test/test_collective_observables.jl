@testset "direct collective expectations and variances" begin
    for (N,d) in ((5,2),(3,3))
        b=PIBasis(N,d);rho=maximally_mixed_state(b)
        X=reshape(ComplexF64.(1:d^2),d,d);X=(X+X')/2
        A=collective_operator(b,X)
        m=collective_moments(rho,X)
        @test m.mean≈expectation(rho,A) atol=2e-10
        @test m.second_moment≈expectation(rho,A*A) atol=2e-10
        @test collective_expectation(rho,X)≈expectation(rho,A) atol=2e-10
        @test collective_variance(rho,X)≈variance(rho,A) atol=2e-10
    end
    sx=ComplexF64[0 1;1 0];sz=ComplexF64[1 0;0 -1]
    for N in 1:10
        b=PIBasis(N,2);rho=iid_pure_state(b,ComplexF64[1,0])
        @test collective_expectation(rho,0.5sz)≈N/2 atol=2e-10
        @test collective_variance(rho,0.5sz)≈0 atol=2e-10
        @test collective_variance(rho,0.5sx)≈N/4 atol=2e-10
    end
    b=PIBasis(2,2);rho=maximally_mixed_state(b)
    cache=OneBodyGeometry(b)
    @test collective_moments(rho,sx;cache=cache)==collective_moments(rho,sx)
    @test collective_covariance_matrix(rho,[sx,sz];cache=cache)≈collective_covariance_matrix(rho,[sx,sz])
    @test_throws DimensionMismatch collective_expectation(rho,zeros(3,3))
    @test_throws ArgumentError collective_variance(rho,ComplexF64[0 1;0 0])
end
