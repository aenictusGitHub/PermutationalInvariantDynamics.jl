@testset "Vectorized superoperators and PI covariance" begin
    A = ComplexF64[1 2im; 3 4-im]
    B = ComplexF64[2-im 1; -im 3]
    X = ComplexF64[1+im 2; -1 3im]
    L = ComplexF64[0 1; 0 0]
    v = vec(X)

    @test left_superoperator(A) * v ≈ vec(A * X)
    @test right_superoperator(A) * v ≈ vec(X * A)
    @test sandwich_superoperator(A, B) * v ≈ vec(A * X * B')
    @test commutator_superoperator(A) * v ≈ vec(-im * (A * X - X * A))
    Q = L' * L
    @test dissipator_superoperator(L) * v ≈ vec(L * X * L' - (Q * X + X * Q) / 2)
    @test issparse(left_superoperator(sparse(A)))
    @test issparse(dissipator_superoperator(sparse(L)))

    @test_throws DimensionMismatch left_superoperator(zeros(2, 3))
    @test_throws DimensionMismatch sandwich_superoperator(zeros(2, 2), zeros(3, 3))

    I2 = Matrix{ComplexF64}(I, 2, 2)
    Z = ComplexF64[1 0; 0 -1]
    sm = ComplexF64[0 1; 0 0]
    collective_Z = kron(I2, Z) + kron(Z, I2)
    site_Z = kron(I2, Z)
    @test is_pi_operator(collective_Z, 2, 2)
    @test is_permutationally_invariant(collective_Z, 2, 2)
    @test !is_pi_operator(site_Z, 2, 2)
    @test is_pi_operator(randn(ComplexF64, 3, 3), 1, 3)
    @test !is_pi_operator(zeros(2, 3), 1, 2)
    @test_throws DimensionMismatch is_pi_operator(zeros(3, 3), 2, 2)

    local_dissipation = dissipator_superoperator(kron(I2, sm)) +
                        dissipator_superoperator(kron(sm, I2))
    one_site_dissipation = dissipator_superoperator(kron(I2, sm))
    collective_dissipation = dissipator_superoperator(kron(I2, sm) + kron(sm, I2))
    @test is_pi_superoperator(local_dissipation, 2, 2)
    @test is_pi_superoperator(collective_dissipation, 2, 2)
    @test !is_pi_superoperator(one_site_dissipation, 2, 2)
    @test is_pi_superoperator(randn(ComplexF64, 9, 9), 1, 3)
    @test_throws DimensionMismatch is_pi_superoperator(zeros(15, 15), 2, 2)

    b = PIBasis(3, 2)
    @test is_pi_operator(collective_operator(b, Z))
    @test is_pi_superoperator(PIModel(b, AbstractPITerm[]))
end
