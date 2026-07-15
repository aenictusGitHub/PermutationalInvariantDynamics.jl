@testset "unitary and antiunitary Liouvillian symmetries" begin
    sx=ComplexF64[0 1;1 0];sy=ComplexF64[0 -im;im 0];sz=ComplexF64[1 0;0 -1];sm=ComplexF64[0 1;0 0]
    b=PIBasis(3,2);decay=PIModel(b,[LocalJump(sm)])
    uz=check_liouvillian_symmetry(decay,sz;kind=:unitary)
    @test uz.symmetric
    @test uz.representation==:pi_liouville
    @test uz.residual<=uz.tolerance
    @test is_liouvillian_symmetric(decay,sz)
    @test !is_liouvillian_symmetric(PIModel(b,[LocalHamiltonian(sx)]),sz)

    real_model=PIModel(b,[LocalJump(sm),LocalHamiltonian(sx)])
    @test check_liouvillian_symmetry(real_model,Matrix{ComplexF64}(I,2,2);kind=:antiunitary).symmetric==false
    dissipative=PIModel(b,[LocalJump(sm)])
    @test is_liouvillian_symmetric(dissipative,Matrix{ComplexF64}(I,2,2);kind=:antiunitary)

    usual=usual_liouvillian_symmetries(decay)
    @test haskey(usual.unitary,:parity_z)
    @test usual.unitary[:parity_z].symmetric
    @test haskey(usual.antiunitary,:spin_time_reversal)

    # Matrix-free validation is reproducible by default, while an assembled
    # matrix is checked by the full commutator rather than random probes.
    Lm=liouvillian(decay;representation=:matrixfree)
    P=matrixfree_symmetry_projector(b,sz;charge=1)
    Pwork=PermutationalInvariantDynamics.SymmetryProjectorWorkspace(P)
    xp=randn(MersenneTwister(41),ComplexF64,length(b));yp=similar(xp)
    PermutationalInvariantDynamics.apply!(yp,P,xp,Pwork)
    @test (@allocated PermutationalInvariantDynamics.apply!(yp,P,xp,Pwork))==0
    mul!(yp,P,xp)
    @test (@allocated mul!(yp,P,xp))<=128
    probe1=PermutationalInvariantDynamics._projected_symmetry_residual(Lm,P)
    probe2=PermutationalInvariantDynamics._projected_symmetry_residual(Lm,P)
    @test probe1==probe2
    @test probe1.validation===:probed
    @test probe1.symmetric
    exact=PermutationalInvariantDynamics._projected_symmetry_residual(
        liouvillian(decay;representation=:sparse),P)
    @test exact.validation===:exact
    @test exact.probes==0
    @test exact.symmetric

    if Threads.nthreads()>=4
        xs=[randn(MersenneTwister(100+i),ComplexF64,length(b)) for i in 1:4]
        refs=[P*x for x in xs]
        shared=[similar(x) for x in xs]
        Threads.@threads for i in 1:4
            mul!(shared[i],P,xs[i])
        end
        @test all(isapprox(shared[i],refs[i];atol=2e-12) for i in 1:4)

        separate=[similar(x) for x in xs]
        works=[PermutationalInvariantDynamics.SymmetryProjectorWorkspace(P) for _ in 1:4]
        Threads.@threads for i in 1:4
            PermutationalInvariantDynamics.apply!(separate[i],P,xs[i],works[i])
        end
        @test all(isapprox(separate[i],refs[i];atol=2e-12) for i in 1:4)
    else
        @test true # exercised under the dedicated four-thread test command
    end

    # A degenerate unitary eigenspace must still produce an orthogonal
    # conjugation projector.  Complex Schur vectors prevent the nonorthogonal
    # degenerate eigenvectors allowed by a generic eigensolver.
    b3=PIBasis(1,3);rng=MersenneTwister(31)
    Q=Matrix(qr(randn(rng,ComplexF64,3,3)).Q)
    Udeg=Q*Diagonal(ComplexF64[1,1,-1])*Q'
    Pdeg=matrixfree_symmetry_projector(b3,Udeg;charge=1)
    n=length(b3);Ipi=Matrix{ComplexF64}(I,n,n)
    Pmat=hcat((Pdeg*view(Ipi,:,j) for j in 1:n)...)
    @test Pmat≈Pmat' atol=3e-11
    @test Pmat*Pmat≈Pmat atol=3e-11
    @test real(tr(Pmat))≈5 atol=3e-11

    # Full one-qubit Liouville-space path agrees with the PI path.
    L=liouvillian(PIModel(PIBasis(1,2),[LocalJump(sm)]);representation=:sparse)
    @test check_liouvillian_symmetry(L,sz;kind=:unitary).symmetric
    @test_throws ArgumentError check_liouvillian_symmetry(L,2sz)
    @test_throws ArgumentError check_liouvillian_symmetry(L,sz;kind=:other)
end
