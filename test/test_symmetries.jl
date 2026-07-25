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

    @testset "projector scalar precision" begin
        sx32=ComplexF32[0 1;1 0]
        sz32=ComplexF32[1 0;0 -1]
        sm32=ComplexF32[0 1;0 0]
        P32=matrixfree_symmetry_projector(b,sz32;charge=1)
        work32=PermutationalInvariantDynamics.SymmetryProjectorWorkspace(P32)
        @test eltype(P32)===ComplexF32
        @test P32.charge===one(ComplexF32)
        @test all(eltype(first(pair))===ComplexF32&&
                  eltype(last(pair))===ComplexF32 for pair in work32.work)

        x32=randn(MersenneTwister(73),ComplexF32,length(b))
        y32=P32*x32
        @test eltype(y32)===ComplexF32
        @test P32*y32≈y32 atol=2f-5
        inplace32=copy(x32)
        PermutationalInvariantDynamics.apply!(
            inplace32,P32,inplace32,work32)
        @test inplace32≈y32 atol=2f-5
        overlapping=vcat(x32,zero(ComplexF32))
        overlapping_source=@view overlapping[1:length(b)]
        overlapping_destination=@view overlapping[2:length(b)+1]
        @test_throws ArgumentError PermutationalInvariantDynamics.apply!(
            overlapping_destination,P32,overlapping_source,work32)
        @test_throws ArgumentError P32*ComplexF64.(x32)

        # A floating charge participates in precision selection, while the
        # exact default charge does not widen a Float32 projector.
        @test eltype(matrixfree_symmetry_projector(
            b,sz32;charge=1.0))===ComplexF64

        model32=PIModel(b,(LocalJump(sm32;rate=1f0),))
        L32=liouvillian(model32;representation=:matrixfree)
        residual32=PermutationalInvariantDynamics._projected_symmetry_residual(
            L32,P32)
        @test residual32.symmetric
        @test residual32.residual isa Float32
        @test residual32.tolerance isa Float32

        # A Float32 projector and an explicitly block-diagonal matrix commute
        # up to ordinary Float32 matrix-product roundoff. Both validation
        # routes must account for that working precision.
        residual_basis=PIBasis(1,2)
        residual_rng=MersenneTwister(81)
        residual_unitary=Matrix(qr(randn(
            residual_rng,ComplexF32,2,2)).Q)
        residual_projector=matrixfree_symmetry_projector(
            residual_basis,residual_unitary)
        residual_dimension=length(residual_basis)
        coordinate_identity=Matrix{ComplexF32}(
            I,residual_dimension,residual_dimension)
        projector_matrix=hcat((residual_projector*
            view(coordinate_identity,:,column)
            for column in 1:residual_dimension)...)
        complement=coordinate_identity-projector_matrix
        range_block=randn(
            residual_rng,ComplexF32,residual_dimension,residual_dimension)
        complement_block=randn(
            residual_rng,ComplexF32,residual_dimension,residual_dimension)
        commuting_matrix=projector_matrix*range_block*projector_matrix+
            complement*complement_block*complement
        exact32=PermutationalInvariantDynamics._projected_symmetry_residual(
            commuting_matrix,residual_projector)
        probed32=PermutationalInvariantDynamics._projected_symmetry_residual(
            commuting_matrix,residual_projector;exact=false)
        @test exact32.symmetric
        @test probed32.symmetric
        @test exact32.tolerance isa Float32
        @test probed32.tolerance isa Float32

        joint32=joint_symmetry_projector(b,(sx32=>1,sz32=>1))
        @test eltype(joint32)===ComplexF32
        @test all(eltype(projector)===ComplexF32
                  for projector in joint32.projectors)
        @test eltype(joint32*x32)===ComplexF32

        # Mixed component precision is promoted once at construction, so all
        # component workspaces remain compatible with the joint buffers.
        mixed=joint_symmetry_projector(
            b,(sx32=>1,ComplexF64.(sz32)=>1))
        @test eltype(mixed)===ComplexF64
        @test all(eltype(projector)===ComplexF64
                  for projector in mixed.projectors)
        mixed_work=JointSymmetryProjectorWorkspace(mixed)
        @test eltype(mixed_work.first)===ComplexF64
        @test eltype(mixed_work.second)===ComplexF64

        # Promotion must retain the accuracy floor of Float32 source data.
        # Otherwise a numerically unitary Q32 is revalidated against Float64
        # roundoff and falsely rejected even beside an exact identity.
        approximate32=Matrix(qr(randn(
            MersenneTwister(75),ComplexF32,2,2)).Q)
        identity64=Matrix{ComplexF64}(I,2,2)
        approximate_mixed=joint_symmetry_projector(
            b,(approximate32=>1,identity64=>1))
        @test eltype(approximate_mixed)===ComplexF64
        @test approximate_mixed.range_dimension>0

        block_rng=MersenneTwister(76)
        pi_pairs=Pair[]
        for (sector_index,partition) in pairs(b.sectors)
            dimension=length(b.patterns[sector_index])
            block=Matrix(qr(randn(
                block_rng,ComplexF32,dimension,dimension)).Q)
            push!(pi_pairs,partition=>block)
        end
        approximate_pi32=operator_from_schur_blocks(b,pi_pairs)
        approximate_pi_mixed=joint_symmetry_projector(
            b,(approximate_pi32=>1,identity64=>1))
        @test eltype(approximate_pi_mixed)===ComplexF64
        @test approximate_pi_mixed.range_dimension>0

        unsupported=try
            matrixfree_symmetry_projector(
                b,BigFloat[1 0;0 -1])
            nothing
        catch error
            error
        end
        @test unsupported isa ArgumentError
        @test occursin("ComplexF32 or ComplexF64",
                       sprint(showerror,unsupported))
    end

    # Full one-qubit Liouville-space path agrees with the PI path.
    L=liouvillian(PIModel(PIBasis(1,2),[LocalJump(sm)]);representation=:sparse)
    @test check_liouvillian_symmetry(L,sz;kind=:unitary).symmetric
    @test_throws ArgumentError check_liouvillian_symmetry(L,2sz)
    @test_throws ArgumentError check_liouvillian_symmetry(L,sz;kind=:other)

    # The assembled checker uses the same working-precision residual floor.
    generic32=Matrix(qr(randn(
        MersenneTwister(91),ComplexF32,3,3)).Q)
    conjugation32=PermutationalInvariantDynamics.sandwich_superoperator(
        generic32)
    commuting32=conjugation32*conjugation32
    assembled32=check_liouvillian_symmetry(
        commuting32,generic32)
    @test assembled32.symmetric
    @test assembled32.residual isa Float32
    @test assembled32.tolerance isa Float32
end
