function _metamorphic_generator_equivalence(
        left::PIModel,right::PIModel;
        seed::Integer,atol::Real=4e-11,rtol::Real=4e-11)
    @test left.basis===right.basis
    sparse_left=liouvillian(left;representation=:sparse)
    sparse_right=liouvillian(right;representation=:sparse)
    @test Matrix(sparse_left)≈Matrix(sparse_right) atol=atol rtol=rtol

    matrixfree_left=liouvillian(left;representation=:matrixfree)
    matrixfree_right=liouvillian(right;representation=:matrixfree)
    rng=MersenneTwister(seed)
    source=randn(rng,ComplexF64,length(left.basis))
    @test matrixfree_left*source≈matrixfree_right*source atol=atol rtol=rtol
    @test adjoint(matrixfree_left)*source≈
          adjoint(matrixfree_right)*source atol=atol rtol=rtol

    @test check_generator(left).trace_preservation_error<=20atol
    @test check_generator(right).trace_preservation_error<=20atol
end

@testset "physical metamorphic generator identities" begin
    basis=PIBasis(3,2)
    identity2=Matrix{ComplexF64}(I,2,2)
    identity4=Matrix{ComplexF64}(I,4,4)
    h=ComplexF64[0.31 0.22+0.13im;0.22-0.13im -0.27]
    jump=ComplexF64[0.11 0.73-0.19im;-0.08im -0.24]
    phase=cis(0.417)

    # Adding a scalar identity to any Hamiltonian changes only the global
    # energy origin. Its commutator must vanish in both prepared backends.
    for (index,constructor) in enumerate((
            operator->LocalHamiltonian(operator;rate=0.37),
            operator->CollectiveHamiltonian(operator;rate=0.37)))
        original=PIModel(basis,(constructor(h),))
        shifted=PIModel(basis,(constructor(h+1.7identity2),))
        _metamorphic_generator_equivalence(
            original,shifted;seed=0x4100+index)
    end

    x=ComplexF64[0 1;1 0]
    z=ComplexF64[1 0;0 -1]
    pair_h=kron(x,x)+0.17kron(z,z)+
           0.11(kron(x,z)+kron(z,x))
    original_pair_h=PIModel(
        basis,(PBodyHamiltonian(pair_h,2;rate=-0.23),))
    shifted_pair_h=PIModel(
        basis,(PBodyHamiltonian(pair_h+0.9identity4,2;rate=-0.23),))
    _metamorphic_generator_equivalence(
        original_pair_h,shifted_pair_h;seed=0x4200)

    # A Lindblad dissipator is invariant under a global phase of its jump.
    # Check ordinary and Appendix-D local/collective channels independently.
    pair_jump=kron(jump,jump)
    jump_constructors=(
        (operator,rate)->LocalJump(operator;rate),
        (operator,rate)->CollectiveJump(operator;rate),
        (operator,rate)->LocalPBodyJump(operator,2;rate),
        (operator,rate)->CollectivePBodyJump(operator,2;rate),
    )
    jump_operators=(jump,jump,pair_jump,pair_jump)
    for index in eachindex(jump_constructors)
        constructor=jump_constructors[index]
        operator=jump_operators[index]
        original=PIModel(basis,(constructor(operator,0.41),))
        rephased=PIModel(basis,(constructor(phase*operator,0.41),))
        _metamorphic_generator_equivalence(
            original,rephased;seed=0x4300+index)
    end

    # Splitting one physical channel into identical channels with additive
    # rates must not depend on deterministic-kernel fusion.
    for index in eachindex(jump_constructors)
        constructor=jump_constructors[index]
        operator=jump_operators[index]
        single=PIModel(basis,(constructor(operator,0.53),))
        split=PIModel(basis,(
            constructor(operator,0.17),
            constructor(operator,0.36),
        ))
        _metamorphic_generator_equivalence(
            single,split;seed=0x4400+index)
    end
end
