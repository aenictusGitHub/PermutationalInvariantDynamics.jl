struct ToyCollectiveJump{O,R} <: AbstractPITerm
    operator::O
    rate::R
end

# An external term lowers through documented dispatch only.  In particular,
# neither PIModel nor the Liouvillian compiler knows this concrete type.
PermutationalInvariantDynamics.term_operator(t::ToyCollectiveJump)=t.operator
PermutationalInvariantDynamics.term_rate(t::ToyCollectiveJump)=t.rate
PermutationalInvariantDynamics.body_order(::ToyCollectiveJump)=1
PermutationalInvariantDynamics.term_scope(::ToyCollectiveJump)=Val(:collective)
PermutationalInvariantDynamics.term_process(::ToyCollectiveJump)=Val(:jump)
PermutationalInvariantDynamics.validate_term(t::ToyCollectiveJump,b::PIBasis)=
    PermutationalInvariantDynamics.validate_term(CollectiveJump(t.operator;rate=t.rate),b)
PermutationalInvariantDynamics.compile_term(t::ToyCollectiveJump,context)=
    PermutationalInvariantDynamics.compile_term(CollectiveJump(t.operator;rate=t.rate),context)
PermutationalInvariantDynamics.rebuild_term(::ToyCollectiveJump,operator,rate)=
    ToyCollectiveJump(operator,rate)

struct IncompleteToyTerm <: AbstractPITerm end

@testset "PI algebra and generators" begin
    b=PIBasis(3,2)
    id=identity_operator(b); mm=maximally_mixed_state(b)
    @test trace(id)≈2^3
    @test trace(mm)≈1
    @test purity(mm)≈1/8
    @test isphysical(mm)
    @test (id*id).data≈id.data atol=2e-14

    # In coefficient coordinates the two identity entries are sqrt(f).  Their
    # raw product overflows at these multiplicities although division by the
    # same sqrt(f) leaves the representable identity coefficient.  Operator
    # algebra must fuse that exceptional scale while ordinary sectors keep the
    # BLAS-first route above.
    for (T,N,tolerance) in ((Float32,200,8eps(Float32)),
                            (Float64,1100,8eps(Float64)))
        central=Partition((N÷2,N÷2))
        large_basis=PIBasis(N,2;sectors=[central.parts])
        large_identity=identity_operator(large_basis;T=T)
        squared=large_identity*large_identity
        @test all(isfinite,squared.data)
        @test squared.data≈large_identity.data rtol=tolerance atol=zero(T)
    end


    # Also force cancellation inside one complex product. The real component
    # is O(2^-24) relative to its two O(1) products and rounds to zero if the
    # fused exponent-range helper is used without the guarded complex check.
    complex_partition=Partition((100,100))
    complex_basis=PIBasis(200,2;sectors=[complex_partition.parts])
    complex_identity=identity_operator(complex_basis;T=Float32)
    coefficient=real(only(complex_identity.data))
    left_value=ComplexF32(coefficient,
        coefficient*(1.0f0+Float32(2.0^-23)))
    right_value=ComplexF32(coefficient,
        coefficient*(1.0f0-Float32(2.0^-24)))
    left_operator=PIOperator(complex_basis,ComplexF32[left_value])
    right_operator=PIOperator(complex_basis,ComplexF32[right_value])
    complex_product=left_operator*right_operator
    complex_reference=setprecision(BigFloat,256) do
        f=BigFloat(symmetric_group_dimension(complex_partition))
        Complex{BigFloat}(left_value)*Complex{BigFloat}(right_value)/sqrt(f)
    end
    @test !iszero(real(complex_reference))
    @test only(complex_product.data)≈ComplexF32(complex_reference) rtol=2f-5 atol=0f0
    X=ComplexF64[0 1;1 0]; Z=ComplexF64[1 0;0 -1]; sm=ComplexF64[0 1;0 0]
    Jx=collective_operator(b,X)
    @test ishermitian(Jx)
    rho=iid_pure_state(b,ComplexF64[1,0]); @test isphysical(rho)
    @test expectation(rho,collective_operator(b,Z))≈3
    m=PIModel(b,[LocalHamiltonian(X),LocalJump(sm;rate=.2),CollectiveJump(sm;rate=.1)])
    L=liouvillian(m;representation=:sparse); Mf=liouvillian(m;representation=:matrixfree)
    @test L*rho.data ≈ Mf*rho.data
    y=similar(rho.data); mul!(y,Mf,rho.data) # compile before allocation audit
    @test (@allocated mul!(y,Mf,rho.data)) <= 512
    @test check_generator(m).trace_preservation_error < 1e-10

    toy=ToyCollectiveJump(sm,0.1)
    toy_model=PIModel(b,[toy]);reference_model=PIModel(b,[CollectiveJump(sm;rate=0.1)])
    toy_sparse=liouvillian(toy_model;representation=:sparse)
    reference_sparse=liouvillian(reference_model;representation=:sparse)
    toy_matrixfree=liouvillian(toy_model;representation=:matrixfree)
    @test Matrix(toy_sparse)≈Matrix(reference_sparse) atol=2e-12
    @test toy_matrixfree*rho.data≈reference_sparse*rho.data atol=2e-12
    @test adjoint(toy_matrixfree)*rho.data≈adjoint(reference_sparse)*rho.data atol=2e-12
    driven_toy=PIModel(b,[ToyCollectiveJump(sm,(t,p)->p.rate*(1+t))])
    frozen_toy=freeze(driven_toy;time=0.25,parameters=(rate=0.4,),representation=:sparse)
    @test Matrix(frozen_toy)≈Matrix(liouvillian(PIModel(b,[CollectiveJump(sm;rate=0.5)]);
                                                representation=:sparse)) atol=2e-12
    @test_throws ArgumentError PIModel(b,[IncompleteToyTerm()])

    basic=steady_state(m;return_info=true)
    @test basic.nullity===nothing
    @test basic.diagnostics===:basic
    ss=steady_state(m;return_info=true,diagnostics=:nullity)
    @test ss.residual < 1e-9
    @test ss.trace_error < 1e-10
    @test ss.nullity == 1
    @test trace(PIState(b,ss.state)) ≈ 1
    @test_throws ArgumentError steady_state(m;diagnostics=:unknown)
    @test_throws ArgumentError steady_state(m;method=:krylov,diagnostics=:nullity)
    @test_throws ArgumentError steady_state(L)
    @test steady_state(Mf) ≈ ss.state atol=1e-8
    se=steady_state(m;method=:eigen,return_info=true)
    @test se.state≈ss.state atol=2e-8
    @test se.method===:eigen && se.converged
    si=steady_state(m;method=:shiftinvert,shift=-1e-3,maxiter=80,atol=1e-12,rtol=1e-10,return_info=true)
    @test si.state≈ss.state atol=2e-8
    @test si.method===:shiftinvert && si.iterations<=80 && si.converged
    @test steady_state(m;method=:krylov,shift=-2e-3,atol=1e-12,rtol=1e-10)≈ss.state atol=2e-8
    @test_throws ArgumentError steady_state(m;method=:shiftinvert,shift=0)

    z=steady_state(spzeros(ComplexF64,length(b),length(b));basis=b,
                   method=:svd,return_info=true)
    @test z.nullity == length(b)
    @test z.trace_error < 1e-10
    @test z.residual == 0

    D=Diagonal(ComplexF64[-1,-0.1+2im,-3])
    @test only(liouvillian_eigenvalues(D,1;which=:LR))≈-0.1+2im
    @test only(liouvillian_eigenvalues(D,1;which=:LM))≈-3
    @test only(liouvillian_eigenvalues(D,1;which=:SM))≈-1
    @test isempty(liouvillian_eigenvalues(D,0;which=:LR))
    @test_throws ArgumentError liouvillian_eigenvalues(D,1;which=:invalid)
    @test_throws ArgumentError liouvillian_eigenvalues(D,-1)
end
