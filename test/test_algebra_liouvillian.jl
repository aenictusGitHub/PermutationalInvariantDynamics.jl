struct ToyCollectiveJump{O,R} <: AbstractPITerm
    operator::O
    rate::R
end

@testset "Custom matrix-free callbacks" begin
    A=ComplexF32[-1 2im 0.5; 3 -4 -im; 0 2 -0.25]
    n=size(A,1);tracevec=ComplexF32[1,0,1]
    forward_calls=Ref(0);adjoint_calls=Ref(0)
    batched_calls=Ref(0);batched_adjoint_calls=Ref(0)
    factor(t,p)=p===nothing ? one(Float32) : p.scale*(one(t)+t)
    action! = (destination,source,time,parameters)->begin
        forward_calls[]+=1
        mul!(destination,factor(time,parameters)*A,source)
    end
    adjoint_action! = (destination,source,time,parameters)->begin
        adjoint_calls[]+=1
        mul!(destination,adjoint(factor(time,parameters)*A),source)
    end
    batched_action! = (destination,source,time,parameters)->begin
        batched_calls[]+=1
        mul!(destination,factor(time,parameters)*A,source)
    end
    batched_adjoint_action! = (destination,source,time,parameters)->begin
        batched_adjoint_calls[]+=1
        mul!(destination,adjoint(factor(time,parameters)*A),source)
    end
    custom=MatrixFreeLiouvillian(n,action!,ComplexF32,tracevec;
        adjoint_action!,batched_action!,batched_adjoint_action!)
    x=ComplexF32[1+im,-2,0.5im];X=hcat(x,2x,-x)

    # Matrix products call the supplied batch once instead of dispatching one
    # synchronized vector callback per right-hand side.
    Y=similar(X);mul!(Y,custom,X)
    @test Y≈A*X
    @test batched_calls[]==1
    @test forward_calls[]==0

    parameters=(scale=0.75f0,);time=0.2f0
    apply!(Y,custom,X,time,parameters)
    @test Y≈factor(time,parameters)*A*X
    @test batched_calls[]==2

    y=similar(x)
    apply_adjoint!(y,custom,x,time,parameters)
    @test y≈adjoint(factor(time,parameters)*A)*x
    @test adjoint_calls[]==1
    @test forward_calls[]==0

    apply_adjoint!(Y,custom,X,time,parameters)
    @test Y≈adjoint(factor(time,parameters)*A)*X
    @test batched_adjoint_calls[]==1
    @test adjoint_calls[]==1

    # Constructing and multiplying the autonomous adjoint must remain
    # matrix-free when an explicit adjoint callback is available.
    custom_adjoint=adjoint(custom)
    mul!(y,custom_adjoint,x)
    @test y≈adjoint(A)*x
    mul!(Y,custom_adjoint,X)
    @test Y≈adjoint(A)*X
    @test forward_calls[]==0
    @test adjoint_calls[]==2
    @test batched_adjoint_calls[]==2

    # A batch-only adjoint can still serve a one-vector request through a
    # one-column wrapper, without probing and materializing the forward map.
    batch_only_calls=Ref(0)
    batch_only_adjoint! = (destination,source,time,parameters)->begin
        batch_only_calls[]+=1
        mul!(destination,adjoint(A),source)
    end
    batch_only=MatrixFreeLiouvillian(n,action!,ComplexF32,tracevec;
        batched_adjoint_action! = batch_only_adjoint!)
    apply_adjoint!(y,batch_only,x,0.3f0,nothing)
    @test y≈adjoint(A)*x
    @test batch_only_calls[]==1

    # The original constructor and column-wise fallbacks remain valid.
    legacy_calls=Ref(0)
    legacy_action! = (destination,source,time,parameters)->begin
        legacy_calls[]+=1
        mul!(destination,A,source)
    end
    legacy=MatrixFreeLiouvillian(n,legacy_action!,ComplexF32,tracevec)
    mul!(Y,legacy,X)
    @test Y≈A*X
    @test legacy_calls[]==size(X,2)

    # A non-PI adapter may retain an opaque plan for metadata and still expose
    # only the vector callback.  It must use the same column fallback rather
    # than being mistaken for a compiled PI LiouvillianPlan.
    adapter_calls=Ref(0)
    adapter_action! = (destination,source,time,parameters)->begin
        adapter_calls[]+=1
        mul!(destination,A,source)
    end
    adapter=MatrixFreeLiouvillian(n,adapter_action!,ComplexF32,tracevec;
        plan=(kind=:external,),workspace=nothing)
    mul!(Y,adapter,X)
    @test Y≈A*X
    @test adapter_calls[]==size(X,2)

    # Freezing a driven custom operator fixes all supplied callbacks, not only
    # its forward vector action.
    driven=MatrixFreeLiouvillian(n,action!,ComplexF32,tracevec;
        autonomous=false,adjoint_action!,batched_action!,
        batched_adjoint_action!)
    @test_throws ArgumentError driven*X
    frozen=freeze(driven;time,parameters)
    @test isautonomous(frozen)
    @test frozen*X≈factor(time,parameters)*A*X
    @test adjoint(frozen)*X≈adjoint(factor(time,parameters)*A)*X

    # Allocating products preserve normal matrix promotion rules.
    wide=custom*Float64[1 2;3 4;5 6]
    @test eltype(wide)===ComplexF64
    @test wide≈A*Float64[1 2;3 4;5 6]
    @test_throws DimensionMismatch mul!(zeros(ComplexF32,n,2),custom,
                                        zeros(ComplexF32,n-1,2))
    @test_throws DimensionMismatch apply_adjoint!(zeros(ComplexF32,n,2),custom,
                                                   zeros(ComplexF32,n,3),0,nothing)

    # Invalid or wider shift-invert inputs are rejected before probing a
    # matrix-free source for explicit materialization.
    callback_counts=(forward_calls[],batched_calls[])
    @test_throws ArgumentError steady_state(custom;method=:shiftinvert,
        shift=big"-0.001",memory_budget=Inf)
    @test (forward_calls[],batched_calls[])==callback_counts
end

@testset "Liouvillian application ownership" begin
    basis=PIBasis(2,2)
    spin=spin_matrices()
    plan=LiouvillianPlan(PIModel(basis,(
        LocalHamiltonian(spin.jx;rate=0.2),
        LocalJump(spin.jm;rate=0.4),
    )))
    workspace=LiouvillianWorkspace(plan)
    vector=ones(ComplexF64,length(basis))
    @test_throws ArgumentError apply!(
        vector,plan,vector,0.0,nothing,workspace)
    @test_throws ArgumentError apply_adjoint!(
        vector,plan,vector,0.0,nothing,workspace)

    matrix=ones(ComplexF64,length(basis),2)
    @test_throws ArgumentError apply!(
        matrix,plan,matrix,0.0,nothing,workspace)
    @test_throws ArgumentError apply_adjoint!(
        matrix,plan,matrix,0.0,nothing,workspace)

    storage=ones(ComplexF64,length(basis)+1)
    source=@view storage[1:length(basis)]
    destination=@view storage[2:length(basis)+1]
    @test Base.mightalias(source,destination)
    @test_throws ArgumentError apply!(
        destination,plan,source,0.0,nothing,workspace)
    @test_throws ArgumentError apply_adjoint!(
        destination,plan,source,0.0,nothing,workspace)

    # The same contract applies before invoking a raw operator-valued fallback.
    fallback=LiouvillianPlan(PIModel(basis,(
        LocalJump((t,p)->spin.jm;rate=0.4),)))
    @test fallback.kernels===nothing
    @test_throws ArgumentError apply!(
        vector,fallback,vector,0.0,nothing,LiouvillianWorkspace(fallback))
end

@testset "Prepared dense-kernel precision" begin
    basis=PIBasis(2,2)
    spin32=spin_matrices(2;T=Float32)
    pair32=kron(spin32.jx,spin32.jx)
    direct32=PIOperator(basis,ComplexF32.(
        collective_operator(basis,spin32.jx).data))
    pair64=ComplexF64.(pair32)
    direct64=PIOperator(basis,ComplexF64.(direct32.data))
    terms32=(
        PBodyHamiltonian(pair32,2;rate=1.0),
        LocalPBodyJump(pair32,2;rate=1.0),
        CollectivePBodyJump(pair32,2;rate=1.0),
        DirectPIHamiltonian(direct32;rate=1.0),
        DirectPIJump(direct32;rate=1.0),
    )
    terms64=(
        PBodyHamiltonian(pair64,2;rate=1.0),
        LocalPBodyJump(pair64,2;rate=1.0),
        CollectivePBodyJump(pair64,2;rate=1.0),
        DirectPIHamiltonian(direct64;rate=1.0),
        DirectPIJump(direct64;rate=1.0),
    )
    source=ComplexF64.(1:length(basis))./(length(basis)+1)
    batch=hcat(source,2source)

    for (term32,term64) in zip(terms32,terms64)
        plan=LiouvillianPlan(
            PIModel(basis,(term32,));fuse_static=Val(false))
        @test eltype(plan)===ComplexF64
        workspace=LiouvillianWorkspace(plan)
        destination=similar(source)
        batch_destination=similar(batch)
        reference=liouvillian(
            PIModel(basis,(term64,));representation=:sparse)

        apply!(destination,plan,source,0.0,nothing,workspace)
        @test destination≈reference*source atol=2e-6 rtol=2e-6
        @test (@allocated apply!(
            destination,plan,source,0.0,nothing,workspace))<=1024

        apply_adjoint!(destination,plan,source,0.0,nothing,workspace)
        @test destination≈adjoint(reference)*source atol=2e-6 rtol=2e-6
        @test (@allocated apply_adjoint!(
            destination,plan,source,0.0,nothing,workspace))<=1024

        # The first batch call grows the explicit fixed-capacity scratch.
        # Subsequent forward and adjoint actions must not repack mixed-precision
        # dense operators in hidden temporary arrays.
        apply!(batch_destination,plan,batch,0.0,nothing,workspace)
        @test batch_destination≈reference*batch atol=2e-6 rtol=2e-6
        @test (@allocated apply!(
            batch_destination,plan,batch,0.0,nothing,workspace))<=1024
        apply_adjoint!(
            batch_destination,plan,batch,0.0,nothing,workspace)
        @test batch_destination≈adjoint(reference)*batch atol=2e-6 rtol=2e-6
        @test (@allocated apply_adjoint!(
            batch_destination,plan,batch,0.0,nothing,workspace))<=1024
    end
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
    nonfinite_toy_operator=copy(sm)
    nonfinite_toy_operator[1,1]=Inf
    @test_throws ArgumentError LiouvillianPlan(PIModel(b,(
        ToyCollectiveJump(nonfinite_toy_operator,0.1),)))
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
    exact_shift=-big(1)//big(1000)
    exact_component=big(1)//big(3)
    exact_zero=big(0)//big(1)
    exact_initial=fill(complex(exact_zero,exact_zero),length(b))
    trace_coordinate=findfirst(!iszero,PermutationalInvariantDynamics._trace_vector(b))
    exact_initial[trace_coordinate]=complex(big(1)//big(1),exact_component)
    exact_si=steady_state(m;method=:shiftinvert,shift=exact_shift,
        initial_state=exact_initial,maxiter=80,atol=1e-12,rtol=1e-10,
        return_info=true)
    @test exact_si.state≈ss.state atol=2e-8
    exact_complex_si=steady_state(m;method=:shiftinvert,
        shift=complex(exact_shift,big(1)//big(10)^6),maxiter=80,
        atol=1e-12,rtol=1e-10,return_info=true)
    @test exact_complex_si.state≈ss.state atol=2e-8
    si32=steady_state(m;method=:shiftinvert,shift=-1f-3,
        initial_state=ComplexF32.(ss.state),maxiter=80,atol=1e-12,
        rtol=1e-10,return_info=true)
    @test si32.state≈ss.state atol=2e-8
    @test steady_state(m;method=:krylov,shift=-2e-3,atol=1e-12,rtol=1e-10)≈ss.state atol=2e-8
    @test_throws ArgumentError steady_state(m;method=:shiftinvert,shift=0)
    @test_throws ArgumentError steady_state(m;method=:shiftinvert,
        shift=Inf)
    @test_throws ArgumentError steady_state(m;method=:shiftinvert,
        shift=complex(-1e-3,Inf))
    @test_throws ArgumentError steady_state(m;method=:shiftinvert,
        shift=big"-0.001")
    @test_throws ArgumentError steady_state(m;method=:shiftinvert,
        shift=-1e-3,initial_state=Complex{BigFloat}.(ss.state))
    underflowing_exact=big(1)//big(10)^1000
    @test_throws ArgumentError steady_state(m;method=:shiftinvert,
        shift=complex(exact_shift,underflowing_exact))
    underflowing_initial=copy(exact_initial)
    underflowing_initial[trace_coordinate]=
        complex(big(1)//big(1),underflowing_exact)
    @test_throws ArgumentError steady_state(m;method=:shiftinvert,
        shift=-1e-3,initial_state=underflowing_initial)
    nonfinite_initial=copy(ss.state);nonfinite_initial[1]=complex(NaN,0)
    @test_throws ArgumentError steady_state(m;method=:shiftinvert,
        shift=-1e-3,initial_state=nonfinite_initial)

    # Julia's factorizing backends do not support arbitrary-precision sparse
    # LU/SVD. Automatic basic solving must select the exact matrix-free Krylov
    # route rather than narrowing or materializing a Float64 surrogate.
    big_generator=Complex{BigFloat}[-1 1;1 -1]
    big_trace=Complex{BigFloat}[1,1]
    big_auto=steady_state(big_generator;trace_vector=big_trace,method=:auto,
        krylovdim=2,maxiter=20,atol=big"1e-40",rtol=big"1e-30",
        return_info=true,memory_budget=Inf)
    @test big_auto.method===:krylov
    @test big_auto.state≈Complex{BigFloat}[0.5,0.5] atol=big"1e-35"
    @test big_auto.residual<=big"1e-35"
    @test_throws ArgumentError steady_state(big_generator;
        trace_vector=big_trace,method=:direct,memory_budget=Inf)
    @test_throws ArgumentError steady_state(big_generator;
        trace_vector=big_trace,method=:shiftinvert,shift=big"-0.001",
        memory_budget=Inf)
    @test_throws ArgumentError steady_state(big_generator;
        trace_vector=big_trace,method=:auto,diagnostics=:nullity,
        memory_budget=Inf)

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
