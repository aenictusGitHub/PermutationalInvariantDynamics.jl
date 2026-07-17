@testset "exact local moments and cumulant bridge" begin
    PID=PermutationalInvariantDynamics
    sx=ComplexF64[0 1;1 0]
    sy=ComplexF64[0 -im;im 0]
    sz=ComplexF64[1 0;0 -1]
    sm=ComplexF64[0 1;0 0]
    sigma=ComplexF64[0.63 0.12+0.07im;0.12-0.07im 0.37]
    basis=PIBasis(4,2)
    product=iid_state(basis,sigma)

    # Distinct-site product moments factorize, including non-Hermitian local
    # operators.  The ordinary tr(A rho) convention is tested explicitly.
    for operators in ((sx,),(sm,),(sx,sz),(sm,sy,sz))
        reference=prod(tr(operator*sigma) for operator in operators)
        @test ordered_local_moment(product,operators)≈reference atol=4e-11
    end
    @test ordered_local_moment(product,())≈trace(product)

    # Independent lower-order APIs and a reduced state give the same ordered
    # local marginals.  GHZ supplies correlations not explained by a product
    # factorization.
    correlated=ghz_state(basis)
    @test ordered_local_moment(correlated,(sx,sx,sx,sx))≈1 atol=5e-11
    @test ordered_local_moment(correlated,(sm,sm,sm,sm))≈0.5 atol=5e-11
    @test ordered_local_moment(correlated,sx)≈tr(sx*one_body_rdm(correlated)) atol=4e-11
    @test ordered_local_moment(correlated,(sx,sy))≈
        two_body_expectation(correlated,sx,sy) atol=5e-11
    reduced=reduced_state(correlated,3)
    @test ordered_local_moment(correlated,(sx,sy,sz))≈
        ordered_local_moment(reduced,(sx,sy,sz)) atol=8e-11
    nonsymmetric_sector=dicke_state(basis,1,0)
    @test ordered_local_moment(nonsymmetric_sector,(sx,sz))≈
        two_body_expectation(nonsymmetric_sector,sx,sz) atol=6e-11
    reduced_sector=reduced_state(nonsymmetric_sector,3)
    @test ordered_local_moment(nonsymmetric_sector,(sm,sy,sz))≈
        ordered_local_moment(reduced_sector,(sm,sy,sz)) atol=1e-10

    moments=ordered_local_moments(product,(x=sx,y=sy,z=sz,m=sm);order=3)
    @test moments isa OrderedLocalMoments
    @test length(moments)==4+10+20
    @test moments[:m,:x,:z]==moments[:z,:m,:x]
    @test moments[(:x,:z)]≈ordered_local_moment(product,(sx,sz))
    @test haskey(moments,(:z,:x))
    @test !haskey(moments,(:unknown,))
    @test !haskey(ordered_local_moments(product,(x=sx,z=sz);
                                       order=3,include_lower=false),(:x,))
    zeroth=ordered_local_moments(product,Pair{Symbol,Matrix{ComplexF64}}[];
                                 order=0)
    @test only(values(zeroth.values))≈trace(product)

    sigma32=ComplexF32.(sigma)
    product32=iid_state(basis,sigma32)
    moments32=ordered_local_moments(product32,(x=ComplexF32.(sx),
                                               z=ComplexF32.(sz));order=2)
    @test eltype(moments32)===ComplexF32
    @test moments32[:x,:z]≈tr(ComplexF32.(sx)*sigma32)*
        tr(ComplexF32.(sz)*sigma32) rtol=3f-4

    # The exact subset normalization is fused into every Appendix-D path.
    # Thus a finite local moment remains representable even when the
    # standalone binomial(N,k) is far beyond Float32.
    trivial_large=PIBasis(1000,1)
    trivial_state=maximally_mixed_state(trivial_large;T=Float32)
    @test ordered_local_moment(
        trivial_state,ntuple(_->ones(ComplexF32,1,1),20))≈1f0 rtol=2f-5

    exact_dictionary=Dict(key=>value for (key,value) in moments.values)
    exact_comparison=compare_cumulant_closure(moments,exact_dictionary)
    @test exact_comparison.within_tolerance
    @test exact_comparison.maximum_absolute_error==0
    perturbed=copy(exact_dictionary)
    perturbed[(:x,)]+=1e-3
    comparison=compare_cumulant_closure(moments,perturbed;atol=1e-8,rtol=1e-8)
    @test !comparison.within_tolerance
    @test comparison.maximum_absolute_error≈1e-3
    @test comparison.count==length(moments)

    pair=kron(sz,sz)
    scheduled_rate=(time,parameters)->parameters.rate*(1+time)
    direct=collective_operator(basis,sz)
    model=PIModel(basis,(
        LocalHamiltonian(sx;rate=scheduled_rate,hbar=2),
        PBodyHamiltonian(pair,2;rate=0.2),
        DirectPIJump(direct;rate=0.1),
    ))
    neutral=cumulant_model_payload(model)
    @test neutral.schema_version==v"1.0.0"
    @test (neutral.N,neutral.d)==(4,2)
    @test neutral.terms[1].process==:hamiltonian
    @test neutral.terms[1].scope==:local
    @test neutral.terms[1].rate_time_dependent
    @test neutral.terms[1].hbar==2
    @test neutral.terms[2].order==2
    @test neutral.terms[2].microscopic
    @test !neutral.terms[3].microscopic
    @test neutral.terms[3].scope==:direct
    evaluated=cumulant_model_payload(model;time=0.5,parameters=(rate=0.4,))
    @test evaluated.terms[1].rate≈0.6
    @test evaluated.terms[1].evaluated_at==0.5

    bridge=cumulant_bridge_payload(model,product,(x=sx,z=sz);order=2)
    @test bridge.model.schema_version==bridge.schema_version
    @test bridge.moments[:x,:z]≈ordered_local_moment(product,(sx,sz))

    other_basis=PIBasis(4,2)
    @test_throws ArgumentError cumulant_bridge_payload(
        model,iid_state(other_basis,sigma),(x=sx,);order=1)
    @test_throws ArgumentError ordered_local_moment(product,(sx,sy,sz,sm,sx))
    @test_throws DimensionMismatch ordered_local_moment(product,ones(3,3))
    @test_throws ArgumentError ordered_local_moments(product,[:x=>sx,:x=>sz];order=1)
    @test_throws KeyError compare_cumulant_closure(moments,Dict{Tuple,ComplexF64}())
    @test_throws ArgumentError quantumcumulants_initial_values(moments,
        Dict((:x,)=>:placeholder))
end

# This smoke test is activated only when an optional-test environment has
# explicitly loaded QuantumCumulants.  The ordinary package test target does
# not acquire or compile that weak dependency.
quantumcumulants_extension=Base.get_extension(
    PermutationalInvariantDynamics,
    :PermutationalInvariantDynamicsQuantumCumulantsExt)
if quantumcumulants_extension!==nothing
    @testset "QuantumCumulants exact p-body normalization" begin
        sm=ComplexF32[0 1;0 0]
        small_basis=PIBasis(2,2)
        small_model=PIModel(small_basis,(
            LocalPBodyJump(kron(sm,sm),2;rate=1f0),))
        small_result=quantumcumulants_model(small_model;order=1)
        @test only(small_result.rates)===0.5f0

        # d=1 keeps the symbolic smoke test tiny while reaching an order whose
        # factorial is not representable as a machine Int.
        SQA=quantumcumulants_extension.SQA
        hilbert=SQA.NLevelSpace(:pid_atom,1,1)
        probe=SQA.Index(hilbert,:z,21,hilbert)
        seed=SQA.IndexedOperator(
            SQA.Transition(hilbert,:σ,1,1),probe)
        large_basis=PIBasis(21,1)
        operator=ones(ComplexF64,1,1)
        hamiltonian_result=quantumcumulants_model(
            PIModel(large_basis,(PBodyHamiltonian(operator,21),));
            order=1,seed_operators=[seed])
        @test length(hamiltonian_result.equations)==1
        local_result=quantumcumulants_model(
            PIModel(large_basis,(LocalPBodyJump(operator,21;rate=1),));
            order=1,seed_operators=[seed])
        @test only(local_result.rates) isa Rational{BigInt}
        @test only(local_result.rates)==
            one(BigInt)//factorial(big(21))
    end
end
