@testset "state validation and cache compatibility" begin
    b=PIBasis(1,2)
    rho=iid_pure_state(b,ComplexF64[1,0])
    @test PermutationalInvariantDynamics.validate_state(rho)===rho
    diagnostics=PermutationalInvariantDynamics.state_diagnostics(rho)
    @test diagnostics.valid
    @test diagnostics.trace_one && diagnostics.hermitian && diagnostics.positive
    @test diagnostics.minimum_eigenvalue≈0 atol=1e-15

    # The requested tolerance must affect PI Hermiticity checks.  This
    # roundoff-sized skew component is accepted only at the loose tolerance.
    almost=copy(rho);p=only(b.sectors);C=coefficient_block(almost,p)
    C[1,2]=5e-9
    @test !ishermitian(almost;atol=1e-10,rtol=0)
    @test ishermitian(almost;atol=1e-8,rtol=0)
    strict=PermutationalInvariantDynamics.state_diagnostics(almost;atol=1e-10,rtol=0)
    @test !strict.hermitian && strict.minimum_eigenvalue===missing
    @test_throws ArgumentError PermutationalInvariantDynamics.validate_state(almost;atol=1e-10,rtol=0)
    @test PermutationalInvariantDynamics.validate_state(almost;atol=1e-8,rtol=0)===almost
    @test isfinite(von_neumann_entropy(almost;atol=1e-8,rtol=0))

    unnormalized=PIState(b,2rho.data)
    @test_throws ArgumentError PermutationalInvariantDynamics.validate_state(unnormalized)
    @test PermutationalInvariantDynamics.validate_state(unnormalized;trace_one=false)===unnormalized

    nonpositive=sector_density_matrix(b,p,ComplexF64[1.1 0;0 -0.1])
    negative_report=PermutationalInvariantDynamics.state_diagnostics(nonpositive)
    @test negative_report.hermitian && !negative_report.positive
    @test negative_report.minimum_eigenvalue≈-0.1
    @test !isphysical(nonpositive)
    @test_throws ArgumentError PermutationalInvariantDynamics.validate_state(nonpositive)
    @test PermutationalInvariantDynamics.validate_state(nonpositive;positive=false)===nonpositive
    @test_throws ArgumentError von_neumann_entropy(nonpositive)
    @test_throws ArgumentError negativity(nonpositive,0)

    roundoff_negative=sector_density_matrix(b,p,ComplexF64[1+5e-11 0;0 -5e-11])
    @test PermutationalInvariantDynamics.validate_state(roundoff_negative)===roundoff_negative
    @test isfinite(fidelity(roundoff_negative,rho))
    full_rank_reference=maximally_mixed_state(b)
    @test isfinite(quantum_relative_entropy(roundoff_negative,full_rank_reference))
    @test isfinite(relative_entropy_decomposition(
        roundoff_negative,full_rank_reference).total)
    @test_throws ArgumentError PermutationalInvariantDynamics.validate_state(roundoff_negative;atol=1e-12,rtol=0)

    # Force the scalable path on these small blocks so its result can be
    # compared directly with the spectral diagnostic above.
    certificate=positivity_diagnostics(rho;method=:auto,dense_threshold=1)
    @test certificate.positive && certificate.method===:cholesky
    @test certificate.minimum_eigenvalue===missing
    @test certificate.certified_lower_bound≈-1e-12
    witness=positivity_diagnostics(nonpositive;method=:cholesky)
    @test !witness.positive
    @test witness.witness_eigenvalue≈-0.1
    @test witness.certified_lower_bound===missing
    @test PermutationalInvariantDynamics.state_diagnostics(rho;
        positivity_method=:cholesky).positivity_method===:cholesky
    @test_throws ArgumentError PermutationalInvariantDynamics.validate_state(
        nonpositive;positivity_method=:cholesky)

    # Exercise the automatic large-block branch on a genuinely larger Schur
    # block, rather than only forcing it on the two-dimensional fixture above.
    # Restricting to the symmetric sector keeps this test polynomial while the
    # physical block itself is 33 x 33.
    large_basis=PIBasis(32,2;sectors=[(32,0)])
    large_state=PIState(large_basis;T=Float64)
    large_block=coefficient_block(large_state,first(large_basis.sectors))
    large_block.=Diagonal(fill(1/size(large_block,1),size(large_block,1)))
    large_certificate=positivity_diagnostics(large_state;method=:auto,
                                             dense_threshold=16)
    @test large_certificate.positive
    @test large_certificate.method===:cholesky
    @test large_certificate.factorized_sectors==1
    @test large_certificate.fallback_eigensolves==0

    @test_throws ArgumentError iid_pure_state(b,ComplexF64[2,0])
    @test iid_pure_state(b,ComplexF64[1+5e-9,0];atol=1e-8,rtol=0) isa PIState

    low_precision_large=PIBasis(2000,2;sectors=[(2000,0)])
    balanced16=ComplexF16[inv(sqrt(Float16(2))),inv(sqrt(Float16(2)))]
    @test_throws ArgumentError iid_pure_state(low_precision_large,balanced16)

    # sqrt(f) itself is beyond Float64 at N=2100, although the stored inverse
    # scale and its O(1) sector probability remain representable.
    central_basis=PIBasis(2100,2;sectors=[(1050,1050)])
    central_partition=only(central_basis.sectors)
    central=basis_state(central_basis,central_partition,
                        only(only(central_basis.patterns)))
    @test trace(central)==1
    @test sector_population(central,central_partition)==1
    @test state_diagnostics(central).valid
    @test_throws ArgumentError physical_block(central,central_partition)
    @test_throws ArgumentError purity(central)
    # A Float64 Liouvillian trace functional would itself require the
    # unrepresentable coefficient sqrt(f).  Compilation must fail explicitly
    # instead of retaining an Inf trace vector; BigFloat remains available.
    @test_throws ArgumentError LiouvillianPlan(PIModel(central_basis,()))
    big_trace_vector=PermutationalInvariantDynamics._trace_vector(
        central_basis,Complex{BigFloat})
    @test all(isfinite,big_trace_vector)

    mixed16=maximally_mixed_state(PIBasis(24,2);T=Float16)
    @test trace(mixed16)≈Float16(1) atol=Float16(2e-2)

    @testset "large multinomial product amplitudes" begin
        # These sizes are just beyond the points where converting the central
        # multinomial coefficient to the destination float produces Inf even
        # though every final occupation amplitude remains finite.  Keep only
        # the symmetric sector so the regression stays polynomial in N.
        for (T,N,rtol) in ((Float16,25,Float16(2e-2)),
                           (Float32,200,Float32(2e-5)),
                           (Float64,1100,2e-11))
            basis=PIBasis(N,2;sectors=[(N,0)])
            psi=Complex{T}[inv(sqrt(T(2))),inv(sqrt(T(2)))]
            product=iid_pure_state(basis,psi)
            block=coefficient_block(product,only(basis.sectors))

            @test eltype(product.data)===Complex{T}
            @test all(isfinite,product.data)

            occupation=N÷2
            pattern_index=findfirst(pattern->content(pattern)==
                (occupation,N-occupation),only(basis.patterns))
            @test pattern_index!==nothing
            reference_probability=setprecision(256) do
                psi_big=Complex{BigFloat}.(psi)
                binomial(big(N),big(occupation))*
                    abs2(psi_big[1])^occupation*
                    abs2(psi_big[2])^(N-occupation)
            end
            @test real(block[pattern_index,pattern_index]) ≈
                T(reference_probability) rtol=rtol atol=zero(T)
            @test abs(imag(block[pattern_index,pattern_index]))<=eps(T)
        end

        # An exactly absent local level must remove every occupation carrying
        # that label without a logarithm, division by zero, or 0*Inf.
        qutrit_basis=PIBasis(20,3;sectors=[(20,0,0)])
        singular=ComplexF32[0.6f0,0,0.8f0im]
        singular_product=iid_pure_state(qutrit_basis,singular)
        singular_block=coefficient_block(
            singular_product,only(qutrit_basis.sectors))
        forbidden=findall(pattern->content(pattern)[2]>0,
                          only(qutrit_basis.patterns))
        @test all(isfinite,singular_product.data)
        @test all(iszero,@view singular_block[forbidden,:])
        @test all(iszero,@view singular_block[:,forbidden])
        @test trace(singular_product)≈1.0f0 atol=2e-5 rtol=2e-5

        # Squaring this component underflows in Float32, but the one-particle
        # occupation coherence is itself representable and must be retained.
        tiny_basis=PIBasis(200,2;sectors=[(200,0)])
        tiny_component=1f-30
        tiny_product=iid_pure_state(
            tiny_basis,ComplexF32[tiny_component,1])
        tiny_block=coefficient_block(tiny_product,only(tiny_basis.sectors))
        patterns=only(tiny_basis.patterns)
        vacuum_index=findfirst(pattern->content(pattern)==(0,200),patterns)
        one_index=findfirst(pattern->content(pattern)==(1,199),patterns)
        @test all(isfinite,tiny_product.data)
        @test abs(tiny_block[one_index,vacuum_index]) ≈
            sqrt(200.0f0)*tiny_component rtol=8eps(Float32) atol=0
    end

    @testset "rank-deficient iid states" begin
        for (d,N) in ((2,4),(3,3))
            basis=PIBasis(N,d)
            psi=ComplexF64[complex(k,k-1) for k in 1:d]
            psi./=norm(psi)
            local_rho=psi*psi'
            product=iid_state(basis,local_rho)
            reference=iid_pure_state(basis,psi)
            @test product.data≈reference.data atol=2e-12 rtol=2e-12
            @test trace(product)≈1 atol=2e-12
            @test isphysical(product)
        end

        # A genuinely mixed but singular qutrit state occupies several Schur
        # sectors and is obtained without a logarithm or regularization.
        basis=PIBasis(4,3)
        local_rho=ComplexF64[0.7 0 0;0 0.3 0;0 0 0]
        product=iid_state(basis,local_rho)
        @test trace(product)≈1 atol=2e-12
        @test isphysical(product)
        @test count(p->real(sector_population(product,p))>1e-12,basis.sectors)>1
        for (sector,patterns) in zip(basis.sectors,basis.patterns)
            expected=Diagonal([prod(real(local_rho[level,level])^occupation
                for (level,occupation) in enumerate(content(pattern)))
                for pattern in patterns])
            @test physical_block(product,sector)≈expected atol=2e-12 rtol=2e-12
        end

        # The CG recurrence itself is scalar generic and does not rely on a
        # LAPACK eigendecomposition.  In particular, a singular BigFloat input
        # remains BigFloat throughout construction.
        setprecision(192) do
            big_local=Complex{BigFloat}[
                big"0.7" 0 0
                0 big"0.3" 0
                0 0 0
            ]
            big_product=iid_state(PIBasis(3,3),big_local)
            @test eltype(big_product.data)===Complex{BigFloat}
            @test abs(trace(big_product)-1)<=big"1e-50"
            @test abs(purity(big_product)-real(LinearAlgebra.tr(big_local*big_local))^3)<=big"1e-50"
            big_certificate=positivity_diagnostics(big_product)
            @test big_certificate.positive
            @test big_certificate.method===:cholesky
            big_bad=copy(big_product)
            first_block=coefficient_block(big_bad,first(big_bad.basis.sectors))
            first_block[1,1]=-one(BigFloat)
            @test !positivity_diagnostics(big_bad;method=:cholesky).positive

            # A tiny rank-one PSD block has a negative Gershgorin lower bound
            # but must be accepted by the generic pivoted semidefinite check.
            tiny_basis=PIBasis(3,2)
            tiny_state=PIState(tiny_basis;T=BigFloat)
            delta=BigFloat(128)*eps(BigFloat)
            symmetric=first(tiny_basis.sectors)
            other=last(tiny_basis.sectors)
            coefficient_block(tiny_state,symmetric).=
                delta.*ones(Complex{BigFloat},4,4)
            fother=BigFloat(symmetric_group_dimension(other))
            other_weight=(one(BigFloat)-4delta)/(2fother)
            coefficient_block(tiny_state,other).=
                sqrt(fother).*Diagonal(fill(other_weight,2))
            @test abs(trace(tiny_state)-1)<=big"1e-50"
            @test positivity_diagnostics(tiny_state;method=:cholesky,
                                         atol=0,rtol=0).positive
        end

        single_product=iid_state(PIBasis(2,2),Float32[0.75 0;0 0.25])
        @test eltype(single_product.data)===ComplexF32

        @test_throws ArgumentError iid_state(basis,2local_rho)
        @test_throws ArgumentError iid_state(basis,ComplexF64[1.01 0 0;0 -0.01 0;0 0 0])

        symmetric_only=PIBasis(3,2;sectors=[(3,0)])
        @test_throws ArgumentError iid_state(symmetric_only,ComplexF64[0.5 0;0 0.5])
        @test trace(iid_state(symmetric_only,ComplexF64[1 0;0 0]))≈1
    end

    # Geometry caches are basis-owned, even when two bases have the same
    # structural labels.  Reusing one across basis objects risks wrong indices.
    b1=PIBasis(3,2);b2=PIBasis(3,2);X=Matrix{ComplexF64}(I,2,2)
    onebody=OneBodyGeometry(b1)
    @test_throws ArgumentError collective_block(b2,X,first(b2.sectors);cache=onebody)

    pair=Matrix{ComplexF64}(I,4,4);pcache=PBodyGeometry(b1,2)
    @test_throws ArgumentError pbody_collective_block(b2,pair,2,first(b2.sectors);cache=pcache)
    @test_throws ArgumentError pbody_collective_operator(b1,pair,1;cache=pcache)
    @test_throws ArgumentError pbody_kernel_operator(b2,pair,pair,2;cache=pcache)
end
