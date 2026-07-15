@testset "large-multiplicity weighted analysis" begin
    sx=ComplexF64[0 1;1 0]/2
    sz=ComplexF64[1 0;0 -1]/2

    # Keep only a fixed j=1 qubit sector.  Its Schur matrices stay 3x3 while
    # f^nu grows exponentially; at N=1100, f^nu is outside Float64 and the
    # physical density eigenvalue 1/f^nu underflows to zero.  All requested
    # observables and information quantities nevertheless remain O(1) (apart
    # from the entropy, which is O(N)) and representable.
    for N in (50,1100,2100)
        p=Partition((N÷2+1,N÷2-1))
        basis=PIBasis(N,2;sectors=[p.parts])
        patterns=only(basis.patterns)
        middle=only(findall(g->content(g)==(N÷2,N÷2),patterns))
        rho=basis_state(basis,p,patterns[middle])
        orthogonal=basis_state(basis,p,patterns[1])
        multiplicity=symmetric_group_dimension(p)
        expected_entropy=setprecision(256) do
            Float64(log(BigFloat(multiplicity))/log(big(2)))
        end

        @test trace(rho)≈1 atol=2e-12
        @test von_neumann_entropy(rho)≈expected_entropy atol=5e-11
        @test renyi_entropy(rho,2)≈expected_entropy atol=5e-11
        @test renyi_entropy(rho,Inf)≈expected_entropy atol=5e-11
        @test fidelity(rho,rho)≈1 atol=3e-12
        @test fidelity(rho,orthogonal)≈0 atol=3e-12
        @test trace_distance(rho,orthogonal)≈1 atol=3e-12
        @test quantum_relative_entropy(rho,rho)≈0 atol=3e-12
        @test quantum_relative_entropy(rho,orthogonal)==Inf

        resolved=only(sector_resolved_entropy(rho))
        @test resolved.probability≈1 atol=2e-12
        @test resolved.irrep_entropy≈0 atol=2e-12
        @test resolved.multiplicity_entropy≈expected_entropy atol=5e-11
        @test entropy_decomposition(rho).total≈expected_entropy atol=5e-11
        @test relative_entropy_decomposition(rho,rho).total≈0 atol=3e-12

        geometry=OneBodyGeometry(basis)
        planx=CollectiveObservablePlan(basis,sx;cache=geometry)
        planz=CollectiveObservablePlan(basis,sz;cache=geometry)
        moments=collective_moments(rho,planx)
        @test moments.mean≈0 atol=3e-12
        @test moments.second_moment≈1 atol=3e-11
        @test collective_variance(rho,planx)≈1 atol=3e-11
        @test qfi(rho,planx)≈4 atol=1e-10
        @test qfim(rho,[planx,planz])≈[4.0 0.0;0.0 0.0] atol=1e-10
        @test wigner_yanase_asymmetry(rho,planx)≈1 atol=3e-11
        @test relative_entropy_of_asymmetry(rho,planx)≈1 atol=5e-11
        @test only(sector_resolved_qfi(rho,planx)).contribution≈4 atol=1e-10
        if N==2100
            # Phase-space analysis uses the same bounded weighted block and
            # must not attempt to materialize the unrepresentable sqrt(f).
            # Use the actual j=0 sector here; `rho` above is j=1,m=0 and its
            # phase-space distributions are not uniform.
            singlet_partition=Partition((N÷2,N÷2))
            singlet_basis=PIBasis(N,2;sectors=[singlet_partition.parts])
            singlet=basis_state(singlet_basis,singlet_partition,
                                only(only(singlet_basis.patterns)))
            singlet_q=spin_husimi_q(singlet,[0.0,1.0],[0.0])
            singlet_w=spin_wigner(singlet,[0.0,1.0],[0.0])
            @test singlet_q.values≈fill(1/(4pi),1,2) atol=2e-15
            @test singlet_w.values≈fill(1/(4pi),1,2) atol=2e-15
        end
        if N<=1100
            # Beyond this point the physical generator is perfectly finite,
            # but its stored equation-(7) coefficient is not representable in
            # Float64. Prepared physical blocks remain the intended API.
            @test wigner_yanase_asymmetry(
                rho,collective_operator(planx))≈1 atol=3e-11
        end
        @test one_body_rdm(rho;cache=geometry)≈Matrix{ComplexF64}(I,2,2)/2 atol=3e-11
    end

    # `atol` is a global state-validation tolerance, not a cutoff applied to
    # every eigenvalue.  This sector has probability below the default atol,
    # but a large (still finite) generator gives it an O(1) QFI contribution.
    # Denominator rank decisions must therefore be relative to the sector.
    basis=PIBasis(4,2;sectors=[(4,0),(3,1)])
    symmetric,small=basis.sectors
    delta=1e-16
    rho=PIState(basis;T=Float64)
    coefficient_block(rho,symmetric)[1,1]=1-delta
    coefficient_block(rho,small)[1,1]=delta/sqrt(3)
    generator_blocks=[
        symmetric=>zeros(ComplexF64,5,5),
        small=>ComplexF64[0 1e8 0;1e8 0 0;0 0 0],
    ]
    generator=operator_from_schur_blocks(basis,generator_blocks)
    @test qfi(rho,generator)≈4 atol=2e-12
    @test qfim(rho,[generator])[1,1]≈4 atol=2e-12

    # Exercise the exact fallback directly: these pair counts overflow Int64,
    # while the normalized Float64 results are benign.
    huge_N=10^12
    ordered=Float64(huge_N)*Float64(huge_N-1)
    square=Float64(huge_N)^2
    @test PermutationalInvariantDynamics._divide_by_particle_pair_factor(
        ordered,huge_N;ordered_distinct=true,context="test")≈1
    @test PermutationalInvariantDynamics._divide_by_particle_pair_factor(
        square,huge_N;ordered_distinct=false,context="test")≈1
    @test PermutationalInvariantDynamics._divide_by_particle_pair_factor(
        Float16(1),1000;ordered_distinct=true,context="test")≈
        Float16(1/(1000*999)) rtol=Float16(1e-3)
    nonexact32=Int(2^24+1)
    @test last(PermutationalInvariantDynamics._exact_particle_count(
        Float32,Int(2^24)))
    @test !last(PermutationalInvariantDynamics._exact_particle_count(
        Float32,nonexact32))
    @test last(PermutationalInvariantDynamics._exact_particle_count(
        Float32,Int(2^24+2)))
    @test !last(PermutationalInvariantDynamics._exact_particle_count(
        Float32,big(nonexact32)))
    exact_ordered=big(nonexact32)*big(nonexact32-1)
    exact_square=big(nonexact32)^2
    @test PermutationalInvariantDynamics._divide_by_particle_pair_factor(
        1.0f0,nonexact32;ordered_distinct=true,context="test")===
        Float32(big(1)//exact_ordered)
    @test PermutationalInvariantDynamics._divide_by_particle_pair_factor(
        1.0f0,nonexact32;ordered_distinct=false,context="test")===
        Float32(big(1)//exact_square)
    @test PermutationalInvariantDynamics._inverse_particle_count(
        Float16,100_000)===Float16(big(1)//big(100_000))
    @test PermutationalInvariantDynamics._inverse_particle_count(
        Float32,nonexact32)===Float32(big(1)//big(nonexact32))
    count_probe=prevfloat(1.0f0)
    @test PermutationalInvariantDynamics._divide_by_particle_count(
        count_probe,nonexact32;prefactor=4,context="test")===
        Float32(Rational{BigInt}(count_probe)*4//big(nonexact32))
    @test PermutationalInvariantDynamics._multiply_by_particle_count(
        count_probe,nonexact32;context="test")===
        Float32(Rational{BigInt}(count_probe)*big(nonexact32))

    large_log_partition=Partition((35_001,34_999))
    large_log_multiplicity=symmetric_group_dimension(large_log_partition)
    expected_log16=Float16(setprecision(256) do
        log(BigFloat(large_log_multiplicity))
    end)
    @test isfinite(expected_log16)
    @test PermutationalInvariantDynamics._log_schur_multiplicity(
        Float16,large_log_partition)===expected_log16

    float32_basis=PIBasis(3,2)
    float32_state=iid_pure_state(float32_basis,ComplexF32[1,0])
    float32_geometry=PermutationalInvariantDynamics._qfim_shared_geometry(
        float32_state,[ComplexF32[0 1;1 0]])
    @test PermutationalInvariantDynamics.geometry_scalar_type(
        float32_geometry)===Float32

    # Binary search must reproduce the former exhaustive scan exactly on
    # small systems, then remain effectively constant-cost at huge N.
    old_depth(F,range,N,atol)=begin
        depth=1
        for k in 1:N-1
            s=fld(N,k)
            r=N-s*k
            F>(s*k^2+r^2)*range^2+atol&&(depth=k+1)
        end
        depth
    end
    for N in 1:80
        for F in (0.0,N/2,N,2N,nextfloat(Float64(N^2)))
            @test PermutationalInvariantDynamics._qfi_entanglement_depth_from_value(
                F,1.0,N,1e-12)==old_depth(F,1.0,N,1e-12)
        end
    end
    @test PermutationalInvariantDynamics._qfi_entanglement_depth_from_value(
        2.0e12,1.0,10^12,0.0)==2
end
