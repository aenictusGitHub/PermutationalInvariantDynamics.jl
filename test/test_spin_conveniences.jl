@testset "spin, Dicke-state, and six-rate conveniences" begin
    @testset "local and collective spin algebra" begin
        for (dimension,R) in ((2,Float32),(3,Float64),(4,BigFloat))
            spin=spin_matrices(dimension;T=R)
            tolerance=R(200)*eps(R)
            @test spin.j==R(dimension-1)/R(2)
            @test eltype(spin.jx)===Complex{R}
            @test spin.jm≈spin.jp' atol=tolerance rtol=tolerance
            @test isapprox(spin.jx*spin.jy-spin.jy*spin.jx,im*spin.jz;
                           atol=tolerance,rtol=tolerance)
            casimir=spin.jx*spin.jx+spin.jy*spin.jy+spin.jz*spin.jz
            @test isapprox(casimir,spin.j*(spin.j+one(R))*I;
                           atol=tolerance,rtol=tolerance)
            @test isapprox(real.(diag(spin.jz)),
                           collect(-spin.j:one(R):spin.j);
                           atol=tolerance,rtol=tolerance)
        end
        qubit=spin_matrices()
        @test qubit.jm==ComplexF64[0 1;0 0]
        @test qubit.jz==ComplexF64[-0.5 0;0 0.5]
        @test_throws ArgumentError spin_matrices(0)
        @test_throws ArgumentError spin_matrices(big(typemax(Int))+1)

        for (N,d) in ((3,2),(2,3))
            basis=PIBasis(N,d)
            cache=OneBodyGeometry(basis)
            spin=spin_matrices(d)
            for (component,local_operator) in
                    ((:x,spin.jx),(:y,spin.jy),(:z,spin.jz),
                     (:plus,spin.jp),(:minus,spin.jm))
                @test isapprox(
                    collective_spin(basis,component;cache=cache).data,
                    collective_operator(basis,local_operator;cache=cache).data;
                    atol=3e-12,rtol=3e-12)
            end
            @test collective_spin(basis,Symbol("+");cache=cache).data≈
                  collective_spin(basis,:plus;cache=cache).data
            @test collective_spin(basis,Symbol("-");cache=cache).data≈
                  collective_spin(basis,:minus;cache=cache).data
            @test_throws ArgumentError collective_spin(basis,:invalid;cache=cache)
        end
    end

    @testset "state constructors" begin
        qutrit_basis=PIBasis(3,3)
        qutrit_level=computational_product_state(qutrit_basis,2;T=Float32)
        @test eltype(qutrit_level.data)===ComplexF32
        @test qutrit_level.data≈iid_pure_state(
            qutrit_basis,ComplexF32[0,1,0]).data atol=2f-6 rtol=2f-6
        @test_throws ArgumentError computational_product_state(qutrit_basis,0)
        @test_throws ArgumentError computational_product_state(qutrit_basis,4)

        N=4
        basis=PIBasis(N,2)
        state=dicke_state(basis,1,0)
        partition=Partition((3,1))
        multiplicity=symmetric_group_dimension(partition)
        @test trace(state)≈1 atol=2e-14
        @test sector_population(state,partition)≈1 atol=2e-14
        @test purity(state)≈1/multiplicity atol=2e-14
        @test expectation(state,collective_spin(basis,:z))≈0 atol=2e-12
        selected=findfirst(pattern->content(pattern)==(2,2),
                           basis.patterns[basis.index[partition]])
        @test physical_block(state,partition)[selected,selected]≈
              inv(multiplicity) atol=2e-14

        ground=computational_product_state(basis,1)
        excited=computational_product_state(basis,2)
        @test dicke_state(basis,N/2,-N/2).data≈ground.data atol=2e-14
        @test dicke_state(basis,N/2,N/2).data≈excited.data atol=2e-14
        diagonal_operator=dicke_operator(basis,1,0,0)
        @test diagonal_operator.data==state.data
        @test eltype(diagonal_operator.data)===eltype(state.data)
        coherence=dicke_operator(basis,1,1,0)
        reverse_coherence=dicke_operator(basis,1,0,1)
        @test adjoint(coherence).data==reverse_coherence.data
        @test physical_block(coherence,partition)[
            findfirst(pattern->content(pattern)==(1,3),
                      basis.patterns[basis.index[partition]]),
            findfirst(pattern->content(pattern)==(2,2),
                      basis.patterns[basis.index[partition]])]≈inv(multiplicity)
        @test_throws ArgumentError dicke_state(basis,0.25,0)
        @test_throws ArgumentError dicke_state(basis,1.5,0.5)
        @test_throws ArgumentError dicke_state(basis,1,1.5)
        @test_throws ArgumentError dicke_state(qutrit_basis,1,0)
        @test_throws ArgumentError dicke_operator(basis,1,0,1.5)
        @test_throws ArgumentError dicke_operator(basis,0.25,0,0)
        @test_throws ArgumentError dicke_operator(qutrit_basis,1,0,0)
        @test_throws ArgumentError dicke_state(PIBasis(2,2),typemin(Int),0)
        @test_throws ArgumentError dicke_state(PIBasis(2,2),1,typemin(Int))
        @test_throws ArgumentError dicke_operator(
            PIBasis(2,2),1,0,typemin(Int))
        symmetric_only=PIBasis(N,2;sectors=[(N,0)])
        @test_throws ArgumentError dicke_state(symmetric_only,1,0)

        huge_multiplicity=PIBasis(200,2;sectors=[(100,100)])
        large_dicke=dicke_state(huge_multiplicity,0,0;T=Float32)
        large_dicke_operator=dicke_operator(
            huge_multiplicity,0,0,0;T=Float32)
        @test all(isfinite,large_dicke.data)
        @test large_dicke.data==large_dicke_operator.data

        unrepresentable_multiplicity=PIBasis(400,2;sectors=[(200,200)])
        @test_throws ArgumentError dicke_state(
            unrepresentable_multiplicity,0,0;T=Float32)
        @test_throws ArgumentError dicke_operator(
            unrepresentable_multiplicity,0,0,0;T=Float32)

        phase=0.37
        ghz=ghz_state(basis;phase=phase)
        @test trace(ghz)≈1 atol=2e-14
        @test purity(ghz)≈1 atol=2e-14
        @test qfi(ghz,spin_matrices(2).jz)≈N^2 atol=2e-10
        @test_throws ArgumentError ghz_state(qutrit_basis)
        @test_throws ArgumentError ghz_state(PIBasis(0,2))

        theta=0.73
        phi=-0.41
        coherent=spin_coherent_state(basis,theta,phi)
        spin=spin_matrices(2)
        expected=(N/2).*(sin(theta)*cos(phi),sin(theta)*sin(phi),cos(theta))
        measured=(collective_expectation(coherent,spin.jx),
                  collective_expectation(coherent,spin.jy),
                  collective_expectation(coherent,spin.jz))
        @test collect(real.(measured))≈collect(expected) atol=3e-12 rtol=3e-12
        @test spin_coherent_state(basis,0,phi).data≈excited.data atol=2e-14
        @test spin_coherent_state(basis,pi,phi).data≈ground.data atol=2e-14
        coherent32=spin_coherent_state(PIBasis(2,2),Float32(0.4),
                                       Float32(-0.2);T=Float32)
        @test eltype(coherent32.data)===ComplexF32
        @test_throws ArgumentError spin_coherent_state(qutrit_basis,0.2,0.1)
        @test_throws ArgumentError spin_coherent_state(basis,Inf,0)
        @test_throws ArgumentError ghz_state(basis;phase=Inf)
        setprecision(128) do
            coherent_big=spin_coherent_state(
                PIBasis(2,2),big"0.4",big"-0.2")
            ghz_big=ghz_state(PIBasis(2,2);phase=big"0.3")
            @test eltype(coherent_big.data)===Complex{BigFloat}
            @test eltype(ghz_big.data)===Complex{BigFloat}
        end
        @test eltype(ghz_state(PIBasis(2,2);T=Float32).data)===ComplexF32
        @test_throws ArgumentError spin_coherent_state(
            PIBasis(2,2),0.4;T=Float32)
        @test_throws ArgumentError ghz_state(
            PIBasis(2,2);phase=0.3,T=Float32)
    end

    @testset "six-rate qubit ensemble" begin
        basis=PIBasis(3,2)
        rates=(emission=0.31,dephasing=0.17,pumping=0.09,
               collective_emission=0.07,collective_dephasing=0.05,
               collective_pumping=0.03)
        model=qubit_ensemble_model(basis;rates...)
        spin=spin_matrices(2)
        reference=PIModel(basis,(
            LocalJump(spin.jm;rate=rates.emission),
            LocalJump(spin.jz;rate=rates.dephasing),
            LocalJump(spin.jp;rate=rates.pumping),
            CollectiveJump(spin.jm;rate=rates.collective_emission),
            CollectiveJump(spin.jz;rate=rates.collective_dephasing),
            CollectiveJump(spin.jp;rate=rates.collective_pumping),
        ))
        @test length(model.terms)==6
        @test isapprox(Matrix(liouvillian(model;representation=:sparse)),
                       Matrix(liouvillian(reference;representation=:sparse));
                       atol=3e-12,rtol=3e-12)

        empty_model=qubit_ensemble_model(basis)
        @test isempty(empty_model.terms)
        restricted=PIBasis(4,2;sectors=[(4,0)])
        collective_only=qubit_ensemble_model(restricted;
            collective_emission=0.2)
        @test length(collective_only.terms)==1
        @test only(collective_only.terms) isa CollectiveJump
        @test_throws ArgumentError qubit_ensemble_model(restricted;emission=0.2)
        @test_throws ArgumentError qubit_ensemble_model(restricted;
            emission=(t,parameters)->zero(t))
        negative=qubit_ensemble_model(basis;emission=-0.2)
        @test only(negative.terms).rate==-0.2

        h=spin.jx
        matrix_hamiltonian=qubit_ensemble_model(basis;hamiltonian=h)
        matrix_reference=PIModel(basis,(CollectiveHamiltonian(h),))
        @test isapprox(
            Matrix(liouvillian(matrix_hamiltonian;representation=:sparse)),
            Matrix(liouvillian(matrix_reference;representation=:sparse));
            atol=3e-12,rtol=3e-12)
        H=collective_spin(basis,:z)
        direct_hamiltonian=qubit_ensemble_model(basis;hamiltonian=H)
        direct_reference=PIModel(basis,(DirectPIHamiltonian(H),))
        @test isapprox(
            Matrix(liouvillian(direct_hamiltonian;representation=:sparse)),
            Matrix(liouvillian(direct_reference;representation=:sparse));
            atol=3e-12,rtol=3e-12)
        nonhermitian=PIOperator(basis)
        coefficient_block(nonhermitian,first(basis.sectors))[1,2]=1
        @test_throws ArgumentError DirectPIHamiltonian(nonhermitian)
        @test DirectPIHamiltonian(nonhermitian;check=false) isa
              DirectPIHamiltonian
        @test_throws ArgumentError qubit_ensemble_model(
            basis;hamiltonian=nonhermitian)
        schedule=InPlaceTimeOperator(h,(destination,t,parameters)->begin
            destination .*= 1+t
        end)
        @test !isautonomous(qubit_ensemble_model(basis;hamiltonian=schedule))
        @test_throws ArgumentError qubit_ensemble_model(basis;
            hamiltonian=(t,parameters)->h)
        @test_throws DimensionMismatch qubit_ensemble_model(basis;
            hamiltonian=zeros(3,3))
        @test_throws ArgumentError qubit_ensemble_model(PIBasis(2,3))

        float32_model=qubit_ensemble_model(2;emission=0.2f0,T=Float32)
        @test eltype(liouvillian(float32_model;representation=:sparse))===ComplexF32
        inferred_float32=qubit_ensemble_model(2;emission=0.2f0)
        @test eltype(liouvillian(inferred_float32;representation=:sparse))===ComplexF32

        # Since the six-rate dephasing jump is jz=sigma_z/2, a one-qubit
        # coherence obeys d rho_ge / dt = -(gamma/2) rho_ge.
        gamma=0.6
        one_qubit=PIBasis(1,2)
        plus=spin_coherent_state(one_qubit,pi/2,0)
        derivative=PIState(one_qubit,
            liouvillian(qubit_ensemble_model(one_qubit;dephasing=gamma);
                        representation=:sparse)*plus.data)
        block=physical_block(derivative,only(one_qubit.sectors))
        @test block[1,2]≈-gamma/4 atol=2e-14
        @test block[2,1]≈-gamma/4 atol=2e-14
    end
end
