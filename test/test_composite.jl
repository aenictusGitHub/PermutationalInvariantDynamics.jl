@testset "Composite PI operator spaces" begin
    bpi=PIBasis(1,2)
    baux=FiniteOperatorBasis(2;label=:cavity)
    basis=CompositePIBasis(bpi,baux)
    @test basis.dimensions==(length(bpi),4)
    @test length(basis)==length(bpi)*4
    @test pi_dimension(basis)==length(basis)

    rho_pi=iid_state(bpi,ComplexF64[0.7 0.1im;-0.1im 0.3])
    rho_aux=ComplexF64[0.6 0.2;0.2 0.4]
    rho=composite_tensor_state(basis,rho_pi,rho_aux)
    # First factor is fastest in composite operator coordinates.
    @test rho.data==kron(vec(rho_aux),rho_pi.data)
    @test trace(rho)≈trace(rho_pi)*LinearAlgebra.tr(rho_aux)
    @test purity(rho)≈purity(rho_pi)*real(LinearAlgebra.tr(rho_aux*rho_aux))

    Xpi=collective_operator(bpi,ComplexF64[0 1;1 0])
    Zaux=ComplexF64[1 0;0 -1]
    observable=composite_tensor_operator(basis,Xpi,Zaux)
    @test observable.data==kron(vec(Zaux),Xpi.data)
    @test expectation(rho,observable)≈
        expectation(rho_pi,Xpi)*LinearAlgebra.tr(adjoint(Zaux)*rho_aux)

    identity=composite_identity_operator(basis)
    @test expectation(rho,identity)≈trace(rho)
    @test trace(identity)≈4

    @test_throws ArgumentError CompositePIBasis()
    @test_throws DimensionMismatch composite_tensor_state(basis,rho_pi)
    @test_throws DimensionMismatch composite_tensor_state(basis,rho_pi,zeros(3,3))

    @testset "preallocated sum of Kronecker maps" begin
        A=ComplexF64[0.2 0.3im 0 0;
                     -0.1im -0.4 0 0;
                     0 0 0.5 0.2;
                     0 0 -0.3 0.1]
        B=ComplexF64[0.1 0.2 0 0;
                     0.3 -0.2 0 0;
                     0 0 0.4 -0.1im;
                     0 0 0.2im -0.5]
        C=ComplexF64[0.3 0 0 0;
                     0 -0.2 0.1 0;
                     0 0.4 0.5 0;
                     0 0 0 -0.1]
        local_term=local_superoperator_term(basis,1,A;coefficient=0.7)
        cross=factorized_superoperator_term(basis,1=>B,2=>C;
                                            coefficient=-0.25im)
        S=CompositeSuperoperator(basis,local_term,cross)
        @test_throws ArgumentError CompositeSuperoperator(
            basis,local_term;T=Float32)
        @test_throws ArgumentError CompositeSuperoperatorWorkspace(S;T=Float32)
        x=ComplexF64.(1:length(basis))./(length(basis)+1)
        y=similar(x)
        workspace=CompositeSuperoperatorWorkspace(S,x)
        apply!(y,S,x,0.0,nothing,workspace)
        reference=(0.7*kron(Matrix{ComplexF64}(I,4,4),A)-
                   0.25im*kron(C,B))*x
        @test y≈reference atol=2e-14 rtol=2e-14
        @test S*x≈reference atol=2e-14 rtol=2e-14
        @test_throws ArgumentError apply!(x,S,x,0.0,nothing,workspace)

        wrapped=composite_matrixfree(S)
        @test wrapped*x≈reference atol=2e-14 rtol=2e-14

        # A callable coefficient uses the explicit-time path and the same
        # preallocated tensor-mode storage.
        driven=factorized_superoperator_term(basis,2=>C;
                                             coefficient=(t,p)->p*t)
        Sd=CompositeSuperoperator(basis,driven)
        wd=CompositeSuperoperatorWorkspace(Sd,x)
        apply!(y,Sd,x,0.4,2.0,wd)
        @test y≈0.8*kron(C,Matrix{ComplexF64}(I,4,4))*x
        @test !isautonomous(Sd)
    end

    @testset "PI Liouvillian lift" begin
        sm=ComplexF64[0 1;0 0]
        model=PIModel(bpi,(LocalJump(sm;rate=0.35),))
        compiled=compile(model;backend=:matrixfree)
        lifted=local_superoperator_term(basis,1,compiled)
        S=CompositeSuperoperator(basis,lifted)
        @test_throws ArgumentError CompositeSuperoperatorWorkspace(S;T=BigFloat)
        x=copy(rho.data);y=similar(x)
        workspace=CompositeSuperoperatorWorkspace(S,x)
        apply!(y,S,x,0.0,nothing,workspace)
        Lpi=Matrix(liouvillian(model;representation=:sparse))
        @test y≈kron(Matrix{ComplexF64}(I,4,4),Lpi)*x atol=3e-13

        # Heterogeneous tuples previously fell back to runtime indexing and
        # allocated despite every numerical kernel being preallocated.
        cross=factorized_superoperator_term(
            basis,
            1=>factor_left_superoperator(
                bpi,collective_operator(bpi,ComplexF64[0 1;0 0])),
            2=>left_superoperator(ComplexF64[0 1;0 0]);
            coefficient=0.13,
        )
        combined=CompositeSuperoperator(basis,lifted,cross)
        combined_workspace=CompositeSuperoperatorWorkspace(combined,x)
        apply!(y,combined,x,0.0,nothing,combined_workspace)
        @test @allocated(apply!(y,combined,x,0.0,nothing,
                                combined_workspace))==0
    end

    @testset "cross-factor Hamiltonian and jump" begin
        # N=1 makes the PI factor physically identical to a two-level matrix,
        # while still exercising compressed PI-coordinate lifts.
        sx=ComplexF64[0 1;1 0]
        sm=ComplexF64[0 1;0 0]
        A=collective_operator(bpi,sx)
        H=composite_hamiltonian_superoperator(basis,1=>A,2=>sx;rate=0.3)
        D=composite_dissipator_superoperator(basis,1=>collective_operator(bpi,sm),
                                             2=>sm;rate=0.2)
        x=copy(rho.data);yh=similar(x);yd=similar(x)
        apply!(yh,H,x,0.0,nothing,CompositeSuperoperatorWorkspace(H,x))
        apply!(yd,D,x,0.0,nothing,CompositeSuperoperatorWorkspace(D,x))

        LA=factor_left_superoperator(bpi,A)
        RA=factor_right_superoperator(bpi,A)
        LB=left_superoperator(sx);RB=right_superoperator(sx)
        Href=-0.3im*(kron(LB,LA)-kron(RB,RA))
        @test yh≈Href*x atol=3e-13

        Jpi=collective_operator(bpi,sm)
        gain=kron(sandwich_superoperator(sm),
                  factor_sandwich_superoperator(bpi,Jpi))
        Qpi=adjoint(Jpi)*Jpi;Qaux=adjoint(sm)*sm
        Dref=0.2*(gain-
            (kron(left_superoperator(Qaux),factor_left_superoperator(bpi,Qpi))+
             kron(right_superoperator(Qaux),factor_right_superoperator(bpi,Qpi)))/2)
        @test yd≈Dref*x atol=3e-13
        @test_throws ArgumentError composite_hamiltonian_superoperator(
            basis,1=>collective_operator(bpi,sm),2=>sx)
        @test_throws ArgumentError composite_hamiltonian_superoperator(
            basis,1=>A,2=>sx;rate=1+im)
        @test_throws ArgumentError composite_dissipator_superoperator(
            basis,1=>Jpi;rate=1+im)

        # The generic RK4 layer discovers the composite workspace and retains
        # trace under a trace-preserving Hamiltonian-plus-jump generator.
        generator=H+D
        evolved=copy(x)
        evolution_workspace=EvolutionWorkspace(generator,x)
        evolve!(evolved,generator,x,(0.0,0.2);steps=16,
                workspace=evolution_workspace)
        @test trace(CompositePIState(basis,evolved))≈trace(rho) atol=2e-12
        evolved_state=time_evolve(generator,rho,(0.0,0.2);steps=16)
        @test evolved_state.data≈evolved atol=2e-14
        @test time_evolve(rho,generator,(0.0,0.2);steps=16).data≈
            evolved atol=2e-14
        sampled=time_evolution(generator,rho,[0.0,0.1,0.2];
                               steps_per_interval=8)
        @test sampled[end].data≈evolved atol=2e-14
        @test time_evolution(rho,generator,[0.0,0.1,0.2];
            steps_per_interval=8)[end].data≈evolved atol=2e-14
        @test pi_dimension(evolved_state)==length(basis)
        @test_throws ArgumentError CompositeSuperoperator(
            basis,first(generator.terms);T=Complex{Int})
    end


    @testset "multiple compressed PI factors" begin
        two_pi=CompositePIBasis(bpi,bpi)
        product=composite_tensor_state(two_pi,rho_pi,rho_pi)
        @test product.data==kron(rho_pi.data,rho_pi.data)
        @test trace(product)≈trace(rho_pi)^2
        localmap=Matrix(liouvillian(PIModel(bpi,(LocalJump(
            ComplexF64[0 1;0 0];rate=0.1),));representation=:sparse))
        term=local_superoperator_term(two_pi,2,localmap)
        S=CompositeSuperoperator(two_pi,term)
        @test S*product.data≈kron(localmap,
                                      Matrix{ComplexF64}(I,length(bpi),length(bpi)))*product.data

        other_basis=PIBasis(1,2)
        other_model=compile(PIModel(other_basis,(LocalJump(
            ComplexF64[0 1;0 0];rate=0.1),));backend=:matrixfree)
        @test_throws ArgumentError local_superoperator_term(two_pi,1,other_model)

        S2=CompositeSuperoperator(two_pi,
            local_superoperator_term(two_pi,1,localmap))
        @test_throws ArgumentError apply!(similar(product.data),S2,product.data,
            0.0,nothing,CompositeSuperoperatorWorkspace(S,product.data))
    end

    @testset "multi-sector PI factor" begin
        multi=PIBasis(2,2)
        @test length(multi.sectors)==2
        finite=FiniteOperatorBasis(2;label=:ancilla)
        composite=CompositePIBasis(multi,finite)
        local_density=ComplexF64[0.65 0.08im;-0.08im 0.35]
        multi_state=iid_state(multi,local_density)
        auxiliary_state=ComplexF64[0.7 0.1;0.1 0.3]
        state=composite_tensor_state(composite,multi_state,auxiliary_state)
        @test trace(state)≈trace(multi_state)*LinearAlgebra.tr(auxiliary_state)

        identity=composite_identity_operator(composite)
        @test expectation(state,identity)≈trace(state)
        @test trace(identity)≈8

        sm=ComplexF64[0 1;0 0]
        model=PIModel(multi,(LocalJump(sm;rate=0.17),))
        compiled=compile(model;backend=:matrixfree)
        lifted=CompositeSuperoperator(composite,
            local_superoperator_term(composite,1,compiled))
        destination=similar(state.data)
        workspace=CompositeSuperoperatorWorkspace(lifted,state.data)
        apply!(destination,lifted,state.data,0.0,nothing,workspace)
        dense_local=Matrix(liouvillian(model;representation=:sparse))
        reference=kron(Matrix{ComplexF64}(I,length(finite),length(finite)),
                       dense_local)*state.data
        @test destination≈reference atol=5e-13 rtol=5e-13
        @test abs(dot(composite_trace_vector(composite),destination))<5e-13
    end

    ambiguities=Test.detect_ambiguities(PermutationalInvariantDynamics;
                                        recursive=true)
    @test !any(pair->occursin("CompositePIState",sprint(show,pair)),ambiguities)
end
