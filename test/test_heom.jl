@testset "PI hierarchy equations of motion" begin
    PIDHEOM=PermutationalInvariantDynamics

    @testset "validation and hierarchy metadata" begin
        basis=PIBasis(1,2)
        spin=spin_matrices()
        Qz=collective_operator(basis,spin.jz)
        Qx=collective_operator(basis,spin.jx)

        @test_throws DimensionMismatch PIDHEOM.HEOMBath(Qz,[0.2],[1.0,2.0])
        @test_throws ArgumentError PIDHEOM.HEOMBath(Qz,Float64[],Float64[])
        @test_throws ArgumentError PIDHEOM.HEOMBath(
            collective_operator(basis,spin.jp),0.2,1.0)
        @test_throws ArgumentError PIDHEOM.HEOMBath(Qz,NaN,1.0)
        @test_throws ArgumentError PIDHEOM.HEOMBath(Qz,0.2,0.0)
        @test_throws ArgumentError PIDHEOM.HEOMBath(Qz,0.2,-1.0+0.2im)
        @test_throws ArgumentError PIDHEOM.HEOMBath(Qz,0.2,1.0;atol=Inf)
        @test_throws ArgumentError PIDHEOM.HEOMBath(Qz,0.2,1.0;rtol=Inf)
        @test_throws DimensionMismatch PIDHEOM.HEOMBath(
            Qz,[0.2],[1.0];right_coefficients=[0.1,0.2])
        @test_throws ArgumentError PIDHEOM.HEOMBath(
            Qz,[0.2],[1.0];right_coefficients=[NaN])

        completed=PIDHEOM.HEOMBath(Qz,0.3,1.2+0.7im)
        @test completed.frequencies==[1.2+0.7im,1.2-0.7im]
        @test completed.coefficients==[0.3,0.0]
        @test completed.right_coefficients==[0.0,0.3]
        paired=PIDHEOM.HEOMBath(Qz,[0.3,0.1im],
            [1.2+0.7im,1.2-0.7im])
        @test paired.right_coefficients==[-0.1im,0.3]
        explicit=PIDHEOM.HEOMBath(Qz,0.3,1.2+0.7im;
            right_coefficients=0.17-0.02im)
        @test explicit.right_coefficients==[0.17-0.02im]
        @test length(explicit.frequencies)==1

        bath_z=PIDHEOM.HEOMBath(Qz,0.2+0.04im,1.1+0.3im;
            right_coefficients=0.2-0.04im)
        bath_x=PIDHEOM.HEOMBath(Qx,0.07,0.8)
        model=PIModel(basis,())
        plan=PIDHEOM.HEOMPlan(model,(bath_z,bath_x);max_depth=2)
        @test PIDHEOM.heom_number_ados(plan)==6
        @test PIDHEOM.heom_multiindices(plan)==
              [[0,0],[0,1],[1,0],[0,2],[1,1],[2,0]]
        detached=PIDHEOM.heom_multiindices(plan)
        detached[1]=[4,4]
        @test plan.multiindices[1]==[0,0]
        @test size(plan)==(6length(basis),6length(basis))
        @test eltype(plan)===ComplexF64
        @test isautonomous(plan)
        @test_throws ArgumentError PIDHEOM.HEOMPlan(
            model,bath_z;max_depth=-1)
        @test_throws ArgumentError PIDHEOM.HEOMPlan(
            model,bath_z;max_depth=1,terminator=:markovian)
        other_basis=PIBasis(1,2)
        @test_throws ArgumentError PIDHEOM.HEOMPlan(
            model,PIDHEOM.HEOMBath(
                collective_operator(other_basis,spin.jz),0.2,1.0);
            max_depth=1)
        @test_throws ArgumentError PIDHEOM.HEOMPlan(
            zeros(ComplexF64,length(basis),length(basis)),bath_z;
            max_depth=1)
        matrix_plan=PIDHEOM.HEOMPlan(
            zeros(ComplexF64,length(basis),length(basis)),bath_z;
            basis,max_depth=1)
        @test size(matrix_plan,1)==2length(basis)
        raw_system=zeros(ComplexF64,length(basis),length(basis))
        copied_plan=PIDHEOM.HEOMPlan(raw_system,bath_z;
            basis,max_depth=1)
        raw_system[1,1]=1
        @test iszero(copied_plan.system[1,1])
        @test_throws ArgumentError PIDHEOM.HEOMPlan(
            model,PIDHEOM.HEOMBath(Qz,0.2,1e308);max_depth=2)
        @test_throws ArgumentError PIDHEOM.HEOMPlan(
            model,PIDHEOM.HEOMBath(Qz,1e308,1.0);max_depth=2)
        inexact_integer=Int64(9_007_199_254_740_993)
        @test_throws ArgumentError PIDHEOM.HEOMPlan(model,
            PIDHEOM.HEOMBath(Qz,inexact_integer,1);max_depth=0)
        @test_throws ArgumentError PIDHEOM.HEOMPlan(model,
            PIDHEOM.HEOMBath(Qz,1,1;
                right_coefficients=inexact_integer);max_depth=0)
        @test_throws ArgumentError PIDHEOM.HEOMPlan(model,
            PIDHEOM.HEOMBath(Qz,1,inexact_integer);max_depth=0)
        empty_basis=PIBasis(0,2;sectors=Tuple{Int,Int}[])
        empty_coupling=PIOperator(empty_basis;T=Float64)
        @test_throws ArgumentError PIDHEOM.HEOMPlan(
            zeros(ComplexF64,0,0),
            PIDHEOM.HEOMBath(empty_coupling,0.2,1.0);
            basis=empty_basis,max_depth=1)

        spin32=spin_matrices(2;T=Float32)
        coupling32=collective_operator(
            basis,spin32.jz;cache=OneBodyGeometry(basis,Float32))
        model32=PIModel(basis,(
            LocalHamiltonian(spin32.jz;rate=0.0f0),))
        plan32=PIDHEOM.HEOMPlan(model32,
            PIDHEOM.HEOMBath(coupling32,0.2f0,1.1f0);max_depth=1)
        state32=PIDHEOM.heom_evolve(plan32,
            iid_pure_state(basis,ComplexF32[1,1]/sqrt(2.0f0)),
            (0.0f0,0.01f0);steps=2)
        @test eltype(plan32)===ComplexF32
        @test eltype(state32.data)===ComplexF32
        @test eltype(PIDHEOM.heom_reduced_state(state32).data)===ComplexF32
        integer_bath_plan=PIDHEOM.HEOMPlan(model32,
            PIDHEOM.HEOMBath(coupling32,1,2);max_depth=1)
        @test eltype(integer_bath_plan)===ComplexF32
        @test PIDHEOM._fixed_liouvillian_scalar_type(
            PIDHEOM.heom_liouvillian(integer_bath_plan))===ComplexF32
        integer_initial=PIDHEOM.heom_initial_state(integer_bath_plan,
            iid_pure_state(basis,ComplexF32[1,0]))
        @test_throws ArgumentError PIDHEOM.heom_evolve!(
            zeros(ComplexF32,length(integer_initial)),integer_bath_plan,
            ComplexF64.(integer_initial.data),(0.0f0,0.01f0);steps=1)
        @test_throws ArgumentError PIDHEOM.heom_evolve!(
            copy(integer_initial.data),integer_bath_plan,integer_initial.data,
            (0.0f0,0.01f0);steps=16_777_217)
    end

    @testset "matrix-free action against a direct hierarchy" begin
        basis=PIBasis(1,2)
        spin=spin_matrices()
        coupling=collective_operator(basis,spin.jz)
        system=liouvillian(PIModel(basis,(
            LocalHamiltonian(spin.jx;rate=0.17),));representation=:sparse)
        coefficients=ComplexF64[0.21+0.06im,0.04-0.02im]
        frequencies=ComplexF64[1.3+0.2im,2.1-0.4im]
        right_coefficients=ComplexF64[0.17-0.01im,0.03+0.05im]
        bath=PIDHEOM.HEOMBath(coupling,coefficients,frequencies;
            right_coefficients)
        plan=PIDHEOM.HEOMPlan(system,bath;basis,max_depth=2)
        workspace=PIDHEOM.HEOMWorkspace(plan)
        dimension=size(plan,1)
        matrixfree=zeros(ComplexF64,dimension,dimension)
        source=zeros(ComplexF64,dimension)
        for column in 1:dimension
            fill!(source,0);source[column]=1
            apply!(view(matrixfree,:,column),plan,source,0.0,nothing,workspace)
        end

        npi=length(basis);number_ados=PIDHEOM.heom_number_ados(plan)
        Q=physical_block(coupling,basis.sectors[1])
        left=left_superoperator(Q);right=right_superoperator(Q)
        commutator=-im*(left-right)
        reference=zeros(ComplexF64,dimension,dimension)
        identity_pi=Matrix{ComplexF64}(I,npi,npi)
        for ado in 1:number_ados
            target=(ado-1)*npi+1:ado*npi
            reference[target,target].=system-plan.decays[ado]*identity_pi
            occupations=plan.multiindices[ado]
            for term in eachindex(coefficients)
                upper=plan.upward[ado,term]
                if !iszero(upper)
                    columns=(upper-1)*npi+1:upper*npi
                    reference[target,columns].+=commutator
                end
                lower=plan.downward[ado,term]
                if occupations[term]>0
                    columns=(lower-1)*npi+1:lower*npi
                    reference[target,columns].+=-im*occupations[term].*(
                        coefficients[term]*left-
                        right_coefficients[term]*right)
                end
            end
        end
        @test matrixfree≈reference atol=2e-14 rtol=2e-14

        rng=MersenneTwister(0x4e10)
        random_hierarchy=randn(rng,ComplexF64,dimension)
        image=similar(random_hierarchy)
        apply!(image,plan,random_hierarchy,0.0,nothing,workspace)
        @test image≈reference*random_hierarchy atol=5e-14 rtol=5e-14
        @test abs(dot(plan.tracevec,image))<5e-14
        @test (@allocated apply!(
            image,plan,random_hierarchy,0.0,nothing,workspace))<=512
        @test_throws ArgumentError apply!(
            random_hierarchy,plan,random_hierarchy,0.0,nothing,workspace)
        overlapping=zeros(ComplexF64,dimension+1)
        @test_throws ArgumentError apply!(view(overlapping,2:dimension+1),
            plan,view(overlapping,1:dimension),0.0,nothing,workspace)
        adapter=PIDHEOM.heom_liouvillian(plan)
        @test size(adapter)==(dimension,dimension)
        batch_source=hcat(random_hierarchy,2random_hierarchy)
        batch_image=similar(batch_source)
        mul!(batch_image,adapter,batch_source)
        @test batch_image≈reference*batch_source atol=8e-14 rtol=8e-14

        # Nontrivial symmetric-group multiplicities exercise the physical
        # C_Q/sqrt(f) coupling scale sector by sector.
        many_basis=PIBasis(3,2)
        many_coupling=collective_operator(many_basis,spin.jx)
        many_plan=PIDHEOM.HEOMPlan(PIModel(many_basis,()),
            PIDHEOM.HEOMBath(many_coupling,0.2,1.0);max_depth=1)
        random_operator=PIOperator(many_basis,
            randn(rng,ComplexF64,length(many_basis)))
        many_source=zeros(ComplexF64,size(many_plan,1))
        many_source[length(many_basis)+1:end].=random_operator.data
        many_image=similar(many_source)
        apply!(many_image,many_plan,many_source,
            PIDHEOM.HEOMWorkspace(many_plan))
        expected=-im*(many_coupling*random_operator-
                      random_operator*many_coupling)
        @test view(many_image,1:length(many_basis))≈expected.data atol=3e-13 rtol=3e-13
    end

    @testset "pure-dephasing reference and extraction" begin
        basis=PIBasis(1,2)
        coupling=collective_operator(basis,spin_matrices().jz)
        coefficient=0.3;frequency=1.2;final_time=0.7
        bath=PIDHEOM.HEOMBath(coupling,coefficient,frequency)
        initial=iid_pure_state(basis,ComplexF64[1,1]/sqrt(2))

        shallow=PIDHEOM.HEOMPlan(PIModel(basis,()),bath;max_depth=2)
        deep=PIDHEOM.HEOMPlan(PIModel(basis,()),bath;max_depth=4)
        shallow_state=PIDHEOM.heom_evolve(
            shallow,initial,(0.0,final_time);steps=100)
        deep_state=PIDHEOM.heom_evolve(
            deep,initial,(0.0,final_time);steps=100)
        exact_coherence=0.5exp(-coefficient/frequency^2*(
            frequency*final_time-1+exp(-frequency*final_time)))
        shallow_coherence=abs(physical_block(
            PIDHEOM.heom_reduced_state(shallow_state),basis.sectors[1])[1,2])
        deep_coherence=abs(physical_block(
            PIDHEOM.heom_reduced_state(deep_state),basis.sectors[1])[1,2])
        @test abs(deep_coherence-exact_coherence)<3e-9
        @test abs(deep_coherence-exact_coherence)<
              abs(shallow_coherence-exact_coherence)
        @test trace(PIDHEOM.heom_reduced_state(deep_state))≈1 atol=2e-13
        @test PIDHEOM.heom_ado(deep_state,[1]) isa PIOperator
        @test_throws ArgumentError PIDHEOM.heom_ado(deep_state,[5])
        @test_throws DimensionMismatch PIDHEOM.heom_ado(deep_state,[1,0])

        # A complex pole is automatically completed by its conjugate in the
        # right correlation. The physical root must remain Hermitian even
        # though individual auxiliary ADOs need not be.
        complex_bath=PIDHEOM.HEOMBath(coupling,0.3,1.2+0.7im)
        complex_plan=PIDHEOM.HEOMPlan(
            PIModel(basis,()),complex_bath;max_depth=4)
        complex_state=PIDHEOM.heom_evolve(
            complex_plan,initial,(0.0,0.3);steps=120)
        complex_root=PIDHEOM.heom_reduced_state(complex_state)
        @test hermiticity_error(complex_root)<3e-12
        @test trace(complex_root)≈1 atol=3e-13

        # Unequal squared coupling eigenvalues make the complex bath-induced
        # phase sensitive to the left/right coefficient placement, not only
        # to their Hermiticity-preserving combination.
        projector=collective_operator(
            basis,ComplexF64[0 0;0 1])
        phase_coefficient=0.21+0.08im
        phase_frequency=1.2+0.7im
        phase_time=0.2
        phase_plan=PIDHEOM.HEOMPlan(PIModel(basis,()),
            PIDHEOM.HEOMBath(projector,phase_coefficient,phase_frequency);
            max_depth=4)
        phase_root=PIDHEOM.heom_reduced_state(PIDHEOM.heom_evolve(
            phase_plan,initial,(0.0,phase_time);steps=80))
        projector_block=physical_block(projector,basis.sectors[1])
        q_left=real(projector_block[1,1])
        q_right=real(projector_block[2,2])
        phase=phase_coefficient*(phase_frequency*phase_time-1+
            exp(-phase_frequency*phase_time))/phase_frequency^2
        initial_coherence=physical_block(initial,basis.sectors[1])[1,2]
        expected_coherence=initial_coherence*exp(-(q_left-q_right)*(
            q_left*phase-q_right*conj(phase)))
        @test physical_block(phase_root,basis.sectors[1])[1,2]≈
              expected_coherence atol=3e-14 rtol=0

        saved=PIDHEOM.heom_time_evolution(
            deep,initial,[0.0,0.2,0.2,0.4];steps_per_interval=20)
        @test length(saved)==4
        @test saved[2].data==saved[3].data
        @test saved[2].data!==saved[3].data
        @test_throws ArgumentError PIDHEOM.heom_time_evolution(
            deep,initial,[NaN];steps_per_interval=1)

        convergence=PIDHEOM.heom_depth_convergence(
            PIModel(basis,()),bath,initial,(0.0,final_time);
            depths=(2,4,6),steps=100,atol=1e-8,rtol=0,consecutive=1)
        @test convergence isa ConvergenceStudyResult
        @test convergence.refinements==[2,4,6]
        @test getproperty.(convergence.results,:ado_count)==[3,5,7]
        @test getproperty.(convergence.results,:dimension)==
              length(basis).*[3,5,7]
        differences=collect(skipmissing(convergence.pairwise_errors))
        @test differences[2]<differences[1]
        @test isequal(convergence.pairwise_converged,
                      Union{Missing,Bool}[missing,false,true])
        @test convergence.converged
        @test maximum(getproperty.(convergence.results,:trace_error))<2e-13
        @test all(getproperty.(convergence.results,:system_prepared_once))
        @test all(getproperty.(convergence.results,:coupling_blocks_shared))
        @test PIDHEOM.heom_reduced_state(
            last(convergence.results).hierarchy).data≈
            convergence.estimates[end].data atol=0 rtol=0
        @test occursin("converged=true",sprint(show,convergence))
        template_report=PIDHEOM.heom_depth_convergence(
            deep,initial,(0.0,final_time);depths=(2,4),steps=100,
            consecutive=1)
        @test all(getproperty.(
            template_report.results,:template_max_depth).==4)
        @test last(template_report.results).hierarchy.plan===deep
        @test_throws ArgumentError PIDHEOM.heom_depth_convergence(
            deep,initial,(0.0,final_time);depths=(2,6),steps=10)
        unnormalized=PIState(basis,2 .* initial.data)
        unnormalized_report=PIDHEOM.heom_depth_convergence(
            PIModel(basis,()),bath,unnormalized,(0.0,0.0);
            depths=(0,1),steps=1,atol=0,rtol=0,consecutive=1)
        @test all(getproperty.(unnormalized_report.results,:initial_trace).≈2)
        @test maximum(getproperty.(
            unnormalized_report.results,:trace_error))<2e-14
        @test unnormalized_report.converged
        @test_throws ArgumentError PIDHEOM.heom_depth_convergence(
            PIModel(basis,()),bath,initial,(0.0,final_time);
            depths=(2,),steps=10)
        @test_throws ArgumentError PIDHEOM.heom_depth_convergence(
            PIModel(basis,()),bath,initial,(0.0,final_time);
            depths=(2,2),steps=10)
        @test_throws ArgumentError PIDHEOM.heom_depth_convergence(
            PIModel(basis,()),bath,initial,(0.0,final_time);
            depths=(0,1),steps=20,atol=0,rtol=0,
            consecutive=1,require_convergence=true)
    end

    @testset "zero coupling and matrix-free stationary hierarchy" begin
        basis=PIBasis(1,2)
        zero_coupling=PIOperator(basis;T=Float64)
        model=qubit_ensemble_model(basis;emission=0.7)
        bath=PIDHEOM.HEOMBath(zero_coupling,0.3,1.2)
        plan=PIDHEOM.HEOMPlan(model,bath;max_depth=2)
        initial=iid_pure_state(basis,ComplexF64[0,1])
        hierarchy=PIDHEOM.heom_initial_state(plan,initial)
        @test PIDHEOM.heom_reduced_state(hierarchy).data==initial.data
        @test iszero(norm(PIDHEOM.heom_ado(hierarchy,[1]).data))
        @test_throws ArgumentError PIDHEOM.heom_evolve!(
            zeros(ComplexF32,length(hierarchy)),plan,hierarchy.data,
            (0.0,0.001);steps=1)

        evolution_workspace=PIDHEOM.HEOMEvolutionWorkspace(plan)
        allocation_destination=copy(hierarchy.data)
        PIDHEOM.heom_evolve!(allocation_destination,plan,hierarchy.data,
            (0.0,0.001);steps=1,workspace=evolution_workspace)
        # Julia 1.10 retains about 0.9 KiB of top-level keyword/testset
        # bookkeeping beyond the roughly 2 KiB local hot-call cost.  This is
        # scalar metadata only; all hierarchy-sized RK4 arrays are reused.
        allocation_limit=VERSION<v"1.11" ? 4096 : 2048
        @test (@allocated heom_evolve!(
            allocation_destination,plan,hierarchy.data,(0.0,0.001);
            steps=1,workspace=evolution_workspace))<=allocation_limit

        evolved=PIDHEOM.heom_evolve(
            plan,hierarchy,(0.0,0.4);steps=80)
        direct=time_evolve(model,initial,(0.0,0.4);steps=80)
        @test PIDHEOM.heom_reduced_state(evolved).data==direct.data
        @test iszero(norm(PIDHEOM.heom_ado(evolved,[1]).data))

        stationary=PIDHEOM.heom_steady_state(
            plan;krylovdim=10,maxiter=100,atol=1e-11,rtol=1e-9)
        reduced=PIDHEOM.heom_reduced_state(stationary)
        @test trace(reduced)≈1 atol=2e-12
        @test iszero(norm(PIDHEOM.heom_ado(stationary,[1]).data))
        system_stationary=PIState(basis,krylov_steady_state(
            liouvillian(model;representation=:matrixfree);
            basis,krylovdim=8,maxiter=100,atol=1e-11,rtol=1e-9))
        @test reduced.data≈system_stationary.data atol=2e-11 rtol=2e-11

        information=PIDHEOM.heom_steady_state(
            plan;krylovdim=10,maxiter=100,atol=1e-11,rtol=1e-9,
            return_info=true)
        @test information.state isa PIDHEOM.HEOMState
        @test information.converged
    end
end
