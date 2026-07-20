@testset "PI hierarchy equations of motion" begin
    PIDHEOM=PermutationalInvariantDynamics

    @testset "physical bath decompositions and local pseudomodes" begin
        basis=PIBasis(1,2)
        spin=spin_matrices()
        coupling=collective_operator(basis,spin.jz)

        drude=PIDHEOM.drude_lorentz_bath(
            coupling,0.3,0.8,2.0;matsubara_terms=2)
        @test drude.metadata.model===:drude_lorentz
        @test drude.metadata.decomposition===:matsubara
        @test drude.coefficients≈ComplexF64[
            0.23309150415611382-0.24im,
            0.16338343532309116,
            0.0776532395879031] rtol=2e-15
        @test drude.frequencies≈ComplexF64[
            0.8,pi,2pi] rtol=2e-15
        @test drude.residue≈0.019270160175333784 rtol=2e-14
        @test PIDHEOM.heom_bath_residue(drude)===drude.residue
        @test PIDHEOM.heom_bath_metadata(drude)===drude.metadata

        pade=PIDHEOM.drude_lorentz_bath(
            coupling,0.3,0.8,2.0;matsubara_terms=2,
            decomposition=:pade)
        @test pade.metadata.decomposition===:pade
        # Independent numerical references from the Hu--Xu--Yan
        # tridiagonal-eigenvalue construction.
        @test pade.coefficients≈ComplexF64[
            0.23309150415611382-0.24im,
            0.16805352709422305,
            0.29576569395822155] rtol=3e-14
        @test pade.frequencies≈ComplexF64[
            0.8,3.1529695721124047,9.749809376461332] rtol=3e-14
        @test pade.residue≈4.799054986737161e-9 rtol=2e-7
        @test_throws ArgumentError PIDHEOM.drude_lorentz_bath(
            coupling,0.3,0.8,2.0;matsubara_terms=2,
            decomposition=:pade,memory_budget=1)

        brownian=PIDHEOM.underdamped_brownian_bath(
            coupling,0.7,0.4,1.2,2.0;matsubara_terms=2)
        @test brownian.metadata.model===:underdamped_brownian
        @test brownian.coefficients≈ComplexF64[
            0.019222556499959145-0.00904883790257897im,
            0.2262853489084457+0.00904883790257897im,
            -0.004874232845971628,
            -0.0007383117865740605] rtol=4e-14
        @test brownian.frequencies≈ComplexF64[
            0.2-1.1832159566199232im,
            0.2+1.1832159566199232im,pi,2pi] rtol=3e-14
        @test brownian.residue≈-3.8952746996584464e-5 rtol=3e-12
        @test brownian.right_coefficients[1]==conj(brownian.coefficients[2])
        @test brownian.right_coefficients[2]==conj(brownian.coefficients[1])

        coupling32=PIOperator(basis,ComplexF32.(coupling.data))
        drude32=PIDHEOM.drude_lorentz_bath(
            coupling32,0.3f0,0.8f0,2.0f0;matsubara_terms=1)
        @test eltype(drude32.coefficients)===ComplexF32
        setprecision(BigFloat,128) do
            coupling_big=PIOperator(basis,Complex{BigFloat}.(coupling.data))
            drude_big=PIDHEOM.drude_lorentz_bath(
                coupling_big,big"0.3",big"0.8",big"2.0";
                matsubara_terms=1)
            @test eltype(drude_big.coefficients)===Complex{BigFloat}
            @test_throws ArgumentError PIDHEOM.drude_lorentz_bath(
                coupling_big,big"0.3",big"0.8",big"2.0";
                matsubara_terms=1,decomposition=:pade)
        end
        @test_throws ArgumentError PIDHEOM.drude_lorentz_bath(
            coupling,0.3,pi,2.0;matsubara_terms=1)
        @test_throws ArgumentError PIDHEOM.drude_lorentz_bath(
            coupling,0.3,0.8,2.0;decomposition=:unknown)
        @test_throws ArgumentError PIDHEOM.underdamped_brownian_bath(
            coupling,0.7,2.4,1.2,2.0)

        sigma_minus=ComplexF64[0 1;0 0]
        embedding=PIDHEOM.independent_local_pseudomode_model(
            2,zeros(2,2),sigma_minus;
            nmax=2,frequency=1.1,coupling_strength=0.2,
            damping=0.4,thermal_occupation=0.3)
        @test embedding.basis.N==2
        @test embedding.basis.d==6
        @test embedding.model.basis===embedding.basis
        @test length(embedding.model.terms)==3
        @test embedding.metadata.embedding===:independent_local_pseudomode
        @test !embedding.metadata.coupling_hermitian
        @test ishermitian(embedding.site_hamiltonian)
        @test embedding.metadata.zero_temperature_pole.coefficient≈0.04
        @test embedding.metadata.zero_temperature_pole.frequency≈0.2+1.1im
        longitudinal=PIDHEOM.independent_local_pseudomode_model(
            2,zeros(2,2),ComplexF64[1 0;0 -1];
            nmax=1,frequency=1.1,coupling_strength=0.2,damping=0.4)
        @test longitudinal.metadata.coupling_hermitian
        @test ishermitian(longitudinal.site_hamiltonian)
        @test_throws ArgumentError PIDHEOM.independent_local_pseudomode_model(
            2,zeros(2,2),Matrix{Float64}(I,2,2);
            nmax=0,frequency=1.1,coupling_strength=0.2,damping=0.4)
    end

    @testset "residue terminator, pruning, and thermal preparation" begin
        basis=PIBasis(1,2)
        spin=spin_matrices()
        coupling=collective_operator(basis,spin.jz)
        bath=PIDHEOM.HEOMBath(coupling,0.0,1.0;residue=0.17)
        system=PIModel(basis,())
        hard=PIDHEOM.HEOMPlan(system,bath;max_depth=0)
        terminated=PIDHEOM.HEOMPlan(
            system,bath;max_depth=0,terminator=:residue)
        source=randn(ComplexF64,length(basis))
        hard_image=similar(source);terminated_image=similar(source)
        PIDHEOM.apply!(hard_image,hard,source,PIDHEOM.HEOMWorkspace(hard))
        PIDHEOM.apply!(terminated_image,terminated,source,
                       PIDHEOM.HEOMWorkspace(terminated))
        jump_plan=PIDHEOM.LiouvillianPlan(
            PIModel(basis,(DirectPIJump(coupling;rate=0.34),)))
        reference=similar(source)
        PIDHEOM.apply!(reference,jump_plan,source,
                       PIDHEOM.LiouvillianWorkspace(jump_plan))
        @test iszero(norm(hard_image))
        @test terminated_image≈reference atol=2e-15 rtol=2e-15
        probe=randn(ComplexF64,length(source))
        adjoint_image=similar(source)
        PIDHEOM.apply_adjoint!(adjoint_image,terminated,probe,
                               PIDHEOM.HEOMWorkspace(terminated))
        @test isapprox(dot(probe,terminated_image),
                       dot(adjoint_image,source);atol=3e-15,rtol=3e-15)

        hard_deep=PIDHEOM.HEOMPlan(system,bath;max_depth=2)
        terminated_deep=PIDHEOM.HEOMPlan(
            system,bath;max_depth=2,terminator=:residue)
        hierarchy_source=randn(ComplexF64,size(hard_deep,1))
        hard_deep_image=similar(hierarchy_source)
        terminated_deep_image=similar(hierarchy_source)
        PIDHEOM.apply!(hard_deep_image,hard_deep,hierarchy_source,
                       PIDHEOM.HEOMWorkspace(hard_deep))
        PIDHEOM.apply!(terminated_deep_image,terminated_deep,
                       hierarchy_source,
                       PIDHEOM.HEOMWorkspace(terminated_deep))
        for ado in 1:PIDHEOM.heom_number_ados(hard_deep)
            range=(ado-1)*length(basis)+1:ado*length(basis)
            PIDHEOM.apply!(reference,jump_plan,view(hierarchy_source,range),
                           PIDHEOM.LiouvillianWorkspace(jump_plan))
            @test isapprox(
                view(terminated_deep_image-hard_deep_image,range),reference;
                atol=3e-15,rtol=3e-15)
        end

        full=PIDHEOM.HEOMPlan(system,
            PIDHEOM.HEOMBath(coupling,[0.2,0.05],[1.0,3.0]);max_depth=4)
        pruned=PIDHEOM.HEOMPlan(system,
            PIDHEOM.HEOMBath(coupling,[0.2,0.05],[1.0,3.0]);max_depth=4,
            importance_cutoff=0.04)
        @test PIDHEOM.heom_number_ados(full)==15
        @test PIDHEOM.heom_number_ados(pruned)<15
        metadata=PIDHEOM.heom_hierarchy_metadata(pruned)
        @test metadata.full_ados==15
        @test metadata.pruning_approximation
        @test length(PIDHEOM.heom_ado_importances(pruned))==
              PIDHEOM.heom_number_ados(pruned)
        retained=Set(Tuple(index) for index in pruned.multiindices)
        for index in pruned.multiindices,term in eachindex(index)
            iszero(index[term])&&continue
            parent=copy(index);parent[term]-=1
            @test Tuple(parent) in retained
        end
        # The pruned map is exactly P*L*P for its retained order ideal: source
        # coordinates of omitted ADOs are zero and omitted output is ignored.
        pruned_source=randn(ComplexF64,size(pruned,1))
        full_source=zeros(ComplexF64,size(full,1))
        npi=length(basis)
        full_lookup=Dict(Tuple(index)=>position
                         for (position,index) in pairs(full.multiindices))
        for (retained_ado,index) in pairs(pruned.multiindices)
            full_ado=full_lookup[Tuple(index)]
            full_source[(full_ado-1)*npi+1:full_ado*npi].=
                view(pruned_source,(retained_ado-1)*npi+1:retained_ado*npi)
        end
        pruned_image=similar(pruned_source);full_image=similar(full_source)
        PIDHEOM.apply!(pruned_image,pruned,pruned_source,
                       PIDHEOM.HEOMWorkspace(pruned))
        PIDHEOM.apply!(full_image,full,full_source,
                       PIDHEOM.HEOMWorkspace(full))
        for (retained_ado,index) in pairs(pruned.multiindices)
            full_ado=full_lookup[Tuple(index)]
            @test isapprox(
                view(pruned_image,
                     (retained_ado-1)*npi+1:retained_ado*npi),
                view(full_image,(full_ado-1)*npi+1:full_ado*npi);
                atol=3e-15,rtol=3e-15)
        end
        pruned_prefix=PIDHEOM._heom_prefix_plan(pruned,2)
        @test all(index->sum(index)<=2,pruned_prefix.multiindices)
        @test pruned_prefix.full_ado_count==6

        H=collective_operator(basis,spin.jz)
        factorized=PIDHEOM.heom_thermal_state(
            hard,H,2.0;preparation=:factorized)
        @test PIDHEOM.heom_reduced_state(factorized).data≈
              thermal_state(H,2.0).data
        relaxed=PIDHEOM.heom_thermal_state(
            hard,H,2.0;preparation=:relaxation,
            relaxation_time=0.1,steps=2)
        @test relaxed.data≈factorized.data atol=0 rtol=0
        damped_plan=PIDHEOM.HEOMPlan(
            PIModel(basis,(LocalJump(ComplexF64[0 1;0 0]),)),bath;
            max_depth=0)
        stationary_preparation=PIDHEOM.heom_thermal_state(
            damped_plan,H,2.0;preparation=:stationary,return_info=true,
            krylovdim=8,maxiter=80,atol=1e-11,rtol=1e-9)
        @test stationary_preparation.state isa PIDHEOM.HEOMState
        @test PIDHEOM.heom_reduced_state(
            stationary_preparation.state).data≈
            iid_pure_state(basis,ComplexF64[1,0]).data atol=2e-10
        physical=PIDHEOM.drude_lorentz_bath(
            coupling,0.2,0.7,3.0;matsubara_terms=0)
        physical_plan=PIDHEOM.HEOMPlan(system,physical;max_depth=0)
        @test_throws ArgumentError PIDHEOM.heom_thermal_state(
            physical_plan,H,2.0)
        @test_throws ArgumentError PIDHEOM.HEOMPlan(
            system,bath;max_depth=1,importance_cutoff=1.1)
    end

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
        plan_workspace=PIDHEOM.HEOMWorkspace(plan)
        @test plan_workspace.system isa PIDHEOM.LiouvillianWorkspace
        @test plan_workspace.system.batch.capacity==
              min(PIDHEOM.heom_number_ados(plan),
                  PIDHEOM._HEOM_SYSTEM_BATCH_WIDTH)
        @test PIDHEOM.heom_number_ados(plan)==6
        @test PIDHEOM.heom_multiindices(plan)==
              [[0,0],[0,1],[1,0],[0,2],[1,1],[2,0]]
        detached=PIDHEOM.heom_multiindices(plan)
        detached[1]=[4,4]
        @test plan.multiindices[1]==[0,0]
        @test size(plan)==(6length(basis),6length(basis))
        @test eltype(plan)===ComplexF64
        @test isautonomous(plan)
        @test plan.scaling===:unscaled
        @test all(isone,plan.ado_scales)
        @test length(plan.topology.lower)==6
        @test length(plan.topology.incident_edges)==12
        @test eltype(plan.topology.lower)===Int32
        @test PIDHEOM.heom_coordinate_scale(plan,[0,0])==1
        @test_throws ArgumentError PIDHEOM.HEOMPlan(
            model,bath_z;max_depth=1,scaling=:unknown)
        @test_throws ArgumentError PIDHEOM.HEOMPlan(
            model,bath_z;max_depth=1,scaling_factors=[1.0,1.0])
        @test_throws DimensionMismatch PIDHEOM.HEOMPlan(
            model,bath_z;max_depth=1,scaling=:scaled,
            scaling_factors=[1.0,2.0])
        @test_throws ArgumentError PIDHEOM.HEOMPlan(
            model,bath_z;max_depth=1,scaling=:scaled,
            scaling_factors=[0.0])
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
        integer_state32=PIDHEOM.HEOMState(
            plan32,zeros(Int,size(plan32,1)))
        inexact_state_data=zeros(Int,size(plan32,1))
        inexact_state_data[1]=16_777_217
        @test eltype(integer_state32.data)===ComplexF32
        @test_throws ArgumentError PIDHEOM.HEOMState(
            plan32,inexact_state_data)
        state32=PIDHEOM.heom_evolve(plan32,
            iid_pure_state(basis,ComplexF32[1,1]/sqrt(2.0f0)),
            (0.0f0,0.01f0);steps=2)
        initial32=iid_pure_state(basis,ComplexF32[1,0])
        problem32=PIDHEOM.heom_problem(
            plan32,initial32,(0.0f0,0.1f0))
        @test eltype(plan32)===ComplexF32
        @test eltype(state32.data)===ComplexF32
        @test eltype(PIDHEOM.heom_reduced_state(state32).data)===ComplexF32
        @test problem32.tspan===(0.0f0,0.1f0)
        @test_throws ArgumentError PIDHEOM.heom_problem(
            plan32,initial32,(0.0,0.1))
        @test_throws ArgumentError PIDHEOM.heom_problem(
            plan32,initial32,(0.0f0,))
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
        @test_throws ArgumentError PIDHEOM.HEOMPlan(model32,
            PIDHEOM.HEOMBath(coupling32,0.2f0,1.1f0);max_depth=1,
            scaling=:scaled,scaling_factors=[0.2])
        @test_throws ArgumentError PIDHEOM.heom_block_preconditioner(
            plan32;regularization=0.1,warn_unamortized=false)
        tiny_regularization=Float64(nextfloat(0.0f0))/2
        @test_throws ArgumentError PIDHEOM.heom_block_preconditioner(
            plan32;regularization=tiny_regularization,
            warn_unamortized=false)
        preconditioner32=PIDHEOM.heom_block_preconditioner(
            plan32;regularization=1.0f-3,operator_scale=1.0,
            warn_unamortized=false)
        @test preconditioner32.operator_scale===1.0f0
        @test_throws ArgumentError PIDHEOM.heom_block_preconditioner(
            plan32;regularization=1.0f-3,operator_scale=0.1,
            warn_unamortized=false)
        @test_throws ArgumentError PIDHEOM.heom_block_preconditioner(
            plan32;expected_reuses=BigInt(typemax(Int))+1,
            warn_unamortized=false)
        @test_throws ArgumentError PIDHEOM.heom_block_preconditioner(
            plan32;expected_solve_applications=BigInt(typemax(Int))+1,
            warn_unamortized=false)
        @test_throws ArgumentError PIDHEOM.heom_block_preconditioner(
            plan32;shift_backend=:unknown,warn_unamortized=false)
        @test_throws ArgumentError PIDHEOM.heom_block_preconditioner(
            plan32;schur_rcond_threshold=1.1f0,warn_unamortized=false)
        @test_throws ArgumentError PIDHEOM.heom_block_preconditioner(
            plan32;schur_residual_tolerance=-eps(Float32),
            warn_unamortized=false)

        # Generic precision deliberately stays on the duplicate-aware LU
        # route; no LAPACK narrowing is used to obtain Schur factors.
        big_coupling=PIOperator(basis;T=BigFloat)
        big_trace=PIDHEOM._trace_vector(basis,Complex{BigFloat})
        big_stationary=big_trace/dot(big_trace,big_trace)
        big_system=-Matrix{Complex{BigFloat}}(I,length(basis),length(basis))+
                   big_stationary*adjoint(big_trace)
        big_plan=PIDHEOM.HEOMPlan(big_system,
            PIDHEOM.HEOMBath(big_coupling,big"0.2",big"1.0");
            basis,max_depth=1)
        big_preconditioner=PIDHEOM.heom_block_preconditioner(
            big_plan;operator_scale=big"1.0",warn_unamortized=false)
        big_cost=PIDHEOM.preconditioner_cost(big_preconditioner)
        @test big_cost.shift_backend===:lu
        @test !big_cost.schur_supported
        @test !big_cost.apply_serialized
        @test_throws ArgumentError PIDHEOM.heom_block_preconditioner(
            big_plan;operator_scale=big"1.0",shift_backend=:schur,
            warn_unamortized=false)
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
        @test length(plan.topology.lower)==6
        @test length(plan.topology.incident_edges)==12
        @test plan.topology.coupling_groups==6
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
                upper_label=copy(occupations);upper_label[term]+=1
                upper=get(plan.index,Tuple(upper_label),0)
                if !iszero(upper)
                    columns=(upper-1)*npi+1:upper*npi
                    reference[target,columns].+=commutator
                end
                if occupations[term]>0
                    lower_label=copy(occupations);lower_label[term]-=1
                    lower=plan.index[Tuple(lower_label)]
                    columns=(lower-1)*npi+1:lower*npi
                    reference[target,columns].+=-im*occupations[term].*(
                        coefficients[term]*left-
                        right_coefficients[term]*right)
                end
            end
        end
        @test matrixfree≈reference atol=2e-14 rtol=2e-14

        scaled_plan=PIDHEOM.HEOMPlan(system,bath;basis,max_depth=2,
            scaling=:scaled,scaling_factors=[0.25,0.5])
        scaled_matrix=zeros(ComplexF64,dimension,dimension)
        scaled_source=zeros(ComplexF64,dimension)
        scaled_workspace=PIDHEOM.HEOMWorkspace(scaled_plan)
        for column in 1:dimension
            fill!(scaled_source,0);scaled_source[column]=1
            apply!(view(scaled_matrix,:,column),scaled_plan,scaled_source,
                   0.0,nothing,scaled_workspace)
        end
        coordinate_scales=repeat(scaled_plan.ado_scales;inner=npi)
        expected_scaled=Diagonal(inv.(coordinate_scales))*reference*
                        Diagonal(coordinate_scales)
        @test scaled_matrix≈expected_scaled atol=5e-14 rtol=5e-14
        @test PIDHEOM.heom_coordinate_scale(scaled_plan,[1,1])≈
              sqrt(0.25*0.5)
        prefix=PIDHEOM._heom_prefix_plan(scaled_plan,1)
        @test prefix.scaling===:scaled
        @test prefix.ado_scales==scaled_plan.ado_scales[1:3]
        @test prefix.pole_scales===scaled_plan.pole_scales

        rng=MersenneTwister(0x4e10)
        random_hierarchy=randn(rng,ComplexF64,dimension)
        image=similar(random_hierarchy)
        apply!(image,plan,random_hierarchy,0.0,nothing,workspace)
        @test image≈reference*random_hierarchy atol=5e-14 rtol=5e-14
        adjoint_source=randn(rng,ComplexF64,dimension)
        adjoint_image=similar(adjoint_source)
        apply_adjoint!(adjoint_image,plan,adjoint_source,0.0,nothing,
                       workspace)
        @test dot(adjoint_source,image)≈
              dot(adjoint_image,random_hierarchy) atol=8e-14 rtol=8e-14
        scaled_input=randn(rng,ComplexF64,dimension)
        scaled_output=similar(scaled_input)
        scaled_adjoint=similar(scaled_input)
        apply!(scaled_output,scaled_plan,scaled_input,scaled_workspace)
        apply_adjoint!(scaled_adjoint,scaled_plan,adjoint_source,
                       scaled_workspace)
        @test dot(adjoint_source,scaled_output)≈
              dot(scaled_adjoint,scaled_input) atol=1e-13 rtol=1e-13
        @test abs(dot(plan.tracevec,image))<5e-14

        # One hierarchy action presents all ADOs as a system-Liouvillian
        # matrix batch. A custom batch callback is therefore evaluated once,
        # rather than once for every ADO, while the exact HEOM result and
        # adjoint remain unchanged.
        vector_calls=Ref(0);batch_calls=Ref(0)
        adjoint_calls=Ref(0);adjoint_batch_calls=Ref(0)
        custom_system=MatrixFreeLiouvillian(npi,
            (destination,input,time,parameters)->begin
                vector_calls[]+=1;mul!(destination,system,input)
            end,ComplexF64,ones(ComplexF64,npi);
            adjoint_action! = (destination,input,time,parameters)->begin
                adjoint_calls[]+=1;mul!(destination,adjoint(system),input)
            end,
            batched_action! = (destination,input,time,parameters)->begin
                batch_calls[]+=1;mul!(destination,system,input)
            end,
            batched_adjoint_action! =
                (destination,input,time,parameters)->begin
                    adjoint_batch_calls[]+=1
                    mul!(destination,adjoint(system),input)
                end)
        custom_plan=PIDHEOM.HEOMPlan(
            custom_system,bath;basis,max_depth=2)
        custom_image=similar(random_hierarchy)
        apply!(custom_image,custom_plan,random_hierarchy,0.0,nothing,
               PIDHEOM.HEOMWorkspace(custom_plan))
        @test custom_image≈reference*random_hierarchy atol=5e-14 rtol=5e-14
        @test batch_calls[]==1
        @test vector_calls[]==0
        custom_adjoint=similar(adjoint_source)
        apply_adjoint!(custom_adjoint,custom_plan,adjoint_source,0.0,nothing,
                       PIDHEOM.HEOMWorkspace(custom_plan))
        @test custom_adjoint≈adjoint(reference)*adjoint_source atol=8e-14 rtol=8e-14
        @test adjoint_batch_calls[]==1
        @test adjoint_calls[]==0
        @test (@allocated apply!(
            image,plan,random_hierarchy,0.0,nothing,workspace))<=512
        @test_throws ArgumentError apply!(
            random_hierarchy,plan,random_hierarchy,0.0,nothing,workspace)
        overlapping=zeros(ComplexF64,dimension+1)
        @test_throws ArgumentError apply!(view(overlapping,2:dimension+1),
            plan,view(overlapping,1:dimension),0.0,nothing,workspace)
        adapter=PIDHEOM.heom_liouvillian(plan)
        @test size(adapter)==(dimension,dimension)
        adapter_adjoint=similar(adjoint_source)
        apply_adjoint!(adapter_adjoint,adapter,adjoint_source)
        @test adapter_adjoint≈adjoint_image atol=0 rtol=0
        batch_source=hcat(random_hierarchy,2random_hierarchy)
        batch_image=similar(batch_source)
        mul!(batch_image,adapter,batch_source)
        @test batch_image≈reference*batch_source atol=8e-14 rtol=8e-14
        adapter_evolution=PermutationalInvariantDynamics.EvolutionWorkspace(
            adapter,random_hierarchy)
        @test adapter_evolution.liouvillian isa PIDHEOM.HEOMWorkspace
        @test adapter_evolution.liouvillian.plan===plan

        unscaled_ado_data=randn(rng,ComplexF64,dimension)
        scaled_state=PIDHEOM.HEOMState(
            scaled_plan,unscaled_ado_data./coordinate_scales)
        for ado in 1:PIDHEOM.heom_number_ados(scaled_plan)
            range=(ado-1)*npi+1:ado*npi
            @test PIDHEOM.heom_ado(scaled_state,ado).data≈
                  unscaled_ado_data[range] atol=3e-16 rtol=3e-16
        end

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
        scaled_deep=PIDHEOM.HEOMPlan(PIModel(basis,()),bath;
            max_depth=4,scaling=:scaled)
        scaled_deep_state=PIDHEOM.heom_evolve(
            scaled_deep,initial,(0.0,final_time);steps=100)
        @test PIDHEOM.heom_reduced_state(scaled_deep_state).data≈
              PIDHEOM.heom_reduced_state(deep_state).data atol=2e-15 rtol=2e-15
        for ado in 1:PIDHEOM.heom_number_ados(deep)
            @test PIDHEOM.heom_ado(scaled_deep_state,ado).data≈
                  PIDHEOM.heom_ado(deep_state,ado).data atol=3e-15 rtol=3e-15
        end
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
        scaled_report=PIDHEOM.heom_depth_convergence(
            PIModel(basis,()),bath,initial,(0.0,0.0);
            depths=(0,1),steps=1,scaling=:scaled,
            scaling_factors=[0.4],consecutive=1)
        @test last(scaled_report.results).hierarchy.plan.scaling===:scaled
        @test last(scaled_report.results).hierarchy.plan.pole_scales==[0.4]
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

        problem=PIDHEOM.heom_problem(plan,hierarchy,(0.0,0.4))
        @test problem.u0==hierarchy.data
        @test problem.u0!==hierarchy.data
        problem_image=similar(problem.u0)
        reference_image=similar(problem.u0)
        problem.f(problem_image,problem.u0,problem.p,0.0)
        apply!(reference_image,plan,problem.u0,0.0,nothing,
               PIDHEOM.HEOMWorkspace(plan))
        @test problem_image==reference_image

        # Generic response/Floquet consumers receive independent HEOM
        # application scratch rather than using the adapter's synchronized
        # compatibility workspace.
        adapter=PIDHEOM.heom_liouvillian(plan)
        response_work=PIDHEOM.ResponseWorkspace(
            adapter;krylovdim=4,mode=:linear)
        @test response_work.action_workspace isa PIDHEOM.HEOMWorkspace
        @test response_work.action_workspace.plan===plan
        response_image=similar(problem.u0)
        apply!(response_image,adapter,problem.u0,0.0,nothing,
               response_work.action_workspace)
        @test response_image==reference_image
        response_adjoint=similar(problem.u0)
        direct_adjoint=similar(problem.u0)
        apply_adjoint!(response_adjoint,adapter,problem.u0,0.0,nothing,
                       response_work.action_workspace)
        apply_adjoint!(direct_adjoint,plan,problem.u0,0.0,nothing,
                       PIDHEOM.HEOMWorkspace(plan))
        @test response_adjoint==direct_adjoint
        @test_throws ArgumentError PIDHEOM.heom_problem(
            plan,PIDHEOM.HEOMState(PIDHEOM.HEOMPlan(
                model,bath;max_depth=1),
                zeros(ComplexF64,2length(basis))),(0.0,0.1))

        evolution_workspace=PIDHEOM.HEOMEvolutionWorkspace(plan)
        hierarchy_dimension=size(plan,1)
        @test sum(length,(evolution_workspace.temporary,
                          evolution_workspace.k1,evolution_workspace.k2,
                          evolution_workspace.k3,evolution_workspace.k4))==
              3hierarchy_dimension
        @test isempty(evolution_workspace.k3)&&
              isempty(evolution_workspace.k4)
        shared_hierarchy=similar(hierarchy.data)
        aliased_evolution=PIDHEOM.HEOMEvolutionWorkspace(
            PIDHEOM.HEOMWorkspace(plan),shared_hierarchy,shared_hierarchy,
            similar(shared_hierarchy),similar(shared_hierarchy,0),
            similar(shared_hierarchy,0))
        @test_throws ArgumentError PIDHEOM.heom_evolve!(
            shared_hierarchy,plan,hierarchy.data,(0.0,0.001);
            steps=1,workspace=aliased_evolution)
        allocation_destination=copy(hierarchy.data)
        PIDHEOM.heom_evolve!(allocation_destination,plan,hierarchy.data,
            (0.0,0.001);steps=1,workspace=evolution_workspace)
        # Julia 1.10 retains top-level keyword/testset and bounded batched-
        # matrix wrapper metadata beyond the local hot-call cost. This is
        # scalar/header storage only; all hierarchy-sized RK4 arrays and
        # Liouvillian batch buffers are reused.
        allocation_limit=VERSION<v"1.11" ? 4608 : 2048
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

        block_preconditioner=PIDHEOM.heom_block_preconditioner(
            plan;warn_unamortized=false)
        @test size(block_preconditioner)==size(plan)
        block_cost=PIDHEOM.preconditioner_cost(block_preconditioner)
        @test block_cost.shift_backend===:schur
        @test block_cost.setup_lu_factorizations==1
        @test block_cost.setup_schur_factorizations==1
        @test block_cost.setup_factorizations==2
        @test block_cost.unique_shift_blocks==2
        @test block_cost.duplicate_shift_blocks==0
        @test block_cost.schur_shift_blocks==2
        @test block_cost.fallback_shift_blocks==0
        @test block_cost.apply_serialized
        @test block_cost.setup_system_rhs_applications==length(basis)
        @test block_cost.setup_system_batch_applications==1
        @test block_cost.setup_application_batches==4
        rhs=randn(MersenneTwister(0x7b10),ComplexF64,size(plan,1))
        preconditioned=similar(rhs)
        ldiv!(preconditioned,block_preconditioner,rhs)
        aliased_preconditioned=copy(rhs)
        ldiv!(aliased_preconditioned,block_preconditioner,
              aliased_preconditioned)
        @test aliased_preconditioned≈preconditioned atol=0 rtol=0
        generator=zeros(ComplexF64,size(plan))
        probe=zeros(ComplexF64,size(plan,1))
        probe_image=similar(probe)
        for column in axes(generator,2)
            fill!(probe,0);probe[column]=1
            apply!(probe_image,plan,probe,PIDHEOM.HEOMWorkspace(plan))
            generator[:,column].=probe_image
        end
        trace_direction=plan.tracevec/dot(plan.tracevec,plan.tracevec)
        trace_fixed=generator/block_preconditioner.operator_scale+
                    trace_direction*adjoint(plan.tracevec)
        @test trace_fixed*preconditioned≈rhs atol=2e-12 rtol=2e-12

        lu_preconditioner=PIDHEOM.heom_block_preconditioner(
            plan;operator_scale=block_preconditioner.operator_scale,
            shift_backend=:lu,warn_unamortized=false)
        lu_preconditioned=similar(rhs)
        ldiv!(lu_preconditioned,lu_preconditioner,rhs)
        @test preconditioned≈lu_preconditioned atol=2e-13 rtol=2e-13
        @test PIDHEOM.preconditioner_cost(lu_preconditioner).shift_backend===:lu
        @test !PIDHEOM.preconditioner_cost(lu_preconditioner).apply_serialized
        nonfinite_rhs=copy(rhs);nonfinite_rhs[length(basis)+1]=Inf
        @test_throws ErrorException ldiv!(
            similar(nonfinite_rhs),block_preconditioner,nonfinite_rhs)

        guarded_fallback=PIDHEOM.heom_block_preconditioner(
            plan;operator_scale=block_preconditioner.operator_scale,
            shift_backend=:schur,schur_rcond_threshold=1.0,
            warn_unamortized=false)
        guarded_cost=PIDHEOM.preconditioner_cost(guarded_fallback)
        @test guarded_cost.shift_backend===:lu
        @test guarded_cost.schur_attempted
        @test guarded_cost.schur_shift_blocks==0
        @test guarded_cost.fallback_shift_blocks==2
        @test guarded_cost.setup_lu_factorizations==3
        guarded_output=similar(rhs)
        ldiv!(guarded_output,guarded_fallback,rhs)
        @test guarded_output≈lu_preconditioned atol=0 rtol=0

        # Repeated pole frequencies create exactly identical shifted ADO
        # blocks. They share LU storage, while every ADO still receives one
        # triangular solve and the preconditioned residual stays exact.
        repeated_bath=PIDHEOM.HEOMBath(zero_coupling,[0.2,0.1],[1.2,1.2];
            right_coefficients=[0.2,0.1])
        repeated_plan=PIDHEOM.HEOMPlan(model,repeated_bath;max_depth=2)
        repeated_preconditioner=PIDHEOM.heom_block_preconditioner(
            repeated_plan;operator_scale=1.0,warn_unamortized=false)
        repeated_cost=PIDHEOM.preconditioner_cost(repeated_preconditioner)
        @test PIDHEOM.heom_number_ados(repeated_plan)==6
        @test repeated_cost.ado_blocks==6
        @test repeated_cost.setup_lu_factorizations==1
        @test repeated_cost.setup_schur_factorizations==1
        @test repeated_cost.setup_factorizations==2
        @test repeated_cost.unique_shift_blocks==2
        @test repeated_cost.duplicate_shift_blocks==3
        @test repeated_cost.reused_shift_blocks==3
        @test repeated_cost.schur_shift_blocks==5
        @test repeated_cost.setup_liouvillian_applications==0
        @test repeated_cost.setup_system_rhs_applications==length(basis)
        @test repeated_cost.stored_factor_coefficients==length(basis)^2
        @test repeated_cost.stored_schur_coefficients==2length(basis)^2
        @test repeated_cost.stored_shift_coefficients==6
        @test repeated_cost.stored_coefficients==
              3length(basis)^2+2length(basis)+6
        @test length(repeated_preconditioner.factor_indices)==6
        @test length(unique(repeated_preconditioner.factor_indices))==2
        preconditioned_information=PIDHEOM.heom_steady_state(
            plan;preconditioner=:block,krylovdim=10,maxiter=100,
            atol=1e-11,rtol=1e-9,return_info=true)
        @test preconditioned_information.preconditioner_cost!==nothing
        @test PIDHEOM.heom_reduced_state(
                  preconditioned_information.state).data≈
              reduced.data atol=2e-11 rtol=2e-11
        @test_throws ArgumentError PIDHEOM.heom_steady_state(
            plan;preconditioner_regularization=1e-8)
    end
end
