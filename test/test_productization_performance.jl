@testset "local operator interchange core" begin
    dense=ComplexF32[0 1;0 0]
    copied=local_operator_matrix(dense;dimension=2)
    @test copied==dense
    @test copied!==dense
    @test eltype(copied)===ComplexF32

    sparse_operator=sparse(dense)
    sparse_copy=local_operator_matrix(sparse_operator)
    @test sparse_copy==sparse_operator
    @test sparse_copy!==sparse_operator
    @test sparse_copy isa SparseMatrixCSC{ComplexF32}

    @test_throws DimensionMismatch local_operator_matrix(zeros(2,3))
    @test_throws ArgumentError local_operator_matrix(dense;dimension=true)
    @test_throws DimensionMismatch local_operator_matrix(dense;dimension=3)
    @test_throws ArgumentError local_operator_matrix(ComplexF64[NaN 0;0 0])
    @test_throws ArgumentError local_operator_matrix(:not_an_operator)
end

@testset "curated workflow namespace" begin
    @test Workflow.PIBasis===PIBasis
    @test Workflow.stationary_state===stationary_state
    @test Workflow.Models===Models
    @test Workflow.verified_solve===verified_solve
    @test Workflow.prepare_geometry===prepare_geometry
    @test Workflow.prepared_reductions===prepared_reductions
    @test Workflow.preparation_cache_summary===preparation_cache_summary
    @test Workflow.ReductionPlanSet===ReductionPlanSet
    @test Workflow.reduction_plan===reduction_plan
    @test Workflow.reduced_states===reduced_states
    @test Workflow.bipartition_negativities===bipartition_negativities
    for name in names(Workflow;all=false,imported=false)
        @test isdefined(PermutationalInvariantDynamics,name)
        @test getfield(Workflow,name)===
            getfield(PermutationalInvariantDynamics,name)
    end
    @test :compile in names(Workflow)
    @test :liouvillian in names(
        PermutationalInvariantDynamics;all=false,imported=false)
    @test !(:liouvillian in names(Workflow))
end

@testset "optional sysimage workload sources" begin
    root=normpath(joinpath(@__DIR__,".."))
    for relative in (
            "scripts/precompile_workload.jl",
            "scripts/build_sysimage.jl")
        source=read(joinpath(root,relative),String)
        @test Meta.parseall(source) isa Expr
    end
end

@testset "affine compiled families" begin
    basis=PIBasis(2,2)
    spin=spin_matrices()
    base=PIModel(basis,(
        LocalHamiltonian(spin.jz;rate=0.1),
    ))
    components=(
        drive=LocalHamiltonian(spin.jx;rate=2.0),
        decay=(LocalJump(spin.jm;rate=0.5),),
    )
    family=compile_affine_family(
        base,components;defaults=(drive=0.25,decay=0.75))
    @test family.names==(:drive,:decay)
    @test family.estimates.geometry_reused

    bound=specialize(
        family,(drive=0.3,decay=0.7);
        backend=:sparse,memory_budget=Inf)
    reference=compile(PIModel(basis,(
        LocalHamiltonian(spin.jz;rate=0.1),
        LocalHamiltonian(spin.jx;rate=0.6),
        LocalJump(spin.jm;rate=0.35),
    ));backend=:sparse,memory_budget=Inf)
    @test Matrix(bound.operator)≈Matrix(reference.operator) atol=2e-13 rtol=2e-13
    @test bound.estimates.affine_parameters==
        (drive=0.3,decay=0.7)
    @test_throws ArgumentError specialize(
        family,(drive=0.2,);memory_budget=Inf)
    @test_throws ArgumentError specialize(
        family,(drive=0.2,decay=0.4,typo=1.0);memory_budget=Inf)
    @test_throws ArgumentError compile_affine_family(
        base,(bad=LocalJump(spin.jm;rate=(t,p)->1.0),))

    mutable_group=AbstractPITerm[
        LocalHamiltonian(spin.jx;rate=0.4)]
    immutable_family=compile_affine_family(
        base,(drive=mutable_group,))
    @test immutable_family.components.drive isa Tuple
    @test length(immutable_family.components.drive)==1
    push!(mutable_group,LocalJump(spin.jm;rate=0.9))
    mutable_group[1]=LocalHamiltonian(spin.jz;rate=8.0)
    immutable_bound=specialize(
        immutable_family,(drive=0.5,);
        backend=:sparse,memory_budget=Inf)
    immutable_reference=compile(PIModel(basis,(
        LocalHamiltonian(spin.jz;rate=0.1),
        LocalHamiltonian(spin.jx;rate=0.2),
    ));backend=:sparse,memory_budget=Inf)
    @test Matrix(immutable_bound.operator)≈
        Matrix(immutable_reference.operator) atol=2e-13 rtol=2e-13

    float32_basis=PIBasis(1,2)
    float32_spin=spin_matrices(2;T=Float32)
    tiny=nextfloat(0.0f0)
    underflow_family=compile_affine_family(
        PIModel(float32_basis,()),
        (decay=LocalJump(float32_spin.jm;rate=tiny),))
    @test_throws ArgumentError specialize(
        underflow_family,(decay=0.5f0,);
        backend=:matrixfree,memory_budget=Inf)
    overflow_family=compile_affine_family(
        PIModel(float32_basis,()),
        (decay=LocalJump(
            float32_spin.jm;rate=floatmax(Float32)),))
    @test_throws ArgumentError specialize(
        overflow_family,(decay=2.0f0,);
        backend=:matrixfree,memory_budget=Inf)
    zero_bound=specialize(
        overflow_family,(decay=0.0f0,);
        backend=:matrixfree,memory_budget=Inf)
    @test only(zero_bound.rates)===0.0f0
end

@testset "threaded prepared Krylov wrapper" begin
    basis=PIBasis(3,2)
    spin=spin_matrices()
    model=PIModel(basis,(
        LocalHamiltonian(spin.jx;rate=0.31),
        LocalJump(spin.jm;rate=0.47),
    ))
    compiled=compile(model;backend=:matrixfree)
    threaded=threaded_matrixfree(compiled;tasks=2)
    @test PermutationalInvariantDynamics._operator_basis(threaded)===basis
    @test PermutationalInvariantDynamics._operator_requires_matrixfree(threaded)
    @test PermutationalInvariantDynamics._operator_trace_functional(threaded)===
        compiled.plan.tracevec

    rng=MersenneTwister(294)
    input=randn(rng,ComplexF64,length(basis))
    expected=similar(input)
    actual=similar(input)
    adjoint_expected=similar(input)
    adjoint_actual=similar(input)
    work=LiouvillianWorkspace(compiled)
    apply!(expected,compiled,input,0.0,nothing,work)
    mul!(actual,threaded,input)
    apply_adjoint!(adjoint_expected,compiled,input,0.0,nothing,work)
    apply_adjoint!(adjoint_actual,threaded,input)
    @test actual≈expected atol=3e-13 rtol=3e-13
    @test adjoint_actual≈adjoint_expected atol=3e-13 rtol=3e-13

    recommendation=recommend_solver(
        threaded;task=:steady_state,memory_budget=Inf,krylovdim=8)
    @test recommendation.backend===:matrixfree
    @test recommendation.algorithm===:gmres
end

@testset "shared multi-bipartition reductions" begin
    basis=PIBasis(4,2)
    rho=iid_pure_state(basis,ComplexF64[1,sqrt(2)]/sqrt(3))
    plans=ReductionPlanSet(basis,(1,2,3))
    @test reduction_plan(plans,2).k==2
    @test plans.estimates.shared_su2_factorials==basis.N+2
    @test_throws ArgumentError ReductionPlanSet(basis,(1,1))
    @test_throws ArgumentError ReductionPlanSet(
        basis,(1,);atol=big"1e-1000")

    qudit_plans=ReductionPlanSet(PIBasis(2,3),(1,))
    @test qudit_plans.estimates.shared_lr_intertwiner_count>0
    @test qudit_plans.estimates.shared_lr_intertwiner_entries>=
        qudit_plans.estimates.shared_lr_intertwiner_count

    work=ReductionWorkspaceSet(plans,rho;mode=:both)
    states=reduced_states(rho,plans;workspace=work)
    purities=reduced_purities(rho,plans;workspace=work)
    negativities=bipartition_negativities(rho,plans;workspace=work)
    for (index,k) in pairs(plans.ks)
        reference_plan=ReductionPlan(basis,k)
        reference_work=ReductionWorkspace(reference_plan,rho;mode=:both)
        reference_state=reduced_state(
            rho,k;plan=reference_plan,workspace=reference_work)
        @test states[index].data≈reference_state.data atol=3e-12 rtol=3e-12
        @test purities[index]≈reduced_purity(
            rho,k;plan=reference_plan,workspace=reference_work) atol=3e-12
        @test negativities[index]≈negativity(
            rho,k;plan=reference_plan,workspace=reference_work) atol=3e-12
    end
end

@testset "curated model zoo" begin
    metadata=Models.catalog()
    @test hasproperty(metadata,:driven_qubits)
    @test metadata.boundary_time_crystal.sectors===:symmetric

    driven=Models.driven_qubits(
        2;drive=0.4,emission=0.2,pumping=0.03)
    @test driven.basis.d==2
    @test length(driven.terms)==3
    @test isautonomous(driven)
    expected_spin=spin_matrices()
    @test PermutationalInvariantDynamics.term_operator(driven.terms[1])≈
        0.4*expected_spin.jx

    twisting=Models.one_axis_twisting(3;chi=0.7)
    @test length(twisting.terms)==1
    @test twisting.terms[1] isa DirectPIHamiltonian

    time_crystal=Models.boundary_time_crystal(
        4;omega=1.2,kappa=0.8)
    @test time_crystal.basis.sectors==[Partition((4,0))]
    @test PermutationalInvariantDynamics.term_rate(
        time_crystal.terms[2])≈0.4
    @test_throws ArgumentError Models.local_pump_decay(2;down=0,up=0)

    float32_models=(
        Models.driven_qubits(2;T=Float32),
        Models.independent_dephasing(2;T=Float32),
        Models.local_pump_decay(2;T=Float32),
        Models.one_axis_twisting(2;T=Float32),
        Models.steady_superradiance(2;T=Float32),
        Models.boundary_time_crystal(2;T=Float32),
    )
    for model in float32_models
        compiled=compile(model;backend=:matrixfree)
        @test compiled.plan.Ttype===ComplexF32
        @test all(model.terms) do term
            rate=PermutationalInvariantDynamics.term_rate(term)
            rate isa Float32||rate isa Integer||rate isa Rational
        end
    end
    @test_throws ArgumentError Models.independent_dephasing(
        2;gamma=nextfloat(0.0),T=Float32)
    @test_throws ArgumentError Models.boundary_time_crystal(
        2;kappa=Inf,T=Float32)
    tiny=nextfloat(0.0f0)
    @test_throws ArgumentError Models.driven_qubits(
        1;drive=tiny,detuning=0,emission=0,pumping=0,dephasing=0,T=Float32)
    @test_throws ArgumentError Models.independent_dephasing(
        2;gamma=tiny,T=Float32)
    @test_throws ArgumentError Models.boundary_time_crystal(
        1;omega=0,kappa=floatmax(Float32),T=Float32)
    zero_model=Models.driven_qubits(
        1;drive=0,detuning=0,emission=0,pumping=0,dephasing=0,T=Float32)
    @test all(iszero,only(zero_model.terms).operator)
    ordinary_dephasing=Models.independent_dephasing(
        2;gamma=0.25f0,T=Float32)
    @test PermutationalInvariantDynamics.term_rate(
        only(ordinary_dephasing.terms))===0.125f0
end
