using Test
using LinearAlgebra
using Random
using PermutationalInvariantDynamics

function _shiftinvert_test_model()
    basis=PIBasis(1,2)
    spin=spin_matrices(2)
    model=PIModel(basis,(
        LocalHamiltonian(0.37spin.jx+0.21spin.jz),
        LocalJump(spin.jm;rate=0.73),
        LocalJump(spin.jp;rate=0.19),
        LocalJump(spin.jz;rate=0.11),
    ))
    L=Matrix(liouvillian(model;representation=:sparse,memory_budget=Inf))
    model,L
end

function _nearest_index(values,target;exclude_zero=false)
    eligible=exclude_zero ?
        findall(value->abs(value)>1e-9,values) : collect(eachindex(values))
    eligible[argmin(abs.(values[eligible].-target))]
end

@testset "Complex trace-deflated inexact shift-invert" begin
    model,L=_shiftinvert_test_model()
    basis=model.basis
    dense=eigen(L)
    oracle_plan=NoJumpIterativePlan(model;memory_budget=Inf)
    tracevec=collect(oracle_plan.tracevec)
    stationary=steady_state(model;method=:direct,memory_budget=Inf)
    rng=MersenneTwister(0x51f7)
    rhs=randn(rng,ComplexF64,length(basis))
    shift=0.41+0.27im
    deflation=0.83

    # Complex right and adjoint inverse applications retain both the original
    # Liouvillian and the exact rank-one deflation convention.  Test both
    # prepared sector backends against bounded dense PI-coordinate oracles.
    for backend in (:schur,:eigen)
        iterative=NoJumpIterativePlan(model;backend,memory_budget=Inf)
        work=NoJumpIterativeWorkspace(iterative;krylovdim=4,recycle_dim=0,
            memory_budget=Inf)
        destination=similar(rhs)

        right=no_jump_iterative_resolvent!(destination,iterative,rhs,work;
            shift,deflation=0,maxiter=100,atol=1e-12,rtol=1e-10,
            memory_budget=Inf)
        @test right.converged
        @test right.shift===ComplexF64(shift)
        @test destination≈(shift*I-L)\rhs atol=2e-10 rtol=2e-9
        @test norm((shift*I-L)*destination-rhs,Inf)≈
            right.residual_inf atol=2e-12 rtol=2e-10

        fill!(destination,0)
        left=no_jump_iterative_resolvent!(destination,iterative,rhs,work;
            shift,deflation=0,adjoint_action=true,maxiter=100,
            atol=1e-12,rtol=1e-10,memory_budget=Inf)
        @test left.converged
        @test left.adjoint_action
        @test destination≈(shift*I-adjoint(L))\rhs atol=2e-10 rtol=2e-9

        # The adjoint rank-one term must use the physical stationary state:
        # (L+delta*|rho_ss><trace|)' =
        # L'+delta*|trace><rho_ss|.  The keyword is deliberately an internal
        # plumbing contract used by the integrated left-mode solver.
        fill!(destination,0)
        left_deflated=no_jump_iterative_resolvent!(
            destination,iterative,rhs,work;shift,deflation,
            adjoint_action=true,adjoint_deflation_functional=stationary,
            maxiter=100,atol=1e-12,rtol=1e-10,memory_budget=Inf)
        adjoint_matrix=shift*I-adjoint(L)-
            deflation*tracevec*adjoint(stationary)
        @test left_deflated.converged
        @test destination≈adjoint_matrix\rhs atol=3e-10 rtol=3e-9
        @test norm(adjoint_matrix*destination-rhs,Inf)<3e-10

        # Reusing the same mutable functional object after changing its
        # entries must refresh the cached Sherman--Morrison denominator.
        mutable_functional=copy(stationary)
        no_jump_iterative_resolvent!(destination,iterative,rhs,work;
            shift,deflation,adjoint_action=true,
            adjoint_deflation_functional=mutable_functional,
            maxiter=100,atol=1e-12,rtol=1e-10,memory_budget=Inf)
        mutable_functional.=iterative.deflation_vector
        mutated=no_jump_iterative_resolvent!(destination,iterative,rhs,work;
            shift,deflation,adjoint_action=true,
            adjoint_deflation_functional=mutable_functional,
            maxiter=100,atol=1e-12,rtol=1e-10,memory_budget=Inf)
        mutated_matrix=shift*I-adjoint(L)-
            deflation*tracevec*adjoint(mutable_functional)
        @test mutated.converged
        @test destination≈mutated_matrix\rhs atol=3e-10 rtol=3e-9

        # A caller may also reuse the custom functional's storage as the
        # destination.  The solve must retain a workspace-owned snapshot for
        # both the Krylov action and its final true-residual check.
        aliased_functional=copy(stationary)
        aliased_matrix=shift*I-adjoint(L)-
            deflation*tracevec*adjoint(aliased_functional)
        aliased=no_jump_iterative_resolvent!(
            aliased_functional,iterative,rhs,work;shift,deflation,
            adjoint_action=true,
            adjoint_deflation_functional=aliased_functional,
            maxiter=100,atol=1e-12,rtol=1e-10,memory_budget=Inf)
        @test aliased.converged
        @test aliased_functional≈aliased_matrix\rhs atol=3e-10 rtol=3e-9
        @test norm(aliased_matrix*aliased_functional-rhs,Inf)<3e-10

        nonfinite_functional=copy(stationary)
        nonfinite_functional[1]=NaN
        @test_throws ArgumentError no_jump_iterative_resolvent!(
            destination,iterative,rhs,work;shift,deflation,
            adjoint_action=true,
            adjoint_deflation_functional=nonfinite_functional,
            maxiter=100,memory_budget=Inf)

        # Workspace storage is task-owned.  Accept only the two deliberate
        # internal buffers used by the stationary solver and reject aliases
        # that could replace an input or invalidate a true-residual check.
        copyto!(work.identity_resolvent,rhs)
        @test_throws ArgumentError no_jump_iterative_resolvent!(
            destination,iterative,work.identity_resolvent,work;
            shift,deflation,maxiter=100,memory_budget=Inf)
        @test_throws ArgumentError no_jump_iterative_resolvent!(
            work.rhs,iterative,rhs,work;shift,deflation,maxiter=100,
            memory_budget=Inf)
        @test_throws ArgumentError no_jump_iterative_resolvent!(
            destination,iterative,rhs,work;shift,deflation,
            adjoint_action=true,
            adjoint_deflation_functional=work.image,
            maxiter=100,memory_budget=Inf)
        @test_throws ArgumentError no_jump_iterative_resolvent!(
            destination,iterative,rhs,work;shift,deflation,
            adjoint_deflation_functional=stationary,
            maxiter=100,memory_budget=Inf)
    end

    target_shift=-0.64+0.32im
    iterative=NoJumpIterativePlan(model;memory_budget=Inf)
    plan=TraceDeflatedShiftInvertPlan(iterative;
        shift=target_shift,deflation=1.2)
    @test size(plan)==size(iterative)
    @test eltype(plan)===ComplexF64
    @test isautonomous(plan)
    @test plan.shift===ComplexF64(target_shift)
    @test plan.deflation===1.2
    @test plan.metadata.algorithm===:inexact_shiftinvert
    @test plan.metadata.outer_restart===:implicit_qr_arnoldi
    @test plan.metadata.trace_deflated
    @test plan.metadata.complex_shift
    @test !plan.metadata.positive_real_contraction_guarantee

    work=TraceDeflatedShiftInvertWorkspace(plan;outer_krylovdim=4,
        inner_krylovdim=4,inner_recycle_dim=0,memory_budget=Inf)
    result=trace_deflated_shiftinvert_spectrum(plan;nev=1,workspace=work,
        krylovdim=4,retained_dimension=2,maxrestarts=8,
        candidate_oversampling=2,inner_maxiter=100,
        inner_atol=1e-12,inner_rtol=1e-10,
        inner_initial_rtol=1e-3,atol=1e-10,rtol=1e-8,
        vectors=true,memory_budget=Inf,rng=MersenneTwister(0x1aa1))
    reference_index=_nearest_index(dense.values,target_shift;exclude_zero=true)
    reference_value=dense.values[reference_index]
    @test result.converged
    @test result.method===:trace_deflated_inexact_shiftinvert_iram
    @test only(result.values)≈reference_value atol=2e-8 rtol=2e-7
    @test only(result.physical_residuals)<2e-9
    @test only(result.relative_residuals)<2e-8
    @test only(result.normalized_trace_errors)<2e-9
    @test norm(L*result.vectors-
        result.vectors*Diagonal(result.values))<2e-8
    @test result.inner_solves>0
    @test result.inner_iterations>0
    @test !isempty(result.inner_tolerance_history)
    @test result.final_inner_rtol<=
        first(result.inner_tolerance_history).inner_rtol
    @test all(diff(getproperty.(result.inner_tolerance_history,
        :next_inner_rtol)).<=0)
    @test all(entry->entry.certified_modes<=entry.candidate_modes,
        result.inner_tolerance_history)
    @test all(entry->entry.maximum_inner_residual_ratio<=1+10eps(),
        result.inner_tolerance_history)
    @test result.original_residual_certification
    @test result.trace_deflated

    huge_initial=fill(ComplexF64(floatmax(Float64),0),length(basis))
    huge_start=trace_deflated_shiftinvert_spectrum(plan;nev=1,
        workspace=work,krylovdim=4,retained_dimension=2,maxrestarts=8,
        candidate_oversampling=2,inner_maxiter=100,
        inner_atol=1e-12,inner_rtol=1e-10,atol=1e-10,rtol=1e-8,
        initial_vector=huge_initial,memory_budget=Inf)
    @test huge_start.converged
    @test only(huge_start.values)≈reference_value atol=2e-8 rtol=2e-7

    allocating=trace_deflated_shiftinvert_spectrum(model;
        shift=target_shift,deflation=1.2,nev=1,krylovdim=4,
        retained_dimension=2,candidate_oversampling=2,maxrestarts=8,
        inner_krylovdim=4,inner_maxiter=100,
        inner_atol=1e-12,inner_rtol=1e-10,atol=1e-10,rtol=1e-8,
        memory_budget=Inf,rng=MersenneTwister(0x1aa1))
    @test only(allocating.values)≈reference_value atol=2e-8 rtol=2e-7

    # Integrated left modes are paired globally, normalized biorthogonally,
    # and independently certified against L'.
    modes=trace_deflated_shiftinvert_spectrum(plan;nev=1,
        krylovdim=4,retained_dimension=2,maxrestarts=8,
        candidate_oversampling=2,inner_krylovdim=4,inner_maxiter=100,
        inner_atol=1e-12,inner_rtol=1e-10,
        inner_initial_rtol=1e-3,atol=1e-10,rtol=1e-8,
        vectors=true,mode_diagnostics=true,memory_budget=Inf,
        rng=MersenneTwister(0xb10f))
    @test modes.converged
    @test modes.left_converged
    @test length(modes.left_values)==1
    @test only(modes.left_values)≈conj(only(modes.values)) atol=2e-8 rtol=2e-7
    right_vector=view(modes.vectors,:,1)
    left_vector=view(modes.left_vectors,:,1)
    @test norm(L*right_vector-only(modes.values)*right_vector)<2e-8
    @test norm(adjoint(L)*left_vector-
        conj(only(modes.values))*left_vector)<2e-8
    @test dot(left_vector,right_vector)≈1 atol=2e-8 rtol=2e-8
    @test abs(dot(tracevec,right_vector))<2e-8
    @test abs(dot(stationary,left_vector))<2e-8
    @test only(modes.condition_numbers)>=1
    @test only(modes.reciprocal_condition_numbers)<=1
    @test modes.mode_diagnostics.original_liouvillian_certified
    @test !hasproperty(modes.mode_diagnostics,:right_vectors)
    @test !hasproperty(modes.mode_diagnostics,:left_vectors)
    @test modes.pairing_converged
    @test modes.diagnostics_complete
    @test only(modes.mode_diagnostics.right_relative_residuals)<2e-8
    @test only(modes.mode_diagnostics.left_relative_residuals)<2e-8

    dense_left=eigen(adjoint(L))
    left_index=_nearest_index(dense_left.values,conj(reference_value))
    dense_right=dense.vectors[:,reference_index]
    dense_left_vector=dense_left.vectors[:,left_index]
    reference_condition=norm(dense_right)*norm(dense_left_vector)/
        abs(dot(dense_left_vector,dense_right))
    @test only(modes.condition_numbers)≈reference_condition atol=3e-7 rtol=3e-6

    # Plans/workspaces are strict about precision, ownership, dimensions, and
    # explicit resource limits.
    @test_throws ArgumentError TraceDeflatedShiftInvertPlan(iterative;
        shift=Inf,deflation=1)
    @test_throws ArgumentError TraceDeflatedShiftInvertPlan(iterative;
        shift=0,deflation=0)
    @test_throws ArgumentError TraceDeflatedShiftInvertPlan(iterative;
        shift=0,deflation=-1)
    @test_throws ArgumentError TraceDeflatedShiftInvertPlan(iterative;
        shift=1.2,deflation=1.2)
    @test_throws ArgumentError TraceDeflatedShiftInvertWorkspace(plan;
        outer_krylovdim=1,memory_budget=Inf)
    @test_throws ArgumentError TraceDeflatedShiftInvertWorkspace(plan;
        inner_krylovdim=0,memory_budget=Inf)
    @test_throws ArgumentError TraceDeflatedShiftInvertWorkspace(plan;
        inner_recycle_dim=-1,memory_budget=Inf)
    @test_throws ArgumentError TraceDeflatedShiftInvertWorkspace(plan;
        outer_krylovdim=4,inner_krylovdim=4,memory_budget=1)
    other=TraceDeflatedShiftInvertPlan(iterative;
        shift=-0.5+0.1im,deflation=1.2)
    @test_throws ArgumentError trace_deflated_shiftinvert_spectrum(other;
        workspace=work,nev=1,krylovdim=4,memory_budget=Inf)
    @test_throws ArgumentError trace_deflated_shiftinvert_spectrum(plan;
        workspace=work,nev=1,krylovdim=4,inner_krylovdim=3,
        memory_budget=Inf)
    @test_throws ArgumentError trace_deflated_shiftinvert_spectrum(plan;
        workspace=work,nev=1,krylovdim=4,inner_recycle_dim=1,
        memory_budget=Inf)
    @test_throws ArgumentError trace_deflated_shiftinvert_spectrum(plan;
        nev=0,memory_budget=Inf)
    @test_throws ArgumentError trace_deflated_shiftinvert_spectrum(plan;
        nev=1,krylovdim=2,candidate_oversampling=2,memory_budget=Inf)
    @test_throws ArgumentError trace_deflated_shiftinvert_spectrum(plan;
        nev=1,krylovdim=4,inner_maxiter=0,memory_budget=Inf)
    @test_throws ArgumentError trace_deflated_shiftinvert_spectrum(plan;
        nev=1,krylovdim=4,inner_safety=0,memory_budget=Inf)
    @test_throws ArgumentError trace_deflated_shiftinvert_spectrum(plan;
        nev=1,krylovdim=4,inner_decay=1,memory_budget=Inf)
    @test_throws ArgumentError trace_deflated_shiftinvert_spectrum(plan;
        nev=1,krylovdim=4,outer_restart=:krylov_schur,memory_budget=Inf)
    @test_throws ArgumentError trace_deflated_shiftinvert_spectrum(iterative;
        shift=target_shift,backend=:eigen,nev=1,krylovdim=4,
        memory_budget=Inf)
    @test_throws ArgumentError trace_deflated_shiftinvert_spectrum(plan;
        nev=1,krylovdim=4,reuse_inner=true,memory_budget=Inf)
    @test_throws ArgumentError trace_deflated_shiftinvert_spectrum(plan;
        nev=1,krylovdim=4,memory_budget=1)
    @test_throws DimensionMismatch trace_deflated_shiftinvert_spectrum(plan;
        nev=1,krylovdim=4,initial_vector=zeros(ComplexF64,3),
        memory_budget=Inf)
    @test_throws ArgumentError trace_deflated_shiftinvert_spectrum(plan;
        nev=1,krylovdim=4,initial_vector=zeros(ComplexF64,4),
        memory_budget=Inf)
    bad_initial=ones(ComplexF64,4);bad_initial[1]=NaN
    @test_throws ArgumentError trace_deflated_shiftinvert_spectrum(plan;
        nev=1,krylovdim=4,initial_vector=bad_initial,memory_budget=Inf)

    # A relaxed Float32 calculation stays in prepared precision.  A wider
    # complex shift is rejected rather than silently narrowing.
    spin32=spin_matrices(2;T=Float32)
    model32=PIModel(PIBasis(1,2),(
        LocalHamiltonian(spin32.jx;rate=0.37f0),
        LocalJump(spin32.jm;rate=0.73f0),
        LocalJump(spin32.jp;rate=0.19f0),))
    iterative32=NoJumpIterativePlan(model32;memory_budget=Inf)
    @test_throws ArgumentError TraceDeflatedShiftInvertPlan(iterative32;
        shift=-0.4+0.2im)
    plan32=TraceDeflatedShiftInvertPlan(iterative32;
        shift=ComplexF32(-0.4f0,0.2f0))
    result32=trace_deflated_shiftinvert_spectrum(plan32;nev=1,
        krylovdim=4,retained_dimension=2,maxrestarts=4,
        inner_krylovdim=4,atol=1f-5,rtol=1f-4,
        inner_atol=1f-6,inner_rtol=1f-5,
        transformed_atol=1f-5,transformed_rtol=1f-4,
        memory_budget=Inf,rng=MersenneTwister(1))
    @test result32.converged
    @test eltype(result32.values)===ComplexF32
end

@testset "Biorthogonal mode assignment and conditioning" begin
    values=ComplexF64[-0.3+0.7im,-1.1-0.2im,-2.4]
    right=Matrix{ComplexF64}(I,3,3)
    permutation=[3,1,2]
    adjoint_values=conj.(values[permutation])
    left=right[:,permutation]
    diagnostics=biorthogonal_mode_diagnostics(values,right,
        adjoint_values,left;pairing_atol=1e-13,pairing_rtol=1e-13)
    @test diagnostics.assignment==[2,3,1]
    @test diagnostics.adjoint_values≈conj.(values) atol=1e-14
    @test diagnostics.overlap_matrix≈I atol=1e-14
    @test diagnostics.biorthogonality_error<1e-14
    @test diagnostics.condition_numbers≈ones(3) atol=1e-14
    @test diagnostics.reciprocal_condition_numbers≈ones(3) atol=1e-14
    @test all(diagnostics.pairing_matched)
    @test diagnostics.pairing_converged
    @test diagnostics.clusters_resolved
    @test diagnostics.diagnostics_complete
    @test all(==(:ok),diagnostics.statuses)
    @test diagnostics.defectiveness===:not_certified

    # A non-normal but diagonalizable pair has condition numbers greater than
    # one.  Passing exact dual bases also checks scale-invariance of the
    # reported conditioning and the returned dot(l,r)=1 normalization.
    epsilon=0.04
    nonnormal_right=ComplexF64[1 1;0 epsilon]
    nonnormal_left=inv(nonnormal_right)'
    nonnormal_values=ComplexF64[-0.5,-0.5+1e-12im]
    nonnormal=biorthogonal_mode_diagnostics(nonnormal_values,
        7nonnormal_right,conj.(nonnormal_values),
        nonnormal_left/11;cluster_atol=1e-10,cluster_rtol=0,
        pairing_atol=1e-10,pairing_rtol=0)
    @test all(nonnormal.condition_numbers.>1)
    @test adjoint(nonnormal.left_vectors)*nonnormal.right_vectors≈I atol=2e-13
    @test length(nonnormal.clusters)==1
    @test only(nonnormal.clusters).size==2
    @test only(nonnormal.clusters).status===:cluster_unmixed
    @test isfinite(only(nonnormal.clusters).projector_condition)
    @test only(nonnormal.clusters).left_rank==2
    @test only(nonnormal.clusters).right_rank==2

    unresolved=biorthogonal_mode_diagnostics(ComplexF64[-0.2],
        reshape(ComplexF64[1,0],2,1),ComplexF64[-0.8],
        reshape(ComplexF64[0,1],2,1);pairing_atol=1e-12,
        pairing_rtol=0)
    @test !only(unresolved.pairing_matched)
    @test !unresolved.pairing_converged
    @test only(unresolved.statuses)===
        :pairing_mismatch_and_defective_or_unresolved
    @test !only(unresolved.normalized)

    rank_deficient_vectors=ComplexF64[1 1;0 0]
    rank_deficient=biorthogonal_mode_diagnostics(
        ComplexF64[-0.2,-0.2],rank_deficient_vectors,
        ComplexF64[-0.2,-0.2],rank_deficient_vectors;
        pairing_atol=1e-12,pairing_rtol=0)
    @test all(rank_deficient.normalized)
    @test !rank_deficient.clusters_resolved
    @test !rank_deficient.diagnostics_complete
    @test only(rank_deficient.clusters).status===:defective_or_unresolved

    @test_throws DimensionMismatch biorthogonal_mode_diagnostics(
        values,right,adjoint_values[1:2],left)
    @test_throws DimensionMismatch biorthogonal_mode_diagnostics(
        values,right[:,1:2],adjoint_values,left)
    @test_throws DimensionMismatch biorthogonal_mode_diagnostics(
        values,right,adjoint_values,vcat(left,zeros(1,3)))
    zero_left=copy(left);fill!(view(zero_left,:,1),0)
    @test_throws ArgumentError biorthogonal_mode_diagnostics(
        values,right,adjoint_values,zero_left)
    bad_values=copy(values);bad_values[1]=ComplexF64(Inf,0)
    @test_throws ArgumentError biorthogonal_mode_diagnostics(
        bad_values,right,adjoint_values,left)
    bad_right=copy(right);bad_right[1,1]=ComplexF64(Inf,0)
    @test_throws ArgumentError biorthogonal_mode_diagnostics(
        values,bad_right,adjoint_values,left)
    huge_right=copy(right);huge_right[:,1].=floatmax(Float64)
    huge=biorthogonal_mode_diagnostics(values,huge_right,
        adjoint_values,left;pairing_atol=1e-13,pairing_rtol=1e-13)
    @test isfinite(norm(view(huge.right_vectors,:,1)))
    @test norm(view(huge.right_vectors,:,1))≈1 atol=2e-15
    @test_throws ArgumentError biorthogonal_mode_diagnostics(
        BigFloat.(real.(values)),BigFloat.(real.(right)),
        BigFloat.(real.(adjoint_values)),BigFloat.(real.(left)))
    @test_throws ArgumentError biorthogonal_mode_diagnostics(
        values,right,adjoint_values,left;pairing_atol=-1)
    @test_throws ArgumentError biorthogonal_mode_diagnostics(
        values,right,adjoint_values,left;memory_budget=1)
end
