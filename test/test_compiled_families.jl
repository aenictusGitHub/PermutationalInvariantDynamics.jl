@testset "compiled parameter families and recycled continuation" begin
    basis=PIBasis(2,2)
    spin=spin_matrices()
    prototype=PIModel(basis,(
        LocalHamiltonian(spin.jx;rate=0.17),
        LocalJump(spin.jm;rate=0.8),
        LocalJump(spin.jp;rate=0.2),
    ))
    family=compile_family(prototype)
    @test family.rate_indices==(1,2,3)
    @test family.default_rates==(0.17,0.8,0.2)
    @test family.estimates.geometry_reused
    family_workspace_bytes=
        PermutationalInvariantDynamics._performance_liouvillian_workspace_bytes(
            family.plan)
    @test family.estimates.matrixfree_workspace_upper_bound==
          family_workspace_bytes
    @test family.estimates.matrixfree_specialization_upper_bound==
          family_workspace_bytes+
          length(basis)*family.estimates.scalar_retained_bytes
    family_sparse_bounds=
        PermutationalInvariantDynamics._performance_sparse_materialization_bounds(
            family.plan)
    @test family.estimates.sparse_structure_supported
    @test family.estimates.sparse_operator_upper_bound==
          family_sparse_bounds.operator_bytes
    @test family.estimates.sparse_contribution_upper_bound==
          family_sparse_bounds.contribution_upper_bound
    @test family.estimates.sparse_retained_nnz_upper_bound==
          family_sparse_bounds.retained_nnz_upper_bound
    @test family.estimates.sparse_assembly_upper_bound==
          family_sparse_bounds.assembly_bytes
    @test family.estimates.sparse_specialization_peak_upper_bound==
          family_sparse_bounds.peak_bytes
    @test family.estimates.matrixfree_specialization_upper_bound<
          family.estimates.sparse_specialization_peak_upper_bound

    rates=(0.23,0.71,0.19)
    specialized=specialize(family,rates)
    reference_model=PIModel(basis,(
        LocalHamiltonian(spin.jx;rate=rates[1]),
        LocalJump(spin.jm;rate=rates[2]),
        LocalJump(spin.jp;rate=rates[3]),
    ))
    reference=liouvillian(reference_model;representation=:sparse)
    @test specialized.plan===family.plan
    @test specialized.backend===:matrixfree
    @test Matrix(liouvillian(specialized;representation=:sparse))≈
          Matrix(reference) atol=2e-13 rtol=2e-13

    vector=ComplexF64.(1:length(basis))./(length(basis)+1)
    output=similar(vector);expected=reference*vector
    workspace=LiouvillianWorkspace(specialized)
    @test apply!(output,specialized,vector,0.0,nothing,workspace)===output
    @test output≈expected atol=2e-13 rtol=2e-13
    @test specialized*vector≈expected atol=2e-13 rtol=2e-13
    apply_adjoint!(output,specialized,vector,0.0,nothing,workspace)
    @test output≈adjoint(reference)*vector atol=2e-13 rtol=2e-13

    block=hcat(vector,2vector)
    block_output=similar(block)
    apply!(block_output,specialized,block,0.0,nothing,workspace)
    @test block_output≈reference*block atol=2e-13 rtol=2e-13
    @test specialized.operator.workspace isa LiouvillianWorkspace
    @test specialized.operator.workspace.batch.capacity==0
    family_batch_growth=
        PermutationalInvariantDynamics._performance_batched_action_growth_bytes(
            specialized,2)
    @test family_batch_growth>0
    mul!(block_output,specialized,block)
    @test block_output≈reference*block atol=2e-13 rtol=2e-13
    @test specialized.operator.workspace.batch.capacity==2
    @test iszero(
        PermutationalInvariantDynamics._performance_batched_action_growth_bytes(
            specialized,2))

    sparse_specialized=specialize(family,rates;backend=:sparse)
    @test sparse_specialized.operator≈reference
    alias_spectrum=pi_liouvillian_spectrum(sparse_specialized;
        method=:ordinary_arnoldi,nev=2,krylovdim=4,
        return_info=true,require_convergence=false)
    @test alias_spectrum.method===:arnoldi
    matrixfree_budget=family.estimates.matrixfree_specialization_upper_bound
    auto_matrixfree=specialize(family,rates;backend=:auto,
        memory_budget=matrixfree_budget)
    @test auto_matrixfree.backend===:matrixfree
    @test auto_matrixfree.estimates.requested_backend===:auto
    auto_sparse=specialize(family,rates;backend=:auto,memory_budget=Inf)
    @test auto_sparse.backend===:sparse
    @test auto_sparse.estimates.budget_status===:disabled
    @test_throws ArgumentError specialize(family,rates;backend=:sparse,
        memory_budget=matrixfree_budget)
    @test_throws ArgumentError specialize(family,rates;backend=:matrixfree,
        memory_budget=matrixfree_budget-1)
    @test_throws ArgumentError liouvillian(specialized;
        representation=:sparse,memory_budget=1)
    @test specialize(family).rates==family.default_rates
    @test_throws DimensionMismatch specialize(family,(1.0,2.0))
    @test_throws ArgumentError specialize(family,(1.0,2.0,Inf))
    @test_throws ArgumentError compile_family(PIModel(basis,(
        LocalJump(spin.jm;rate=(time,parameters)->one(time)),)))
    @test_throws ArgumentError compile_family(prototype;rate_indices=Int[])
    @test_throws ArgumentError compile_family(prototype;rate_indices=(1,1))
    @test_throws BoundsError compile_family(prototype;rate_indices=(4,))

    one_rate=compile_family(prototype;rate_indices=(3,))
    @test specialize(one_rate,0.31).rates==(0.31,)
    @test_throws ArgumentError specialize(family,0.31)
    @test_throws ArgumentError ParameterScanPlan([0.2],family;
        specialize_options=(backend=:matrixfree,memory_budget=1024,))

    # A scan specialization reuses both the immutable Schur plan and the
    # GCRO near-zero subspace. Every changed operator is reapplied to the
    # retained directions before they are accepted.
    parameters=[0.16,0.19,0.22]
    scan=ParameterScanPlan(parameters,family;
        rate_builder=p->(0.17,0.8,p),
        algorithm=RecycledGMRESAlgorithm(
            krylovdim=8,maxiter=300,recycle_dim=3),
        continuation=true)
    scan_workspace=ParameterScanWorkspace()
    result=parameter_scan(scan;workspace=scan_workspace,on_error=:throw)
    @test all(point->point.status===:success,result.points)
    @test !result.points[1].warm_started
    @test result.points[2].warm_started
    @test result.points[2].workspace_reused
    @test result.points[2].diagnostics.solver.krylov_variant===:recycled
    @test result.points[2].diagnostics.solver.recycled_initially
    @test all(point->point.diagnostics.compile.geometry_reused,result.points)
    @test !result.points[1].diagnostics.compile.scan_liouvillian_workspace_reused
    @test result.points[2].diagnostics.compile.scan_liouvillian_workspace_reused
    @test all(point->point.diagnostics.compile.specialization_workspace_bytes==0,
              result.points)

    # Harmonic continuation keeps the selected slow subspace rather than
    # collapsing it to one combined starting vector.
    small_basis=PIBasis(1,2)
    small_spin=spin_matrices()
    small_family=compile_family(PIModel(small_basis,(
        LocalJump(small_spin.jm;rate=1.0),
        LocalJump(small_spin.jp;rate=0.2),
    )))
    harmonic=HarmonicArnoldiAlgorithm(
        nev=2,krylovdim=4,thickdim=2,maxrestarts=4)
    spectral_scan=ParameterScanPlan([0.2,0.25],small_family;
        rate_builder=p->(1.0,p),task=:spectrum,algorithm=harmonic,
        spectrum_target=:near_zero,nev=2,save_vectors=false,
        solver_options=(require_convergence=false,))
    spectral_result=parameter_scan(spectral_scan;on_error=:throw)
    @test spectral_result.points[2].warm_started
    @test spectral_result.points[2].diagnostics.solver.initial_retained_dimension==2
    @test spectral_result.restart_seed isa Matrix
end
