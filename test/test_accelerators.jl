@testset "Accelerator capability and guarded preflight" begin
    capability=accelerator_capability(:cuda)
    @test capability.backend===:cuda
    @test !capability.extension_loaded
    @test !capability.functional
    @test capability.supports_sparse_pi
    @test !capability.supports_matrixfree_pi
    @test capability.transfer_policy===:explicit_once
    @test capability.reason===:extension_not_loaded
    index_limit=BigInt(typemax(Int32))
    @test PermutationalInvariantDynamics._accelerator_sparse_indices_supported(
        Int32,1,index_limit-1)
    @test !PermutationalInvariantDynamics._accelerator_sparse_indices_supported(
        Int32,1,index_limit)

    unknown=accelerator_capability(:not_a_backend)
    @test !unknown.functional
    @test unknown.reason===:unknown_backend

    basis=PIBasis(2,2)
    spin=spin_matrices(2)
    model=PIModel(
        basis,
        (
            CollectiveHamiltonian(spin.jz),
            LocalJump(spin.jm;rate=0.2),
        ),
    )
    compiled=compile(model;backend=:matrixfree)
    report=accelerator_preflight(
        compiled;
        rhs_columns=3,
        memory_budget=Inf,
        device_memory_budget=Inf,
    )
    @test report.basis===basis
    @test report.scalar_type===eltype(compiled)
    @test report.dimension==length(basis)
    @test report.sectors==length(basis.sectors)
    @test report.rhs_columns==3
    @test report.rhs_kind===:matrix
    @test report.source_backend===:matrixfree
    @test report.autonomous
    @test report.sparse_materialization_supported
    @test report.materialization_required
    @test report.exact_nnz===nothing
    @test report.retained_nnz_upper_bound>0
    @test report.device_vector_bytes==
        2big(length(basis))*3*sizeof(eltype(compiled))
    @test report.device_peak_bytes==
        report.device_sparse_operator_bytes+report.device_vector_bytes
    @test report.combined_peak_bytes==
        report.host_materialization_peak_bytes+report.device_peak_bytes
    @test report.fits_memory_budget
    @test report.fits_device_memory_budget
    @test :backend_unavailable in report.issues
    @test !report.ready
    @test_throws ArgumentError accelerate(
        compiled;memory_budget=Inf,device_memory_budget=Inf)

    sparse_compiled=compile(model;backend=:sparse,memory_budget=Inf)
    sparse_report=accelerator_preflight(
        sparse_compiled;memory_budget=Inf,device_memory_budget=Inf)
    @test sparse_report.source_backend===:sparse
    @test !sparse_report.materialization_required
    @test sparse_report.exact_nnz==nnz(sparse_compiled.operator)
    @test sparse_report.retained_nnz_upper_bound==
        sparse_report.exact_nnz

    @eval PermutationalInvariantDynamics begin
        function _accelerator_extension_capability(
                ::Val{:test_vector_only})
            AcceleratorCapability(
                :test_vector_only,true,true,true,false,(:vector,),
                (Float64,ComplexF64),Int64,:explicit_once,:available,
                "test-only vector backend")
        end
    end
    vector_report=accelerator_preflight(
        compiled;backend=:test_vector_only,rhs_columns=1,
        memory_budget=Inf,device_memory_budget=Inf)
    @test vector_report.rhs_kind===:vector
    @test !(:rhs_kind_unsupported in vector_report.issues)
    explicit_matrix_report=accelerator_preflight(
        compiled;backend=:test_vector_only,rhs_columns=1,rhs_kind=:matrix,
        memory_budget=Inf,device_memory_budget=Inf)
    @test explicit_matrix_report.rhs_kind===:matrix
    @test :rhs_kind_unsupported in explicit_matrix_report.issues
    matrix_report=accelerator_preflight(
        compiled;backend=:test_vector_only,rhs_columns=2,
        memory_budget=Inf,device_memory_budget=Inf)
    @test :rhs_kind_unsupported in matrix_report.issues
    @test !matrix_report.ready

    @eval PermutationalInvariantDynamics begin
        function _accelerator_extension_capability(
                ::Val{:test_matrix_only})
            AcceleratorCapability(
                :test_matrix_only,true,true,true,false,(:matrix,),
                (Float64,ComplexF64),Int64,:explicit_once,:available,
                "test-only matrix backend")
        end
        function _accelerator_extension_capability(
                ::Val{:test_implicit_transfer})
            AcceleratorCapability(
                :test_implicit_transfer,true,true,true,false,
                (:vector,:matrix),(Float64,ComplexF64),Int64,
                :implicit_per_action,:available,
                "test-only backend with an invalid transfer policy")
        end
    end
    matrix_width_one=accelerator_preflight(
        compiled;backend=:test_matrix_only,rhs_columns=1,rhs_kind=:matrix,
        memory_budget=Inf,device_memory_budget=Inf)
    @test matrix_width_one.rhs_kind===:matrix
    @test !(:rhs_kind_unsupported in matrix_width_one.issues)
    matrix_auto=accelerator_preflight(
        compiled;backend=:test_matrix_only,rhs_columns=1,
        memory_budget=Inf,device_memory_budget=Inf)
    @test matrix_auto.rhs_kind===:vector
    @test :rhs_kind_unsupported in matrix_auto.issues
    implicit_transfer=accelerator_preflight(
        compiled;backend=:test_implicit_transfer,
        memory_budget=Inf,device_memory_budget=Inf)
    @test :unsupported_transfer_policy in implicit_transfer.issues
    @test !implicit_transfer.ready

    tiny=accelerator_preflight(
        compiled;memory_budget=0,device_memory_budget=0)
    @test !tiny.fits_memory_budget
    @test !tiny.fits_device_memory_budget
    @test :memory_budget_exceeded in tiny.issues
    @test :device_memory_budget_exceeded in tiny.issues

    @test_throws ArgumentError accelerator_preflight(model)
    @test_throws ArgumentError accelerator_preflight(compiled;rhs_columns=0)
    @test_throws ArgumentError accelerator_preflight(
        compiled;rhs_columns=2,rhs_kind=:vector)
    @test_throws ArgumentError accelerator_preflight(
        compiled;rhs_kind=:tensor)
    @test_throws ArgumentError accelerator_preflight(
        compiled;memory_budget=-1)
    @test_throws ArgumentError accelerator_preflight(
        compiled;device_memory_budget=NaN)

    schedule=t->spin.jz
    driven=LiouvillianPlan(PIModel(
        basis,(LocalHamiltonian(InPlaceTimeOperator(
            spin.jz,(destination,t,p)->copyto!(destination,schedule(t)))),)))
    driven_report=accelerator_preflight(
        driven;memory_budget=Inf,device_memory_budget=Inf)
    @test !driven_report.autonomous
    @test :nonautonomous_source in driven_report.issues
end
