const _COLLECTIVE_FASTPATH_PID = PermutationalInvariantDynamics

# Independent small-system oracle for the one-row Schur irrep.  It uses only
# the public GT contents/order and the bosonic action
# a_i^dagger*a_j |n> = sqrt((n_i+1)n_j) |n+e_i-e_j>.
function _collective_fastpath_occupation_block(basis, operator)
    @assert length(basis.sectors) == 1
    occupations = content.(only(basis.patterns))
    lookup = Dict(occupation => index
                  for (index, occupation) in pairs(occupations))
    R = typeof(real(zero(eltype(operator))))
    S = promote_type(Complex{R}, eltype(operator))
    block = zeros(S, length(occupations), length(occupations))
    for (column, occupation) in pairs(occupations)
        for local_label in 1:basis.d
            block[column, column] +=
                R(occupation[local_label]) * operator[local_label, local_label]
        end
        for source_label in 1:basis.d
            source_count = occupation[source_label]
            iszero(source_count) && continue
            for target_label in 1:basis.d
                target_label == source_label && continue
                target = ntuple(label -> occupation[label] +
                    (label == target_label ? 1 : 0) -
                    (label == source_label ? 1 : 0), basis.d)
                coefficient = sqrt(R((occupation[target_label] + 1) *
                                     source_count))
                block[lookup[target], column] +=
                    coefficient * operator[target_label, source_label]
            end
        end
    end
    block
end

function _collective_fastpath_blockdiag_reference(basis, operator, kind)
    cache = OneBodyGeometry(basis)
    result = zeros(ComplexF64, length(basis), length(basis))
    for (sector_index, partition) in pairs(basis.sectors)
        block = collective_block(basis, operator, partition; cache)
        sector_map = kind === :commutator ?
            commutator_superoperator(block) : dissipator_superoperator(block)
        indices = basis.offsets[sector_index]:(basis.offsets[sector_index + 1] - 1)
        result[indices, indices] .= sector_map
    end
    result
end

@testset "fully symmetric collective occupation lowering" begin
    PID = _COLLECTIVE_FASTPATH_PID

    # For N=1, ascending GT order is the reverse of the local matrix order.
    # This catches an otherwise tempting, but incorrect, direct matrix copy.
    one_body = ComplexF64[
        0.2 + 0.1im  0.3 - 0.4im  -0.1 + 0.2im
        0.7 + 0.2im -0.6 + 0.3im   0.5 - 0.1im
       -0.2 + 0.8im  0.4 + 0.6im   0.9 - 0.2im
    ]
    basis1 = PIBasis(1, 3; sectors=[(1, 0, 0)])
    geometry1 = PID._SymmetricCollectiveGeometry(basis1, Float64)
    block1 = PID._symmetric_collective_block(
        basis1, one_body, only(basis1.sectors), geometry1)
    @test block1 ≈ reverse(one_body; dims=(1, 2)) atol=2e-15 rtol=2e-15

    # Qubit and qutrit blocks agree with a separately assembled occupation
    # action, including complex off-diagonal entries.
    cases = (
        (4, 2, ComplexF64[0.2 + 0.1im 0.7 - 0.3im;
                          -0.4 + 0.2im -0.5 + 0.6im]),
        (3, 3, one_body),
    )
    for (N, d, operator) in cases
        sector = ntuple(index -> index == 1 ? N : 0, d)
        basis = PIBasis(N, d; sectors=[sector])
        geometry = PID._SymmetricCollectiveGeometry(basis, Float64)
        actual = PID._symmetric_collective_block(
            basis, operator, only(basis.sectors), geometry)
        expected = _collective_fastpath_occupation_block(basis, operator)
        @test actual ≈ expected atol=2e-13 rtol=2e-13

        identity_block = PID._symmetric_collective_block(
            basis, Matrix{ComplexF64}(I, d, d), only(basis.sectors), geometry)
        @test identity_block ≈ N * Matrix{ComplexF64}(I, size(identity_block)...)

        second = reverse(operator; dims=1) +
                 im * reverse(operator; dims=2)
        lifted_first = expected
        lifted_second = _collective_fastpath_occupation_block(basis, second)
        lifted_commutator = _collective_fastpath_occupation_block(
            basis, operator * second - second * operator)
        @test lifted_first * lifted_second - lifted_second * lifted_first ≈
              lifted_commutator atol=2e-12 rtol=2e-12
    end
end

@testset "collective sparse and matrix-free fast paths" begin
    PID = _COLLECTIVE_FASTPATH_PID
    rng = MersenneTwister(0x51c011ec)

    for (N, d) in ((4, 2), (3, 3))
        sector = ntuple(index -> index == 1 ? N : 0, d)
        basis = PIBasis(N, d; sectors=[sector])
        raw_h = randn(rng, ComplexF64, d, d)
        hamiltonian = (raw_h + raw_h') / 2
        jump = randn(rng, ComplexF64, d, d)
        hamiltonian_rate = 0.17
        jump_rate = 0.23
        model = PIModel(basis, (
            CollectiveHamiltonian(hamiltonian; rate=hamiltonian_rate),
            CollectiveJump(jump; rate=jump_rate),
        ))

        context = PID.TermCompileContext(model)
        @test context.onebody isa PID._SymmetricCollectiveGeometry
        lifted_h = _collective_fastpath_occupation_block(basis, hamiltonian)
        lifted_l = _collective_fastpath_occupation_block(basis, jump)
        dense_reference =
            hamiltonian_rate * commutator_superoperator(lifted_h) +
            jump_rate * dissipator_superoperator(lifted_l)

        sparse_generator = liouvillian(
            model; representation=:sparse, memory_budget=Inf)
        @test issparse(sparse_generator)
        @test Matrix(sparse_generator) ≈ dense_reference atol=3e-12 rtol=3e-12

        plan = LiouvillianPlan(model)
        workspace = LiouvillianWorkspace(plan)
        input = randn(rng, ComplexF64, length(basis))
        output = similar(input)
        apply!(output, plan, input, 0.0, nothing, workspace)
        @test output ≈ dense_reference * input atol=4e-12 rtol=4e-12
        apply_adjoint!(output, plan, input, 0.0, nothing, workspace)
        @test output ≈ adjoint(dense_reference) * input atol=4e-12 rtol=4e-12

        matrixfree = liouvillian(
            model; representation=:matrixfree, memory_budget=Inf)
        @test matrixfree * input ≈ dense_reference * input atol=4e-12 rtol=4e-12
        @test adjoint(matrixfree) * input ≈
              adjoint(dense_reference) * input atol=4e-12 rtol=4e-12
    end

    # Sparse-first block assembly must retain ladder sparsity and avoid a
    # dense m^2-by-m^2 temporary.  This is an allocation contract, not a
    # machine-dependent timing assertion.
    m = 24
    ladder = spzeros(ComplexF64, m, m)
    for column in 2:m
        ladder[column - 1, column] = sqrt(column - 1)
    end
    loss = ladder' * ladder
    sparse_commutator = PID._sparse_commutator_block(ladder)
    sparse_dissipator = PID._sparse_dissipator_block(ladder, loss)
    @test issparse(sparse_commutator)
    @test issparse(sparse_dissipator)
    @test Matrix(sparse_commutator) ≈ commutator_superoperator(Matrix(ladder))
    @test Matrix(sparse_dissipator) ≈ dissipator_superoperator(Matrix(ladder))
    @test nnz(sparse_commutator) <= 4m^2
    @test nnz(sparse_dissipator) <= 3m^2
    PID._sparse_dissipator_block(ladder, loss)
    GC.gc()
    allocated = @allocated PID._sparse_dissipator_block(ladder, loss)
    dense_temporary_bytes = m^4 * sizeof(ComplexF64)
    @test allocated < dense_temporary_bytes
end

@testset "collective sparse resource safeguards" begin
    PID = _COLLECTIVE_FASTPATH_PID
    spin = spin_matrices()

    # A large Dicke ladder has O(N^2) PI coordinates but only O(N^2)
    # Liouvillian support.  The prepared support estimate must therefore let
    # :auto select sparse without charging a fictitious dense n_PI^2 matrix.
    basis = PIBasis(64, 2; sectors=[(64, 0)])
    model = PIModel(basis, (CollectiveJump(spin.jm; rate=0.3),))
    plan = LiouvillianPlan(model)
    bounds = PID._performance_sparse_materialization_bounds(plan)
    @test bounds.structured
    @test bounds.contribution_upper_bound < big(length(basis))^2
    materialized = PID._matrix_from_plan(plan)
    @test nnz(materialized) <= bounds.retained_nnz_upper_bound
    @test Base.summarysize(materialized) <= bounds.operator_bytes + 1024

    preparation = PID._model_preparation_bytes(model)
    sparse_peak = BigInt(Base.summarysize(plan)) + bounds.peak_bytes
    budget = Int(max(preparation, sparse_peak) + 4096)
    dense_operator = big(length(basis))^2 *
        (sizeof(ComplexF64) + sizeof(Int)) +
        (big(length(basis)) + 1) * sizeof(Int)
    @test sparse_peak < BigInt(Base.summarysize(plan)) + 3dense_operator
    compiled = compile(model; backend=:auto, memory_budget=budget)
    @test compiled.backend === :sparse
    @test compiled.estimates.sparse_structure_supported
    @test compiled.estimates.sparse_retained_nnz_upper_bound ==
          bounds.retained_nnz_upper_bound
    sparse_report = recommend_solver(
        compiled; algorithm=:direct, memory_budget=Inf)
    @test sparse_report.sparse_structure_supported
    @test sparse_report.sparse_operator_upper_bytes ==
          Base.summarysize(compiled.operator)
    @test iszero(sparse_report.sparse_materialization_peak_upper_bytes)

    prepared_matrixfree = compile(
        model; backend=:matrixfree, memory_budget=Inf)
    prepared_bounds = PID._performance_sparse_materialization_bounds(
        prepared_matrixfree.plan)
    prepared_report = recommend_solver(
        prepared_matrixfree; algorithm=:direct, memory_budget=Inf)
    @test prepared_report.sparse_structure_supported
    @test prepared_report.sparse_operator_upper_bytes ==
          prepared_bounds.operator_bytes
    @test prepared_report.sparse_materialization_peak_upper_bytes ==
          prepared_bounds.peak_bytes
    @test prepared_report.resources.setup.bytes ==
          prepared_bounds.peak_bytes - prepared_bounds.operator_bytes

    # Local gains and multi-sector collective terms still need the complete
    # one-box geometry.  A budget that covers the lightweight symmetric setup
    # is intentionally not presented as sufficient for that general route.
    small_symmetric_basis = PIBasis(12, 2; sectors=[(12, 0)])
    symmetric_model = PIModel(small_symmetric_basis,
        (CollectiveJump(spin.jm),))
    full_basis = PIBasis(12, 2)
    local_model = PIModel(full_basis, (LocalJump(spin.jm),))
    symmetric_preparation = PID._model_preparation_bytes(symmetric_model)
    local_preparation = PID._model_preparation_bytes(local_model)
    @test symmetric_preparation < local_preparation
    @test isnothing(PID._require_model_preparation_budget(
        symmetric_model, symmetric_preparation))
    @test_throws ArgumentError PID._require_model_preparation_budget(
        local_model, symmetric_preparation)
    symmetric_report = recommend_solver(symmetric_model; memory_budget=Inf)
    local_report = recommend_solver(local_model; memory_budget=Inf)
    symmetric_geometry = PID._estimate_symmetric_collective_geometry(
        small_symmetric_basis)
    @test symmetric_report.geometry_setup_upper_bytes ==
          symmetric_geometry.setup_bytes
    @test local_report.geometry_setup_upper_bytes ==
          estimate_geometry_bytes(full_basis).setup_bytes

    # A driven operator may change support, so it must retain the safe dense
    # fallback even though its block construction uses occupation geometry.
    prototype = ComplexF64[0 1; 0 0]
    schedule = InPlaceTimeOperator(
        prototype, (destination, time, parameters) -> nothing)
    driven_plan = LiouvillianPlan(PIModel(small_symmetric_basis,
        (CollectiveJump(schedule),)))
    driven_bounds = PID._performance_sparse_materialization_bounds(driven_plan)
    @test !driven_bounds.structured
    @test driven_bounds.retained_nnz_upper_bound ==
          big(length(small_symmetric_basis))^2
end

@testset "driven collective occupation lowering" begin
    PID = _COLLECTIVE_FASTPATH_PID
    rng = MersenneTwister(0xd71a)
    basis = PIBasis(3, 3; sectors=[(3, 0, 0)])
    raw_h0 = randn(rng, ComplexF64, 3, 3)
    raw_h1 = randn(rng, ComplexF64, 3, 3)
    h0 = (raw_h0 + raw_h0') / 2
    h1 = (raw_h1 + raw_h1') / 2
    l0 = randn(rng, ComplexF64, 3, 3)
    l1 = randn(rng, ComplexF64, 3, 3)
    h_schedule = InPlaceTimeOperator(h0, (destination, time, parameters) -> begin
        @. destination = cos(time) * h0 + parameters.mix * sin(time) * h1
        nothing
    end)
    l_schedule = InPlaceTimeOperator(l0, (destination, time, parameters) -> begin
        @. destination = l0 + parameters.mix * time * l1
        nothing
    end)
    model = PIModel(basis, (
        CollectiveHamiltonian(h_schedule; rate=0.19),
        CollectiveJump(l_schedule; rate=0.07),
    ))
    context = PID.TermCompileContext(model)
    @test context.onebody isa PID._SymmetricCollectiveGeometry

    time = 0.31
    parameters = (mix=0.43,)
    instantaneous_h = cos(time) * h0 + parameters.mix * sin(time) * h1
    instantaneous_l = l0 + parameters.mix * time * l1
    lifted_h = _collective_fastpath_occupation_block(basis, instantaneous_h)
    lifted_l = _collective_fastpath_occupation_block(basis, instantaneous_l)
    reference = 0.19 * commutator_superoperator(lifted_h) +
                0.07 * dissipator_superoperator(lifted_l)

    plan = LiouvillianPlan(model)
    workspace = LiouvillianWorkspace(plan)
    input = randn(rng, ComplexF64, length(basis))
    output = similar(input)
    apply!(output, plan, input, time, parameters, workspace)
    @test output ≈ reference * input atol=5e-12 rtol=5e-12
    apply!(output, plan, input, time, parameters, workspace)
    @test (@allocated apply!(
        output, plan, input, time, parameters, workspace)) == 0
    apply_adjoint!(output, plan, input, time, parameters, workspace)
    @test output ≈ adjoint(reference) * input atol=5e-12 rtol=5e-12

    frozen = freeze(model; time, parameters, representation=:sparse)
    @test Matrix(frozen) ≈ reference atol=5e-12 rtol=5e-12
end

@testset "collective fast-path precision and fallbacks" begin
    PID = _COLLECTIVE_FASTPATH_PID

    # A mixed-sector collective model and every local gain model retain the
    # complete one-box geometry.  Their numerical lowering remains unchanged.
    mixed_basis = PIBasis(4, 2)
    lowering = ComplexF64[0 1; 0 0]
    mixed_model = PIModel(mixed_basis, (
        CollectiveHamiltonian(ComplexF64[0.2 0.3; 0.3 -0.1]; rate=0.11),
        CollectiveJump(lowering; rate=0.17),
    ))
    mixed_context = PID.TermCompileContext(mixed_model)
    @test mixed_context.onebody isa OneBodyGeometry
    mixed_reference =
        0.11 * _collective_fastpath_blockdiag_reference(
            mixed_basis, ComplexF64[0.2 0.3; 0.3 -0.1], :commutator) +
        0.17 * _collective_fastpath_blockdiag_reference(
            mixed_basis, lowering, :dissipator)
    @test Matrix(liouvillian(mixed_model; representation=:sparse,
                            memory_budget=Inf)) ≈ mixed_reference atol=3e-12 rtol=3e-12

    local_model = PIModel(mixed_basis, (
        LocalJump(lowering; rate=0.13),
        CollectiveJump(lowering; rate=0.07),
    ))
    @test PID.TermCompileContext(local_model).onebody isa OneBodyGeometry
    local_sparse = liouvillian(
        local_model; representation=:sparse, memory_budget=Inf)
    local_matrixfree = liouvillian(
        local_model; representation=:matrixfree, memory_budget=Inf)
    local_input = ComplexF64.(1:length(mixed_basis))
    @test local_matrixfree * local_input ≈ local_sparse * local_input

    symmetric_basis = PIBasis(4, 2; sectors=[(4, 0)])
    @test_throws ArgumentError PIModel(
        symmetric_basis, (LocalJump(lowering),))
    local_hamiltonian = PIModel(symmetric_basis, (
        LocalHamiltonian(ComplexF64[0 1; 1 0]),))
    @test PID.TermCompileContext(local_hamiltonian).onebody isa
          PID._SymmetricCollectiveGeometry

    basis32 = PIBasis(5, 2; sectors=[(5, 0)])
    operator32 = ComplexF32[0.2f0 0.7f0 - 0.1f0im;
                            -0.3f0 + 0.2f0im -0.4f0]
    model32 = PIModel(basis32, (CollectiveJump(operator32; rate=0.2f0),))
    context32 = PID.TermCompileContext(model32)
    @test context32.onebody isa PID._SymmetricCollectiveGeometry
    block32 = only(PID._collective_blocks(operator32, context32))
    @test eltype(block32) === ComplexF32
    @test isapprox(block32,
        _collective_fastpath_occupation_block(basis32, operator32);
        atol=2f-5, rtol=2f-5)
    @test eltype(LiouvillianPlan(model32)) === ComplexF32

    setprecision(128) do
        basis_big = PIBasis(3, 2; sectors=[(3, 0)])
        operator_big = Complex{BigFloat}[
            big"0.125" complex(big"0.75", big"0.2")
            complex(big"0.0", -big"0.4") -big"0.375"
        ]
        model_big = PIModel(
            basis_big, (CollectiveJump(operator_big; rate=big"0.2"),))
        context_big = PID.TermCompileContext(model_big)
        @test context_big.onebody isa PID._SymmetricCollectiveGeometry
        block_big = only(PID._collective_blocks(operator_big, context_big))
        @test eltype(block_big) === Complex{BigFloat}
        @test block_big ≈
              _collective_fastpath_occupation_block(basis_big, operator_big)
        @test eltype(LiouvillianPlan(model_big)) === Complex{BigFloat}
    end

    # Float16 uses the native occupation path below the established
    # cancellation threshold, and the guarded general/wider path above it.
    small16 = PIBasis(8, 2; sectors=[(8, 0)])
    operator16 = Complex{Float16}[0 1; 1 0]
    context16 = PID.TermCompileContext(
        PIModel(small16, (CollectiveJump(operator16),)))
    @test context16.onebody isa PID._SymmetricCollectiveGeometry
    @test all(isfinite, only(PID._collective_blocks(operator16, context16)))

    risk16 = PIBasis(34, 2; sectors=[(34, 0)])
    risk_model = PIModel(risk16, (CollectiveJump(operator16),))
    risk_context = PID.TermCompileContext(risk_model)
    @test risk_context.onebody isa OneBodyGeometry
    @test all(isfinite, only(PID._collective_blocks(operator16, risk_context)))
    risk_schedule = InPlaceTimeOperator(
        operator16, (destination, time, parameters) -> nothing)
    error = try
        LiouvillianPlan(PIModel(risk16, (CollectiveJump(risk_schedule),)))
        nothing
    catch caught
        caught
    end
    @test error isa ArgumentError
    @test occursin("wider InPlaceTimeOperator prototype", sprint(showerror, error))
end
