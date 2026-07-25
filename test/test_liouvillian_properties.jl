using Test
using LinearAlgebra
using Random
using SparseArrays
using PermutationalInvariantDynamics

const PIDLiouvillianProperties = PermutationalInvariantDynamics

function _property_random_matrix(rng, ::Type{R}, rows, columns=rows) where {R}
    randn(rng, Complex{R}, rows, columns)
end

function _property_random_hermitian(rng, ::Type{R}, dimension) where {R}
    matrix = _property_random_matrix(rng, R, dimension)
    (matrix + matrix') / R(2)
end

function _property_random_pi_operator(rng, basis, ::Type{R};
                                      hermitian=false) where {R}
    operator = PIOperator(basis; T=R)
    for partition in basis.sectors
        block = coefficient_block(operator, partition)
        random_block = _property_random_matrix(
            rng, R, size(block, 1), size(block, 2))
        block .= hermitian ?
            (random_block + random_block') / R(2) : random_block
    end
    operator
end

function _property_model(
    rng, N, d, ::Type{R}; direct_terms=false, pbody_terms=true) where {R}
    basis = PIBasis(N, d)
    local_h = _property_random_hermitian(rng, R, d)
    collective_h = _property_random_hermitian(rng, R, d)
    local_jump = _property_random_matrix(rng, R, d)
    collective_jump = _property_random_matrix(rng, R, d)

    pair_left = _property_random_hermitian(rng, R, d)
    pair_right = _property_random_hermitian(rng, R, d)
    # This is invariant under exchange of the two particle slots, as required
    # by the Appendix-D two-body constructors.
    pair_h = (kron(pair_left, pair_right) +
              kron(pair_right, pair_left)) / R(2)
    pair_seed = _property_random_matrix(rng, R, d^2)
    pair_swap =
        PIDLiouvillianProperties._tensor_swap_permutation(2, d, 1)
    pair_jump =
        (pair_seed + pair_seed[pair_swap, pair_swap]) / R(2)

    onebody_terms = (
        LocalHamiltonian(local_h; rate=R(0.17)),
        CollectiveHamiltonian(collective_h; rate=-R(0.11)),
        LocalJump(local_jump; rate=R(0.23)),
        # Negative deterministic rates are permitted for time-local
        # generators. They must not change any linear-algebra invariant.
        CollectiveJump(collective_jump; rate=-R(0.07)),
    )
    higher_terms = pbody_terms ? (
        PBodyHamiltonian(pair_h, 2; rate=R(0.13)),
        LocalPBodyJump(pair_jump, 2; rate=R(0.05)),
        CollectivePBodyJump(pair_jump, 2; rate=-R(0.03)),
    ) : ()
    common_terms = (onebody_terms..., higher_terms...)

    if direct_terms
        direct_h = _property_random_pi_operator(
            rng, basis, R; hermitian=true)
        direct_jump = _property_random_pi_operator(rng, basis, R)
        return PIModel(basis, (
            common_terms...,
            DirectPIHamiltonian(direct_h; rate=R(0.09)),
            DirectPIJump(direct_jump; rate=-R(0.02)),
        ))
    end
    PIModel(basis, common_terms)
end

function _test_liouvillian_representation_properties(
    rng, model::PIModel, ::Type{R}) where {R}

    basis = model.basis
    dimension = length(basis)
    tolerance = R === Float32 ? R(3e-4) : R(3e-11)

    fused = LiouvillianPlan(model)
    # Lower both representations from the same immutable physical term plan.
    # This exercises the two execution backends without rebuilding expensive
    # Appendix-D geometry for each property probe.
    sparse_generator = PIDLiouvillianProperties._matrix_from_plan(fused)
    matrix_free = PIDLiouvillianProperties._matrixfree_liouvillian(fused)

    @test issparse(sparse_generator)
    @test issparse(fused.tracevec)
    @test eltype(sparse_generator) === Complex{R}
    @test eltype(fused) === Complex{R}
    @test eltype(matrix_free) === Complex{R}

    source = randn(rng, Complex{R}, dimension)
    probe = randn(rng, Complex{R}, dimension)
    batch = randn(rng, Complex{R}, dimension, 3)
    sparse_forward = sparse_generator * source
    sparse_adjoint = adjoint(sparse_generator) * probe
    sparse_batch = sparse_generator * batch
    sparse_adjoint_batch = adjoint(sparse_generator) * batch

    workspace = LiouvillianWorkspace(fused)
    forward = similar(source)
    adjoint_image = similar(probe)
    batch_forward = similar(batch)
    batch_adjoint = similar(batch)

    apply!(forward, fused, source, workspace)
    apply_adjoint!(adjoint_image, fused, probe, workspace)
    apply!(batch_forward, fused, batch, workspace)
    apply_adjoint!(batch_adjoint, fused, batch, workspace)

    @test forward ≈ sparse_forward atol=tolerance rtol=tolerance
    @test adjoint_image ≈ sparse_adjoint atol=tolerance rtol=tolerance
    @test batch_forward ≈ sparse_batch atol=tolerance rtol=tolerance
    @test batch_adjoint ≈ sparse_adjoint_batch atol=tolerance rtol=tolerance

    repeated = similar(batch)
    repeated_adjoint = similar(batch)
    for column in axes(batch, 2)
        apply!(
            view(repeated, :, column), fused, view(batch, :, column),
            workspace)
        apply_adjoint!(
            view(repeated_adjoint, :, column), fused,
            view(batch, :, column), workspace)
    end
    @test batch_forward ≈ repeated atol=tolerance rtol=tolerance
    @test batch_adjoint ≈ repeated_adjoint atol=tolerance rtol=tolerance

    # This duality check catches conjugation or left/right ordering bugs even
    # when forward and adjoint implementations share prepared data.
    duality_atol =
        tolerance * max(one(R), norm(forward), norm(adjoint_image))
    @test isapprox(
        dot(probe, forward), dot(adjoint_image, source);
        atol=duality_atol, rtol=tolerance)

    @test isapprox(
        matrix_free * source, sparse_forward;
        atol=tolerance, rtol=tolerance)
    @test isapprox(
        adjoint(matrix_free) * probe, sparse_adjoint;
        atol=tolerance, rtol=tolerance)
    @test isapprox(
        matrix_free * batch, sparse_batch;
        atol=tolerance, rtol=tolerance)
    @test isapprox(
        adjoint(matrix_free) * batch, sparse_adjoint_batch;
        atol=tolerance, rtol=tolerance)

    # Trace preservation is checked through the sparse physical trace
    # functional retained by the prepared plan, both explicitly and through
    # the matrix-free adjoint action.
    trace_vector = fused.tracevec
    trace_adjoint_reference = adjoint(sparse_generator) * trace_vector
    trace_adjoint = zeros(Complex{R}, dimension)
    apply_adjoint!(
        trace_adjoint, fused, trace_vector, LiouvillianWorkspace(fused))
    trace_scale = max(one(R), norm(sparse_generator, Inf))
    @test norm(trace_adjoint_reference, Inf) <= tolerance * trace_scale
    @test isapprox(
        trace_adjoint, trace_adjoint_reference;
        atol=tolerance * trace_scale, rtol=tolerance)

    # Every real-rate Hamiltonian/Lindblad-like contribution preserves the
    # Hermitian operator subspace, including a negative deterministic rate.
    hermitian_source = _property_random_pi_operator(
        rng, basis, R; hermitian=true)
    hermitian_image_data = zeros(Complex{R}, dimension)
    apply!(
        hermitian_image_data, fused, hermitian_source.data,
        LiouvillianWorkspace(fused))
    hermitian_image = PIOperator(basis, hermitian_image_data)
    @test ishermitian(
        hermitian_image; atol=tolerance * max(one(R), norm(hermitian_image_data)),
        rtol=tolerance)
    @test abs(trace(hermitian_image)) <=
        tolerance * max(one(R), norm(hermitian_image_data))
end

@testset "randomized Liouvillian representation properties" begin
    # Fresh deterministic RNGs keep the generated cases independent of test
    # ordering and of any Krylov breakdown-recovery draws elsewhere.
    _test_liouvillian_representation_properties(
        MersenneTwister(0x51a7_0001),
        _property_model(
            MersenneTwister(0x51a7_1001), 3, 2, Float64;
            direct_terms=true),
        Float64)

    _test_liouvillian_representation_properties(
        MersenneTwister(0x51a7_0002),
        _property_model(
            MersenneTwister(0x51a7_1002), 2, 3, Float32;
            pbody_terms=false),
        Float32)
end

@testset "exact-rate Float32 lowering remains narrow" begin
    basis = PIBasis(2, 2)
    spin = spin_matrices(2; T=Float32)
    exact_model = PIModel(basis, (
        LocalHamiltonian(spin.jx; rate=big(2) // big(7),
                         hbar=big(3) // big(2)),
        LocalJump(spin.jm; rate=-big(1) // big(11)),
    ))
    rounded_model = PIModel(basis, (
        LocalHamiltonian(spin.jx; rate=Float32(2 // 7),
                         hbar=Float32(3 // 2)),
        LocalJump(spin.jm; rate=-Float32(1 // 11)),
    ))

    exact_plan = LiouvillianPlan(exact_model)
    exact_sparse = PIDLiouvillianProperties._matrix_from_plan(exact_plan)
    rounded_sparse = liouvillian(
        rounded_model; representation=:sparse, memory_budget=Inf)
    @test eltype(exact_plan) === ComplexF32
    @test eltype(exact_sparse) === ComplexF32
    @test Matrix(exact_sparse) ≈ Matrix(rounded_sparse) atol=8f-7 rtol=8f-7

    rng = MersenneTwister(0x51a7_2001)
    batch = randn(rng, ComplexF32, length(basis), 2)
    output = similar(batch)
    apply!(
        output, exact_plan, batch, 0f0, nothing,
        LiouvillianWorkspace(exact_plan))
    @test output ≈ exact_sparse * batch atol=3f-6 rtol=3f-6
end
