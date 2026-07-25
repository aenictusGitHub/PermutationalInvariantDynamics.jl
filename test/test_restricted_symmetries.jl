const _PIDRestrictedSymmetry = PermutationalInvariantDynamics
if !isdefined(_PIDRestrictedSymmetry,:SymmetryCoordinateRestriction)
    Base.include(_PIDRestrictedSymmetry,
        joinpath(@__DIR__,"..","src","restricted_symmetries.jl"))
end

@testset "certified compressed symmetry coordinates" begin
    PID=_PIDRestrictedSymmetry
    basis=PIBasis(1,3)
    symmetry=Diagonal(ComplexF64[1,1,-1])
    restriction=PID.diagonal_symmetry_restriction(
        basis,symmetry;charge=1)
    @test length(restriction)==4
    @test size(restriction)==(length(basis),4)
    indices=PID.retained_indices(restriction)
    @test issorted(indices)
    @test length(unique(indices))==length(indices)
    indices[1]=length(basis)
    @test PID.retained_indices(restriction)[1]!=indices[1]

    reduced=ComplexF64[1+im,2-im,3+2im,4]
    ambient=fill(ComplexF64(NaN),length(basis))
    PID.embed!(ambient,restriction,reduced)
    @test ambient[PID.retained_indices(restriction)]==reduced
    @test all(iszero,ambient[.!restriction.mask])
    roundtrip=similar(reduced)
    PID.restrict!(roundtrip,restriction,ambient)
    @test roundtrip==reduced

    explicit=PID.SymmetryCoordinateRestriction(
        basis,reverse(PID.retained_indices(restriction));label=:test)
    @test PID.retained_indices(explicit)==PID.retained_indices(restriction)
    @test_throws ArgumentError PID.SymmetryCoordinateRestriction(basis,[1,1])
    @test_throws BoundsError PID.SymmetryCoordinateRestriction(basis,[0])
    @test_throws ArgumentError PID.diagonal_symmetry_restriction(
        basis,ones(ComplexF64,3,3);charge=1)

    # The jump acts only inside the + Hilbert-charge subspace.  The full
    # qutrit generator has an additional stationary state in the - subspace,
    # whereas the compressed + block has the unique state |1><1|.
    jump=zeros(ComplexF64,3,3);jump[1,2]=1
    model=PIModel(basis,(LocalJump(jump),))
    source=compile(model;backend=:matrixfree)
    certificate=PID.restriction_invariance(source,restriction)
    @test certificate.invariant
    @test certificate.validation===:exhaustive_coordinate_probe
    @test certificate.applications==length(restriction)
    @test certificate.leakage_norm<=certificate.tolerance

    operator=PID.RestrictedLiouvillian(source,restriction)
    @test size(operator)==(4,4)
    @test eltype(operator)===ComplexF64
    @test operator.backend===:lowered
    @test operator.compressed_source!==nothing
    @test operator.compatibility_workspace===nothing
    @test operator.certificate.invariant
    @test PID.restricted_trace_vector(restriction,ComplexF64)==operator.tracevec
    @test operator.tracevec isa SparseVector
    @test PID._operator_trace_functional(operator)===operator.tracevec
    @test PID._operator_trace_vector(operator)===operator.tracevec
    @test PID._operator_has_adjoint(operator)
    @test PID._linear_operator_workspace(operator) isa
          PID._RestrictedKernelWorkspace

    sparse_full=Matrix(liouvillian(model;representation=:sparse))
    reference=sparse_full[PID.retained_indices(restriction),
                          PID.retained_indices(restriction)]
    rng=MersenneTwister(910)
    input=randn(rng,ComplexF64,length(restriction))
    output=similar(input)
    work=PID.RestrictedLiouvillianWorkspace(source,restriction)
    PID.apply!(output,operator,input,work)
    @test output≈reference*input atol=3e-13
    reduced_work=PID.RestrictedLiouvillianWorkspace(operator)
    @test !hasproperty(reduced_work,:ambient_input)
    @test_throws ArgumentError PID.restriction_full_residual(
        operator,input;workspace=reduced_work)
    PID.apply!(output,operator,input,reduced_work)
    @test output≈reference*input atol=3e-13
    @test operator*input≈reference*input atol=3e-13
    PID.apply_adjoint!(output,operator,input,work)
    @test output≈adjoint(reference)*input atol=3e-13
    PID.apply_adjoint!(output,operator,input,reduced_work)
    @test output≈adjoint(reference)*input atol=3e-13
    @test adjoint(operator)*input≈adjoint(reference)*input atol=3e-13

    sparse_source=liouvillian(model;representation=:sparse)
    compressed=PID.RestrictedLiouvillian(sparse_source,restriction)
    @test compressed.backend===:compressed
    @test compressed.certificate.validation===:exhaustive_matrix_scan
    @test compressed.certificate.applications==0
    @test compressed.compatibility_workspace===nothing
    @test PID._linear_operator_workspace(compressed)===nothing
    @test PID._operator_has_adjoint(compressed)
    @test size(compressed.compressed_source)==(length(restriction),length(restriction))
    @test compressed*input≈reference*input atol=3e-13
    PID.apply!(output,compressed,input,0.0,nothing)
    @test output≈reference*input atol=3e-13
    PID.apply_adjoint!(output,compressed,input,0.0,nothing)
    @test output≈adjoint(reference)*input atol=3e-13
    batch=randn(rng,ComplexF64,length(restriction),3)
    @test compressed*batch≈reference*batch atol=3e-13
    @test adjoint(compressed)*batch≈adjoint(reference)*batch atol=3e-13

    restricted_period=floquet_map(operator,0.01;steps=4)
    @test restricted_period.tracevec≈operator.tracevec
    period_work=FloquetWorkspace(restricted_period)
    @test period_work.source_workspace isa PID._RestrictedKernelWorkspace
    PID.apply!(output,restricted_period,input,period_work)
    @test output≈exp(0.01 * reference)*input atol=2e-12 rtol=2e-12
    PID.apply_adjoint!(output,restricted_period,input,period_work)
    @test output≈exp(0.01 * adjoint(reference))*input atol=2e-12 rtol=2e-12
    lowered_batch=similar(batch)
    PID.apply!(lowered_batch,operator,batch,0.0,nothing,reduced_work)
    @test lowered_batch≈reference*batch atol=3e-13
    PID.apply_adjoint!(lowered_batch,operator,batch,0.0,nothing,reduced_work)
    @test lowered_batch≈adjoint(reference)*batch atol=3e-13
    compressed_direct=steady_state(compressed;
        trace_vector=compressed.tracevec,method=:direct,
        return_info=true,atol=1e-12,rtol=1e-10)
    @test compressed_direct.residual<2e-10
    @test compressed_direct.trace_error<2e-11
    @test_throws ArgumentError PID.restriction_invariance(
        sparse_source,restriction;atol=floatmax(Float64),
        rtol=floatmax(Float64))
    nonfinite_source=copy(sparse_source)
    nonfinite_source[PID.retained_indices(restriction)[1],
                     PID.retained_indices(restriction)[1]]=Inf
    @test_throws ArgumentError PID.restriction_invariance(
        nonfinite_source,restriction)
    @test_throws ArgumentError PID.RestrictedLiouvillian(
        source,restriction;backend=:compressed)
    @test PID.RestrictedLiouvillian(
        sparse_source,restriction;backend=:embedded).backend===:embedded
    @test_throws ArgumentError PID.RestrictedLiouvillian(
        sparse_source,restriction;backend=:invalid)

    # An invariant but non-Cartesian explicit mask cannot be expressed as a
    # ket-pattern rectangle. It remains on the certified embedded fallback.
    qubit_basis=PIBasis(1,2);z=ComplexF64[1 0;0 -1]
    diagonal_only=PID.SymmetryCoordinateRestriction(qubit_basis,[1,4])
    diagonal_source=compile(PIModel(qubit_basis,(LocalJump(z),));
                            backend=:matrixfree)
    diagonal_operator=PID.RestrictedLiouvillian(
        diagonal_source,diagonal_only)
    @test diagonal_operator.backend===:embedded
    @test_throws ArgumentError PID.RestrictedLiouvillian(
        diagonal_source,diagonal_only;backend=:lowered)

    # Exercise every fixed lowered kernel family, including rectangular ket
    # and bra subblocks in Hamiltonian and collective dissipator actions.
    diagonal_h=Diagonal(ComplexF64[0.3,-0.2,0.7])
    diagonal_c=Diagonal(ComplexF64[0.1,0.4,0.9])
    mixed_model=PIModel(basis,(
        LocalHamiltonian(Matrix(diagonal_h);rate=0.37),
        CollectiveJump(Matrix(diagonal_c);rate=0.23),
        LocalJump(jump;rate=0.11)))
    mixed_source=compile(mixed_model;backend=:matrixfree)
    mixed_operator=PID.RestrictedLiouvillian(mixed_source,restriction)
    @test mixed_operator.backend===:lowered
    mixed_reference=Matrix(liouvillian(mixed_model;representation=:sparse))[
        PID.retained_indices(restriction),PID.retained_indices(restriction)]
    mixed_work=PID.RestrictedLiouvillianWorkspace(mixed_operator)
    PID.apply!(output,mixed_operator,input,mixed_work)
    @test output≈mixed_reference*input atol=5e-13
    PID.apply_adjoint!(output,mixed_operator,input,mixed_work)
    @test output≈adjoint(mixed_reference)*input atol=5e-13

    # Multi-sector qubit parity exercises genuinely rectangular ket/bra
    # blocks. Equal-charge density restrictions share their left/right block
    # storage, while off-diagonal charge restrictions retain distinct shapes.
    many_basis=PIBasis(4,2)
    parity=Diagonal(ComplexF64[1,-1])
    z=ComplexF64[1 0;0 -1]
    many_model=PIModel(many_basis,(
        LocalHamiltonian(z;rate=0.19),
        CollectiveJump(z;rate=0.07),
        LocalJump(z;rate=0.05)))
    many_source=compile(many_model;backend=:matrixfree)
    density_restriction=PID.diagonal_symmetry_restriction(
        many_basis,parity;charge=1)
    density_operator=PID.RestrictedLiouvillian(
        many_source,density_restriction)
    @test density_operator.backend===:lowered
    for kernel in density_operator.compressed_source.plan.kernels
        if hasproperty(kernel,:left_blocks)
            @test all(kernel.left_blocks[index]===kernel.right_blocks[index]
                      for index in eachindex(kernel.left_blocks))
        end
        if hasproperty(kernel,:qblocks)
            @test all(first(pair)===last(pair) for pair in kernel.qblocks)
        end
    end

    rectangular_restriction=PID.diagonal_symmetry_restriction(
        many_basis,parity;ket_charge=1,bra_charge=-1)
    rectangular_operator=PID.RestrictedLiouvillian(
        many_source,rectangular_restriction)
    @test rectangular_operator.backend===:lowered
    @test all(iszero,rectangular_operator.tracevec)
    many_sparse=Matrix(liouvillian(many_model;representation=:sparse))
    rectangular_indices=PID.retained_indices(rectangular_restriction)
    rectangular_reference=many_sparse[rectangular_indices,rectangular_indices]
    rectangular_input=randn(rng,ComplexF64,length(rectangular_restriction))
    rectangular_output=similar(rectangular_input)
    rectangular_work=PID.RestrictedLiouvillianWorkspace(rectangular_operator)
    PID.apply!(rectangular_output,rectangular_operator,rectangular_input,
               rectangular_work)
    @test rectangular_output≈rectangular_reference*rectangular_input atol=8e-13
    PID.apply_adjoint!(rectangular_output,rectangular_operator,
                       rectangular_input,rectangular_work)
    @test rectangular_output≈adjoint(rectangular_reference)*rectangular_input atol=8e-13

    # Local p-body gains must remain rectangular Schur sandwiches after
    # restriction. Expanding them into reduced coordinate triplets would
    # restore quartic setup/storage precisely where symmetry reduction is
    # intended to help.
    pair_z=kron(z,z)
    pbody_model=PIModel(many_basis,(
        LocalPBodyJump(pair_z,2;rate=0.03),))
    pbody_source=compile(pbody_model;backend=:matrixfree)
    pbody_operator=PID.RestrictedLiouvillian(
        pbody_source,density_restriction)
    @test pbody_operator.backend===:lowered
    @test first(pbody_operator.compressed_source.plan.kernels) isa
        PID._RestrictedFactorizedLocalPBodyJumpKernel
    @test !hasproperty(
        first(pbody_operator.compressed_source.plan.kernels),:I)
    pbody_indices=PID.retained_indices(density_restriction)
    pbody_reference=Matrix(liouvillian(
        pbody_model;representation=:sparse))[pbody_indices,pbody_indices]
    pbody_input=randn(rng,ComplexF64,length(density_restriction))
    pbody_output=similar(pbody_input)
    pbody_work=PID.RestrictedLiouvillianWorkspace(pbody_operator)
    PID.apply!(pbody_output,pbody_operator,pbody_input,pbody_work)
    @test pbody_output≈pbody_reference*pbody_input atol=2e-12
    PID.apply_adjoint!(
        pbody_output,pbody_operator,pbody_input,pbody_work)
    @test pbody_output≈adjoint(pbody_reference)*pbody_input atol=2e-12

    # Response solvers must retain the reduced application scratch rather
    # than reintroducing two ambient PI vectors behind the lowered operator.
    response_work=ResponseWorkspace(rectangular_operator;
        krylovdim=8,mode=:linear)
    @test response_work.action_workspace isa PID._RestrictedKernelWorkspace
    @test !hasproperty(response_work.action_workspace,:ambient_input)
    PID._response_apply!(rectangular_output,rectangular_operator,
                         rectangular_input,response_work)
    @test rectangular_output≈rectangular_reference*rectangular_input atol=8e-13
    compressed_response=ResponseWorkspace(compressed;krylovdim=4,mode=:linear)
    @test compressed_response.action_workspace===nothing

    stationary=PID.restricted_steady_state(operator;return_info=true,
        atol=1e-12,rtol=1e-10,krylovdim=4,maxiter=40)
    @test stationary.state isa PIState
    @test abs(trace(stationary.state)-1)<2e-11
    @test stationary.full_residual<2e-10
    @test stationary.leakage_residual<2e-12
    @test diagnostics(stationary.state).valid
    residual=PID.restriction_full_residual(
        operator,stationary.reduced_state;workspace=work)
    @test residual.residual≈stationary.full_residual atol=1e-14
    @test residual.trace≈1 atol=2e-11

    offdiagonal=PID.diagonal_symmetry_restriction(
        basis,symmetry;ket_charge=1,bra_charge=-1)
    @test all(iszero,PID.restricted_trace_vector(offdiagonal,ComplexF64))

    # A jump from the selected + support into the - support must be rejected
    # by the exhaustive leakage certificate rather than silently projected.
    leaking_jump=zeros(ComplexF64,3,3);leaking_jump[3,1]=1
    leaking=compile(PIModel(basis,(LocalJump(leaking_jump),));backend=:matrixfree)
    leaking_report=PID.restriction_invariance(leaking,restriction)
    @test !leaking_report.invariant
    @test leaking_report.leakage_norm>leaking_report.tolerance
    @test_throws ArgumentError PID.RestrictedLiouvillian(leaking,restriction)

    # Dominant storage and trace data follow the source precision.
    symmetry32=Diagonal(ComplexF32[1,1,-1])
    restriction32=PID.diagonal_symmetry_restriction(
        basis,symmetry32;charge=1f0,atol=1f-6,rtol=1f-6)
    jump32=ComplexF32.(jump)
    source32=compile(PIModel(basis,(LocalJump(jump32;rate=1f0),));
                     backend=:matrixfree)
    operator32=PID.RestrictedLiouvillian(source32,restriction32;
                                         rtol=1f-5)
    @test eltype(operator32)===ComplexF32
    @test eltype(operator32.tracevec)===ComplexF32
    result32=PID.restricted_steady_state(operator32;return_info=true,
        atol=1f-5,rtol=1f-4,krylovdim=4,maxiter=40)
    @test result32.state isa PIState{Float32}
    @test result32.full_normalized_residual<=2f-4
end
