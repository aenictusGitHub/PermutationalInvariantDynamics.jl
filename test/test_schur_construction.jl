@testset "Schur block construction and metadata" begin
    basis=PIBasis(3,2)
    operator=PIOperator(basis;T=Float32)
    for (sector_index,partition) in pairs(basis.sectors)
        block=coefficient_block(operator,partition)
        for column in axes(block,2),row in axes(block,1)
            block[row,column]=ComplexF32(row+10column+100sector_index,
                                         row-column)
        end
    end

    physical_pairs=collect(each_schur_block(operator))
    @test first.(physical_pairs)==basis.sectors
    physical_roundtrip=operator_from_schur_blocks(basis,physical_pairs)
    @test eltype(physical_roundtrip.data)===ComplexF32
    @test physical_roundtrip.data≈operator.data atol=2eps(Float32) rtol=2eps(Float32)

    coefficient_roundtrip=operator_from_schur_blocks(
        basis,each_schur_block(operator;representation=:coefficient);
        representation=:coefficient)
    @test coefficient_roundtrip.data==operator.data

    # Physical blocks are detached, while explicit coefficient blocks retain
    # the documented mutable-view semantics.
    first_partition=first(basis.sectors)
    detached=last(first(physical_pairs))
    original=operator.data[1]
    detached[1,1]+=1
    @test operator.data[1]==original
    coefficient_pair=first(each_schur_block(operator;representation=:coefficient))
    last(coefficient_pair)[1,1]+=1
    @test operator.data[1]==original+1

    # A sector with f^nu > 1 checks the physical/coefficient conversion rather
    # than only the multiplicity-one symmetric block.
    mixed_partition=Partition((2,1))
    mixed_block=ComplexF32[1 2im;-3im 4]
    scaled=operator_from_schur_blocks(basis,[mixed_partition=>mixed_block])
    @test coefficient_block(scaled,mixed_partition)≈sqrt(Float32(2)).*mixed_block
    @test physical_block(scaled,mixed_partition)≈mixed_block
    @test all(iszero,coefficient_block(scaled,first_partition))

    # Tuple labels are accepted, inputs are copied, and an empty collection
    # gives a well-defined zero ComplexF64 object.
    source=copy(mixed_block)
    tuple_labeled=operator_from_schur_blocks(basis,[(2,1)=>source])
    source[1,1]=99
    @test physical_block(tuple_labeled,mixed_partition)[1,1]≈1
    empty_operator=operator_from_schur_blocks(basis,Pair[])
    @test eltype(empty_operator.data)===ComplexF64
    @test all(iszero,empty_operator.data)
    @test eltype(operator_from_schur_blocks(
        basis,Pair[];T=Float32).data)===ComplexF32
    @test eltype(state_from_schur_blocks(
        basis,Pair[];T=BigFloat).data)===Complex{BigFloat}
    @test_throws ArgumentError operator_from_schur_blocks(
        basis,[mixed_partition=>ComplexF64.(mixed_block)];T=Float32)

    # State construction uses identical block conventions but validates only
    # when explicitly requested.
    rho=maximally_mixed_state(basis;T=Float32)
    rebuilt=state_from_schur_blocks(
        basis,each_schur_block(rho);validate=true,
        atol=Float32(1e-6),rtol=Float32(1e-5))
    @test rebuilt.data≈rho.data atol=2eps(Float32) rtol=2eps(Float32)
    invalid=state_from_schur_blocks(
        PIBasis(1,2),[Partition((1,0))=>ComplexF64[1.1 0;0 -0.1]])
    @test !isphysical(invalid)
    @test_throws ArgumentError state_from_schur_blocks(
        invalid.basis,each_schur_block(invalid);validate=true,
        atol=1e-12,rtol=0)
    @test_throws ArgumentError state_from_schur_blocks(
        basis,Pair[];validate=false,atol=1e-12)

    # Construction rejects ambiguous or incompatible input instead of
    # overwriting, reshaping, or dropping it.
    dimension=length(basis.patterns[basis.index[mixed_partition]])
    @test_throws ArgumentError operator_from_schur_blocks(
        basis,[mixed_partition=>mixed_block,mixed_partition=>mixed_block])
    @test_throws DimensionMismatch operator_from_schur_blocks(
        basis,[mixed_partition=>zeros(ComplexF64,dimension+1,dimension+1)])
    @test_throws ArgumentError operator_from_schur_blocks(
        basis,[mixed_partition=>1])
    @test_throws ArgumentError operator_from_schur_blocks(
        basis,[:invalid=>mixed_block])
    @test_throws ArgumentError operator_from_schur_blocks(
        basis,[Partition((3,0,0))=>mixed_block])
    @test_throws ArgumentError operator_from_schur_blocks(
        basis,[mixed_partition=>mixed_block];representation=:invalid)
    @test_throws ArgumentError collect(each_schur_block(
        operator;representation=:invalid))
    restricted=PIBasis(3,2;sectors=[(3,0)])
    @test_throws ArgumentError operator_from_schur_blocks(
        restricted,[mixed_partition=>mixed_block])

    metadata=sector_metadata(basis)
    @test getproperty.(metadata,:index)==[1,2]
    @test getproperty.(metadata,:partition)==basis.sectors
    @test getproperty.(metadata,:block_dimension)==[4,2]
    @test getproperty.(metadata,:coordinate_dimension)==[16,4]
    @test getproperty.(metadata,:coordinate_range)==[1:16,17:20]
    @test getproperty.(metadata,:multiplicity)==BigInt[1,2]
    @test getproperty.(metadata,:hilbert_dimension)==BigInt[4,4]
    @test getproperty.(metadata,:spin)==[3//2,1//2]
    @test sum(item.hilbert_dimension for item in metadata)==big(2)^basis.N

    qutrit_basis=PIBasis(3,3)
    qutrit_metadata=sector_metadata(qutrit_basis)
    @test getproperty.(qutrit_metadata,:block_dimension)==[10,8,1]
    @test getproperty.(qutrit_metadata,:multiplicity)==BigInt[1,2,1]
    @test all(ismissing,getproperty.(qutrit_metadata,:spin))
    @test sum(item.hilbert_dimension for item in qutrit_metadata)==big(3)^3
    @test sum(item.coordinate_dimension for item in qutrit_metadata)==length(qutrit_basis)

    restricted_metadata=sector_metadata(restricted)
    @test only(restricted_metadata).hilbert_dimension==4
    @test sum(item.hilbert_dimension for item in restricted_metadata)<big(2)^3
    empty_basis=PIBasis(0,2;sectors=Tuple{Int,Int}[])
    @test isempty(sector_metadata(empty_basis))

    # Exact multiplicities must not narrow even when they exceed machine Int.
    large_multiplicity_basis=PIBasis(100,2;sectors=[(50,50)])
    large_metadata=only(sector_metadata(large_multiplicity_basis))
    @test large_metadata.multiplicity isa BigInt
    @test large_metadata.multiplicity==symmetric_group_dimension(Partition((50,50)))

    # The multiplicity itself exceeds Float32 here, but its square root is
    # representable and is the only scale required by coefficient blocks.
    scaled_basis=PIBasis(200,2;sectors=[(100,100)])
    scaled_partition=only(scaled_basis.sectors)
    zero_operator=PIOperator(scaled_basis;T=Float32)
    @test physical_block(zero_operator,scaled_partition)==zeros(ComplexF32,1,1)
    scaled_operator=operator_from_schur_blocks(
        scaled_basis,[scaled_partition=>ones(ComplexF32,1,1)])
    @test physical_block(scaled_operator,scaled_partition)≈ones(ComplexF32,1,1)

    # A final square-root scale outside Float32 still raises explicitly.
    unrepresentable_basis=PIBasis(400,2;sectors=[(200,200)])
    unrepresentable_partition=only(unrepresentable_basis.sectors)
    @test_throws ArgumentError physical_block(
        PIOperator(unrepresentable_basis;T=Float32),unrepresentable_partition)

    setprecision(192) do
        big_basis=PIBasis(3,2;sectors=[(2,1)])
        half_im=Complex{BigFloat}(0,big"0.5")
        big_block=Complex{BigFloat}[big"1.25" half_im;
                                    -half_im big"0.75"]
        big_operator=operator_from_schur_blocks(big_basis,[mixed_partition=>big_block])
        @test eltype(big_operator.data)===Complex{BigFloat}
        @test physical_block(big_operator,mixed_partition)≈big_block rtol=eps(BigFloat)*8
    end
end
