const _BlockEntropyPID=PermutationalInvariantDynamics
if !isdefined(_BlockEntropyPID,:HilbertBlockEntropyPlan)
    Base.include(_BlockEntropyPID,
        joinpath(@__DIR__,"..","src","block_entropy.jl"))
end

# Deliberately exponential, test-only oracle.  It is used below only for
# N=3, d=2, hence constructs an 8-by-8 density matrix.
function _block_entropy_dense_oracle(A)
    PID=_BlockEntropyPID
    rows=Matrix{Float64}[]
    labels=NamedTuple[]
    offset=1
    for sector in partitions(A.basis.N,A.basis.d)
        paths=PID._removal_paths(sector,A.basis.N)
        @assert length(paths)==symmetric_group_dimension(sector)
        for path in paths
            tensor=PID._path_isometry(path)
            block=reshape(tensor,size(tensor,1),A.basis.d^A.basis.N)
            range=offset:offset+size(block,1)-1
            push!(rows,block)
            push!(labels,(;sector,range))
            offset=last(range)+1
        end
    end
    transform=reduce(vcat,rows)
    T=promote_type(eltype(A.data),Float64)
    schur=zeros(T,size(transform))
    for item in labels
        schur[item.range,item.range].=Matrix(physical_block(A,item.sector))
    end
    transform'*schur*transform
end

function _block_entropy_random_state(basis,plan,rng;T=Float64)
    rho=PIState(basis;T)
    for (sector_index,sector) in pairs(basis.sectors)
        block=coefficient_block(rho,sector)
        fill!(block,zero(eltype(block)))
        for indices in plan.blocks[sector_index]
            dimension=length(indices)
            random=randn(rng,Complex{T},dimension,dimension)
            positive=random*random'+T(0.2)*I
            index_vector=collect(indices)
            block[index_vector,index_vector].=positive
        end
    end
    _BlockEntropyPID.normalize!(rho)
end

@testset "certified Hilbert-block entropy" begin
    PID=_BlockEntropyPID
    @test Workflow.HilbertBlockEntropyPlan===PID.HilbertBlockEntropyPlan
    @test Workflow.block_entropy_diagnostics===PID.block_entropy_diagnostics
    @test Workflow.block_von_neumann_entropy===
        PID.block_von_neumann_entropy
    rng=MersenneTwister(0xb10ce7)
    basis=PIBasis(3,2)
    parity=Diagonal(ComplexF64[1,-1])
    plan=PID.HilbertBlockEntropyPlan(basis,parity)

    @test plan.block_count>length(basis.sectors)
    @test plan.largest_block<plan.largest_sector
    @test plan.split_cubic_work<plan.unsplit_cubic_work
    @test occursin("HilbertBlockEntropyPlan",sprint(show,plan))
    @test plan.metadata.kind===:diagonal_local_unitary
    @test !plan.metadata.model_symmetry_certified
    @test all(all(>(0),membership) for membership in plan.membership)

    rho=_block_entropy_random_state(basis,plan,rng)
    workspace=PID.HilbertBlockEntropyWorkspace(plan,Float64)
    @test occursin("HilbertBlockEntropyWorkspace",sprint(show,workspace))
    result=PID.block_von_neumann_entropy(
        rho,plan;workspace,return_info=true)
    @test result.value≈von_neumann_entropy(rho) atol=8e-12
    @test result.diagnostics.valid
    @test result.diagnostics.reason===:exact
    @test result.diagnostics.block_reason===:exact
    @test result.diagnostics.exactly_block_diagonal
    @test result.diagnostics.estimated_cubic_fraction<1
    @test PID.block_entropy_diagnostics(rho,plan;workspace).valid
    @test_throws ArgumentError PID.HilbertBlockEntropyWorkspace(
        plan,Float64;memory_budget=1)
    @test_throws ArgumentError PID.block_von_neumann_entropy(
        rho,plan;memory_budget=1)
    @test_throws ArgumentError PID.block_von_neumann_entropy(
        rho,plan;workspace,memory_budget=1)

    dense=_block_entropy_dense_oracle(rho)
    dense_values=eigvals(Hermitian((dense+dense')/2))
    dense_entropy=-sum(value*log2(value) for value in dense_values if value>0)
    @test result.value≈dense_entropy atol=2e-11

    # The explicit route is order independent and yields the same immutable
    # partition.  It is the escape hatch for externally certified symmetries.
    entries=NamedTuple[]
    for (sector_index,sector) in pairs(basis.sectors)
        for (group_index,indices) in pairs(plan.blocks[sector_index])
            push!(entries,(sector=sector,indices=reverse(collect(indices)),
                           label=(sector_index,group_index)))
        end
    end
    reverse!(entries)
    explicit=PID.HilbertBlockEntropyPlan(basis,entries;label=:external)
    @test explicit.blocks==plan.blocks
    @test explicit.label===:external
    @test PID.block_von_neumann_entropy(rho,explicit)≈result.value atol=8e-12
    @test_throws ArgumentError PID.block_von_neumann_entropy(
        rho,explicit;workspace)

    missing=copy(entries)
    pop!(missing)
    @test_throws ArgumentError PID.HilbertBlockEntropyPlan(basis,missing)
    overlap=copy(entries)
    push!(overlap,first(entries))
    @test_throws ArgumentError PID.HilbertBlockEntropyPlan(basis,overlap)
    @test_throws ArgumentError PID.HilbertBlockEntropyPlan(
        basis,[(4,0)=>[1]])
    @test_throws ArgumentError PID.HilbertBlockEntropyPlan(
        basis,ComplexF64[0 1;1 0])
    @test_throws ArgumentError PID.HilbertBlockEntropyPlan(
        basis,Diagonal(ComplexF64[1,2]))
    @test_throws ArgumentError PID.HilbertBlockEntropyPlan(
        PIBasis(10_000,1),Diagonal(ComplexF64[0.9]);atol=0,rtol=0.2)

    root=cis(2pi/3)
    root_plan=PID.HilbertBlockEntropyPlan(
        PIBasis(6,2;sectors=[(6,0)]),Diagonal(ComplexF64[1,root]))
    @test sort(length.(only(root_plan.blocks)))==[2,2,3]
    chain=Diagonal(ComplexF64[1,cis(0.075)])
    @test_throws ArgumentError PID.HilbertBlockEntropyPlan(
        PIBasis(2,2;sectors=[(2,0)]),chain;atol=0.08,rtol=0)

    other_basis=PIBasis(3,2)
    other_rho=_block_entropy_random_state(
        other_basis,PID.HilbertBlockEntropyPlan(other_basis,parity),rng)
    @test_throws ArgumentError PID.block_von_neumann_entropy(other_rho,plan)

    # A small cross-charge coherence remains a valid density matrix, but the
    # strict default refuses to reinterpret it as block diagonal.  Explicit
    # structural tolerance reports that the returned value belongs to the
    # block-diagonal projection.
    leaked=copy(rho)
    sector_index=findfirst(groups->length(groups)>1,plan.blocks)
    sector=basis.sectors[sector_index]
    left=first(plan.blocks[sector_index][1])
    right=first(plan.blocks[sector_index][2])
    epsilon=1e-10
    leaked_block=coefficient_block(leaked,sector)
    leaked_block[left,right]+=epsilon
    leaked_block[right,left]+=epsilon
    @test isphysical(leaked)
    strict=PID.block_entropy_diagnostics(leaked,plan)
    @test !strict.valid
    @test strict.reason===:offblock_leakage
    @test_throws ArgumentError PID.block_von_neumann_entropy(leaked,plan)
    approximate=PID.block_von_neumann_entropy(
        leaked,plan;block_atol=2epsilon,return_info=true)
    @test approximate.diagnostics.valid
    @test approximate.diagnostics.block_reason===:within_tolerance
    @test !approximate.diagnostics.exactly_block_diagonal
    projected=copy(leaked)
    projected_block=coefficient_block(projected,sector)
    projected_block[left,right]=projected_block[right,left]=0
    @test approximate.value≈von_neumann_entropy(projected) atol=8e-12

    # Scalar precision follows the state.  The split analysis does not narrow
    # a Float32 density to Float64.
    basis32=PIBasis(3,2)
    default_plan32=PID.HilbertBlockEntropyPlan(
        basis32,Diagonal(ComplexF32[1,-1]))
    @test default_plan32.block_count>length(basis32.sectors)
    plan32=PID.HilbertBlockEntropyPlan(
        basis32,Diagonal(ComplexF32[1,-1]);atol=1f-6,rtol=1f-5)
    rho32=_block_entropy_random_state(basis32,plan32,rng;T=Float32)
    workspace32=PID.HilbertBlockEntropyWorkspace(plan32,Float32)
    value32=PID.block_von_neumann_entropy(
        rho32,plan32;workspace=workspace32)
    @test value32 isa Float32
    @test value32≈von_neumann_entropy(rho32) atol=8f-5
    @test_throws ArgumentError PID.block_von_neumann_entropy(
        rho32,plan32;base=2.0)

    parity160=setprecision(BigFloat,160) do
        Diagonal(Complex{BigFloat}[one(BigFloat),-one(BigFloat)])
    end
    setprecision(BigFloat,256) do
        plan160=PID.HilbertBlockEntropyPlan(basis32,parity160)
        @test all(value->precision(real(value))==160&&
                         precision(imag(value))==160,
                  plan160.metadata.diagonal)
    end
    mixed_parity=Diagonal(Complex{BigFloat}[
        setprecision(BigFloat,128) do
            complex(one(BigFloat),zero(BigFloat))
        end,
        setprecision(BigFloat,256) do
            complex(-one(BigFloat),zero(BigFloat))
        end,
    ])
    @test_throws ArgumentError PID.HilbertBlockEntropyPlan(
        basis32,mixed_parity)
end

@testset "certified strong-symmetry integration" begin
    PID=_BlockEntropyPID
    basis=PIBasis(2,2)
    parity=Diagonal(ComplexF64[1,-1])
    model=PIModel(basis,(LocalJump(Matrix(parity)),))
    report=strong_symmetry_report(
        model;candidates=(parity_z=parity,))
    reduction=strong_symmetry_reduction(
        model;report,candidate=:parity_z)
    plan=PID.HilbertBlockEntropyPlan(reduction)
    @test plan.basis===basis
    @test plan.metadata.model_symmetry_certified
    @test plan.metadata.certificate.candidate===:parity_z

    direct=PID.HilbertBlockEntropyPlan(basis,parity)
    @test plan.blocks==direct.blocks
    rho=_block_entropy_random_state(basis,plan,MersenneTwister(0x51a7))
    @test PID.block_von_neumann_entropy(rho,plan)≈
          von_neumann_entropy(rho) atol=8e-12
end
