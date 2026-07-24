# Exact tracing of one internal factor from every identical local supersite.
#
# The PI operator space is the symmetric tensor power of the one-site
# Hilbert--Schmidt operator space.  Normalized symmetric occupations of local
# matrix units therefore form an orthonormal basis with exactly
# `length(PIBasis(N,d))` elements.  A polarized version of the `iid_state`
# Schur recurrence constructs those operators directly in equation-(7)
# coordinates, without a `d^N` tensor product.

"""
    LocalFactorTracePlan(source_basis, local_dimensions;
                         traced_factor=2, T=Float64,
                         memory_budget=512*1024^2,
                         atol=nothing, rtol=nothing)
    LocalFactorTracePlan(rho, local_dimensions; kwargs...)

Prepare the exact PI map which traces the same internal tensor factor from
every identical supersite. `local_dimensions=(d1,d2)` follows Julia's
`kron(factor1,factor2)` ordering and must satisfy
`d1*d2 == source_basis.d`. `traced_factor` is `1` or `2`.

The output basis is always the complete `PIBasis(source_basis.N,dkeep)`.
Local tracing can populate several output Young sectors even when the source
basis is restricted to one sector.

The plan stores two read-only exact-support sparse rectangular transforms.
Reusing it reduces each state with two sparse matrix-vector products. Setup
constructs one occupation column at a time and discards only values satisfying
`iszero`; it never applies a numerical dropping tolerance. Setup and
application never construct a `source_basis.d^N` state, a full-system density
matrix, or a list of local operator strings.

`memory_budget` guards both retained storage and conservative setup
temporaries; pass `Inf` only as an explicit opt-out. The `rho` constructor
infers `T` from the state's real floating-point component type.
"""
struct LocalFactorTracePlan{R<:AbstractFloat,B<:PIBasis,O<:PIBasis,L,Q,E}
    basis::B
    output_basis::O
    local_dimensions::NTuple{2,Int}
    traced_factor::Int
    kept_factor::Int
    lifted_columns::L
    output_columns::Q
    estimates::E
end

function show(io::IO,plan::LocalFactorTracePlan)
    print(io,"LocalFactorTracePlan(N=$(plan.basis.N), ",
          "local_dimensions=$(plan.local_dimensions), ",
          "traced_factor=$(plan.traced_factor), ",
          "input_dimension=$(length(plan.basis)), ",
          "output_dimension=$(length(plan.output_basis)), ",
          "scalar_type=$(_real_float_type(eltype(plan.lifted_columns))))")
end

mutable struct _LocalFactorColumnBuilder{R<:AbstractFloat,D,L,Q,B,A}
    basis::B
    letters::A
    pattern_cache::Dict{Partition{D},Vector{GTPattern{D,L}}}
    transition_cache::Dict{
        Tuple{Partition{D},Partition{D}},Vector{Matrix{R}}}
    edge_cache::Dict{
        Tuple{Partition{D},Partition{D}},Vector{Matrix{Complex{R}}}}
    block_cache::Dict{
        Tuple{Partition{D},NTuple{Q,Int}},Matrix{Complex{R}}}
end

function _LocalFactorColumnBuilder(
        basis::B,letters::NTuple{Q,Vector{Tuple{Int,Int}}},
        ::Type{R}) where {D,L,B<:PIBasis{D,L},Q,R<:AbstractFloat}
    _LocalFactorColumnBuilder{R,D,L,Q,B,typeof(letters)}(
        basis,letters,
        Dict{Partition{D},Vector{GTPattern{D,L}}}(),
        Dict{Tuple{Partition{D},Partition{D}},Vector{Matrix{R}}}(),
        Dict{Tuple{Partition{D},Partition{D}},
             Vector{Matrix{Complex{R}}}}(),
        Dict{Tuple{Partition{D},NTuple{Q,Int}},
             Matrix{Complex{R}}}())
end

function _local_factor_edge_maps!(
        builder::_LocalFactorColumnBuilder{R,D},lower::Partition{D},
        upper::Partition{D}) where {R,D}
    key=(lower,upper)
    get!(builder.edge_cache,key) do
        maps=_pbody_edge_transitions!(
            builder.transition_cache,builder.pattern_cache,lower,upper,R)
        [Matrix{Complex{R}}(map) for map in maps]
    end
end

function _local_factor_polarized_block(
        builder::_LocalFactorColumnBuilder{R,D,L,Q},
        partition::Partition{D},counts::NTuple{Q,Int}) where {R,D,L,Q}
    key=(partition,counts)
    cached=get(builder.block_cache,key,nothing)
    cached===nothing||return cached
    sum(counts)==weight(partition)||error(
        "internal local-factor occupation/partition weight mismatch")
    if iszero(weight(partition))
        all(iszero,counts)||error(
            "internal nonzero local-factor occupation at the empty partition")
        block=reshape(Complex{R}[one(R)],1,1)
        builder.block_cache[key]=block
        return block
    end

    parent=remove_corner(partition,first(removable_corners(partition)))
    edges=_local_factor_edge_maps!(builder,parent,partition)
    endpoint_dimension=size(first(edges),1)
    parent_dimension=size(first(edges),2)
    block=zeros(Complex{R},endpoint_dimension,endpoint_dimension)
    temporary=zeros(Complex{R},endpoint_dimension,parent_dimension)
    one_complex=one(Complex{R})
    for letter in 1:Q
        iszero(counts[letter])&&continue
        parent_counts=ntuple(
            index->counts[index]-(index==letter),Val(Q))
        parent_block=_local_factor_polarized_block(
            builder,parent,parent_counts)
        for (row_label,column_label) in builder.letters[letter]
            mul!(temporary,edges[row_label],parent_block)
            mul!(block,temporary,adjoint(edges[column_label]),
                 one_complex,one_complex)
        end
    end
    builder.block_cache[key]=block
    block
end

function _local_factor_compositions(total::Int,::Val{Q}) where Q
    total>=0||throw(ArgumentError("occupation weight must be nonnegative"))
    Q>=1||throw(ArgumentError(
        "the local operator alphabet must be nonempty"))
    output=NTuple{Q,Int}[]
    current=zeros(Int,Q)
    function recurse(position::Int,left::Int)
        if position==Q
            current[position]=left
            push!(output,Tuple(current))
            return
        end
        for value in 0:left
            current[position]=value
            recurse(position+1,left-value)
        end
    end
    recurse(1,total)
    output
end

function _local_factor_column!(
        destination::AbstractVector{Complex{R}},
        builder::_LocalFactorColumnBuilder{R,D,L,Q},
        counts::NTuple{Q,Int}) where {R,D,L,Q}
    length(destination)==length(builder.basis)||throw(DimensionMismatch(
        "local-factor occupation column has the wrong length"))
    sum(counts)==builder.basis.N||error(
        "internal local-factor final occupation weight mismatch")
    empty!(builder.block_cache)
    multinomial=exact_multinomial(counts)
    for (sector_index,partition) in pairs(builder.basis.sectors)
        block=_local_factor_polarized_block(builder,partition,counts)
        scale=_prepare_exact_scale(
            R,symmetric_group_dimension(partition),multinomial,Val(true);
            context="normalized local-factor occupation in sector $partition")
        offset=builder.basis.offsets[sector_index]
        @inbounds for index in eachindex(block)
            destination[offset+index-1]=_apply_prepared_exact_scale(
                block[index],scale;
                context="normalized local-factor occupation in sector $partition")
        end
    end
    destination
end

function _local_factor_matrix_unit_letters(dimension::Int)
    count=Base.checked_mul(dimension,dimension)
    ntuple(Val(count)) do letter
        row=mod1(letter,dimension)
        column=(letter-1)÷dimension+1
        Tuple{Int,Int}[(row,column)]
    end
end

function _local_factor_lifted_letters(
        local_dimensions::NTuple{2,Int},traced_factor::Int)
    d1,d2=local_dimensions
    kept_dimension=traced_factor==2 ? d1 : d2
    traced_dimension=traced_factor==2 ? d2 : d1
    count=Base.checked_mul(kept_dimension,kept_dimension)
    ntuple(Val(count)) do letter
        kept_row=mod1(letter,kept_dimension)
        kept_column=(letter-1)÷kept_dimension+1
        entries=Vector{Tuple{Int,Int}}(undef,traced_dimension)
        for traced_label in 1:traced_dimension
            if traced_factor==2
                # kron(E_kept,I_traced): the second factor is fastest.
                row=traced_label+d2*(kept_row-1)
                column=traced_label+d2*(kept_column-1)
            else
                # kron(I_traced,E_kept): the second factor is fastest.
                row=kept_row+d2*(traced_label-1)
                column=kept_column+d2*(traced_label-1)
            end
            entries[traced_label]=(row,column)
        end
        entries
    end
end

function _local_factor_sparse_csc_bytes(
        columns::Integer,nonzeros::Integer,::Type{T};
        bigfloat_precision::Integer=precision(BigFloat)) where T
    count=BigInt(nonzeros)
    _performance_entries_bytes(
        count,T;bigfloat_precision)+
        (count+BigInt(columns)+1)*sizeof(Int)
end

function _local_factor_chain_partitions(basis::PIBasis{D}) where D
    chains=Vector{Vector{Partition{D}}}(undef,length(basis.sectors))
    edge_set=Set{Tuple{Partition{D},Partition{D}}}()
    partition_set=Set{Partition{D}}()
    for (sector_index,sector) in pairs(basis.sectors)
        chain=Partition{D}[sector]
        push!(partition_set,sector)
        partition=sector
        while weight(partition)>0
            parent=remove_corner(
                partition,first(removable_corners(partition)))
            push!(edge_set,(parent,partition))
            push!(partition_set,parent)
            push!(chain,parent)
            partition=parent
        end
        chains[sector_index]=chain
    end
    chains,edge_set,partition_set
end

function _local_factor_suboccupation_bound(
        N::Int,weight_value::Int,alphabet_size::Int)
    left=exact_binomial(
        BigInt(weight_value)+alphabet_size-1,weight_value)
    removed=N-weight_value
    right=exact_binomial(BigInt(removed)+alphabet_size-1,removed)
    min(left,right)
end

# Conservative peak live storage for one polarized-column recurrence. The
# recurrence follows one deterministic corner-removal chain from each retained
# source sector. For a partition of weight w, a fixed top occupation can reach
# at most the smaller of all weight-w suboccupations and all distributions of
# the N-w removed letters. This bounds every cached dense Schur block without
# charging the dense rectangular transform that is never constructed.
function _local_factor_builder_peak_bytes(
        basis::PIBasis{D,L},alphabet_size::Int,::Type{R};
        bigfloat_precision::Integer=precision(BigFloat)) where
        {D,L,R<:AbstractFloat}
    chains,edges,partitions=_local_factor_chain_partitions(basis)
    dimensions=Dict(
        partition=>BigInt(unitary_group_dimension(partition))
        for partition in partitions)
    complex_bytes=_scalar_retained_bytes(
        Complex{R};bigfloat_precision)
    real_bytes=_scalar_retained_bytes(R;bigfloat_precision)
    int_bytes=BigInt(sizeof(Int))

    block_entries=big(0)
    block_count=big(0)
    for chain in chains,partition in chain
        partition_weight=weight(partition)
        count=_local_factor_suboccupation_bound(
            basis.N,partition_weight,alphabet_size)
        dimension=dimensions[partition]
        block_entries+=count*dimension^2
        block_count+=count
    end

    transition_entries=big(0)
    pattern_entries=big(0)
    for partition in partitions
        pattern_entries+=dimensions[partition]
    end
    for (lower,upper) in edges
        transition_entries+=
            BigInt(D)*dimensions[lower]*dimensions[upper]
    end

    # transition_cache keeps real edge matrices and edge_cache keeps their
    # Complex copies. At most one dense temporary accompanies every active
    # recursion depth; charging all transition cells is a strict upper bound.
    numeric=(block_entries+transition_entries)*complex_bytes+
            transition_entries*real_bytes
    pattern_payload=pattern_entries*BigInt(L)*int_bytes
    # Dictionary buckets, vectors, matrix headers and hash-table slack are
    # allocator dependent. A deliberately padded structural envelope follows
    # the same policy as the prepared one-box cache estimate.
    containers=512*(block_count+
                    2BigInt(D)*length(edges)+
                    length(partitions)+8)
    2*(numeric+pattern_payload+containers)+65536
end

function _local_factor_sparse_transform(
        basis::PIBasis,letters,counts,::Type{R};
        memory_budget,retained_before_bytes::Integer=0,
        setup_fixed_bytes::Integer=0,
        context::AbstractString) where R<:AbstractFloat
    row_count=length(basis)
    column_count=length(counts)
    T=Complex{R}
    precision_bits=R===BigFloat ? precision(BigFloat) : precision(R)
    builder_peak=_local_factor_builder_peak_bytes(
        basis,length(letters),R;bigfloat_precision=precision_bits)
    column_bytes=_performance_entries_bytes(
        row_count,T;bigfloat_precision=precision_bits)
    empty_csc_bytes=_local_factor_sparse_csc_bytes(
        column_count,0,T;bigfloat_precision=precision_bits)
    outer_chunk_bytes=
        BigInt(2column_count)*sizeof(Ptr{Cvoid})+
        2*(BigInt(column_count)+1)*256
    minimum_peak=BigInt(retained_before_bytes)+
                 BigInt(setup_fixed_bytes)+builder_peak+
                 column_bytes+empty_csc_bytes+outer_chunk_bytes
    _require_performance_budget(
        "$context setup",minimum_peak,memory_budget;guidance=
        "Reduce N/local dimensions, use a restricted source PIBasis, or increase memory_budget.")

    builder=_LocalFactorColumnBuilder(basis,letters,R)
    column=zeros(T,row_count)
    column_pointers=Vector{Int}(undef,column_count+1)
    # Retain exact-sized per-column chunks during construction. Growing
    # `Vector`s do not expose their allocated capacity, so a `push!`/`sizehint!`
    # route cannot preflight a reallocation peak reliably. The final CSC
    # arrays are allocated once, after the exact total support is known.
    column_rows=Vector{Vector{Int}}(undef,column_count)
    column_values=Vector{Vector{T}}(undef,column_count)
    column_pointers[1]=1
    maximum_peak=minimum_peak
    accumulated_nonzeros=0
    chunk_payload_bytes=big(0)
    for (column_index,occupation) in pairs(counts)
        fill!(column,zero(T))
        _local_factor_column!(column,builder,occupation)
        column_nonzeros=count(!iszero,column)
        total_nonzeros=Base.checked_add(
            accumulated_nonzeros,column_nonzeros)
        new_chunk_payload=
            _performance_entries_bytes(
                column_nonzeros,T;
                bigfloat_precision=precision_bits)+
            BigInt(column_nonzeros)*sizeof(Int)
        candidate_chunk_payload=chunk_payload_bytes+new_chunk_payload
        candidate_peak=BigInt(retained_before_bytes)+
                       BigInt(setup_fixed_bytes)+builder_peak+
                       column_bytes+empty_csc_bytes+
                       outer_chunk_bytes+candidate_chunk_payload
        maximum_peak=max(maximum_peak,candidate_peak)
        _require_performance_budget(
            "$context setup",candidate_peak,memory_budget;guidance=
            "Reduce N/local dimensions, use a restricted source PIBasis, or increase memory_budget.")
        rows=Vector{Int}(undef,column_nonzeros)
        column_data=Vector{T}(undef,column_nonzeros)
        destination_index=0
        @inbounds for row in eachindex(column)
            value=column[row]
            iszero(value)&&continue
            destination_index+=1
            rows[destination_index]=row
            column_data[destination_index]=value
        end
        destination_index==column_nonzeros||error(
            "internal local-factor sparse support count mismatch")
        column_rows[column_index]=rows
        column_values[column_index]=column_data
        accumulated_nonzeros=total_nonzeros
        chunk_payload_bytes=candidate_chunk_payload
        column_pointers[column_index+1]=total_nonzeros+1
    end
    final_csc_bytes=_local_factor_sparse_csc_bytes(
        column_count,accumulated_nonzeros,T;
        bigfloat_precision=precision_bits)
    chunk_actual_bytes=
        BigInt(Base.summarysize(column_rows))+
        BigInt(Base.summarysize(column_values))
    final_allocation_peak=BigInt(retained_before_bytes)+
                          BigInt(setup_fixed_bytes)+builder_peak+
                          column_bytes+empty_csc_bytes+
                          chunk_actual_bytes+final_csc_bytes
    maximum_peak=max(maximum_peak,final_allocation_peak)
    _require_performance_budget(
        "$context setup",final_allocation_peak,memory_budget;guidance=
        "Reduce N/local dimensions, use a restricted source PIBasis, or increase memory_budget.")
    row_indices=Vector{Int}(undef,accumulated_nonzeros)
    values=Vector{T}(undef,accumulated_nonzeros)
    for column_index in 1:column_count
        destination=column_pointers[column_index]
        rows=column_rows[column_index]
        column_data=column_values[column_index]
        copyto!(row_indices,destination,rows,1,length(rows))
        copyto!(values,destination,column_data,1,length(column_data))
    end
    matrix=SparseMatrixCSC(
        row_count,column_count,column_pointers,row_indices,values)
    retained_bytes=max(
        _local_factor_sparse_csc_bytes(
            column_count,nnz(matrix),T;
            bigfloat_precision=precision_bits),
        BigInt(Base.summarysize(matrix)))
    actual_peak=BigInt(retained_before_bytes)+
                BigInt(setup_fixed_bytes)+
                BigInt(Base.summarysize(builder))+
                BigInt(Base.summarysize(column))+
                chunk_actual_bytes+
                BigInt(Base.summarysize(matrix))
    maximum_peak=max(maximum_peak,actual_peak)
    _require_performance_budget(
        "$context setup",actual_peak,memory_budget;guidance=
        "Reduce N/local dimensions, use a restricted source PIBasis, or increase memory_budget.")
    matrix,(retained_bytes=BigInt(retained_bytes),
            nonzeros=BigInt(nnz(matrix)),
            setup_peak_bytes=maximum_peak,
            builder_peak_bytes=builder_peak)
end

# Compute `opnorm(Q'Q-I,Inf)` without materializing `Q'Q`. Sparse matrix
# multiplication can turn a sparse square transform into a dense Gram matrix,
# even when only its infinity norm is needed. The row adjacency below stores
# integer references into the original CSC values. Each Gram column is then
# accumulated into one stamped dense work vector and immediately reduced into
# the final row sums.
function _local_factor_streamed_gram_validation(
        matrix::SparseMatrixCSC{Complex{R},Int};
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,
        base_bytes::Integer=0,
        context::AbstractString="local-factor output transform") where
        R<:AbstractFloat
    row_count,column_count=size(matrix)
    row_count==column_count||throw(DimensionMismatch(
        "local-factor orthonormality validation requires a square transform"))
    all(isfinite,matrix.nzval)||throw(ErrorException(
        "local-factor output transform contains nonfinite coefficients"))
    nonzero_count=nnz(matrix)
    precision_bits=R===BigFloat ? precision(BigFloat) : precision(R)
    integer_entries=
        BigInt(row_count)+1+
        2BigInt(nonzero_count)+
        BigInt(row_count)+
        2BigInt(column_count)
    numeric_bytes=
        _performance_entries_bytes(
            column_count,Complex{R};
            bigfloat_precision=precision_bits)+
        _performance_entries_bytes(
            column_count,R;
            bigfloat_precision=precision_bits)
    estimated_workspace_bytes=
        integer_entries*sizeof(Int)+numeric_bytes+8*256
    estimated_peak=BigInt(base_bytes)+estimated_workspace_bytes
    _require_performance_budget(
        "$context streamed Gram validation",estimated_peak,memory_budget;
        guidance="Reduce N/local dimensions or increase memory_budget.")

    # `row_offsets[row]+1:row_offsets[row+1]` enumerates every nonzero
    # in that matrix row. Entries are appended in ascending source-column
    # order, preserving deterministic accumulation.
    row_offsets=zeros(Int,row_count+1)
    @inbounds for row in matrix.rowval
        row_offsets[row+1]=Base.checked_add(row_offsets[row+1],1)
    end
    @inbounds for row in 1:row_count
        row_offsets[row+1]=Base.checked_add(
            row_offsets[row],row_offsets[row+1])
    end
    row_columns=Vector{Int}(undef,nonzero_count)
    value_pointers=Vector{Int}(undef,nonzero_count)
    next_slot=Vector{Int}(undef,row_count)
    @inbounds for row in 1:row_count
        next_slot[row]=row_offsets[row]+1
    end
    @inbounds for column in 1:column_count
        for pointer in nzrange(matrix,column)
            row=matrix.rowval[pointer]
            slot=next_slot[row]
            row_columns[slot]=column
            value_pointers[slot]=pointer
            next_slot[row]=slot+1
        end
    end

    accumulator=zeros(Complex{R},column_count)
    stamps=zeros(Int,column_count)
    touched=Vector{Int}(undef,column_count)
    row_sums=zeros(R,row_count)
    @inbounds for column in 1:column_count
        for pointer in nzrange(matrix,column)
            row=matrix.rowval[pointer]
            row_sums[row]+=abs(matrix.nzval[pointer])
        end
    end
    transform_inf_norm=maximum(row_sums;init=zero(R))
    gram_scale=max(one(R),transform_inf_norm^2)
    isfinite(transform_inf_norm)&&isfinite(gram_scale)||throw(ErrorException(
        "local-factor output transform norm is nonfinite or overflowed"))
    fill!(row_sums,zero(R))

    pair_products=big(0)
    @inbounds for row in 1:row_count
        degree=row_offsets[row+1]-row_offsets[row]
        pair_products+=BigInt(degree)^2
    end
    @inbounds for column in 1:column_count
        touched_count=0
        for matrix_pointer in nzrange(matrix,column)
            row=matrix.rowval[matrix_pointer]
            right_value=matrix.nzval[matrix_pointer]
            for row_slot in row_offsets[row]+1:row_offsets[row+1]
                gram_row=row_columns[row_slot]
                contribution=
                    conj(matrix.nzval[value_pointers[row_slot]])*
                    right_value
                if stamps[gram_row]!=column
                    stamps[gram_row]=column
                    accumulator[gram_row]=contribution
                    touched_count+=1
                    touched[touched_count]=gram_row
                else
                    accumulator[gram_row]+=contribution
                end
            end
        end
        diagonal_seen=false
        for touched_index in 1:touched_count
            gram_row=touched[touched_index]
            value=accumulator[gram_row]
            if gram_row==column
                diagonal_seen=true
                value-=one(Complex{R})
            end
            row_sums[gram_row]+=abs(value)
        end
        diagonal_seen||(row_sums[column]+=one(R))
    end
    residual=maximum(row_sums;init=zero(R))
    isfinite(residual)||throw(ErrorException(
        "local-factor output transform Gram residual is nonfinite or overflowed"))
    actual_workspace_bytes=BigInt(Base.summarysize((
        row_offsets,row_columns,value_pointers,next_slot,accumulator,
        stamps,touched,row_sums)))
    actual_peak=BigInt(base_bytes)+actual_workspace_bytes
    _require_performance_budget(
        "$context streamed Gram validation",actual_peak,memory_budget;
        guidance="Reduce N/local dimensions or increase memory_budget.")
    residual,gram_scale,(
        workspace_bytes=max(estimated_workspace_bytes,
                            actual_workspace_bytes),
        workspace_entries=integer_entries+
                          2BigInt(column_count),
        pair_products,
        peak_bytes=max(estimated_peak,actual_peak))
end

function _local_factor_dimensions(
        basis::PIBasis,local_dimensions,traced_factor::Integer)
    local_dimensions isa Tuple&&length(local_dimensions)==2&&
        all(value->value isa Integer,local_dimensions)||throw(ArgumentError(
        "local_dimensions must be a tuple of two positive integers"))
    d1=Int(local_dimensions[1]);d2=Int(local_dimensions[2])
    d1>=1&&d2>=1||throw(ArgumentError(
        "local_dimensions must contain positive dimensions"))
    product=try
        Base.checked_mul(d1,d2)
    catch error
        error isa OverflowError||rethrow()
        throw(ArgumentError("the product of local_dimensions exceeds Int"))
    end
    product==basis.d||throw(DimensionMismatch(
        "prod(local_dimensions)=$product must equal source_basis.d=$(basis.d)"))
    traced=Int(traced_factor)
    traced in (1,2)||throw(ArgumentError(
        "traced_factor must be 1 or 2"))
    kept=3-traced
    kept_dimension=kept==1 ? d1 : d2
    (d1,d2),traced,kept,kept_dimension
end

function _local_factor_setup_tolerances(
        ::Type{R},atol,rtol,N::Int,dimension::Int) where R<:AbstractFloat
    default_rtol=max(R(2e-10),
        R(256)*eps(R)*R(max(1,N,dimension)))
    a=atol===nothing ? zero(R) : R(atol)
    r=rtol===nothing ? default_rtol : R(rtol)
    isfinite(a)&&a>=zero(R)||throw(ArgumentError(
        "local-factor trace atol must be finite and nonnegative"))
    isfinite(r)&&r>=zero(R)||throw(ArgumentError(
        "local-factor trace rtol must be finite and nonnegative"))
    a,r
end

function LocalFactorTracePlan(
        basis::B,local_dimensions;traced_factor::Integer=2,
        T::Type{R}=Float64,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,
        atol=nothing,rtol=nothing) where
        {D,L,B<:PIBasis{D,L},R<:AbstractFloat}
    isconcretetype(R)||throw(ArgumentError(
        "T must be a concrete AbstractFloat type"))
    dimensions,traced,kept,kept_dimension=
        _local_factor_dimensions(basis,local_dimensions,traced_factor)
    input_size=length(basis)
    CT=Complex{R}
    precision_bits=R===BigFloat ? precision(BigFloat) : precision(R)
    rounding_mode=R===BigFloat ? rounding(BigFloat) : nothing
    alphabet_size_big=BigInt(kept_dimension)^2
    output_size_big=exact_binomial(
        BigInt(basis.N)+alphabet_size_big-1,basis.N)

    output_size_big<=typemax(Int)||throw(ArgumentError(
        "the kept-factor PI dimension exceeds Int indexing capacity"))
    alphabet_size_big<=typemax(Int)||throw(ArgumentError(
        "the kept-factor operator alphabet exceeds Int indexing capacity"))
    output_size=Int(output_size_big)
    alphabet_size=Int(alphabet_size_big)
    counts_bytes=output_size_big*alphabet_size_big*sizeof(Int)+256
    kept_pattern_entries=
        BigInt(kept_dimension)*(BigInt(kept_dimension)+1)÷2
    output_basis_setup_bytes=
        output_size_big*(kept_pattern_entries*sizeof(Int)+512)+65536
    setup_fixed_bytes=counts_bytes+output_basis_setup_bytes
    minimum_sparse_bytes=
        _local_factor_sparse_csc_bytes(
            output_size,0,CT;bigfloat_precision=precision_bits)+
        _local_factor_sparse_csc_bytes(
            output_size,0,CT;bigfloat_precision=precision_bits)+
        _performance_entries_bytes(
            max(input_size,output_size),CT;
            bigfloat_precision=precision_bits)
    _require_performance_budget(
        "local-factor trace plan setup",
        minimum_sparse_bytes+setup_fixed_bytes,memory_budget;
        guidance="Reduce N/local dimensions or increase memory_budget.")
    output_basis=PIBasis(basis.N,kept_dimension)
    length(output_basis)==output_size||error(
        "internal local-factor PI dimension mismatch")
    counts=_local_factor_compositions(basis.N,Val(alphabet_size))
    length(counts)==output_size||error(
        "internal local-factor occupation/PI dimension mismatch")

    output_columns,output_statistics=_local_factor_sparse_transform(
        output_basis,_local_factor_matrix_unit_letters(kept_dimension),
        counts,R;memory_budget,setup_fixed_bytes,
        context="local-factor output transform")
    lifted_columns,lifted_statistics=_local_factor_sparse_transform(
        basis,_local_factor_lifted_letters(dimensions,traced),counts,R;
        memory_budget,
        retained_before_bytes=output_statistics.retained_bytes,
        setup_fixed_bytes,
        context="local-factor lifted transform")
    all(isfinite,lifted_columns.nzval)||throw(ErrorException(
        "local-factor lifted transform contains nonfinite coefficients"))
    retained_entries=output_statistics.nonzeros+
                     lifted_statistics.nonzeros
    retained_bytes=output_statistics.retained_bytes+
                   lifted_statistics.retained_bytes
    dense_entries=BigInt(input_size)*output_size_big+
                  output_size_big^2

    setup_atol,setup_rtol=_local_factor_setup_tolerances(
        R,atol,rtol,basis.N,basis.d)
    validation_base_bytes=setup_fixed_bytes+retained_bytes
    gram_residual,gram_scale,gram_statistics=
        _local_factor_streamed_gram_validation(
            output_columns;memory_budget,
            base_bytes=validation_base_bytes,
            context="local-factor trace plan")
    validation_vector_bytes=_performance_entries_bytes(
        3(BigInt(input_size)+output_size_big),CT;
        bigfloat_precision=precision_bits)
    validation_peak=validation_base_bytes+validation_vector_bytes
    _require_performance_budget(
        "local-factor trace plan validation",validation_peak,memory_budget;
        guidance="Reduce N/local dimensions or increase memory_budget.")
    gram_residual<=setup_atol+setup_rtol*gram_scale||throw(ErrorException(
        "local-factor output occupation transform lost orthonormality: " *
        "residual=$gram_residual, tolerance=$(setup_atol+setup_rtol*gram_scale)"))

    output_trace=_trace_vector(output_basis,CT)
    input_trace=_trace_vector(basis,CT)
    occupation_trace=adjoint(output_columns)*output_trace
    pulled_trace=lifted_columns*occupation_trace
    trace_residual=norm(pulled_trace-input_trace,Inf)
    trace_scale=max(one(R),norm(input_trace,Inf),
                    norm(pulled_trace,Inf))
    isfinite(trace_residual)&&isfinite(trace_scale)||throw(ErrorException(
        "local-factor trace-preservation residual is nonfinite or overflowed"))
    trace_residual<=setup_atol+setup_rtol*trace_scale||throw(ErrorException(
        "local-factor trace map is not trace preserving within setup " *
        "tolerance: residual=$trace_residual, " *
        "tolerance=$(setup_atol+setup_rtol*trace_scale)"))

    actual_validation_peak=
        BigInt(Base.summarysize(counts))+
        BigInt(Base.summarysize(output_basis))+
        retained_bytes+
        BigInt(Base.summarysize(output_trace))+
        BigInt(Base.summarysize(input_trace))+
        BigInt(Base.summarysize(occupation_trace))+
        BigInt(Base.summarysize(pulled_trace))
    peak_bytes=max(
        output_statistics.setup_peak_bytes,
        lifted_statistics.setup_peak_bytes,
        gram_statistics.peak_bytes,
        validation_peak,actual_validation_peak)
    _require_performance_budget(
        "local-factor trace plan validation",actual_validation_peak,
        memory_budget;guidance=
        "Reduce N/local dimensions or increase memory_budget.")
    peak_entries=retained_entries+
                 3(BigInt(input_size)+output_size_big)+
                 gram_statistics.workspace_entries
    estimates=(input_dimension=input_size,output_dimension=output_size,
        dense_entries,retained_entries,retained_bytes,
        lifted_nonzeros=lifted_statistics.nonzeros,
        output_nonzeros=output_statistics.nonzeros,
        peak_entries,peak_bytes,
        output_setup_peak_bytes=output_statistics.setup_peak_bytes,
        lifted_setup_peak_bytes=lifted_statistics.setup_peak_bytes,
        counts_setup_bytes=counts_bytes,
        output_basis_setup_bytes,
        gram_validation=:streamed_sparse_columns,
        gram_workspace_bytes=gram_statistics.workspace_bytes,
        gram_pair_products=gram_statistics.pair_products,
        memory_budget=_memory_budget_bytes(memory_budget),
        scalar_type=CT,precision_bits,rounding_mode,
        storage=:exact_support_sparse_csc,
        gram_residual,trace_residual)
    LocalFactorTracePlan{R,B,typeof(output_basis),typeof(lifted_columns),
        typeof(output_columns),typeof(estimates)}(
        basis,output_basis,dimensions,traced,kept,lifted_columns,
        output_columns,estimates)
end

function LocalFactorTracePlan(
        rho::PIState,local_dimensions;
        T::Type{R}=_real_float_type(eltype(rho.data)),
        kwargs...) where R<:AbstractFloat
    source_type=_real_float_type(eltype(rho.data))
    R===source_type||throw(ArgumentError(
        "T=$R does not match source-state scalar type $source_type; " *
        "convert the state explicitly before preparing its local-factor trace plan"))
    if R===BigFloat
        precision_bounds=_local_factor_precision_bounds(rho.data)
        precision_bounds[1]==precision_bounds[2]||throw(ArgumentError(
            "source state BigFloat storage has mixed precision range " *
            "$precision_bounds; rebuild it at one precision"))
        input_precision=precision_bounds[2]
        if precision(BigFloat)!=input_precision
            return setprecision(BigFloat,input_precision) do
                LocalFactorTracePlan(
                    rho.basis,local_dimensions;T,kwargs...)
            end
        end
    end
    LocalFactorTracePlan(rho.basis,local_dimensions;T,kwargs...)
end

"""
    LocalFactorTraceWorkspace(plan)

Allocate the one output-occupation vector needed by repeated applications of
a [`LocalFactorTracePlan`](@ref). A workspace belongs to one exact plan and
must be owned by one task at a time.
"""
mutable struct LocalFactorTraceWorkspace{T,P<:LocalFactorTracePlan}
    plan::P
    occupation_coordinates::Vector{T}
end

function LocalFactorTraceWorkspace(plan::LocalFactorTracePlan)
    T=eltype(plan.lifted_columns)
    if _real_float_type(T)===BigFloat
        precision_bits=plan.estimates.precision_bits
        rounding_mode=plan.estimates.rounding_mode
        if precision(BigFloat)!=precision_bits||
                rounding(BigFloat)!=rounding_mode
            return setrounding(BigFloat,rounding_mode) do
                setprecision(BigFloat,precision_bits) do
                    LocalFactorTraceWorkspace(plan)
                end
            end
        end
    end
    LocalFactorTraceWorkspace{T,typeof(plan)}(
        plan,zeros(T,length(plan.output_basis)))
end

function _local_factor_precision_bounds(values)
    minimum_precision=typemax(Int)
    maximum_precision=0
    for value in values
        value_precision=max(
            precision(real(value)),precision(imag(value)))
        minimum_precision=min(minimum_precision,value_precision)
        maximum_precision=max(maximum_precision,value_precision)
    end
    isempty(values) ? (precision(BigFloat),precision(BigFloat)) :
        (minimum_precision,maximum_precision)
end

function show(io::IO,workspace::LocalFactorTraceWorkspace)
    print(io,"LocalFactorTraceWorkspace(output_dimension=",
          length(workspace.occupation_coordinates),
          ", scalar_type=",eltype(workspace.occupation_coordinates),")")
end

function _check_local_factor_trace_resources(
        output::PIState,source::PIState,plan::LocalFactorTracePlan,
        workspace::LocalFactorTraceWorkspace)
    source.basis===plan.basis||throw(ArgumentError(
        "LocalFactorTracePlan was prepared for a different source PIBasis"))
    output.basis===plan.output_basis||throw(ArgumentError(
        "output state must use the LocalFactorTracePlan output_basis object"))
    workspace.plan===plan||throw(ArgumentError(
        "LocalFactorTraceWorkspace was prepared for a different plan"))
    output===source&&throw(ArgumentError(
        "local-factor trace output must not alias its source"))
    T=eltype(plan.lifted_columns)
    eltype(source.data)===T||throw(ArgumentError(
        "source state scalar type $(eltype(source.data)) does not match " *
        "LocalFactorTracePlan scalar type $T; rebuild the plan from the state " *
        "or with T=$(_real_float_type(eltype(source.data)))"))
    eltype(output.data)===T||throw(ArgumentError(
        "output state scalar type $(eltype(output.data)) does not match " *
        "LocalFactorTracePlan scalar type $T"))
    eltype(workspace.occupation_coordinates)===T||error(
        "internal local-factor workspace scalar mismatch")
    if _real_float_type(T)===BigFloat
        required=plan.estimates.precision_bits
        for (name,values) in (
                ("source state",source.data),
                ("output state",output.data),
                ("workspace",workspace.occupation_coordinates))
            bounds=_local_factor_precision_bounds(values)
            bounds==(required,required)||throw(ArgumentError(
                "$name BigFloat storage has precision range $bounds, but " *
                "the LocalFactorTracePlan requires $required bits; " *
                "rebuild it in the plan precision"))
        end
    end
    nothing
end

"""
    local_factor_trace!(output, source, plan, workspace;
                        check=true,
                        atol=_analysis_atol(source),
                        rtol=_state_rtol(source))

Trace the selected local factor from every supersite of `source`, writing the
result into `output`. The plan and task-owned workspace are reused without
hidden state-sized allocation. `output` must use `plan.output_basis`.

With `check=true`, both input and output are validated at the requested
tolerance. Set `check=false` only when validation is already provided by the
surrounding prepared workflow and the allocation-free contraction itself is
required. The routine never normalizes, symmetrizes, clips eigenvalues, or
otherwise repairs either state.
"""
function local_factor_trace!(
        output::PIState,source::PIState,plan::LocalFactorTracePlan,
        workspace::LocalFactorTraceWorkspace;
        check::Bool=true,
        atol::Real=_analysis_atol(source),
        rtol::Real=_state_rtol(source))
    isfinite(atol)&&atol>=0||throw(ArgumentError(
        "atol must be finite and nonnegative"))
    isfinite(rtol)&&rtol>=0||throw(ArgumentError(
        "rtol must be finite and nonnegative"))
    _check_local_factor_trace_resources(output,source,plan,workspace)
    if _real_float_type(eltype(plan.lifted_columns))===BigFloat
        precision_bits=plan.estimates.precision_bits
        rounding_mode=plan.estimates.rounding_mode
        if precision(BigFloat)!=precision_bits||
                rounding(BigFloat)!=rounding_mode
            return setrounding(BigFloat,rounding_mode) do
                setprecision(BigFloat,precision_bits) do
                    local_factor_trace!(
                        output,source,plan,workspace;
                        check,atol,rtol)
                end
            end
        end
    end
    check&&validate_state(source;atol,rtol)
    mul!(workspace.occupation_coordinates,
         adjoint(plan.lifted_columns),source.data)
    mul!(output.data,plan.output_columns,
         workspace.occupation_coordinates)
    check&&validate_state(output;atol,rtol)
    output
end

"""
    local_factor_trace(source, plan; workspace=nothing, kwargs...)
    local_factor_trace(source, local_dimensions;
                       traced_factor=2, memory_budget=512*1024^2,
                       setup_atol=nothing, setup_rtol=nothing, kwargs...)

Return the PI state obtained by tracing one internal tensor factor from every
identical supersite. Pass a prepared plan and workspace for parameter scans.

This operation is different from [`reduced_state`](@ref): local-factor tracing
keeps all `N` particles and reduces each particle's internal Hilbert space,
whereas `reduced_state` keeps only a selected number of particles.
"""
function local_factor_trace(
        source::PIState,plan::LocalFactorTracePlan;
        workspace=nothing,
        check::Bool=true,
        atol::Real=_analysis_atol(source),
        rtol::Real=_state_rtol(source))
    if _real_float_type(eltype(plan.lifted_columns))===BigFloat
        precision_bits=plan.estimates.precision_bits
        rounding_mode=plan.estimates.rounding_mode
        if precision(BigFloat)!=precision_bits||
                rounding(BigFloat)!=rounding_mode
            return setrounding(BigFloat,rounding_mode) do
                setprecision(BigFloat,precision_bits) do
                    local_factor_trace(
                        source,plan;workspace,check,atol,rtol)
                end
            end
        end
    end
    work=workspace===nothing ? LocalFactorTraceWorkspace(plan) : workspace
    work isa LocalFactorTraceWorkspace||throw(ArgumentError(
        "workspace must be a LocalFactorTraceWorkspace"))
    T=_real_float_type(eltype(plan.lifted_columns))
    output=PIState(plan.output_basis;T)
    local_factor_trace!(output,source,plan,work;check,atol,rtol)
end

function local_factor_trace(
        source::PIState,local_dimensions;
        traced_factor::Integer=2,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,
        setup_atol=nothing,setup_rtol=nothing,
        check::Bool=true,
        atol::Real=_analysis_atol(source),
        rtol::Real=_state_rtol(source))
    plan=LocalFactorTracePlan(
        source,local_dimensions;traced_factor,memory_budget,
        atol=setup_atol,rtol=setup_rtol)
    workspace=LocalFactorTraceWorkspace(plan)
    local_factor_trace(source,plan;workspace,check,atol,rtol)
end
