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

The plan stores two read-only rectangular transforms. Reusing it reduces each
state with two dense matrix-vector products. Setup and application never
construct a `source_basis.d^N` state, a full-system density matrix, or a list
of local operator strings. The retained coefficient count is

`length(source_basis)*length(output_basis) + length(output_basis)^2`.

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
    alphabet_size_big=BigInt(kept_dimension)^2
    output_size_big=exact_binomial(
        BigInt(basis.N)+alphabet_size_big-1,basis.N)
    retained_entries=BigInt(input_size)*output_size_big+
                     output_size_big^2
    # Besides the retained transforms, setup owns their Schur recurrences,
    # normalized columns, and the output Gram check. This deliberately
    # conservative factor guards those transient matrices without pretending
    # to predict allocator-specific object headers.
    peak_entries=3retained_entries+
                 4(BigInt(input_size)+output_size_big)
    retained_bytes=_performance_entries_bytes(retained_entries,CT)
    peak_bytes=_performance_entries_bytes(peak_entries,CT)
    _require_performance_budget(
        "local-factor trace plan setup",peak_bytes,memory_budget;guidance=
        "Reduce N/local dimensions or prepare only when the rectangular PI transform fits.")

    output_size_big<=typemax(Int)||throw(ArgumentError(
        "the kept-factor PI dimension exceeds Int indexing capacity"))
    alphabet_size_big<=typemax(Int)||throw(ArgumentError(
        "the kept-factor operator alphabet exceeds Int indexing capacity"))
    output_size=Int(output_size_big)
    alphabet_size=Int(alphabet_size_big)
    output_basis=PIBasis(basis.N,kept_dimension)
    length(output_basis)==output_size||error(
        "internal local-factor PI dimension mismatch")
    counts=_local_factor_compositions(basis.N,Val(alphabet_size))
    length(counts)==output_size||error(
        "internal local-factor occupation/PI dimension mismatch")

    output_columns=zeros(CT,output_size,output_size)
    output_builder=_LocalFactorColumnBuilder(
        output_basis,_local_factor_matrix_unit_letters(kept_dimension),R)
    for (column,occupation) in pairs(counts)
        _local_factor_column!(
            view(output_columns,:,column),output_builder,occupation)
    end

    lifted_columns=zeros(CT,input_size,output_size)
    input_builder=_LocalFactorColumnBuilder(
        basis,_local_factor_lifted_letters(dimensions,traced),R)
    for (column,occupation) in pairs(counts)
        _local_factor_column!(
            view(lifted_columns,:,column),input_builder,occupation)
    end

    setup_atol,setup_rtol=_local_factor_setup_tolerances(
        R,atol,rtol,basis.N,basis.d)
    gram=adjoint(output_columns)*output_columns
    gram_residual=opnorm(
        gram-Matrix{CT}(I,output_size,output_size),Inf)
    gram_scale=max(one(R),opnorm(output_columns,Inf)^2)
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
    trace_residual<=setup_atol+setup_rtol*trace_scale||throw(ErrorException(
        "local-factor trace map is not trace preserving within setup " *
        "tolerance: residual=$trace_residual, " *
        "tolerance=$(setup_atol+setup_rtol*trace_scale)"))

    estimates=(input_dimension=input_size,output_dimension=output_size,
        retained_entries,retained_bytes,peak_entries,peak_bytes,
        memory_budget=_memory_budget_bytes(memory_budget),
        scalar_type=CT,gram_residual,trace_residual)
    LocalFactorTracePlan{R,B,typeof(output_basis),typeof(lifted_columns),
        typeof(output_columns),typeof(estimates)}(
        basis,output_basis,dimensions,traced,kept,lifted_columns,
        output_columns,estimates)
end

function LocalFactorTracePlan(
        rho::PIState,local_dimensions;
        T::Type{R}=_real_float_type(eltype(rho.data)),
        kwargs...) where R<:AbstractFloat
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
    LocalFactorTraceWorkspace{T,typeof(plan)}(
        plan,zeros(T,length(plan.output_basis)))
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
    atol>=0||throw(ArgumentError("atol must be nonnegative"))
    rtol>=0||throw(ArgumentError("rtol must be nonnegative"))
    _check_local_factor_trace_resources(output,source,plan,workspace)
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
