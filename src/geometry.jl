struct _PackedOneBoxTransitions{T<:AbstractFloat}
    offsets::Vector{Int}
    # (upper-pattern index, one-based local label, coefficient), grouped by
    # lower-pattern index.  Structural zeros are not retained.
    terms::Vector{Tuple{Int,Int,T}}
end

"""
    OneBoxCGCache(basis; max_depth=min(1, basis.N), T=Float64,
                  memory_budget=64 * 1024^2)

Precompute the nonzero one-box Gelfand--Tsetlin Clebsch--Gordan transitions
reachable from the sectors of `basis` through at most `max_depth` successive
box removals.  Compatible parent candidates are selected by their content;
the cache retains packed sparse transition tuples, never dense Cartesian
tables of structural zeros.

The cache is immutable in use, belongs to the exact `basis` object supplied
at construction, and may be shared by read-only geometry constructors.  Its
floating type is fixed by `T`.  A `BigFloat` cache also records the active
precision and rounding mode and refuses reuse under different arithmetic
settings, preventing an apparently high-precision computation from consuming
incompatibly rounded stored coefficients.  `max_depth=0` is valid and stores
no transitions.

Construction is guarded by `memory_budget` (64 MiB by default); pass `Inf` to
opt out explicitly.  `candidate_count`, `coefficient_count`, and
`estimated_bytes` report respectively the compatible candidates evaluated,
the retained nonzeros, and a conservative retained-storage estimate.  No
global mutable cache is used.
"""
struct OneBoxCGCache{T,D,L,B<:PIBasis{D,L},Q}
    basis::B
    max_depth::Int
    precision_bits::Int
    rounding_mode::Q
    patterns::Dict{Partition{D},Vector{GTPattern{D,L}}}
    pattern_indices::Dict{Partition{D},Dict{GTPattern{D,L},Int}}
    depths::Dict{Partition{D},Int}
    transitions::Dict{Tuple{Partition{D},Partition{D}},
                      _PackedOneBoxTransitions{T}}
    candidate_count::Int
    coefficient_count::Int
    estimated_bytes::BigInt
end

_cgc_cache_scalar_type(::OneBoxCGCache{T}) where T=T
_cgc_cache_scalar_type(cache)=throw(ArgumentError(
    "cache must be a OneBoxCGCache or nothing, got $(typeof(cache))"))
_resolve_cgc_scalar_type(::Nothing,::OneBoxCGCache{T}) where T=T
function _resolve_cgc_scalar_type(::Type{T},
        ::OneBoxCGCache{R}) where {T<:AbstractFloat,R<:AbstractFloat}
    isconcretetype(T)||throw(ArgumentError(
        "T must be a concrete AbstractFloat type, got $T"))
    T===R||throw(ArgumentError(
        "OneBoxCGCache stores $R coefficients but T=$T was requested"))
    # Return the cache's type parameter, which remains inferable even though
    # Julia keyword NamedTuples type a supplied `T` value only as `DataType`.
    R
end
geometry_scalar_type(::OneBoxCGCache{T}) where T=T

function _onebox_memory_budget(memory_budget)
    memory_budget isa Real&&!isa(memory_budget,Bool)||throw(ArgumentError(
        "memory_budget must be a nonnegative byte count or Inf"))
    isnan(memory_budget)&&throw(ArgumentError("memory_budget must not be NaN"))
    memory_budget<0&&throw(ArgumentError("memory_budget must be nonnegative"))
    isinf(memory_budget)&&return nothing
    try
        floor(BigInt,memory_budget)
    catch error
        throw(ArgumentError("memory_budget is not a representable byte count: "*
                            sprint(showerror,error)))
    end
end

function _check_onebox_memory_budget(estimated::BigInt,memory_budget)
    budget=_onebox_memory_budget(memory_budget)
    budget===nothing&&return estimated
    estimated<=budget||throw(ArgumentError(
        "OneBoxCGCache requires an estimated $estimated retained bytes, "*
        "exceeding memory_budget=$budget; reduce max_depth, use a restricted "*
        "PIBasis, increase memory_budget, or pass memory_budget=Inf explicitly"))
    estimated
end

function _onebox_domain(b::PIBasis{D},max_depth::Int) where D
    levels=Vector{Vector{Partition{D}}}(undef,max_depth+1)
    levels[1]=copy(b.sectors)
    depths=Dict{Partition{D},Int}(p=>0 for p in levels[1])
    edges=Vector{Tuple{Partition{D},Partition{D}}}()
    for depth in 1:max_depth
        next=Partition{D}[]
        seen=Set{Partition{D}}()
        for upper in levels[depth],corner in removable_corners(upper)
            lower=remove_corner(upper,corner)
            push!(edges,(lower,upper))
            if !(lower in seen)
                push!(seen,lower);push!(next,lower)
            end
        end
        sort!(next;by=p->p.parts,rev=true)
        levels[depth+1]=next
        for partition in next
            depths[partition]=depth
        end
    end
    unique!(edges)
    levels,depths,edges
end

function _onebox_pattern_storage_estimate(::Type{R},::Val{L},pattern_count,
        edge_count,candidate_count,partition_count,precision_bits) where
        {R<:AbstractFloat,L}
    int_bytes=big(sizeof(Int));pointer_bytes=big(sizeof(Ptr{Cvoid}))
    scalar_bytes=_scalar_retained_bytes(R;
        bigfloat_precision=precision_bits)
    # Includes packed tuples and offsets plus deliberately conservative
    # dictionary/load-factor and container allowances for patterns and edges.
    structural=big(candidate_count)*(2int_bytes+scalar_bytes)+
        big(pattern_count)*(big(L)*int_bytes+12int_bytes+4pointer_bytes)+
        big(edge_count)*(8int_bytes+4pointer_bytes)+
        big(partition_count)*(big(L)*int_bytes+12int_bytes+4pointer_bytes)
    # `summarysize` also counts dictionary hash tables, bucket slack, array
    # headers, and allocator alignment whose exact sizes vary across supported
    # Julia releases. A factor-two envelope plus a fixed small-cache allowance
    # keeps the public estimate conservative without inspecting or allocating
    # the coefficient table first.
    2structural+65536
end

# Allocation-free structural upper bound used by guarded high-level model
# preparation.  Content filtering makes the retained table much smaller in
# practice; the dense compatible-parent bound is deliberately conservative so
# the automatic small-system cache never hides memory outside the command's
# declared budget.
function _estimate_onebox_cache_upper(b::PIBasis{D,L},max_depth::Integer,
        ::Type{R};precision_bits::Integer=
            (R===BigFloat ? precision(BigFloat) : precision(R))) where
        {D,L,R<:AbstractFloat}
    0<=max_depth<=b.N||throw(ArgumentError(
        "max_depth must satisfy 0 ≤ max_depth ≤ N=$(b.N)"))
    levels,_,edges=_onebox_domain(b,Int(max_depth))
    partitions_at_depth=reduce(vcat,levels;init=Partition{D}[])
    dimensions=Dict(p=>unitary_group_dimension(p)
                    for p in partitions_at_depth)
    pattern_count=sum(values(dimensions);init=big(0))
    candidate_upper=sum((big(D)*dimensions[lower]*dimensions[upper]
                         for (lower,upper) in edges);init=big(0))
    _onebox_pattern_storage_estimate(R,Val(L),pattern_count,length(edges),
        candidate_upper,length(partitions_at_depth),Int(precision_bits))
end

function _onebox_candidate_count(lower_patterns,upper_patterns)
    D=length(shape(first(lower_patterns)))
    upper_by_weight=Dict{NTuple{D,Int},Int}()
    for upper in upper_patterns
        key=content(upper)
        upper_by_weight[key]=get(upper_by_weight,key,0)+1
    end
    count=0
    for lower in lower_patterns
        lower_weight=content(lower)
        for local_label in 1:D
            parent_weight=ntuple(k->lower_weight[k]+(k==local_label),D)
            count=Base.checked_add(count,get(upper_by_weight,parent_weight,0))
        end
    end
    count
end

function _build_onebox_transitions(lower_patterns::Vector{GTPattern{D,L}},
        upper_patterns::Vector{GTPattern{D,L}},::Type{R}) where
        {D,L,R<:AbstractFloat}
    upper_by_weight=Dict{NTuple{D,Int},Vector{Int}}()
    for (upper_index,upper) in pairs(upper_patterns)
        push!(get!(()->Int[],upper_by_weight,content(upper)),upper_index)
    end
    offsets=Vector{Int}(undef,length(lower_patterns)+1);offsets[1]=1
    terms=Tuple{Int,Int,R}[]
    for (lower_index,lower) in pairs(lower_patterns)
        lower_weight=content(lower)
        for local_label in 1:D
            parent_weight=ntuple(k->lower_weight[k]+(k==local_label),D)
            upper_indices=get(upper_by_weight,parent_weight,nothing)
            upper_indices===nothing&&continue
            for upper_index in upper_indices
                value=_cgc_uncached(lower,local_label-1,
                    upper_patterns[upper_index],R)
                iszero(value)||push!(terms,(upper_index,local_label,value))
            end
        end
        offsets[lower_index+1]=length(terms)+1
    end
    _PackedOneBoxTransitions{R}(offsets,terms)
end

OneBoxCGCache(b::PIBasis;max_depth::Integer=min(1,b.N),T=Float64,
        memory_budget=64*1024^2)=
    OneBoxCGCache(b,T;max_depth,memory_budget)

function OneBoxCGCache(b::B,::Type{R};max_depth::Integer=min(1,b.N),
        memory_budget=64*1024^2) where
        {D,L,B<:PIBasis{D,L},R<:AbstractFloat}
    isconcretetype(R)||throw(ArgumentError(
        "T must be a concrete AbstractFloat type, got $R"))
    0<=max_depth<=b.N||throw(ArgumentError(
        "max_depth must satisfy 0 ≤ max_depth ≤ N=$(b.N)"))
    depth=Int(max_depth);precision_bits=R===BigFloat ? precision(BigFloat) :
        precision(R)
    rounding_mode=R===BigFloat ? rounding(BigFloat) : nothing
    levels,depths,edges=_onebox_domain(b,depth)
    partitions_at_depth=reduce(vcat,levels;init=Partition{D}[])

    # Guard the unavoidable pattern/index storage before enumerating any
    # descendant GT-pattern table. Top-level patterns remain basis-owned.
    exact_pattern_count=sum((unitary_group_dimension(p)
                             for p in partitions_at_depth);init=big(0))
    exact_pattern_count<=typemax(Int)||throw(ArgumentError(
        "OneBoxCGCache pattern count exceeds Int indexing; reduce max_depth "*
        "or restrict the basis"))
    minimum_estimate=_onebox_pattern_storage_estimate(R,Val(L),
        exact_pattern_count,length(edges),0,length(partitions_at_depth),
        precision_bits)
    _check_onebox_memory_budget(minimum_estimate,memory_budget)

    patterns=Dict{Partition{D},Vector{GTPattern{D,L}}}()
    for p in partitions_at_depth
        sector_index=get(b.index,p,0)
        patterns[p]=get(depths,p,typemax(Int))==0&&sector_index>0 ?
            b.patterns[sector_index] : gt_patterns(p)
    end
    candidate_count=0
    for (lower,upper) in edges
        candidate_count=Base.checked_add(candidate_count,
            _onebox_candidate_count(patterns[lower],patterns[upper]))
    end
    estimate=_onebox_pattern_storage_estimate(R,Val(L),
        exact_pattern_count,length(edges),candidate_count,
        length(partitions_at_depth),precision_bits)
    _check_onebox_memory_budget(estimate,memory_budget)

    indices=Dict{Partition{D},Dict{GTPattern{D,L},Int}}(
        p=>Dict(pattern=>index for (index,pattern) in pairs(patterns[p]))
        for p in partitions_at_depth)
    transitions=Dict{Tuple{Partition{D},Partition{D}},
                     _PackedOneBoxTransitions{R}}()
    coefficient_count=0
    for edge in edges
        lower,upper=edge
        table=_build_onebox_transitions(patterns[lower],patterns[upper],R)
        transitions[edge]=table
        coefficient_count=Base.checked_add(coefficient_count,length(table.terms))
    end
    OneBoxCGCache{R,D,L,B,typeof(rounding_mode)}(
        b,depth,precision_bits,rounding_mode,patterns,indices,depths,
        transitions,candidate_count,coefficient_count,estimate)
end

function _check_onebox_cache_precision(cache::OneBoxCGCache{R},
                                       ::Type{T}) where
        {R<:AbstractFloat,T<:AbstractFloat}
    R===T||throw(ArgumentError(
        "OneBoxCGCache stores $R coefficients but T=$T was requested"))
    if R===BigFloat&&cache.precision_bits!=precision(BigFloat)
        throw(ArgumentError(
            "OneBoxCGCache was constructed at BigFloat precision "*
            "$(cache.precision_bits), but the active precision is "*
            "$(precision(BigFloat)); rebuild the cache at the requested precision"))
    end
    if R===BigFloat&&cache.rounding_mode!=rounding(BigFloat)
        throw(ArgumentError(
            "OneBoxCGCache was constructed with BigFloat rounding mode "*
            "$(cache.rounding_mode), but the active mode is "*
            "$(rounding(BigFloat)); rebuild the cache with the requested "*
            "rounding mode"))
    end
    cache
end

function _check_onebox_coefficient_cache(cache::OneBoxCGCache,b::PIBasis,
        required_depth::Integer,::Type{R}) where R<:AbstractFloat
    cache.basis===b||throw(ArgumentError(
        "OneBoxCGCache was constructed for a different PIBasis; construct "*
        "or reuse a cache owned by this exact basis"))
    cache.max_depth>=required_depth||throw(ArgumentError(
        "OneBoxCGCache max_depth=$(cache.max_depth) does not cover required "*
        "depth $required_depth"))
    _check_onebox_cache_precision(cache,R)
end

_check_onebox_coefficient_cache(cache,b::PIBasis,required_depth::Integer,
        ::Type{R}) where R<:AbstractFloat=throw(ArgumentError(
    "coefficient_cache must be a OneBoxCGCache or nothing, got "*
    "$(typeof(cache))"))

function _cached_partition_depth(cache::OneBoxCGCache,p::Partition,role)
    depth=get(cache.depths,p,nothing)
    depth!==nothing&&return depth
    implied=cache.basis.N-weight(p)
    if implied<0||implied>cache.max_depth
        throw(ArgumentError(
            "$role partition $p lies at depth $implied outside "*
            "OneBoxCGCache max_depth=$(cache.max_depth)"))
    end
    throw(ArgumentError(
        "$role partition $p is not reachable from the retained sectors of "*
        "the exact PIBasis owned by this OneBoxCGCache"))
end

function _cached_cgc(cache::OneBoxCGCache{R,D,L},mu::GTPattern{D,L},
        j::Integer,lam::GTPattern{D,L},::Type{T}) where
        {R<:AbstractFloat,T<:AbstractFloat,D,L}
    _check_onebox_cache_precision(cache,T)
    lower=shape(mu);upper=shape(lam)
    _cached_partition_depth(cache,lower,"lower")
    _cached_partition_depth(cache,upper,"upper")
    0<=j<D||return zero(T)
    weight(upper)==weight(lower)+1||return zero(T)
    table=get(cache.transitions,(lower,upper),nothing)
    table===nothing&&return zero(T)
    lower_index=get(cache.pattern_indices[lower],mu,0)
    upper_index=get(cache.pattern_indices[upper],lam,0)
    (lower_index>0&&upper_index>0)||throw(ArgumentError(
        "CG pattern is outside the canonical pattern tables of this cache"))
    @inbounds for term_index in
            table.offsets[lower_index]:(table.offsets[lower_index+1]-1)
        candidate_upper,candidate_label,value=table.terms[term_index]
        candidate_upper==upper_index&&candidate_label==j+1&&return value
    end
    zero(T)
end

function _cached_cgc(cache::OneBoxCGCache,mu::GTPattern,j::Integer,
                     lam::GTPattern,::Type{T}) where T<:AbstractFloat
    _check_onebox_cache_precision(cache,T)
    throw(ArgumentError(
        "CG patterns have a local dimension incompatible with this OneBoxCGCache"))
end

function Base.show(io::IO,cache::OneBoxCGCache{T}) where T
    print(io,"OneBoxCGCache(N=$(cache.basis.N), d=$(cache.basis.d), "*
        "max_depth=$(cache.max_depth), T=$T, "*
        "coefficients=$(cache.coefficient_count))")
end

struct _OneBodyContractionView{T} <: AbstractVector{Tuple{Int,Int,T}}
    terms::Vector{Tuple{Int,Int,T}}
    first::Int
    last::Int
end

Base.IndexStyle(::Type{<:_OneBodyContractionView})=IndexLinear()
Base.size(view::_OneBodyContractionView)=(length(view),)
Base.length(view::_OneBodyContractionView)=max(view.last-view.first+1,0)
@inline function Base.getindex(view::_OneBodyContractionView,index::Int)
    @boundscheck checkbounds(view,index)
    @inbounds view.terms[view.first+index-1]
end
@inline function Base.iterate(view::_OneBodyContractionView,
                              position::Int=view.first)
    position>view.last&&return nothing
    @inbounds (view.terms[position],position+1)
end

# One table replaces a matrix of heap-allocated tiny vectors by two contiguous
# arrays.  `table[a,c]` remains an `AbstractVector`, so contraction hot paths
# and the public read-only geometry semantics do not change.
struct _PackedOneBodyContractions{T} <:
       AbstractMatrix{_OneBodyContractionView{T}}
    nrows::Int
    ncols::Int
    offsets::Vector{Int}
    terms::Vector{Tuple{Int,Int,T}}
end

Base.IndexStyle(::Type{<:_PackedOneBodyContractions})=IndexLinear()
Base.size(table::_PackedOneBodyContractions)=(table.nrows,table.ncols)
@inline function Base.getindex(table::_PackedOneBodyContractions{T},
                               index::Int) where T
    @boundscheck checkbounds(table,index)
    @inbounds _OneBodyContractionView{T}(
        table.terms,table.offsets[index],table.offsets[index+1]-1)
end
@inline function Base.getindex(table::_PackedOneBodyContractions,
                               row::Int,column::Int)
    @boundscheck checkbounds(table,row,column)
    @inbounds table[row+(column-1)*table.nrows]
end

struct _OneBodyConnections{D}
    nonempty::Dict{Tuple{Int,Int},Vector{Partition{D}}}
    empty::Vector{Partition{D}}
    nsectors::Int
end

function Base.getindex(connections::_OneBodyConnections,
                       key::Tuple{Int,Int})
    1<=key[1]<=connections.nsectors&&1<=key[2]<=connections.nsectors||
        throw(KeyError(key))
    get(connections.nonempty,key,connections.empty)
end
Base.length(connections::_OneBodyConnections)=connections.nsectors^2

function _pack_onebody_contractions(left,right,nrows::Int,ncols::Int,
                                    ::Type{T}) where T
    ncells=Base.checked_mul(nrows,ncols)
    counts=zeros(Int,ncells)
    @inbounds for child_index in eachindex(left)
        for (a,i,x) in left[child_index],(c,j,y) in right[child_index]
            value=x*y
            iszero(value)&&continue
            counts[a+(c-1)*nrows]+=1
        end
    end
    offsets=Vector{Int}(undef,ncells+1);offsets[1]=1
    @inbounds for index in 1:ncells
        offsets[index+1]=Base.checked_add(offsets[index],counts[index])
    end
    terms=Vector{Tuple{Int,Int,T}}(undef,offsets[end]-1)
    cursor=copy(@view offsets[1:end-1])
    @inbounds for child_index in eachindex(left)
        for (a,i,x) in left[child_index],(c,j,y) in right[child_index]
            value=x*y
            iszero(value)&&continue
            cell=a+(c-1)*nrows
            terms[cursor[cell]]=(i,j,value)
            cursor[cell]+=1
        end
    end
    _PackedOneBodyContractions{T}(nrows,ncols,offsets,terms)
end

"""
    OneBodyGeometry(basis; T=Float64, coefficient_cache=nothing)

Precompute the one-box Clebsch--Gordan contractions, multiplicity scales, and
sector connections shared by local and collective one-particle operations.
Contraction tables retain one contiguous tuple array and column-major offsets;
logical empty cells and absent sector pairs therefore require no individual
heap vectors or dictionary entries.
The cache is read-only after construction and is tied to the exact `PIBasis`
object supplied at construction; it may be reused to prepare many operators.
Pass a basis-owned [`OneBoxCGCache`](@ref) with `max_depth >= 1` for `N > 0`
through `coefficient_cache` to reuse precomputed CG coefficients across
several geometry constructions. A zero-particle basis requires depth zero.
The default preserves call-local construction and does not create global
state.
"""
struct OneBodyGeometry{T,D,L,B<:PIBasis{D,L}}
    basis::B
    # (lambda sector, mu partition, nu sector) => packed tables indexed (a,c),
    # each exposing a sparse dyadic vector (i,j,value) without per-cell storage.
    contractions::Dict{Tuple{Int,Partition{D},Int},_PackedOneBodyContractions{T}}
    scales::Dict{Tuple{Int,Partition{D},Int},T}
    connections::_OneBodyConnections{D}
end
OneBodyGeometry(b::PIBasis;T=Float64,coefficient_cache=nothing)=
    OneBodyGeometry(b,T;coefficient_cache)

# For a fixed child pattern and local label, content conservation leaves only
# parent patterns of one known weight.  Building these sparse transitions once
# removes the previous parent-pair x child-pattern x d^2 repetition of the
# exact-rational CG calculation.
function _onebody_transitions(parent_patterns::Vector{GTPattern{D,L}},
                              child_patterns::Vector{GTPattern{D,L}},
                              ::Type{R}) where {D,L,R<:AbstractFloat}
    parent_by_weight=Dict{NTuple{D,Int},Vector{Int}}()
    for (parent_index,parent) in pairs(parent_patterns)
        push!(get!(()->Int[],parent_by_weight,content(parent)),parent_index)
    end
    Transition=Tuple{Int,Int,R} # parent index, one-based local label, CG
    transitions=[Transition[] for _ in eachindex(child_patterns)]
    for (child_index,child) in pairs(child_patterns)
        child_weight=content(child)
        for local_label in 0:D-1
            parent_weight=ntuple(
                level->child_weight[level]+(level==local_label+1),D)
            parent_indices=get(parent_by_weight,parent_weight,nothing)
            parent_indices===nothing&&continue
            for parent_index in parent_indices
                value=_cgc_uncached(
                    child,local_label,parent_patterns[parent_index],R)
                iszero(value)||push!(transitions[child_index],
                                     (parent_index,local_label+1,value))
            end
        end
    end
    transitions
end

function _onebody_transitions(parent_patterns::Vector{GTPattern{D,L}},
        child_patterns::Vector{GTPattern{D,L}},::Type{R},
        cache::OneBoxCGCache{R,D,L}) where {D,L,R<:AbstractFloat}
    parent_partition=shape(first(parent_patterns))
    child_partition=shape(first(child_patterns))
    cached_parent=get(cache.patterns,parent_partition,nothing)
    cached_child=get(cache.patterns,child_partition,nothing)
    cached_parent==parent_patterns&&cached_child==child_patterns||throw(
        ArgumentError("OneBoxCGCache pattern ordering is incompatible with geometry"))
    table=get(cache.transitions,(child_partition,parent_partition),nothing)
    table===nothing&&throw(ArgumentError(
        "OneBoxCGCache does not contain required transition "*
        "$child_partition -> $parent_partition"))
    Transition=Tuple{Int,Int,R}
    transitions=[Transition[] for _ in eachindex(child_patterns)]
    @inbounds for child_index in eachindex(child_patterns)
        for term_index in
                table.offsets[child_index]:(table.offsets[child_index+1]-1)
            push!(transitions[child_index],table.terms[term_index])
        end
    end
    transitions
end

_onebody_transitions(parent_patterns,child_patterns,::Type{R},::Nothing) where
    {R<:AbstractFloat}=_onebody_transitions(parent_patterns,child_patterns,R)

function OneBodyGeometry(b::B,::Type{R};coefficient_cache=nothing) where
        {D,L,B<:PIBasis{D,L},R<:AbstractFloat}
    coefficient_cache===nothing||_check_onebox_coefficient_cache(
        coefficient_cache,b,iszero(b.N) ? 0 : 1,R)
    Term=Tuple{Int,Int,R}
    dict=Dict{Tuple{Int,Partition{D},Int},_PackedOneBodyContractions{R}}()
    scales=Dict{Tuple{Int,Partition{D},Int},R}()
    removed=[[remove_corner(p,r) for r in removable_corners(p)] for p in b.sectors]
    pattern_cache=Dict{Partition{D},Vector{GTPattern{D,L}}}()

    # Each (parent sector, child partition) transition table is shared by all
    # right-hand sectors meeting that child.  It is construction scratch and
    # is not retained by the finished read-only geometry.
    transition_cache=Dict{Tuple{Int,Partition{D}},Vector{Vector{Tuple{Int,Int,R}}}}()
    for (sector_index,children) in pairs(removed),mu in children
        child_patterns=get!(() -> gt_patterns(mu),pattern_cache,mu)
        transition_cache[(sector_index,mu)]=
            _onebody_transitions(b.patterns[sector_index],child_patterns,R,
                                 coefficient_cache)
    end

    connection_data=Dict{Tuple{Int,Int},Vector{Partition{D}}}()
    for (li,l) in pairs(b.sectors),(ni,n) in pairs(b.sectors)
        common=Partition{D}[mu for mu in removed[li] if mu in removed[ni]]
        isempty(common)||setindex!(connection_data,common,(li,ni))
        for mu in common
            left=transition_cache[(li,mu)];right=transition_cache[(ni,mu)]
            arr=_pack_onebody_contractions(left,right,
                length(b.patterns[li]),length(b.patterns[ni]),R)
            key=(li,mu,ni);dict[key]=arr
            left_weight=_one_box_branch_weight(l,mu)
            right_weight=_one_box_branch_weight(n,mu)
            scale_squared=left_weight*right_weight
            scales[key]=_checked_sqrt_exact_ratio(
                R,numerator(scale_squared),denominator(scale_squared);
                context="one-body Schur branching scale")
        end
    end
    connections=_OneBodyConnections{D}(
        connection_data,Partition{D}[],length(b.sectors))
    OneBodyGeometry{R,D,L,B}(b,dict,scales,connections)
end

# Structural pre-construction estimate.  Counts and byte estimates are exact
# BigInts and require no CG evaluation or geometry construction.  A transition
# candidate is identified only from GT-pattern content: for a child pattern and
# local label, every parent pattern of the required content is a possible
# nonzero CG transition.  The constructor evaluates exactly those candidates.
# Pairing the left and right candidate counts child by child therefore gives a
# rigorous upper bound on retained contraction tuples, including qudit weight
# multiplicities (where the simpler d^2*dim(mu) count is not an upper bound).
#
# Tuple/vector overhead is deliberately conservative and describes peak live
# structural storage, excluding the already caller-owned basis but not
# heap-backed scalar payloads.  Backing-store allowances cover dictionary and
# transition-vector capacity; packed contraction tuples themselves are stored
# in exactly sized contiguous vectors.  This is not cumulative
# garbage-collector allocation volume.
function _estimate_onebody_geometry(b::PIBasis{D,L},
                                    ::Type{R}=Float64;
                                    bigfloat_precision::Integer=
                                        precision(BigFloat)) where {D,L,R<:AbstractFloat}
    removed=[[remove_corner(p,r) for r in removable_corners(p)] for p in b.sectors]
    pattern_cache=Dict{Partition{D},Vector{GTPattern{D,L}}}()
    candidate_counts=Dict{Tuple{Int,Partition{D}},Vector{Int}}()
    transition_candidate_count=big(0)
    transition_vector_count=big(0)
    for (sector_index,children) in pairs(removed),mu in children
        child_patterns=get!(() -> gt_patterns(mu),pattern_cache,mu)
        parent_by_weight=Dict{NTuple{D,Int},Int}()
        for parent in b.patterns[sector_index]
            parent_weight=content(parent)
            parent_by_weight[parent_weight]=get(parent_by_weight,parent_weight,0)+1
        end
        counts=Vector{Int}(undef,length(child_patterns))
        for (child_index,child) in pairs(child_patterns)
            child_weight=content(child)
            count=0
            for local_label in 1:D
                parent_weight=ntuple(
                    level->child_weight[level]+(level==local_label),D)
                count+=get(parent_by_weight,parent_weight,0)
            end
            counts[child_index]=count
            transition_candidate_count+=count
        end
        candidate_counts[(sector_index,mu)]=counts
        transition_vector_count+=length(child_patterns)
    end

    connection_count=big(0);connected_pair_count=big(0)
    cell_count=big(0);term_upper=big(0);largest_cell_count=big(0)
    for li in eachindex(b.sectors),ni in eachindex(b.sectors)
        pair_connected=false
        for mu in removed[li]
            mu in removed[ni]||continue
            pair_connected=true
            connection_count+=1
            table_cells=big(length(b.patterns[li]))*length(b.patterns[ni])
            cell_count+=table_cells
            largest_cell_count=max(largest_cell_count,table_cells)
            left_counts=candidate_counts[(li,mu)]
            right_counts=candidate_counts[(ni,mu)]
            term_upper+=sum((big(left)*right for (left,right) in
                             zip(left_counts,right_counts));init=big(0))
        end
        connected_pair_count+=pair_connected
    end
    removal_count=sum(length,removed;init=0)
    unique_children=collect(keys(pattern_cache))
    child_pattern_count=sum(length(pattern_cache[mu]) for mu in unique_children;
                            init=0)

    pointer_bytes=big(sizeof(Ptr{Cvoid}));int_bytes=big(sizeof(Int))
    value_bytes=_scalar_retained_bytes(R;bigfloat_precision)
    container_header=8int_bytes
    dictionary_entry=16int_bytes
    initial_capacity=big(8)
    tuple_bytes=2int_bytes+value_bytes
    partition_bytes=big(D)*int_bytes
    pattern_bytes_per_entry=big(L)*int_bytes
    sector_pairs=big(length(b.sectors))^2

    # Retained contraction tables use one offset per logical matrix cell and
    # one contiguous tuple array.  There are no per-cell Vector headers or
    # pointers.  The constant allowances cover Julia container headers and
    # dictionary load-factor slack without relying on allocator internals.
    contraction_dictionary=container_header+connection_count*(
        dictionary_entry+2int_bytes+partition_bytes+pointer_bytes)
    contraction_tables=connection_count*(2container_header+4int_bytes)+
        int_bytes*(cell_count+connection_count)+2tuple_bytes*term_upper
    scale_dictionary=container_header+connection_count*(
        dictionary_entry+2int_bytes+partition_bytes+value_bytes)
    connection_dictionary=container_header+connected_pair_count*(
        dictionary_entry+2int_bytes+container_header)+
        partition_bytes*(connection_count+initial_capacity*connected_pair_count)
    retained_bytes=container_header+contraction_dictionary+
        contraction_tables+scale_dictionary+connection_dictionary

    # Construction scratch retained until the constructor returns: every
    # (sector,child) transition table, the unique child GT-pattern cache, and
    # the largest parent-content dictionary used to stage one table.
    transition_dictionary=container_header+big(removal_count)*(
        dictionary_entry+int_bytes+partition_bytes+pointer_bytes)
    transition_tables=container_header*big(removal_count)+
        pointer_bytes*transition_vector_count+
        container_header*transition_vector_count+
        initial_capacity*tuple_bytes*transition_candidate_count
    pattern_dictionary=container_header+big(length(unique_children))*(
        dictionary_entry+partition_bytes+pointer_bytes)
    pattern_storage=container_header*big(length(unique_children))+
        initial_capacity*pattern_bytes_per_entry*big(child_pattern_count)
    largest_parent_pattern_count=maximum(length,b.patterns;init=0)
    parent_weight_scratch=container_header+big(largest_parent_pattern_count)*(
        dictionary_entry+big(D)*int_bytes+pointer_bytes+container_header+
        initial_capacity*int_bytes)
    packing_scratch=2int_bytes*largest_cell_count+container_header
    removed_storage=container_header+container_header*big(length(b.sectors))+
        initial_capacity*partition_bytes*big(removal_count)
    setup_bytes=retained_bytes+transition_dictionary+transition_tables+
        pattern_dictionary+pattern_storage+parent_weight_scratch+
        removed_storage+packing_scratch
    (sector_pairs=big(length(b.sectors))^2,
     removal_count=big(removal_count),connection_count,connected_pair_count,
     cell_count,
     contraction_terms_upper=term_upper,
     transition_terms_upper=transition_candidate_count,
     cgc_evaluations_upper=transition_candidate_count,
     transition_vector_count,
     child_pattern_count=big(child_pattern_count),
     retained_bytes,setup_bytes)
end
geometry_scalar_type(::OneBodyGeometry{T}) where T=T
function _check_geometry_basis(cache::OneBodyGeometry,b::PIBasis)
    cache.basis===b||throw(ArgumentError("OneBodyGeometry was constructed for a different PIBasis; construct or reuse a cache owned by this exact basis"))
    cache
end

# The irrep carried by the one-row partition `(N,0,...)` is the symmetric
# N-boson occupation space.  Collective one-body operators never leave a
# Schur sector, so a basis containing only this irrep does not need the much
# larger one-box removal/subduction geometry used by local gain terms.  Keep a
# small, basis-owned occupation lookup instead.  This is deliberately an
# internal compilation geometry: mixed-sector and local processes continue to
# use `OneBodyGeometry` and its guarded recoupling path.
_is_fully_symmetric_partition(b::PIBasis,p::Partition)=
    p.parts[1]==b.N&&all(iszero,p.parts[2:end])
_has_single_fully_symmetric_sector(b::PIBasis)=
    length(b.sectors)==1&&_is_fully_symmetric_partition(b,only(b.sectors))

struct _SymmetricCollectiveGeometry{T,D,L,B<:PIBasis{D,L},O,I,W,F}
    basis::B
    occupations::O
    indices::I
    diagonal_factors::W
    transition_offsets::Vector{Int}
    transitions::F
end

function _SymmetricCollectiveGeometry(b::B,::Type{T}) where
        {D,L,B<:PIBasis{D,L},T<:AbstractFloat}
    isconcretetype(T)||throw(ArgumentError(
        "collective geometry requires a concrete floating-point type, got $T"))
    _has_single_fully_symmetric_sector(b)||throw(ArgumentError(
        "symmetric collective geometry requires the sole retained sector " *
        "($(b.N),0,...)"))
    _needs_wide_collective(b,T)&&throw(ArgumentError(
        "symmetric collective geometry at N=$(b.N) cannot certify native $T " *
        "occupation factors; use a wider geometry scalar type"))
    occupations=NTuple{D,Int}[content(pattern) for pattern in only(b.patterns)]
    indices=Dict{NTuple{D,Int},Int}(
        occupation=>index for (index,occupation) in pairs(occupations))
    length(indices)==length(occupations)||error(
        "symmetric-sector GT contents are not unique")
    diagonal_factors=NTuple{D,T}[ntuple(label->_symmetric_occupation_factor(
        T,occupation[label]),D)
        for occupation in occupations]
    transition_offsets=Vector{Int}(undef,length(occupations)+1)
    transition_offsets[1]=1
    transitions=Tuple{Int,Int,Int,T}[]
    for (column,occupation) in pairs(occupations)
        for source_label in 1:D
            source_count=occupation[source_label]
            iszero(source_count)&&continue
            for target_label in 1:D
                target_label==source_label&&continue
                raised=Base.checked_add(occupation[target_label],1)
                factor_squared=_checked_occupation_product(
                    raised,source_count)
                target=ntuple(label->occupation[label]+
                    (label==target_label ? 1 : 0)-
                    (label==source_label ? 1 : 0),D)
                push!(transitions,(indices[target],target_label,source_label,
                    _symmetric_transition_factor(T,factor_squared)))
            end
        end
        transition_offsets[column+1]=length(transitions)+1
    end
    _SymmetricCollectiveGeometry{T,D,L,B,typeof(occupations),typeof(indices),
        typeof(diagonal_factors),typeof(transitions)}(
        b,occupations,indices,diagonal_factors,transition_offsets,transitions)
end


function _symmetric_occupation_factor(::Type{T},count::Integer) where
        T<:AbstractFloat
    if T===Float16||T===Float32||T===Float64
        # `_needs_wide_collective` bounds N below the exact-integer range of
        # each IEEE type, so this conversion is both exact and allocation-free.
        return T(count)
    end
    _checked_exact_ratio(T,count,1;
        context="symmetric collective occupation factor")
end

function _symmetric_transition_factor(::Type{T},factor_squared::Integer) where
        T<:AbstractFloat
    if T===Float16||T===Float32||T===Float64
        # Under the same gate, the product is at most N^2 and remains an exact
        # IEEE integer before the correctly rounded square root.
        result=sqrt(T(factor_squared))
        isfinite(result)&&!iszero(result)||throw(ArgumentError(
            "symmetric collective transition factor is outside the nonzero " *
            "finite range of $T; use a wider geometry scalar type"))
        return result
    end
    _checked_sqrt_exact_ratio(T,factor_squared,1;
        context="symmetric collective transition factor")
end

geometry_scalar_type(::_SymmetricCollectiveGeometry{T}) where T=T
function _check_geometry_basis(cache::_SymmetricCollectiveGeometry,b::PIBasis)
    cache.basis===b||throw(ArgumentError(
        "symmetric collective geometry was constructed for a different " *
        "PIBasis; construct or reuse a cache owned by this exact basis"))
    cache
end

function _contract(terms::AbstractVector{<:Tuple{Int,Int,T}},X) where T
    S=promote_type(T,eltype(X))
    sum(conj(z)*X[i,j] for (i,j,z) in terms;init=zero(S))
end

function _collective_block_fast(b::PIBasis,X,p::Partition,
                                cache::OneBodyGeometry{T}) where T
    s=_sidx(b,p);n=length(b.patterns[s])
    K=zeros(promote_type(Complex{T},eltype(X)),n,n)
    for a in 1:n,c in 1:n,mu in cache.connections[(s,s)]
        r=cache.scales[(s,mu,s)]
        K[a,c]+=r*_contract(cache.contractions[(s,mu,s)][a,c],X)
    end
    K
end

function _checked_occupation_product(left::Int,right::Int)
    try
        Base.checked_mul(left,right)
    catch error
        error isa OverflowError||rethrow()
        big(left)*right
    end
end

function _symmetric_scaled_component(value::Real,factor::T,
        numerator_value::Integer,::Val{Root},context) where
        {T<:AbstractFloat,Root}
    result=value*factor
    R=_real_float_type(typeof(result))
    invalid=!isfinite(result)||(!iszero(value)&&iszero(result))
    endpoint=(R===Float16||R===Float32||R===Float64)&&
        (abs(result)==floatmax(R)||abs(result)==nextfloat(zero(R)))
    if invalid||endpoint
        # The ordinary path above is allocation-free.  Only a true floating
        # boundary pays for the exact rational certificate and its wider-type
        # guidance; return the original precision-compatible product after a
        # successful certificate.
        if Root
            _checked_mul_sqrt_exact_ratio(R,value,numerator_value,1;context)
        else
            _checked_mul_exact_ratio(R,value,numerator_value,1;context)
        end
    end
    result
end

function _symmetric_scaled_component(value::Complex,factor::T,
        numerator_value::Integer,root,context) where T<:AbstractFloat
    complex(_symmetric_scaled_component(
                real(value),factor,numerator_value,root,context),
            _symmetric_scaled_component(
                imag(value),factor,numerator_value,root,context))
end

# In the symmetric irrep, Gamma(X)=sum_ab X_ab a_a^dagger a_b.  The stored
# GT order need not coincide with lexicographic occupation order (already for
# N=1 it reverses the local matrix axes), hence the explicit content lookup.
# Exact integer factors remain fused with matrix entries so large occupations
# never create an avoidable `Inf*0` or a separately converted coefficient.
function _fill_symmetric_collective_block!(K,
        cache::_SymmetricCollectiveGeometry{T,D},X) where {T,D}
    b=cache.basis
    size(X)==(D,D)||throw(DimensionMismatch(
        "local collective operator must be $D×$D"))
    size(K)==(length(cache.occupations),length(cache.occupations))||
        throw(DimensionMismatch("collective Schur block has the wrong dimensions"))
    fill!(K,zero(eltype(K)))
    for (column,occupation) in pairs(cache.occupations)
        diagonal=zero(eltype(K))
        @inbounds for local_label in 1:D
            count=occupation[local_label]
            (iszero(count)||iszero(X[local_label,local_label]))&&continue
            diagonal+=_symmetric_scaled_component(
                X[local_label,local_label],
                cache.diagonal_factors[column][local_label],count,Val(false),
                "symmetric collective diagonal contribution")
        end
        K[column,column]=diagonal
        transition_range=cache.transition_offsets[column]:(cache.transition_offsets[column+1]-1)
        @inbounds for transition_index in transition_range
            row,target_label,source_label,factor=
                cache.transitions[transition_index]
            value=X[target_label,source_label]
            iszero(value)&&continue
            factor_squared=_checked_occupation_product(
                Base.checked_add(occupation[target_label],1),
                occupation[source_label])
            K[row,column]+=_symmetric_scaled_component(
                value,factor,factor_squared,Val(true),
                "symmetric collective transition contribution")
        end
    end
    all(isfinite,K)||throw(ArgumentError(
        "collective Schur block is outside the finite range of $(eltype(K)); " *
        "use a wider operator/geometry scalar type"))
    K
end

function _symmetric_collective_block(b::PIBasis,X,p::Partition,
        cache::_SymmetricCollectiveGeometry{T}) where T
    _check_geometry_basis(cache,b)
    p==only(b.sectors)||throw(ArgumentError(
        "partition $p is not retained by the symmetric collective geometry"))
    S=promote_type(Complex{T},eltype(X))
    K=Matrix{S}(undef,length(cache.occupations),length(cache.occupations))
    _fill_symmetric_collective_block!(K,cache,X)
end

function _collective_wider_type(::Type{T},N::Int) where T<:AbstractFloat
    # Choose the narrowest standard type for which cancellation of O(N)
    # branches into an O(1) generator loses no more than sqrt(eps).  A single
    # promotion step is insufficient, e.g. Float16 at N=60_000 or Float32 at
    # N=10^15.
    if T===Float16 && Float32(N)<=inv(sqrt(eps(Float32)))
        return Float32
    end
    if (T===Float16||T===Float32) && Float64(N)<=inv(sqrt(eps(Float64)))
        return Float64
    end
    BigFloat
end

function _needs_wide_collective(b::PIBasis,::Type{T}) where T<:AbstractFloat
    T===BigFloat&&return false
    threshold=inv(sqrt(eps(T)))
    particle_count=try
        T(b.N)
    catch
        T(Inf)
    end
    !isfinite(particle_count)||particle_count>threshold
end

function _convert_checked_geometry_value(::Type{S},source;
        context::AbstractString) where S
    R=_real_float_type(S)
    converted=try
        S(source)
    catch error
        throw(ArgumentError(
            "$context is not representable in $S; use a wider operator/geometry " *
            "scalar type: $(sprint(showerror,error))"))
    end
    isfinite(converted)||throw(ArgumentError(
        "$context is outside the finite range of $S; use a wider " *
        "operator/geometry scalar type"))
    source_components=(real(source),imag(source))
    converted_components=(real(converted),imag(converted))
    any((!iszero(original)&&iszero(stored) for
         (original,stored) in zip(source_components,converted_components)))&&
        throw(ArgumentError(
            "$context is outside the nonzero finite range of $S; use a wider " *
            "operator/geometry scalar type"))
    if R===Float16||R===Float32||R===Float64
        minimum_exact=Rational{BigInt}(nextfloat(zero(R)))
        maximum_exact=Rational{BigInt}(floatmax(R))
        for (original,stored) in zip(source_components,converted_components)
            iszero(original)&&continue
            magnitude=abs(stored)
            (magnitude==nextfloat(zero(R))||magnitude==floatmax(R))||continue
            minimum_exact<=abs(Rational{BigInt}(original))<=maximum_exact||
                throw(ArgumentError(
                    "$context is outside the nonzero finite range of $S; use a " *
                    "wider operator/geometry scalar type"))
        end
    end
    converted
end

# Recompute only the requested sector in a wider geometry precision.  This is
# activated beyond N*eps(T)~sqrt(eps(T)), where O(N) branch contributions can
# otherwise lose an O(1) traceless collective generator.  Feasible sectors at
# such enormous N have small retained irrep dimensions; ordinary sizes never
# pay for this secondary geometry.
function _collective_block_wide(b::PIBasis,X,p::Partition,::Type{W},::Type{S}) where
        {W<:AbstractFloat,S}
    compute=function ()
        restricted=PIBasis(b.N,b.d;sectors=[p.parts])
        wide_cache=OneBodyGeometry(restricted,W)
        wide_block=_collective_block_fast(restricted,X,p,wide_cache)
        result=Matrix{S}(undef,size(wide_block)...)
        for index in eachindex(result,wide_block)
            result[index]=_convert_checked_geometry_value(
                S,wide_block[index];context="collective Schur block entry")
        end
        result
    end
    if W===BigFloat
        # Guard bits beyond the N^2 cancellation requirement make this robust
        # even when the process-wide BigFloat precision was set unusually low.
        required=max(precision(BigFloat),2ndigits(big(max(b.N,1));base=2)+32)
        return setprecision(compute,required)
    end
    compute()
end

function _local_kernel_operation_count(cache::OneBodyGeometry,l::Partition,
        a,b,n::Partition,c,d)
    B=cache.basis;li=_sidx(B,l);ni=_sidx(B,n);operations=big(1)
    for mu in cache.connections[(li,ni)]
        arr=cache.contractions[(li,mu,ni)]
        operations+=2big(length(arr[a,c]))+2big(length(arr[b,d]))+8
    end
    operations
end

function _local_kernel_value_and_absolute(cache::OneBodyGeometry,X,Y,
        l::Partition,a,b,n::Partition,c,d)
    B=cache.basis;li=_sidx(B,l);ni=_sidx(B,n)
    S=promote_type(geometry_scalar_type(cache),eltype(X),eltype(Y))
    R=_real_float_type(S);z=zero(S);absolute_sum=zero(R)
    for mu in cache.connections[(li,ni)]
        key=(li,mu,ni);arr=cache.contractions[key]
        contribution=cache.scales[key]*_contract(arr[a,c],X)*
            conj(_contract(arr[b,d],Y))
        z+=contribution;absolute_sum+=abs(contribution)
    end
    z,absolute_sum
end

function _wide_local_kernel_element(cache::OneBodyGeometry,X,Y,l::Partition,
        a,b,n::Partition,c,d,::Type{S}) where S
    T=geometry_scalar_type(cache);W=_collective_wider_type(T,cache.basis.N)
    compute=function ()
        sectors=l==n ? [l.parts] : [l.parts,n.parts]
        restricted=PIBasis(cache.basis.N,cache.basis.d;sectors)
        wide_cache=OneBodyGeometry(restricted,W)
        WX=eltype(X)<:Complex ? Complex{W} : W
        WY=eltype(Y)<:Complex ? Complex{W} : W
        Xwide=Matrix{WX}(X);Ywide=Matrix{WY}(Y)
        value,absolute_sum=_local_kernel_value_and_absolute(
            wide_cache,Xwide,Ywide,l,a,b,n,c,d)
        # With no nonzero summand this is a structural zero, independent of
        # floating accumulation. Nonempty near-cancellations are never given
        # the same treatment merely because they fall inside an error bound.
        iszero(absolute_sum)&&return zero(S)
        operation_count=_local_kernel_operation_count(
            wide_cache,l,a,b,n,c,d)
        if abs(value)<=W(64)*W(operation_count)*eps(W)*absolute_sum
            throw(ArgumentError(
                "local one-body kernel element remains indistinguishable from " *
                "zero after guarded-wide accumulation; use a wider " *
                "operator/geometry scalar type"))
        end
        _convert_checked_geometry_value(
            S,value;context="local one-body kernel element")
    end
    if W===BigFloat
        required=max(precision(BigFloat),
            2ndigits(big(max(cache.basis.N,1));base=2)+32)
        return setprecision(compute,required)
    end
    compute()
end

"""
    local_kernel_element(cache, X, Y, l, a, b, n, c, d)

Return the PI-coordinate matrix element from input block coordinate
``(n,c,d)`` to output coordinate ``(l,a,b)`` of the local map
``rho -> sum_i X_i rho Y_i^dagger``. Schur-block indices are one-based.
"""
function local_kernel_element(cache::OneBodyGeometry{T},X,Y,l::Partition,a,b,n::Partition,c,d) where T
    B=cache.basis;li=_sidx(B,l);ni=_sidx(B,n)
    S=promote_type(T,eltype(X),eltype(Y))
    if !_needs_wide_collective(B,T)
        z=zero(S)
        for mu in cache.connections[(li,ni)]
            key=(li,mu,ni);arr=cache.contractions[key]
            z+=cache.scales[key]*_contract(arr[a,c],X)*
                conj(_contract(arr[b,d],Y))
        end
        return z
    end
    z,absolute_sum=_local_kernel_value_and_absolute(
        cache,X,Y,l,a,b,n,c,d)
    operations=_local_kernel_operation_count(cache,l,a,b,n,c,d)
    R=_real_float_type(S)
    uncertified=!isfinite(z)||!isfinite(absolute_sum)||
        (!iszero(absolute_sum)&&(
            absolute_sum>R(8)*abs(z)||
            R(64)*R(operations)*R(eps(T))*absolute_sum>=
                (iszero(z) ? nextfloat(zero(R)) : eps(abs(z)))/R(4)))
    # Julia 1.10 does not infer the guarded-wide closure's scalar return type
    # through this data-dependent branch.  Both paths are contractually `S`;
    # keep that contract explicit without converting or widening the fast path.
    (uncertified ? _wide_local_kernel_element(
        cache,X,Y,l,a,b,n,c,d,S) : z)::S
end

function _collective_geometry(b::PIBasis,::Type{T},cache) where
        T<:AbstractFloat
    if cache===nothing
        if _has_single_fully_symmetric_sector(b)&&
           !_needs_wide_collective(b,T)
            return _SymmetricCollectiveGeometry(b,T)
        end
        return OneBodyGeometry(b,T)
    end
    _check_geometry_basis(cache,b)
end

"""
    collective_block(basis, X, partition; cache=nothing)

Return the physical Schur block in `partition` of the collective one-body
operator ``sum_i X_i``. A sole fully symmetric sector uses a lightweight
occupation-number lift. General Schur sectors use `OneBodyGeometry`; pass a
shared cache when evaluating several blocks or operators on the same exact
basis.
"""
function collective_block(b::PIBasis,X,p::Partition;cache=nothing)
    cache=_collective_geometry(b,Float64,cache)
    cache isa _SymmetricCollectiveGeometry&&
        return _symmetric_collective_block(b,X,p,cache)
    T=geometry_scalar_type(cache)
    S=promote_type(Complex{T},eltype(X))
    if _needs_wide_collective(b,T)
        W=_collective_wider_type(T,b.N)
        return _collective_block_wide(b,X,p,W,S)
    end
    K=_collective_block_fast(b,X,p,cache)
    all(isfinite,K)||throw(ArgumentError(
        "collective Schur block is outside the finite range of $S; "*
        "use a wider operator/geometry scalar type"))
    K
end

"""
    collective_operator(basis, X; cache=nothing)

Construct the PI operator representing ``sum_i X_i`` without forming a
``d^N`` computational-space matrix. A sole fully symmetric sector uses a
lightweight occupation-number lift; otherwise pass a shared `OneBodyGeometry`
to amortize general Schur recoupling setup.
"""
function collective_operator(b::PIBasis,X;cache=nothing)
    cache=_collective_geometry(b,Float64,cache)
    T=promote_type(geometry_scalar_type(cache),_real_float_type(eltype(X)))
    a=PIOperator(b;T=T)
    for p in b.sectors
        C=coefficient_block(a,p)
        B=collective_block(b,X,p;cache=cache)
        scale=try
            _schur_multiplicity_scale(T,p)
        catch error
            error isa ArgumentError||rethrow()
            nothing
        end
        if scale===nothing
            f=symmetric_group_dimension(p)
            C.=_checked_mul_sqrt_exact_ratio(
                B,f,one(f);
                context="stored collective-operator coefficients in sector $p")
        else
            C.=scale.*B
        end
        all(isfinite,C)||throw(ArgumentError(
            "stored collective-operator coefficients in sector $p are "*
            "outside the finite range of Complex{$T}; use a wider scalar type"))
    end
    a
end

function _mean_local_coefficient_block_wide(b::PIBasis,X,p::Partition,
        ::Type{W},::Type{S}) where {W<:AbstractFloat,S}
    compute=function ()
        restricted=PIBasis(b.N,b.d;sectors=[p.parts])
        wide_cache=OneBodyGeometry(restricted,W)
        extensive=_collective_block_fast(restricted,X,p,wide_cache)
        multiplicity=symmetric_group_dimension(p)
        denominator=big(b.N)^2
        coefficient=_checked_mul_sqrt_exact_ratio(
            W,extensive,multiplicity,denominator;
            context="mean-local stored Schur coefficients")
        result=Matrix{S}(undef,size(coefficient)...)
        for index in eachindex(result,coefficient)
            result[index]=_convert_checked_geometry_value(
                S,coefficient[index];
                context="mean-local stored Schur coefficient")
        end
        result
    end
    if W===BigFloat
        required=max(precision(BigFloat),
            2ndigits(big(max(b.N,1));base=2)+32)
        return setprecision(compute,required)
    end
    compute()
end

"""
    mean_local_operator(basis, X; cache=nothing)

Construct the particle-averaged one-body observable
``(1/N) sum_i X_i``. Its equation-(7) coefficients are scaled directly by
``sqrt(f^nu)/N``; the extensive stored collective operator is never formed.
The sole fully symmetric sector uses the occupation-number lift.
"""
function mean_local_operator(b::PIBasis,X;cache=nothing)
    b.N>0||throw(ArgumentError(
        "the particle-averaged local operator requires N > 0"))
    cache=_collective_geometry(b,Float64,cache)
    T=promote_type(geometry_scalar_type(cache),_real_float_type(eltype(X)))
    S=Complex{T}
    result=PIOperator(b;T=T)
    needs_wide=_needs_wide_collective(b,geometry_scalar_type(cache))
    W=needs_wide ? _collective_wider_type(
        geometry_scalar_type(cache),b.N) : geometry_scalar_type(cache)
    for p in b.sectors
        destination=coefficient_block(result,p)
        if needs_wide
            destination.=_mean_local_coefficient_block_wide(b,X,p,W,S)
        else
            extensive=cache isa _SymmetricCollectiveGeometry ?
                _symmetric_collective_block(b,X,p,cache) :
                _collective_block_fast(b,X,p,cache)
            multiplicity=symmetric_group_dimension(p)
            multiplicity_scale=try
                _checked_sqrt_exact_integer(T,multiplicity;
                    context="square root of the sector multiplicity for $p")
            catch error
                error isa ArgumentError||rethrow()
                nothing
            end
            if multiplicity_scale!==nothing
                direct_scale=multiplicity_scale/T(b.N)
                direct=direct_scale.*extensive
                if _ordinary_scaled_value_safe(direct,extensive)
                    destination.=direct
                    continue
                end
            end
            destination.=_checked_mul_sqrt_exact_ratio(
                T,extensive,multiplicity,big(b.N)^2;
                context="mean-local stored Schur coefficients in sector $p")
        end
    end
    result
end
