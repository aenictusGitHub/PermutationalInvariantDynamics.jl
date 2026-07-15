"""
    OneBodyGeometry(basis; T=Float64)

Precompute the one-box Clebsch--Gordan contractions, multiplicity scales, and
sector connections shared by local and collective one-particle operations.
The cache is read-only after construction and is tied to the exact `PIBasis`
object supplied at construction; it may be reused to prepare many operators.
"""
struct OneBodyGeometry{T,D,L,B<:PIBasis{D,L}}
    basis::B
    # (lambda sector, mu partition, nu sector) => matrices indexed (a,c), each a sparse dyadic tuple (i,j,value)
    contractions::Dict{Tuple{Int,Partition{D},Int},Array{Vector{Tuple{Int,Int,T}},2}}
    scales::Dict{Tuple{Int,Partition{D},Int},T}
    connections::Dict{Tuple{Int,Int},Vector{Partition{D}}}
end
OneBodyGeometry(b::PIBasis;T=Float64)=OneBodyGeometry(b,T)

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
            for parent_index in get(parent_by_weight,parent_weight,Int[])
                value=cgc(child,local_label,parent_patterns[parent_index];T=R)
                iszero(value)||push!(transitions[child_index],
                                     (parent_index,local_label+1,value))
            end
        end
    end
    transitions
end

function OneBodyGeometry(b::B,::Type{R}) where {D,L,B<:PIBasis{D,L},R<:AbstractFloat}
    Term=Tuple{Int,Int,R}
    dict=Dict{Tuple{Int,Partition{D},Int},Array{Vector{Term},2}}()
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
            _onebody_transitions(b.patterns[sector_index],child_patterns,R)
    end

    connections=Dict{Tuple{Int,Int},Vector{Partition{D}}}()
    for (li,l) in pairs(b.sectors),(ni,n) in pairs(b.sectors)
        common=Partition{D}[mu for mu in removed[li] if mu in removed[ni]]
        connections[(li,ni)]=common
        for mu in common
            left=transition_cache[(li,mu)];right=transition_cache[(ni,mu)]
            # All structurally empty cells share one empty read-only vector.
            # A private vector is created only when the first term is found.
            empty_terms=Term[]
            arr=fill(empty_terms,length(b.patterns[li]),length(b.patterns[ni]))
            for child_index in eachindex(left)
                for (a,i,x) in left[child_index],(c,j,y) in right[child_index]
                    value=x*y
                    iszero(value)&&continue
                    terms=arr[a,c]
                    if terms===empty_terms
                        terms=Term[]
                        arr[a,c]=terms
                    end
                    push!(terms,(i,j,value))
                end
            end
            key=(li,mu,ni);dict[key]=arr
            left_weight=_one_box_branch_weight(l,mu)
            right_weight=_one_box_branch_weight(n,mu)
            scale_squared=left_weight*right_weight
            scales[key]=_checked_sqrt_exact_ratio(
                R,numerator(scale_squared),denominator(scale_squared);
                context="one-body Schur branching scale")
        end
    end
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
# heap-backed scalar payloads.  Backing-store allowances use eight entries per
# retained candidate; this bounds both the small-vector initial capacity and
# geometric growth.  It is not cumulative garbage-collector allocation volume.
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

    connection_count=big(0);cell_count=big(0);term_upper=big(0)
    for li in eachindex(b.sectors),ni in eachindex(b.sectors)
        for mu in removed[li]
            mu in removed[ni]||continue
            connection_count+=1
            cell_count+=big(length(b.patterns[li]))*length(b.patterns[ni])
            left_counts=candidate_counts[(li,mu)]
            right_counts=candidate_counts[(ni,mu)]
            term_upper+=sum((big(left)*right for (left,right) in
                             zip(left_counts,right_counts));init=big(0))
        end
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
    nonempty_upper=min(cell_count,term_upper)
    sector_pairs=big(length(b.sectors))^2

    # Retained contraction arrays and their per-cell sparse tuple vectors.
    contraction_dictionary=container_header+connection_count*(
        dictionary_entry+2int_bytes+partition_bytes+pointer_bytes)
    contraction_arrays=connection_count*container_header+pointer_bytes*cell_count
    contraction_vectors=container_header*(connection_count+nonempty_upper)+
        initial_capacity*tuple_bytes*term_upper
    scale_dictionary=container_header+connection_count*(
        dictionary_entry+2int_bytes+partition_bytes+value_bytes)
    connection_dictionary=container_header+sector_pairs*(
        dictionary_entry+2int_bytes+container_header)+
        initial_capacity*partition_bytes*connection_count
    retained_bytes=container_header+contraction_dictionary+contraction_arrays+
        contraction_vectors+scale_dictionary+connection_dictionary

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
    removed_storage=container_header+container_header*big(length(b.sectors))+
        initial_capacity*partition_bytes*big(removal_count)
    setup_bytes=retained_bytes+transition_dictionary+transition_tables+
        pattern_dictionary+pattern_storage+parent_weight_scratch+removed_storage
    (sector_pairs=big(length(b.sectors))^2,
     removal_count=big(removal_count),connection_count,cell_count,
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

"""
    collective_block(basis, X, partition; cache=OneBodyGeometry(basis))

Return the physical Schur block in `partition` of the collective one-body
operator ``sum_i X_i``. Reuse `cache` when evaluating several blocks or
operators on the same exact basis.
"""
function collective_block(b::PIBasis,X,p::Partition;cache=OneBodyGeometry(b))
    _check_geometry_basis(cache,b)
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
    collective_operator(basis, X; cache=OneBodyGeometry(basis))

Construct the PI operator representing ``sum_i X_i`` without forming a
``d^N`` computational-space matrix.
"""
function collective_operator(b::PIBasis,X;cache=OneBodyGeometry(b))
    _check_geometry_basis(cache,b)
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
    mean_local_operator(basis, X; cache=OneBodyGeometry(basis))

Construct the particle-averaged one-body observable
``(1/N) sum_i X_i``. Its equation-(7) coefficients are scaled directly by
``sqrt(f^nu)/N``; the extensive stored collective operator is never formed.
"""
function mean_local_operator(b::PIBasis,X;cache=OneBodyGeometry(b))
    b.N>0||throw(ArgumentError(
        "the particle-averaged local operator requires N > 0"))
    _check_geometry_basis(cache,b)
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
            extensive=_collective_block_fast(b,X,p,cache)
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
