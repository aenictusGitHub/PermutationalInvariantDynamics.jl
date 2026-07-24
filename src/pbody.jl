struct _PackedPathIsometry{T,Ti<:Integer} <: AbstractArray{T,3}
    data::SparseMatrixCSC{T,Ti}
    endpoint_dimension::Int
    centre_dimension::Int
    word_count::Int
end

Base.size(U::_PackedPathIsometry)=
    (U.endpoint_dimension,U.centre_dimension,U.word_count)
Base.axes(U::_PackedPathIsometry)=map(Base.OneTo,size(U))
Base.IndexStyle(::Type{<:_PackedPathIsometry})=IndexCartesian()
@inline function Base.getindex(U::_PackedPathIsometry,row::Int,
                               centre::Int,word::Int)
    @boundscheck checkbounds(U,row,centre,word)
    @inbounds U.data[row,centre+(word-1)*U.centre_dimension]
end

function Base.Array(U::_PackedPathIsometry{T}) where T
    reshape(Matrix(U.data),size(U))
end

"""
    PBodyGeometry(basis, p; T=Float64, coefficient_cache=nothing)

Cache Appendix-D removal paths, exact path weights, and successive one-box CG
isometries for symmetric `p`-body processes. The retained path isometries store
only exact nonzero support in CSC form over the combined
`(centre pattern, local word)` column. Pass a basis-owned
[`OneBoxCGCache`](@ref) whose `max_depth >= p` through `coefficient_cache` to
reuse precomputed sparse coefficients. Without it, construction still uses
content-indexed compatible candidates and shares each one-box edge within the
geometry; it never evaluates the dense Cartesian set of structural zeros.
"""
struct PBodyGeometry{T,D,L,B<:PIBasis{D,L}}
    basis::B
    p::Int
    paths::Dict{Partition{D},Vector{Vector{Partition{D}}}}
    isometries::Dict{
        Tuple{Vararg{Partition{D}}},_PackedPathIsometry{T,Int}}
    path_weights::Dict{Tuple{Vararg{Partition{D}}},Rational{BigInt}}
    estimates::NamedTuple
end
geometry_scalar_type(::PBodyGeometry{T}) where T=T

function _removal_paths(endpoint::Partition{D},p::Int) where D
    p==0&&return [[endpoint]]
    out=Vector{Vector{Partition{D}}}()
    for r in removable_corners(endpoint),tail in _removal_paths(remove_corner(endpoint,r),p-1)
        push!(out,vcat(tail,[endpoint])) # center -> ... -> endpoint
    end
    out
end

_path_isometry(path::AbstractVector{<:Partition};T=Float64)=
    _path_isometry(path,T)

function _pbody_word_count(local_dimension::Int,p::Int)
    local_dimension>0||throw(ArgumentError(
        "the p-body local dimension must be positive"))
    p>=0||throw(ArgumentError("the p-body order must be nonnegative"))
    words=1
    for _ in 1:p
        words=try
            Base.checked_mul(words,local_dimension)
        catch error
            error isa OverflowError||rethrow()
            throw(ArgumentError(
                "the local p-body tensor dimension exceeds Int indexing"))
        end
    end
    words
end

function _pbody_combined_columns(centre_dimension::Int,words::Int)
    centre_dimension>=0||throw(ArgumentError(
        "the p-body centre dimension must be nonnegative"))
    words>=0||throw(ArgumentError(
        "the p-body word count must be nonnegative"))
    try
        Base.checked_mul(centre_dimension,words)
    catch error
        error isa OverflowError||rethrow()
        throw(ArgumentError(
            "the packed p-body column dimension exceeds Int indexing; "*
            "reduce the retained basis or the p-body order"))
    end
end

# Coefficient-free structural bound used by guarded high-level compilation.
# It follows only the Young-lattice removal graph and exact U(d) irrep
# dimensions.  Treating every dense-equivalent path-isometry entry as a
# possible CSC nonzero is conservative; importantly, it is still a sum over
# retained removal paths rather than a PI-coordinate-squared bound.
function _estimate_pbody_geometry(
        b::PIBasis{D,L},p::Integer,::Type{R}=Float64;
        bigfloat_precision::Integer=precision(BigFloat)) where
        {D,L,R<:AbstractFloat}
    1<=p<=b.N||throw(ArgumentError("p must satisfy 1 ≤ p ≤ N"))
    order=Int(p)
    levels,_,edges=_onebox_domain(b,order)
    dimensions=Dict(
        partition=>unitary_group_dimension(partition)
        for level in levels for partition in level)
    words=big(D)^order
    pattern_cache=Dict{Partition{D},Vector{GTPattern{D,L}}}()
    content_cache=Dict{Partition{D},Dict{NTuple{D,Int},BigInt}}()
    function content_histogram(partition)
        get!(content_cache,partition) do
            sector_index=get(b.index,partition,0)
            patterns=sector_index==0 ?
                get!(() -> gt_patterns(partition),pattern_cache,partition) :
                b.patterns[sector_index]
            histogram=Dict{NTuple{D,Int},BigInt}()
            for pattern in patterns
                key=content(pattern)
                histogram[key]=get(histogram,key,big(0))+1
            end
            histogram
        end
    end
    function support_upper(endpoint,centre)
        result=big(0)
        endpoint_histogram=content_histogram(endpoint)
        centre_histogram=content_histogram(centre)
        for (centre_content,centre_count) in centre_histogram,
            (endpoint_content,endpoint_count) in endpoint_histogram
            difference=ntuple(
                index->endpoint_content[index]-centre_content[index],D)
            any(value->value<0,difference)&&continue
            sum(difference)==order||continue
            result+=centre_count*endpoint_count*
                exact_multinomial(difference)
        end
        result
    end
    path_count=big(0)
    dense_entries=big(0)
    retained_entries_upper=big(0)
    column_pointer_entries=big(0)
    largest_path_entries=big(0)
    largest_centre_dimension=big(0)
    for endpoint in b.sectors
        counts=Dict{Partition{D},BigInt}(endpoint=>big(1))
        for _ in 1:order
            next=Dict{Partition{D},BigInt}()
            for (upper,count) in counts
                for corner in removable_corners(upper)
                    lower=remove_corner(upper,corner)
                    next[lower]=get(next,lower,big(0))+count
                end
            end
            counts=next
        end
        endpoint_dimension=dimensions[endpoint]
        for (centre,multiplicity) in counts
            centre_dimension=dimensions[centre]
            path_entries=endpoint_dimension*centre_dimension*words
            path_support=support_upper(endpoint,centre)
            path_count+=multiplicity
            dense_entries+=multiplicity*path_entries
            retained_entries_upper+=multiplicity*path_support
            column_pointer_entries+=
                multiplicity*(centre_dimension*words+1)
            largest_path_entries=max(largest_path_entries,path_support)
            largest_centre_dimension=max(
                largest_centre_dimension,centre_dimension)
        end
    end

    int_bytes=big(sizeof(Int))
    pointer_bytes=big(sizeof(Ptr{Cvoid}))
    header=8int_bytes
    dictionary_entry=16int_bytes
    scalar_bytes=_scalar_retained_bytes(
        R;bigfloat_precision)
    partition_bytes=big(D)*int_bytes
    path_partition_entries=path_count*big(order+1)

    sparse_payload=retained_entries_upper*(scalar_bytes+int_bytes)+
        column_pointer_entries*int_bytes
    # Three dictionaries (`paths`, `isometries`, and exact path weights), the
    # path vectors and tuple keys, and exact-rational payload.  The exact
    # payload bound grows with every branch factor rather than assuming
    # machine-sized integers.
    branch_bits=big(order)*big(D)*
        max(big(1),ndigits(big(b.N+D+1);base=2))
    exact_weight_bytes=path_count*(8header+2cld(branch_bits,8))
    path_metadata=path_count*(
        8header+3dictionary_entry+2pointer_bytes)+
        path_partition_entries*partition_bytes+
        big(length(b.sectors))*(4header+dictionary_entry)
    retained_bytes=16header+sparse_payload+exact_weight_bytes+path_metadata

    transition_entries=sum(
        (big(D)*dimensions[lower]*dimensions[upper]
         for (lower,upper) in edges);init=big(0))
    transition_bytes=transition_entries*scalar_bytes+
        big(length(edges))*big(D)*(header+pointer_bytes)
    partition_count=sum(length,levels;init=0)
    pattern_count=sum(
        (dimensions[partition] for level in levels for partition in level);
        init=big(0))
    pattern_bytes=pattern_count*(big(L)*int_bytes+4int_bytes)+
        big(partition_count)*(header+dictionary_entry+partition_bytes)
    maximum_dimension=maximum(values(dimensions);init=big(0))
    propagation_bytes=
        2maximum_dimension*largest_centre_dimension*scalar_bytes
    # While one path is packed, its COO triplets coexist with the already
    # retained CSC maps and the dense propagation buffers. Sparse assembly may
    # transiently duplicate those triplets, hence the factor two.
    packing_bytes=2largest_path_entries*(
        scalar_bytes+2int_bytes)+8header
    setup_bytes=retained_bytes+transition_bytes+pattern_bytes+
        propagation_bytes+packing_bytes
    (;retained_bytes,setup_bytes,path_count,dense_entries,
      retained_entries_upper,column_pointer_entries,
      transition_entries,pattern_count,words,scalar_type=R,
      scalar_retained_bytes=scalar_bytes,
      scalar_storage_estimate=_scalar_storage_estimate(R),
      bigfloat_precision_assumption=
          _scalar_precision_assumption(R,bigfloat_precision),
      estimate=:conservative_structural_upper_bound)
end

# One path is the product of p one-box CG maps.  The older implementation
# rebuilt dictionaries of GT-pattern amplitudes separately for every centre
# pattern and every local word.  Besides hashing immutable patterns in the
# innermost loop, that allocated hundreds of times more memory than the final
# isometry.  Cache every edge map once and propagate all centre columns at
# once through two reusable dense buffers instead.
function _pbody_pattern_table!(cache,partition)
    get!(()->gt_patterns(partition),cache,partition)
end

function _pbody_edge_transitions!(transition_cache,pattern_cache,
        lower::Partition{D},upper::Partition{D},::Type{R},
        coefficient_cache=nothing) where
        {D,R<:AbstractFloat}
    key=(lower,upper)
    get!(transition_cache,key) do
        lower_patterns=_pbody_pattern_table!(pattern_cache,lower)
        upper_patterns=_pbody_pattern_table!(pattern_cache,upper)
        maps=[zeros(R,length(upper_patterns),length(lower_patterns))
              for _ in 1:D]
        table=if coefficient_cache===nothing
            _build_onebox_transitions(lower_patterns,upper_patterns,R)
        else
            cached=get(coefficient_cache.transitions,(lower,upper),nothing)
            cached===nothing&&throw(ArgumentError(
                "OneBoxCGCache does not contain required transition "*
                "$lower -> $upper"))
            cached
        end
        @inbounds for lower_index in eachindex(lower_patterns)
            for term_index in
                    table.offsets[lower_index]:(table.offsets[lower_index+1]-1)
                upper_index,local_label,value=table.terms[term_index]
                maps[local_label][upper_index,lower_index]=value
            end
        end
        maps
    end
end


function _path_isometry(path::AbstractVector{Partition{D}},::Type{R},
        pattern_cache,transition_cache,coefficient_cache=nothing) where
        {D,R<:AbstractFloat}
    p=length(path)-1
    p>=0||throw(ArgumentError("a p-body path must contain at least one partition"))
    patterns=[_pbody_pattern_table!(pattern_cache,partition) for partition in path]
    dimensions=map(length,patterns)
    local_dimension=D
    words=_pbody_word_count(local_dimension,p)
    centre_dimension=first(dimensions)
    endpoint_dimension=last(dimensions)
    U=zeros(R,endpoint_dimension,centre_dimension,words)
    p==0&&begin
        @inbounds for index in 1:centre_dimension
            U[index,index,1]=one(R)
        end
        return U
    end
    transitions=[_pbody_edge_transitions!(transition_cache,pattern_cache,
        path[stage],path[stage+1],R,coefficient_cache) for stage in 1:p]
    maximum_dimension=maximum(dimensions)
    first_buffer=zeros(R,maximum_dimension,centre_dimension)
    second_buffer=similar(first_buffer)
    powers=Vector{Int}(undef,p)
    power=1
    for stage in 1:p
        powers[stage]=power
        stage<p&&(power=Base.checked_mul(power,local_dimension))
    end
    for word in 0:words-1
        current=first_buffer;next=second_buffer
        fill!(current,zero(R))
        @inbounds for index in 1:centre_dimension
            current[index,index]=one(R)
        end
        current_dimension=centre_dimension
        for stage in 1:p
            next_dimension=dimensions[stage+1]
            local_label=(word÷powers[stage])%local_dimension
            transition=transitions[stage][local_label+1]
            mul!(view(next,1:next_dimension,1:centre_dimension),transition,
                 view(current,1:current_dimension,1:centre_dimension))
            current,next=next,current
            current_dimension=next_dimension
        end
        copyto!(view(U,:,:,word+1),
                view(current,1:endpoint_dimension,1:centre_dimension))
    end
    U
end

function _packed_path_isometry(path::AbstractVector{Partition{D}},::Type{R},
        pattern_cache,transition_cache,coefficient_cache=nothing) where
        {D,R<:AbstractFloat}
    p=length(path)-1
    p>=0||throw(ArgumentError(
        "a p-body path must contain at least one partition"))
    patterns=[_pbody_pattern_table!(pattern_cache,partition)
              for partition in path]
    dimensions=map(length,patterns)
    local_dimension=D
    words=_pbody_word_count(local_dimension,p)
    centre_dimension=first(dimensions)
    endpoint_dimension=last(dimensions)
    combined_columns=_pbody_combined_columns(centre_dimension,words)
    rows=Int[]
    columns=Int[]
    values=R[]
    if p==0
        sizehint!(rows,centre_dimension)
        sizehint!(columns,centre_dimension)
        sizehint!(values,centre_dimension)
        @inbounds for index in 1:centre_dimension
            push!(rows,index)
            push!(columns,index)
            push!(values,one(R))
        end
        data=sparse(rows,columns,values,endpoint_dimension,centre_dimension)
        return _PackedPathIsometry(
            data,endpoint_dimension,centre_dimension,words)
    end
    transitions=[_pbody_edge_transitions!(transition_cache,pattern_cache,
        path[stage],path[stage+1],R,coefficient_cache) for stage in 1:p]
    maximum_dimension=maximum(dimensions)
    first_buffer=zeros(R,maximum_dimension,centre_dimension)
    second_buffer=similar(first_buffer)
    powers=Vector{Int}(undef,p)
    power=1
    for stage in 1:p
        powers[stage]=power
        stage<p&&(power=Base.checked_mul(power,local_dimension))
    end
    for word in 0:words-1
        current=first_buffer
        next=second_buffer
        fill!(current,zero(R))
        @inbounds for index in 1:centre_dimension
            current[index,index]=one(R)
        end
        current_dimension=centre_dimension
        for stage in 1:p
            next_dimension=dimensions[stage+1]
            local_label=(word÷powers[stage])%local_dimension
            transition=transitions[stage][local_label+1]
            mul!(view(next,1:next_dimension,1:centre_dimension),transition,
                 view(current,1:current_dimension,1:centre_dimension))
            current,next=next,current
            current_dimension=next_dimension
        end
        column_offset=Base.checked_mul(word,centre_dimension)
        @inbounds for centre in 1:centre_dimension,
                      row in 1:endpoint_dimension
            value=current[row,centre]
            iszero(value)&&continue
            push!(rows,row)
            push!(columns,column_offset+centre)
            push!(values,value)
        end
    end
    data=sparse(rows,columns,values,endpoint_dimension,
                combined_columns)
    _PackedPathIsometry(data,endpoint_dimension,centre_dimension,words)
end

function _pbody_geometry_estimates(isometries)
    retained_entries=sum((BigInt(nnz(U.data)) for U in values(isometries));
                         init=big(0))
    dense_entries=sum((prod(BigInt.(size(U))) for U in values(isometries));
                      init=big(0))
    retained_bytes=BigInt(Base.summarysize(isometries))
    scalar_type=isempty(isometries) ? Float64 :
        eltype(first(values(isometries)))
    precision_bits=if scalar_type===BigFloat
        maximum((precision(value) for U in values(isometries)
                 for value in U.data.nzval);init=precision(BigFloat))
    else
        precision(scalar_type)
    end
    dense_payload_bytes=BigInt(dense_entries)*
        _scalar_retained_bytes(
            scalar_type;bigfloat_precision=precision_bits)
    (;storage=:exact_support_sparse_csc,
      retained_entries,
      dense_entries,
      retained_bytes,
      dense_payload_bytes,
      scalar_type,
      precision_bits)
end


function _path_isometry(path::AbstractVector{Partition{D}},::Type{R}) where
        {D,R<:AbstractFloat}
    L=D*(D+1)÷2
    pattern_cache=Dict{Partition{D},Vector{GTPattern{D,L}}}()
    transition_cache=Dict{Tuple{Partition{D},Partition{D}},Vector{Matrix{R}}}()
    _path_isometry(path,R,pattern_cache,transition_cache)
end

PBodyGeometry(b::PIBasis,p::Integer;T=Float64,coefficient_cache=nothing)=
    PBodyGeometry(b,p,T;coefficient_cache)
function PBodyGeometry(b::B,p::Integer,::Type{R};coefficient_cache=nothing) where
        {D,L,B<:PIBasis{D,L},R<:AbstractFloat}
    1<=p<=b.N||throw(ArgumentError("p must satisfy 1 ≤ p ≤ N"))
    coefficient_cache===nothing||_check_onebox_coefficient_cache(
        coefficient_cache,b,Int(p),R)
    paths=Dict{Partition{D},Vector{Vector{Partition{D}}}}(
        q=>_removal_paths(q,Int(p)) for q in b.sectors)
    Iso=_PackedPathIsometry{R,Int}
    iso=Dict{Tuple{Vararg{Partition{D}}},Iso}()
    pattern_cache=coefficient_cache===nothing ?
        Dict{Partition{D},Vector{GTPattern{D,L}}}() :
        copy(coefficient_cache.patterns)
    transition_cache=Dict{Tuple{Partition{D},Partition{D}},Vector{Matrix{R}}}()
    for ps in values(paths),path in ps
        iso[Tuple(path)]=_packed_path_isometry(
            path,R,pattern_cache,transition_cache,coefficient_cache)
    end
    path_weights=Dict{Tuple{Vararg{Partition{D}}},Rational{BigInt}}()
    for ps in values(paths),path in ps
        path_weights[Tuple(path)]=_subset_path_weight(path)
    end
    estimates=_pbody_geometry_estimates(iso)
    PBodyGeometry{R,D,L,B}(
        b,Int(p),paths,iso,path_weights,estimates)
end

function _check_pbody_geometry(cache::PBodyGeometry,b::PIBasis,p::Integer)
    cache.basis===b||throw(ArgumentError("PBodyGeometry was constructed for a different PIBasis; construct or reuse a cache owned by this exact basis"))
    cache.p==p||throw(ArgumentError("PBodyGeometry has p=$(cache.p), but the requested process has p=$p"))
    cache
end

function _check_pbody_operator(cache::PBodyGeometry,X)
    dp=_pbody_word_count(cache.basis.d,cache.p)
    size(X)==(dp,dp)||throw(DimensionMismatch("p-body operator must be $dp×$dp"))
    d=cache.basis.d;p=cache.p
    for k in 1:p-1
        R=_real_float_type(eltype(X));perm=_tensor_swap_permutation(p,d,k);maxdiff=zero(R)
        for j in 1:dp,i in 1:dp;maxdiff=max(maxdiff,abs(X[i,j]-X[perm[i],perm[j]]));end
        maxdiff<=R(1e-10)*max(norm(X,Inf),one(R))||throw(ArgumentError("p-body operator must be invariant under permutations of its p particles"))
    end
    nothing
end

function _path_contractions(UL::AbstractArray{T,3},UR::AbstractArray{T,3},X) where T
    dl,dc,dp=size(UL);dr,dc2,dp2=size(UR);dc==dc2&&dp==dp2||throw(DimensionMismatch())
    A=zeros(promote_type(Complex{T},eltype(X)),dl,dr)
    for a in 1:dl,c in 1:dr,w in 1:dc,i in 1:dp,j in 1:dp
        A[a,c]+=UL[a,w,i]*UR[c,w,j]*X[i,j]
    end
    A
end

function _path_contractions(
        UL::_PackedPathIsometry{T},UR::_PackedPathIsometry{T},X) where T
    dl,dc,dp=size(UL)
    dr,dc2,dp2=size(UR)
    dc==dc2&&dp==dp2||throw(DimensionMismatch())
    A=zeros(promote_type(Complex{T},eltype(X)),dl,dr)
    left=UL.data
    right=UR.data
    @inbounds for w in 1:dc,j in 1:dp
        right_column=w+(j-1)*dc
        for right_pointer in nzrange(right,right_column)
            output_column=right.rowval[right_pointer]
            right_value=right.nzval[right_pointer]
            for i in 1:dp
                x=X[i,j]
                iszero(x)&&continue
                left_column=w+(i-1)*dc
                for left_pointer in nzrange(left,left_column)
                    output_row=left.rowval[left_pointer]
                    A[output_row,output_column]+=
                        left.nzval[left_pointer]*right_value*x
                end
            end
        end
    end
    A
end

function _pbody_cancellation_possible(cache::PBodyGeometry,lambda::Partition,
                                      ::Type{R}) where R<:AbstractFloat
    R===BigFloat&&return false
    threshold=Rational{BigInt}(inv(sqrt(eps(R))))
    any(path->cache.path_weights[Tuple(path)]>threshold,cache.paths[lambda])
end

_pbody_kernel_cancellation_possible(cache::PBodyGeometry,l::Partition,
    n::Partition,::Type{R}) where R<:AbstractFloat =
    _pbody_cancellation_possible(cache,l,R)||
    _pbody_cancellation_possible(cache,n,R)

function _pbody_kernel_guard_bits(cache::PBodyGeometry,l::Partition,
                                  n::Partition,::Type{R}) where R<:AbstractFloat
    scale_bits=0
    for sector in (l,n),path in cache.paths[sector]
        scale=cache.path_weights[Tuple(path)]
        # This is a conservative upper bound on the binary digits contributed
        # by sqrt(weight(left)*weight(right)): the geometric mean cannot exceed
        # the largest endpoint path weight.
        bits=max(0,ndigits(numerator(scale);base=2)-
                   ndigits(denominator(scale);base=2)+1)
        scale_bits=max(scale_bits,bits)
    end
    precision(R)+scale_bits+32
end

function _convert_checked_pbody_value(::Type{R},value;
                                      context::AbstractString) where R<:AbstractFloat
    converted=try
        Complex{R}(value)
    catch error
        throw(ArgumentError(
            "$context is not representable in $R; use a wider scalar type: "*
            sprint(showerror,error)))
    end
    isfinite(converted)||throw(ArgumentError(
        "$context is outside the finite range of $R; use a wider scalar type"))
    ((!iszero(real(value))&&iszero(real(converted)))||
     (!iszero(imag(value))&&iszero(imag(converted))))&&throw(ArgumentError(
        "$context is outside the nonzero finite range of $R; use a wider scalar type"))
    if R===Float16||R===Float32||R===Float64
        minimum_exact=Rational{BigInt}(nextfloat(zero(R)))
        maximum_exact=Rational{BigInt}(floatmax(R))
        for (source_component,converted_component) in
            ((real(value),real(converted)),(imag(value),imag(converted)))
            iszero(source_component)&&continue
            magnitude=abs(converted_component)
            (magnitude==nextfloat(zero(R))||magnitude==floatmax(R))||continue
            exact_magnitude=abs(Rational{BigInt}(source_component))
            minimum_exact<=exact_magnitude<=maximum_exact||throw(ArgumentError(
                "$context is outside the nonzero finite range of $R; use a wider scalar type"))
        end
    end
    converted
end

function _convert_checked_pbody_block(::Type{R},block;
                                      context::AbstractString) where R<:AbstractFloat
    output=zeros(Complex{R},size(block))
    @inbounds for index in eachindex(output,block)
        output[index]=_convert_checked_pbody_value(R,block[index];context)
    end
    output
end

function _pbody_pair_contractions(cache::PBodyGeometry,X,Y,l::Partition{D},
                                  n::Partition{D},::Type{W}) where
        {D,W<:AbstractFloat}
    Xwide=Matrix{Complex{W}}(X);Ywide=Matrix{Complex{W}}(Y)
    left_paths=cache.paths[l];right_paths=cache.paths[n]
    L=D*(D+1)÷2
    pattern_cache=Dict{Partition{D},Vector{GTPattern{D,L}}}()
    transition_cache=Dict{Tuple{Partition{D},Partition{D}},Vector{Matrix{W}}}()
    left_isometries=[_path_isometry(path,W,pattern_cache,transition_cache)
                     for path in left_paths]
    right_isometries=l==n ? left_isometries :
        [_path_isometry(path,W,pattern_cache,transition_cache)
         for path in right_paths]
    Pair=Tuple{_PreparedExactScale{W,true},Matrix{Complex{W}},Matrix{Complex{W}}}
    contractions=Pair[]
    for (left_index,left_path) in pairs(left_paths),
        (right_index,right_path) in pairs(right_paths)
        left_path[1]==right_path[1]||continue
        scale_squared=cache.path_weights[Tuple(left_path)]*
            cache.path_weights[Tuple(right_path)]
        scale=_prepare_exact_scale(W,numerator(scale_squared),
            denominator(scale_squared),Val(true);
            context="wide local p-body path-pair contribution")
        UL=left_isometries[left_index];UR=right_isometries[right_index]
        push!(contractions,(scale,_path_contractions(UL,UR,Xwide),
                                  _path_contractions(UL,UR,Ywide)))
    end
    contractions
end

function _pbody_kernel_value_and_absolute(contractions,a,bb,c,d)
    CT=promote_type(eltype(contractions[1][2]),eltype(contractions[1][3]))
    R=_real_float_type(CT);z=zero(CT);absolute_sum=zero(R)
    for (scale,AX,AY) in contractions
        contribution=_apply_prepared_exact_scale_product(
            AX[a,c],conj(AY[bb,d]),scale;
            context="wide local p-body path-pair contribution")
        z+=contribution;absolute_sum+=abs(contribution)
    end
    z,absolute_sum
end

function _wide_pbody_zero_bound(cache::PBodyGeometry,l::Partition,
                                n::Partition,::Type{W},absolute_sum,
                                pair_count::Integer) where W<:AbstractFloat
    primitive_terms=big(1)
    for sector in (l,n),path in cache.paths[sector]
        U=cache.isometries[Tuple(path)]
        primitive_terms=max(primitive_terms,
            big(size(U,2))*big(size(U,3))^2)
    end
    # Each pair contains two contractions, one product, and one accumulation.
    # The deliberately generous constant also covers complex arithmetic. This
    # is a forward roundoff bound, not a certificate that the mathematical sum
    # is zero. A value inside it must therefore raise rather than be clipped.
    operation_bound=big(max(pair_count,1))*(2primitive_terms+16)
    W(64)*W(operation_bound)*eps(W)*absolute_sum
end

function _pbody_native_sum_uncertified(cache::PBodyGeometry,l::Partition,
        n::Partition,::Type{R},value,absolute_sum,pair_count::Integer) where
        R<:AbstractFloat
    (!isfinite(value)||!isfinite(absolute_sum))&&return true
    iszero(absolute_sum)&&return false
    absolute_sum>R(8)*abs(value)&&return true
    # Even without a small final value, large path contributions can leave the
    # native sum uncertain by one or more ulps (for example D[I] at N=10^8).
    # Recompute whenever the conservative forward-error estimate cannot certify
    # the rounded working-precision result.
    roundoff_bound=_wide_pbody_zero_bound(
        cache,l,n,R,absolute_sum,pair_count)
    spacing=iszero(value) ? nextfloat(zero(R)) : eps(abs(value))
    roundoff_bound>=spacing/R(4)
end

function _convert_checked_pbody_kernel_sum(::Type{R},value,absolute_sum,
        cache::PBodyGeometry,l::Partition,n::Partition,pair_count::Integer;
        context::AbstractString) where R<:AbstractFloat
    W=_real_float_type(typeof(value))
    # No nonzero path-pair contribution is a structural zero. Any nonempty
    # sum inside the forward-error interval is unresolved and must raise.
    iszero(absolute_sum)&&return zero(Complex{R})
    if abs(value)<=_wide_pbody_zero_bound(
            cache,l,n,W,absolute_sum,pair_count)
        throw(ArgumentError(
            "$context remains indistinguishable from zero after guarded-wide " *
            "accumulation; use a wider scalar type"))
    end
    _convert_checked_pbody_value(R,value;context)
end

function _wide_pbody_kernel_element(cache::PBodyGeometry,X,Y,l::Partition,
        a::Int,bb::Int,n::Partition,c::Int,d::Int,::Type{R}) where R<:AbstractFloat
    evaluate=function (::Type{W}) where W<:AbstractFloat
        contractions=_pbody_pair_contractions(cache,X,Y,l,n,W)
        value,absolute_sum=_pbody_kernel_value_and_absolute(
            contractions,a,bb,c,d)
        _convert_checked_pbody_kernel_sum(R,value,absolute_sum,cache,l,n,
            length(contractions);context="local p-body kernel element")
    end
    guard_requirement=_pbody_kernel_guard_bits(cache,l,n,R)
    if R===Float16&&guard_requirement<=precision(Float64)
        return evaluate(Float64)
    end
    required=max(precision(BigFloat),256,guard_requirement)
    setprecision(BigFloat,required) do
        evaluate(BigFloat)
    end
end

function _wide_pbody_kernel_group(cache::PBodyGeometry,X,Y,l::Partition,
        n::Partition,::Type{R},native_values,severe_mask) where R<:AbstractFloat
    function evaluate(::Type{W}) where W<:AbstractFloat
        contractions=_pbody_pair_contractions(cache,X,Y,l,n,W)
        nl=length(cache.basis.patterns[_sidx(cache.basis,l)])
        nn=length(cache.basis.patterns[_sidx(cache.basis,n)])
        values=copy(native_values)
        for a in 1:nl,bb in 1:nl,c in 1:nn,d in 1:nn
            row=a+(bb-1)*nl;column=c+(d-1)*nn
            severe_mask[row,column]||continue
            value,absolute_sum=_pbody_kernel_value_and_absolute(
                contractions,a,bb,c,d)
            values[row,column]=_convert_checked_pbody_kernel_sum(
                R,value,absolute_sum,cache,l,n,length(contractions);
                context="local p-body kernel element")
        end
        values
    end
    guard_requirement=_pbody_kernel_guard_bits(cache,l,n,R)
    R===Float16&&guard_requirement<=precision(Float64)&&return evaluate(Float64)
    required=max(precision(BigFloat),256,guard_requirement)
    setprecision(BigFloat,required) do
        evaluate(BigFloat)
    end
end

function _wide_pbody_collective_block(cache::PBodyGeometry,X,
        lambda::Partition{D},::Type{R}) where {D,R<:AbstractFloat}
    W=R===Float16 ? Float64 : BigFloat
    evaluate=() -> begin
        Xwide=Matrix{Complex{W}}(X)
        n=length(cache.basis.patterns[_sidx(cache.basis,lambda)])
        block=zeros(Complex{W},n,n)
        L=D*(D+1)÷2
        pattern_cache=Dict{Partition{D},Vector{GTPattern{D,L}}}()
        transition_cache=
            Dict{Tuple{Partition{D},Partition{D}},Vector{Matrix{W}}}()
        for path in cache.paths[lambda]
            U=_path_isometry(path,W,pattern_cache,transition_cache)
            exact_scale=cache.path_weights[Tuple(path)]
            block .+=_checked_mul_exact_ratio(
                _path_contractions(U,U,Xwide),numerator(exact_scale),
                denominator(exact_scale);
                context="wide collective p-body path contribution")
        end
        _convert_checked_pbody_block(R,block;
            context="collective p-body block")
    end
    if W===BigFloat
        largest_scale_bits=maximum(cache.paths[lambda];init=0) do path
            scale=cache.path_weights[Tuple(path)]
            max(0,ndigits(numerator(scale);base=2)-
                  ndigits(denominator(scale);base=2)+1)
        end
        # Retain the requested output bits after losing as many leading bits as
        # the largest exact path scale can create through cancellation.
        requested_precision=max(precision(BigFloat),256,
            precision(R)+largest_scale_bits+32)
        return setprecision(BigFloat,requested_precision) do
            evaluate()
        end
    end
    evaluate()
end

"""Appendix-D collective sum `sum_{n1<...<np} Xp^(n1,...,np)` in one Schur block."""
function pbody_collective_block(cache::PBodyGeometry{T},X,lambda::Partition;check::Bool=true) where T
    check&&_check_pbody_operator(cache,X);s=_sidx(cache.basis,lambda);n=length(cache.basis.patterns[s]);K=zeros(promote_type(Complex{T},eltype(X)),n,n)
    R=_real_float_type(eltype(K))
    cancellation_check=_pbody_cancellation_possible(cache,lambda,R)
    absolute_sum=cancellation_check ? zeros(R,n,n) : nothing
    for path in cache.paths[lambda]
        exact_scale=cache.path_weights[Tuple(path)]
        U=cache.isometries[Tuple(path)]
        contraction=_path_contractions(U,U,X)
        prepared_scale=_prepare_exact_scale(R,numerator(exact_scale),
            denominator(exact_scale),Val(false);
            context="collective p-body path contribution")
        contribution=try
            prepared_scale.direct ? prepared_scale.factor.*contraction :
                _apply_prepared_exact_scale(contraction,prepared_scale;
                    context="collective p-body path contribution")
        catch error
            error isa ArgumentError||rethrow()
            return _wide_pbody_collective_block(cache,X,lambda,R)
        end
        K .+=contribution
        cancellation_check&&(absolute_sum .+=abs.(contribution))
    end
    if cancellation_check
        severe=any(index->!iszero(absolute_sum[index])&&
            absolute_sum[index]>R(8)*abs(K[index]),
            eachindex(K))
        severe&&return _wide_pbody_collective_block(cache,X,lambda,R)
    end
    K
end
function pbody_collective_block(b::PIBasis,X,p::Integer,lambda::Partition;cache=PBodyGeometry(b,p))
    _check_pbody_geometry(cache,b,p)
    pbody_collective_block(cache,X,lambda)
end

"""Construct the collective sum of a permutation-symmetric `p`-particle operator."""
function pbody_collective_operator(b::PIBasis,X,p::Integer;cache=PBodyGeometry(b,p))
    _check_pbody_geometry(cache,b,p)
    _check_pbody_operator(cache,X)
    T=promote_type(geometry_scalar_type(cache),_real_float_type(eltype(X)));A=PIOperator(b;T=T)
    for q in b.sectors
        coefficient_block(A,q).=_multiply_by_schur_multiplicity_scale(
            pbody_collective_block(cache,X,q;check=false),T,q)
    end
    A
end

"""
    pbody_kernel_element(cache, X, Y, l, a, b, n, c, d)

Return one PI-coordinate matrix element of the Appendix-D local `p`-body map
``rho -> sum_S X_S rho Y_S^dagger``, where `S` runs over unordered
`p`-particle subsets. The input coordinate is ``(n,c,d)`` and the output
coordinate is ``(l,a,b)``. Both `X` and `Y` must be permutation invariant on
their `p` tensor factors.
"""
function pbody_kernel_element(cache::PBodyGeometry{T},X,Y,l::Partition,a::Int,bb::Int,n::Partition,c::Int,d::Int) where T
    _check_pbody_operator(cache,X);_check_pbody_operator(cache,Y)
    CT=promote_type(Complex{T},eltype(X),eltype(Y));Rscale=_real_float_type(CT)
    z=zero(CT)
    cancellation_check=_pbody_kernel_cancellation_possible(cache,l,n,Rscale)
    if !cancellation_check
        for lp in cache.paths[l],rp in cache.paths[n]
            lp[1]==rp[1]||continue
            scale_squared=cache.path_weights[Tuple(lp)]*
                cache.path_weights[Tuple(rp)]
            UL=cache.isometries[Tuple(lp)];UR=cache.isometries[Tuple(rp)]
            AX=_path_contractions(UL,UR,X);AY=_path_contractions(UL,UR,Y)
            scale=_prepare_exact_scale(Rscale,numerator(scale_squared),
                denominator(scale_squared),Val(true);
                context="local p-body path-pair contribution")
            z+=scale.direct ? scale.factor*AX[a,c]*conj(AY[bb,d]) :
                _apply_prepared_exact_scale_product(
                    AX[a,c],conj(AY[bb,d]),scale;
                    context="local p-body path-pair contribution")
        end
        return z
    end
    absolute_sum=zero(Rscale);pair_count=0
    for lp in cache.paths[l],rp in cache.paths[n]
        lp[1]==rp[1]||continue
        scale_squared=cache.path_weights[Tuple(lp)]*cache.path_weights[Tuple(rp)]
        UL=cache.isometries[Tuple(lp)];UR=cache.isometries[Tuple(rp)]
        AX=_path_contractions(UL,UR,X);AY=_path_contractions(UL,UR,Y)
        scale=_prepare_exact_scale(Rscale,numerator(scale_squared),
            denominator(scale_squared),Val(true);
            context="local p-body path-pair contribution")
        contribution=try
            _apply_prepared_exact_scale_product(
                AX[a,c],conj(AY[bb,d]),scale;
                context="local p-body path-pair contribution")
        catch error
            error isa ArgumentError||rethrow()
            cancellation_check||rethrow()
            return _wide_pbody_kernel_element(cache,X,Y,l,a,bb,n,c,d,Rscale)
        end
        z+=contribution
        if cancellation_check
            absolute_sum+=abs(contribution);pair_count+=1
        end
    end
    cancellation_check&&_pbody_native_sum_uncertified(
        cache,l,n,Rscale,z,absolute_sum,pair_count)&&
        return _wide_pbody_kernel_element(cache,X,Y,l,a,bb,n,c,d,Rscale)
    z
end

function pbody_kernel_triplets(cache::PBodyGeometry{T},X,Y) where T
    _check_pbody_operator(cache,X);_check_pbody_operator(cache,Y)
    TX=promote_type(Complex{T},eltype(X));TY=promote_type(Complex{T},eltype(Y))
    CT=promote_type(TX,TY);b=cache.basis;I=Int[];J=Int[];V=CT[]
    for (li,l) in pairs(b.sectors),(ni,n) in pairs(b.sectors)
        isempty(intersect(first.(cache.paths[l]),first.(cache.paths[n])))&&continue
        nl=length(b.patterns[li]);nn=length(b.patterns[ni])
        Rscale=_real_float_type(CT)
        ScaleType=_PreparedExactScale{Rscale,true}
        contractions=Tuple{ScaleType,Matrix{TX},Matrix{TY}}[]
        for lp in cache.paths[l],rp in cache.paths[n]
            lp[1]==rp[1]||continue
            scale_squared=cache.path_weights[Tuple(lp)]*cache.path_weights[Tuple(rp)]
            scale=_prepare_exact_scale(Rscale,numerator(scale_squared),
                denominator(scale_squared),Val(true);
                context="local p-body path-pair contribution")
            UL=cache.isometries[Tuple(lp)];UR=cache.isometries[Tuple(rp)]
            push!(contractions,(scale,_path_contractions(UL,UR,X),_path_contractions(UL,UR,Y)))
        end
        cancellation_check=_pbody_kernel_cancellation_possible(cache,l,n,Rscale)
        if !cancellation_check
            for a in 1:nl,bb in 1:nl,c in 1:nn,d in 1:nn
                z=sum(q->q[1].direct ?
                        q[1].factor*q[2][a,c]*conj(q[3][bb,d]) :
                        _apply_prepared_exact_scale_product(
                            q[2][a,c],conj(q[3][bb,d]),q[1];
                            context="local p-body path-pair contribution"),
                      contractions;init=zero(CT));iszero(z)&&continue
                push!(I,b.offsets[li]+a-1+(bb-1)*nl)
                push!(J,b.offsets[ni]+c-1+(d-1)*nn);push!(V,z)
            end
            continue
        end
        native_values=zeros(CT,nl*nl,nn*nn)
        severe_mask=falses(nl*nl,nn*nn);severe=false
        for a in 1:nl,bb in 1:nl,c in 1:nn,d in 1:nn
            row=a+(bb-1)*nl;column=c+(d-1)*nn
            z=zero(CT);absolute_sum=zero(Rscale);failed=false
            for (scale,AX,AY) in contractions
                contribution=try
                    _apply_prepared_exact_scale_product(
                        AX[a,c],conj(AY[bb,d]),scale;
                        context="local p-body path-pair contribution")
                catch error
                    error isa ArgumentError||rethrow()
                    failed=true;break
                end
                z+=contribution;absolute_sum+=abs(contribution)
            end
            native_values[row,column]=z
            coordinate_severe=failed||_pbody_native_sum_uncertified(
                cache,l,n,Rscale,z,absolute_sum,length(contractions))
            severe_mask[row,column]=coordinate_severe
            severe|=coordinate_severe
        end
        values=severe ? _wide_pbody_kernel_group(
            cache,X,Y,l,n,Rscale,native_values,severe_mask) : native_values
        for row in axes(values,1),column in axes(values,2)
            z=values[row,column];iszero(z)&&continue
            push!(I,b.offsets[li]+row-1);push!(J,b.offsets[ni]+column-1);push!(V,z)
        end
    end
    (;I,J,V)
end

"""Sparse PI-coordinate representation of the Appendix-D superoperator `K_X,Y`."""
function pbody_kernel_operator(b::PIBasis,X,Y,p::Integer;cache=PBodyGeometry(b,p))
    _check_pbody_geometry(cache,b,p)
    t=pbody_kernel_triplets(cache,X,Y);sparse(t.I,t.J,t.V,length(b),length(b))
end
