# Identical composite supersites and finite-cutoff local pseudomodes.
#
# The exact PI representation treats one physical system together with all of
# its local auxiliaries as one particle.  Consequently the package uses one
# `PIBasis(N, prod(factor_dimensions))`; it never tensors N system Hilbert
# spaces or represents the local modes as independent global PI factors.

"""
    PISupersite(N, factor_dimensions;
                labels=nothing, specifications=nothing,
                sectors=nothing, T=Float64,
                memory_budget=512*1024^2)
    PISupersite(basis, factor_dimensions; kwargs...)

Describe one identical composite particle and construct its exact PI basis.
`factor_dimensions=(d1,d2,...)` follows Julia's
`kron(factor1,factor2,...)` ordering, so the last internal factor is fastest.
The local dimension is `D=prod(factor_dimensions)` and the complete PI
coordinate count is `binomial(N+D^2-1,N)`, not `D^(2N)`.

`labels` selects internal factors in [`lift_supersite_operator`](@ref).
`specifications` may retain immutable factor metadata; ordinary generic
supersites use `nothing`. The default `memory_budget` checks at least the
basis metadata, internal identities, and one complex PI coordinate vector
before returning. Pass `Inf` only as an explicit opt-out.

For identical systems with local bosonic modes, prefer
[`pseudomode_supersite`](@ref), which installs a `:system` factor followed by
one or more [`BosonicPseudomode`](@ref) specifications.
"""
struct PISupersite{R<:AbstractFloat,B<:PIBasis,D,L,S,I,E}
    basis::B
    factor_dimensions::D
    factor_labels::L
    factor_specifications::S
    identities::I
    estimates::E
end

eltype(::PISupersite{R}) where R=Complex{R}

function show(io::IO,site::PISupersite)
    print(io,"PISupersite(N=$(site.basis.N), ",
          "factor_dimensions=$(site.factor_dimensions), ",
          "local_dimension=$(site.basis.d), ",
          "pi_dimension=$(length(site.basis)))")
end

@inline _supersite_isfinite(value::Real)=isfinite(value)
@inline _supersite_isfinite(value::Complex)=
    isfinite(real(value))&&isfinite(imag(value))
@inline _supersite_isfinite(value)=false
@inline _supersite_stored_values(
    value::Union{SparseMatrixCSC,SparseVector})=nonzeros(value)
@inline function _supersite_stored_values(value::AbstractArray)
    issparse(value) ? nonzeros(sparse(value)) : value
end

function _supersite_checked_real(
        ::Type{R},value,context::AbstractString) where R<:AbstractFloat
    value isa Real&&!(value isa Bool)||throw(ArgumentError(
        "$context must be a real number"))
    converted=try
        R(value)
    catch error
        throw(ArgumentError(
            "$context is not representable in $R: " *
            sprint(showerror,error)))
    end
    isfinite(converted)||throw(ArgumentError(
        "$context is outside the finite range of $R; use a wider scalar type"))
    if !iszero(value)&&iszero(converted)
        throw(ArgumentError(
            "$context is outside the nonzero finite range of $R; " *
            "use a wider scalar type"))
    end
    if value isa Integer&&BigInt(converted)!=BigInt(value)
        throw(ArgumentError(
            "$context is not exactly representable in $R; " *
            "pass it in a wider floating type"))
    end
    if (value isa Integer||value isa Rational)&&!iszero(value)&&
            (R===Float16||R===Float32||R===Float64)
        exact=abs(Rational{BigInt}(value))
        minimum_exact=Rational{BigInt}(nextfloat(zero(R)))
        maximum_exact=Rational{BigInt}(floatmax(R))
        minimum_exact<=exact<=maximum_exact||throw(ArgumentError(
            "$context is outside the nonzero finite range of $R; " *
            "use a wider scalar type"))
    end
    converted
end

function _supersite_checked_complex(
        ::Type{R},value,context::AbstractString) where R<:AbstractFloat
    value isa Number&&!(value isa Bool)||throw(ArgumentError(
        "$context must be a number"))
    Complex{R}(
        _supersite_checked_real(R,real(value),"$context real part"),
        _supersite_checked_real(R,imag(value),"$context imaginary part"))
end

function _supersite_checked_product(left,right,context::AbstractString)
    result=left*right
    _supersite_isfinite(result)||throw(ArgumentError(
        "$context is outside the finite range of $(typeof(result)); " *
        "use a wider scalar type"))
    (!iszero(left)&&!iszero(right)&&iszero(result))&&throw(ArgumentError(
        "$context is outside the nonzero finite range of $(typeof(result)); " *
        "use a wider scalar type"))
    result
end

function _supersite_checked_increment(value,context::AbstractString)
    result=value+one(value)
    _supersite_isfinite(result)||throw(ArgumentError(
        "$context is outside the finite range of $(typeof(result)); " *
        "use a wider scalar type"))
    result
end

@inline _supersite_value_precision(::Any)=0
@inline _supersite_value_precision(value::BigFloat)=precision(value)
@inline _supersite_value_precision(value::Complex{BigFloat})=
    max(precision(real(value)),precision(imag(value)))

function _supersite_array_precision(operator::AbstractArray)
    _real_float_type(eltype(operator))===BigFloat||return 0
    values=_supersite_stored_values(operator)
    maximum(_supersite_value_precision,values;init=0)
end

function _supersite_value_or_array_precision(value)
    value isa AbstractArray ? _supersite_array_precision(value) :
                              _supersite_value_precision(value)
end

function _with_supersite_precision(
        f,::Type{R},precision_bits::Integer,
        rounding_mode) where R<:AbstractFloat
    if R===BigFloat
        return setrounding(BigFloat,rounding_mode) do
            setprecision(BigFloat,Int(precision_bits)) do
                f()
            end
        end
    end
    f()
end

@inline function _supersite_rounding_mode(
        site::PISupersite,::Type{R}) where R<:AbstractFloat
    R===BigFloat||return nothing
    site.estimates.rounding_mode===nothing ?
        rounding(BigFloat) : site.estimates.rounding_mode
end

function _supersite_promote_parameter_type(::Type{R},value) where
        R<:AbstractFloat
    component=typeof(real(value))
    component<:Union{Integer,Rational} ? R :
        promote_type(R,_real_float_type(typeof(value)))
end

function _supersite_checked_dimensions(factor_dimensions)
    raw=Tuple(factor_dimensions)
    isempty(raw)&&throw(ArgumentError(
        "a supersite must contain at least one internal factor"))
    dimensions=map(raw) do dimension
        dimension isa Integer&&!(dimension isa Bool)||throw(ArgumentError(
            "every supersite factor dimension must be an integer"))
        dimension>=1||throw(ArgumentError(
            "every supersite factor dimension must be positive"))
        BigInt(dimension)<=typemax(Int)||throw(ArgumentError(
            "a supersite factor dimension exceeds Int indexing"))
        Int(dimension)
    end
    local_dimension=prod(BigInt,dimensions)
    local_dimension<=typemax(Int)||throw(ArgumentError(
        "the supersite local dimension exceeds Int indexing"))
    Tuple(dimensions),Int(local_dimension)
end

function _supersite_labels(labels,count::Int)
    resolved=labels===nothing ?
        ntuple(index->Symbol(:factor,index),count) : Tuple(labels)
    length(resolved)==count||throw(DimensionMismatch(
        "labels must contain one entry per supersite factor"))
    all(label->label isa Symbol,resolved)||throw(ArgumentError(
        "supersite factor labels must be Symbols"))
    length(unique(resolved))==count||throw(ArgumentError(
        "supersite factor labels must be unique"))
    resolved
end

function _supersite_specifications(specifications,count::Int)
    resolved=specifications===nothing ? ntuple(_->nothing,count) :
        Tuple(specifications)
    length(resolved)==count||throw(DimensionMismatch(
        "specifications must contain one entry per supersite factor"))
    resolved
end

function _supersite_sector_geometry(N::Int,d::Int,sectors)
    partitions_checked=if sectors===nothing
        partitions(N,d)
    else
        try
            Partition{d}[Partition(Tuple(sector)) for sector in sectors]
        catch error
            throw(ArgumentError(
                "could not interpret the requested supersite sectors: " *
                sprint(showerror,error)))
        end
    end
    all(partition->weight(partition)==N,partitions_checked)||
        throw(ArgumentError(
        "all supersite sectors must partition N into $d parts"))
    length(unique(partitions_checked))==length(partitions_checked)||
        throw(ArgumentError("duplicate supersite sector"))
    sort!(partitions_checked;by=partition->partition.parts,rev=true)
    dimensions=BigInt[
        unitary_group_dimension(partition)
        for partition in partitions_checked]
    coordinate_dimension=sum(
        (dimension^2 for dimension in dimensions);init=big(0))
    pattern_count=sum(dimensions;init=big(0))
    (;sectors=partitions_checked,coordinate_dimension,pattern_count)
end

function _supersite_basis_preflight_bytes(d::Int,geometry)
    sector_count=BigInt(length(geometry.sectors))
    pattern_entries=BigInt(d)*(d+1)÷2
    pattern_payload=
        geometry.pattern_count*pattern_entries*sizeof(Int)
    partition_payload=sector_count*BigInt(d)*sizeof(Int)
    # Include the stored sector vector, dictionary keys, offsets, outer
    # pattern-vector references, and conservative Julia-array headers. The
    # exact post-construction `summarysize` check remains authoritative.
    structural_payload=
        3partition_payload+
        (geometry.pattern_count+6sector_count+1)*sizeof(Int)+
        256*(sector_count+1)
    pattern_payload+structural_payload
end

function _supersite_sparse_csc_bytes(
        columns::Integer,nonzeros::Integer,::Type{T};
        bigfloat_precision::Integer=precision(BigFloat)) where T
    count=BigInt(nonzeros)
    _performance_entries_bytes(count,T;bigfloat_precision)+
        (count+BigInt(columns)+1)*sizeof(Int)
end

function _supersite_identity_bytes(
        dimensions,::Type{T};
        bigfloat_precision::Integer=precision(BigFloat)) where T
    sum((_supersite_sparse_csc_bytes(
             dimension,dimension,T;bigfloat_precision)
         for dimension in dimensions);
        init=big(0))
end

function PISupersite(
        N::Integer,factor_dimensions;
        labels=nothing,specifications=nothing,sectors=nothing,
        T=Float64,memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    N isa Integer&&!(N isa Bool)||throw(ArgumentError(
        "N must be an integer"))
    N>=0||throw(ArgumentError("N must be nonnegative"))
    BigInt(N)<=typemax(Int)||throw(ArgumentError(
        "N must be representable as an Int"))
    dimensions,local_dimension=
        _supersite_checked_dimensions(factor_dimensions)
    resolved_labels=_supersite_labels(labels,length(dimensions))
    resolved_specifications=
        _supersite_specifications(specifications,length(dimensions))
    R=_real_float_type(T)
    R<:AbstractFloat||throw(ArgumentError(
        "the supersite scalar type must promote to an AbstractFloat"))
    precision_bits=R===BigFloat ? precision(BigFloat) : precision(R)
    rounding_mode=R===BigFloat ? rounding(BigFloat) : nothing
    identity_bytes=_supersite_identity_bytes(
        dimensions,Complex{R};
        bigfloat_precision=precision_bits)
    # For the complete basis, its exact PI coordinate dimension is available
    # without enumerating a single Young sector. Reject an impossible request
    # before `partitions(N,D)` or any GT-pattern metadata is materialized.
    complete_coordinate_dimension=sectors===nothing ?
        commutant_dimension(Int(N),local_dimension) : nothing
    if complete_coordinate_dimension!==nothing
        initial_coordinate_bytes=_performance_entries_bytes(
            complete_coordinate_dimension,Complex{R};
            bigfloat_precision=precision_bits)
        _require_performance_budget(
            "PI supersite construction",
            initial_coordinate_bytes+identity_bytes,memory_budget;
            guidance="Reduce N, a local cutoff, or retain explicit sectors.")
    end
    geometry=_supersite_sector_geometry(
        Int(N),local_dimension,sectors)
    coordinate_dimension=geometry.coordinate_dimension
    complete_coordinate_dimension===nothing||
        coordinate_dimension==complete_coordinate_dimension||
        error("internal complete supersite coordinate-count mismatch")
    coordinate_bytes=_performance_entries_bytes(
        coordinate_dimension,Complex{R};
        bigfloat_precision=precision_bits)
    basis_preflight_bytes=
        _supersite_basis_preflight_bytes(local_dimension,geometry)
    preflight_bytes=
        coordinate_bytes+identity_bytes+basis_preflight_bytes
    _require_performance_budget(
        "PI supersite construction",preflight_bytes,memory_budget;
        guidance="Reduce N, a local cutoff, or the retained sector set.")

    basis=PIBasis(
        Int(N),local_dimension;
        sectors=(partition.parts for partition in geometry.sectors))
    retained_bytes=BigInt(Base.summarysize(basis))+coordinate_bytes+
                   identity_bytes
    _require_performance_budget(
        "PI supersite retained data",retained_bytes,memory_budget;
        guidance="Reduce N, a local cutoff, or the retained sector set.")
    identities=ntuple(length(dimensions)) do index
        dimension=dimensions[index]
        spdiagm(0=>fill(one(Complex{R}),dimension))
    end
    estimates=(pi_dimension=coordinate_dimension,
               coordinate_bytes=coordinate_bytes,
               identity_bytes=identity_bytes,
               basis_preflight_bytes=basis_preflight_bytes,
               basis_bytes=BigInt(Base.summarysize(basis)),
               retained_bytes=retained_bytes,
               precision_bits=precision_bits,
               rounding_mode=rounding_mode,
               memory_budget=memory_budget)
    PISupersite{R,typeof(basis),typeof(dimensions),
        typeof(resolved_labels),typeof(resolved_specifications),
        typeof(identities),typeof(estimates)}(
        basis,dimensions,resolved_labels,resolved_specifications,
        identities,estimates)
end

function PISupersite(
        basis::PIBasis,factor_dimensions;
        labels=nothing,specifications=nothing,T=Float64,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    dimensions,local_dimension=
        _supersite_checked_dimensions(factor_dimensions)
    local_dimension==basis.d||throw(DimensionMismatch(
        "the product of factor_dimensions must equal basis.d=$(basis.d)"))
    resolved_labels=_supersite_labels(labels,length(dimensions))
    resolved_specifications=
        _supersite_specifications(specifications,length(dimensions))
    R=_real_float_type(T)
    R<:AbstractFloat||throw(ArgumentError(
        "the supersite scalar type must promote to an AbstractFloat"))
    precision_bits=R===BigFloat ? precision(BigFloat) : precision(R)
    rounding_mode=R===BigFloat ? rounding(BigFloat) : nothing
    coordinate_bytes=_performance_entries_bytes(
        length(basis),Complex{R};
        bigfloat_precision=precision_bits)
    identity_bytes=_supersite_identity_bytes(
        dimensions,Complex{R};
        bigfloat_precision=precision_bits)
    basis_bytes=BigInt(Base.summarysize(basis))
    retained_bytes=basis_bytes+coordinate_bytes+identity_bytes
    _require_performance_budget(
        "PI supersite retained data",retained_bytes,memory_budget;
        guidance="Reduce N, a local cutoff, or the retained sector set.")
    identities=ntuple(length(dimensions)) do index
        dimension=dimensions[index]
        spdiagm(0=>fill(one(Complex{R}),dimension))
    end
    estimates=(pi_dimension=BigInt(length(basis)),
               coordinate_bytes=coordinate_bytes,
               identity_bytes=identity_bytes,
               basis_bytes=basis_bytes,
               retained_bytes=retained_bytes,
               precision_bits=precision_bits,
               rounding_mode=rounding_mode,
               memory_budget=memory_budget)
    PISupersite{R,typeof(basis),typeof(dimensions),
        typeof(resolved_labels),typeof(resolved_specifications),
        typeof(identities),typeof(estimates)}(
        basis,dimensions,resolved_labels,resolved_specifications,
        identities,estimates)
end

function _supersite_factor_index(site::PISupersite,factor)
    if factor isa Integer&&!(factor isa Bool)
        1<=factor<=length(site.factor_dimensions)||throw(BoundsError(
            site.factor_dimensions,factor))
        return Int(factor)
    end
    factor isa Symbol||throw(ArgumentError(
        "a supersite factor must be selected by integer or Symbol"))
    index=findfirst(isequal(factor),site.factor_labels)
    index===nothing&&throw(ArgumentError(
        "unknown supersite factor $factor; available labels are " *
        join(site.factor_labels,", ")))
    index
end

function _supersite_operator_type(site::PISupersite,operators)
    R=_real_float_type(eltype(site))
    for operator in operators
        operator isa AbstractMatrix||throw(ArgumentError(
            "every supersite component must be a matrix"))
        R=promote_type(R,_real_float_type(eltype(operator)))
    end
    R<:AbstractFloat||throw(ArgumentError(
        "supersite operators must have floating-point-compatible scalars"))
    R
end

@inline _supersite_structural_nnz(operator::AbstractArray)=
    issparse(operator) ? nnz(operator) : count(!iszero,operator)

function _supersite_checked_kron(left,right,context::AbstractString)
    expected=BigInt(_supersite_structural_nnz(left))*
             _supersite_structural_nnz(right)
    result=kron(left,right)
    issparse(result)&&dropzeros!(result)
    values=issparse(result) ? nonzeros(result) : result
    all(_supersite_isfinite,values)||throw(ArgumentError(
        "$context overflowed; use a wider scalar type"))
    BigInt(_supersite_structural_nnz(result))==expected||
        throw(ArgumentError(
        "$context lost a required nonzero product; use a wider scalar type"))
    result
end

function _supersite_checked_outer(ket,context::AbstractString)
    expected=BigInt(_supersite_structural_nnz(ket))^2
    result=ket*ket'
    all(_supersite_isfinite,result)||throw(ArgumentError(
        "$context overflowed; use a wider scalar type"))
    BigInt(_supersite_structural_nnz(result))==expected||
        throw(ArgumentError(
        "$context lost a required nonzero product; use a wider scalar type"))
    result
end

function _supersite_checked_scale(
        rate,operator::AbstractMatrix,context::AbstractString)
    expected=iszero(rate) ? big(0) :
        BigInt(_supersite_structural_nnz(operator))
    result=rate*operator
    issparse(result)&&dropzeros!(result)
    all(_supersite_isfinite,
        _supersite_stored_values(result))||throw(ArgumentError(
        "$context overflowed; use a wider scalar type"))
    BigInt(_supersite_structural_nnz(result))==expected||
        throw(ArgumentError(
        "$context lost a required nonzero product; " *
        "use a wider scalar type"))
    result
end

function _supersite_add_hamiltonian(
        hamiltonian,rate,operator,context::AbstractString)
    result=hamiltonian+
        _supersite_checked_scale(rate,operator,context)
    issparse(result)&&dropzeros!(result)
    all(_supersite_isfinite,
        _supersite_stored_values(result))||throw(ArgumentError(
        "$context produced a nonfinite accumulated Hamiltonian; " *
        "use a wider scalar type"))
    result
end

function _supersite_matrix_storage_bytes(
        dimension::Integer,nonzeros::Integer,sparse_storage::Bool,
        ::Type{T};
        bigfloat_precision::Integer=precision(BigFloat)) where T
    sparse_storage ?
        _supersite_sparse_csc_bytes(
            dimension,nonzeros,T;bigfloat_precision) :
        _performance_entries_bytes(
            BigInt(dimension)^2,T;bigfloat_precision)
end

function _supersite_tensor_peak_bytes(
        dimensions,operators,::Type{T};
        bigfloat_precision::Integer=precision(BigFloat)) where T
    current_dimension=BigInt(first(dimensions))
    current_sparse=issparse(first(operators))
    current_nonzeros=BigInt(
        _supersite_structural_nnz(first(operators)))
    current_bytes=_supersite_matrix_storage_bytes(
        current_dimension,current_nonzeros,current_sparse,T;
        bigfloat_precision)
    peak=current_bytes
    for (dimension,operator) in zip(
            Base.tail(dimensions),Base.tail(operators))
        component_dimension=BigInt(dimension)
        component_sparse=issparse(operator)
        component_nonzeros=BigInt(
            _supersite_structural_nnz(operator))
        component_bytes=_supersite_matrix_storage_bytes(
            component_dimension,component_nonzeros,component_sparse,T;
            bigfloat_precision)
        output_dimension=current_dimension*component_dimension
        output_sparse=current_sparse||component_sparse
        output_nonzeros=current_nonzeros*component_nonzeros
        output_bytes=_supersite_matrix_storage_bytes(
            output_dimension,output_nonzeros,output_sparse,T;
            bigfloat_precision)
        output_peak=output_sparse ? 3output_bytes : output_bytes
        peak=max(peak,current_bytes+component_bytes+output_peak)
        current_dimension=output_dimension
        current_sparse=output_sparse
        current_nonzeros=output_nonzeros
        current_bytes=output_bytes
    end
    peak
end

function _supersite_converted_component(
        operator::AbstractMatrix,::Type{R};
        context::AbstractString="supersite operator") where R<:AbstractFloat
    if issparse(operator)
        source=operator isa SparseMatrixCSC ? operator : sparse(operator)
        source_values=nonzeros(source)
        converted_values=Vector{Complex{R}}(undef,length(source_values))
        @inbounds for index in eachindex(source_values)
            converted_values[index]=_supersite_checked_complex(
                R,source_values[index],"$context entry")
        end
        converted=SparseMatrixCSC(
            size(source,1),size(source,2),copy(source.colptr),
            copy(source.rowval),converted_values)
        dropzeros!(converted)
        return converted
    end
    converted=Matrix{Complex{R}}(undef,size(operator))
    @inbounds for index in eachindex(converted,operator)
        converted[index]=_supersite_checked_complex(
            R,operator[index],"$context entry")
    end
    converted
end

"""
    supersite_tensor_operator(site, components...;
                              memory_budget=512*1024^2)

Form one local operator `kron(components...)` in the internal ordering recorded
by `site`. Exactly one square component is required per factor. This allocates
only the `site.basis.d`-dimensional *local* matrix; it never constructs an
`N`-particle Hilbert-space operator.
"""
function supersite_tensor_operator(
        site::PISupersite,components...;
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    resolved=length(components)==1&&first(components) isa Tuple ?
        first(components) : components
    length(resolved)==length(site.factor_dimensions)||
        throw(DimensionMismatch(
        "one operator is required for every supersite factor"))
    resolved=map(resolved) do operator
        operator isa AbstractArray&&issparse(operator)&&
            !(operator isa Union{SparseMatrixCSC,SparseVector}) ?
            sparse(operator) : operator
    end
    for (index,(operator,dimension)) in
            enumerate(zip(resolved,site.factor_dimensions))
        operator isa AbstractMatrix||throw(ArgumentError(
            "supersite component $index must be a matrix"))
        size(operator)==(dimension,dimension)||throw(DimensionMismatch(
            "supersite component $index must be $dimension×$dimension"))
        values=_supersite_stored_values(operator)
        all(_supersite_isfinite,values)||throw(ArgumentError(
            "supersite component $index contains a nonfinite value"))
    end
    R=_supersite_operator_type(site,resolved)
    # Each Kronecker step briefly retains its previous result, one converted
    # component, and the new result. Track exact dense/sparse structure through
    # the declared factor order so a sparse identity or ladder operator does
    # not receive a fictitious dense D-by-D charge.
    precision_bits=R===BigFloat ?
        max(site.estimates.precision_bits,
            maximum(_supersite_array_precision,resolved;init=0)) :
        precision(R)
    rounding_mode=_supersite_rounding_mode(site,R)
    _with_supersite_precision(R,precision_bits,rounding_mode) do
        estimated=_supersite_tensor_peak_bytes(
            site.factor_dimensions,resolved,Complex{R};
            bigfloat_precision=precision_bits)
        _require_performance_budget(
            "local supersite tensor operator",estimated,memory_budget;
            guidance="Reduce an internal local dimension.")
        result=_supersite_converted_component(
            first(resolved),R;context="supersite component 1")
        for (tail_index,operator) in enumerate(Base.tail(resolved))
            component_index=tail_index+1
            component=_supersite_converted_component(
                operator,R;
                context="supersite component $component_index")
            expected_nonzeros=
                BigInt(_supersite_structural_nnz(result))*
                _supersite_structural_nnz(component)
            result=kron(result,component)
            issparse(result)&&dropzeros!(result)
            all(_supersite_isfinite,
                _supersite_stored_values(result))||
                throw(ArgumentError(
                "the local supersite tensor operator overflowed in " *
                "$R; use a wider scalar type"))
            actual_nonzeros=BigInt(
                _supersite_structural_nnz(result))
            actual_nonzeros==expected_nonzeros||throw(ArgumentError(
                "the local supersite tensor operator lost a required " *
                "nonzero product in $R; use a wider scalar type"))
        end
        result
    end
end

"""
    lift_supersite_operator(site, operator; factor,
                            memory_budget=512*1024^2)

Lift a matrix acting on one internal factor to the whole local supersite,
inserting identities on every other factor. `factor` is an integer or one of
`site.factor_labels`.
"""
function lift_supersite_operator(
        site::PISupersite,operator::AbstractMatrix;
        factor,memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    selected=_supersite_factor_index(site,factor)
    components=ntuple(
        index->index==selected ? operator : site.identities[index],
        length(site.factor_dimensions))
    supersite_tensor_operator(site,components;memory_budget)
end

"""
    lift_system_operator(site, operator; memory_budget=512*1024^2)

Lift a one-system operator to a pseudomode supersite. The system must be the
factor labeled `:system`.
"""
lift_system_operator(
    site::PISupersite,operator::AbstractMatrix;
    memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)=
    lift_supersite_operator(
        site,operator;factor=:system,memory_budget)

function _supersite_checked_power(base::Int,power::Int,context::String)
    result=big(base)^power
    result<=typemax(Int)||throw(ArgumentError(
        "$context exceeds Int indexing"))
    Int(result)
end

@inline function _supersite_decode_equal_base!(
        digits::Vector{Int},encoded::Int,base::Int)
    value=encoded
    @inbounds for index in length(digits):-1:1
        value,remainder=divrem(value,base)
        digits[index]=remainder+1
    end
    iszero(value)||error("internal supersite mixed-radix overflow")
    digits
end

@inline function _supersite_interleaved_index(
        system_digits::Vector{Int},auxiliary_digits::Vector{Int},
        auxiliary_dimension::Int,local_dimension::Int)
    encoded=0
    @inbounds for particle in eachindex(system_digits,auxiliary_digits)
        local_index=(system_digits[particle]-1)*auxiliary_dimension+
                    auxiliary_digits[particle]
        encoded=encoded*local_dimension+(local_index-1)
    end
    encoded+1
end

"""
    lift_system_pbody_operator(site, operator, p;
                               memory_budget=512*1024^2)

Lift a system-only `p`-particle matrix to the paired supersite ordering by
tensoring the identity on every particle's local auxiliaries. The input order
is `system_1 ⊗ ... ⊗ system_p`; the output order is
`(system_1⊗aux_1) ⊗ ... ⊗ (system_p⊗aux_p)`.

The result is sparse and has exactly `nnz(operator)*d_aux^p` structural
entries. This avoids the incorrect grouped ordering produced by
`kron(operator, I_aux^⊗p)` and never constructs an `N`-particle operator.
"""
function lift_system_pbody_operator(
        site::PISupersite,operator::AbstractMatrix,p::Integer;
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    p isa Integer&&!(p isa Bool)||throw(ArgumentError(
        "the body order must be an integer"))
    p>=1||throw(ArgumentError("the body order must be positive"))
    BigInt(p)<=typemax(Int)||throw(ArgumentError(
        "the body order exceeds Int indexing"))
    order=Int(p)
    order<=site.basis.N||throw(ArgumentError(
        "the body order $order exceeds N=$(site.basis.N)"))
    system_factor=_supersite_factor_index(site,:system)
    system_factor==1||throw(ArgumentError(
        "system p-body lifting requires :system to be the first factor"))
    ds=site.factor_dimensions[1]
    system_tensor_dimension=
        _supersite_checked_power(ds,order,"the system p-body dimension")
    size(operator)==(system_tensor_dimension,system_tensor_dimension)||
        throw(DimensionMismatch(
        "the system p-body operator must be " *
        "$system_tensor_dimension×$system_tensor_dimension"))
    sparse_source=issparse(operator) ?
        (operator isa SparseMatrixCSC ? operator : sparse(operator)) :
        nothing
    operator_values=sparse_source===nothing ?
        operator : nonzeros(sparse_source)
    all(_supersite_isfinite,operator_values)||throw(ArgumentError(
        "the system p-body operator contains a nonfinite value"))
    R=promote_type(_real_float_type(eltype(site)),
                   _real_float_type(eltype(operator)))
    precision_bits=R===BigFloat ?
        max(site.estimates.precision_bits,
            _supersite_array_precision(operator)) :
        precision(R)
    auxiliary_dimension=div(site.basis.d,ds)
    auxiliary_configurations=_supersite_checked_power(
        auxiliary_dimension,order,
        "the auxiliary p-body identity dimension")
    output_dimension=_supersite_checked_power(
        site.basis.d,order,"the supersite p-body dimension")
    input_entries=sparse_source!==nothing ?
        count(!iszero,nonzeros(sparse_source)) :
        count(!iszero,operator)
    output_entries=BigInt(input_entries)*auxiliary_configurations
    output_entries<=typemax(Int)||throw(ArgumentError(
        "the lifted system p-body operator has too many sparse entries " *
        "for Int indexing"))
    output_count=Int(output_entries)
    scalar_bytes=_scalar_retained_bytes(
        Complex{R};bigfloat_precision=precision_bits)
    int_bytes=BigInt(sizeof(Int))
    # I, J, V assembly plus the resulting CSC arrays and the converted input.
    estimate=output_entries*(2scalar_bytes+4int_bytes)+
             BigInt(output_dimension+1)*2int_bytes+
             BigInt(input_entries)*(2scalar_bytes+4int_bytes)
    _require_performance_budget(
        "system p-body supersite lifting",estimate,memory_budget;
        guidance="Reduce p, a pseudomode cutoff, or the system-operator support.")

    rounding_mode=_supersite_rounding_mode(site,R)
    _with_supersite_precision(R,precision_bits,rounding_mode) do
        converted=_supersite_converted_component(
            sparse_source===nothing ? operator : sparse_source,R;
            context="system p-body operator")
        converted=SparseMatrixCSC{Complex{R},Int}(converted)
        dropzeros!(converted)
        rows_system,columns_system,values=findnz(converted)
        length(values)==input_entries||error(
            "internal p-body sparse-support count mismatch")
        row_indices=Vector{Int}(undef,output_count)
        column_indices=Vector{Int}(undef,output_count)
        lifted_values=Vector{Complex{R}}(undef,output_count)
        row_digits=Vector{Int}(undef,order)
        column_digits=similar(row_digits)
        auxiliary_digits=similar(row_digits)
        destination=1
        @inbounds for source_index in eachindex(values)
            _supersite_decode_equal_base!(
                row_digits,rows_system[source_index]-1,ds)
            _supersite_decode_equal_base!(
                column_digits,columns_system[source_index]-1,ds)
            for auxiliary_code in 0:auxiliary_configurations-1
                _supersite_decode_equal_base!(
                    auxiliary_digits,auxiliary_code,
                    auxiliary_dimension)
                row_indices[destination]=_supersite_interleaved_index(
                    row_digits,auxiliary_digits,auxiliary_dimension,
                    site.basis.d)
                column_indices[destination]=
                    _supersite_interleaved_index(
                        column_digits,auxiliary_digits,
                        auxiliary_dimension,site.basis.d)
                lifted_values[destination]=values[source_index]
                destination+=1
            end
        end
        sparse(row_indices,column_indices,lifted_values,
               output_dimension,output_dimension)
    end
end

function _supersite_rebuild_term(term::LocalHamiltonian,operator)
    LocalHamiltonian(operator;rate=term.rate,hbar=term.hbar,check=false)
end
function _supersite_rebuild_term(term::CollectiveHamiltonian,operator)
    CollectiveHamiltonian(operator;rate=term.rate,hbar=term.hbar,check=false)
end
_supersite_rebuild_term(term::LocalJump,operator)=
    LocalJump(operator;rate=term.rate)
_supersite_rebuild_term(term::CollectiveJump,operator)=
    CollectiveJump(operator;rate=term.rate)
function _supersite_rebuild_term(term::PBodyHamiltonian,operator)
    PBodyHamiltonian(operator,term.p;rate=term.rate,hbar=term.hbar,
                     check=false)
end
_supersite_rebuild_term(term::LocalPBodyJump,operator)=
    LocalPBodyJump(operator,term.p;rate=term.rate)
_supersite_rebuild_term(term::CollectivePBodyJump,operator)=
    CollectivePBodyJump(operator,term.p;rate=term.rate)

"""
    lift_system_term(site, term; memory_budget=512*1024^2)

Lift a fixed microscopic system term to a paired supersite while preserving
its local/collective scope, body order, rate, and Hamiltonian `hbar`.
One-body and symmetric `p`-body built-in terms are supported. Direct
Schur-space terms and operator schedules are rejected because they do not
define a fixed microscopic system matrix to lift.
"""
function lift_system_term(
        site::PISupersite,term::AbstractPITerm;
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    term isa Union{DirectPIHamiltonian,DirectPIJump}&&throw(ArgumentError(
        "a direct PI term is already basis-specific and cannot be lifted " *
        "from a system basis to a supersite basis"))
    term isa Union{LocalHamiltonian,CollectiveHamiltonian,LocalJump,
                   CollectiveJump,PBodyHamiltonian,LocalPBodyJump,
                   CollectivePBodyJump}||throw(ArgumentError(
        "lift_system_term supports fixed built-in microscopic terms"))
    operator=term_operator(term)
    operator isa AbstractMatrix||throw(ArgumentError(
        "lift_system_term requires a fixed matrix operator; define a " *
        "preallocated schedule directly in the supersite dimension"))
    lifted=body_order(term)==1 ?
        lift_system_operator(site,operator;memory_budget) :
        lift_system_pbody_operator(
            site,operator,body_order(term);memory_budget)
    _supersite_rebuild_term(term,lifted)
end

function _supersite_converted_vector(
        vector::AbstractVector,::Type{R};
        context::AbstractString="supersite state") where R<:AbstractFloat
    converted=Vector{Complex{R}}(undef,length(vector))
    @inbounds for index in eachindex(converted,vector)
        converted[index]=_supersite_checked_complex(
            R,vector[index],"$context entry")
    end
    converted
end

function _supersite_iid_memory_upper(
        site::PISupersite,::Type{R},precision_bits::Int;
        pure::Bool) where R<:AbstractFloat
    basis=site.basis
    local_dimension=BigInt(basis.d)
    output_entries=BigInt(length(basis))
    output_bytes=_performance_entries_bytes(
        output_entries,Complex{R};
        bigfloat_precision=precision_bits)
    if pure
        symmetric=Partition(
            ntuple(index->index==1 ? basis.N : 0,basis.d))
        symmetric_dimension=unitary_group_dimension(symmetric)
        table_entries=BigInt(max(basis.d-1,0))*
            BigInt(basis.N+1)*BigInt(basis.N+2)÷2
        W=_iid_amplitude_work_type(R)
        recurrence_bytes=
            _performance_entries_bytes(
                symmetric_dimension^2+
                3symmetric_dimension+8local_dimension,
                Complex{R};bigfloat_precision=precision_bits)+
            _performance_entries_bytes(
                table_entries+4local_dimension,W;
                bigfloat_precision=precision_bits)
        setup_peak=output_bytes+recurrence_bytes+
            512BigInt(basis.d+1)
        return (;setup_peak_bytes=setup_peak,output_bytes,
                cached_block_bytes=big(0),
                isometry_bytes=big(0),
                scratch_bytes=recurrence_bytes,
                pure=true,precision_bits)
    end

    PartitionType=eltype(basis.sectors)
    seen_blocks=Set{PartitionType}()
    seen_edges=Set{Tuple{PartitionType,PartitionType}}()
    block_entries=big(0)
    isometry_entries=big(0)
    scratch_entries=big(0)
    for sector in basis.sectors
        lambda=sector
        while true
            if !(lambda in seen_blocks)
                push!(seen_blocks,lambda)
                dimension=unitary_group_dimension(lambda)
                block_entries+=dimension^2
            end
            iszero(weight(lambda))&&break
            mu=remove_corner(
                lambda,first(removable_corners(lambda)))
            edge=(mu,lambda)
            if !(edge in seen_edges)
                push!(seen_edges,edge)
                parent_dimension=unitary_group_dimension(mu)
                block_dimension=unitary_group_dimension(lambda)
                isometry_entries+=
                    block_dimension*parent_dimension*basis.d
                scratch_entries=max(
                    scratch_entries,
                    2block_dimension^2+
                    2block_dimension*parent_dimension)
            end
            lambda=mu
        end
    end
    cached_block_bytes=_performance_entries_bytes(
        block_entries,Complex{R};
        bigfloat_precision=precision_bits)
    isometry_bytes=_performance_entries_bytes(
        isometry_entries,R;bigfloat_precision=precision_bits)
    scratch_bytes=_performance_entries_bytes(
        scratch_entries+8local_dimension^2,Complex{R};
        bigfloat_precision=precision_bits)
    dictionary_bytes=
        512BigInt(length(seen_blocks)+length(seen_edges)+2)
    setup_peak=output_bytes+cached_block_bytes+isometry_bytes+
        scratch_bytes+dictionary_bytes
    (;setup_peak_bytes=setup_peak,output_bytes,cached_block_bytes,
      isometry_bytes,scratch_bytes,pure=false,precision_bits)
end

"""
    supersite_iid_state(site, local_state;
                        memory_budget=512*1024^2, kwargs...)

Construct `N` identical copies of an arbitrary local supersite ket or density
matrix. Unlike [`supersite_product_state`](@ref), `local_state` may already
contain correlations or entanglement between the system and its local
auxiliaries. The recurrence acts directly in PI coordinates.
"""
function supersite_iid_state(
        site::PISupersite,local_state;
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,kwargs...)
    local_state isa Union{AbstractVector,AbstractMatrix}||
        throw(ArgumentError(
        "local_state must be a supersite ket or density matrix"))
    if local_state isa AbstractVector
        length(local_state)==site.basis.d||throw(DimensionMismatch(
            "the local supersite ket must have length $(site.basis.d)"))
    else
        size(local_state)==(site.basis.d,site.basis.d)||
            throw(DimensionMismatch(
            "the local supersite density matrix must be " *
            "$(site.basis.d)×$(site.basis.d)"))
    end
    all(_supersite_isfinite,
        _supersite_stored_values(local_state))||throw(ArgumentError(
        "the local supersite state contains a nonfinite value"))
    R=promote_type(_real_float_type(eltype(site)),
                   _real_float_type(eltype(local_state)))
    precision_bits=R===BigFloat ?
        max(site.estimates.precision_bits,
            _supersite_array_precision(local_state)) :
        precision(R)
    rounding_mode=_supersite_rounding_mode(site,R)
    resources=_supersite_iid_memory_upper(
        site,R,precision_bits;
        pure=local_state isa AbstractVector)
    _require_performance_budget(
        "iid supersite-state construction",resources.setup_peak_bytes,
        memory_budget;
        guidance="Reduce N or an internal local dimension.")
    _with_supersite_precision(R,precision_bits,rounding_mode) do
        if local_state isa AbstractVector
            converted=_supersite_converted_vector(
                local_state,R;context="local supersite ket")
            return iid_pure_state(site.basis,converted;kwargs...)
        end
        converted=_supersite_converted_component(
            local_state,R;context="local supersite density matrix")
        iid_state(site.basis,converted;kwargs...)
    end
end

"""
    supersite_product_state(site, factor_states...;
                            memory_budget=512*1024^2, kwargs...)

Construct the iid `N`-supersite product state from one local state per
internal factor. If every factor is a ket, [`iid_pure_state`](@ref) is used;
otherwise kets are converted to rank-one density matrices and
[`iid_state`](@ref) is used. Only a local vector or density matrix is
tensored—never a `site.basis.d^N` object.
"""
function supersite_product_state(
        site::PISupersite,factor_states...;
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,kwargs...)
    states=length(factor_states)==1&&first(factor_states) isa Tuple ?
        first(factor_states) : factor_states
    length(states)==length(site.factor_dimensions)||
        throw(DimensionMismatch(
        "one local state is required for every supersite factor"))
    all_pure=true
    R=_real_float_type(eltype(site))
    for (index,(state,dimension)) in
            enumerate(zip(states,site.factor_dimensions))
        if state isa AbstractVector
            length(state)==dimension||throw(DimensionMismatch(
                "factor-state ket $index must have length $dimension"))
        elseif state isa AbstractMatrix
            size(state)==(dimension,dimension)||throw(DimensionMismatch(
                "factor-state density matrix $index must be " *
                "$dimension×$dimension"))
            all_pure=false
        else
            throw(ArgumentError(
                "factor state $index must be a ket or density matrix"))
        end
        all(_supersite_isfinite,
            _supersite_stored_values(state))||throw(ArgumentError(
            "factor state $index contains a nonfinite value"))
        R=promote_type(R,_real_float_type(eltype(state)))
    end
    local_entries=BigInt(site.basis.d)^(all_pure ? 1 : 2)
    precision_bits=R===BigFloat ?
        max(site.estimates.precision_bits,
            maximum(_supersite_array_precision,states;init=0)) :
        precision(R)
    rounding_mode=_supersite_rounding_mode(site,R)
    iid_resources=_supersite_iid_memory_upper(
        site,R,precision_bits;pure=all_pure)
    # The local tensor recurrence briefly owns the preceding and new
    # Kronecker products while the iid constructor owns its local input.
    local_construction_bytes=_performance_entries_bytes(
        3local_entries,Complex{R};
        bigfloat_precision=precision_bits)
    estimate=iid_resources.setup_peak_bytes+local_construction_bytes
    _require_performance_budget(
        "supersite product-state construction",estimate,memory_budget;
        guidance="Reduce N or an internal local dimension.")
    _with_supersite_precision(R,precision_bits,rounding_mode) do
        if all_pure
            local_state=_supersite_converted_vector(
                first(states),R;context="factor-state ket 1")
            for (tail_index,state) in enumerate(Base.tail(states))
                converted=_supersite_converted_vector(
                    state,R;
                    context="factor-state ket $(tail_index+1)")
                local_state=_supersite_checked_kron(
                    local_state,converted,
                    "factor-state ket Kronecker product")
            end
            return supersite_iid_state(
                site,local_state;memory_budget,kwargs...)
        end
        density_components=ntuple(length(states)) do index
            state=states[index]
            if state isa AbstractVector
                ket=_supersite_converted_vector(
                    state,R;context="factor-state ket $index")
                _supersite_checked_outer(
                    ket,"factor-state density matrix $index")
            else
                _supersite_converted_component(
                    state,R;context="factor-state density matrix $index")
            end
        end
        local_state=first(density_components)
        for state in Base.tail(density_components)
            local_state=_supersite_checked_kron(
                local_state,state,
                "factor-state density-matrix Kronecker product")
        end
        supersite_iid_state(
            site,local_state;memory_budget,kwargs...)
    end
end

"""
    BosonicPseudomode(nmax;
                      frequency=0, damping=0,
                      thermal_occupation=0, label=:mode, T=nothing,
                      memory_budget=512*1024^2)

Describe one identical truncated bosonic mode with occupations `0:nmax`.
The immutable specification retains its identity, annihilation, creation,
number, parity, top-level projector, and vacuum ket. `frequency` is allowed
to be signed for rotating-frame models; `damping` and
`thermal_occupation` must be finite and nonnegative.
All fixed mode matrices use exact sparse support, and `memory_budget` guards
their aggregate retained storage plus sparse-assembly temporaries before
allocation.

The package dissipator convention is
`D[a](rho)=a*rho*a' - {a'a,rho}/2`, so `damping=kappa` gives an amplitude
pole with real decay `kappa/2`.
"""
struct BosonicPseudomode{R<:AbstractFloat,L,M,V,Q}
    label::L
    nmax::Int
    levels::Int
    frequency::R
    damping::R
    thermal_occupation::R
    identity::M
    annihilation::M
    creation::M
    number_operator::M
    parity::M
    top_projector::M
    vacuum::V
    precision_bits::Int
    rounding_mode::Q
end

eltype(::BosonicPseudomode{R}) where R=Complex{R}

function show(io::IO,mode::BosonicPseudomode)
    print(io,"BosonicPseudomode(label=$(mode.label), nmax=$(mode.nmax), ",
          "frequency=$(mode.frequency), damping=$(mode.damping), ",
          "thermal_occupation=$(mode.thermal_occupation))")
end

function _pseudomode_inferred_real_type(values)
    types=Type[]
    for value in values
        value isa AbstractFloat&&push!(types,typeof(value))
    end
    isempty(types) ? nothing : foldl(promote_type,types)
end

function BosonicPseudomode(
        nmax::Integer;frequency=0,damping=0,thermal_occupation=0,
        label::Symbol=:mode,T=nothing,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    nmax isa Integer&&!(nmax isa Bool)||throw(ArgumentError(
        "nmax must be an integer"))
    nmax>=0||throw(ArgumentError("nmax must be nonnegative"))
    BigInt(nmax)<typemax(Int)||throw(ArgumentError(
        "nmax is too large for local indexing"))
    all(value->value isa Real&&!(value isa Bool),
        (frequency,damping,thermal_occupation))||throw(ArgumentError(
        "frequency, damping, and thermal_occupation must be real numbers"))
    inferred=_pseudomode_inferred_real_type(
        (frequency,damping,thermal_occupation))
    R=T===nothing ? (inferred===nothing ? Float64 : inferred) :
                    _real_float_type(T)
    R<:AbstractFloat||throw(ArgumentError(
        "the pseudomode scalar type must be an AbstractFloat"))
    T!==nothing&&inferred!==nothing&&
        promote_type(R,inferred)!==R&&throw(ArgumentError(
        "T=$R would narrow a pseudomode parameter of type $inferred"))
    precision_bits=R===BigFloat ?
        max(precision(BigFloat),
            _supersite_value_precision(frequency),
            _supersite_value_precision(damping),
            _supersite_value_precision(thermal_occupation)) :
        precision(R)
    rounding_mode=R===BigFloat ? rounding(BigFloat) : nothing
    levels=Int(nmax)+1
    matrix_nonzeros=(
        levels,Int(nmax),Int(nmax),Int(nmax),levels,1)
    retained_bytes=sum(
        (_supersite_sparse_csc_bytes(
             levels,count,Complex{R};
             bigfloat_precision=precision_bits)
         for count in matrix_nonzeros);init=big(0))+
        _performance_entries_bytes(
            levels,Complex{R};
            bigfloat_precision=precision_bits)
    largest_triplet=BigInt(max(Int(nmax),levels))
    assembly_scratch_bytes=
        2largest_triplet*sizeof(Int)+
        _performance_entries_bytes(
            largest_triplet,Complex{R};
            bigfloat_precision=precision_bits)
    peak_bytes=retained_bytes+assembly_scratch_bytes
    _require_performance_budget(
        "bosonic pseudomode construction",peak_bytes,memory_budget;
        guidance="Reduce nmax or use a smaller scalar precision.")
    _with_supersite_precision(R,precision_bits,rounding_mode) do
        omega=_supersite_checked_real(R,frequency,"frequency")
        kappa=_supersite_checked_real(R,damping,"damping")
        occupation=_supersite_checked_real(
            R,thermal_occupation,"thermal_occupation")
        kappa>=zero(R)||throw(ArgumentError(
            "pseudomode damping must be nonnegative"))
        occupation>=zero(R)||throw(ArgumentError(
            "pseudomode thermal_occupation must be nonnegative"))
        identity=spdiagm(
            0=>fill(one(Complex{R}),levels))
        ladder_rows=collect(1:Int(nmax))
        ladder_columns=collect(2:levels)
        ladder_values=Complex{R}[
            _checked_sqrt_exact_integer(
                R,BigInt(number);
                context="pseudomode ladder coefficient sqrt($number)")
            for number in 1:Int(nmax)]
        annihilation=sparse(
            ladder_rows,ladder_columns,ladder_values,levels,levels)
        creation=copy(adjoint(annihilation))
        number_operator=sparse(
            collect(2:levels),collect(2:levels),
            Complex{R}[
                _supersite_checked_real(
                    R,number,"pseudomode occupation label")
                for number in 1:Int(nmax)],
            levels,levels)
        parity=spdiagm(
            0=>Complex{R}[
                isodd(number) ? -one(R) : one(R)
                for number in 0:Int(nmax)])
        top_projector=sparse(
            [levels],[levels],Complex{R}[one(R)],levels,levels)
        vacuum=zeros(Complex{R},levels)
        vacuum[1]=one(R)
        BosonicPseudomode{
            R,typeof(label),typeof(identity),typeof(vacuum),
            typeof(rounding_mode)}(
            label,Int(nmax),levels,omega,kappa,occupation,identity,
            annihilation,creation,number_operator,parity,top_projector,
            vacuum,precision_bits,rounding_mode)
    end
end

function _pseudomode_tuple(modes)
    resolved=modes isa BosonicPseudomode ? (modes,) :
        modes isa Tuple ? modes : Tuple(modes)
    isempty(resolved)&&throw(ArgumentError(
        "at least one BosonicPseudomode is required"))
    all(mode->mode isa BosonicPseudomode,resolved)||throw(ArgumentError(
        "every local mode specification must be a BosonicPseudomode"))
    resolved
end

"""
    pseudomode_supersite(N, system_dimension, modes...;
                         sectors=nothing, T=nothing,
                         memory_budget=512*1024^2)
    pseudomode_supersite(basis, system_dimension, modes...; kwargs...)

Construct one exact PI supersite containing a `system_dimension`-level system
followed by one or more identical-per-site truncated pseudomodes. Modes are
local to each system: the returned basis is
`PIBasis(N, system_dimension*prod(mode.levels))`, which fully exploits
permutation symmetry of the `N` system+mode tuples.
"""
function pseudomode_supersite(
        N::Integer,system_dimension::Integer,modes...;
        sectors=nothing,T=nothing,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    system_dimension isa Integer&&!(system_dimension isa Bool)||
        throw(ArgumentError("system_dimension must be an integer"))
    system_dimension>=1||throw(ArgumentError(
        "system_dimension must be positive"))
    BigInt(system_dimension)<=typemax(Int)||throw(ArgumentError(
        "system_dimension exceeds Int indexing"))
    resolved=length(modes)==1&&
             !(first(modes) isa BosonicPseudomode) ?
        _pseudomode_tuple(first(modes)) : _pseudomode_tuple(modes)
    labels=(:system,(mode.label for mode in resolved)...)
    length(unique(labels))==length(labels)||throw(ArgumentError(
        "pseudomode labels must be unique and different from :system"))
    inferred=foldl(promote_type,
        (_real_float_type(eltype(mode)) for mode in resolved))
    R=T===nothing ? inferred : _real_float_type(T)
    T!==nothing&&promote_type(R,inferred)!==R&&throw(ArgumentError(
        "T=$R would narrow a pseudomode scalar type $inferred"))
    dimensions=(Int(system_dimension),
                (mode.levels for mode in resolved)...)
    specifications=(nothing,resolved...)
    precision_bits=R===BigFloat ?
        max(precision(BigFloat),
            maximum(mode->mode.precision_bits,resolved;init=0)) :
        precision(R)
    rounding_mode=R===BigFloat ? rounding(BigFloat) : nothing
    _with_supersite_precision(R,precision_bits,rounding_mode) do
        PISupersite(
            N,dimensions;labels,specifications,sectors,T=R,memory_budget)
    end
end

function pseudomode_supersite(
        basis::PIBasis,system_dimension::Integer,modes...;
        T=nothing,memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    system_dimension isa Integer&&!(system_dimension isa Bool)||
        throw(ArgumentError("system_dimension must be an integer"))
    system_dimension>=1||throw(ArgumentError(
        "system_dimension must be positive"))
    BigInt(system_dimension)<=typemax(Int)||throw(ArgumentError(
        "system_dimension exceeds Int indexing"))
    resolved=length(modes)==1&&
             !(first(modes) isa BosonicPseudomode) ?
        _pseudomode_tuple(first(modes)) : _pseudomode_tuple(modes)
    labels=(:system,(mode.label for mode in resolved)...)
    length(unique(labels))==length(labels)||throw(ArgumentError(
        "pseudomode labels must be unique and different from :system"))
    inferred=foldl(promote_type,
        (_real_float_type(eltype(mode)) for mode in resolved))
    R=T===nothing ? inferred : _real_float_type(T)
    T!==nothing&&promote_type(R,inferred)!==R&&throw(ArgumentError(
        "T=$R would narrow a pseudomode scalar type $inferred"))
    dimensions=(Int(system_dimension),
                (mode.levels for mode in resolved)...)
    specifications=(nothing,resolved...)
    precision_bits=R===BigFloat ?
        max(precision(BigFloat),
            maximum(mode->mode.precision_bits,resolved;init=0)) :
        precision(R)
    rounding_mode=R===BigFloat ? rounding(BigFloat) : nothing
    _with_supersite_precision(R,precision_bits,rounding_mode) do
        PISupersite(
            basis,dimensions;labels,specifications,T=R,memory_budget)
    end
end

function _pseudomodes(site::PISupersite)
    length(site.factor_specifications)>=2||throw(ArgumentError(
        "the supersite does not contain a pseudomode"))
    modes=Base.tail(site.factor_specifications)
    all(mode->mode isa BosonicPseudomode,modes)||throw(ArgumentError(
        "the supersite was not constructed by pseudomode_supersite"))
    modes
end

function _pseudomode_factor_index(site::PISupersite,mode)
    modes=_pseudomodes(site)
    if mode isa Integer&&!(mode isa Bool)
        1<=mode<=length(modes)||throw(BoundsError(modes,mode))
        return Int(mode)+1
    end
    mode isa Symbol||throw(ArgumentError(
        "a pseudomode must be selected by integer or Symbol"))
    index=findfirst(isequal(mode),site.factor_labels)
    index===nothing&&throw(ArgumentError(
        "unknown pseudomode label $mode"))
    index==1&&throw(ArgumentError(":system is not a pseudomode"))
    index
end

"""
    lift_pseudomode_operator(site, operator; mode=1,
                             memory_budget=512*1024^2)

Lift an operator on one local truncated mode to the complete supersite.
`mode` is a one-based mode number or its symbolic label.
"""
function lift_pseudomode_operator(
        site::PISupersite,operator::AbstractMatrix;
        mode=1,memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    factor=_pseudomode_factor_index(site,mode)
    lift_supersite_operator(site,operator;factor,memory_budget)
end

"""
    pseudomode_operators(site, mode=1; memory_budget=512*1024^2)

Return lifted supersite operators `(annihilation, creation, number_operator,
parity, top_projector)` for one local pseudomode.
"""
function pseudomode_operators(
        site::PISupersite,mode=1;
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    factor=_pseudomode_factor_index(site,mode)
    specification=site.factor_specifications[factor]
    R=promote_type(_real_float_type(eltype(site)),
                   _real_float_type(eltype(specification)))
    precision_bits=R===BigFloat ?
        max(site.estimates.precision_bits,
            specification.precision_bits) : precision(R)
    rounding_mode=_supersite_rounding_mode(site,R)
    if R===BigFloat&&
            (precision(BigFloat)!=precision_bits||
             rounding(BigFloat)!=rounding_mode)
        return _with_supersite_precision(
            R,precision_bits,rounding_mode) do
            pseudomode_operators(site,mode;memory_budget)
        end
    end
    source_operators=(
        specification.annihilation,specification.creation,
        specification.number_operator,specification.parity,
        specification.top_projector)
    output_bytes=BigInt[]
    lift_peaks=BigInt[]
    for operator in source_operators
        components=ntuple(
            index->index==factor ? operator : site.identities[index],
            length(site.factor_dimensions))
        output_nonzeros=
            BigInt(_supersite_structural_nnz(operator))*
            div(BigInt(site.basis.d),BigInt(specification.levels))
        push!(output_bytes,_supersite_sparse_csc_bytes(
            site.basis.d,output_nonzeros,Complex{R};
            bigfloat_precision=precision_bits))
        push!(lift_peaks,_supersite_tensor_peak_bytes(
            site.factor_dimensions,components,Complex{R};
            bigfloat_precision=precision_bits))
    end
    retained=sum(output_bytes;init=big(0))
    peak=retained+maximum(lift_peaks;init=big(0))
    _require_performance_budget(
        "lifted pseudomode operators",peak,memory_budget;
        guidance="Reduce the pseudomode cutoff or another local factor.")
    lift(operator)=lift_supersite_operator(
        site,operator;factor,memory_budget=Inf)
    (annihilation=lift(specification.annihilation),
     creation=lift(specification.creation),
     number_operator=lift(specification.number_operator),
     parity=lift(specification.parity),
     top_projector=lift(specification.top_projector))
end

"""
    PseudomodeCoupling(operator; mode=1, strength=1,
                       counterrotating_strength=0,
                       memory_budget=512*1024^2)

Specify a system--pseudomode interaction. The rotating-wave part is

`g * L⊗a' + conj(g) * L'⊗a`,

with `g=strength`. The optional counter-rotating part is

`h * L⊗a + conj(h) * L'⊗a'`.

`L` may be non-Hermitian and `g,h` may be complex. The constructor copies the
finite square system matrix. [`pseudomode_coupling_terms`](@ref) decomposes
each complex coefficient into real scalar rates multiplying Hermitian local
operators, making the result compatible with prepared PI model families.
"""
struct PseudomodeCoupling{R<:AbstractFloat,O,M,Q}
    operator::O
    mode::M
    strength::Complex{R}
    counterrotating_strength::Complex{R}
    precision_bits::Int
    rounding_mode::Q
end

eltype(::PseudomodeCoupling{R}) where R=Complex{R}

function show(io::IO,coupling::PseudomodeCoupling)
    print(io,"PseudomodeCoupling(mode=$(coupling.mode), ",
          "strength=$(coupling.strength), ",
          "counterrotating_strength=$(coupling.counterrotating_strength))")
end

function PseudomodeCoupling(
        operator::AbstractMatrix;mode=1,strength=1,
        counterrotating_strength=0,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    rows,columns=size(operator)
    rows==columns&&rows>0||throw(DimensionMismatch(
        "a pseudomode coupling operator must be nonempty and square"))
    all(_supersite_isfinite,
        _supersite_stored_values(operator))||throw(ArgumentError(
        "a pseudomode coupling operator must contain only finite values"))
    mode isa Union{Integer,Symbol}&&!(mode isa Bool)||throw(ArgumentError(
        "mode must be a one-based integer or Symbol"))
    if mode isa Integer
        mode>=1||throw(ArgumentError(
            "an integer pseudomode selector must be positive"))
        BigInt(mode)<=typemax(Int)||throw(ArgumentError(
            "the pseudomode selector exceeds Int indexing"))
        mode=Int(mode)
    end
    strength isa Number&&!(strength isa Bool)&&
        counterrotating_strength isa Number&&
        !(counterrotating_strength isa Bool)||
        throw(ArgumentError("pseudomode coupling strengths must be numbers"))
    _supersite_isfinite(strength)&&
        _supersite_isfinite(counterrotating_strength)||throw(ArgumentError(
        "pseudomode coupling strengths must be finite"))
    R=_real_float_type(eltype(operator))
    R=_supersite_promote_parameter_type(R,strength)
    R=_supersite_promote_parameter_type(R,counterrotating_strength)
    precision_bits=R===BigFloat ?
        max(precision(BigFloat),
            _supersite_array_precision(operator),
            _supersite_value_precision(strength),
            _supersite_value_precision(counterrotating_strength)) :
        precision(R)
    rounding_mode=R===BigFloat ? rounding(BigFloat) : nothing
    entries=_supersite_structural_nnz(operator)
    retained_bytes=_supersite_matrix_storage_bytes(
        size(operator,2),entries,issparse(operator),Complex{R};
        bigfloat_precision=precision_bits)
    _require_performance_budget(
        "pseudomode coupling construction",retained_bytes,memory_budget;
        guidance="Reduce the system dimension or operator support.")
    _with_supersite_precision(R,precision_bits,rounding_mode) do
        stored=_supersite_converted_component(
            operator,R;context="pseudomode coupling operator")
        converted_strength=_supersite_checked_complex(
            R,strength,"pseudomode coupling strength")
        converted_counterrotating=_supersite_checked_complex(
            R,counterrotating_strength,
            "counter-rotating pseudomode coupling strength")
        PseudomodeCoupling{
            R,typeof(stored),typeof(mode),typeof(rounding_mode)}(
            stored,mode,converted_strength,converted_counterrotating,
            precision_bits,rounding_mode)
    end
end

function _pseudomode_interaction_operator(
        site::PISupersite,coupling::PseudomodeCoupling,
        creation::Bool;memory_budget)
    factor=_pseudomode_factor_index(site,coupling.mode)
    specification=site.factor_specifications[factor]
    mode_operator=creation ? specification.creation :
                             specification.annihilation
    components=ntuple(length(site.factor_dimensions)) do index
        index==1&&return coupling.operator
        index==factor&&return mode_operator
        site.identities[index]
    end
    supersite_tensor_operator(site,components;memory_budget)
end

function _pseudomode_quadrature_terms(
        operator,strength;retain_zero_components::Bool=false)
    iszero(strength)&&!retain_zero_components&&return ()
    terms=()
    real_strength=real(strength)
    imaginary_strength=imag(strength)
    if !iszero(real_strength)||retain_zero_components
        quadrature=operator+operator'
        issparse(quadrature)&&dropzeros!(quadrature)
        all(_supersite_isfinite,
            _supersite_stored_values(quadrature))||
            throw(ArgumentError(
            "a pseudomode coupling quadrature overflowed; " *
            "use a wider scalar type"))
        terms=(terms...,LocalHamiltonian(
            quadrature;rate=real_strength,check=false))
    end
    if !iszero(imaginary_strength)||retain_zero_components
        quadrature=im*(operator-operator')
        issparse(quadrature)&&dropzeros!(quadrature)
        all(_supersite_isfinite,
            _supersite_stored_values(quadrature))||
            throw(ArgumentError(
            "a pseudomode coupling quadrature overflowed; " *
            "use a wider scalar type"))
        terms=(terms...,LocalHamiltonian(
            quadrature;rate=imaginary_strength,check=false))
    end
    terms
end

"""
    pseudomode_coupling_terms(site, coupling;
                              retain_zero_components=false,
                              memory_budget=512*1024^2)

Build the Hermitian local Hamiltonian terms for one
[`PseudomodeCoupling`](@ref). Fixed local tensor matrices are prepared once;
the real and imaginary coupling strengths remain scalar term rates. Set
`retain_zero_components=true` when a later [`compile_family`](@ref)
specialization must vary a component whose prototype rate is zero.
"""
function pseudomode_coupling_terms(
        site::PISupersite,coupling::PseudomodeCoupling;
        retain_zero_components::Bool=false,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    size(coupling.operator)==
        (site.factor_dimensions[1],site.factor_dimensions[1])||
        throw(DimensionMismatch(
        "the coupling operator must match the supersite system dimension"))
    R=promote_type(_real_float_type(eltype(site)),
                   _real_float_type(eltype(coupling)))
    precision_bits=R===BigFloat ?
        max(site.estimates.precision_bits,coupling.precision_bits) :
        precision(R)
    rounding_mode=_supersite_rounding_mode(site,R)
    if R===BigFloat&&
            (precision(BigFloat)!=precision_bits||
             rounding(BigFloat)!=rounding_mode)
        return _with_supersite_precision(
            R,precision_bits,rounding_mode) do
            pseudomode_coupling_terms(
                site,coupling;retain_zero_components,memory_budget)
        end
    end
    iszero(coupling.strength)&&
        iszero(coupling.counterrotating_strength)&&
        !retain_zero_components&&return ()
    terms=()
    if !iszero(coupling.strength)||retain_zero_components
        rotating=_pseudomode_interaction_operator(
            site,coupling,true;memory_budget)
        terms=_pseudomode_quadrature_terms(
            rotating,coupling.strength;retain_zero_components)
    end
    if !iszero(coupling.counterrotating_strength)||retain_zero_components
        counterrotating=_pseudomode_interaction_operator(
            site,coupling,false;memory_budget)
        terms=(terms...,_pseudomode_quadrature_terms(
            counterrotating,coupling.counterrotating_strength;
            retain_zero_components)...)
    end
    terms
end

"""
    pseudomode_damping_terms(site, mode=1;
                             retain_zero_terms=false,
                             memory_budget=512*1024^2)

Return independent local loss and, when needed, thermal-gain terms for one
pseudomode. Rates are `kappa*(nbar+1)` and `kappa*nbar` in the package's
standard dissipator convention. A zero damping returns an empty tuple.
`retain_zero_terms=true` keeps both loss and gain prototypes for a prepared
scalar-rate family.
"""
function pseudomode_damping_terms(
        site::PISupersite,mode=1;
        retain_zero_terms::Bool=false,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    factor=_pseudomode_factor_index(site,mode)
    specification=site.factor_specifications[factor]
    R=promote_type(_real_float_type(eltype(site)),
                   _real_float_type(eltype(specification)))
    precision_bits=R===BigFloat ?
        max(site.estimates.precision_bits,
            specification.precision_bits) : precision(R)
    rounding_mode=_supersite_rounding_mode(site,R)
    if R===BigFloat&&
            (precision(BigFloat)!=precision_bits||
             rounding(BigFloat)!=rounding_mode)
        return _with_supersite_precision(
            R,precision_bits,rounding_mode) do
            pseudomode_damping_terms(
                site,mode;retain_zero_terms,memory_budget)
        end
    end
    iszero(specification.damping)&&!retain_zero_terms&&return ()
    operators=pseudomode_operators(site,mode;memory_budget)
    _pseudomode_damping_terms(
        specification.damping,specification.thermal_occupation,operators;
        retain_gain=retain_zero_terms)
end

function _pseudomode_damping_terms(
        damping,thermal_occupation,operators;retain_gain::Bool=false)
    stimulated=_supersite_checked_increment(
        thermal_occupation,
        "thermal occupation plus one")
    loss_rate=_supersite_checked_product(
        damping,stimulated,
        "pseudomode thermal-loss rate")
    loss=LocalJump(
        operators.annihilation;
        rate=loss_rate)
    iszero(thermal_occupation)&&!retain_gain&&return (loss,)
    gain_rate=_supersite_checked_product(
        damping,thermal_occupation,
        "pseudomode thermal-gain rate")
    gain=LocalJump(
        operators.creation;
        rate=gain_rate)
    (loss,gain)
end

_pseudomode_damping_terms(specification,operators)=
    _pseudomode_damping_terms(
        specification.damping,specification.thermal_occupation,operators)

function _pseudomode_argument_tuple(value,required_type,name)
    value===nothing&&return ()
    resolved=if value isa required_type
        (value,)
    elseif value isa Tuple
        value
    else
        try
            Tuple(value)
        catch error
            throw(ArgumentError(
                "$name must be one $required_type or an iterable of them: " *
                sprint(showerror,error)))
        end
    end
    all(item->item isa required_type,resolved)||throw(ArgumentError(
        "every $name must be a $required_type"))
    resolved
end

function _pseudomode_parameter_tuple(value,defaults,name;
                                     nonnegative::Bool=false)
    count=length(defaults)
    resolved=if value===nothing
        Tuple(defaults)
    elseif count==1&&value isa Real&&!(value isa Bool)
        (value,)
    else
        try
            Tuple(value)
        catch error
            throw(ArgumentError(
                "$name must be one value per pseudomode: " *
                sprint(showerror,error)))
        end
    end
    length(resolved)==count||throw(DimensionMismatch(
        "$name must contain one value per pseudomode"))
    for parameter in resolved
        parameter isa Real&&!(parameter isa Bool)||throw(ArgumentError(
            "$name values must be real numbers"))
        _supersite_isfinite(parameter)||throw(ArgumentError(
            "$name values must be finite"))
        nonnegative&&parameter<0&&throw(ArgumentError(
            "$name values must be nonnegative"))
    end
    resolved
end

function _pseudomode_promote_term_type(
        ::Type{R},term::AbstractPITerm) where R<:AbstractFloat
    term isa _BuiltinPITerm||return R
    S=R
    operator=_operator_prototype(term_operator(term))
    if operator isa AbstractArray
        S=promote_type(S,_real_float_type(eltype(operator)))
    elseif operator isa AbstractPIOperator
        S=promote_type(S,_real_float_type(eltype(operator.data)))
    end
    rate=term_rate(term)
    rate isa Number&&(S=_supersite_promote_parameter_type(S,rate))
    if term isa _HamiltonianPITerm
        S=_supersite_promote_parameter_type(S,term_hbar(term))
    end
    S
end

function _pseudomode_term_precision(term::AbstractPITerm)
    term isa _BuiltinPITerm||return 0
    operator=_operator_prototype(term_operator(term))
    operator_precision=operator isa AbstractArray ?
        _supersite_array_precision(operator) :
        operator isa AbstractPIOperator ?
            _supersite_array_precision(operator.data) : 0
    rate=term_rate(term)
    rate_precision=rate isa Number ?
        _supersite_value_precision(rate) : 0
    hbar_precision=term isa _HamiltonianPITerm ?
        _supersite_value_precision(term_hbar(term)) : 0
    max(operator_precision,rate_precision,hbar_precision)
end

function _pseudomode_check_term_scalars(
        terms,::Type{R}) where R<:AbstractFloat
    for (index,term) in pairs(terms)
        term isa _BuiltinPITerm||continue
        rate=term_rate(term)
        rate isa Real&&_supersite_checked_real(
            R,rate,"term $index rate")
        term isa _HamiltonianPITerm&&
            _supersite_checked_real(
                R,term_hbar(term),"term $index hbar")
    end
    nothing
end

@inline function _pseudomode_quadrature_count(
        strength,retain_zero_terms::Bool)
    retain_zero_terms ? 2 :
        Int(!iszero(real(strength)))+
        Int(!iszero(imag(strength)))
end

function _pseudomode_model_memory_upper(
        site::PISupersite,system_hamiltonian,resolved_modes,
        resolved_couplings,resolved_system_terms,
        converted_system_rate,converted_frequencies,
        retain_zero_terms::Bool,::Type{R},precision_bits::Int) where
        R<:AbstractFloat
    local_dimension=BigInt(site.basis.d)
    system_dimension=BigInt(site.factor_dimensions[1])
    auxiliary_dimension=div(local_dimension,system_dimension)
    matrix_bytes=BigInt[]
    site_hamiltonian_nonzeros=big(0)

    system_nonzeros=BigInt(
        _supersite_structural_nnz(system_hamiltonian))
    lifted_system_nonzeros=system_nonzeros*auxiliary_dimension
    push!(matrix_bytes,_supersite_sparse_csc_bytes(
        local_dimension,lifted_system_nonzeros,Complex{R};
        bigfloat_precision=precision_bits))
    if !iszero(system_hamiltonian)&&
            (!iszero(converted_system_rate)||retain_zero_terms)
        site_hamiltonian_nonzeros+=lifted_system_nonzeros
    end

    for (mode_index,mode) in pairs(resolved_modes)
        other_factors=div(local_dimension,BigInt(mode.levels))
        counts=BigInt[
            mode.nmax,mode.nmax,mode.nmax,mode.levels,1]
        for count in counts
            push!(matrix_bytes,_supersite_sparse_csc_bytes(
                local_dimension,count*other_factors,Complex{R};
                bigfloat_precision=precision_bits))
        end
        if !iszero(converted_frequencies[mode_index])||
                retain_zero_terms
            site_hamiltonian_nonzeros+=
                BigInt(mode.nmax)*other_factors
        end
    end

    for coupling in resolved_couplings
        factor=_pseudomode_factor_index(site,coupling.mode)
        mode=site.factor_specifications[factor]
        other_modes=div(
            auxiliary_dimension,BigInt(mode.levels))
        interaction_nonzeros=
            BigInt(_supersite_structural_nnz(coupling.operator))*
            BigInt(mode.nmax)*other_modes
        quadrature_nonzeros=min(
            local_dimension^2,2interaction_nonzeros)
        count=_pseudomode_quadrature_count(
            coupling.strength,retain_zero_terms)+
            _pseudomode_quadrature_count(
                coupling.counterrotating_strength,
                retain_zero_terms)
        for _ in 1:count
            push!(matrix_bytes,_supersite_sparse_csc_bytes(
                local_dimension,quadrature_nonzeros,Complex{R};
                bigfloat_precision=precision_bits))
        end
        site_hamiltonian_nonzeros+=BigInt(count)*quadrature_nonzeros
    end

    for term in resolved_system_terms
        term isa Union{
            LocalHamiltonian,CollectiveHamiltonian,LocalJump,
            CollectiveJump,PBodyHamiltonian,LocalPBodyJump,
            CollectivePBodyJump}||throw(ArgumentError(
            "lift_system_term supports fixed built-in microscopic terms"))
        operator=term_operator(term)
        operator isa AbstractMatrix||throw(ArgumentError(
            "lift_system_term requires a fixed matrix operator; define a " *
            "preallocated schedule directly in the supersite dimension"))
        order=body_order(term)
        order<=site.basis.N||throw(ArgumentError(
            "the body order $order exceeds N=$(site.basis.N)"))
        output_dimension=local_dimension^order
        output_dimension<=typemax(Int)||throw(ArgumentError(
            "the lifted system p-body dimension exceeds Int indexing"))
        output_nonzeros=
            BigInt(_supersite_structural_nnz(operator))*
            auxiliary_dimension^order
        push!(matrix_bytes,_supersite_sparse_csc_bytes(
            output_dimension,output_nonzeros,Complex{R};
            bigfloat_precision=precision_bits))
    end

    site_hamiltonian_nonzeros=min(
        local_dimension^2,site_hamiltonian_nonzeros)
    site_hamiltonian_bytes=_supersite_sparse_csc_bytes(
        local_dimension,site_hamiltonian_nonzeros,Complex{R};
        bigfloat_precision=precision_bits)
    generated_retained=sum(matrix_bytes;init=big(0))+
        site_hamiltonian_bytes+
        512BigInt(length(matrix_bytes)+length(resolved_system_terms)+1)
    largest=maximum(matrix_bytes;init=big(0))
    setup_peak=BigInt(site.estimates.retained_bytes)+
        generated_retained+3largest
    (;site_bytes=BigInt(site.estimates.retained_bytes),
      generated_retained_bytes=generated_retained,
      site_hamiltonian_bytes,
      largest_generated_matrix_bytes=largest,
      setup_peak_bytes=setup_peak,
      precision_bits)
end

"""
    pseudomode_model(site, system_hamiltonian;
                     couplings=(), system_terms=(), supersite_terms=(),
                     system_rate=1, frequencies=nothing,
                     dampings=nothing, thermal_occupations=nothing,
                     retain_zero_terms=false,
                     memory_budget=512*1024^2)
    pseudomode_model(N, system_hamiltonian, modes;
                     couplings=(), system_terms=(), supersite_terms=(),
                     system_rate=1, frequencies=nothing,
                     dampings=nothing, thermal_occupations=nothing,
                     retain_zero_terms=false, sectors=nothing,
                     memory_budget=512*1024^2)

Build a time-local PI model for `N` identical quantum systems, each coupled to
the same finite collection of local truncated pseudomodes.

The returned named tuple contains the reusable [`PISupersite`](@ref), exact PI
`basis`, validated `model`, lifted system and mode operators, separated model
term groups, the base one-supersite Hamiltonian, and cutoff metadata.
`system_terms` are fixed built-in microscopic terms in the bare-system
dimension and are lifted automatically; this includes uniform all-to-all
[`PBodyHamiltonian`](@ref) interactions. `supersite_terms` are already in the
combined local dimension.

`site_hamiltonian` (also returned as `base_site_hamiltonian`) combines only
`system_hamiltonian`, mode frequencies, and pseudomode couplings. It cannot
include genuinely collective or `p`-body `system_terms`; those remain in
`lifted_system_terms`. Supplied `supersite_terms` are returned unchanged.

Local Hamiltonians, couplings, frequencies, and damping channels remain
separate terms so `compile`, `compile_family`, and matrix-free solvers can
reuse their exact Schur geometry. Prepare `site` once and call the first
method throughout a scan to preserve exact basis identity. The `N` method is
a convenience constructor. Independent local damping normally requires the
complete Schur-sector basis; an incompatible `sectors` restriction is rejected
by [`PIModel`](@ref). Set `retain_zero_terms=true` before
[`compile_family`](@ref) when a scan must vary rates that are zero in the
prototype.
"""
function pseudomode_model(
        site::PISupersite,system_hamiltonian::AbstractMatrix;
        couplings=(),system_terms=(),supersite_terms=(),system_rate=1,
        frequencies=nothing,dampings=nothing,thermal_occupations=nothing,
        retain_zero_terms::Bool=false,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    site.basis.N>=1||throw(ArgumentError(
        "a pseudomode model requires at least one identical system"))
    rows,columns=size(system_hamiltonian)
    rows==columns&&rows>0||throw(DimensionMismatch(
        "system_hamiltonian must be nonempty and square"))
    rows==site.factor_dimensions[1]||throw(DimensionMismatch(
        "system_hamiltonian must match the supersite system dimension " *
        "$(site.factor_dimensions[1])"))
    all(_supersite_isfinite,
        _supersite_stored_values(system_hamiltonian))||throw(ArgumentError(
        "system_hamiltonian must contain only finite values"))
    ishermitian(system_hamiltonian)||throw(ArgumentError(
        "system_hamiltonian must be Hermitian"))
    system_rate isa Real&&!(system_rate isa Bool)||throw(ArgumentError(
        "system_rate must be a real number"))
    _supersite_isfinite(system_rate)||throw(ArgumentError(
        "system_rate must be finite"))
    resolved_modes=_pseudomodes(site)
    resolved_frequencies=_pseudomode_parameter_tuple(
        frequencies,(mode.frequency for mode in resolved_modes),
        "frequencies")
    resolved_dampings=_pseudomode_parameter_tuple(
        dampings,(mode.damping for mode in resolved_modes),
        "dampings";nonnegative=true)
    resolved_occupations=_pseudomode_parameter_tuple(
        thermal_occupations,
        (mode.thermal_occupation for mode in resolved_modes),
        "thermal_occupations";nonnegative=true)
    resolved_couplings=_pseudomode_argument_tuple(
        couplings,PseudomodeCoupling,"coupling specification")
    resolved_system_terms=_pseudomode_argument_tuple(
        system_terms,AbstractPITerm,"system term")
    resolved_supersite_terms=_pseudomode_argument_tuple(
        supersite_terms,AbstractPITerm,"supersite term")
    R=promote_type(_real_float_type(eltype(site)),
                   _real_float_type(eltype(system_hamiltonian)))
    for mode in resolved_modes
        R=promote_type(R,_real_float_type(eltype(mode)))
    end
    for coupling in resolved_couplings
        R=promote_type(R,_real_float_type(eltype(coupling)))
    end
    for term in (resolved_system_terms...,resolved_supersite_terms...)
        R=_pseudomode_promote_term_type(R,term)
    end
    R=_supersite_promote_parameter_type(R,system_rate)
    for value in (resolved_frequencies...,resolved_dampings...,
                  resolved_occupations...)
        R=_supersite_promote_parameter_type(R,value)
    end
    precision_bits=R===BigFloat ?
        max(site.estimates.precision_bits,
            _supersite_array_precision(system_hamiltonian),
            maximum(mode->mode.precision_bits,resolved_modes;init=0),
            maximum(coupling->coupling.precision_bits,
                    resolved_couplings;init=0),
            maximum(_pseudomode_term_precision,
                    (resolved_system_terms...,
                     resolved_supersite_terms...);init=0),
            _supersite_value_precision(system_rate),
            maximum(_supersite_value_precision,
                    (resolved_frequencies...,resolved_dampings...,
                     resolved_occupations...);init=0)) :
        precision(R)
    rounding_mode=_supersite_rounding_mode(site,R)
    if R===BigFloat&&
            (precision(BigFloat)!=precision_bits||
             rounding(BigFloat)!=rounding_mode)
        return _with_supersite_precision(
            R,precision_bits,rounding_mode) do
            pseudomode_model(
                site,system_hamiltonian;
                couplings=resolved_couplings,
                system_terms=resolved_system_terms,
                supersite_terms=resolved_supersite_terms,
                system_rate,frequencies=resolved_frequencies,
                dampings=resolved_dampings,
                thermal_occupations=resolved_occupations,
                retain_zero_terms,memory_budget)
        end
    end
    converted_system_rate=_supersite_checked_real(
        R,system_rate,"system Hamiltonian rate")
    converted_frequencies=Tuple(
        _supersite_checked_real(
            R,value,"pseudomode frequency")
        for value in resolved_frequencies)
    converted_dampings=Tuple(
        _supersite_checked_real(
            R,value,"pseudomode damping")
        for value in resolved_dampings)
    converted_occupations=Tuple(
        _supersite_checked_real(
            R,value,"pseudomode thermal occupation")
        for value in resolved_occupations)
    _pseudomode_check_term_scalars(
        (resolved_system_terms...,resolved_supersite_terms...),R)
    resources=_pseudomode_model_memory_upper(
        site,system_hamiltonian,resolved_modes,resolved_couplings,
        resolved_system_terms,converted_system_rate,
        converted_frequencies,retain_zero_terms,R,precision_bits)
    _require_performance_budget(
        "pseudomode model construction",resources.setup_peak_bytes,
        memory_budget;
        guidance="Reduce N, a cutoff, p-body order, or operator support.")
    Hsystem=_supersite_converted_component(
        system_hamiltonian,R;context="system Hamiltonian")
    lifted_system_hamiltonian=lift_system_operator(
        site,Hsystem;memory_budget=Inf)

    local_hamiltonian_terms=()
    site_hamiltonian=spzeros(
        Complex{R},site.basis.d,site.basis.d)
    if !iszero(Hsystem)&&
            (!iszero(converted_system_rate)||retain_zero_terms)
        rate=converted_system_rate
        local_hamiltonian_terms=(LocalHamiltonian(
            lifted_system_hamiltonian;rate,check=false),)
        site_hamiltonian=_supersite_add_hamiltonian(
            site_hamiltonian,rate,lifted_system_hamiltonian,
            "system Hamiltonian scaling")
    end

    mode_operators=map(1:length(resolved_modes)) do mode_index
        pseudomode_operators(site,mode_index;memory_budget=Inf)
    end
    for (frequency,operators) in zip(
            converted_frequencies,mode_operators)
        if !iszero(frequency)||retain_zero_terms
            term=LocalHamiltonian(
                operators.number_operator;rate=frequency,check=false)
            local_hamiltonian_terms=(local_hamiltonian_terms...,term)
            site_hamiltonian=_supersite_add_hamiltonian(
                site_hamiltonian,frequency,operators.number_operator,
                "pseudomode frequency scaling")
        end
    end
    coupling_terms=()
    for coupling in resolved_couplings
        prepared=pseudomode_coupling_terms(
            site,coupling;retain_zero_components=retain_zero_terms,
            memory_budget=Inf)
        coupling_terms=(coupling_terms...,prepared...)
        for term in prepared
            rate=_supersite_checked_real(
                R,term.rate,"pseudomode coupling quadrature rate")
            site_hamiltonian=_supersite_add_hamiltonian(
                site_hamiltonian,rate,term.operator,
                "pseudomode coupling scaling")
        end
    end
    damping_terms=()
    for (damping,occupation,operators) in zip(
            converted_dampings,converted_occupations,mode_operators)
        prepared=iszero(damping)&&!retain_zero_terms ? () :
            _pseudomode_damping_terms(
                damping,occupation,operators;
                retain_gain=retain_zero_terms)
        damping_terms=(damping_terms...,prepared...)
    end
    lifted_system_terms=map(resolved_system_terms) do term
        lift_system_term(site,term;memory_budget=Inf)
    end
    terms=(local_hamiltonian_terms...,coupling_terms...,damping_terms...,
           lifted_system_terms...,resolved_supersite_terms...)
    model=PIModel(site.basis,terms)
    metadata=(
        embedding=:identical_local_pseudomodes,
        exact_permutation_symmetry=true,
        cutoff_approximation=true,
        system_dimension=rows,
        mode_count=length(resolved_modes),
        oscillator_cutoffs=Tuple(mode.nmax for mode in resolved_modes),
        mode_levels=Tuple(mode.levels for mode in resolved_modes),
        local_dimension=site.basis.d,
        pi_dimension=length(site.basis),
        full_hilbert_dimension=BigInt(site.basis.d)^site.basis.N,
        ordering=:system_then_local_modes,
        dissipator_convention=:standard,
        frequencies=converted_frequencies,
        dampings=converted_dampings,
        thermal_occupations=converted_occupations,
        precision_bits=precision_bits,
        rounding_mode=rounding_mode,
        retained_zero_terms=retain_zero_terms,
        resource_estimates=resources,
        modes=resolved_modes)
    (;supersite=site,basis=site.basis,model,
      site_hamiltonian,base_site_hamiltonian=site_hamiltonian,
      lifted_system_hamiltonian,
      mode_operators=Tuple(mode_operators),
      local_hamiltonian_terms,coupling_terms,damping_terms,
      lifted_system_terms,
      supersite_terms=resolved_supersite_terms,
      resource_estimates=resources,metadata)
end

function pseudomode_model(
        N::Integer,system_hamiltonian::AbstractMatrix,modes;
        couplings=(),system_terms=(),supersite_terms=(),system_rate=1,
        frequencies=nothing,dampings=nothing,thermal_occupations=nothing,
        retain_zero_terms::Bool=false,
        sectors=nothing,topology::Symbol=:local,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    N isa Integer&&!(N isa Bool)||throw(ArgumentError(
        "N must be an integer"))
    N>=1||throw(ArgumentError(
        "a pseudomode model requires at least one identical system"))
    rows,columns=size(system_hamiltonian)
    rows==columns&&rows>0||throw(DimensionMismatch(
        "system_hamiltonian must be nonempty and square"))
    all(_supersite_isfinite,
        _supersite_stored_values(system_hamiltonian))||throw(ArgumentError(
        "system_hamiltonian must contain only finite values"))
    ishermitian(system_hamiltonian)||throw(ArgumentError(
        "system_hamiltonian must be Hermitian"))
    resolved_modes=_pseudomode_tuple(modes)
    resolved_couplings=_pseudomode_argument_tuple(
        couplings,PseudomodeCoupling,"coupling specification")
    resolved_system_terms=_pseudomode_argument_tuple(
        system_terms,AbstractPITerm,"system term")
    resolved_supersite_terms=_pseudomode_argument_tuple(
        supersite_terms,AbstractPITerm,"supersite term")
    resolved_frequencies=_pseudomode_parameter_tuple(
        frequencies,(mode.frequency for mode in resolved_modes),
        "frequencies")
    resolved_dampings=_pseudomode_parameter_tuple(
        dampings,(mode.damping for mode in resolved_modes),
        "dampings";nonnegative=true)
    resolved_occupations=_pseudomode_parameter_tuple(
        thermal_occupations,
        (mode.thermal_occupation for mode in resolved_modes),
        "thermal_occupations";nonnegative=true)
    topology in (:local,:global)||throw(ArgumentError(
        "topology must be :local or :global"))
    if topology===:global
        length(resolved_modes)==1||throw(ArgumentError(
            "topology=:global currently describes exactly one shared pseudomode"))
        isempty(resolved_supersite_terms)||throw(ArgumentError(
            "supersite_terms are defined only for replicated local " *
            "pseudomodes; put system terms in system_terms for " *
            "topology=:global"))
        return global_pseudomode_model(
            N,system_hamiltonian,only(resolved_modes);
            couplings=resolved_couplings,
            system_terms=resolved_system_terms,system_rate,sectors,
            frequency=only(resolved_frequencies),
            damping=only(resolved_dampings),
            thermal_occupation=only(resolved_occupations),
            retain_zero_terms,memory_budget)
    end
    R=_real_float_type(eltype(system_hamiltonian))
    for mode in resolved_modes
        R=promote_type(R,_real_float_type(eltype(mode)))
    end
    for coupling in resolved_couplings
        R=promote_type(R,_real_float_type(eltype(coupling)))
    end
    for term in (resolved_system_terms...,resolved_supersite_terms...)
        R=_pseudomode_promote_term_type(R,term)
    end
    system_rate isa Real&&!(system_rate isa Bool)||throw(ArgumentError(
        "system_rate must be a real number"))
    _supersite_isfinite(system_rate)||throw(ArgumentError(
        "system_rate must be finite"))
    R=_supersite_promote_parameter_type(R,system_rate)
    for value in (resolved_frequencies...,resolved_dampings...,
                  resolved_occupations...)
        R=_supersite_promote_parameter_type(R,value)
    end
    precision_bits=R===BigFloat ?
        max(precision(BigFloat),
            _supersite_array_precision(system_hamiltonian),
            maximum(mode->mode.precision_bits,resolved_modes;init=0),
            maximum(coupling->coupling.precision_bits,
                    resolved_couplings;init=0),
            maximum(_pseudomode_term_precision,
                    (resolved_system_terms...,
                     resolved_supersite_terms...);init=0),
            _supersite_value_precision(system_rate),
            maximum(_supersite_value_precision,
                    (resolved_frequencies...,resolved_dampings...,
                     resolved_occupations...);init=0)) :
        precision(R)
    rounding_mode=R===BigFloat ? rounding(BigFloat) : nothing
    if R===BigFloat&&precision(BigFloat)!=precision_bits
        return _with_supersite_precision(
            R,precision_bits,rounding_mode) do
            pseudomode_model(
                N,system_hamiltonian,resolved_modes;
                couplings=resolved_couplings,
                system_terms=resolved_system_terms,
                supersite_terms=resolved_supersite_terms,
                system_rate,frequencies=resolved_frequencies,
                dampings=resolved_dampings,
                thermal_occupations=resolved_occupations,
                retain_zero_terms,sectors,topology,memory_budget)
        end
    end
    _supersite_checked_real(
        R,system_rate,"system Hamiltonian rate")
    for (name,values) in (
            ("pseudomode frequency",resolved_frequencies),
            ("pseudomode damping",resolved_dampings),
            ("pseudomode thermal occupation",resolved_occupations))
        for value in values
            _supersite_checked_real(R,value,name)
        end
    end
    site=pseudomode_supersite(
        N,rows,resolved_modes;sectors,T=R,memory_budget)
    pseudomode_model(
        site,system_hamiltonian;
        couplings=resolved_couplings,
        system_terms=resolved_system_terms,
        supersite_terms=resolved_supersite_terms,
        system_rate,frequencies=resolved_frequencies,
        dampings=resolved_dampings,
        thermal_occupations=resolved_occupations,retain_zero_terms,
        memory_budget)
end

"""
    independent_local_pseudomode_model(N, system_hamiltonian, modes;
                                       kwargs...)

General multi-mode alias for [`pseudomode_model`](@ref). The historical
single-mode matrix signature remains available with keywords `nmax`,
`frequency`, `coupling_strength`, and `damping`.
"""
function independent_local_pseudomode_model(
        N::Integer,system_hamiltonian::AbstractMatrix,
        modes::Union{BosonicPseudomode,Tuple,AbstractVector};
        kwargs...)
    pseudomode_model(N,system_hamiltonian,modes;kwargs...)
end

function independent_local_pseudomode_model(
        site::PISupersite,system_hamiltonian::AbstractMatrix;
        kwargs...)
    pseudomode_model(site,system_hamiltonian;kwargs...)
end

"""
    pseudomode_product_state(site, system_state;
                             mode_states=nothing,
                             memory_budget=512*1024^2, kwargs...)

Construct an iid state of identical system+pseudomode supersites. By default
every mode starts in its vacuum ket. Supply one mode state for a single mode
or a tuple of mode kets/density matrices for multiple modes.
"""
function pseudomode_product_state(
        site::PISupersite,system_state;
        mode_states=nothing,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,kwargs...)
    modes=_pseudomodes(site)
    resolved=if mode_states===nothing
        Tuple(mode.vacuum for mode in modes)
    elseif length(modes)==1&&
           mode_states isa Union{AbstractVector,AbstractMatrix}
        (mode_states,)
    else
        Tuple(mode_states)
    end
    length(resolved)==length(modes)||throw(DimensionMismatch(
        "mode_states must contain one state per pseudomode"))
    supersite_product_state(
        site,(system_state,resolved...);
        memory_budget,kwargs...)
end

"""
    pseudomode_trace_plan(site;
                          T=real(eltype(site)),
                          memory_budget=512*1024^2, kwargs...)

Prepare the exact PI map that traces every local pseudomode from an identical
system+pseudomode supersite. All modes are combined into the trailing local
factor, so the output is the complete `PIBasis(N, system_dimension)`.
"""
function pseudomode_trace_plan(
        site::PISupersite;
        T=_real_float_type(eltype(site)),
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,kwargs...)
    _pseudomodes(site)
    R=_real_float_type(T)
    R<:AbstractFloat||throw(ArgumentError(
        "the pseudomode trace scalar type must be an AbstractFloat"))
    precision_bits=R===BigFloat ?
        max(precision(BigFloat),site.estimates.precision_bits) :
        precision(R)
    rounding_mode=_supersite_rounding_mode(site,R)
    if R===BigFloat&&
            (precision(BigFloat)!=precision_bits||
             rounding(BigFloat)!=rounding_mode)
        return _with_supersite_precision(
            R,precision_bits,rounding_mode) do
            pseudomode_trace_plan(
                site;T=R,memory_budget,kwargs...)
        end
    end
    system_dimension=site.factor_dimensions[1]
    auxiliary_dimension=div(site.basis.d,system_dimension)
    LocalFactorTracePlan(
        site.basis,(system_dimension,auxiliary_dimension);
        traced_factor=2,T=R,memory_budget,kwargs...)
end

function _check_pseudomode_trace_plan(
        site::PISupersite,prepared::LocalFactorTracePlan)
    system_dimension=site.factor_dimensions[1]
    auxiliary_dimension=div(site.basis.d,system_dimension)
    prepared.basis===site.basis||throw(ArgumentError(
        "the pseudomode trace plan belongs to a different source basis"))
    prepared.local_dimensions==
        (system_dimension,auxiliary_dimension)||throw(ArgumentError(
        "the trace plan does not use this supersite's system/mode split"))
    prepared.traced_factor==2||throw(ArgumentError(
        "the trace plan must trace the combined pseudomode factor"))
    prepared.output_basis.N==site.basis.N&&
        prepared.output_basis.d==system_dimension||throw(ArgumentError(
        "the trace plan has an incompatible output basis"))
    prepared
end

"""
    trace_pseudomodes!(output, rho, site, plan, workspace; kwargs...)

Trace every local pseudomode into a caller-owned spin/qudit PI state. `plan`
must come from [`pseudomode_trace_plan`](@ref), and `workspace` must be a
task-owned [`LocalFactorTraceWorkspace`](@ref). This is the allocation-free
repeated-application route used in parameter scans.
"""
function trace_pseudomodes!(
        output::PIState,rho::PIState,site::PISupersite,
        plan::LocalFactorTracePlan,
        workspace::LocalFactorTraceWorkspace;kwargs...)
    rho.basis===site.basis||throw(ArgumentError(
        "the state belongs to a different supersite basis"))
    _check_pseudomode_trace_plan(site,plan)
    local_factor_trace!(output,rho,plan,workspace;kwargs...)
end

"""
    trace_pseudomodes(rho, site;
                      plan=nothing, workspace=nothing,
                      memory_budget=512*1024^2, kwargs...)

Trace all local pseudomodes from every identical supersite directly in PI
coordinates. Reuse a [`pseudomode_trace_plan`](@ref) and one task-owned
[`LocalFactorTraceWorkspace`](@ref) across a scan.
"""
function trace_pseudomodes(
        rho::PIState,site::PISupersite;
        plan=nothing,workspace=nothing,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,kwargs...)
    rho.basis===site.basis||throw(ArgumentError(
        "the state belongs to a different supersite basis"))
    R=_real_float_type(eltype(rho.data))
    if R===BigFloat
        precision_bounds=_local_factor_precision_bounds(rho.data)
        precision_bounds[1]==precision_bounds[2]||throw(ArgumentError(
            "source state BigFloat storage has mixed precision range " *
            "$precision_bounds; rebuild it at one precision"))
        rho_precision=precision_bounds[2]
        precision_bits=plan===nothing ?
            max(site.estimates.precision_bits,rho_precision) :
            plan isa LocalFactorTracePlan ?
                plan.estimates.precision_bits : rho_precision
        rounding_mode=plan isa LocalFactorTracePlan ?
            plan.estimates.rounding_mode :
            _supersite_rounding_mode(site,R)
        if precision(BigFloat)!=precision_bits||
                rounding(BigFloat)!=rounding_mode
            return _with_supersite_precision(
                R,precision_bits,rounding_mode) do
                trace_pseudomodes(
                    rho,site;plan,workspace,memory_budget,kwargs...)
            end
        end
    end
    prepared=plan===nothing ?
        pseudomode_trace_plan(
            site;T=R,memory_budget) :
        plan
    prepared isa LocalFactorTracePlan||throw(ArgumentError(
        "plan must be a LocalFactorTracePlan"))
    _check_pseudomode_trace_plan(site,prepared)
    local_factor_trace(rho,prepared;workspace,kwargs...)
end
