"""
    FiniteOperatorBasis(d; label=:auxiliary)

Operator-space factor for an ordinary finite `d`-dimensional auxiliary
Hilbert space.  Its `d^2` coordinates are `vec(A)` in Julia column-major
order.  This factor is intended for small cavities, ancillas, or other finite
systems coupled to one or more compressed [`PIBasis`](@ref) factors.
"""
struct FiniteOperatorBasis
    d::Int
    label::Symbol
    function FiniteOperatorBasis(d::Integer;label::Symbol=:auxiliary)
        d>0||throw(ArgumentError("the finite Hilbert-space dimension must be positive"))
        d<=isqrt(typemax(Int))||throw(OverflowError("the finite operator-space dimension d^2 overflows Int"))
        new(Int(d),label)
    end
end

length(b::FiniteOperatorBasis)=b.d^2
show(io::IO,b::FiniteOperatorBasis)=print(io,
    "FiniteOperatorBasis(d=$(b.d), label=$(repr(b.label)), dimension=$(length(b)))")

const _CompositeFactorBasis=Union{PIBasis,FiniteOperatorBasis}

"""
    CompositePIBasis(factors...)

Tensor product of compressed PI operator spaces and ordinary finite operator
spaces.  At least one factor is required.  The first declared factor is the
fastest-varying index in the flattened coordinate vector.  Consequently a
factorized coordinate vector is ordered as
`kron(x_last, ..., x_second, x_first)`.

Only retained PI coordinates and finite auxiliary `d^2` coordinates are
formed; a PI factor's full `d^N` Hilbert space is never reconstructed.
"""
struct CompositePIBasis{F<:Tuple,D<:Tuple}
    factors::F
    dimensions::D
    dimension::Int
end

function CompositePIBasis(factors::_CompositeFactorBasis...)
    isempty(factors)&&throw(ArgumentError("a composite basis needs at least one factor"))
    dims=map(length,factors)
    total=1
    for n in dims
        total=Base.checked_mul(total,n)
    end
    CompositePIBasis(factors,dims,total)
end

length(b::CompositePIBasis)=b.dimension
show(io::IO,b::CompositePIBasis)=print(io,
    "CompositePIBasis(factors=$(length(b.factors)), dimension=$(length(b)))")

abstract type AbstractCompositePIOperator{T} end

# Public composite containers copy caller-owned coordinate vectors.  Internal
# constructors which have just allocated and completely initialized a fresh
# `Vector` may transfer that ownership instead, avoiding an otherwise redundant
# full-coordinate copy.  Keep the tag private so the public copy contract does
# not depend on caller discipline.
struct _OwnedCompositeCoordinates end
const _OWNED_COMPOSITE_COORDINATES=_OwnedCompositeCoordinates()

"""
    CompositePIOperator(basis, data)
    CompositePIOperator(basis; T=Float64)

An operator in the tensor product of the factor operator-coordinate spaces.
PI factors retain their equation-(7) coefficient convention; finite factors
use column-major matrix units.  The first basis factor varies fastest.
"""
struct CompositePIOperator{T<:AbstractFloat,B<:CompositePIBasis} <:
       AbstractCompositePIOperator{T}
    basis::B
    data::Vector{Complex{T}}
    function CompositePIOperator(b::CompositePIBasis,
                                 data::AbstractVector{Complex{T}}) where
            T<:AbstractFloat
        length(data)==length(b)||throw(DimensionMismatch(
            "composite coefficient vector has the wrong length"))
        new{T,typeof(b)}(b,collect(data))
    end
    function CompositePIOperator(b::CompositePIBasis,
                                 data::Vector{Complex{T}},
                                 ::_OwnedCompositeCoordinates) where
            T<:AbstractFloat
        length(data)==length(b)||throw(DimensionMismatch(
            "composite coefficient vector has the wrong length"))
        new{T,typeof(b)}(b,data)
    end
end

"""
    CompositePIState(basis, data)
    CompositePIState(basis; T=Float64)

Density-state container in the same tensor-product coordinates as
[`CompositePIOperator`](@ref).  Construction checks dimensions only and does
not normalize, symmetrize, or repair the input.
"""
struct CompositePIState{T<:AbstractFloat,B<:CompositePIBasis} <:
       AbstractCompositePIOperator{T}
    basis::B
    data::Vector{Complex{T}}
    function CompositePIState(b::CompositePIBasis,
                              data::AbstractVector{Complex{T}}) where
            T<:AbstractFloat
        length(data)==length(b)||throw(DimensionMismatch(
            "composite coefficient vector has the wrong length"))
        new{T,typeof(b)}(b,collect(data))
    end
    function CompositePIState(b::CompositePIBasis,
                              data::Vector{Complex{T}},
                              ::_OwnedCompositeCoordinates) where
            T<:AbstractFloat
        length(data)==length(b)||throw(DimensionMismatch(
            "composite coefficient vector has the wrong length"))
        new{T,typeof(b)}(b,data)
    end
end

_owned_composite_operator(b::CompositePIBasis,data::Vector{<:Complex})=
    CompositePIOperator(b,data,_OWNED_COMPOSITE_COORDINATES)
_owned_composite_state(b::CompositePIBasis,data::Vector{<:Complex})=
    CompositePIState(b,data,_OWNED_COMPOSITE_COORDINATES)

CompositePIOperator(b::CompositePIBasis;T=Float64)=
    _owned_composite_operator(b,zeros(Complex{T},length(b)))
CompositePIState(b::CompositePIBasis;T=Float64)=
    _owned_composite_state(b,zeros(Complex{T},length(b)))
copy(A::CompositePIOperator)=CompositePIOperator(A.basis,A.data)
copy(rho::CompositePIState)=CompositePIState(rho.basis,rho.data)
eltype(A::AbstractCompositePIOperator)=eltype(A.data)

function _same_composite_basis(A,B)
    A.basis===B.basis||throw(ArgumentError("incompatible composite bases"))
end

function _composite_component_vector(factor::PIBasis,A::AbstractPIOperator,
                                     state::Bool)
    A.basis===factor||throw(ArgumentError(
        "a PI tensor component uses a different PIBasis object"))
    state&&!(A isa PIState)&&throw(ArgumentError(
        "a PI component of a composite state must be a PIState"))
    A.data
end

function _composite_component_vector(factor::FiniteOperatorBasis,
                                     A::AbstractMatrix,state::Bool)
    size(A)==(factor.d,factor.d)||throw(DimensionMismatch(
        "finite tensor component has size $(size(A)); expected ($(factor.d), $(factor.d))"))
    vec(A)
end

_composite_component_vector(factor,component,state::Bool)=throw(ArgumentError(
    "component $(typeof(component)) is incompatible with factor $(typeof(factor))"))

function _composite_tensor_data(b::CompositePIBasis,components,state::Bool)
    length(components)==length(b.factors)||throw(DimensionMismatch(
        "one tensor component is required for each composite factor"))
    vectors=map((factor,component)->
        _composite_component_vector(factor,component,state),b.factors,components)
    R=mapreduce(v->_real_float_type(eltype(v)),promote_type,vectors)
    data=Complex{R}[one(R)]
    # Appending a factor makes it the slower index.  This is `kron(v,data)`
    # without constructing any operator-space Kronecker matrix.
    for vector in vectors
        old=data
        data=Vector{Complex{R}}(undef,Base.checked_mul(length(old),length(vector)))
        @inbounds for j in eachindex(vector),i in eachindex(old)
            data[i+(j-1)*length(old)]=old[i]*vector[j]
        end
    end
    data
end

"""
    composite_tensor_operator(basis, components...)

Construct a factorized composite operator.  A PI component is a
[`PIOperator`](@ref); a finite component is a square matrix.  The returned
coordinates use `kron(x_last, ..., x_first)` ordering.
"""
function composite_tensor_operator(b::CompositePIBasis,components...)
    _owned_composite_operator(b,_composite_tensor_data(b,components,false))
end

"""
    composite_tensor_state(basis, components...)

Construct a factorized composite state from one [`PIState`](@ref) per PI
factor and one density matrix per finite factor.  Inputs are copied and are
not normalized implicitly.
"""
function composite_tensor_state(b::CompositePIBasis,components...)
    _owned_composite_state(b,_composite_tensor_data(b,components,true))
end

function _finite_identity(b::FiniteOperatorBasis,::Type{T}) where T<:AbstractFloat
    Matrix{Complex{T}}(I,b.d,b.d)
end

"""Construct the identity operator on all retained composite factors."""
function composite_identity_operator(b::CompositePIBasis;T=Float64)
    components=map(factor->factor isa PIBasis ?
        identity_operator(factor;T=T) : _finite_identity(factor,T),b.factors)
    composite_tensor_operator(b,components...)
end

function _factor_trace_vector(b::PIBasis,::Type{T}) where T<:AbstractFloat
    _trace_vector(b,Complex{T})
end
function _factor_trace_vector(b::FiniteOperatorBasis,::Type{T}) where T<:AbstractFloat
    result=zeros(Complex{T},length(b))
    @inbounds for i in 1:b.d
        result[i+(i-1)*b.d]=one(T)
    end
    result
end

"""
    composite_trace_vector(basis; T=Float64)

Return the tensor-product physical trace functional in composite coordinate
order.  For a PI factor this includes its exact Schur multiplicity weights;
for a finite factor it is ordinary matrix trace.
"""
function composite_trace_vector(b::CompositePIBasis;T=Float64)
    vectors=map(factor->_factor_trace_vector(factor,T),b.factors)
    data=Complex{T}[one(T)]
    for vector in vectors
        old=data
        data=Vector{Complex{T}}(undef,Base.checked_mul(length(old),length(vector)))
        @inbounds for j in eachindex(vector),i in eachindex(old)
            data[i+(j-1)*length(old)]=old[i]*vector[j]
        end
    end
    data
end

function _composite_trace_groups(b::PIBasis)
    [(diagonal=[b.offsets[s]-1+i+(i-1)*length(b.patterns[s])
                for i in eachindex(b.patterns[s])],
      multiplicity=symmetric_group_dimension(p))
     for (s,p) in pairs(b.sectors)]
end
_composite_trace_groups(b::FiniteOperatorBasis)=[(
    diagonal=[i+(i-1)*b.d for i in 1:b.d],multiplicity=big(1))]

function _composite_diagonal_sum(data,combo,strides,factor::Int,index::Int)
    factor>length(combo)&&return data[index]
    total=zero(eltype(data))
    @inbounds for coordinate in combo[factor].diagonal
        total+=_composite_diagonal_sum(data,combo,strides,factor+1,
                                      index+(coordinate-1)*strides[factor])
    end
    total
end

"""
Return the physical trace of a composite state or operator.

The implementation contracts only joint diagonal coordinates and fuses the
exact product of Schur multiplicities with each sector-tuple trace.  It does
not allocate a full composite trace vector.
"""
function trace(A::AbstractCompositePIOperator)
    R=_real_float_type(eltype(A.data))
    groups=map(_composite_trace_groups,A.basis.factors)
    strides=ntuple(i->i==1 ? 1 : prod(A.basis.dimensions[1:i-1]),
                   length(A.basis.factors))
    total=zero(Complex{R});correction=zero(Complex{R})
    for combo in Iterators.product(groups...)
        block_trace=_composite_diagonal_sum(A.data,combo,strides,1,1)
        multiplicity=prod(group->group.multiplicity,combo;init=big(1))
        contribution=_checked_mul_sqrt_exact_ratio(
            block_trace,multiplicity,big(1);
            context="composite trace contribution")
        updated=total+contribution
        correction+=abs(total)>=abs(contribution) ?
            (total-updated)+contribution : (contribution-updated)+total
        total=updated
    end
    total+correction
end

function _composite_reduced_factor_index(
        basis::CompositePIBasis,factor::Integer)
    factor isa Integer&&!(factor isa Bool)||throw(ArgumentError(
        "the retained composite factor must be an integer"))
    1<=factor<=length(basis.factors)||throw(BoundsError(
        basis.factors,factor))
    Int(factor)
end

"""
    CompositeReductionPlan(basis, factor; T=Float64,
                           memory_budget=512*1024^2)
    CompositeReductionPlan(rho, factor; kwargs...)

Prepare the exact contraction which traces every composite factor except
`factor`. The plan packs only joint physical-diagonal source offsets, group
boundaries, and exact products of Schur multiplicities. It never constructs
the full composite trace vector or a Hilbert-space density matrix.

For ordinary representable multiplicities, their square roots are converted
once and the repeated contraction is allocation-free at machine precision. If
a square root is not independently representable, the plan retains its exact
`BigInt` multiplicity and a binary-scaled factor so application can fuse that
scale with the traced value without premature overflow or underflow.

The plan is immutable and may be shared. It is tied to the exact
[`CompositePIBasis`](@ref), selected factor, scalar type, and, for `BigFloat`,
the captured precision and rounding mode.
"""
struct CompositeReductionPlan{
        R<:AbstractFloat,B<:CompositePIBasis,K,O,G,M,S,P,E,Q}
    basis::B
    kept_factor::Int
    kept_basis::K
    kept_dimension::Int
    kept_stride::Int
    traced_offsets::O
    group_boundaries::G
    exact_multiplicities::M
    scales::S
    prepared_scales::P
    direct_scales::Bool
    estimates::E
    precision_bits::Int
    rounding_mode::Q
end

function show(io::IO,plan::CompositeReductionPlan{R}) where R
    print(io,
        "CompositeReductionPlan(kept_factor=$(plan.kept_factor), " *
        "input_dimension=$(length(plan.basis)), " *
        "output_dimension=$(plan.kept_dimension), " *
        "traced_offsets=$(length(plan.traced_offsets)), scalar_type=Complex{$R})")
end

function _composite_reduction_strides(basis::CompositePIBasis)
    stride=1
    ntuple(length(basis.factors)) do factor
        current=stride
        stride=Base.checked_mul(stride,basis.dimensions[factor])
        current
    end
end

function _composite_reduction_group_multiplicity(
        combo,kept_factor::Int)
    multiplicity=big(1)
    for factor in eachindex(combo)
        factor==kept_factor&&continue
        multiplicity*=combo[factor].multiplicity
    end
    multiplicity
end

function _composite_reduction_group_diagonals(
        combo,kept_factor::Int)
    count=big(1)
    for factor in eachindex(combo)
        factor==kept_factor&&continue
        count*=length(combo[factor].diagonal)
    end
    count
end

function _append_composite_reduction_offsets!(
        offsets::Vector{Int},combo,strides,kept_factor::Int,
        factor::Int,offset::Int)
    if factor>length(combo)
        push!(offsets,offset)
        return offsets
    end
    if factor==kept_factor
        return _append_composite_reduction_offsets!(
            offsets,combo,strides,kept_factor,factor+1,offset)
    end
    @inbounds for coordinate in combo[factor].diagonal
        _append_composite_reduction_offsets!(
            offsets,combo,strides,kept_factor,factor+1,
            offset+(coordinate-1)*strides[factor])
    end
    offsets
end

function CompositeReductionPlan(
        basis::B,factor::Integer;T::Type{R}=Float64,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET) where
        {B<:CompositePIBasis,R<:AbstractFloat}
    isconcretetype(R)||throw(ArgumentError(
        "T must be a concrete AbstractFloat type"))
    kept_factor=_composite_reduced_factor_index(basis,factor)
    kept_basis=basis.factors[kept_factor]
    kept_dimension=basis.dimensions[kept_factor]
    strides=_composite_reduction_strides(basis)
    groups=ntuple(length(basis.factors)) do index
        index==kept_factor ? (nothing,) :
            Tuple(_composite_trace_groups(basis.factors[index]))
    end
    combinations=Iterators.product(groups...)

    group_count=big(1)
    for group in groups
        group_count*=length(group)
    end
    group_count<=typemax(Int)||throw(ArgumentError(
        "the number of composite reduction multiplicity groups exceeds Int indexing capacity"))
    traced_count=big(0)
    exact_payload_bytes=big(0)
    for combo in combinations
        traced_count+=_composite_reduction_group_diagonals(
            combo,kept_factor)
        multiplicity=_composite_reduction_group_multiplicity(
            combo,kept_factor)
        exact_payload_bytes+=cld(
            BigInt(max(1,ndigits(multiplicity;base=2))),8)
    end
    traced_count<=typemax(Int)||throw(ArgumentError(
        "the packed composite reduction diagonal count exceeds Int indexing capacity"))

    precision_bits=R===BigFloat ? precision(BigFloat) : precision(R)
    rounding_mode=R===BigFloat ? rounding(BigFloat) : nothing
    int_bytes=BigInt(sizeof(Int))
    pointer_bytes=BigInt(sizeof(Ptr{Cvoid}))
    scalar_bytes=_scalar_retained_bytes(
        R;bigfloat_precision=precision_bits)
    # `big(::BigInt)` preserves object identity: each public exact
    # multiplicity and the numerator in its prepared scale share one GMP
    # payload. The separate exact-multiplicity vector therefore adds only
    # references and gives callers direct, stable metadata without extracting
    # a private prepared-scale field. Object/header allowances keep this a
    # conservative guard rather than an allocator-specific byte promise.
    exact_bytes=exact_payload_bytes+
        group_count*(8pointer_bytes+6int_bytes)
    # Vector/BigInt headers and the immutable plan object dominate tiny plans.
    # This fixed allowance keeps the public guard conservative without making
    # large packed-offset estimates allocator-specific.
    container_bytes=128int_bytes
    retained_bytes=
        (traced_count+group_count+1)*int_bytes+
        group_count*(4scalar_bytes+2int_bytes)+exact_bytes+
        container_bytes
    setup_peak_bytes=2retained_bytes+
        (traced_count+group_count)*int_bytes
    _require_performance_budget(
        "composite reduction plan setup",setup_peak_bytes,memory_budget;
        guidance="Reduce the number or dimensions of traced factors.")

    offsets=Int[]
    sizehint!(offsets,Int(traced_count))
    boundaries=Int[1]
    sizehint!(boundaries,Int(group_count)+1)
    exact_multiplicities=BigInt[]
    sizehint!(exact_multiplicities,Int(group_count))
    scales=Vector{R}()
    sizehint!(scales,Int(group_count))
    prepared_scales=Vector{
        _PreparedExactScale{R,true}}()
    sizehint!(prepared_scales,Int(group_count))
    direct_scales=true
    for combo in combinations
        multiplicity=_composite_reduction_group_multiplicity(
            combo,kept_factor)
        push!(exact_multiplicities,multiplicity)
        prepared=_prepare_exact_scale(
            R,multiplicity,big(1),Val(true);
            context="composite reduction multiplicity scale")
        push!(prepared_scales,prepared)
        push!(scales,prepared.direct ? prepared.factor : zero(R))
        direct_scales&=prepared.direct
        _append_composite_reduction_offsets!(
            offsets,combo,strides,kept_factor,1,0)
        push!(boundaries,length(offsets)+1)
    end
    length(offsets)==Int(traced_count)||error(
        "internal composite reduction diagonal-count mismatch")
    length(exact_multiplicities)==Int(group_count)||error(
        "internal composite reduction group-count mismatch")

    estimates=(
        input_dimension=length(basis),
        output_dimension=kept_dimension,
        traced_diagonal_count=traced_count,
        multiplicity_group_count=group_count,
        retained_bytes,setup_peak_bytes,
        memory_budget=_memory_budget_bytes(memory_budget),
        scalar_type=Complex{R},precision_bits,rounding_mode,
        direct_scales)
    CompositeReductionPlan{
        R,B,typeof(kept_basis),typeof(offsets),typeof(boundaries),
        typeof(exact_multiplicities),typeof(scales),
        typeof(prepared_scales),typeof(estimates),typeof(rounding_mode)}(
        basis,kept_factor,kept_basis,kept_dimension,
        strides[kept_factor],offsets,boundaries,
        exact_multiplicities,scales,prepared_scales,direct_scales,
        estimates,precision_bits,rounding_mode)
end

function CompositeReductionPlan(
        rho::CompositePIState,factor::Integer;
        T::Type{R}=_real_float_type(eltype(rho.data)),
        kwargs...) where R<:AbstractFloat
    source_type=_real_float_type(eltype(rho.data))
    R===source_type||throw(ArgumentError(
        "T=$R does not match source-state scalar type $source_type; " *
        "convert the state explicitly before preparing its reduction plan"))
    if R===BigFloat
        bounds=_composite_reduction_precision_bounds(rho.data)
        bounds[1]==bounds[2]||throw(ArgumentError(
            "source-state BigFloat storage has mixed precision range " *
            "$bounds; rebuild it at one precision"))
        input_precision=bounds[1]
        if precision(BigFloat)!=input_precision
            return setprecision(BigFloat,input_precision) do
                CompositeReductionPlan(rho.basis,factor;T,kwargs...)
            end
        end
    end
    CompositeReductionPlan(rho.basis,factor;T,kwargs...)
end

function _composite_reduction_precision_bounds(values)
    isempty(values)&&return (precision(BigFloat),precision(BigFloat))
    minimum_precision=typemax(Int)
    maximum_precision=0
    for value in values
        value_precision=max(
            precision(real(value)),precision(imag(value)))
        minimum_precision=min(minimum_precision,value_precision)
        maximum_precision=max(maximum_precision,value_precision)
    end
    minimum_precision,maximum_precision
end

function _check_composite_reduction_resources(
        destination::AbstractArray,rho::CompositePIState,
        plan::CompositeReductionPlan{R}) where R
    rho.basis===plan.basis||throw(ArgumentError(
        "CompositeReductionPlan was prepared for a different CompositePIBasis"))
    length(destination)==plan.kept_dimension||throw(DimensionMismatch(
        "the reduced-state destination has the wrong length"))
    Base.mightalias(destination,rho.data)&&throw(ArgumentError(
        "composite_reduced_state! requires a destination which does not alias the source"))
    required=Complex{R}
    eltype(rho.data)===required||throw(ArgumentError(
        "source state scalar type $(eltype(rho.data)) does not match " *
        "CompositeReductionPlan scalar type $required; rebuild the plan from the state"))
    eltype(destination)===required||throw(ArgumentError(
        "destination scalar type $(eltype(destination)) does not match " *
        "CompositeReductionPlan scalar type $required"))
    if R===BigFloat
        for (name,values) in (
                ("source state",rho.data),
                ("destination",destination))
            bounds=_composite_reduction_precision_bounds(values)
            bounds==(plan.precision_bits,plan.precision_bits)||throw(
                ArgumentError(
                    "$name BigFloat storage has precision range $bounds, " *
                    "but the CompositeReductionPlan requires " *
                    "$(plan.precision_bits) bits"))
        end
    end
    nothing
end

@inline function _composite_reduction_group_sum(
        source,offsets,begins::Int,stops::Int,kept_offset::Int)
    total=zero(eltype(source))
    @inbounds for index in begins:stops
        total+=source[kept_offset+offsets[index]+1]
    end
    total
end

@inline function _composite_reduction_direct_scale(
        input::Real,factor::R,
        prepared::_PreparedExactScale{R,true}) where R<:AbstractFloat
    iszero(input)&&return R(input)*factor
    value=R(input)
    result=value*factor
    fixed_ieee=R===Float16||R===Float32||R===Float64
    endpoint=fixed_ieee&&(
        abs(result)==floatmax(R)||
        abs(result)==nextfloat(zero(R)))
    if !isfinite(result)||iszero(result)||endpoint
        return _apply_prepared_exact_scale(
            value,prepared;
            context="composite reduced-state trace contribution")
    end
    result
end

@inline function _composite_reduction_direct_scale(
        input::Complex,factor::R,
        prepared::_PreparedExactScale{R,true}) where R<:AbstractFloat
    complex(
        _composite_reduction_direct_scale(
            real(input),factor,prepared),
        _composite_reduction_direct_scale(
            imag(input),factor,prepared))
end

function _composite_reduced_state_direct!(
        destination,rho::CompositePIState,
        plan::CompositeReductionPlan)
    source=rho.data
    boundaries=plan.group_boundaries
    offsets=plan.traced_offsets
    scales=plan.scales
    prepared_scales=plan.prepared_scales
    @inbounds for kept_coordinate in 1:plan.kept_dimension
        kept_offset=(kept_coordinate-1)*plan.kept_stride
        total=zero(eltype(destination))
        correction=zero(eltype(destination))
        for group in eachindex(scales)
            block_trace=_composite_reduction_group_sum(
                source,offsets,boundaries[group],
                boundaries[group+1]-1,kept_offset)
            contribution=_composite_reduction_direct_scale(
                block_trace,scales[group],prepared_scales[group])
            updated=total+contribution
            correction+=abs(total)>=abs(contribution) ?
                (total-updated)+contribution :
                (contribution-updated)+total
            total=updated
        end
        destination[kept_coordinate]=total+correction
    end
    destination
end

function _composite_reduced_state_scaled!(
        destination,rho::CompositePIState,
        plan::CompositeReductionPlan)
    source=rho.data
    boundaries=plan.group_boundaries
    offsets=plan.traced_offsets
    scales=plan.prepared_scales
    @inbounds for kept_coordinate in 1:plan.kept_dimension
        kept_offset=(kept_coordinate-1)*plan.kept_stride
        total=zero(eltype(destination))
        correction=zero(eltype(destination))
        for group in eachindex(scales)
            block_trace=_composite_reduction_group_sum(
                source,offsets,boundaries[group],
                boundaries[group+1]-1,kept_offset)
            contribution=_apply_prepared_exact_scale(
                block_trace,scales[group];
                context="composite reduced-state trace contribution")
            updated=total+contribution
            correction+=abs(total)>=abs(contribution) ?
                (total-updated)+contribution :
                (contribution-updated)+total
            total=updated
        end
        destination[kept_coordinate]=total+correction
    end
    destination
end

function _composite_reduced_state_data!(
        destination::AbstractArray,rho::CompositePIState,
        plan::CompositeReductionPlan{R}) where R
    _check_composite_reduction_resources(destination,rho,plan)
    if R===BigFloat&&
            (precision(BigFloat)!=plan.precision_bits||
             rounding(BigFloat)!=plan.rounding_mode)
        return setrounding(BigFloat,plan.rounding_mode) do
            setprecision(BigFloat,plan.precision_bits) do
                _composite_reduced_state_data!(destination,rho,plan)
            end
        end
    end
    plan.direct_scales ?
        _composite_reduced_state_direct!(destination,rho,plan) :
        _composite_reduced_state_scaled!(destination,rho,plan)
end

function _composite_reduced_state_data!(
        destination::AbstractArray,rho::CompositePIState,
        kept_factor::Int)
    plan=CompositeReductionPlan(rho,kept_factor)
    _composite_reduced_state_data!(destination,rho,plan)
end

"""
    composite_reduced_state!(destination, rho, factor;
                             memory_budget=512*1024^2)
    composite_reduced_state!(destination, rho, plan)

Trace every composite factor except `factor` directly in tensor-product
operator coordinates. For a retained [`PIBasis`](@ref), `destination` must be
a [`PIState`](@ref) on the exact retained basis. For a retained
[`FiniteOperatorBasis`](@ref), it must be a square matrix of that factor's
Hilbert-space dimension.

The contraction visits only physical diagonal coordinates of traced factors,
uses exact products of Schur multiplicities, and never reconstructs a
many-particle Hilbert space or a full tensor-product trace vector. Source and
destination must not alias.
"""
function composite_reduced_state!(
        destination::PIState,rho::CompositePIState,factor::Integer;
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    selected=_composite_reduced_factor_index(rho.basis,factor)
    retained=rho.basis.factors[selected]
    retained isa PIBasis||throw(ArgumentError(
        "a PIState destination requires retaining a PIBasis factor"))
    destination.basis===retained||throw(ArgumentError(
        "the reduced PI destination belongs to a different PIBasis object"))
    plan=CompositeReductionPlan(
        rho,selected;memory_budget)
    composite_reduced_state!(destination,rho,plan)
end

function composite_reduced_state!(
        destination::PIState,rho::CompositePIState,
        plan::CompositeReductionPlan)
    plan.kept_basis isa PIBasis||throw(ArgumentError(
        "a PIState destination requires a plan retaining a PIBasis factor"))
    destination.basis===plan.kept_basis||throw(ArgumentError(
        "the reduced PI destination belongs to a different PIBasis object"))
    _composite_reduced_state_data!(destination.data,rho,plan)
    destination
end

function composite_reduced_state!(
        destination::AbstractMatrix,rho::CompositePIState,factor::Integer;
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    selected=_composite_reduced_factor_index(rho.basis,factor)
    retained=rho.basis.factors[selected]
    retained isa FiniteOperatorBasis||throw(ArgumentError(
        "a matrix destination requires retaining a FiniteOperatorBasis factor"))
    size(destination)==(retained.d,retained.d)||throw(DimensionMismatch(
        "the reduced finite-factor destination must be $(retained.d)×$(retained.d)"))
    plan=CompositeReductionPlan(
        rho,selected;memory_budget)
    composite_reduced_state!(destination,rho,plan)
end

function composite_reduced_state!(
        destination::AbstractMatrix,rho::CompositePIState,
        plan::CompositeReductionPlan)
    retained=plan.kept_basis
    retained isa FiniteOperatorBasis||throw(ArgumentError(
        "a matrix destination requires a plan retaining a FiniteOperatorBasis factor"))
    size(destination)==(retained.d,retained.d)||throw(DimensionMismatch(
        "the reduced finite-factor destination must be $(retained.d)×$(retained.d)"))
    _composite_reduced_state_data!(destination,rho,plan)
    destination
end

"""
    composite_reduced_state(rho, factor; memory_budget=512*1024^2)
    composite_reduced_state(rho, plan)

Return the state of one retained composite factor after tracing all other
factors. A PI factor returns a [`PIState`](@ref), while an ordinary finite
factor returns its dense density matrix. No normalization or state repair is
performed.
"""
function composite_reduced_state(
        rho::CompositePIState,factor::Integer;
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    plan=CompositeReductionPlan(
        rho,factor;memory_budget)
    composite_reduced_state(rho,plan)
end

function composite_reduced_state(
        rho::CompositePIState,plan::CompositeReductionPlan{R}) where R
    if R===BigFloat&&
            (precision(BigFloat)!=plan.precision_bits||
             rounding(BigFloat)!=plan.rounding_mode)
        return setrounding(BigFloat,plan.rounding_mode) do
            setprecision(BigFloat,plan.precision_bits) do
                # Allocate both PI and finite destinations inside the exact
                # arithmetic context captured by the immutable plan. Entering
                # only inside the in-place contraction would leave newly
                # allocated BigFloat storage at the ambient precision.
                composite_reduced_state(rho,plan)
            end
        end
    end
    retained=plan.kept_basis
    if retained isa PIBasis
        destination=PIState(retained;T=R)
        return composite_reduced_state!(destination,rho,plan)
    end
    destination=zeros(Complex{R},retained.d,retained.d)
    composite_reduced_state!(destination,rho,plan)
end

"""Normalize a composite state by its physical trace, without other repair."""
function normalize!(rho::CompositePIState)
    z=trace(rho)
    iszero(z)&&throw(ArgumentError("cannot normalize a zero-trace composite state"))
    rho.data./=z
    rho
end

"""Return `tr(rho^2)` from the orthonormal composite coordinates."""
purity(rho::CompositePIState)=real(dot(rho.data,rho.data))

function _factor_adjoint_coordinate(b::FiniteOperatorBasis,index::Int)
    row=mod(index-1,b.d)+1
    column=div(index-1,b.d)+1
    column+(row-1)*b.d
end

function _factor_adjoint_coordinate(b::PIBasis,index::Int)
    sector=searchsortedlast(b.offsets,index)
    sector<=length(b.sectors)||throw(BoundsError(b,index))
    n=length(b.patterns[sector])
    local_index=index-b.offsets[sector]
    row=mod(local_index,n)+1
    column=div(local_index,n)+1
    b.offsets[sector]+column-1+(row-1)*n
end

function _composite_adjoint_coordinate(b::CompositePIBasis,index::Int)
    1<=index<=length(b)||throw(BoundsError(b,index))
    remaining=index-1
    destination=1
    stride=1
    @inbounds for factor_index in eachindex(b.factors)
        n=b.dimensions[factor_index]
        coordinate=mod(remaining,n)+1
        remaining=div(remaining,n)
        adjoint_coordinate=_factor_adjoint_coordinate(
            b.factors[factor_index],coordinate)
        destination+=(adjoint_coordinate-1)*stride
        stride*=n
    end
    destination
end

function _composite_hermiticity_metrics(A::AbstractCompositePIOperator)
    R=_real_float_type(eltype(A.data))
    error=zero(R)
    scale=zero(R)
    @inbounds for index in eachindex(A.data)
        adjoint_index=_composite_adjoint_coordinate(A.basis,index)
        error=max(error,abs(A.data[index]-conj(A.data[adjoint_index])))
        scale=max(scale,abs(A.data[index]))
    end
    (;error,scale)
end

"""Test Hermiticity directly in the tensor-product operator coordinates."""
function ishermitian(A::AbstractCompositePIOperator;
                     atol::Real=0,
                     rtol::Real=sqrt(eps(_real_float_type(eltype(A.data)))))
    atol>=0||throw(ArgumentError("atol must be nonnegative"))
    rtol>=0||throw(ArgumentError("rtol must be nonnegative"))
    metrics=_composite_hermiticity_metrics(A)
    metrics.error<=atol+rtol*metrics.scale
end

"""
    expectation(rho, A)

Return the Hilbert--Schmidt expectation `tr(A' * rho)`.  Since both the PI
equation-(7) basis and finite matrix units are orthonormal, this is the direct
coordinate dot product.  For Hermitian `A` it is the usual expectation value.
"""
function expectation(rho::CompositePIState,A::CompositePIOperator)
    _same_composite_basis(rho,A)
    dot(A.data,rho.data)
end

"""
    CompositeSuperoperatorTerm(basis, actions; coefficient=1)

One factorized superoperator term. `actions` is a tuple with one entry per
factor; use `nothing` for an identity map.  Each other entry must be a square
operator-space map of the corresponding factor dimension.  The coefficient
may be a number or a `value_at`-compatible time-dependent schedule. Matrix
actions are copied at construction so the resulting term is read-only.
"""
struct CompositeSuperoperatorTerm{B,A<:Tuple,C}
    basis::B
    actions::A
    coefficient::C
end

_validate_composite_factor_action(::FiniteOperatorBasis,action,index)=nothing
_validate_composite_factor_action(::PIBasis,::AbstractMatrix,index)=nothing
function _validate_composite_factor_action(b::PIBasis,plan::LiouvillianPlan,index)
    plan.basis===b||throw(ArgumentError(
        "factor $index LiouvillianPlan belongs to a different PIBasis object"))
end
function _validate_composite_factor_action(b::PIBasis,
                                           compiled::CompiledPIModel,index)
    compiled.plan.basis===b||throw(ArgumentError(
        "factor $index compiled model belongs to a different PIBasis object"))
end
function _validate_composite_factor_action(b::PIBasis,L::MatrixFreeLiouvillian,
                                           index)
    L.plan===nothing&&throw(ArgumentError(
        "factor $index custom MatrixFreeLiouvillian has no PIBasis provenance; "*
        "pass a matrix, LiouvillianPlan, or compiled PI Liouvillian"))
    L.plan.basis===b||throw(ArgumentError(
        "factor $index matrix-free Liouvillian belongs to a different PIBasis object"))
end
_validate_composite_factor_action(::PIBasis,action,index)=nothing
_stored_composite_action(action::AbstractMatrix)=copy(action)
_stored_composite_action(action)=action

function CompositeSuperoperatorTerm(b::CompositePIBasis,actions::Tuple;
                                    coefficient=1)
    length(actions)==length(b.factors)||throw(DimensionMismatch(
        "one action entry is required for each composite factor"))
    stored_actions=map(action->action===nothing ? nothing :
                       _stored_composite_action(action),actions)
    for (i,action) in pairs(stored_actions)
        action===nothing&&continue
        size(action)==(b.dimensions[i],b.dimensions[i])||throw(DimensionMismatch(
            "factor $i action has size $(size(action)); expected ($(b.dimensions[i]), $(b.dimensions[i]))"))
        _validate_composite_factor_action(b.factors[i],action,i)
    end
    CompositeSuperoperatorTerm{typeof(b),typeof(stored_actions),typeof(coefficient)}(
        b,stored_actions,coefficient)
end

"""
    factorized_superoperator_term(basis, factor=>action...; coefficient=1)

Construct a term by naming only its nonidentity factor actions.  Duplicate or
out-of-range factor indices are rejected.
"""
function factorized_superoperator_term(b::CompositePIBasis,pairs::Pair...;
                                       coefficient=1)
    actions=Any[nothing for _ in b.factors]
    seen=falses(length(actions))
    for pair in pairs
        i=Int(first(pair))
        1<=i<=length(actions)||throw(BoundsError(b.factors,i))
        seen[i]&&throw(ArgumentError("factor $i is specified more than once"))
        seen[i]=true
        actions[i]=last(pair)
    end
    CompositeSuperoperatorTerm(b,Tuple(actions);coefficient=coefficient)
end

"""Lift one factor action while leaving every other factor unchanged."""
local_superoperator_term(b::CompositePIBasis,factor::Integer,action;
                         coefficient=1)=
    factorized_superoperator_term(b,Int(factor)=>action;coefficient=coefficient)

_composite_action_eltype(action)=eltype(action)
_composite_action_autonomous(::AbstractMatrix)=true
_composite_action_autonomous(action)=isautonomous(action)

function _term_eltype(term::CompositeSuperoperatorTerm)
    types=Type[]
    term.coefficient isa Number&&push!(types,typeof(term.coefficient))
    for action in term.actions
        action===nothing||push!(types,_composite_action_eltype(action))
    end
    isempty(types) ? nothing : foldl(promote_type,types)
end

_term_autonomous(term::CompositeSuperoperatorTerm)=
    term.coefficient isa Number&&all(action->action===nothing||
        _composite_action_autonomous(action),term.actions)

_fixed_composite_action_type(::AbstractMatrix)=nothing
_fixed_composite_action_type(plan::LiouvillianPlan)=eltype(plan)
_fixed_composite_action_type(compiled::CompiledPIModel)=eltype(compiled)
_fixed_composite_action_type(L::MatrixFreeLiouvillian)=
    L.plan===nothing ? nothing : eltype(L)
_fixed_composite_action_type(action)=nothing

function _composite_coordinate_type(T)
    T isa Type||throw(ArgumentError(
        "the composite scalar precision must be a type"))
    if T<:AbstractFloat
        return Complex{T}
    end
    R=try
        _real_float_type(T)
    catch
        nothing
    end
    T<:Complex{<:AbstractFloat}&&R!==nothing||throw(ArgumentError(
        "the composite scalar precision must be an AbstractFloat type or a Complex floating type"))
    T
end

"""
    CompositeSuperoperator(basis, terms...; T=nothing)

Read-only matrix-free sum of factorized superoperators.  No Kronecker matrix
is formed.  `T` is required only when the scalar type cannot be inferred, for
example for an empty sum or a driven identity term.
"""
struct CompositeSuperoperator{B,TT<:Tuple,T}
    basis::B
    terms::TT
    Ttype::Type{T}
    autonomous::Bool
end

function CompositeSuperoperator(b::CompositePIBasis,
                                terms::CompositeSuperoperatorTerm...;T=nothing)
    all(term->term.basis===b,terms)||throw(ArgumentError(
        "every term must use the exact CompositePIBasis object"))
    inferred=filter(!isnothing,map(_term_eltype,terms))
    inferred_scalar=isempty(inferred) ? nothing : foldl(promote_type,inferred)
    raw_scalar = if T===nothing
        isempty(inferred)&&throw(ArgumentError(
            "cannot infer the scalar type; pass T for an empty or callable-only sum"))
        inferred_scalar
    else
        requested=_composite_coordinate_type(T)
        if inferred_scalar!==nothing
            required=_composite_coordinate_type(_real_float_type(inferred_scalar))
            promote_type(requested,required)==requested||throw(ArgumentError(
                "explicit composite scalar type $requested would narrow term data of type $required"))
        end
        requested
    end
    # Composite states always use complex operator coordinates.  A real map
    # must therefore still own complex scratch: it can act on a density matrix
    # with nonzero coherences without an inexact scatter into real buffers.
    scalar = _composite_coordinate_type(_real_float_type(raw_scalar))
    scalar<:Number||throw(ArgumentError("the composite scalar type must be numeric"))
    for term in terms,action in term.actions
        action===nothing&&continue
        fixed_type=_fixed_composite_action_type(action)
        fixed_type===nothing&&continue
        _real_float_type(fixed_type)==_real_float_type(scalar)||throw(ArgumentError(
            "a prepared factor action has fixed scalar type $fixed_type but the "*
            "composite coordinates use $scalar; compile that PI action at the "*
            "composite precision"))
    end
    CompositeSuperoperator{typeof(b),typeof(terms),scalar}(
        b,terms,scalar,all(_term_autonomous,terms))
end

size(S::CompositeSuperoperator)=(length(S.basis),length(S.basis))
size(S::CompositeSuperoperator,i::Integer)=i in (1,2) ? length(S.basis) : 1
eltype(S::CompositeSuperoperator)=S.Ttype
isautonomous(S::CompositeSuperoperator)=S.autonomous
_fixed_composite_action_type(S::CompositeSuperoperator)=eltype(S)

function +(A::CompositeSuperoperator,B::CompositeSuperoperator)
    A.basis===B.basis||throw(ArgumentError("incompatible composite bases"))
    CompositeSuperoperator(A.basis,A.terms...,B.terms...;
        T=promote_type(eltype(A),eltype(B)))
end

struct _CompositeFactorWorkspace{V,W}
    input::V
    output::V
    action_workspace::W
end
struct _CompositeTermWorkspace{W<:Tuple}
    factors::W
end

_composite_nested_workspace(::AbstractMatrix)=nothing
_composite_nested_workspace(plan::LiouvillianPlan)=LiouvillianWorkspace(plan)
_composite_nested_workspace(compiled::CompiledPIModel)=LiouvillianWorkspace(compiled)
_composite_nested_workspace(L::MatrixFreeLiouvillian)=
    L.plan===nothing ? nothing : LiouvillianWorkspace(L.plan)
_composite_nested_workspace(S::CompositeSuperoperator)=
    CompositeSuperoperatorWorkspace(S)
_composite_nested_workspace(action)=nothing

function _factor_workspace(action,n,::Type{T}) where T
    action===nothing&&return nothing
    _CompositeFactorWorkspace(zeros(T,n),zeros(T,n),
                              _composite_nested_workspace(action))
end

"""
    CompositeSuperoperatorWorkspace(S[, source])
    CompositeSuperoperatorWorkspace(S; T=eltype(S))

Caller-owned full-vector, tensor-fiber, and nested Liouvillian scratch for
[`apply!`](@ref).  Reuse it sequentially; use one workspace per concurrent
task.
"""
struct CompositeSuperoperatorWorkspace{S,V,W<:Tuple}
    superoperator::S
    buffer1::V
    buffer2::V
    terms::W
end

function CompositeSuperoperatorWorkspace(S::CompositeSuperoperator;
                                         T=eltype(S))
    resolved_type=_composite_coordinate_type(T)
    promote_type(resolved_type,eltype(S))==resolved_type||throw(ArgumentError(
        "workspace scalar type $resolved_type would narrow composite superoperator $(eltype(S))"))
    if _real_float_type(resolved_type)!=_real_float_type(eltype(S))&&
       any(action->action!==nothing&&
           _fixed_composite_action_type(action)!==nothing,
           (action for term in S.terms for action in term.actions))
        throw(ArgumentError(
            "a wider composite workspace is incompatible with a fixed-precision "*
            "prepared PI factor action; compile that action and construct the "*
            "composite superoperator at the wider precision"))
    end
    term_workspaces=map(S.terms) do term
        factors=map((action,n)->_factor_workspace(action,n,resolved_type),
                    term.actions,S.basis.dimensions)
        _CompositeTermWorkspace(factors)
    end
    CompositeSuperoperatorWorkspace(S,zeros(resolved_type,length(S.basis)),
                                    zeros(resolved_type,length(S.basis)),term_workspaces)
end
CompositeSuperoperatorWorkspace(S::CompositeSuperoperator,
                                source::AbstractVector)=
    CompositeSuperoperatorWorkspace(S;T=promote_type(eltype(S),eltype(source)))

struct _CompositeFactorBatchWorkspace{M,W}
    input::M
    output::M
    action_workspace::W
end
struct _CompositeTermBatchWorkspace{W<:Tuple}
    factors::W
end

function _composite_nested_batch_workspace(action,capacity::Int)
    work=_composite_nested_workspace(action)
    if work isa LiouvillianWorkspace
        _ensure_batch_capacity!(work.batch,capacity)
    elseif work isa CompositeSuperoperatorWorkspace
        return CompositeSuperoperatorBatchWorkspace(
            work.superoperator;capacity,T=eltype(work.buffer1))
    end
    work
end

function _factor_batch_workspace(action,n,capacity::Int,::Type{T}) where T
    action===nothing&&return nothing
    _CompositeFactorBatchWorkspace(
        zeros(T,n,capacity),zeros(T,n,capacity),
        _composite_nested_batch_workspace(action,capacity))
end

"""
    CompositeSuperoperatorBatchWorkspace(S; capacity, T=eltype(S))

Task-owned, fixed-capacity scratch for matrix right-hand sides of a
[`CompositeSuperoperator`](@ref). The workspace batches equal tensor fibers
from all supplied right-hand sides through each factor map. It never forms a
global Kronecker matrix and never grows after construction: applying more than
`capacity` columns raises.
"""
struct CompositeSuperoperatorBatchWorkspace{S,M,W<:Tuple}
    superoperator::S
    capacity::Int
    buffer1::M
    buffer2::M
    terms::W
end

function CompositeSuperoperatorBatchWorkspace(
        S::CompositeSuperoperator;capacity::Integer,T=eltype(S))
    capacity isa Integer&&!(capacity isa Bool)&&capacity>0||
        throw(ArgumentError("batch capacity must be a positive integer"))
    BigInt(capacity)<=typemax(Int)||throw(ArgumentError(
        "batch capacity must be representable as an Int"))
    cap=Int(capacity)
    resolved_type=_composite_coordinate_type(T)
    promote_type(resolved_type,eltype(S))==resolved_type||throw(ArgumentError(
        "workspace scalar type $resolved_type would narrow composite superoperator $(eltype(S))"))
    if _real_float_type(resolved_type)!=_real_float_type(eltype(S))&&
       any(action->action!==nothing&&
           _fixed_composite_action_type(action)!==nothing,
           (action for term in S.terms for action in term.actions))
        throw(ArgumentError(
            "a wider composite workspace is incompatible with a fixed-precision "*
            "prepared PI factor action; compile that action and construct the "*
            "composite superoperator at the wider precision"))
    end
    term_workspaces=map(S.terms) do term
        factors=map((action,n)->_factor_batch_workspace(
                        action,n,cap,resolved_type),
                    term.actions,S.basis.dimensions)
        _CompositeTermBatchWorkspace(factors)
    end
    CompositeSuperoperatorBatchWorkspace(
        S,cap,zeros(resolved_type,length(S.basis),cap),
        zeros(resolved_type,length(S.basis),cap),term_workspaces)
end

CompositeSuperoperatorBatchWorkspace(
    S::CompositeSuperoperator,source::AbstractMatrix)=
    CompositeSuperoperatorBatchWorkspace(
        S;capacity=size(source,2),
        T=promote_type(eltype(S),eltype(source)))

function _composite_factor_apply!(y,A::AbstractMatrix,x,t,p,work)
    mul!(y,A,x)
end
function _composite_factor_apply!(y,A::LiouvillianPlan,x,t,p,
                                  work::LiouvillianWorkspace)
    apply!(y,A,x,t,p,work)
end
function _composite_factor_apply!(y,A::CompiledPIModel,x,t,p,
                                  work::LiouvillianWorkspace)
    apply!(y,A,x,t,p,work)
end
function _composite_factor_apply!(y,A::MatrixFreeLiouvillian,x,t,p,
                                  work)
    work===nothing ? apply!(y,A,x,t,p) : apply!(y,A,x,t,p,work)
end
function _composite_factor_apply!(
        y,A::CompositeSuperoperator,x,t,p,
        work::CompositeSuperoperatorWorkspace)
    apply!(y,A,x,t,p,work)
end
function _composite_factor_apply!(y,A,x,t,p,work)
    apply!(y,A,x,t,p)
end

function _composite_factor_apply_adjoint!(
        y,A::AbstractMatrix,x,t,p,work)
    mul!(y,adjoint(A),x)
end
function _composite_factor_apply_adjoint!(
        y,A::LiouvillianPlan,x,t,p,work::LiouvillianWorkspace)
    apply_adjoint!(y,A,x,t,p,work)
end
function _composite_factor_apply_adjoint!(
        y,A::CompiledPIModel,x,t,p,work::LiouvillianWorkspace)
    apply_adjoint!(y,A,x,t,p,work)
end
function _composite_factor_apply_adjoint!(
        y,A::MatrixFreeLiouvillian,x,t,p,work)
    if A.plan===nothing&&getfield(A,:adjoint_action!)===nothing&&
            getfield(A,:batched_adjoint_action!)===nothing
        throw(ArgumentError(
            "a factor MatrixFreeLiouvillian has no explicit adjoint action"))
    end
    work===nothing ? apply_adjoint!(y,A,x,t,p) :
                     apply_adjoint!(y,A,x,t,p,work)
end
function _composite_factor_apply_adjoint!(
        y,A::CompositeSuperoperator,x,t,p,
        work::CompositeSuperoperatorWorkspace)
    apply_adjoint!(y,A,x,t,p,work)
end
function _composite_factor_apply_adjoint!(y,A,x,t,p,work)
    applicable(apply_adjoint!,y,A,x,t,p)||throw(ArgumentError(
        "composite factor action $(typeof(A)) has no explicit adjoint application"))
    apply_adjoint!(y,A,x,t,p)
end

function _apply_tensor_mode_generic!(destination,action,source,factor::Int,dims,
                                     t,p,work::_CompositeFactorWorkspace)
    stride=1
    @inbounds for i in 1:factor-1
        stride*=dims[i]
    end
    n=dims[factor]
    outer=length(source)÷(stride*n)
    @inbounds for block in 0:outer-1,inner in 1:stride
        base=block*stride*n+inner
        for j in 1:n
            work.input[j]=source[base+(j-1)*stride]
        end
        _composite_factor_apply!(work.output,action,work.input,t,p,
                                 work.action_workspace)
        for j in 1:n
            destination[base+(j-1)*stride]=work.output[j]
        end
    end
    destination
end

function _apply_tensor_mode_adjoint!(
        destination,action,source,factor::Int,dims,t,p,
        work::_CompositeFactorWorkspace)
    stride=1
    @inbounds for i in 1:factor-1
        stride*=dims[i]
    end
    n=dims[factor]
    outer=length(source)÷(stride*n)
    @inbounds for block in 0:outer-1,inner in 1:stride
        base=block*stride*n+inner
        for j in 1:n
            work.input[j]=source[base+(j-1)*stride]
        end
        _composite_factor_apply_adjoint!(
            work.output,action,work.input,t,p,work.action_workspace)
        for j in 1:n
            destination[base+(j-1)*stride]=work.output[j]
        end
    end
    destination
end

@inline function _apply_tensor_mode!(destination,action,source,factor::Int,dims,
                                     t,p,work::_CompositeFactorWorkspace)
    _apply_tensor_mode_generic!(destination,action,source,factor,dims,t,p,work)
end

# `reshape(vector, ...)` creates small wrappers which escape through BLAS on
# supported Julia releases.  Calling the same GEMM ABI on the contiguous vector
# storage keeps the explicit-workspace application path allocation-free.  The
# generic false return retains the gather/GEMV/scatter fallback for non-BLAS
# scalar types.
_composite_first_mode_gemm!(destination,action,source,n,columns)=false
for (gemm,elty) in ((:dgemm_,:Float64),(:sgemm_,:Float32),
                    (:zgemm_,:ComplexF64),(:cgemm_,:ComplexF32))
    @eval function _composite_first_mode_gemm!(
            destination::StridedVector{$elty},action::StridedMatrix{$elty},
            source::StridedVector{$elty},n::Int,columns::Int)
        stride(destination,1)==1&&stride(source,1)==1&&stride(action,1)==1||
            return false
        BI=LinearAlgebra.BLAS.BlasInt
        m=BI(n);batch=BI(columns);k=BI(n)
        lda=BI(max(1,stride(action,2)));ldb=BI(max(1,n));ldc=BI(max(1,n))
        alpha=one($elty);beta=zero($elty);trans=UInt8('N')
        GC.@preserve destination action source begin
            ccall((LinearAlgebra.BLAS.@blasfunc($gemm),
                   LinearAlgebra.BLAS.libblastrampoline),Cvoid,
                  (Ref{UInt8},Ref{UInt8},Ref{LinearAlgebra.BLAS.BlasInt},
                   Ref{LinearAlgebra.BLAS.BlasInt},
                   Ref{LinearAlgebra.BLAS.BlasInt},Ref{$elty},Ptr{$elty},
                   Ref{LinearAlgebra.BLAS.BlasInt},Ptr{$elty},
                   Ref{LinearAlgebra.BLAS.BlasInt},Ref{$elty},Ptr{$elty},
                   Ref{LinearAlgebra.BLAS.BlasInt},Clong,Clong),
                  trans,trans,m,batch,k,alpha,pointer(action),lda,
                  pointer(source),ldb,beta,pointer(destination),ldc,1,1)
        end
        true
    end
end

# The first declared factor is contiguous in the composite coordinate vector.
# For a homogeneous dense matrix action, all of its tensor fibers are therefore
# the columns of one reshaped matrix and can be applied by one GEMM.  Preserve
# the gather/GEMV/scatter implementation above for later factors, non-strided
# inputs, sparse/custom actions, and mixed scalar types (the latter avoids the
# substantial Julia-1.10 mixed-real/complex packing fallback).
@inline function _apply_tensor_mode!(destination::StridedVector{T},
                                     action::StridedMatrix{T},
                                     source::StridedVector{T},factor::Int,dims,
                                     t,p,work::_CompositeFactorWorkspace) where T
    factor==1||return _apply_tensor_mode_generic!(
        destination,action,source,factor,dims,t,p,work)
    n=dims[1]
    columns=length(source)÷n
    _composite_first_mode_gemm!(destination,action,source,n,columns)&&
        return destination
    _apply_tensor_mode_generic!(destination,action,source,factor,dims,t,p,work)
end

function _composite_factor_apply_batch!(
        y,A::AbstractMatrix,x,t,p,work)
    mul!(y,A,x)
end
function _composite_factor_apply_batch!(
        y,A::LiouvillianPlan,x,t,p,work::LiouvillianWorkspace)
    apply!(y,A,x,t,p,work)
end
function _composite_factor_apply_batch!(
        y,A::CompiledPIModel,x,t,p,work::LiouvillianWorkspace)
    apply!(y,A,x,t,p,work)
end
function _composite_factor_apply_batch!(
        y,A::MatrixFreeLiouvillian,x,t,p,work)
    work===nothing ? apply!(y,A,x,t,p) : apply!(y,A,x,t,p,work)
end
function _composite_factor_apply_batch!(
        y,A::CompositeSuperoperator,x,t,p,
        work::CompositeSuperoperatorBatchWorkspace)
    apply!(y,A,x,t,p,work)
end
function _composite_factor_apply_batch!(y,A,x,t,p,work)
    if work!==nothing&&applicable(apply!,y,A,x,t,p,work)
        return apply!(y,A,x,t,p,work)
    elseif applicable(apply!,y,A,x,t,p)
        return apply!(y,A,x,t,p)
    end
    for column in axes(x,2)
        _composite_factor_apply!(
            view(y,:,column),A,view(x,:,column),t,p,work)
    end
    y
end

function _composite_factor_apply_adjoint_batch!(
        y,A::AbstractMatrix,x,t,p,work)
    mul!(y,adjoint(A),x)
end
function _composite_factor_apply_adjoint_batch!(
        y,A::LiouvillianPlan,x,t,p,work::LiouvillianWorkspace)
    apply_adjoint!(y,A,x,t,p,work)
end
function _composite_factor_apply_adjoint_batch!(
        y,A::CompiledPIModel,x,t,p,work::LiouvillianWorkspace)
    apply_adjoint!(y,A,x,t,p,work)
end
function _composite_factor_apply_adjoint_batch!(
        y,A::MatrixFreeLiouvillian,x,t,p,work)
    if A.plan===nothing&&getfield(A,:adjoint_action!)===nothing&&
            getfield(A,:batched_adjoint_action!)===nothing
        throw(ArgumentError(
            "a factor MatrixFreeLiouvillian has no explicit adjoint action"))
    end
    work===nothing ? apply_adjoint!(y,A,x,t,p) :
                     apply_adjoint!(y,A,x,t,p,work)
end
function _composite_factor_apply_adjoint_batch!(
        y,A::CompositeSuperoperator,x,t,p,
        work::CompositeSuperoperatorBatchWorkspace)
    apply_adjoint!(y,A,x,t,p,work)
end
function _composite_factor_apply_adjoint_batch!(y,A,x,t,p,work)
    if work!==nothing&&applicable(apply_adjoint!,y,A,x,t,p,work)
        return apply_adjoint!(y,A,x,t,p,work)
    elseif applicable(apply_adjoint!,y,A,x,t,p)
        return apply_adjoint!(y,A,x,t,p)
    end
    for column in axes(x,2)
        _composite_factor_apply_adjoint!(
            view(y,:,column),A,view(x,:,column),t,p,work)
    end
    y
end

function _apply_tensor_mode_batch_generic!(
        destination,action,source,factor::Int,dims,t,p,
        work::_CompositeFactorBatchWorkspace,columns::Int,
        adjoint_action::Bool)
    stride=1
    @inbounds for index in 1:factor-1
        stride*=dims[index]
    end
    n=dims[factor]
    outer=size(source,1)÷(stride*n)
    input=view(work.input,:,1:columns)
    output=view(work.output,:,1:columns)
    @inbounds for block in 0:outer-1,inner in 1:stride
        base=block*stride*n+inner
        for column in 1:columns,j in 1:n
            input[j,column]=source[base+(j-1)*stride,column]
        end
        if adjoint_action
            _composite_factor_apply_adjoint_batch!(
                output,action,input,t,p,work.action_workspace)
        else
            _composite_factor_apply_batch!(
                output,action,input,t,p,work.action_workspace)
        end
        for column in 1:columns,j in 1:n
            destination[base+(j-1)*stride,column]=output[j,column]
        end
    end
    destination
end

@inline function _apply_tensor_mode_batch!(
        destination,action,source,factor::Int,dims,t,p,
        work::_CompositeFactorBatchWorkspace,columns::Int)
    _apply_tensor_mode_batch_generic!(
        destination,action,source,factor,dims,t,p,work,columns,false)
end

@inline function _apply_tensor_mode_batch!(
        destination::StridedMatrix{T},action::StridedMatrix{T},
        source::StridedMatrix{T},factor::Int,dims,t,p,
        work::_CompositeFactorBatchWorkspace,columns::Int) where T
    if factor==1
        n=dims[1]
        mul!(reshape(destination,n,:),action,reshape(source,n,:))
        return destination
    end
    _apply_tensor_mode_batch_generic!(
        destination,action,source,factor,dims,t,p,work,columns,false)
end

@inline function _apply_tensor_mode_adjoint_batch!(
        destination,action,source,factor::Int,dims,t,p,
        work::_CompositeFactorBatchWorkspace,columns::Int)
    _apply_tensor_mode_batch_generic!(
        destination,action,source,factor,dims,t,p,work,columns,true)
end

@inline function _apply_tensor_mode_adjoint_batch!(
        destination::StridedMatrix{T},action::StridedMatrix{T},
        source::StridedMatrix{T},factor::Int,dims,t,p,
        work::_CompositeFactorBatchWorkspace,columns::Int) where T
    if factor==1
        n=dims[1]
        mul!(reshape(destination,n,:),adjoint(action),reshape(source,n,:))
        return destination
    end
    _apply_tensor_mode_batch_generic!(
        destination,action,source,factor,dims,t,p,work,columns,true)
end

function _checked_composite_coefficient(::Type{T},value) where T
    value isa Number||throw(ArgumentError(
        "a composite term coefficient must evaluate to a number"))
    converted=try
        T(value)
    catch error
        throw(ArgumentError("composite term coefficient is not representable in $T: "*
                            sprint(showerror,error)))
    end
    isfinite(converted)||throw(ArgumentError(
        "composite term coefficient must be finite"))
    converted==value||throw(ArgumentError(
        "composite term coefficient would be narrowed to $T; use a wider workspace"))
    converted
end

@inline function _apply_composite_actions!(source,buffer1,buffer2,
        ::Tuple{},::Tuple{},dims,factor::Int,use_second::Bool,t,p)
    source
end

@inline function _apply_composite_actions!(source,buffer1,buffer2,
        actions::Tuple{Nothing,Vararg{Any}},
        workspaces::Tuple{Nothing,Vararg{Any}},dims,factor::Int,
        use_second::Bool,t,p)
    _apply_composite_actions!(source,buffer1,buffer2,Base.tail(actions),
        Base.tail(workspaces),dims,factor+1,use_second,t,p)
end

@inline function _apply_composite_actions!(source,buffer1,buffer2,
        actions::Tuple{A,Vararg{Any}},workspaces::Tuple{W,Vararg{Any}},dims,
        factor::Int,use_second::Bool,t,p) where {A,W}
    destination=use_second ? buffer2 : buffer1
    _apply_tensor_mode!(destination,first(actions),source,factor,dims,t,p,
                        first(workspaces))
    _apply_composite_actions!(destination,buffer1,buffer2,Base.tail(actions),
        Base.tail(workspaces),dims,factor+1,!use_second,t,p)
end

@inline function _apply_composite_adjoint_actions!(
        source,buffer1,buffer2,::Tuple{},::Tuple{},dims,
        factor::Int,use_second::Bool,t,p)
    source
end

@inline function _apply_composite_adjoint_actions!(
        source,buffer1,buffer2,
        actions::Tuple{Nothing,Vararg{Any}},
        workspaces::Tuple{Nothing,Vararg{Any}},dims,factor::Int,
        use_second::Bool,t,p)
    _apply_composite_adjoint_actions!(
        source,buffer1,buffer2,Base.tail(actions),
        Base.tail(workspaces),dims,factor+1,use_second,t,p)
end

@inline function _apply_composite_adjoint_actions!(
        source,buffer1,buffer2,
        actions::Tuple{A,Vararg{Any}},
        workspaces::Tuple{W,Vararg{Any}},dims,factor::Int,
        use_second::Bool,t,p) where {A,W}
    destination=use_second ? buffer2 : buffer1
    _apply_tensor_mode_adjoint!(
        destination,first(actions),source,factor,dims,t,p,
        first(workspaces))
    _apply_composite_adjoint_actions!(
        destination,buffer1,buffer2,Base.tail(actions),
        Base.tail(workspaces),dims,factor+1,!use_second,t,p)
end

@inline _apply_composite_batch_actions!(
    source,buffer1,buffer2,::Tuple{},::Tuple{},dims,
    factor::Int,use_second::Bool,t,p,columns::Int)=source

@inline function _apply_composite_batch_actions!(
        source,buffer1,buffer2,
        actions::Tuple{Nothing,Vararg{Any}},
        workspaces::Tuple{Nothing,Vararg{Any}},dims,factor::Int,
        use_second::Bool,t,p,columns::Int)
    _apply_composite_batch_actions!(
        source,buffer1,buffer2,Base.tail(actions),
        Base.tail(workspaces),dims,factor+1,use_second,t,p,columns)
end

@inline function _apply_composite_batch_actions!(
        source,buffer1,buffer2,
        actions::Tuple{A,Vararg{Any}},
        workspaces::Tuple{W,Vararg{Any}},dims,factor::Int,
        use_second::Bool,t,p,columns::Int) where {A,W}
    destination=use_second ? buffer2 : buffer1
    _apply_tensor_mode_batch!(
        destination,first(actions),source,factor,dims,t,p,
        first(workspaces),columns)
    _apply_composite_batch_actions!(
        destination,buffer1,buffer2,Base.tail(actions),
        Base.tail(workspaces),dims,factor+1,!use_second,t,p,columns)
end

@inline _apply_composite_adjoint_batch_actions!(
    source,buffer1,buffer2,::Tuple{},::Tuple{},dims,
    factor::Int,use_second::Bool,t,p,columns::Int)=source

@inline function _apply_composite_adjoint_batch_actions!(
        source,buffer1,buffer2,
        actions::Tuple{Nothing,Vararg{Any}},
        workspaces::Tuple{Nothing,Vararg{Any}},dims,factor::Int,
        use_second::Bool,t,p,columns::Int)
    _apply_composite_adjoint_batch_actions!(
        source,buffer1,buffer2,Base.tail(actions),
        Base.tail(workspaces),dims,factor+1,use_second,t,p,columns)
end

@inline function _apply_composite_adjoint_batch_actions!(
        source,buffer1,buffer2,
        actions::Tuple{A,Vararg{Any}},
        workspaces::Tuple{W,Vararg{Any}},dims,factor::Int,
        use_second::Bool,t,p,columns::Int) where {A,W}
    destination=use_second ? buffer2 : buffer1
    _apply_tensor_mode_adjoint_batch!(
        destination,first(actions),source,factor,dims,t,p,
        first(workspaces),columns)
    _apply_composite_adjoint_batch_actions!(
        destination,buffer1,buffer2,Base.tail(actions),
        Base.tail(workspaces),dims,factor+1,!use_second,t,p,columns)
end

@inline function _apply_composite_term!(y,S,x,term,termwork,t,p,
                                        buffer1,buffer2)
    source=_apply_composite_actions!(x,buffer1,buffer2,term.actions,
        termwork.factors,S.basis.dimensions,1,false,t,p)
    coefficient=_checked_composite_coefficient(eltype(y),
        value_at(term.coefficient,t,p))
    @inbounds @simd for i in eachindex(y,source)
        y[i]+=coefficient*source[i]
    end
    y
end

@inline function _apply_composite_adjoint_term!(
        y,S,x,term,termwork,t,p,buffer1,buffer2)
    source=_apply_composite_adjoint_actions!(
        x,buffer1,buffer2,term.actions,termwork.factors,
        S.basis.dimensions,1,false,t,p)
    coefficient=_checked_composite_coefficient(
        eltype(y),conj(value_at(term.coefficient,t,p)))
    @inbounds @simd for i in eachindex(y,source)
        y[i]+=coefficient*source[i]
    end
    y
end


@inline _apply_composite_terms!(y,S,x,::Tuple{},::Tuple{},t,p,
                                buffer1,buffer2)=y
@inline function _apply_composite_terms!(y,S,x,
        terms::Tuple{Term,Vararg{Any}},
        workspaces::Tuple{Work,Vararg{Any}},t,p,buffer1,buffer2) where
        {Term,Work}
    _apply_composite_term!(y,S,x,first(terms),first(workspaces),t,p,
                           buffer1,buffer2)
    _apply_composite_terms!(y,S,x,Base.tail(terms),Base.tail(workspaces),
                            t,p,buffer1,buffer2)
end

@inline _apply_composite_adjoint_terms!(
    y,S,x,::Tuple{},::Tuple{},t,p,buffer1,buffer2)=y
@inline function _apply_composite_adjoint_terms!(
        y,S,x,terms::Tuple{Term,Vararg{Any}},
        workspaces::Tuple{Work,Vararg{Any}},t,p,buffer1,buffer2) where
        {Term,Work}
    _apply_composite_adjoint_term!(
        y,S,x,first(terms),first(workspaces),t,p,buffer1,buffer2)
    _apply_composite_adjoint_terms!(
        y,S,x,Base.tail(terms),Base.tail(workspaces),
        t,p,buffer1,buffer2)
end

@inline function _apply_composite_batch_term!(
        y,S,x,term,termwork,t,p,buffer1,buffer2,columns::Int)
    source=_apply_composite_batch_actions!(
        x,buffer1,buffer2,term.actions,termwork.factors,
        S.basis.dimensions,1,false,t,p,columns)
    coefficient=_checked_composite_coefficient(
        eltype(y),value_at(term.coefficient,t,p))
    @inbounds for column in 1:columns
        @simd for index in axes(y,1)
            y[index,column]+=coefficient*source[index,column]
        end
    end
    y
end

@inline function _apply_composite_adjoint_batch_term!(
        y,S,x,term,termwork,t,p,buffer1,buffer2,columns::Int)
    source=_apply_composite_adjoint_batch_actions!(
        x,buffer1,buffer2,term.actions,termwork.factors,
        S.basis.dimensions,1,false,t,p,columns)
    coefficient=_checked_composite_coefficient(
        eltype(y),conj(value_at(term.coefficient,t,p)))
    @inbounds for column in 1:columns
        @simd for index in axes(y,1)
            y[index,column]+=coefficient*source[index,column]
        end
    end
    y
end

@inline _apply_composite_batch_terms!(
    y,S,x,::Tuple{},::Tuple{},t,p,buffer1,buffer2,columns::Int)=y
@inline function _apply_composite_batch_terms!(
        y,S,x,terms::Tuple{Term,Vararg{Any}},
        workspaces::Tuple{Work,Vararg{Any}},t,p,buffer1,buffer2,
        columns::Int) where {Term,Work}
    _apply_composite_batch_term!(
        y,S,x,first(terms),first(workspaces),t,p,
        buffer1,buffer2,columns)
    _apply_composite_batch_terms!(
        y,S,x,Base.tail(terms),Base.tail(workspaces),t,p,
        buffer1,buffer2,columns)
end

@inline _apply_composite_adjoint_batch_terms!(
    y,S,x,::Tuple{},::Tuple{},t,p,buffer1,buffer2,columns::Int)=y
@inline function _apply_composite_adjoint_batch_terms!(
        y,S,x,terms::Tuple{Term,Vararg{Any}},
        workspaces::Tuple{Work,Vararg{Any}},t,p,buffer1,buffer2,
        columns::Int) where {Term,Work}
    _apply_composite_adjoint_batch_term!(
        y,S,x,first(terms),first(workspaces),t,p,
        buffer1,buffer2,columns)
    _apply_composite_adjoint_batch_terms!(
        y,S,x,Base.tail(terms),Base.tail(workspaces),t,p,
        buffer1,buffer2,columns)
end

"""
    apply!(destination, S, source, time, parameters, workspace)
    apply!(destination, S, source, workspace)

Apply a sum-of-Kronecker composite superoperator without forming any
Kronecker matrix.  The explicit workspace is preallocated and task-owned.
Source and destination must not alias.
"""
function apply!(y::AbstractVector,S::CompositeSuperoperator,x::AbstractVector,
                t,p,work::CompositeSuperoperatorWorkspace)
    n=length(S.basis)
    length(x)==n&&length(y)==n||throw(DimensionMismatch(
        "composite superoperator vector has the wrong length"))
    Base.mightalias(y,x)&&throw(ArgumentError(
        "composite apply! requires nonaliasing source and destination"))
    work.superoperator===S||throw(ArgumentError(
        "composite workspace belongs to a different superoperator"))
    length(work.buffer1)==n&&length(work.buffer2)==n||throw(DimensionMismatch(
        "composite workspace has the wrong dimension"))
    length(work.terms)==length(S.terms)||throw(DimensionMismatch(
        "composite workspace belongs to a different term plan"))
    required=promote_type(eltype(S),eltype(x))
    promote_type(eltype(y),required)==eltype(y)||throw(ArgumentError(
        "destination element type $(eltype(y)) would narrow composite output $required"))
    eltype(work.buffer1)==eltype(y)||throw(ArgumentError(
        "composite workspace and destination scalar types differ"))
    fill!(y,zero(eltype(y)))
    _apply_composite_terms!(y,S,x,S.terms,work.terms,t,p,
                            work.buffer1,work.buffer2)
    y
end

"""
    apply_adjoint!(destination, S, source, time, parameters, workspace)
    apply_adjoint!(destination, S, source, workspace)

Apply the adjoint of a factorized [`CompositeSuperoperator`](@ref) without
forming a Kronecker matrix. Matrix factors use their adjoint maps and prepared
PI factors delegate to their allocation-conscious `apply_adjoint!` methods.
The scalar coefficient of every product term is conjugated. Reuse the same
task-owned [`CompositeSuperoperatorWorkspace`](@ref) used by forward
application.
"""
function apply_adjoint!(
        y::AbstractVector,S::CompositeSuperoperator,x::AbstractVector,
        t,p,work::CompositeSuperoperatorWorkspace)
    n=length(S.basis)
    length(x)==n&&length(y)==n||throw(DimensionMismatch(
        "composite superoperator vector has the wrong length"))
    Base.mightalias(y,x)&&throw(ArgumentError(
        "composite apply_adjoint! requires nonaliasing source and destination"))
    work.superoperator===S||throw(ArgumentError(
        "composite workspace belongs to a different superoperator"))
    length(work.buffer1)==n&&length(work.buffer2)==n||throw(DimensionMismatch(
        "composite workspace has the wrong dimension"))
    length(work.terms)==length(S.terms)||throw(DimensionMismatch(
        "composite workspace belongs to a different term plan"))
    required=promote_type(eltype(S),eltype(x))
    promote_type(eltype(y),required)==eltype(y)||throw(ArgumentError(
        "destination element type $(eltype(y)) would narrow composite adjoint output $required"))
    eltype(work.buffer1)==eltype(y)||throw(ArgumentError(
        "composite workspace and destination scalar types differ"))
    fill!(y,zero(eltype(y)))
    _apply_composite_adjoint_terms!(
        y,S,x,S.terms,work.terms,t,p,work.buffer1,work.buffer2)
    y
end

function _check_composite_batch_application(
        y::AbstractMatrix,S::CompositeSuperoperator,x::AbstractMatrix,
        work::CompositeSuperoperatorBatchWorkspace,operation::AbstractString)
    n=length(S.basis)
    size(x,1)==n&&size(y)==size(x)||throw(DimensionMismatch(
        "composite $operation matrix has incompatible dimensions"))
    Base.mightalias(y,x)&&throw(ArgumentError(
        "composite $operation requires nonaliasing source and destination"))
    work.superoperator===S||throw(ArgumentError(
        "composite batch workspace belongs to a different superoperator"))
    columns=size(x,2)
    columns<=work.capacity||throw(ArgumentError(
        "composite batch has $columns columns but workspace capacity is "*
        "$(work.capacity); construct a larger CompositeSuperoperatorBatchWorkspace"))
    size(work.buffer1)==(n,work.capacity)&&
        size(work.buffer2)==(n,work.capacity)||
        throw(DimensionMismatch(
            "composite batch workspace has incompatible full-coordinate buffers"))
    length(work.terms)==length(S.terms)||throw(DimensionMismatch(
        "composite batch workspace belongs to a different term plan"))
    required=promote_type(eltype(S),eltype(x))
    promote_type(eltype(y),required)==eltype(y)||throw(ArgumentError(
        "destination element type $(eltype(y)) would narrow composite output $required"))
    eltype(work.buffer1)==eltype(y)||throw(ArgumentError(
        "composite batch workspace and destination scalar types differ"))
    columns
end

"""
    apply!(destination, S, source, time, parameters,
           workspace::CompositeSuperoperatorBatchWorkspace)

Apply a composite generator to several right-hand sides through fixed-capacity
task-owned scratch. Factor schedules and term coefficients are evaluated once
per tensor fiber and batch rather than once per right-hand side.
"""
function apply!(
        y::AbstractMatrix,S::CompositeSuperoperator,x::AbstractMatrix,
        t,p,work::CompositeSuperoperatorBatchWorkspace)
    columns=_check_composite_batch_application(
        y,S,x,work,"apply!")
    fill!(y,zero(eltype(y)))
    buffer1=view(work.buffer1,:,1:columns)
    buffer2=view(work.buffer2,:,1:columns)
    _apply_composite_batch_terms!(
        y,S,x,S.terms,work.terms,t,p,buffer1,buffer2,columns)
    y
end

"""
    apply_adjoint!(destination, S, source, time, parameters,
                   workspace::CompositeSuperoperatorBatchWorkspace)

Apply the exact adjoint to several right-hand sides using the same
fixed-capacity batched tensor-fiber layout as [`apply!`](@ref).
"""
function apply_adjoint!(
        y::AbstractMatrix,S::CompositeSuperoperator,x::AbstractMatrix,
        t,p,work::CompositeSuperoperatorBatchWorkspace)
    columns=_check_composite_batch_application(
        y,S,x,work,"apply_adjoint!")
    fill!(y,zero(eltype(y)))
    buffer1=view(work.buffer1,:,1:columns)
    buffer2=view(work.buffer2,:,1:columns)
    _apply_composite_adjoint_batch_terms!(
        y,S,x,S.terms,work.terms,t,p,buffer1,buffer2,columns)
    y
end


# Let the generic fixed-step evolution layer discover and reuse composite
# tensor-mode scratch without teaching the read-only superoperator about
# mutable storage.
_linear_operator_workspace(S::CompositeSuperoperator)=
    CompositeSuperoperatorWorkspace(S)
_linear_operator_batch_workspace(
    S::CompositeSuperoperator,columns::Integer,::Type{T}) where T=
    CompositeSuperoperatorBatchWorkspace(S;capacity=columns,T)

const _CompositeMatrixFreeLiouvillian=
    MatrixFreeLiouvillian{F,T,V,P,W,K,A,B,C} where
        {F,T,V,P,W<:CompositeSuperoperatorWorkspace,K,A,B,C}

# `composite_matrixfree` is a plan-less compatibility wrapper, but its retained
# vector workspace still identifies the exact immutable composite plan.  Let
# prepared consumers recover fresh task-owned vector or fixed-capacity matrix
# scratch instead of falling back to the synchronized columnwise callbacks.
function _linear_operator_workspace(
        source::_CompositeMatrixFreeLiouvillian)
    CompositeSuperoperatorWorkspace(
        source.workspace.superoperator;T=eltype(source))
end

function _linear_operator_batch_workspace(
        source::_CompositeMatrixFreeLiouvillian,
        columns::Integer,::Type{T}) where T
    CompositeSuperoperatorBatchWorkspace(
        source.workspace.superoperator;capacity=columns,T)
end

function apply!(
        y::AbstractVector,source::_CompositeMatrixFreeLiouvillian,
        x::AbstractVector,t,p,work::CompositeSuperoperatorWorkspace)
    apply!(y,source.workspace.superoperator,x,t,p,work)
end

function apply!(
        y::AbstractMatrix,source::_CompositeMatrixFreeLiouvillian,
        x::AbstractMatrix,t,p,work::CompositeSuperoperatorBatchWorkspace)
    apply!(y,source.workspace.superoperator,x,t,p,work)
end

function apply_adjoint!(
        y::AbstractVector,source::_CompositeMatrixFreeLiouvillian,
        x::AbstractVector,t,p,work::CompositeSuperoperatorWorkspace)
    apply_adjoint!(y,source.workspace.superoperator,x,t,p,work)
end

function apply_adjoint!(
        y::AbstractMatrix,source::_CompositeMatrixFreeLiouvillian,
        x::AbstractMatrix,t,p,work::CompositeSuperoperatorBatchWorkspace)
    apply_adjoint!(y,source.workspace.superoperator,x,t,p,work)
end

const _CompositeEvolutionOperator=Union{AbstractMatrix,
    MatrixFreeLiouvillian,CompositeSuperoperator}

function evolve!(destination::CompositePIState,L::_CompositeEvolutionOperator,
                 source::CompositePIState,
                 tspan;kwargs...)
    destination.basis===source.basis||throw(ArgumentError(
        "source and destination use incompatible composite bases"))
    L isa CompositeSuperoperator&&L.basis!==source.basis&&throw(ArgumentError(
        "composite state and superoperator use incompatible bases"))
    evolve!(destination.data,L,source.data,tspan;kwargs...)
    destination
end

"""Return the composite state obtained by propagating `rho` over `tspan`."""
function time_evolve(L::_CompositeEvolutionOperator,rho::CompositePIState,
                     tspan;kwargs...)
    output=copy(rho)
    evolve!(output,L,rho,tspan;kwargs...)
end
time_evolve(rho::CompositePIState,L::_CompositeEvolutionOperator,
            tspan;kwargs...)=
    time_evolve(L,rho,tspan;kwargs...)

"""Return composite states at ordered times using one reusable workspace."""
function time_evolution(L::_CompositeEvolutionOperator,rho::CompositePIState,
                        times;
                        steps_per_interval::Integer=64,parameters=nothing)
    ts=collect(times)
    isempty(ts)&&return CompositePIState[]
    all(diff(ts).>=0)||throw(ArgumentError("times must be nondecreasing"))
    steps_per_interval>0||throw(ArgumentError(
        "steps_per_interval must be positive"))
    current=copy(rho)
    workspace=EvolutionWorkspace(L,current.data)
    output=typeof(current)[copy(current)]
    for index in 2:length(ts)
        ts[index]==ts[index-1]||evolve!(current,L,current,
            (ts[index-1],ts[index]);steps=steps_per_interval,
            parameters=parameters,workspace=workspace)
        push!(output,copy(current))
    end
    output
end
time_evolution(rho::CompositePIState,L::_CompositeEvolutionOperator,
               times;kwargs...)=
    time_evolution(L,rho,times;kwargs...)

# These methods are defined before the generic high-level fallbacks are
# included, and make memory estimators work directly on composite objects.
pi_dimension(b::CompositePIBasis)=length(b)
pi_dimension(A::AbstractCompositePIOperator)=length(A.data)
pi_dimension(S::CompositeSuperoperator)=length(S.basis)

function apply!(y,S::CompositeSuperoperator,x,
                work::CompositeSuperoperatorWorkspace)
    isautonomous(S)||throw(ArgumentError(
        "autonomous apply! was requested for a driven composite superoperator"))
    apply!(y,S,x,0.0,nothing,work)
end

function apply_adjoint!(y,S::CompositeSuperoperator,x,
                        work::CompositeSuperoperatorWorkspace)
    isautonomous(S)||throw(ArgumentError(
        "autonomous apply_adjoint! was requested for a driven composite superoperator"))
    apply_adjoint!(y,S,x,0.0,nothing,work)
end

function apply!(y::AbstractMatrix,S::CompositeSuperoperator,x::AbstractMatrix,
                work::CompositeSuperoperatorBatchWorkspace)
    isautonomous(S)||throw(ArgumentError(
        "autonomous batched apply! was requested for a driven composite superoperator"))
    apply!(y,S,x,0.0,nothing,work)
end

function apply_adjoint!(
        y::AbstractMatrix,S::CompositeSuperoperator,x::AbstractMatrix,
        work::CompositeSuperoperatorBatchWorkspace)
    isautonomous(S)||throw(ArgumentError(
        "autonomous batched apply_adjoint! was requested for a driven composite superoperator"))
    apply_adjoint!(y,S,x,0.0,nothing,work)
end

function mul!(y,S::CompositeSuperoperator,x)
    isautonomous(S)||throw(ArgumentError(
        "mul! requires an autonomous composite superoperator"))
    if x isa AbstractMatrix
        apply!(y,S,x,CompositeSuperoperatorBatchWorkspace(S,x))
    else
        apply!(y,S,x,CompositeSuperoperatorWorkspace(S,x))
    end
end
*(S::CompositeSuperoperator,x::AbstractVector)=
    mul!(similar(x,promote_type(eltype(S),eltype(x)),length(S.basis)),S,x)
*(S::CompositeSuperoperator,x::AbstractMatrix)=
    mul!(similar(x,promote_type(eltype(S),eltype(x)),
                 length(S.basis),size(x,2)),S,x)

function _pi_factor_superoperator(A::PIOperator,B::PIOperator,kind::Symbol)
    _samebasis(A,B)
    T=promote_type(eltype(A.data),eltype(B.data))
    rows=Int[];columns=Int[];values=T[]
    b=A.basis
    for p in b.sectors
        R=_real_float_type(T)
        # These are internal contractions.  Fuse the inverse multiplicity
        # scale with the stored coefficient block instead of imposing the
        # public `physical_block` requirement that sqrt(f^nu) itself fit in R.
        left=_divide_by_schur_multiplicity_scale(
            Matrix(coefficient_block(A,p)),R,p)
        sparse_left=sparse(left)
        block = if kind===:left
            left_superoperator(sparse_left)
        elseif kind===:right
            right_superoperator(sparse_left)
        else
            right=_divide_by_schur_multiplicity_scale(
                Matrix(coefficient_block(B,p)),R,p)
            sandwich_superoperator(sparse_left,sparse(right))
        end
        s=b.index[p];offset=b.offsets[s]-1
        I,J,V=findnz(sparse(block))
        append!(rows,I.+offset);append!(columns,J.+offset);append!(values,V)
    end
    sparse(rows,columns,values,length(b),length(b))
end

"""Return the factor-coordinate map `X -> A*X` without a full PI Hilbert space."""
factor_left_superoperator(b::PIBasis,A::PIOperator)=
    (A.basis===b||throw(ArgumentError("operator and PI factor bases differ"));
     _pi_factor_superoperator(A,A,:left))
function factor_left_superoperator(b::FiniteOperatorBasis,A::AbstractMatrix)
    size(A)==(b.d,b.d)||throw(DimensionMismatch("finite operator has the wrong size"))
    left_superoperator(A)
end

"""Return the factor-coordinate map `X -> X*A` without a full PI Hilbert space."""
factor_right_superoperator(b::PIBasis,A::PIOperator)=
    (A.basis===b||throw(ArgumentError("operator and PI factor bases differ"));
     _pi_factor_superoperator(A,A,:right))
function factor_right_superoperator(b::FiniteOperatorBasis,A::AbstractMatrix)
    size(A)==(b.d,b.d)||throw(DimensionMismatch("finite operator has the wrong size"))
    right_superoperator(A)
end

"""Return the factor-coordinate map `X -> A*X*B'`."""
factor_sandwich_superoperator(b::PIBasis,A::PIOperator,B::PIOperator=A)=
    (A.basis===b&&B.basis===b||throw(ArgumentError("operator and PI factor bases differ"));
     _pi_factor_superoperator(A,B,:sandwich))
function factor_sandwich_superoperator(b::FiniteOperatorBasis,A::AbstractMatrix,
                                       B::AbstractMatrix=A)
    size(A)==(b.d,b.d)&&size(B)==(b.d,b.d)||throw(DimensionMismatch(
        "finite operators have the wrong size"))
    sandwich_superoperator(A,B)
end

function _factor_operator_pairs(b::CompositePIBasis,pairs)
    seen=falses(length(b.factors));result=Pair{Int,Any}[]
    for pair in pairs
        i=Int(first(pair));1<=i<=length(b.factors)||throw(BoundsError(b.factors,i))
        seen[i]&&throw(ArgumentError("factor $i is specified more than once"))
        seen[i]=true;push!(result,i=>last(pair))
    end
    result
end

_scaled_composite_coefficient(rate,scale)=rate isa Number ? scale*rate :
    ((t,p)->scale*value_at(rate,t,p))

function _validated_composite_real_rate(rate,kind::AbstractString)
    if rate isa Number
        isreal(rate)||throw(ArgumentError("a $kind rate must be real"))
        return real(rate)
    end
    (t,p)->begin
        evaluated=value_at(rate,t,p)
        evaluated isa Number&&isreal(evaluated)||throw(ArgumentError(
            "a $kind rate must evaluate to a real number"))
        real(evaluated)
    end
end

_factor_operator_ishermitian(::PIBasis,A::PIOperator)=ishermitian(A)
_factor_operator_ishermitian(::FiniteOperatorBasis,A::AbstractMatrix)=
    LinearAlgebra.ishermitian(A)
_factor_operator_ishermitian(factor,A)=false

"""
    composite_hamiltonian_superoperator(
        basis, factor=>operator...; rate=1, check=true)

Matrix-free commutator generated by a tensor-product Hamiltonian on the named
factors.  It is represented as two products of factor left/right maps.
"""
function composite_hamiltonian_superoperator(b::CompositePIBasis,pairs::Pair...;
                                             rate=1,check::Bool=true)
    ops=_factor_operator_pairs(b,pairs);isempty(ops)&&throw(ArgumentError(
        "a composite Hamiltonian needs at least one nonidentity factor"))
    if check
        for pair in ops
            i=first(pair)
            _factor_operator_ishermitian(b.factors[i],last(pair))||
                throw(ArgumentError(
                    "Hamiltonian operator on factor $i must be Hermitian"))
        end
    end
    resolved_rate=_validated_composite_real_rate(rate,"Hamiltonian")
    leftpairs=map(pair->first(pair)=>factor_left_superoperator(
        b.factors[first(pair)],last(pair)),ops)
    rightpairs=map(pair->first(pair)=>factor_right_superoperator(
        b.factors[first(pair)],last(pair)),ops)
    left=factorized_superoperator_term(b,leftpairs...;
        coefficient=_scaled_composite_coefficient(resolved_rate,-im))
    right=factorized_superoperator_term(b,rightpairs...;
        coefficient=_scaled_composite_coefficient(resolved_rate,im))
    CompositeSuperoperator(b,left,right)
end

_factor_product(A::PIOperator)=adjoint(A)*A
_factor_product(A::AbstractMatrix)=adjoint(A)*A

function _validated_composite_dissipator_rate(rate)
    _validated_composite_real_rate(rate,"dissipator")
end

"""
    composite_dissipator_superoperator(basis, factor=>jump...; rate=1)

Matrix-free Lindblad dissipator for a tensor-product jump on the named
factors.  The gain and two anticommutator pieces are three products of factor
sandwich/left/right maps.
"""
function composite_dissipator_superoperator(b::CompositePIBasis,pairs::Pair...;
                                            rate=1)
    ops=_factor_operator_pairs(b,pairs);isempty(ops)&&throw(ArgumentError(
        "a composite jump needs at least one nonidentity factor"))
    resolved_rate=_validated_composite_dissipator_rate(rate)
    gainpairs=map(pair->first(pair)=>factor_sandwich_superoperator(
        b.factors[first(pair)],last(pair)),ops)
    products=map(pair->first(pair)=>_factor_product(last(pair)),ops)
    leftpairs=map(pair->first(pair)=>factor_left_superoperator(
        b.factors[first(pair)],last(pair)),products)
    rightpairs=map(pair->first(pair)=>factor_right_superoperator(
        b.factors[first(pair)],last(pair)),products)
    gain=factorized_superoperator_term(b,gainpairs...;coefficient=resolved_rate)
    lossleft=factorized_superoperator_term(b,leftpairs...;
        coefficient=_scaled_composite_coefficient(resolved_rate,-1//2))
    lossright=factorized_superoperator_term(b,rightpairs...;
        coefficient=_scaled_composite_coefficient(resolved_rate,-1//2))
    CompositeSuperoperator(b,gain,lossleft,lossright)
end

"""
    composite_matrixfree(S; T=eltype(S))

Wrap a composite sum as the package's synchronized
[`MatrixFreeLiouvillian`](@ref).  The wrapper owns one compatibility
workspace; explicit parallel hot loops should call [`apply!`](@ref) with one
`CompositeSuperoperatorWorkspace` per task instead. Prepared package
consumers, including `sensitivity_problem`, recover fresh task-owned vector or
fixed-capacity batch scratch from the wrapper. Bare compatibility matrix
products retain the synchronized column fallback.
"""
function composite_matrixfree(S::CompositeSuperoperator;T=eltype(S))
    resolved_type=_composite_coordinate_type(T)
    workspace=CompositeSuperoperatorWorkspace(S;T=resolved_type)
    action! = (y,x,t,p)->apply!(y,S,x,t,p,workspace)
    adjoint_action! =
        (y,x,t,p)->apply_adjoint!(y,S,x,t,p,workspace)
    MatrixFreeLiouvillian(length(S.basis),action!,resolved_type,
        composite_trace_vector(S.basis;T=_real_float_type(resolved_type));
        autonomous=isautonomous(S),workspace,adjoint_action!)
end

_performance_composite_factor_action_bytes(
    ::Nothing,dimension,::Type{T}) where T=big(0)
_performance_composite_factor_action_bytes(
    ::AbstractMatrix,dimension,::Type{T}) where T=big(0)
_performance_composite_factor_action_bytes(
    action::LiouvillianPlan,dimension,::Type{T}) where T=
    _performance_source_action_bytes(action,T)
_performance_composite_factor_action_bytes(
    action::CompiledPIModel,dimension,::Type{T}) where T=
    _performance_source_action_bytes(action,T)
_performance_composite_factor_action_bytes(
    action::MatrixFreeLiouvillian,dimension,::Type{T}) where T=
    _performance_source_action_bytes(action,T)
_performance_composite_factor_action_bytes(
    action,dimension,::Type{T}) where T=
    _performance_array_bytes(dimension,T,0;linear_arrays=16)

function _performance_composite_workspace_bytes(
        superoperator::CompositeSuperoperator,::Type{T},
        columns::Integer) where T
    columns>0||throw(ArgumentError(
        "composite workspace column count must be positive"))
    total=_performance_entries_bytes(
        2*BigInt(length(superoperator.basis))*columns,T)
    for term in superoperator.terms
        for (action,dimension) in
                zip(term.actions,superoperator.basis.dimensions)
            action===nothing&&continue
            total+=_performance_entries_bytes(
                2*BigInt(dimension)*columns,T)
            total+=_performance_linear_operator_workspace_bytes(
                action;batch_columns=columns)
        end
    end
    total
end

function _performance_linear_operator_workspace_bytes(
        superoperator::CompositeSuperoperator;
        batch_columns::Integer=0)
    batch_columns>=0||throw(ArgumentError(
        "batch_columns must be nonnegative"))
    columns=batch_columns==0 ? 1 : batch_columns
    _performance_composite_workspace_bytes(
        superoperator,eltype(superoperator),columns)
end

function _performance_linear_operator_workspace_bytes(
        source::_CompositeMatrixFreeLiouvillian;
        batch_columns::Integer=0)
    batch_columns>=0||throw(ArgumentError(
        "batch_columns must be nonnegative"))
    columns=batch_columns==0 ? 1 : batch_columns
    _performance_composite_workspace_bytes(
        source.workspace.superoperator,eltype(source),columns)
end

function _performance_composite_action_bytes(
        superoperator::CompositeSuperoperator,::Type{T}) where T
    total=big(0)
    for term in superoperator.terms
        for (action,dimension) in
                zip(term.actions,superoperator.basis.dimensions)
            total+=_performance_composite_factor_action_bytes(
                action,dimension,T)
        end
    end
    total
end

# A composite compatibility wrapper already retains its complete tensor-mode
# workspace. Iterative solvers charge only any nested fallback action
# transient, rather than a second generic full-coordinate callback allowance.
function _performance_source_action_bytes(
        source::MatrixFreeLiouvillian{
            F,T,V,P,W,K,A,B,C},::Type{S}) where
        {F,T,V,P,W<:CompositeSuperoperatorWorkspace,K,A,B,C,S}
    _performance_composite_action_bytes(
        source.workspace.superoperator,S)
end

_operator_requires_matrixfree(
    ::MatrixFreeLiouvillian{
        F,T,V,P,W,K,A,B,C}) where
    {F,T,V,P,W<:CompositeSuperoperatorWorkspace,K,A,B,C}=true
