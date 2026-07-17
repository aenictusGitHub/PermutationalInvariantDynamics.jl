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
end

CompositePIOperator(b::CompositePIBasis;T=Float64)=
    CompositePIOperator(b,zeros(Complex{T},length(b)))
CompositePIState(b::CompositePIBasis;T=Float64)=
    CompositePIState(b,zeros(Complex{T},length(b)))
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
    CompositePIOperator(b,_composite_tensor_data(b,components,false))
end

"""
    composite_tensor_state(basis, components...)

Construct a factorized composite state from one [`PIState`](@ref) per PI
factor and one density matrix per finite factor.  Inputs are copied and are
not normalized implicitly.
"""
function composite_tensor_state(b::CompositePIBasis,components...)
    CompositePIState(b,_composite_tensor_data(b,components,true))
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

"""Normalize a composite state by its physical trace, without other repair."""
function normalize!(rho::CompositePIState)
    z=trace(rho)
    iszero(z)&&throw(ArgumentError("cannot normalize a zero-trace composite state"))
    rho.data./=z
    rho
end

"""Return `tr(rho^2)` from the orthonormal composite coordinates."""
purity(rho::CompositePIState)=real(dot(rho.data,rho.data))

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
function _composite_factor_apply!(y,A,x,t,p,work)
    apply!(y,A,x,t,p)
end

function _apply_tensor_mode!(destination,action,source,factor::Int,dims,
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


# Let the generic fixed-step evolution layer discover and reuse composite
# tensor-mode scratch without teaching the read-only superoperator about
# mutable storage.
_liouvillian_workspace(S::CompositeSuperoperator)=
    CompositeSuperoperatorWorkspace(S)

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

function mul!(y,S::CompositeSuperoperator,x)
    isautonomous(S)||throw(ArgumentError(
        "mul! requires an autonomous composite superoperator"))
    apply!(y,S,x,CompositeSuperoperatorWorkspace(S,x))
end
*(S::CompositeSuperoperator,x::AbstractVector)=
    mul!(similar(x,promote_type(eltype(S),eltype(x)),length(S.basis)),S,x)

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
`CompositeSuperoperatorWorkspace` per task instead.
"""
function composite_matrixfree(S::CompositeSuperoperator;T=eltype(S))
    resolved_type=_composite_coordinate_type(T)
    workspace=CompositeSuperoperatorWorkspace(S;T=resolved_type)
    action! = (y,x,t,p)->apply!(y,S,x,t,p,workspace)
    MatrixFreeLiouvillian(length(S.basis),action!,resolved_type,
        composite_trace_vector(S.basis;T=_real_float_type(resolved_type));
        autonomous=isautonomous(S))
end
