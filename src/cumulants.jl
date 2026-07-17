# Exact permutation-symmetric local moments and a dependency-neutral bridge to
# higher-order cumulant-closure packages.  The numerical path below contracts
# Appendix-D collective blocks directly with multiplicity-weighted state
# blocks.  Its exponential object is the requested local order, d^k, never the
# full-system Hilbert dimension d^N.

const _CUMULANT_BRIDGE_SCHEMA_VERSION = v"1.0.0"

"""
    OrderedLocalMoments

Exact distinct-particle local moments of a `PIState`, grouped by permutation
symmetry.  `table[:x, :y]` represents
`tr(rho * X^(1) * Y^(2))`; particle labels are distinct and the lookup order
is immaterial for a PI state.  The stored dictionary therefore contains one
canonical multiset key instead of all permutations of that key.

The table records moments only in the PI representation.  Its setup scales
with the requested order through `d^k` local tensors and Appendix-D Schur
geometry, not with the full Hilbert dimension `d^N`.
"""
struct OrderedLocalMoments{T,O,V,I}
    N::Int
    d::Int
    max_order::Int
    includes_lower_orders::Bool
    labels::Vector{Symbol}
    operators::O
    values::V
    label_indices::I
end

eltype(::Type{<:OrderedLocalMoments{T}}) where T=T
eltype(::OrderedLocalMoments{T}) where T=T
length(table::OrderedLocalMoments)=length(table.values)
Base.keys(table::OrderedLocalMoments)=keys(table.values)

function show(io::IO,table::OrderedLocalMoments)
    print(io,"OrderedLocalMoments(N=$(table.N), d=$(table.d), order=$(table.max_order), moments=$(length(table)))")
end

function _canonical_moment_key(table::OrderedLocalMoments,key)
    labels = if key isa Symbol
        (key,)
    elseif key isa Tuple
        key
    else
        try
            Tuple(key)
        catch error
            throw(ArgumentError("a moment key must be a Symbol or an iterable of Symbols: $(sprint(showerror,error))"))
        end
    end
    all(label->label isa Symbol,labels)||throw(ArgumentError(
        "every local-moment label must be a Symbol"))
    all(label->haskey(table.label_indices,label),labels)||throw(KeyError(
        first(label for label in labels if !haskey(table.label_indices,label))))
    Tuple(sort!(collect(labels);by=label->table.label_indices[label]))
end

getindex(table::OrderedLocalMoments,key::Tuple)=
    table.values[_canonical_moment_key(table,key)]
getindex(table::OrderedLocalMoments,labels::Symbol...)=
    table.values[_canonical_moment_key(table,labels)]
function Base.haskey(table::OrderedLocalMoments,key)
    canonical=try
        _canonical_moment_key(table,key)
    catch error
        error isa KeyError&&return false
        rethrow()
    end
    haskey(table.values,canonical)
end

# The package tensor convention makes particle one the fastest index.  If a
# permutation-symmetric tensor on q-1 sites is stored in `result`, then
# kron(Aq,result) places Aq on site q.  Averaging the q swaps (q,r) inserts Aq
# in every possible position and inductively gives the uniform average over
# all q! permutations.  This costs q tensor permutations rather than q! tensor
# products and treats repeated operators without a special case.
function _particle_swap_permutation(p::Int,d::Int,left::Int,right::Int,D::Int)
    1<=left<=p&&1<=right<=p||throw(BoundsError())
    left==right&&return collect(1:D)
    stride_left=d^(left-1);stride_right=d^(right-1)
    permutation=Vector{Int}(undef,D)
    @inbounds for index in 0:D-1
        a=(index÷stride_left)%d;b=(index÷stride_right)%d
        permutation[index+1]=index+1+(b-a)*stride_left+(a-b)*stride_right
    end
    permutation
end

function _symmetric_local_tensor(operators::AbstractVector{<:AbstractMatrix},
                                 d::Int,::Type{C}) where C
    isempty(operators)&&return reshape(C[one(C)],1,1)
    result=Matrix{C}(operators[1])
    R=_real_float_type(C)
    for order in 2:length(operators)
        base=kron(Matrix{C}(operators[order]),result)
        dimension=size(base,1)
        averaged=zeros(C,dimension,dimension)
        @inbounds for position in 1:order
            if position==order
                averaged .+= base
            else
                permutation=_particle_swap_permutation(
                    order,d,position,order,dimension)
                for column in 1:dimension,row in 1:dimension
                    averaged[permutation[row],permutation[column]]+=base[row,column]
                end
            end
        end
        scale=_prepare_exact_scale(R,one(BigInt),big(order),Val(false);
            context="normalization of a symmetrized local-moment tensor")
        result=scale.direct ? (averaged .*= scale.factor) :
            _apply_prepared_exact_scale(averaged,scale;
                context="normalization of a symmetrized local-moment tensor")
    end
    result
end

@inline function _trace_product(A::AbstractMatrix,B::AbstractMatrix)
    size(A)==size(B)||throw(DimensionMismatch("Schur blocks must have equal sizes"))
    value=zero(promote_type(eltype(A),eltype(B)))
    @inbounds for column in axes(A,2),row in axes(A,1)
        value+=A[row,column]*B[column,row]
    end
    value
end

# Divide the exact Appendix-D path weight by binomial(N,k) before converting or
# multiplying it.  Forming the extensive collective block first can overflow
# even when the requested local moment is O(1).  This normalized variant keeps
# the final contraction bounded and retains the guarded-wide cancellation path
# used by the production p-body block builder.
function _wide_normalized_pbody_block(cache::PBodyGeometry,X,
                                      sector::Partition,subset_count::BigInt,
                                      ::Type{R}) where R<:AbstractFloat
    W=R===Float16 ? Float64 : BigFloat
    evaluate=() -> begin
        Xwide=Matrix{Complex{W}}(X)
        n=length(cache.basis.patterns[_sidx(cache.basis,sector)])
        block=zeros(Complex{W},n,n)
        for path in cache.paths[sector]
            U=_path_isometry(path,W)
            exact_scale=cache.path_weights[Tuple(path)]/subset_count
            block .+=_checked_mul_exact_ratio(
                _path_contractions(U,U,Xwide),numerator(exact_scale),
                denominator(exact_scale);
                context="wide normalized local-moment path contribution")
        end
        _convert_checked_pbody_block(R,block;
            context="normalized local-moment Schur block")
    end
    if W===BigFloat
        largest_scale_bits=maximum(cache.paths[sector];init=0) do path
            scale=cache.path_weights[Tuple(path)]/subset_count
            max(0,ndigits(numerator(scale);base=2)-
                  ndigits(denominator(scale);base=2)+1)
        end
        requested_precision=max(precision(BigFloat),256,
            precision(R)+largest_scale_bits+32)
        return setprecision(BigFloat,requested_precision) do
            evaluate()
        end
    end
    evaluate()
end

function _normalized_pbody_block(cache::PBodyGeometry{T},X,
                                 sector::Partition,
                                 subset_count::BigInt) where T
    sector_index=_sidx(cache.basis,sector)
    n=length(cache.basis.patterns[sector_index])
    block=zeros(promote_type(Complex{T},eltype(X)),n,n)
    R=_real_float_type(eltype(block))
    cancellation_check=_pbody_cancellation_possible(cache,sector,R)
    absolute_sum=cancellation_check ? zeros(R,n,n) : nothing
    for path in cache.paths[sector]
        exact_scale=cache.path_weights[Tuple(path)]/subset_count
        U=cache.isometries[Tuple(path)]
        contraction=_path_contractions(U,U,X)
        prepared_scale=_prepare_exact_scale(R,numerator(exact_scale),
            denominator(exact_scale),Val(false);
            context="normalized local-moment path contribution")
        contribution=try
            prepared_scale.direct ? prepared_scale.factor.*contraction :
                _apply_prepared_exact_scale(contraction,prepared_scale;
                    context="normalized local-moment path contribution")
        catch error
            error isa ArgumentError||rethrow()
            return _wide_normalized_pbody_block(
                cache,X,sector,subset_count,R)
        end
        block .+=contribution
        cancellation_check&&(absolute_sum .+=abs.(contribution))
    end
    if cancellation_check
        severe=any(index->!iszero(absolute_sum[index])&&
            absolute_sum[index]>R(8)*abs(block[index]),eachindex(block))
        severe&&return _wide_normalized_pbody_block(
            cache,X,sector,subset_count,R)
    end
    block
end

function _moment_scalar_type(rho::PIState,operators,cache)
    T=eltype(rho.data)
    for operator in operators
        operator isa AbstractMatrix||throw(ArgumentError(
            "every local operator must be a matrix"))
        T=promote_type(T,eltype(operator))
    end
    cache===nothing||(T=promote_type(T,Complex{geometry_scalar_type(cache)}))
    R=_real_float_type(T)
    promote_type(T,Complex{R})
end

function _ordered_local_moment(rho::PIState,
                               operators::AbstractVector{<:AbstractMatrix},
                               cache)
    order=length(operators)
    order==0&&return trace(rho)
    order<=rho.basis.N||throw(ArgumentError(
        "moment order $order exceeds N=$(rho.basis.N) distinct particles"))
    for operator in operators
        size(operator)==(rho.basis.d,rho.basis.d)||throw(DimensionMismatch(
            "each local operator must be $(rho.basis.d)×$(rho.basis.d)"))
    end
    C=_moment_scalar_type(rho,operators,cache)
    matrices=Matrix{C}[Matrix{C}(operator) for operator in operators]
    R=_real_float_type(C)
    geometry = if cache===nothing
        PBodyGeometry(rho.basis,order,R)
    else
        cache isa PBodyGeometry||throw(ArgumentError(
            "cache must be a PBodyGeometry for the requested moment order"))
        _check_pbody_geometry(cache,rho.basis,order)
    end
    local_tensor=_symmetric_local_tensor(matrices,rho.basis.d,C)
    subset_count=exact_binomial(rho.basis.N,order)
    total=zero(C)
    for sector in rho.basis.sectors
        block=_normalized_pbody_block(
            geometry,local_tensor,sector,subset_count)
        total+=_trace_product(block,_multiplicity_weighted_block(rho,sector))
    end
    total
end

"""
    ordered_local_moment(rho, operators; cache=nothing)

Return the exact PI expectation of local operators acting on an ordered tuple
of distinct particles.  A single matrix computes a one-site moment; a tuple or
vector `(A, B, ...)` computes
`tr(rho * A^(1) * B^(2) * ...)`.  Permutation invariance makes this value
independent of how the distinct particle labels are assigned, but operators
on the same particle are deliberately not combined by this API.

The implementation symmetrizes only the `d^k` local tensor, contracts
Appendix-D Schur blocks, and divides by the exact `binomial(N,k)` subset count.
It never constructs a `d^N` state.  Reuse a `PBodyGeometry(basis,k)` through
`cache` for repeated moments of the same order.
"""
ordered_local_moment(rho::PIState,operator::AbstractMatrix;cache=nothing)=
    _ordered_local_moment(rho,AbstractMatrix[operator],cache)

function ordered_local_moment(rho::PIState,operators;cache=nothing)
    operators isa AbstractMatrix&&return ordered_local_moment(rho,operators;cache)
    collected=collect(operators)
    all(operator->operator isa AbstractMatrix,collected)||throw(ArgumentError(
        "every local operator must be a matrix"))
    _ordered_local_moment(rho,AbstractMatrix[collected...],cache)
end

function _labeled_local_operators(rho::PIState,operators)
    entries=try
        operators isa Union{NamedTuple,AbstractDict} ?
            collect(pairs(operators)) : collect(operators)
    catch error
        throw(ArgumentError(
            "operators must be a NamedTuple, dictionary, or iterable of Symbol=>matrix pairs: $(sprint(showerror,error))"))
    end
    # Hash-map iteration is deliberately not part of the neutral schema.  A
    # dictionary alphabet therefore receives a deterministic symbolic order;
    # NamedTuples and explicit pair iterables retain the caller's order.
    operators isa AbstractDict&&sort!(entries;by=entry->string(first(entry)))
    labels=Symbol[];raw=AbstractMatrix[];seen=Set{Symbol}()
    for entry in entries
        entry isa Pair||throw(ArgumentError(
            "operators must be an iterable of Symbol=>matrix pairs"))
        label=first(entry);label isa Symbol||throw(ArgumentError(
            "local-operator labels must be Symbols"))
        label in seen&&throw(ArgumentError("duplicate local-operator label $label"))
        operator=last(entry);operator isa AbstractMatrix||throw(ArgumentError(
            "operator $label must be a matrix"))
        size(operator)==(rho.basis.d,rho.basis.d)||throw(DimensionMismatch(
            "operator $label must be $(rho.basis.d)×$(rho.basis.d)"))
        push!(seen,label);push!(labels,label);push!(raw,operator)
    end
    labels,raw
end

function _canonical_multiset_keys!(destination,labels,order,start,prefix)
    if length(prefix)==order
        push!(destination,Tuple(prefix));return destination
    end
    for index in start:length(labels)
        push!(prefix,labels[index])
        _canonical_multiset_keys!(destination,labels,order,index,prefix)
        pop!(prefix)
    end
    destination
end

"""
    ordered_local_moments(rho, operators; order, include_lower=true)

Extract all exact permutation-symmetric distinct-particle moments over a
labeled local-operator alphabet.  `operators` may be a named tuple, dictionary,
or iterable of `Symbol => matrix` pairs.  With `include_lower=true`, orders
one through `order` are returned; otherwise only the selected order is
returned.  For `order=0`, the sole empty-key value is `trace(rho)`.

Only one canonical key is evaluated for each operator multiset.  Consequently
an alphabet of size `m` requires `binomial(m+k-1,k)` contractions at exact
order `k`, rather than `m^k`, and one `PBodyGeometry` is shared by every
moment of that order.  The largest local scratch is `d^order` by `d^order`;
there is no full-Hilbert reconstruction.
"""
function ordered_local_moments(rho::PIState,operators;
                               order::Integer,include_lower::Bool=true)
    order>=0||throw(ArgumentError("order must be nonnegative"))
    order<=rho.basis.N||throw(ArgumentError(
        "moment order $order exceeds N=$(rho.basis.N) distinct particles"))
    labels,raw=_labeled_local_operators(rho,operators)
    order>0&&isempty(labels)&&throw(ArgumentError(
        "at least one labeled local operator is required for a positive order"))
    C=_moment_scalar_type(rho,raw,nothing)
    stored=Dict{Symbol,Matrix{C}}(
        label=>Matrix{C}(operator) for (label,operator) in zip(labels,raw))
    values=Dict{Tuple{Vararg{Symbol}},C}()
    if order==0
        values[()]=C(trace(rho))
    else
        first_order=include_lower ? 1 : Int(order)
        R=_real_float_type(C)
        for current_order in first_order:Int(order)
            geometry=PBodyGeometry(rho.basis,current_order,R)
            moment_keys=Tuple{Vararg{Symbol}}[]
            _canonical_multiset_keys!(
                moment_keys,labels,current_order,1,Symbol[])
            for key in moment_keys
                matrices=AbstractMatrix[stored[label] for label in key]
                values[key]=C(_ordered_local_moment(rho,matrices,geometry))
            end
        end
    end
    indices=Dict(label=>index for (index,label) in pairs(labels))
    OrderedLocalMoments{C,typeof(stored),typeof(values),typeof(indices)}(
        rho.basis.N,rho.basis.d,Int(order),include_lower,
        copy(labels),stored,values,indices)
end

"""
    CumulantTermPayload

Dependency-neutral metadata for one microscopic `AbstractPITerm`.  `process`,
`scope`, and `order` follow the public term hooks.  `operator` is a detached
fixed operator, an in-place schedule prototype, or `nothing` when an
unevaluated allocating schedule has no prototype; `operator_schedule` retains
that schedule.  Direct PI terms have `microscopic=false` because their Schur
blocks do not determine a unique local cumulant hierarchy.
"""
struct CumulantTermPayload{O,S,R,H,E}
    process::Symbol
    scope::Symbol
    order::Int
    operator::O
    operator_schedule::S
    rate::R
    hbar::H
    microscopic::Bool
    operator_time_dependent::Bool
    rate_time_dependent::Bool
    evaluated_at::E
end

"""
    CumulantModelPayload

Versioned, dependency-neutral description of a `PIModel` for an external
cumulant-closure adapter.  It records `N`, `d`, copied microscopic operator
prototypes, rate specifications, body orders, tensor-index ordering, and the
package Hamiltonian/dissipator conventions.  No symbolic package types are
stored in the core payload.
"""
struct CumulantModelPayload{Terms,Metadata}
    schema_version::VersionNumber
    N::Int
    d::Int
    terms::Terms
    metadata::Metadata
end

"""
    CumulantBridgePayload

Combined versioned model metadata and exact [`OrderedLocalMoments`](@ref) for
initialization, validation, or convergence checks of an order-`k` cumulant
closure.
"""
struct CumulantBridgePayload{M,Q,Metadata}
    schema_version::VersionNumber
    model::M
    moments::Q
    metadata::Metadata
end

_val_symbol(::Val{value}) where value=value

function _detached_payload_operator(operator)
    operator===nothing&&return nothing
    operator isa Tuple&&return map(_detached_payload_operator,operator)
    applicable(copy,operator)||throw(ArgumentError(
        "a cumulant payload operator of type $(typeof(operator)) must support copy"))
    copy(operator)
end

function _cumulant_term_payload(term::AbstractPITerm,time,parameters)
    process=_val_symbol(term_process(term));scope=_val_symbol(term_scope(term))
    raw_operator=term_operator(term);raw_rate=term_rate(term)
    operator_time_dependent=raw_operator isa Function
    rate_time_dependent=raw_rate isa Function
    if time===nothing
        if raw_operator isa InPlaceTimeOperator
            operator=_detached_payload_operator(raw_operator.prototype)
            operator_schedule=raw_operator
        elseif operator_time_dependent
            operator=nothing;operator_schedule=raw_operator
        else
            operator=_detached_payload_operator(raw_operator)
            operator_schedule=nothing
        end
        rate=raw_rate;evaluated_at=nothing
    else
        evaluated=value_at(raw_operator,time,parameters)
        evaluated isa Function&&throw(ArgumentError(
            "the operator schedule for $(typeof(term)) did not evaluate to a fixed operator"))
        operator=_detached_payload_operator(evaluated)
        operator_schedule=nothing
        rate=value_at(raw_rate,time,parameters)
        rate isa Number||throw(ArgumentError(
            "the rate schedule for $(typeof(term)) did not evaluate to a number"))
        evaluated_at=time
    end
    hbar=process==:hamiltonian&&applicable(term_hbar,term) ? term_hbar(term) : nothing
    CumulantTermPayload(process,scope,Int(body_order(term)),operator,
        operator_schedule,rate,hbar,scope!=:direct,
        operator_time_dependent,rate_time_dependent,evaluated_at)
end

"""
    cumulant_model_payload(model; time=nothing, parameters=nothing)

Export the microscopic content of a `PIModel` into the versioned neutral
`CumulantModelPayload` schema.  With no `time`, fixed operators are copied,
`InPlaceTimeOperator` prototypes are copied while their schedules are retained,
and allocating operator/rate schedules remain unevaluated.  Supplying `time`
evaluates every schedule with the package `value_at(x,time,parameters)`
convention and requires fixed numerical results.

The payload is intended for package-specific adapters which build automatic
order-`k` closures.  A term with `microscopic=false` is a direct Schur-space
term and must not be assigned a guessed local realization.
"""
function cumulant_model_payload(model::PIModel;time=nothing,parameters=nothing)
    terms=map(term->_cumulant_term_payload(term,time,parameters),model.terms)
    metadata=(tensor_index_order=:particle_one_fastest,
              local_matrix_indices=:one_based,
              subset_sum_convention=:unordered_distinct_subsets,
              ordered_moment_convention=:distinct_particle_falling_factorial,
              dissipator_convention=:LrhoLdagger_minus_half_anticommutator,
              evaluated=time!==nothing,
              evaluated_at=time)
    CumulantModelPayload(_CUMULANT_BRIDGE_SCHEMA_VERSION,
        model.basis.N,model.basis.d,terms,metadata)
end

"""
    cumulant_bridge_payload(model, rho, operators; order,
                            include_lower=true, time=nothing,
                            parameters=nothing)

Create one versioned object containing neutral microscopic model metadata and
exact PI local moments through the requested closure order.  `rho` and
`model` must own the identical `PIBasis`; this prevents a payload from mixing
different sector restrictions or conventions.
"""
function cumulant_bridge_payload(model::PIModel,rho::PIState,operators;
                                 order::Integer,include_lower::Bool=true,
                                 time=nothing,parameters=nothing)
    model.basis===rho.basis||throw(ArgumentError(
        "model and state must use the exact same PIBasis"))
    model_payload=cumulant_model_payload(model;time,parameters)
    moments=ordered_local_moments(rho,operators;order,include_lower)
    metadata=(moment_order=Int(order),
              includes_lower_orders=include_lower,
              exact_pi_reference=true)
    CumulantBridgePayload(_CUMULANT_BRIDGE_SCHEMA_VERSION,
        model_payload,moments,metadata)
end

"""
    CumulantComparison

Error summary returned by [`compare_cumulant_closure`](@ref).  `rows` retains
the exact value, closure value, absolute error, relative error, and tolerance
decision for every compared canonical moment key.
"""
struct CumulantComparison{R,Rows}
    count::Int
    maximum_absolute_error::R
    root_mean_square_error::R
    maximum_relative_error::R
    within_tolerance::Bool
    rows::Rows
end

function show(io::IO,result::CumulantComparison)
    print(io,"CumulantComparison(count=$(result.count), max_error=$(result.maximum_absolute_error), within_tolerance=$(result.within_tolerance))")
end

function _closure_moment(closure::OrderedLocalMoments,key)
    closure[key]
end
function _closure_moment(closure::AbstractDict,key)
    haskey(closure,key)&&return closure[key]
    length(key)==1&&haskey(closure,only(key))&&return closure[only(key)]
    throw(KeyError(key))
end
function _closure_moment(closure,key)
    applicable(closure,key)&&return closure(key)
    applicable(closure,key...)&&return closure(key...)
    throw(ArgumentError(
        "closure output must be an OrderedLocalMoments table, dictionary, or callable accepting moment keys"))
end

"""
    compare_cumulant_closure(exact, closure; atol=nothing, rtol=nothing)

Compare exact PI moments with a closure output.  `closure` may be another
`OrderedLocalMoments`, a dictionary keyed by canonical label tuples (a single
`Symbol` is also accepted for first order), or a callable.  Missing moments
raise instead of being silently omitted.  Defaults are `atol=0` and
`rtol=sqrt(eps(R))` in the promoted real scalar type `R`.

The per-row test is `abs(closure-exact) <= atol + rtol*abs(exact)`.  Relative
error is measured against `abs(exact)` and is `Inf` for a nonzero error at an
exact zero.
"""
function compare_cumulant_closure(exact::OrderedLocalMoments,closure;
                                  atol=nothing,rtol=nothing)
    moment_keys=collect(keys(exact.values))
    sort!(moment_keys;by=key->(length(key),
        Tuple(exact.label_indices[label] for label in key)))
    closure_values=Any[]
    value_type=eltype(exact)
    for key in moment_keys
        value=_closure_moment(closure,key)
        value isa Number||throw(ArgumentError(
            "closure value for $key must be numeric"))
        push!(closure_values,value)
        value_type=promote_type(value_type,typeof(value))
    end
    initial_real=_real_float_type(value_type)
    atol_value=atol===nothing ? zero(initial_real) : atol
    rtol_value=rtol===nothing ? sqrt(eps(initial_real)) : rtol
    R=promote_type(initial_real,typeof(float(atol_value)),typeof(float(rtol_value)))
    absolute_tolerance=R(atol_value);relative_tolerance=R(rtol_value)
    isfinite(absolute_tolerance)&&absolute_tolerance>=zero(R)||throw(ArgumentError(
        "atol must be finite and nonnegative"))
    isfinite(relative_tolerance)&&relative_tolerance>=zero(R)||throw(ArgumentError(
        "rtol must be finite and nonnegative"))
    C=promote_type(value_type,Complex{R})
    rows=NamedTuple[]
    maximum_absolute_error=zero(R);sum_squares=zero(R)
    maximum_relative_error=zero(R);all_within=true
    for (index,key) in pairs(moment_keys)
        reference=C(exact.values[key]);prediction=C(closure_values[index])
        error=R(abs(prediction-reference))
        relative_error=iszero(reference) ?
            (iszero(error) ? zero(R) : R(Inf)) : error/R(abs(reference))
        within=error<=absolute_tolerance+relative_tolerance*R(abs(reference))
        push!(rows,(labels=key,exact=reference,closure=prediction,
                    absolute_error=error,relative_error=relative_error,
                    within_tolerance=within))
        maximum_absolute_error=max(maximum_absolute_error,error)
        sum_squares+=abs2(error)
        maximum_relative_error=max(maximum_relative_error,relative_error)
        all_within&=within
    end
    rms=isempty(rows) ? zero(R) : sqrt(sum_squares/R(length(rows)))
    CumulantComparison(length(rows),maximum_absolute_error,rms,
        maximum_relative_error,all_within,rows)
end

"""
    quantumcumulants_initial_values(moments, symbolic_map)

Map neutral exact moment keys to user-supplied QuantumCumulants.jl symbolic
averages.  This optional function becomes available after
`import QuantumCumulants`; the package extension validates every symbolic
order with QuantumCumulants' public `get_order` API and returns a dictionary
suitable for assembling initial values.

`symbolic_map` maps a local label tuple (or a single first-order `Symbol`) to
the corresponding symbolic average.  Symbolic Hamiltonian/jump construction
is intentionally left to the adapter or research script because it depends on
the selected QuantumCumulants Hilbert and index spaces.  The core package does
not guess that mapping and does not depend on QuantumCumulants.
"""
function quantumcumulants_initial_values(moments::OrderedLocalMoments,
                                          symbolic_map)
    extension=Base.get_extension(@__MODULE__,
        :PermutationalInvariantDynamicsQuantumCumulantsExt)
    extension===nothing&&throw(ArgumentError(
        "QuantumCumulants is not loaded; run `import QuantumCumulants` before requesting symbolic initial values"))
    extension.quantumcumulants_initial_values(moments,symbolic_map)
end

"""
    quantumcumulants_model(model; order=2, time=nothing, parameters=nothing,
                           complete=false, scale=false, kwargs...)

Automatically lower the microscopic terms of a [`PIModel`](@ref) to indexed
SecondQuantizedAlgebra operators and construct QuantumCumulants equations.
This optional method becomes available after `import QuantumCumulants`; the
core package retains no symbolic-algebra dependency.

One-body Hamiltonians and local or collective jumps are lowered directly.
Permutation-symmetric `p`-body terms use pairwise-distinct symbolic indices
and the exact `1/factorial(p)` conversion from ordered tuples to the package's
unordered-subset convention. Fixed correlated one-body baths are factorized
into independent symbolic jump channels. Direct PI terms are rejected because
Schur blocks do not specify a unique microscopic realization. Time-dependent
operators, rates, or Kossakowski matrices require an explicit `time` (and
optional `parameters`) at which they can be evaluated.

The returned named tuple retains the symbolic Hilbert space, summation and
probe indices, Hamiltonian, jumps, rates, seed operators, equations, and
lowering metadata. By default the equations contain the requested seed
moments but are not recursively completed or permutation-scaled; set
`complete=true` and/or `scale=true` explicitly. Extra keywords are forwarded
to the optional adapter, including `seed_operators`, `space_name`,
`transition_name`, and `ground_state`.

This is a cumulant closure, not exact finite-`N` PI dynamics. Validate its
selected order against [`ordered_local_moments`](@ref) or exact PI evolution.
"""
function quantumcumulants_model(model::PIModel;kwargs...)
    extension=Base.get_extension(@__MODULE__,
        :PermutationalInvariantDynamicsQuantumCumulantsExt)
    extension===nothing&&throw(ArgumentError(
        "QuantumCumulants is not loaded; run `import QuantumCumulants` before requesting automatic symbolic lowering"))
    extension.quantumcumulants_model(model;kwargs...)
end
