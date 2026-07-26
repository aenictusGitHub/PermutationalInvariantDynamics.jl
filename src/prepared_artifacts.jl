"""
    PreparedGeometryBundle

Immutable, shareable collection of expensive representation data prepared for
one exact [`PIBasis`](@ref). A bundle may contain a [`OneBoxCGCache`](@ref), a
complete [`OneBodyGeometry`](@ref), selected [`PBodyGeometry`](@ref) objects,
and one [`ReductionPlanSet`](@ref). It never contains mutable numerical
workspaces, model callbacks, solver state, or locks.

Use [`prepare_geometry`](@ref) for one bundle or
[`prepare_geometry!`](@ref) with a user-owned [`PreparationCache`](@ref) when
the same request may recur.
"""
struct PreparedGeometryBundle{T,B,L,Q,C,O,P,G,R,E}
    basis::B
    basis_layout::L
    scalar_type::Type{T}
    precision_bits::Int
    rounding_mode::Q
    coefficient_cache::C
    one_body::O
    pbody_orders::P
    pbody_geometries::G
    reduction_plans::R
    estimates::E
end

function Base.show(io::IO,bundle::PreparedGeometryBundle)
    reductions=bundle.reduction_plans
    reduction_count=reductions===nothing ? 0 : length(reductions.plans)
    print(io,
        "PreparedGeometryBundle(N=$(bundle.basis.N), d=$(bundle.basis.d), " *
        "T=$(bundle.scalar_type), one_body=$(bundle.one_body!==nothing), " *
        "pbody_orders=$(bundle.pbody_orders), reductions=$reduction_count, " *
        "retained_bytes=$(bundle.estimates.retained_bytes))")
end

function _prepared_basis_layout(basis::PIBasis)
    # PIBasis is an immutable shell around mutable vectors. The ordinary
    # prepared APIs require exact object identity; retaining this exact layout
    # snapshot additionally makes accidental mutation fail before reuse.
    (N=basis.N,
     d=basis.d,
     sectors=Tuple(partition.parts for partition in basis.sectors),
     patterns=Tuple(Tuple(pattern.entries for pattern in patterns)
                    for patterns in basis.patterns),
     offsets=Tuple(basis.offsets),
     index=Tuple((partition.parts,get(basis.index,partition,nothing))
                 for partition in basis.sectors))
end

function _prepared_basis_layout_matches(layout,basis::PIBasis)
    layout.N==basis.N&&layout.d==basis.d||return false
    length(layout.sectors)==length(basis.sectors)||return false
    length(layout.patterns)==length(basis.patterns)||return false
    length(layout.offsets)==length(basis.offsets)||return false
    length(layout.index)==length(basis.index)||return false
    @inbounds for index in eachindex(basis.sectors)
        layout.sectors[index]==basis.sectors[index].parts||return false
        stored_patterns=layout.patterns[index]
        current_patterns=basis.patterns[index]
        length(stored_patterns)==length(current_patterns)||return false
        for pattern_index in eachindex(current_patterns)
            stored_patterns[pattern_index]==
                current_patterns[pattern_index].entries||return false
        end
    end
    @inbounds for index in eachindex(basis.offsets)
        layout.offsets[index]==basis.offsets[index]||return false
    end
    @inbounds for index in eachindex(basis.sectors)
        partition=basis.sectors[index]
        layout.index[index]==
            (partition.parts,get(basis.index,partition,nothing))||return false
    end
    true
end

function _check_prepared_basis_layout(bundle::PreparedGeometryBundle,
                                      basis::PIBasis)
    bundle.basis===basis||throw(ArgumentError(
        "PreparedGeometryBundle belongs to a different PIBasis object; " *
        "prepare a bundle for this exact basis"))
    _prepared_basis_layout_matches(bundle.basis_layout,basis)||throw(ArgumentError(
        "the PIBasis layout changed after PreparedGeometryBundle " *
        "construction; rebuild the basis and its prepared bundle"))
    bundle
end

function _check_prepared_arithmetic(bundle::PreparedGeometryBundle{T}) where T
    if T===BigFloat
        precision(BigFloat)==bundle.precision_bits||throw(ArgumentError(
            "PreparedGeometryBundle was constructed at BigFloat precision " *
            "$(bundle.precision_bits), but the active precision is " *
            "$(precision(BigFloat)); rebuild or enter the matching precision"))
        rounding(BigFloat)==bundle.rounding_mode||throw(ArgumentError(
            "PreparedGeometryBundle was constructed with BigFloat rounding " *
            "mode $(bundle.rounding_mode), but the active mode is " *
            "$(rounding(BigFloat)); rebuild or enter the matching mode"))
    end
    bundle
end

"""
    validate_prepared_geometry(bundle[, basis])

Validate exact basis ownership, the retained basis layout, scalar precision,
and every contained geometry/plan link. The check does not allocate numerical
scratch or modify the bundle.
"""
function validate_prepared_geometry(bundle::PreparedGeometryBundle,
                                    basis::PIBasis=bundle.basis)
    _check_prepared_basis_layout(bundle,basis)
    _check_prepared_arithmetic(bundle)
    length(bundle.pbody_orders)==length(bundle.pbody_geometries)||throw(
        DimensionMismatch(
            "PreparedGeometryBundle has inconsistent p-body order and " *
            "geometry storage"))
    all(order->order isa Integer&&!(order isa Bool)&&1<=order<=basis.N,
        bundle.pbody_orders)||throw(ArgumentError(
            "PreparedGeometryBundle p-body orders must be integers in 1:N"))
    allunique(bundle.pbody_orders)||throw(ArgumentError(
        "PreparedGeometryBundle p-body orders must be unique"))
    coefficient_cache=bundle.coefficient_cache
    if coefficient_cache!==nothing
        required_depth=max(
            bundle.one_body===nothing ? 0 : min(1,basis.N),
            maximum(bundle.pbody_orders;init=0))
        _check_onebox_coefficient_cache(
            coefficient_cache,basis,required_depth,bundle.scalar_type)
    end
    if bundle.one_body!==nothing
        _check_geometry_basis(bundle.one_body,basis)
        geometry_scalar_type(bundle.one_body)===bundle.scalar_type||throw(
            ArgumentError("one-body geometry scalar type is inconsistent " *
                          "with its PreparedGeometryBundle"))
    end
    for (order,geometry) in
            zip(bundle.pbody_orders,bundle.pbody_geometries)
        _check_pbody_geometry(geometry,basis,order)
        geometry_scalar_type(geometry)===bundle.scalar_type||throw(
            ArgumentError("p-body geometry scalar type is inconsistent " *
                          "with its PreparedGeometryBundle"))
    end
    reductions=bundle.reduction_plans
    if reductions!==nothing
        reductions.basis===basis||throw(ArgumentError(
            "ReductionPlanSet belongs to a different PIBasis object"))
        _check_reduction_plan_set(reductions)
    end
    bundle
end

function _preparation_budget(memory_budget)
    memory_budget isa Real&&!(memory_budget isa Bool)||throw(ArgumentError(
        "memory_budget must be a nonnegative number of bytes or Inf"))
    isnan(memory_budget)&&throw(ArgumentError(
        "memory_budget must not be NaN"))
    memory_budget>=0||throw(ArgumentError(
        "memory_budget must be nonnegative"))
    isinf(memory_budget)&&return nothing
    try
        floor(BigInt,memory_budget)
    catch error
        throw(ArgumentError(
            "memory_budget is not a representable byte count: " *
            sprint(showerror,error)))
    end
end

function _check_preparation_budget(required::BigInt,budget,
                                   operation::AbstractString)
    budget===nothing&&return required
    required<=budget||throw(ArgumentError(
        "$operation requires an estimated $required retained/peak bytes, " *
        "exceeding memory_budget=$budget; reduce the selected body orders " *
        "or bipartitions, restrict the basis, raise memory_budget, or pass " *
        "memory_budget=Inf explicitly"))
    required
end

function _prepared_orders(basis::PIBasis,orders)
    prepared=Int[]
    for order in orders
        order isa Integer&&!(order isa Bool)||throw(ArgumentError(
            "p-body orders must be integers"))
        1<=order<=basis.N||throw(ArgumentError(
            "each p-body order must satisfy 1 ≤ p ≤ N=$(basis.N)"))
        push!(prepared,Int(order))
    end
    allunique(prepared)||throw(ArgumentError(
        "p-body orders must be unique"))
    Tuple(prepared)
end

function _prepared_reduction_ks(basis::PIBasis,ks)
    ks===nothing&&return nothing
    prepared=Int[]
    for k in ks
        k isa Integer&&!(k isa Bool)||throw(ArgumentError(
            "reduction subsystem sizes must be integers"))
        0<=k<=basis.N||throw(ArgumentError(
            "each reduction size must satisfy 0 ≤ k ≤ N=$(basis.N)"))
        push!(prepared,Int(k))
    end
    isempty(prepared)&&return nothing
    allunique(prepared)||throw(ArgumentError(
        "reduction subsystem sizes must be unique"))
    Tuple(prepared)
end

function _prepared_scalar_context(::Type{T}) where T
    T<:AbstractFloat&&isconcretetype(T)||throw(ArgumentError(
        "T must be a concrete AbstractFloat type, got $T"))
    precision_bits=T===BigFloat ? precision(BigFloat) : precision(T)
    rounding_mode=T===BigFloat ? rounding(BigFloat) : nothing
    precision_bits,rounding_mode
end

function _preparation_request(basis::PIBasis,::Type{T};
        one_body::Bool=true,pbody_orders=(),
        reduction_ks=nothing,reduction_atol::Real=2e-11,
        coefficient_cache=:auto) where T
    precision_bits,rounding_mode=_prepared_scalar_context(T)
    orders=_prepared_orders(basis,pbody_orders)
    ks=_prepared_reduction_ks(basis,reduction_ks)
    tolerance=_reduction_set_tolerance(reduction_atol)
    depth=max(one_body ? min(1,basis.N) : 0,
              maximum(orders;init=0))
    cache_mode=if coefficient_cache===:auto
        :auto
    elseif coefficient_cache===nothing
        :none
    elseif coefficient_cache isa OneBoxCGCache
        _check_onebox_coefficient_cache(
            coefficient_cache,basis,depth,T)
        (:external,objectid(coefficient_cache))
    else
        throw(ArgumentError(
            "coefficient_cache must be :auto, nothing, or a " *
            "basis-owned OneBoxCGCache"))
    end
    signature=(
        scalar_type=T,
        precision_bits,
        rounding_mode,
        one_body,
        pbody_orders=orders,
        reduction_ks=ks,
        reduction_atol=tolerance,
        coefficient_cache=cache_mode)
    (;signature,orders,ks,depth,precision_bits,rounding_mode,
      reduction_atol=tolerance,coefficient_cache)
end

function _known_geometry_estimates(basis::PIBasis,::Type{T},request) where T
    precision_bits=request.precision_bits
    coefficient_retained=if request.depth==0||
            request.coefficient_cache===nothing
        big(0)
    elseif request.coefficient_cache===:auto
        _estimate_onebox_cache_upper(
            basis,request.depth,T;precision_bits)
    else
        BigInt(Base.summarysize(request.coefficient_cache))
    end
    onebody=request.signature.one_body ?
        _estimate_onebody_geometry(
            basis,T;bigfloat_precision=precision_bits) : nothing
    pbody=Tuple(_estimate_pbody_geometry(
        basis,order,T;bigfloat_precision=precision_bits)
        for order in request.orders)

    retained=coefficient_retained
    setup_peak=coefficient_retained
    if onebody!==nothing
        setup_peak=max(setup_peak,retained+onebody.setup_bytes)
        retained+=onebody.retained_bytes
    end
    for estimate in pbody
        setup_peak=max(setup_peak,retained+estimate.setup_bytes)
        retained+=estimate.retained_bytes
    end
    (coefficient_retained_bytes=coefficient_retained,
     onebody=onebody,
     pbody=pbody,
     known_retained_upper_bytes=retained,
     known_setup_peak_bytes=setup_peak,
     reduction_setup_preflight=request.ks===nothing ? :not_requested :
         :not_available)
end

function _build_prepared_geometry(basis::PIBasis,::Type{T},request;
                                  memory_budget=512*1024^2) where T
    budget=_preparation_budget(memory_budget)
    known=_known_geometry_estimates(basis,T,request)
    _check_preparation_budget(
        known.known_setup_peak_bytes,budget,
        "PreparedGeometryBundle setup")
    if request.ks!==nothing&&basis.d>2&&budget!==nothing
        throw(ArgumentError(
            "PreparedGeometryBundle cannot preflight the transient sparse-" *
            "QR/nullspace setup required by qudit ReductionPlanSet. Pass " *
            "memory_budget=Inf explicitly after assessing that setup, or " *
            "prepare the one- and p-body geometry without reduction_ks."))
    end

    coefficient_cache=if request.depth==0||
            request.coefficient_cache===nothing
        nothing
    elseif request.coefficient_cache===:auto
        OneBoxCGCache(
            basis,T;max_depth=request.depth,
            memory_budget=budget===nothing ? Inf : budget)
    else
        request.coefficient_cache
    end
    onebody=request.signature.one_body ?
        OneBodyGeometry(basis,T;coefficient_cache) : nothing
    pbody=Tuple(PBodyGeometry(
        basis,order,T;coefficient_cache)
        for order in request.orders)

    # ReductionPlanSet currently has no allocation-free qudit SPQR setup
    # estimator. Build it only after all rigorously estimated geometry has
    # passed the command budget, then check its exact retained size before it
    # becomes part of the returned bundle or a PreparationCache entry.
    reductions=request.ks===nothing ? nothing :
        ReductionPlanSet(
            basis,request.ks;atol=request.reduction_atol)
    layout=_prepared_basis_layout(basis)
    payload=(layout,coefficient_cache,onebody,pbody,reductions)
    retained_bytes=BigInt(Base.summarysize(payload))
    _check_preparation_budget(
        retained_bytes,budget,
        "PreparedGeometryBundle retained storage")
    estimates=merge(known,(
        retained_bytes,
        pbody_orders=request.orders,
        reduction_ks=request.ks,
        precision_bits=request.precision_bits,
        scalar_type=T))
    bundle=PreparedGeometryBundle(
        basis,layout,T,request.precision_bits,request.rounding_mode,
        coefficient_cache,onebody,request.orders,pbody,reductions,estimates)
    validate_prepared_geometry(bundle,basis)
end

"""
    prepare_geometry(basis; T=Float64, one_body=true, pbody_orders=(),
                     reduction_ks=nothing, reduction_atol=2e-11,
                     coefficient_cache=:auto,
                     memory_budget=512 * 1024^2)

Prepare one immutable bundle without a store. `coefficient_cache=:auto`
retains a depth-bounded, basis-owned [`OneBoxCGCache`](@ref) shared by the
requested geometries. Pass `nothing` to use call-local coefficient
construction, or pass an existing compatible cache.

Known one-body and p-body setup peaks are checked before construction.
Retained storage, including an optional `ReductionPlanSet`, is checked before
the bundle is returned. `Inf` is the only explicit budget opt-out.
"""
function prepare_geometry(basis::PIBasis;T=Float64,one_body::Bool=true,
        pbody_orders=(),reduction_ks=nothing,reduction_atol::Real=2e-11,
        coefficient_cache=:auto,memory_budget=512*1024^2)
    request=_preparation_request(
        basis,T;one_body,pbody_orders,reduction_ks,reduction_atol,
        coefficient_cache)
    _build_prepared_geometry(basis,T,request;memory_budget)
end

"""
    onebody_geometry(bundle[, basis])

Return the complete prepared one-body geometry after validating its exact
basis and arithmetic context. Raise when the bundle was prepared without it.
"""
function onebody_geometry(bundle::PreparedGeometryBundle,
                          basis::PIBasis=bundle.basis)
    validate_prepared_geometry(bundle,basis)
    bundle.one_body===nothing&&throw(ArgumentError(
        "PreparedGeometryBundle does not contain one-body geometry"))
    bundle.one_body
end

"""
    pbody_geometry(bundle, p[, basis])

Return the prepared geometry for body order `p`, preserving the order supplied
to `prepare_geometry`.
"""
function pbody_geometry(bundle::PreparedGeometryBundle,p::Integer,
                        basis::PIBasis=bundle.basis)
    validate_prepared_geometry(bundle,basis)
    position=findfirst(==(p),bundle.pbody_orders)
    position===nothing&&throw(ArgumentError(
        "PreparedGeometryBundle does not contain p-body order $p"))
    bundle.pbody_geometries[position]
end

"""
    prepared_reductions(bundle[, basis])

Return the bundle's `ReductionPlanSet` after exact validation. Raise when no
particle reductions were requested.
"""
function prepared_reductions(bundle::PreparedGeometryBundle,
                             basis::PIBasis=bundle.basis)
    validate_prepared_geometry(bundle,basis)
    bundle.reduction_plans===nothing&&throw(ArgumentError(
        "PreparedGeometryBundle does not contain a ReductionPlanSet"))
    bundle.reduction_plans
end

"""
    PreparationCache(; memory_budget=512 * 1024^2)

Explicit, user-owned store for [`PreparedGeometryBundle`](@ref) objects.
Entries are partitioned by exact `PIBasis` object identity and by scalar,
precision, rounding, geometry-selection, and reduction-selection request.
The cache is synchronized for concurrent lookup/construction, but prepared
workspaces remain task-owned as usual.

The byte budget applies to the conservative sum of each retained bundle's
standalone `summarysize`; shared basis/cache objects may therefore be counted
more than once. This favors a predictable upper bound over hidden retention.
No process-global cache is used.
"""
mutable struct PreparationCache
    memory_budget::Union{Nothing,BigInt}
    retained_bytes::BigInt
    entries::IdDict{Any,Dict{Any,Any}}
    lock::ReentrantLock
end

function PreparationCache(;memory_budget=512*1024^2)
    PreparationCache(
        _preparation_budget(memory_budget),big(0),
        IdDict{Any,Dict{Any,Any}}(),ReentrantLock())
end

function Base.show(io::IO,cache::PreparationCache)
    entries=sum(length,values(cache.entries);init=0)
    budget=cache.memory_budget===nothing ? "Inf" :
        string(cache.memory_budget)
    print(io,
        "PreparationCache(entries=$entries, retained_bytes=" *
        "$(cache.retained_bytes), memory_budget=$budget)")
end

function _cache_remaining(cache::PreparationCache)
    cache.memory_budget===nothing&&return nothing
    max(big(0),cache.memory_budget-cache.retained_bytes)
end

"""
    prepare_geometry!(cache, basis; kwargs...)

Return the exactly matching cached bundle, or prepare and retain one new
bundle. Construction is serialized per cache so two tasks cannot duplicate an
expensive miss. Cache capacity is checked before insertion; failure leaves the
cache unchanged.
"""
function prepare_geometry!(cache::PreparationCache,basis::PIBasis;
        T=Float64,one_body::Bool=true,pbody_orders=(),
        reduction_ks=nothing,reduction_atol::Real=2e-11,
        coefficient_cache=:auto)
    request=_preparation_request(
        basis,T;one_body,pbody_orders,reduction_ks,reduction_atol,
        coefficient_cache)
    lock(cache.lock) do
        basis_entries=get(cache.entries,basis,nothing)
        if basis_entries!==nothing
            bundle=get(basis_entries,request.signature,nothing)
            if bundle!==nothing
                if request.coefficient_cache isa OneBoxCGCache
                    bundle.coefficient_cache===request.coefficient_cache||
                        throw(ArgumentError(
                            "cached preparation signature collided with a " *
                            "different external OneBoxCGCache"))
                end
                return validate_prepared_geometry(bundle,basis)
            end
        end
        remaining=_cache_remaining(cache)
        bundle=_build_prepared_geometry(
            basis,T,request;
            memory_budget=remaining===nothing ? Inf : remaining)
        bytes=BigInt(Base.summarysize(bundle))
        _check_preparation_budget(
            cache.retained_bytes+bytes,cache.memory_budget,
            "PreparationCache insertion")
        basis_entries===nothing&&begin
            basis_entries=Dict{Any,Any}()
            cache.entries[basis]=basis_entries
        end
        basis_entries[request.signature]=bundle
        cache.retained_bytes+=bytes
        bundle
    end
end

"""
    evict_prepared_geometry!(cache, bundle)

Remove the exact bundle object from `cache`. Return `true` when it was present
and `false` otherwise.
"""
function evict_prepared_geometry!(
        cache::PreparationCache,bundle::PreparedGeometryBundle)
    lock(cache.lock) do
        basis_entries=get(cache.entries,bundle.basis,nothing)
        basis_entries===nothing&&return false
        for (signature,value) in basis_entries
            value===bundle||continue
            delete!(basis_entries,signature)
            cache.retained_bytes=max(
                big(0),cache.retained_bytes-BigInt(Base.summarysize(value)))
            isempty(basis_entries)&&delete!(cache.entries,bundle.basis)
            return true
        end
        false
    end
end

"""
    clear_preparation_cache!(cache)

Drop all retained bundle references and reset accounting. Existing bundle
objects held by callers remain valid.
"""
function clear_preparation_cache!(cache::PreparationCache)
    lock(cache.lock) do
        empty!(cache.entries)
        cache.retained_bytes=big(0)
    end
    cache
end

"""
    preparation_cache_summary(cache)

Return stable cache accounting without exposing its mutable dictionaries.
"""
function preparation_cache_summary(cache::PreparationCache)
    lock(cache.lock) do
        (basis_count=length(cache.entries),
         entry_count=sum(length,values(cache.entries);init=0),
         retained_bytes=cache.retained_bytes,
         memory_budget=cache.memory_budget===nothing ?
             Inf : cache.memory_budget)
    end
end
