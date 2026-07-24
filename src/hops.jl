"""
    HOPSBath(coupling, coefficients, frequencies; label=:bath)

Finite exponential correlation for one shared Gaussian bath in the hierarchy
of pure states (HOPS),

```math
C(t)=\\sum_k c_k\\exp(-\\nu_k t),\\qquad t\\geq0.
```

`coupling` is a fixed PI operator ``L``. Unlike [`HEOMBath`](@ref), it need
not be Hermitian: the linear HOPS hierarchy uses ``L`` in its noise and
downward terms and ``L^\\dagger`` in its upward term. Every coefficient must
be finite and every pole must be finite with strictly positive real part.
The constructor copies all input data.

The bath describes one *shared* noise realization multiplying a PI operator.
Independent local colored noises are not PI path by path and cannot be
represented by this backend.
"""
struct HOPSBath{O,C,F,L}
    coupling::O
    coefficients::C
    frequencies::F
    label::L
end

function HOPSBath(coupling::PIOperator,coefficients::AbstractVector,
                  frequencies::AbstractVector;label=:bath)
    length(coefficients)==length(frequencies)||throw(DimensionMismatch(
        "HOPS bath coefficients and frequencies must have equal lengths"))
    isempty(coefficients)&&throw(ArgumentError(
        "a HOPS bath must contain at least one exponential term"))
    cs=collect(coefficients)
    nus=collect(frequencies)
    for (index,value) in pairs(cs)
        value isa Number&&_heom_isfinite(value)||throw(ArgumentError(
            "HOPS bath coefficient $index must be a finite number"))
    end
    for (index,value) in pairs(nus)
        value isa Number&&_heom_isfinite(value)||throw(ArgumentError(
            "HOPS bath frequency $index must be a finite number"))
        real(value)>0||throw(ArgumentError(
            "HOPS bath frequency $index must have strictly positive real part"))
    end
    all(_heom_isfinite,coupling.data)||throw(ArgumentError(
        "HOPS bath coupling coefficients must be finite"))
    HOPSBath{typeof(coupling),typeof(cs),typeof(nus),typeof(label)}(
        copy(coupling),cs,nus,label)
end

HOPSBath(coupling::PIOperator,coefficient::Number,frequency::Number;kwargs...)=
    HOPSBath(coupling,[coefficient],[frequency];kwargs...)

show(io::IO,bath::HOPSBath)=print(io,
    "HOPSBath(N=$(bath.coupling.basis.N), d=$(bath.coupling.basis.d), " *
    "exponentials=$(length(bath.coefficients)), label=$(bath.label))")

@inline function _hops_canonical_pole(value::Complex{R}) where R
    Complex{R}(iszero(real(value)) ? zero(R) : real(value),
               iszero(imag(value)) ? zero(R) : imag(value))
end

function _hops_aggregate_poles(coefficients,frequencies,::Type{T}) where T
    values=Dict{T,T}()
    for index in eachindex(coefficients)
        pole=_hops_canonical_pole(_heom_checked_convert(
            T,frequencies[index],"HEOM-to-HOPS correlation pole $index"))
        coefficient=_heom_checked_convert(
            T,coefficients[index],
            "HEOM-to-HOPS correlation coefficient $index")
        values[pole]=get(values,pole,zero(T))+coefficient
    end
    values
end

"""
    HOPSBath(bath::HEOMBath; check_right=true, atol=0, rtol=0,
             label=bath.metadata.model)

Convert the physical left correlation of a compatible [`HEOMBath`](@ref) to
HOPS form. A nonzero HEOM white-noise residue is rejected because the linear
colored-noise hierarchy does not unravel that time-local correction.
`check_right=true` also verifies that the HEOM same-pole right correlation is
the exact conjugate of the left correlation (after grouping duplicate poles).
Set explicit nonzero tolerances only when accepting a known roundoff-level
decomposition discrepancy.
"""
function HOPSBath(bath::HEOMBath;check_right::Bool=true,
        atol::Real=0,rtol::Real=0,
        label=get(bath.metadata,:model,:heom_bath))
    isfinite(atol)&&atol>=0||throw(ArgumentError(
        "HEOM-to-HOPS atol must be finite and nonnegative"))
    isfinite(rtol)&&rtol>=0||throw(ArgumentError(
        "HEOM-to-HOPS rtol must be finite and nonnegative"))
    iszero(bath.residue)||throw(ArgumentError(
        "an HEOM bath with a nonzero white-noise residue cannot be converted to linear HOPS; represent that Markovian correction separately"))
    if check_right
        T=_complex_float_type(promote_type(
            eltype(bath.coupling.data),
            mapreduce(typeof,promote_type,bath.coefficients),
            mapreduce(typeof,promote_type,bath.right_coefficients),
            mapreduce(typeof,promote_type,bath.frequencies)))
        left=_hops_aggregate_poles(bath.coefficients,bath.frequencies,T)
        right=_hops_aggregate_poles(
            bath.right_coefficients,bath.frequencies,T)
        poles=Set{T}()
        union!(poles,keys(left));union!(poles,keys(right))
        for pole in keys(left)
            push!(poles,_hops_canonical_pole(conj(pole)))
        end
        R=_real_float_type(T)
        for pole in poles
            actual=get(right,pole,zero(T))
            expected=conj(get(
                left,_hops_canonical_pole(conj(pole)),zero(T)))
            tolerance=R(atol)+R(rtol)*max(abs(actual),abs(expected))
            abs(actual-expected)<=tolerance||throw(ArgumentError(
                "HEOM right correlation is incompatible with the physical conjugate left correlation at pole $pole; construct HOPSBath directly only if an independent valid HOPS covariance is intended"))
        end
    end
    HOPSBath(bath.coupling,bath.coefficients,bath.frequencies;label)
end

function _hops_bath_tuple(baths)
    baths isa HOPSBath&&return (baths,)
    tuple=Tuple(baths)
    isempty(tuple)&&throw(ArgumentError("at least one HOPS bath is required"))
    all(bath->bath isa HOPSBath,tuple)||throw(ArgumentError(
        "every bath must be a HOPSBath"))
    tuple
end

function _hops_scalar_type(hamiltonian,baths)
    T=_complex_float_type(eltype(hamiltonian.data))
    for bath in baths
        T=promote_type(T,eltype(bath.coupling.data))
        for value in bath.coefficients
            T=_heom_promote_value_type(T,value)
        end
        for value in bath.frequencies
            T=_heom_promote_value_type(T,value)
        end
    end
    _complex_float_type(T)
end

function _hops_precision_context(
        hamiltonian,baths,scaling_factors,importance_cutoff,
        ::Type{T}) where T
    R=_real_float_type(T)
    R===BigFloat||return precision(R),nothing
    bits=max(precision(BigFloat),
             _supersite_array_precision(hamiltonian.data))
    for bath in baths
        bits=max(bits,_supersite_array_precision(bath.coupling.data))
        for value in bath.coefficients
            bits=max(bits,_supersite_value_precision(value))
        end
        for value in bath.frequencies
            bits=max(bits,_supersite_value_precision(value))
        end
    end
    bits=max(bits,_supersite_value_precision(importance_cutoff))
    if scaling_factors!==nothing
        for value in scaling_factors
            bits=max(bits,_supersite_value_precision(value))
        end
    end
    bits,rounding(BigFloat)
end

function _hops_with_precision(f,plan)
    _with_supersite_precision(
        f,_real_float_type(plan.Ttype),
        plan.precision_bits,plan.rounding_mode)
end

@inline _hops_value_has_precision(value,bits)=true
@inline _hops_value_has_precision(value::BigFloat,bits)=
    precision(value)==bits
@inline _hops_value_has_precision(value::Complex{BigFloat},bits)=
    precision(real(value))==bits&&precision(imag(value))==bits

function _hops_require_array_precision(
        array,plan,description::AbstractString)
    _real_float_type(plan.Ttype)===BigFloat||return array
    all(value->_hops_value_has_precision(value,plan.precision_bits),array)||
        throw(ArgumentError(
        "$description does not use the prepared BigFloat precision " *
        "$(plan.precision_bits); rebuild it in the plan precision context"))
    array
end

@inline _hops_context_value(value,::Type{T}) where T=value
@inline function _hops_context_value(
        value,::Type{Complex{BigFloat}})
    converted=Complex{BigFloat}(value)
    Complex{BigFloat}(
        real(converted)+zero(BigFloat),
        imag(converted)+zero(BigFloat))
end

function _hops_plan_bytes(basis,::Type{T},number_ados::BigInt,
        poles::Int,depth::Int,baths::Int,
        precision_bits::Integer) where T
    npi=BigInt(length(basis))
    # One Hamiltonian and one coupling per bath. Non-Hermitian adjoints use
    # transposed block application and are not retained as duplicate blocks.
    block_entries=(BigInt(1)+BigInt(baths))*npi
    multiindex_entries=number_ados*BigInt(poles)
    edge_bound=BigInt(poles)*number_ados
    # Scaling level matrices are setup transients even though only the packed
    # edge coefficients and coordinate scales survive in the plan.
    real_entries=2BigInt(depth)*BigInt(poles)+number_ados+
                 BigInt(poles)+edge_bound
    complex_entries=block_entries+3BigInt(poles)+number_ados+edge_bound
    _performance_entries_bytes(
        complex_entries,T;bigfloat_precision=precision_bits)+
        _performance_entries_bytes(
            real_entries,_real_float_type(T);
            bigfloat_precision=precision_bits)+
        # Multi-indices, packed edges/incidences, bath-edge lists, lookup keys,
        # vector headers, and conservative Dict payload/capacity. This is an
        # intentionally generous setup/retention estimate rather than an
        # object-size promise.
        (12edge_bound+8multiindex_entries+32number_ados+
         16BigInt(poles)+8BigInt(baths))*BigInt(sizeof(Int))
end

function _hops_retained_node_limit(
        basis,::Type{T},poles,depth,baths,precision_bits,
        memory_budget,nweak) where T
    coordinate_limit=typemax(Int)÷max(nweak,1)
    limit=_performance_memory_limit(memory_budget)
    limit===nothing&&return coordinate_limit
    base=_hops_plan_bytes(
        basis,T,big(0),poles,depth,baths,precision_bits)
    one_node=_hops_plan_bytes(
        basis,T,big(1),poles,depth,baths,precision_bits)
    per_node=one_node-base
    per_node>0||error("internal HOPS memory estimate is not increasing")
    limit>=one_node||_require_performance_budget(
        "HOPS plan preparation",one_node,memory_budget;
        guidance="Reduce max_depth or the number of correlation poles.")
    budget_nodes=Int(min(
        BigInt(coordinate_limit),(limit-base)÷per_node))
    max(budget_nodes,1)
end

function _hops_pruned_multiindices(
        coefficients,frequencies,max_depth::Int,cutoff,
        metric::Symbol,::Type{R},node_limit::Int) where R
    metric===:normalized_coupling_decay||throw(ArgumentError(
        "importance_metric must be :normalized_coupling_decay"))
    cutoff isa Real&&isfinite(cutoff)&&0<cutoff<=1||throw(ArgumentError(
        "a pruned HOPS importance_cutoff must lie in (0,1]"))
    cutoffR=R(cutoff)
    cutoffR==cutoff||throw(ArgumentError(
        "importance_cutoff is not exactly representable in $R"))
    K=length(coefficients)
    weights=_heom_importance_weights(
        coefficients,coefficients,frequencies,R)
    root=zeros(Int,K)
    indices=Vector{Vector{Int}}([root])
    importances=R[one(R)]
    lookup=Dict{Tuple,Int}(Tuple(root)=>1)
    frontier=Vector{Vector{Int}}([root])
    candidate_limit=BigInt(max(K,1))*BigInt(node_limit)
    for depth in 1:max_depth
        candidates=Set{Tuple}()
        for parent in frontier,term in 1:K
            candidate=copy(parent)
            candidate[term]+=1
            key=Tuple(candidate)
            if !in(key,candidates)&&BigInt(length(candidates))>=candidate_limit
                throw(ArgumentError(
                    "HOPS importance-pruning setup exceeds its memory-budget node bound; increase importance_cutoff, reduce max_depth/poles, or pass a larger memory_budget"))
            end
            push!(candidates,key)
        end
        ordered=sort!(collect(candidates))
        next_frontier=Vector{Vector{Int}}()
        for key in ordered
            candidate=collect(key)
            all_parents=true
            for term in 1:K
                iszero(candidate[term])&&continue
                parent=copy(candidate)
                parent[term]-=1
                if !haskey(lookup,Tuple(parent))
                    all_parents=false
                    break
                end
            end
            all_parents||continue
            importance=_heom_multiindex_importance(candidate,weights,R)
            importance>=cutoffR||continue
            length(indices)<node_limit||throw(ArgumentError(
                "the retained HOPS importance hierarchy exceeds the memory-budget node bound; increase importance_cutoff, reduce max_depth/poles, or pass a larger memory_budget"))
            push!(indices,candidate)
            push!(importances,importance)
            lookup[key]=length(indices)
            push!(next_frontier,candidate)
        end
        frontier=next_frontier
        isempty(frontier)&&break
    end
    indices,importances,cutoffR
end

"""
    HOPSPlan(hamiltonian, baths; max_depth, scaling=:scaled,
             scaling_factors=nothing, importance_cutoff=0,
             importance_metric=:normalized_coupling_decay,
             memory_budget=512MiB)

Prepare the linear hierarchy of pure states for a fixed Hermitian PI
Hamiltonian and one or more [`HOPSBath`](@ref)s. Each hierarchy node is a
pseudo-ket in ``directsum_nu U_nu`` and therefore has
[`weak_pi_dimension`](@ref) amplitudes rather than `length(basis)` density
coordinates.

For flattened pole index ``k`` and multi-index ``n``, the unscaled equation is

```math
\\dot\\psi_n=(-iH-n\\cdot\\nu+\\sum_bL_bz_b^*)\\psi_n
 +\\sum_k n_kc_kL_{b(k)}\\psi_{n-e_k}
 -\\sum_kL_{b(k)}^\\dagger\\psi_{n+e_k}.
```

`max_depth` is a hard zero boundary. A positive `importance_cutoff` applies
the same deterministic downward-closed heuristic as PI--HEOM and is an
additional explicit approximation. With `scaling=:scaled`, the stored
coordinate is divided by
``prod_k sqrt(n_k! a_k^n_k)``. This is an exact diagonal similarity of the
retained hierarchy; the default positive scale is `abs(c_k)`. Duplicate
equal poles within one bath are combined exactly, and an exactly cancelled
pole is removed before hierarchy construction.

The plan is read-only and shareable. Mutable hierarchy, RK4, coupling, and
colored-noise storage belongs to [`HOPSWorkspace`](@ref). The memory budget is
enforced before hierarchy enumeration; `Inf` is the explicit opt-out.
"""
struct HOPSPlan{B,H,C,V,E,I,D,G,R,M,T,Q}
    basis::B
    hamiltonian_blocks::H
    coupling_blocks::C
    coefficients::V
    frequencies::V
    exponent_baths::E
    bath_edges::Vector{Vector{Int}}
    multiindices::I
    index::D
    topology::G
    edge_downward::V
    edge_upward::Vector{R}
    ado_scales::Vector{R}
    pole_scales::Vector{R}
    decays::V
    offsets::Vector{Int}
    bath_labels::M
    ado_importances::Vector{R}
    full_ado_count::BigInt
    max_depth::Int
    nweak::Int
    Ttype::Type{T}
    scaling::Symbol
    importance_cutoff::R
    importance_metric::Symbol
    internal_noise_supported::Bool
    coupling_selfadjoint::BitVector
    precision_bits::Int
    rounding_mode::Q
end

function HOPSPlan(hamiltonian::PIOperator,baths;max_depth::Integer,
        scaling::Symbol=:scaled,scaling_factors=nothing,
        importance_cutoff::Real=0,
        importance_metric::Symbol=:normalized_coupling_decay,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,
        atol::Real=0,
        rtol=nothing)
    max_depth isa Bool&&throw(ArgumentError("max_depth must not be a Bool"))
    max_depth>=0||throw(ArgumentError("max_depth must be nonnegative"))
    BigInt(max_depth)<=typemax(Int)||throw(ArgumentError(
        "max_depth must be representable as an Int"))
    bath_tuple=_hops_bath_tuple(baths)
    basis=hamiltonian.basis
    for (number,bath) in pairs(bath_tuple)
        bath.coupling.basis===basis||throw(ArgumentError(
            "HOPS bath $number uses a different PI basis"))
    end
    T=_hops_scalar_type(hamiltonian,bath_tuple)
    R=_real_float_type(T)
    precision_bits,rounding_mode=
        _hops_precision_context(
            hamiltonian,bath_tuple,scaling_factors,
            importance_cutoff,T)
    if R===BigFloat&&precision(BigFloat)!=precision_bits
        return _with_supersite_precision(
            R,precision_bits,rounding_mode) do
            HOPSPlan(hamiltonian,bath_tuple;
                max_depth,scaling,scaling_factors,importance_cutoff,
                importance_metric,memory_budget,atol,rtol)
        end
    end
    resolved_rtol=rtol===nothing ? sqrt(eps(R)) : rtol
    resolved_rtol isa Real||throw(ArgumentError(
        "rtol must be a real number"))
    isfinite(atol)&&atol>=0||throw(ArgumentError(
        "atol must be finite and nonnegative"))
    isfinite(resolved_rtol)&&resolved_rtol>=0||throw(ArgumentError(
        "rtol must be finite and nonnegative"))
    ishermitian(hamiltonian;atol,rtol=resolved_rtol)||throw(ArgumentError(
        "the HOPS system Hamiltonian must be Hermitian"))
    all(_heom_isfinite,hamiltonian.data)||throw(ArgumentError(
        "HOPS Hamiltonian coefficients must be finite"))
    prepared_importance_cutoff=R===BigFloat ?
        R(importance_cutoff)+zero(R) : importance_cutoff
    prepared_scaling_factors=if R===BigFloat&&scaling_factors!==nothing
        [begin
             value isa Real ? R(value)+zero(R) : value
         end for value in scaling_factors]
    else
        scaling_factors
    end
    coefficients=T[]
    frequencies=T[]
    exponent_baths=Int[]
    for (bath_number,bath) in pairs(bath_tuple)
        aggregated=Dict{T,T}()
        pole_order=T[]
        for term in eachindex(bath.coefficients)
            coefficient=_heom_checked_convert(
                T,bath.coefficients[term],
                "HOPS bath $bath_number coefficient $term")
            frequency=_heom_checked_convert(
                T,bath.frequencies[term],
                "HOPS bath $bath_number frequency $term")
            coefficient=_hops_context_value(coefficient,T)
            frequency=_hops_context_value(frequency,T)
            real(frequency)>0||throw(ArgumentError(
                "HOPS bath $bath_number frequency $term lost its positive real part in $T; use wider precision"))
            frequency=_hops_canonical_pole(frequency)
            if !haskey(aggregated,frequency)
                aggregated[frequency]=zero(T)
                push!(pole_order,frequency)
            end
            combined=aggregated[frequency]+coefficient
            _heom_isfinite(combined)||throw(ArgumentError(
                "combined HOPS coefficient at bath $bath_number pole $frequency is not finite in $T"))
            aggregated[frequency]=combined
        end
        for frequency in pole_order
            coefficient=aggregated[frequency]
            # An exact zero pole contributes neither noise nor hierarchy
            # generation. Removing it is algebraic, not numerical pruning.
            iszero(coefficient)&&continue
            push!(coefficients,coefficient)
            push!(frequencies,frequency)
            push!(exponent_baths,bath_number)
        end
    end
    K=length(coefficients)
    depth=Int(max_depth)
    full_ado_count=exact_binomial(BigInt(K)+BigInt(depth),BigInt(depth))
    nweak=weak_pi_dimension(basis)
    importance_metric===:normalized_coupling_decay||throw(ArgumentError(
        "importance_metric must be :normalized_coupling_decay"))
    isfinite(prepared_importance_cutoff)&&
        0<=prepared_importance_cutoff<=1||
        throw(ArgumentError(
        "importance_cutoff must be finite and lie between zero and one"))

    # HOPS has one left correlation. Passing it in both amplitude positions
    # reuses the HEOM importance/scaling geometry without changing values.
    multiindices,ado_importances,prepared_cutoff=if iszero(
            prepared_importance_cutoff)
        full_ado_count<=typemax(Int)||throw(ArgumentError(
            "the requested complete HOPS hierarchy has $full_ado_count nodes, exceeding Int indexing; use an explicit positive importance_cutoff or reduce max_depth/poles"))
        full_ado_count*BigInt(nweak)<=typemax(Int)||throw(ArgumentError(
            "the requested complete HOPS coordinate dimension exceeds Int indexing; use an explicit positive importance_cutoff or a smaller basis"))
        estimate=_hops_plan_bytes(
            basis,T,full_ado_count,K,depth,length(bath_tuple),
            precision_bits)
        _require_performance_budget(
            "HOPS plan preparation",estimate,memory_budget;
            guidance="Reduce max_depth or the number of correlation poles.")
        _heom_pruned_multiindices(
            coefficients,coefficients,frequencies,depth,
            prepared_importance_cutoff,importance_metric,R)
    else
        node_limit=_hops_retained_node_limit(
            basis,T,K,depth,length(bath_tuple),precision_bits,
            memory_budget,nweak)
        _hops_pruned_multiindices(
            coefficients,frequencies,depth,prepared_importance_cutoff,
            importance_metric,R,node_limit)
    end
    retained_count=BigInt(length(multiindices))
    retained_count*BigInt(nweak)<=typemax(Int)||throw(ArgumentError(
        "the retained HOPS coordinate dimension exceeds Int indexing"))
    retained_estimate=_hops_plan_bytes(
        basis,T,retained_count,K,depth,length(bath_tuple),
        precision_bits)
    _require_performance_budget(
        "HOPS retained plan",retained_estimate,memory_budget;
        guidance="Increase importance_cutoff or reduce max_depth/poles.")
    lookup=Dict{Tuple,Int}(Tuple(index)=>position
                           for (position,index) in pairs(multiindices))
    topology=_heom_packed_topology(
        multiindices,lookup,exponent_baths,depth)
    bath_edges=[Int[] for _ in eachindex(bath_tuple)]
    for edge in eachindex(topology.lower)
        term=Int(topology.term[edge])
        push!(bath_edges[exponent_baths[term]],edge)
    end
    ado_scales,pole_scales,upward,downward=
        _heom_scaling_data(coefficients,coefficients,multiindices,lookup,
            depth,scaling,prepared_scaling_factors,T)
    edge_downward=Vector{T}(undef,length(topology.lower))
    edge_upward=Vector{R}(undef,length(topology.lower))
    for edge in eachindex(topology.lower)
        term=Int(topology.term[edge])
        level=Int(topology.level[edge])
        downward_value=coefficients[term]*downward[level,term]
        _heom_isfinite(downward_value)||throw(ArgumentError(
            "HOPS downward edge coefficient $edge is not finite in $T"))
        !iszero(coefficients[term])&&
            !iszero(downward[level,term])&&
            iszero(downward_value)&&throw(ArgumentError(
            "HOPS downward edge coefficient $edge underflowed in $T; use wider precision"))
        edge_downward[edge]=downward_value
        edge_upward[edge]=upward[level,term]
    end
    decays=Vector{T}(undef,length(multiindices))
    for (ado,index) in pairs(multiindices)
        value=zero(T)
        for term in eachindex(index)
            value+=T(index[term])*frequencies[term]
        end
        _heom_isfinite(value)||throw(ArgumentError(
            "HOPS hierarchy decay $ado is not finite in $T"))
        decays[ado]=value
    end
    hamiltonian_blocks=_heom_coupling_blocks(basis,hamiltonian,T)
    coupling_blocks=Vector{Vector{Matrix{T}}}(undef,length(bath_tuple))
    coupling_selfadjoint=falses(length(bath_tuple))
    for (bath_number,bath) in pairs(bath_tuple)
        blocks=_heom_coupling_blocks(basis,bath.coupling,T)
        coupling_blocks[bath_number]=blocks
        coupling_selfadjoint[bath_number]=
            ishermitian(bath.coupling;atol=0,rtol=0)
    end
    internal_noise_supported=all(value->
        iszero(imag(value))&&real(value)>=zero(R),coefficients)
    HOPSPlan{typeof(basis),typeof(hamiltonian_blocks),
        typeof(coupling_blocks),typeof(coefficients),
        typeof(exponent_baths),typeof(multiindices),typeof(lookup),
        typeof(topology),R,typeof(map(b->b.label,bath_tuple)),T,
        typeof(rounding_mode)}(
        basis,hamiltonian_blocks,coupling_blocks,
        coefficients,frequencies,exponent_baths,bath_edges,
        multiindices,lookup,
        topology,edge_downward,edge_upward,ado_scales,pole_scales,decays,
        _weak_pi_offsets(basis),map(b->b.label,bath_tuple),
        ado_importances,full_ado_count,depth,nweak,T,scaling,
        prepared_cutoff,importance_metric,internal_noise_supported,
        coupling_selfadjoint,precision_bits,rounding_mode)
end

HOPSPlan(hamiltonian::PIOperator,bath::HEOMBath;kwargs...)=
    HOPSPlan(hamiltonian,HOPSBath(bath);kwargs...)
HOPSPlan(hamiltonian::PIOperator,
         baths::Tuple{Vararg{HEOMBath}};kwargs...)=
    HOPSPlan(hamiltonian,map(HOPSBath,baths);kwargs...)
HOPSPlan(hamiltonian::PIOperator,
         baths::AbstractVector{<:HEOMBath};kwargs...)=
    HOPSPlan(hamiltonian,map(HOPSBath,baths);kwargs...)

size(plan::HOPSPlan)=begin
    dimension=Base.checked_mul(plan.nweak,length(plan.multiindices))
    (dimension,dimension)
end
size(plan::HOPSPlan,index::Integer)=index in (1,2) ? size(plan)[index] : 1
eltype(plan::HOPSPlan)=plan.Ttype
show(io::IO,plan::HOPSPlan)=print(io,
    "HOPSPlan(N=$(plan.basis.N), d=$(plan.basis.d), " *
    "nodes=$(length(plan.multiindices)), dimension=$(size(plan,1)), " *
    "scaling=$(plan.scaling))")

"""Return the number of retained pure-state hierarchy nodes."""
hops_number_auxiliaries(plan::HOPSPlan)=length(plan.multiindices)

"""Return detached copies of the retained HOPS hierarchy multi-indices."""
hops_multiindices(plan::HOPSPlan)=[copy(index) for index in plan.multiindices]

"""
    hops_auxiliary_importances(plan)

Return the dimensionless heuristic score associated with every retained HOPS
node. The root score is one. Scores determine an explicit positive
`importance_cutoff`; they are diagnostics, not error bounds.
"""
function hops_auxiliary_importances(plan::HOPSPlan)
    if _real_float_type(plan.Ttype)===BigFloat&&
            (precision(BigFloat)!=plan.precision_bits||
             rounding(BigFloat)!=plan.rounding_mode)
        return _hops_with_precision(plan) do
            hops_auxiliary_importances(plan)
        end
    end
    isempty(plan.ado_importances)||
        return copy(plan.ado_importances)
    R=_real_float_type(plan.Ttype)
    weights=_heom_importance_weights(
        plan.coefficients,plan.coefficients,plan.frequencies,R)
    R[_heom_multiindex_importance(index,weights,R)
      for index in plan.multiindices]
end

"""
    hops_hierarchy_metadata(plan)

Return hierarchy size, truncation, scaling, noise, and PI-coordinate metadata
without materializing a hierarchy state.
"""
function hops_hierarchy_metadata(plan::HOPSPlan)
    (;retained_auxiliaries=length(plan.multiindices),
      full_auxiliaries=plan.full_ado_count,
      poles=length(plan.coefficients),
      baths=length(plan.coupling_blocks),
      max_depth=plan.max_depth,
      weak_pi_dimension=plan.nweak,
      coordinate_dimension=size(plan,1),
      scaling=plan.scaling,
      importance_cutoff=plan.importance_cutoff,
      importance_metric=plan.importance_metric,
      pruning_approximation=!iszero(plan.importance_cutoff),
      internal_noise_supported=plan.internal_noise_supported,
      precision_bits=plan.precision_bits,
      rounding_mode=plan.rounding_mode,
      equation=:linear)
end

function _hops_auxiliary_index(plan::HOPSPlan,label)
    if label isa Integer&&!(label isa Bool)
        1<=label<=length(plan.multiindices)||throw(BoundsError(
            plan.multiindices,label))
        return Int(label)
    end
    label isa AbstractVector||throw(ArgumentError(
        "HOPS auxiliary label must be a one-based index or occupation vector"))
    length(label)==length(plan.coefficients)||throw(DimensionMismatch(
        "HOPS occupation label must contain one entry per prepared pole"))
    occupations=Vector{Int}(undef,length(label))
    for index in eachindex(label)
        value=label[index]
        value isa Integer&&!(value isa Bool)&&value>=0||
            throw(ArgumentError(
            "HOPS occupation entries must be nonnegative integers"))
        BigInt(value)<=typemax(Int)||throw(ArgumentError(
            "HOPS occupation entry exceeds Int indexing"))
        occupations[index]=Int(value)
    end
    position=get(plan.index,Tuple(occupations),0)
    position>0||throw(ArgumentError(
        "the requested HOPS auxiliary is not retained by this hierarchy"))
    position
end

"""
    hops_coordinate_scale(plan, label)

Return the positive scalar ``s_n`` relating one stored scaled auxiliary root
to the conventional unscaled root, ``psi_n=s_n*psihat_n``. `label` may be a
one-based retained-node index or its pole-occupation vector. The root and
every node of an unscaled plan return one.
"""
hops_coordinate_scale(plan::HOPSPlan,label)=
    plan.ado_scales[_hops_auxiliary_index(plan,label)]

function _hops_workspace_bytes(plan::HOPSPlan)
    hierarchy_entries=BigInt(size(plan,1))
    pole_entries=3BigInt(length(plan.coefficients))
    noise_entries=3BigInt(length(plan.coupling_blocks))
    # Current, stage, slope, next, and one reusable coupling action.
    hierarchy_vectors=5
    _performance_entries_bytes(
        BigInt(hierarchy_vectors)*hierarchy_entries+
        pole_entries+noise_entries+
        BigInt(plan.nweak),plan.Ttype;
        bigfloat_precision=plan.precision_bits)
end

"""
    HOPSWorkspace(plan; memory_budget=512MiB)

Task-owned mutable storage for one [`HOPSPlan`](@ref). The workspace retains
four low-storage RK4 hierarchy vectors, one reusable coupling-action
hierarchy, three OU component stages, and three bath-noise stages. Reuse it
across sequential paths. It must not be used concurrently.
"""
struct HOPSWorkspace{P,V,N}
    plan::P
    current::V
    stage::V
    slope::V
    next::V
    coupling::V
    ou_current::N
    ou_midpoint::N
    ou_endpoint::N
    noise_start::N
    noise_midpoint::N
    noise_endpoint::N
    root_buffer::N
end

@inline function _hops_workspace_similar(array)
    isbitstype(eltype(array)) ? similar(array) :
        zeros(eltype(array),size(array))
end

@generated function _hops_check_scratch_aliases(arrays::T) where T<:Tuple
    checks=Expr[]
    for left in 1:fieldcount(T)-1, right in left+1:fieldcount(T)
        push!(checks,quote
            Base.mightalias(arrays[$left],arrays[$right])&&
                throw(ArgumentError(
                "HOPS workspace scratch arrays must not alias"))
        end)
    end
    quote
        $(checks...)
        nothing
    end
end

function HOPSWorkspace(plan::HOPSPlan;
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    if _real_float_type(plan.Ttype)===BigFloat&&
            (precision(BigFloat)!=plan.precision_bits||
             rounding(BigFloat)!=plan.rounding_mode)
        return _hops_with_precision(plan) do
            HOPSWorkspace(plan;memory_budget)
        end
    end
    estimate=_hops_workspace_bytes(plan)
    _require_performance_budget("HOPS workspace",estimate,memory_budget;
        guidance="Reduce max_depth or the number of correlation poles.")
    vector=zeros(
        plan.Ttype,plan.nweak,length(plan.multiindices))
    pole_vector=zeros(plan.Ttype,length(plan.coefficients))
    bath_vector=zeros(plan.Ttype,length(plan.coupling_blocks))
    root_buffer=zeros(plan.Ttype,plan.nweak)
    HOPSWorkspace(
        plan,vector,
        _hops_workspace_similar(vector),
        _hops_workspace_similar(vector),
        _hops_workspace_similar(vector),
        _hops_workspace_similar(vector),
        pole_vector,
        _hops_workspace_similar(pole_vector),
        _hops_workspace_similar(pole_vector),
        bath_vector,
        _hops_workspace_similar(bath_vector),
        _hops_workspace_similar(bath_vector),
        root_buffer)
end

function _check_hops_workspace(work::HOPSWorkspace,plan::HOPSPlan)
    work.plan===plan||throw(ArgumentError(
        "HOPS workspace was prepared for a different plan"))
    expected=size(plan,1)
    for vector in (work.current,work.stage,work.slope,work.next,
                   work.coupling)
        length(vector)==expected||throw(DimensionMismatch(
            "HOPS workspace has an incompatible hierarchy vector"))
        eltype(vector)===plan.Ttype||throw(ArgumentError(
            "HOPS workspace has incompatible scalar precision"))
        _hops_require_array_precision(
            vector,plan,"HOPS workspace hierarchy storage")
    end
    npoles=length(plan.coefficients)
    nbaths=length(plan.coupling_blocks)
    for vector in (work.ou_current,work.ou_midpoint,work.ou_endpoint)
        length(vector)==npoles||throw(DimensionMismatch(
            "HOPS workspace has an incompatible OU-component vector"))
        eltype(vector)===plan.Ttype||throw(ArgumentError(
            "HOPS workspace OU components have incompatible scalar precision"))
        _hops_require_array_precision(
            vector,plan,"HOPS workspace OU storage")
    end
    for vector in
            (work.noise_start,work.noise_midpoint,work.noise_endpoint)
        length(vector)==nbaths||throw(DimensionMismatch(
            "HOPS workspace has an incompatible bath-noise vector"))
        eltype(vector)===plan.Ttype||throw(ArgumentError(
            "HOPS workspace bath noise has incompatible scalar precision"))
        _hops_require_array_precision(
            vector,plan,"HOPS workspace bath-noise storage")
    end
    length(work.root_buffer)==plan.nweak||throw(DimensionMismatch(
        "HOPS workspace has an incompatible root buffer"))
    eltype(work.root_buffer)===plan.Ttype||throw(ArgumentError(
        "HOPS workspace root buffer has incompatible scalar precision"))
    _hops_require_array_precision(
        work.root_buffer,plan,"HOPS workspace root storage")
    common_arrays=(
        work.current,work.stage,work.slope,work.next,work.coupling,
        work.ou_current,work.ou_midpoint,work.ou_endpoint,
        work.noise_start,work.noise_midpoint,work.noise_endpoint,
        work.root_buffer)
    _hops_check_scratch_aliases(common_arrays)
    work
end

function _hops_check_noise(plan::HOPSPlan,noise)
    length(noise)==length(plan.coupling_blocks)||throw(DimensionMismatch(
        "HOPS noise must contain one value per bath"))
    for (index,value) in pairs(noise)
        value isa Number&&_heom_isfinite(value)||throw(ArgumentError(
            "HOPS noise value $index must be finite"))
        converted=try
            plan.Ttype(value)
        catch error
            error isa InexactError||error isa OverflowError||rethrow()
            throw(ArgumentError(
                "HOPS noise value $index is not representable in $(plan.Ttype)"))
        end
        converted==value||throw(ArgumentError(
            "HOPS noise value $index would narrow in $(plan.Ttype)"))
        if _real_float_type(plan.Ttype)===BigFloat&&
                !_hops_value_has_precision(value,plan.precision_bits)
            throw(ArgumentError(
                "HOPS noise value $index does not use the prepared BigFloat precision $(plan.precision_bits)"))
        end
    end
    noise
end

@inline function _hops_block_gemm!(destination,A,source,alpha,beta)
    mul!(destination,A,source,alpha,beta)
end

@inline function _hops_block_gemm!(
        destination::StridedMatrix{Complex{R}},
        A::StridedMatrix{Complex{R}},
        source::StridedMatrix{Complex{R}},
        alpha::Complex{R},beta::Complex{R}) where
        R<:Union{Float32,Float64}
    LinearAlgebra.BLAS.gemm!('N','N',alpha,A,source,beta,destination)
end

@inline function _hops_block_gemm_adjoint!(
        destination,A,source,alpha,beta)
    mul!(destination,adjoint(A),source,alpha,beta)
end

@inline function _hops_block_gemm_adjoint!(
        destination::StridedMatrix{Complex{R}},
        A::StridedMatrix{Complex{R}},
        source::StridedMatrix{Complex{R}},
        alpha::Complex{R},beta::Complex{R}) where
        R<:Union{Float32,Float64}
    LinearAlgebra.BLAS.gemm!('C','N',alpha,A,source,beta,destination)
end

function _hops_apply_blocks_batch!(destination,blocks,source,offsets,
                                    alpha,beta)
    for sector in eachindex(blocks)
        range=_weak_sector_range(offsets,sector)
        _hops_block_gemm!(view(destination,range,:),blocks[sector],
                          view(source,range,:),alpha,beta)
    end
    destination
end

function _hops_apply_blocks_adjoint_batch!(
        destination,blocks,source,offsets,alpha,beta)
    for sector in eachindex(blocks)
        range=_weak_sector_range(offsets,sector)
        _hops_block_gemm_adjoint!(
            view(destination,range,:),blocks[sector],
            view(source,range,:),alpha,beta)
    end
    destination
end

"""
    hops_rhs!(destination, plan, source, noise, workspace)

Apply the conditioned linear HOPS right-hand side for one already prepared
vector of bath-noise values. `source` and `destination` contain all hierarchy
nodes in node-major order; `noise[b]` is ``z_b(t)`` and the equation applies
its complex conjugate.

The operation is deterministic and uses direct BLAS on the prepared Schur
blocks for ordinary machine precision. The public flat-vector wrapper may
create only bounded reshape headers; the internal workspace-matrix path used
by trajectory RK4 is allocation-free after warm-up. It never draws random
numbers. `destination` must not alias `source`, and the explicit workspace
must be task-owned.
"""
function _hops_rhs_matrices!(Y,plan::HOPSPlan,X,noise,
                             work::HOPSWorkspace)
    _hops_check_noise(plan,noise)
    all(_heom_isfinite,X)||throw(ArgumentError(
        "HOPS source contains nonfinite amplitudes"))
    naux=length(plan.multiindices)
    LX=work.coupling
    fill!(Y,zero(eltype(Y)))
    alpha=convert(plan.Ttype,-1im)
    _hops_apply_blocks_batch!(
        Y,plan.hamiltonian_blocks,X,plan.offsets,alpha,zero(alpha))
    @inbounds for ado in 1:naux
        decay=plan.decays[ado]
        for row in 1:plan.nweak
            Y[row,ado]-=decay*X[row,ado]
        end
    end
    topology=plan.topology
    for bath in eachindex(plan.coupling_blocks)
        _hops_apply_blocks_batch!(
            LX,plan.coupling_blocks[bath],X,plan.offsets,
            one(plan.Ttype),zero(plan.Ttype))
        noise_scale=conj(plan.Ttype(noise[bath]))
        if !iszero(noise_scale)
            @inbounds for index in eachindex(Y)
                Y[index]+=noise_scale*work.coupling[index]
            end
        end
        if plan.coupling_selfadjoint[bath]
            @inbounds for edge in plan.bath_edges[bath]
                lower=Int(topology.lower[edge])
                upper=Int(topology.upper[edge])
                downward=plan.edge_downward[edge]
                upward=plan.edge_upward[edge]
                for row in 1:plan.nweak
                    Y[row,upper]+=downward*LX[row,lower]
                    Y[row,lower]-=upward*LX[row,upper]
                end
            end
        else
            @inbounds for edge in plan.bath_edges[bath]
                lower=Int(topology.lower[edge])
                upper=Int(topology.upper[edge])
                downward=plan.edge_downward[edge]
                for row in 1:plan.nweak
                    Y[row,upper]+=downward*LX[row,lower]
                end
            end
            _hops_apply_blocks_adjoint_batch!(
                LX,plan.coupling_blocks[bath],X,plan.offsets,
                one(plan.Ttype),zero(plan.Ttype))
            @inbounds for edge in plan.bath_edges[bath]
                lower=Int(topology.lower[edge])
                upper=Int(topology.upper[edge])
                upward=plan.edge_upward[edge]
                for row in 1:plan.nweak
                    Y[row,lower]-=upward*LX[row,upper]
                end
            end
        end
    end
    all(_heom_isfinite,Y)||throw(ArgumentError(
        "conditioned HOPS right-hand side produced nonfinite amplitudes"))
    Y
end

"""
    hops_rhs!(destination, plan, source, noise, workspace)

Apply the deterministic conditioned linear-HOPS hierarchy action to flat
node-major vectors. `noise` contains one current ``z_b`` value per shared
bath. Random sampling is never performed by this function; see
[`hops_trajectory`](@ref) for prepared colored-noise paths.
"""
function hops_rhs!(destination::AbstractVector,plan::HOPSPlan,
                   source::AbstractVector,noise,
                   work::HOPSWorkspace)
    if _real_float_type(plan.Ttype)===BigFloat&&
            (precision(BigFloat)!=plan.precision_bits||
             rounding(BigFloat)!=plan.rounding_mode)
        return _hops_with_precision(plan) do
            hops_rhs!(destination,plan,source,noise,work)
        end
    end
    Base.mightalias(destination,source)&&throw(ArgumentError(
        "HOPS source and destination must not alias"))
    _check_hops_workspace(work,plan)
    length(source)==size(plan,1)||throw(DimensionMismatch(
        "HOPS source has the wrong coordinate dimension"))
    length(destination)==size(plan,1)||throw(DimensionMismatch(
        "HOPS destination has the wrong coordinate dimension"))
    eltype(source)===plan.Ttype&&eltype(destination)===plan.Ttype||
        throw(ArgumentError(
            "HOPS source and destination must use $(plan.Ttype)"))
    _hops_require_array_precision(
        source,plan,"HOPS source hierarchy")
    (Base.mightalias(source,work.coupling)||
     Base.mightalias(destination,work.coupling))&&throw(ArgumentError(
        "HOPS source and destination must not alias coupling-action scratch"))
    noise isa AbstractArray&&Base.mightalias(destination,noise)&&
        throw(ArgumentError(
            "HOPS destination must not alias the current noise vector"))
    X=reshape(source,plan.nweak,length(plan.multiindices))
    Y=reshape(destination,plan.nweak,length(plan.multiindices))
    _hops_rhs_matrices!(Y,plan,X,noise,work)
    destination
end

"""
    HOPSRootKet(basis, data)

Unnormalized direct-sum Schur-irrep root of a linear HOPS hierarchy. Its
sector slice ``psi_nu`` contributes
``psi_nu*psi_nu'/sqrt(f^nu)`` to the PI coefficient block. Unlike
[`WeakPIPseudoKet`](@ref), construction does not require unit norm.
"""
struct HOPSRootKet{T<:AbstractFloat,B<:PIBasis}
    basis::B
    data::Vector{Complex{T}}
    function HOPSRootKet(basis::B,data::AbstractVector{Complex{T}}) where
            {T<:AbstractFloat,B<:PIBasis}
        length(data)==weak_pi_dimension(basis)||throw(DimensionMismatch(
            "HOPS root has the wrong direct-sum Schur dimension"))
        all(isfinite,data)||throw(ArgumentError(
            "HOPS root amplitudes must be finite"))
        new{T,B}(basis,collect(data))
    end
end

copy(root::HOPSRootKet)=HOPSRootKet(root.basis,root.data)
eltype(root::HOPSRootKet)=eltype(root.data)
length(root::HOPSRootKet)=length(root.data)
show(io::IO,root::HOPSRootKet)=print(io,
    "HOPSRootKet(N=$(root.basis.N), d=$(root.basis.d), " *
    "dimension=$(length(root)), weight=$(real(dot(root.data,root.data))))")

"""
    HOPSTrajectory

Saved root pseudo-kets and bath-noise values from one linear PI-HOPS path.
Only the root is retained at requested output times; auxiliary hierarchy
states remain in the reusable workspace. `noise[b,j]` is the shared process
``z_b`` at `times[j]`.
"""
struct HOPSTrajectory{R<:AbstractFloat,S<:HOPSRootKet,M<:AbstractMatrix}
    times::Vector{R}
    states::Vector{S}
    noise::M
end

Base.length(path::HOPSTrajectory)=length(path.times)
Base.firstindex(path::HOPSTrajectory)=firstindex(path.times)
Base.lastindex(path::HOPSTrajectory)=lastindex(path.times)
Base.getindex(path::HOPSTrajectory,index::Integer)=path.states[index]
show(io::IO,path::HOPSTrajectory)=print(io,
    "HOPSTrajectory($(length(path.times)) saved roots, " *
    "$(size(path.noise,1)) shared baths)")

function _hops_density_data!(destination,basis,root,offsets,scales)
    length(destination)==length(basis)||throw(DimensionMismatch(
        "HOPS density destination has the wrong PI dimension"))
    fill!(destination,zero(eltype(destination)))
    for (sector,partition) in pairs(basis.sectors)
        dimension=length(basis.patterns[sector])
        block=reshape(view(destination,
            basis.offsets[sector]:basis.offsets[sector+1]-1),
            dimension,dimension)
        amplitudes=view(root,_weak_sector_range(offsets,sector))
        _weak_density_outer!(block,amplitudes,scales[sector];
                             accumulate=false)
    end
    destination
end

"""
    hops_density(root)
    hops_density(trajectory, index)

Construct the unnormalized PI density contribution
``|psi_0><psi_0|`` of one linear-HOPS root. The physical Monte Carlo
estimator averages these returned coefficient vectors without normalizing
individual paths. Consequently `trace(hops_density(root)) == norm(root)^2`
up to roundoff.
"""
function hops_density(root::HOPSRootKet{R}) where R<:AbstractFloat
    if R===BigFloat
        precision_bits=max(
            precision(BigFloat),
            _supersite_array_precision(root.data))
        if precision(BigFloat)!=precision_bits
            return _with_supersite_precision(
                R,precision_bits,rounding(BigFloat)) do
                hops_density(root)
            end
        end
    end
    output=zeros(Complex{R},length(root.basis))
    _hops_density_data!(output,root.basis,root.data,
        _weak_pi_offsets(root.basis),_weak_pi_density_scales(root.basis,R))
    PIState(root.basis,output)
end
hops_density(path::HOPSTrajectory,index::Integer)=
    hops_density(path.states[index])

@inline function _hops_unit_complex_gaussian(rng::AbstractRNG,
                                              ::Type{R}) where
        R<:AbstractFloat
    normalization=inv(sqrt(R(2)))
    try
        Complex{R}(randn(rng,R)*normalization,
                   randn(rng,R)*normalization)
    catch error
        error isa MethodError||rethrow()
        throw(ArgumentError(
            "the selected RNG cannot generate Gaussian samples in $R"))
    end
end

@inline function _hops_unit_complex_gaussian(
        rng::AbstractRNG,::Type{BigFloat})
    # Direct polar Box--Muller sampling avoids relying on a BigFloat randn
    # method. The returned proper complex Gaussian has E[abs2(z)]=1.
    uniform=rand(rng,BigFloat)
    while iszero(uniform)
        uniform=rand(rng,BigFloat)
    end
    phase=BigFloat(2)*BigFloat(pi)*rand(rng,BigFloat)
    radius=sqrt(-log(uniform))
    Complex{BigFloat}(radius*cos(phase),radius*sin(phase))
end

function _hops_sum_bath_noise!(destination,components,plan::HOPSPlan)
    fill!(destination,zero(eltype(destination)))
    @inbounds for term in eachindex(components)
        destination[plan.exponent_baths[term]]+=components[term]
    end
    destination
end

function _hops_initialize_ou!(work::HOPSWorkspace,rng::AbstractRNG)
    plan=work.plan
    plan.internal_noise_supported||throw(ArgumentError(
        "this HOPS correlation contains a complex or negative exponential coefficient; pass a deterministic external noise provider for the total bath covariance"))
    R=_real_float_type(plan.Ttype)
    @inbounds for term in eachindex(work.ou_current)
        coefficient=R(real(plan.coefficients[term]))
        amplitude=sqrt(coefficient)
        isfinite(amplitude)||throw(ArgumentError(
            "stationary HOPS noise amplitude $term is not finite in $R"))
        work.ou_current[term]=amplitude*
            _hops_unit_complex_gaussian(rng,R)
    end
    _hops_sum_bath_noise!(work.noise_start,work.ou_current,plan)
    work
end

function _hops_advance_ou!(destination,source,h,plan::HOPSPlan,
                           rng::AbstractRNG)
    R=_real_float_type(plan.Ttype)
    @inbounds for term in eachindex(source)
        pole=plan.frequencies[term]
        coefficient=R(real(plan.coefficients[term]))
        decay=exp(-pole*h)
        fraction=-expm1(-R(2)*R(real(pole))*h)
        fraction>=zero(R)&&isfinite(fraction)||throw(ArgumentError(
            "HOPS OU innovation fraction $term is invalid"))
        variance=coefficient*fraction
        isfinite(variance)&&variance>=zero(R)||throw(ArgumentError(
            "HOPS OU innovation variance $term is invalid in $R"))
        coefficient>zero(R)&&h>zero(R)&&iszero(variance)&&throw(ArgumentError(
            "HOPS OU innovation variance $term underflowed in $R; use wider precision or a larger integration step"))
        destination[term]=decay*source[term]+sqrt(variance)*
            _hops_unit_complex_gaussian(rng,R)
    end
    destination
end

function _hops_external_noise!(destination,provider,time,plan::HOPSPlan)
    if all(bath->applicable(provider,time,bath),eachindex(destination))
        @inbounds for bath in eachindex(destination)
            destination[bath]=provider(time,bath)
        end
    elseif applicable(provider,destination,time)
        provider(destination,time)
    else
        throw(ArgumentError(
            "external HOPS noise must support noise(time,bath_index) or noise!(destination,time); type at least one argument when an untyped two-argument callable would be ambiguous"))
    end
    _hops_check_noise(plan,destination)
    @inbounds for bath in eachindex(destination)
        destination[bath]=plan.Ttype(destination[bath])
    end
    destination
end

function _hops_rk4_step!(plan::HOPSPlan,work::HOPSWorkspace,h,
        noise_start,noise_midpoint,noise_endpoint)
    current=work.current
    stage=work.stage
    slope=work.slope
    next=work.next
    half=h/2
    _hops_rhs_matrices!(slope,plan,current,noise_start,work)
    @inbounds for index in eachindex(current)
        next[index]=current[index]+(h/6)*slope[index]
        stage[index]=current[index]+half*slope[index]
    end
    _hops_rhs_matrices!(slope,plan,stage,noise_midpoint,work)
    @inbounds for index in eachindex(current)
        next[index]+=(h/3)*slope[index]
        stage[index]=current[index]+half*slope[index]
    end
    _hops_rhs_matrices!(slope,plan,stage,noise_midpoint,work)
    @inbounds for index in eachindex(current)
        next[index]+=(h/3)*slope[index]
        stage[index]=current[index]+h*slope[index]
    end
    _hops_rhs_matrices!(slope,plan,stage,noise_endpoint,work)
    @inbounds for index in eachindex(current)
        next[index]+=(h/6)*slope[index]
    end
    all(_heom_isfinite,next)||throw(ErrorException(
        "linear HOPS integration produced nonfinite amplitudes"))
    copyto!(current,next)
    work
end

function _hops_times(times,::Type{R},dt) where R<:AbstractFloat
    raw=collect(times)
    isempty(raw)&&throw(ArgumentError(
        "at least one HOPS output time is required"))
    converted=Vector{R}(undef,length(raw))
    for index in eachindex(raw)
        value=_trajectory_real_input(
            R,raw[index],"HOPS output time at index $index")
        converted[index]=R===BigFloat ? value+zero(R) : value
    end
    all(diff(converted).>=zero(R))||throw(ArgumentError(
        "HOPS output times must be nondecreasing"))
    raw_step=_trajectory_real_input(R,dt,"HOPS dt")
    step=R===BigFloat ? raw_step+zero(R) : raw_step
    step>zero(R)||throw(ArgumentError("HOPS dt must be positive"))
    converted,step
end

function _hops_validate_initial(plan::HOPSPlan,
                                initial::WeakPIPseudoKet)
    initial.basis===plan.basis||throw(ArgumentError(
        "HOPS initial pseudo-ket and plan use different PI bases"))
    eltype(initial.data)===plan.Ttype||throw(ArgumentError(
        "HOPS initial pseudo-ket scalar type $(eltype(initial.data)) does not match prepared precision $(plan.Ttype)"))
    all(isfinite,initial.data)||throw(ArgumentError(
        "HOPS initial pseudo-ket contains nonfinite amplitudes"))
    _hops_require_array_precision(
        initial.data,plan,"HOPS initial pseudo-ket")
    initial
end

function _hops_reset_state!(work::HOPSWorkspace,
                            initial_data::AbstractVector)
    fill!(work.current,zero(eltype(work.current)))
    length(initial_data)==work.plan.nweak||throw(DimensionMismatch(
        "HOPS initial root has the wrong direct-sum dimension"))
    copyto!(view(work.current,:,1),initial_data)
    fill!(work.stage,zero(eltype(work.stage)))
    fill!(work.slope,zero(eltype(work.slope)))
    fill!(work.next,zero(eltype(work.next)))
    work
end

function _hops_integrate!(record!,plan::HOPSPlan,
        initial::WeakPIPseudoKet,times,dt,work::HOPSWorkspace,
        rng::AbstractRNG,noise_provider)
    _check_hops_workspace(work,plan)
    _hops_validate_initial(plan,initial)
    _hops_integrate_data!(
        record!,plan,initial.data,times,dt,work,rng,noise_provider)
end

function _hops_integrate_data!(record!,plan::HOPSPlan,
        initial_data::AbstractVector,times,dt,work::HOPSWorkspace,
        rng::AbstractRNG,noise_provider)
    _check_hops_workspace(work,plan)
    length(initial_data)==plan.nweak||throw(DimensionMismatch(
        "HOPS initial root has the wrong direct-sum dimension"))
    eltype(initial_data)===plan.Ttype||throw(ArgumentError(
        "HOPS initial root has incompatible scalar precision"))
    all(isfinite,initial_data)||throw(ArgumentError(
        "HOPS initial root contains nonfinite amplitudes"))
    _hops_require_array_precision(
        initial_data,plan,"HOPS initial root")
    _hops_reset_state!(work,initial_data)
    builtin=noise_provider===nothing
    if builtin
        _hops_initialize_ou!(work,rng)
    else
        _hops_external_noise!(
            work.noise_start,noise_provider,times[1],plan)
    end
    record!(view(work.current,:,1),work.noise_start,1)
    time=times[1]
    for output_index in 2:length(times)
        target=times[output_index]
        while time<target
            h,lands_on_target=_trajectory_step_to_target(
                time,target,dt)
            h>zero(h)||throw(ErrorException(
                "HOPS integration step did not advance time"))
            midpoint=time+h/2
            endpoint=lands_on_target ? target : time+h
            if builtin
                _hops_advance_ou!(
                    work.ou_midpoint,work.ou_current,h/2,plan,rng)
                _hops_advance_ou!(
                    work.ou_endpoint,work.ou_midpoint,h/2,plan,rng)
                _hops_sum_bath_noise!(
                    work.noise_start,work.ou_current,plan)
                _hops_sum_bath_noise!(
                    work.noise_midpoint,work.ou_midpoint,plan)
                _hops_sum_bath_noise!(
                    work.noise_endpoint,work.ou_endpoint,plan)
            else
                _hops_external_noise!(
                    work.noise_start,noise_provider,time,plan)
                _hops_external_noise!(
                    work.noise_midpoint,noise_provider,midpoint,plan)
                _hops_external_noise!(
                    work.noise_endpoint,noise_provider,endpoint,plan)
            end
            _hops_rk4_step!(plan,work,h,work.noise_start,
                            work.noise_midpoint,work.noise_endpoint)
            if builtin
                copyto!(work.ou_current,work.ou_endpoint)
            end
            time=endpoint
        end
        # For a repeated output time the start buffer still contains its
        # current value. For an advanced interval, endpoint is the current
        # noise by construction.
        current_noise=target==times[1] ? work.noise_start :
            work.noise_endpoint
        record!(view(work.current,:,1),current_noise,output_index)
    end
    work
end

"""
    hops_trajectory(plan, initial, times; dt, rng=Random.default_rng(),
                    noise=nothing, workspace=nothing,
                    memory_budget=512MiB)

Integrate one fixed-step linear PI-HOPS path. `initial` is a normalized
[`WeakPIPseudoKet`](@ref); all auxiliary hierarchy nodes start at zero. The
returned [`HOPSTrajectory`](@ref) retains only root pseudo-kets and bath-noise
values at `times`.

When every exponential coefficient is real and nonnegative, `noise=nothing`
uses the exact stationary complex Ornstein--Uhlenbeck transition at every
half step. Otherwise pass a deterministic provider supporting either
`noise(time, bath_index)` or in-place `noise!(destination, time)`. The
provider must represent the covariance of the *total* bath correlation and
must return the same value whenever queried at the same time. Adaptive
integration is intentionally absent because rejected stochastic steps need a
consistent OU/Brownian bridge.
"""
function hops_trajectory(plan::HOPSPlan,
        initial::WeakPIPseudoKet{R},times;dt::Real,
        rng::AbstractRNG=Random.default_rng(),noise=nothing,
        workspace=nothing,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET) where
        R<:AbstractFloat
    if R===BigFloat&&
            (precision(BigFloat)!=plan.precision_bits||
             rounding(BigFloat)!=plan.rounding_mode)
        return _hops_with_precision(plan) do
            hops_trajectory(
                plan,initial,times;dt,rng,noise,workspace,memory_budget)
        end
    end
    ts,step=_hops_times(times,R,dt)
    _hops_validate_initial(plan,initial)
    output_entries=BigInt(plan.nweak)*BigInt(length(ts))+
                   BigInt(length(plan.coupling_blocks))*BigInt(length(ts))
    estimate=_hops_workspace_bytes(plan)+
        _performance_entries_bytes(
            output_entries,plan.Ttype;
            bigfloat_precision=plan.precision_bits)+
        _performance_entries_bytes(
            length(ts),R;bigfloat_precision=plan.precision_bits)
    _require_performance_budget("HOPS trajectory",estimate,memory_budget;
        guidance="Reuse a workspace and request fewer saved output times.")
    work=if workspace===nothing
        HOPSWorkspace(plan;memory_budget=Inf)
    else
        workspace isa HOPSWorkspace||throw(ArgumentError(
            "workspace must be a HOPSWorkspace"))
        _check_hops_workspace(workspace,plan)
    end
    Root=HOPSRootKet{R,typeof(plan.basis)}
    states=Vector{Root}(undef,length(ts))
    noise_history=Matrix{plan.Ttype}(
        undef,length(plan.coupling_blocks),length(ts))
    function record!(root,current_noise,index)
        states[index]=HOPSRootKet(plan.basis,root)
        copyto!(view(noise_history,:,index),current_noise)
        nothing
    end
    _hops_integrate!(record!,plan,initial,ts,step,work,rng,noise)
    HOPSTrajectory(ts,states,noise_history)
end

"""
    HOPSInitialEnsemble(rho; atol=..., rtol=...)
    hops_initial_ensemble(plan, rho; kwargs...)

Prepare a categorical pure-root ensemble for a general normalized PI density
operator. Each multiplicity-weighted Schur block
``sqrt(f^nu) C_nu`` is diagonalized. Its strictly positive eigenvalues are
the component weights and its eigenvectors are roots supported in that Schur
sector. No multiplicity-tableau sampling and no `d^N` state are required.

The input must be Hermitian within the explicit tolerances, have unit trace
within those tolerances, and have no negative computed eigenvalue. Negative
eigenvalues are never clipped. Zero eigenvalues are omitted exactly.
"""
struct HOPSInitialEnsemble{R<:AbstractFloat,B<:PIBasis}
    basis::B
    sectors::Vector{Int}
    amplitudes::Vector{Vector{Complex{R}}}
    weights::Vector{R}
    cumulative::Vector{R}
    source_trace::R
end

function HOPSInitialEnsemble(rho::PIState{R};
        atol=nothing,
        rtol=nothing,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET) where R<:AbstractFloat
    precision_bits=R===BigFloat ?
        max(precision(BigFloat),_supersite_array_precision(rho.data)) :
        precision(R)
    if R===BigFloat&&precision(BigFloat)!=precision_bits
        return _with_supersite_precision(
            R,precision_bits,rounding(BigFloat)) do
            HOPSInitialEnsemble(rho;atol,rtol,memory_budget)
        end
    end
    resolved_atol=atol===nothing ? _analysis_atol(rho) : atol
    resolved_rtol=rtol===nothing ? _state_rtol(rho) : rtol
    resolved_atol isa Real||throw(ArgumentError(
        "HOPS initial-ensemble atol must be a real number"))
    resolved_rtol isa Real||throw(ArgumentError(
        "HOPS initial-ensemble rtol must be a real number"))
    isfinite(resolved_atol)&&resolved_atol>=0||throw(ArgumentError(
        "HOPS initial-ensemble atol must be finite and nonnegative"))
    isfinite(resolved_rtol)&&resolved_rtol>=0||throw(ArgumentError(
        "HOPS initial-ensemble rtol must be finite and nonnegative"))
    ishermitian(rho;atol=resolved_atol,rtol=resolved_rtol)||
        throw(ArgumentError(
        "HOPS initial density operator must be Hermitian"))
    estimate=_performance_entries_bytes(
        4BigInt(length(rho.basis)),eltype(rho.data);
        bigfloat_precision=precision_bits)
    _require_performance_budget(
        "HOPS initial Schur eigenensemble",estimate,memory_budget;
        guidance="Use a restricted invariant Schur basis when physically appropriate.")
    z=trace(rho)
    tolerance=R(resolved_atol)+R(resolved_rtol)
    abs(imag(z))<=tolerance&&abs(real(z)-one(R))<=tolerance||
        throw(ArgumentError(
            "HOPS initial density operator must have unit real trace; trace=$z"))
    sectors=Int[]
    amplitudes=Vector{Vector{Complex{R}}}()
    weights=R[]
    for (sector,partition) in pairs(rho.basis.sectors)
        block=_multiplicity_weighted_block(rho,partition)
        scale=norm(block,Inf)
        norm(block-block',Inf)<=
            R(resolved_atol)+R(resolved_rtol)*scale||
            throw(ArgumentError(
                "HOPS initial block $partition is not Hermitian"))
        iszero(scale)&&continue
        decomposition=_hermitian_eigen(Hermitian((block+block')/2);
            operation="HOPS initial Schur eigenensemble")
        minimum(decomposition.values)>=zero(R)||throw(ArgumentError(
            "HOPS initial block $partition has a negative computed eigenvalue; the state is not PSD in its working precision"))
        for column in eachindex(decomposition.values)
            weight=decomposition.values[column]
            iszero(weight)&&continue
            isfinite(weight)&&weight>zero(R)||throw(ArgumentError(
                "HOPS initial component weight is not positive and finite"))
            push!(sectors,sector)
            push!(amplitudes,collect(view(decomposition.vectors,:,column)))
            push!(weights,weight)
        end
    end
    isempty(weights)&&throw(ArgumentError(
        "HOPS initial density operator has no positive spectral component"))
    total=sum(weights)
    isfinite(total)&&total>zero(R)||throw(ArgumentError(
        "HOPS initial component weight is outside the finite range of $R"))
    abs(total-one(R))<=tolerance||throw(ArgumentError(
        "HOPS initial spectral weights sum to $total instead of one"))
    cumulative=similar(weights)
    running=zero(R)
    @inbounds for index in eachindex(weights)
        running+=weights[index]/total
        cumulative[index]=running
    end
    cumulative[end]=one(R)
    HOPSInitialEnsemble(
        rho.basis,sectors,amplitudes,weights,cumulative,R(real(z)))
end

"""
    hops_initial_ensemble(plan, rho; kwargs...)

Prepare a [`HOPSInitialEnsemble`](@ref) and require exact PI-basis and scalar
compatibility with `plan`.
"""
function hops_initial_ensemble(plan::HOPSPlan,rho::PIState;kwargs...)
    if _real_float_type(plan.Ttype)===BigFloat&&
            (precision(BigFloat)!=plan.precision_bits||
             rounding(BigFloat)!=plan.rounding_mode)
        return _hops_with_precision(plan) do
            hops_initial_ensemble(plan,rho;kwargs...)
        end
    end
    rho.basis===plan.basis||throw(ArgumentError(
        "HOPS plan and initial density operator use different PI bases"))
    eltype(rho.data)===plan.Ttype||throw(ArgumentError(
        "HOPS initial density scalar type $(eltype(rho.data)) does not match prepared precision $(plan.Ttype)"))
    _hops_require_array_precision(
        rho.data,plan,"HOPS initial density operator")
    HOPSInitialEnsemble(rho;kwargs...)
end

function _hops_sample_initial!(destination,
        ensemble::HOPSInitialEnsemble{R},offsets,
        rng::AbstractRNG) where R
    fill!(destination,zero(eltype(destination)))
    draw=rand(rng,R)
    component=searchsortedfirst(ensemble.cumulative,draw)
    component=min(component,length(ensemble.cumulative))
    sector=ensemble.sectors[component]
    copyto!(view(destination,_weak_sector_range(offsets,sector)),
            ensemble.amplitudes[component])
    destination
end

"""
    HOPSBatchWorkspace(plan; workers=Threads.nthreads(),
                       memory_budget=512MiB)

Reusable pool for [`hops_average`](@ref). The immutable plan is shared;
every worker owns an independent [`HOPSWorkspace`](@ref) and RNG. A batch
workspace is mutable and must not serve concurrent ensemble calls.
"""
struct HOPSBatchWorkspace{P,W,G,S}
    plan::P
    workers::W
    rngs::G
    seeds::S
end

function HOPSBatchWorkspace(plan::HOPSPlan;
        workers::Integer=Threads.nthreads(),
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    if _real_float_type(plan.Ttype)===BigFloat&&
            (precision(BigFloat)!=plan.precision_bits||
             rounding(BigFloat)!=plan.rounding_mode)
        return _hops_with_precision(plan) do
            HOPSBatchWorkspace(plan;workers,memory_budget)
        end
    end
    workers isa Bool&&throw(ArgumentError(
        "HOPS worker count must not be a Bool"))
    workers>0||throw(ArgumentError("HOPS worker count must be positive"))
    BigInt(workers)<=typemax(Int)||throw(ArgumentError(
        "HOPS worker count exceeds Int indexing"))
    count=Int(workers)
    estimate=BigInt(count)*_hops_workspace_bytes(plan)
    _require_performance_budget("HOPS batch workspace",estimate,memory_budget;
        guidance="Reduce workers, max_depth, or the number of bath poles.")
    workspaces=[HOPSWorkspace(plan;memory_budget=Inf) for _ in 1:count]
    rngs=[MersenneTwister(0) for _ in 1:count]
    HOPSBatchWorkspace(plan,workspaces,rngs,UInt64[])
end

function _check_hops_batch_workspace(batch::HOPSBatchWorkspace,
                                     plan::HOPSPlan)
    batch.plan===plan||throw(ArgumentError(
        "HOPS batch workspace was prepared for a different plan"))
    isempty(batch.workers)&&throw(ArgumentError(
        "HOPS batch workspace has no workers"))
    length(batch.workers)==length(batch.rngs)||throw(ArgumentError(
        "HOPS batch workspace has inconsistent worker and RNG storage"))
    batch.seeds isa Vector{UInt64}||throw(ArgumentError(
        "HOPS batch workspace has incompatible seed storage"))
    for work in batch.workers
        _check_hops_workspace(work,plan)
    end
    batch
end

"""
    HOPSEnsembleResult

Optional diagnostic result from
`hops_average(...; return_info=true)`. `states` are the unmodified averages
of unnormalized root outer products. `sample_spread[j]` is the unbiased
Hilbert--Schmidt sample spread at `times[j]`, and `standard_error[j]` is its
norm divided by the square root of the independent path count. These are
Monte Carlo diagnostics, not hierarchy- or time-step-error estimates.

Indexing and iteration traverse `states`.
"""
struct HOPSEnsembleResult{R<:AbstractFloat,S<:PIState,M}
    times::Vector{R}
    states::Vector{S}
    trajectory_count::Int
    sample_spread::Vector{R}
    standard_error::Vector{R}
    metadata::M
end

Base.length(result::HOPSEnsembleResult)=length(result.states)
Base.firstindex(result::HOPSEnsembleResult)=firstindex(result.states)
Base.lastindex(result::HOPSEnsembleResult)=lastindex(result.states)
Base.getindex(result::HOPSEnsembleResult,index::Integer)=result.states[index]
Base.iterate(result::HOPSEnsembleResult,args...)=
    iterate(result.states,args...)
show(io::IO,result::HOPSEnsembleResult)=print(io,
    "HOPSEnsembleResult($(result.trajectory_count) paths, " *
    "$(length(result.times)) output times)")

function _hops_average_memory_bytes(
        plan,ntimes,trajectories,active_workers,retained_workers)
    npi=BigInt(length(plan.basis))
    time_count=BigInt(ntimes)
    worker_count=BigInt(active_workers)
    workspace_bytes=BigInt(retained_workers)*_hops_workspace_bytes(plan)
    # Per worker and time: one coefficient mean plus one scalar M2; one
    # density buffer per worker. Final states coexist with worker means while
    # the reduction is assembled.
    complex_entries=worker_count*(npi*time_count+npi)+npi*time_count
    real_entries=worker_count*time_count+3time_count
    workspace_bytes+
        _performance_entries_bytes(
            complex_entries,plan.Ttype;
            bigfloat_precision=plan.precision_bits)+
        _performance_entries_bytes(
            real_entries,_real_float_type(plan.Ttype);
            bigfloat_precision=plan.precision_bits)+
        BigInt(trajectories)*BigInt(sizeof(UInt64))
end

struct _HOPSStateAccumulatorRecorder{A,V,S,P}
    accumulators::A
    density_buffer::V
    scales::S
    plan::P
end

@inline function (recorder::_HOPSStateAccumulatorRecorder)(
        root,_noise,index)
    plan=recorder.plan
    _hops_density_data!(
        recorder.density_buffer,plan.basis,root,
        plan.offsets,recorder.scales)
    _accumulate_state!(
        recorder.accumulators[index],recorder.density_buffer)
    nothing
end

function _hops_path_accumulate!(
        recorder::_HOPSStateAccumulatorRecorder,
        plan,initial,times,dt,work,rng)
    _hops_integrate!(recorder,plan,initial,times,dt,work,rng,nothing)
    nothing
end

function _hops_path_accumulate!(recorder::_HOPSStateAccumulatorRecorder,
        plan,initial::HOPSInitialEnsemble,times,dt,work,rng)
    _hops_sample_initial!(work.root_buffer,initial,plan.offsets,rng)
    _hops_integrate_data!(
        recorder,plan,work.root_buffer,times,dt,work,rng,nothing)
    nothing
end

"""
    hops_average(plan, initial, times, trajectories;
                 dt, seed=0, threaded=false, workspace=nothing,
                 return_info=false, memory_budget=512MiB)
    hops_average(plan, rho::PIState, times, trajectories; ...)

Average independent linear PI-HOPS paths without retaining their hierarchy or
root histories. Every sampled root is converted to its unnormalized PI outer
product before online Welford reduction. Individual roots and the final
average are never normalized or positivity-repaired.

The fixed seed assigns one `UInt64` stream to every trajectory index.
`threaded=true` shares the read-only plan and uses task-owned workspaces with
deterministic balanced partitions. Floating reductions are merged in worker
order and can differ by roundoff when the active worker count changes. Pass a
reusable
[`HOPSBatchWorkspace`](@ref) to amortize hierarchy storage across ensembles.

By default the function returns `Vector{PIState}`. Set `return_info=true` for
a [`HOPSEnsembleResult`](@ref) carrying Hilbert--Schmidt Monte Carlo
uncertainties. Time step, hierarchy depth/cutoff, correlation decomposition,
and path count require separate convergence checks.

A general normalized `PIState` is accepted through an automatically prepared
[`HOPSInitialEnsemble`](@ref). Its Schur eigencomponent is sampled before the
stationary bath noise on each independently seeded path. Use
`initial_atol`/`initial_rtol` only to acknowledge known roundoff-level input
errors; negative computed eigenvalues are never clipped.
"""
function hops_average(plan::HOPSPlan,
        initial::Union{WeakPIPseudoKet{R},HOPSInitialEnsemble{R}},
        times,trajectories::Integer;
        dt::Real,seed::Integer=0,threaded::Bool=false,workspace=nothing,
        return_info::Bool=false,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET) where
        R<:AbstractFloat
    if R===BigFloat&&
            (precision(BigFloat)!=plan.precision_bits||
             rounding(BigFloat)!=plan.rounding_mode)
        return _hops_with_precision(plan) do
            hops_average(
                plan,initial,times,trajectories;
                dt,seed,threaded,workspace,return_info,memory_budget)
        end
    end
    trajectories isa Bool&&throw(ArgumentError(
        "HOPS trajectory count must not be a Bool"))
    trajectories>0||throw(ArgumentError(
        "HOPS trajectory count must be positive"))
    BigInt(trajectories)<=typemax(Int)||throw(ArgumentError(
        "HOPS trajectory count exceeds Int indexing"))
    count=Int(trajectories)
    ts,step=_hops_times(times,R,dt)
    if initial isa WeakPIPseudoKet
        _hops_validate_initial(plan,initial)
    else
        initial.basis===plan.basis||throw(ArgumentError(
            "HOPS initial ensemble and plan use different PI bases"))
        R===_real_float_type(plan.Ttype)||throw(ArgumentError(
            "HOPS initial ensemble has incompatible scalar precision"))
        _hops_require_array_precision(
            initial.weights,plan,"HOPS initial-ensemble weights")
        _hops_require_array_precision(
            initial.cumulative,plan,
            "HOPS initial-ensemble cumulative weights")
        for amplitudes in initial.amplitudes
            _hops_require_array_precision(
                amplitudes,plan,"HOPS initial-ensemble amplitudes")
        end
    end
    desired_workers=threaded ? min(count,Threads.nthreads()) : 1
    batch=nothing
    single=nothing
    available_workers=desired_workers
    if workspace===nothing
        # The aggregate guard below counts these workspaces before allocation.
    elseif workspace isa HOPSBatchWorkspace
        batch=_check_hops_batch_workspace(workspace,plan)
        available_workers=threaded ?
            min(desired_workers,length(batch.workers)) : 1
    elseif workspace isa HOPSWorkspace
        threaded&&throw(ArgumentError(
            "threaded HOPS averages require a HOPSBatchWorkspace"))
        single=_check_hops_workspace(workspace,plan)
        available_workers=1
    else
        throw(ArgumentError(
            "workspace must be a HOPSWorkspace or HOPSBatchWorkspace"))
    end
    retained_workers=if workspace===nothing
        available_workers
    elseif batch===nothing
        1
    else
        length(batch.workers)
    end
    estimate=_hops_average_memory_bytes(
        plan,length(ts),count,available_workers,retained_workers)
    _require_performance_budget("HOPS trajectory average",estimate,
        memory_budget;guidance=
        "Reduce workers, saved times, trajectory count, or hierarchy depth.")
    if workspace===nothing
        batch=HOPSBatchWorkspace(
            plan;workers=available_workers,memory_budget=Inf)
    end
    workers=batch===nothing ? (single,) :
        batch.workers
    rngs=batch===nothing ? (MersenneTwister(seed),) : batch.rngs
    master=first(rngs)
    Random.seed!(master,seed)
    seeds=if batch===nothing
        rand(master,UInt64,count)
    else
        resize!(batch.seeds,count)
        rand!(master,batch.seeds)
        batch.seeds
    end
    prototype=zeros(plan.Ttype,length(plan.basis))
    accumulators=[
        [_OnlineStateAccumulator(prototype) for _ in eachindex(ts)]
        for _ in 1:available_workers]
    density_buffers=[similar(prototype) for _ in 1:available_workers]
    scales=_weak_pi_density_scales(plan.basis,R)
    recorders=[
        _HOPSStateAccumulatorRecorder(
            accumulators[index],density_buffers[index],scales,plan)
        for index in 1:available_workers]

    if available_workers==1
        work=first(workers)
        rng=first(rngs)
        recorder=first(recorders)
        for trajectory in 1:count
            Random.seed!(rng,seeds[trajectory])
            _hops_path_accumulate!(
                recorder,plan,initial,ts,step,work,rng)
        end
    else
        @sync for worker_index in 1:available_workers
            let work=workers[worker_index],rng=rngs[worker_index],
                recorder=recorders[worker_index]
                Threads.@spawn begin
                    # Every path uses the same fixed grid and hierarchy, so a
                    # static strided partition is balanced and avoids atomic
                    # scheduling overhead. It also fixes the reduction groups
                    # for a given worker count.
                    for trajectory in
                            worker_index:available_workers:count
                        Random.seed!(rng,seeds[trajectory])
                        _hops_path_accumulate!(
                            recorder,plan,initial,ts,step,work,rng)
                    end
                end
            end
        end
    end
    merged=accumulators[1]
    for worker_index in 2:available_workers
        for time_index in eachindex(ts)
            _merge_states!(
                merged[time_index],accumulators[worker_index][time_index])
        end
    end
    states=Vector{PIState{R,typeof(plan.basis)}}(undef,length(ts))
    spread=Vector{R}(undef,length(ts))
    stderr=Vector{R}(undef,length(ts))
    countR=_checked_statistics_count(R,count,"HOPS trajectory")
    for time_index in eachindex(ts)
        accumulator=merged[time_index]
        accumulator.count==count||throw(ErrorException(
            "internal HOPS average retained $(accumulator.count) paths instead of $count"))
        states[time_index]=PIState(plan.basis,accumulator.mean)
        if count>1
            denominator=_checked_statistics_count(
                R,count-1,"HOPS trajectory")
            spread[time_index]=sqrt(accumulator.m2/denominator)
            stderr[time_index]=spread[time_index]/sqrt(countR)
        else
            spread[time_index]=zero(R)
            stderr[time_index]=zero(R)
        end
    end
    result=HOPSEnsembleResult(ts,states,count,spread,stderr,
        (;equation=:linear,noise=:stationary_ou,
          hierarchy=hops_hierarchy_metadata(plan),
          normalization=:unnormalized_root_outer_product))
    return_info ? result : states
end

function hops_average(plan::HOPSPlan,rho::PIState,
        times,trajectories::Integer;
        initial_atol=nothing,
        initial_rtol=nothing,kwargs...)
    if _real_float_type(plan.Ttype)===BigFloat&&
            (precision(BigFloat)!=plan.precision_bits||
             rounding(BigFloat)!=plan.rounding_mode)
        return _hops_with_precision(plan) do
            hops_average(
                plan,rho,times,trajectories;
                initial_atol,initial_rtol,kwargs...)
        end
    end
    memory_budget=get(
        kwargs,:memory_budget,_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    initial=hops_initial_ensemble(
        plan,rho;atol=initial_atol,rtol=initial_rtol,memory_budget)
    hops_average(plan,initial,times,trajectories;kwargs...)
end
