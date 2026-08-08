"""
    HilbertBlockEntropyPlan(basis, blocks; label=:explicit)
    HilbertBlockEntropyPlan(basis, local_unitary; atol=1e-12, rtol=1e-10,
                            label=:diagonal_strong_symmetry)
    HilbertBlockEntropyPlan(reduction::StrongSymmetryReduction; kwargs...)

Prepare a Hilbert-space block partition for symmetry-aware total-state
entropy.  The plan is tied to one exact [`PIBasis`](@ref), owns detached
read-only local Gelfand--Tsetlin index groups, and may be shared between tasks.

The explicit constructor accepts an iterable of `sector => indices` pairs or
named tuples `(sector=..., indices=..., label=...)`.  Indices are local to the
sector's Schur block.  Every Hilbert-space index in every retained sector must
occur exactly once; incomplete or overlapping partitions are rejected.

The matrix constructor requires a diagonal local unitary and groups GT
patterns by the charge of `local_unitary^tensor N`.  Construction validates
unitarity and rejects numerically ambiguous charge clusters.  The
`StrongSymmetryReduction` constructor additionally records that the model
symmetry was certified term by term.  This model certificate alone never
asserts that an arbitrary state is block diagonal: [`block_entropy_diagnostics`](@ref)
and [`block_von_neumann_entropy`](@ref) always check the supplied state.

For block sizes ``n_b`` inside Schur sectors of sizes ``m_nu``, the spectral
work is proportional to `sum(n_b^3)` rather than `sum(m_nu^3)`.  No `d^N`
state, symmetry matrix, or full-sector eigensystem is constructed.
"""
struct HilbertBlockEntropyPlan{B,G,J,Q,L,M}
    basis::B
    blocks::G
    membership::J
    block_labels::Q
    label::L
    metadata::M
    block_count::Int
    largest_block::Int
    largest_sector::Int
    unsplit_cubic_work::BigInt
    split_cubic_work::BigInt
end

function Base.show(io::IO,plan::HilbertBlockEntropyPlan)
    print(io,"HilbertBlockEntropyPlan($(plan.block_count) blocks, largest=" *
             "$(plan.largest_block), unsplit_largest=$(plan.largest_sector), " *
             "label=$(plan.label))")
end

"""
    HilbertBlockEntropyWorkspace(plan, T=Float64;
                                 memory_budget=512*1024^2)

Allocate one task-owned dense scratch block with the size of the largest
prepared Hilbert block. Reuse it across calls to
[`block_von_neumann_entropy`](@ref) or
[`block_entropy_diagnostics`](@ref). The eigensolver overwrites this storage;
the immutable plan remains safe to share between tasks. Entropy evaluation
also guards conservative eigensolver work arrays against its `memory_budget`.
"""
mutable struct HilbertBlockEntropyWorkspace{T<:AbstractFloat,P,M,Q}
    plan::P
    block::M
    precision_bits::Int
    rounding_mode::Q
end

function HilbertBlockEntropyWorkspace(
        plan::HilbertBlockEntropyPlan,
        ::Type{T}=Float64;
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET) where
        T<:AbstractFloat
    isconcretetype(T)||throw(ArgumentError(
        "Hilbert-block entropy workspace requires a concrete floating type"))
    entries=BigInt(plan.largest_block)^2
    peak=_performance_entries_bytes(entries,Complex{T})
    _require_performance_budget(
        "Hilbert-block entropy workspace",peak,memory_budget;guidance=
        "Use a finer certified symmetry partition or raise the explicit budget.")
    precision_bits=T===BigFloat ? precision(BigFloat) : 0
    rounding_mode=T===BigFloat ? rounding(BigFloat) : nothing
    HilbertBlockEntropyWorkspace{
        T,typeof(plan),Matrix{Complex{T}},typeof(rounding_mode)}(
        plan,zeros(Complex{T},plan.largest_block,plan.largest_block),
        precision_bits,rounding_mode)
end


function Base.show(io::IO,workspace::HilbertBlockEntropyWorkspace{T}) where T
    print(io,"HilbertBlockEntropyWorkspace(maximum_block=" *
             "$(size(workspace.block,1)), scalar_type=$T)")
end

function _check_hilbert_block_workspace(
        workspace::HilbertBlockEntropyWorkspace{T},
        plan::HilbertBlockEntropyPlan,::Type{R}) where
        {T<:AbstractFloat,R<:AbstractFloat}
    workspace.plan===plan||throw(ArgumentError(
        "HilbertBlockEntropyWorkspace was prepared for a different plan"))
    T===R||throw(ArgumentError(
        "workspace scalar type $T does not match state scalar type $R"))
    size(workspace.block)==(plan.largest_block,plan.largest_block)||
        throw(DimensionMismatch(
            "Hilbert-block entropy workspace has the wrong dimensions"))
    if R===BigFloat
        bounds=_local_factor_precision_bounds(workspace.block)
        bounds==(workspace.precision_bits,workspace.precision_bits)||
            throw(ArgumentError(
                "Hilbert-block workspace has precision range $bounds, but " *
                "its recorded precision is $(workspace.precision_bits) bits"))
    end
    workspace
end

function _hilbert_block_eigvals!(block)
    try
        eigvals!(Hermitian(block))
    catch error
        error isa MethodError||rethrow()
        R=_real_float_type(eltype(block))
        throw(ArgumentError(
            "Hilbert-block entropy eigensystems are unavailable for scalar " *
            "type $R with Julia's active LinearAlgebra backend; use " *
            "Float32/Float64 data or load a compatible generic eigensolver"))
    end
end

function _hilbert_block_tolerance(::Type{R},value,label) where R<:AbstractFloat
    converted=_checked_prepared_real(value,R,label)
    converted>=zero(R)||throw(ArgumentError(
        "$label must be nonnegative"))
    converted
end

function _hilbert_block_sector(basis::PIBasis,raw)
    partition=raw isa Partition ? raw : try
        Partition(Tuple(raw))
    catch
        throw(ArgumentError(
            "an explicit entropy block sector must be a Partition or tuple"))
    end
    index=get(basis.index,partition,0)
    index>0||throw(ArgumentError(
        "explicit entropy block sector $partition is absent from the basis"))
    basis.sectors[index]
end

function _hilbert_block_entry(entry)
    if entry isa Pair
        value=last(entry)
        if value isa NamedTuple
            hasproperty(value,:indices)||throw(ArgumentError(
                "an explicit entropy block pair with named data needs `indices`"))
            return first(entry),getproperty(value,:indices),
                   hasproperty(value,:label) ? getproperty(value,:label) : nothing
        end
        return first(entry),value,nothing
    elseif entry isa NamedTuple
        hasproperty(entry,:sector)&&hasproperty(entry,:indices)||
            throw(ArgumentError(
                "an explicit entropy block needs `sector` and `indices`"))
        return getproperty(entry,:sector),getproperty(entry,:indices),
               hasproperty(entry,:label) ? getproperty(entry,:label) : nothing
    end
    throw(ArgumentError(
        "explicit entropy blocks must be pairs or named tuples"))
end

function _hilbert_block_plan(basis::PIBasis,groups,labels,label,metadata)
    # Deep-copy into plan-owned arrays. Large charge partitions can contain
    # thousands of blocks; encoding those blocks as one giant tuple makes the
    # tuple length part of the compiler type and creates pathological setup
    # allocation. These arrays are read-only by contract, like the sparse and
    # dense arrays owned by the other immutable prepared plans.
    stored_groups=[[copy(group) for group in sector_groups]
                   for sector_groups in groups]
    stored_labels=[copy(sector_labels) for sector_labels in labels]
    membership=map(stored_groups,basis.patterns) do sector_groups,patterns
        values=zeros(Int,length(patterns))
        for (block_index,group) in pairs(sector_groups),index in group
            values[index]=block_index
        end
        all(>(0),values)||error(
            "internal error: Hilbert-block partition is not exhaustive")
        values
    end
    block_count=sum(length,stored_groups;init=0)
    largest_block=0
    split=big(0)
    for sector_groups in stored_groups,group in sector_groups
        largest_block=max(largest_block,length(group))
        split+=BigInt(length(group))^3
    end
    largest_sector=maximum(length,basis.patterns;init=0)
    unsplit=big(0)
    for patterns in basis.patterns
        unsplit+=BigInt(length(patterns))^3
    end
    HilbertBlockEntropyPlan(
        basis,stored_groups,membership,stored_labels,label,metadata,block_count,
        largest_block,largest_sector,unsplit,split)
end

function _hilbert_block_prepared_scale(::Type{R},sector) where
        R<:AbstractFloat
    _prepare_exact_scale(
        R,symmetric_group_dimension(sector),one(BigInt),Val(true);
        context="multiplicity-weighted Hilbert block for $sector")
end

@inline function _hilbert_block_weighted_entry(value,scale)
    _apply_prepared_exact_scale(value,scale;
        context="multiplicity-weighted Hilbert-block entry")
end

function _hilbert_block_entropy_peak_bytes(
        plan::HilbertBlockEntropyPlan,::Type{R}) where R<:AbstractFloat
    n=plan.largest_block
    # LAPACK's divide-and-conquer Hermitian drivers may retain quadratic
    # complex and real work in addition to the caller-owned overwritten
    # block. Four complex n-by-n arrays plus linear scratch is a conservative
    # cross-backend bound for the exact block route.
    _performance_array_bytes(n,Complex{R},4;linear_arrays=8)
end

function _hilbert_full_positivity_peak_bytes(
        plan::HilbertBlockEntropyPlan,::Type{R}) where R<:AbstractFloat
    n=plan.largest_sector
    _performance_array_bytes(n,Complex{R},4;linear_arrays=8)
end

function HilbertBlockEntropyPlan(basis::PIBasis,explicit_blocks;
                                 label=:explicit)
    groups=[Vector{Vector{Int}}() for _ in basis.sectors]
    labels=[Any[] for _ in basis.sectors]
    for (entry_index,entry) in enumerate(explicit_blocks)
        raw_sector,raw_indices,raw_label=_hilbert_block_entry(entry)
        sector=_hilbert_block_sector(basis,raw_sector)
        sector_index=basis.index[sector]
        dimension=length(basis.patterns[sector_index])
        indices=Int[]
        for raw_index in raw_indices
            raw_index isa Integer&&!(raw_index isa Bool)||throw(ArgumentError(
                "explicit entropy block indices must be integers"))
            1<=raw_index<=dimension||throw(BoundsError(1:dimension,raw_index))
            push!(indices,Int(raw_index))
        end
        isempty(indices)&&throw(ArgumentError(
            "an explicit entropy block must contain at least one index"))
        sort!(indices)
        allunique(indices)||throw(ArgumentError(
            "indices within an explicit entropy block must be distinct"))
        push!(groups[sector_index],indices)
        push!(labels[sector_index],raw_label===nothing ?
              (sector=sector,entry=entry_index) : raw_label)
    end
    for (sector_index,sector) in pairs(basis.sectors)
        dimension=length(basis.patterns[sector_index])
        counts=zeros(Int,dimension)
        for group in groups[sector_index],index in group
            counts[index]+=1
        end
        missing=findall(iszero,counts)
        repeated=findall(>(1),counts)
        isempty(missing)||throw(ArgumentError(
            "explicit entropy blocks omit GT indices $missing in sector $sector"))
        isempty(repeated)||throw(ArgumentError(
            "explicit entropy blocks overlap at GT indices $repeated in sector $sector"))
        order=sortperm(groups[sector_index];by=first)
        groups[sector_index]=groups[sector_index][order]
        labels[sector_index]=labels[sector_index][order]
    end
    _hilbert_block_plan(basis,groups,labels,label,
        (kind=:explicit_partition,model_symmetry_certified=false))
end

function _hilbert_charge_match(left,right,atol,rtol)
    abs(left-right)<=atol+rtol*max(abs(left),abs(right),one(atol))
end

function _hilbert_charge_group_is_clique(phases,group,atol,rtol)
    length(group)<=1&&return true
    minimum_real=minimum(index->real(phases[index]),group)
    maximum_real=maximum(index->real(phases[index]),group)
    minimum_imag=minimum(index->imag(phases[index]),group)
    maximum_imag=maximum(index->imag(phases[index]),group)
    # The bounding-box diagonal is an inexpensive upper bound on every pair
    # distance. It certifies the usual repeated-root charge groups in O(m).
    guaranteed=atol+rtol
    hypot(maximum_real-minimum_real,maximum_imag-minimum_imag)<=guaranteed&&
        return true
    # Only a numerically marginal cluster reaches this cold ambiguity check.
    for right_position in 2:length(group)
        right=group[right_position]
        for left_position in 1:right_position-1
            left=group[left_position]
            _hilbert_charge_match(phases[left],phases[right],atol,rtol)||
                return false
        end
    end
    true
end

function _hilbert_charge_groups(phases,atol,rtol)
    dimension=length(phases)
    dimension==0&&return Vector{Vector{Int}}()
    # Unit-circle sorting replaces the former all-pairs connected-component
    # search. Equal charges are contiguous. Rotate after the largest angular
    # gap so a charge close to the -pi/pi branch cut is not split.
    order=sortperm(eachindex(phases);by=index->angle(phases[index]))
    if dimension>1
        angles=map(index->angle(phases[index]),order)
        two_pi=oftype(angles[1],2)*oftype(angles[1],pi)
        gap_index=dimension
        largest_gap=(angles[1]+two_pi)-angles[end]
        for index in 1:dimension-1
            gap=angles[index+1]-angles[index]
            if gap>largest_gap
                largest_gap=gap
                gap_index=index
            end
        end
        gap_index<dimension&&
            (order=vcat(view(order,gap_index+1:dimension),
                        view(order,1:gap_index)))
    end
    groups=Vector{Vector{Int}}()
    current=Int[first(order)]
    for position in 2:dimension
        target=order[position]
        matches_first=_hilbert_charge_match(
            phases[first(current)],phases[target],atol,rtol)
        matches_last=_hilbert_charge_match(
            phases[last(current)],phases[target],atol,rtol)
        if matches_first&&matches_last
            push!(current,target)
        elseif matches_first||matches_last
            throw(ArgumentError(
                "diagonal symmetry charges form an ambiguous tolerance " *
                "cluster; reduce atol/rtol or provide an explicit block partition"))
        else
            any(index->_hilbert_charge_match(
                    phases[index],phases[target],atol,rtol),current)&&
                throw(ArgumentError(
                    "diagonal symmetry charges form an ambiguous tolerance " *
                    "cluster; reduce atol/rtol or provide an explicit block partition"))
            _hilbert_charge_group_is_clique(phases,current,atol,rtol)||
                throw(ArgumentError(
                    "diagonal symmetry charges form an ambiguous tolerance " *
                    "cluster; reduce atol/rtol or provide an explicit block partition"))
            sort!(current)
            push!(groups,current)
            current=Int[target]
        end
    end
    _hilbert_charge_group_is_clique(phases,current,atol,rtol)||
        throw(ArgumentError(
            "diagonal symmetry charges form an ambiguous tolerance cluster; " *
            "reduce atol/rtol or provide an explicit block partition"))
    sort!(current);push!(groups,current)
    sort!(groups;by=first)
    groups
end

_hilbert_block_stored_matrix_values(matrix::SparseMatrixCSC)=nonzeros(matrix)
_hilbert_block_stored_matrix_values(matrix::Diagonal)=matrix.diag
_hilbert_block_stored_matrix_values(matrix)=matrix

function _diagonal_hilbert_block_plan(basis::PIBasis,
        local_unitary::AbstractMatrix;atol::Real,rtol::Real,label,
        model_symmetry_certified,certificate)
    size(local_unitary)==(basis.d,basis.d)||throw(DimensionMismatch(
        "the local symmetry must be $(basis.d) by $(basis.d)"))
    CT=_complex_float_type(eltype(local_unitary))
    R=_real_float_type(CT)
    if R===BigFloat
        bounds=_local_factor_precision_bounds(
            _hilbert_block_stored_matrix_values(local_unitary))
        bounds[1]==bounds[2]||throw(ArgumentError(
            "local symmetry BigFloat storage has mixed precision $bounds"))
        if precision(BigFloat)!=bounds[1]
            return setprecision(BigFloat,bounds[1]) do
                _diagonal_hilbert_block_plan(
                    basis,local_unitary;atol,rtol,label,
                    model_symmetry_certified,certificate)
            end
        end
    end
    atolR=_hilbert_block_tolerance(R,atol,"atol")
    rtolR=_hilbert_block_tolerance(R,rtol,"rtol")
    matrix=Matrix{CT}(local_unitary)
    all(value->isfinite(real(value))&&isfinite(imag(value)),matrix)||
        throw(ArgumentError("the local symmetry must contain only finite values"))
    diagonal=diag(matrix)
    scale=max(norm(matrix,Inf),one(R))
    norm(matrix-Diagonal(diagonal),Inf)<=atolR+rtolR*scale||
        throw(ArgumentError(
            "Hilbert-space charge blocks require a diagonal local unitary"))
    all(value->abs(abs(value)-one(R))<=atolR+rtolR,diagonal)||
        throw(ArgumentError(
            "every diagonal local-symmetry eigenvalue must have unit modulus"))

    groups=Vector{Vector{Vector{Int}}}(undef,length(basis.sectors))
    labels=Vector{Vector{Any}}(undef,length(basis.sectors))
    for (sector_index,patterns) in pairs(basis.patterns)
        phases=Vector{CT}(undef,length(patterns))
        for (pattern_index,pattern) in pairs(patterns)
            occupations=content(pattern)
            phase=one(CT)
            @inbounds for level in eachindex(diagonal)
                exponent=occupations[level]
                iszero(exponent)&&continue
                factor=diagonal[level]^exponent
                isfinite(factor)||throw(ArgumentError(
                    "an N-particle symmetry charge is nonfinite; use a " *
                    "more accurate local unitary or wider scalar type"))
                !iszero(diagonal[level])&&iszero(factor)&&throw(ArgumentError(
                    "a nonzero N-particle symmetry charge underflows; use a " *
                    "more accurate local unitary or wider scalar type"))
                previous=phase
                phase*=factor
                isfinite(phase)||throw(ArgumentError(
                    "an N-particle symmetry charge is nonfinite; use a " *
                    "more accurate local unitary or wider scalar type"))
                !iszero(previous)&&!iszero(factor)&&iszero(phase)&&
                    throw(ArgumentError(
                        "a nonzero N-particle symmetry charge underflows; " *
                        "use a more accurate local unitary or wider scalar type"))
            end
            abs(abs(phase)-one(R))<=atolR+rtolR||throw(ArgumentError(
                "the powered N-particle symmetry charge is not unit modulus " *
                "within tolerance; use a more accurate local unitary or an " *
                "explicit Hilbert-block partition"))
            phases[pattern_index]=phase
        end
        sector_groups=_hilbert_charge_groups(phases,atolR,rtolR)
        groups[sector_index]=sector_groups
        labels[sector_index]=Any[phases[first(group)] for group in sector_groups]
    end
    metadata=(kind=:diagonal_local_unitary,
              diagonal=Tuple(diagonal),unitary_atol=atolR,
              unitary_rtol=rtolR,model_symmetry_certified,certificate)
    _hilbert_block_plan(basis,groups,labels,label,metadata)
end

function HilbertBlockEntropyPlan(basis::PIBasis,
        local_unitary::AbstractMatrix;
        atol::Real=1//1_000_000_000_000,
        rtol::Real=1//10_000_000_000,
        label=:diagonal_strong_symmetry)
    _diagonal_hilbert_block_plan(
        basis,local_unitary;atol,rtol,label,
        model_symmetry_certified=false,certificate=:state_checked_at_evaluation)
end

function HilbertBlockEntropyPlan(reduction::StrongSymmetryReduction;
        atol::Real=1//1_000_000_000_000,
        rtol::Real=1//10_000_000_000,label=nothing)
    candidate=reduction.candidate
    candidate.status===true||throw(ArgumentError(
        "the StrongSymmetryReduction does not retain a certified candidate"))
    selected_label=label===nothing ?
        (kind=:certified_strong_symmetry,candidate=candidate.name) : label
    _diagonal_hilbert_block_plan(
        reduction.model.basis,candidate.unitary;atol,rtol,
        label=selected_label,model_symmetry_certified=true,
        certificate=(validation=candidate.validation,
                     candidate=candidate.name,
                     trace_bearing_sectors=length(reduction.sectors)))
end

"""Diagnostics returned by symmetry-aware Hilbert-block entropy analysis."""
struct HilbertBlockEntropyDiagnostics{Z,R<:AbstractFloat,L}
    plan_label::L
    trace_value::Z
    trace_error::R
    trace_tolerance::R
    hermiticity_error::R
    hermiticity_tolerance::R
    offblock_error::R
    offblock_tolerance::R
    minimum_block_eigenvalue::R
    maximum_positivity_tolerance::R
    trace_one::Bool
    hermitian::Bool
    block_diagonal::Bool
    exactly_block_diagonal::Bool
    projected_positive::Bool
    input_positive::Bool
    positivity_certification::Symbol
    valid::Bool
    reason::Symbol
    block_reason::Symbol
    block_count::Int
    largest_block::Int
    largest_sector::Int
    unsplit_cubic_work::BigInt
    split_cubic_work::BigInt
    estimated_cubic_fraction::R
end

function Base.show(io::IO,diagnostics::HilbertBlockEntropyDiagnostics)
    print(io,"HilbertBlockEntropyDiagnostics(valid=$(diagnostics.valid), " *
             "reason=$(diagnostics.reason), blocks=$(diagnostics.block_count), " *
             "estimated_cubic_fraction=$(diagnostics.estimated_cubic_fraction))")
end

function _hilbert_block_entropy_analysis(rho::PIState,
        plan::HilbertBlockEntropyPlan;atol::Real,rtol::Real,
        block_atol::Real,block_rtol::Real,base::Real,compute_entropy::Bool,
        workspace,memory_budget)
    rho.basis===plan.basis||throw(ArgumentError(
        "HilbertBlockEntropyPlan belongs to a different PIBasis"))
    R=_real_float_type(eltype(rho.data))
    if R===BigFloat
        bounds=_local_factor_precision_bounds(rho.data)
        bounds[1]==bounds[2]||throw(ArgumentError(
            "Hilbert-block entropy input has mixed BigFloat precision $bounds"))
        target_precision=bounds[1]
        target_rounding=workspace isa HilbertBlockEntropyWorkspace ?
            workspace.rounding_mode : rounding(BigFloat)
        if precision(BigFloat)!=target_precision||
                rounding(BigFloat)!=target_rounding
            return setrounding(BigFloat,target_rounding) do
                setprecision(BigFloat,target_precision) do
                    _hilbert_block_entropy_analysis(
                        rho,plan;atol,rtol,block_atol,block_rtol,base,
                        compute_entropy,workspace,memory_budget)
                end
            end
        end
        workspace isa HilbertBlockEntropyWorkspace&&
            workspace.precision_bits!=target_precision&&throw(ArgumentError(
                "Hilbert-block workspace precision " *
                "$(workspace.precision_bits) does not match state precision " *
                "$target_precision"))
    end
    atolR=_hilbert_block_tolerance(R,atol,"atol")
    rtolR=_hilbert_block_tolerance(R,rtol,"rtol")
    block_atolR=_hilbert_block_tolerance(R,block_atol,"block_atol")
    block_rtolR=_hilbert_block_tolerance(R,block_rtol,"block_rtol")
    base isa Real&&isfinite(base)&&base>0&&base!=1||throw(ArgumentError(
        "invalid logarithm base"))
    baseR=_checked_prepared_real(base,R,"logarithm base")
    baseR>zero(R)&&baseR!=one(R)||throw(ArgumentError(
        "logarithm base is not representable in $R"))
    _require_performance_budget(
        "Hilbert-block entropy spectral analysis",
        _hilbert_block_entropy_peak_bytes(plan,R),memory_budget;guidance=
        "Use a finer certified symmetry partition or raise the explicit budget.")
    work=workspace===nothing ?
        HilbertBlockEntropyWorkspace(plan,R;memory_budget) : workspace
    work isa HilbertBlockEntropyWorkspace||throw(ArgumentError(
        "workspace must be a HilbertBlockEntropyWorkspace"))
    _check_hilbert_block_workspace(work,plan,R)

    trace_value=trace(rho)
    isfinite(trace_value)||throw(ArgumentError(
        "block entropy encountered a nonfinite state trace"))
    trace_error=R(abs(trace_value-one(trace_value)))
    trace_tolerance=atolR+rtolR
    hermiticity_error=zero(R)
    hermiticity_scale=zero(R)
    offblock_error=zero(R)
    block_scale=zero(R)
    minimum_eigenvalue=R(Inf)
    maximum_positivity_tolerance=zero(R)
    projected_positive=true
    entropy=zero(R)

    for (sector_index,sector) in pairs(plan.basis.sectors)
        block=coefficient_block(rho,sector)
        multiplicity_scale=_hilbert_block_prepared_scale(R,sector)
        local_blocks=plan.blocks[sector_index]
        membership=plan.membership[sector_index]
        local_hermiticity=zero(R)
        local_scale=zero(R)
        local_offblock=zero(R)
        @inbounds for row in axes(block,1)
            hermiticity_row=zero(R)
            scale_row=zero(R)
            offblock_row=zero(R)
            for column in axes(block,2)
                value=_hilbert_block_weighted_entry(
                    block[row,column],multiplicity_scale)
                isfinite(real(value))&&isfinite(imag(value))||
                    throw(ArgumentError(
                        "block entropy encountered nonfinite data in sector $sector"))
                magnitude=R(abs(value))
                scale_row+=magnitude
                transpose_value=_hilbert_block_weighted_entry(
                    block[column,row],multiplicity_scale)
                hermiticity_row+=R(abs(value-conj(transpose_value)))
                membership[row]==membership[column]||
                    (offblock_row+=magnitude)
            end
            local_scale=max(local_scale,scale_row)
            local_hermiticity=max(local_hermiticity,hermiticity_row)
            local_offblock=max(local_offblock,offblock_row)
        end
        hermiticity_error=max(hermiticity_error,local_hermiticity)
        hermiticity_scale=max(hermiticity_scale,local_scale)
        block_scale=max(block_scale,local_scale)
        offblock_error=max(offblock_error,local_offblock)

        log_multiplicity=_log_schur_multiplicity(R,sector)
        for indices in local_blocks
            dimension=length(indices)
            subblock=@view work.block[1:dimension,1:dimension]
            @inbounds for column in 1:dimension,row in 1:dimension
                source_row=indices[row];source_column=indices[column]
                value=_hilbert_block_weighted_entry(
                    block[source_row,source_column],multiplicity_scale)
                transpose_value=_hilbert_block_weighted_entry(
                    block[source_column,source_row],multiplicity_scale)
                subblock[row,column]=(value+conj(transpose_value))/R(2)
            end
            values=_hilbert_block_eigvals!(subblock)
            spectral_scale=maximum(abs,values;init=zero(R))
            positivity_tolerance=atolR+rtolR*spectral_scale
            maximum_positivity_tolerance=max(
                maximum_positivity_tolerance,positivity_tolerance)
            local_minimum=minimum(values;init=R(Inf))
            minimum_eigenvalue=min(minimum_eigenvalue,local_minimum)
            local_minimum>=-positivity_tolerance||(projected_positive=false)
            if compute_entropy
                probability=zero(R)
                for value in values
                    value>zero(R)||continue
                    probability+=value
                    entropy-=value*log(value)
                end
                entropy+=probability*log_multiplicity
            end
        end
    end

    hermiticity_tolerance=atolR+rtolR*hermiticity_scale
    offblock_tolerance=block_atolR+block_rtolR*max(block_scale,one(R))
    trace_one=trace_error<=trace_tolerance
    hermitian=hermiticity_error<=hermiticity_tolerance
    block_diagonal=offblock_error<=offblock_tolerance
    exactly_block_diagonal=iszero(offblock_error)
    input_positive=false
    positivity_certification=:not_checked
    if exactly_block_diagonal
        input_positive=projected_positive
        positivity_certification=:prepared_hilbert_blocks
    elseif trace_one&&hermitian&&block_diagonal
        # A tolerance-sized off-block perturbation can make an otherwise PSD
        # block projection indefinite.  Certify the unmodified input on this
        # explicitly approximate path.  Exact support never pays for this
        # fallback full-sector factorization.
        _require_performance_budget(
            "tolerance-projected Hilbert-block positivity certification",
            _hilbert_full_positivity_peak_bytes(plan,R),memory_budget;guidance=
            "Use exact block support, a finer basis, or raise the explicit budget.")
        positivity=positivity_diagnostics(
            rho;atol=atolR,rtol=rtolR,method=:auto)
        input_positive=positivity.positive
        positivity_certification=Symbol(:full_state_,positivity.method)
    end
    valid=trace_one&&hermitian&&block_diagonal&&projected_positive&&
          input_positive
    block_reason=exactly_block_diagonal ? :exact :
                 block_diagonal ? :within_tolerance : :offblock_leakage
    reason=!trace_one ? :trace_not_one :
           !hermitian ? :not_hermitian :
           !block_diagonal ? :offblock_leakage :
           !(projected_positive&&input_positive) ? :not_positive :
           block_reason
    work_fraction=_checked_exact_ratio(
        R,plan.split_cubic_work,plan.unsplit_cubic_work;
        context="estimated Hilbert-block cubic-work fraction")
    diagnostics=HilbertBlockEntropyDiagnostics(
        plan.label,trace_value,trace_error,trace_tolerance,
        hermiticity_error,hermiticity_tolerance,offblock_error,
        offblock_tolerance,minimum_eigenvalue,
        maximum_positivity_tolerance,trace_one,hermitian,block_diagonal,
        exactly_block_diagonal,projected_positive,input_positive,
        positivity_certification,valid,reason,block_reason,
        plan.block_count,plan.largest_block,plan.largest_sector,
        plan.unsplit_cubic_work,plan.split_cubic_work,work_fraction)
    entropy_value=compute_entropy ? entropy/log(baseR) : missing
    entropy_value,diagnostics
end

"""
    block_entropy_diagnostics(rho, plan; atol, rtol,
                              block_atol=0, block_rtol=0,
                              workspace=nothing,
                              memory_budget=512*1024^2)

Check whether `rho` is a trace-one, Hermitian, positive state compatible with
the prepared Hilbert-space block partition. With exact block support,
positivity is tested by diagonalizing only the prepared subblocks. An explicit
tolerance-projected interpretation additionally certifies positivity of the
unmodified state. The state is never normalized, symmetrized, clipped, or
otherwise modified.

Block certification is structurally strict by default.  Passing a nonzero
`block_atol` or `block_rtol` explicitly accepts tolerance-sized off-block
coherences; `block_reason == :within_tolerance` then records that subsequent
entropy evaluation describes the block-diagonal projection rather than the
unmodified input exactly.
"""
function block_entropy_diagnostics(rho::PIState,
        plan::HilbertBlockEntropyPlan;
        atol::Real=_analysis_atol(rho),rtol::Real=_state_rtol(rho),
        block_atol::Real=0,block_rtol::Real=0,workspace=nothing,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    _,diagnostics=_hilbert_block_entropy_analysis(
        rho,plan;atol,rtol,block_atol,block_rtol,base=2,
        compute_entropy=false,workspace,memory_budget)
    diagnostics
end

"""
    block_von_neumann_entropy(rho, plan; base=2, atol, rtol,
                              block_atol=0, block_rtol=0,
                              workspace=nothing,
                              memory_budget=512*1024^2,
                              return_info=false)

Compute total von Neumann entropy using a certified Hilbert-space block
partition.  Each multiplicity-weighted Schur block is split into the prepared
charge/support blocks, and only those smaller Hermitian eigensystems are
solved.  Symmetric-group multiplicities are included exactly as in
[`von_neumann_entropy`](@ref).

The state and its block structure are validated during the same pass.  Exact
block support is required by default.  Explicit nonzero `block_atol` or
`block_rtol` values opt into the block-diagonal projection described by
[`block_entropy_diagnostics`](@ref); this is reported and never silent.
`return_info=true` returns `(value, diagnostics)`.
"""
function block_von_neumann_entropy(rho::PIState,
        plan::HilbertBlockEntropyPlan;base::Real=2,
        atol::Real=_analysis_atol(rho),rtol::Real=_state_rtol(rho),
        block_atol::Real=0,block_rtol::Real=0,
        workspace=nothing,memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,
        return_info::Bool=false)
    value,diagnostics=_hilbert_block_entropy_analysis(
        rho,plan;atol,rtol,block_atol,block_rtol,base,
        compute_entropy=true,workspace,memory_budget)
    diagnostics.trace_one||throw(ArgumentError(
        "state is not trace one: error=$(diagnostics.trace_error), " *
        "tolerance=$(diagnostics.trace_tolerance)"))
    diagnostics.hermitian||throw(ArgumentError(
        "state is not Hermitian: error=$(diagnostics.hermiticity_error), " *
        "tolerance=$(diagnostics.hermiticity_tolerance)"))
    diagnostics.block_diagonal||throw(ArgumentError(
        "state is not compatible with the prepared Hilbert-space blocks: " *
        "off-block error=$(diagnostics.offblock_error), " *
        "tolerance=$(diagnostics.offblock_tolerance)"))
    diagnostics.projected_positive&&diagnostics.input_positive||throw(ArgumentError(
        "state is not positive within the requested tolerances: " *
        "minimum eigenvalue=$(diagnostics.minimum_block_eigenvalue), " *
        "maximum block tolerance=$(diagnostics.maximum_positivity_tolerance), " *
        "certification=$(diagnostics.positivity_certification)"))
    return_info ? (value=value,diagnostics=diagnostics) : value
end
