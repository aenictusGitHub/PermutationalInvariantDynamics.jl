"""
    population_dimension(basis)

Return the number ``sum_nu dim(U_nu)`` of diagonal GT-pattern populations in
`basis`.  This differs from `length(basis)`, which counts all
``sum_nu dim(U_nu)^2`` PI operator coordinates.
"""
population_dimension(b::PIBasis)=sum(length,b.patterns;init=0)

"""
    PopulationInvarianceReport

Result of testing whether a PI generator preserves Schur-basis diagonality.
`reason == :certified` denotes exact zero leakage at the selected scalar
precision. `:within_tolerance` is used only when the caller explicitly
supplies a nonzero tolerance that accepts a nonzero residual;
`:offdiagonal_leakage` rejects the restriction. Nonfinite generator data or
tolerances throw and are never certified.
"""
struct PopulationInvarianceReport{R<:AbstractFloat}
    invariant::Union{Bool,Missing}
    maximum_leakage::Union{R,Missing}
    scale::Union{R,Missing}
    tolerance::Union{R,Missing}
    population_dimension::Int
    pi_dimension::Int
    components::Int
    failing_component::Union{Nothing,Int}
    reason::Symbol
end

function show(io::IO,report::PopulationInvarianceReport)
    print(io,"PopulationInvarianceReport(invariant=$(report.invariant), " *
             "reason=$(report.reason), population_dimension=$(report.population_dimension), " *
             "pi_dimension=$(report.pi_dimension))")
end

# A unit entry in the public population vector is the full physical
# probability of one GT pattern, including all f^nu multiplicity copies.  Its
# stored equation-(7) coefficient is therefore 1/sqrt(f^nu).  Keeping these
# conversions in one immutable map avoids repeating convention-sensitive
# indexing throughout compilation and state conversion.
struct _PopulationCoordinateMap{R<:AbstractFloat}
    diagonal_indices::Vector{Int}
    diagonal_lookup::Dict{Int,Int}
    scales::Vector{R}
    inverse_scales::Vector{R}
    sector_offsets::Vector{Int}
end

function _population_coordinate_map(b::PIBasis,::Type{R}) where R<:AbstractFloat
    np=population_dimension(b)
    indices=Vector{Int}(undef,np)
    lookup=Dict{Int,Int}()
    scales=Vector{R}(undef,np)
    inverse_scales=Vector{R}(undef,np)
    sector_offsets=Vector{Int}(undef,length(b.sectors)+1)
    population_index=1
    for (sector_index,partition) in pairs(b.sectors)
        sector_offsets[sector_index]=population_index
        n=length(b.patterns[sector_index])
        multiplicity_scale=_schur_multiplicity_scale(R,partition)
        for pattern_index in 1:n
            coordinate=b.offsets[sector_index]+(pattern_index-1)*(n+1)
            indices[population_index]=coordinate
            lookup[coordinate]=population_index
            scales[population_index]=multiplicity_scale
            inverse_scales[population_index]=inv(multiplicity_scale)
            population_index+=1
        end
    end
    sector_offsets[end]=population_index
    _PopulationCoordinateMap(indices,lookup,scales,inverse_scales,sector_offsets)
end

function _population_tolerances(::Type{R},atol,rtol) where R<:AbstractFloat
    absolute=atol===nothing ? zero(R) : R(atol)
    relative=rtol===nothing ? zero(R) : R(rtol)
    isfinite(absolute)||throw(ArgumentError("atol must be finite"))
    isfinite(relative)||throw(ArgumentError("rtol must be finite"))
    absolute>=zero(R)||throw(ArgumentError("atol must be nonnegative"))
    relative>=zero(R)||throw(ArgumentError("rtol must be nonnegative"))
    absolute,relative
end

@inline function _push_population_raw!(I,J,V,row,column,value,::Type{T}) where T
    converted=convert(T,value)
    isfinite(converted)||throw(ArgumentError(
        "population restriction encountered a nonfinite generator entry"))
    iszero(converted)&&return nothing
    push!(I,row);push!(J,column);push!(V,converted)
    nothing
end

function _empty_population_raw(b,map,::Type{T}) where T
    spzeros(T,length(b),length(map.diagonal_indices))
end

# Build the unscaled map from physical populations to full PI coefficient
# derivatives.  These routines work directly from prepared kernels: they do
# not materialize the n_PI by n_PI Liouvillian and are substantially cheaper
# than probing every population with a full matrix-free application.
function _population_raw_action(kernel::HamiltonianPIKernel,b,map,::Type{T}) where T
    I=Int[];J=Int[];V=T[]
    for sector_index in eachindex(b.sectors)
        K=kernel.blocks[sector_index]
        n=length(b.patterns[sector_index]);offset=b.offsets[sector_index]
        population_offset=map.sector_offsets[sector_index]
        for source in 1:n
            column=population_offset+source-1
            inverse_scale=map.inverse_scales[column]
            for row_index in 1:n
                value=(-1im)*K[row_index,source]*inverse_scale
                coordinate=offset+row_index-1+(source-1)*n
                _push_population_raw!(I,J,V,coordinate,column,value,T)
            end
            for column_index in 1:n
                value=(1im)*K[source,column_index]*inverse_scale
                coordinate=offset+source-1+(column_index-1)*n
                _push_population_raw!(I,J,V,coordinate,column,value,T)
            end
        end
    end
    sparse(I,J,V,length(b),length(map.diagonal_indices))
end

function _append_anticommutator_raw!(I,J,V,Q,b,map,sector_index,::Type{T}) where T
    n=length(b.patterns[sector_index]);offset=b.offsets[sector_index]
    population_offset=map.sector_offsets[sector_index]
    half=one(T)/2
    for source in 1:n
        column=population_offset+source-1
        inverse_scale=map.inverse_scales[column]
        for row_index in 1:n
            coordinate=offset+row_index-1+(source-1)*n
            _push_population_raw!(I,J,V,coordinate,column,
                                  -half*Q[row_index,source]*inverse_scale,T)
        end
        for column_index in 1:n
            coordinate=offset+source-1+(column_index-1)*n
            _push_population_raw!(I,J,V,coordinate,column,
                                  -half*Q[source,column_index]*inverse_scale,T)
        end
    end
    nothing
end

function _population_raw_action(kernel::DissipatorPIKernel,b,map,::Type{T}) where T
    I=Int[];J=Int[];V=T[]
    for sector_index in eachindex(b.sectors)
        K=kernel.blocks[sector_index];Q=kernel.qblocks[sector_index]
        n=length(b.patterns[sector_index]);offset=b.offsets[sector_index]
        population_offset=map.sector_offsets[sector_index]
        for source in 1:n
            column=population_offset+source-1
            inverse_scale=map.inverse_scales[column]
            for output_column in 1:n
                right=conj(K[output_column,source]);iszero(right)&&continue
                for output_row in 1:n
                    left=K[output_row,source];iszero(left)&&continue
                    coordinate=offset+output_row-1+(output_column-1)*n
                    _push_population_raw!(I,J,V,coordinate,column,
                                          left*right*inverse_scale,T)
                end
            end
        end
        _append_anticommutator_raw!(I,J,V,Q,b,map,sector_index,T)
    end
    sparse(I,J,V,length(b),length(map.diagonal_indices))
end

function _population_raw_action(kernel::LocalJumpPIKernel,b,map,::Type{T}) where T
    I=Int[];J=Int[];V=T[]
    for index in eachindex(kernel.gain.V)
        population_column=get(map.diagonal_lookup,kernel.gain.J[index],0)
        iszero(population_column)&&continue
        value=kernel.gain.V[index]*map.inverse_scales[population_column]
        _push_population_raw!(I,J,V,kernel.gain.I[index],population_column,value,T)
    end
    for sector_index in eachindex(b.sectors)
        _append_anticommutator_raw!(I,J,V,kernel.qblocks[sector_index],
                                    b,map,sector_index,T)
    end
    sparse(I,J,V,length(b),length(map.diagonal_indices))
end

function _population_raw_action(kernel::FactorizedLocalJumpPIKernel,b,map,
                                ::Type{T}) where T
    I=Int[];J=Int[];V=T[]
    @inbounds for branch_index in eachindex(kernel.branches.entries)
        branch=kernel.branches.entries[branch_index]
        li=branch.output_sector;ni=branch.input_sector
        nl=length(b.patterns[li]);nn=length(b.patterns[ni])
        contraction=kernel.contractions[branch_index]
        population_offset=map.sector_offsets[ni]
        for source in 1:nn
            column=population_offset+source-1
            factor=branch.scale*map.inverse_scales[column]
            for output_column in 1:nl,output_row in 1:nl
                value=factor*contraction[output_row,source]*
                    conj(contraction[output_column,source])
                coordinate=b.offsets[li]+output_row-1+(output_column-1)*nl
                _push_population_raw!(I,J,V,coordinate,column,value,T)
            end
        end
    end
    for sector_index in eachindex(b.sectors)
        _append_anticommutator_raw!(I,J,V,kernel.qblocks[sector_index],
                                    b,map,sector_index,T)
    end
    sparse(I,J,V,length(b),length(map.diagonal_indices))
end

function _population_raw_action(kernel::FactorizedLocalPBodyJumpPIKernel,b,map,
                                ::Type{T}) where T
    I=Int[];J=Int[];V=T[]
    @inbounds for (li,ni,first_pair,last_pair) in kernel.groups
        nl=length(b.patterns[li]);nn=length(b.patterns[ni])
        population_offset=map.sector_offsets[ni]
        for source in 1:nn
            column=population_offset+source-1
            inverse_scale=map.inverse_scales[column]
            for pair in first_pair:last_pair
                contraction=kernel.contractions[pair]
                exact_scale=kernel.pair_scales[pair]
                for output_column in 1:nl,output_row in 1:nl
                    primitive=contraction[output_row,source]*
                        conj(contraction[output_column,source])
                    value=inverse_scale*(exact_scale.direct ?
                        exact_scale.factor*primitive :
                        _apply_prepared_exact_scale(primitive,exact_scale;
                            context="population local p-body gain"))
                    coordinate=b.offsets[li]+output_row-1+
                        (output_column-1)*nl
                    _push_population_raw!(I,J,V,coordinate,column,value,T)
                end
            end
        end
    end
    for sector_index in eachindex(b.sectors)
        _append_anticommutator_raw!(I,J,V,kernel.qblocks[sector_index],
                                    b,map,sector_index,T)
    end
    sparse(I,J,V,length(b),length(map.diagonal_indices))
end

function _population_kernel_scale(kernel::HamiltonianPIKernel,t,p,::Type{T}) where T
    value=value_at(kernel.scale,t,p)
    value isa Number||throw(ArgumentError(
        "a Hamiltonian rate must evaluate to a number, got $(typeof(value))"))
    promote_type(T,typeof(value))===T||throw(ArgumentError(
        "evaluated Hamiltonian rate type $(typeof(value)) is wider than population plan type $T"))
    converted=convert(T,value)
    isfinite(converted)||throw(ArgumentError(
        "Hamiltonian rate must evaluate to a finite number"))
    converted
end
function _population_kernel_scale(kernel::Union{DissipatorPIKernel,LocalJumpPIKernel,
        FactorizedLocalJumpPIKernel,FactorizedLocalPBodyJumpPIKernel},
                                  t,p,::Type{T}) where T
    value=_evaluated_dissipative_rate(kernel.scale,t,p)
    promote_type(T,typeof(value))===T||throw(ArgumentError(
        "evaluated dissipative rate type $(typeof(value)) is wider than population plan type $T"))
    converted=convert(T,value)
    isfinite(converted)||throw(ArgumentError(
        "dissipative rate must evaluate to a finite number"))
    converted
end

function _append_scaled_population_sparse!(I,J,V,raw,scale,::Type{T}) where T
    rows=rowvals(raw);values=nonzeros(raw)
    for column in axes(raw,2),index in nzrange(raw,column)
        _push_population_raw!(I,J,V,rows[index],column,
                              scale*values[index],T)
    end
    nothing
end

function _autonomous_population_raw(kernels,b,map,::Type{T}) where T
    I=Int[];J=Int[];V=T[]
    for kernel in kernels
        scale=_population_kernel_scale(kernel,0.0,nothing,T)
        iszero(scale)&&continue
        raw=_population_raw_action(kernel,b,map,T)
        _append_scaled_population_sparse!(I,J,V,raw,scale,T)
    end
    nrows=length(b);ncolumns=length(map.diagonal_indices)
    sparse(I,J,V,nrows,ncolumns)
end

function _extract_population_matrix(raw,b,map,absolute,relative,components;
                                    failing_component=nothing)
    R=eltype(map.scales);T=eltype(raw)
    I=Int[];J=Int[];V=T[]
    maximum_leakage=zero(R);scale=zero(R)
    rows=rowvals(raw);values=nonzeros(raw)
    for column in axes(raw,2),index in nzrange(raw,column)
        value=values[index]
        isfinite(value)||throw(ArgumentError(
            "population restriction encountered a nonfinite generator entry"))
        magnitude=abs(value)
        isfinite(magnitude)||throw(ArgumentError(
            "population restriction encountered a nonfinite generator magnitude"))
        scale=max(scale,R(magnitude))
        population_row=get(map.diagonal_lookup,rows[index],0)
        if iszero(population_row)
            maximum_leakage=max(maximum_leakage,R(magnitude))
        else
            projected=map.scales[population_row]*value
            isfinite(projected)||throw(ArgumentError(
                "population restriction encountered a nonfinite projected entry"))
            iszero(projected)||begin
                push!(I,population_row);push!(J,column);push!(V,projected)
            end
        end
    end
    tolerance=absolute+relative*scale
    isfinite(tolerance)||throw(ArgumentError(
        "population-invariance tolerance became nonfinite"))
    invariant=maximum_leakage<=tolerance
    reason=invariant ? (iszero(maximum_leakage) ? :certified : :within_tolerance) :
                       :offdiagonal_leakage
    report=PopulationInvarianceReport{R}(invariant,maximum_leakage,scale,tolerance,
        length(map.diagonal_indices),length(b),components,
        invariant ? nothing : failing_component,
        reason)
    matrix=sparse(I,J,V,length(map.diagonal_indices),length(map.diagonal_indices))
    matrix,report
end

function _missing_population_report(b,::Type{R},components,reason) where R<:AbstractFloat
    PopulationInvarianceReport{R}(missing,missing,missing,missing,
        population_dimension(b),length(b),components,nothing,reason)
end

struct _PopulationKernel{M,S}
    matrix::M
    scale::S
    process::Symbol
end

function _population_component(kernel,b,map,T,absolute,relative,components,index)
    if kernel.scale isa Number&&iszero(kernel.scale)
        matrix=spzeros(T,length(map.diagonal_indices),length(map.diagonal_indices))
        R=eltype(map.scales)
        report=PopulationInvarianceReport{R}(true,zero(R),zero(R),absolute,
            length(map.diagonal_indices),length(b),components,nothing,:certified)
        return (_PopulationKernel(matrix,kernel.scale,
            kernel isa HamiltonianPIKernel ? :hamiltonian : :dissipative),report)
    end
    raw=_population_raw_action(kernel,b,map,T)
    matrix,report=_extract_population_matrix(raw,b,map,absolute,relative,components;
                                             failing_component=index)
    (_PopulationKernel(matrix,kernel.scale,
        kernel isa HamiltonianPIKernel ? :hamiltonian : :dissipative),report)
end

function _aggregate_component_report(reports,b,map,absolute,relative)
    first_failure=findfirst(report->report.invariant!==true,reports)
    if first_failure!==nothing
        failed=reports[first_failure]
        return PopulationInvarianceReport{eltype(map.scales)}(failed.invariant,
            failed.maximum_leakage,failed.scale,failed.tolerance,
            length(map.diagonal_indices),length(b),length(reports),first_failure,
            failed.reason)
    end
    maximum_leakage=maximum((report.maximum_leakage for report in reports);
                            init=zero(eltype(map.scales)))
    scale=maximum((report.scale for report in reports);init=zero(eltype(map.scales)))
    tolerance=absolute+relative*scale
    reason=any(report->report.reason===:within_tolerance,reports) ?
        :within_tolerance : :certified
    PopulationInvarianceReport{eltype(map.scales)}(true,maximum_leakage,scale,tolerance,
        length(map.diagonal_indices),length(b),length(reports),nothing,reason)
end

function _compile_population_plan(plan::LiouvillianPlan;atol=nothing,rtol=nothing)
    b=plan.basis;T=plan.Ttype;R=_real_float_type(T)
    absolute,relative=_population_tolerances(R,atol,rtol)
    components=plan.kernels===nothing ? 0 : length(plan.kernels)
    plan.kernels===nothing&&return (nothing,nothing,
        _missing_population_report(b,R,components,:uncompiled_operator_schedule))
    # A caller may hand us an already fused autonomous plan without its source
    # model. Probe only the physical population columns matrix-free; model and
    # compiled-model constructors below deliberately rebuild term-resolved
    # kernels so their stricter component diagnostics remain available.
    if any(kernel->kernel isa FusedStaticPIKernel,plan.kernels)
        return _compile_population_operator(plan,b;atol,rtol)
    end
    all(kernel->kernel isa AbstractStaticPIKernel,plan.kernels)||return (nothing,nothing,
        _missing_population_report(b,R,components,:operator_time_dependence))
    coordinate_map=_population_coordinate_map(b,R)
    if plan.autonomous
        raw=_autonomous_population_raw(plan.kernels,b,coordinate_map,T)
        matrix,report=_extract_population_matrix(raw,b,coordinate_map,absolute,relative,components)
        kernels=(_PopulationKernel(matrix,one(T),:generic),)
        return kernels,coordinate_map,report
    end
    component_indices=ntuple(identity,components)
    compiled=Base.map(plan.kernels,component_indices) do kernel,index
        _population_component(kernel,b,coordinate_map,T,absolute,relative,components,index)
    end
    reports=Base.map(last,compiled)
    report=_aggregate_component_report(reports,b,coordinate_map,absolute,relative)
    kernels=Base.map(first,compiled)
    kernels,coordinate_map,report
end

function _raw_from_matrix(L::AbstractMatrix,b,map,::Type{T}) where T
    size(L)==(length(b),length(b))||throw(DimensionMismatch(
        "Liouvillian matrix dimensions do not match the PI basis"))
    selected=sparse(T.(L[:,map.diagonal_indices]))
    values=nonzeros(selected)
    for column in axes(selected,2),index in nzrange(selected,column)
        values[index]*=map.inverse_scales[column]
    end
    selected
end

function _raw_from_operator(L,b,map,::Type{T}) where T
    size(L)==(length(b),length(b))||throw(DimensionMismatch(
        "Liouvillian dimensions do not match the PI basis"))
    input=zeros(T,length(b));output=similar(input)
    work=L isa LiouvillianPlan ? LiouvillianWorkspace(L) : nothing
    I=Int[];J=Int[];V=T[]
    for column in eachindex(map.diagonal_indices)
        fill!(input,zero(T));input[map.diagonal_indices[column]]=map.inverse_scales[column]
        if work===nothing
            mul!(output,L,input)
        else
            apply!(output,L,input,work)
        end
        for row in eachindex(output)
            iszero(output[row])&&continue
            push!(I,row);push!(J,column);push!(V,output[row])
        end
    end
    sparse(I,J,V,length(b),length(map.diagonal_indices))
end

function _compile_population_operator(L,b;atol=nothing,rtol=nothing)
    isautonomous(L)||return (nothing,nothing,
        _missing_population_report(b,_real_float_type(eltype(L)),1,:time_dependent_operator))
    T=_complex_float_type(eltype(L));R=_real_float_type(T)
    map=_population_coordinate_map(b,R)
    absolute,relative=_population_tolerances(R,atol,rtol)
    raw=L isa AbstractMatrix ? _raw_from_matrix(L,b,map,T) : _raw_from_operator(L,b,map,T)
    matrix,report=_extract_population_matrix(raw,b,map,absolute,relative,1)
    (_PopulationKernel(matrix,one(T),:generic),),map,report
end

"""
    PopulationPlan(source[, basis]; atol=nothing, rtol=nothing)

Compile the restriction of a PI Liouvillian to physical Schur-diagonal
populations.  Construction succeeds only after certifying that the diagonal
subspace is invariant.  Fixed operators with scalar time-dependent rates are
compiled term by term and remain time dependent without rebuilding their
sparse population matrices.  Operator-valued schedules must first be frozen
at an explicit time. The default `atol=nothing, rtol=nothing` is structurally
strict (both effective tolerances are zero), so a weak but nonzero
coherence-generating term is never discarded. Supplying a nonzero tolerance
is an explicit approximate-projection request and is recorded as
`invariance.reason == :within_tolerance` whenever it accepts nonzero leakage.

Only the population generator and basis are retained. Full-PI coordinate
lookup data are compile-local and scale with `population_dimension(basis)`,
not `length(basis)`.

The current plan compiler forms the standalone coordinate conversion
`sqrt(f^nu)` in its working real type; model/compiled-plan sources also inherit
the parent Liouvillian trace functional. If a required standalone scale is
outside that type's finite range, construction raises with wider-type
guidance. This is a plan-construction limit, not a limitation of
[`diagonal_populations`](@ref) or [`state_from_populations`](@ref), whose
prepared scaled products can remain finite without representing the factor by
itself.
"""
struct PopulationPlan{B,K,T,R}
    basis::B
    kernels::K
    Ttype::Type{T}
    autonomous::Bool
    invariance::R
end

function _population_plan(plan::LiouvillianPlan;atol=nothing,rtol=nothing)
    kernels,map,report=_compile_population_plan(plan;atol=atol,rtol=rtol)
    report.invariant===true||throw(ArgumentError(report.invariant===missing ?
        "Schur-diagonal invariance could not be certified ($(report.reason)); freeze operator-valued time dependence at an explicit time" :
        "Liouvillian does not preserve Schur-diagonal states: maximum leakage=$(report.maximum_leakage), tolerance=$(report.tolerance)"))
    PopulationPlan(plan.basis,kernels,plan.Ttype,plan.autonomous,report)
end

PopulationPlan(model::PIModel;kwargs...)=
    _population_plan(_term_resolved_liouvillian_plan(model);kwargs...)
PopulationPlan(compiled::CompiledPIModel;kwargs...)=_population_plan(
    compiled.plan.kernels!==nothing&&
    any(kernel->kernel isa FusedStaticPIKernel,compiled.plan.kernels) ?
        _term_resolved_liouvillian_plan(compiled.model) : compiled.plan;kwargs...)
PopulationPlan(plan::LiouvillianPlan;kwargs...)=_population_plan(plan;kwargs...)
function PopulationPlan(L::MatrixFreeLiouvillian,b::PIBasis=L.plan===nothing ?
                        throw(ArgumentError("a custom matrix-free Liouvillian requires an explicit PI basis")) : L.plan.basis;
                        atol=nothing,rtol=nothing)
    if L.plan!==nothing
        b===L.plan.basis||throw(ArgumentError(
            "the supplied basis is not the matrix-free Liouvillian's exact PI basis"))
        return _population_plan(L.plan;atol=atol,rtol=rtol)
    end
    kernels,map,report=_compile_population_operator(L,b;atol=atol,rtol=rtol)
    report.invariant===true||throw(ArgumentError(report.invariant===missing ?
        "Schur-diagonal invariance could not be certified ($(report.reason))" :
        "Liouvillian does not preserve Schur-diagonal states: maximum leakage=$(report.maximum_leakage), tolerance=$(report.tolerance)"))
    PopulationPlan(b,kernels,_complex_float_type(eltype(L)),true,report)
end
function PopulationPlan(L::AbstractMatrix,b::PIBasis;atol=nothing,rtol=nothing)
    kernels,map,report=_compile_population_operator(L,b;atol=atol,rtol=rtol)
    report.invariant===true||throw(ArgumentError(
        "Liouvillian does not preserve Schur-diagonal states: maximum leakage=$(report.maximum_leakage), tolerance=$(report.tolerance)"))
    PopulationPlan(b,kernels,_complex_float_type(eltype(L)),true,report)
end

size(plan::PopulationPlan)=(population_dimension(plan.basis),population_dimension(plan.basis))
size(plan::PopulationPlan,index::Integer)=index in (1,2) ? population_dimension(plan.basis) : 1
eltype(plan::PopulationPlan)=plan.Ttype
isautonomous(plan::PopulationPlan)=plan.autonomous
function show(io::IO,plan::PopulationPlan)
    print(io,"PopulationPlan(N=$(plan.basis.N), d=$(plan.basis.d), " *
             "dimension=$(size(plan,1)), components=$(length(plan.kernels)), " *
             "autonomous=$(isautonomous(plan)))")
end

"""
    population_invariance(source[, basis]; atol=nothing, rtol=nothing)

Return a [`PopulationInvarianceReport`](@ref) without constructing a usable
plan. `invariant == missing` means that a global certificate is unavailable,
most commonly because an operator-valued schedule has not been frozen.
Defaults are structurally strict; nonzero `atol` or `rtol` explicitly permit
an approximate projection and are reflected in the report's `reason`.
"""
function population_invariance(model::PIModel;kwargs...)
    _,_,report=_compile_population_plan(
        _term_resolved_liouvillian_plan(model);kwargs...);report
end
function population_invariance(compiled::CompiledPIModel;kwargs...)
    plan=compiled.plan.kernels!==nothing&&
        any(kernel->kernel isa FusedStaticPIKernel,compiled.plan.kernels) ?
            _term_resolved_liouvillian_plan(compiled.model) : compiled.plan
    _,_,report=_compile_population_plan(plan;kwargs...);report
end
function population_invariance(plan::LiouvillianPlan;kwargs...)
    _,_,report=_compile_population_plan(plan;kwargs...);report
end
function population_invariance(plan::PopulationPlan;kwargs...)
    isempty(kwargs)||throw(ArgumentError(
        "a PopulationPlan already stores its invariance certificate; rebuild the plan to use different tolerances"))
    plan.invariance
end
function population_invariance(L::MatrixFreeLiouvillian,b::PIBasis=L.plan===nothing ?
                               throw(ArgumentError("a custom matrix-free Liouvillian requires an explicit PI basis")) : L.plan.basis;
                               kwargs...)
    if L.plan!==nothing
        b===L.plan.basis||throw(ArgumentError(
            "the supplied basis is not the matrix-free Liouvillian's exact PI basis"))
        _,_,report=_compile_population_plan(L.plan;kwargs...);return report
    end
    _,_,report=_compile_population_operator(L,b;kwargs...);report
end
function population_invariance(L::AbstractMatrix,b::PIBasis;kwargs...)
    _,_,report=_compile_population_operator(L,b;kwargs...);report
end

"""
Reusable scratch for population-generator application and fixed-step RK4
evolution. New workspaces retain three full integration arrays (`stage`, `k1`,
and `k2`) plus the independent kernel-application buffer. The legacy `k3` and
`k4` fields are empty compatibility placeholders; full-sized legacy values are
also accepted.
"""
struct PopulationWorkspace{V}
    stage::V
    k1::V
    k2::V
    k3::V
    k4::V
    kernel::V
end

function PopulationWorkspace(plan::PopulationPlan)
    T=eltype(plan);n=size(plan,1)
    PopulationWorkspace(zeros(T,n),zeros(T,n),zeros(T,n),T[],T[],zeros(T,n))
end
function PopulationWorkspace(plan::PopulationPlan,source::AbstractVector)
    length(source)==size(plan,1)||throw(DimensionMismatch("population vector has the wrong length"))
    T=promote_type(eltype(plan),eltype(source));n=size(plan,1)
    PopulationWorkspace(zeros(T,n),zeros(T,n),zeros(T,n),T[],T[],zeros(T,n))
end

function _check_population_workspace(work::PopulationWorkspace,plan,source,destination)
    n=size(plan,1)
    all(vector->length(vector)==n,(work.stage,work.k1,work.k2,work.kernel))&&
        all(vector->length(vector) in (0,n),(work.k3,work.k4))||
        throw(DimensionMismatch("population workspace has the wrong dimension"))
    required=promote_type(eltype(plan),eltype(source))
    promote_type(eltype(work.kernel),required)===eltype(work.kernel)||throw(ArgumentError(
        "population workspace scalar type $(eltype(work.kernel)) cannot represent the plan and source types"))
    promote_type(eltype(destination),required)===eltype(destination)||throw(ArgumentError(
        "population destination scalar type $(eltype(destination)) cannot represent the plan and source types"))
    nothing
end

@inline _apply_population_kernels!(y,tmp,x,::Tuple{},t,p,T)=y
@inline function _apply_population_kernels!(y,tmp,x,kernels::Tuple{K,Vararg{Any}},t,p,T) where K
    kernel=first(kernels)
    scale = kernel.process===:dissipative ? begin
        value=_evaluated_dissipative_rate(kernel.scale,t,p)
        promote_type(T,typeof(value))===T||throw(ArgumentError(
            "evaluated population rate type $(typeof(value)) is wider than workspace type $T"))
        converted=convert(T,value)
        isfinite(converted)||throw(ArgumentError(
            "evaluated population rate must be finite"))
        converted
    end : begin
        value=value_at(kernel.scale,t,p)
        value isa Number||throw(ArgumentError("population rate must evaluate to a number"))
        promote_type(T,typeof(value))===T||throw(ArgumentError(
            "evaluated population rate type $(typeof(value)) is wider than workspace type $T"))
        converted=convert(T,value)
        isfinite(converted)||throw(ArgumentError(
            "evaluated population rate must be finite"))
        converted
    end
    iszero(scale)&&return _apply_population_kernels!(
        y,tmp,x,Base.tail(kernels),t,p,T)
    mul!(tmp,kernel.matrix,x)
    @inbounds for index in eachindex(y);y[index]+=scale*tmp[index];end
    _apply_population_kernels!(y,tmp,x,Base.tail(kernels),t,p,T)
end

"""Apply a population plan with caller-owned scratch."""
function apply!(destination::AbstractVector,plan::PopulationPlan,source::AbstractVector,
                time,parameters,work::PopulationWorkspace)
    length(source)==size(plan,1)&&length(destination)==size(plan,1)||
        throw(DimensionMismatch("population vector has the wrong length"))
    Base.mightalias(destination,source)&&throw(ArgumentError(
        "population apply! destination must not alias its source"))
    _check_population_workspace(work,plan,source,destination)
    fill!(destination,zero(eltype(destination)))
    _apply_population_kernels!(destination,work.kernel,source,plan.kernels,
                               time,parameters,eltype(work.kernel))
end

function apply!(destination::AbstractVector,plan::PopulationPlan,source::AbstractVector,
                work::PopulationWorkspace)
    isautonomous(plan)||throw(ArgumentError(
        "autonomous population application requires a fixed-rate plan; use the explicit-time apply! method"))
    apply!(destination,plan,source,0.0,nothing,work)
end
apply!(destination::AbstractVector,plan::PopulationPlan,source::AbstractVector,time,parameters)=
    apply!(destination,plan,source,time,parameters,PopulationWorkspace(plan,source))

function mul!(destination::AbstractVector,plan::PopulationPlan,source::AbstractVector)
    isautonomous(plan)||throw(ArgumentError(
        "mul! requires an autonomous population plan; use apply! with explicit time and parameters"))
    apply!(destination,plan,source,0.0,nothing,PopulationWorkspace(plan,source))
end
function *(plan::PopulationPlan,source::AbstractVector)
    destination=zeros(promote_type(eltype(plan),eltype(source)),size(plan,1))
    mul!(destination,plan,source)
end

function _population_sparse_generator(plan,time,parameters)
    T=eltype(plan);I=Int[];J=Int[];V=T[]
    for kernel in plan.kernels
        scale = kernel.process===:dissipative ?
            _evaluated_dissipative_rate(kernel.scale,time,parameters) :
            value_at(kernel.scale,time,parameters)
        scale isa Number||throw(ArgumentError("population rate must evaluate to a number"))
        promote_type(T,typeof(scale))===T||throw(ArgumentError(
            "evaluated population rate type $(typeof(scale)) is wider than population plan type $T"))
        converted=convert(T,scale)
        isfinite(converted)||throw(ArgumentError(
            "evaluated population rate must be finite"))
        iszero(converted)&&continue
        _append_scaled_population_sparse!(I,J,V,kernel.matrix,converted,T)
    end
    sparse(I,J,V,size(plan,1),size(plan,2))
end

"""
    population_generator(plan; representation=:matrixfree, time=nothing,
                         parameters=nothing)

Return the reusable `PopulationPlan` itself for `:matrixfree`, or assemble its
sparse reduced generator.  A driven plan requires an explicit `time` for
sparse assembly.
"""
function population_generator(plan::PopulationPlan;representation=:matrixfree,
                              time=nothing,parameters=nothing)
    representation in (:matrixfree,:sparse)||throw(ArgumentError(
        "representation must be :matrixfree or :sparse"))
    representation===:matrixfree&&return plan
    if !isautonomous(plan)&&time===nothing
        throw(ArgumentError("a driven population generator requires an explicit time"))
    end
    _population_sparse_generator(plan,time===nothing ? 0.0 : time,parameters)
end
function population_generator(model::PIModel;representation=:matrixfree,time=nothing,
                              parameters=nothing,atol=nothing,rtol=nothing)
    population_generator(PopulationPlan(model;atol=atol,rtol=rtol);
                         representation=representation,time=time,parameters=parameters)
end
function population_generator(compiled::CompiledPIModel;representation=:matrixfree,
                              time=nothing,parameters=nothing,atol=nothing,rtol=nothing)
    population_generator(PopulationPlan(compiled;atol=atol,rtol=rtol);
                         representation=representation,time=time,parameters=parameters)
end

function _schur_diagonal_metrics(operator::AbstractPIOperator)
    R=_real_float_type(eltype(operator.data));error=zero(R);scale=zero(R)
    for partition in operator.basis.sectors
        # Diagonal populations live in the bounded coordinates
        # sqrt(f)C=f*rho.  Measuring diagonality there avoids requiring the
        # standalone sqrt(f), and matches the values returned below.
        block=coefficient_block(operator,partition)
        direct_weight=try
            _schur_multiplicity_scale(R,partition)
        catch caught
            caught isa ArgumentError||rethrow()
            nothing
        end
        if direct_weight!==nothing
            for column in axes(block,2),row in axes(block,1)
                source=block[row,column]
                entry=source*direct_weight
                if !_ordinary_scaled_value_safe(entry,source)
                    entry=_checked_mul_sqrt_exact_ratio(
                        source,symmetric_group_dimension(partition),one(BigInt);
                        context="Schur-diagonal weighted metric in sector $partition")
                end
                value=R(abs(entry));scale=max(scale,value)
                row==column|| (error=max(error,value))
            end
        else
            multiplicity=symmetric_group_dimension(partition)
            weight=_prepare_exact_scale(R,multiplicity,one(BigInt),Val(true);
                context="Schur-diagonal weighted metric in sector $partition")
            for column in axes(block,2),row in axes(block,1)
                entry=_apply_prepared_exact_scale(block[row,column],weight;
                    context="Schur-diagonal weighted metric in sector $partition")
                value=R(abs(entry));scale=max(scale,value)
                row==column|| (error=max(error,value))
            end
        end
    end
    error,scale
end

"""
    diagonal_populations(A; check=true, atol=nothing, rtol=nothing)
    diagonal_populations!(destination, A; ...)

Extract multiplicity-weighted physical diagonal entries from every Schur
sector in basis order.  Their sum equals `trace(A)`.  By default appreciable
off-diagonal entries raise rather than being discarded; `check=false` is an
explicit projection request. `atol` and `rtol` measure off-diagonal entries in
these same bounded multiplicity-weighted coordinates.
"""
function diagonal_populations!(destination::AbstractVector,
                               operator::AbstractPIOperator;check::Bool=true,
                               atol=nothing,rtol=nothing)
    b=operator.basis;length(destination)==population_dimension(b)||
        throw(DimensionMismatch("population destination has the wrong length"))
    promote_type(eltype(destination),eltype(operator.data))===eltype(destination)||
        throw(ArgumentError("population destination scalar type $(eltype(destination)) " *
                            "cannot represent operator scalar type $(eltype(operator.data))"))
    R=_real_float_type(eltype(operator.data));absolute,relative=_population_tolerances(R,atol,rtol)
    if check
        error,scale=_schur_diagonal_metrics(operator)
        error<=absolute+relative*scale||throw(ArgumentError(
            "operator is not Schur diagonal: off-diagonal error=$error, tolerance=$(absolute+relative*scale)"))
    end
    output_index=1
    for partition in b.sectors
        block=coefficient_block(operator,partition)
        direct_weight=try
            _schur_multiplicity_scale(R,partition)
        catch caught
            caught isa ArgumentError||rethrow()
            nothing
        end
        if direct_weight!==nothing
            for index in axes(block,1)
                source=block[index,index]
                entry=source*direct_weight
                if !_ordinary_scaled_value_safe(entry,source)
                    entry=_checked_mul_sqrt_exact_ratio(
                        source,symmetric_group_dimension(partition),one(BigInt);
                        context="Schur-diagonal population in sector $partition")
                end
                destination[output_index]=entry
                output_index+=1
            end
        else
            multiplicity=symmetric_group_dimension(partition)
            weight=_prepare_exact_scale(R,multiplicity,one(BigInt),Val(true);
                context="Schur-diagonal population in sector $partition")
            for index in axes(block,1)
                destination[output_index]=_apply_prepared_exact_scale(
                    block[index,index],weight;
                    context="Schur-diagonal population in sector $partition")
                output_index+=1
            end
        end
    end
    destination
end
"""
    diagonal_populations(A; check=true, atol=nothing, rtol=nothing)

Allocate and return the multiplicity-weighted physical Schur-diagonal entries
of `A`. See [`diagonal_populations!`](@ref) for validation and projection
semantics.
"""
function diagonal_populations(operator::AbstractPIOperator;kwargs...)
    destination=zeros(eltype(operator.data),population_dimension(operator.basis))
    diagonal_populations!(destination,operator;kwargs...)
end

"""
    state_from_populations(basis, populations; validate=false, validation_kwargs...)

Construct a Schur-diagonal `PIState` from multiplicity-weighted physical
populations.  No normalization, positivity repair, or clipping is performed.
Set `validate=true` to call [`validate_state`](@ref) on the result.
"""
function state_from_populations(b::PIBasis,populations::AbstractVector;
                                validate::Bool=false,validation_kwargs...)
    length(populations)==population_dimension(b)||throw(DimensionMismatch(
        "population vector has the wrong length"))
    !validate&&!isempty(validation_kwargs)&&throw(ArgumentError(
        "validation keywords require validate=true"))
    R=_real_float_type(eltype(populations));rho=PIState(b;T=R)
    input_index=1
    for partition in b.sectors
        block=coefficient_block(rho,partition)
        direct_inverse=try
            # Match the allocation-light coordinate-map convention on ordinary
            # sectors: form the representable forward scale once and invert it.
            # If either the scale or a scaled entry is exceptional, the exact
            # fused branch below remains authoritative.
            inv(_schur_multiplicity_scale(R,partition))
        catch caught
            caught isa ArgumentError||rethrow()
            nothing
        end
        if direct_inverse!==nothing
            for index in axes(block,1)
                source=populations[input_index]
                entry=source*direct_inverse
                if !_ordinary_scaled_value_safe(entry,source)
                    entry=_checked_mul_sqrt_exact_ratio(
                        source,one(BigInt),symmetric_group_dimension(partition);
                        context="stored population coefficient in sector $partition")
                end
                block[index,index]=entry
                input_index+=1
            end
        else
            multiplicity=symmetric_group_dimension(partition)
            inverse_weight=_prepare_exact_scale(
                R,one(BigInt),multiplicity,Val(true);
                context="stored population coefficient in sector $partition")
            for index in axes(block,1)
                block[index,index]=_apply_prepared_exact_scale(
                    populations[input_index],inverse_weight;
                    context="stored population coefficient in sector $partition")
                input_index+=1
            end
        end
    end
    validate ? validate_state(rho;validation_kwargs...) : rho
end

function _check_population_evolution_workspace(work,plan,source,destination)
    _check_population_workspace(work,plan,source,destination)
    active=(work.stage,work.k1,work.k2,work.kernel)
    for i in eachindex(active)
        Base.mightalias(active[i],destination)&&throw(ArgumentError(
            "population evolution destination must not alias workspace scratch"))
        for j in 1:i-1
            Base.mightalias(active[i],active[j])&&throw(ArgumentError(
                "population workspace scratch arrays must not alias"))
        end
    end
    nothing
end

"""
    evolve_populations!(destination, plan, source, tspan; steps=256,
                        parameters=nothing, workspace=nothing)

Propagate a population vector with preallocated three-scratch fourth-order
Runge--Kutta evolution. `destination` may alias `source`, but it must not alias
workspace scratch; one workspace may be reused sequentially but not
concurrently.
"""
function evolve_populations!(destination::AbstractVector,plan::PopulationPlan,
                             source::AbstractVector,tspan;steps::Integer=256,
                             parameters=nothing,workspace=nothing)
    length(source)==size(plan,1)&&length(destination)==size(plan,1)||
        throw(DimensionMismatch("population vector has the wrong length"))
    steps>0||throw(ArgumentError("steps must be positive"))
    t0,t1=tspan;t1>=t0||throw(ArgumentError("tspan must be ordered"))
    R=_real_float_type(promote_type(eltype(plan),eltype(source),eltype(destination)))
    for (name,value) in (("initial time",t0),("final time",t1))
        if value isa AbstractFloat
            promote_type(R,typeof(value))===R||throw(ArgumentError(
                "$name type $(typeof(value)) is wider than the in-place population state precision $R"))
        end
        converted=try
            convert(R,value)
        catch
            throw(ArgumentError("$name is not representable in $R"))
        end
        isfinite(converted)&&converted==value||throw(ArgumentError(
            "$name is not represented exactly and finitely in $R"))
    end
    work=workspace===nothing ? PopulationWorkspace(plan,source) : workspace
    destination===source||!Base.mightalias(destination,source)||throw(
        ArgumentError("population evolution permits exact in-place use but not partially overlapping source and destination storage"))
    _check_population_evolution_workspace(work,plan,source,destination)
    destination===source||copyto!(destination,source)
    time=convert(R,t0);step=(convert(R,t1)-time)/R(steps)
    for _ in 1:steps
        apply!(work.k1,plan,destination,time,parameters,work)
        copyto!(work.k2,work.k1)
        @. work.stage=destination+(step/2)*work.k1
        apply!(work.k1,plan,work.stage,time+step/2,parameters,work)
        @. work.k2=work.k2+2work.k1
        @. work.stage=destination+(step/2)*work.k1
        apply!(work.k1,plan,work.stage,time+step/2,parameters,work)
        @. work.k2=work.k2+2work.k1
        @. work.stage=destination+step*work.k1
        apply!(work.k1,plan,work.stage,time+step,parameters,work)
        @. work.k2=work.k2+work.k1
        @. destination=destination+(step/6)*work.k2
        time+=step
    end
    destination
end

"""Saved population vectors and their certified population plan."""
struct PopulationSolution{T,V,P}
    times::Vector{T}
    populations::Vector{V}
    plan::P
end
Base.length(solution::PopulationSolution)=length(solution.populations)
Base.getindex(solution::PopulationSolution,index::Integer)=solution.populations[index]
Base.firstindex(solution::PopulationSolution)=firstindex(solution.populations)
Base.lastindex(solution::PopulationSolution)=lastindex(solution.populations)
Base.iterate(solution::PopulationSolution,args...)=iterate(solution.populations,args...)
state(solution::PopulationSolution,index::Integer)=
    state_from_populations(solution.plan.basis,solution.populations[index])
state_at(solution::PopulationSolution,time::Real)=
    state(solution,_saved_time_index(solution.times,time))
state(solution::PopulationSolution,time::Real)=state_at(solution,time)
function show(io::IO,solution::PopulationSolution)
    print(io,"PopulationSolution($(length(solution)) vectors, dimension=$(size(solution.plan,1)), " *
             "t=$(first(solution.times))…$(last(solution.times)))")
end

"""
    solve_populations(plan, initial, tspan; saveat=nothing,
                      steps_per_interval=64, parameters=nothing)

Evolve and save only certified Schur-diagonal physical populations.  Passing
a `PIState` extracts its populations after checking that it is diagonal.
"""
function solve_populations(plan::PopulationPlan,initial::AbstractVector,tspan;
                           saveat=nothing,steps_per_interval::Integer=64,
                           parameters=nothing)
    steps_per_interval>0||throw(ArgumentError("steps_per_interval must be positive"))
    times=_saved_times(tspan,saveat)
    length(initial)==size(plan,1)||throw(DimensionMismatch("population vector has the wrong length"))
    time_type=_real_float_type(eltype(times))
    T=promote_type(eltype(plan),eltype(initial),Complex{time_type})
    current=T.(initial)
    work=PopulationWorkspace(plan,current);saved=typeof(current)[copy(current)]
    for index in 2:length(times)
        times[index]==times[index-1]||evolve_populations!(current,plan,current,
            (times[index-1],times[index]);steps=steps_per_interval,
            parameters=parameters,workspace=work)
        push!(saved,copy(current))
    end
    PopulationSolution(times,saved,plan)
end
function solve_populations(plan::PopulationPlan,initial::PIState,tspan;kwargs...)
    initial.basis===plan.basis||throw(ArgumentError("initial state uses a different PI basis"))
    solve_populations(plan,diagonal_populations(initial),tspan;kwargs...)
end
solve_populations(model::PIModel,initial,tspan;plan_kwargs=NamedTuple(),kwargs...)=
    solve_populations(PopulationPlan(model;plan_kwargs...),initial,tspan;kwargs...)
solve_populations(compiled::CompiledPIModel,initial,tspan;plan_kwargs=NamedTuple(),kwargs...)=
    solve_populations(PopulationPlan(compiled;plan_kwargs...),initial,tspan;kwargs...)

"""
    stationary_populations(plan; method=:auto, kwargs...)

Solve the trace-one stationary problem in the reduced population space.  The
returned vector uses physical populations and therefore has trace functional
`ones(size(plan,1))`.
"""
function stationary_populations(plan::PopulationPlan;method=:auto,kwargs...)
    isautonomous(plan)||throw(ArgumentError(
        "stationary populations require an autonomous plan; freeze the source first"))
    matrix=population_generator(plan;representation=:sparse)
    trace_vector=ones(eltype(plan),size(plan,1))
    steady_state(matrix;trace_vector=trace_vector,method=method,kwargs...)
end
stationary_populations(model::PIModel;plan_kwargs=NamedTuple(),kwargs...)=
    stationary_populations(PopulationPlan(model;plan_kwargs...);kwargs...)
stationary_populations(compiled::CompiledPIModel;plan_kwargs=NamedTuple(),kwargs...)=
    stationary_populations(PopulationPlan(compiled;plan_kwargs...);kwargs...)
