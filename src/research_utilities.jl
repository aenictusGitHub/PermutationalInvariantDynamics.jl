"""
    spectral_trace(rho, f; atol=_analysis_atol(rho), rtol=_state_rtol(rho))

Evaluate `tr(f(rho))` directly from multiplicity-compressed physical Schur
spectra.  Each eigenvalue of sector `nu` is counted with its exact
symmetric-group multiplicity `f^nu`; no `d^N` spectrum is expanded.

The callback `f` is evaluated on physical density eigenvalues.  Generic
callbacks cannot be algebraically rescaled, so an eigenvalue or final weighted
term outside the state's floating type raises with wider-precision guidance.
Use specialized routines such as [`von_neumann_entropy`](@ref) and
[`renyi_entropy`](@ref) for their overflow-safe logarithmic formulas.
"""
function spectral_trace(rho::PIState,f;
        atol::Real=_analysis_atol(rho),rtol::Real=_state_rtol(rho))
    validate_state(rho;atol,rtol)
    R=_real_float_type(eltype(rho.data));result=nothing
    for sector in rho.basis.sectors
        E=_weighted_sector_eigvals(rho,sector;atol,rtol,
                                  operation="spectral trace")
        multiplicity=symmetric_group_dimension(sector)
        inverse_scale=_prepare_exact_scale(R,one(BigInt),multiplicity,
            Val(false);context="physical eigenvalue in spectral_trace")
        multiplicity_scale=_prepare_exact_scale(R,multiplicity,one(BigInt),
            Val(false);context="sector multiplicity in spectral_trace")
        for weighted_eigenvalue in E.values
            physical=_apply_prepared_exact_scale(weighted_eigenvalue,
                inverse_scale;context="physical eigenvalue in spectral_trace")
            value=f(physical)
            value isa Number||throw(ArgumentError(
                "spectral_trace callback must return a number"))
            isfinite(value)||throw(ArgumentError(
                "spectral_trace callback returned a nonfinite value"))
            weighted=_apply_prepared_exact_scale(value,multiplicity_scale;
                context="multiplicity-weighted spectral_trace contribution")
            result=result===nothing ? weighted : result+weighted
        end
    end
    result===nothing ? zero(R) : result
end

"""Metadata for one public Schur/GT population coordinate."""
struct PopulationCoordinate{P,G}
    index::Int
    pi_index::Int
    sector_index::Int
    pattern_index::Int
    sector::P
    pattern::G
    multiplicity::BigInt
end

"""
    PopulationCoordinates(basis)
    each_population_coordinate(basis)

Allocation-free iterable over the population coordinates used by
[`PopulationPlan`](@ref).  Entries follow sector order and then stored
GT-pattern order.  `index` addresses a population vector, `pi_index` is the
corresponding diagonal PI coefficient, and `multiplicity` is exact `BigInt`.
"""
struct PopulationCoordinates{B}
    basis::B
end

PopulationCoordinates(basis::PIBasis)=PopulationCoordinates{typeof(basis)}(basis)
"""Return [`PopulationCoordinates`](@ref) for `basis`."""
each_population_coordinate(basis::PIBasis)=PopulationCoordinates(basis)
Base.length(coordinates::PopulationCoordinates)=population_dimension(coordinates.basis)
Base.eltype(::Type{PopulationCoordinates{PIBasis{D,L}}}) where {D,L}=
    PopulationCoordinate{Partition{D},GTPattern{D,L}}
Base.keys(coordinates::PopulationCoordinates)=Base.OneTo(length(coordinates))
Base.eachindex(coordinates::PopulationCoordinates)=Base.OneTo(length(coordinates))

function _population_coordinate_at(b::PIBasis,index::Int)
    1<=index<=population_dimension(b)||throw(BoundsError(b,index))
    remaining=index
    for sector_index in eachindex(b.sectors)
        n=length(b.patterns[sector_index])
        if remaining<=n
            pi_index=b.offsets[sector_index]+(remaining-1)*(n+1)
            sector=b.sectors[sector_index]
            return PopulationCoordinate(index,pi_index,sector_index,remaining,
                sector,b.patterns[sector_index][remaining],
                symmetric_group_dimension(sector))
        end
        remaining-=n
    end
    error("unreachable population coordinate")
end

@inline function _population_coordinate_at(b::PIBasis,index::Int,
        sector_index::Int,pattern_index::Int)
    n=length(b.patterns[sector_index])
    pi_index=b.offsets[sector_index]+(pattern_index-1)*(n+1)
    sector=b.sectors[sector_index]
    PopulationCoordinate(index,pi_index,sector_index,pattern_index,
        sector,b.patterns[sector_index][pattern_index],
        symmetric_group_dimension(sector))
end

Base.getindex(coordinates::PopulationCoordinates,index::Integer)=
    _population_coordinate_at(coordinates.basis,Int(index))
Base.firstindex(::PopulationCoordinates)=1
Base.lastindex(coordinates::PopulationCoordinates)=length(coordinates)
function Base.iterate(coordinates::PopulationCoordinates)
    isempty(coordinates.basis.sectors)&&return nothing
    coordinate=_population_coordinate_at(coordinates.basis,1,1,1)
    coordinate,(1,2,2)
end
function Base.iterate(coordinates::PopulationCoordinates,
        state::NTuple{3,Int})
    sector_index,pattern_index,index=state
    index>length(coordinates)&&return nothing
    while pattern_index>length(coordinates.basis.patterns[sector_index])
        sector_index+=1
        sector_index>length(coordinates.basis.sectors)&&return nothing
        pattern_index=1
    end
    coordinate=_population_coordinate_at(coordinates.basis,index,
                                          sector_index,pattern_index)
    coordinate,(sector_index,pattern_index+1,index+1)
end

"""One directed off-diagonal transition of a reduced population generator."""
struct PopulationTransition{T,C}
    source::C
    destination::C
    rate::T
end

"""
    population_transitions(plan; time=nothing, parameters=nothing,
                           atol=0)

Return directed off-diagonal transitions of a certified population generator.
Rates are read from the sparse reduced generator at the requested time;
diagonal escape terms are omitted.  `source` and `destination` contain full
[`PopulationCoordinate`](@ref) metadata.  Values with magnitude at or below
the nonnegative absolute `atol` are omitted without modifying the generator.
"""
function population_transitions(plan::PopulationPlan;time=nothing,
        parameters=nothing,atol::Real=0)
    atol>=0&&isfinite(atol)||throw(ArgumentError(
        "population transition atol must be finite and nonnegative"))
    generator=population_generator(plan;representation=:sparse,time,parameters)
    coordinates=PopulationCoordinates(plan.basis)
    coordinate_cache=collect(coordinates)
    I,J,V=findnz(generator)
    T=eltype(V);C=eltype(coordinate_cache)
    transitions=PopulationTransition{T,C}[]
    tolerance=_real_float_type(T)(atol)
    for index in eachindex(V)
        I[index]==J[index]&&continue
        abs(V[index])<=tolerance&&continue
        push!(transitions,PopulationTransition(coordinate_cache[J[index]],
                                                coordinate_cache[I[index]],V[index]))
    end
    transitions
end

population_transitions(model::PIModel;kwargs...)=
    population_transitions(PopulationPlan(model);kwargs...)
population_transitions(compiled::CompiledPIModel;kwargs...)=
    population_transitions(PopulationPlan(compiled);kwargs...)
