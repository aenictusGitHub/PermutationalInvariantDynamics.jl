# Generalized U(d)-coherent-state Husimi data for qudit PI states.  A point is
# represented by a local unitary U (or a Hermitian generator H with
# U=exp(-im*H)).  In every Schur sector nu, the same local transformation is
# lifted through the physical collective block and applied to the extremal GT
# vector.  Schur orthogonality then gives
#
#   integral dU dim(U_nu) <nu,U|rho_bar_nu|nu,U> = tr(rho_bar_nu),
#
# for normalized Haar measure.  This convention treats every retained irrep
# independently and never embeds the state in the d^N Hilbert space.

"""
    QuditHusimiPlan(basis, points; representation=:unitary,
                    sectors=:all, T=nothing, atol=nothing, rtol=nothing)

Prepare generalized coherent vectors for a qudit Husimi-Q transform.  Each
point is either a `d`-by-`d` unitary (`representation=:unitary`) or a Hermitian
generator `H` (`representation=:generator`) representing the local unitary
`exp(-im*H)`.  A single matrix or an iterable of matrices is accepted.

For a sector `nu`, the fiducial vector is the first vector in the package's
ascending GT order, whose content is `reverse(nu.parts)`.  The lifted local
unitary acts through the physical collective one-body generator.  The plan is
read-only, tied to the exact `PIBasis`, and may be reused for many states.

Unitary input currently uses the LAPACK Schur factorization and therefore
supports `Float32` and `Float64` real precision.  Use Hermitian generators for
other scalar types.  An explicit `T` may widen floating-point coordinates but
may not silently narrow them; integer and exact coordinates remain accepted.
No `d^N` object is constructed.
"""
struct QuditHusimiPlan{B,P,V,R<:AbstractFloat}
    basis::B
    sectors::Vector{P}
    sector_indices::Vector{Int}
    coherent_vectors::V
    point_count::Int
    representation::Symbol
    real_type::Type{R}
end

function show(io::IO,plan::QuditHusimiPlan)
    print(io,"QuditHusimiPlan(d=",plan.basis.d,
          ", points=",plan.point_count,
          ", sectors=",length(plan.sectors),
          ", representation=",plan.representation,")")
end

"""
    QuditHusimiData

Generalized qudit Husimi-Q values with respect to normalized Haar measure on
the local unitary group.  `values[point]` is the sum over selected Schur
sectors.  When `resolved=true`, `sector_values[sector, point]` retains each
sector density separately.  Every sector density integrates to its entry in
`populations`; consequently the aggregate integrates to the selected physical
population.

The unitary parametrization is generally redundant because a highest-weight
state has a nontrivial stabilizer.  This does not change the normalized Haar
identity used by the transform.
"""
struct QuditHusimiData{R<:AbstractFloat,P,S,M}
    values::Vector{R}
    sectors::Vector{P}
    multiplicities::Vector{BigInt}
    irrep_dimensions::Vector{Int}
    populations::Vector{R}
    sector_values::S
    normalization::Symbol
    metadata::M
end

function show(io::IO,data::QuditHusimiData)
    print(io,"QuditHusimiData(points=",length(data.values),
          ", sectors=",length(data.sectors),
          ", resolved=",data.sector_values!==nothing,")")
end

function _qudit_phase_space_points(points,d::Int)
    collection=points isa AbstractMatrix ? (points,) : try
        Tuple(points)
    catch error
        throw(ArgumentError(
            "phase-space points must be a matrix or an iterable of matrices: $(sprint(showerror,error))"))
    end
    isempty(collection)&&throw(ArgumentError(
        "at least one qudit phase-space point is required"))
    for point in collection
        point isa AbstractMatrix||throw(ArgumentError(
            "every qudit phase-space point must be a matrix"))
        size(point)==(d,d)||throw(DimensionMismatch(
            "every qudit phase-space point must be $d-by-$d"))
        all(isfinite,point)||throw(ArgumentError(
            "qudit phase-space points must contain only finite values"))
    end
    collection
end

function _qudit_phase_space_real_type(points,T)
    if T!==nothing
        T isa Type&&T<:AbstractFloat&&isconcretetype(T)||throw(ArgumentError(
            "T must be a concrete AbstractFloat type"))
        for point in points,value in point
            coordinate_type=value isa AbstractFloat ? typeof(value) :
                value isa Complex&&real(value) isa AbstractFloat ?
                    typeof(real(value)) : nothing
            coordinate_type===nothing&&continue
            promote_type(T,coordinate_type)===T||throw(ArgumentError(
                "phase-space coordinate scalar type $coordinate_type would narrow in $T precision; "*
                "omit T or use a wider type"))
        end
        R=T
    else
        R=nothing
        for point in points
            candidate=_real_float_type(eltype(point))
            R=R===nothing ? candidate : promote_type(R,candidate)
        end
    end
    R<:AbstractFloat&&isconcretetype(R)||throw(ArgumentError(
        "phase-space point precision must promote to a concrete AbstractFloat type"))
    R
end

function _qudit_phase_space_real_input(::Type{R},value,label) where R<:AbstractFloat
    value isa Real||throw(ArgumentError("$label must be real"))
    converted=try
        R(value)
    catch error
        throw(ArgumentError("$label cannot be represented in $R: $(sprint(showerror,error))"))
    end
    isfinite(converted)||throw(ArgumentError("$label must be finite"))
    value isa AbstractFloat&&promote_type(R,typeof(value))!==R&&throw(ArgumentError(
        "$label scalar type $(typeof(value)) would narrow in $R precision"))
    converted
end

function _qudit_phase_space_sector_indices(basis::PIBasis,sectors)
    indices=Int[]
    if sectors===:all
        append!(indices,eachindex(basis.sectors))
    elseif sectors isa Partition ||
           (sectors isa Tuple&&length(sectors)==basis.d&&
            all(x->x isa Integer,sectors)) ||
           (sectors isa AbstractVector&&length(sectors)==basis.d&&
            all(x->x isa Integer,sectors))
        push!(indices,basis.index[_schur_block_partition(basis,sectors)])
    else
        sectors isa Symbol&&throw(ArgumentError(
            "sectors must be :all, one retained partition, or an iterable of retained partitions"))
        for label in sectors
            partition=_schur_block_partition(basis,label)
            push!(indices,basis.index[partition])
        end
    end
    isempty(indices)&&throw(ArgumentError(
        "at least one Schur sector must be selected"))
    length(unique(indices))==length(indices)||throw(ArgumentError(
        "duplicate Schur sector in qudit phase-space selection"))
    sort!(indices)
end

function _qudit_phase_space_tolerances(::Type{R},atol,rtol) where R
    default=sqrt(eps(R))
    a=atol===nothing ? zero(R) : _qudit_phase_space_real_input(R,atol,"atol")
    r=rtol===nothing ? default : _qudit_phase_space_real_input(R,rtol,"rtol")
    a>=zero(R)||throw(ArgumentError("atol must be nonnegative"))
    r>=zero(R)||throw(ArgumentError("rtol must be nonnegative"))
    a,r
end

function _validate_qudit_husimi_state(rho,atol,rtol)
    if atol===nothing&&rtol===nothing
        validate_state(rho)
    elseif atol===nothing
        validate_state(rho;atol=_analysis_atol(rho),rtol=rtol)
    elseif rtol===nothing
        validate_state(rho;atol=atol,rtol=_state_rtol(rho))
    else
        validate_state(rho;atol,rtol)
    end
end

function _unitary_phase_space_generator(point,::Type{R},atol,rtol) where R
    R in (Float32,Float64)||throw(ArgumentError(
        "unitary qudit phase-space points require Float32 or Float64; "*
        "use representation=:generator for $R precision"))
    C=Complex{R};U=Matrix{C}(point);d=size(U,1)
    identity_matrix=Matrix{C}(I,d,d)
    residual=opnorm(adjoint(U)*U-identity_matrix,Inf)
    scale=max(one(R),opnorm(U,Inf)^2)
    residual<=atol+rtol*scale||throw(ArgumentError(
        "qudit phase-space point is not unitary within tolerance: residual=$residual"))
    factor=schur(U)
    diagonal=diag(factor.T)
    # For U=exp(-im*H), the principal Hermitian generator has eigenvalues
    # -angle(lambda).  Constructing through the Schur vectors is stable for a
    # validated normal matrix and avoids a general non-Hermitian matrix log.
    phases=R[-angle(value) for value in diagonal]
    raw=Matrix(factor.Z)*Diagonal(phases)*adjoint(Matrix(factor.Z))
    Matrix{C}(Hermitian(raw))
end

function _generator_phase_space_matrix(point,::Type{R},atol,rtol) where R
    C=Complex{R};H=Matrix{C}(point)
    residual=opnorm(H-adjoint(H),Inf)
    scale=max(one(R),opnorm(H,Inf))
    residual<=atol+rtol*scale||throw(ArgumentError(
        "qudit phase-space generator is not Hermitian within tolerance: residual=$residual"))
    # The accepted skew part is conversion roundoff only.  `Hermitian` fixes a
    # single triangle for the internal matrix exponential; invalid inputs have
    # already been rejected above rather than silently repaired.
    Matrix{C}(Hermitian(H))
end

function QuditHusimiPlan(basis::PIBasis,points;representation::Symbol=:unitary,
        sectors=:all,T=nothing,atol=nothing,rtol=nothing)
    representation in (:unitary,:generator)||throw(ArgumentError(
        "representation must be :unitary or :generator"))
    collection=_qudit_phase_space_points(points,basis.d)
    R=_qudit_phase_space_real_type(collection,T)
    a,r=_qudit_phase_space_tolerances(R,atol,rtol)
    generators=map(collection) do point
        representation===:unitary ?
            _unitary_phase_space_generator(point,R,a,r) :
            _generator_phase_space_matrix(point,R,a,r)
    end
    indices=_qudit_phase_space_sector_indices(basis,sectors)
    selected=basis.sectors[indices]
    geometry=OneBodyGeometry(basis;T=R)
    C=Complex{R};npoints=length(generators)
    coherent=Vector{Matrix{C}}(undef,length(indices))
    for (selected_index,sector_index) in pairs(indices)
        partition=basis.sectors[sector_index]
        patterns=basis.patterns[sector_index]
        content(first(patterns))==reverse(partition.parts)||error(
            "internal error: the first GT vector is not the extremal fiducial vector")
        dimension=length(patterns)
        vectors=Matrix{C}(undef,dimension,npoints)
        for point_index in eachindex(generators)
            block=collective_block(basis,generators[point_index],partition;
                                   cache=geometry)
            lifted=exp(-im*Matrix{C}(block))
            vector=view(vectors,:,point_index)
            copyto!(vector,view(lifted,:,1))
            vector_norm=norm(vector)
            isfinite(vector_norm)&&vector_norm>zero(R)||error(
                "internal error: a lifted coherent vector has invalid norm")
            vector./=vector_norm
        end
        coherent[selected_index]=vectors
    end
    QuditHusimiPlan(basis,collect(selected),indices,coherent,npoints,
                    representation,R)
end

"""
    qudit_husimi_q(rho, points; representation=:unitary, sectors=:all,
                   resolved=false, plan=nothing, atol=nothing, rtol=nothing)
    qudit_husimi_q(rho, plan; resolved=false, atol=nothing, rtol=nothing)

Evaluate the generalized `U(d)` coherent-state Husimi-Q density of a PI state.
For normalized Haar measure `dU`, sector `nu` uses

```math
Q_\\nu(U)=\\dim(U_\\nu)\\langle\\nu,U|\\bar\\rho_\\nu|\\nu,U\\rangle,
\\qquad \\bar\\rho_\\nu=\\sqrt{f^\\nu}C_\\nu,
```

and therefore `integral Q_nu(U) dU` equals the physical population of that
sector.  The aggregate is the sum over selected sectors.  Values are returned
without clipping; invalid input states are rejected by `validate_state`.

Construct and reuse a [`QuditHusimiPlan`](@ref) when evaluating several states
on the same phase-space points.
"""
function qudit_husimi_q(rho::PIState,plan::QuditHusimiPlan;
        resolved::Bool=false,atol=nothing,rtol=nothing)
    rho.basis===plan.basis||throw(ArgumentError(
        "state and qudit Husimi plan use incompatible PI bases"))
    _validate_qudit_husimi_state(rho,atol,rtol)
    R=plan.real_type
    promote_type(_real_float_type(eltype(rho.data)),R)===R||throw(ArgumentError(
        "state precision cannot be represented by the qudit Husimi plan without narrowing"))
    nsectors=length(plan.sectors);npoints=plan.point_count
    values=zeros(R,npoints)
    sector_values=resolved ? Matrix{R}(undef,nsectors,npoints) : nothing
    populations=Vector{R}(undef,nsectors)
    multiplicities=Vector{BigInt}(undef,nsectors)
    dimensions=Vector{Int}(undef,nsectors)
    for selected_index in 1:nsectors
        sector_index=plan.sector_indices[selected_index]
        partition=plan.sectors[selected_index]
        block,population=_multiplicity_weighted_block(rho,sector_index,R)
        populations[selected_index]=population
        multiplicities[selected_index]=symmetric_group_dimension(partition)
        dimension=size(block,1);dimensions[selected_index]=dimension
        scale=R(dimension)
        isfinite(scale)||throw(ArgumentError(
            "sector dimension is outside the finite range of $R"))
        vectors=plan.coherent_vectors[selected_index]
        temporary=Vector{Complex{R}}(undef,dimension)
        for point_index in 1:npoints
            vector=view(vectors,:,point_index)
            mul!(temporary,block,vector)
            value=scale*real(dot(vector,temporary))
            values[point_index]+=value
            resolved&&(sector_values[selected_index,point_index]=value)
        end
    end
    metadata=(measure=:normalized_haar,
              representation=plan.representation,
              point_count=npoints,
              local_dimension=rho.basis.d,
              selected_population=sum(populations),
              orbit_parameterization=:local_unitary)
    QuditHusimiData(values,copy(plan.sectors),multiplicities,dimensions,
                    populations,sector_values,:normalized_haar,metadata)
end

function qudit_husimi_q(rho::PIState,points;representation::Symbol=:unitary,
        sectors=:all,resolved::Bool=false,plan=nothing,T=nothing,
        atol=nothing,rtol=nothing)
    prepared=if plan===nothing
        QuditHusimiPlan(rho.basis,points;representation,sectors,T,atol,rtol)
    else
        plan isa QuditHusimiPlan||throw(ArgumentError(
            "plan must be a QuditHusimiPlan"))
        points===nothing||throw(ArgumentError(
            "phase-space points are already stored in the supplied plan; call qudit_husimi_q(rho, plan)"))
        plan
    end
    qudit_husimi_q(rho,prepared;resolved,atol,rtol)
end
