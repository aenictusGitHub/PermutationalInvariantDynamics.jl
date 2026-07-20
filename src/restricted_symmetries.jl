"""
    SymmetryCoordinateRestriction(basis, indices; label=:explicit)

Describe an orthonormal coordinate restriction of a PI operator space.  The
retained `indices` are copied, checked to be distinct PI coefficient
coordinates, and stored in ascending order.  The restriction is therefore the
isometry whose columns are the selected standard basis vectors; it never
constructs a dense projection matrix.

This low-level constructor is also the escape hatch for a support rule derived
outside the package.  Constructing a restriction does *not* claim that a
Liouvillian preserves it.  Use [`restriction_invariance`](@ref), or construct a
[`RestrictedLiouvillian`](@ref), which performs an exhaustive leakage check.
"""
struct SymmetryCoordinateRestriction{B,I,M,L}
    basis::B
    indices::I
    mask::M
    label::L
end

function SymmetryCoordinateRestriction(basis::PIBasis, indices; label=:explicit)
    ambient=length(basis)
    retained=Int[]
    for raw in indices
        raw isa Integer && !(raw isa Bool) || throw(ArgumentError(
            "restriction indices must be integer coordinates"))
        1<=raw<=ambient || throw(BoundsError(1:ambient,raw))
        push!(retained,Int(raw))
    end
    isempty(retained)&&throw(ArgumentError(
        "a symmetry-coordinate restriction must retain at least one coordinate"))
    sort!(retained)
    for index in 2:length(retained)
        retained[index]!=retained[index-1]||throw(ArgumentError(
            "restriction indices must be distinct"))
    end
    mask=falses(ambient)
    mask[retained].=true
    SymmetryCoordinateRestriction(basis,retained,mask,label)
end

Base.length(restriction::SymmetryCoordinateRestriction)=length(restriction.indices)
Base.size(restriction::SymmetryCoordinateRestriction)=(length(restriction.basis),
                                                        length(restriction))
Base.size(restriction::SymmetryCoordinateRestriction,index::Integer)=
    index==1 ? length(restriction.basis) : index==2 ? length(restriction) : 1

"""Return a detached, ascending copy of the retained PI coordinate indices."""
retained_indices(restriction::SymmetryCoordinateRestriction)=copy(restriction.indices)

function Base.show(io::IO,restriction::SymmetryCoordinateRestriction)
    print(io,"SymmetryCoordinateRestriction($(length(restriction)) / ",
          "$(length(restriction.basis)) coordinates, label=$(restriction.label))")
end

function _restriction_tolerance(::Type{R},value,label) where R<:AbstractFloat
    value isa Real&&isfinite(value)&&value>=0||throw(ArgumentError(
        "$label must be a finite nonnegative real number"))
    converted=R(value)
    isfinite(converted)||throw(ArgumentError("$label is not finite in $R"))
    !iszero(value)&&iszero(converted)&&throw(ArgumentError(
        "$label underflows in $R; use a wider scalar type"))
    converted
end

"""
    diagonal_symmetry_restriction(basis, local_unitary;
        charge=nothing, ket_charge=nothing, bra_charge=nothing,
        atol=1e-12, rtol=1e-10, label=:diagonal_strong_symmetry)

Construct the exact PI-coordinate support associated with separate Hilbert-
space ket and bra charges of `local_unitary^tensor N`.  `local_unitary` must be
diagonal in the package's local computational basis.  With `charge=q`, both
charges are set to `q`; otherwise `bra_charge` defaults to `ket_charge`, and
the latter defaults to one.

For a GT pattern with local occupation tuple `n`, its charge is evaluated as
`prod(diag(local_unitary).^n)`.  A coefficient coordinate is retained exactly
when its row pattern has `ket_charge` and its column pattern has `bra_charge`.
Thus a density operator supported on one strong-symmetry Hilbert block uses
equal ket and bra charges.  Different charges describe an off-diagonal
operator block and generally have an identically zero physical trace.

Only the support is constructed here.  This does not infer strong symmetry
from a model; [`RestrictedLiouvillian`](@ref) exhaustively certifies that the
supplied autonomous generator leaves the support invariant.
"""
function diagonal_symmetry_restriction(basis::PIBasis,local_unitary::AbstractMatrix;
        charge=nothing,ket_charge=nothing,bra_charge=nothing,
        atol::Real=1e-12,rtol::Real=1e-10,
        label=:diagonal_strong_symmetry)
    size(local_unitary)==(basis.d,basis.d)||throw(DimensionMismatch(
        "the local symmetry must be $(basis.d) by $(basis.d)"))
    if charge!==nothing
        ket_charge===nothing&&bra_charge===nothing||throw(ArgumentError(
            "pass either charge or separate ket_charge/bra_charge values"))
        ket_charge=charge;bra_charge=charge
    else
        ket_charge===nothing&&(ket_charge=1)
        bra_charge===nothing&&(bra_charge=ket_charge)
    end
    CT=promote_type(_complex_float_type(eltype(local_unitary)),
                    _complex_float_type(typeof(ket_charge)),
                    _complex_float_type(typeof(bra_charge)))
    R=_real_float_type(CT)
    atolR=_restriction_tolerance(R,atol,"atol")
    rtolR=_restriction_tolerance(R,rtol,"rtol")
    matrix=Matrix{CT}(local_unitary)
    all(value->isfinite(real(value))&&isfinite(imag(value)),matrix)||
        throw(ArgumentError("the local symmetry must contain only finite values"))
    diagonal=diag(matrix)
    offdiagonal=norm(matrix-Diagonal(diagonal),Inf)
    matrix_scale=max(norm(matrix,Inf),one(R))
    offdiagonal<=atolR+rtolR*matrix_scale||throw(ArgumentError(
        "the strong-symmetry coordinate rule requires a diagonal local unitary"))
    for value in diagonal
        abs(abs(value)-one(R))<=atolR+rtolR||throw(ArgumentError(
            "the diagonal local symmetry must be unitary"))
    end
    qket=CT(ket_charge);qbra=CT(bra_charge)
    for (name,value) in (("ket_charge",qket),("bra_charge",qbra))
        isfinite(real(value))&&isfinite(imag(value))||throw(ArgumentError(
            "$name must be finite"))
        abs(abs(value)-one(R))<=atolR+rtolR||throw(ArgumentError(
            "$name must have unit modulus"))
    end
    charge_match(value,target)=
        abs(value-target)<=atolR+rtolR*max(abs(target),one(R))
    indices=Int[]
    for (sector,patterns) in pairs(basis.patterns)
        dimension=length(patterns)
        offset=basis.offsets[sector]-1
        phases=Vector{CT}(undef,dimension)
        for (pattern_index,pattern) in pairs(patterns)
            occupations=content(pattern)
            phase=one(CT)
            @inbounds for level in eachindex(diagonal)
                phase*=diagonal[level]^occupations[level]
            end
            phases[pattern_index]=phase
        end
        rows=findall(value->charge_match(value,qket),phases)
        columns=findall(value->charge_match(value,qbra),phases)
        for column in columns,row in rows
            push!(indices,offset+row+(column-1)*dimension)
        end
    end
    isempty(indices)&&throw(ArgumentError(
        "the requested ket/bra charge block is absent from the retained PI basis"))
    metadata=(kind=:diagonal_strong_symmetry,ket_charge=qket,
              bra_charge=qbra,user_label=label)
    SymmetryCoordinateRestriction(basis,indices;label=metadata)
end

"""
    restricted_trace_vector(restriction, [T=ComplexF64])

Return the equation-(7) physical trace functional on the retained coordinates.
The scalar type is explicit because Schur multiplicity square roots must remain
representable in the working precision.  An off-diagonal ket/bra charge block
legitimately returns the zero functional.
"""
function restricted_trace_vector(restriction::SymmetryCoordinateRestriction,
                                 ::Type{T}=ComplexF64) where T
    isconcretetype(T)&&T<:Number||throw(ArgumentError(
        "the trace scalar type must be a concrete number type"))
    R=_real_float_type(T)
    R<:AbstractFloat||throw(ArgumentError(
        "the trace scalar type must have an AbstractFloat real component"))
    full=_trace_vector(restriction.basis,T)
    full[restriction.indices]
end

"""Restrict a full PI coefficient vector into preallocated reduced storage."""
function restrict!(destination::AbstractVector,
                   restriction::SymmetryCoordinateRestriction,
                   source::AbstractVector)
    length(source)==length(restriction.basis)||throw(DimensionMismatch(
        "full PI source has the wrong length"))
    length(destination)==length(restriction)||throw(DimensionMismatch(
        "reduced destination has the wrong length"))
    input=Base.mightalias(destination,source) ? copy(source) : source
    @inbounds for reduced_index in eachindex(restriction.indices)
        destination[reduced_index]=input[restriction.indices[reduced_index]]
    end
    destination
end

"""Embed reduced coordinates into a preallocated, zero-filled full PI vector."""
function embed!(destination::AbstractVector,
                restriction::SymmetryCoordinateRestriction,
                source::AbstractVector)
    length(destination)==length(restriction.basis)||throw(DimensionMismatch(
        "full PI destination has the wrong length"))
    length(source)==length(restriction)||throw(DimensionMismatch(
        "reduced source has the wrong length"))
    input=Base.mightalias(destination,source) ? copy(source) : source
    fill!(destination,zero(eltype(destination)))
    @inbounds for reduced_index in eachindex(restriction.indices)
        destination[restriction.indices[reduced_index]]=input[reduced_index]
    end
    destination
end

"""
    RestrictedLiouvillianWorkspace(source, restriction)
    RestrictedLiouvillianWorkspace(operator)

Caller-owned scratch for restricted Liouvillian certification and application.
The source/restriction constructor owns two ambient vectors needed by
exhaustive matrix-free leakage checks and by the embedded fallback. The
prepared-operator constructor instead returns reduced Schur-block scratch for
a `:lowered` operator, so hot applications retain no ambient vectors.
"""
struct RestrictedLiouvillianWorkspace{R,V,W}
    restriction::R
    ambient_input::V
    ambient_output::V
    source_workspace::W
end

function _restricted_source_workspace(source)
    _linear_operator_workspace(source)
end

function RestrictedLiouvillianWorkspace(source,
        restriction::SymmetryCoordinateRestriction)
    size(source)==(length(restriction.basis),length(restriction.basis))||
        throw(DimensionMismatch(
        "the source Liouvillian and restriction have incompatible dimensions"))
    T=_complex_float_type(eltype(source))
    vector=zeros(T,length(restriction.basis))
    RestrictedLiouvillianWorkspace(restriction,vector,similar(vector),
                                   _restricted_source_workspace(source))
end

function _check_restricted_workspace(work::RestrictedLiouvillianWorkspace,
                                     source,restriction)
    work.restriction===restriction||throw(ArgumentError(
        "restricted workspace belongs to a different coordinate restriction"))
    ambient=length(restriction.basis)
    length(work.ambient_input)==ambient&&length(work.ambient_output)==ambient||
        throw(DimensionMismatch("restricted workspace has the wrong ambient dimension"))
    expected=_complex_float_type(eltype(source))
    eltype(work.ambient_input)===expected&&eltype(work.ambient_output)===expected||
        throw(ArgumentError("restricted workspace has an incompatible scalar type"))
    work
end

function _restricted_source_apply!(destination,source,input,time,parameters,work)
    if work===nothing
        return source isa AbstractMatrix ? mul!(destination,source,input) :
               apply!(destination,source,input,time,parameters)
    end
    apply!(destination,source,input,time,parameters,work)
end

function _restricted_source_adjoint!(destination,source,input,time,parameters,work)
    _operator_has_adjoint(source)||throw(ArgumentError(
        "the source does not provide a matrix-free adjoint action"))
    source isa AbstractMatrix&&return mul!(destination,adjoint(source),input)
    work===nothing ? apply_adjoint!(destination,source,input,time,parameters) :
        apply_adjoint!(destination,source,input,time,parameters,work)
end

"""Numerical certificate for exhaustive coordinate-subspace leakage testing."""
struct RestrictionInvarianceReport{R}
    invariant::Bool
    leakage_norm::R
    relative_leakage::R
    action_norm::R
    maximum_column_leakage::R
    tolerance::R
    applications::Int
    validation::Symbol
end

function Base.show(io::IO,report::RestrictionInvarianceReport)
    print(io,"RestrictionInvarianceReport(invariant=$(report.invariant), ",
          "leakage=$(report.leakage_norm), tolerance=$(report.tolerance), ",
          "applications=$(report.applications))")
end

"""
    restriction_invariance(source, restriction;
        atol=0, rtol=nothing, workspace=nothing)

Exhaustively certify numerical invariance of the selected coordinate range.
Explicit matrices are scanned directly, while a matrix-free autonomous
`source` is applied to every retained coordinate vector.  The
reported `leakage_norm` is the Frobenius norm of `(I-P)*source*P`, while
`action_norm` is the Frobenius norm of `source*P`.  No global Liouvillian or
projection matrix is materialized.  `rtol=nothing` selects `100eps(R)` in the
source's real working precision.

This is an exhaustive floating-point certificate, not a symbolic theorem.  A
driven source must first be made autonomous with [`freeze`](@ref); testing one
instant would not certify invariance for all later applications.
"""
function restriction_invariance(source,
        restriction::SymmetryCoordinateRestriction;
        atol::Real=0,rtol=nothing,workspace=nothing)
    _require_autonomous(source,"symmetry-coordinate invariance certification")
    ambient=length(restriction.basis)
    size(source)==(ambient,ambient)||throw(DimensionMismatch(
        "the source Liouvillian and restriction have incompatible dimensions"))
    work=workspace===nothing ?
        RestrictedLiouvillianWorkspace(source,restriction) :
        _check_restricted_workspace(workspace,source,restriction)
    R=_real_float_type(eltype(work.ambient_input))
    atolR=_restriction_tolerance(R,atol,"atol")
    rtolR=rtol===nothing ? R(100)*eps(R) :
        _restriction_tolerance(R,rtol,"rtol")
    input=work.ambient_input;output=work.ambient_output
    action_norm=zero(R);leakage_norm=zero(R);maximum_column=zero(R)
    for coordinate in restriction.indices
        fill!(input,zero(eltype(input)));input[coordinate]=one(eltype(input))
        _restricted_source_apply!(output,source,input,zero(R),nothing,
                                  work.source_workspace)
        column_norm=norm(output)
        column_leakage=zero(R)
        @inbounds for index in eachindex(output)
            restriction.mask[index]||
                (column_leakage=hypot(column_leakage,abs(output[index])))
        end
        action_norm=hypot(action_norm,column_norm)
        leakage_norm=hypot(leakage_norm,column_leakage)
        maximum_column=max(maximum_column,column_leakage)
    end
    isfinite(action_norm)&&isfinite(leakage_norm)&&isfinite(maximum_column)||
        throw(ArgumentError(
            "the source produced nonfinite data during restriction certification"))
    tolerance=atolR+rtolR*action_norm
    isfinite(tolerance)||throw(ArgumentError(
        "the restriction tolerance overflows the source working precision"))
    relative=iszero(action_norm) ?
        (iszero(leakage_norm) ? zero(R) : R(Inf)) : leakage_norm/action_norm
    RestrictionInvarianceReport(leakage_norm<=tolerance,leakage_norm,
        relative,action_norm,maximum_column,tolerance,length(restriction),
        :exhaustive_coordinate_probe)
end

function _matrix_restriction_report(action_norm,leakage_norm,maximum_column,
        atolR,rtolR)
    isfinite(action_norm)&&isfinite(leakage_norm)&&isfinite(maximum_column)||
        throw(ArgumentError(
            "the source contains nonfinite data in the selected columns"))
    R=typeof(action_norm)
    tolerance=atolR+rtolR*action_norm
    isfinite(tolerance)||throw(ArgumentError(
        "the restriction tolerance overflows the source working precision"))
    relative=iszero(action_norm) ?
        (iszero(leakage_norm) ? zero(R) : R(Inf)) : leakage_norm/action_norm
    RestrictionInvarianceReport(leakage_norm<=tolerance,leakage_norm,
        relative,action_norm,maximum_column,tolerance,0,
        :exhaustive_matrix_scan)
end

# Explicit matrices already expose their columns. Scan them directly instead
# of performing one allocation-prone matrix-vector product per retained
# coordinate. This remains the exact Frobenius leakage certificate used by the
# generic route and makes sparse strong-symmetry setup proportional to the
# selected stored nonzeros.
function restriction_invariance(source::AbstractMatrix,
        restriction::SymmetryCoordinateRestriction;
        atol::Real=0,rtol=nothing,workspace=nothing)
    _require_autonomous(source,"symmetry-coordinate invariance certification")
    ambient=length(restriction.basis)
    size(source)==(ambient,ambient)||throw(DimensionMismatch(
        "the source Liouvillian and restriction have incompatible dimensions"))
    workspace===nothing||_check_restricted_workspace(
        workspace,source,restriction)
    R=_real_float_type(_complex_float_type(eltype(source)))
    atolR=_restriction_tolerance(R,atol,"atol")
    rtolR=rtol===nothing ? R(100)*eps(R) :
        _restriction_tolerance(R,rtol,"rtol")
    action_norm=zero(R);leakage_norm=zero(R);maximum_column=zero(R)
    @inbounds for column in restriction.indices
        column_norm=zero(R);column_leakage=zero(R)
        for row in axes(source,1)
            magnitude=abs(source[row,column])
            column_norm=hypot(column_norm,magnitude)
            restriction.mask[row]||
                (column_leakage=hypot(column_leakage,magnitude))
        end
        action_norm=hypot(action_norm,column_norm)
        leakage_norm=hypot(leakage_norm,column_leakage)
        maximum_column=max(maximum_column,column_leakage)
    end
    _matrix_restriction_report(action_norm,leakage_norm,maximum_column,
        atolR,rtolR)
end

# A diagonal Hilbert-space charge restriction has a Cartesian support inside
# each Schur block: selected ket GT patterns times selected bra GT patterns.
# Keeping that rectangular description lets the ordinary prepared Schur
# kernels act directly on reduced coordinates.  Explicit coordinate masks that
# are not Cartesian remain valid, but use the embedded fallback below.
struct _RestrictedSectorBlock
    sector::Int
    rows::Vector{Int}
    columns::Vector{Int}
    offset::Int
end

function _restricted_sector_blocks(restriction::SymmetryCoordinateRestriction)
    basis=restriction.basis
    blocks=_RestrictedSectorBlock[]
    indices=restriction.indices
    for sector in eachindex(basis.sectors)
        dimension=length(basis.patterns[sector])
        first_coordinate=basis.offsets[sector]
        last_coordinate=first_coordinate+dimension^2-1
        first_reduced=searchsortedfirst(indices,first_coordinate)
        last_reduced=searchsortedlast(indices,last_coordinate)
        first_reduced<=last_reduced||continue
        rows=Int[];columns=Int[]
        for reduced in first_reduced:last_reduced
            local_coordinate=indices[reduced]-first_coordinate
            push!(rows,mod(local_coordinate,dimension)+1)
            push!(columns,div(local_coordinate,dimension)+1)
        end
        sort!(unique!(rows));sort!(unique!(columns))
        length(rows)*length(columns)==last_reduced-first_reduced+1||return nothing
        reduced=first_reduced
        for column in columns,row in rows
            indices[reduced]==first_coordinate+row-1+(column-1)*dimension||
                return nothing
            reduced+=1
        end
        push!(blocks,_RestrictedSectorBlock(sector,rows,columns,first_reduced))
    end
    blocks
end

struct _RestrictedHamiltonianKernel{L,R,S}
    left_blocks::L
    right_blocks::R
    scale::S
end

struct _RestrictedDissipatorKernel{L,R,Q,S}
    left_blocks::L
    right_blocks::R
    qblocks::Q
    scale::S
end


struct _RestrictedLocalJumpKernel{Q,I,J,V,S}
    qblocks::Q
    I::I
    J::J
    V::V
    scale::S
end


struct _RestrictedKernelPlan{B,G,K,T}
    basis::B
    sectors::G
    kernels::K
    dimension::Int
    Ttype::Type{T}
end

Base.size(plan::_RestrictedKernelPlan)=(plan.dimension,plan.dimension)
Base.size(plan::_RestrictedKernelPlan,index::Integer)=
    index in (1,2) ? plan.dimension : 1
Base.eltype(plan::_RestrictedKernelPlan)=plan.Ttype
isautonomous(::_RestrictedKernelPlan)=true

function _restricted_source_plan(source)
    source isa LiouvillianPlan&&return source
    if source isa CompiledPIModel
        kernels=source.plan.kernels
        return kernels!==nothing&&
            any(kernel->kernel isa FusedStaticPIKernel,kernels) ?
                _term_resolved_liouvillian_plan(source.model) : source.plan
    end
    source isa MatrixFreeLiouvillian&&source.plan isa LiouvillianPlan&&
        return source.plan
    nothing
end

function _restricted_static_kernels(kernels::Tuple)
    supported=Union{HamiltonianPIKernel,DissipatorPIKernel,LocalJumpPIKernel,
                    FactorizedLocalJumpPIKernel,
                    FactorizedLocalPBodyJumpPIKernel}
    all(kernel->kernel isa supported,kernels)
end

function _restricted_static_kernels(kernels)
    false
end

function _restricted_block_pairs(blocks,sectors)
    map(sectors) do geometry
        block=blocks[geometry.sector]
        left=Matrix(block[geometry.rows,geometry.rows])
        # Equal ket and bra charges are the ordinary density-operator case.
        # Share the detached restricted block instead of retaining an
        # identical second copy for every prepared kernel and Schur sector.
        right=geometry.rows==geometry.columns ? left :
            Matrix(block[geometry.columns,geometry.columns])
        (left,right)
    end
end

function _lower_restricted_kernel(kernel::HamiltonianPIKernel,sectors,
                                  reverse_lookup,basis)
    pairs=_restricted_block_pairs(kernel.blocks,sectors)
    _RestrictedHamiltonianKernel(map(first,pairs),map(last,pairs),kernel.scale)
end

function _lower_restricted_kernel(kernel::DissipatorPIKernel,sectors,
                                  reverse_lookup,basis)
    pairs=_restricted_block_pairs(kernel.blocks,sectors)
    qpairs=_restricted_block_pairs(kernel.qblocks,sectors)
    _RestrictedDissipatorKernel(map(first,pairs),map(last,pairs),qpairs,
                                kernel.scale)
end

function _lower_restricted_kernel(kernel::LocalJumpPIKernel,sectors,
                                  reverse_lookup,basis)
    qpairs=_restricted_block_pairs(kernel.qblocks,sectors)
    I=Int[];J=Int[];V=eltype(kernel.gain.V)[]
    for index in eachindex(kernel.gain.V)
        output=get(reverse_lookup,kernel.gain.I[index],0)
        input=get(reverse_lookup,kernel.gain.J[index],0)
        iszero(output)||iszero(input)||begin
            push!(I,output);push!(J,input);push!(V,kernel.gain.V[index])
        end
    end
    _RestrictedLocalJumpKernel(qpairs,I,J,V,kernel.scale)
end

function _lower_restricted_kernel(kernel::FactorizedLocalJumpPIKernel,sectors,
                                  reverse_lookup,b)
    qpairs=_restricted_block_pairs(kernel.qblocks,sectors)
    T=promote_type(eltype(first(kernel.contractions)),
                   eltype(first(kernel.qblocks)))
    I=Int[];J=Int[];V=T[]
    @inbounds for branch_index in eachindex(kernel.branches.entries)
        branch=kernel.branches.entries[branch_index]
        li=branch.output_sector;ni=branch.input_sector
        nl=length(b.patterns[li]);nn=length(b.patterns[ni])
        contraction=kernel.contractions[branch_index]
        for bb in 1:nl,a in 1:nl,d in 1:nn,c in 1:nn
            output=get(reverse_lookup,b.offsets[li]+a-1+(bb-1)*nl,0)
            input=get(reverse_lookup,b.offsets[ni]+c-1+(d-1)*nn,0)
            (iszero(output)||iszero(input))&&continue
            value=branch.scale*contraction[a,c]*conj(contraction[bb,d])
            iszero(value)&&continue
            push!(I,output);push!(J,input);push!(V,value)
        end
    end
    _RestrictedLocalJumpKernel(qpairs,I,J,V,kernel.scale)
end

function _lower_restricted_kernel(kernel::FactorizedLocalPBodyJumpPIKernel,
                                  sectors,reverse_lookup,b)
    qpairs=_restricted_block_pairs(kernel.qblocks,sectors)
    T=promote_type(eltype(first(kernel.contractions)),
                   eltype(first(kernel.qblocks)))
    I=Int[];J=Int[];V=T[]
    @inbounds for (li,ni,first_pair,last_pair) in kernel.groups
        nl=length(b.patterns[li]);nn=length(b.patterns[ni])
        for pair in first_pair:last_pair
            contraction=kernel.contractions[pair];exact_scale=kernel.pair_scales[pair]
            for bb in 1:nl,a in 1:nl,d in 1:nn,c in 1:nn
                output=get(reverse_lookup,b.offsets[li]+a-1+(bb-1)*nl,0)
                input=get(reverse_lookup,b.offsets[ni]+c-1+(d-1)*nn,0)
                (iszero(output)||iszero(input))&&continue
                primitive=contraction[a,c]*conj(contraction[bb,d])
                value=exact_scale.direct ? exact_scale.factor*primitive :
                    _apply_prepared_exact_scale(primitive,exact_scale;
                        context="restricted local p-body gain")
                iszero(value)&&continue
                push!(I,output);push!(J,input);push!(V,value)
            end
        end
    end
    _RestrictedLocalJumpKernel(qpairs,I,J,V,kernel.scale)
end

function _lower_restricted_kernels(::Tuple{},sectors,reverse_lookup,basis)
    ()
end

function _lower_restricted_kernels(kernels::Tuple{K,Vararg{Any}},sectors,
                                   reverse_lookup,basis) where K
    (_lower_restricted_kernel(first(kernels),sectors,reverse_lookup,basis),
     _lower_restricted_kernels(Base.tail(kernels),sectors,reverse_lookup,
                               basis)...)
end

function _restricted_kernel_plan(source,
        restriction::SymmetryCoordinateRestriction)
    plan=_restricted_source_plan(source)
    plan===nothing&&return nothing
    plan.kernels===nothing&&return nothing
    _restricted_static_kernels(plan.kernels)||return nothing
    sectors=_restricted_sector_blocks(restriction)
    sectors===nothing&&return nothing
    reverse_lookup=Dict{Int,Int}(ambient=>reduced
        for (reduced,ambient) in pairs(restriction.indices))
    kernels=_lower_restricted_kernels(
        plan.kernels,sectors,reverse_lookup,plan.basis)
    _RestrictedKernelPlan(plan.basis,sectors,kernels,length(restriction),
                          plan.Ttype)
end

struct _RestrictedKernelWorkspace{P,W,T}
    plan::P
    blocks::W
    Ttype::Type{T}
end

function _RestrictedKernelWorkspace(plan::_RestrictedKernelPlan)
    T=plan.Ttype
    blocks=map(plan.sectors) do geometry
        shape=(length(geometry.rows),length(geometry.columns))
        (zeros(T,shape),zeros(T,shape),zeros(T,shape))
    end
    _RestrictedKernelWorkspace(plan,blocks,T)
end

function _check_restricted_kernel_workspace(work::_RestrictedKernelWorkspace,
                                            plan::_RestrictedKernelPlan)
    work.plan===plan||throw(ArgumentError(
        "restricted-kernel workspace belongs to a different lowered plan"))
    work.Ttype===plan.Ttype||throw(ArgumentError(
        "restricted-kernel workspace has an incompatible scalar type"))
    work
end

@inline function _restricted_copy_block!(destination,source,offset)
    copyto!(destination,1,source,offset,length(destination))
    destination
end

function _apply_restricted_kernel!(destination,input,
        kernel::_RestrictedHamiltonianKernel,plan,time,parameters,work)
    scale=convert(plan.Ttype,value_at(kernel.scale,time,parameters))
    for index in eachindex(plan.sectors)
        geometry=plan.sectors[index];left,right,state=work.blocks[index]
        _restricted_copy_block!(state,input,geometry.offset)
        mul!(left,kernel.left_blocks[index],state)
        mul!(right,state,kernel.right_blocks[index])
        @inbounds for coordinate in eachindex(left,right)
            destination[geometry.offset+coordinate-1]+=
                (-1im*scale)*(left[coordinate]-right[coordinate])
        end
    end
    destination
end

function _apply_restricted_kernel!(destination,input,
        kernel::_RestrictedDissipatorKernel,plan,time,parameters,work)
    scale=convert(plan.Ttype,
        _evaluated_dissipative_rate(kernel.scale,time,parameters))
    for index in eachindex(plan.sectors)
        geometry=plan.sectors[index];left,right,state=work.blocks[index]
        _restricted_copy_block!(state,input,geometry.offset)
        left_jump=kernel.left_blocks[index];right_jump=kernel.right_blocks[index]
        mul!(left,left_jump,state);mul!(right,left,adjoint(right_jump))
        @inbounds for coordinate in eachindex(right)
            destination[geometry.offset+coordinate-1]+=scale*right[coordinate]
        end
        qleft,qright=kernel.qblocks[index]
        mul!(left,qleft,state);mul!(right,state,qright)
        @inbounds for coordinate in eachindex(left,right)
            destination[geometry.offset+coordinate-1]-=
                (scale/2)*(left[coordinate]+right[coordinate])
        end
    end
    destination
end

function _apply_restricted_kernel!(destination,input,
        kernel::_RestrictedLocalJumpKernel,plan,time,parameters,work)
    scale=convert(plan.Ttype,
        _evaluated_dissipative_rate(kernel.scale,time,parameters))
    @inbounds for index in eachindex(kernel.V)
        destination[kernel.I[index]]+=scale*kernel.V[index]*input[kernel.J[index]]
    end
    for index in eachindex(plan.sectors)
        geometry=plan.sectors[index];left,right,state=work.blocks[index]
        _restricted_copy_block!(state,input,geometry.offset)
        qleft,qright=kernel.qblocks[index]
        mul!(left,qleft,state);mul!(right,state,qright)
        @inbounds for coordinate in eachindex(left,right)
            destination[geometry.offset+coordinate-1]-=
                (scale/2)*(left[coordinate]+right[coordinate])
        end
    end
    destination
end

function _apply_restricted_adjoint_kernel!(destination,input,
        kernel::_RestrictedHamiltonianKernel,plan,time,parameters,work)
    scale=conj(convert(plan.Ttype,value_at(kernel.scale,time,parameters)))
    for index in eachindex(plan.sectors)
        geometry=plan.sectors[index];left,right,state=work.blocks[index]
        _restricted_copy_block!(state,input,geometry.offset)
        mul!(left,adjoint(kernel.left_blocks[index]),state)
        mul!(right,state,adjoint(kernel.right_blocks[index]))
        @inbounds for coordinate in eachindex(left,right)
            destination[geometry.offset+coordinate-1]+=
                (1im*scale)*(left[coordinate]-right[coordinate])
        end
    end
    destination
end

function _apply_restricted_adjoint_kernel!(destination,input,
        kernel::_RestrictedDissipatorKernel,plan,time,parameters,work)
    scale=conj(convert(plan.Ttype,
        _evaluated_dissipative_rate(kernel.scale,time,parameters)))
    for index in eachindex(plan.sectors)
        geometry=plan.sectors[index];left,right,state=work.blocks[index]
        _restricted_copy_block!(state,input,geometry.offset)
        left_jump=kernel.left_blocks[index];right_jump=kernel.right_blocks[index]
        mul!(left,adjoint(left_jump),state);mul!(right,left,right_jump)
        @inbounds for coordinate in eachindex(right)
            destination[geometry.offset+coordinate-1]+=scale*right[coordinate]
        end
        qleft,qright=kernel.qblocks[index]
        mul!(left,adjoint(qleft),state);mul!(right,state,adjoint(qright))
        @inbounds for coordinate in eachindex(left,right)
            destination[geometry.offset+coordinate-1]-=
                (scale/2)*(left[coordinate]+right[coordinate])
        end
    end
    destination
end

function _apply_restricted_adjoint_kernel!(destination,input,
        kernel::_RestrictedLocalJumpKernel,plan,time,parameters,work)
    scale=conj(convert(plan.Ttype,
        _evaluated_dissipative_rate(kernel.scale,time,parameters)))
    @inbounds for index in eachindex(kernel.V)
        destination[kernel.J[index]]+=scale*conj(kernel.V[index])*input[kernel.I[index]]
    end
    for index in eachindex(plan.sectors)
        geometry=plan.sectors[index];left,right,state=work.blocks[index]
        _restricted_copy_block!(state,input,geometry.offset)
        qleft,qright=kernel.qblocks[index]
        mul!(left,adjoint(qleft),state);mul!(right,state,adjoint(qright))
        @inbounds for coordinate in eachindex(left,right)
            destination[geometry.offset+coordinate-1]-=
                (scale/2)*(left[coordinate]+right[coordinate])
        end
    end
    destination
end

_apply_restricted_kernels!(destination,input,::Tuple{},plan,time,parameters,work)=
    destination
function _apply_restricted_kernels!(destination,input,
        kernels::Tuple{K,Vararg{Any}},plan,time,parameters,work) where K
    _apply_restricted_kernel!(destination,input,first(kernels),plan,time,parameters,work)
    _apply_restricted_kernels!(destination,input,Base.tail(kernels),plan,
                               time,parameters,work)
end

_apply_restricted_adjoint_kernels!(destination,input,::Tuple{},plan,time,
                                   parameters,work)=destination
function _apply_restricted_adjoint_kernels!(destination,input,
        kernels::Tuple{K,Vararg{Any}},plan,time,parameters,work) where K
    _apply_restricted_adjoint_kernel!(destination,input,first(kernels),plan,
                                      time,parameters,work)
    _apply_restricted_adjoint_kernels!(destination,input,Base.tail(kernels),plan,
                                       time,parameters,work)
end

function apply!(destination::AbstractVector,plan::_RestrictedKernelPlan,
                input::AbstractVector,time,parameters,
                work::_RestrictedKernelWorkspace)
    length(input)==plan.dimension&&length(destination)==plan.dimension||
        throw(DimensionMismatch("lowered restricted vector has the wrong length"))
    Base.mightalias(destination,input)&&throw(ArgumentError(
        "lowered restricted source and destination must not share storage"))
    _check_restricted_kernel_workspace(work,plan)
    promote_type(plan.Ttype,eltype(input))===plan.Ttype||throw(ArgumentError(
        "lowered restricted input cannot be represented without narrowing"))
    promote_type(plan.Ttype,eltype(destination))===eltype(destination)||
        throw(ArgumentError(
            "lowered restricted destination cannot represent the plan precision"))
    fill!(destination,zero(eltype(destination)))
    _apply_restricted_kernels!(destination,input,plan.kernels,plan,time,
                               parameters,work)
end

function apply_adjoint!(destination::AbstractVector,plan::_RestrictedKernelPlan,
        input::AbstractVector,time,parameters,work::_RestrictedKernelWorkspace)
    length(input)==plan.dimension&&length(destination)==plan.dimension||
        throw(DimensionMismatch("lowered restricted adjoint vector has the wrong length"))
    Base.mightalias(destination,input)&&throw(ArgumentError(
        "lowered restricted adjoint source and destination must not share storage"))
    _check_restricted_kernel_workspace(work,plan)
    fill!(destination,zero(eltype(destination)))
    _apply_restricted_adjoint_kernels!(destination,input,plan.kernels,plan,time,
                                       parameters,work)
end

struct _LoweredRestrictedOperator{P,W,K}
    plan::P
    workspace::W
    lock::K
end

function _LoweredRestrictedOperator(plan::_RestrictedKernelPlan)
    _LoweredRestrictedOperator(plan,_RestrictedKernelWorkspace(plan),ReentrantLock())
end

Base.size(operator::_LoweredRestrictedOperator)=size(operator.plan)
Base.size(operator::_LoweredRestrictedOperator,index::Integer)=size(operator.plan,index)
Base.eltype(operator::_LoweredRestrictedOperator)=eltype(operator.plan)
isautonomous(::_LoweredRestrictedOperator)=true

function LinearAlgebra.mul!(destination::AbstractVector,
        operator::_LoweredRestrictedOperator,input::AbstractVector)
    lock(operator.lock)
    try
        apply!(destination,operator.plan,input,zero(_real_float_type(eltype(operator))),
               nothing,operator.workspace)
    finally
        unlock(operator.lock)
    end
end

function LinearAlgebra.mul!(destination::AbstractMatrix,
        operator::_LoweredRestrictedOperator,input::AbstractMatrix)
    size(input,1)==size(operator,2)&&
        size(destination)==(size(operator,1),size(input,2))||
        throw(DimensionMismatch("lowered restricted batch has the wrong dimensions"))
    lock(operator.lock)
    try
        for column in axes(input,2)
            apply!(view(destination,:,column),operator.plan,
                   view(input,:,column),zero(_real_float_type(eltype(operator))),
                   nothing,operator.workspace)
        end
    finally
        unlock(operator.lock)
    end
    destination
end

struct _AdjointLoweredRestrictedOperator{O}
    parent::O
end
Base.size(operator::_AdjointLoweredRestrictedOperator)=reverse(size(operator.parent))
Base.size(operator::_AdjointLoweredRestrictedOperator,index::Integer)=
    size(operator.parent,3-index)
Base.eltype(operator::_AdjointLoweredRestrictedOperator)=eltype(operator.parent)
Base.adjoint(operator::_LoweredRestrictedOperator)=
    _AdjointLoweredRestrictedOperator(operator)
Base.adjoint(operator::_AdjointLoweredRestrictedOperator)=operator.parent
function LinearAlgebra.mul!(destination::AbstractVector,
        operator::_AdjointLoweredRestrictedOperator,input::AbstractVector)
    parent=operator.parent
    lock(parent.lock)
    try
        apply_adjoint!(destination,parent.plan,input,
            zero(_real_float_type(eltype(parent))),nothing,parent.workspace)
    finally
        unlock(parent.lock)
    end
end
function LinearAlgebra.mul!(destination::AbstractMatrix,
        operator::_AdjointLoweredRestrictedOperator,input::AbstractMatrix)
    parent=operator.parent
    size(input,1)==size(parent,1)&&
        size(destination)==(size(parent,2),size(input,2))||
        throw(DimensionMismatch("lowered restricted adjoint batch has the wrong dimensions"))
    lock(parent.lock)
    try
        for column in axes(input,2)
            apply_adjoint!(view(destination,:,column),parent.plan,
                view(input,:,column),zero(_real_float_type(eltype(parent))),
                nothing,parent.workspace)
        end
    finally
        unlock(parent.lock)
    end
    destination
end

function restriction_invariance(source::SparseMatrixCSC,
        restriction::SymmetryCoordinateRestriction;
        atol::Real=0,rtol=nothing,workspace=nothing)
    ambient=length(restriction.basis)
    size(source)==(ambient,ambient)||throw(DimensionMismatch(
        "the source Liouvillian and restriction have incompatible dimensions"))
    workspace===nothing||_check_restricted_workspace(
        workspace,source,restriction)
    R=_real_float_type(_complex_float_type(eltype(source)))
    atolR=_restriction_tolerance(R,atol,"atol")
    rtolR=rtol===nothing ? R(100)*eps(R) :
        _restriction_tolerance(R,rtol,"rtol")
    action_norm=zero(R);leakage_norm=zero(R);maximum_column=zero(R)
    rows=rowvals(source);values=nonzeros(source)
    @inbounds for column in restriction.indices
        column_norm=zero(R);column_leakage=zero(R)
        for pointer in nzrange(source,column)
            magnitude=abs(values[pointer])
            column_norm=hypot(column_norm,magnitude)
            restriction.mask[rows[pointer]]||
                (column_leakage=hypot(column_leakage,magnitude))
        end
        action_norm=hypot(action_norm,column_norm)
        leakage_norm=hypot(leakage_norm,column_leakage)
        maximum_column=max(maximum_column,column_leakage)
    end
    _matrix_restriction_report(action_norm,leakage_norm,maximum_column,
        atolR,rtolR)
end

"""
    RestrictedLiouvillian(source, restriction;
                          atol=0, rtol=nothing, backend=:auto)

Prepare the compressed operator `P' * source * P` after exhaustively
certifying that the autonomous source leaves the coordinate range invariant.
For a sparse or dense matrix source, `backend=:auto` stores and applies the
actual compressed matrix `source[indices,indices]`; neither the reduced
Krylov iteration, one operator application, nor the compatibility wrapper
then retains or touches ambient vectors.
For a prepared matrix-free source whose fixed kernels have Cartesian Schur
support, `backend=:auto` lowers those kernels directly to the selected ket and
bra GT-pattern rectangles.  This `:lowered` backend retains no ambient
application vectors and is the preferred strong-symmetry path.  Unsupported
prepared or explicit coordinate masks retain the `:embedded` fallback, whose
large working vectors remain caller-owned through
[`RestrictedLiouvillianWorkspace`](@ref). `backend=:embedded` may be selected
explicitly; `backend=:compressed` requires an `AbstractMatrix` source, and
`backend=:lowered` requires a compatible prepared `LiouvillianPlan`.

Compatibility embedded `mul!` calls use a locked workspace; parallel code
should call `apply!` with one explicit workspace per task.

The stored trace functional is the exact equation-(7) trace vector restricted
in the source working precision.  A restriction with a zero trace functional
is valid for decay-mode calculations but is rejected by
[`restricted_steady_state`](@ref).
"""
struct RestrictedLiouvillian{S,R,T,V,C,M,W,K}
    source::S
    restriction::R
    Ttype::Type{T}
    tracevec::V
    certificate::C
    compressed_source::M
    backend::Symbol
    compatibility_workspace::W
    lock::K
end

function RestrictedLiouvillian(source,
        restriction::SymmetryCoordinateRestriction;
        atol::Real=0,rtol=nothing,workspace=nothing,
        backend::Symbol=:auto)
    backend in (:auto,:compressed,:lowered,:embedded)||throw(ArgumentError(
        "restricted Liouvillian backend must be :auto, :compressed, :lowered, or :embedded"))
    backend===:compressed&&!(source isa AbstractMatrix)&&throw(ArgumentError(
        "backend=:compressed requires an AbstractMatrix source"))
    _require_autonomous(source,"restricted Liouvillian construction")
    work=if workspace!==nothing
        _check_restricted_workspace(workspace,source,restriction)
    elseif source isa AbstractMatrix
        nothing
    else
        RestrictedLiouvillianWorkspace(source,restriction)
    end
    certificate=restriction_invariance(source,restriction;
        atol,rtol,workspace=work)
    certificate.invariant||throw(ArgumentError(
        "the requested coordinate range is not invariant: " *
        "leakage=$(certificate.leakage_norm), " *
        "tolerance=$(certificate.tolerance)"))
    # Only copy the retained Schur subblocks after the exhaustive certificate
    # has accepted the range. A leaking user mask therefore pays no lowered-
    # plan allocation before the constructor raises.
    lowered_plan=source isa AbstractMatrix ? nothing :
        _restricted_kernel_plan(source,restriction)
    backend===:lowered&&lowered_plan===nothing&&throw(ArgumentError(
        "backend=:lowered requires fixed prepared kernels and Cartesian Schur-block support"))
    selected_backend=backend===:auto ?
        (source isa AbstractMatrix ? :compressed :
         lowered_plan===nothing ? :embedded : :lowered) : backend
    T=_complex_float_type(eltype(source))
    tracevec=restricted_trace_vector(restriction,T)
    compressed=selected_backend===:compressed ?
        source[restriction.indices,restriction.indices] :
        selected_backend===:lowered ? _LoweredRestrictedOperator(lowered_plan) :
        nothing
    # A truly compressed matrix application needs no ambient scratch. For an
    # embedded backend, a caller-supplied certification workspace remains
    # caller-owned and the compatibility path owns distinct locked scratch.
    compatibility=if selected_backend in (:compressed,:lowered)
        nothing
    elseif workspace===nothing&&work!==nothing
        work
    else
        RestrictedLiouvillianWorkspace(source,restriction)
    end
    RestrictedLiouvillian(source,restriction,T,tracevec,certificate,compressed,
                          selected_backend,compatibility,ReentrantLock())
end

Base.size(operator::RestrictedLiouvillian)=(length(operator.restriction),
                                            length(operator.restriction))
Base.size(operator::RestrictedLiouvillian,index::Integer)=
    index in (1,2) ? length(operator.restriction) : 1
Base.eltype(operator::RestrictedLiouvillian)=operator.Ttype
isautonomous(operator::RestrictedLiouvillian)=isautonomous(operator.source)

function Base.show(io::IO,operator::RestrictedLiouvillian)
    print(io,"RestrictedLiouvillian($(size(operator,1)) / ",
          "$(length(operator.restriction.basis)) coordinates, ",
          "backend=$(operator.backend), ",
          "leakage=$(operator.certificate.leakage_norm))")
end

"""
    RestrictedLiouvillianWorkspace(operator::RestrictedLiouvillian)

Construct task-owned application scratch selected for an already prepared
restricted operator.  A `:lowered` operator returns reduced Schur-block
scratch and therefore retains no ambient PI vectors.  Embedded operators
retain the legacy full-coordinate workspace.  Explicit matrices need no
workspace and are rejected by this constructor.
"""
function RestrictedLiouvillianWorkspace(operator::RestrictedLiouvillian)
    if operator.backend===:lowered
        return _RestrictedKernelWorkspace(operator.compressed_source.plan)
    elseif operator.backend===:embedded
        return RestrictedLiouvillianWorkspace(operator.source,
                                               operator.restriction)
    end
    throw(ArgumentError(
        "a compressed matrix restriction does not require an application workspace"))
end

# Participate in the common matrix-free source protocol. A compressed matrix
# needs no mutable scratch; lowered and embedded backends receive one
# task-owned workspace selected by the constructor above.
_operator_trace_vector(operator::RestrictedLiouvillian)=operator.tracevec
_linear_operator_workspace(operator::RestrictedLiouvillian)=
    operator.backend===:compressed ? nothing :
    RestrictedLiouvillianWorkspace(operator)
_operator_has_adjoint(operator::RestrictedLiouvillian)=
    operator.compressed_source!==nothing||_operator_has_adjoint(operator.source)

function _performance_linear_operator_workspace_bytes(
        operator::RestrictedLiouvillian;batch_columns::Integer=0)
    batch_columns>=0||throw(ArgumentError(
        "batch_columns must be nonnegative"))
    operator.backend===:compressed&&return big(0)
    if operator.backend===:lowered
        return _performance_array_bytes(
            size(operator,1),eltype(operator),0;linear_arrays=3)
    end
    ambient=length(operator.restriction.basis)
    _performance_array_bytes(ambient,eltype(operator),0;linear_arrays=2)+
        _performance_linear_operator_workspace_bytes(
            operator.source;batch_columns=0)
end

_performance_source_action_bytes(operator::RestrictedLiouvillian,
        ::Type{T}) where T=operator.backend===:embedded ?
    _performance_source_action_bytes(operator.source,T) : big(0)

function _check_restricted_apply_types(destination,operator,input)
    T=eltype(operator)
    promote_type(T,eltype(input))===T||throw(ArgumentError(
        "the reduced input cannot be represented in the source working precision without narrowing"))
    promote_type(T,eltype(destination))===eltype(destination)||throw(ArgumentError(
        "the reduced destination cannot represent the source working precision"))
    nothing
end

function _check_restricted_input_type(operator,input)
    T=eltype(operator)
    promote_type(T,eltype(input))===T||throw(ArgumentError(
        "the reduced input cannot be represented in the source working precision without narrowing"))
    nothing
end

"""Apply a restricted Liouvillian using caller-owned full-coordinate scratch."""
function apply!(destination::AbstractVector,
        operator::RestrictedLiouvillian,input::AbstractVector,time,parameters)
    operator.backend===:compressed||throw(ArgumentError(
        "a lowered or embedded restricted Liouvillian requires an explicit task-owned workspace"))
    length(input)==size(operator,2)&&length(destination)==size(operator,1)||
        throw(DimensionMismatch("restricted Liouvillian vector has the wrong length"))
    _check_restricted_apply_types(destination,operator,input)
    mul!(destination,operator.compressed_source,input)
end

function apply!(destination::AbstractVector,operator::RestrictedLiouvillian,
                input::AbstractVector,time,parameters,
                work::RestrictedLiouvillianWorkspace)
    length(input)==size(operator,2)&&length(destination)==size(operator,1)||
        throw(DimensionMismatch("restricted Liouvillian vector has the wrong length"))
    _check_restricted_workspace(work,operator.source,operator.restriction)
    _check_restricted_apply_types(destination,operator,input)
    if operator.compressed_source!==nothing
        return mul!(destination,operator.compressed_source,input)
    end
    embed!(work.ambient_input,operator.restriction,input)
    _restricted_source_apply!(work.ambient_output,operator.source,
        work.ambient_input,time,parameters,work.source_workspace)
    restrict!(destination,operator.restriction,work.ambient_output)
end

function apply!(destination::AbstractVector,operator::RestrictedLiouvillian,
                input::AbstractVector,time,parameters,
                work::_RestrictedKernelWorkspace)
    operator.backend===:lowered||throw(ArgumentError(
        "reduced Schur-block scratch requires a :lowered restricted Liouvillian"))
    _check_restricted_apply_types(destination,operator,input)
    apply!(destination,operator.compressed_source.plan,input,time,parameters,work)
end

function apply!(destination::AbstractMatrix,operator::RestrictedLiouvillian,
                input::AbstractMatrix,time,parameters,
                work::RestrictedLiouvillianWorkspace)
    size(input,1)==size(operator,2)&&
        size(destination)==(size(operator,1),size(input,2))||
        throw(DimensionMismatch("restricted Liouvillian batch has the wrong dimensions"))
    _check_restricted_workspace(work,operator.source,operator.restriction)
    _check_restricted_apply_types(destination,operator,input)
    operator.compressed_source!==nothing&&
        return mul!(destination,operator.compressed_source,input)
    for column in axes(input,2)
        apply!(view(destination,:,column),operator,view(input,:,column),
               time,parameters,work)
    end
    destination
end

function apply!(destination::AbstractMatrix,operator::RestrictedLiouvillian,
                input::AbstractMatrix,time,parameters,
                work::_RestrictedKernelWorkspace)
    operator.backend===:lowered||throw(ArgumentError(
        "reduced Schur-block scratch requires a :lowered restricted Liouvillian"))
    size(input,1)==size(operator,2)&&
        size(destination)==(size(operator,1),size(input,2))||
        throw(DimensionMismatch("restricted Liouvillian batch has the wrong dimensions"))
    _check_restricted_apply_types(destination,operator,input)
    for column in axes(input,2)
        apply!(view(destination,:,column),operator.compressed_source.plan,
               view(input,:,column),time,parameters,work)
    end
    destination
end

function apply!(destination::AbstractVector,operator::RestrictedLiouvillian,
                input::AbstractVector,work::RestrictedLiouvillianWorkspace)
    _require_autonomous(operator,"restricted Liouvillian application")
    apply!(destination,operator,input,zero(_real_float_type(eltype(operator))),
           nothing,work)
end

function apply!(destination::AbstractVector,operator::RestrictedLiouvillian,
                input::AbstractVector,work::_RestrictedKernelWorkspace)
    _require_autonomous(operator,"restricted Liouvillian application")
    apply!(destination,operator,input,
           zero(_real_float_type(eltype(operator))),nothing,work)
end

function LinearAlgebra.mul!(destination::AbstractVector,
                            operator::RestrictedLiouvillian,
                            input::AbstractVector)
    _require_autonomous(operator,"restricted Liouvillian multiplication")
    if operator.compressed_source!==nothing
        length(input)==size(operator,2)&&length(destination)==size(operator,1)||
            throw(DimensionMismatch(
                "restricted Liouvillian vector has the wrong length"))
        _check_restricted_apply_types(destination,operator,input)
        return mul!(destination,operator.compressed_source,input)
    end
    lock(operator.lock)
    try
        apply!(destination,operator,input,operator.compatibility_workspace)
    finally
        unlock(operator.lock)
    end
end

function LinearAlgebra.mul!(destination::AbstractMatrix,
        operator::RestrictedLiouvillian,input::AbstractMatrix)
    _require_autonomous(operator,"restricted Liouvillian multiplication")
    if operator.compressed_source!==nothing
        size(input,1)==size(operator,2)&&
            size(destination)==(size(operator,1),size(input,2))||
            throw(DimensionMismatch(
                "restricted Liouvillian batch has the wrong dimensions"))
        _check_restricted_apply_types(destination,operator,input)
        return mul!(destination,operator.compressed_source,input)
    end
    lock(operator.lock)
    try
        apply!(destination,operator,input,
            zero(_real_float_type(eltype(operator))),nothing,
            operator.compatibility_workspace)
    finally
        unlock(operator.lock)
    end
end

Base.:*(operator::RestrictedLiouvillian,input::AbstractVector)=
    mul!(similar(input,promote_type(eltype(operator),eltype(input)),size(operator,1)),
         operator,input)
Base.:*(operator::RestrictedLiouvillian,input::AbstractMatrix)=
    mul!(similar(input,promote_type(eltype(operator),eltype(input)),
                 size(operator,1),size(input,2)),operator,input)

function _materialize(operator::RestrictedLiouvillian)
    operator.compressed_source isa AbstractMatrix&&return operator.compressed_source
    n=size(operator,1)
    matrix=Matrix{eltype(operator)}(undef,n,n)
    source=zeros(eltype(operator),n)
    for column in 1:n
        fill!(source,zero(eltype(source)));source[column]=one(eltype(source))
        mul!(view(matrix,:,column),operator,source)
    end
    matrix
end

"""Apply the adjoint compressed action when the source provides one."""
function apply_adjoint!(destination::AbstractVector,
        operator::RestrictedLiouvillian,input::AbstractVector,time,parameters)
    operator.backend===:compressed||throw(ArgumentError(
        "a lowered or embedded restricted Liouvillian requires an explicit task-owned workspace"))
    length(input)==size(operator,1)&&length(destination)==size(operator,2)||
        throw(DimensionMismatch("restricted adjoint vector has the wrong length"))
    _check_restricted_apply_types(destination,operator,input)
    mul!(destination,adjoint(operator.compressed_source),input)
end

function apply_adjoint!(destination::AbstractVector,
        operator::RestrictedLiouvillian,input::AbstractVector,time,parameters,
        work::RestrictedLiouvillianWorkspace)
    length(input)==size(operator,1)&&length(destination)==size(operator,2)||
        throw(DimensionMismatch("restricted adjoint vector has the wrong length"))
    _check_restricted_workspace(work,operator.source,operator.restriction)
    _check_restricted_apply_types(destination,operator,input)
    if operator.compressed_source!==nothing
        return mul!(destination,adjoint(operator.compressed_source),input)
    end
    embed!(work.ambient_input,operator.restriction,input)
    _restricted_source_adjoint!(work.ambient_output,operator.source,
        work.ambient_input,time,parameters,work.source_workspace)
    restrict!(destination,operator.restriction,work.ambient_output)
end

function apply_adjoint!(destination::AbstractVector,
        operator::RestrictedLiouvillian,input::AbstractVector,time,parameters,
        work::_RestrictedKernelWorkspace)
    operator.backend===:lowered||throw(ArgumentError(
        "reduced Schur-block scratch requires a :lowered restricted Liouvillian"))
    _check_restricted_apply_types(destination,operator,input)
    apply_adjoint!(destination,operator.compressed_source.plan,input,time,
                   parameters,work)
end

function apply_adjoint!(destination::AbstractMatrix,
        operator::RestrictedLiouvillian,input::AbstractMatrix,time,parameters,
        work::RestrictedLiouvillianWorkspace)
    size(input,1)==size(operator,1)&&
        size(destination)==(size(operator,2),size(input,2))||
        throw(DimensionMismatch("restricted adjoint batch has the wrong dimensions"))
    _check_restricted_workspace(work,operator.source,operator.restriction)
    _check_restricted_apply_types(destination,operator,input)
    operator.compressed_source!==nothing&&
        return mul!(destination,adjoint(operator.compressed_source),input)
    for column in axes(input,2)
        apply_adjoint!(view(destination,:,column),operator,
            view(input,:,column),time,parameters,work)
    end
    destination
end

function apply_adjoint!(destination::AbstractMatrix,
        operator::RestrictedLiouvillian,input::AbstractMatrix,time,parameters,
        work::_RestrictedKernelWorkspace)
    operator.backend===:lowered||throw(ArgumentError(
        "reduced Schur-block scratch requires a :lowered restricted Liouvillian"))
    size(input,1)==size(operator,1)&&
        size(destination)==(size(operator,2),size(input,2))||
        throw(DimensionMismatch("restricted adjoint batch has the wrong dimensions"))
    _check_restricted_apply_types(destination,operator,input)
    for column in axes(input,2)
        apply_adjoint!(view(destination,:,column),operator.compressed_source.plan,
            view(input,:,column),time,parameters,work)
    end
    destination
end

function apply_adjoint!(destination::AbstractVector,
        operator::RestrictedLiouvillian,input::AbstractVector,
        work::RestrictedLiouvillianWorkspace)
    _require_autonomous(operator,"restricted adjoint application")
    apply_adjoint!(destination,operator,input,
        zero(_real_float_type(eltype(operator))),nothing,work)
end

function apply_adjoint!(destination::AbstractVector,
        operator::RestrictedLiouvillian,input::AbstractVector,
        work::_RestrictedKernelWorkspace)
    _require_autonomous(operator,"restricted adjoint application")
    apply_adjoint!(destination,operator,input,
        zero(_real_float_type(eltype(operator))),nothing,work)
end

struct _AdjointRestrictedLiouvillian{L}
    parent::L
end

Base.size(operator::_AdjointRestrictedLiouvillian)=reverse(size(operator.parent))
Base.size(operator::_AdjointRestrictedLiouvillian,index::Integer)=
    size(operator.parent,3-index)
Base.eltype(operator::_AdjointRestrictedLiouvillian)=eltype(operator.parent)
isautonomous(operator::_AdjointRestrictedLiouvillian)=isautonomous(operator.parent)

function Base.adjoint(operator::RestrictedLiouvillian)
    _operator_has_adjoint(operator.source)||throw(ArgumentError(
        "the source does not provide a matrix-free adjoint action"))
    _AdjointRestrictedLiouvillian(operator)
end
Base.adjoint(operator::_AdjointRestrictedLiouvillian)=operator.parent

function LinearAlgebra.mul!(destination::AbstractVector,
        operator::_AdjointRestrictedLiouvillian,input::AbstractVector)
    parent=operator.parent
    if parent.compressed_source!==nothing
        length(input)==size(parent,1)&&length(destination)==size(parent,2)||
            throw(DimensionMismatch(
                "restricted adjoint vector has the wrong length"))
        _check_restricted_apply_types(destination,parent,input)
        return mul!(destination,adjoint(parent.compressed_source),input)
    end
    lock(parent.lock)
    try
        apply_adjoint!(destination,parent,input,parent.compatibility_workspace)
    finally
        unlock(parent.lock)
    end
end
function LinearAlgebra.mul!(destination::AbstractMatrix,
        operator::_AdjointRestrictedLiouvillian,input::AbstractMatrix)
    parent=operator.parent
    if parent.compressed_source!==nothing
        size(input,1)==size(parent,1)&&
            size(destination)==(size(parent,2),size(input,2))||
            throw(DimensionMismatch(
                "restricted adjoint batch has the wrong dimensions"))
        _check_restricted_apply_types(destination,parent,input)
        return mul!(destination,adjoint(parent.compressed_source),input)
    end
    lock(parent.lock)
    try
        apply_adjoint!(destination,parent,input,
            zero(_real_float_type(eltype(parent))),nothing,
            parent.compatibility_workspace)
    finally
        unlock(parent.lock)
    end
end
Base.:*(operator::_AdjointRestrictedLiouvillian,input::AbstractVector)=
    mul!(similar(input,promote_type(eltype(operator),eltype(input)),size(operator,1)),
         operator,input)
Base.:*(operator::_AdjointRestrictedLiouvillian,input::AbstractMatrix)=
    mul!(similar(input,promote_type(eltype(operator),eltype(input)),
                 size(operator,1),size(input,2)),operator,input)

"""
    restriction_full_residual(operator, reduced_vector;
                              eigenvalue=0, workspace=nothing)

Embed a reduced vector, evaluate the original full-coordinate source, and
return its full, retained, and leakage residual norms for
`source*x = eigenvalue*x`.  The returned physical trace uses the restricted
equation-(7) trace functional.  This helper is intentionally evaluated in the
ambient PI space, so it independently checks the compressed result without a
`d^N` reconstruction.

This diagnostic necessarily needs ambient PI scratch. The workspace returned
by `RestrictedLiouvillianWorkspace(operator)` for a `:lowered` operator is
reduced-only and is therefore rejected; construct
`RestrictedLiouvillianWorkspace(operator.source, operator.restriction)` when
reusing this diagnostic.
"""
function restriction_full_residual(operator::RestrictedLiouvillian,
        reduced_vector::AbstractVector;eigenvalue=0,workspace=nothing)
    length(reduced_vector)==size(operator,2)||throw(DimensionMismatch(
        "the reduced vector has the wrong length"))
    workspace isa _RestrictedKernelWorkspace&&throw(ArgumentError(
        "restriction_full_residual requires ambient PI scratch; construct " *
        "RestrictedLiouvillianWorkspace(operator.source, operator.restriction)"))
    work=workspace===nothing ?
        RestrictedLiouvillianWorkspace(operator.source,operator.restriction) :
        _check_restricted_workspace(workspace,operator.source,
                                    operator.restriction)
    _check_restricted_input_type(operator,reduced_vector)
    T=eltype(operator)
    promote_type(T,typeof(eigenvalue))===T||throw(ArgumentError(
        "eigenvalue cannot be represented in the source working precision without narrowing"))
    lambda=T(eigenvalue)
    embed!(work.ambient_input,operator.restriction,reduced_vector)
    _restricted_source_apply!(work.ambient_output,operator.source,
        work.ambient_input,zero(_real_float_type(T)),nothing,
        work.source_workspace)
    @inbounds for index in eachindex(work.ambient_output)
        work.ambient_output[index]-=lambda*work.ambient_input[index]
    end
    R=_real_float_type(T);inside=zero(R);outside=zero(R)
    @inbounds for index in eachindex(work.ambient_output)
        if operator.restriction.mask[index]
            inside=hypot(inside,abs(work.ambient_output[index]))
        else
            outside=hypot(outside,abs(work.ambient_output[index]))
        end
    end
    residual=hypot(inside,outside);vector_norm=norm(work.ambient_input)
    relative=iszero(vector_norm) ?
        (iszero(residual) ? zero(R) : R(Inf)) : residual/vector_norm
    (;residual,relative_residual=relative,inside_residual=inside,
      leakage_residual=outside,vector_norm,
      trace=dot(operator.tracevec,reduced_vector),eigenvalue=lambda)
end

"""
    restricted_steady_state(operator; return_info=false, ...)

Solve the trace-fixed stationary equation directly in the certified compressed
coordinates using the existing restarted matrix-free GMRES implementation.
The returned value is an ambient [`PIState`](@ref), with exact zeros outside
the selected support.  `initial_state` may be either an ambient `PIState` or a
reduced coordinate vector.

After GMRES converges, the solution is embedded and checked with
[`restriction_full_residual`](@ref).  A full residual exceeding the same
operator-scaled tolerance raises.  The method certifies the linear stationary
equation and physical trace, not uniqueness or positivity; use the usual state
diagnostics separately when those claims are required.  A preconditioner, when
supplied, must act in the reduced coordinate space.
"""
function restricted_steady_state(operator::RestrictedLiouvillian;
        initial_state=nothing,krylovdim::Integer=30,maxiter::Integer=500,
        atol::Real=1e-10,rtol::Real=1e-8,workspace=nothing,
        preconditioner=nothing,operator_scale=nothing,
        return_info::Bool=false)
    _require_autonomous(operator,"restricted steady-state solving")
    norm(operator.tracevec)>zero(_real_float_type(eltype(operator)))||
        throw(ArgumentError(
        "the selected ket/bra charge block has an identically zero physical trace"))
    reduced_initial=if initial_state isa PIState
        initial_state.basis===operator.restriction.basis||throw(ArgumentError(
            "the initial state and restriction use incompatible PI bases"))
        vector=zeros(promote_type(eltype(operator),eltype(initial_state.data)),
                     size(operator,1))
        restrict!(vector,operator.restriction,initial_state.data)
    else
        initial_state
    end
    result=krylov_steady_state(operator;trace_vector=operator.tracevec,
        initial_state=reduced_initial,krylovdim,maxiter,atol,rtol,workspace,
        preconditioner,operator_scale,return_info=true)
    # Use detached residual scratch.  The compatibility workspace is shared by
    # synchronized `mul!` calls and must not be accessed outside its lock.
    full_report=restriction_full_residual(operator,result.state)
    R=_real_float_type(eltype(result.state))
    tolerance=R(atol)+R(rtol)*max(norm(result.state),one(R))
    normalized_full=full_report.residual/result.operator_scale
    normalized_full<=tolerance||throw(ArgumentError(
        "the compressed steady state fails the full-coordinate residual check: " *
        "normalized_residual=$normalized_full, tolerance=$tolerance"))
    full=zeros(eltype(result.state),length(operator.restriction.basis))
    embed!(full,operator.restriction,result.state)
    state=PIState(operator.restriction.basis,full)
    info=merge(result,(state,reduced_state=result.state,
        full_residual=full_report.residual,
        full_normalized_residual=normalized_full,
        leakage_residual=full_report.leakage_residual,
        restriction_certificate=operator.certificate))
    return_info ? info : state
end
