function _symmetry_roundoff_floor(::Type{R},roundoff) where R<:AbstractFloat
    floor=roundoff===nothing ? R(100)*eps(R) : R(roundoff)
    isfinite(floor)&&floor>=zero(R)||throw(ArgumentError(
        "symmetry roundoff floor must be finite and nonnegative"))
    max(floor,R(100)*eps(R))
end

function _check_unitary_matrix(U;atol,rtol,roundoff=nothing)
    size(U,1)==size(U,2)||throw(DimensionMismatch("symmetry operator must be square"));n=size(U,1)
    R=_real_float_type(eltype(U))
    absolute,relative=_symmetry_tolerances(R,atol,rtol)
    roundoff_floor=_symmetry_roundoff_floor(R,roundoff)
    gram=U'*U;scale=max(norm(gram,Inf),one(R))
    norm(gram-I,Inf)<=absolute+(relative+roundoff_floor)*scale||
        throw(ArgumentError("symmetry operator must be unitary"))
    n
end

function _symmetry_lapack_complex_type(::Type{T}) where T
    R=_real_float_type(T)
    R===Float32&&return ComplexF32
    R===Float64&&return ComplexF64
    throw(ArgumentError(
        "unitary symmetry eigensystems require a Float32 or Float64 LAPACK "*
        "scalar type; convert the symmetry explicitly to ComplexF32 or ComplexF64"))
end

function _symmetry_tolerances(::Type{R},atol,rtol) where R<:AbstractFloat
    absolute=R(atol);relative=R(rtol)
    isfinite(absolute)&&absolute>=zero(R)||throw(ArgumentError(
        "symmetry atol must be finite and nonnegative"))
    isfinite(relative)&&relative>=zero(R)||throw(ArgumentError(
        "symmetry rtol must be finite and nonnegative"))
    absolute,relative
end

"""Return a unitary eigenbasis, including inside degenerate eigenspaces."""
function _orthonormal_unitary_eigensystem(U;atol,rtol,roundoff=nothing)
    _check_unitary_matrix(U;atol=atol,rtol=rtol,roundoff)
    # Eigenvectors returned by a generic eigensolver need not be orthogonal
    # inside a degenerate eigenspace.  Complex Schur vectors are orthonormal by
    # construction, and a normal (in particular unitary) matrix has diagonal
    # complex Schur form up to roundoff.
    T=_symmetry_lapack_complex_type(eltype(U));RT=_real_float_type(T)
    absolute,relative=_symmetry_tolerances(RT,atol,rtol)
    roundoff_floor=_symmetry_roundoff_floor(RT,roundoff)
    F=schur(Matrix{T}(U));schur_form=Matrix(F.T);W=Matrix(F.Z)
    vals=collect(diag(schur_form))
    offdiag=schur_form-Diagonal(vals)
    scale=max(norm(schur_form),one(RT))
    tol=RT(100)*(absolute+relative*scale)+roundoff_floor*scale
    norm(offdiag)<=tol||throw(ArgumentError("failed to construct orthonormal eigenspaces for the unitary symmetry"))
    vals,W
end

function _local_unitary_blocks(
        b::PIBasis,U;atol,rtol,cache=nothing,roundoff=nothing)
    _check_unitary_matrix(U;atol=atol,rtol=rtol,roundoff)==b.d||
        throw(DimensionMismatch("local symmetry must be $(b.d)×$(b.d)"))
    vals,W=_orthonormal_unitary_eigensystem(
        U;atol=atol,rtol=rtol,roundoff)
    T=eltype(W);R=_real_float_type(T)
    H=W*Diagonal(angle.(vals))*W';H=(H+H')/R(2)
    imaginary_unit=complex(zero(R),one(R))
    absolute,relative=_symmetry_tolerances(R,atol,rtol)
    roundoff_floor=_symmetry_roundoff_floor(R,roundoff)
    reconstructed=exp(imaginary_unit*H)
    scale=max(norm(U,Inf),one(R))
    norm(reconstructed-U,Inf)<=R(10)*
        (absolute+(relative+roundoff_floor)*scale)||
        throw(ArgumentError("failed to construct a Hermitian generator for the unitary"))
    cache===nothing&&(cache=OneBodyGeometry(b,R))
    [exp(im*collective_block(b,H,p;cache=cache)) for p in b.sectors]
end

function _pi_unitary_blocks(
        b::PIBasis,U;atol,rtol,cache=nothing,roundoff=nothing)
    if U isa PIOperator
        U.basis===b||throw(ArgumentError("symmetry operator uses an incompatible PI basis"))
        blocks=[Matrix(physical_block(U,p)) for p in b.sectors]
        all(V->_check_unitary_matrix(
            V;atol=atol,rtol=rtol,roundoff)>0,blocks)||error("unreachable")
        blocks
    elseif U isa AbstractMatrix
        _local_unitary_blocks(
            b,U;atol=atol,rtol=rtol,cache=cache,roundoff)
    else
        throw(ArgumentError("PI symmetry must be a local unitary matrix or PIOperator"))
    end
end

function _pi_conjugation_superoperator(b::PIBasis,U;atol,rtol,cache=nothing)
    blocks=_pi_unitary_blocks(b,U;atol=atol,rtol=rtol,cache=cache)
    blockdiag((sparse(kron(conj(V),V)) for V in blocks)...)
end

"""Immutable mathematical data for one unitary-conjugation charge projector."""
struct SymmetryProjectorData{B,C,W,M}
    basis::B
    charge::C
    eigenvectors::W
    masks::M
end

"""Caller-owned scratch for allocation-free symmetry projection."""
struct SymmetryProjectorWorkspace{T}
    work::Vector{Tuple{Matrix{T},Matrix{T}}}
end

"""Matrix-free projector with a locked compatibility workspace."""
struct MatrixFreeSymmetryProjector{D,W,K}
    data::D
    compatibility_workspace::W
    lock::K
end

function Base.getproperty(P::MatrixFreeSymmetryProjector,name::Symbol)
    name in (:basis,:charge,:eigenvectors,:masks) ? getproperty(getfield(P,:data),name) : getfield(P,name)
end
Base.propertynames(::MatrixFreeSymmetryProjector,private::Bool=false)=
    private ? (:basis,:charge,:eigenvectors,:masks,:data,:compatibility_workspace,:lock) :
              (:basis,:charge,:eigenvectors,:masks)

# The Boolean masks are exact coordinates in the sector-local eigenbases, so
# their total population is the exact rank of this orthogonal projector.
_projector_range_dimension(P::MatrixFreeSymmetryProjector,n::Integer)=
    sum(count(mask) for mask in P.masks)

_projector_charges(P::MatrixFreeSymmetryProjector)=(P.charge,)

function SymmetryProjectorWorkspace(P::MatrixFreeSymmetryProjector)
    T=eltype(P)
    SymmetryProjectorWorkspace{T}([(zeros(T,size(V)),zeros(T,size(V)))
                                   for V in P.eigenvectors])
end

size(P::MatrixFreeSymmetryProjector)=(length(P.basis),length(P.basis))
size(P::MatrixFreeSymmetryProjector,i::Integer)=i in (1,2) ? length(P.basis) : 1
eltype(P::MatrixFreeSymmetryProjector)=eltype(first(P.eigenvectors))

function _check_symmetry_projector_workspace(P,work::SymmetryProjectorWorkspace)
    length(work.work)==length(P.eigenvectors)||throw(DimensionMismatch("symmetry-projector workspace has the wrong number of sectors"))
    for s in eachindex(P.eigenvectors)
        n=size(P.eigenvectors[s],1);A,B=work.work[s]
        size(A)==(n,n)&&size(B)==(n,n)||throw(DimensionMismatch("symmetry-projector workspace has an incompatible sector block"))
        eltype(A)===eltype(P)&&eltype(B)===eltype(P)||throw(ArgumentError(
            "symmetry-projector workspace has incompatible scalar precision"))
    end
    work
end

function _check_symmetry_projector_vector_types(P,destination,source)
    Base.mightalias(destination,source)&&destination!==source&&throw(
        ArgumentError("partially aliased symmetry-projector source and "*
                      "destination are unsupported; use distinct arrays or "*
                      "the exact in-place form apply!(x, P, x, workspace)"))
    T=eltype(P)
    promote_type(T,eltype(source))===T||throw(ArgumentError(
        "symmetry projection would narrow the source precision; rebuild the "*
        "projector from a symmetry operator with at least the source precision"))
    promote_type(T,eltype(destination))===eltype(destination)||throw(
        ArgumentError("symmetry-projector destination cannot represent $T values"))
    nothing
end

"""Apply `P` using caller-owned scratch; one workspace may not be shared concurrently."""
function apply!(y,P::MatrixFreeSymmetryProjector,x,
                work::SymmetryProjectorWorkspace)
    length(x)==length(P.basis)&&length(y)==length(P.basis)||throw(DimensionMismatch("projector vector has wrong length"))
    _check_symmetry_projector_vector_types(P,y,x)
    _check_symmetry_projector_workspace(P,work)
    for s in eachindex(P.basis.sectors)
        r=P.basis.offsets[s]:P.basis.offsets[s+1]-1;n=length(P.basis.patterns[s])
        X=reshape(view(x,r),n,n);Y=reshape(view(y,r),n,n)
        V=P.eigenvectors[s];mask=P.masks[s];A,B=work.work[s]
        copyto!(B,X);mul!(A,adjoint(V),B);mul!(B,A,V)
        @inbounds for i in eachindex(B);B[i]=mask[i] ? B[i] : zero(eltype(B));end
        mul!(A,V,B);mul!(B,A,adjoint(V));copyto!(Y,B)
    end
    y
end

# The compatibility path serializes access to its allocation-free workspace.
# Concurrent hot loops should instead use one explicit workspace per task.
function mul!(y,P::MatrixFreeSymmetryProjector,x)
    lock(P.lock)
    try
        apply!(y,P,x,P.compatibility_workspace)
    finally
        unlock(P.lock)
    end
end

*(P::MatrixFreeSymmetryProjector,x)=
    mul!(similar(x,promote_type(eltype(P),eltype(x)),length(P.basis)),P,x)

function _symmetry_charge_is_exact(charge)
    charge isa Number||return false
    real_part=real(charge);imaginary_part=imag(charge)
    (real_part isa Integer||real_part isa Rational)&&
        (imaginary_part isa Integer||imaginary_part isa Rational)
end

function _symmetry_projector_type(::Type{T},charge) where T
    charge in (:trivial,:stationary,:identity)&&return T
    charge isa Symbol&&throw(ArgumentError(
        "unknown symmetry charge symbol $charge"))
    charge isa Number||throw(ArgumentError(
        "a unitary symmetry charge must be numeric"))
    _symmetry_charge_is_exact(charge)&&return T
    promote_type(T,_symmetry_lapack_complex_type(typeof(charge)))
end

function _symmetry_operator_type(U)
    if U isa PIOperator
        _symmetry_lapack_complex_type(eltype(U))
    elseif U isa AbstractMatrix
        _symmetry_lapack_complex_type(eltype(U))
    else
        throw(ArgumentError("PI symmetry must be a local unitary matrix or PIOperator"))
    end
end

function _pi_unitary_blocks_at_type(
        b::PIBasis,U,::Type{T};atol,rtol,roundoff=nothing) where T
    if U isa AbstractMatrix
        _local_unitary_blocks(
            b,Matrix{T}(U);atol=atol,rtol=rtol,roundoff)
    else
        [Matrix{T}(block) for block in
            _pi_unitary_blocks(b,U;atol=atol,rtol=rtol,roundoff)]
    end
end

function _matrixfree_symmetry_projector_from_blocks(
        b::PIBasis,blocks,charge,::Type{T};
        atol,rtol,roundoff=nothing) where T
    T in (ComplexF32,ComplexF64)||throw(ArgumentError(
        "symmetry projector storage must use ComplexF32 or ComplexF64"))
    R=_real_float_type(T);absolute,relative=_symmetry_tolerances(R,atol,rtol)
    roundoff_floor=_symmetry_roundoff_floor(R,roundoff)
    q=charge in (:trivial,:stationary,:identity) ? one(T) : T(charge)
    isfinite(real(q))&&isfinite(imag(q))||throw(ArgumentError(
        "a unitary symmetry charge must be finite"))
    selection_tolerance=absolute+relative+roundoff_floor
    abs(abs(q)-one(R))<=selection_tolerance||throw(ArgumentError(
        "a unitary symmetry charge must have unit modulus"))
    vecs=Matrix{T}[];masks=BitMatrix[]
    found=false
    for block in blocks
        vals,W=_orthonormal_unitary_eigensystem(
            Matrix{T}(block);atol=absolute,rtol=relative,roundoff)
        n=length(vals);mask=falses(n,n)
        for j in 1:n,i in 1:n
            # vec column (i,j) transforms with eigenvalue vals[i]*conj(vals[j]).
            mask[i,j]=abs(vals[i]*conj(vals[j])-q)<=selection_tolerance
            found|=mask[i,j]
        end
        push!(vecs,W);push!(masks,mask)
    end
    found||throw(ArgumentError("requested charge is absent from the PI conjugation representation"))
    data=SymmetryProjectorData(b,q,vecs,masks)
    work=SymmetryProjectorWorkspace{T}(
        [(zeros(T,size(V)),zeros(T,size(V))) for V in vecs])
    MatrixFreeSymmetryProjector(data,work,ReentrantLock())
end

"""
    matrixfree_symmetry_projector(basis, U; charge=1)

Construct an orthogonal matrix-free projector onto the eigenspace with
`U*rho*U' = charge*rho`. `U` may be a local unitary or a block PI unitary.
Only sector-sized unitary matrices and Boolean masks are stored. The standard
`mul!` path serializes a reusable compatibility workspace. For parallel or
repeated explicit application, allocate one `SymmetryProjectorWorkspace` per
task and call `apply!(y, P, x, workspace)`. LAPACK-backed construction
preserves `ComplexF32` or `ComplexF64`; other scalar types are rejected with
explicit conversion guidance. Exact in-place application
`apply!(x, P, x, workspace)` is supported, but distinct partially overlapping
views are rejected.
"""
function matrixfree_symmetry_projector(b::PIBasis,U;charge=1,
                                       atol::Real=1e-12,rtol::Real=1e-10)
    T=_symmetry_projector_type(_symmetry_operator_type(U),charge)
    blocks=_pi_unitary_blocks_at_type(b,U,T;atol,rtol)
    _matrixfree_symmetry_projector_from_blocks(
        b,blocks,charge,T;atol,rtol)
end

"""Caller-owned scratch for a simultaneous weak-symmetry projector."""
struct JointSymmetryProjectorWorkspace{W,V}
    component_workspaces::W
    first::V
    second::V
end

"""
    JointSymmetryProjector

Matrix-free orthogonal projector onto the intersection of several commuting
unitary-conjugation charge sectors.  `charges` records the requested charge of
each component and `range_dimension` is the exact numerical rank certified at
construction.
"""
struct JointSymmetryProjector{B,P,C,W,K}
    basis::B
    projectors::P
    charges::C
    range_dimension::Int
    compatibility_workspace::W
    lock::K
end

Base.size(P::JointSymmetryProjector)=(length(P.basis),length(P.basis))
Base.size(P::JointSymmetryProjector,index::Integer)=index in (1,2) ?
    length(P.basis) : 1
Base.eltype(P::JointSymmetryProjector)=eltype(first(P.projectors))
Base.getproperty(P::JointSymmetryProjector,name::Symbol)=
    name===:charge ? getfield(P,:charges) : getfield(P,name)
Base.propertynames(::JointSymmetryProjector,private::Bool=false)=private ?
    (:basis,:projectors,:charges,:charge,:range_dimension,
     :compatibility_workspace,:lock) :
    (:basis,:projectors,:charges,:charge,:range_dimension)

_projector_range_dimension(P::JointSymmetryProjector,n::Integer)=P.range_dimension
_projector_charges(P::JointSymmetryProjector)=P.charges

_projector_has_only_trivial_charges(P,tolerance::Real)=
    all(charge->abs(charge-one(charge))<=tolerance,_projector_charges(P))

function JointSymmetryProjectorWorkspace(P::JointSymmetryProjector)
    T=eltype(P)
    JointSymmetryProjectorWorkspace(
        Tuple(SymmetryProjectorWorkspace(projector) for projector in P.projectors),
        zeros(T,length(P.basis)),zeros(T,length(P.basis)))
end

function _check_joint_workspace(P,work::JointSymmetryProjectorWorkspace)
    length(work.component_workspaces)==length(P.projectors)||throw(DimensionMismatch(
        "joint-symmetry workspace has the wrong component count"))
    length(work.first)==length(P.basis)&&length(work.second)==length(P.basis)||
        throw(DimensionMismatch("joint-symmetry workspace has the wrong dimension"))
    work
end

"""Apply a joint projector with caller-owned scratch."""
function apply!(destination,P::JointSymmetryProjector,source,
                work::JointSymmetryProjectorWorkspace)
    length(destination)==length(source)==length(P.basis)||throw(DimensionMismatch(
        "joint-symmetry projector vector has wrong length"))
    _check_symmetry_projector_vector_types(P,destination,source)
    _check_joint_workspace(P,work)
    if length(P.projectors)==1
        return apply!(destination,first(P.projectors),source,
                      first(work.component_workspaces))
    end
    apply!(work.first,P.projectors[1],source,work.component_workspaces[1])
    current=work.first
    for index in 2:length(P.projectors)
        target=current===work.first ? work.second : work.first
        apply!(target,P.projectors[index],current,
               work.component_workspaces[index])
        current=target
    end
    copyto!(destination,current);destination
end

function mul!(destination,P::JointSymmetryProjector,source)
    lock(P.lock)
    try
        apply!(destination,P,source,P.compatibility_workspace)
    finally
        unlock(P.lock)
    end
end
Base.:*(P::JointSymmetryProjector,source)=
    mul!(similar(source,promote_type(eltype(P),eltype(source)),
                 length(P.basis)),P,source)

function _joint_symmetry_specifications(specifications)
    single_tuple=specifications isa Tuple&&length(specifications)==2&&
        first(specifications) isa Union{AbstractMatrix,PIOperator}
    raw=specifications isa Pair||single_tuple ? (specifications,) :
        specifications isa Tuple ? specifications : Tuple(specifications)
    isempty(raw)&&throw(ArgumentError(
        "at least one unitary/charge pair is required"))
    Tuple(begin
        if specification isa Pair
            (first(specification),last(specification))
        elseif specification isa Tuple&&length(specification)==2
            specification
        else
            throw(ArgumentError(
                "each joint symmetry specification must be operator=>charge or (operator, charge)"))
        end
    end for specification in raw)
end

function _check_conjugation_commutation(
        blocks;atol,rtol,roundoff=nothing)
    for left in 1:length(blocks)-1,right in left+1:length(blocks),
        sector in eachindex(blocks[left])
        U=blocks[left][sector];V=blocks[right][sector]
        R=_real_float_type(promote_type(eltype(U),eltype(V)))
        absolute,relative=_symmetry_tolerances(R,atol,rtol)
        roundoff_floor=_symmetry_roundoff_floor(R,roundoff)
        commutator=U*V*adjoint(U)*adjoint(V);n=size(U,1)
        phase=tr(commutator)/n
        iszero(phase)&&throw(ArgumentError(
            "unitary conjugation commutator has zero phase"))
        phase/=abs(phase)
        scale=max(norm(commutator,Inf),one(real(phase)))
        norm(commutator-phase*I,Inf)<=
            absolute+(relative+roundoff_floor)*scale||throw(ArgumentError(
            "requested unitary conjugation symmetries do not commute on sector $sector"))
    end
    nothing
end

function _joint_projector_rank(
        projectors,basis,atol,rtol;roundoff=nothing)
    temporary=JointSymmetryProjector(basis,projectors,
        Tuple(P.charge for P in projectors),0,nothing,ReentrantLock())
    workspace=JointSymmetryProjectorWorkspace(temporary)
    n=length(basis);T=eltype(temporary);R=_real_float_type(T)
    absolute,relative=_symmetry_tolerances(R,atol,rtol)
    roundoff_floor=_symmetry_roundoff_floor(R,roundoff)
    source=zeros(T,n);destination=similar(source)
    projector_trace=zero(R)
    for index in 1:n
        fill!(source,zero(T));source[index]=one(T)
        apply!(destination,temporary,source,workspace)
        projector_trace+=real(destination[index])
    end
    rounded=round(Int,projector_trace)
    rank_scale=max(R(rounded),one(R))
    abs(projector_trace-R(rounded))<=
        absolute+(relative+roundoff_floor)*rank_scale||
        throw(ArgumentError(
        "failed to certify an integer joint-projector rank; trace=$projector_trace"))
    rounded>0||throw(ArgumentError(
        "the requested simultaneous symmetry-charge sector is empty"))
    rounded
end

"""
    joint_symmetry_projector(basis, specifications; atol=1e-12, rtol=1e-10)

Construct the intersection projector for several commuting weak unitary
symmetries.  Each specification is `U => charge` or `(U, charge)`, where `U`
has the same meaning as in [`matrixfree_symmetry_projector`](@ref).  Pairwise
commutation of the induced conjugations is certified sector by sector; merely
approximately compatible noncommuting projectors are rejected.

The resulting projector plugs directly into harmonic Arnoldi and the PI gap
API by passing it as `symmetry=projector`.  One explicit
[`JointSymmetryProjectorWorkspace`](@ref) is required per concurrent task.
"""
function joint_symmetry_projector(basis::PIBasis,specifications;
        atol::Real=1e-12,rtol::Real=1e-10)
    specs=_joint_symmetry_specifications(specifications)
    T=foldl(promote_type,
        (_symmetry_projector_type(
            _symmetry_operator_type(first(spec)),last(spec))
         for spec in specs))
    R=_real_float_type(T)
    source_roundoff=maximum(begin
        operator_floor=R(100)*R(eps(_real_float_type(
            _symmetry_operator_type(first(spec)))))
        charge=last(spec)
        (charge in (:trivial,:stationary,:identity)||
         _symmetry_charge_is_exact(charge)) ? operator_floor :
            max(operator_floor,R(100)*R(eps(_real_float_type(
                _symmetry_lapack_complex_type(typeof(charge))))))
    end for spec in specs)
    blocks=Tuple(_pi_unitary_blocks_at_type(
        basis,first(spec),T;atol,rtol,roundoff=source_roundoff)
        for spec in specs)
    _check_conjugation_commutation(
        blocks;atol,rtol,roundoff=source_roundoff)
    projectors=Tuple(_matrixfree_symmetry_projector_from_blocks(
        basis,blocks[index],last(specs[index]),T;
        atol,rtol,roundoff=source_roundoff)
        for index in eachindex(specs))
    rank=_joint_projector_rank(
        projectors,basis,atol,rtol;roundoff=source_roundoff)
    provisional=JointSymmetryProjector(basis,projectors,
        Tuple(P.charge for P in projectors),rank,nothing,ReentrantLock())
    workspace=JointSymmetryProjectorWorkspace(provisional)
    JointSymmetryProjector(basis,projectors,provisional.charges,rank,
                           workspace,ReentrantLock())
end

function _symmetry_residual_roundoff(::Type{R},dimension::Integer) where
        R<:AbstractFloat
    # A dense commutator contains length-dimension inner products. Retain the
    # same 100-epsilon floor used by projector construction for small systems,
    # and the standard dimension*epsilon accumulation scale when it is larger.
    dimension_roundoff=R(max(dimension,100))*eps(R)
    isfinite(dimension_roundoff)||throw(ArgumentError(
        "symmetry residual dimension is outside the finite range of $R"))
    max(_symmetry_roundoff_floor(R,nothing),dimension_roundoff)
end

function _projected_symmetry_residual(L,P;probes::Integer=3,
                                      rng=Random.MersenneTwister(0),
                                      atol::Real=1e-12,rtol::Real=1e-10,
                                      exact::Bool=L isa AbstractMatrix)
    n=size(L,1);size(L,2)==n||throw(DimensionMismatch("Liouvillian must be square"))
    size(P)==(n,n)||throw(DimensionMismatch("projector and Liouvillian dimensions differ"))
    T=eltype(P);R=_real_float_type(T)
    promote_type(T,_complex_float_type(eltype(L)))===T||throw(ArgumentError(
        "symmetry residual evaluation would narrow the Liouvillian precision; "*
        "rebuild the projector at the Liouvillian precision"))
    absolute,relative=_symmetry_tolerances(R,atol,rtol)
    roundoff_floor=_symmetry_residual_roundoff(R,n)
    pwork=P isa JointSymmetryProjector ?
        JointSymmetryProjectorWorkspace(P) : SymmetryProjectorWorkspace(P)
    if exact
        Q=Matrix{T}(undef,n,n);e=zeros(T,n)
        for j in 1:n
            fill!(e,zero(T));e[j]=one(T)
            apply!(view(Q,:,j),P,e,pwork)
        end
        residual_matrix=L*Q-Q*L
        residual=norm(residual_matrix)
        scale=max(norm(L)*norm(Q),one(_real_float_type(T)))
        tol=absolute+(relative+roundoff_floor)*scale
        return (symmetric=residual<=tol,residual,
                relative_residual=residual/scale,tolerance=tol,
                validation=:exact,probes=0)
    end
    probes>0||throw(ArgumentError("probes must be positive"))
    x=zeros(T,n);a=similar(x);b=similar(x);c=similar(x)
    worst=zero(_real_float_type(T));scale=one(_real_float_type(T))
    for _ in 1:probes
        randn!(rng,x);apply!(a,P,x,pwork);mul!(b,L,a)
        mul!(c,L,x);apply!(c,P,c,pwork)
        worst=max(worst,norm(b-c));scale=max(scale,norm(b),norm(c))
    end
    tol=absolute+(relative+roundoff_floor)*scale
    (symmetric=worst<=tol,residual=worst,relative_residual=worst/scale,
     tolerance=tol,validation=:probed,probes=Int(probes))
end

"""
    check_liouvillian_symmetry(L, U; kind=:unitary, basis=nothing,
                               atol=1e-12, rtol=1e-10)

Check a weak Liouvillian symmetry. For `kind=:unitary`, test covariance under
`rho -> U*rho*U'`. For `kind=:antiunitary`, test covariance under
`rho -> U*conj(rho)*U'`, i.e. the antiunitary `U*K` in the chosen basis.
With a `PIBasis`, a local `d×d` unitary is lifted as `U^⊗N` using Schur blocks;
a `PIOperator` may instead specify sector-dependent unitary blocks.
"""
function check_liouvillian_symmetry(x,U;kind=:unitary,basis=nothing,
                                    atol::Real=1e-12,rtol::Real=1e-10)
    kind in (:unitary,:antiunitary)||throw(ArgumentError("kind must be :unitary or :antiunitary"))
    if x isa PIModel
        basis===nothing&&(basis=x.basis);L=liouvillian(x;representation=:sparse)
    else
        L=x
    end
    M=_materialize(L);n=size(M,1);size(M,2)==n||throw(DimensionMismatch("Liouvillian must be square"))
    if basis===nothing
        D=_check_unitary_matrix(U;atol=atol,rtol=rtol);D^2==n||throw(DimensionMismatch("full-space Liouvillian dimension must equal size(U,1)^2"))
        S=sandwich_superoperator(U)
    else
        length(basis)==n||throw(DimensionMismatch("PI basis and Liouvillian dimensions differ"))
        S=_pi_conjugation_superoperator(basis,U;atol=atol,rtol=rtol)
    end
    residual_matrix = kind===:unitary ? M*S-S*M : M*S-S*conj(M)
    T=promote_type(_complex_float_type(eltype(M)),
                   _complex_float_type(eltype(S)))
    R=_real_float_type(T)
    absolute,relative=_symmetry_tolerances(R,atol,rtol)
    roundoff_floor=_symmetry_residual_roundoff(R,n)
    residual=norm(residual_matrix)
    scale=max(norm(M)*norm(S),one(R))
    tol=absolute+(relative+roundoff_floor)*scale
    (;symmetric=residual<=tol,kind,residual,relative_residual=residual/scale,
      tolerance=tol,dimension=n,representation=basis===nothing ? :full_liouville : :pi_liouville)
end

"""Return only the Boolean `symmetric` field from `check_liouvillian_symmetry`."""
is_liouvillian_symmetric(args...;kwargs...)=check_liouvillian_symmetry(args...;kwargs...).symmetric
_symmetry_candidate_pairs(x)=x isa AbstractVector{<:Pair} ? collect(x) : collect(pairs(x))
function _usual_unitary_candidates(d)
    omega=exp(2pi*im/d);clock=Diagonal(ComplexF64[omega^(j-1) for j in 1:d]);shift=zeros(ComplexF64,d,d)
    for j in 1:d;shift[mod1(j+1,d),j]=1;end
    units=Pair{Symbol,Any}[:clock_phase=>clock,:cyclic_shift=>shift]
    if d==2
        push!(units,:parity_x=>ComplexF64[0 1;1 0]);push!(units,:parity_z=>ComplexF64[1 0;0 -1])
    end
    units
end

"""
    usual_liouvillian_symmetries(model_or_L; basis=nothing,
                                 unitary_candidates=nothing,
                                 antiunitary_candidates=nothing)

Check named common weak symmetries. Defaults are the local clock/phase and
cyclic-shift unitaries plus complex conjugation; qubits additionally include
Pauli parity axes and spin time reversal `i*sigma_y*K`. Candidate collections
may be dictionaries, named tuples, or vectors of pairs.
"""
function usual_liouvillian_symmetries(x;basis=nothing,unitary_candidates=nothing,
                                      antiunitary_candidates=nothing,atol::Real=1e-12,rtol::Real=1e-10)
    x isa PIModel&&basis===nothing&&(basis=x.basis)
    basis===nothing&&throw(ArgumentError("pass basis=... when checking an already constructed PI Liouvillian"))
    d=basis.d
    if unitary_candidates===nothing
        units=_usual_unitary_candidates(d)
    else
        units=_symmetry_candidate_pairs(unitary_candidates)
    end
    if antiunitary_candidates===nothing
        antis=Pair{Symbol,Any}[:complex_conjugation=>Matrix{ComplexF64}(I,d,d)]
        d==2&&push!(antis,:spin_time_reversal=>ComplexF64[0 1;-1 0])
    else
        antis=_symmetry_candidate_pairs(antiunitary_candidates)
    end
    unitary=Dict(name=>check_liouvillian_symmetry(x,U;kind=:unitary,basis=basis,atol=atol,rtol=rtol) for (name,U) in units)
    antiunitary=Dict(name=>check_liouvillian_symmetry(x,U;kind=:antiunitary,basis=basis,atol=atol,rtol=rtol) for (name,U) in antis)
    (;unitary,antiunitary)
end
