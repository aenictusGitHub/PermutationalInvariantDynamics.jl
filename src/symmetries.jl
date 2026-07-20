function _check_unitary_matrix(U;atol,rtol)
    size(U,1)==size(U,2)||throw(DimensionMismatch("symmetry operator must be square"));n=size(U,1)
    isapprox(U'*U,I;atol=atol,rtol=rtol)||throw(ArgumentError("symmetry operator must be unitary"));n
end

"""Return a unitary eigenbasis, including inside degenerate eigenspaces."""
function _orthonormal_unitary_eigensystem(U;atol,rtol)
    _check_unitary_matrix(U;atol=atol,rtol=rtol)
    # Eigenvectors returned by a generic eigensolver need not be orthogonal
    # inside a degenerate eigenspace.  Complex Schur vectors are orthonormal by
    # construction, and a normal (in particular unitary) matrix has diagonal
    # complex Schur form up to roundoff.
    T=promote_type(eltype(U),ComplexF64)
    F=schur(Matrix{T}(U));R=Matrix(F.T);W=Matrix(F.Z);vals=collect(diag(R))
    offdiag=R-Diagonal(vals);scale=max(norm(R),1.0)
    tol=100*(atol+rtol*scale+eps(Float64)*scale)
    norm(offdiag)<=tol||throw(ArgumentError("failed to construct orthonormal eigenspaces for the unitary symmetry"))
    vals,W
end

function _local_unitary_blocks(b::PIBasis,U;atol,rtol,cache=nothing)
    _check_unitary_matrix(U;atol=atol,rtol=rtol)==b.d||throw(DimensionMismatch("local symmetry must be $(b.d)×$(b.d)"))
    vals,W=_orthonormal_unitary_eigensystem(U;atol=atol,rtol=rtol)
    H=W*Diagonal(angle.(vals))*W';H=(H+H')/2
    isapprox(exp(im*H),U;atol=10atol,rtol=10rtol)||throw(ArgumentError("failed to construct a Hermitian generator for the unitary"))
    cache===nothing&&(cache=OneBodyGeometry(b))
    [exp(im*collective_block(b,H,p;cache=cache)) for p in b.sectors]
end

function _pi_unitary_blocks(b::PIBasis,U;atol,rtol,cache=nothing)
    if U isa PIOperator
        U.basis===b||throw(ArgumentError("symmetry operator uses an incompatible PI basis"))
        blocks=[Matrix(physical_block(U,p)) for p in b.sectors]
        all(V->_check_unitary_matrix(V;atol=atol,rtol=rtol)>0,blocks)||error("unreachable")
        blocks
    elseif U isa AbstractMatrix
        _local_unitary_blocks(b,U;atol=atol,rtol=rtol,cache=cache)
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
eltype(::MatrixFreeSymmetryProjector)=ComplexF64

function _check_symmetry_projector_workspace(P,work::SymmetryProjectorWorkspace)
    length(work.work)==length(P.eigenvectors)||throw(DimensionMismatch("symmetry-projector workspace has the wrong number of sectors"))
    for s in eachindex(P.eigenvectors)
        n=size(P.eigenvectors[s],1);A,B=work.work[s]
        size(A)==(n,n)&&size(B)==(n,n)||throw(DimensionMismatch("symmetry-projector workspace has an incompatible sector block"))
    end
    work
end

"""Apply `P` using caller-owned scratch; one workspace may not be shared concurrently."""
function apply!(y,P::MatrixFreeSymmetryProjector,x,
                work::SymmetryProjectorWorkspace)
    length(x)==length(P.basis)&&length(y)==length(P.basis)||throw(DimensionMismatch("projector vector has wrong length"))
    _check_symmetry_projector_workspace(P,work)
    for s in eachindex(P.basis.sectors)
        r=P.basis.offsets[s]:P.basis.offsets[s+1]-1;n=length(P.basis.patterns[s])
        X=reshape(view(x,r),n,n);Y=reshape(view(y,r),n,n)
        V=P.eigenvectors[s];mask=P.masks[s];A,B=work.work[s]
        mul!(A,adjoint(V),X);mul!(B,A,V)
        @inbounds for i in eachindex(B);B[i]=mask[i] ? B[i] : zero(eltype(B));end
        mul!(A,V,B);mul!(Y,A,adjoint(V))
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

*(P::MatrixFreeSymmetryProjector,x)=mul!(similar(x,length(P.basis)),P,x)

"""
    matrixfree_symmetry_projector(basis, U; charge=1)

Construct an orthogonal matrix-free projector onto the eigenspace with
`U*rho*U' = charge*rho`. `U` may be a local unitary or a block PI unitary.
Only sector-sized unitary matrices and Boolean masks are stored. The standard
`mul!` path serializes a reusable compatibility workspace. For parallel or
repeated explicit application, allocate one `SymmetryProjectorWorkspace` per
task and call `apply!(y, P, x, workspace)`.
"""
function matrixfree_symmetry_projector(b::PIBasis,U;charge=1,
                                       atol::Real=1e-12,rtol::Real=1e-10)
    blocks=_pi_unitary_blocks(b,U;atol=atol,rtol=rtol)
    q=charge in (:trivial,:stationary,:identity) ? one(ComplexF64) : ComplexF64(charge)
    abs(abs(q)-1)<=atol+rtol||throw(ArgumentError("a unitary symmetry charge must have unit modulus"))
    vecs=Matrix{ComplexF64}[];masks=BitMatrix[]
    found=false
    for V in blocks
        vals,W=_orthonormal_unitary_eigensystem(V;atol=atol,rtol=rtol);n=length(vals)
        mask=falses(n,n)
        for j in 1:n,i in 1:n
            # vec column (i,j) transforms with eigenvalue vals[i]*conj(vals[j]).
            mask[i,j]=abs(vals[i]*conj(vals[j])-q)<=atol+rtol
            found|=mask[i,j]
        end
        push!(vecs,W);push!(masks,mask)
    end
    found||throw(ArgumentError("requested charge is absent from the PI conjugation representation"))
    data=SymmetryProjectorData(b,q,vecs,masks)
    work=SymmetryProjectorWorkspace{ComplexF64}([(zeros(ComplexF64,size(V)),zeros(ComplexF64,size(V))) for V in vecs])
    MatrixFreeSymmetryProjector(data,work,ReentrantLock())
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
Base.eltype(::JointSymmetryProjector)=ComplexF64
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
    JointSymmetryProjectorWorkspace(
        Tuple(SymmetryProjectorWorkspace(projector) for projector in P.projectors),
        zeros(ComplexF64,length(P.basis)),zeros(ComplexF64,length(P.basis)))
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
    mul!(similar(source,ComplexF64,length(P.basis)),P,source)

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

function _check_conjugation_commutation(blocks;atol,rtol)
    for left in 1:length(blocks)-1,right in left+1:length(blocks),
        sector in eachindex(blocks[left])
        U=blocks[left][sector];V=blocks[right][sector]
        commutator=U*V*adjoint(U)*adjoint(V);n=size(U,1)
        phase=tr(commutator)/n
        iszero(phase)&&throw(ArgumentError(
            "unitary conjugation commutator has zero phase"))
        phase/=abs(phase)
        scale=max(norm(commutator,Inf),one(real(phase)))
        norm(commutator-phase*I,Inf)<=atol+rtol*scale||throw(ArgumentError(
            "requested unitary conjugation symmetries do not commute on sector $sector"))
    end
    nothing
end

function _joint_projector_rank(projectors,basis,atol,rtol)
    temporary=JointSymmetryProjector(basis,projectors,
        Tuple(P.charge for P in projectors),0,nothing,ReentrantLock())
    workspace=JointSymmetryProjectorWorkspace(temporary)
    n=length(basis);source=zeros(ComplexF64,n);destination=similar(source)
    projector_trace=0.0
    for index in 1:n
        fill!(source,0);source[index]=1
        apply!(destination,temporary,source,workspace)
        projector_trace+=real(destination[index])
    end
    rounded=round(Int,projector_trace)
    abs(projector_trace-rounded)<=atol+rtol*max(rounded,1)||throw(ArgumentError(
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
    blocks=Tuple(_pi_unitary_blocks(basis,first(spec);atol,rtol)
                 for spec in specs)
    _check_conjugation_commutation(blocks;atol,rtol)
    projectors=Tuple(matrixfree_symmetry_projector(basis,first(spec);
        charge=last(spec),atol,rtol) for spec in specs)
    rank=_joint_projector_rank(projectors,basis,atol,rtol)
    provisional=JointSymmetryProjector(basis,projectors,
        Tuple(P.charge for P in projectors),rank,nothing,ReentrantLock())
    workspace=JointSymmetryProjectorWorkspace(provisional)
    JointSymmetryProjector(basis,projectors,provisional.charges,rank,
                           workspace,ReentrantLock())
end

function _projected_symmetry_residual(L,P;probes::Integer=3,
                                      rng=Random.MersenneTwister(0),
                                      atol::Real=1e-12,rtol::Real=1e-10,
                                      exact::Bool=L isa AbstractMatrix)
    n=size(L,1);size(L,2)==n||throw(DimensionMismatch("Liouvillian must be square"))
    size(P)==(n,n)||throw(DimensionMismatch("projector and Liouvillian dimensions differ"))
    pwork=P isa JointSymmetryProjector ?
        JointSymmetryProjectorWorkspace(P) : SymmetryProjectorWorkspace(P)
    if exact
        Q=Matrix{eltype(P)}(undef,n,n);e=zeros(eltype(P),n)
        for j in 1:n
            fill!(e,zero(eltype(e)));e[j]=one(eltype(e))
            apply!(view(Q,:,j),P,e,pwork)
        end
        R=L*Q-Q*L
        residual=norm(R);scale=max(norm(L)*norm(Q),1.0);tol=atol+rtol*scale
        return (symmetric=residual<=tol,residual,
                relative_residual=residual/scale,tolerance=tol,
                validation=:exact,probes=0)
    end
    probes>0||throw(ArgumentError("probes must be positive"))
    x=zeros(ComplexF64,n);a=similar(x);b=similar(x);c=similar(x);worst=0.0;scale=1.0
    for _ in 1:probes
        randn!(rng,x);apply!(a,P,x,pwork);mul!(b,L,a)
        mul!(c,L,x);apply!(c,P,c,pwork)
        worst=max(worst,norm(b-c));scale=max(scale,norm(b),norm(c))
    end
    tol=atol+rtol*scale
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
    residual=norm(residual_matrix);scale=max(norm(M)*norm(S),1.0);tol=atol+rtol*scale
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
