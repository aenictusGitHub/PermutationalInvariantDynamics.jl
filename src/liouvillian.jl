"""
    MatrixFreeLiouvillian(n, action!, T, trace_vector;
                          autonomous=true, plan=nothing, workspace=nothing,
                          adjoint_action! = nothing,
                          batched_action! = nothing,
                          batched_adjoint_action! = nothing)

Matrix-free `n x n` Liouvillian whose callback implements
`action!(destination, source, time, parameters)`. `T` is its scalar type and
`trace_vector` encodes the physical trace functional. A custom, plan-less
operator may additionally provide callbacks with the same four-argument
signature for its adjoint and for matrix batches. Batched callbacks receive
`destination` and `source` matrices whose columns are independent vectors.
Missing batched callbacks fall back to the corresponding vector callback;
missing adjoint callbacks retain the autonomous materialization fallback.

Compatibility calls are synchronized through one lock shared by all supplied
callbacks. Compiled PI hot loops should use `compile`, `apply!`, and one
explicit `LiouvillianWorkspace` per task.
"""
struct MatrixFreeLiouvillian{F,T,V,P,W,K,A,B,C}
    n::Int
    action!::F
    Ttype::Type{T}
    tracevec::V
    autonomous::Bool
    plan::P
    workspace::W
    lock::K
    adjoint_action!::A
    batched_action!::B
    batched_adjoint_action!::C
end

function _synchronized_liouvillian_callback(callback,callback_lock)
    callback===nothing&&return nothing
    function synchronized_callback!(destination,source,time,parameters)
        lock(callback_lock)
        try
            callback(destination,source,time,parameters)
        finally
            unlock(callback_lock)
        end
    end
end

function MatrixFreeLiouvillian(n::Integer, action!, ::Type{T}, tracevec;
                               autonomous::Bool=true,plan=nothing,
                               workspace=nothing,
                               adjoint_action! = nothing,
                               batched_action! = nothing,
                               batched_adjoint_action! = nothing) where T
    n > 0 || throw(ArgumentError("Liouvillian dimension must be positive"))
    length(tracevec) == n || throw(DimensionMismatch("trace vector has the wrong length"))
    action_lock=ReentrantLock()
    # `action!` is a compatibility surface used by older integrations.  Keep
    # it safe when one compiled Liouvillian is shared between tasks; new hot
    # loops should pass an explicit LiouvillianWorkspace to `apply!` instead.
    safe_action! = _synchronized_liouvillian_callback(action!,action_lock)
    safe_adjoint_action! = _synchronized_liouvillian_callback(
        adjoint_action!,action_lock)
    safe_batched_action! = _synchronized_liouvillian_callback(
        batched_action!,action_lock)
    safe_batched_adjoint_action! = _synchronized_liouvillian_callback(
        batched_adjoint_action!,action_lock)
    MatrixFreeLiouvillian{typeof(safe_action!),T,typeof(tracevec),
                          typeof(plan),typeof(workspace),typeof(action_lock),
                          typeof(safe_adjoint_action!),typeof(safe_batched_action!),
                          typeof(safe_batched_adjoint_action!)}(
        Int(n),safe_action!,T,tracevec,autonomous,plan,workspace,action_lock,
        safe_adjoint_action!,safe_batched_action!,safe_batched_adjoint_action!)
end

size(L::MatrixFreeLiouvillian)=(L.n,L.n); eltype(L::MatrixFreeLiouvillian)=L.Ttype
size(L::MatrixFreeLiouvillian,i::Integer)=i in (1,2) ? L.n : 1

"""Return whether a model or Liouvillian is independent of time and parameters."""
isautonomous(::AbstractMatrix) = true
isautonomous(L::MatrixFreeLiouvillian) = L.autonomous
isautonomous(model::PIModel)=all(term_isautonomous,model.terms)

function _require_autonomous(L, operation::AbstractString)
    isautonomous(L) && return L
    throw(ArgumentError("$operation requires a time-independent Liouvillian; " *
                        "call freeze(...; time=..., parameters=...) first, or use " *
                        "evolve!/dynamics_problem for explicitly time-dependent evolution"))
end

function mul!(y,L::MatrixFreeLiouvillian,x)
    _require_autonomous(L, "mul!")
    L.action!(y,x,0.0,nothing)
    y
end

# Match ordinary matrix multiplication: allocating products widen their
# destination to hold both the operator and source scalar types.  In
# particular, a Float64 compiled plan may safely act on a Float32 source;
# allocating a Float32 destination here would otherwise trip the explicit
# no-narrowing guard in `apply!`.
_product_destination(operator,source,dims...)=
    similar(source,promote_type(eltype(operator),eltype(source)),dims...)

*(L::MatrixFreeLiouvillian,x::AbstractVector)=
    mul!(_product_destination(L,x,L.n),L,x)

*(L::MatrixFreeLiouvillian,X::AbstractMatrix)=
    mul!(_product_destination(L,X,L.n,size(X,2)),L,X)

function mul!(Y::AbstractMatrix,L::MatrixFreeLiouvillian,X::AbstractMatrix)
    _require_autonomous(L,"mul!")
    size(X,1)==L.n||throw(DimensionMismatch("matrix input has the wrong leading dimension"))
    size(Y)==(L.n,size(X,2))||throw(DimensionMismatch("matrix output has the wrong dimensions"))
    if L.batched_action! !== nothing
        L.batched_action!(Y,X,0.0,nothing)
    elseif L.plan isa LiouvillianPlan
        lock(L.lock)
        try
            apply!(Y,L.plan,X,0.0,nothing,L.workspace)
        finally
            unlock(L.lock)
        end
    else
        # MatrixFreeLiouvillian is also used as an adapter for non-PI plans
        # such as HEOM.  Their vector callback is the compatibility contract;
        # only a compiled PI LiouvillianPlan implies the sectorwise batch
        # kernel below.  Other adapters retain the documented column fallback.
        for j in axes(X,2)
            L.action!(view(Y,:,j),view(X,:,j),0.0,nothing)
        end
    end
    Y
end

# Common explicit-time action used by both the fixed-step and SciML adapters.
_liouvillian_action!(y,L::AbstractMatrix,x,t,p)=mul!(y,L,x)
_liouvillian_action!(y,L::MatrixFreeLiouvillian,x,t,p)=L.action!(y,x,t,p)

function _append_sparse_block!(rows,columns,values,M,offset,scale)
    ii,jj,vv=findnz(M)
    @inbounds for index in eachindex(vv)
        value=convert(eltype(values),scale*vv[index])
        iszero(value)&&continue
        push!(rows,ii[index]+offset)
        push!(columns,jj[index]+offset)
        push!(values,value)
    end
    nothing
end

# Sparse Liouvillian construction must start from exact Schur-block support.
# Building a dense m^2-by-m^2 Kronecker product and sparsifying afterwards
# makes a simple Dicke ladder allocate O(m^4) temporary storage.  These
# helpers preserve the same column-major vec identities and exact `iszero`
# semantics while never materializing that dense intermediate.
_exact_sparse_block(matrix::SparseMatrixCSC)=matrix
_exact_sparse_block(matrix)=sparse(matrix)

function _sparse_commutator_block(K)
    S=_exact_sparse_block(K)
    result=-im*(left_superoperator(S)-right_superoperator(S))
    dropzeros!(result)
end

function _sparse_dissipator_block(K,Q)
    S=_exact_sparse_block(K);QS=_exact_sparse_block(Q)
    result=sandwich_superoperator(S)-
        (left_superoperator(QS)+right_superoperator(QS))/2
    dropzeros!(result)
end


function _block_superop(b::PIBasis,blocks,kind;qblocks=nothing)
    kind in (:commutator,:dissipator)||throw(ArgumentError(
        "block superoperator kind must be :commutator or :dissipator"))
    kind===:dissipator&&qblocks===nothing&&throw(ArgumentError(
        "dissipator block assembly requires prepared K'K blocks"))
    T=isempty(blocks) ? ComplexF64 : eltype(first(blocks))
    rows=Int[];columns=Int[];values=T[]
    for s in eachindex(b.sectors)
        K=blocks[s];offset=b.offsets[s]-1
        M=kind===:commutator ? _sparse_commutator_block(K) :
            _sparse_dissipator_block(K,qblocks[s])
        _append_sparse_block!(rows,columns,values,M,offset,one(T))
    end
    sparse(rows,columns,values,length(b),length(b))
end

function _direct_blocks(b,o::PIOperator)
    RT=_real_float_type(eltype(o.data))
    [_divide_by_schur_multiplicity_scale(Matrix(coefficient_block(o,p)),RT,p)
     for p in b.sectors]
end
_direct_blocks(b,o)=throw(ArgumentError("direct PI terms require a PIOperator, got $(typeof(o))"))

function _matrix_at(model,t,p)
    frozen=PIModel(model.basis,map(term->freeze_term(term,t,p),model.terms))
    _matrix_from_plan(LiouvillianPlan(frozen))
end

function _local_kernel_triplets(b,cache,X,Y)
    T=promote_type(Complex{geometry_scalar_type(cache)},eltype(X),eltype(Y))
    I=Int[]; J=Int[]; V=T[]
    for (li,l) in pairs(b.sectors),(ni,n) in pairs(b.sectors)
        common=cache.connections[(li,ni)]
        isempty(common)&&continue
        nl=length(b.patterns[li]); nn=length(b.patterns[ni])
        for a in 1:nl,bb in 1:nl,c in 1:nn,d in 1:nn
            z=local_kernel_element(cache,X,Y,l,a,bb,n,c,d); iszero(z)&&continue
            push!(I,b.offsets[li]+a-1+(bb-1)*nl)
            push!(J,b.offsets[ni]+c-1+(d-1)*nn); push!(V,z)
        end
    end
    (;I,J,V)
end

# For one common child `mu`, the local gain from input sector `nu` to output
# sector `lambda` factorizes as
#
#     X_nu -> r_(lambda,mu,nu) C_(lambda,mu,nu) X_nu C'_(lambda,mu,nu),
#
# where every entry of the rectangular C matrix is one contraction of the
# evaluated local operator with the read-only one-box geometry.  Retaining
# these branches avoids the previous quartic table of all `(a,b,c,d)` PI
# coordinates.  It also preserves interference between arbitrary entries of
# the local operator inside each C entry, including entries that are zero in
# the schedule prototype and become nonzero later.
struct _LocalGainBranch{T}
    output_sector::Int
    input_sector::Int
    scale::T
    table::_PackedOneBodyContractions{T}
end

struct _LocalGainBranches{T}
    entries::Vector{_LocalGainBranch{T}}
    maximum_block_dimension::Int
end

struct _StaticLocalGainBranch{T}
    output_sector::Int
    input_sector::Int
    scale::T
end
struct _StaticLocalGainBranches{T}
    entries::Vector{_StaticLocalGainBranch{T}}
    maximum_block_dimension::Int
end

# Fixed one-body channels often inherit exact selection-rule zeros from their
# local operator.  For example, every Schur contraction of a qubit lowering
# operator contains only one shifted diagonal.  Retain an exact support only
# when a setup-time arithmetic estimate predicts less work than the dense
# rectangular sandwich.  The dense matrix remains the canonical factor for
# explicit materialization and for consumers which need arbitrary indexing;
# the support is only O(nnz(C)) and never becomes a quartic gain map.
struct _StaticOneBodyContraction{T,M<:AbstractMatrix{T}} <: AbstractMatrix{T}
    matrix::M
    output_rows::Vector{Int}
    input_columns::Vector{Int}
    values::Vector{T}
    use_support::Bool
end

Base.size(contraction::_StaticOneBodyContraction)=size(contraction.matrix)
Base.axes(contraction::_StaticOneBodyContraction)=axes(contraction.matrix)
Base.IndexStyle(::Type{<:_StaticOneBodyContraction})=IndexCartesian()
@inline Base.getindex(contraction::_StaticOneBodyContraction,row::Int,
                      column::Int)=contraction.matrix[row,column]

function _static_onebody_contraction(matrix::M) where
        {T,M<:AbstractMatrix{T}}
    output_rows=Int[];input_columns=Int[];values=T[]
    @inbounds for column in axes(matrix,2),row in axes(matrix,1)
        value=matrix[row,column]
        iszero(value)&&continue
        push!(output_rows,row);push!(input_columns,column);push!(values,value)
    end
    nonzeros=length(values)
    output_dimension,input_dimension=size(matrix)
    # A support pair evaluates one scalar contribution.  Give that loop a
    # factor-two penalty relative to the usual two dense matrix products so
    # borderline/dense contractions continue to use the tuned dense path.
    sparse_cost=UInt128(2)*UInt128(nonzeros)*UInt128(nonzeros)
    dense_cost=UInt128(output_dimension)*UInt128(input_dimension)*
        UInt128(output_dimension+input_dimension)
    use_support=sparse_cost<=dense_cost
    if !use_support
        empty!(output_rows);empty!(input_columns);empty!(values)
    end
    _StaticOneBodyContraction(matrix,output_rows,input_columns,values,
                              use_support)
end

# Julia 1.10 does not always preserve the concrete wrapper element type
# through `map(_static_onebody_contraction, contractions)`.  Construct the
# vector explicitly so a fixed local channel remains fully inferred on every
# supported Julia release.
function _static_onebody_contractions(contractions::AbstractVector{M}) where
        {T,M<:AbstractMatrix{T}}
    prepared=Vector{_StaticOneBodyContraction{T,M}}(
        undef,length(contractions))
    @inbounds for index in eachindex(contractions)
        prepared[index]=_static_onebody_contraction(contractions[index])
    end
    prepared
end

function _static_local_gain_branches(branches::_LocalGainBranches{T}) where T
    entries=_StaticLocalGainBranch{T}[
        _StaticLocalGainBranch(branch.output_sector,branch.input_sector,
                               branch.scale)
        for branch in branches.entries]
    _StaticLocalGainBranches(entries,branches.maximum_block_dimension)
end

function _local_gain_branches(b,cache::OneBodyGeometry{T}) where T
    entries=_LocalGainBranch{T}[]
    for li in eachindex(b.sectors),ni in eachindex(b.sectors)
        for mu in cache.connections[(li,ni)]
            key=(li,mu,ni)
            push!(entries,_LocalGainBranch(
                li,ni,cache.scales[key],cache.contractions[key]))
        end
    end
    _LocalGainBranches(entries,maximum(length,b.patterns;init=1))
end

_exact_real(x)=x isa Integer||x isa Rational

# Divide two exact real values without converting either potentially enormous
# operand separately. Ordinary floating rates retain the direct converted
# division path below; this setup-only fallback is for Integer/Rational data
# whose exact quotient may be representable even when either operand is not.
function _checked_exact_rate_quotient(rate,c,::Type{R}) where R<:AbstractFloat
    iszero(c)&&throw(ArgumentError("Hamiltonian hbar must be nonzero"))
    # Preserve the allocation-free common path for small exactly representable
    # integers (in particular the default rate=hbar=1). Only fall back to
    # BigInt rational cancellation when converting an operand first could lose
    # information or leave the finite range.
    converted_rate=try
        R(rate)
    catch
        nothing
    end
    converted_hbar=try
        R(c)
    catch
        nothing
    end
    if converted_rate!==nothing&&converted_hbar!==nothing&&
       isfinite(converted_rate)&&isfinite(converted_hbar)&&
       converted_rate==rate&&converted_hbar==c
        quotient=converted_rate/converted_hbar
        if isfinite(quotient)&&(iszero(converted_rate)||!iszero(quotient))
            return quotient
        end
    end
    numerator_value=Rational{BigInt}(rate)
    denominator_value=Rational{BigInt}(c)
    quotient=numerator_value/denominator_value
    magnitude=_checked_exact_ratio(R,abs(numerator(quotient)),denominator(quotient);
        context="Hamiltonian rate/hbar quotient")
    quotient<0 ? -magnitude : magnitude
end

function _scaled_rate(rate,c,::Type{R}) where R<:AbstractFloat
    iszero(c)&&throw(ArgumentError("Hamiltonian hbar must be nonzero"))
    if rate isa Number
        return _exact_real(rate)&&_exact_real(c) ?
            _checked_exact_rate_quotient(rate,c,R) : R(rate)/R(c)
    end
    (t,p)->begin
        evaluated=value_at(rate,t,p)
        _exact_real(evaluated)&&_exact_real(c) ?
            _checked_exact_rate_quotient(evaluated,c,R) : R(evaluated)/R(c)
    end
end
abstract type AbstractStaticPIKernel end
struct HamiltonianPIKernel{B,S} <: AbstractStaticPIKernel
    blocks::B; scale::S
end
struct DissipatorPIKernel{B,Q,S} <: AbstractStaticPIKernel
    blocks::B; qblocks::Q; scale::S
end
struct LocalJumpPIKernel{Q,G,S} <: AbstractStaticPIKernel
    qblocks::Q; gain::G; scale::S
end

# Fixed local channels use the same rectangular Schur contractions as the
# preallocated operator-schedule path.  Keeping the contractions themselves,
# rather than their four-index PI-coordinate expansion, makes retained setup
# storage quadratic in the Schur-block dimensions.  Sparse materialization
# expands these factors only when it is explicitly requested.
struct FactorizedLocalJumpPIKernel{Q,B,C,S} <: AbstractStaticPIKernel
    qblocks::Q
    branches::B
    contractions::C
    scale::S
end

struct FactorizedLocalPBodyJumpPIKernel{Q,G,C,P,S} <: AbstractStaticPIKernel
    qblocks::Q
    groups::G
    contractions::C
    pair_scales::P
    scale::S
end

# Autonomous fixed kernels can share their diagonal Schur work.  Hamiltonian
# blocks and the complete anticommutator loss are accumulated once at setup;
# gain maps stay as independent channels so no spurious cross terms are
# introduced.  The small gain records expose their output/input sector
# indices through `branches` or `groups`, which also permits deterministic
# target-sector parallel application.
struct _FusedCollectiveGain{B,S}
    blocks::B
    scale::S
end
struct _FusedOneBodyGain{B,C,S}
    branches::B
    contractions::C
    scale::S
end
struct _FusedPBodyGain{G,C,P,S}
    groups::G
    contractions::C
    pair_scales::P
    scale::S
end
struct FusedStaticPIKernel{H,Q,C,O,P} <: AbstractStaticPIKernel
    hamiltonian_blocks::H
    loss_blocks::Q
    collective_gains::C
    onebody_gains::O
    pbody_gains::P
end


# In-place schedules keep all mutable evaluated data outside the plan. The
# builders below are read-only handles to prepared representation geometry.
abstract type AbstractDynamicPIKernel end
struct CollectiveOneBodyBlockBuilder{T,D,L,B<:PIBasis{D,L}}
    geometry::Union{OneBodyGeometry{T,D,L,B},
                    _SymmetricCollectiveGeometry{T,D,L,B}}
end
CollectiveOneBodyBlockBuilder(
    geometry::OneBodyGeometry{T,D,L,B}) where {T,D,L,B}=
    CollectiveOneBodyBlockBuilder{T,D,L,B}(geometry)
CollectiveOneBodyBlockBuilder(
    geometry::_SymmetricCollectiveGeometry{T,D,L,B}) where {T,D,L,B}=
    CollectiveOneBodyBlockBuilder{T,D,L,B}(geometry)
struct CollectivePBodyBlockBuilder{G,P,E,O}
    geometry::G
    permutations::P
    block_entries::E
    operation_counts::O
    cancellation_risk::Bool
end
struct DirectPIBlockBuilder{B,S}
    basis::B
    inverse_scales::S
end
struct InPlaceHamiltonianPIKernel{S,B,R} <: AbstractDynamicPIKernel
    schedule::S;builder::B;scale::R
end
struct InPlaceDissipatorPIKernel{S,B,R} <: AbstractDynamicPIKernel
    schedule::S;builder::B;scale::R
end
struct InPlaceLocalJumpPIKernel{S,G,R,B} <: AbstractDynamicPIKernel
    schedule::S;geometry::G;scale::R
    branches::B
end
struct InPlaceLocalPBodyJumpPIKernel{S,B,R,G,L,U,Q} <: AbstractDynamicPIKernel
    schedule::S;builder::B;scale::R
    groups::G
    left_isometries::L;right_isometries::U;pair_scales::Q
end
struct InPlaceCorrelatedCollectiveJumpPIKernel{S,B,R,A,Q} <: AbstractDynamicPIKernel
    schedule::S
    channel_blocks::B
    scale::R
    atol::A
    rtol::Q
end
struct InPlaceCorrelatedLocalJumpPIKernel{S,O,G,R,B,A,Q} <: AbstractDynamicPIKernel
    schedule::S
    operators::O
    geometry::G
    scale::R
    branches::B
    atol::A
    rtol::Q
end

struct InPlaceHamiltonianKernelWorkspace{O,B,C}
    operator::O;blocks::B;cancellation_scratch::C
end
struct InPlaceDissipatorKernelWorkspace{O,B,Q,C}
    operator::O;blocks::B;qblocks::Q;cancellation_scratch::C
end
struct InPlaceLocalJumpKernelWorkspace{O,Q,B,C,S}
    operator::O
    qoperator::Q
    qblocks::B
    contractions::C
    gain_scratch::S
end
struct InPlaceLocalPBodyJumpKernelWorkspace{O,Q,B,C,S,A}
    operator::O;qoperator::Q;qblocks::B;contractions::C;gain_scratch::S
    cancellation_scratch::A
end
struct StaticFactorizedGainKernelWorkspace{S}
    gain_scratch::S
end
struct InPlaceCorrelatedCollectiveJumpKernelWorkspace{K,F,R,B,Q,S,N}
    kossakowski::K
    factor::F
    residual::R
    effective_blocks::B
    qblocks::Q
    qscratch::S
    rank::N
end
struct InPlaceCorrelatedLocalJumpKernelWorkspace{K,F,R,O,Q,S,B,C,G,N}
    kossakowski::K
    factor::F
    residual::R
    effective_operators::O
    qoperator::Q
    qscratch::S
    qblocks::B
    contractions::C
    gain_scratch::G
    rank::N
end

function _evaluated_dissipative_rate(rate,t,p)
    evaluated=value_at(rate,t,p)
    evaluated isa Real||throw(ArgumentError(
        "a dissipative rate must evaluate to a real number, got $(typeof(evaluated))"))
    isfinite(evaluated)||throw(ArgumentError(
        "a dissipative rate must evaluate to a finite number"))
    evaluated
end

function _apply_kernel!(y,x,ker::HamiltonianPIKernel,b,t,p,work)
    scale=convert(eltype(work[1][1]),value_at(ker.scale,t,p))
    for s in eachindex(b.sectors)
        n=length(b.patterns[s]);off=b.offsets[s];A,B,X=work[s];K=ker.blocks[s]
        copyto!(X,1,x,off,n*n)
        mul!(A,K,X); mul!(B,X,K)
        @inbounds for j in eachindex(A);y[off+j-1]+=(-1im*scale)*(A[j]-B[j]);end
    end
end
function _apply_kernel!(y,x,ker::DissipatorPIKernel,b,t,p,work)
    scale=convert(eltype(work[1][1]),_evaluated_dissipative_rate(ker.scale,t,p))
    for s in eachindex(b.sectors)
        n=length(b.patterns[s]);off=b.offsets[s];A,B,X=work[s]
        copyto!(X,1,x,off,n*n)
        K=ker.blocks[s]; Q=ker.qblocks[s]; mul!(A,K,X); mul!(B,A,adjoint(K))
        @inbounds for j in eachindex(B);y[off+j-1]+=scale*B[j];end
        mul!(A,Q,X); mul!(B,X,Q)
        @inbounds for j in eachindex(A);y[off+j-1]-=(scale/2)*(A[j]+B[j]);end
    end
end
function _apply_kernel!(y,x,ker::LocalJumpPIKernel,b,t,p,work)
    scale=convert(eltype(work[1][1]),_evaluated_dissipative_rate(ker.scale,t,p))
    @inbounds for q in eachindex(ker.gain.V)
        y[ker.gain.I[q]] += scale*ker.gain.V[q]*x[ker.gain.J[q]]
    end
    for s in eachindex(b.sectors)
        n=length(b.patterns[s]);off=b.offsets[s];A,B,X=work[s];Q=ker.qblocks[s]
        copyto!(X,1,x,off,n*n)
        mul!(A,Q,X); mul!(B,X,Q)
        @inbounds for j in eachindex(A);y[off+j-1]-=(scale/2)*(A[j]+B[j]);end
    end
end

function _apply_adjoint_kernel!(y,x,ker::HamiltonianPIKernel,b,t,p,work)
    scale=conj(convert(eltype(work[1][1]),value_at(ker.scale,t,p)))
    for s in eachindex(b.sectors)
        n=length(b.patterns[s]);off=b.offsets[s];A,B,X=work[s];K=ker.blocks[s]
        copyto!(X,1,x,off,n*n)
        mul!(A,adjoint(K),X);mul!(B,X,adjoint(K))
        @inbounds for j in eachindex(A);y[off+j-1]+=(1im*scale)*(A[j]-B[j]);end
    end
end

function _apply_adjoint_kernel!(y,x,ker::DissipatorPIKernel,b,t,p,work)
    scale=conj(convert(eltype(work[1][1]),_evaluated_dissipative_rate(ker.scale,t,p)))
    for s in eachindex(b.sectors)
        n=length(b.patterns[s]);off=b.offsets[s];A,B,X=work[s]
        copyto!(X,1,x,off,n*n)
        K=ker.blocks[s];Q=ker.qblocks[s]
        mul!(A,adjoint(K),X);mul!(B,A,K)
        @inbounds for j in eachindex(B);y[off+j-1]+=scale*B[j];end
        mul!(A,adjoint(Q),X);mul!(B,X,adjoint(Q))
        @inbounds for j in eachindex(A);y[off+j-1]-=(scale/2)*(A[j]+B[j]);end
    end
end

function _apply_adjoint_kernel!(y,x,ker::LocalJumpPIKernel,b,t,p,work)
    scale=conj(convert(eltype(work[1][1]),_evaluated_dissipative_rate(ker.scale,t,p)))
    @inbounds for q in eachindex(ker.gain.V)
        y[ker.gain.J[q]]+=scale*conj(ker.gain.V[q])*x[ker.gain.I[q]]
    end
    for s in eachindex(b.sectors)
        n=length(b.patterns[s]);off=b.offsets[s];A,B,X=work[s];Q=ker.qblocks[s]
        copyto!(X,1,x,off,n*n)
        mul!(A,adjoint(Q),X);mul!(B,X,adjoint(Q))
        @inbounds for j in eachindex(A);y[off+j-1]-=(scale/2)*(A[j]+B[j]);end
    end
end

"""Read-only caches supplied to `compile_term` implementations."""
struct TermCompileContext{B,G,P,C,T}
    basis::B
    onebody::G
    pbody::P
    coefficient_cache::C
    geometry_type::Type{T}
end

# Preserve the established internal/custom-term testing constructor.  The
# additional field is setup-only coefficient reuse; omitting it must retain
# the former uncached semantics.
TermCompileContext(basis,onebody,pbody,geometry_type::Type)=
    TermCompileContext(basis,onebody,pbody,nothing,geometry_type)

_operator_real_type(::Function)=Float64
_operator_real_type(operator::InPlaceTimeOperator)=
    _operator_real_type(operator.prototype)
_operator_real_type(operator::AbstractPIOperator)=_real_float_type(eltype(operator.data))
_operator_real_type(operator::AbstractArray)=_real_float_type(eltype(operator))
_operator_real_type(operators::Tuple)=isempty(operators) ? Float64 :
    foldl(promote_type,map(_operator_real_type,operators))
_operator_real_type(operator)=Float64

_term_geometry_type(term::AbstractPITerm)=_operator_real_type(term_operator(term))
function _term_geometry_type(term::_CorrelatedOneBodyJumps)
    operator_type=_operator_real_type(term.operators)
    prototype=_operator_prototype(term.kossakowski)
    matrix_type=prototype isa AbstractMatrix ? _operator_real_type(prototype) :
                                               operator_type
    result=promote_type(operator_type,matrix_type)
    result=_promote_correlated_tolerance_type(result,term.atol)
    _promote_correlated_tolerance_type(result,term.rtol)
end

function _model_geometry_type(model::PIModel)
    types=map(_term_geometry_type,model.terms)
    isempty(types) ? Float64 : foldl(promote_type,types)
end

function _model_onebox_requirements(model::PIModel,
        ::Type{T}=_model_geometry_type(model)) where T<:AbstractFloat
    symmetric_collective_available=
        _has_single_fully_symmetric_sector(model.basis)&&
        !_needs_wide_collective(model.basis,T)
    needs_full_onebody=any(model.terms) do term
        _term_requires_onebody_geometry(term)&&
            !_term_supports_diagonal_collective_geometry(term)
    end
    has_collective_onebody=any(
        _term_supports_diagonal_collective_geometry,model.terms)
    uses_symmetric_collective=symmetric_collective_available&&
        !needs_full_onebody&&has_collective_onebody
    uses_diagonal_onebody=!symmetric_collective_available&&
        !needs_full_onebody&&has_collective_onebody
    needs_onebody=needs_full_onebody||uses_diagonal_onebody
    pbody_orders=unique(Int[body_order(term) for term in model.terms
                            if term isa _PBodyPITerm])
    required_depth=max(needs_onebody ? 1 : 0,maximum(pbody_orders;init=0))
    geometry_families=(needs_onebody ? 1 : 0)+length(pbody_orders)
    (;needs_onebody,needs_full_onebody,uses_diagonal_onebody,
      uses_symmetric_collective,pbody_orders,required_depth,
      geometry_families)
end

function _small_onebox_autocache(b::PIBasis)
    (b.d==1&&b.N<=64)||(b.d==2&&b.N<=32)||
        (b.d==3&&b.N<=8)||(b.d==4&&b.N<=5)
end

function _compile_coefficient_cache(model::PIModel,::Type{T},requirements,
                                    supplied) where T<:AbstractFloat
    if supplied!==nothing
        _check_onebox_coefficient_cache(
            supplied,model.basis,requirements.required_depth,T)
        return supplied
    end
    requirements.geometry_families>1||return nothing
    _small_onebox_autocache(model.basis)||return nothing
    # This is construction scratch shared only while lowering the model.  The
    # finished geometries retain their packed contractions/isometries, not the
    # primitive table.  The strict cap keeps the automatic path confined to
    # the intended small-(N,d) regime; larger studies can prepare and pass an
    # explicit cache with a user-selected memory budget.
    OneBoxCGCache(model.basis;max_depth=requirements.required_depth,T,
                  memory_budget=64*1024^2)
end

function TermCompileContext(model::PIModel{B};coefficient_cache=nothing) where
        {D,L,B<:PIBasis{D,L}}
    b=model.basis;T=_model_geometry_type(model)
    requirements=_model_onebox_requirements(model,T)
    coefficients=_compile_coefficient_cache(
        model,T,requirements,coefficient_cache)
    onebody=if requirements.needs_full_onebody
        OneBodyGeometry(b,T;coefficient_cache=coefficients)
    elseif requirements.uses_symmetric_collective
        _SymmetricCollectiveGeometry(b,T)
    elseif requirements.uses_diagonal_onebody
        _diagonal_onebody_geometry(
            b,T;coefficient_cache=coefficients)
    else
        nothing
    end
    pbody=Dict{Int,PBodyGeometry{T,D,L,B}}()
    TermCompileContext(b,onebody,pbody,coefficients,T)
end

# Direct-PI and Appendix-D terms have their own lowering geometry.  Avoid the
# much larger one-box setup for models containing only those built-ins.  A
# custom term remains conservative because its delegated lowering is unknown.
_term_requires_onebody_geometry(::AbstractPITerm)=true
_term_requires_onebody_geometry(::Union{DirectPIHamiltonian,DirectPIJump,
    PBodyHamiltonian,LocalPBodyJump,CollectivePBodyJump})=false

# These built-ins only need diagonal Schur blocks of a collective one-body
# lift.  On the sole fully symmetric irrep they can use occupation-number
# geometry.  A custom term remains conservative because its delegated
# `compile_term` implementation may also require local sector-changing gains.
_term_supports_symmetric_collective_geometry(::AbstractPITerm)=false
_term_supports_symmetric_collective_geometry(::Union{LocalHamiltonian,
    CollectiveHamiltonian,CollectiveJump,CorrelatedCollectiveJumps})=true
_term_supports_diagonal_collective_geometry(term::AbstractPITerm)=
    _term_supports_symmetric_collective_geometry(term)

function _pbody_geometry!(context::TermCompileContext,order::Integer)
    get!(()->PBodyGeometry(context.basis,order,context.geometry_type;
                          coefficient_cache=context.coefficient_cache),
         context.pbody,Int(order))
end

function _pbody_block_builder(context::TermCompileContext,term)
    geometry=_pbody_geometry!(context,body_order(term))
    prototype=_operator_prototype(term_operator(term))
    _check_pbody_operator(geometry,prototype)
    permutations=[_tensor_swap_permutation(geometry.p,geometry.basis.d,k)
                  for k in 1:geometry.p-1]
    block_entries=map(geometry.basis.sectors) do sector
        [(geometry.isometries[Tuple(path)],begin
              exact_scale=geometry.path_weights[Tuple(path)]
              _prepare_exact_scale(context.geometry_type,numerator(exact_scale),
                  denominator(exact_scale),Val(false);
                  context="collective p-body path contribution")
          end) for path in geometry.paths[sector]]
    end
    cancellation_risk=any(entries->any(entry->!entry[2].direct,entries),block_entries)||
        any(entries->any(entry->begin
                scale=entry[2]
                scale.numerator//scale.denominator>
                    Rational{BigInt}(inv(sqrt(eps(context.geometry_type))))
            end,entries),block_entries)
    operation_counts=map(block_entries) do entries
        sum(entries;init=big(1)) do entry
            U=entry[1]
            BigInt(size(U,2))*BigInt(size(U,3))^2+1
        end
    end
    CollectivePBodyBlockBuilder(
        geometry,permutations,block_entries,operation_counts,
        cancellation_risk)
end

function _check_dynamic_pbody_operator(builder::CollectivePBodyBlockBuilder,X)
    geometry=builder.geometry;dp=geometry.basis.d^geometry.p
    size(X)==(dp,dp)||throw(DimensionMismatch(
        "p-body operator must be $dp×$dp"))
    R=_real_float_type(eltype(X));tolerance=R(1e-10)*max(norm(X,Inf),one(R))
    for permutation in builder.permutations
        maxdiff=zero(R)
        @inbounds for column in 1:dp,row in 1:dp
            maxdiff=max(maxdiff,abs(X[row,column]-
                X[permutation[row],permutation[column]]))
        end
        maxdiff<=tolerance||throw(ArgumentError(
            "an in-place p-body callback produced an operator that is not permutation invariant"))
    end
    nothing
end
_check_dynamic_pbody_operator(builder,X)=nothing

function _dynamic_pbody_block_uncertified(value,absolute_sum,
        operation_count::Integer)
    R=_real_float_type(typeof(value))
    (!isfinite(value)||!isfinite(absolute_sum))&&return true
    iszero(absolute_sum)&&return false
    absolute_sum>R(8)*abs(value)&&return true
    forward_bound=R(64)*R(max(operation_count,1))*eps(R)*absolute_sum
    spacing=iszero(value) ? nextfloat(zero(R)) : eps(abs(value))
    forward_bound>=spacing/R(4)
end

function _fill_dynamic_blocks!(
        blocks,builder::CollectivePBodyBlockBuilder,X,
        cancellation_scratch)
    geometry=builder.geometry;b=geometry.basis
    for s in eachindex(b.sectors)
        K=blocks[s];fill!(K,zero(eltype(K)))
        if builder.cancellation_risk
            size(cancellation_scratch,1)>=size(K,1)&&
                size(cancellation_scratch,2)>=size(K,2)||
                throw(DimensionMismatch(
                    "dynamic p-body cancellation scratch is too small"))
            @inbounds for column in axes(K,2),row in axes(K,1)
                cancellation_scratch[row,column]=
                    zero(eltype(cancellation_scratch))
            end
            for (U,scale) in builder.block_entries[s]
                _accumulate_dynamic_pbody_path_certified!(
                    K,cancellation_scratch,U,X,scale)
            end
            operation_count=builder.operation_counts[s]
            for column in axes(K,2),row in axes(K,1)
                _dynamic_pbody_block_uncertified(
                    K[row,column],cancellation_scratch[row,column],
                    operation_count)&&
                    throw(ArgumentError(
                        "the evaluated dynamic p-body block cannot certify its " *
                        "working-precision result through large-N path accumulation; " *
                        "use a wider InPlaceTimeOperator prototype scalar type"))
            end
        else
            for (U,scale) in builder.block_entries[s]
                _accumulate_dynamic_pbody_path!(K,U,X,scale)
            end
        end
    end
    blocks
end

function _fill_dynamic_blocks!(
        blocks,builder::CollectivePBodyBlockBuilder,X)
    builder.cancellation_risk&&throw(ArgumentError(
        "a cancellation-risk dynamic p-body builder requires preallocated "*
        "certification scratch"))
    _fill_dynamic_blocks!(blocks,builder,X,nothing)
end

function _accumulate_dynamic_pbody_path!(
        destination,U::_PackedPathIsometry,X,scale)
    rows,centers,local_dimension=size(U)
    size(destination)==(rows,rows)||throw(DimensionMismatch(
        "dynamic p-body destination has the wrong dimensions"))
    support=U.data
    @inbounds for centre in 1:centers,j in 1:local_dimension
        right_column=centre+(j-1)*centers
        for right_pointer in nzrange(support,right_column)
            output_column=support.rowval[right_pointer]
            right_value=support.nzval[right_pointer]
            for i in 1:local_dimension
                x=X[i,j]
                iszero(x)&&continue
                left_column=centre+(i-1)*centers
                for left_pointer in nzrange(support,left_column)
                    output_row=support.rowval[left_pointer]
                    primitive=support.nzval[left_pointer]*right_value*x
                    contribution=scale.direct ? scale.factor*primitive :
                        _apply_prepared_exact_scale(
                            primitive,scale;
                            context="dynamic collective p-body path contribution")
                    destination[output_row,output_column]+=contribution
                end
            end
        end
    end
    destination
end

function _accumulate_dynamic_pbody_path_certified!(
        destination,absolute_sum,U::_PackedPathIsometry,X,scale)
    rows,centers,local_dimension=size(U)
    size(destination)==(rows,rows)||throw(DimensionMismatch(
        "dynamic p-body destination has the wrong dimensions"))
    size(absolute_sum,1)>=rows&&size(absolute_sum,2)>=rows||
        throw(DimensionMismatch(
            "dynamic p-body certification scratch is too small"))
    support=U.data
    @inbounds for centre in 1:centers,j in 1:local_dimension
        right_column=centre+(j-1)*centers
        for right_pointer in nzrange(support,right_column)
            output_column=support.rowval[right_pointer]
            right_value=support.nzval[right_pointer]
            for i in 1:local_dimension
                x=X[i,j]
                iszero(x)&&continue
                left_column=centre+(i-1)*centers
                for left_pointer in nzrange(support,left_column)
                    output_row=support.rowval[left_pointer]
                    primitive=support.nzval[left_pointer]*right_value*x
                    contribution=scale.direct ? scale.factor*primitive :
                        _apply_prepared_exact_scale(
                            primitive,scale;
                            context="dynamic collective p-body path contribution")
                    destination[output_row,output_column]+=contribution
                    absolute_sum[output_row,output_column]+=
                        abs(contribution)
                end
            end
        end
    end
    destination
end

function _path_contractions!(destination,UL,UR,X)
    fill!(destination,zero(eltype(destination)))
    left_rows,centers,local_dimension=size(UL)
    right_rows,centers_right,local_dimension_right=size(UR)
    centers==centers_right&&local_dimension==local_dimension_right||
        throw(DimensionMismatch("incompatible p-body path isometries"))
    size(destination)==(left_rows,right_rows)||throw(DimensionMismatch(
        "p-body contraction destination has the wrong dimensions"))
    @inbounds for column in 1:right_rows,row in 1:left_rows,w in 1:centers,
                  j in 1:local_dimension,i in 1:local_dimension
        destination[row,column]+=UL[row,w,i]*UR[column,w,j]*X[i,j]
    end
    destination
end

function _path_contractions!(destination,
        UL::_PackedPathIsometry,UR::_PackedPathIsometry,X)
    fill!(destination,zero(eltype(destination)))
    left_rows,centers,local_dimension=size(UL)
    right_rows,centers_right,local_dimension_right=size(UR)
    centers==centers_right&&local_dimension==local_dimension_right||
        throw(DimensionMismatch("incompatible p-body path isometries"))
    size(destination)==(left_rows,right_rows)||throw(DimensionMismatch(
        "p-body contraction destination has the wrong dimensions"))
    left=UL.data
    right=UR.data
    @inbounds for centre in 1:centers,j in 1:local_dimension
        right_column=centre+(j-1)*centers
        for right_pointer in nzrange(right,right_column)
            output_column=right.rowval[right_pointer]
            right_value=right.nzval[right_pointer]
            for i in 1:local_dimension
                x=X[i,j]
                iszero(x)&&continue
                left_column=centre+(i-1)*centers
                for left_pointer in nzrange(left,left_column)
                    output_row=left.rowval[left_pointer]
                    destination[output_row,output_column]+=
                        left.nzval[left_pointer]*right_value*x
                end
            end
        end
    end
    destination
end

function _pbody_gain_factorization_data(builder::CollectivePBodyBlockBuilder)
    geometry=builder.geometry;b=geometry.basis;T=geometry_scalar_type(geometry)
    groups=NTuple{4,Int}[]
    I=eltype(values(geometry.isometries))
    left_isometries=I[]
    right_isometries=I[]
    pair_scales=_PreparedExactScale{T,true}[]
    for (li,left_sector) in pairs(b.sectors),(ni,right_sector) in pairs(b.sectors)
        first_pair=length(pair_scales)+1
        for left_path in geometry.paths[left_sector],right_path in geometry.paths[right_sector]
            left_path[1]==right_path[1]||continue
            push!(left_isometries,geometry.isometries[Tuple(left_path)])
            push!(right_isometries,geometry.isometries[Tuple(right_path)])
            scale_squared=geometry.path_weights[Tuple(left_path)]*
                geometry.path_weights[Tuple(right_path)]
            push!(pair_scales,_prepare_exact_scale(T,numerator(scale_squared),
                denominator(scale_squared),Val(true);
                context="local p-body path-pair contribution"))
        end
        last_pair=length(pair_scales)
        first_pair<=last_pair||continue
        push!(groups,(li,ni,first_pair,last_pair))
    end
    (;groups,left_isometries,right_isometries,pair_scales)
end

function _pbody_gain_factorization(builder::CollectivePBodyBlockBuilder)
    geometry=builder.geometry;T=geometry_scalar_type(geometry)
    builder.cancellation_risk&&throw(ArgumentError(
        "dynamic local p-body gain factors are cancellation-prone at scalar " *
        "type $T, including when every individual factor is representable; " *
        "use a wider InPlaceTimeOperator prototype scalar type"))
    result=_pbody_gain_factorization_data(builder)
    pair_scales=result.pair_scales
    any(scale->!scale.direct,pair_scales)&&throw(ArgumentError(
        "dynamic local p-body gain factors exceed the nonzero finite range of $T; " *
        "use a wider InPlaceTimeOperator prototype scalar type so the preallocated " *
        "quadratic contraction scratch cannot underflow before exact rescaling"))
    result
end

function _fill_dynamic_blocks!(blocks,builder::CollectiveOneBodyBlockBuilder,X)
    # Keep the builder's type independent of the runtime sector selection so
    # plan construction remains inferable, then cross a function barrier that
    # specializes on the concrete geometry.  Accessing the Union-valued field
    # throughout this loop otherwise boxes it once per driven application.
    _fill_dynamic_collective_blocks!(blocks,builder.geometry,X)
end

@noinline _symmetric_collective_block_count_error()=throw(DimensionMismatch(
    "symmetric collective geometry requires exactly one Schur block"))

function _fill_dynamic_collective_blocks!(blocks,
        cache::_SymmetricCollectiveGeometry,X)
    length(blocks)==1||_symmetric_collective_block_count_error()
    # `only(blocks)` still materializes a tiny iterator state on Julia 1.10;
    # the validated direct index is allocation-free on every supported line.
    _fill_symmetric_collective_block!(@inbounds(blocks[1]),cache,X)
    blocks
end

function _fill_dynamic_collective_blocks!(blocks,cache::OneBodyGeometry,X)
    b=cache.basis
    for s in eachindex(b.sectors)
        K=blocks[s];fill!(K,zero(eltype(K)))
        n=length(b.patterns[s])
        @inbounds for a in 1:n,c in 1:n,mu in cache.connections[(s,s)]
            key=(s,mu,s)
            K[a,c]+=cache.scales[key]*_contract(cache.contractions[key][a,c],X)
        end
    end
    blocks
end

function _fill_dynamic_blocks!(blocks,builder::DirectPIBlockBuilder,
                               operator::AbstractPIOperator)
    b=builder.basis
    operator.basis===b||throw(ArgumentError(
        "an in-place direct PI operator belongs to a different basis"))
    for s in eachindex(b.sectors)
        K=blocks[s];off=b.offsets[s];count=length(K)
        prepared=builder.inverse_scales[s]
        if prepared.use_divisor
            copyto!(K,1,operator.data,off,count)
            divisor=prepared.divisor
            @inbounds for index in eachindex(K);K[index]/=divisor;end
        else
            @inbounds for index in eachindex(K)
                K[index]=_apply_prepared_exact_scale(
                    operator.data[off+index-1],prepared.inverse;
                    context="dynamic direct-PI physical Schur block")
            end
        end
    end
    blocks
end

_fill_dynamic_blocks!(blocks,builder,X,cancellation_scratch)=
    _fill_dynamic_blocks!(blocks,builder,X)

function _dynamic_onebody_builder(
        context::TermCompileContext{B,G,P,C,T}) where
        {D,L,B<:PIBasis{D,L},G,P,C,T}
    cache=context.onebody
    cache===nothing&&error(
        "internal error: collective one-body geometry was not prepared")
    cache isa OneBodyGeometry&&_needs_wide_collective(cache.basis,T)&&
        throw(ArgumentError(
            "preallocated collective one-body blocks at N=$(cache.basis.N) cannot " *
            "certify large-N cancellation in fixed $T scratch; use a wider " *
            "InPlaceTimeOperator prototype scalar type"))
    CollectiveOneBodyBlockBuilder{T,D,L,B}(cache)
end

function _direct_pi_block_builder(context::TermCompileContext,::Type{R}) where
        R<:AbstractFloat
    scales=map(context.basis.sectors) do sector
        multiplicity=symmetric_group_dimension(sector)
        divisor=try
            # Reuse the exact hook dimension already needed by the exceptional
            # fused inverse below.  Calling `_schur_multiplicity_scale` here
            # would evaluate the hook formula a second time even on the
            # ordinary direct-division path.
            _checked_sqrt_exact_integer(R,multiplicity;
                context="square root of the sector multiplicity for $sector")
        catch error
            error isa ArgumentError||rethrow()
            nothing
        end
        # When the divisor has a safe reciprocal, build the direct prepared
        # record from that already-computed value instead of running the exact
        # square-root conversion a second time. Only exceptional sectors need
        # the binary-scaled BigInt fallback.
        inverse_factor=divisor===nothing ? nothing : inv(divisor)
        use_divisor=inverse_factor!==nothing&&
            _ordinary_scaled_component_safe(inverse_factor,one(R))
        inverse=if !use_divisor
            _prepare_exact_scale(R,one(BigInt),multiplicity,Val(true);
                context="dynamic direct-PI inverse Schur-multiplicity scale")
        else
            _PreparedExactScale{R,true}(
                one(BigInt),multiplicity,true,inverse_factor,inverse_factor,0)
        end
        (use_divisor=use_divisor,
         divisor=divisor===nothing ? one(R) : divisor,
         inverse)
    end
    DirectPIBlockBuilder(context.basis,scales)
end

_dynamic_builder(context::TermCompileContext,::Union{LocalHamiltonian,
    CollectiveHamiltonian,CollectiveJump})=_dynamic_onebody_builder(context)
_dynamic_builder(context::TermCompileContext,t::Union{PBodyHamiltonian,
    CollectivePBodyJump})=_pbody_block_builder(context,t)

function _collective_blocks(operator::AbstractMatrix{O},
        context::TermCompileContext{B,G,P,C,R}) where {O,B,G,P,C,R}
    S=promote_type(Complex{R},O)
    blocks=Vector{Matrix{S}}(undef,length(context.basis.sectors))
    for index in eachindex(context.basis.sectors)
        blocks[index]=if context.onebody isa _SymmetricCollectiveGeometry
            _symmetric_collective_block(context.basis,operator,
                context.basis.sectors[index],context.onebody)
        else
            collective_block(context.basis,operator,
                context.basis.sectors[index];cache=context.onebody)
        end
    end
    blocks
end

function _sparse_collective_blocks(operator::AbstractMatrix{O},
        context::TermCompileContext{B,G,P,C,R}) where {O,B,G,P,C,R}
    S=promote_type(Complex{R},O)
    blocks=Vector{SparseMatrixCSC{S,Int}}(
        undef,length(context.basis.sectors))
    for index in eachindex(context.basis.sectors)
        blocks[index]=_collective_sparse_block(
            context.basis,operator,context.basis.sectors[index];
            cache=context.onebody)
    end
    blocks
end
_direct_term_blocks(operator,context)=_direct_blocks(context.basis,operator)

# Fixed collective one-body lifts have the exact sparse support of the U(d)
# generators in every Schur irrep. Retaining it lets the existing preallocated
# matrix-multiplication kernels use sparse-dense `mul!` without changing
# arithmetic or allocating in hot loops. Applying this representation to all
# fixed collective one-body terms also keeps plan types independent of the
# runtime sector selection. Dynamic schedules remain dense because their
# support may change at every evaluation.
_prepare_collective_action(
    blocks::AbstractVector{<:SparseMatrixCSC})=blocks
_prepare_collective_action(blocks)=[sparse(block) for block in blocks]

function _compile_hamiltonian(term,context,blocks)
    R=_real_float_type(eltype(first(blocks)))
    HamiltonianPIKernel(blocks,_scaled_rate(term_rate(term),term_hbar(term),R))
end
function _compile_collective_hamiltonian(term,context,blocks)
    _compile_hamiltonian(term,context,_prepare_collective_action(blocks))
end
function compile_term(t::Union{LocalHamiltonian,CollectiveHamiltonian},
                      context::TermCompileContext)
    operator=term_operator(t);R=context.geometry_type
    operator isa InPlaceTimeOperator && return InPlaceHamiltonianPIKernel(
        operator,_dynamic_builder(context,t),_scaled_rate(term_rate(t),term_hbar(t),R))
    _compile_collective_hamiltonian(
        t,context,_sparse_collective_blocks(operator,context))
end
function compile_term(t::DirectPIHamiltonian,context::TermCompileContext)
    operator=term_operator(t);R=context.geometry_type
    if operator isa InPlaceTimeOperator
        scale=_scaled_rate(term_rate(t),term_hbar(t),R)
        block_type=_real_float_type(_scale_promoted_type(Complex{R},scale))
        return InPlaceHamiltonianPIKernel(
            operator,_direct_pi_block_builder(context,block_type),scale)
    end
    _compile_hamiltonian(t,context,_direct_term_blocks(operator,context))
end

function _compile_dissipator(term,blocks)
    DissipatorPIKernel(blocks,[K'*K for K in blocks],term_rate(term))
end
function _compile_collective_dissipator(term,blocks,context)
    # Fixed collective blocks already retain exact CSC support.  Sparse Gram
    # products avoid the otherwise dominant dense K'K temporary while keeping
    # exact structural zeros; dynamic schedules continue to use dense
    # preallocated multiplication in their task-owned workspace.
    qblocks=map(blocks) do K
        Q=K'*K
        dropzeros!(Q)
        Q
    end
    blocks=_prepare_collective_action(blocks)
    qblocks=_prepare_collective_action(qblocks)
    DissipatorPIKernel(blocks,qblocks,term_rate(term))
end
function compile_term(t::CollectiveJump,context::TermCompileContext)
    operator=term_operator(t)
    operator isa InPlaceTimeOperator && return InPlaceDissipatorPIKernel(
        operator,_dynamic_builder(context,t),term_rate(t))
    _compile_collective_dissipator(
        t,_sparse_collective_blocks(operator,context),context)
end
function compile_term(t::DirectPIJump,context::TermCompileContext)
    operator=term_operator(t)
    if operator isa InPlaceTimeOperator
        scale=term_rate(t);R=context.geometry_type
        block_type=_real_float_type(_scale_promoted_type(Complex{R},scale))
        return InPlaceDissipatorPIKernel(
            operator,_direct_pi_block_builder(context,block_type),scale)
    end
    _compile_dissipator(t,_direct_term_blocks(operator,context))
end

function compile_term(t::LocalJump,context::TermCompileContext)
    operator=term_operator(t)
    if operator isa InPlaceTimeOperator
        _dynamic_onebody_builder(context)
        branches=_local_gain_branches(context.basis,context.onebody)
        return InPlaceLocalJumpPIKernel(operator,context.onebody,term_rate(t),
            branches)
    end
    Q=_sparse_collective_blocks(operator'*operator,context)
    prepared_branches=_local_gain_branches(context.basis,context.onebody)
    T=promote_type(eltype(first(Q)),eltype(operator))
    contractions=_local_gain_contraction_workspace(prepared_branches,T)
    _fill_local_gain_contractions!(contractions,prepared_branches,operator)
    branches=_static_local_gain_branches(prepared_branches)
    static_contractions=_static_onebody_contractions(contractions)
    FactorizedLocalJumpPIKernel(Q,branches,static_contractions,term_rate(t))
end

function _effective_correlated_operator(coefficients,operators,
        ::Type{T}) where T<:Number
    effective=zeros(T,size(first(operators)))
    @inbounds for operator_index in eachindex(operators)
        coefficient=coefficients[operator_index]
        operator=operators[operator_index]
        for index in eachindex(effective,operator)
            effective[index]+=coefficient*operator[index]
        end
    end
    effective
end

function _effective_correlated_operators(term::_CorrelatedOneBodyJumps,
        ::Type{R}) where R<:AbstractFloat
    factor=term.factor
    factor===nothing&&throw(ArgumentError(
        "a callable Kossakowski matrix must be frozen or use InPlaceTimeOperator"))
    operators=term.operators
    isempty(factor)&&return (zeros(Complex{R},size(first(operators))),)
    map(coefficients->_effective_correlated_operator(
            coefficients,operators,Complex{R}),factor)
end

function compile_term(term::_CorrelatedOneBodyJumps,
                      context::TermCompileContext)
    matrix=term.kossakowski
    rate=_prepared_correlated_rate(term.rate,context.geometry_type)
    if matrix isa InPlaceTimeOperator
        if term isa CorrelatedCollectiveJumps
            channel_blocks=map(operator->_collective_blocks(operator,context),
                               term.operators)
            return InPlaceCorrelatedCollectiveJumpPIKernel(
                matrix,channel_blocks,rate,term.atol,term.rtol)
        end
        _dynamic_onebody_builder(context)
        branches=_local_gain_branches(context.basis,context.onebody)
        return InPlaceCorrelatedLocalJumpPIKernel(
            matrix,term.operators,context.onebody,rate,
            branches,term.atol,term.rtol)
    end
    matrix isa AbstractMatrix||throw(ArgumentError(
        "a raw callable Kossakowski matrix must use the freeze fallback"))
    effective=_effective_correlated_operators(term,context.geometry_type)
    if term isa CorrelatedLocalJumps
        map(operator->compile_term(LocalJump(operator;rate=rate),context),
            effective)
    else
        map(operator->compile_term(CollectiveJump(operator;rate=rate),context),
            effective)
    end
end

function _pbody_blocks(term,context,operator=term_operator(term))
    geometry=_pbody_geometry!(context,body_order(term));_check_pbody_operator(geometry,operator)
    [pbody_collective_block(geometry,operator,p;check=false) for p in context.basis.sectors]
end

function compile_term(t::PBodyHamiltonian,context::TermCompileContext)
    operator=term_operator(t);R=context.geometry_type
    operator isa InPlaceTimeOperator && return InPlaceHamiltonianPIKernel(
        operator,_dynamic_builder(context,t),_scaled_rate(term_rate(t),term_hbar(t),R))
    _compile_hamiltonian(t,context,_pbody_blocks(t,context))
end
function compile_term(t::CollectivePBodyJump,context::TermCompileContext)
    operator=term_operator(t)
    operator isa InPlaceTimeOperator && return InPlaceDissipatorPIKernel(
        operator,_dynamic_builder(context,t),term_rate(t))
    _compile_dissipator(t,_pbody_blocks(t,context))
end

function compile_term(t::LocalPBodyJump,context::TermCompileContext)
    operator=term_operator(t)
    if operator isa InPlaceTimeOperator
        builder=_pbody_block_builder(context,t)
        structure=_pbody_gain_factorization(builder)
        return InPlaceLocalPBodyJumpPIKernel(operator,builder,term_rate(t),
            structure.groups,structure.left_isometries,
            structure.right_isometries,structure.pair_scales)
    end
    geometry=_pbody_geometry!(context,body_order(t));Qop=operator'*operator
    _check_pbody_operator(geometry,Qop)
    Q=[pbody_collective_block(geometry,Qop,p;check=false) for p in context.basis.sectors]
    # The factorized fixed path deliberately retains the guarded triplet
    # implementation as an oracle/fallback for exceptionally large path
    # scales.  In that regime runtime accumulation through native rectangular
    # scratch could lose a cancellation that the setup-time wide accumulator
    # certifies.  Ordinary small and medium models take the compact path.
    builder=_pbody_block_builder(context,t)
    if builder.cancellation_risk
        return LocalJumpPIKernel(Q,
            pbody_kernel_triplets(geometry,operator,operator),term_rate(t))
    end
    structure=_pbody_gain_factorization_data(builder)
    if any(scale->!scale.direct,structure.pair_scales)
        return LocalJumpPIKernel(Q,
            pbody_kernel_triplets(geometry,operator,operator),term_rate(t))
    end
    T=promote_type(eltype(first(Q)),eltype(operator))
    contractions=Matrix{T}[
        _path_contractions(structure.left_isometries[index],
                           structure.right_isometries[index],operator)
        for index in eachindex(structure.pair_scales)]
    FactorizedLocalPBodyJumpPIKernel(Q,structure.groups,contractions,
                                    structure.pair_scales,term_rate(t))
end

_compiled_kernel_tuple(kernel::Tuple)=kernel
_compiled_kernel_tuple(kernel)=(kernel,)
_compile_term_sequence(::Tuple{},context)=()
function _compile_term_sequence(terms::Tuple{T,Vararg{Any}},context) where T
    current=_compiled_kernel_tuple(compile_term(first(terms),context))
    (current...,_compile_term_sequence(Base.tail(terms),context)...)
end

_fusable_fixed_kernel(::HamiltonianPIKernel{B,S}) where {B,S<:Number}=true
_fusable_fixed_kernel(::DissipatorPIKernel{B,Q,S}) where {B,Q,S<:Real}=true
_fusable_fixed_kernel(::FactorizedLocalJumpPIKernel{Q,B,C,S}) where
    {Q,B,C,S<:Real}=true
_fusable_fixed_kernel(::FactorizedLocalPBodyJumpPIKernel{Q,G,C,P,S}) where
    {Q,G,C,P,S<:Real}=true
_fusable_fixed_kernel(kernel)=false

_partition_fusable_kernels(::Tuple{})=((),())
function _partition_fusable_kernels(kernels::Tuple{K,Vararg{Any}}) where K
    selected,remaining=_partition_fusable_kernels(Base.tail(kernels))
    if _fusable_fixed_kernel(first(kernels))
        ((first(kernels),selected...),remaining)
    else
        (selected,(first(kernels),remaining...))
    end
end

_select_kernel_type(::Tuple{},::Type{S}) where S=()
function _select_kernel_type(kernels::Tuple{K,Vararg{Any}},
                             ::Type{S}) where {K,S}
    selected=_select_kernel_type(Base.tail(kernels),S)
    K<:S ? (first(kernels),selected...) : selected
end

function _fused_zero_blocks(b,::Type{T}) where T
    [zeros(T,length(b.patterns[index]),length(b.patterns[index]))
     for index in eachindex(b.sectors)]
end

_fusion_source_blocks(kernel,::Val{:hamiltonian})=kernel.blocks
_fusion_source_blocks(kernel,::Val{:loss})=kernel.qblocks
_sparse_fusion_blocks_trait(
    ::AbstractVector{<:SparseMatrixCSC})=Val(true)
_sparse_fusion_blocks_trait(blocks)=Val(false)
_sparse_fusion_and(::Val{true},::Val{true})=Val(true)
_sparse_fusion_and(left,right)=Val(false)
_sparse_fusion_trait(::Tuple{},field)=Val(true)
function _sparse_fusion_trait(kernels::Tuple{K,Vararg{Any}},
                              field) where K
    _sparse_fusion_and(
        _sparse_fusion_blocks_trait(
            _fusion_source_blocks(first(kernels),field)),
        _sparse_fusion_trait(Base.tail(kernels),field))
end

# Add fixed sparse Schur blocks column by column.  The accumulator is only one
# Schur-vector wide, and contributions to each coordinate are visited in the
# same kernel order as the former dense-block loop.  Exact cancellations are
# removed; no numerical dropping tolerance is introduced.
function _fused_sparse_scaled_blocks(kernels::Tuple,b,::Type{T},
                                     field::Val) where T
    prepared=Vector{SparseMatrixCSC{T,Int}}(
        undef,length(b.sectors))
    for sector in eachindex(b.sectors)
        dimension=length(b.patterns[sector])
        column_offsets=Vector{Int}(undef,dimension+1)
        column_offsets[1]=1
        row_indices=Int[]
        nonzeros=T[]
        support_upper_bound=try
            foldl(kernels;init=0) do total,kernel
                Base.checked_add(total,
                    nnz(_fusion_source_blocks(kernel,field)[sector]))
            end
        catch error
            error isa OverflowError||rethrow()
            nothing
        end
        if support_upper_bound!==nothing
            sizehint!(row_indices,support_upper_bound)
            sizehint!(nonzeros,support_upper_bound)
        end
        accumulator=zeros(T,dimension)
        marked=falses(dimension)
        touched=Int[]
        if support_upper_bound!==nothing
            sizehint!(touched,min(dimension,support_upper_bound))
        end
        for column in 1:dimension
            empty!(touched)
            for kernel in kernels
                scale=convert(T,kernel.scale)
                source=_fusion_source_blocks(kernel,field)[sector]
                @inbounds for pointer in
                        source.colptr[column]:(source.colptr[column+1]-1)
                    row=source.rowval[pointer]
                    if !marked[row]
                        marked[row]=true
                        push!(touched,row)
                    end
                    accumulator[row]+=
                        scale*convert(T,source.nzval[pointer])
                end
            end
            sort!(touched)
            @inbounds for row in touched
                value=accumulator[row]
                if !iszero(value)
                    push!(row_indices,row)
                    push!(nonzeros,value)
                end
                accumulator[row]=zero(T)
                marked[row]=false
            end
            column_offsets[column+1]=length(nonzeros)+1
        end
        prepared[sector]=SparseMatrixCSC(
            dimension,dimension,column_offsets,row_indices,nonzeros)
    end
    prepared
end

function _fused_dense_scaled_blocks(kernels::Tuple,b,::Type{T},
                                    field::Val) where T
    blocks=_fused_zero_blocks(b,T)
    for kernel in kernels
        scale=convert(T,kernel.scale)
        sources=_fusion_source_blocks(kernel,field)
        for sector in eachindex(blocks),index in eachindex(blocks[sector])
            blocks[sector][index]+=scale*sources[sector][index]
        end
    end
    blocks
end

_fused_scaled_blocks(::Tuple{},b,::Type{T},field) where T=nothing
function _fused_scaled_blocks(kernels::Tuple,b,::Type{T},field) where T
    _fused_scaled_blocks(
        kernels,b,T,field,_sparse_fusion_trait(kernels,field))
end
_fused_scaled_blocks(kernels,b,::Type{T},field,::Val{true}) where T=
    _fused_sparse_scaled_blocks(kernels,b,T,field)
_fused_scaled_blocks(kernels,b,::Type{T},field,::Val{false}) where T=
    _fused_dense_scaled_blocks(kernels,b,T,field)

_fused_owned_matrix(matrix::Matrix{T},::Type{T}) where T=matrix
_fused_owned_matrix(matrix::SparseMatrixCSC{T,Int},::Type{T}) where T=matrix
_fused_owned_matrix(matrix::SparseMatrixCSC,::Type{T}) where T=
    SparseMatrixCSC{T,Int}(matrix)
_fused_owned_matrix(matrix,::Type{T}) where T=Matrix{T}(matrix)
_fused_owned_matrix(contraction::_StaticOneBodyContraction{T},
                    ::Type{T}) where T=contraction
function _fused_owned_matrix(contraction::_StaticOneBodyContraction,
                             ::Type{T}) where T
    _static_onebody_contraction(Matrix{T}(contraction.matrix))
end

function _fused_owned_blocks(blocks::AbstractVector{<:SparseMatrixCSC},
        ::Type{T}) where T
    prepared=Vector{SparseMatrixCSC{T,Int}}(undef,length(blocks))
    @inbounds for index in eachindex(blocks)
        prepared[index]=_fused_owned_matrix(blocks[index],T)
    end
    prepared
end
function _fused_owned_blocks(blocks::AbstractVector,::Type{T}) where T
    prepared=Vector{Matrix{T}}(undef,length(blocks))
    @inbounds for index in eachindex(blocks)
        prepared[index]=_fused_owned_matrix(blocks[index],T)
    end
    prepared
end

function _fused_owned_onebody_contractions(contractions::V,::Type{T}) where
        {T,M,C<:_StaticOneBodyContraction{T,M},V<:AbstractVector{C}}
    contractions
end
function _fused_owned_onebody_contractions(contractions::AbstractVector,
                                            ::Type{T}) where T
    prepared=Vector{_StaticOneBodyContraction{T,Matrix{T}}}(
        undef,length(contractions))
    @inbounds for index in eachindex(contractions)
        prepared[index]=_fused_owned_matrix(contractions[index],T)
    end
    prepared
end

_fused_collective_gains(::Tuple{},::Type{T}) where T=()
function _fused_collective_gains(kernels::Tuple{K,Vararg{Any}},
                                 ::Type{T}) where {K,T}
    kernel=first(kernels)
    gain=_FusedCollectiveGain(_fused_owned_blocks(kernel.blocks,T),
                              convert(T,kernel.scale))
    (gain,_fused_collective_gains(Base.tail(kernels),T)...)
end

_fused_onebody_gains(::Tuple{},::Type{T}) where T=()
function _fused_onebody_gains(kernels::Tuple{K,Vararg{Any}},
                              ::Type{T}) where {K,T}
    kernel=first(kernels)
    gain=_FusedOneBodyGain(kernel.branches,
        _fused_owned_onebody_contractions(kernel.contractions,T),
        convert(T,kernel.scale))
    (gain,_fused_onebody_gains(Base.tail(kernels),T)...)
end

_fused_pbody_gains(::Tuple{},::Type{T}) where T=()
function _fused_pbody_gains(kernels::Tuple{K,Vararg{Any}},
                            ::Type{T}) where {K,T}
    kernel=first(kernels)
    gain=_FusedPBodyGain(kernel.groups,
        _fused_owned_blocks(kernel.contractions,T),kernel.pair_scales,
        convert(T,kernel.scale))
    (gain,_fused_pbody_gains(Base.tail(kernels),T)...)
end

function _fuse_fixed_kernels(kernels::Tuple,b)
    selected,remaining=_partition_fusable_kernels(kernels)
    _fuse_selected_fixed_kernels(selected,remaining,kernels,b)
end

# Dispatch on tuple length rather than branching on `length(selected)`.  This
# keeps Julia 1.10 from inferring a union of the fused and unfused plan types.
_fuse_selected_fixed_kernels(::Tuple{},remaining,kernels,b)=kernels
_fuse_selected_fixed_kernels(::Tuple{K},remaining,kernels,b) where K=kernels
function _fuse_selected_fixed_kernels(
        selected::Tuple{K1,K2,Vararg{Any}},remaining,kernels,b) where {K1,K2}
    T=foldl(promote_type,map(_kernel_scalar_type,selected))

    hamiltonians=_select_kernel_type(selected,HamiltonianPIKernel)
    hamiltonian_blocks=_fused_scaled_blocks(
        hamiltonians,b,T,Val(:hamiltonian))

    DissipativeKernel=Union{DissipatorPIKernel,FactorizedLocalJumpPIKernel,
                            FactorizedLocalPBodyJumpPIKernel}
    dissipative=_select_kernel_type(selected,DissipativeKernel)
    # Type-based partitioning keeps plan construction fully inferred. Validate
    # the values before absorbing their rates so nonfinite fixed inputs cannot
    # evade the ordinary dissipative application check.
    foreach(kernel->_evaluated_dissipative_rate(kernel.scale,0.0,nothing),
            dissipative)
    loss_blocks=_fused_scaled_blocks(
        dissipative,b,T,Val(:loss))

    collective=_select_kernel_type(selected,DissipatorPIKernel)
    onebody=_select_kernel_type(selected,FactorizedLocalJumpPIKernel)
    pbody=_select_kernel_type(selected,FactorizedLocalPBodyJumpPIKernel)
    collective_gains=_fused_collective_gains(collective,T)
    onebody_gains=_fused_onebody_gains(onebody,T)
    pbody_gains=_fused_pbody_gains(pbody,T)

    fused=FusedStaticPIKernel(hamiltonian_blocks,loss_blocks,
                              collective_gains,onebody_gains,pbody_gains)
    (fused,remaining...)
end

function _static_kernels_unfused(model;coefficient_cache=nothing)
    context=TermCompileContext(model;coefficient_cache)
    _compile_term_sequence(model.terms,context)
end
_static_fusion_flag(flag::Val{F}) where F=Val(F)
_static_fusion_flag(flag::Bool)=Val(flag)
_apply_static_fusion(kernels,b,::Val{true})=
    _fuse_fixed_kernels(kernels,b)
_apply_static_fusion(kernels,b,::Val{false})=kernels

function _static_kernels(model;coefficient_cache=nothing,
                         fuse_static=Val(true))
    kernels=_static_kernels_unfused(model;coefficient_cache)
    _apply_static_fusion(
        kernels,model.basis,_static_fusion_flag(fuse_static))
end

"""
    LiouvillianPlan(model; coefficient_cache=nothing)

Immutable prepared term data and geometry for a PI Liouvillian. A supplied
[`OneBoxCGCache`](@ref) is shared by every compatible one- and `p`-body
geometry built for the model; it must belong to the exact basis and cover the
required body-order depth and scalar type. When it is omitted, compilation
automatically prepares a bounded temporary cache for small models that need
more than one geometry family. Fixed local gains retain rectangular Schur
factors, and compatible autonomous numeric kernels share one effective
Hamiltonian and anticommutator loss. Gain channels remain separate. The
internal `fuse_static=false` diagnostic route retains one kernel sequence per
model term for reference and channel-resolved consumers.
"""
struct LiouvillianPlan{B,K,V,M,T}
    basis::B
    kernels::K
    tracevec::V
    fallback_model::M
    Ttype::Type{T}
    autonomous::Bool
end

_scale_promoted_type(::Type{T},scale::S) where {T,S<:Number}=promote_type(T,S)
_scale_promoted_type(::Type{T},scale) where T=T
_kernel_scalar_type(kernel::HamiltonianPIKernel)=_scale_promoted_type(eltype(first(kernel.blocks)),kernel.scale)
_kernel_scalar_type(kernel::DissipatorPIKernel)=_scale_promoted_type(eltype(first(kernel.blocks)),kernel.scale)
_kernel_scalar_type(kernel::LocalJumpPIKernel)=_scale_promoted_type(
    promote_type(eltype(kernel.gain.V),eltype(first(kernel.qblocks))),kernel.scale)
_kernel_scalar_type(kernel::FactorizedLocalJumpPIKernel)=_scale_promoted_type(
    promote_type(eltype(first(kernel.contractions)),
                 eltype(first(kernel.qblocks))),kernel.scale)
_kernel_scalar_type(kernel::FactorizedLocalPBodyJumpPIKernel)=_scale_promoted_type(
    promote_type(eltype(first(kernel.contractions)),
                 eltype(first(kernel.qblocks))),kernel.scale)
function _kernel_scalar_type(kernel::FusedStaticPIKernel)
    kernel.hamiltonian_blocks!==nothing&&
        return eltype(first(kernel.hamiltonian_blocks))
    kernel.loss_blocks!==nothing&&return eltype(first(kernel.loss_blocks))
    !isempty(kernel.collective_gains)&&
        return eltype(first(first(kernel.collective_gains).blocks))
    !isempty(kernel.onebody_gains)&&
        return eltype(first(first(kernel.onebody_gains).contractions))
    eltype(first(first(kernel.pbody_gains).contractions))
end
_schedule_scalar_type(::InPlaceTimeOperator{O}) where
    {T,O<:AbstractMatrix{T}}=Complex{_real_float_type(T)}
_schedule_scalar_type(::InPlaceTimeOperator{O}) where
    {R,O<:AbstractPIOperator{R}}=Complex{R}
_schedule_scalar_type(schedule::InPlaceTimeOperator)=begin
    prototype=schedule.prototype
    R=prototype isa AbstractPIOperator ? _real_float_type(eltype(prototype.data)) :
      _real_float_type(eltype(prototype))
    Complex{R}
end
_kernel_scalar_type(kernel::InPlaceHamiltonianPIKernel)=
    _scale_promoted_type(_schedule_scalar_type(kernel.schedule),kernel.scale)
_kernel_scalar_type(kernel::InPlaceDissipatorPIKernel)=
    _scale_promoted_type(_schedule_scalar_type(kernel.schedule),kernel.scale)
_kernel_scalar_type(kernel::InPlaceLocalJumpPIKernel)=
    _scale_promoted_type(_schedule_scalar_type(kernel.schedule),kernel.scale)
_kernel_scalar_type(kernel::InPlaceLocalPBodyJumpPIKernel)=
    _scale_promoted_type(_schedule_scalar_type(kernel.schedule),kernel.scale)
function _kernel_scalar_type(kernel::InPlaceCorrelatedCollectiveJumpPIKernel)
    block_type=eltype(first(first(kernel.channel_blocks)))
    _scale_promoted_type(promote_type(_schedule_scalar_type(kernel.schedule),
                                      block_type),kernel.scale)
end
function _kernel_scalar_type(kernel::InPlaceCorrelatedLocalJumpPIKernel)
    geometry_type=Complex{geometry_scalar_type(kernel.geometry)}
    _scale_promoted_type(promote_type(_schedule_scalar_type(kernel.schedule),
                                      geometry_type),kernel.scale)
end

function _liouvillian_plan_from_kernels(model,kernels)
    fixed_operators=all(term_has_fixed_operator,model.terms)
    fallback=fixed_operators ? nothing : model
    T = kernels===nothing||isempty(kernels) ? ComplexF64 :
        foldl(promote_type,(_kernel_scalar_type(k) for k in kernels))
    LiouvillianPlan(model.basis,kernels,_trace_vector(model.basis,T),
                    fallback,T,isautonomous(model))
end

function LiouvillianPlan(model::PIModel;coefficient_cache=nothing,
                         fuse_static=Val(true))
    prepared_operators=all(t->term_has_fixed_operator(t)||term_has_preallocated_operator(t),
                           model.terms)
    kernels=prepared_operators ?
        _static_kernels(model;coefficient_cache,fuse_static) : nothing
    _liouvillian_plan_from_kernels(model,kernels)
end

# Term-resolved consumers (quantum trajectories, population certificates,
# and channel metadata) require a one-to-one model-term/kernel sequence.  A
# separate branch-free helper keeps their constructors fully inferred while
# ordinary deterministic plans retain autonomous fusion by default.
function _term_resolved_liouvillian_plan(model::PIModel;
                                         coefficient_cache=nothing)
    prepared_operators=all(t->term_has_fixed_operator(t)||term_has_preallocated_operator(t),
                           model.terms)
    kernels=prepared_operators ?
        _static_kernels_unfused(model;coefficient_cache) : nothing
    _liouvillian_plan_from_kernels(model,kernels)
end

size(plan::LiouvillianPlan)=(length(plan.basis),length(plan.basis))
size(plan::LiouvillianPlan,i::Integer)=i in (1,2) ? length(plan.basis) : 1
eltype(plan::LiouvillianPlan)=plan.Ttype
isautonomous(plan::LiouvillianPlan)=plan.autonomous

mutable struct _LiouvillianBatchScratch{T}
    maximum_block_dimension::Int
    capacity::Int
    input::Matrix{T}
    left::Matrix{T}
    right::Matrix{T}
end

function _LiouvillianBatchScratch(::Type{T},maximum_block_dimension::Int) where T
    empty=zeros(T,maximum_block_dimension,0)
    _LiouvillianBatchScratch{T}(maximum_block_dimension,0,empty,
        similar(empty),similar(empty))
end

function _ensure_batch_capacity!(scratch::_LiouvillianBatchScratch{T},
                                 columns::Int) where T
    columns<=scratch.capacity&&return scratch
    width=Base.checked_mul(scratch.maximum_block_dimension,columns)
    scratch.input=zeros(T,scratch.maximum_block_dimension,width)
    scratch.left=similar(scratch.input)
    scratch.right=similar(scratch.input)
    scratch.capacity=columns
    scratch
end

"""
Per-task mutable scratch for applying a `LiouvillianPlan`.

Vector application retains three matrices per Schur sector. Batched
application grows an additional three largest-sector buffers on first use and
then reuses them. This keeps ordinary vector workspaces lean while allowing
sectorwise matrix--matrix kernels for multiple right-hand sides.
"""
struct LiouvillianWorkspace{B,W,K,T,S}
    basis::B
    blocks::W
    kernel_workspaces::K
    Ttype::Type{T}
    batch::S
end

_operator_workspace(prototype::AbstractMatrix)=Matrix(prototype)
_operator_workspace(prototype::AbstractPIOperator)=copy(prototype)
function _dynamic_block_workspace(b,T)
    [zeros(T,length(b.patterns[s]),length(b.patterns[s]))
     for s in eachindex(b.sectors)]
end
function _dynamic_pbody_cancellation_workspace(builder,b,::Type{T}) where T
    required=builder isa CollectivePBodyBlockBuilder&&
        builder.cancellation_risk
    largest=required ? maximum(length,b.patterns;init=0) : 0
    zeros(_real_float_type(T),largest,largest)
end
_kernel_workspace(::AbstractStaticPIKernel,b,T)=nothing
function _kernel_workspace(kernel::Union{FactorizedLocalJumpPIKernel,
        FactorizedLocalPBodyJumpPIKernel},b,T)
    largest=maximum(length,b.patterns;init=1)
    StaticFactorizedGainKernelWorkspace(zeros(T,largest,largest))
end
function _kernel_workspace(kernel::FusedStaticPIKernel,b,T)
    has_rectangular=!isempty(kernel.onebody_gains)||!isempty(kernel.pbody_gains)
    largest=has_rectangular ? maximum(length,b.patterns;init=1) : 0
    StaticFactorizedGainKernelWorkspace(zeros(T,largest,largest))
end
function _kernel_workspace(kernel::InPlaceHamiltonianPIKernel,b,T)
    InPlaceHamiltonianKernelWorkspace(_operator_workspace(kernel.schedule.prototype),
        _dynamic_block_workspace(b,T),
        _dynamic_pbody_cancellation_workspace(kernel.builder,b,T))
end
function _kernel_workspace(kernel::InPlaceDissipatorPIKernel,b,T)
    blocks=_dynamic_block_workspace(b,T)
    InPlaceDissipatorKernelWorkspace(_operator_workspace(kernel.schedule.prototype),
        blocks,_dynamic_block_workspace(b,T),
        _dynamic_pbody_cancellation_workspace(kernel.builder,b,T))
end
function _local_gain_contraction_workspace(branches,::Type{T},
                                           channels::Int=1) where T
    count=Base.checked_mul(channels,length(branches.entries))
    contractions=Vector{Matrix{T}}(undef,count)
    index=0
    for _ in 1:channels,branch in branches.entries
        index+=1
        contractions[index]=zeros(T,size(branch.table))
    end
    contractions
end
function _kernel_workspace(kernel::InPlaceLocalJumpPIKernel,b,T)
    operator=_operator_workspace(kernel.schedule.prototype)
    qoperator=similar(operator,size(operator))
    InPlaceLocalJumpKernelWorkspace(operator,qoperator,
        _dynamic_block_workspace(b,T),
        _local_gain_contraction_workspace(kernel.branches,T),
        zeros(T,kernel.branches.maximum_block_dimension,
                kernel.branches.maximum_block_dimension))
end
function _kernel_workspace(kernel::InPlaceLocalPBodyJumpPIKernel,b,T)
    operator=_operator_workspace(kernel.schedule.prototype)
    qoperator=similar(operator,size(operator))
    contractions=[zeros(T,size(kernel.left_isometries[index],1),
                            size(kernel.right_isometries[index],1))
                  for index in eachindex(kernel.pair_scales)]
    largest_block=maximum(length,b.patterns;init=0)
    InPlaceLocalPBodyJumpKernelWorkspace(operator,qoperator,
        _dynamic_block_workspace(b,T),contractions,
        zeros(T,largest_block,largest_block),
        _dynamic_pbody_cancellation_workspace(kernel.builder,b,T))
end
function _kernel_workspace(kernel::InPlaceCorrelatedCollectiveJumpPIKernel,b,T)
    m=length(kernel.channel_blocks)
    InPlaceCorrelatedCollectiveJumpKernelWorkspace(
        _operator_workspace(kernel.schedule.prototype),zeros(T,m,m),zeros(T,m,m),
        [_dynamic_block_workspace(b,T) for _ in 1:m],
        _dynamic_block_workspace(b,T),_dynamic_block_workspace(b,T),Ref(0))
end
function _kernel_workspace(kernel::InPlaceCorrelatedLocalJumpPIKernel,b,T)
    m=length(kernel.operators);d=size(first(kernel.operators),1)
    InPlaceCorrelatedLocalJumpKernelWorkspace(
        _operator_workspace(kernel.schedule.prototype),zeros(T,m,m),zeros(T,m,m),
        [zeros(T,d,d) for _ in 1:m],zeros(T,d,d),zeros(T,d,d),
        _dynamic_block_workspace(b,T),
        _local_gain_contraction_workspace(kernel.branches,T,m),
        zeros(T,kernel.branches.maximum_block_dimension,
                kernel.branches.maximum_block_dimension),Ref(0))
end

function LiouvillianWorkspace(plan::LiouvillianPlan)
    T=plan.Ttype
    work=[(zeros(T,length(plan.basis.patterns[s]),length(plan.basis.patterns[s])),
           zeros(T,length(plan.basis.patterns[s]),length(plan.basis.patterns[s])),
           zeros(T,length(plan.basis.patterns[s]),length(plan.basis.patterns[s])))
          for s in eachindex(plan.basis.sectors)]
    kernel_workspaces=plan.kernels===nothing ? nothing :
        map(kernel->_kernel_workspace(kernel,plan.basis,T),plan.kernels)
    maximum_block=maximum(length,plan.basis.patterns;init=1)
    batch=_LiouvillianBatchScratch(T,maximum_block)
    LiouvillianWorkspace(plan.basis,work,kernel_workspaces,T,batch)
end

function _check_liouvillian_workspace(work::LiouvillianWorkspace,plan::LiouvillianPlan)
    work.basis===plan.basis||throw(ArgumentError("Liouvillian workspace belongs to a different PI basis"))
    work.Ttype===plan.Ttype||throw(ArgumentError("Liouvillian workspace has an incompatible scalar type"))
    work
end

@inline function _pack_batch_blocks!(packed,X,offset::Int,dimension::Int,
                                     columns::Int)
    @inbounds for rhs in 1:columns,column in 1:dimension,row in 1:dimension
        packed[row,(rhs-1)*dimension+column]=
            X[offset+row-1+(column-1)*dimension,rhs]
    end
    packed
end

@inline function _pack_adjoint_batch_blocks!(packed,X,offset::Int,
                                             dimension::Int,columns::Int)
    @inbounds for rhs in 1:columns,column in 1:dimension,row in 1:dimension
        packed[row,(rhs-1)*dimension+column]=conj(
            X[offset+column-1+(row-1)*dimension,rhs])
    end
    packed
end


@inline function _pack_adjoint_contiguous_blocks!(packed,source,
        dimension::Int,columns::Int)
    @inbounds for rhs in 1:columns,column in 1:dimension,row in 1:dimension
        packed[row,(rhs-1)*dimension+column]=conj(
            source[column,(rhs-1)*dimension+row])
    end
    packed
end


function _batch_add_left_right!(Y,X,offset::Int,dimension::Int,
        left_operator,right_operator,left_scale,right_scale,
        scratch::_LiouvillianBatchScratch)
    columns=size(X,2);columns==0&&return Y
    width=dimension*columns
    input=@view scratch.input[1:dimension,1:width]
    left=@view scratch.left[1:dimension,1:width]
    right=@view scratch.right[1:dimension,1:width]
    _pack_batch_blocks!(input,X,offset,dimension,columns)
    mul!(left,left_operator,input)
    _pack_adjoint_batch_blocks!(input,X,offset,dimension,columns)
    mul!(right,adjoint(right_operator),input)
    @inbounds for rhs in 1:columns,column in 1:dimension,row in 1:dimension
        coordinate=offset+row-1+(column-1)*dimension
        Y[coordinate,rhs]+=left_scale*left[row,(rhs-1)*dimension+column]+
            right_scale*conj(right[column,(rhs-1)*dimension+row])
    end
    Y
end


function _batch_add_sandwich!(Y,X,offset::Int,dimension::Int,
        left_operator,right_operator,scale,
        scratch::_LiouvillianBatchScratch)
    columns=size(X,2);columns==0&&return Y
    width=dimension*columns
    input=@view scratch.input[1:dimension,1:width]
    left=@view scratch.left[1:dimension,1:width]
    right=@view scratch.right[1:dimension,1:width]
    _pack_batch_blocks!(input,X,offset,dimension,columns)
    mul!(left,left_operator,input)
    _pack_adjoint_contiguous_blocks!(input,left,dimension,columns)
    mul!(right,adjoint(right_operator),input)
    @inbounds for rhs in 1:columns,column in 1:dimension,row in 1:dimension
        coordinate=offset+row-1+(column-1)*dimension
        Y[coordinate,rhs]+=scale*conj(
            right[column,(rhs-1)*dimension+row])
    end
    Y
end


@inline function _pack_adjoint_rectangular_blocks!(packed,source,
        output_dimension::Int,input_dimension::Int,columns::Int)
    @inbounds for rhs in 1:columns,column in 1:output_dimension,
                  row in 1:input_dimension
        packed[row,(rhs-1)*output_dimension+column]=conj(
            source[column,(rhs-1)*input_dimension+row])
    end
    packed
end


function _batch_add_rectangular_sandwich!(Y,X,output_offset::Int,
        input_offset::Int,output_dimension::Int,input_dimension::Int,
        operator,scale,scratch::_LiouvillianBatchScratch)
    columns=size(X,2);columns==0&&return Y
    input_width=input_dimension*columns
    input=@view scratch.input[1:input_dimension,1:input_width]
    left=@view scratch.left[1:output_dimension,1:input_width]
    _pack_batch_blocks!(input,X,input_offset,input_dimension,columns)
    mul!(left,operator,input)

    output_width=output_dimension*columns
    adjoint_input=@view scratch.input[1:input_dimension,1:output_width]
    right=@view scratch.right[1:output_dimension,1:output_width]
    _pack_adjoint_rectangular_blocks!(
        adjoint_input,left,output_dimension,input_dimension,columns)
    mul!(right,operator,adjoint_input)
    @inbounds for rhs in 1:columns,column in 1:output_dimension,
                  row in 1:output_dimension
        coordinate=output_offset+row-1+(column-1)*output_dimension
        Y[coordinate,rhs]+=scale*conj(
            right[column,(rhs-1)*output_dimension+row])
    end
    Y
end

function _batch_add_supported_onebody_sandwich!(Y,X,output_offset::Int,
        input_offset::Int,contraction::_StaticOneBodyContraction,scale;
        adjoint::Bool=false)
    rows=contraction.output_rows;columns=contraction.input_columns
    values=contraction.values;right_hand_sides=size(X,2)
    if adjoint
        output_dimension=size(contraction,2)
        input_dimension=size(contraction,1)
        @inbounds for rhs in 1:right_hand_sides,right in eachindex(values)
            output_column=columns[right]
            input_column=rows[right]
            right_value=values[right]
            for left in eachindex(values)
                output_row=columns[left]
                input_row=rows[left]
                output_coordinate=output_offset+output_row-1+
                    (output_column-1)*output_dimension
                input_coordinate=input_offset+input_row-1+
                    (input_column-1)*input_dimension
                Y[output_coordinate,rhs]+=scale*conj(values[left])*
                    right_value*X[input_coordinate,rhs]
            end
        end
    else
        output_dimension=size(contraction,1)
        input_dimension=size(contraction,2)
        @inbounds for rhs in 1:right_hand_sides,right in eachindex(values)
            output_column=rows[right]
            input_column=columns[right]
            right_value=values[right]
            for left in eachindex(values)
                output_row=rows[left]
                input_row=columns[left]
                output_coordinate=output_offset+output_row-1+
                    (output_column-1)*output_dimension
                input_coordinate=input_offset+input_row-1+
                    (input_column-1)*input_dimension
                Y[output_coordinate,rhs]+=scale*values[left]*
                    conj(right_value)*X[input_coordinate,rhs]
            end
        end
    end
    Y
end

function _batch_add_rectangular_sandwich!(Y,X,output_offset::Int,
        input_offset::Int,output_dimension::Int,input_dimension::Int,
        contraction::_StaticOneBodyContraction,scale,
        scratch::_LiouvillianBatchScratch)
    contraction.use_support&&return _batch_add_supported_onebody_sandwich!(
        Y,X,output_offset,input_offset,contraction,scale)
    _batch_add_rectangular_sandwich!(Y,X,output_offset,input_offset,
        output_dimension,input_dimension,contraction.matrix,scale,scratch)
end

function _batch_add_rectangular_sandwich!(Y,X,output_offset::Int,
        input_offset::Int,output_dimension::Int,input_dimension::Int,
        contraction::LinearAlgebra.Adjoint{T,S},scale,
        scratch::_LiouvillianBatchScratch) where
        {T,S<:_StaticOneBodyContraction}
    parent_contraction=parent(contraction)
    if parent_contraction.use_support
        return _batch_add_supported_onebody_sandwich!(Y,X,output_offset,
            input_offset,parent_contraction,scale;adjoint=true)
    end
    _batch_add_rectangular_sandwich!(Y,X,output_offset,input_offset,
        output_dimension,input_dimension,
        LinearAlgebra.adjoint(parent_contraction.matrix),scale,scratch)
end


function _apply_kernel_batch!(Y,X,kernel::HamiltonianPIKernel,b,t,p,scratch)
    scale=convert(eltype(scratch.input),value_at(kernel.scale,t,p))
    for sector in eachindex(b.sectors)
        dimension=length(b.patterns[sector]);offset=b.offsets[sector]
        block=kernel.blocks[sector]
        _batch_add_left_right!(Y,X,offset,dimension,block,block,
            -1im*scale,1im*scale,scratch)
    end
    Y
end


function _apply_adjoint_kernel_batch!(Y,X,kernel::HamiltonianPIKernel,b,t,p,
                                      scratch)
    scale=conj(convert(eltype(scratch.input),value_at(kernel.scale,t,p)))
    for sector in eachindex(b.sectors)
        dimension=length(b.patterns[sector]);offset=b.offsets[sector]
        block=adjoint(kernel.blocks[sector])
        _batch_add_left_right!(Y,X,offset,dimension,block,block,
            1im*scale,-1im*scale,scratch)
    end
    Y
end


function _apply_kernel_batch!(Y,X,kernel::DissipatorPIKernel,b,t,p,scratch)
    scale=convert(eltype(scratch.input),
        _evaluated_dissipative_rate(kernel.scale,t,p))
    for sector in eachindex(b.sectors)
        dimension=length(b.patterns[sector]);offset=b.offsets[sector]
        block=kernel.blocks[sector];qblock=kernel.qblocks[sector]
        _batch_add_sandwich!(Y,X,offset,dimension,block,adjoint(block),
            scale,scratch)
        _batch_add_left_right!(Y,X,offset,dimension,qblock,qblock,
            -scale/2,-scale/2,scratch)
    end
    Y
end


function _apply_adjoint_kernel_batch!(Y,X,kernel::DissipatorPIKernel,b,t,p,
                                      scratch)
    scale=conj(convert(eltype(scratch.input),
        _evaluated_dissipative_rate(kernel.scale,t,p)))
    for sector in eachindex(b.sectors)
        dimension=length(b.patterns[sector]);offset=b.offsets[sector]
        block=kernel.blocks[sector];qblock=adjoint(kernel.qblocks[sector])
        _batch_add_sandwich!(Y,X,offset,dimension,adjoint(block),block,
            scale,scratch)
        _batch_add_left_right!(Y,X,offset,dimension,qblock,qblock,
            -scale/2,-scale/2,scratch)
    end
    Y
end


function _apply_kernel_batch!(Y,X,kernel::LocalJumpPIKernel,b,t,p,scratch)
    scale=convert(eltype(scratch.input),
        _evaluated_dissipative_rate(kernel.scale,t,p))
    @inbounds for rhs in axes(X,2),index in eachindex(kernel.gain.V)
        Y[kernel.gain.I[index],rhs]+=scale*kernel.gain.V[index]*
            X[kernel.gain.J[index],rhs]
    end
    for sector in eachindex(b.sectors)
        dimension=length(b.patterns[sector]);offset=b.offsets[sector]
        qblock=kernel.qblocks[sector]
        _batch_add_left_right!(Y,X,offset,dimension,qblock,qblock,
            -scale/2,-scale/2,scratch)
    end
    Y
end


function _apply_adjoint_kernel_batch!(Y,X,kernel::LocalJumpPIKernel,b,t,p,
                                      scratch)
    scale=conj(convert(eltype(scratch.input),
        _evaluated_dissipative_rate(kernel.scale,t,p)))
    @inbounds for rhs in axes(X,2),index in eachindex(kernel.gain.V)
        Y[kernel.gain.J[index],rhs]+=scale*conj(kernel.gain.V[index])*
            X[kernel.gain.I[index],rhs]
    end
    for sector in eachindex(b.sectors)
        dimension=length(b.patterns[sector]);offset=b.offsets[sector]
        qblock=adjoint(kernel.qblocks[sector])
        _batch_add_left_right!(Y,X,offset,dimension,qblock,qblock,
            -scale/2,-scale/2,scratch)
    end
    Y
end

function _fixed_liouvillian_scalar_type(L)
    L isa LiouvillianPlan&&return L.Ttype
    L isa MatrixFreeLiouvillian&&L.plan!==nothing&&return L.plan.Ttype
    L isa CompiledPIModel&&L.backend===:matrixfree&&return L.plan.Ttype
    L isa AdjointMatrixFreeLiouvillian&&return _fixed_liouvillian_scalar_type(L.parent)
    nothing
end

function _check_liouvillian_source_precision(L,::Type{T},context) where T
    fixed=_fixed_liouvillian_scalar_type(L)
    fixed===nothing&&return T
    promote_type(fixed,T)===fixed||throw(ArgumentError(
        "a compiled matrix-free PI Liouvillian with scalar type $fixed cannot accept $context scalar type $T without narrowing its block scratch; compile the model at the wider precision"))
    T
end

function _check_liouvillian_apply_types(destination,source,plan::LiouvillianPlan)
    plan_type=plan.Ttype;source_type=eltype(source);destination_type=eltype(destination)
    _check_liouvillian_source_precision(plan,source_type,"source")
    promote_type(destination_type,plan_type)===destination_type||throw(ArgumentError(
        "Liouvillian destination scalar type $destination_type cannot represent plan scalar type $plan_type"))
    nothing
end

_prepare_kernel!(::AbstractStaticPIKernel,::Nothing,b,t,p)=nothing
_prepare_kernel!(::Union{FactorizedLocalJumpPIKernel,
                         FactorizedLocalPBodyJumpPIKernel,
                         FusedStaticPIKernel},
                 ::StaticFactorizedGainKernelWorkspace,b,t,p)=nothing
_dynamic_ishermitian(operator::AbstractMatrix)=ishermitian(operator)
function _dynamic_ishermitian(operator::AbstractPIOperator)
    for sector in operator.basis.sectors
        block=coefficient_block(operator,sector)
        @inbounds for column in axes(block,2),row in 1:column
            block[row,column]==conj(block[column,row])||return false
        end
    end
    true
end
function _prepare_kernel!(kernel::InPlaceHamiltonianPIKernel,
                          work::InPlaceHamiltonianKernelWorkspace,b,t,p)
    _evaluate_time_operator!(work.operator,kernel.schedule,t,p)
    _check_dynamic_pbody_operator(kernel.builder,work.operator)
    _dynamic_ishermitian(work.operator)||throw(ArgumentError(
        "an in-place Hamiltonian callback produced a non-Hermitian operator"))
    _fill_dynamic_blocks!(
        work.blocks,kernel.builder,work.operator,
        work.cancellation_scratch)
    nothing
end
function _prepare_kernel!(kernel::InPlaceDissipatorPIKernel,
                          work::InPlaceDissipatorKernelWorkspace,b,t,p)
    _evaluate_time_operator!(work.operator,kernel.schedule,t,p)
    _check_dynamic_pbody_operator(kernel.builder,work.operator)
    _fill_dynamic_blocks!(
        work.blocks,kernel.builder,work.operator,
        work.cancellation_scratch)
    for s in eachindex(work.blocks)
        mul!(work.qblocks[s],adjoint(work.blocks[s]),work.blocks[s])
    end
    nothing
end
function _prepare_kernel!(kernel::InPlaceLocalPBodyJumpPIKernel,
                          work::InPlaceLocalPBodyJumpKernelWorkspace,b,t,p)
    _evaluate_time_operator!(work.operator,kernel.schedule,t,p)
    _check_dynamic_pbody_operator(kernel.builder,work.operator)
    mul!(work.qoperator,adjoint(work.operator),work.operator)
    _fill_dynamic_blocks!(
        work.qblocks,kernel.builder,work.qoperator,
        work.cancellation_scratch)
    for index in eachindex(work.contractions)
        _path_contractions!(work.contractions[index],kernel.left_isometries[index],
                            kernel.right_isometries[index],work.operator)
    end
    nothing
end

function _fill_local_gain_contractions!(destination,branches,operator,
                                        channel::Int=1)
    branch_count=length(branches.entries)
    first_index=(channel-1)*branch_count
    @inbounds for branch_index in 1:branch_count
        branch=branches.entries[branch_index]
        contraction=destination[first_index+branch_index]
        table=branch.table
        for column in axes(contraction,2),row in axes(contraction,1)
            contraction[row,column]=_contract(table[row,column],operator)
        end
    end
    destination
end

function _prepare_kernel!(kernel::InPlaceLocalJumpPIKernel,
                          work::InPlaceLocalJumpKernelWorkspace,b,t,p)
    _evaluate_time_operator!(work.operator,kernel.schedule,t,p)
    mul!(work.qoperator,adjoint(work.operator),work.operator)
    _fill_dynamic_blocks!(work.qblocks,
        CollectiveOneBodyBlockBuilder(kernel.geometry),work.qoperator)
    _fill_local_gain_contractions!(
        work.contractions,kernel.branches,work.operator)
    nothing
end
function _prepare_kernel!(kernel::InPlaceCorrelatedCollectiveJumpPIKernel,
        work::InPlaceCorrelatedCollectiveJumpKernelWorkspace,b,t,p)
    _evaluate_time_operator!(work.kossakowski,kernel.schedule,t,p)
    rank=_factor_kossakowski!(work.factor,work.residual,work.kossakowski,
                              kernel.atol,kernel.rtol)
    work.rank[]=rank
    for sector in eachindex(b.sectors)
        fill!(work.qblocks[sector],zero(eltype(work.qblocks[sector])))
    end
    for channel in 1:rank
        effective=work.effective_blocks[channel]
        for sector in eachindex(b.sectors)
            block=effective[sector];fill!(block,zero(eltype(block)))
            @inbounds for operator_index in eachindex(kernel.channel_blocks)
                coefficient=work.factor[operator_index,channel]
                source=kernel.channel_blocks[operator_index][sector]
                for index in eachindex(block,source)
                    block[index]+=coefficient*source[index]
                end
            end
            scratch=work.qscratch[sector]
            mul!(scratch,adjoint(block),block)
            @inbounds for index in eachindex(work.qblocks[sector],scratch)
                work.qblocks[sector][index]+=scratch[index]
            end
        end
    end
    nothing
end

function _prepare_kernel!(kernel::InPlaceCorrelatedLocalJumpPIKernel,
        work::InPlaceCorrelatedLocalJumpKernelWorkspace,b,t,p)
    _evaluate_time_operator!(work.kossakowski,kernel.schedule,t,p)
    rank=_factor_kossakowski!(work.factor,work.residual,work.kossakowski,
                              kernel.atol,kernel.rtol)
    work.rank[]=rank
    fill!(work.qoperator,zero(eltype(work.qoperator)))
    for channel in 1:rank
        effective=work.effective_operators[channel]
        fill!(effective,zero(eltype(effective)))
        @inbounds for operator_index in eachindex(kernel.operators)
            coefficient=work.factor[operator_index,channel]
            operator=kernel.operators[operator_index]
            for index in eachindex(effective,operator)
                effective[index]+=coefficient*operator[index]
            end
        end
        mul!(work.qscratch,adjoint(effective),effective)
        @inbounds for index in eachindex(work.qoperator,work.qscratch)
            work.qoperator[index]+=work.qscratch[index]
        end
        _fill_local_gain_contractions!(
            work.contractions,kernel.branches,effective,channel)
    end
    _fill_dynamic_blocks!(work.qblocks,
        CollectiveOneBodyBlockBuilder(kernel.geometry),work.qoperator)
    nothing
end

# BLAS dispatch dominates the arithmetic for the small rectangular Schur
# blocks common at modest N and d.  This fixed-loop sandwich preserves the
# compact factors and O(n^3) arithmetic without retaining a quartic map.  Once
# either dimension is larger, the ordinary two-GEMM route wins and remains the
# default.
const _RECTANGULAR_GAIN_MICROKERNEL_DIMENSION=8
function _rectangular_sandwich!(target,scratch,operator,source)
    output_dimension,input_dimension=size(operator)
    if max(output_dimension,input_dimension)<=
            _RECTANGULAR_GAIN_MICROKERNEL_DIMENSION
        @inbounds for column in 1:input_dimension,row in 1:output_dimension
            value=zero(eltype(scratch))
            for inner in 1:input_dimension
                value+=operator[row,inner]*source[inner,column]
            end
            scratch[row,column]=value
        end
        @inbounds for column in 1:output_dimension,row in 1:output_dimension
            value=zero(eltype(target))
            for inner in 1:input_dimension
                value+=scratch[row,inner]*conj(operator[column,inner])
            end
            target[row,column]=value
        end
    else
        mul!(scratch,operator,source)
        mul!(target,scratch,LinearAlgebra.adjoint(operator))
    end
    target
end

function _add_rectangular_sandwich!(destination,output_offset,target,scratch,
        operator,source,scale;adjoint::Bool=false)
    effective=adjoint ? LinearAlgebra.adjoint(operator) : operator
    _rectangular_sandwich!(target,scratch,effective,source)
    @inbounds for index in eachindex(target)
        destination[output_offset+index-1]+=scale*target[index]
    end
    destination
end

function _add_rectangular_sandwich!(destination,output_offset,target,scratch,
        contraction::_StaticOneBodyContraction,source,scale;
        adjoint::Bool=false)
    if !contraction.use_support
        return _add_rectangular_sandwich!(destination,output_offset,target,
            scratch,contraction.matrix,source,scale;adjoint)
    end
    rows=contraction.output_rows;columns=contraction.input_columns
    values=contraction.values
    if adjoint
        output_dimension=size(contraction,2)
        @inbounds for right in eachindex(values)
            output_column=columns[right]
            input_column=rows[right]
            right_value=values[right]
            for left in eachindex(values)
                output_row=columns[left]
                input_row=rows[left]
                coordinate=output_offset+output_row-1+
                    (output_column-1)*output_dimension
                destination[coordinate]+=scale*conj(values[left])*
                    right_value*source[input_row,input_column]
            end
        end
    else
        output_dimension=size(contraction,1)
        @inbounds for right in eachindex(values)
            output_column=rows[right]
            input_column=columns[right]
            right_value=values[right]
            for left in eachindex(values)
                output_row=rows[left]
                input_row=columns[left]
                coordinate=output_offset+output_row-1+
                    (output_column-1)*output_dimension
                destination[coordinate]+=scale*values[left]*
                    conj(right_value)*source[input_row,input_column]
            end
        end
    end
    destination
end

function _apply_factorized_onebody_gain!(y,branches,contractions,
        gain_scratch,b,scale,work,channels::Int;adjoint::Bool=false)
    branch_count=length(branches.entries)
    @inbounds for branch_index in 1:branch_count
        branch=branches.entries[branch_index]
        li=branch.output_sector;ni=branch.input_sector
        nl=length(b.patterns[li]);nn=length(b.patterns[ni])
        source_sector=adjoint ? li : ni
        target_sector=adjoint ? ni : li
        source=work[source_sector][3]
        target=work[target_sector][1]
        target_offset=b.offsets[target_sector]
        scratch=adjoint ? @view(gain_scratch[1:nn,1:nl]) :
                          @view(gain_scratch[1:nl,1:nn])
        branch_scale=scale*branch.scale
        for channel in 1:channels
            contraction=contractions[(channel-1)*branch_count+branch_index]
            _add_rectangular_sandwich!(y,target_offset,target,scratch,
                contraction,source,branch_scale;adjoint)
        end
    end
    nothing
end

function _apply_factorized_onebody_gain_batch!(Y,X,branches,contractions,
        b,scale,scratch,channels::Int;adjoint::Bool=false)
    branch_count=length(branches.entries)
    @inbounds for branch_index in 1:branch_count
        branch=branches.entries[branch_index]
        li=branch.output_sector;ni=branch.input_sector
        nl=length(b.patterns[li]);nn=length(b.patterns[ni])
        output_sector=adjoint ? ni : li
        input_sector=adjoint ? li : ni
        output_dimension=adjoint ? nn : nl
        input_dimension=adjoint ? nl : nn
        branch_scale=scale*branch.scale
        for channel in 1:channels
            contraction=contractions[(channel-1)*branch_count+branch_index]
            operator=adjoint ? LinearAlgebra.adjoint(contraction) : contraction
            _batch_add_rectangular_sandwich!(Y,X,
                b.offsets[output_sector],b.offsets[input_sector],
                output_dimension,input_dimension,operator,branch_scale,scratch)
        end
    end
    nothing
end

function _apply_factorized_pbody_gain_batch!(Y,X,groups,contractions,
        pair_scales,b,scale,scratch;adjoint::Bool=false)
    @inbounds for (li,ni,first_pair,last_pair) in groups
        output_sector=adjoint ? ni : li
        input_sector=adjoint ? li : ni
        output_dimension=length(b.patterns[output_sector])
        input_dimension=length(b.patterns[input_sector])
        for pair in first_pair:last_pair
            contraction=contractions[pair]
            operator=adjoint ? LinearAlgebra.adjoint(contraction) : contraction
            exact_scale=pair_scales[pair]
            pair_scale=exact_scale.direct ? scale*exact_scale.factor :
                _apply_prepared_exact_scale(scale,exact_scale;
                    context="batched local p-body gain contribution")
            _batch_add_rectangular_sandwich!(Y,X,
                b.offsets[output_sector],b.offsets[input_sector],
                output_dimension,input_dimension,operator,pair_scale,scratch)
        end
    end
    nothing
end

function _apply_local_jump_anticommutator_batch!(Y,X,qblocks,b,scale,scratch;
                                                 adjoint::Bool=false)
    for sector in eachindex(b.sectors)
        dimension=length(b.patterns[sector]);offset=b.offsets[sector]
        qblock=adjoint ? LinearAlgebra.adjoint(qblocks[sector]) : qblocks[sector]
        _batch_add_left_right!(Y,X,offset,dimension,qblock,qblock,
            -scale/2,-scale/2,scratch)
    end
    nothing
end

_apply_prepared_kernel!(y,x,kernel::AbstractStaticPIKernel,::Nothing,b,t,p,work)=
    _apply_kernel!(y,x,kernel,b,t,p,work)
_apply_adjoint_prepared_kernel!(y,x,kernel::AbstractStaticPIKernel,::Nothing,b,t,p,work)=
    _apply_adjoint_kernel!(y,x,kernel,b,t,p,work)

function _apply_prepared_kernel!(y,x,kernel::InPlaceHamiltonianPIKernel,
                                 prepared::InPlaceHamiltonianKernelWorkspace,
                                 b,t,p,work)
    _apply_kernel!(y,x,HamiltonianPIKernel(prepared.blocks,kernel.scale),b,t,p,work)
end
function _apply_adjoint_prepared_kernel!(y,x,kernel::InPlaceHamiltonianPIKernel,
                                         prepared::InPlaceHamiltonianKernelWorkspace,
                                         b,t,p,work)
    _apply_adjoint_kernel!(y,x,HamiltonianPIKernel(prepared.blocks,kernel.scale),b,t,p,work)
end
function _apply_prepared_kernel!(y,x,kernel::InPlaceDissipatorPIKernel,
                                 prepared::InPlaceDissipatorKernelWorkspace,
                                 b,t,p,work)
    _apply_kernel!(y,x,DissipatorPIKernel(prepared.blocks,prepared.qblocks,kernel.scale),
                   b,t,p,work)
end
function _apply_adjoint_prepared_kernel!(y,x,kernel::InPlaceDissipatorPIKernel,
                                         prepared::InPlaceDissipatorKernelWorkspace,
                                         b,t,p,work)
    _apply_adjoint_kernel!(y,x,
        DissipatorPIKernel(prepared.blocks,prepared.qblocks,kernel.scale),b,t,p,work)
end
function _apply_prepared_kernel!(y,x,kernel::InPlaceLocalJumpPIKernel,
                                 prepared::InPlaceLocalJumpKernelWorkspace,
                                 b,t,p,work)
    scale=convert(eltype(work[1][1]),
                  _evaluated_dissipative_rate(kernel.scale,t,p))
    _copy_input_blocks!(work,x,b)
    _apply_factorized_onebody_gain!(y,kernel.branches,prepared.contractions,
        prepared.gain_scratch,b,scale,work,1)
    _apply_local_jump_anticommutator!(
        y,prepared.qblocks,b,scale,work)
end
function _apply_adjoint_prepared_kernel!(y,x,kernel::InPlaceLocalJumpPIKernel,
                                         prepared::InPlaceLocalJumpKernelWorkspace,
                                         b,t,p,work)
    scale=conj(convert(eltype(work[1][1]),
                       _evaluated_dissipative_rate(kernel.scale,t,p)))
    _copy_input_blocks!(work,x,b)
    _apply_factorized_onebody_gain!(y,kernel.branches,prepared.contractions,
        prepared.gain_scratch,b,scale,work,1;adjoint=true)
    _apply_local_jump_anticommutator!(
        y,prepared.qblocks,b,scale,work;adjoint=true)
end
function _apply_prepared_kernel!(y,x,
        kernel::InPlaceCorrelatedCollectiveJumpPIKernel,
        prepared::InPlaceCorrelatedCollectiveJumpKernelWorkspace,b,t,p,work)
    scale=convert(eltype(work[1][1]),
                  _evaluated_dissipative_rate(kernel.scale,t,p))
    for sector in eachindex(b.sectors)
        n=length(b.patterns[sector]);offset=b.offsets[sector]
        left,right,input=work[sector]
        copyto!(input,1,x,offset,n*n)
        for channel in 1:prepared.rank[]
            block=prepared.effective_blocks[channel][sector]
            mul!(left,block,input);mul!(right,left,adjoint(block))
            @inbounds for index in eachindex(right)
                y[offset+index-1]+=scale*right[index]
            end
        end
        qblock=prepared.qblocks[sector]
        mul!(left,qblock,input);mul!(right,input,qblock)
        @inbounds for index in eachindex(left)
            y[offset+index-1]-=(scale/2)*(left[index]+right[index])
        end
    end
    nothing
end
function _apply_adjoint_prepared_kernel!(y,x,
        kernel::InPlaceCorrelatedCollectiveJumpPIKernel,
        prepared::InPlaceCorrelatedCollectiveJumpKernelWorkspace,b,t,p,work)
    scale=conj(convert(eltype(work[1][1]),
                       _evaluated_dissipative_rate(kernel.scale,t,p)))
    for sector in eachindex(b.sectors)
        n=length(b.patterns[sector]);offset=b.offsets[sector]
        left,right,input=work[sector]
        copyto!(input,1,x,offset,n*n)
        for channel in 1:prepared.rank[]
            block=prepared.effective_blocks[channel][sector]
            mul!(left,adjoint(block),input);mul!(right,left,block)
            @inbounds for index in eachindex(right)
                y[offset+index-1]+=scale*right[index]
            end
        end
        qblock=prepared.qblocks[sector]
        mul!(left,adjoint(qblock),input);mul!(right,input,adjoint(qblock))
        @inbounds for index in eachindex(left)
            y[offset+index-1]-=(scale/2)*(left[index]+right[index])
        end
    end
    nothing
end
function _apply_prepared_kernel!(y,x,kernel::InPlaceCorrelatedLocalJumpPIKernel,
        prepared::InPlaceCorrelatedLocalJumpKernelWorkspace,b,t,p,work)
    scale=convert(eltype(work[1][1]),
                  _evaluated_dissipative_rate(kernel.scale,t,p))
    _copy_input_blocks!(work,x,b)
    _apply_factorized_onebody_gain!(y,kernel.branches,prepared.contractions,
        prepared.gain_scratch,b,scale,work,prepared.rank[])
    _apply_local_jump_anticommutator!(
        y,prepared.qblocks,b,scale,work)
end
function _apply_adjoint_prepared_kernel!(y,x,
        kernel::InPlaceCorrelatedLocalJumpPIKernel,
        prepared::InPlaceCorrelatedLocalJumpKernelWorkspace,b,t,p,work)
    scale=conj(convert(eltype(work[1][1]),
                       _evaluated_dissipative_rate(kernel.scale,t,p)))
    _copy_input_blocks!(work,x,b)
    _apply_factorized_onebody_gain!(y,kernel.branches,prepared.contractions,
        prepared.gain_scratch,b,scale,work,prepared.rank[];adjoint=true)
    _apply_local_jump_anticommutator!(
        y,prepared.qblocks,b,scale,work;adjoint=true)
end

# Batched prepared paths reuse the dynamic data evaluated above, but contract
# all right-hand sides sectorwise. Rare specialized kernels without a batch
# contraction retain a correctness-preserving column fallback; they still
# prepare operator schedules only once per batch.
_apply_prepared_batch_kernel!(Y,X,kernel::AbstractStaticPIKernel,::Nothing,
        b,t,p,work::LiouvillianWorkspace)=
    _apply_kernel_batch!(Y,X,kernel,b,t,p,work.batch)
_apply_adjoint_prepared_batch_kernel!(Y,X,kernel::AbstractStaticPIKernel,
        ::Nothing,b,t,p,work::LiouvillianWorkspace)=
    _apply_adjoint_kernel_batch!(Y,X,kernel,b,t,p,work.batch)

function _apply_prepared_batch_kernel!(Y,X,kernel::InPlaceHamiltonianPIKernel,
        prepared::InPlaceHamiltonianKernelWorkspace,b,t,p,work)
    _apply_kernel_batch!(Y,X,
        HamiltonianPIKernel(prepared.blocks,kernel.scale),b,t,p,work.batch)
end
function _apply_adjoint_prepared_batch_kernel!(Y,X,
        kernel::InPlaceHamiltonianPIKernel,
        prepared::InPlaceHamiltonianKernelWorkspace,b,t,p,work)
    _apply_adjoint_kernel_batch!(Y,X,
        HamiltonianPIKernel(prepared.blocks,kernel.scale),b,t,p,work.batch)
end
function _apply_prepared_batch_kernel!(Y,X,kernel::InPlaceDissipatorPIKernel,
        prepared::InPlaceDissipatorKernelWorkspace,b,t,p,work)
    _apply_kernel_batch!(Y,X,DissipatorPIKernel(
        prepared.blocks,prepared.qblocks,kernel.scale),b,t,p,work.batch)
end
function _apply_adjoint_prepared_batch_kernel!(Y,X,
        kernel::InPlaceDissipatorPIKernel,
        prepared::InPlaceDissipatorKernelWorkspace,b,t,p,work)
    _apply_adjoint_kernel_batch!(Y,X,DissipatorPIKernel(
        prepared.blocks,prepared.qblocks,kernel.scale),b,t,p,work.batch)
end
function _apply_prepared_batch_kernel!(Y,X,kernel::InPlaceLocalJumpPIKernel,
        prepared::InPlaceLocalJumpKernelWorkspace,b,t,p,work)
    scale=convert(eltype(work.batch.input),
                  _evaluated_dissipative_rate(kernel.scale,t,p))
    _apply_factorized_onebody_gain_batch!(Y,X,kernel.branches,
        prepared.contractions,b,scale,work.batch,1)
    _apply_local_jump_anticommutator_batch!(
        Y,X,prepared.qblocks,b,scale,work.batch)
end
function _apply_adjoint_prepared_batch_kernel!(Y,X,
        kernel::InPlaceLocalJumpPIKernel,
        prepared::InPlaceLocalJumpKernelWorkspace,b,t,p,work)
    scale=conj(convert(eltype(work.batch.input),
                       _evaluated_dissipative_rate(kernel.scale,t,p)))
    _apply_factorized_onebody_gain_batch!(Y,X,kernel.branches,
        prepared.contractions,b,scale,work.batch,1;adjoint=true)
    _apply_local_jump_anticommutator_batch!(
        Y,X,prepared.qblocks,b,scale,work.batch;adjoint=true)
end
function _apply_prepared_batch_kernel!(Y,X,
        kernel::InPlaceCorrelatedLocalJumpPIKernel,
        prepared::InPlaceCorrelatedLocalJumpKernelWorkspace,b,t,p,work)
    scale=convert(eltype(work.batch.input),
                  _evaluated_dissipative_rate(kernel.scale,t,p))
    _apply_factorized_onebody_gain_batch!(Y,X,kernel.branches,
        prepared.contractions,b,scale,work.batch,prepared.rank[])
    _apply_local_jump_anticommutator_batch!(
        Y,X,prepared.qblocks,b,scale,work.batch)
end
function _apply_adjoint_prepared_batch_kernel!(Y,X,
        kernel::InPlaceCorrelatedLocalJumpPIKernel,
        prepared::InPlaceCorrelatedLocalJumpKernelWorkspace,b,t,p,work)
    scale=conj(convert(eltype(work.batch.input),
                       _evaluated_dissipative_rate(kernel.scale,t,p)))
    _apply_factorized_onebody_gain_batch!(Y,X,kernel.branches,
        prepared.contractions,b,scale,work.batch,prepared.rank[];adjoint=true)
    _apply_local_jump_anticommutator_batch!(
        Y,X,prepared.qblocks,b,scale,work.batch;adjoint=true)
end

function _apply_prepared_batch_kernel!(Y,X,
        kernel::FactorizedLocalJumpPIKernel,
        prepared::StaticFactorizedGainKernelWorkspace,b,t,p,work)
    scale=convert(eltype(work.batch.input),
                  _evaluated_dissipative_rate(kernel.scale,t,p))
    _apply_factorized_onebody_gain_batch!(Y,X,kernel.branches,
        kernel.contractions,b,scale,work.batch,1)
    _apply_local_jump_anticommutator_batch!(
        Y,X,kernel.qblocks,b,scale,work.batch)
end
function _apply_adjoint_prepared_batch_kernel!(Y,X,
        kernel::FactorizedLocalJumpPIKernel,
        prepared::StaticFactorizedGainKernelWorkspace,b,t,p,work)
    scale=conj(convert(eltype(work.batch.input),
                       _evaluated_dissipative_rate(kernel.scale,t,p)))
    _apply_factorized_onebody_gain_batch!(Y,X,kernel.branches,
        kernel.contractions,b,scale,work.batch,1;adjoint=true)
    _apply_local_jump_anticommutator_batch!(
        Y,X,kernel.qblocks,b,scale,work.batch;adjoint=true)
end

function _apply_prepared_batch_kernel!(Y,X,
        kernel::FactorizedLocalPBodyJumpPIKernel,
        prepared::StaticFactorizedGainKernelWorkspace,b,t,p,work)
    scale=convert(eltype(work.batch.input),
                  _evaluated_dissipative_rate(kernel.scale,t,p))
    _apply_factorized_pbody_gain_batch!(Y,X,kernel.groups,
        kernel.contractions,kernel.pair_scales,b,scale,work.batch)
    _apply_local_jump_anticommutator_batch!(
        Y,X,kernel.qblocks,b,scale,work.batch)
end
function _apply_adjoint_prepared_batch_kernel!(Y,X,
        kernel::FactorizedLocalPBodyJumpPIKernel,
        prepared::StaticFactorizedGainKernelWorkspace,b,t,p,work)
    scale=conj(convert(eltype(work.batch.input),
                       _evaluated_dissipative_rate(kernel.scale,t,p)))
    _apply_factorized_pbody_gain_batch!(Y,X,kernel.groups,
        kernel.contractions,kernel.pair_scales,b,scale,work.batch;adjoint=true)
    _apply_local_jump_anticommutator_batch!(
        Y,X,kernel.qblocks,b,scale,work.batch;adjoint=true)
end

function _apply_prepared_batch_kernel!(Y,X,kernel::FusedStaticPIKernel,
        prepared::StaticFactorizedGainKernelWorkspace,b,t,p,work)
    if kernel.hamiltonian_blocks!==nothing
        for sector in eachindex(b.sectors)
            dimension=length(b.patterns[sector]);offset=b.offsets[sector]
            block=kernel.hamiltonian_blocks[sector]
            _batch_add_left_right!(Y,X,offset,dimension,block,block,
                -1im,1im,work.batch)
        end
    end
    for gain in kernel.collective_gains,sector in eachindex(b.sectors)
        dimension=length(b.patterns[sector]);offset=b.offsets[sector]
        block=gain.blocks[sector]
        _batch_add_sandwich!(Y,X,offset,dimension,block,adjoint(block),
                             gain.scale,work.batch)
    end
    if kernel.loss_blocks!==nothing
        _apply_local_jump_anticommutator_batch!(
            Y,X,kernel.loss_blocks,b,one(eltype(work.batch.input)),work.batch)
    end
    for gain in kernel.onebody_gains
        _apply_factorized_onebody_gain_batch!(Y,X,gain.branches,
            gain.contractions,b,gain.scale,work.batch,1)
    end
    for gain in kernel.pbody_gains
        _apply_factorized_pbody_gain_batch!(Y,X,gain.groups,
            gain.contractions,gain.pair_scales,b,gain.scale,work.batch)
    end
    Y
end

function _apply_adjoint_prepared_batch_kernel!(Y,X,
        kernel::FusedStaticPIKernel,
        prepared::StaticFactorizedGainKernelWorkspace,b,t,p,work)
    if kernel.hamiltonian_blocks!==nothing
        for sector in eachindex(b.sectors)
            dimension=length(b.patterns[sector]);offset=b.offsets[sector]
            block=adjoint(kernel.hamiltonian_blocks[sector])
            _batch_add_left_right!(Y,X,offset,dimension,block,block,
                1im,-1im,work.batch)
        end
    end
    for gain in kernel.collective_gains,sector in eachindex(b.sectors)
        dimension=length(b.patterns[sector]);offset=b.offsets[sector]
        block=gain.blocks[sector]
        _batch_add_sandwich!(Y,X,offset,dimension,adjoint(block),block,
                             conj(gain.scale),work.batch)
    end
    if kernel.loss_blocks!==nothing
        _apply_local_jump_anticommutator_batch!(
            Y,X,kernel.loss_blocks,b,one(eltype(work.batch.input)),work.batch;
            adjoint=true)
    end
    for gain in kernel.onebody_gains
        _apply_factorized_onebody_gain_batch!(Y,X,gain.branches,
            gain.contractions,b,conj(gain.scale),work.batch,1;adjoint=true)
    end
    for gain in kernel.pbody_gains
        _apply_factorized_pbody_gain_batch!(Y,X,gain.groups,
            gain.contractions,gain.pair_scales,b,conj(gain.scale),work.batch;
            adjoint=true)
    end
    Y
end

function _apply_prepared_batch_kernel!(Y,X,kernel,prepared,b,t,p,work)
    for column in axes(X,2)
        _apply_prepared_kernel!(view(Y,:,column),view(X,:,column),kernel,
            prepared,b,t,p,work.blocks)
    end
    Y
end
function _apply_adjoint_prepared_batch_kernel!(Y,X,kernel,prepared,b,t,p,work)
    for column in axes(X,2)
        _apply_adjoint_prepared_kernel!(view(Y,:,column),view(X,:,column),
            kernel,prepared,b,t,p,work.blocks)
    end
    Y
end

# Appendix-D local gain maps have a Kraus-like factorization for every
# compatible pair of Schur paths.  If `C` is the evaluated rectangular path
# contraction from input sector `nu` to output sector `lambda`, its complete
# four-index contribution is
#
#     X_nu -> q * C * X_nu * C'
#
# with the exact symmetric-group scale `q`.  Applying that factorization
# directly avoids retaining the O(n_PI^2) coordinate I/J/value tables used by
# a generic sparse gain map.  The caller-owned block and rectangular scratch
# matrices make both the forward and Frobenius-adjoint paths allocation-free.
function _copy_input_blocks!(work,x,b)
    for s in eachindex(b.sectors)
        n=length(b.patterns[s]);off=b.offsets[s]
        copyto!(work[s][3],1,x,off,n*n)
    end
    nothing
end

function _apply_factorized_pbody_gain!(y,groups,contractions,pair_scales,
        gain_scratch,b,scale,work)
    @inbounds for (li,ni,first_pair,last_pair) in groups
        nl=length(b.patterns[li]);nn=length(b.patterns[ni])
        input=work[ni][3];output=work[li][1]
        scratch=@view gain_scratch[1:nl,1:nn]
        output_offset=b.offsets[li]
        for pair in first_pair:last_pair
            contraction=contractions[pair]
            _rectangular_sandwich!(output,scratch,contraction,input)
            exact_scale=pair_scales[pair]
            if exact_scale.direct
                pair_scale=scale*exact_scale.factor
                for index in eachindex(output)
                    y[output_offset+index-1]+=pair_scale*output[index]
                end
            else
                for index in eachindex(output)
                    contribution=scale*output[index]
                    y[output_offset+index-1]+=_apply_prepared_exact_scale(
                        contribution,exact_scale;
                        context="dynamic local p-body gain contribution")
                end
            end
        end
    end
    nothing
end

function _apply_adjoint_factorized_pbody_gain!(y,groups,contractions,
        pair_scales,gain_scratch,b,scale,work)
    @inbounds for (li,ni,first_pair,last_pair) in groups
        nl=length(b.patterns[li]);nn=length(b.patterns[ni])
        input=work[li][3];output=work[ni][1]
        scratch=@view gain_scratch[1:nn,1:nl]
        output_offset=b.offsets[ni]
        for pair in first_pair:last_pair
            contraction=contractions[pair]
            _rectangular_sandwich!(output,scratch,adjoint(contraction),input)
            exact_scale=pair_scales[pair]
            if exact_scale.direct
                pair_scale=scale*exact_scale.factor
                for index in eachindex(output)
                    y[output_offset+index-1]+=pair_scale*output[index]
                end
            else
                for index in eachindex(output)
                    contribution=scale*output[index]
                    y[output_offset+index-1]+=_apply_prepared_exact_scale(
                        contribution,exact_scale;
                        context="adjoint dynamic local p-body gain contribution")
                end
            end
        end
    end
    nothing
end

function _apply_local_jump_anticommutator!(y,qblocks,b,scale,work;adjoint=false)
    for s in eachindex(b.sectors)
        off=b.offsets[s];A,B,X=work[s];Q=qblocks[s]
        effective_q=adjoint ? LinearAlgebra.adjoint(Q) : Q
        mul!(A,effective_q,X);mul!(B,X,effective_q)
        @inbounds for index in eachindex(A)
            y[off+index-1]-=(scale/2)*(A[index]+B[index])
        end
    end
    nothing
end

function _apply_prepared_kernel!(y,x,kernel::FactorizedLocalJumpPIKernel,
        prepared::StaticFactorizedGainKernelWorkspace,b,t,p,work)
    scale=convert(eltype(work[1][1]),
                  _evaluated_dissipative_rate(kernel.scale,t,p))
    _copy_input_blocks!(work,x,b)
    _apply_factorized_onebody_gain!(y,kernel.branches,kernel.contractions,
        prepared.gain_scratch,b,scale,work,1)
    _apply_local_jump_anticommutator!(y,kernel.qblocks,b,scale,work)
end

function _apply_adjoint_prepared_kernel!(y,x,
        kernel::FactorizedLocalJumpPIKernel,
        prepared::StaticFactorizedGainKernelWorkspace,b,t,p,work)
    scale=conj(convert(eltype(work[1][1]),
                       _evaluated_dissipative_rate(kernel.scale,t,p)))
    _copy_input_blocks!(work,x,b)
    _apply_factorized_onebody_gain!(y,kernel.branches,kernel.contractions,
        prepared.gain_scratch,b,scale,work,1;adjoint=true)
    _apply_local_jump_anticommutator!(
        y,kernel.qblocks,b,scale,work;adjoint=true)
end

function _apply_prepared_kernel!(y,x,kernel::FactorizedLocalPBodyJumpPIKernel,
        prepared::StaticFactorizedGainKernelWorkspace,b,t,p,work)
    scale=convert(eltype(work[1][1]),
                  _evaluated_dissipative_rate(kernel.scale,t,p))
    _copy_input_blocks!(work,x,b)
    _apply_factorized_pbody_gain!(y,kernel.groups,kernel.contractions,
        kernel.pair_scales,prepared.gain_scratch,b,scale,work)
    _apply_local_jump_anticommutator!(y,kernel.qblocks,b,scale,work)
end

function _apply_adjoint_prepared_kernel!(y,x,
        kernel::FactorizedLocalPBodyJumpPIKernel,
        prepared::StaticFactorizedGainKernelWorkspace,b,t,p,work)
    scale=conj(convert(eltype(work[1][1]),
                       _evaluated_dissipative_rate(kernel.scale,t,p)))
    _copy_input_blocks!(work,x,b)
    _apply_adjoint_factorized_pbody_gain!(y,kernel.groups,kernel.contractions,
        kernel.pair_scales,prepared.gain_scratch,b,scale,work)
    _apply_local_jump_anticommutator!(
        y,kernel.qblocks,b,scale,work;adjoint=true)
end

function _apply_prepared_kernel!(y,x,kernel::FusedStaticPIKernel,
        prepared::StaticFactorizedGainKernelWorkspace,b,t,p,work)
    _copy_input_blocks!(work,x,b)
    if kernel.hamiltonian_blocks!==nothing
        for sector in eachindex(b.sectors)
            offset=b.offsets[sector];left,right,input=work[sector]
            block=kernel.hamiltonian_blocks[sector]
            mul!(left,block,input);mul!(right,input,block)
            @inbounds for index in eachindex(left)
                y[offset+index-1]+=-1im*(left[index]-right[index])
            end
        end
    end
    for gain in kernel.collective_gains
        for sector in eachindex(b.sectors)
            offset=b.offsets[sector];left,right,input=work[sector]
            block=gain.blocks[sector]
            mul!(left,block,input);mul!(right,left,adjoint(block))
            @inbounds for index in eachindex(right)
                y[offset+index-1]+=gain.scale*right[index]
            end
        end
    end
    if kernel.loss_blocks!==nothing
        _apply_local_jump_anticommutator!(
            y,kernel.loss_blocks,b,one(eltype(work[1][1])),work)
    end
    for gain in kernel.onebody_gains
        _apply_factorized_onebody_gain!(y,gain.branches,gain.contractions,
            prepared.gain_scratch,b,gain.scale,work,1)
    end
    for gain in kernel.pbody_gains
        _apply_factorized_pbody_gain!(y,gain.groups,gain.contractions,
            gain.pair_scales,prepared.gain_scratch,b,gain.scale,work)
    end
    nothing
end

function _apply_adjoint_prepared_kernel!(y,x,kernel::FusedStaticPIKernel,
        prepared::StaticFactorizedGainKernelWorkspace,b,t,p,work)
    _copy_input_blocks!(work,x,b)
    if kernel.hamiltonian_blocks!==nothing
        for sector in eachindex(b.sectors)
            offset=b.offsets[sector];left,right,input=work[sector]
            block=adjoint(kernel.hamiltonian_blocks[sector])
            mul!(left,block,input);mul!(right,input,block)
            @inbounds for index in eachindex(left)
                y[offset+index-1]+=1im*(left[index]-right[index])
            end
        end
    end
    for gain in kernel.collective_gains
        scale=conj(gain.scale)
        for sector in eachindex(b.sectors)
            offset=b.offsets[sector];left,right,input=work[sector]
            block=gain.blocks[sector]
            mul!(left,adjoint(block),input);mul!(right,left,block)
            @inbounds for index in eachindex(right)
                y[offset+index-1]+=scale*right[index]
            end
        end
    end
    if kernel.loss_blocks!==nothing
        _apply_local_jump_anticommutator!(y,kernel.loss_blocks,b,
            one(eltype(work[1][1])),work;adjoint=true)
    end
    for gain in kernel.onebody_gains
        _apply_factorized_onebody_gain!(y,gain.branches,gain.contractions,
            prepared.gain_scratch,b,conj(gain.scale),work,1;adjoint=true)
    end
    for gain in kernel.pbody_gains
        _apply_adjoint_factorized_pbody_gain!(y,gain.groups,gain.contractions,
            gain.pair_scales,prepared.gain_scratch,b,conj(gain.scale),work)
    end
    nothing
end

function _apply_prepared_kernel!(y,x,kernel::InPlaceLocalPBodyJumpPIKernel,
        prepared::InPlaceLocalPBodyJumpKernelWorkspace,b,t,p,work)
    scale=convert(eltype(work[1][1]),
                  _evaluated_dissipative_rate(kernel.scale,t,p))
    _copy_input_blocks!(work,x,b)
    _apply_factorized_pbody_gain!(y,kernel.groups,prepared.contractions,
        kernel.pair_scales,prepared.gain_scratch,b,scale,work)
    _apply_local_jump_anticommutator!(y,prepared.qblocks,b,scale,work)
end
function _apply_adjoint_prepared_kernel!(y,x,kernel::InPlaceLocalPBodyJumpPIKernel,
        prepared::InPlaceLocalPBodyJumpKernelWorkspace,b,t,p,work)
    scale=conj(convert(eltype(work[1][1]),
                       _evaluated_dissipative_rate(kernel.scale,t,p)))
    _copy_input_blocks!(work,x,b)
    _apply_adjoint_factorized_pbody_gain!(y,kernel.groups,prepared.contractions,
        kernel.pair_scales,prepared.gain_scratch,b,scale,work)
    _apply_local_jump_anticommutator!(y,prepared.qblocks,b,scale,work;adjoint=true)
end

# Recursion over heterogeneous tuples lets Julia specialize every term and
# matching workspace. A normal loop would box differently typed kernels.
@inline _prepare_kernels!(::Tuple{},::Tuple{},b,t,p)=nothing
@inline function _prepare_kernels!(kernels::Tuple{K,Vararg{Any}},
                                   prepared::Tuple{W,Vararg{Any}},b,t,p) where {K,W}
    _prepare_kernel!(first(kernels),first(prepared),b,t,p)
    _prepare_kernels!(Base.tail(kernels),Base.tail(prepared),b,t,p)
end

@inline _apply_kernels!(y,x,::Tuple{},::Tuple{},b,t,p,work)=nothing
@inline function _apply_kernels!(y,x,kernels::Tuple{K,Vararg{Any}},
                                 prepared::Tuple{W,Vararg{Any}},b,t,p,work) where {K,W}
    _apply_prepared_kernel!(y,x,first(kernels),first(prepared),b,t,p,work)
    _apply_kernels!(y,x,Base.tail(kernels),Base.tail(prepared),b,t,p,work)
end

@inline _apply_adjoint_kernels!(y,x,::Tuple{},::Tuple{},b,t,p,work)=nothing
@inline function _apply_adjoint_kernels!(y,x,kernels::Tuple{K,Vararg{Any}},
        prepared::Tuple{W,Vararg{Any}},b,t,p,work) where {K,W}
    _apply_adjoint_prepared_kernel!(y,x,first(kernels),first(prepared),b,t,p,work)
    _apply_adjoint_kernels!(y,x,Base.tail(kernels),Base.tail(prepared),b,t,p,work)
end

@inline _apply_batch_kernels!(Y,X,::Tuple{},::Tuple{},b,t,p,work)=nothing
@inline function _apply_batch_kernels!(Y,X,kernels::Tuple{K,Vararg{Any}},
        prepared::Tuple{W,Vararg{Any}},b,t,p,work) where {K,W}
    _apply_prepared_batch_kernel!(Y,X,first(kernels),first(prepared),
                                  b,t,p,work)
    _apply_batch_kernels!(Y,X,Base.tail(kernels),Base.tail(prepared),
                          b,t,p,work)
end

@inline _apply_adjoint_batch_kernels!(Y,X,::Tuple{},::Tuple{},b,t,p,work)=nothing
@inline function _apply_adjoint_batch_kernels!(Y,X,
        kernels::Tuple{K,Vararg{Any}},prepared::Tuple{W,Vararg{Any}},
        b,t,p,work) where {K,W}
    _apply_adjoint_prepared_batch_kernel!(Y,X,first(kernels),first(prepared),
                                          b,t,p,work)
    _apply_adjoint_batch_kernels!(Y,X,Base.tail(kernels),Base.tail(prepared),
                                  b,t,p,work)
end

"""Apply a compiled Liouvillian using caller-owned scratch."""
function _apply_prepared_vector!(y,plan,x,t,p,work)
    fill!(y,zero(eltype(y)))
    _apply_kernels!(y,x,plan.kernels,work.kernel_workspaces,
                    plan.basis,t,p,work.blocks)
    y
end

function apply!(y::AbstractVector,plan::LiouvillianPlan,x::AbstractVector,t,p,
                work::LiouvillianWorkspace)
    n=length(plan.basis)
    length(x)==n&&length(y)==n||throw(DimensionMismatch("Liouvillian vector has the wrong length"))
    _check_liouvillian_workspace(work,plan)
    _check_liouvillian_apply_types(y,x,plan)
    if plan.kernels===nothing
        # Raw operator functions retain the compatibility fallback. Use
        # InPlaceTimeOperator to prepare built-in schedules in caller scratch.
        return mul!(y,_matrix_at(plan.fallback_model,t,p),x)
    end
    _prepare_kernels!(plan.kernels,work.kernel_workspaces,plan.basis,t,p)
    _apply_prepared_vector!(y,plan,x,t,p,work)
end

function apply!(Y::AbstractMatrix,plan::LiouvillianPlan,X::AbstractMatrix,t,p,
                work::LiouvillianWorkspace)
    n=length(plan.basis)
    size(X,1)==n||throw(DimensionMismatch("matrix input has the wrong leading dimension"))
    size(Y)==(n,size(X,2))||throw(DimensionMismatch("matrix output has the wrong dimensions"))
    _check_liouvillian_workspace(work,plan)
    _check_liouvillian_apply_types(Y,X,plan)
    if plan.kernels===nothing
        # Freeze/lower once for the entire batch, rather than once per column.
        return mul!(Y,_matrix_at(plan.fallback_model,t,p),X)
    end
    _prepare_kernels!(plan.kernels,work.kernel_workspaces,plan.basis,t,p)
    fill!(Y,zero(eltype(Y)))
    _ensure_batch_capacity!(work.batch,size(X,2))
    _apply_batch_kernels!(Y,X,plan.kernels,work.kernel_workspaces,
                          plan.basis,t,p,work)
    Y
end

function apply!(y,plan::LiouvillianPlan,x,work::LiouvillianWorkspace)
    _require_autonomous(plan,"apply!")
    apply!(y,plan,x,0.0,nothing,work)
end

apply!(y,plan::LiouvillianPlan,x,t,p)=apply!(y,plan,x,t,p,LiouvillianWorkspace(plan))
function apply!(y,plan::LiouvillianPlan,x)
    _require_autonomous(plan,"apply!")
    apply!(y,plan,x,0.0,nothing)
end

"""
    apply_adjoint!(destination, L, input, time, parameters, workspace)
    apply_adjoint!(destination, L, input[, workspace])

Apply the adjoint PI Liouvillian in orthonormal coefficient coordinates.
`L` may be a prepared plan, compiled model, or matrix-free Liouvillian;
vector and batched matrix inputs are supported. The explicit-time form accepts
driven generators. Reuse one compatible `LiouvillianWorkspace` per task to
avoid scratch allocation; overloads without `time` require an autonomous
generator.
"""
function apply_adjoint!(y::AbstractVector,plan::LiouvillianPlan,x::AbstractVector,
                        t,p,work::LiouvillianWorkspace)
    n=length(plan.basis)
    length(x)==n&&length(y)==n||throw(DimensionMismatch("Liouvillian vector has the wrong length"))
    _check_liouvillian_workspace(work,plan)
    _check_liouvillian_apply_types(y,x,plan)
    if plan.kernels===nothing
        M=_matrix_at(plan.fallback_model,t,p)
        return mul!(y,adjoint(M),x)
    end
    _prepare_kernels!(plan.kernels,work.kernel_workspaces,plan.basis,t,p)
    fill!(y,zero(eltype(y)))
    _apply_adjoint_kernels!(y,x,plan.kernels,work.kernel_workspaces,
                            plan.basis,t,p,work.blocks)
    y
end


function apply_adjoint!(Y::AbstractMatrix,plan::LiouvillianPlan,X::AbstractMatrix,
                        t,p,work::LiouvillianWorkspace)
    n=length(plan.basis)
    size(X,1)==n||throw(DimensionMismatch("matrix input has the wrong leading dimension"))
    size(Y)==(n,size(X,2))||throw(DimensionMismatch("matrix output has the wrong dimensions"))
    _check_liouvillian_workspace(work,plan)
    _check_liouvillian_apply_types(Y,X,plan)
    if plan.kernels===nothing
        M=_matrix_at(plan.fallback_model,t,p)
        return mul!(Y,adjoint(M),X)
    end
    _prepare_kernels!(plan.kernels,work.kernel_workspaces,plan.basis,t,p)
    fill!(Y,zero(eltype(Y)))
    _ensure_batch_capacity!(work.batch,size(X,2))
    _apply_adjoint_batch_kernels!(Y,X,plan.kernels,work.kernel_workspaces,
                                  plan.basis,t,p,work)
    Y
end

function apply_adjoint!(y,plan::LiouvillianPlan,x,work::LiouvillianWorkspace)
    _require_autonomous(plan,"apply_adjoint!")
    apply_adjoint!(y,plan,x,0.0,nothing,work)
end
function apply_adjoint!(y,plan::LiouvillianPlan,x)
    _require_autonomous(plan,"apply_adjoint!")
    apply_adjoint!(y,plan,x,0.0,nothing,LiouvillianWorkspace(plan))
end

function _matrixfree_liouvillian(plan::LiouvillianPlan)
    work=LiouvillianWorkspace(plan)
    action! = (y,x,t,p)->apply!(y,plan,x,t,p,work)
    MatrixFreeLiouvillian(length(plan.basis),action!,plan.Ttype,copy(plan.tracevec);
                          autonomous=plan.autonomous,plan=plan,workspace=work)
end

function _local_jump_matrix(plan,ker,scale)
    n=length(plan.basis);gain=sparse(ker.gain.I,ker.gain.J,scale.*ker.gain.V,n,n)
    gain-_loss_matrix(plan,ker.qblocks,scale)
end

function _loss_matrix(plan,qblocks,scale)
    b=plan.basis;n=length(b)
    rows=Int[];columns=Int[];values=plan.Ttype[]
    for s in eachindex(b.sectors)
        offset=b.offsets[s]-1;Q=_exact_sparse_block(qblocks[s])
        M=(left_superoperator(Q)+right_superoperator(Q))/2
        dropzeros!(M)
        _append_sparse_block!(rows,columns,values,M,offset,scale)
    end
    sparse(rows,columns,values,n,n)
end

function _factorized_onebody_gain_matrix(plan,branches,contractions,scale)
    b=plan.basis;T=plan.Ttype;I=Int[];J=Int[];V=T[]
    @inbounds for branch_index in eachindex(branches.entries)
        branch=branches.entries[branch_index]
        li=branch.output_sector;ni=branch.input_sector
        nl=length(b.patterns[li]);nn=length(b.patterns[ni])
        contraction=contractions[branch_index]
        factor=scale*branch.scale
        for bb in 1:nl,a in 1:nl,d in 1:nn,c in 1:nn
            value=factor*contraction[a,c]*conj(contraction[bb,d])
            iszero(value)&&continue
            push!(I,b.offsets[li]+a-1+(bb-1)*nl)
            push!(J,b.offsets[ni]+c-1+(d-1)*nn)
            push!(V,convert(T,value))
        end
    end
    sparse(I,J,V,length(b),length(b))
end

function _factorized_pbody_gain_matrix(plan,groups,contractions,pair_scales,scale)
    b=plan.basis;T=plan.Ttype;I=Int[];J=Int[];V=T[]
    @inbounds for (li,ni,first_pair,last_pair) in groups
        nl=length(b.patterns[li]);nn=length(b.patterns[ni])
        for pair in first_pair:last_pair
            contraction=contractions[pair];exact_scale=pair_scales[pair]
            for bb in 1:nl,a in 1:nl,d in 1:nn,c in 1:nn
                primitive=contraction[a,c]*conj(contraction[bb,d])
                value=scale*(exact_scale.direct ? exact_scale.factor*primitive :
                    _apply_prepared_exact_scale(primitive,exact_scale;
                        context="sparse local p-body gain materialization"))
                iszero(value)&&continue
                push!(I,b.offsets[li]+a-1+(bb-1)*nl)
                push!(J,b.offsets[ni]+c-1+(d-1)*nn)
                push!(V,convert(T,value))
            end
        end
    end
    sparse(I,J,V,length(b),length(b))
end

function _collective_gain_matrix(plan,blocks,scale)
    b=plan.basis;n=length(b)
    rows=Int[];columns=Int[];values=plan.Ttype[]
    for sector in eachindex(b.sectors)
        offset=b.offsets[sector]-1
        sector_map=sandwich_superoperator(
            _exact_sparse_block(blocks[sector]))
        _append_sparse_block!(rows,columns,values,sector_map,offset,scale)
    end
    sparse(rows,columns,values,n,n)
end

_kernel_matrix(plan,kernel::HamiltonianPIKernel,scale)=
    scale*_block_superop(plan.basis,kernel.blocks,:commutator)
_kernel_matrix(plan,kernel::DissipatorPIKernel,scale)=
    scale*_block_superop(plan.basis,kernel.blocks,:dissipator;
                         qblocks=kernel.qblocks)
_kernel_matrix(plan,kernel::LocalJumpPIKernel,scale)=
    _local_jump_matrix(plan,kernel,scale)
_kernel_matrix(plan,kernel::FactorizedLocalJumpPIKernel,scale)=
    _factorized_onebody_gain_matrix(plan,kernel.branches,kernel.contractions,scale)-
    _loss_matrix(plan,kernel.qblocks,scale)
_kernel_matrix(plan,kernel::FactorizedLocalPBodyJumpPIKernel,scale)=
    _factorized_pbody_gain_matrix(plan,kernel.groups,kernel.contractions,
                                  kernel.pair_scales,scale)-
    _loss_matrix(plan,kernel.qblocks,scale)
function _kernel_matrix(plan,kernel::FusedStaticPIKernel,scale)
    n=length(plan.basis);M=spzeros(plan.Ttype,n,n)
    if kernel.hamiltonian_blocks!==nothing
        M+=_block_superop(plan.basis,kernel.hamiltonian_blocks,:commutator)
    end
    if kernel.loss_blocks!==nothing
        M-=_loss_matrix(plan,kernel.loss_blocks,one(plan.Ttype))
    end
    for gain in kernel.collective_gains
        M+=_collective_gain_matrix(plan,gain.blocks,gain.scale)
    end
    for gain in kernel.onebody_gains
        M+=_factorized_onebody_gain_matrix(plan,gain.branches,
                                            gain.contractions,gain.scale)
    end
    for gain in kernel.pbody_gains
        M+=_factorized_pbody_gain_matrix(plan,gain.groups,gain.contractions,
            gain.pair_scales,gain.scale)
    end
    scale*M
end

_materialized_kernel_scale(kernel::HamiltonianPIKernel)=value_at(kernel.scale,0.0,nothing)
_materialized_kernel_scale(kernel::Union{DissipatorPIKernel,LocalJumpPIKernel,
        FactorizedLocalJumpPIKernel,FactorizedLocalPBodyJumpPIKernel})=
    _evaluated_dissipative_rate(kernel.scale,0.0,nothing)
_materialized_kernel_scale(::FusedStaticPIKernel)=1

_add_kernel_matrices(M,plan,::Tuple{})=M
function _add_kernel_matrices(M,plan,kernels::Tuple{K,Vararg{Any}}) where K
    kernel=first(kernels);scale=convert(plan.Ttype,_materialized_kernel_scale(kernel))
    _add_kernel_matrices(M+_kernel_matrix(plan,kernel,scale),plan,Base.tail(kernels))
end

function _matrix_from_plan(plan::LiouvillianPlan)
    _require_autonomous(plan,"sparse materialization")
    plan.kernels===nothing&&throw(ArgumentError("operator-valued plans must be frozen before sparse materialization"))
    n=length(plan.basis)
    _add_kernel_matrices(spzeros(plan.Ttype,n,n),plan,plan.kernels)
end

"""
    liouvillian(model; representation=:matrixfree,
                memory_budget=512*1024^2, coefficient_cache=nothing)
    liouvillian(compiled; representation=compiled.backend)

Return a PI-coordinate Liouvillian as either a sparse matrix or a
`MatrixFreeLiouvillian`. Model construction lowers from the same prepared
term plan in both representations. Sparse materialization requires an
autonomous model; use `freeze` at an explicit time or the matrix-free backend
for driven dynamics. Its conservative live sparse-assembly bound is checked
before the coordinate matrix is allocated. Pass `memory_budget=Inf` only as
an explicit opt-out. Pass a compatible [`OneBoxCGCache`](@ref) to reuse
one-box coefficients across model preparations; otherwise small mixed
one-/`p`-body models prepare and share one automatically. Prefer `compile`
when the generator will be reused.
"""
function liouvillian(model::PIModel;representation=:matrixfree,
                     memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,
                     coefficient_cache=nothing)
    representation in (:sparse,:matrixfree)||throw(ArgumentError("representation must be :sparse or :matrixfree"))
    _require_model_preparation_budget(model,memory_budget;
        operation="PI Liouvillian preparation",coefficient_cache)
    plan=LiouvillianPlan(model;coefficient_cache)
    if representation===:sparse
        sparse_bounds=_performance_sparse_materialization_bounds(plan)
        estimate=BigInt(Base.summarysize(plan))+sparse_bounds.peak_bytes
        _require_performance_budget("sparse Liouvillian materialization",
            estimate,memory_budget;guidance="Use representation=:matrixfree.")
        return _matrix_from_plan(plan)
    end
    estimate=BigInt(Base.summarysize(plan))+
        _performance_liouvillian_workspace_bytes(plan)
    _require_performance_budget("matrix-free Liouvillian workspace",estimate,
        memory_budget;guidance="Reduce the retained basis/model size.")
    _matrixfree_liouvillian(plan)
end

"""A model, its reusable Liouvillian plan, and an automatically selected backend."""
struct CompiledPIModel{M,P,O,E}
    model::M
    plan::P
    operator::O
    backend::Symbol
    estimates::E
end

size(compiled::CompiledPIModel)=size(compiled.operator)
size(compiled::CompiledPIModel,i::Integer)=size(compiled.operator,i)
eltype(compiled::CompiledPIModel)=eltype(compiled.operator)
isautonomous(compiled::CompiledPIModel)=isautonomous(compiled.plan)

function _memory_budget_bytes(memory_budget)
    memory_budget isa Real&&!(memory_budget isa Bool)||throw(ArgumentError(
        "memory_budget must be a real number of bytes, not a Bool"))
    isnan(memory_budget)&&throw(ArgumentError("memory_budget cannot be NaN"))
    memory_budget>=0||throw(ArgumentError("memory_budget must be nonnegative"))
    isfinite(memory_budget) ? Int(min(floor(BigInt,memory_budget),BigInt(typemax(Int)))) : typemax(Int)
end

# High-level routines that may retain dense PI-coordinate arrays share one
# conservative default.  `Inf` is the explicit opt-out: unlike conversion to
# `typemax(Int)`, it must also permit estimates larger than the addressable
# integer range so the guard itself never becomes the artificial limit.
const _DEFAULT_HIGHLEVEL_MEMORY_BUDGET=512*1024^2

function _performance_memory_limit(memory_budget)
    _memory_budget_bytes(memory_budget) # common type/sign validation
    isfinite(memory_budget) ? BigInt(floor(BigInt,memory_budget)) : nothing
end

function _performance_array_bytes(dimension::Integer,::Type{T},
        square_arrays::Integer;linear_arrays::Integer=0,
        bigfloat_precision::Integer=precision(BigFloat)) where T
    dimension>=0||throw(ArgumentError("dimension must be nonnegative"))
    square_arrays>=0&&linear_arrays>=0||throw(ArgumentError(
        "performance array counts must be nonnegative"))
    entries=BigInt(square_arrays)*BigInt(dimension)^2+
            BigInt(linear_arrays)*BigInt(dimension)
    entries*_scalar_retained_bytes(T;bigfloat_precision)
end

function _performance_entries_bytes(entries::Integer,::Type{T};
        bigfloat_precision::Integer=precision(BigFloat)) where T
    entries>=0||throw(ArgumentError("storage entry count must be nonnegative"))
    BigInt(entries)*_scalar_retained_bytes(T;bigfloat_precision)
end

function _performance_gmres_bytes(dimension::Integer,::Type{T},
        krylovdim::Integer;recycle_dim::Integer=0,
        bigfloat_precision::Integer=precision(BigFloat)) where T
    dimension>0&&krylovdim>0&&recycle_dim>=0||throw(ArgumentError(
        "GMRES dimensions must be positive and recycle_dim nonnegative"))
    n=BigInt(dimension);m=min(n,BigInt(krylovdim));r=min(n,BigInt(recycle_dim))
    # Ordinary GMRES owns one basis/Hessenberg plus six full vectors. GCRO
    # additionally retains U, C, candidate, and AU (four n-by-r blocks), its
    # r-by-m coupling, and projected dense factors.
    complex_entries=n*(m+6)+(m+1)*m+2m+1+
                    4n*r+r*m+2r*r
    _performance_entries_bytes(complex_entries,T;bigfloat_precision)+
        _performance_entries_bytes(m,_real_float_type(T);bigfloat_precision)
end

function _performance_arnoldi_bytes(dimension::Integer,::Type{T},
        krylovdim::Integer;mode::Symbol=:full,
        bigfloat_precision::Integer=precision(BigFloat)) where T
    dimension>0&&krylovdim>0||throw(ArgumentError(
        "Arnoldi dimensions must be positive"))
    mode in (:ordinary,:full)||throw(ArgumentError(
        "Arnoldi performance mode must be :ordinary or :full"))
    n=BigInt(dimension);m=min(n,BigInt(krylovdim))
    entries=mode===:ordinary ? n*(m+3)+(m+1)*m :
        n*(5m+3)+3m*m+m
    _performance_entries_bytes(entries,T;bigfloat_precision)
end

_performance_source_action_bytes(source::AbstractMatrix,::Type{T}) where T=big(0)

function _performance_operator_workspace_bytes(prototype;
        bigfloat_precision::Integer=precision(BigFloat))
    storage=prototype isa AbstractPIOperator ? prototype.data : prototype
    _performance_entries_bytes(length(storage),eltype(storage);
                               bigfloat_precision)
end

_performance_kernel_workspace_bytes(::AbstractStaticPIKernel,basis,::Type{T};
    bigfloat_precision::Integer=precision(BigFloat)) where T=big(0)

function _performance_kernel_workspace_bytes(
        ::Union{FactorizedLocalJumpPIKernel,FactorizedLocalPBodyJumpPIKernel},
        basis,::Type{T};bigfloat_precision::Integer=precision(BigFloat)) where T
    largest=maximum(length,basis.patterns;init=1)
    _performance_entries_bytes(BigInt(largest)^2,T;bigfloat_precision)
end

function _performance_kernel_workspace_bytes(kernel::FusedStaticPIKernel,
        basis,::Type{T};bigfloat_precision::Integer=precision(BigFloat)) where T
    isempty(kernel.onebody_gains)&&isempty(kernel.pbody_gains)&&return big(0)
    largest=maximum(length,basis.patterns;init=1)
    _performance_entries_bytes(BigInt(largest)^2,T;bigfloat_precision)
end

function _performance_kernel_workspace_bytes(kernel::InPlaceHamiltonianPIKernel,
        basis,::Type{T};bigfloat_precision::Integer=precision(BigFloat)) where T
    bytes=_performance_operator_workspace_bytes(kernel.schedule.prototype;
        bigfloat_precision)+
        _performance_entries_bytes(length(basis),T;bigfloat_precision)
    if kernel.builder isa CollectivePBodyBlockBuilder&&
            kernel.builder.cancellation_risk
        largest=maximum(length,basis.patterns;init=0)
        bytes+=_performance_entries_bytes(
            BigInt(largest)^2,_real_float_type(T);bigfloat_precision)
    end
    bytes
end

function _performance_kernel_workspace_bytes(kernel::InPlaceDissipatorPIKernel,
        basis,::Type{T};bigfloat_precision::Integer=precision(BigFloat)) where T
    bytes=_performance_operator_workspace_bytes(kernel.schedule.prototype;
        bigfloat_precision)+
        _performance_entries_bytes(2BigInt(length(basis)),T;
                                   bigfloat_precision)
    if kernel.builder isa CollectivePBodyBlockBuilder&&
            kernel.builder.cancellation_risk
        largest=maximum(length,basis.patterns;init=0)
        bytes+=_performance_entries_bytes(
            BigInt(largest)^2,_real_float_type(T);bigfloat_precision)
    end
    bytes
end

function _performance_branch_contraction_entries(branches)
    sum((BigInt(length(branch.table)) for branch in branches.entries);
        init=big(0))
end

function _performance_kernel_workspace_bytes(kernel::InPlaceLocalJumpPIKernel,
        basis,::Type{T};bigfloat_precision::Integer=precision(BigFloat)) where T
    operator_bytes=_performance_operator_workspace_bytes(
        kernel.schedule.prototype;bigfloat_precision)
    entries=BigInt(length(basis))+
        _performance_branch_contraction_entries(kernel.branches)+
        BigInt(kernel.branches.maximum_block_dimension)^2
    2operator_bytes+_performance_entries_bytes(entries,T;bigfloat_precision)
end

function _performance_kernel_workspace_bytes(
        kernel::InPlaceLocalPBodyJumpPIKernel,basis,::Type{T};
        bigfloat_precision::Integer=precision(BigFloat)) where T
    operator_bytes=_performance_operator_workspace_bytes(
        kernel.schedule.prototype;bigfloat_precision)
    contractions=sum((BigInt(size(kernel.left_isometries[index],1))*
                      BigInt(size(kernel.right_isometries[index],1))
                      for index in eachindex(kernel.pair_scales));init=big(0))
    largest=maximum(length,basis.patterns;init=0)
    entries=BigInt(length(basis))+contractions+BigInt(largest)^2
    bytes=2operator_bytes+
        _performance_entries_bytes(entries,T;bigfloat_precision)
    if kernel.builder.cancellation_risk
        bytes+=_performance_entries_bytes(
            BigInt(largest)^2,_real_float_type(T);bigfloat_precision)
    end
    bytes
end

function _performance_kernel_workspace_bytes(
        kernel::InPlaceCorrelatedCollectiveJumpPIKernel,basis,::Type{T};
        bigfloat_precision::Integer=precision(BigFloat)) where T
    m=BigInt(length(kernel.channel_blocks));n=BigInt(length(basis))
    _performance_operator_workspace_bytes(kernel.schedule.prototype;
        bigfloat_precision)+
        _performance_entries_bytes(2m^2+(m+2)n,T;bigfloat_precision)
end

function _performance_kernel_workspace_bytes(
        kernel::InPlaceCorrelatedLocalJumpPIKernel,basis,::Type{T};
        bigfloat_precision::Integer=precision(BigFloat)) where T
    m=BigInt(length(kernel.operators));d=BigInt(size(first(kernel.operators),1))
    n=BigInt(length(basis));contractions=_performance_branch_contraction_entries(
        kernel.branches)
    entries=2m^2+(m+2)d^2+n+m*contractions+
        BigInt(kernel.branches.maximum_block_dimension)^2
    _performance_operator_workspace_bytes(kernel.schedule.prototype;
        bigfloat_precision)+
        _performance_entries_bytes(entries,T;bigfloat_precision)
end

function _performance_liouvillian_workspace_bytes(plan::LiouvillianPlan;
        batch_columns::Integer=0,
        bigfloat_precision::Integer=precision(BigFloat))
    batch_columns>=0||throw(ArgumentError(
        "batch_columns must be nonnegative"))
    n=BigInt(length(plan.basis))
    largest=BigInt(maximum(length,plan.basis.patterns;init=1))
    bytes=_performance_entries_bytes(
        3n+3largest^2*BigInt(batch_columns),plan.Ttype;
        bigfloat_precision)
    plan.kernels===nothing&&return bytes
    for kernel in plan.kernels
        bytes+=_performance_kernel_workspace_bytes(
            kernel,plan.basis,plan.Ttype;bigfloat_precision)
    end
    bytes
end

function _performance_liouvillian_batch_payload_bytes(basis::PIBasis,
        ::Type{T},batch_columns::Integer;
        bigfloat_precision::Integer=precision(BigFloat)) where T
    batch_columns>=0||throw(ArgumentError(
        "batch_columns must be nonnegative"))
    largest=BigInt(maximum(length,basis.patterns;init=1))
    _performance_entries_bytes(
        3largest^2*BigInt(batch_columns),T;bigfloat_precision)
end

_performance_liouvillian_batch_payload_bytes(plan::LiouvillianPlan,
        batch_columns::Integer;
        bigfloat_precision::Integer=precision(BigFloat))=
    _performance_liouvillian_batch_payload_bytes(
        plan.basis,plan.Ttype,batch_columns;bigfloat_precision)

function _performance_liouvillian_fallback_bytes(plan::LiouvillianPlan;
        bigfloat_precision::Integer=precision(BigFloat))
    plan.kernels===nothing||return big(0)
    n=BigInt(length(plan.basis))
    scalar_bytes=_scalar_retained_bytes(plan.Ttype;bigfloat_precision)
    sparse=n^2*(scalar_bytes+2sizeof(Int))+(n+1)*sizeof(Int)
    _model_preparation_bytes(plan.fallback_model;bigfloat_precision)+3sparse
end

# Count exact stored mathematical support without allocating a sparse copy.
# Prepared standard kernels already own all matrices inspected here, so this
# setup-only pass is linear in their retained block payload.  Explicit stored
# zeros are not charged as nonzeros because every sparse materializer below
# applies the same exact `iszero` rule; no numerical tolerance is involved.
function _performance_matrix_nonzeros(matrix)
    count=big(0)
    @inbounds for value in matrix
        iszero(value)||(count+=1)
    end
    count
end
# Iterating an `AbstractArray` visits every logical entry.  Avoid turning this
# setup estimate into an O(m^2) scan when the symmetric collective fast path
# has already retained an exact sparse Schur block.
function _performance_matrix_nonzeros(matrix::SparseMatrixCSC)
    BigInt(count(!iszero,nonzeros(matrix)))
end

function _performance_commutator_contributions(blocks)
    sum(eachindex(blocks);init=big(0)) do sector
        block=blocks[sector]
        2BigInt(size(block,1))*_performance_matrix_nonzeros(block)
    end
end

function _performance_loss_contributions(blocks)
    sum(eachindex(blocks);init=big(0)) do sector
        block=blocks[sector]
        2BigInt(size(block,1))*_performance_matrix_nonzeros(block)
    end
end

function _performance_collective_gain_contributions(blocks)
    sum(blocks;init=big(0)) do block
        support=_performance_matrix_nonzeros(block)
        support^2
    end
end

function _performance_factorized_gain_contributions(contractions)
    sum(contractions;init=big(0)) do contraction
        support=_performance_matrix_nonzeros(contraction)
        support^2
    end
end

_performance_sparse_kernel_contributions(kernel::HamiltonianPIKernel)=
    _performance_commutator_contributions(kernel.blocks)
function _performance_sparse_kernel_contributions(kernel::DissipatorPIKernel)
    _performance_collective_gain_contributions(kernel.blocks)+
        _performance_loss_contributions(kernel.qblocks)
end
function _performance_sparse_kernel_contributions(kernel::LocalJumpPIKernel)
    gain=count(!iszero,kernel.gain.V)
    BigInt(gain)+_performance_loss_contributions(kernel.qblocks)
end
function _performance_sparse_kernel_contributions(
        kernel::FactorizedLocalJumpPIKernel)
    _performance_factorized_gain_contributions(kernel.contractions)+
        _performance_loss_contributions(kernel.qblocks)
end
function _performance_sparse_kernel_contributions(
        kernel::FactorizedLocalPBodyJumpPIKernel)
    _performance_factorized_gain_contributions(kernel.contractions)+
        _performance_loss_contributions(kernel.qblocks)
end
function _performance_sparse_kernel_contributions(kernel::FusedStaticPIKernel)
    contributions=big(0)
    kernel.hamiltonian_blocks===nothing||
        (contributions+=_performance_commutator_contributions(
            kernel.hamiltonian_blocks))
    kernel.loss_blocks===nothing||
        (contributions+=_performance_loss_contributions(kernel.loss_blocks))
    for gain in kernel.collective_gains
        contributions+=_performance_collective_gain_contributions(gain.blocks)
    end
    for gain in kernel.onebody_gains
        contributions+=_performance_factorized_gain_contributions(
            gain.contractions)
    end
    for gain in kernel.pbody_gains
        contributions+=_performance_factorized_gain_contributions(
            gain.contractions)
    end
    contributions
end
_performance_sparse_kernel_contributions(::AbstractDynamicPIKernel)=nothing
_performance_sparse_kernel_contributions(kernel)=nothing

_performance_sparse_plan_contributions(::Tuple{})=big(0)
function _performance_sparse_plan_contributions(
        kernels::Tuple{K,Vararg{Any}}) where K
    head=_performance_sparse_kernel_contributions(first(kernels))
    head===nothing&&return nothing
    tail=_performance_sparse_plan_contributions(Base.tail(kernels))
    tail===nothing ? nothing : head+tail
end

"""
Conservative storage bounds for sparse materialization of a prepared PI plan.

For standard fixed kernels, `contribution_upper_bound` counts the exact-support
triplets generated before duplicate coordinates are coalesced.  Unknown or
dynamic kernels retain the dense-coordinate fallback.  `operator_bytes` bounds
the retained CSC result, while `peak_bytes` includes a factor-sixteen allowance
for simultaneous sparse Kronecker operands/results, global triplets, CSC
conversion, kernel-by-kernel accumulation, and allocator/version variation.
The prepared plan itself is not included in either quantity.
"""
function _performance_sparse_materialization_bounds(plan::LiouvillianPlan;
        bigfloat_precision::Integer=precision(BigFloat))
    n=BigInt(length(plan.basis));dense_entries=n^2
    contributions=plan.kernels===nothing ? nothing :
        _performance_sparse_plan_contributions(plan.kernels)
    structured=contributions!==nothing
    contribution_upper=structured ? contributions : dense_entries
    retained_entries=min(contribution_upper,dense_entries)
    scalar_bytes=_scalar_retained_bytes(plan.Ttype;bigfloat_precision)
    int_bytes=BigInt(sizeof(Int));column_bytes=(n+1)*int_bytes
    operator_bytes=retained_entries*(scalar_bytes+int_bytes)+column_bytes
    assembly_bytes=contribution_upper*(scalar_bytes+2int_bytes)+column_bytes
    # Sparse block expressions can transiently retain both Kronecker operands,
    # a gain/loss result, exact triplets, the converted CSC block, and the
    # accumulated result.  Sixteen complete largest-payload allowances remain a
    # conservative live bound without returning to the old dense n_PI^2 guess.
    peak_bytes=16max(operator_bytes,assembly_bytes)
    (;structured,contribution_upper_bound=contribution_upper,
      retained_nnz_upper_bound=retained_entries,operator_bytes,
      assembly_bytes,peak_bytes)
end

function _performance_source_action_bytes(plan::LiouvillianPlan,
        ::Type{T}) where T
    _performance_liouvillian_fallback_bytes(plan)
end

function _performance_source_action_bytes(source::CompiledPIModel,
        ::Type{T}) where T
    source.backend===:sparse ? big(0) :
        _performance_liouvillian_fallback_bytes(source.plan)
end
function _performance_source_action_bytes(source::MatrixFreeLiouvillian,
        ::Type{T}) where T
    # Package-prepared adapters retain their explicit application workspace in
    # the operator itself.  Callers (and scan resource reports) already count
    # that retained object, so charging another generic 16-vector action bound
    # here would double count it.  Plan-less callbacks without an exposed
    # workspace keep the conservative fallback below.
    if source.workspace isa LiouvillianWorkspace
        return source.plan isa LiouvillianPlan ?
            _performance_liouvillian_fallback_bytes(source.plan) : big(0)
    end
    n=size(source,1)
    _performance_array_bytes(n,T,0;linear_arrays=16)
end
function _performance_source_action_bytes(source,::Type{T}) where T
    n=size(source,1)
    _performance_array_bytes(n,T,0;linear_arrays=16)
end

function _performance_budget_fits(estimated_bytes::Integer,memory_budget)
    estimated_bytes>=0||throw(ArgumentError(
        "estimated memory must be nonnegative"))
    limit=_performance_memory_limit(memory_budget)
    limit===nothing||BigInt(estimated_bytes)<=limit
end


function _require_performance_budget(operation::AbstractString,
        estimated_bytes::Integer,memory_budget;guidance::AbstractString="")
    _performance_budget_fits(estimated_bytes,memory_budget)&&return nothing
    limit=_performance_memory_limit(memory_budget)
    suffix=isempty(guidance) ? "" : " $guidance"
    throw(ArgumentError(
        "$operation requires an estimated peak of at least $estimated_bytes bytes, "*
        "which exceeds memory_budget=$limit bytes.$suffix Pass memory_budget=Inf "*
        "to opt in explicitly after checking available RAM."))
end

function _estimate_symmetric_collective_geometry(b::PIBasis,
        ::Type{T}=Float64;
        bigfloat_precision::Integer=precision(BigFloat)) where
        T<:AbstractFloat
    _has_single_fully_symmetric_sector(b)||throw(ArgumentError(
        "symmetric collective geometry requires one fully symmetric sector"))
    occupation_count=BigInt(length(only(b.patterns)))
    int_bytes=BigInt(sizeof(Int));header=8int_bytes
    tuple_bytes=BigInt(b.d)*int_bytes
    scalar_bytes=_scalar_retained_bytes(T;bigfloat_precision)
    transition_count=sum(only(b.patterns);init=big(0)) do pattern
        BigInt(count(!iszero,content(pattern)))*BigInt(b.d-1)
    end
    # The retained object owns one exactly sized occupation vector and a Dict
    # from each occupation tuple to its block index, plus preconverted diagonal
    # factors, packed off-diagonal transitions, and their column offsets.  A
    # factor-two capacity allowance covers hash-table load-factor slack and the
    # push!-grown transition vector; the setup allowance also covers tuple,
    # index, and transition-vector staging.
    # The caller-owned basis is referenced rather than counted a second time.
    dictionary_entry=16int_bytes+tuple_bytes+int_bytes
    occupation_bytes=occupation_count*tuple_bytes
    diagonal_bytes=occupation_count*BigInt(b.d)*scalar_bytes
    offset_bytes=(occupation_count+1)*int_bytes
    transition_bytes=2transition_count*(3int_bytes+scalar_bytes)
    retained_bytes=8header+occupation_bytes+2occupation_count*dictionary_entry+
        diagonal_bytes+offset_bytes+transition_bytes
    setup_bytes=retained_bytes+4header+2occupation_bytes+diagonal_bytes+
        offset_bytes+2transition_bytes
    (;retained_bytes,setup_bytes,occupation_count,
      transition_count,scalar_type=T,scalar_retained_bytes=scalar_bytes,
      scalar_storage_estimate=_scalar_storage_estimate(T),
      bigfloat_precision_assumption=
          _scalar_precision_assumption(T,bigfloat_precision),
      estimate=:conservative_structural_upper_bound)
end

function _estimate_model_geometry(model::PIModel;
        bigfloat_precision::Integer=precision(BigFloat))
    R=_model_geometry_type(model)
    requirements=_model_onebox_requirements(model,R)
    onebody=if requirements.needs_full_onebody
        _estimate_onebody_geometry(
            model.basis,R;bigfloat_precision)
    elseif requirements.uses_symmetric_collective
        _estimate_symmetric_collective_geometry(
            model.basis,R;bigfloat_precision)
    elseif requirements.uses_diagonal_onebody
        _estimate_onebody_geometry(
            model.basis,R;diagonal_only=true,bigfloat_precision)
    else
        nothing
    end
    pbody=map(requirements.pbody_orders) do order
        order=>_estimate_pbody_geometry(
            model.basis,order,R;bigfloat_precision)
    end
    retained_bytes=(onebody===nothing ? big(0) :
                    big(onebody.retained_bytes))+
        sum((big(entry.second.retained_bytes) for entry in pbody);
            init=big(0))
    setup_bytes=(onebody===nothing ? big(0) :
                 big(onebody.setup_bytes))+
        sum((big(entry.second.setup_bytes) for entry in pbody);
            init=big(0))
    (;retained_bytes,setup_bytes,onebody,pbody,
      pbody_orders=requirements.pbody_orders,
      geometry_families=requirements.geometry_families,
      estimate=:sum_of_conservative_structural_upper_bounds)
end

function _model_preparation_bytes(model::PIModel;
        linear_arrays::Integer=16,
        bigfloat_precision::Integer=precision(BigFloat),
        coefficient_cache=nothing)
    n=length(model.basis);R=_model_geometry_type(model)
    T=Complex{R}
    requirements=_model_onebox_requirements(model,R)
    geometry=_estimate_model_geometry(
        model;bigfloat_precision).setup_bytes
    automatic_coefficients=coefficient_cache===nothing&&
        requirements.geometry_families>1&&_small_onebox_autocache(model.basis) ?
        _estimate_onebox_cache_upper(model.basis,requirements.required_depth,R;
            precision_bits=bigfloat_precision) : big(0)
    geometry+automatic_coefficients+
        _performance_array_bytes(n,T,0;linear_arrays,bigfloat_precision)
end

function _require_model_preparation_budget(model::PIModel,memory_budget;
        operation::AbstractString="PI model preparation",
        bigfloat_precision::Integer=precision(BigFloat),
        coefficient_cache=nothing)
    estimate=_model_preparation_bytes(
        model;bigfloat_precision,coefficient_cache)
    _require_performance_budget(operation,estimate,memory_budget;guidance=
        "Reduce the retained basis/model size or increase the budget.")
end

"""
    compile(model; backend=:auto, memory_budget=512*1024^2,
            bigfloat_precision=precision(BigFloat), coefficient_cache=nothing)

Prepare all fixed Schur geometry once and choose a sparse or matrix-free
backend. For standard fixed kernels, `backend=:auto` bounds sparse storage
from their prepared exact block support; dynamic, custom, or otherwise unknown
kernels retain a conservative dense-coordinate fallback. Driven models always
remain matrix-free. The returned `CompiledPIModel` can be passed directly to
`apply!`, `evolve!`, and `dynamics_problem`.

The sparse bound includes the prepared plan and simultaneous sparse assembly
temporaries. Matrix-free preparation separately bounds its retained plan and
compatibility action scratch. Automatic selection uses matrix-free work when
the sparse route does not fit, and raises if that bounded alternative also
exceeds the budget. Pass `memory_budget=Inf` as an explicit opt-out.

For fixed-size isbits scalar types, storage bounds retain the exact inline
`sizeof(T)` accounting. `BigFloat` and `Complex{BigFloat}` use an explicitly
conservative retained-storage bound at `bigfloat_precision`; pass the maximum
precision intended for generated matrix entries when it differs from the
active process precision.

Pass a compatible [`OneBoxCGCache`](@ref) to share one-box Clebsch--Gordan
coefficients across one- and `p`-body geometry construction and across calls.
An explicit cache is caller-owned and is therefore excluded from the
preparation memory estimate. If it is omitted, a small model with multiple
geometry families may build one bounded temporary cache automatically; its
storage is included in the estimate.
"""
function compile(model::PIModel;backend=:auto,memory_budget=512*1024^2,
                 bigfloat_precision::Integer=precision(BigFloat),
                 coefficient_cache=nothing)
    backend in (:auto,:sparse,:matrixfree)||throw(ArgumentError("backend must be :auto, :sparse, or :matrixfree"))
    _require_model_preparation_budget(model,memory_budget;
        operation="compiled PI model preparation",bigfloat_precision,
        coefficient_cache)
    budget=_memory_budget_bytes(memory_budget)
    plan=LiouvillianPlan(model;coefficient_cache)
    n=length(model.basis)
    T=plan.Ttype
    scalar_bytes=_scalar_retained_bytes(T;bigfloat_precision)
    sparse_bounds=_performance_sparse_materialization_bounds(
        plan;bigfloat_precision)
    sparse_operator_big=sparse_bounds.operator_bytes
    sparse_upper=Int(min(sparse_operator_big,BigInt(typemax(Int))))
    plan_bytes=Base.summarysize(plan)
    sparse_total=Int(min(BigInt(plan_bytes)+sparse_operator_big,
                         BigInt(typemax(Int))))
    sparse_peak_big=BigInt(plan_bytes)+sparse_bounds.peak_bytes
    sparse_peak=Int(min(sparse_peak_big,BigInt(typemax(Int))))
    matrixfree_workspace_big=_performance_liouvillian_workspace_bytes(
        plan;bigfloat_precision)
    matrixfree_total_big=BigInt(plan_bytes)+matrixfree_workspace_big
    matrixfree_total=Int(min(matrixfree_total_big,BigInt(typemax(Int))))
    chosen = backend===:auto ?
        (plan.autonomous&&_performance_budget_fits(sparse_peak_big,memory_budget) ?
            :sparse : :matrixfree) : backend
    chosen===:sparse&&!plan.autonomous&&throw(ArgumentError("a time-dependent model cannot use the sparse backend; freeze it at an explicit time or use backend=:matrixfree"))
    chosen===:sparse&&_require_performance_budget(
        "sparse compiled Liouvillian materialization",sparse_peak_big,
        memory_budget;guidance="Use backend=:matrixfree.")
    chosen===:matrixfree&&_require_performance_budget(
        "matrix-free compiled Liouvillian workspace",matrixfree_total_big,
        memory_budget;guidance="Reduce the retained basis/model size.")
    operator=chosen===:sparse ? _matrix_from_plan(plan) : _matrixfree_liouvillian(plan)
    estimates=(scalar_type=T,dimension=n,plan_bytes=plan_bytes,
               scalar_retained_bytes=scalar_bytes,
               scalar_storage_estimate=_scalar_storage_estimate(T),
               bigfloat_precision_assumption=
                   _scalar_precision_assumption(T,bigfloat_precision),
               sparse_upper_bound=sparse_upper,sparse_operator_upper_bound=sparse_upper,
               sparse_structure_supported=sparse_bounds.structured,
               sparse_contribution_upper_bound=Int(min(
                   sparse_bounds.contribution_upper_bound,
                   BigInt(typemax(Int)))),
               sparse_retained_nnz_upper_bound=Int(min(
                   sparse_bounds.retained_nnz_upper_bound,
                   BigInt(typemax(Int)))),
               sparse_assembly_upper_bound=Int(min(
                   sparse_bounds.assembly_bytes,BigInt(typemax(Int)))),
               sparse_compiled_upper_bound=sparse_total,
               sparse_materialization_peak_upper_bound=sparse_peak,
               matrixfree_workspace_upper_bound=Int(min(
                   matrixfree_workspace_big,BigInt(typemax(Int)))),
               matrixfree_compiled_estimate=matrixfree_total,memory_budget=budget,
               requested_backend=backend,chosen_backend=chosen)
    CompiledPIModel(model,plan,operator,chosen,estimates)
end

function liouvillian(compiled::CompiledPIModel;representation=compiled.backend,
                     memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    representation in (:sparse,:matrixfree)||throw(ArgumentError("representation must be :sparse or :matrixfree"))
    representation===compiled.backend&&return compiled.operator
    if representation===:sparse
        precision_assumption=
            compiled.estimates.bigfloat_precision_assumption
        sparse_precision=precision_assumption===nothing ?
            precision(BigFloat) : precision_assumption
        estimate=_performance_sparse_materialization_bounds(
            compiled.plan;bigfloat_precision=sparse_precision).peak_bytes
        _require_performance_budget("sparse Liouvillian materialization",
            estimate,memory_budget;guidance="Keep representation=:matrixfree.")
        return _matrix_from_plan(compiled.plan)
    end
    estimate=BigInt(compiled.estimates.plan_bytes)+
        _performance_liouvillian_workspace_bytes(compiled.plan)
    _require_performance_budget("matrix-free Liouvillian compatibility workspace",
        estimate,memory_budget;guidance="Keep representation=:sparse or use a caller-owned plan workspace.")
    _matrixfree_liouvillian(compiled.plan)
end

LiouvillianWorkspace(compiled::CompiledPIModel)=LiouvillianWorkspace(compiled.plan)
function LiouvillianWorkspace(L::MatrixFreeLiouvillian)
    L.plan isa LiouvillianPlan||throw(ArgumentError(
        "this MatrixFreeLiouvillian does not wrap a compiled PI LiouvillianPlan"))
    LiouvillianWorkspace(L.plan)
end

function apply!(y,L::MatrixFreeLiouvillian,x,t,p,work::LiouvillianWorkspace)
    L.plan isa LiouvillianPlan ? apply!(y,L.plan,x,t,p,work) :
        L.action!(y,x,t,p)
end
function apply!(Y::AbstractMatrix,L::MatrixFreeLiouvillian,X::AbstractMatrix,t,p,
                work::LiouvillianWorkspace)
    L.plan isa LiouvillianPlan ? apply!(Y,L.plan,X,t,p,work) :
        throw(ArgumentError(
            "only a compiled PI MatrixFreeLiouvillian supports a LiouvillianWorkspace batch"))
end
function apply!(Y::AbstractMatrix,L::MatrixFreeLiouvillian,X::AbstractMatrix,t,p)
    size(X,1)==L.n||throw(DimensionMismatch("matrix input has the wrong leading dimension"))
    size(Y)==(L.n,size(X,2))||throw(DimensionMismatch("matrix output has the wrong dimensions"))
    if L.batched_action! !== nothing
        L.batched_action!(Y,X,t,p)
    elseif L.plan isa LiouvillianPlan
        lock(L.lock)
        try
            apply!(Y,L.plan,X,t,p,L.workspace)
        finally
            unlock(L.lock)
        end
    else
        for j in axes(X,2)
            L.action!(view(Y,:,j),view(X,:,j),t,p)
        end
    end
    Y
end
apply!(y,L::MatrixFreeLiouvillian,x,t,p)=L.action!(y,x,t,p)
apply!(y,L::MatrixFreeLiouvillian,x)=mul!(y,L,x)

function apply!(y,L::AbstractMatrix,x,t,p,work=nothing)
    mul!(y,L,x)
end

function apply!(y,compiled::CompiledPIModel,x,t,p,work::LiouvillianWorkspace)
    compiled.backend===:matrixfree ? apply!(y,compiled.plan,x,t,p,work) : mul!(y,compiled.operator,x)
end
function apply!(y,compiled::CompiledPIModel,x,t,p)
    compiled.backend===:matrixfree ? apply!(y,compiled.operator,x,t,p) : mul!(y,compiled.operator,x)
end
apply!(y,compiled::CompiledPIModel,x)=mul!(y,compiled,x)

_liouvillian_action!(y,compiled::CompiledPIModel,x,t,p)=apply!(y,compiled,x,t,p)

function mul!(y,compiled::CompiledPIModel,x)
    _require_autonomous(compiled,"mul!")
    mul!(y,compiled.operator,x)
end
*(compiled::CompiledPIModel,x::AbstractVector)=
    mul!(_product_destination(compiled,x,size(compiled,1)),compiled,x)

*(compiled::CompiledPIModel,X::AbstractMatrix)=
    mul!(_product_destination(compiled,X,size(compiled,1),size(X,2)),compiled,X)

function mul!(Y::AbstractMatrix,compiled::CompiledPIModel,X::AbstractMatrix)
    _require_autonomous(compiled,"mul!")
    mul!(Y,compiled.operator,X)
end

function apply_adjoint!(y,L::MatrixFreeLiouvillian,x,t,p,
                        work::LiouvillianWorkspace)
    if L.plan===nothing
        _require_autonomous(L,"apply_adjoint!")
        return mul!(y,adjoint(Matrix(_materialize(L))),x)
    end
    apply_adjoint!(y,L.plan,x,t,p,work)
end
function apply_adjoint!(Y::AbstractMatrix,L::MatrixFreeLiouvillian,X::AbstractMatrix,
                        t,p,work::LiouvillianWorkspace)
    L.plan===nothing&&throw(ArgumentError(
        "a custom MatrixFreeLiouvillian does not support explicit batched adjoint workspaces"))
    apply_adjoint!(Y,L.plan,X,t,p,work)
end

function _apply_custom_adjoint!(y::AbstractVector,L::MatrixFreeLiouvillian,
                                x::AbstractVector,t,p)
    length(x)==L.n&&length(y)==L.n||
        throw(DimensionMismatch("Liouvillian vector has the wrong length"))
    if L.adjoint_action! !== nothing
        L.adjoint_action!(y,x,t,p)
    elseif L.batched_adjoint_action! !== nothing
        L.batched_adjoint_action!(reshape(y,L.n,1),reshape(x,L.n,1),t,p)
    else
        _require_autonomous(L,"apply_adjoint!")
        mul!(y,adjoint(Matrix(_materialize(L))),x)
    end
    y
end

function _apply_custom_adjoint!(Y::AbstractMatrix,L::MatrixFreeLiouvillian,
                                X::AbstractMatrix,t,p)
    size(X,1)==L.n||throw(DimensionMismatch("matrix input has the wrong leading dimension"))
    size(Y)==(L.n,size(X,2))||throw(DimensionMismatch("matrix output has the wrong dimensions"))
    if L.batched_adjoint_action! !== nothing
        L.batched_adjoint_action!(Y,X,t,p)
    elseif L.adjoint_action! !== nothing
        for j in axes(X,2)
            L.adjoint_action!(view(Y,:,j),view(X,:,j),t,p)
        end
    else
        _require_autonomous(L,"apply_adjoint!")
        mul!(Y,adjoint(Matrix(_materialize(L))),X)
    end
    Y
end

function apply_adjoint!(y::AbstractVector,L::MatrixFreeLiouvillian,
                        x::AbstractVector,t,p)
    if L.plan===nothing
        return _apply_custom_adjoint!(y,L,x,t,p)
    end
    lock(L.lock)
    try
        apply_adjoint!(y,L.plan,x,t,p,L.workspace)
    finally
        unlock(L.lock)
    end
end
function apply_adjoint!(Y::AbstractMatrix,L::MatrixFreeLiouvillian,
                        X::AbstractMatrix,t,p)
    if L.plan===nothing
        return _apply_custom_adjoint!(Y,L,X,t,p)
    end
    lock(L.lock)
    try
        apply_adjoint!(Y,L.plan,X,t,p,L.workspace)
    finally
        unlock(L.lock)
    end
end
function apply_adjoint!(y,L::MatrixFreeLiouvillian,x)
    _require_autonomous(L,"apply_adjoint!")
    apply_adjoint!(y,L,x,0.0,nothing)
end

function apply_adjoint!(y,compiled::CompiledPIModel,x,t,p,
                        work::LiouvillianWorkspace)
    compiled.backend===:matrixfree ? apply_adjoint!(y,compiled.plan,x,t,p,work) :
        mul!(y,adjoint(compiled.operator),x)
end
function apply_adjoint!(Y::AbstractMatrix,compiled::CompiledPIModel,X::AbstractMatrix,
                        t,p,work::LiouvillianWorkspace)
    compiled.backend===:matrixfree ?
        apply_adjoint!(Y,compiled.plan,X,t,p,work) :
        mul!(Y,adjoint(compiled.operator),X)
end

function apply_adjoint!(y,compiled::CompiledPIModel,x,t,p)
    compiled.backend===:matrixfree ? apply_adjoint!(y,compiled.operator,x,t,p) :
        mul!(y,adjoint(compiled.operator),x)
end
function apply_adjoint!(y,compiled::CompiledPIModel,x)
    _require_autonomous(compiled,"apply_adjoint!")
    apply_adjoint!(y,compiled,x,0.0,nothing)
end

struct AdjointMatrixFreeLiouvillian{L,M}
    parent::L
    fallback::M
end

size(A::AdjointMatrixFreeLiouvillian)=reverse(size(A.parent))
size(A::AdjointMatrixFreeLiouvillian,i::Integer)=i in (1,2) ? size(A.parent,3-i) : 1
eltype(A::AdjointMatrixFreeLiouvillian)=eltype(A.parent)
isautonomous(A::AdjointMatrixFreeLiouvillian)=isautonomous(A.parent)

function adjoint(L::MatrixFreeLiouvillian)
    _require_autonomous(L,"adjoint")
    explicit_adjoint=L.adjoint_action! !== nothing ||
                     L.batched_adjoint_action! !== nothing
    fallback=L.plan===nothing&&!explicit_adjoint ?
        adjoint(Matrix(_materialize(L))) : nothing
    AdjointMatrixFreeLiouvillian(L,fallback)
end
adjoint(A::AdjointMatrixFreeLiouvillian)=A.parent

function mul!(y,A::AdjointMatrixFreeLiouvillian,x)
    if A.fallback===nothing
        apply_adjoint!(y,A.parent,x,0.0,nothing)
    else
        mul!(y,A.fallback,x)
    end
end
*(A::AdjointMatrixFreeLiouvillian,x::AbstractVector)=
    mul!(_product_destination(A,x,size(A,1)),A,x)

*(A::AdjointMatrixFreeLiouvillian,X::AbstractMatrix)=
    mul!(_product_destination(A,X,size(A,1),size(X,2)),A,X)

function mul!(Y::AbstractMatrix,A::AdjointMatrixFreeLiouvillian,X::AbstractMatrix)
    size(X,1)==size(A,2)||throw(DimensionMismatch("matrix input has the wrong leading dimension"))
    size(Y)==(size(A,1),size(X,2))||throw(DimensionMismatch("matrix output has the wrong dimensions"))
    if A.fallback===nothing
        apply_adjoint!(Y,A.parent,X,0.0,nothing)
    else
        mul!(Y,A.fallback,X)
    end
    Y
end

function freeze(compiled::CompiledPIModel;time,parameters=nothing,
                representation=:matrixfree)
    freeze(compiled.model;time=time,parameters=parameters,representation=representation)
end

# These typed forwarding methods are defined before spectra.jl is included;
# its general methods extend the same generic functions later in module load.
function pi_liouvillian_spectrum(compiled::CompiledPIModel;method=:dense,
        basis=compiled.plan.basis,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,kwargs...)
    method isa Symbol||throw(ArgumentError("method must be a Symbol"))
    method=_canonical_spectrum_algorithm(method)
    representation=method in (:arnoldi,:block_arnoldi,:harmonic,:iram,:jd) ?
        :matrixfree : compiled.backend
    source=liouvillian(compiled;representation=representation,memory_budget)
    pi_liouvillian_spectrum(source;method=method,basis=basis,memory_budget,
                            kwargs...)
end

function pi_liouvillian_gap(compiled::CompiledPIModel;method=:dense,
        basis=compiled.plan.basis,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,kwargs...)
    method isa Symbol||throw(ArgumentError("method must be a Symbol"))
    method=_canonical_spectrum_algorithm(method)
    representation=method in (:arnoldi,:block_arnoldi,:harmonic,:iram,:jd) ?
        :matrixfree : compiled.backend
    source=liouvillian(compiled;representation=representation,memory_budget)
    pi_liouvillian_gap(source;method=method,basis=basis,memory_budget,
                       kwargs...)
end

"""
    freeze(model; time, parameters=nothing, representation=:matrixfree)

Evaluate every time- or parameter-dependent term at the explicitly supplied
`time` and return an autonomous Liouvillian. This is the required bridge from
a driven model to stationary-state or spectral linear algebra; it never
silently assumes `time == 0`.
"""
function freeze(model::PIModel; time, parameters=nothing, representation=:matrixfree)
    representation in (:matrixfree,:sparse) ||
        throw(ArgumentError("representation must be :sparse or :matrixfree"))
    frozen_model=PIModel(model.basis,(freeze_term(term,time,parameters) for term in model.terms))
    liouvillian(frozen_model;representation=representation)
end

function freeze(L::MatrixFreeLiouvillian; time, parameters=nothing,
                representation=:matrixfree)
    representation in (:matrixfree,:sparse) ||
        throw(ArgumentError("representation must be :sparse or :matrixfree"))
    frozen = if isautonomous(L)
        L
    else
        action! = (y,x,t,p)->L.action!(y,x,time,parameters)
        adjoint_action! = L.adjoint_action! === nothing ? nothing :
            (y,x,t,p)->L.adjoint_action!(y,x,time,parameters)
        batched_action! = L.batched_action! === nothing ? nothing :
            (Y,X,t,p)->L.batched_action!(Y,X,time,parameters)
        batched_adjoint_action! = L.batched_adjoint_action! === nothing ? nothing :
            (Y,X,t,p)->L.batched_adjoint_action!(Y,X,time,parameters)
        MatrixFreeLiouvillian(L.n,action!,L.Ttype,copy(L.tracevec);
            autonomous=true,adjoint_action!,batched_action!,
            batched_adjoint_action!)
    end
    representation===:matrixfree ? frozen : sparse(_materialize(frozen))
end

"""Coordinate vector `t` such that `dot(t, rho.data) == trace(rho)`."""
function _trace_vector(b::PIBasis,::Type{T}=ComplexF64) where T
    t=zeros(T,length(b))
    R=_real_float_type(T)
    for (s,p) in pairs(b.sectors)
        n=length(b.patterns[s]);off=b.offsets[s]-1
        scale=_schur_multiplicity_scale(R,p)
        for i in 1:n
            t[off+i+(i-1)*n]=scale
        end
    end
    t
end

function _materialize(L)
    L isa AbstractMatrix && return L
    L isa MatrixFreeLiouvillian && _require_autonomous(L, "materialization")
    L isa LiouvillianPlan && return _matrix_from_plan(L)
    L isa CompiledPIModel && return L.backend===:sparse ? L.operator : _matrix_from_plan(L.plan)
    L isa MatrixFreeLiouvillian && L.plan!==nothing && return _matrix_from_plan(L.plan)
    n=size(L,1); M=Matrix{eltype(L)}(undef,n,n); e=zeros(eltype(L),n)
    for j in 1:n
        fill!(e,0); e[j]=1; mul!(view(M,:,j),L,e)
    end
    M
end

"""
    steady_state(L; basis=nothing, trace_vector=nothing, method=:auto,
                 shift=nothing, maxiter=200, initial_state=nothing,
                 atol=1e-10, rtol=1e-8, return_info=false,
                 diagnostics=:basic, krylovdim=30, recycle_dim=0, workspace=nothing,
                 preconditioner=nothing, memory_budget=512*1024^2)

Solve `L*rho = 0` subject to the physical trace constraint. Passing a
`PIModel`, a `basis`, or a matrix-free Liouvillian supplies the exact
equation-(7) trace functional. `method=:direct` uses a bordered sparse solve;
`:svd` returns the minimum-norm member of a possibly degenerate stationary
manifold; `:eigen` selects the dense eigenvector closest to zero; and
`:shiftinvert` performs sparse inverse iteration near `shift`. `:krylov`
(`:gmres`) applies a restarted matrix-free GMRES solve with a rank-one trace
constraint and never assembles the Liouvillian. Set `recycle_dim>0` to retain
an augmentation space in a compatible `RecycledGMRESWorkspace` across a
sequence of related solves.
`:auto` validates the direct solve before falling back to SVD.
`diagnostics=:basic` reports residual and trace checks without an extra dense
factorization. Request `diagnostics=:nullity` only when the numerical
stationary-space dimension is needed; this performs an SVD for methods that
do not already require one.

Dense/factorizing routes are preflighted against `memory_budget` using a
conservative PI-coordinate work-array estimate. When `method=:auto` and basic
diagnostics do not fit, the routine selects matrix-free Krylov instead of
starting a dense fallback. An explicitly requested factorizing method raises
before materialization; pass `memory_budget=Inf` only to opt in after checking
available RAM.
For `preconditioner=:schur`, the preflight also includes the block extraction
and LU-factor setup peak. Storage owned by an arbitrary caller-supplied
preconditioner is not introspectable and remains the caller's responsibility.

For the `steady_state(model::PIModel; ...)` route, `coefficient_cache` may be
a compatible [`OneBoxCGCache`](@ref). It accelerates preparation only and
does not change the stationary solve or its numerical tolerances.
"""
function steady_state(L; basis=nothing, trace_vector=nothing, method=:auto,
                      shift=nothing,maxiter::Integer=200,initial_state=nothing,
                      atol=1e-10,rtol=1e-8,return_info=false,
                      diagnostics=:basic,
                      krylovdim::Integer=30,recycle_dim::Integer=0,
                      workspace=nothing,preconditioner=nothing,
                      preconditioner_regularization::Real=0,
                      memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    method isa Symbol||throw(ArgumentError("method must be a Symbol"))
    method=_stationary_solver_method(method)
    diagnostics in (:basic,:nullity) || throw(ArgumentError("diagnostics must be :basic or :nullity"))
    maxiter>0||throw(ArgumentError("maxiter must be positive"))
    L isa MatrixFreeLiouvillian && _require_autonomous(L, "steady_state")
    n=size(L,1);size(L,2)==n||throw(DimensionMismatch("L must be square"))
    source_type=_complex_float_type(eltype(L))
    dense_scalar=promote_type(source_type,ComplexF64)
    dense_arrays=diagnostics===:nullity ? 10 :
        method in (:svd,:eigen,:auto) ? 8 : 6
    dense_estimate=_performance_array_bytes(n,dense_scalar,dense_arrays;
                                             linear_arrays=8)
    if method===:auto&&diagnostics===:basic&&
            !_performance_budget_fits(dense_estimate,memory_budget)
        method=:krylov
    elseif method!==:krylov
        _require_performance_budget("steady-state factorization",dense_estimate,
            memory_budget;guidance=
            "Use method=:krylov for a bounded matrix-free solve.")
    end
    if method===:krylov
        trace_type=trace_vector!==nothing ? eltype(trace_vector) :
            L isa MatrixFreeLiouvillian ? eltype(L.tracevec) : source_type
        initial_type=initial_state isa PIState ? eltype(initial_state.data) :
            initial_state===nothing ? trace_type : eltype(initial_state)
        solver_type=promote_type(source_type,trace_type,initial_type)
        solver_type=_promote_krylov_scalar_type(solver_type,
                                                 preconditioner_regularization)
        if preconditioner!==nothing&&!(preconditioner isa Symbol)
            solver_type=_promote_krylov_operator_type(solver_type,
                                                       preconditioner)
        end
        if workspace!==nothing
            solver_type=promote_type(solver_type,eltype(workspace.V))
        end
        effective_krylovdim=workspace===nothing ? krylovdim :
            size(workspace.H,2)
        effective_recycle_dim=workspace isa RecycledGMRESWorkspace ?
            size(workspace.U,2) : recycle_dim
        krylov_estimate=_performance_gmres_bytes(n,solver_type,
            effective_krylovdim;recycle_dim=effective_recycle_dim)+
            _performance_source_action_bytes(L,solver_type)+
            (L isa LiouvillianPlan ?
                _performance_linear_operator_workspace_bytes(L) : big(0))
        if preconditioner===:schur
            basis===nothing&&throw(ArgumentError(
                "preconditioner=:schur requires basis=... or a PIModel"))
            coefficients=sum(BigInt(basis.offsets[index+1]-
                basis.offsets[index])^2 for index in eachindex(basis.sectors);
                init=big(0))
            krylov_estimate+=_performance_entries_bytes(
                3BigInt(coefficients)+4BigInt(n),solver_type)
        end
        _require_performance_budget("matrix-free steady-state Krylov workspace",
            krylov_estimate,memory_budget;guidance=
            "Reduce krylovdim/recycle_dim or increase the budget.")
        diagnostics===:nullity && throw(ArgumentError("diagnostics=:nullity is not available for the matrix-free Krylov steady-state solve; use method=:svd on a manageable problem"))
        # A LiouvillianPlan deliberately exposes explicit-workspace `apply!`
        # rather than synchronized `mul!`.  Construct its bounded compatibility
        # adapter only after the common Krylov/action-workspace preflight; dense
        # model routes therefore continue to avoid this allocation entirely.
        krylov_source=L isa LiouvillianPlan ? _matrixfree_liouvillian(L) : L
        P = if preconditioner===:schur
            basis===nothing&&throw(ArgumentError("preconditioner=:schur requires basis=... or a PIModel"))
            schur_sector_preconditioner(krylov_source,basis;
                trace_vector=trace_vector,
                regularization=preconditioner_regularization)
        elseif preconditioner isa Symbol
            throw(ArgumentError("unknown Krylov preconditioner $preconditioner"))
        else
            preconditioner
        end
        return krylov_steady_state(krylov_source;basis=basis,
            trace_vector=trace_vector,
            initial_state=initial_state,krylovdim=krylovdim,recycle_dim=recycle_dim,
            workspace=workspace,
            preconditioner=P,maxiter=maxiter,atol=atol,rtol=rtol,
            return_info=return_info)
    end
    M=_materialize(L)
    t = trace_vector !== nothing ? collect(trace_vector) :
        basis !== nothing ? _trace_vector(basis,promote_type(eltype(M),ComplexF64)) :
        L isa MatrixFreeLiouvillian ? collect(L.tracevec) : nothing
    t===nothing && throw(ArgumentError("the physical trace is ambiguous; pass basis=... or trace_vector=..."))
    length(t)==n||throw(DimensionMismatch("trace vector has wrong length"))
    CT=promote_type(eltype(M),eltype(t),ComplexF64); Mc=CT.(M); tc=CT.(t)
    scale=max(opnorm(Mc,Inf),1); tol=atol+rtol*scale
    chosen=method; x=nothing;iterations=0;eigenvalue=nothing;converged=false
    if method===:shiftinvert
        σ=shift===nothing ? -max(sqrt(eps(Float64))*scale,atol) : ComplexF64(shift)
        iszero(σ)&&throw(ArgumentError("shift-invert requires a nonzero shift near the stationary eigenvalue"))
        A=sparse(Mc)-σ*I;fac=lu(A)
        x = initial_state isa PIState ? ComplexF64.(initial_state.data) :
            initial_state===nothing ? tc/dot(tc,tc) : ComplexF64.(initial_state)
        length(x)==n||throw(DimensionMismatch("initial_state has wrong length"));z=dot(tc,x)
        abs(z)>tol||throw(ArgumentError("initial_state must have nonzero physical trace"));x./=z
        for it in 1:maxiter
            y=fac\x;z=dot(tc,y);abs(z)>eps(Float64)||throw(ArgumentError("shift-invert iterate has numerically zero trace"));y./=z
            x=y;iterations=it;res=norm(Mc*x);converged=res<=tol*max(norm(x),1)
            converged&&break
        end
        eigenvalue=dot(x,Mc*x)/dot(x,x)
        converged||throw(ArgumentError("shift-invert did not converge in $maxiter iterations; residual=$(norm(Mc*x))"))
    elseif method===:eigen
        E=eigen(Matrix(Mc));order=sortperm(abs.(E.values));idx=findfirst(i->abs(dot(tc,E.vectors[:,i]))>tol,order)
        idx===nothing&&throw(ArgumentError("no traceful eigenvector was found"));j=order[idx];x=E.vectors[:,j];x./=dot(tc,x)
        eigenvalue=E.values[j];iterations=1;converged=norm(Mc*x)<=tol*max(norm(x),1)
        converged||throw(ArgumentError("nearest traceful eigenvector fails the stationary residual tolerance"))
    end
    if method in (:auto,:direct)
        K=[sparse(Mc) sparse(tc); sparse(adjoint(tc)) spzeros(CT,1,1)]
        rhs=zeros(CT,n+1); rhs[end]=1
        try
            candidate=K\rhs; x=candidate[1:n]
            ok=norm(Mc*x)<=tol*max(norm(x),1) && abs(dot(tc,x)-1)<=atol+rtol
            if !ok
                method===:direct && throw(ArgumentError("bordered steady-state solve failed residual checks"))
                x=nothing
            else
                chosen=:direct
                converged=true
            end
        catch err
            method===:direct && rethrow(err)
            x=nothing
        end
    end
    S = ((method in (:auto,:svd) && x===nothing) || diagnostics===:nullity) ? svd(Matrix(Mc)) : nothing
    svtol = S===nothing ? 0.0 : atol+rtol*(isempty(S.S) ? 0 : maximum(S.S))
    nullity = S===nothing ? nothing : count(<=(svtol),S.S)
    if x===nothing
        nullity>0||throw(ArgumentError("Liouvillian has no numerical nullspace at the requested tolerance"))
        Z=S.V[:,end-nullity+1:end]; w=adjoint(Z)*tc
        norm(w)>tol||throw(ArgumentError("stationary nullspace contains no unit-trace state"))
        x=Z*(w/dot(w,w)); chosen=:svd
        converged=true
    end
    residual=norm(Mc*x); terr=abs(dot(tc,x)-1)
    info=(state=x,residual=residual,trace_error=terr,nullity=nullity,method=chosen,
          iterations=iterations,eigenvalue=eigenvalue,converged=converged,
          diagnostics=diagnostics)
    return_info ? info : x
end
function steady_state(model::PIModel;method=:auto,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,
        coefficient_cache=nothing,kwargs...)
    _require_autonomous(model, "steady_state")
    _require_model_preparation_budget(model,memory_budget;
        operation="steady-state model preparation",coefficient_cache)
    # Keep preparation matrix free until the common budget guard has selected
    # a route. This prevents `method=:auto` from allocating the sparse PI
    # matrix before it has a chance to choose bounded GMRES.
    plan=LiouvillianPlan(model;coefficient_cache)
    steady_state(plan;basis=model.basis,trace_vector=plan.tracevec,
                 method=method,memory_budget,kwargs...)
end

function steady_state(compiled::CompiledPIModel;method=:auto,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,kwargs...)
    _require_autonomous(compiled,"steady_state")
    method isa Symbol||throw(ArgumentError("method must be a Symbol"))
    method=_stationary_solver_method(method)
    representation=method===:krylov ? :matrixfree : compiled.backend
    L=liouvillian(compiled;representation=representation,memory_budget)
    steady_state(L;basis=compiled.plan.basis,trace_vector=compiled.plan.tracevec,
                 method=method,memory_budget,kwargs...)
end

"""Return selected Liouvillian eigenvalues according to `:LR`, `:LM`, or `:SM`."""
function liouvillian_eigenvalues(L,k::Integer;which=:LR,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    k >= 0 || throw(ArgumentError("k must be nonnegative"))
    which in (:LR,:LM,:SM) || throw(ArgumentError("which must be :LR, :LM, or :SM"))
    n=size(L,1);size(L,2)==n||throw(DimensionMismatch("L must be square"))
    T=promote_type(_complex_float_type(eltype(L)),ComplexF64)
    estimate=_performance_array_bytes(n,T,5;linear_arrays=4)
    _require_performance_budget("complete Liouvillian eigendecomposition",
        estimate,memory_budget;guidance=
        "Use krylov_liouvillian_spectrum for selected eigenvalues.")
    values=eigvals(Matrix(_materialize(L)))
    order = which===:LR ? sortperm(values;by=real,rev=true) :
            which===:LM ? sortperm(values;by=abs,rev=true) :
                          sortperm(values;by=abs)
    values[order[1:min(k,length(order))]]
end
