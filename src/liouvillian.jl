"""
    MatrixFreeLiouvillian(n, action!, T, trace_vector;
                          autonomous=true, plan=nothing, workspace=nothing)

Matrix-free `n x n` Liouvillian whose callback implements
`action!(destination, source, time, parameters)`. `T` is its scalar type and
`trace_vector` encodes the physical trace functional. Compatibility calls are
synchronized; compiled PI hot loops should use `compile`, `apply!`, and one
explicit `LiouvillianWorkspace` per task.
"""
struct MatrixFreeLiouvillian{F,T,V,P,W,K}
    n::Int
    action!::F
    Ttype::Type{T}
    tracevec::V
    autonomous::Bool
    plan::P
    workspace::W
    lock::K
end

function MatrixFreeLiouvillian(n::Integer, action!, ::Type{T}, tracevec;
                               autonomous::Bool=true,plan=nothing,
                               workspace=nothing) where T
    n > 0 || throw(ArgumentError("Liouvillian dimension must be positive"))
    length(tracevec) == n || throw(DimensionMismatch("trace vector has the wrong length"))
    action_lock=ReentrantLock()
    # `action!` is a compatibility surface used by older integrations.  Keep
    # it safe when one compiled Liouvillian is shared between tasks; new hot
    # loops should pass an explicit LiouvillianWorkspace to `apply!` instead.
    safe_action! = function (y,x,t,p)
        lock(action_lock)
        try
            action!(y,x,t,p)
        finally
            unlock(action_lock)
        end
    end
    MatrixFreeLiouvillian{typeof(safe_action!),T,typeof(tracevec),
                          typeof(plan),typeof(workspace),typeof(action_lock)}(
        Int(n),safe_action!,T,tracevec,autonomous,plan,workspace,action_lock)
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
    if L.plan===nothing
        for j in axes(X,2)
            L.action!(view(Y,:,j),view(X,:,j),0.0,nothing)
        end
    else
        lock(L.lock)
        try
            for j in axes(X,2)
                apply!(view(Y,:,j),L.plan,view(X,:,j),0.0,nothing,L.workspace)
            end
        finally
            unlock(L.lock)
        end
    end
    Y
end

# Common explicit-time action used by both the fixed-step and SciML adapters.
_liouvillian_action!(y,L::AbstractMatrix,x,t,p)=mul!(y,L,x)
_liouvillian_action!(y,L::MatrixFreeLiouvillian,x,t,p)=L.action!(y,x,t,p)

function _block_superop(b::PIBasis,blocks,kind)
    T=isempty(blocks) ? ComplexF64 : eltype(first(blocks))
    rows=Int[];cols=Int[];V=T[]
    for (s,p) in pairs(b.sectors)
        K=blocks[s]; n=size(K,1); off=b.offsets[s]-1
        M = kind===:commutator ? commutator_superoperator(K) : dissipator_superoperator(K)
        ii,jj,vv=findnz(sparse(M));append!(rows,ii.+off);append!(cols,jj.+off);append!(V,vv)
    end
    sparse(rows,cols,V,length(b),length(b))
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

# The nonzero numerical pattern of a local gain map depends on the evaluated
# jump matrix.  An in-place operator schedule therefore retains every
# representation-theoretically allowed coordinate and fills only its values
# in the task-local workspace.  This preserves interference between all
# entries of L(t), including entries that vanish in the prototype.
function _local_gain_structure(b,cache)
    I=Int[];J=Int[]
    for (li,l) in pairs(b.sectors),(ni,n) in pairs(b.sectors)
        isempty(cache.connections[(li,ni)])&&continue
        nl=length(b.patterns[li]);nn=length(b.patterns[ni])
        for a in 1:nl,bb in 1:nl,c in 1:nn,d in 1:nn
            push!(I,b.offsets[li]+a-1+(bb-1)*nl)
            push!(J,b.offsets[ni]+c-1+(d-1)*nn)
        end
    end
    (;I,J)
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


# In-place schedules keep all mutable evaluated data outside the plan. The
# builders below are read-only handles to prepared representation geometry.
abstract type AbstractDynamicPIKernel end
struct CollectiveOneBodyBlockBuilder{G}
    geometry::G
end
struct CollectivePBodyBlockBuilder{G,P,E}
    geometry::G
    permutations::P
    block_entries::E
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
struct InPlaceLocalJumpPIKernel{S,G,R,I,J} <: AbstractDynamicPIKernel
    schedule::S;geometry::G;scale::R
    I::I;J::J
end
struct InPlaceLocalPBodyJumpPIKernel{S,B,R,G,L,U,Q} <: AbstractDynamicPIKernel
    schedule::S;builder::B;scale::R
    groups::G
    left_isometries::L;right_isometries::U;pair_scales::Q
end

struct InPlaceHamiltonianKernelWorkspace{O,B}
    operator::O;blocks::B
end
struct InPlaceDissipatorKernelWorkspace{O,B,Q}
    operator::O;blocks::B;qblocks::Q
end
struct InPlaceLocalJumpKernelWorkspace{O,Q,B,V}
    operator::O;qoperator::Q;qblocks::B;values::V
end
struct InPlaceLocalPBodyJumpKernelWorkspace{O,Q,B,C,S}
    operator::O;qoperator::Q;qblocks::B;contractions::C;gain_scratch::S
end

function _evaluated_dissipative_rate(rate,t,p)
    evaluated=value_at(rate,t,p)
    evaluated isa Real||throw(ArgumentError(
        "a dissipative rate must evaluate to a real number, got $(typeof(evaluated))"))
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
struct TermCompileContext{B,G,P,T}
    basis::B
    onebody::G
    pbody::P
    geometry_type::Type{T}
end

_operator_real_type(::Function)=Float64
_operator_real_type(operator::InPlaceTimeOperator)=
    _operator_real_type(operator.prototype)
_operator_real_type(operator::AbstractPIOperator)=_real_float_type(eltype(operator.data))
_operator_real_type(operator::AbstractArray)=_real_float_type(eltype(operator))
_operator_real_type(operator)=Float64

function _model_geometry_type(model::PIModel)
    types=map(t->_operator_real_type(term_operator(t)),model.terms)
    isempty(types) ? Float64 : foldl(promote_type,types)
end

function TermCompileContext(model::PIModel{B}) where {D,L,B<:PIBasis{D,L}}
    b=model.basis;T=_model_geometry_type(model)
    onebody=any(_term_requires_onebody_geometry,model.terms) ? OneBodyGeometry(b,T) : nothing
    pbody=Dict{Int,PBodyGeometry{T,D,L,B}}()
    TermCompileContext(b,onebody,pbody,T)
end

# Direct-PI and Appendix-D terms have their own lowering geometry.  Avoid the
# much larger one-box setup for models containing only those built-ins.  A
# custom term remains conservative because its delegated lowering is unknown.
_term_requires_onebody_geometry(::AbstractPITerm)=true
_term_requires_onebody_geometry(::Union{DirectPIHamiltonian,DirectPIJump,
    PBodyHamiltonian,LocalPBodyJump,CollectivePBodyJump})=false

function _pbody_geometry!(context::TermCompileContext,order::Integer)
    get!(()->PBodyGeometry(context.basis,order,context.geometry_type),context.pbody,Int(order))
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
    CollectivePBodyBlockBuilder(geometry,permutations,block_entries,
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

function _fill_dynamic_blocks!(blocks,builder::CollectivePBodyBlockBuilder,X)
    geometry=builder.geometry;b=geometry.basis
    for s in eachindex(b.sectors)
        K=blocks[s];fill!(K,zero(eltype(K)))
        for (U,scale) in builder.block_entries[s]
            rows,centers,local_dimension=size(U)
            if scale.direct
                factor=scale.factor
                @inbounds for column in 1:rows,row in 1:rows,w in 1:centers,
                              j in 1:local_dimension,i in 1:local_dimension
                    K[row,column]+=factor*U[row,w,i]*U[column,w,j]*X[i,j]
                end
            else
                @inbounds for column in 1:rows,row in 1:rows,w in 1:centers,
                              j in 1:local_dimension,i in 1:local_dimension
                    contribution=U[row,w,i]*U[column,w,j]*X[i,j]
                    K[row,column]+=_apply_prepared_exact_scale(
                        contribution,scale;
                        context="dynamic collective p-body path contribution")
                end
            end
        end
        if builder.cancellation_risk
            operation_count=sum(builder.block_entries[s];init=1) do entry
                U=entry[1]
                size(U,2)*size(U,3)^2+1
            end
            for column in axes(K,2),row in axes(K,1)
                absolute_sum=zero(_real_float_type(eltype(K)))
                for (U,exact_scale) in builder.block_entries[s]
                    rows,centers,local_dimension=size(U)
                    for w in 1:centers,j in 1:local_dimension,i in 1:local_dimension
                        primitive=U[row,w,i]*U[column,w,j]*X[i,j]
                        contribution=exact_scale.direct ? exact_scale.factor*primitive :
                            _apply_prepared_exact_scale(primitive,exact_scale;
                                context="dynamic collective p-body cancellation check")
                        absolute_sum+=abs(contribution)
                    end
                end
                _dynamic_pbody_block_uncertified(
                    K[row,column],absolute_sum,operation_count)&&
                    throw(ArgumentError(
                        "the evaluated dynamic p-body block cannot certify its " *
                        "working-precision result through large-N path accumulation; " *
                        "use a wider InPlaceTimeOperator prototype scalar type"))
            end
        end
    end
    blocks
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

function _pbody_gain_factorization(builder::CollectivePBodyBlockBuilder)
    geometry=builder.geometry;b=geometry.basis;T=geometry_scalar_type(geometry)
    builder.cancellation_risk&&throw(ArgumentError(
        "dynamic local p-body gain factors are cancellation-prone at scalar " *
        "type $T, including when every individual factor is representable; " *
        "use a wider InPlaceTimeOperator prototype scalar type"))
    groups=NTuple{4,Int}[]
    left_isometries=Array{T,3}[];right_isometries=Array{T,3}[]
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
    any(scale->!scale.direct,pair_scales)&&throw(ArgumentError(
        "dynamic local p-body gain factors exceed the nonzero finite range of $T; " *
        "use a wider InPlaceTimeOperator prototype scalar type so the preallocated " *
        "quadratic contraction scratch cannot underflow before exact rescaling"))
    (;groups,left_isometries,right_isometries,pair_scales)
end

function _fill_dynamic_blocks!(blocks,builder::CollectiveOneBodyBlockBuilder,X)
    cache=builder.geometry;b=cache.basis
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

function _dynamic_onebody_builder(context::TermCompileContext)
    cache=context.onebody;T=geometry_scalar_type(cache)
    _needs_wide_collective(cache.basis,T)&&throw(ArgumentError(
        "preallocated collective one-body blocks at N=$(cache.basis.N) cannot " *
        "certify large-N cancellation in fixed $T scratch; use a wider " *
        "InPlaceTimeOperator prototype scalar type"))
    CollectiveOneBodyBlockBuilder(cache)
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

_collective_blocks(operator,context)=
    [collective_block(context.basis,operator,p;cache=context.onebody) for p in context.basis.sectors]
_direct_term_blocks(operator,context)=_direct_blocks(context.basis,operator)

function _compile_hamiltonian(term,context,blocks)
    R=_real_float_type(eltype(first(blocks)))
    HamiltonianPIKernel(blocks,_scaled_rate(term_rate(term),term_hbar(term),R))
end
function compile_term(t::Union{LocalHamiltonian,CollectiveHamiltonian},
                      context::TermCompileContext)
    operator=term_operator(t);R=context.geometry_type
    operator isa InPlaceTimeOperator && return InPlaceHamiltonianPIKernel(
        operator,_dynamic_builder(context,t),_scaled_rate(term_rate(t),term_hbar(t),R))
    _compile_hamiltonian(t,context,_collective_blocks(operator,context))
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
function compile_term(t::CollectiveJump,context::TermCompileContext)
    operator=term_operator(t)
    operator isa InPlaceTimeOperator && return InPlaceDissipatorPIKernel(
        operator,_dynamic_builder(context,t),term_rate(t))
    _compile_dissipator(t,_collective_blocks(operator,context))
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
        structure=_local_gain_structure(context.basis,context.onebody)
        return InPlaceLocalJumpPIKernel(operator,context.onebody,term_rate(t),
            structure.I,structure.J)
    end
    Q=_collective_blocks(operator'*operator,context)
    LocalJumpPIKernel(Q,_local_kernel_triplets(context.basis,context.onebody,
                                               operator,operator),term_rate(t))
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
    LocalJumpPIKernel(Q,pbody_kernel_triplets(geometry,operator,operator),term_rate(t))
end

function _static_kernels(model)
    context=TermCompileContext(model)
    map(t->compile_term(t,context),model.terms)
end

"""Immutable prepared term data and geometry for a PI Liouvillian."""
struct LiouvillianPlan{B,K,V,M,T}
    basis::B
    kernels::K
    tracevec::V
    fallback_model::M
    Ttype::Type{T}
    autonomous::Bool
end

_scale_promoted_type(T,scale)=scale isa Number ? promote_type(T,typeof(scale)) : T
_kernel_scalar_type(kernel::HamiltonianPIKernel)=_scale_promoted_type(eltype(first(kernel.blocks)),kernel.scale)
_kernel_scalar_type(kernel::DissipatorPIKernel)=_scale_promoted_type(eltype(first(kernel.blocks)),kernel.scale)
_kernel_scalar_type(kernel::LocalJumpPIKernel)=_scale_promoted_type(
    promote_type(eltype(kernel.gain.V),eltype(first(kernel.qblocks))),kernel.scale)
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

function LiouvillianPlan(model::PIModel)
    fixed_operators=all(term_has_fixed_operator,model.terms)
    prepared_operators=all(t->term_has_fixed_operator(t)||term_has_preallocated_operator(t),
                           model.terms)
    kernels=prepared_operators ? _static_kernels(model) : nothing
    fallback=fixed_operators ? nothing : model
    T = kernels===nothing||isempty(kernels) ? ComplexF64 :
        foldl(promote_type,(_kernel_scalar_type(k) for k in kernels))
    LiouvillianPlan(model.basis,kernels,_trace_vector(model.basis,T),
                    fallback,T,isautonomous(model))
end

size(plan::LiouvillianPlan)=(length(plan.basis),length(plan.basis))
size(plan::LiouvillianPlan,i::Integer)=i in (1,2) ? length(plan.basis) : 1
eltype(plan::LiouvillianPlan)=plan.Ttype
isautonomous(plan::LiouvillianPlan)=plan.autonomous

"""Per-task mutable scratch for applying a `LiouvillianPlan`."""
struct LiouvillianWorkspace{B,W,K,T}
    basis::B
    blocks::W
    kernel_workspaces::K
    Ttype::Type{T}
end

_operator_workspace(prototype::AbstractMatrix)=Matrix(prototype)
_operator_workspace(prototype::AbstractPIOperator)=copy(prototype)
function _dynamic_block_workspace(b,T)
    [zeros(T,length(b.patterns[s]),length(b.patterns[s]))
     for s in eachindex(b.sectors)]
end
_kernel_workspace(::AbstractStaticPIKernel,b,T)=nothing
function _kernel_workspace(kernel::InPlaceHamiltonianPIKernel,b,T)
    InPlaceHamiltonianKernelWorkspace(_operator_workspace(kernel.schedule.prototype),
                                      _dynamic_block_workspace(b,T))
end
function _kernel_workspace(kernel::InPlaceDissipatorPIKernel,b,T)
    blocks=_dynamic_block_workspace(b,T)
    InPlaceDissipatorKernelWorkspace(_operator_workspace(kernel.schedule.prototype),
                                     blocks,_dynamic_block_workspace(b,T))
end
function _kernel_workspace(kernel::InPlaceLocalJumpPIKernel,b,T)
    operator=_operator_workspace(kernel.schedule.prototype)
    qoperator=similar(operator,size(operator))
    InPlaceLocalJumpKernelWorkspace(operator,qoperator,
        _dynamic_block_workspace(b,T),zeros(T,length(kernel.I)))
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
        zeros(T,largest_block,largest_block))
end

function LiouvillianWorkspace(plan::LiouvillianPlan)
    T=plan.Ttype
    work=[(zeros(T,length(plan.basis.patterns[s]),length(plan.basis.patterns[s])),
           zeros(T,length(plan.basis.patterns[s]),length(plan.basis.patterns[s])),
           zeros(T,length(plan.basis.patterns[s]),length(plan.basis.patterns[s])))
          for s in eachindex(plan.basis.sectors)]
    kernel_workspaces=plan.kernels===nothing ? nothing :
        map(kernel->_kernel_workspace(kernel,plan.basis,T),plan.kernels)
    LiouvillianWorkspace(plan.basis,work,kernel_workspaces,T)
end

function _check_liouvillian_workspace(work::LiouvillianWorkspace,plan::LiouvillianPlan)
    work.basis===plan.basis||throw(ArgumentError("Liouvillian workspace belongs to a different PI basis"))
    work.Ttype===plan.Ttype||throw(ArgumentError("Liouvillian workspace has an incompatible scalar type"))
    work
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
    _fill_dynamic_blocks!(work.blocks,kernel.builder,work.operator)
    nothing
end
function _prepare_kernel!(kernel::InPlaceDissipatorPIKernel,
                          work::InPlaceDissipatorKernelWorkspace,b,t,p)
    _evaluate_time_operator!(work.operator,kernel.schedule,t,p)
    _check_dynamic_pbody_operator(kernel.builder,work.operator)
    _fill_dynamic_blocks!(work.blocks,kernel.builder,work.operator)
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
    _fill_dynamic_blocks!(work.qblocks,kernel.builder,work.qoperator)
    for index in eachindex(work.contractions)
        _path_contractions!(work.contractions[index],kernel.left_isometries[index],
                            kernel.right_isometries[index],work.operator)
    end
    nothing
end
function _prepare_kernel!(kernel::InPlaceLocalJumpPIKernel,
                          work::InPlaceLocalJumpKernelWorkspace,b,t,p)
    _evaluate_time_operator!(work.operator,kernel.schedule,t,p)
    mul!(work.qoperator,adjoint(work.operator),work.operator)
    _fill_dynamic_blocks!(work.qblocks,
        CollectiveOneBodyBlockBuilder(kernel.geometry),work.qoperator)
    index=0
    @inbounds for (li,l) in pairs(b.sectors),(ni,n) in pairs(b.sectors)
        isempty(kernel.geometry.connections[(li,ni)])&&continue
        nl=length(b.patterns[li]);nn=length(b.patterns[ni])
        for a in 1:nl,bb in 1:nl,c in 1:nn,d in 1:nn
            index+=1
            work.values[index]=local_kernel_element(kernel.geometry,
                work.operator,work.operator,l,a,bb,n,c,d)
        end
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
    gain=(I=kernel.I,J=kernel.J,V=prepared.values)
    _apply_kernel!(y,x,LocalJumpPIKernel(prepared.qblocks,gain,kernel.scale),b,t,p,work)
end
function _apply_adjoint_prepared_kernel!(y,x,kernel::InPlaceLocalJumpPIKernel,
                                         prepared::InPlaceLocalJumpKernelWorkspace,
                                         b,t,p,work)
    gain=(I=kernel.I,J=kernel.J,V=prepared.values)
    _apply_adjoint_kernel!(y,x,LocalJumpPIKernel(prepared.qblocks,gain,kernel.scale),
                           b,t,p,work)
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

function _apply_factorized_pbody_gain!(y,kernel,prepared,b,scale,work)
    @inbounds for (li,ni,first_pair,last_pair) in kernel.groups
        nl=length(b.patterns[li]);nn=length(b.patterns[ni])
        input=work[ni][3];output=work[li][1]
        scratch=@view prepared.gain_scratch[1:nl,1:nn]
        output_offset=b.offsets[li]
        for pair in first_pair:last_pair
            contraction=prepared.contractions[pair]
            mul!(scratch,contraction,input)
            mul!(output,scratch,adjoint(contraction))
            exact_scale=kernel.pair_scales[pair]
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

function _apply_adjoint_factorized_pbody_gain!(y,kernel,prepared,b,scale,work)
    @inbounds for (li,ni,first_pair,last_pair) in kernel.groups
        nl=length(b.patterns[li]);nn=length(b.patterns[ni])
        input=work[li][3];output=work[ni][1]
        scratch=@view prepared.gain_scratch[1:nn,1:nl]
        output_offset=b.offsets[ni]
        for pair in first_pair:last_pair
            contraction=prepared.contractions[pair]
            mul!(scratch,adjoint(contraction),input)
            mul!(output,scratch,contraction)
            exact_scale=kernel.pair_scales[pair]
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

function _apply_prepared_kernel!(y,x,kernel::InPlaceLocalPBodyJumpPIKernel,
        prepared::InPlaceLocalPBodyJumpKernelWorkspace,b,t,p,work)
    scale=convert(eltype(work[1][1]),
                  _evaluated_dissipative_rate(kernel.scale,t,p))
    _copy_input_blocks!(work,x,b)
    _apply_factorized_pbody_gain!(y,kernel,prepared,b,scale,work)
    _apply_local_jump_anticommutator!(y,prepared.qblocks,b,scale,work)
end
function _apply_adjoint_prepared_kernel!(y,x,kernel::InPlaceLocalPBodyJumpPIKernel,
        prepared::InPlaceLocalPBodyJumpKernelWorkspace,b,t,p,work)
    scale=conj(convert(eltype(work[1][1]),
                       _evaluated_dissipative_rate(kernel.scale,t,p)))
    _copy_input_blocks!(work,x,b)
    _apply_adjoint_factorized_pbody_gain!(y,kernel,prepared,b,scale,work)
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
    for j in axes(X,2)
        _apply_prepared_vector!(view(Y,:,j),plan,view(X,:,j),t,p,work)
    end
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
    for j in axes(X,2)
        output=view(Y,:,j);input=view(X,:,j)
        fill!(output,zero(eltype(output)))
        _apply_adjoint_kernels!(output,input,plan.kernels,work.kernel_workspaces,
                                plan.basis,t,p,work.blocks)
    end
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
    anti=spzeros(plan.Ttype,n,n)
    for s in eachindex(plan.basis.sectors)
        off=plan.basis.offsets[s]-1;Q=ker.qblocks[s]
        M=(left_superoperator(Q)+right_superoperator(Q))/2
        ii,jj,vv=findnz(sparse(M));anti+=sparse(ii.+off,jj.+off,scale.*vv,n,n)
    end
    gain-anti
end

_kernel_matrix(plan,kernel::HamiltonianPIKernel,scale)=
    scale*_block_superop(plan.basis,kernel.blocks,:commutator)
_kernel_matrix(plan,kernel::DissipatorPIKernel,scale)=
    scale*_block_superop(plan.basis,kernel.blocks,:dissipator)
_kernel_matrix(plan,kernel::LocalJumpPIKernel,scale)=
    _local_jump_matrix(plan,kernel,scale)

_materialized_kernel_scale(kernel::HamiltonianPIKernel)=value_at(kernel.scale,0.0,nothing)
_materialized_kernel_scale(kernel::Union{DissipatorPIKernel,LocalJumpPIKernel})=
    _evaluated_dissipative_rate(kernel.scale,0.0,nothing)

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
    liouvillian(model; representation=:matrixfree)
    liouvillian(compiled; representation=compiled.backend)

Return a PI-coordinate Liouvillian as either a sparse matrix or a
`MatrixFreeLiouvillian`. Model construction lowers from the same prepared
term plan in both representations. Sparse materialization requires an
autonomous model; use `freeze` at an explicit time or the matrix-free backend
for driven dynamics. Prefer `compile` when the generator will be reused.
"""
function liouvillian(model::PIModel;representation=:matrixfree)
    representation in (:sparse,:matrixfree)||throw(ArgumentError("representation must be :sparse or :matrixfree"))
    plan=LiouvillianPlan(model)
    representation===:sparse ? _matrix_from_plan(plan) : _matrixfree_liouvillian(plan)
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
    memory_budget isa Real||throw(ArgumentError("memory_budget must be a number of bytes"))
    memory_budget>=0||throw(ArgumentError("memory_budget must be nonnegative"))
    isfinite(memory_budget) ? Int(min(floor(BigInt,memory_budget),BigInt(typemax(Int)))) : typemax(Int)
end

"""
    compile(model; backend=:auto, memory_budget=512*1024^2,
            bigfloat_precision=precision(BigFloat))

Prepare all fixed Schur geometry once and choose a sparse or matrix-free
backend. `backend=:auto` uses a conservative sparse-storage upper bound and
always keeps driven models matrix-free. The returned `CompiledPIModel` can be
passed directly to `apply!`, `evolve!`, and `dynamics_problem`.

For fixed-size isbits scalar types, storage bounds retain the exact inline
`sizeof(T)` accounting. `BigFloat` and `Complex{BigFloat}` use an explicitly
conservative retained-storage bound at `bigfloat_precision`; pass the maximum
precision intended for generated matrix entries when it differs from the
active process precision.
"""
function compile(model::PIModel;backend=:auto,memory_budget=512*1024^2,
                 bigfloat_precision::Integer=precision(BigFloat))
    backend in (:auto,:sparse,:matrixfree)||throw(ArgumentError("backend must be :auto, :sparse, or :matrixfree"))
    budget=_memory_budget_bytes(memory_budget);plan=LiouvillianPlan(model);n=length(model.basis)
    T=plan.Ttype
    scalar_bytes=_scalar_retained_bytes(T;bigfloat_precision)
    dense_entries=BigInt(n)^2
    sparse_upper_big=dense_entries*(scalar_bytes+2sizeof(Int))
    sparse_upper=Int(min(sparse_upper_big,BigInt(typemax(Int))))
    plan_bytes=Base.summarysize(plan)
    sparse_total=Int(min(BigInt(plan_bytes)+sparse_upper_big,BigInt(typemax(Int))))
    matrixfree_total=Int(min(BigInt(plan_bytes)+3BigInt(n)*scalar_bytes,BigInt(typemax(Int))))
    chosen = backend===:auto ? (plan.autonomous&&sparse_total<=budget ? :sparse : :matrixfree) : backend
    chosen===:sparse&&!plan.autonomous&&throw(ArgumentError("a time-dependent model cannot use the sparse backend; freeze it at an explicit time or use backend=:matrixfree"))
    operator=chosen===:sparse ? _matrix_from_plan(plan) : _matrixfree_liouvillian(plan)
    estimates=(scalar_type=T,dimension=n,plan_bytes=plan_bytes,
               scalar_retained_bytes=scalar_bytes,
               scalar_storage_estimate=_scalar_storage_estimate(T),
               bigfloat_precision_assumption=
                   _scalar_precision_assumption(T,bigfloat_precision),
               sparse_upper_bound=sparse_upper,sparse_operator_upper_bound=sparse_upper,
               sparse_compiled_upper_bound=sparse_total,
               matrixfree_compiled_estimate=matrixfree_total,memory_budget=budget,
               requested_backend=backend,chosen_backend=chosen)
    CompiledPIModel(model,plan,operator,chosen,estimates)
end

function liouvillian(compiled::CompiledPIModel;representation=compiled.backend)
    representation in (:sparse,:matrixfree)||throw(ArgumentError("representation must be :sparse or :matrixfree"))
    representation===compiled.backend&&return compiled.operator
    representation===:sparse ? _matrix_from_plan(compiled.plan) : _matrixfree_liouvillian(compiled.plan)
end

LiouvillianWorkspace(compiled::CompiledPIModel)=LiouvillianWorkspace(compiled.plan)
function LiouvillianWorkspace(L::MatrixFreeLiouvillian)
    L.plan===nothing&&throw(ArgumentError("this custom MatrixFreeLiouvillian has no compiled plan workspace"))
    LiouvillianWorkspace(L.plan)
end

function apply!(y,L::MatrixFreeLiouvillian,x,t,p,work::LiouvillianWorkspace)
    L.plan===nothing ? L.action!(y,x,t,p) : apply!(y,L.plan,x,t,p,work)
end
function apply!(Y::AbstractMatrix,L::MatrixFreeLiouvillian,X::AbstractMatrix,t,p,
                work::LiouvillianWorkspace)
    L.plan===nothing ? error("a custom MatrixFreeLiouvillian does not support explicit batched workspaces") :
        apply!(Y,L.plan,X,t,p,work)
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

function apply_adjoint!(y,L::MatrixFreeLiouvillian,x,t,p)
    if L.plan===nothing
        _require_autonomous(L,"apply_adjoint!")
        return mul!(y,adjoint(Matrix(_materialize(L))),x)
    end
    lock(L.lock)
    try
        apply_adjoint!(y,L.plan,x,t,p,L.workspace)
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
    fallback=L.plan===nothing ? adjoint(Matrix(_materialize(L))) : nothing
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
    for j in axes(X,2)
        mul!(view(Y,:,j),A,view(X,:,j))
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
                                 basis=compiled.plan.basis,kwargs...)
    representation=method in (:krylov,:arnoldi,:harmonic) ? :matrixfree : compiled.backend
    source=liouvillian(compiled;representation=representation)
    pi_liouvillian_spectrum(source;method=method,basis=basis,kwargs...)
end

function pi_liouvillian_gap(compiled::CompiledPIModel;method=:dense,
                            basis=compiled.plan.basis,kwargs...)
    representation=method in (:krylov,:arnoldi,:harmonic) ? :matrixfree : compiled.backend
    source=liouvillian(compiled;representation=representation)
    pi_liouvillian_gap(source;method=method,basis=basis,kwargs...)
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
        MatrixFreeLiouvillian(L.n,action!,L.Ttype,copy(L.tracevec);autonomous=true)
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
                 diagnostics=:basic, krylovdim=30, workspace=nothing,
                 preconditioner=nothing)

Solve `L*rho = 0` subject to the physical trace constraint. Passing a
`PIModel`, a `basis`, or a matrix-free Liouvillian supplies the exact
equation-(7) trace functional. `method=:direct` uses a bordered sparse solve;
`:svd` returns the minimum-norm member of a possibly degenerate stationary
manifold; `:eigen` selects the dense eigenvector closest to zero; and
`:shiftinvert` performs sparse inverse iteration near `shift`. `:krylov`
(`:gmres`) applies a restarted matrix-free GMRES solve with a rank-one trace
constraint and never assembles the Liouvillian.
`:auto` validates the direct solve before falling back to SVD.
`diagnostics=:basic` reports residual and trace checks without an extra dense
factorization. Request `diagnostics=:nullity` only when the numerical
stationary-space dimension is needed; this performs an SVD for methods that
do not already require one.
"""
function steady_state(L; basis=nothing, trace_vector=nothing, method=:auto,
                      shift=nothing,maxiter::Integer=200,initial_state=nothing,
                      atol=1e-10,rtol=1e-8,return_info=false,
                      diagnostics=:basic,
                      krylovdim::Integer=30,workspace=nothing,preconditioner=nothing,
                      preconditioner_regularization::Real=0)
    method in (:shift_invert,:inverse_iteration)&&(method=:shiftinvert)
    method===:gmres&&(method=:krylov)
    method in (:auto,:direct,:svd,:eigen,:shiftinvert,:krylov) || throw(ArgumentError("method must be :auto, :direct, :svd, :eigen, :shiftinvert, or :krylov"))
    diagnostics in (:basic,:nullity) || throw(ArgumentError("diagnostics must be :basic or :nullity"))
    maxiter>0||throw(ArgumentError("maxiter must be positive"))
    L isa MatrixFreeLiouvillian && _require_autonomous(L, "steady_state")
    if method===:krylov
        diagnostics===:nullity && throw(ArgumentError("diagnostics=:nullity is not available for the matrix-free Krylov steady-state solve; use method=:svd on a manageable problem"))
        P = if preconditioner===:schur
            basis===nothing&&throw(ArgumentError("preconditioner=:schur requires basis=... or a PIModel"))
            schur_sector_preconditioner(L,basis;trace_vector=trace_vector,
                regularization=preconditioner_regularization)
        elseif preconditioner isa Symbol
            throw(ArgumentError("unknown Krylov preconditioner $preconditioner"))
        else
            preconditioner
        end
        return krylov_steady_state(L;basis=basis,trace_vector=trace_vector,
            initial_state=initial_state,krylovdim=krylovdim,workspace=workspace,
            preconditioner=P,maxiter=maxiter,atol=atol,rtol=rtol,
            return_info=return_info)
    end
    M=_materialize(L); n=size(M,1); size(M,2)==n||throw(DimensionMismatch("L must be square"))
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
function steady_state(model::PIModel;method=:auto,kwargs...)
    _require_autonomous(model, "steady_state")
    representation=method in (:krylov,:gmres) ? :matrixfree : :sparse
    steady_state(liouvillian(model;representation=representation);basis=model.basis,method=method,kwargs...)
end

function steady_state(compiled::CompiledPIModel;method=:auto,kwargs...)
    _require_autonomous(compiled,"steady_state")
    representation=method in (:krylov,:gmres) ? :matrixfree : compiled.backend
    L=liouvillian(compiled;representation=representation)
    steady_state(L;basis=compiled.plan.basis,trace_vector=compiled.plan.tracevec,
                 method=method,kwargs...)
end

"""Return selected Liouvillian eigenvalues according to `:LR`, `:LM`, or `:SM`."""
function liouvillian_eigenvalues(L,k::Integer;which=:LR)
    k >= 0 || throw(ArgumentError("k must be nonnegative"))
    which in (:LR,:LM,:SM) || throw(ArgumentError("which must be :LR, :LM, or :SM"))
    values=eigvals(Matrix(_materialize(L)))
    order = which===:LR ? sortperm(values;by=real,rev=true) :
            which===:LM ? sortperm(values;by=abs,rev=true) :
                          sortperm(values;by=abs)
    values[order[1:min(k,length(order))]]
end
