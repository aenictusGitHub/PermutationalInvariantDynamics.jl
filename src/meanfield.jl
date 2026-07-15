import SciMLBase

# Product-state (first-cumulant) closure.  The rules below deliberately lower
# from the public term semantics rather than from Schur kernels: their cost is
# controlled by the local dimension and body order, and is independent of the
# number of Schur sectors.
abstract type _AbstractMeanFieldRule end

struct _MFHamiltonianRule{O,R,H,C} <: _AbstractMeanFieldRule
    operator::O
    rate::R
    hbar::H
    order::Int
    count::C
end

struct _MFLocalJumpRule{O,Q,A,R,C} <: _AbstractMeanFieldRule
    operator::O
    quadratic::Q
    adjoint_operator::A
    rate::R
    order::Int
    count::C
end

struct _MFCollectiveJumpRule{O,Q,A,R,C} <: _AbstractMeanFieldRule
    operator::O
    quadratic::Q
    adjoint_operator::A
    rate::R
    finite_cross_count::C
    thermodynamic_cross_count::C
end

# For J=sum_{|S|=p} L_S, D[J] contains ordered pairs of subsets (S,T).
# After tracing to particle 1, only pairs with 1 in S union T survive.  Their
# contractions depend only on |S intersection T| and can be expressed through
# the effective operators
#
#   E_m(sigma)=Tr_{m+1:p}[L (I^m tensor sigma^(p-m))], 0 <= m <= p.
#
# Keeping the exact overlap counts in the immutable rule lets the hot path use
# only matrices of dimension at most d^p, rather than constructing operators on
# the union of two p-particle subsets.
struct _MFCollectivePBodyJumpRule{O,R,S,C,T} <: _AbstractMeanFieldRule
    operator::O
    rate::R
    order::Int
    same_tag_overlap_counts::S
    one_tag_overlap_counts::C
    thermodynamic_cross_count::T
end

"""
    MeanFieldPlan(model::PIModel; limit=:finite)
    MeanFieldPlan(N, d, terms; limit=:finite)

Prepare the finite-`N` product-state closure
`tr_2...N(L(sigma^otimes N))`.  `limit=:thermodynamic` retains the leading
`N^(p-1)/(p-1)!` subset count for local/Hamiltonian `p`-body terms.  For a
collective `p`-body jump it keeps the leading disjoint-subset field with count
`N^(2p-1)/((p-1)!p!)` and drops subleading overlap/self terms.  Rates are used
exactly as supplied; no Kac normalization is inferred or inserted.

Operators are copied into the immutable plan.  Fixed one- and symmetric
`p`-body Hamiltonians, independent jumps, and collective `p`-body jumps are
supported.  Direct PI terms, operator-valued time functions, and custom terms
without an explicit mean-field lowering are rejected explicitly.
"""
struct MeanFieldPlan{Rules}
    N::Int
    d::Int
    rules::Rules
    limit::Symbol
    max_order::Int
end

function _mf_check_rate(rate)
    rate isa Union{Number,Function} ||
        throw(ArgumentError("mean-field term rates must be numbers or scalar time-dependent functions"))
    rate
end

function _mf_check_operator(operator, d::Int, p::Int, name::AbstractString;
                            hermitian::Bool=false, symmetric::Bool=false,
                            copy_operator::Bool=true)
    operator isa Function &&
        throw(ArgumentError("mean-field plans require fixed operators; operator-valued time functions are unsupported"))
    operator isa AbstractMatrix ||
        throw(ArgumentError("$name must be a fixed matrix"))
    D=d^p
    size(operator)==(D,D) || throw(DimensionMismatch("$name must be $D×$D"))
    hermitian && !ishermitian(operator) &&
        throw(ArgumentError("mean-field Hamiltonians must be Hermitian"))
    if symmetric && p>1
        R=_real_float_type(eltype(operator))
        tolerance=R(1e-10)*max(R(norm(operator,Inf)),one(R))
        for k in 1:p-1
            permutation=_tensor_swap_permutation(p,d,k)
            difference=zero(R)
            @inbounds for j in 1:D, i in 1:D
                difference=max(difference,R(abs(operator[i,j]-operator[permutation[i],permutation[j]])))
            end
            difference<=tolerance ||
                throw(ArgumentError("p-body operator must be invariant under permutations of its p particles"))
        end
    end
    copy_operator ? copy(operator) : operator
end

_mf_subset_count(N,p,::Val{:finite})=binomial(big(N-1),big(p-1))
_mf_subset_count(N,p,::Val{:thermodynamic})=
    big(N)^(p-1)//factorial(big(p-1))

function _mf_collective_pbody_counts(N,p,operator,rate)
    containing=binomial(big(N-1),big(p-1))
    # Both ordered subsets contain the tagged particle; r is their total
    # overlap, including that particle.
    same_tag=ntuple(p) do r
        count=containing*binomial(big(p-1),big(r-1))*
              binomial(big(N-p),big(p-r))
        _mf_numeric_count(count,operator,rate)
    end
    # Exactly the first ordered subset contains the tagged particle; r is the
    # overlap among the remaining particles.  The reverse orientation has the
    # same count and is applied separately to preserve the cross dissipator.
    one_tag=ntuple(p) do index
        r=index-1
        count=containing*binomial(big(p-1),big(r))*
              binomial(big(N-p),big(p-r))
        _mf_numeric_count(count,operator,rate)
    end
    leading=big(N)^(2p-1)//
            (factorial(big(p-1))*factorial(big(p)))
    same_tag,one_tag,_mf_numeric_count(leading,operator,rate)
end

function _mf_numeric_count(count,operator,rate,extra=1)
    T=rate isa Number ? promote_type(eltype(operator),typeof(rate),typeof(extra)) :
                       promote_type(eltype(operator),typeof(extra))
    R=_real_float_type(T);value=convert(R,count)
    isfinite(value) || throw(ArgumentError(
        "mean-field combinatorial factor is not representable in $R; use BigFloat operator/rate data"))
    value
end

function _mf_lower(term,N,d,limit)
    term isa AbstractPITerm ||
        throw(ArgumentError("mean-field terms must subtype AbstractPITerm, got $(typeof(term))"))
    term isa Union{DirectPIHamiltonian,DirectPIJump} &&
        throw(ArgumentError("direct PI terms have no body-order provenance and are unsupported by the mean-field closure; use equivalent local/p-body terms"))
    term_operator(term) isa Function &&
        throw(ArgumentError("mean-field plans require fixed operators; operator-valued time functions are unsupported"))
    rate=_mf_check_rate(term_rate(term))
    if term isa Union{LocalHamiltonian,CollectiveHamiltonian}
        H=_mf_check_operator(term_operator(term),d,1,"one-particle Hamiltonian";hermitian=true)
        return _MFHamiltonianRule(H,rate,term.hbar,1,_mf_numeric_count(1,H,rate,term.hbar))
    elseif term isa PBodyHamiltonian
        1<=term.p<=N || throw(ArgumentError("term body order $(term.p) must satisfy 1 ≤ p ≤ N=$N"))
        H=_mf_check_operator(term_operator(term),d,term.p,"$(term.p)-particle Hamiltonian";
                             hermitian=true,symmetric=true)
        count=_mf_numeric_count(_mf_subset_count(N,term.p,Val(limit)),H,rate,term.hbar)
        return _MFHamiltonianRule(H,rate,term.hbar,term.p,count)
    elseif term isa LocalJump
        L=_mf_check_operator(term_operator(term),d,1,"one-particle jump")
        return _MFLocalJumpRule(L,copy(adjoint(L)*L),copy(adjoint(L)),rate,1,
                                _mf_numeric_count(1,L,rate))
    elseif term isa LocalPBodyJump
        1<=term.p<=N || throw(ArgumentError("term body order $(term.p) must satisfy 1 ≤ p ≤ N=$N"))
        L=_mf_check_operator(term_operator(term),d,term.p,"$(term.p)-particle jump";symmetric=true)
        count=_mf_numeric_count(_mf_subset_count(N,term.p,Val(limit)),L,rate)
        return _MFLocalJumpRule(L,copy(adjoint(L)*L),copy(adjoint(L)),rate,
                                term.p,count)
    elseif term isa CollectiveJump
        L=_mf_check_operator(term_operator(term),d,1,"one-particle collective jump")
        return _MFCollectiveJumpRule(L,copy(adjoint(L)*L),copy(adjoint(L)),rate,
            _mf_numeric_count(N-1,L,rate),_mf_numeric_count(N,L,rate))
    elseif term isa CollectivePBodyJump
        1<=term.p<=N || throw(ArgumentError("term body order $(term.p) must satisfy 1 ≤ p ≤ N=$N"))
        if term.p==1
            L=_mf_check_operator(term_operator(term),d,1,"one-particle collective jump")
            return _MFCollectiveJumpRule(L,copy(adjoint(L)*L),copy(adjoint(L)),rate,
                _mf_numeric_count(N-1,L,rate),_mf_numeric_count(N,L,rate))
        end
        L=_mf_check_operator(term_operator(term),d,term.p,
                             "$(term.p)-particle collective jump";symmetric=true)
        same_tag,one_tag,leading=_mf_collective_pbody_counts(N,term.p,L,rate)
        return _MFCollectivePBodyJumpRule(L,rate,term.p,same_tag,one_tag,leading)
    elseif term isa AbstractPITerm
        throw(ArgumentError("custom PI term $(typeof(term)) has no mean-field lowering rule"))
    end
    throw(ArgumentError("mean-field terms must subtype AbstractPITerm, got $(typeof(term))"))
end

function MeanFieldPlan(N::Integer,d::Integer,terms;limit=:finite)
    N>=1 || throw(ArgumentError("mean-field dynamics require N ≥ 1"))
    d>=1 || throw(ArgumentError("local dimension must be positive"))
    limit in (:finite,:thermodynamic) ||
        throw(ArgumentError("limit must be :finite or :thermodynamic"))
    ts=Tuple(terms)
    rules=map(term->_mf_lower(term,Int(N),Int(d),limit),ts)
    max_order=isempty(rules) ? 1 : maximum(rule->rule isa _MFCollectiveJumpRule ? 1 : rule.order,rules)
    MeanFieldPlan{typeof(rules)}(Int(N),Int(d),rules,limit,max_order)
end

MeanFieldPlan(model::PIModel;limit=:finite)=
    MeanFieldPlan(model.basis.N,model.basis.d,model.terms;limit=limit)

isautonomous(plan::MeanFieldPlan)=all(rule->rule.rate isa Number,plan.rules)

function _mf_rule_type(rule::_AbstractMeanFieldRule)
    T=rule isa _MFCollectiveJumpRule ?
        promote_type(eltype(rule.operator),typeof(rule.finite_cross_count)) :
      rule isa _MFCollectivePBodyJumpRule ?
        promote_type(eltype(rule.operator),typeof(rule.thermodynamic_cross_count)) :
        promote_type(eltype(rule.operator),typeof(rule.count))
    rule.rate isa Number && (T=promote_type(T,typeof(rule.rate)))
    rule isa _MFHamiltonianRule && (T=promote_type(T,typeof(rule.hbar)))
    T
end

function _mf_workspace_type(plan::MeanFieldPlan,sigma)
    T=eltype(sigma)
    for rule in plan.rules
        T=promote_type(T,_mf_rule_type(rule))
    end
    R=_real_float_type(T)
    Complex{R}
end

"""Mutable, per-task scratch for mean-field contractions and RK4 evolution."""
struct MeanFieldWorkspace{T,P}
    plan::P
    powers::Vector{Matrix{T}}
    effective_operators::Vector{Matrix{T}}
    tmp1::Matrix{T}
    tmp2::Matrix{T}
    tmp3::Matrix{T}
    stage::Matrix{T}
    k1::Matrix{T}
    k2::Matrix{T}
    k3::Matrix{T}
    k4::Matrix{T}
end
eltype(::MeanFieldWorkspace{T}) where T=T

function MeanFieldWorkspace(plan::MeanFieldPlan,sigma::AbstractMatrix)
    size(sigma)==(plan.d,plan.d) || throw(DimensionMismatch("one-particle state must be $(plan.d)×$(plan.d)"))
    T=_mf_workspace_type(plan,sigma)
    powers=[zeros(T,plan.d^p,plan.d^p) for p in 1:plan.max_order]
    collective_order=maximum((rule.order for rule in plan.rules
                              if rule isa _MFCollectivePBodyJumpRule);init=0)
    effective_operators=collective_order==0 ? Matrix{T}[] :
        [zeros(T,plan.d^p,plan.d^p) for p in 0:collective_order]
    D=plan.d^plan.max_order
    MeanFieldWorkspace{T,typeof(plan)}(plan,powers,effective_operators,
        zeros(T,D,D),zeros(T,D,D),zeros(T,D,D),
        zeros(T,plan.d,plan.d),zeros(T,plan.d,plan.d),zeros(T,plan.d,plan.d),
        zeros(T,plan.d,plan.d),zeros(T,plan.d,plan.d))
end

function _mf_check_workspace(work::MeanFieldWorkspace,plan::MeanFieldPlan,sigma,dest=nothing)
    work.plan===plan || throw(ArgumentError("mean-field workspace belongs to a different plan"))
    size(sigma)==(plan.d,plan.d) || throw(DimensionMismatch("one-particle state must be $(plan.d)×$(plan.d)"))
    dest===nothing || size(dest)==size(sigma) || throw(DimensionMismatch("mean-field derivative has the wrong dimensions"))
    Twork=eltype(work.stage);Tstate=eltype(sigma)
    promote_type(Tstate,Twork)===Twork || throw(ArgumentError(
        "mean-field workspace element type $Twork cannot represent state values of type $Tstate"))
    if dest!==nothing
        Tdest=eltype(dest)
        promote_type(Tdest,Twork)===Tdest || throw(ArgumentError(
            "mean-field derivative element type $Tdest cannot represent workspace values of type $Twork"))
    end
    work
end

function _mf_build_powers!(work::MeanFieldWorkspace,sigma,maximum_order::Int)
    copyto!(work.powers[1],sigma)
    d=work.plan.d
    for p in 2:maximum_order
        previous=work.powers[p-1]; current=work.powers[p]; Dprev=size(previous,1)
        @inbounds for slow_col in 1:d, fast_col in 1:Dprev,
                      slow_row in 1:d, fast_row in 1:Dprev
            row=fast_row+(slow_row-1)*Dprev
            col=fast_col+(slow_col-1)*Dprev
            current[row,col]=sigma[slow_row,slow_col]*previous[fast_row,fast_col]
        end
    end
    work
end

@inline function _mf_rate(rule,t,parameters,work)
    rate=value_at(rule.rate,t,parameters)
    rate isa Number || throw(ArgumentError("a mean-field term rate must evaluate to a scalar number"))
    T=eltype(work.stage)
    promote_type(typeof(rate),T)===T || throw(ArgumentError(
        "evaluated rate type $(typeof(rate)) cannot be represented by mean-field workspace type $T; construct the plan/state at the required precision"))
    rate
end

function _mf_partial_trace_add!(dest,A,d,p,scale)
    environment=d^(p-1)
    @inbounds for col in 1:d,row in 1:d
        value=zero(eltype(dest))
        for e in 0:environment-1
            value+=A[row+d*e,col+d*e]
        end
        dest[row,col]+=scale*value
    end
    dest
end

function _mf_apply!(dest,rule::_MFHamiltonianRule,plan,sigma,t,parameters,work)
    D=plan.d^rule.order; R=work.powers[rule.order]
    A=@view work.tmp1[1:D,1:D]; B=@view work.tmp2[1:D,1:D]
    mul!(A,rule.operator,R);mul!(B,R,rule.operator)
    rate=_mf_rate(rule,t,parameters,work)
    scale=(-1im)*(rate/rule.hbar)*rule.count
    environment=plan.d^(rule.order-1)
    @inbounds for col in 1:plan.d,row in 1:plan.d
        value=zero(eltype(dest))
        for e in 0:environment-1
            i=row+plan.d*e;j=col+plan.d*e
            value+=A[i,j]-B[i,j]
        end
        dest[row,col]+=scale*value
    end
end

function _mf_apply!(dest,rule::_MFLocalJumpRule,plan,sigma,t,parameters,work)
    D=plan.d^rule.order;R=work.powers[rule.order]
    A=@view work.tmp1[1:D,1:D];B=@view work.tmp2[1:D,1:D];C=@view work.tmp3[1:D,1:D]
    rate=_mf_rate(rule,t,parameters,work);scale=rate*rule.count
    mul!(A,rule.operator,R);mul!(B,A,rule.adjoint_operator)
    _mf_partial_trace_add!(dest,B,plan.d,rule.order,scale)
    mul!(A,rule.quadratic,R);mul!(C,R,rule.quadratic)
    _mf_partial_trace_add!(dest,A,plan.d,rule.order,-scale/2)
    _mf_partial_trace_add!(dest,C,plan.d,rule.order,-scale/2)
end

function _mf_expectation(sigma,X)
    z=zero(promote_type(eltype(sigma),eltype(X)))
    @inbounds for j in axes(X,2),i in axes(X,1)
        z+=X[i,j]*sigma[j,i]
    end
    z
end

function _mf_apply!(dest,rule::_MFCollectiveJumpRule,plan,sigma,t,parameters,work)
    d=plan.d;A=@view work.tmp1[1:d,1:d];B=@view work.tmp2[1:d,1:d]
    rate=_mf_rate(rule,t,parameters,work)
    if plan.limit===:finite
        mul!(A,rule.operator,sigma);mul!(B,A,rule.adjoint_operator)
        @inbounds for i in eachindex(dest);dest[i]+=rate*B[i];end
        mul!(A,rule.quadratic,sigma);mul!(B,sigma,rule.quadratic)
        @inbounds for i in eachindex(dest);dest[i]-=(rate/2)*(A[i]+B[i]);end
        cross_count=rule.finite_cross_count
    else
        cross_count=rule.thermodynamic_cross_count
    end
    alpha=_mf_expectation(sigma,rule.operator)
    cross_scale=rate*cross_count/2
    mul!(A,rule.operator,sigma);mul!(B,sigma,rule.operator)
    @inbounds for i in eachindex(dest);dest[i]+=cross_scale*conj(alpha)*(A[i]-B[i]);end
    mul!(A,rule.adjoint_operator,sigma);mul!(B,sigma,rule.adjoint_operator)
    @inbounds for i in eachindex(dest);dest[i]-=cross_scale*alpha*(A[i]-B[i]);end
end

function _mf_effective_operator!(dest,L,plan,work,p::Int,m::Int)
    d=plan.d;retained=d^m;environment=d^(p-m)
    size(dest)==(retained,retained) ||
        throw(DimensionMismatch("effective-operator workspace has the wrong dimensions"))
    fill!(dest,zero(eltype(dest)))
    if m==p
        copyto!(dest,L)
        return dest
    end
    environment_state=work.powers[p-m]
    # The tagged/retained particles are the fast tensor indices.  This is
    # Tr_env[L (I_retained tensor sigma_env)] in that ordering.
    @inbounds for col in 1:retained,row in 1:retained,
                      environment_row in 1:environment,
                      environment_col in 1:environment
        full_row=row+retained*(environment_row-1)
        full_col=col+retained*(environment_col-1)
        dest[row,col]+=L[full_row,full_col]*
                       environment_state[environment_col,environment_row]
    end
    dest
end

function _mf_collective_effective_operators!(rule::_MFCollectivePBodyJumpRule,
                                             plan,work)
    for m in 0:rule.order
        _mf_effective_operator!(work.effective_operators[m+1],rule.operator,
                                plan,work,rule.order,m)
    end
    work
end

function _mf_embed_environment_operator!(dest,B,d::Int,q::Int)
    total=d^q;environment=d^(q-1)
    size(dest)==(total,total) ||
        throw(DimensionMismatch("embedded effective-operator workspace has the wrong dimensions"))
    fill!(dest,zero(eltype(dest)))
    # Particle 1 is the fast tensor index, hence I_1 tensor B is kron(B,I).
    @inbounds for environment_col in 1:environment,
                      environment_row in 1:environment,local_index in 1:d
        row=local_index+d*(environment_row-1)
        col=local_index+d*(environment_col-1)
        dest[row,col]=B[environment_row,environment_col]
    end
    dest
end

function _mf_cross_dissipator_add!(dest,A,B,R,d::Int,q::Int,scale,work)
    D=d^q
    X=@view work.tmp1[1:D,1:D]
    Y=@view work.tmp2[1:D,1:D]
    mul!(X,A,R);mul!(Y,X,adjoint(B))
    _mf_partial_trace_add!(dest,Y,d,q,scale)
    mul!(X,adjoint(B),A);mul!(Y,X,R)
    _mf_partial_trace_add!(dest,Y,d,q,-scale/2)
    mul!(Y,R,X)
    _mf_partial_trace_add!(dest,Y,d,q,-scale/2)
    dest
end

function _mf_collective_overlap_add!(dest,rule,plan,work,r::Int,count,rate)
    iszero(count) && return dest
    q=r+1;D=plan.d^q
    A=work.effective_operators[q+1]
    Bsmall=work.effective_operators[r+1]
    B=@view work.tmp3[1:D,1:D]
    _mf_embed_environment_operator!(B,Bsmall,plan.d,q)
    R=work.powers[q]
    scale=rate*count
    _mf_cross_dissipator_add!(dest,A,B,R,plan.d,q,scale,work)
    _mf_cross_dissipator_add!(dest,B,A,R,plan.d,q,scale,work)
end

function _mf_apply!(dest,rule::_MFCollectivePBodyJumpRule,plan,sigma,t,
                    parameters,work)
    _mf_collective_effective_operators!(rule,plan,work)
    rate=_mf_rate(rule,t,parameters,work);p=rule.order
    if plan.limit===:finite
        # Both ordered subsets contain particle 1.  Their common contracted
        # operator is E_r and their contribution is D[E_r](sigma^tensor r).
        for r in 1:p
            count=rule.same_tag_overlap_counts[r]
            iszero(count) && continue
            A=work.effective_operators[r+1]
            _mf_cross_dissipator_add!(dest,A,A,work.powers[r],plan.d,r,
                                      rate*count,work)
        end
        # Exactly one ordered subset contains particle 1.  Add both
        # orientations of the generalized cross dissipator.
        for r in 0:p-1
            _mf_collective_overlap_add!(dest,rule,plan,work,r,
                                        rule.one_tag_overlap_counts[r+1],rate)
        end
    else
        # The disjoint r=0 class is O(N^(2p-1)); every overlap and self term is
        # subleading.  This reduces to the familiar collective one-body field
        # when p=1.
        _mf_collective_overlap_add!(dest,rule,plan,work,0,
                                    rule.thermodynamic_cross_count,rate)
    end
end

@inline _mf_apply_rules!(dest,::Tuple{},plan,sigma,t,parameters,work)=nothing
@inline function _mf_apply_rules!(dest,rules::Tuple{R,Vararg{Any}},plan,sigma,t,parameters,work) where R
    _mf_apply!(dest,first(rules),plan,sigma,t,parameters,work)
    _mf_apply_rules!(dest,Base.tail(rules),plan,sigma,t,parameters,work)
end

"""Apply the prepared product-state mean-field vector field without allocating scratch."""
function meanfield_rhs!(dest::AbstractMatrix,plan::MeanFieldPlan,sigma::AbstractMatrix,
                        t,parameters,work::MeanFieldWorkspace)
    _mf_check_workspace(work,plan,sigma,dest)
    Base.mightalias(dest,sigma) &&
        throw(ArgumentError("mean-field derivative storage must not alias the input state"))
    fill!(dest,zero(eltype(dest)))
    _mf_build_powers!(work,sigma,plan.max_order)
    _mf_apply_rules!(dest,plan.rules,plan,sigma,t,parameters,work)
    dest
end

meanfield_rhs!(dest,plan,sigma,work::MeanFieldWorkspace;time=0,parameters=nothing)=
    meanfield_rhs!(dest,plan,sigma,time,parameters,work)

"""Allocate and return the one-particle mean-field derivative."""
function meanfield_rhs(plan::MeanFieldPlan,sigma::AbstractMatrix;time=0,parameters=nothing)
    work=MeanFieldWorkspace(plan,sigma);dest=similar(work.stage)
    meanfield_rhs!(dest,plan,sigma,time,parameters,work)
end

"""Construct an in-place SciML `ODEProblem` for a prepared mean-field plan."""
function meanfield_problem(plan::MeanFieldPlan,sigma0::AbstractMatrix,tspan;parameters=nothing)
    work=MeanFieldWorkspace(plan,sigma0);u0=Matrix{eltype(work.stage)}(sigma0)
    f! = (du,u,p,t)->meanfield_rhs!(du,plan,u,t,p,work)
    SciMLBase.ODEProblem(f!,u0,tspan,parameters)
end

"""Propagate a one-particle density matrix with a preallocated fixed-step RK4 kernel."""
function meanfield_evolve!(dest::AbstractMatrix,plan::MeanFieldPlan,src::AbstractMatrix,tspan;
                           steps::Integer=256,parameters=nothing,workspace=nothing)
    steps>0 || throw(ArgumentError("steps must be positive"))
    size(dest)==size(src) || throw(DimensionMismatch("source and destination dimensions differ"))
    work=workspace===nothing ? MeanFieldWorkspace(plan,src) : workspace
    _mf_check_workspace(work,plan,src,dest)
    promote_type(eltype(dest),eltype(work.stage))===eltype(work.stage) ||
        throw(ArgumentError("mean-field evolution destination precision must match its workspace"))
    dest===src || copyto!(dest,src)
    t0,t1=tspan;t1>=t0 || throw(ArgumentError("tspan must be ordered"));h=(t1-t0)/steps;t=t0
    for _ in 1:steps
        meanfield_rhs!(work.k1,plan,dest,t,parameters,work)
        @. work.stage=dest+(h/2)*work.k1
        meanfield_rhs!(work.k2,plan,work.stage,t+h/2,parameters,work)
        @. work.stage=dest+(h/2)*work.k2
        meanfield_rhs!(work.k3,plan,work.stage,t+h/2,parameters,work)
        @. work.stage=dest+h*work.k3
        meanfield_rhs!(work.k4,plan,work.stage,t+h,parameters,work)
        @. dest=dest+(h/6)*(work.k1+2work.k2+2work.k3+work.k4)
        t+=h
    end
    dest
end

"""Saved fixed-step product-closure trajectory."""
struct MeanFieldResult{T,S}
    times::Vector{T}
    states::S
    limit::Symbol
    algorithm::Symbol
end
Base.length(result::MeanFieldResult)=length(result.states)
Base.getindex(result::MeanFieldResult,i::Integer)=result.states[i]
Base.firstindex(result::MeanFieldResult)=firstindex(result.states)
Base.lastindex(result::MeanFieldResult)=lastindex(result.states)
Base.iterate(result::MeanFieldResult,args...)=iterate(result.states,args...)
state(result::MeanFieldResult,i::Integer)=result.states[i]

function _meanfield_saved_times(tspan,saveat)
    t0,t1=tspan;t1>=t0||throw(ArgumentError("tspan must be ordered"))
    saveat===nothing&&return [float(t0),float(t1)]
    if saveat isa Real
        saveat>0||throw(ArgumentError("saveat must be positive"))
        ts=collect(float(t0):float(saveat):float(t1))
        (isempty(ts)||ts[end]<t1)&&push!(ts,float(t1))
        return ts
    end
    ts=float.(collect(saveat));isempty(ts)&&throw(ArgumentError("saveat cannot be empty"))
    first(ts)==t0&&last(ts)==t1||throw(ArgumentError("explicit saveat times must include both endpoints of tspan"))
    all(diff(ts).>=0)||throw(ArgumentError("saveat times must be nondecreasing"))
    ts
end

"""Solve mean-field dynamics and retain matrices at the requested sampling times."""
function solve_meanfield(plan::MeanFieldPlan,sigma0::AbstractMatrix,tspan;saveat=nothing,
                         steps_per_interval::Integer=64,parameters=nothing)
    steps_per_interval>0 || throw(ArgumentError("steps_per_interval must be positive"))
    times=_meanfield_saved_times(tspan,saveat);work=MeanFieldWorkspace(plan,sigma0)
    current=Matrix{eltype(work.stage)}(sigma0);states=typeof(current)[copy(current)]
    for i in 2:length(times)
        times[i]==times[i-1] || meanfield_evolve!(current,plan,current,(times[i-1],times[i]);
            steps=steps_per_interval,parameters=parameters,workspace=work)
        push!(states,copy(current))
    end
    MeanFieldResult(times,states,plan.limit,:rk4)
end

function _mf_hermitian_basis(d::Int,::Type{T}) where T
    B=Matrix{T}[];s=sqrt(real(one(T))+real(one(T)))
    for j in 2:d,i in 1:j-1
        x=zeros(T,d,d);x[i,j]=inv(s);x[j,i]=inv(s);push!(B,x)
        y=zeros(T,d,d);y[i,j]=-im/s;y[j,i]=im/s;push!(B,y)
    end
    for k in 1:d-1
        x=zeros(T,d,d);den=sqrt(real(T(k*(k+1))))
        for i in 1:k;x[i,i]=inv(den);end
        x[k+1,k+1]=-k/den;push!(B,x)
    end
    B
end

"""Real Jacobian on the traceless-Hermitian one-particle tangent space."""
function meanfield_jacobian(plan::MeanFieldPlan,sigma::AbstractMatrix;time=0,
                            parameters=nothing,epsilon=nothing)
    work=MeanFieldWorkspace(plan,sigma);T=eltype(work.stage);R=_real_float_type(T)
    basis=_mf_hermitian_basis(plan.d,T);n=length(basis);J=zeros(R,n,n)
    scale=max(norm(sigma),one(R))
    step=epsilon===nothing ? cbrt(eps(R))*scale : R(epsilon)
    step>0 || throw(ArgumentError("epsilon must be positive"))
    plus=Matrix{T}(undef,plan.d,plan.d);minus=similar(plus);fp=similar(plus);fm=similar(plus)
    for col in 1:n
        @. plus=sigma+step*basis[col];@. minus=sigma-step*basis[col]
        meanfield_rhs!(fp,plan,plus,time,parameters,work)
        meanfield_rhs!(fm,plan,minus,time,parameters,work)
        @. fp=(fp-fm)/(2step)
        for row in 1:n;J[row,col]=real(_mf_expectation(fp,basis[row]));end
    end
    J
end

"""Linear-stability data for a candidate mean-field stationary state."""
function meanfield_stability(plan::MeanFieldPlan,sigma::AbstractMatrix;atol::Real=1e-10,kwargs...)
    atol>=0 || throw(ArgumentError("atol must be nonnegative"))
    J=meanfield_jacobian(plan,sigma;kwargs...);values=eigvals(J)
    abscissa=isempty(values) ? -Inf : maximum(real,values)
    (;jacobian=J,eigenvalues=values,spectral_abscissa=abscissa,stable=abscissa < -atol)
end

"""Find the stationary state in the basin of `sigma0` by fixed-step relaxation."""
function meanfield_stationary_state(plan::MeanFieldPlan,sigma0::AbstractMatrix;dt::Real=0.01,
                                    max_steps::Integer=100_000,tol::Real=1e-10,
                                    parameters=nothing,workspace=nothing,return_info::Bool=false)
    isautonomous(plan) ||
        throw(ArgumentError("meanfield_stationary_state requires an autonomous plan"))
    dt>0 || throw(ArgumentError("dt must be positive"));max_steps>0 || throw(ArgumentError("max_steps must be positive"))
    tol>=0 || throw(ArgumentError("tol must be nonnegative"))
    work=workspace===nothing ? MeanFieldWorkspace(plan,sigma0) : workspace
    state_matrix=Matrix{eltype(work.stage)}(sigma0);residual=Inf;iterations=0
    for k in 0:Int(max_steps)
        meanfield_rhs!(work.k1,plan,state_matrix,k*dt,parameters,work)
        residual=norm(work.k1);iterations=k
        residual<=tol && break
        k==max_steps && break
        meanfield_evolve!(state_matrix,plan,state_matrix,(k*dt,(k+1)*dt);steps=1,
                          parameters=parameters,workspace=work)
    end
    info=(;state=state_matrix,residual,converged=residual<=tol,iterations,
           time=iterations*dt,limit=plan.limit)
    return_info && return info
    info.converged || error("mean-field relaxation did not converge in $max_steps steps (residual=$residual); use return_info=true to inspect the result")
    state_matrix
end

"""Product-state expectation `tr(sigma*X)` of a one-particle observable."""
function meanfield_expectation(sigma::AbstractMatrix,X::AbstractMatrix)
    size(sigma)==size(X) || throw(DimensionMismatch("state and observable dimensions differ"))
    size(sigma,1)==size(sigma,2) ||
        throw(DimensionMismatch("state and observable matrices must be square"))
    _mf_expectation(sigma,X)
end

"""Mean, second moment, and variance of `sum_i X_i` in a product state."""
function meanfield_collective_moments(N::Integer,sigma::AbstractMatrix,X::AbstractMatrix)
    N>=0 || throw(ArgumentError("N must be nonnegative"));size(sigma)==size(X) || throw(DimensionMismatch("state and observable dimensions differ"))
    size(sigma,1)==size(sigma,2) ||
        throw(DimensionMismatch("state and observable matrices must be square"))
    mu=_mf_expectation(sigma,X);local_second=_mf_expectation(sigma,X*X)
    R=_real_float_type(promote_type(typeof(mu),typeof(local_second)))
    n,count_is_exact=_exact_particle_count(R,N)
    if count_is_exact
        # Associating each extensive factor with one local expectation avoids
        # the overflowing intermediate n*(n-1) for a bounded normalized
        # observable, while retaining the former allocation-free small-N path.
        # An integer beyond the consecutive range of `R` must use the exact
        # prepared factors below rather than silently substituting its rounded
        # floating-point neighbour.
        mean=n*mu
        local_term=n*local_second
        pair_term=mean*((n-one(R))*mu)
        second_moment=local_term+pair_term
        variance=local_term-mean*mu
        if all(isfinite,(mean,second_moment,variance))&&
           (iszero(mu)||!iszero(mean))
            return (;mean,second_moment,variance)
        end
    end
    exact_n=big(N)
    scale_n=_prepare_exact_scale(R,exact_n,one(BigInt),Val(false);
        context="mean-field collective multiplicity")
    mean=_apply_prepared_exact_scale(mu,scale_n;
        context="mean-field collective mean")
    local_term=_apply_prepared_exact_scale(local_second,scale_n;
        context="mean-field collective local second moment")
    scale_pairs=_prepare_exact_scale(R,exact_n*big(N-1),one(BigInt),Val(false);
        context="mean-field collective pair multiplicity")
    pair_term=_apply_prepared_exact_scale_product(mu,mu,scale_pairs;
        context="mean-field collective pair second moment")
    second_moment=local_term+pair_term
    repeated_mean=_apply_prepared_exact_scale_product(mu,mu,scale_n;
        context="mean-field collective variance product")
    variance=local_term-repeated_mean
    (;mean,second_moment,variance)
end
meanfield_collective_moments(plan::MeanFieldPlan,sigma,X)=meanfield_collective_moments(plan.N,sigma,X)
meanfield_collective_moments(sigma::AbstractMatrix,X::AbstractMatrix,N::Integer)=
    meanfield_collective_moments(N,sigma,X)

"""Expectation of the unordered subset sum of a symmetric `p`-body observable."""
function meanfield_pbody_expectation(N::Integer,sigma::AbstractMatrix,X::AbstractMatrix,p::Integer)
    1<=p<=N || throw(ArgumentError("p must satisfy 1 ≤ p ≤ N"));d=size(sigma,1)
    size(sigma,2)==d || throw(DimensionMismatch("one-particle state must be square"))
    checked=_mf_check_operator(X,d,Int(p),"p-body observable";
                               symmetric=true,copy_operator=false)
    T=promote_type(eltype(sigma),eltype(checked));power=Matrix{T}(sigma)
    for _ in 2:p;power=kron(sigma,power);end # newly added particle is the slow tensor index
    count=exact_binomial(N,p)
    _checked_mul_exact_ratio(_mf_expectation(power,checked),count,one(BigInt);
        context="mean-field p-body subset expectation")
end
meanfield_pbody_expectation(plan::MeanFieldPlan,sigma,X,p::Integer)=
    meanfield_pbody_expectation(plan.N,sigma,X,p)
meanfield_pbody_expectation(sigma::AbstractMatrix,X::AbstractMatrix,p::Integer,N::Integer)=
    meanfield_pbody_expectation(N,sigma,X,p)
