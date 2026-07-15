"""
    AbstractPITerm

Supertype for physical contributions to a `PIModel`. A custom subtype must
implement the qualified extension hooks `term_operator`, `term_rate`,
`body_order`, `term_scope`, `term_process`, `validate_term`, and
`compile_term(term, context)`. The supported compilation pattern is to build
an equivalent built-in term and delegate to its `compile_term` method; compiled
kernel types are internal. Sector-changing terms should also implement
`required_sectors`, and driven terms should implement `rebuild_term` when they
must support `freeze`.

`term_scope` returns `Val(:local)`, `Val(:collective)`, or `Val(:direct)`, and
`term_process` returns `Val(:hamiltonian)` or `Val(:jump)`. Missing hooks raise
explicitly rather than being inferred from field names.
"""
abstract type AbstractPITerm end

"""
    value_at(x, time, parameters)

Return a constant specification unchanged, or evaluate a callable
specification as `x(time, parameters)`. This is the scalar/operator schedule
convention used by PI model terms.
"""
value_at(x,t,p)=x
value_at(x::Function,t,p)=x(t,p)
_check_square(X,d,name)=size(X)==(d,d)||throw(DimensionMismatch("$name must be $d×$d"))

"""
    InPlaceTimeOperator(prototype, update!)

Describe an operator-valued time dependence without allocating a new operator
at every Liouvillian application. `prototype` fixes the operator's shape,
basis, and scalar type. Before each evaluation the caller-owned destination is
reset to `prototype`, then `update!(destination, time, parameters)` must mutate
it in place and return either `destination` or `nothing`.

All built-in Hamiltonian and jump terms lower this schedule into mutable
buffers owned by `LiouvillianWorkspace`, including Appendix-D p-body terms;
the schedule itself and the compiled plan remain read-only and may be shared
between tasks. Use one workspace per task. Calling the schedule directly, as
done by `freeze`, intentionally returns a fresh evaluated operator.
"""
struct InPlaceTimeOperator{O,F} <: Function
    prototype::O
    update!::F
    function InPlaceTimeOperator(prototype::O,update!::F) where {O,F}
        prototype isa Union{AbstractMatrix,AbstractPIOperator} ||
            throw(ArgumentError("an in-place time-operator prototype must be a matrix or PI operator"))
        applicable(copy,prototype) ||
            throw(ArgumentError("the in-place time-operator prototype must support copy"))
        stored=copy(prototype)
        new{typeof(stored),F}(stored,update!)
    end
end

_operator_prototype(operator)=operator
_operator_prototype(operator::InPlaceTimeOperator)=operator.prototype

function _evaluate_time_operator!(destination::AbstractMatrix,
                                  schedule::InPlaceTimeOperator,t,p)
    size(destination)==size(schedule.prototype) ||
        throw(DimensionMismatch("in-place time-operator destination has the wrong dimensions"))
    copyto!(destination,schedule.prototype)
    result=schedule.update!(destination,t,p)
    result===nothing||result===destination||throw(ArgumentError(
        "an in-place time-operator callback must return its destination or nothing"))
    size(destination)==size(schedule.prototype)||throw(DimensionMismatch(
        "an in-place time-operator callback changed the operator dimensions"))
    destination
end

function _evaluate_time_operator!(destination::AbstractPIOperator,
                                  schedule::InPlaceTimeOperator,t,p)
    prototype=schedule.prototype
    destination.basis===prototype.basis||throw(ArgumentError(
        "in-place PI time-operator destination belongs to a different basis"))
    copyto!(destination.data,prototype.data)
    result=schedule.update!(destination,t,p)
    result===nothing||result===destination||throw(ArgumentError(
        "an in-place time-operator callback must return its destination or nothing"))
    destination.basis===prototype.basis||throw(ArgumentError(
        "an in-place PI time-operator callback changed the operator basis"))
    length(destination.data)==length(prototype.data)||throw(DimensionMismatch(
        "an in-place PI time-operator callback changed the coefficient length"))
    destination
end

function (schedule::InPlaceTimeOperator)(t,p)
    destination=copy(schedule.prototype)
    _evaluate_time_operator!(destination,schedule,t,p)
end

value_at(schedule::InPlaceTimeOperator,t,p)=schedule(t,p)

term_operator(t::AbstractPITerm)=throw(ArgumentError("$(typeof(t)) must implement term_operator(term)"))
term_rate(t::AbstractPITerm)=throw(ArgumentError("$(typeof(t)) must implement term_rate(term)"))
body_order(t::AbstractPITerm)=throw(ArgumentError("$(typeof(t)) must implement body_order(term)"))
term_scope(t::AbstractPITerm)=throw(ArgumentError("$(typeof(t)) must implement term_scope(term)"))
term_process(t::AbstractPITerm)=throw(ArgumentError("$(typeof(t)) must implement term_process(term)"))
validate_term(t::AbstractPITerm,b::PIBasis)=throw(ArgumentError("unsupported PI term $(typeof(t)); implement validate_term(term, basis) and compile_term(term, context)"))
compile_term(t::AbstractPITerm,context)=throw(ArgumentError("unsupported PI term $(typeof(t)); implement compile_term(term, context)"))
required_sectors(t::AbstractPITerm,b::PIBasis{D}) where D=Partition{D}[]

term_has_fixed_operator(t::AbstractPITerm)=!(term_operator(t) isa Function)
term_has_preallocated_operator(t::AbstractPITerm)=term_operator(t) isa InPlaceTimeOperator
term_isautonomous(t::AbstractPITerm)=term_has_fixed_operator(t)&&term_rate(t) isa Number

"""
    LocalHamiltonian(H; rate=1, hbar=1, check=true)

Hamiltonian term `-(im/hbar) * rate * [sum_i H^(i), rho]` generated by a
one-particle operator. A callable rate or operator follows the package's
`(time, parameters)` convention; `check=true` verifies a fixed matrix is
Hermitian.
"""
struct LocalHamiltonian{O,R,H<:Real} <: AbstractPITerm; operator::O; rate::R; hbar::H; end
LocalHamiltonian(H;rate=1,hbar=1,check=true)=begin P=_operator_prototype(H);check&&P isa AbstractMatrix&&!ishermitian(P)&&throw(ArgumentError("Hamiltonian prototype must be Hermitian"));LocalHamiltonian(H,rate,hbar) end

"""
    CollectiveHamiltonian(H; rate=1, hbar=1, check=true)

Hamiltonian term generated by the collective one-body operator
`sum_i H^(i)`. It has the same microscopic Hamiltonian action as
`LocalHamiltonian` and is marked as a collective term for model
classification. `check=true` verifies a fixed matrix is Hermitian.
"""
struct CollectiveHamiltonian{O,R,H<:Real} <: AbstractPITerm; operator::O; rate::R; hbar::H; end
CollectiveHamiltonian(H;rate=1,hbar=1,check=true)=begin P=_operator_prototype(H);check&&P isa AbstractMatrix&&!ishermitian(P)&&throw(ArgumentError("Hamiltonian prototype must be Hermitian"));CollectiveHamiltonian(H,rate,hbar) end

"""
    LocalJump(L; rate=1)

Independent one-particle dissipative channels
`rate * sum_i D[L^(i)]`, with `D[L](rho) = L*rho*L' - {L'*L,rho}/2`.
"""
struct LocalJump{O,R} <: AbstractPITerm; operator::O; rate::R; end
LocalJump(L;rate=1)=LocalJump(L,rate)

"""
    CollectiveJump(L; rate=1)

Collective dissipative channel `rate * D[sum_i L^(i)]`.
"""
struct CollectiveJump{O,R} <: AbstractPITerm; operator::O; rate::R; end
CollectiveJump(L;rate=1)=CollectiveJump(L,rate)

"""
    DirectPIHamiltonian(H; rate=1, hbar=1, check=true)

Hamiltonian commutator generated directly by a `PIOperator` on the model's
exact `PIBasis`. Use this when the Schur blocks are already known rather than
from a one- or p-body microscopic decomposition. `check=true` verifies the
fixed `PIOperator` prototype is Hermitian.
"""
struct DirectPIHamiltonian{O,R,H<:Real} <: AbstractPITerm; operator::O; rate::R; hbar::H; end
DirectPIHamiltonian(H;rate=1,hbar=1,check=true)=begin
    prototype=_operator_prototype(H)
    check&&prototype isa AbstractPIOperator&&!ishermitian(prototype)&&
        throw(ArgumentError("Hamiltonian prototype must be Hermitian"))
    DirectPIHamiltonian(H,rate,hbar)
end

"""
    DirectPIJump(L; rate=1)

Dissipative channel `rate * D[L]` for a `PIOperator` on the model's exact
`PIBasis`.
"""
struct DirectPIJump{O,R} <: AbstractPITerm; operator::O; rate::R; end
DirectPIJump(L;rate=1)=DirectPIJump(L,rate)

"""
    PBodyHamiltonian(H, p; rate=1, hbar=1, check=true)

Hamiltonian generated by the symmetric sum of `H` over all unordered
`p`-particle subsets. `H` is a permutation-symmetric `d^p x d^p` operator;
`check=true` verifies a fixed matrix is Hermitian.
"""
struct PBodyHamiltonian{O,R,H<:Real} <: AbstractPITerm; operator::O; p::Int; rate::R; hbar::H; end
PBodyHamiltonian(H,p::Integer;rate=1,hbar=1,check=true)=begin
    P=_operator_prototype(H);p>=1||throw(ArgumentError("p must be positive"));check&&P isa AbstractMatrix&&!ishermitian(P)&&throw(ArgumentError("Hamiltonian prototype must be Hermitian"));PBodyHamiltonian(H,Int(p),rate,hbar)
end
"""
    LocalPBodyJump(L, p; rate=1)

Sum of independent channels `rate * sum_S D[L_S]` over unordered
`p`-particle subsets `S`.
"""
struct LocalPBodyJump{O,R} <: AbstractPITerm; operator::O; p::Int; rate::R; end
LocalPBodyJump(L,p::Integer;rate=1)=(p>=1||throw(ArgumentError("p must be positive"));LocalPBodyJump(L,Int(p),rate))

"""
    CollectivePBodyJump(L, p; rate=1)

Collective channel `rate * D[sum_S L_S]`, where the coherent sum runs over
all unordered `p`-particle subsets.
"""
struct CollectivePBodyJump{O,R} <: AbstractPITerm; operator::O; p::Int; rate::R; end
CollectivePBodyJump(L,p::Integer;rate=1)=(p>=1||throw(ArgumentError("p must be positive"));CollectivePBodyJump(L,Int(p),rate))

const _OneBodyPITerm=Union{LocalHamiltonian,CollectiveHamiltonian,LocalJump,
                           CollectiveJump,DirectPIHamiltonian,DirectPIJump}
const _PBodyPITerm=Union{PBodyHamiltonian,LocalPBodyJump,CollectivePBodyJump}
const _BuiltinPITerm=Union{_OneBodyPITerm,_PBodyPITerm}
const _HamiltonianPITerm=Union{LocalHamiltonian,CollectiveHamiltonian,
                               DirectPIHamiltonian,PBodyHamiltonian}
const _JumpPITerm=Union{LocalJump,CollectiveJump,DirectPIJump,
                        LocalPBodyJump,CollectivePBodyJump}

term_operator(t::_BuiltinPITerm)=t.operator
term_rate(t::_BuiltinPITerm)=t.rate
body_order(::_OneBodyPITerm)=1
body_order(t::_PBodyPITerm)=t.p
term_scope(::Union{LocalHamiltonian,LocalJump,LocalPBodyJump})=Val(:local)
term_scope(::Union{CollectiveHamiltonian,CollectiveJump,PBodyHamiltonian,
                   CollectivePBodyJump})=Val(:collective)
term_scope(::Union{DirectPIHamiltonian,DirectPIJump})=Val(:direct)
term_process(::_HamiltonianPITerm)=Val(:hamiltonian)
term_process(::_JumpPITerm)=Val(:jump)
term_hbar(t::_HamiltonianPITerm)=t.hbar

function validate_term(t::_BuiltinPITerm,b::PIBasis)
    order=body_order(t)
    1<=order<=b.N||throw(ArgumentError("term body order $order must satisfy 1 ≤ p ≤ N=$(b.N)"))
    operator=_operator_prototype(term_operator(t))
    if term_scope(t) isa Val{:direct}
        operator isa AbstractPIOperator||throw(ArgumentError("direct PI terms require a PIOperator"))
    end
    operator isa AbstractMatrix&&_check_square(operator,b.d^order,
        order==1 ? "one-particle operator" : "$order-particle operator")
    operator isa AbstractPIOperator&&operator.basis!==b&&
        throw(ArgumentError("term uses incompatible basis"))
    nothing
end

required_sectors(::LocalJump,b::PIBasis)=
    unique(q for p in b.sectors for q in minus_plus_neighbors(p))

# Two endpoint sectors of a local p-body map are coupled exactly when their
# Young diagrams have a common ancestor after removing p boxes. Generate the
# descendants of just those ancestors instead of scanning `partitions(N,d)`;
# the cost is controlled by the body order and retained sectors, not by N.
function _pbody_addition_descendants(mu::Partition{D},p::Integer) where D
    current=Set{Partition{D}}((mu,))
    for _ in 1:p
        current=Set(add_corner(sector,row) for sector in current
                    for row in addable_corners(sector))
    end
    sort!(collect(current);by=sector->sector.parts,rev=true)
end

function required_sectors(t::LocalPBodyJump,b::PIBasis{D}) where D
    descendants=Dict{Partition{D},Vector{Partition{D}}}()
    required=Set{Partition{D}}()
    for sector in b.sectors,path in _removal_paths(sector,t.p)
        center=first(path)
        union!(required,get!(() -> _pbody_addition_descendants(center,t.p),
                            descendants,center))
    end
    sort!(collect(required);by=sector->sector.parts,rev=true)
end

# The Schur--Weyl identity sum_lambda dim(U_lambda)^2 = binomial(N+d^2-1,N)
# detects completeness without materializing every partition.  Pattern lists
# have already been constructed by PIBasis, so the retained side costs one
# short exact sum and remains safe even when the complete sector count is
# astronomically large.
function _basis_is_complete(b::PIBasis)
    retained=sum((big(length(patterns))^2 for patterns in b.patterns);
                 init=big(0))
    retained==commutant_dimension(b.N,b.d)
end

function rebuild_term(t::AbstractPITerm,operator,rate)
    throw(ArgumentError("time-dependent term $(typeof(t)) must implement rebuild_term(term, operator, rate) to support freeze"))
end
rebuild_term(t::LocalHamiltonian,o,r)=LocalHamiltonian(o;rate=r,hbar=t.hbar)
rebuild_term(t::CollectiveHamiltonian,o,r)=CollectiveHamiltonian(o;rate=r,hbar=t.hbar)
rebuild_term(::LocalJump,o,r)=LocalJump(o;rate=r)
rebuild_term(::CollectiveJump,o,r)=CollectiveJump(o;rate=r)
rebuild_term(t::DirectPIHamiltonian,o,r)=DirectPIHamiltonian(o;rate=r,hbar=t.hbar)
rebuild_term(::DirectPIJump,o,r)=DirectPIJump(o;rate=r)
rebuild_term(t::PBodyHamiltonian,o,r)=PBodyHamiltonian(o,t.p;rate=r,hbar=t.hbar)
rebuild_term(t::LocalPBodyJump,o,r)=LocalPBodyJump(o,t.p;rate=r)
rebuild_term(t::CollectivePBodyJump,o,r)=CollectivePBodyJump(o,t.p;rate=r)

function freeze_term(term::AbstractPITerm,t,p)
    old_operator=term_operator(term);old_rate=term_rate(term)
    operator=value_at(old_operator,t,p);rate=value_at(old_rate,t,p)
    rate isa Number||throw(ArgumentError("a frozen term rate must evaluate to a number, got $(typeof(rate))"))
    operator isa Function&&throw(ArgumentError("a frozen term operator must evaluate to a fixed operator"))
    operator===old_operator&&rate===old_rate ? term : rebuild_term(term,operator,rate)
end

"""
    PIModel(basis, terms)

Validated immutable PI model. Terms are stored as a concrete tuple and must
subtype `AbstractPITerm`. Construction checks operator dimensions, basis
ownership, body orders, and whether a restricted basis contains every sector
required by local processes.
"""
struct PIModel{B<:PIBasis,Terms<:Tuple}
    basis::B
    terms::Terms
    function PIModel(b::PIBasis,terms)
        ts=Tuple(terms)
        all(t->t isa AbstractPITerm,ts)||throw(ArgumentError("every model term must subtype AbstractPITerm"))
        foreach(t->validate_term(t,b),ts)
        restricted=!_basis_is_complete(b)
        if restricted
            missing=setdiff(unique(q for t in ts for q in required_sectors(t,b)),b.sectors)
            isempty(missing)||throw(ArgumentError("local terms require missing sectors: $(join(missing, ", "))"))
        end
        new{typeof(b),typeof(ts)}(b,ts)
    end
end
"""Return a compact named-tuple summary of a `PIModel`."""
model_summary(m::PIModel)=(N=m.basis.N,d=m.basis.d,dimension=length(m.basis),
                          terms=length(m.terms),autonomous=isautonomous(m))
"""
    basis_summary(basis; T=ComplexF64,
                  bigfloat_precision=precision(BigFloat))

Return sector, retained-coordinate dimension, and memory information for a
`PIBasis`.  `state_bytes` describes one vector on the *retained* basis, so it
also remains accurate for sector-restricted bases.  The legacy
[`estimate_memory`](@ref) function instead estimates a complete PI basis from
only `(N,d)` and therefore cannot account for sector restrictions.
Fixed-size isbits scalars use exact inline byte accounting. Heap-backed
`BigFloat` values use the conservative retained-storage bound at
`bigfloat_precision` shared by [`estimate_state_bytes`](@ref).
"""
function basis_summary(b::PIBasis;T=ComplexF64,
                       bigfloat_precision::Integer=precision(BigFloat))
    (N=b.N,d=b.d,sectors=length(b.sectors),dimension=length(b),
     state_bytes=big(length(b))*_scalar_retained_bytes(
         T;bigfloat_precision),basis_bytes=Base.summarysize(b))
end
