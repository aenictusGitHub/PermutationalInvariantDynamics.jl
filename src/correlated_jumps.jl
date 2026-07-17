@doc raw"""
    CorrelatedLocalJumps(operators, kossakowski; rate=1,
                         atol=nothing, rtol=nothing)

Correlated independent one-particle Lindblad channels

```math
\mathrm{rate}\sum_i\sum_{a,b}\Gamma_{ab}
\left(L_a^{(i)}\rho L_b^{(i)\dagger}
-\frac{1}{2}\{L_b^\dagger L_a,\rho\}^{(i)}\right),
```

where `operators` is a nonempty collection of equally sized one-particle
matrices and `kossakowski` is a Hermitian positive-semidefinite matrix
`Γ` with one row and column per operator. A fixed `Γ` is copied, validated,
and factorized once at construction. A raw `(time, parameters) -> Γ` function
uses the allocating compatibility path; wrap a matrix prototype and mutating
callback in [`InPlaceTimeOperator`](@ref) for task-local, preallocated
evaluation.

`atol` and `rtol` control explicit Hermiticity and positivity acceptance.
Their defaults are exact zero; only working-precision roundoff generated
inside the factorization is tolerated. Positive eigenchannels are not dropped
by a numerical rank cutoff. The common scalar `rate` must be real, finite, and
representable in the prepared precision when evaluated, but may be negative
for deliberately non-CP time-local generators.
"""
struct CorrelatedLocalJumps{O,K,F,R,A,B} <: AbstractPITerm
    operators::O
    kossakowski::K
    factor::F
    rate::R
    atol::A
    rtol::B
end

@doc raw"""
    CorrelatedCollectiveJumps(operators, kossakowski; rate=1,
                              atol=nothing, rtol=nothing)

Correlated collective one-body Lindblad channels

```math
\mathrm{rate}\sum_{a,b}\Gamma_{ab}
\left(J_a\rho J_b^\dagger
-\frac{1}{2}\{J_b^\dagger J_a,\rho\}\right),\qquad
J_a=\sum_i L_a^{(i)}.
```

The operator, Kossakowski-matrix, tolerance, and time-dependence conventions
are the same as for [`CorrelatedLocalJumps`](@ref). Collective channels stay
within each retained Schur sector, whereas correlated local channels may
couple sectors.
"""
struct CorrelatedCollectiveJumps{O,K,F,R,A,B} <: AbstractPITerm
    operators::O
    kossakowski::K
    factor::F
    rate::R
    atol::A
    rtol::B
end

const _CorrelatedOneBodyJumps=Union{CorrelatedLocalJumps,
                                    CorrelatedCollectiveJumps}

struct _CheckedCorrelatedRate{S,R<:AbstractFloat} <: Function
    schedule::S
    real_type::Type{R}
end
function (rate::_CheckedCorrelatedRate{S,R})(time,parameters) where {S,R}
    value=value_at(rate.schedule,time,parameters)
    value isa Real||throw(ArgumentError(
        "a correlated dissipative rate must evaluate to a real number"))
    isfinite(value)||throw(ArgumentError(
        "a correlated dissipative rate must evaluate to a finite number"))
    promote_type(R,typeof(value))===R||throw(ArgumentError(
        "evaluated correlated rate type $(typeof(value)) is wider than prepared precision $R"))
    value
end
_prepared_correlated_rate(rate::Number,::Type{R}) where R<:AbstractFloat=rate
_prepared_correlated_rate(rate,::Type{R}) where R<:AbstractFloat=
    _CheckedCorrelatedRate(rate,R)

_correlated_tolerance_type(::Nothing)=Union{}
_correlated_tolerance_type(x::Integer)=Union{}
_correlated_tolerance_type(x)=_real_float_type(typeof(x))
_promote_correlated_tolerance_type(::Type{R},::Nothing) where R=R
_promote_correlated_tolerance_type(::Type{R},::Integer) where R=R
_promote_correlated_tolerance_type(::Type{R},value) where R=
    promote_type(R,_real_float_type(typeof(value)))

function _checked_correlated_tolerance(value,name)
    value===nothing&&return nothing
    value isa Real||throw(ArgumentError("$name must be a nonnegative real number"))
    isfinite(value)&&value>=0||throw(ArgumentError(
        "$name must be finite and nonnegative"))
    value
end

function _correlated_factor_type(matrix,atol,rtol)
    base=_real_float_type(eltype(matrix))
    with_absolute=_promote_correlated_tolerance_type(base,atol)
    Complex{_promote_correlated_tolerance_type(with_absolute,rtol)}
end

function _correlated_tolerances(::Type{R},atol,rtol) where R<:AbstractFloat
    absolute=atol===nothing ? zero(R) : R(atol)
    relative=rtol===nothing ? zero(R) : R(rtol)
    isfinite(absolute)&&absolute>=0||throw(ArgumentError(
        "Kossakowski atol is not a finite nonnegative $R value"))
    isfinite(relative)&&relative>=0||throw(ArgumentError(
        "Kossakowski rtol is not a finite nonnegative $R value"))
    absolute,relative
end


# Pivoted residual Cholesky is used instead of an eigendecomposition so the
# validation/factorization path remains available for BigFloat on Julia 1.10.
# The caller supplies all scratch on dynamic paths. The returned factor obeys
# A = F*F' to working precision, with every strictly positive residual pivot
# retained. Only a nonpositive residual caused by arithmetic roundoff may be
# treated as zero without an explicit user tolerance.
function _factor_kossakowski!(factor::AbstractMatrix{T},
                              residual::AbstractMatrix{T},matrix,
                              atol=nothing,rtol=nothing) where T<:Complex
    m=size(matrix,1)
    size(matrix)==(m,m)||throw(DimensionMismatch(
        "Kossakowski matrix must be square"))
    size(factor)==(m,m)&&size(residual)==(m,m)||throw(DimensionMismatch(
        "Kossakowski factorization scratch has the wrong dimensions"))
    R=typeof(real(zero(T)));absolute,relative=
        _correlated_tolerances(R,atol,rtol)
    scale=zero(R)
    @inbounds for column in 1:m,row in 1:m
        value=matrix[row,column]
        value isa Number||throw(ArgumentError(
            "Kossakowski entries must be numbers"))
        isfinite(value)||throw(ArgumentError(
            "Kossakowski entries must be finite"))
        converted=try
            T(value)
        catch
            throw(ArgumentError(
                "a Kossakowski entry is not representable in factorization type $T"))
        end
        isfinite(converted)||throw(ArgumentError(
            "a Kossakowski entry is outside the finite range of factorization type $T; use wider input precision"))
        residual[row,column]=converted
        scale=max(scale,abs(converted))
    end
    tolerance=absolute+relative*scale
    isfinite(tolerance)||throw(ArgumentError(
        "Kossakowski validation tolerance overflowed"))
    @inbounds for column in 1:m,row in 1:column
        if iszero(tolerance)
            matrix[row,column]==conj(matrix[column,row])||throw(ArgumentError(
                "Kossakowski matrix must be exactly Hermitian when atol=rtol=0"))
        end
        difference=abs(residual[row,column]-conj(residual[column,row]))
        difference<=tolerance||throw(ArgumentError(
            "Kossakowski matrix must be Hermitian (maximum allowed discrepancy is $tolerance)"))
    end
    fill!(factor,zero(T))
    # Symmetrization is performed only after the original input has passed the
    # requested Hermiticity check. It prevents accepted roundoff-sized skew
    # components from contaminating the PSD factorization.
    @inbounds for column in 1:m,row in 1:column
        average=(residual[row,column]+conj(residual[column,row]))/2
        residual[row,column]=average
        residual[column,row]=conj(average)
    end
    roundoff=R(64)*R(max(m,1))*eps(R)*scale
    psd_tolerance=max(tolerance,roundoff)
    @inbounds for index in 1:m
        diagonal=real(residual[index,index])
        diagonal>=-tolerance||throw(ArgumentError(
            "Kossakowski matrix is not positive semidefinite"))
        residual[index,index]=T(diagonal)
    end

    rank=0
    for column in 1:m
        pivot_index=1
        pivot=real(residual[1,1])
        @inbounds for index in 2:m
            candidate=real(residual[index,index])
            if candidate>pivot
                pivot=candidate;pivot_index=index
            end
        end
        if !(pivot>zero(R))
            maximum_residual=zero(R)
            @inbounds for value in residual
                maximum_residual=max(maximum_residual,abs(value))
            end
            maximum_residual<=psd_tolerance||throw(ArgumentError(
                "Kossakowski matrix is not positive semidefinite"))
            break
        end
        rank+=1
        root=sqrt(pivot)
        @inbounds for row in 1:m
            factor[row,column]=residual[row,pivot_index]/root
        end
        @inbounds for right in 1:m,left in 1:m
            residual[left,right]-=
                factor[left,column]*conj(factor[right,column])
        end
        # The pivot row and column are analytically zero. Setting them exactly
        # avoids selecting the same pivot again due solely to cancellation.
        @inbounds for index in 1:m
            residual[pivot_index,index]=zero(T)
            residual[index,pivot_index]=zero(T)
            residual[index,index]=T(real(residual[index,index]))
        end
        minimum_diagonal=minimum(index->real(residual[index,index]),1:m)
        minimum_diagonal>=-psd_tolerance||throw(ArgumentError(
            "Kossakowski matrix is not positive semidefinite"))
    end
    rank
end

function _factor_kossakowski(matrix,m::Integer,atol=nothing,rtol=nothing)
    size(matrix)==(m,m)||throw(DimensionMismatch(
        "Kossakowski matrix must have size $m×$m"))
    eltype(matrix)<:Number||throw(ArgumentError(
        "Kossakowski entries must be numbers"))
    T=_correlated_factor_type(matrix,atol,rtol)
    factor=zeros(T,m,m);residual=similar(factor)
    rank=_factor_kossakowski!(factor,residual,matrix,atol,rtol)
    ntuple(column->copy(@view factor[:,column]),rank)
end

function _stored_correlated_operators(operators)
    operators isa AbstractMatrix&&throw(ArgumentError(
        "operators must be a collection of one-particle matrices, not one matrix"))
    values=try
        Tuple(operators)
    catch
        throw(ArgumentError("operators must be a finite collection of matrices"))
    end
    isempty(values)&&throw(ArgumentError(
        "at least one correlated jump operator is required"))
    all(operator->operator isa AbstractMatrix,values)||throw(ArgumentError(
        "every correlated jump operator must be a matrix"))
    all(operator->eltype(operator)<:Number,values)||throw(ArgumentError(
        "correlated jump matrices must have a numeric element type"))
    first_size=size(first(values))
    first_size[1]==first_size[2]||throw(DimensionMismatch(
        "correlated jump operators must be square"))
    all(operator->size(operator)==first_size,values)||throw(DimensionMismatch(
        "all correlated jump operators must have the same dimensions"))
    for operator in values,value in operator
        value isa Number||throw(ArgumentError(
            "correlated jump operator entries must be numbers"))
        isfinite(value)||throw(ArgumentError(
            "correlated jump operator entries must be finite"))
    end
    map(copy,values)
end

function _construct_correlated_jumps(::Type{C},operators,kossakowski;
        rate=1,atol=nothing,rtol=nothing) where C<:_CorrelatedOneBodyJumps
    stored_operators=_stored_correlated_operators(operators)
    m=length(stored_operators)
    _checked_correlated_tolerance(atol,"atol")
    _checked_correlated_tolerance(rtol,"rtol")
    if rate isa Number
        rate isa Real||throw(ArgumentError(
            "a correlated dissipative rate must be real"))
        isfinite(rate)||throw(ArgumentError(
            "a correlated dissipative rate must be finite"))
    end
    if kossakowski isa AbstractMatrix
        stored=copy(kossakowski)
        factor=_factor_kossakowski(stored,m,atol,rtol)
    elseif kossakowski isa InPlaceTimeOperator
        kossakowski.prototype isa AbstractMatrix||throw(ArgumentError(
            "an in-place Kossakowski schedule requires a matrix prototype"))
        size(kossakowski.prototype)==(m,m)||throw(DimensionMismatch(
            "Kossakowski matrix prototype must have size $m×$m"))
        # Validate the reset prototype now; every evaluated matrix is checked
        # again in its task-local Liouvillian workspace.
        _factor_kossakowski(kossakowski.prototype,m,atol,rtol)
        stored=kossakowski;factor=nothing
    elseif kossakowski isa Function
        stored=kossakowski;factor=nothing
    else
        throw(ArgumentError(
            "kossakowski must be a matrix or a (time, parameters) schedule"))
    end
    C(stored_operators,stored,factor,rate,atol,rtol)
end

CorrelatedLocalJumps(operators,kossakowski;kwargs...)=
    _construct_correlated_jumps(CorrelatedLocalJumps,operators,kossakowski;kwargs...)
CorrelatedCollectiveJumps(operators,kossakowski;kwargs...)=
    _construct_correlated_jumps(CorrelatedCollectiveJumps,operators,kossakowski;kwargs...)

term_operator(term::_CorrelatedOneBodyJumps)=term.operators
term_rate(term::_CorrelatedOneBodyJumps)=term.rate
body_order(::_CorrelatedOneBodyJumps)=1
term_scope(::CorrelatedLocalJumps)=Val(:local)
term_scope(::CorrelatedCollectiveJumps)=Val(:collective)
term_process(::_CorrelatedOneBodyJumps)=Val(:jump)

term_has_fixed_operator(term::_CorrelatedOneBodyJumps)=
    term.kossakowski isa AbstractMatrix
term_has_preallocated_operator(term::_CorrelatedOneBodyJumps)=
    term.kossakowski isa InPlaceTimeOperator
term_isautonomous(term::_CorrelatedOneBodyJumps)=
    term_has_fixed_operator(term)&&term.rate isa Number

function validate_term(term::_CorrelatedOneBodyJumps,basis::PIBasis)
    for operator in term.operators
        _check_square(operator,basis.d,"correlated one-particle operator")
    end
    prototype=_operator_prototype(term.kossakowski)
    prototype isa AbstractMatrix&&size(prototype)!=(length(term.operators),
                                                   length(term.operators))&&
        throw(DimensionMismatch("Kossakowski matrix has the wrong dimensions"))
    nothing
end

required_sectors(::CorrelatedLocalJumps,b::PIBasis)=
    unique(q for p in b.sectors for q in minus_plus_neighbors(p))

function freeze_term(term::_CorrelatedOneBodyJumps,time,parameters)
    matrix=value_at(term.kossakowski,time,parameters)
    matrix isa AbstractMatrix||throw(ArgumentError(
        "a Kossakowski schedule must evaluate to a matrix"))
    rate=value_at(term.rate,time,parameters)
    rate isa Real||throw(ArgumentError(
        "a correlated dissipative rate must evaluate to a real number"))
    isfinite(rate)||throw(ArgumentError(
        "a correlated dissipative rate must evaluate to a finite number"))
    if matrix===term.kossakowski
        rate===term.rate&&return term
        return term isa CorrelatedLocalJumps ?
            CorrelatedLocalJumps(term.operators,term.kossakowski,term.factor,
                                 rate,term.atol,term.rtol) :
            CorrelatedCollectiveJumps(term.operators,term.kossakowski,
                                      term.factor,rate,term.atol,term.rtol)
    end
    constructor=term isa CorrelatedLocalJumps ? CorrelatedLocalJumps :
                                               CorrelatedCollectiveJumps
    constructor(term.operators,matrix;rate=rate,atol=term.atol,rtol=term.rtol)
end
