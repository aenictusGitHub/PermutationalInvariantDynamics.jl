"""
    StrongSymmetryReport

Term-by-term report for diagonal *strong* unitary symmetries of a
[`PIModel`](@ref).  Each entry of `candidates` contains `name`, `unitary`,
`status`, `terms`, and the Hilbert-space `charges` present in the retained
Schur basis.  `status` is `true`, `false`, or `missing`; `missing` is
inconclusive and is never treated as a symmetry certificate.

Unlike [`usual_liouvillian_symmetries`](@ref), this report tests commutation
with every Hamiltonian and jump operator separately.  A jump which merely
changes by a phase is therefore a weak covariance symmetry, but not a strong
symmetry.
"""
struct StrongSymmetryReport{M,B,C,D}
    model::M
    basis::B
    candidates::C
    discovery::D
end

function Base.show(io::IO,report::StrongSymmetryReport)
    certified=count(candidate->candidate.status===true,report.candidates)
    inconclusive=count(candidate->ismissing(candidate.status),
                       report.candidates)
    print(io,"StrongSymmetryReport($(length(report.candidates)) candidates, ",
          "$certified certified, $inconclusive inconclusive)")
end

"""
    StrongSymmetryReduction

Prepared collection of every trace-bearing Hilbert-charge block of one
certified diagonal strong symmetry.  `sectors` contains one named tuple per
charge with its [`SymmetryCoordinateRestriction`](@ref) and exhaustively
certified [`RestrictedLiouvillian`](@ref).  No stationary sector is selected
implicitly.
"""
struct StrongSymmetryReduction{M,S,C,Q,R}
    model::M
    source::S
    candidate::C
    sectors::Q
    resources::R
end

function Base.show(io::IO,reduction::StrongSymmetryReduction)
    print(io,"StrongSymmetryReduction(candidate=$(reduction.candidate.name), ",
          "$(length(reduction.sectors)) trace-bearing sectors)")
end

function _strong_checked_diagonal(basis::PIBasis,U::AbstractMatrix;
                                  atol::Real,rtol::Real)
    size(U)==(basis.d,basis.d)||throw(DimensionMismatch(
        "a local strong-symmetry candidate must be $(basis.d) by $(basis.d)"))
    T=_complex_float_type(eltype(U))
    R=_real_float_type(T)
    absolute=_restriction_tolerance(R,atol,"atol")
    relative=_restriction_tolerance(R,rtol,"rtol")
    matrix=Matrix{T}(U)
    all(value->isfinite(real(value))&&isfinite(imag(value)),matrix)||
        throw(ArgumentError(
            "a local strong-symmetry candidate must contain only finite values"))
    diagonal=diag(matrix)
    scale=max(norm(matrix,Inf),one(R))
    norm(matrix-Diagonal(diagonal),Inf)<=absolute+relative*scale||
        throw(ArgumentError(
            "automatic strong-symmetry reduction currently requires a " *
            "diagonal local unitary"))
    tolerance=absolute+relative
    all(value->abs(abs(value)-one(R))<=tolerance,diagonal)||
        throw(ArgumentError(
            "every diagonal strong-symmetry eigenvalue must have unit modulus"))
    matrix,diagonal,absolute,relative
end

function _strong_phase_for_index(index::Integer,diagonal,p::Integer,d::Integer)
    value=one(eltype(diagonal))
    remaining=index-1
    @inbounds for _ in 1:p
        value*=diagonal[mod(remaining,d)+1]
        remaining=div(remaining,d)
    end
    value
end

function _strong_matrix_commutation(operator::AbstractMatrix,diagonal,
                                    p::Integer,d::Integer;
                                    atol::Real,rtol::Real)
    expected=big(d)^p
    expected<=typemax(Int)||throw(ArgumentError(
        "the local p-body dimension exceeds Int"))
    dimension=Int(expected)
    size(operator)==(dimension,dimension)||throw(DimensionMismatch(
        "a $p-body operator must be $dimension by $dimension"))
    T=promote_type(_complex_float_type(eltype(operator)),eltype(diagonal))
    R=_real_float_type(T)
    absolute=_restriction_tolerance(R,atol,"atol")
    relative=_restriction_tolerance(R,rtol,"rtol")
    phases=Vector{T}(undef,dimension)
    diagonalT=T.(diagonal)
    @inbounds for index in 1:dimension
        phases[index]=_strong_phase_for_index(index,diagonalT,p,d)
    end
    residual=zero(R)
    scale=zero(R)
    if operator isa SparseMatrixCSC
        rows,columns,values=findnz(operator)
        @inbounds for index in eachindex(values)
            value=T(values[index])
            isfinite(value)||throw(ArgumentError(
                "strong-symmetry certification encountered a nonfinite operator entry"))
            scale=hypot(scale,abs(value))
            residual=hypot(residual,
                abs((phases[rows[index]]-phases[columns[index]])*value))
        end
    else
        @inbounds for column in axes(operator,2),row in axes(operator,1)
            value=T(operator[row,column])
            isfinite(value)||throw(ArgumentError(
                "strong-symmetry certification encountered a nonfinite operator entry"))
            scale=hypot(scale,abs(value))
            residual=hypot(residual,
                abs((phases[row]-phases[column])*value))
        end
    end
    comparison_scale=max(scale,one(R))
    roundoff=R(max(dimension,100))*eps(R)*comparison_scale
    tolerance=absolute+relative*comparison_scale+roundoff
    (status=residual<=tolerance,reason=:termwise_commutation,
     residual,relative_residual=residual/comparison_scale,tolerance,
     validation=:local_support_scan)
end

function _strong_pi_commutation(operator::AbstractPIOperator,basis::PIBasis,
                                diagonal;atol::Real,rtol::Real)
    operator.basis===basis||throw(ArgumentError(
        "a direct PI term uses an incompatible basis"))
    T=promote_type(_complex_float_type(eltype(operator)),eltype(diagonal))
    R=_real_float_type(T)
    absolute=_restriction_tolerance(R,atol,"atol")
    relative=_restriction_tolerance(R,rtol,"rtol")
    diagonalT=T.(diagonal)
    residual=zero(R)
    scale=zero(R)
    maximum_dimension=0
    for (sector,patterns) in pairs(basis.patterns)
        dimension=length(patterns)
        maximum_dimension=max(maximum_dimension,dimension)
        phases=Vector{T}(undef,dimension)
        @inbounds for (index,pattern) in pairs(patterns)
            occupations=content(pattern)
            phase=one(T)
            for level in eachindex(diagonalT)
                phase*=diagonalT[level]^occupations[level]
            end
            phases[index]=phase
        end
        block=coefficient_block(operator,basis.sectors[sector])
        @inbounds for column in 1:dimension,row in 1:dimension
            value=T(block[row,column])
            isfinite(value)||throw(ArgumentError(
                "strong-symmetry certification encountered a nonfinite PI entry"))
            scale=hypot(scale,abs(value))
            residual=hypot(residual,
                abs((phases[row]-phases[column])*value))
        end
    end
    comparison_scale=max(scale,one(R))
    roundoff=R(max(maximum_dimension,100))*eps(R)*comparison_scale
    tolerance=absolute+relative*comparison_scale+roundoff
    (status=residual<=tolerance,reason=:termwise_commutation,
     residual,relative_residual=residual/comparison_scale,tolerance,
     validation=:schur_support_scan)
end

_strong_zero_rate(term)=term_rate(term) isa Number&&iszero(term_rate(term))

function _strong_missing_term(reason::Symbol)
    (status=missing,reason,residual=nothing,relative_residual=nothing,
     tolerance=nothing,validation=:inconclusive)
end

function _strong_correlated_effective_operators(term::_CorrelatedOneBodyJumps)
    factor=term.factor
    factor===nothing&&return nothing
    isempty(factor)&&return ()
    operator_type=foldl(promote_type,
        (_complex_float_type(eltype(operator)) for operator in term.operators))
    factor_type=_complex_float_type(eltype(first(factor)))
    T=promote_type(operator_type,factor_type)
    map(coefficients->_effective_correlated_operator(
        coefficients,term.operators,T),factor)
end

function _strong_term_commutation(term::_CorrelatedOneBodyJumps,basis,
                                  diagonal;atol,rtol)
    _strong_zero_rate(term)&&return
        (status=true,reason=:zero_rate,residual=zero(_real_float_type(
            eltype(diagonal))),relative_residual=zero(_real_float_type(
            eltype(diagonal))),tolerance=zero(_real_float_type(
            eltype(diagonal))),validation=:exact_zero_rate)
    effective=_strong_correlated_effective_operators(term)
    operators=effective===nothing ? term.operators : effective
    isempty(operators)&&return
        (status=true,reason=:zero_kossakowski_rank,
         residual=zero(_real_float_type(eltype(diagonal))),
         relative_residual=zero(_real_float_type(eltype(diagonal))),
         tolerance=zero(_real_float_type(eltype(diagonal))),
         validation=:exact_zero_channel)
    reports=map(operator->_strong_matrix_commutation(
        operator,diagonal,1,basis.d;atol,rtol),operators)
    failed=any(report->report.status===false,reports)
    # For a fixed Kossakowski matrix the factorized effective channels are the
    # actual Lindblad operators, so a failure disproves strong commutation.
    # For a schedule, commuting with every listed seed is sufficient, whereas
    # a failing seed is only inconclusive until the schedule is frozen.
    status=failed ? (effective===nothing ? missing : false) : true
    R=_real_float_type(eltype(diagonal))
    residual=maximum((R(report.residual) for report in reports);init=zero(R))
    relative=maximum((R(report.relative_residual) for report in reports);
                     init=zero(R))
    tolerance=maximum((R(report.tolerance) for report in reports);init=zero(R))
    reason=status===true ? :all_channel_operators_commute :
           status===false ? :effective_channel_does_not_commute :
                            :kossakowski_schedule_requires_freeze
    (status,reason,
     residual,relative_residual=relative,tolerance,
     validation=effective===nothing ? :sufficient_seed_support_scan :
                                     :effective_channel_support_scan)
end

function _strong_term_commutation(term::AbstractPITerm,basis,diagonal;
                                  atol,rtol)
    _strong_zero_rate(term)&&return
        (status=true,reason=:zero_rate,residual=zero(_real_float_type(
            eltype(diagonal))),relative_residual=zero(_real_float_type(
            eltype(diagonal))),tolerance=zero(_real_float_type(
            eltype(diagonal))),validation=:exact_zero_rate)
    operator=try
        term_operator(term)
    catch
        return _strong_missing_term(:missing_term_operator)
    end
    operator isa Function&&return _strong_missing_term(
        :operator_schedule_requires_freeze)
    if operator isa AbstractMatrix
        order=try
            body_order(term)
        catch
            return _strong_missing_term(:missing_body_order)
        end
        return _strong_matrix_commutation(
            operator,diagonal,order,basis.d;atol,rtol)
    elseif operator isa AbstractPIOperator
        return _strong_pi_commutation(operator,basis,diagonal;atol,rtol)
    end
    _strong_missing_term(:unsupported_operator_representation)
end

function _strong_pattern_charges(basis::PIBasis,diagonal;
                                 atol::Real,rtol::Real)
    T=eltype(diagonal)
    R=_real_float_type(T)
    absolute=_restriction_tolerance(R,atol,"atol")
    relative=_restriction_tolerance(R,rtol,"rtol")
    charges=T[]
    for patterns in basis.patterns,pattern in patterns
        occupations=content(pattern)
        charge=one(T)
        @inbounds for level in eachindex(diagonal)
            charge*=diagonal[level]^occupations[level]
        end
        match=any(existing->abs(charge-existing)<=
            absolute+relative*max(abs(existing),one(R)),charges)
        match||push!(charges,charge)
    end
    sort!(charges;by=value->(angle(value),real(value),imag(value)))
end

function _strong_equal_charge_dimensions(basis::PIBasis,diagonal,charges;
                                         atol::Real,rtol::Real)
    T=promote_type(eltype(diagonal),eltype(charges))
    R=_real_float_type(T)
    absolute=_restriction_tolerance(R,atol,"atol")
    relative=_restriction_tolerance(R,rtol,"rtol")
    diagonalT=T.(diagonal)
    map(charges) do raw_target
        target=T(raw_target)
        dimension=big(0)
        for patterns in basis.patterns
            count_in_sector=0
            for pattern in patterns
                occupations=content(pattern)
                phase=one(T)
                @inbounds for level in eachindex(diagonalT)
                    phase*=diagonalT[level]^occupations[level]
                end
                abs(phase-target)<=absolute+
                    relative*max(abs(target),one(R))&&
                    (count_in_sector+=1)
            end
            dimension+=BigInt(count_in_sector)^2
        end
        dimension<=typemax(Int)||throw(ArgumentError(
            "a strong-symmetry charge dimension exceeds Int"))
        Int(dimension)
    end
end

function _strong_xor!(left::BitVector,right::BitVector)
    @inbounds for index in eachindex(left)
        left[index]=xor(left[index],right[index])
    end
    left
end

# Incremental reduced row-echelon basis over GF(2).  At most `d` rows are
# retained, irrespective of the number of nonzero p-body matrix entries.
function _strong_gf2_insert!(pivots::Vector{Union{Nothing,BitVector}},
                             row::BitVector)
    # First reduce against every existing pivot.  Choosing a new pivot before
    # visiting later columns would leave other pivot variables in this row and
    # invalidate the direct nullspace back-substitution below.
    for column in eachindex(row)
        pivot=pivots[column]
        pivot===nothing&&continue
        row[column]&&_strong_xor!(row,pivot)
    end
    new_column=findfirst(row)
    new_column===nothing&&return false
    for previous in eachindex(pivots)
        existing=pivots[previous]
        existing===nothing&&continue
        existing[new_column]&&_strong_xor!(existing,row)
    end
    pivots[new_column]=copy(row)
    true
end

function _strong_support_equation!(equation::BitVector,row_index::Integer,
                                   column_index::Integer,d::Integer,
                                   p::Integer)
    fill!(equation,false)
    row=row_index-1
    column=column_index-1
    @inbounds for _ in 1:p
        row_level=mod(row,d)+1
        column_level=mod(column,d)+1
        equation[row_level]=!equation[row_level]
        equation[column_level]=!equation[column_level]
        row=div(row,d)
        column=div(column,d)
    end
    equation
end

function _strong_add_matrix_constraints!(pivots,operator::AbstractMatrix,
                                         d::Integer,p::Integer)
    equation=falses(d)
    if operator isa SparseMatrixCSC
        rows,columns,values=findnz(operator)
        @inbounds for index in eachindex(values)
            iszero(values[index])&&continue
            _strong_support_equation!(
                equation,rows[index],columns[index],d,p)
            _strong_gf2_insert!(pivots,equation)
        end
    else
        @inbounds for column in axes(operator,2),row in axes(operator,1)
            iszero(operator[row,column])&&continue
            _strong_support_equation!(equation,row,column,d,p)
            _strong_gf2_insert!(pivots,equation)
        end
    end
    pivots
end

function _strong_binary_constraints!(pivots,term::_CorrelatedOneBodyJumps,
                                     basis)
    _strong_zero_rate(term)&&return true
    effective=_strong_correlated_effective_operators(term)
    operators=effective===nothing ? term.operators : effective
    for operator in operators
        _strong_add_matrix_constraints!(pivots,operator,basis.d,1)
    end
    effective!==nothing
end

function _strong_binary_constraints!(pivots,term::AbstractPITerm,basis)
    _strong_zero_rate(term)&&return true
    operator=try
        term_operator(term)
    catch
        return false
    end
    operator isa AbstractMatrix||return false
    operator isa Function&&return false
    order=try
        body_order(term)
    catch
        return false
    end
    _strong_add_matrix_constraints!(pivots,operator,basis.d,order)
    true
end

function _strong_gf2_nullspace(pivots)
    pivot_columns=findall(index->pivots[index]!==nothing,eachindex(pivots))
    pivot_mask=falses(length(pivots))
    pivot_mask[pivot_columns].=true
    vectors=BitVector[]
    for free in eachindex(pivots)
        pivot_mask[free]&&continue
        vector=falses(length(pivots))
        vector[free]=true
        for pivot_column in pivot_columns
            vector[pivot_column]=pivots[pivot_column][free]
        end
        push!(vectors,vector)
    end
    vectors
end

function _strong_discovered_binary_candidates(model::PIModel)
    d=model.basis.d
    pivots=Vector{Union{Nothing,BitVector}}(undef,d)
    fill!(pivots,nothing)
    complete=true
    for term in model.terms
        complete&=_strong_binary_constraints!(pivots,term,model.basis)
    end
    nullspace=_strong_gf2_nullspace(pivots)
    # Local U and -U induce the same conjugation and the same Hilbert-charge
    # partition up to relabeling.  Canonicalize every vector to first bit zero,
    # which quotients the unavoidable all-ones null vector.
    canonical=BitVector[]
    quotient_pivots=Vector{Union{Nothing,BitVector}}(undef,d)
    fill!(quotient_pivots,nothing)
    for vector in nullspace
        vector[1]&&_strong_xor!(vector,trues(d))
        any(vector)||continue
        _strong_gf2_insert!(quotient_pivots,copy(vector))||continue
        push!(canonical,copy(vector))
    end
    candidates=Pair{Symbol,Any}[]
    for (index,bits) in pairs(canonical)
        diagonal=ComplexF64[bits[level] ? -1 : 1 for level in 1:d]
        push!(candidates,Symbol("binary_parity_$index")=>Diagonal(diagonal))
    end
    candidates,complete
end

function _strong_usual_diagonal_candidates(model::PIModel)
    d=model.basis.d
    clock=if d==2
        # Keep qubit parity exact.  exp(pi*im) carries a tiny imaginary part
        # whose Nth power eventually spoils charge clustering at large N.
        Diagonal(ComplexF64[1,-1])
    else
        omega=exp(2pi*im/d)
        Diagonal(ComplexF64[omega^(level-1) for level in 1:d])
    end
    standard=Pair{Symbol,Any}[
        (d==2 ? :parity_z : :clock_phase)=>clock]
    discovered,complete=_strong_discovered_binary_candidates(model)
    append!(standard,discovered)
    standard,complete
end

function _strong_same_unitary(left,right;atol,rtol)
    size(left)==size(right)||return false
    T=promote_type(eltype(left),eltype(right))
    R=_real_float_type(T)
    left_diagonal=T.(diag(left))
    right_diagonal=T.(diag(right))
    left_diagonal./=left_diagonal[1]
    right_diagonal./=right_diagonal[1]
    norm(left_diagonal-right_diagonal,Inf)<=
        R(atol)+R(rtol)*max(norm(left_diagonal,Inf),one(R))
end

function _strong_candidate_pairs(model::PIModel,candidates)
    if candidates===:usual
        return _strong_usual_diagonal_candidates(model)
    elseif candidates isa AbstractMatrix
        return (Pair{Symbol,Any}[:candidate=>candidates],true)
    end
    (_symmetry_candidate_pairs(candidates),true)
end

"""
    strong_symmetry_report(model; candidates=:usual,
                           atol=1e-12, rtol=1e-10)

Discover and certify diagonal strong unitary symmetries without constructing a
`d^N` operator.  `candidates=:usual` tests the local clock (Pauli-Z parity for
qubits) and derives additional binary `±1` parity candidates from exact
nonzero supports of the model's microscopic one- and p-body operators.  An
explicit matrix or a named collection of matrices may be supplied instead.

Every candidate is checked term by term.  Hamiltonians and every jump
operator must commute separately with `U^tensor N`.  The candidate `status`
is `true`, `false`, or `missing`; scheduled operators that have not been
[`freeze`](@ref)d are reported as `missing`.  Scalar time-dependent rates do
not obstruct certification because they cannot change a commutator.

The nullspace search is complete only *within binary diagonal sign
symmetries*, and only when
`report.discovery.microscopic_support_complete=true`.  The usual clock is an
additional tested candidate, not part of that completeness claim.  An
incomplete search means that additional symmetries may exist; it does not
weaken any reported `true` certificate.
"""
function strong_symmetry_report(model::PIModel;candidates=:usual,
                                atol::Real=1e-12,rtol::Real=1e-10)
    pairs,discovery_complete=_strong_candidate_pairs(model,candidates)
    checked=Any[]
    for (raw_name,U) in pairs
        U isa AbstractMatrix||throw(ArgumentError(
            "strong-symmetry candidate $raw_name must be a matrix"))
        name=raw_name isa Symbol ? raw_name : Symbol(string(raw_name))
        matrix,diagonal,_,_=_strong_checked_diagonal(
            model.basis,U;atol,rtol)
        any(existing->_strong_same_unitary(
            existing.unitary,matrix;atol,rtol),checked)&&continue
        term_reports=map(enumerate(model.terms)) do indexed
            term_index,term=indexed
            report=_strong_term_commutation(
                term,model.basis,diagonal;atol,rtol)
            merge((term_index,term_type=typeof(term)),report)
        end
        statuses=getproperty.(term_reports,:status)
        status=any(value->value===false,statuses) ? false :
               any(ismissing,statuses) ? missing : true
        charges=_strong_pattern_charges(
            model.basis,diagonal;atol,rtol)
        push!(checked,(name,unitary=copy(matrix),status,
            terms=Tuple(term_reports),charges=Tuple(charges),
            trace_bearing_sectors=length(charges),
            validation=:termwise_strong_commutation))
    end
    discovery=(requested=candidates===:usual ? :usual : :explicit,
               complete_within=candidates===:usual ? :binary_sign :
                                                    :supplied_candidates,
               microscopic_support_complete=discovery_complete,
               method=candidates===:usual ?
                   :exact_binary_support_nullspace : :supplied_candidates)
    StrongSymmetryReport(model,model.basis,Tuple(checked),discovery)
end

function _strong_select_candidate(report::StrongSymmetryReport,selection)
    certified=filter(candidate->candidate.status===true,report.candidates)
    if selection===:auto
        useful=filter(candidate->length(candidate.charges)>1,certified)
        isempty(useful)&&throw(ArgumentError(
            "no nontrivial certified diagonal strong symmetry was found; " *
            "inspect strong_symmetry_report(model) for false or missing candidates"))
        best=first(useful)
        for candidate in Iterators.drop(useful,1)
            length(candidate.charges)>length(best.charges)&&(best=candidate)
        end
        return best
    end
    matches=filter(candidate->candidate.name==selection,report.candidates)
    isempty(matches)&&throw(ArgumentError(
        "strong-symmetry candidate $selection is absent from the report"))
    candidate=only(matches)
    candidate.status===true||throw(ArgumentError(
        "strong-symmetry candidate $selection is not certified " *
        "(status=$(candidate.status))"))
    candidate
end

"""
    strong_symmetry_reduction(model; candidate=:auto, report=nothing,
                              memory_budget=512*1024^2, ...)

Compile `model` matrix-free once and construct an exhaustively certified
[`RestrictedLiouvillian`](@ref) for *every* trace-bearing charge of one
diagonal strong symmetry.  `candidate=:auto` chooses the certified candidate
with the largest number of charge blocks; it never chooses one stationary
charge block.  Pass a candidate name from [`strong_symmetry_report`](@ref) to
make the symmetry choice explicit.

`model` must be autonomous.  Driven operator schedules must first be frozen at
an explicit time.  Each returned reduced operator independently certifies
ambient-coordinate leakage, and supported fixed kernels use the lowered
matrix-free backend.
"""
function strong_symmetry_reduction(model::PIModel;candidate=:auto,
        report=nothing,atol::Real=1e-12,rtol::Real=1e-10,
        invariance_atol::Real=0,invariance_rtol=nothing,
        restriction_backend::Symbol=:auto,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET)
    restriction_backend in (:auto,:lowered,:embedded)||throw(ArgumentError(
        "restriction_backend must be :auto, :lowered, or :embedded"))
    _require_autonomous(model,"automatic strong-symmetry reduction")
    prepared_report=report===nothing ?
        strong_symmetry_report(model;atol,rtol) : report
    prepared_report isa StrongSymmetryReport||throw(ArgumentError(
        "report must be a StrongSymmetryReport"))
    prepared_report.model===model||throw(ArgumentError(
        "the strong-symmetry report belongs to a different PIModel; " *
        "recompute it after changing any term"))
    selected=_strong_select_candidate(prepared_report,candidate)
    # Candidate matrices are ordinary Julia arrays and can be mutated by a
    # caller after report construction.  Recheck the selected copy at this
    # ownership boundary before treating it as a prepared reduction.
    refreshed=strong_symmetry_report(model;candidates=[
        selected.name=>copy(selected.unitary)],atol,rtol)
    selected=only(refreshed.candidates)
    selected.status===true||throw(ArgumentError(
        "the selected strong-symmetry candidate no longer passes " *
        "termwise certification"))
    _require_model_preparation_budget(
        model,memory_budget;
        operation="automatic strong-symmetry source preparation")
    # Strong reduction is a term-resolved consumer.  Preparing this unfused
    # plan once avoids rebuilding it independently for every charge when the
    # ordinary deterministic compiler would otherwise fuse static kernels.
    source=_term_resolved_liouvillian_plan(model)
    source_bytes=BigInt(Base.summarysize(source))
    source_workspace=_performance_linear_operator_workspace_bytes(source)
    _require_performance_budget(
        "automatic strong-symmetry matrix-free source",
        source_bytes+source_workspace,memory_budget;guidance=
        "Reduce the retained basis/model size or raise the explicit budget.")
    _,selected_diagonal,_,_=_strong_checked_diagonal(
        model.basis,selected.unitary;atol,rtol)
    charge_dimensions=_strong_equal_charge_dimensions(
        model.basis,selected_diagonal,selected.charges;atol,rtol)
    certification_transient=_performance_array_bytes(
        length(model.basis),eltype(source),0;linear_arrays=2)+
        source_workspace
    source_plan_supported=source.kernels!==nothing&&
        _restricted_static_kernels(source.kernels)
    predicted_backends=map(charge_dimensions) do _
        # Equal ket/bra charges constructed by
        # diagonal_symmetry_restriction are Cartesian in every Schur block.
        can_lower=source_plan_supported
        if restriction_backend===:lowered
            can_lower||throw(ArgumentError(
                "backend=:lowered requires fixed supported kernels and " *
                "Cartesian charge support"))
            :lowered
        elseif restriction_backend===:embedded
            :embedded
        elseif restriction_backend===:auto
            can_lower ? :lowered : :embedded
        else
            error("unreachable restriction backend")
        end
    end
    lowered_count=count(==(:lowered),predicted_backends)
    embedded_count=count(==(:embedded),predicted_backends)
    reduced_workspace_bound=_performance_array_bytes(
        sum(charge_dimensions),
        eltype(source),0;linear_arrays=3)
    # A sliced lowered plan cannot retain more dense kernel payload than one
    # conservative copy of the source plan per charge. Embedded compatibility
    # operators retain two ambient vectors and one fresh source workspace per
    # charge. Certification uses one additional such transient sequentially.
    sliced_plan_bound=BigInt(lowered_count)*source_bytes
    embedded_retained_bound=BigInt(embedded_count)*certification_transient
    # The full source workspace is a conservative upper bound for every
    # sliced factorized-gain scratch buffer. Count it explicitly in addition
    # to the three reduced Schur-block vectors above.
    lowered_gain_scratch_bound=BigInt(lowered_count)*source_workspace
    # Account conservatively for tuple/vector/dictionary headers in each
    # sliced plan. Their payload is already covered above, but `summarysize`
    # includes small non-scalar structural objects as well.
    structural_overhead_bound=BigInt(length(charge_dimensions))*8192
    # Each restriction stores ascending Int coordinates and one ambient
    # BitVector.  Guard this before those potentially large arrays are built.
    restriction_bytes_bound=sum(
        (2BigInt(sizeof(Int))*BigInt(dimension)+
         cld(BigInt(length(model.basis)),big(8))
         for dimension in charge_dimensions);init=big(0))
    retained_minimum=source_bytes+restriction_bytes_bound+
        reduced_workspace_bound+sliced_plan_bound+embedded_retained_bound+
        lowered_gain_scratch_bound+structural_overhead_bound
    lowering_setup_transient=source_plan_supported ? source_bytes : big(0)
    preflight_peak=retained_minimum+certification_transient+
        lowering_setup_transient
    _require_performance_budget(
        "automatic strong-symmetry reduction preflight",preflight_peak,
        memory_budget;guidance=
        "Select one named symmetry with fewer charge blocks or raise the " *
        "explicit memory budget.")
    restrictions=map(zip(selected.charges,charge_dimensions)) do item
        charge,expected_dimension=item
        restriction=diagonal_symmetry_restriction(
            model.basis,selected.unitary;charge,atol,rtol,
            label=(kind=:automatic_strong_symmetry,
                   candidate=selected.name,charge))
        length(restriction)==expected_dimension||error(
            "internal error: charge-dimension preflight disagrees with " *
            "restriction construction")
        (charge=charge,restriction=restriction)
    end
    sectors=map(restrictions) do item
        charge=item.charge
        restriction=item.restriction
        operator=RestrictedLiouvillian(
            source,restriction;atol=invariance_atol,
            rtol=invariance_rtol,backend=restriction_backend)
        norm(operator.tracevec)>zero(_real_float_type(eltype(operator)))||
            error("internal error: an equal-charge restriction has zero trace")
        (charge,dimension=length(restriction),restriction,operator,
         trace_norm=norm(operator.tracevec),
         leakage_certificate=operator.certificate)
    end
    retained_source=BigInt(Base.summarysize(source))
    retained_sectors=sum(sectors;init=big(0)) do sector
        operator=sector.operator
        BigInt(Base.summarysize(sector.restriction))+
        BigInt(Base.summarysize(operator.tracevec))+
        BigInt(Base.summarysize(operator.certificate))+
        BigInt(Base.summarysize(operator.compressed_source))+
        BigInt(Base.summarysize(operator.compatibility_workspace))
    end
    retained_total=retained_source+retained_sectors
    _require_performance_budget(
        "automatic strong-symmetry retained reductions",retained_total,
        memory_budget;guidance=
        "Select one named symmetry with fewer charge blocks, use a larger " *
        "explicit budget, or construct individual restrictions manually.")
    resources=(memory_budget=_memory_budget_bytes(memory_budget),
               retained_source_bytes=retained_source,
               retained_sector_bytes=retained_sectors,
               retained_bytes=retained_total,
               preflight_peak_upper_bound=preflight_peak,
               certification_transient_bytes=certification_transient,
               lowering_setup_transient_bytes=lowering_setup_transient,
               lowered_gain_scratch_bound,
               structural_overhead_bound,
               predicted_backends=Tuple(predicted_backends),
               accounting=:source_plus_all_charge_restrictions,
               setup_guards=:compiled_source_and_exhaustive_per_sector)
    StrongSymmetryReduction(model,source,selected,Tuple(sectors),resources)
end

"""
    strong_symmetry_steady_states(reduction; embed_states=true,
                                  memory_budget=512*1024^2, kwargs...)

Solve the trace-fixed stationary equation in every trace-bearing charge block
of a [`StrongSymmetryReduction`](@ref).  The result always contains one entry
per charge; it never silently chooses a sector.  By default every entry
includes the ambient [`PIState`](@ref), reduced coordinates, invariance
certificate, and the independent full-coordinate residual produced by
[`restricted_steady_state`](@ref).
Set `embed_states=false` to retain only reduced coordinates and diagnostics;
one ambient state is still a bounded transient while its full residual is
validated.

This computes one stationary solution per charge block.  It does not certify
uniqueness or positivity inside a block.
"""
function strong_symmetry_steady_states(reduction::StrongSymmetryReduction;
        embed_states::Bool=true,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,kwargs...)
    haskey(kwargs,:return_info)&&throw(ArgumentError(
        "strong_symmetry_steady_states always returns full solver metadata; " *
        "do not pass return_info"))
    for key in (:workspace,:preconditioner)
        haskey(kwargs,key)&&throw(ArgumentError(
            "$key cannot be shared across distinct strong-charge operators; " *
            "solve each reduction.sectors entry explicitly to supply " *
            "sector-owned $key"))
    end
    krylovdim=Int(get(kwargs,:krylovdim,30))
    krylovdim>0||throw(ArgumentError("krylovdim must be positive"))
    T=eltype(reduction.source)
    ambient=length(reduction.model.basis)
    reduced_output=_performance_entries_bytes(
        sum(sector.dimension for sector in reduction.sectors),T)
    ambient_output=embed_states ? _performance_entries_bytes(
        BigInt(ambient)*length(reduction.sectors),T) : big(0)
    result_metadata=BigInt(length(reduction.sectors))*4096
    one_ambient_transient=embed_states ? big(0) :
        _performance_entries_bytes(ambient,T)
    per_sector_peak=maximum((
        _performance_gmres_bytes(sector.dimension,T,krylovdim)+
        _performance_linear_operator_workspace_bytes(sector.operator)+
        reduction.resources.certification_transient_bytes+
        one_ambient_transient
        for sector in reduction.sectors);init=big(0))
    aggregate_peak=BigInt(reduction.resources.retained_bytes)+
        reduced_output+ambient_output+result_metadata+per_sector_peak
    _require_performance_budget(
        "strong-symmetry stationary ensemble",aggregate_peak,memory_budget;
        guidance="Use embed_states=false, reduce krylovdim, solve selected " *
                 "charge sectors explicitly, or raise the explicit budget.")
    results=map(reduction.sectors) do sector
        # A one-coordinate trace-bearing charge sector has an identically
        # zero trace-preserving generator.  Its trace constraint nevertheless
        # fixes the state uniquely; give GMRES the natural unit scale instead
        # of asking the generic operator probe to infer a scale from zero.
        solved=if sector.dimension==1&&!haskey(kwargs,:operator_scale)
            R=_real_float_type(eltype(sector.operator))
            restricted_steady_state(
                sector.operator;return_info=true,operator_scale=one(R),
                kwargs...)
        else
            restricted_steady_state(
                sector.operator;return_info=true,kwargs...)
        end
        returned=embed_states ? solved : merge(solved,(state=nothing,))
        merge((charge=sector.charge,dimension=sector.dimension,
               backend=sector.operator.backend),returned)
    end
    (candidate=reduction.candidate.name,
     unitary=copy(reduction.candidate.unitary),
     complete_trace_bearing_charge_enumeration=true,
     offdiagonal_charge_blocks_included=false,
     global_stationary_manifold_certified=false,
     selected_stationary_sector=nothing,
     resources=(memory_budget=_memory_budget_bytes(memory_budget),
                aggregate_peak_upper_bound=aggregate_peak,
                retained_reduction_bytes=reduction.resources.retained_bytes,
                reduced_output_bytes=reduced_output,
                ambient_output_bytes=ambient_output,
                result_metadata_bytes=result_metadata,
                maximum_sector_solve_peak=per_sector_peak,
                embedded_states=embed_states),
     sectors=Tuple(results))
end

"""
    strong_symmetry_spectra(reduction; method=:krylov, nev=nothing,
                            krylovdim=nothing, vectors=false,
                            validate_full=true, ...)

Compute a Liouvillian spectrum separately in every trace-bearing strong-charge
block.  By default right Ritz vectors are retained internally long enough to
evaluate [`restriction_full_residual`](@ref) against the original ambient
Liouvillian, then discarded unless `vectors=true`.  A failed full residual
raises instead of accepting a reduced-space Ritz pair.

Selected iterative spectra remain charge-resolved partial spectra; the return
value does not relabel their union as a certified global gap. Off-diagonal
ket/bra charge blocks are not trace bearing and are deliberately omitted, so
even complete per-sector diagonal spectra are not the global Liouvillian
spectrum.
When `nev` is omitted, each block requests at most six modes; an explicit
`nev` is never reduced silently.
"""
function strong_symmetry_spectra(reduction::StrongSymmetryReduction;
        method=:krylov,nev::Union{Nothing,Integer}=nothing,
        krylovdim::Union{Nothing,Integer}=nothing,
        vectors::Bool=false,validate_full::Bool=true,
        atol::Real=1e-10,rtol::Real=1e-8,
        full_atol::Real=atol,full_rtol::Real=rtol,
        memory_budget=_DEFAULT_HIGHLEVEL_MEMORY_BUDGET,kwargs...)
    haskey(kwargs,:return_info)&&throw(ArgumentError(
        "strong_symmetry_spectra always returns full solver metadata; " *
        "do not pass return_info"))
    for key in (:workspace,:operator_workspace,:initial_vector,
                :initial_subspace,:preconditioner)
        haskey(kwargs,key)&&throw(ArgumentError(
            "$key cannot be shared across distinct strong-charge operators; " *
            "solve each reduction.sectors entry explicitly to supply " *
            "sector-owned $key"))
    end
    method isa Symbol||throw(ArgumentError("method must be a Symbol"))
    selected_method=_canonical_spectrum_algorithm(method)
    selected_method===:auto&&throw(ArgumentError(
        "strong_symmetry_spectra requires an explicit method"))
    sector_nevs=map(sector->
        nev===nothing ? min(6,sector.dimension) : Int(nev),
        reduction.sectors)
    all(index->0<sector_nevs[index]<=reduction.sectors[index].dimension,
        eachindex(sector_nevs))||throw(ArgumentError(
        "nev must lie between 1 and every selected charge-sector dimension"))
    sector_krylovdims=map(sector_nevs) do sector_nev
        krylovdim===nothing ? max(20,2sector_nev+4) : Int(krylovdim)
    end
    all(>(0),sector_krylovdims)||throw(ArgumentError(
        "krylovdim must be positive"))
    retain_vectors=vectors||validate_full
    T=eltype(reduction.source)
    solver_peaks=map(eachindex(reduction.sectors)) do index
        sector=reduction.sectors[index]
        if selected_method===:dense
            _performance_array_bytes(
                sector.dimension,T,retain_vectors ? 7 : 5;
                linear_arrays=6)
        else
            _selected_spectrum_workspace_bytes(
                sector.operator,selected_method,
                sector_krylovdims[index],sector_nevs[index];
                vectors=retain_vectors,kwargs...)
        end
    end
    full_residual_peak=validate_full ?
        reduction.resources.certification_transient_bytes : big(0)
    maximum_solver_peak=maximum(solver_peaks;init=big(0))+
        full_residual_peak
    retained_counts=selected_method===:dense ?
        map(sector->sector.dimension,reduction.sectors) : sector_nevs
    retained_modes=sum(retained_counts)
    values_output=_performance_entries_bytes(retained_modes,T)
    vectors_output=vectors ? sum(
        (_performance_entries_bytes(
            BigInt(reduction.sectors[index].dimension)*retained_counts[index],T)
         for index in eachindex(reduction.sectors));init=big(0)) : big(0)
    residual_metadata=validate_full ? BigInt(retained_modes)*4096 : big(0)
    result_metadata=BigInt(length(reduction.sectors))*4096
    aggregate_peak=BigInt(reduction.resources.retained_bytes)+
        values_output+vectors_output+residual_metadata+result_metadata+
        maximum_solver_peak
    _require_performance_budget(
        "strong-symmetry sector spectra",aggregate_peak,memory_budget;
        guidance="Reduce nev/krylovdim, disable full-residual validation, " *
                 "solve selected charge sectors explicitly, or raise the " *
                 "explicit budget.")
    results=map(eachindex(reduction.sectors)) do sector_index
        sector=reduction.sectors[sector_index]
        sector_nev=sector_nevs[sector_index]
        sector_krylovdim=sector_krylovdims[sector_index]
        spectrum=pi_liouvillian_spectrum(
            sector.operator;method=selected_method,vectors=retain_vectors,
            nev=sector_nev,krylovdim=sector_krylovdim,
            return_info=true,atol,rtol,memory_budget,kwargs...)
        full_reports=if validate_full
            work=RestrictedLiouvillianWorkspace(
                sector.operator.source,sector.operator.restriction)
            reports=map(eachindex(spectrum.values)) do index
                restriction_full_residual(
                    sector.operator,view(spectrum.vectors,:,index);
                    eigenvalue=spectrum.values[index],workspace=work)
            end
            R=_real_float_type(eltype(sector.operator))
            absolute=_restriction_tolerance(R,full_atol,"full_atol")
            relative=_restriction_tolerance(R,full_rtol,"full_rtol")
            for report in reports
                tolerance=absolute+relative*max(abs(report.eigenvalue),one(R))
                report.relative_residual<=tolerance||throw(ArgumentError(
                    "a reduced strong-symmetry eigenpair fails the " *
                    "full-coordinate residual check: relative_residual=" *
                    "$(report.relative_residual), tolerance=$tolerance"))
            end
            Tuple(reports)
        else
            nothing
        end
        returned=vectors ? spectrum : merge(spectrum,(vectors=nothing,))
        (charge=sector.charge,dimension=sector.dimension,
         backend=sector.operator.backend,spectrum=returned,
         full_residuals=full_reports,validated_full=validate_full,
         scope=selected_method===:dense ? :complete_trace_bearing_charge_sector :
                                         :partial_trace_bearing_charge_sector)
    end
    (candidate=reduction.candidate.name,
     unitary=copy(reduction.candidate.unitary),
     complete_trace_bearing_charge_enumeration=true,
     offdiagonal_charge_blocks_included=false,
     global_spectrum=false,
     resources=(memory_budget=_memory_budget_bytes(memory_budget),
                aggregate_peak_upper_bound=aggregate_peak,
                retained_reduction_bytes=reduction.resources.retained_bytes,
                values_output_bytes=values_output,
                vectors_output_bytes=vectors_output,
                residual_metadata_bytes=residual_metadata,
                result_metadata_bytes=result_metadata,
                maximum_sector_solver_peak=maximum_solver_peak),
     sectors=Tuple(results))
end
