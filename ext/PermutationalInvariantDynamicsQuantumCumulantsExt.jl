module PermutationalInvariantDynamicsQuantumCumulantsExt

using PermutationalInvariantDynamics
using LinearAlgebra
import QuantumCumulants

const PID=PermutationalInvariantDynamics
const SQA=QuantumCumulants.SecondQuantizedAlgebra

function _neutral_key(moments,key)
    key isa Symbol ?
        PermutationalInvariantDynamics._canonical_moment_key(moments,(key,)) :
        PermutationalInvariantDynamics._canonical_moment_key(moments,key)
end

function quantumcumulants_initial_values(
        moments::PermutationalInvariantDynamics.OrderedLocalMoments,
        symbolic_map)
    entries=try
        symbolic_map isa Union{NamedTuple,AbstractDict} ?
            collect(pairs(symbolic_map)) : collect(symbolic_map)
    catch error
        throw(ArgumentError(
            "symbolic_map must be a dictionary or named collection of neutral-key => symbolic-average entries: $(sprint(showerror,error))"))
    end
    output=Dict{Any,eltype(moments)}()
    for entry in entries
        neutral=first(entry);symbolic=last(entry)
        key=_neutral_key(moments,neutral)
        haskey(moments,key)||throw(KeyError(key))
        symbolic_order=try
            QuantumCumulants.get_order(symbolic)
        catch error
            throw(ArgumentError(
                "QuantumCumulants.get_order failed for the symbolic target of $key: $(sprint(showerror,error))"))
        end
        symbolic_order==length(key)||throw(ArgumentError(
            "neutral moment $key has order $(length(key)), but its QuantumCumulants target has order $symbolic_order"))
        haskey(output,symbolic)&&throw(ArgumentError(
            "multiple neutral moments map to the same QuantumCumulants symbolic average"))
        output[symbolic]=moments[key]
    end
    output
end

@inline function _tensor_level(index::Int,site::Int,d::Int)
    ((index-1)÷d^(site-1))%d+1
end

function _indexed_matrix_operator(matrix::AbstractMatrix,hilbert,indices,
                                  d::Int,transition_name::Symbol)
    p=length(indices);dimension=d^p
    size(matrix)==(dimension,dimension)||throw(DimensionMismatch(
        "a $p-body symbolic operator must be $dimension×$dimension"))
    seed=SQA.IndexedOperator(
        SQA.Transition(hilbert,transition_name,1,1),first(indices))
    expression=zero(matrix[begin])*seed
    @inbounds for column in 1:dimension,row in 1:dimension
        coefficient=matrix[row,column]
        iszero(coefficient)&&continue
        factors=map(1:p) do site
            a=_tensor_level(row,site,d);b=_tensor_level(column,site,d)
            SQA.IndexedOperator(
                SQA.Transition(hilbert,transition_name,a,b),indices[site])
        end
        product=foldl(*,factors)
        expression+=coefficient*product
    end
    expression
end

function _distinct_expression(expression,indices)
    length(indices)<=1&&return expression
    pairs=Tuple{SQA.Index,SQA.Index}[
        (indices[left],indices[right])
        for right in 2:length(indices) for left in 1:right-1]
    SQA.assume_distinct_index(expression,pairs)
end

function _unordered_sum(expression,indices)
    distinct=_distinct_expression(expression,indices)
    summed=foldl((value,index)->SQA.Σ(value,index),indices;init=distinct)
    summed*(one(BigInt)//factorial(big(length(indices))))
end

function _scale_local_pbody_rate(rate,p::Int)
    denominator=factorial(big(p))
    if rate isa AbstractFloat||
            (rate isa Complex&&real(rate) isa AbstractFloat)
        return PID._checked_mul_exact_ratio(rate,one(BigInt),denominator;
            context="QuantumCumulants local p-body subset rate")
    end
    rate*(one(BigInt)//denominator)
end

function _check_symbolic_pbody_symmetry(matrix,p::Int,d::Int)
    p<=1&&return matrix
    R=PID._real_float_type(eltype(matrix))
    tolerance=R(1e-10)*max(norm(matrix,Inf),one(R))
    dimension=d^p
    for adjacent in 1:p-1
        permutation=PID._tensor_swap_permutation(p,d,adjacent)
        maximum_difference=zero(R)
        @inbounds for column in 1:dimension,row in 1:dimension
            maximum_difference=max(maximum_difference,
                abs(matrix[row,column]-
                    matrix[permutation[row],permutation[column]]))
        end
        maximum_difference<=tolerance||throw(ArgumentError(
            "automatic QuantumCumulants lowering requires each p-body operator to be permutation symmetric"))
    end
    matrix
end

function _fixed_payload_value(value,schedule,time,description)
    if schedule!==nothing
        time===nothing&&throw(ArgumentError(
            "$description is time dependent; pass an explicit time and parameters"))
        throw(ArgumentError(
            "$description remained unevaluated after an explicit time was supplied"))
    end
    value===nothing&&throw(ArgumentError(
        "$description has no fixed prototype; pass an explicit time and parameters"))
    value
end

function _renamed_adjoint(expression,indices,partner_indices)
    renamed=SQA.change_index(expression,
        Dict(indices[position]=>partner_indices[position]
             for position in eachindex(indices)))
    adjoint(renamed)
end

function _append_payload_term!(hamiltonians,jumps,jumps_dagger,rates,payload,
                               hilbert,indices,partner_indices,d,
                               transition_name,time)
    payload.microscopic||throw(ArgumentError(
        "direct PI terms cannot be lowered automatically to QuantumCumulants; provide an explicit microscopic symbolic model"))
    operator=_fixed_payload_value(payload.operator,payload.operator_schedule,
                                  time,"term operator")
    operator isa AbstractMatrix||throw(ArgumentError(
        "automatic QuantumCumulants lowering requires matrix-valued microscopic operators"))
    rate=payload.rate
    rate isa Number||throw(ArgumentError(
        "term rate is time dependent; pass an explicit time and parameters"))
    p=payload.order
    selected=indices[1:p]
    _check_symbolic_pbody_symmetry(operator,p,d)
    raw=_indexed_matrix_operator(operator,hilbert,selected,d,transition_name)
    if payload.process===:hamiltonian
        payload.hbar isa Number&&!iszero(payload.hbar)||throw(ArgumentError(
            "Hamiltonian hbar must be a nonzero number for symbolic lowering"))
        push!(hamiltonians,(rate/payload.hbar)*_unordered_sum(raw,selected))
    elseif payload.process===:jump
        if payload.scope===:local
            jump=_distinct_expression(raw,selected)
            push!(jumps,jump);push!(jumps_dagger,adjoint(jump))
            push!(rates,_scale_local_pbody_rate(rate,p))
        elseif payload.scope===:collective
            jump=_unordered_sum(raw,selected)
            push!(jumps,jump)
            push!(jumps_dagger,_renamed_adjoint(
                jump,selected,partner_indices[1:p]))
            push!(rates,rate)
        else
            throw(ArgumentError("unsupported microscopic jump scope $(payload.scope)"))
        end
    else
        throw(ArgumentError("unsupported microscopic process $(payload.process)"))
    end
    nothing
end

function _evaluated_correlated_term(term,time,parameters)
    fixed_matrix=term.kossakowski isa AbstractMatrix
    fixed_rate=term.rate isa Number
    if fixed_matrix&&fixed_rate
        return term
    end
    time===nothing&&throw(ArgumentError(
        "time-dependent correlated jumps require an explicit time and parameters for symbolic lowering"))
    matrix=PID.value_at(term.kossakowski,time,parameters)
    rate=PID.value_at(term.rate,time,parameters)
    constructor=term isa PID.CorrelatedLocalJumps ?
        PID.CorrelatedLocalJumps : PID.CorrelatedCollectiveJumps
    constructor(term.operators,matrix;rate,atol=term.atol,rtol=term.rtol)
end

function _append_correlated_term!(jumps,jumps_dagger,rates,term,hilbert,
                                  index,partner_index,d,transition_name,time,
                                  parameters)
    fixed=_evaluated_correlated_term(term,time,parameters)
    factor=fixed.factor
    factor isa Tuple||throw(ArgumentError(
        "correlated jump factorization is unavailable after evaluation"))
    for coefficients in factor
        any(value->!iszero(value),coefficients)||continue
        matrix=zeros(promote_type(eltype(coefficients),
            mapreduce(eltype,promote_type,fixed.operators)),d,d)
        for operator in eachindex(fixed.operators)
            matrix .+= coefficients[operator].*fixed.operators[operator]
        end
        raw=_indexed_matrix_operator(matrix,hilbert,(index,),d,transition_name)
        if fixed isa PID.CorrelatedLocalJumps
            push!(jumps,raw);push!(jumps_dagger,adjoint(raw))
        else
            jump=SQA.Σ(raw,index)
            push!(jumps,jump)
            push!(jumps_dagger,_renamed_adjoint(
                jump,(index,),(partner_index,)))
        end
        push!(rates,fixed.rate)
    end
    nothing
end

function _default_seed_operators(hilbert,probe,d,transition_name,ground_state)
    [SQA.IndexedOperator(SQA.Transition(hilbert,transition_name,a,b),probe)
     for b in 1:d for a in 1:d if !(a==ground_state&&b==ground_state)]
end

function quantumcumulants_model(model::PID.PIModel;order::Integer=2,
        time=nothing,parameters=nothing,complete::Bool=false,
        scale::Bool=false,seed_operators=nothing,
        space_name::Symbol=:pid_atom,transition_name::Symbol=:σ,
        ground_state::Integer=1,complete_options=NamedTuple(),
        scale_options=NamedTuple())
    order>=1||throw(ArgumentError("cumulant order must be positive"))
    d=model.basis.d;N=model.basis.N
    1<=ground_state<=d||throw(ArgumentError(
        "ground_state must lie in 1:$d"))
    complete_options isa NamedTuple||throw(ArgumentError(
        "complete_options must be a NamedTuple"))
    scale_options isa NamedTuple||throw(ArgumentError(
        "scale_options must be a NamedTuple"))
    maximum_order=maximum((PID.body_order(term) for term in model.terms);init=1)
    hilbert=SQA.NLevelSpace(space_name,d,Int(ground_state))
    indices=[SQA.Index(hilbert,Symbol(:i,site),N,hilbert)
             for site in 1:maximum_order]
    partner_indices=[SQA.Index(hilbert,Symbol(:j,site),N,hilbert)
                     for site in 1:maximum_order]
    probe=SQA.Index(hilbert,:z,N,hilbert)
    payload=PID.cumulant_model_payload(model;time,parameters)
    hamiltonians=Any[];jumps=Any[];jumps_dagger=Any[];rates=Any[]
    for (term,description) in zip(model.terms,payload.terms)
        if term isa Union{PID.CorrelatedLocalJumps,
                          PID.CorrelatedCollectiveJumps}
            _append_correlated_term!(jumps,jumps_dagger,rates,term,hilbert,
                indices[1],partner_indices[1],d,transition_name,time,parameters)
        else
            _append_payload_term!(hamiltonians,jumps,jumps_dagger,rates,
                description,hilbert,indices,partner_indices,d,
                transition_name,time)
        end
    end
    seeds=seed_operators===nothing ?
        _default_seed_operators(hilbert,probe,d,transition_name,
                                Int(ground_state)) : collect(seed_operators)
    isempty(seeds)&&throw(ArgumentError("seed_operators cannot be empty"))
    reference=SQA.IndexedOperator(
        SQA.Transition(hilbert,transition_name,1,1),probe)
    hamiltonian=isempty(hamiltonians) ? 0*reference : sum(hamiltonians)
    equations=QuantumCumulants.meanfield(seeds,hamiltonian,jumps;
        Jdagger=jumps_dagger,rates,order=Int(order))
    complete&&(equations=QuantumCumulants.complete(
        equations;complete_options...))
    scale&&(equations=QuantumCumulants.scale(equations;scale_options...))
    metadata=(N=N,d=d,order=Int(order),subset_convention=:unordered,
        cumulant_approximation=true,completed=complete,scaled=scale,
        evaluated_at=time,correlated_channels=count(term->term isa Union{
            PID.CorrelatedLocalJumps,PID.CorrelatedCollectiveJumps},model.terms))
    (;hilbert,indices,partner_indices,probe_index=probe,
      seed_operators=seeds,hamiltonian,jumps,jumps_dagger,rates,equations,
      metadata)
end

end
