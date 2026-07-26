"""
    local_operator_matrix(operator; dimension=nothing)

Return a detached matrix representation of a one-particle operator for use in
PI model terms. Plain dense and sparse matrices are copied without changing
their scalar type or sparse support. `dimension`, when supplied, validates the
expected local Hilbert-space dimension.

Optional package extensions add this method for QuantumOptics.jl and
QuantumToolbox.jl operator objects. Only a genuine one-site operator is
accepted: kets, superoperators, rectangular maps, and full-system operators
whose dimension does not match the requested local dimension are rejected.
No full-Hilbert reconstruction or basis-order conversion is performed.
"""
function local_operator_matrix(operator::AbstractMatrix;
                               dimension::Union{Nothing,Integer}=nothing)
    rows,columns=size(operator)
    rows==columns||throw(DimensionMismatch(
        "a local quantum operator must be square; received size $(size(operator))"))
    rows>0||throw(ArgumentError("a local quantum operator cannot be empty"))
    if dimension!==nothing
        dimension isa Bool&&throw(ArgumentError(
            "the requested local operator dimension must be an integer, not Bool"))
        dimension>0||throw(ArgumentError(
            "the requested local operator dimension must be positive"))
        rows==dimension||throw(DimensionMismatch(
            "the operator has local dimension $rows, but dimension=$dimension was requested"))
    end
    all(isfinite,operator)||throw(ArgumentError(
        "a local quantum operator must contain only finite entries"))
    copy(operator)
end

function local_operator_matrix(operator;
                               dimension::Union{Nothing,Integer}=nothing)
    throw(ArgumentError(
        "unsupported local operator type $(typeof(operator)); pass its matrix "*
        "representation, or load a supported optional interoperability package"))
end
