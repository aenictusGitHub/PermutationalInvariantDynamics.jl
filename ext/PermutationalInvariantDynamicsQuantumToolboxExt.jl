module PermutationalInvariantDynamicsQuantumToolboxExt

import PermutationalInvariantDynamics
import QuantumToolbox

const PID=PermutationalInvariantDynamics
const QT=QuantumToolbox
const _QTOperator=QT.QuantumObject{<:QT.Operator}

PID.local_operator_matrix(operator::_QTOperator;
                          dimension::Union{Nothing,Integer}=nothing)=
    PID.local_operator_matrix(operator.data;dimension)

PID.LocalHamiltonian(operator::_QTOperator;kwargs...)=
    PID.LocalHamiltonian(PID.local_operator_matrix(operator);kwargs...)
PID.CollectiveHamiltonian(operator::_QTOperator;kwargs...)=
    PID.CollectiveHamiltonian(PID.local_operator_matrix(operator);kwargs...)
PID.LocalJump(operator::_QTOperator;kwargs...)=
    PID.LocalJump(PID.local_operator_matrix(operator);kwargs...)
PID.CollectiveJump(operator::_QTOperator;kwargs...)=
    PID.CollectiveJump(PID.local_operator_matrix(operator);kwargs...)

function PID.collective_operator(basis::PID.PIBasis,
                                 operator::_QTOperator;
                                 cache=nothing)
    PID.collective_operator(
        basis,PID.local_operator_matrix(operator;dimension=basis.d);cache)
end

end
