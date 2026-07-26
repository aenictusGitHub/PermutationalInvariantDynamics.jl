module PermutationalInvariantDynamicsQuantumOpticsExt

import PermutationalInvariantDynamics
import QuantumOptics

const PID=PermutationalInvariantDynamics
const QO=QuantumOptics
const QOB=QO.QuantumOpticsBase

function PID.local_operator_matrix(operator::QO.DataOperator;
                                   dimension::Union{Nothing,Integer}=nothing)
    QOB.samebases(operator)||throw(ArgumentError(
        "a QuantumOptics local operator must use the same left and right " *
        "Hilbert-space basis; maps between different bases are not supported"))
    PID.local_operator_matrix(operator.data;dimension)
end

PID.LocalHamiltonian(operator::QO.DataOperator;kwargs...)=
    PID.LocalHamiltonian(PID.local_operator_matrix(operator);kwargs...)
PID.CollectiveHamiltonian(operator::QO.DataOperator;kwargs...)=
    PID.CollectiveHamiltonian(PID.local_operator_matrix(operator);kwargs...)
PID.LocalJump(operator::QO.DataOperator;kwargs...)=
    PID.LocalJump(PID.local_operator_matrix(operator);kwargs...)
PID.CollectiveJump(operator::QO.DataOperator;kwargs...)=
    PID.CollectiveJump(PID.local_operator_matrix(operator);kwargs...)

function PID.collective_operator(basis::PID.PIBasis,
                                 operator::QO.DataOperator;
                                 cache=nothing)
    PID.collective_operator(
        basis,PID.local_operator_matrix(operator;dimension=basis.d);cache)
end

end
