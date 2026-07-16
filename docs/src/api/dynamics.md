# Dynamics and evolution

## Liouvillian preparation and application

```@docs
liouvillian
MatrixFreeLiouvillian
LiouvillianPlan
LiouvillianWorkspace
CompiledPIModel
compile
apply!
apply_adjoint!
isautonomous
freeze
dynamics_problem
PISolution
state
EvolutionWorkspace
evolve!
time_evolve
time_evolution
```

## Certified Schur-diagonal populations

```@docs
population_dimension
PopulationInvarianceReport
population_invariance
PopulationPlan
PopulationWorkspace
diagonal_populations
diagonal_populations!
state_from_populations
population_generator
evolve_populations!
PopulationSolution
solve_populations
stationary_populations
```

## Quantum trajectories

```@docs
QuantumTrajectory
TrajectoryPlan
TrajectoryWorkspace
TrajectoryBatchWorkspace
quantum_trajectory
quantum_trajectories
trajectory_average
jump_statistics
trajectory_observable_statistics
trajectory_statistics
```

## Mean-field predictions

```@docs
MeanFieldPlan
MeanFieldWorkspace
MeanFieldResult
meanfield_rhs!
meanfield_rhs
meanfield_problem
meanfield_evolve!
solve_meanfield
meanfield_jacobian
meanfield_stability
meanfield_stationary_state
meanfield_expectation
meanfield_collective_moments
meanfield_pbody_expectation
```

## Floquet analysis

```@docs
floquet_propagator
floquet_multipliers
floquet_exponents
floquet_gap
floquet_steady_state
stroboscopic_evolution
floquet_evolve
```
