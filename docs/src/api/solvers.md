# Stationary states, spectra, and research solvers

## Stationary-state and Krylov algorithms

```@docs
steady_state
krylov_steady_state
KrylovWorkspace
ArnoldiWorkspace
JacobiDavidsonWorkspace
SchurSectorPreconditioner
schur_sector_preconditioner
preconditioner_cost
krylov_liouvillian_spectrum
harmonic_arnoldi_spectrum
implicitly_restarted_arnoldi_spectrum
jacobi_davidson_spectrum
liouvillian_eigenvalues
liouvillian_gap
```

## PI spectra

```@docs
pi_liouvillian_spectrum
pi_density_spectrum
pi_liouvillian_gap
pi_density_operator_spectrum
density_operator_spectrum
```

## Evans uniqueness and weak symmetries

```@docs
evans_uniqueness
has_unique_steady_state_evans
check_liouvillian_symmetry
is_liouvillian_symmetric
usual_liouvillian_symmetries
MatrixFreeSymmetryProjector
SymmetryProjectorWorkspace
matrixfree_symmetry_projector
```

Simultaneous commuting-charge projectors are documented in
[Research utilities and control](../research_utilities.md).

## Response and sensitivities

```@docs
liouvillian_modes
resolvent_norm
adjoint_evolve
sensitivity_problem
sensitivity_state
classical_fisher_information
observable_decay_modes
integrated_correlation_time
steady_state_susceptibility
pseudospectral_abscissa
qfim_from_derivatives
```

## Sizing and summaries

```@docs
estimate_basis_size
estimate_memory
basis_summary
model_summary
value_at
```

## High-level algorithms and results

```@docs
AbstractPIAlgorithm
AutoAlgorithm
DirectAlgorithm
SVDAlgorithm
EigenAlgorithm
ShiftInvertAlgorithm
GMRESAlgorithm
HarmonicArnoldiAlgorithm
SteadyStateResult
DynamicsResult
SpectrumResult
stationary_state
solve_dynamics
liouvillian_spectrum
diagnostics
pi_dimension
estimate_state_bytes
estimate_basis_bytes
estimate_liouvillian_bytes
estimate_geometry_bytes
estimate_solver_bytes
recommend_solver
```
