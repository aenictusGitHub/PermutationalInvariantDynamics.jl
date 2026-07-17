# Research utilities, channels, and control

This experimental surface collects analysis and workflow tools which operate
directly on retained PI coordinates. None reconstructs a `d^N` density matrix.
The runnable overview is
[`examples/research_utilities.jl`](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/blob/main/examples/research_utilities.jl).

## Compressed spectra and populations

`spectral_trace(rho,f)` evaluates ``\mathrm{tr}[f(\rho)]`` from physical
Schur eigenvalues with exact multiplicities. Specialized entropy routines
remain preferable when logarithmic rescaling is needed. Population-coordinate
iterators expose the exact sector, GT pattern, multiplicity, population index,
and diagonal PI index; `population_transitions` adds the directed off-diagonal
rates of a certified `PopulationPlan`.

## Channels, measurements, and tomography

`PIChannel` and `MatrixFreePIChannel` act in equation-(7) coefficient
coordinates. `check_pi_channel` certifies trace preservation and, for a
materialized map, complete positivity on the retained direct-sum Schur
algebra. A certificate says nothing about omitted sectors.

PI POVMs use the same scope. Sampling builds one cumulative categorical table;
maximum-likelihood tomography uses a diluted, positivity-preserving
``R\rho R`` iteration and always reports convergence.

## Checkpoints

The built-in `:pid` format records schema version, scalar type, restricted
partitions, coefficients, optional time, and string metadata. Loading never
normalizes or repairs the state. JLD2 and HDF5 are weak dependencies activated
only when explicitly imported.

## Joint symmetries and gradients

`joint_symmetry_projector` intersects commuting unitary-conjugation charge
sectors without constructing a reduced Liouvillian. Construction verifies
commutation and certifies the intersection rank. Use one
`JointSymmetryProjectorWorkspace` per concurrent task.

`implicit_steady_state_gradient` solves the trace-fixed tangent equation with
matrix-free GMRES. `checkpointed_adjoint_gradient` propagates piecewise
controls with RK4, retains only periodic forward checkpoints, recomputes each
segment once, and uses endpoint-trapezoidal gradient quadrature. Converge the
Krylov tolerance or control grid for the intended observable.

## API

```@docs
spectral_trace
PopulationCoordinate
PopulationCoordinates
each_population_coordinate
PopulationTransition
population_transitions
AbstractPIChannel
PIChannel
MatrixFreePIChannel
apply_channel!
apply_channel
compose_channels
channel_adjoint
identity_channel
kraus_channel
PIChannelCheck
check_pi_channel
pi_povm_probabilities
PIPOVMSample
sample_pi_povm
PITomographyResult
maximum_likelihood_tomography
PI_CHECKPOINT_VERSION
PIStateCheckpoint
checkpoint_state
save_checkpoint
load_checkpoint
JointSymmetryProjector
JointSymmetryProjectorWorkspace
joint_symmetry_projector
SteadyStateGradientPlan
SteadyStateGradientWorkspace
SteadyStateGradientResult
implicit_steady_state_gradient
AdjointControlResult
checkpointed_adjoint_gradient
```
