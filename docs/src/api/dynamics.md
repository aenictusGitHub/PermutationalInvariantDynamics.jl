# Dynamics and evolution

## Liouvillian preparation and application

Use `InPlaceTimeOperator` for an operator-valued drive that is evaluated many
times. The compiled plan keeps representation geometry read-only and each
`LiouvillianWorkspace` owns the evaluated matrices. In particular, a driven
`LocalJump` is prepared as rectangular common-child Schur contractions and is
applied as a sum of `C * X * C'` products. It does not retain a table over all
pairs of PI coordinates. Driven `CorrelatedLocalJumps` use the same
factorization for every active Kossakowski channel, so arbitrary interference
between the supplied local operators is preserved. Vector, adjoint, and
batched applications all reuse this workspace; a batch evaluates the schedule
once and contracts all right-hand sides with matrix--matrix kernels.

For repeated small-system representation setup, a basis-owned
`OneBoxCGCache` may be supplied through `coefficient_cache` to
`LiouvillianPlan`, `compile`, `liouvillian`, `steady_state`, or
`compile_family`. Compatible models then reuse its one-box coefficients while
building every required one-body and Appendix-D geometry. Exact basis,
removal-depth, and scalar-precision mismatches raise. Compilation already
creates a bounded transient cache for supported small bases when more than one
geometry family is needed; an explicit cache is useful across independent
compilations.

Fixed local jump terms use compact rectangular Schur contractions in a
matrix-free plan. Fixed numeric Hamiltonian contributions and
anticommutator losses are also combined at compilation, while every physical
gain channel remains distinct. This lowers retained memory and repeated block
work without changing rates or introducing jump-channel cross terms. Calling
`liouvillian(...; representation=:sparse)` expands the same prepared factors
only for that explicit representation. Channel-resolved trajectory and
population commands automatically use their term-resolved lowering, so their
physical channel metadata is unaffected by deterministic-kernel fusion.

For a sufficiently large collection of Schur sectors, single-vector
application can opt into Julia-task parallelism with
`ThreadedLiouvillianWorkspace` and `threaded_apply!`. Setup assigns each
complete output sector to exactly one worker using a deterministic
cubic-block-cost partition. Workers only read the input and prepared kernel
data; there are no atomic output updates or scheduler-ordered reductions. Operator and rate
schedules are evaluated once before spawning. The adjoint route follows the
same ownership rule, while one task uses the ordinary serial fast path.

```julia
plan = LiouvillianPlan(model)
work = ThreadedLiouvillianWorkspace(plan; tasks=Threads.nthreads())
destination = similar(source)
threaded_apply!(destination, plan, source, work)
```

The workspace is caller-owned, reusable sequentially, and guarded against
concurrent use. This API does not alter global BLAS threading: benchmark
single-level BLAS and Julia-task configurations separately, because nested
threading can be slower for small Schur blocks. Raw allocating operator
functions cannot provide a prepared target-sector kernel and are rejected;
wrap such schedules in `InPlaceTimeOperator` first.

```@docs
liouvillian
MatrixFreeLiouvillian
LiouvillianPlan
LiouvillianWorkspace
ThreadedLiouvillianWorkspace
CompiledPIModel
compile
CompiledPIModelFamily
SpecializedPIModel
compile_family
specialize
apply!
apply_adjoint!
threaded_apply!
threaded_apply_adjoint!
isautonomous
freeze
dynamics_problem
PISolution
state
state_at
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
TrajectorySteadyStateResult
TrajectoryPlan
TrajectoryWorkspace
TrajectoryBatchWorkspace
quantum_trajectory
quantum_trajectories
trajectory_steady_state
trajectory_average
jump_statistics
trajectory_observable_statistics
trajectory_statistics
```

## Weak-PI pseudo-ket trajectories

```@docs
WeakPIPseudoKet
weak_pi_dimension
weak_pi_density
weak_pi_pseudoket
weak_pi_expectation
WeakPIKrausBranch
WeakPIJumpRecord
WeakPIQuantumTrajectory
WeakPITrajectoryPlan
WeakPITrajectoryWorkspace
WeakPITrajectoryBatchWorkspace
WeakPIBatchMeansDiagnostics
weak_pi_quantum_trajectory
weak_pi_quantum_trajectories
weak_pi_trajectory_steady_state
weak_pi_trajectory_average
weak_pi_trajectory_statistics
```

## Diffusive conditional dynamics

See the [diffusive-monitoring guide](../diffusive_monitoring.md) for the
canonical API documentation and the normalized Itô convention.

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
FloquetMap
FloquetWorkspace
FloquetBatchWorkspace
floquet_map
restricted_floquet_map
selected_floquet_multipliers
floquet_propagator
floquet_multipliers
floquet_exponents
floquet_gap
floquet_steady_state
stroboscopic_evolution
floquet_evolve
```
