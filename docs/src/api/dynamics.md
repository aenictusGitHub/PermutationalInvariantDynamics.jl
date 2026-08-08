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

Fixed floating rates and `hbar` participate in the prepared plan's scalar
precision; they are never first narrowed to the operator prototype type.
When this widens a dense fixed kernel, its blocks and contractions are
converted once during preparation so preallocated application does not create
mixed-precision packing buffers; sparse one-body support remains sparse.
Callable scalar rates must evaluate to finite real values representable by the
precision selected when the plan was compiled. A wider value, or a nonzero
value that would underflow, raises with guidance to recompile at wider
precision. Negative rates remain valid for deterministic time-local
generators, although such a generator need not define a completely positive
map. Fixed operators, in-place prototypes, and every evaluated in-place
operator must contain only finite coefficients. `apply!` and `apply_adjoint!`
require non-aliasing source and destination arrays, including overlapping
views.

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

For an autonomous prepared generator, the high-level solver can replace many
fixed RK4 applications by adaptive restarted-Arnoldi exponential actions:

```julia
solution = solve_dynamics(
    prepared, rho0, (0.0, 20.0);
    saveat=0.1,
    algorithm=ExpvAlgorithm(
        krylovdim=30,
        atol=1e-11,
        rtol=1e-9,
    ),
)
```

The output grid remains exact: each adjacent interval is propagated as
$\exp(\Delta t\,\mathcal L)\rho$. One `KrylovExpvWorkspace` and one
task-owned Liouvillian application workspace are reused across all intervals.
Rejected time slices only reevaluate the small projected exponential and
reuse their Arnoldi factorization. Driven generators and parameter-dependent
applications are rejected because one fixed exponential does not represent
their dynamics. Keep the default RK4 route for driven models, or use
`dynamics_problem` with an adaptive SciML solver.

## Physical fully symmetric ket dynamics

For closed pure-state dynamics confined to the sole fully symmetric irrep,
the ket-native plan avoids the quadratic density coordinate. The immutable
plan is shareable; fixed-step and Krylov workspaces are task-owned. Read
[Symmetric pure kets and block-resolved
entropy](../symmetric_kets_and_block_entropy.md) before substituting this
physical ket for a weak-PI or HOPS pseudo-ket.

```@docs
SymmetricKetHamiltonianPlan
SymmetricKetWorkspace
apply_symmetric_hamiltonian!
evolve_symmetric_ket!
time_evolve_symmetric_ket
krylov_evolve_symmetric_ket!
krylov_time_evolve_symmetric_ket
symmetric_ket_expectation
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

## PI hierarchy of pure states

For finite-memory environments represented by exponential correlations,
PI--HOPS evolves a hierarchy of weak-PI pseudo-kets instead of density-valued
ADOs. The exact PI backend requires shared baths with collective/PI coupling
operators: independent local colored noises generally break permutation
symmetry on each realization even when their ensemble average is PI.

Prepare one immutable `HOPSPlan`, then give each simultaneous stochastic path
its own `HOPSWorkspace` and RNG. Linear HOPS returns an unnormalized root
pseudo-ket. `hops_density` forms its outer-product contribution and
`hops_average` averages those contributions without pathwise normalization.
Converge the exponential decomposition, hierarchy depth, time step, and
trajectory count independently. See [PI hierarchy of pure states
(HOPS)](../hops.md) for conventions and a complete workflow.

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
