# Reproducible experiments

The experiment layer packages one calculation together with the evidence
needed to interpret it:

```text
PIExperiment
    │
    ├── plan_experiment      resource estimate and selected exact route
    │
    └── verified_solve
            ├── high-level solver result
            ├── requested observables
            ├── physical and numerical checks
            └── environment and model provenance
```

It does not introduce another numerical backend. It coordinates
[`stationary_state`](@ref), [`solve_dynamics`](@ref),
[`recommend_solver`](@ref), [`state_diagnostics`](@ref), and the convergence
study functions while preserving their precision, memory, and ownership
contracts.

## A stationary-state experiment

```julia
using PermutationalInvariantDynamics

N = 6
basis = PIBasis(N, 2)
spin = spin_matrices(2)

model = PIModel(basis, (
    LocalHamiltonian(0.7 * spin.jx),
    LocalJump(spin.jm; rate=0.12),
    LocalJump(spin.jp; rate=0.02),
))

experiment = PIExperiment(
    model;
    task=:steady_state,
    algorithm=AutoAlgorithm(),
    observables=(
        polarization=collective_operator(basis, spin.jz) / N,
    ),
    memory_budget=512 * 1024^2,
    metadata=(
        project="driven ensemble scan",
        parameter_set="figure-2-point-17",
    ),
)
```

Inspect the planned route before allocating solver storage:

```julia
plan = explain_experiment(experiment)

plan.backend
plan.selected_algorithm
plan.recommendation.reason
plan.recommendation.known_peak_bytes
plan.recommendation.budget_status
```

The planning call validates the specification and invokes the ordinary
resource preflight. It does not compile or solve the model.

For a raw `PIModel`, construction snapshots every fixed built-in term
operator, including an `InPlaceTimeOperator` prototype, before retaining the
model. Later mutation of the caller's matrix therefore cannot change the
experiment. Prepared immutable sources are shared. Custom extension terms and
callable schedules retain their documented caller-owned behavior; because
their captured state cannot be inspected generically, the provenance digest
is marked incomplete.

Run the calculation with its requested checks:

```julia
result = verified_solve(experiment)

rho_ss = result.solution.state
result.observables[:polarization]
result.report
result.provenance
```

For a steady state, the report retains:

- the solver residual, physical trace error, iteration status, and resource
  preflight;
- trace, Hermiticity, and positivity diagnostics for the returned state;
- whether a complete PI basis or an explicitly declared sector restriction
  was used;
- package, Julia, platform, thread-count, and user metadata;
- the ambient `BigFloat` precision and rounding mode, together with the
  per-value precision of any retained `BigFloat` data;
- a deterministic structural checksum of supported model and experiment
  fields.

No failed physical state is normalized, symmetrized, or clipped.

## Exactness is explicit

The default

```julia
representation = :complete_pi
```

requires every Schur sector in the PI basis. This is an exact representation
of the declared PI model, up to the reported numerical solver tolerance.

A deliberately restricted model must say so:

```julia
restricted = PIExperiment(
    restricted_model;
    representation=:declared_sectors,
)
```

Its execution report contains

```julia
restricted_plan.exactness.physical_approximation ==
    :user_declared_sector_restriction
```

The experiment layer never infers that omitted sectors are harmless. Use the
package's symmetry and population certification tools when a physical
invariance statement is required.

Likewise, `algorithm=:auto` may select an exact sparse or matrix-free route
within the ordinary high-level policy. An explicit direct, GMRES, or
exponential-action request is never replaced silently.

## Dynamics and time-step evidence

```julia
rho0 = iid_pure_state(basis, ComplexF64[1, 0])

dynamics_experiment = PIExperiment(
    model;
    task=:dynamics,
    initial_state=rho0,
    tspan=(0.0, 5.0),
    saveat=0.0:0.1:5.0,
    steps_per_interval=32,
    observables=(
        polarization=collective_operator(basis, spin.jz) / N,
    ),
)

dynamics = verified_solve(dynamics_experiment)
```

All retained states are checked for trace, Hermiticity, and positivity.
Without a refinement request, the report deliberately says

```julia
dynamics.report.refinement_converged === missing
```

because checking physicality at one resolution does not establish
time-discretization convergence.

Request a study explicitly when final-state accuracy needs refinement
evidence:

```julia
refinement = RefinementSpec(
    :steps_per_interval,
    (16, 32, 64, 128);
    atol=1e-9,
    rtol=1e-7,
    consecutive=2,
    require_convergence=true,
)

verified_dynamics = PIExperiment(
    model;
    task=:dynamics,
    initial_state=rho0,
    tspan=(0.0, 5.0),
    saveat=0.0:0.1:5.0,
    observables=(
        polarization=collective_operator(basis, spin.jz) / N,
    ),
    verification=VerificationSpec(refinement=refinement),
)

result = verified_solve(verified_dynamics)
result.report.refinement_converged
result.report.evidence.refinement.pairwise_errors
```

Each listed number is the number of fixed RK4 steps between adjacent saved
times. The comparison uses the retained final PI states. Consequently,
time-step refinement currently requires `save_states=true`.

Before the first refinement solve, the experiment preflight includes the
cumulative retained outputs from earlier levels plus the peak storage of the
next level. It never starts a refinement sequence that is already known to
exceed `memory_budget`.

State-free observable dynamics remain available when physical-state
validation is intentionally disabled:

```julia
streaming = PIExperiment(
    model;
    task=:dynamics,
    initial_state=rho0,
    tspan=(0.0, 5.0),
    saveat=0.0:0.01:5.0,
    observables=(polarization=collective_operator(basis, spin.jz) / N,),
    save_states=false,
    verification=VerificationSpec(require_physical=false),
)
```

The report then marks `physical_valid=missing`; it does not infer physicality
from the observable series.

## GMRES refinement

Krylov-dimension refinement is available only when GMRES was explicitly
requested:

```julia
gmres_refinement = RefinementSpec(
    :krylov_dimension,
    (20, 30, 40);
    atol=1e-9,
    rtol=1e-7,
    consecutive=1,
)

experiment = PIExperiment(
    model;
    task=:steady_state,
    algorithm=GMRESAlgorithm(maxiter=500),
    verification=VerificationSpec(refinement=gmres_refinement),
)
```

This restriction is intentional. A refinement request must not turn a direct
or automatic request into a different solver behind the user's back.

## Saving a result

```julia
save_experiment("run-017.pidrun", result)
archive = load_experiment(
    "run-017.pidrun";
    memory_budget=512 * 1024^2,
)
```

The `.pidrun` path is a versioned directory containing:

- one portable `.pid` checkpoint for each retained PI state;
- precision-preserving binary time and observable arrays;
- verification, route, environment, checksum, and user metadata.

Existing paths are never overwritten. The archive contains numerical results
and provenance, not executable Julia code. In particular, loading it does not
deserialize or execute model closures, workspaces, factorizations, or callback
objects. Archive schema 2 also flattens solver, physical-state, and refinement
evidence into inert string metadata; large state/resource payloads embedded in
diagnostics are deliberately omitted. Schema-1 archives remain readable.
Before allocating state or observable arrays, `load_experiment` validates
declared lengths against each file payload and checks the combined retained
state, basis, time, and observable estimate against `memory_budget`. It also
requires consistent state bases, task/state/time cardinalities, ordered finite
times, unique observable names, and scalar-versus-series metadata. The loader
does not validate physical positivity automatically; apply `validate_state`
when importing an untrusted numerical result.

```julia
archive.states
archive.times
archive.observables
archive.metadata["structural_digest"]
archive.metadata["verified"]
```

For a callable rate or another opaque user object, provenance records its type
but cannot inspect captured closure state. In that case
`result.provenance.digest_complete` is false. Record a source revision,
configuration-file checksum, or parameter-set identifier in `metadata`.

## Present limitations

The initial experiment layer intentionally covers deterministic PI
steady-state and dynamics calculations that return `PIState` objects.

It does not yet coordinate:

- composite or shared-pseudomode state physicality;
- HEOM/HOPS hierarchy and pseudomode-cutoff refinement;
- stochastic trajectory confidence intervals;
- spectral target refinement;
- executable model serialization.

Those workflows already have specialized plans and diagnostics. They should
be integrated only with task-specific evidence rather than treated as though
a deterministic PI-state residual certified them.

## API

```@docs
RefinementSpec
VerificationSpec
PIExperiment
ExperimentExecutionPlan
plan_experiment
explain_experiment
ExperimentProvenance
ExperimentReport
ExperimentResult
verified_solve
PI_EXPERIMENT_ARCHIVE_VERSION
ExperimentArchive
save_experiment
load_experiment
```
