# Diffusive conditional dynamics

`diffusive_trajectory` evolves a PI density operator conditioned on collective
homodyne or heterodyne records.  The unconditional `PIModel` and the monitored
operators are deliberately separate:

```math
d\rho_c = \mathcal L[\rho_c]dt
 + \sqrt{\eta}\,\mathcal H[e^{-i\phi}c]\rho_c\,dW,
```

where

```math
\mathcal H[a]\rho=a\rho+\rho a^\dagger
-\mathrm{tr}[(a+a^\dagger)\rho]\rho.
```

Thus `model` must already contain the dissipator associated with `c`.  A
monitor adds the innovation and the measurement record only; it never inserts
a second dissipator implicitly.

## Prepared workflow

```julia
basis = PIBasis(20, 2)
sm = ComplexF64[0 1; 0 0]
gamma = 0.2
model = PIModel(basis, (CollectiveJump(sm; rate=gamma),))
rho0 = computational_product_state(basis, 2)

# The collapse operator is sqrt(gamma) * J_-.
monitor = homodyne_monitor(sqrt(gamma) * sm;
    efficiency=0.75, phase=0, label=:emission)
plan = DiffusivePlan(model, monitor)
workspace = DiffusiveWorkspace(plan, rho0)

result = diffusive_trajectory(
    plan, rho0, range(0, 1; length=101);
    dt=0.001, workspace, rng=MersenneTwister(4),
    observables=(magnetization=collective_spin(basis, :z),),
)
```

For repeated or threaded ensembles, prepare the complete sampling request
once. `DiffusiveBatchPlan` validates and stores the output grid, time step,
and observable plans; `DiffusiveBatchWorkspace` owns one numerical workspace
and random stream per worker:

```julia
batch = DiffusiveBatchPlan(plan, rho0, range(0, 1; length=101);
    dt=0.001,
    observables=(magnetization=collective_spin(basis, :z),))
batch_workspace = DiffusiveBatchWorkspace(batch, rho0; workers=Threads.nthreads())
paths = diffusive_trajectories(batch, rho0, 1000;
    workspace=batch_workspace, seed=7, threaded=true)
```

The immutable batch plan is shareable. Its workspace is mutable, reusable
sequentially, and must not be shared by concurrent callers.

A one-particle monitor matrix is always lifted to the collective operator
``\sum_i c^{(i)}``.  Passing a prepared `PIOperator` monitors that PI operator
directly.  Particle-resolved local records reveal particle labels and do not
preserve the PI conditional state, so this API rejects them rather than
silently averaging a non-PI measurement.

`heterodyne_monitor` produces orthogonal I/Q records with two independent real
Wiener processes and strength ``\sqrt{\eta/2}`` per quadrature.  Efficiencies
and phases may be scalar schedules `(time, parameters)`.  Their values are
validated at every step.

## Records, ensembles, and numerical scope

`DiffusiveTrajectory.records` and `.innovations` are cumulative at the output
times.  Their rows are identified by `record_labels`; heterodyne labels carry
`:I` and `:Q` suffixes.  `save_states=false` discards conditional state history
while retaining these records and optional observable samples.

```julia
paths = diffusive_trajectories(
    plan, rho0, range(0, 1; length=51), 1000;
    dt=0.001, seed=7, threaded=true,
)
rho_mean = diffusive_average(paths)
```

Streams are derived from trajectory index, so serial and threaded calls are
reproducible.  Each worker owns one `DiffusiveWorkspace`; the immutable plan is
shared.

The current integrator is normalized Euler--Maruyama.  It is preallocated and
converges to the Itô SME, but finite steps do not by themselves certify
positivity.  Converge `dt`, and call `validate_state` when strict conditional
state auditing is required.  The ensemble mean converges to the unconditional
master equation already stored in the model.

## Confidence-controlled stopping

`adaptive_quantum_trajectories`,
`adaptive_weak_pi_quantum_trajectories`, and
`adaptive_diffusive_trajectories` process deterministic batches until every
requested observable at every saved time has a confidence half-width below
`atol + rtol*abs(mean)`. They retain only online means and variances, not
trajectory state histories. The result reports
`converged=false` and `stopping_reason=:maximum_trajectories` when the sample
cap is reached; it never silently accepts the cap as convergence.

Stopping uses a finite-horizon simultaneous empirical-Bernstein bound. For
each observable and saved time, the implementation combines its sample
variance with a certified spectral-range bound of the Hermitian PI operator.
The failure probability `1-confidence` is divided across all requested
observable/time pairs and every possible batch check up to
`max_trajectories`. Consequently, seeing no rare jump in an early batch does
not produce a zero-width certificate for a nonconstant observable. This bound
controls the complete adaptive stopping request; the pointwise normal
intervals retained in `result.observables` remain useful descriptive output
but are not the stopping certificate. The simultaneous half-width and worst
tolerance ratio used at each check are stored in `convergence_history`.
`metadata.effective_independent_samples` is the number of separately seeded
trajectories used by the bound; this independence statement does not cover
time-integration or finite-time initialization bias.
Every sampled value is checked against the padded spectral bound. A value
outside it raises instead of producing an invalid confidence claim. This
observed-value check is not, by itself, a bounded-support proof for the
finite-step diffusive sampler: Euler--Maruyama is not positivity preserving
and its Gaussian increments have unbounded support. The diffusive
empirical-Bernstein result is therefore a Monte Carlo stopping diagnostic
conditional on a separate `dt` convergence and conditional-state physicality
study, not an unconditional coverage certificate. Quantum-jump paths likewise
require their deterministic no-jump integrator tolerances to be converged.

```julia
adaptive = adaptive_diffusive_trajectories(batch, rho0;
    min_trajectories=128, max_trajectories=20_000,
    batch_size=64, confidence=0.95, rtol=0.02,
    seed=7, threaded=true)
```

The quantum-jump variant takes a model or `TrajectoryPlan`, `times`, `dt`, and
an observable collection. Both routes preserve trajectory-index-derived
random streams, so serial and threaded runs of the same request use the same
samples at each checked batch boundary. Parallel accumulation can change the
last floating-point bits because worker partial sums are merged in a different
order; scientific reproducibility should therefore compare at the requested
statistical tolerance rather than assume bitwise identity.

The stopping certificate controls Monte Carlo sampling only. Converge the
quantum-jump integrator or diffusive Euler--Maruyama step separately, and do
not reuse sampling convergence as evidence for a hierarchy, finite-size, or
model approximation. Optional quantum-jump summaries can still retain pooled
waiting times; disable them when that jump-count-scaled storage is
unnecessary. Adaptive stopping currently covers density-valued PI jump paths,
weak-PI pseudo-kets, and collective diffusive paths, but not Distributed
workers.

The runnable
[`wiseman_milburn_homodyne.jl`](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/blob/main/examples/wiseman_milburn_homodyne.jl)
compares the conditional paths, their ensemble average, and the PI master
equation.

## API

```@docs
DiffusiveMonitor
homodyne_monitor
heterodyne_monitor
DiffusivePlan
DiffusiveWorkspace
DiffusiveBatchPlan
DiffusiveBatchWorkspace
DiffusiveTrajectory
diffusive_trajectory
diffusive_trajectories
diffusive_average
AdaptiveTrajectoryResult
adaptive_quantum_trajectories
adaptive_weak_pi_quantum_trajectories
adaptive_diffusive_trajectories
```
