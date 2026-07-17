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
DiffusiveTrajectory
diffusive_trajectory
diffusive_trajectories
diffusive_average
```
