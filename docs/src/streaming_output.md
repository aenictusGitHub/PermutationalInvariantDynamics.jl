# Streaming and observable-only output

Fixed-step deterministic evolution and PI quantum trajectories can record
small analysis products while discarding sampled density-state histories. The
numerical evolution is unchanged: streaming controls only what is retained at
requested output times.

## Deterministic dynamics

Pass one or more named observables to `solve_dynamics`:

```julia
result = solve_dynamics(
    prepared, rho0, (0.0, 1.0);
    saveat=0.05,
    observables=(energy=H, polarization=sz),
    save_states=false,
)
energy = result.observables[:energy]
```

The observable collection may be a named tuple, dictionary, pair collection,
or a single local matrix/`PIOperator`. Local matrices represent collective
sums. With `save_states=false`, one mutable PI state is propagated and no
sampled state vector is copied into the result. Set `save_states=true` to
retain states and expectations in the same run.

## Trajectory ensembles

For trajectories, the corresponding call returns online Monte Carlo means,
unbiased variances, standard errors, and normal confidence intervals:

```julia
ensemble = quantum_trajectories(
    plan, rho0, times, 10_000;
    dt=0.002, seed=1, threaded=true,
    observables=(magnetization=Jz,),
    save_states=false,
)
magnetization = ensemble.observables.observables[:magnetization]
```

Each task owns its trajectory workspace, observable buffer, and Welford
accumulator. The accumulators are merged in worker order; no mutable numerical
scratch is shared between tasks. Random streams remain indexed by trajectory,
as in the history-returning API.

Jump summaries are enabled by default and retain pooled waiting times, giving
storage proportional to the number of recorded waiting intervals rather than
PI-state history size. Use `jump_statistics=false` when those data are not
needed. At least one observable is required when `save_states=false`; the
no-observable call deliberately keeps the legacy vector-of-trajectories return
type. In threaded runs pooled waiting samples have no trajectory-order
guarantee; their values and aggregate statistics remain tied to the
trajectory-indexed random streams. `confidence` selects the reported normal
confidence interval and must lie strictly between zero and one.

## Trajectory steady-state estimates

For an autonomous model, `trajectory_steady_state` estimates a stationary
density operator without retaining complete paths, saved state histories, or
jump records:

```julia
rho_mc = trajectory_steady_state(
    plan, rho0;
    trajectories=2_000,
    settling_time=50.0,
    dt=0.002,
    samples_per_trajectory=11,
    sampling_interval=2.0,
    seed=1,
    threaded=true,
)
```

Each independent realization first evolves through `settling_time`. The
selected post-settling states are averaged *within that path*, and those path
means are then averaged across trajectories. This ordering is important:
states separated by `sampling_interval` on one path are correlated and are
not counted as independent Monte Carlo realizations. The default
`samples_per_trajectory=1` uses only the state at the end of the settling
interval and does not require `sampling_interval`. At least two independent
paths are required.

By default the function returns the approximate `PIState`. Set
`return_info=true` to obtain a `TrajectorySteadyStateResult` containing the
state, the Hilbert--Schmidt sample spread and standard error across independent
path means, the Liouvillian residual and relative residual, and the trace
error. Named Hermitian `observables` additionally obtain path-level means,
unbiased variances, standard errors, and normal confidence intervals:

```julia
estimate = trajectory_steady_state(
    plan, rho0;
    trajectories=2_000,
    settling_time=50.0,
    dt=0.002,
    samples_per_trajectory=11,
    sampling_interval=2.0,
    observables=(magnetization=Jz,),
    confidence=0.95,
    return_info=true,
)
magnetization = estimate.observables.observables[:magnetization]
```

Observable statistics belong to the detailed result, so an `observables`
request requires `return_info=true` rather than computing a report that would
then be discarded.

This routine requires an autonomous fixed-operator `TrajectoryPlan`; a model
or compiled source is prepared into that form automatically. It deliberately
does not claim convergence. Increase `settling_time`, vary
`sampling_interval`, converge the fixed step or adaptive controls, and increase
`trajectories` separately. A small residual is a diagnostic of the averaged
state, not a proof of uniqueness or of equilibration. When the Liouvillian has
several stationary states, its ensemble mean can retain dependence on `rho0`.
Different valid unravelings of one fixed Lindbladian have the same ensemble
mean, although their path distributions, variances, and finite Monte Carlo
realizations can differ.

The direct-sum Schur pseudo-ket backend has the analogous history-free
`weak_pi_trajectory_steady_state`. It converts every sampled pseudo-ket to PI
density coordinates before forming within-path means; averaging pseudo-ket
amplitudes would give the wrong mixed state. See [Weak-PI Schur pseudo-ket
trajectories](weak_pi_trajectories.md#History-free-stationary-density-estimate)
for its formula, supported channels, and fixed-step restrictions.

The runnable
[`streaming_output.jl`](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/blob/main/examples/streaming_output.jl)
example compares both routes with an exactly soluble emission model.

## Result types

```@docs
DynamicsStreamResult
TrajectoryEnsembleResult
```
