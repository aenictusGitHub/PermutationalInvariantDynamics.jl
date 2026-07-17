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

The runnable
[`streaming_output.jl`](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/blob/main/examples/streaming_output.jl)
example compares both routes with an exactly soluble emission model.

## Result types

```@docs
DynamicsStreamResult
TrajectoryEnsembleResult
```
