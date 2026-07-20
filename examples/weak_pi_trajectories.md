# Weak-PI pseudo-ket trajectories

Source: [`weak_pi_trajectories.jl`](weak_pi_trajectories.jl)

## Literature model

Zhang, Zhang, and Mølmer,
[*New J. Phys.* **20**, 112001 (2018)](https://doi.org/10.1088/1367-2630/aaec36),
studied collective cavity decay together with independent free-space decay,

```math
\dot\rho=\Gamma_c\mathcal D[J_-]\rho
+\gamma_l\sum_{i=1}^N\mathcal D[\sigma_-^{(i)}]\rho.
```

Their Dicke pseudo-state algorithm resolves an individual local event into
allowed total-spin shifts. This example tests the package's corresponding
opt-in weak-PI backend for the article's ``\gamma_l/\Gamma_c=1`` case. It
does not assign a label to the emitting particle. Instead, it factorizes the
already permutation-averaged PI gain into a complete set of Schur-sector
Kraus branches.

The two recorded fluxes are

```math
I_c=\Gamma_c\langle J_+J_-\rangle,
\qquad
I_{\mathrm{fs}}=\gamma_l\left(N/2+\langle J_z\rangle\right).
```

The default ``N=6`` and 400 paths per unraveling make this a fast backend
regression. The paper used ``N=50`` and 512 pseudo-state paths, so this is not
a digitization of its figure.

## Four solution routes

The script compares four calculations of the same ensemble dynamics:

1. `solve_populations` evolves the certified Schur/GT-diagonal population
   generator. This specialization is valid because decay from the fully
   excited state preserves the diagonal subspace.
2. `solve_dynamics` evolves the general matrix-free PI density state and
   agrees with the population result to numerical integration precision.
3. `quantum_trajectories` records density-valued PI paths. An unresolved
   local event combines all local Kraus outcomes and can make a conditional
   path mixed.
4. `weak_pi_quantum_trajectories` samples one Schur Kraus branch and retains
   a direct-sum Schur-irrep pseudo-ket.

```julia
rho0 = iid_pure_state(basis, ComplexF64[0, 1])
psi0 = weak_pi_pseudoket(rho0)

prepared = compile(model; backend=:matrixfree)
weak_plan = WeakPITrajectoryPlan(prepared)
density_plan = TrajectoryPlan(prepared)

weak_paths = weak_pi_quantum_trajectories(
    weak_plan, psi0, times, 400;
    dt=0.005, max_jump_probability=0.025,
    seed=2026, threaded=Threads.nthreads() > 1,
)
```

Both stochastic batches use the same output grid, number of paths, threading
choice, fixed-step size, and maximum jump probability. This makes the printed
per-path wall times a useful local comparison of the prepared backends. The
timing is deliberately not an assertion: it depends on hardware, thread
count, jump history, and Julia version. One path is run first to compile the
typed kernels, and the measurement excludes plan construction, statistics,
and plotting. Both backends support adaptive continuous-event trajectories;
that different integrator is not used in the timing panel. The retained-
history comparison uses `Base.summarysize` on the two
equal-size returned batches; it includes saved state wrappers and jump records
but excludes prepared plans, reusable worker scratch, and transient peak RAM.
It is a like-for-like history comparison, not the minimum possible output
memory: density-valued trajectories also support observable-only
`save_states=false` streaming, whereas the current weak-PI ensemble API
retains its pseudo-ket paths before postprocessing.

## Accuracy and sector checks

The example checks the exact initial values ``I_c(0)=N\Gamma_c`` and
``I_{\mathrm{fs}}(0)=N\gamma_l``, conservation of the population sum, and
agreement between the two deterministic PI solvers. Both stochastic fluxes
must lie within six reported standard errors plus a small integration floor
of the deterministic reference, and their averaged PI density states are
compared directly with the full PI solution.

`weak_pi_trajectory_statistics` converts pseudo-ket outer products to PI
density blocks only while forming ensemble averages. It also records every
``J\rightarrow J'`` transition. The script verifies that collective-channel
jumps preserve ``J`` and that every sector-changing record belongs to the
local channel.

## Representation comparison

For a complete qubit basis, the coordinate counts are

```math
P_{\mathrm{weak}}
=\sum_\nu\dim U_\nu
=\left(\left\lfloor N/2\right\rfloor+1\right)
 \left(\left\lceil N/2\right\rceil+1\right),
```

```math
P_{\mathrm{PI\ density}}
=\sum_\nu(\dim U_\nu)^2
=\binom{N+3}{3}.
```

A labeled-particle ket and density matrix instead contain ``2^N`` and
``4^N`` complex coordinates. At the default size these counts are:

| representation | coordinates |
|---|---:|
| weak-PI pseudo-ket | 16 |
| certified populations in this diagonal model | 16 |
| PI density state | 84 |
| full labeled-particle ket | 64 |
| full density matrix | 4096 |

The PI density vector can be longer than a full ket at very small ``N``
because the two objects solve different tasks: one stores an operator algebra,
the other one pure labeled state. Its comparison at equal task is with the
full density matrix; the weak pseudo-ket is the compressed trajectory-state
comparison with a full ket.

At the article's ``N=50``, the weak and PI density counts are 676 and 23,426,
whereas the labeled ket already has ``2^{50}`` amplitudes. These are snapshot
payload counts, not total plan or workspace memory. In particular, prepared
Kraus maps and integration scratch also consume memory, returned histories
multiply the payload by the number of paths and saved times, and the
population backend is available only after its invariance certificate passes.
The scaling curves are evaluated from these formulas; the example never
constructs a full-Hilbert ket or density matrix.

## Figures

With CairoMakie available, the first figure contains:

- cavity and free-space fluxes with 95% confidence bands for both
  unravelings and the population master-equation curve;
- the PI coefficient-vector error of both ensemble averages, together with
  the deterministic population/full-PI discrepancy;
- a heatmap of sampled source-``J`` to target-``J'`` local sector changes per
  trajectory.

It is saved as `weak_pi_trajectories_zhang_molmer.{pdf,png}`. A second figure,
`weak_pi_trajectories_method_comparison.{pdf,png}`, shows exact coordinate
scaling through ``N=50`` and the warmed per-path wall time measured for the
two fixed-step trajectory backends on the current run. Its third panel shows
the retained memory of equal path ensembles with the same saved times.

```sh
julia --project=examples examples/weak_pi_trajectories.jl
```

Running from the root environment still executes every numerical assertion
when CairoMakie is unavailable; only rendering is skipped. Converge `dt`,
`max_jump_probability`, and the number of trajectories before using the
curves as research data. Monte Carlo errors decrease only as the inverse
square root of the number of paths.

Set `PI_EXAMPLE_PLOTS=0` to skip even the optional Makie import during a
numerical-only compatibility or timing run:

```sh
PI_EXAMPLE_PLOTS=0 julia --project=. examples/weak_pi_trajectories.jl
```

## Event-driven, confidence-controlled, and stationary paths

The executable also exercises the advanced weak-PI routes on small batches.
Continuous event times use adaptive Dormand--Prince propagation of the
normalized pseudo-ket and accumulated hazard:

```julia
adaptive = adaptive_weak_pi_quantum_trajectories(
    weak_plan, psi0, times;
    observables, algorithm=:event, dt=0.02, dtmax=0.05,
    min_trajectories=16, max_trajectories=32, batch_size=8,
    atol=0.75, rtol=0, seed=3026)
```

Observables are contracted directly from pseudo-kets; no path or PI-density
history is retained. The simultaneous empirical-Bernstein rule controls
Monte Carlo dispersion only. `adaptive.converged=false` at the configured
maximum is a valid result and means that more independent paths are required.
Integration and finite-time biases remain separate.

For an autonomous model, a density-valued stationary estimate can be streamed
from weak-PI paths:

```julia
stationary = weak_pi_trajectory_steady_state(
    weak_plan, psi0;
    trajectories=8, settling_time=5.0,
    samples_per_trajectory=8, sampling_interval=0.1, batch_size=4,
    algorithm=:event, dt=0.02, dtmax=0.05,
    return_info=true)
```

Each path first averages its post-settling density reconstructions; the
primary standard error is then computed across the eight independent path
means. `metadata.batch_means` separately treats the sixteen complete temporal
batches as approximately independent and reports an autocorrelation-aware
diagnostic. Its assumptions explicitly leave burn-in, finite-window, and
integration bias uncontrolled. Batch-length and total-window refinements are
therefore required before interpreting it as a stationary error bar.

## Interpretation boundary

A `WeakPIPseudoKet` is not a pure wavefunction of labeled atoms. Its sector
block represents

```math
C_\nu=|\psi_\nu\rangle\langle\psi_\nu|/\sqrt{f^\nu},
```

and relative phases between different sectors are discarded by the PI
algebra. The selected subduction Kraus decomposition is one valid unresolved
measurement record. Kraus rotations preserve the master equation and all
ensemble-linear fluxes while changing individual pseudo-paths and possibly
their higher trajectory statistics. The example therefore validates ensemble
quantities and sector-selection metadata; it does not claim path-by-path
identity with the article, the density-valued unraveling, or a labeled-emitter
trajectory.
