# Weak-PI Schur pseudo-ket trajectories

The weak-PI trajectory backend is an opt-in pure-state unraveling inside the
direct sum of retained Schur irreps,

```math
\mathcal H_{\rm weak}=\bigoplus_\nu U_\nu,
\qquad
|\psi\rangle=\bigoplus_\nu|\psi_\nu\rangle.
```

Its dimension is $\sum_\nu\dim U_\nu$. This is generally much smaller than
both the PI density-coordinate dimension $\sum_\nu(\dim U_\nu)^2$ and the
full labeled-particle dimension $d^N$. Production code constructs neither a
$d^N$ ket nor a $d^N\times d^N$ operator.

For a genuine physical pure ket confined to the sole fully symmetric irrep,
use [`SymmetricKet`](@ref) instead. The distinctions are summarized in
[Symmetric pure kets and block-resolved
entropy](symmetric_kets_and_block_entropy.md).

The starting literature includes Yuan Zhang, Yu-Xiang Zhang, and Klaus
Mølmer,
[*Monte-Carlo simulations of superradiant lasing*,
New J. Phys. **20**, 112001
(2018)](https://doi.org/10.1088/1367-2630/aaec36), whose reduced-Dicke
quantum-jump construction resolves collective and local decay through
total-spin-sector changes. Related weak-symmetry work by Elliot W. Lloyd,
Aleksandra A. Ziolkowska, and Jonathan Keeling,
[*Permutation-symmetric quantum trajectories*,
arXiv:2605.11103 (2026)](https://arxiv.org/abs/2605.11103), which develops
weak-permutation-symmetric stochastic unravelings for emitters coupled to a
common system such as a cavity. The single-ensemble backend documented here
implements Schur-sector pseudo-ket trajectories; the distinction from the
shared-system cases in that work is stated under
[scope and convergence](#Scope-and-convergence).

## What the pseudo-ket means

A `WeakPIPseudoKet` maps to a physical PI density state through

```math
C_\nu=\frac{|\psi_\nu\rangle\langle\psi_\nu|}{\sqrt{f^\nu}},
\qquad
\rho_\nu=\frac{C_\nu}{\sqrt{f^\nu}}.
```

Consequently $\sum_\nu\langle\psi_\nu|\psi_\nu\rangle=1$ is the physical
trace condition. Relative phases between different $\nu$ sectors disappear
from every PI density operator and observable; they are not physical
coherences. A weak-PI pseudo-ket is therefore not a pure state of $N$
labeled particles. It is also distinct from the density-valued conditional
state returned by `quantum_trajectory`.

`weak_pi_pseudoket(rho)` accepts only states whose multiplicity-weighted block
$\sqrt{f^\nu}C_\nu$ has rank at most one in every occupied sector. A general
mixed state is rejected rather than purified or approximated silently.
`weak_pi_density(psi)` performs the reverse conversion.

## Sector-changing local Kraus branches

For a fixed one-particle local channel $L$, the PI gain map between input
sector $\nu$ and output sector $\lambda$ factorizes over common one-box
children $\mu\vdash N-1$:

```math
G_{\lambda\nu}(C_\nu)
=\sum_\mu s_{\lambda\mu\nu}
 A_{\lambda\mu\nu} C_\nu
 A_{\lambda\mu\nu}^\dagger.
```

The pseudo-density block is
$\bar\rho_\nu=\sqrt{f^\nu}C_\nu$. Its Kraus matrices are therefore

```math
K_{\lambda\mu\nu}
=\sqrt{s_{\lambda\mu\nu}
       \sqrt{\frac{f^\lambda}{f^\nu}}}\,
 A_{\lambda\mu\nu}.
```

`WeakPITrajectoryPlan` constructs these matrices directly from the cached
one-box Clebsch--Gordan contractions. It does not form and diagonalize a dense
Choi matrix. At setup it verifies, separately for every physical channel and
source sector,

```math
\sum_{\lambda,\mu}K_{\lambda\mu\nu}^\dagger
K_{\lambda\mu\nu}=Q_\nu,
```

where $Q_\nu$ is the prepared physical Schur block entering the no-jump
generator. This makes the ensemble gain exactly the same CP map as the
density-valued PI trajectory backend, up to floating-point setup error.

Collective, direct-PI, and collective $p$-body jump operators preserve the
Schur sector and contribute one Kraus branch per nonzero sector block. Fixed
one-body `LocalJump` channels may change sectors. The construction is the same
for qubits and qudits.

The Kraus decomposition defines one legitimate unresolved measurement record;
it does not reconstruct a particle label. Another Kraus rotation gives the
same master equation and all ensemble-linear observables but can change
individual paths and trajectory-level variances.

## Prepared workflow

```julia
using PermutationalInvariantDynamics
using Random

N = 6
basis = PIBasis(N, 2)
sm = ComplexF64[0 1; 0 0]
model = PIModel(basis, (
    CollectiveJump(sm; rate=0.2),
    LocalJump(sm; rate=0.8),
))

rho0 = iid_pure_state(basis, ComplexF64[0, 1])
psi0 = weak_pi_pseudoket(rho0)
plan = WeakPITrajectoryPlan(model)
batch = WeakPITrajectoryBatchWorkspace(plan, psi0)

times = collect(0.0:0.1:1.0)
paths = weak_pi_quantum_trajectories(
    plan, psi0, times, 500;
    dt=0.005, max_jump_probability=0.03,
    seed=2026, threaded=Threads.nthreads() > 1,
    workspace=batch,
)

summary = weak_pi_trajectory_statistics(
    paths;
    observables=(excitation=adjoint(sm) * sm,),
    nchannels=2,
)
rho_mean = summary.average_states[end]
```

Use continuous hazard roots by selecting the adaptive algorithm:

```julia
event_paths = weak_pi_quantum_trajectories(
    plan, psi0, times, 500;
    algorithm=:event, dt=0.02, dtmax=0.05,
    abstol=1e-9, reltol=1e-7,
    event_time_tolerance=1e-10,
    seed=2026, workspace=batch,
)
```

The aliases `:adaptive` and `:event_driven` select the same algorithm. An
embedded Dormand--Prince 5(4) trial advances the normalized pseudo-ket and
the integrated jump hazard with common stages. A crossing is located from
the unchanged step-start state, then the ordinary sector-resolved Kraus
branch is sampled at the physical event time. Event times are therefore not
restricted to step endpoints.

Plans are immutable and shareable. A `WeakPITrajectoryWorkspace` owns the
Runge--Kutta, intensity, and selected-branch scratch for one path. A
`WeakPITrajectoryBatchWorkspace` gives every worker independent scratch and an
independent RNG while retaining trajectory-index-derived seeds. Serial and
threaded batches are therefore reproducible for a fixed seed.

`WeakPITrajectoryWorkspace(mode=:full)` is the default and supports both
algorithms. Fixed propagation uses three full-vector RK4 registers.
`mode=:fixed` omits `k3`, `k4`, and the six adaptive-only vectors and rejects
an event-driven call rather than allocating scratch lazily. Batch workspaces
accept the same mode keyword.

Every no-jump stage first combines the rate-weighted channel loss blocks into
one effective Schur operator, requiring one matrix-vector action per sector.
Individual channel and Kraus-branch intensities are evaluated only after an
event must be selected. For event-driven paths, the accepted seven DOPRI
stages supply a quartic continuous extension. Four scalar hazard coefficients
are retained for bisection and the root pseudo-ket is reconstructed once from
the retained stages; root localization performs no additional RHS trials.

`WeakPIJumpRecord` stores the model channel, flattened Kraus branch, source
partition, target partition, and the $N-1$ child partition for a local
event. `weak_pi_trajectory_statistics` reports ordinary channel counts plus
the sampled Schur-sector transition counts. Observable statistics accept the
same Hermitian local matrices and PI operators as the density trajectory
tools. `weak_pi_expectation` evaluates one pseudo-ket directly.

## History-free stationary density estimate

For an autonomous model, `weak_pi_trajectory_steady_state` estimates the
initial-state-selected stationary density operator without retaining
pseudo-ket histories or jump records:

```julia
estimate = weak_pi_trajectory_steady_state(
    plan, psi0;
    trajectories=2_000,
    settling_time=50.0,
    samples_per_trajectory=11,
    sampling_interval=2.0,
    dt=0.002,
    max_jump_probability=0.03,
    seed=2026,
    threaded=Threads.nthreads() > 1,
    workspace=batch,
    observables=(excitation=adjoint(sm) * sm,),
    return_info=true,
)
rho_stationary = estimate.state
```

The estimator also accepts `algorithm=:event`. For a time-series error
diagnostic, request complete batches:

```julia
estimate = weak_pi_trajectory_steady_state(
    plan, psi0;
    trajectories=500,
    settling_time=50.0,
    samples_per_trajectory=40,
    sampling_interval=0.5,
    batch_size=10,
    dt=0.01,
    return_info=true,
)
batch_report = estimate.metadata.batch_means
batch_report.standard_error
batch_report.effective_independent_samples
```

`batch_size` requires `return_info=true`, because the diagnostic is returned
only through `estimate.metadata.batch_means`. The primary
`estimate.standard_error` continues to use independently seeded path means.
The optional report first averages `batch_size` consecutive time
samples and treats the complete batch means as approximately independent.
Increase the batch length and total window until the estimate stabilizes.
This does not certify burn-in or finite-window bias; those qualifications are
recorded explicitly in `batch_report.assumptions`.

The density conversion must precede every average. For independent path
$r$ and its $K$ selected post-settling times, the path mean is

```math
\overline C^{(r)}_\nu
=\frac{1}{K}\sum_{k=1}^K
  \frac{|\psi^{(r)}_\nu(t_k)\rangle
        \langle\psi^{(r)}_\nu(t_k)|}{\sqrt{f^\nu}},
\qquad
\widehat C_\nu=\frac{1}{M}\sum_{r=1}^M\overline C^{(r)}_\nu.
```

It would be incorrect to average pseudo-ket amplitudes before taking their
outer products: the density map is nonlinear, sector-relative phases are
unphysical, and the stationary estimate is generally mixed. The function
therefore returns a `PIState`, never a `WeakPIPseudoKet`.

Time samples from one realization can be autocorrelated. They are first
combined into $\overline C^{(r)}$, and only the $M$ independent path means
enter the uncertainty estimate. Because the equation-(7) PI coefficient basis
is Hilbert--Schmidt orthonormal,

```math
s_{\mathrm{HS}}
=\sqrt{\frac{1}{M-1}\sum_{r=1}^M
 \left\|\overline C^{(r)}-\widehat C\right\|_2^2},
\qquad
\mathrm{SE}_{\mathrm{HS}}=\frac{s_{\mathrm{HS}}}{\sqrt M}.
```

At least two independent paths are required. With `return_info=true`, the
shared `TrajectorySteadyStateResult` reports this sample spread and standard
error, optional Hermitian-observable statistics across the same path means,
the density Liouvillian residual, relative residual, and trace error. These
are diagnostics, not a `converged` claim. The averaged state is not silently
normalized, symmetrized, or positivity-repaired.

Its `metadata` identifies `backend=:weak_pi` and the selected algorithm,
records the effective worker and sampling controls, and exposes
`pseudo_ket_dimension` and `density_dimension`. The reduction and uncertainty
labels are respectively `:post_settling_density_mean` and
`:independent_path_mean`.

The state/density reduction is history-free: each worker retains one weak
pseudo-ket, one PI-coordinate path mean, and online accumulators,
independently of the number of paths and sampling times. Reproducible
trajectory-indexed streams retain one `UInt64` seed per path. Producing a full
density estimate still requires forming every retained Schur outer product at
each requested sample. Optional observables are reduced from the same path
density mean and therefore use the same independent-path sampling unit.

## Confidence-controlled weak-PI ensembles

Observable-only calculations can stop at deterministic batch boundaries once
all simultaneous empirical-Bernstein widths meet their target:

```julia
adaptive = adaptive_weak_pi_quantum_trajectories(
    plan, psi0, times;
    observables=(excitation=adjoint(sm) * sm,),
    algorithm=:event, dt=0.02,
    min_trajectories=128,
    max_trajectories=10_000,
    batch_size=64,
    confidence=0.95,
    atol=2e-3,
    rtol=1e-2,
    seed=2026,
)
```

Observables are contracted directly from pseudo-kets; no state history or
sampled PI density operator is retained. Each separately seeded trajectory
is one independent sample for the stopping certificate, reported by
`adaptive.metadata.effective_independent_samples`. The certificate controls
Monte Carlo dispersion only. It does not cover integrator, finite-time
initialization, finite-window, or model-truncation bias.

## Scope and convergence

The fixed weak-PI integrator is RK4 with a maximum jump probability; the event
integrator is adaptive Dormand--Prince with continuous hazard roots. Converge
the controls of the selected algorithm before interpreting Monte Carlo error
bars. Rates must evaluate to finite, nonnegative real values representable in
the prepared precision; time grids and controls may not silently narrow.
Hamiltonian rates must likewise be finite and real.

Fixed operator-valued collective/direct jumps and fixed one-body local jumps
are supported. Operator-valued schedules and `LocalPBodyJump` are rejected.
The composite backend can represent a finite cavity and density-valued
tensor-product quantum jumps, but a composite pseudo-ket trajectory compiler
is still separate future work.

The stationary estimator additionally requires an autonomous source and a
`WeakPIPseudoKet` initial state. A `PIState` can be converted with
`weak_pi_pseudoket` only when every occupied multiplicity-weighted block has
rank at most one; a general mixed state is not silently purified. Separately
refine the settling time, within-path sampling interval and window length,
the selected integration controls, and independent trajectory count. The
primary Monte Carlo uncertainty does not include burn-in, finite-window,
integration, or model-truncation bias. Optional batch means diagnose temporal
autocorrelation only under their explicit approximately-independent-batches
assumption. A nonunique stationary manifold can retain dependence on the initial state or
strong-symmetry component. Kraus rotations preserve the ensemble master
equation but can alter finite-sample variance.

The runnable
[`weak_pi_trajectories.jl`](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/blob/main/examples/weak_pi_trajectories.jl) example
uses the Zhang--Zhang--Mølmer decay model cited above.
For the paper's $\gamma_l/\Gamma_c=1$ case it compares four routes:
certified population dynamics, general matrix-free PI master evolution,
density-valued PI trajectories, and weak-PI pseudo-kets. The stochastic
backends use equal ensemble sizes and identical fixed-step controls. Lloyd,
Ziolkowska, and Keeling's related 2026 paper cited above provides the broader
common-system weak-symmetry context, but this example does not reproduce its
shared-cavity calculations.

Two optional CairoMakie figures show the fluxes and 95% confidence bands,
PI-state ensemble errors, sampled $J\rightarrow J'$ sector changes,
representation-size scaling, and a warmed per-path timing from the current
run. An equal-ensemble retained-history panel complements the asymptotic
coordinate counts; it excludes plans, worker scratch, and transient peak RAM.
For observable-only work, the density-valued backend can instead use
`save_states=false`; the ordinary weak-PI batch/statistics interface retains
paths before forming statistics. `weak_pi_trajectory_steady_state` is the
history-free weak-PI route when the full stationary density is required.
The timing is descriptive rather than a regression threshold. The scaling
panel uses the exact complete-qubit counts

```math
\sum_\nu\dim U_\nu
=\left(\left\lfloor N/2\right\rfloor+1\right)
 \left(\left\lceil N/2\right\rceil+1\right),
\qquad
\sum_\nu(\dim U_\nu)^2=\binom{N+3}{3},
```

alongside $2^N$ and $4^N$. These curves are evaluated as formulas; no
full-Hilbert state is constructed. The paired
[`examples/weak_pi_trajectories.md`](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/blob/main/examples/weak_pi_trajectories.md)
guide gives the numerical checks, timing scope, figure panels, and
interpretation boundaries in detail. Set `PI_EXAMPLE_PLOTS=0` for a
numerical-only compatibility or timing run that skips the optional Makie
import.
