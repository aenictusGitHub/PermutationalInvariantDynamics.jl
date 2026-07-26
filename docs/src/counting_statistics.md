# Full counting statistics and dynamical large deviations

The counting-statistics backend resolves selected physical jump channels of
an autonomous PI model without constructing a full-Hilbert-space state or a
dense PI Liouvillian.

## Convention

For channel counts $K_a(t)$ and signed increments $q_a$, the moment-generating
function is

```math
Z(s,t)=\mathbb E\!\left[
\exp\!\left(s\sum_a q_a K_a(t)\right)\right].
```

Only the jump gain is tilted:

```math
\mathcal L_s=\mathcal L+
\sum_a\left(e^{s q_a}-1\right)\mathcal J_a.
```

The anticommutator loss remains unchanged. Consequently,
$Z(s,t)=\mathrm{tr}[\exp(t\mathcal L_s)\rho(0)]$. The scaled
cumulant-generating function (SCGF) is the eigenvalue with largest real part,

```math
\theta(s)=\lim_{t\to\infty}\frac{\log Z(s,t)}{t}.
```

Its first two derivatives at zero are the stationary current and
zero-frequency noise. The implementation uses the real
moment-generating field $s$. A complex field can be used with
`tilted_liouvillian` and `finite_time_mgf` for characteristic functions, but
`counting_scgf` deliberately requires a real field so that largest-real-part
selection follows the large-deviation branch.

## Prepared workflow

```julia
using PermutationalInvariantDynamics

basis = PIBasis(8, 2)
spin = spin_matrices()
model = PIModel(basis, (
    LocalJump(spin.jm; rate=0.7),
    LocalJump(spin.jp; rate=0.3),
))

compiled = compile(model; backend=:matrixfree)

# Channel numbers match TrajectoryPlan and recorded jump histories.
counting = TiltedLiouvillianPlan(
    compiled;
    channels=[1],
    increments=[1],
)
workspace = TiltedLiouvillianWorkspace(counting)

rho0 = computational_product_state(basis, 2)
Z = finite_time_mgf(
    counting, rho0, 4.0, 0.1;
    workspace,
    krylovdim=40,
)

theta = counting_scgf(counting, 0.1; krylovdim=50)
transport = counting_cumulants(counting; krylovdim=50)
@show transport.current transport.noise transport.fano
```

A `LocalJump` is one unresolved PI channel: it counts jumps summed over all
identical particles. A collective channel is likewise one channel. Fixed
correlated reservoirs are counted after their prepared positive
factorization, matching the channel indices recorded by quantum
trajectories. Selecting the same channel twice is rejected; use a non-unit
increment to assign its charge.

Plans accept only autonomous, fixed-operator trajectory lowerings. Freeze a
driven model at an explicit time and parameter value before constructing the
plan. Counting also enforces finite, nonnegative rates, even though the
deterministic time-local solver permits negative rates.

## Matrix-free application

For a custom eigensolver, prepare the operator once:

```julia
Ltilted = tilted_liouvillian(counting, 0.1; workspace)
y = similar(rho0.data)
mul!(y, Ltilted, rho0.data)
```

The operator precomputes the factors `expm1(s*q)` and reuses one synchronized
workspace. Concurrent tasks should share `counting`, not `workspace`; give
each task a separate `TiltedLiouvillianWorkspace` and call `apply_tilted!`.
The adjoint action is also matrix-free.

## Large-deviation curve

```julia
curve = counting_scgf_curve(
    counting, range(-0.5, 0.5; length=41);
    krylovdim=60,
)

currents = range(0.0, 0.5; length=51)
rate = large_deviation_rate_function(curve, currents)
```

The rate function is the discrete Legendre--Fenchel estimate

```math
I(j)=\max_s\{s j-\theta(s)\}.
```

Inspect `rate.boundary_maxima`. A maximum on the edge of the supplied field
grid means that the continuous supremum has not been resolved; the routine
reports this rather than extrapolating or silently extending the grid.

## Reliability and convergence

- `finite_time_mgf` reports adaptive Krylov exponential-action diagnostics
  with `return_info=true`.
- `counting_scgf(...; return_info=true)` reports the dominant Ritz residual.
- `counting_cumulants` compares steps $h$ and $h/2$ and returns Richardson
  discrepancy estimates. Refine `step` and `krylovdim` independently.
- A nonzero imaginary part of the real-field SCGF is retained. Cumulant and
  Legendre routines reject it when it exceeds the Ritz-residual reliability
  scale.
- `memory_budget` guards model-to-counting preparation, Krylov and
  tilted-application workspaces, and the retained output of an SCGF field
  curve. The guard runs before a known-length field iterator is consumed.
- `finite_time_mgf(...; return_info=false)` returns only the scalar and does
  not copy the propagated PI vector into a state. With `return_info=true`, the
  additional detached tilted state is deliberately retained and included in
  the memory estimate.
- Products `s*q` and weights `expm1(s*q)` are checked for nonfinite values and
  underflow in the prepared precision before they enter the generator.

Finite-time trajectory counts offer an independent stochastic cross-check:
for a stored trajectory, evaluate
`exp(s * sum(q[channel] for channel in trajectory.jump_channels))` and
average this weight over independent paths. The deterministic tilted result
should lie within the reported Monte Carlo confidence interval after
time-step and ensemble-size convergence.

## API

```@docs
TiltedLiouvillianPlan
TiltedLiouvillianWorkspace
TiltedLiouvillian
apply_tilted!
apply_tilted_adjoint!
tilted_liouvillian
counting_scgf
counting_cumulants
counting_scgf_curve
large_deviation_rate_function
finite_time_mgf
```
