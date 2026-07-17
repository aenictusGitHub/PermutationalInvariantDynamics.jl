# Weak-PI Schur pseudo-ket trajectories

The weak-PI trajectory backend is an opt-in pure-state unraveling inside the
direct sum of retained Schur irreps,

```math
\mathcal H_{\rm weak}=\bigoplus_\nu U_\nu,
\qquad
|\psi\rangle=\bigoplus_\nu|\psi_\nu\rangle.
```

Its dimension is ``\sum_\nu\dim U_\nu``. This is generally much smaller than
both the PI density-coordinate dimension ``\sum_\nu(\dim U_\nu)^2`` and the
full labeled-particle dimension ``d^N``. Production code constructs neither a
``d^N`` ket nor a ``d^N\times d^N`` operator.

## What the pseudo-ket means

A `WeakPIPseudoKet` maps to a physical PI density state through

```math
C_\nu=\frac{|\psi_\nu\rangle\langle\psi_\nu|}{\sqrt{f^\nu}},
\qquad
\rho_\nu=\frac{C_\nu}{\sqrt{f^\nu}}.
```

Consequently ``\sum_\nu\langle\psi_\nu|\psi_\nu\rangle=1`` is the physical
trace condition. Relative phases between different ``\nu`` sectors disappear
from every PI density operator and observable; they are not physical
coherences. A weak-PI pseudo-ket is therefore not a pure state of ``N``
labeled particles. It is also distinct from the density-valued conditional
state returned by `quantum_trajectory`.

`weak_pi_pseudoket(rho)` accepts only states whose multiplicity-weighted block
``\sqrt{f^\nu}C_\nu`` has rank at most one in every occupied sector. A general
mixed state is rejected rather than purified or approximated silently.
`weak_pi_density(psi)` performs the reverse conversion.

## Sector-changing local Kraus branches

For a fixed one-particle local channel ``L``, the PI gain map between input
sector ``\nu`` and output sector ``\lambda`` factorizes over common one-box
children ``\mu\vdash N-1``:

```math
G_{\lambda\nu}(C_\nu)
=\sum_\mu s_{\lambda\mu\nu}
 A_{\lambda\mu\nu} C_\nu
 A_{\lambda\mu\nu}^\dagger.
```

The pseudo-density block is
``\bar\rho_\nu=\sqrt{f^\nu}C_\nu``. Its Kraus matrices are therefore

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

where ``Q_\nu`` is the prepared physical Schur block entering the no-jump
generator. This makes the ensemble gain exactly the same CP map as the
density-valued PI trajectory backend, up to floating-point setup error.

Collective, direct-PI, and collective ``p``-body jump operators preserve the
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

Plans are immutable and shareable. A `WeakPITrajectoryWorkspace` owns the
Runge--Kutta, intensity, and selected-branch scratch for one path. A
`WeakPITrajectoryBatchWorkspace` gives every worker independent scratch and an
independent RNG while retaining trajectory-index-derived seeds. Serial and
threaded batches are therefore reproducible for a fixed seed.

`WeakPIJumpRecord` stores the model channel, flattened Kraus branch, source
partition, target partition, and the ``N-1`` child partition for a local
event. `weak_pi_trajectory_statistics` reports ordinary channel counts plus
the sampled Schur-sector transition counts. Observable statistics accept the
same Hermitian local matrices and PI operators as the density trajectory
tools. `weak_pi_expectation` evaluates one pseudo-ket directly.

## Scope and convergence

The current weak-PI integrator is fixed-step RK4 with a maximum jump
probability. Converge both `dt` and `max_jump_probability` before interpreting
Monte Carlo error bars. Rates must evaluate to finite, nonnegative real values
representable in the prepared precision; time grids and controls may not
silently narrow. Hamiltonian rates must likewise be finite and real.

Fixed operator-valued collective/direct jumps and fixed one-body local jumps
are supported. Operator-valued schedules and `LocalPBodyJump` are rejected.
The deterministic composite basis can represent a finite cavity, but a
composite pseudo-ket trajectory compiler is still separate future work.

The runnable
[`weak_pi_trajectories.jl`](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/blob/main/examples/weak_pi_trajectories.jl) example
uses the decay model of Zhang, Zhang, and Mølmer,
[*New J. Phys.* **20**, 112001 (2018)](https://doi.org/10.1088/1367-2630/aaec36).
For the paper's ``\gamma_l/\Gamma_c=1`` case it compares four routes:
certified population dynamics, general matrix-free PI master evolution,
density-valued PI trajectories, and weak-PI pseudo-kets. The stochastic
backends use equal ensemble sizes and identical fixed-step controls.

Two optional CairoMakie figures show the fluxes and 95% confidence bands,
PI-state ensemble errors, sampled ``J\rightarrow J'`` sector changes,
representation-size scaling, and a warmed per-path timing from the current
run. An equal-ensemble retained-history panel complements the asymptotic
coordinate counts; it excludes plans, worker scratch, and transient peak RAM.
For observable-only work, the density-valued backend can instead use
`save_states=false`; the current weak-PI batch interface retains paths before
forming statistics.
The timing is descriptive rather than a regression threshold. The scaling
panel uses the exact complete-qubit counts

```math
\sum_\nu\dim U_\nu
=\left(\left\lfloor N/2\right\rfloor+1\right)
 \left(\left\lceil N/2\right\rceil+1\right),
\qquad
\sum_\nu(\dim U_\nu)^2=\binom{N+3}{3},
```

alongside ``2^N`` and ``4^N``. These curves are evaluated as formulas; no
full-Hilbert state is constructed. The paired
[`examples/weak_pi_trajectories.md`](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/blob/main/examples/weak_pi_trajectories.md)
guide gives the numerical checks, timing scope, figure panels, and
interpretation boundaries in detail. Set `PI_EXAMPLE_PLOTS=0` for a
numerical-only compatibility or timing run that skips the optional Makie
import.
