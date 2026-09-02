# No-jump-resolvent iterative solvers

The no-jump solver family implements the construction of Beugnot, Gregory,
Robin, and Tilloy, *Well-conditioned iterative methods for large open quantum
systems*, [arXiv:2608.30860 (2026)](https://arxiv.org/abs/2608.30860), directly
in PI Schur coordinates. It provides five related calculations:

- a reusable sectorwise no-jump resolvent;
- stationary states from right-preconditioned GMRES or a CPTP fixed-point map;
- selected nonzero slow Liouvillian modes from nested shift-invert Arnoldi or
  complex-shift, trace-deflated inexact IRAM;
- paired left/right-mode conditioning and non-normality diagnostics; and
- first-order implicit-Euler dynamics for stiff, low-accuracy propagation.

These are specialized research solvers, not automatic replacements for
[`stationary_state`](@ref), [`liouvillian_spectrum`](@ref), or the higher-order
dynamics integrators. Their advantage is largest when repeated no-jump solves
are much cheaper than a factorization of the complete PI Liouvillian.
They are currently in the advanced reusable API tier and are imported from
`PermutationalInvariantDynamics`, not from the stable `Workflow` namespace.

## Applicable models

The prepared split accepts a `PIModel`, `CompiledPIModel`,
`SpecializedPIModel`, or `LiouvillianPlan` whose resulting generator is a
fixed GKSL generator with a Hermitian Hamiltonian and finite nonnegative jump
rates. Built-in local, collective, and Appendix-D local jump terms are
supported. Hermiticity is
checked again on the total prepared Schur-block Hamiltonian, so disabling a
term constructor's `check` keyword cannot attach the CPTP guarantees to an
invalid generator.

For a driven `PIModel`, select the instantaneous generator explicitly while
the physical terms are still available:

```julia
instantaneous_plan = NoJumpIterativePlan(
    driven_model; time=1.25, parameters=params, backend=:schur)
instantaneous_resolvent = NoJumpResolventPlan(
    driven_model; time=1.25, parameters=params)
```

The same `time` and `parameters` keywords are accepted by
`no_jump_iterative_steady_state`, `no_jump_iterative_liouvillian_spectrum`, and
`no_jump_iterative_implicit_euler` when their source is the original `PIModel`. Every
term is evaluated first and the resulting autonomous model is then lowered,
so the exact jump/no-jump split is retained. This answers only the stationary,
spectral, or resolvent question for the generator at that instant; it is not a
steady state of the driven evolution. If `no_jump_iterative_implicit_euler` receives these
keywords, every step likewise uses that one frozen instantaneous generator;
it does not follow the original time-dependent schedule. Do not pass
`freeze(model; ...)` here:
that general helper intentionally returns only a linear operator and no longer
contains the physical split required by this method. Evaluation keywords are
rejected for autonomous models and already prepared sources rather than being
silently ignored.

The following cases are deliberately rejected:

- negative deterministic rates, because the CPTP and contraction statements
  no longer apply;
- opaque compiled kernels that do not expose their physical jump/loss split;
- composite, global-pseudomode, HEOM, and HOPS sources; and
- a zero-shift solve when strict stability of the no-jump generator cannot be
  certified.

Steady-state uniqueness is an assumption, not a result of this solver. Use
[`evans_uniqueness`](@ref) separately when its hypotheses apply, and always
inspect the returned physical residual and state diagnostics.

## PI form of the no-jump resolvent

Write the Lindbladian as

```math
\mathcal L=\mathcal S+\mathcal K,
\qquad
\mathcal S(X)=GX+XG^\dagger,
\qquad
G=-\mathrm i H-\frac12\sum_j L_j^\dagger L_j,
```

where $\mathcal K(X)=\sum_j L_j X L_j^\dagger$ contains the jump gains. A PI
Hilbert-space operator has one physical block in every retained Schur sector,
so

```math
\mathcal S_\nu(X_\nu)=G_\nu X_\nu+X_\nu G_\nu^\dagger.
```

Consequently the shifted inverse

```math
R_\lambda^{\mathcal S}=(\lambda-\mathcal S)^{-1}
```

is a collection of independent sector-sized Sylvester solves. Neither a
$d^N$ Hilbert-space object nor a full PI-coordinate inverse is formed. If the
Schur-irrep dimensions are $r_\nu$, preparation and repeated dense-block work
scale with sums of $r_\nu^3$, while retained block storage scales with sums of
$r_\nu^2$.

[`NoJumpResolventPlan`](@ref) prepares these factorizations. The default
`backend=:schur` uses unitary Schur factors and a preallocated
Bartels--Stewart recurrence; it remains valid for defective $G_\nu$ blocks.
`backend=:eigen` follows the paper's four-matrix-multiplication route and may
be faster on suitable hardware, but it rejects singular or excessively
ill-conditioned eigenvector matrices.

```julia
using PermutationalInvariantDynamics

basis = PIBasis(8, 2)
spin = spin_matrices(2)
model = PIModel(basis, (
    LocalHamiltonian(spin.jx; rate=0.25),
    LocalJump(spin.jm; rate=0.7),
    LocalJump(spin.jp; rate=0.2),
))

resolvent_plan = NoJumpResolventPlan(model; backend=:schur)
resolvent_work = NoJumpResolventWorkspace(resolvent_plan)
source = maximally_mixed_state(basis)
shifted = no_jump_resolvent(
    resolvent_plan, source; shift=0.4, workspace=resolvent_work)
```

Use [`no_jump_resolvent!`](@ref) with separate source and destination vectors
inside a hot loop. Source and destination may not alias.

## One prepared solver family

[`NoJumpIterativePlan`](@ref) combines the no-jump factorization, the exact matrix-free
PI Liouvillian, the physical trace functional, and a trace-one identity
direction. [`NoJumpIterativeWorkspace`](@ref) owns all mutable sector, Liouvillian, and
recycled-GMRES scratch.

```julia
plan = NoJumpIterativePlan(model; backend=:schur)
workspace = NoJumpIterativeWorkspace(plan; krylovdim=40, recycle_dim=8)
```

Plans are immutable and may be shared. A workspace retains warm starts and
recycled directions and must belong to only one task at a time.
`plan.metadata.generator_mode` is `:autonomous` for an originally autonomous
source and `:frozen_instantaneous` for an explicitly sampled driven model.

### Stationary state by right-preconditioned GMRES

The linear route solves a trace-deflated equation. This implementation uses
$v=I/D$, with $\mathrm{Tr}(v)=1$, and applies

```math
A_{\lambda,\delta}
=\lambda-\mathcal L-\delta\lvert v\rangle\langle\mathrm{Tr}\rvert.
```

The paper instead writes the unnormalized identity. Rescaling its deflation
coefficient by $D$ gives the convention above, while keeping the deflated
stationary eigenvalue independent of the Hilbert dimension. The rank-one term
is included exactly in the no-jump right preconditioner through a
Sherman--Morrison correction.

```julia
steady = no_jump_iterative_steady_state(
    plan;
    method=:gmres,
    workspace,
    deflation=1.0,
    maxiter=500,
    atol=1e-10,
    rtol=1e-8,
    return_info=true,
)

rho_ss = steady.state
steady.physical_residual_inf
steady.state_diagnostics.valid
```

The returned convergence decision includes a fresh residual of the original,
undeflated $\mathcal L$. A small transformed or preconditioned residual alone
is never accepted as physical convergence.

### Stationary state from the CPTP fixed-point map

When $G$ is strictly stable, define

```math
\Phi=\mathcal K R_0^{\mathcal S}
=I+\mathcal L R_0^{\mathcal S}.
```

This map is CPTP, and its fixed point reconstructs the stationary density
operator. Other eigenvalues may also lie on the unit circle, so the
implementation uses restarted Arnoldi and selects the Ritz value nearest one;
it does not use a power iteration.

The matrix-free action applies the prepared channel gain map
$\mathcal K R_0^{\mathcal S}$ directly. It does not evaluate the algebraically
equivalent $I+\mathcal L R_0^{\mathcal S}$, avoiding cancellation between the
identity and no-jump loss terms. `plan.metadata.fixed_point_action` and the
returned solver diagnostics report `:direct_gain`. The `:cptp_fixed_point`
guarantee is present only when setup certified that every no-jump Schur block
is strictly stable; dark-state plans retain only the positive-shift
contraction guarantee.

```julia
fixed = no_jump_iterative_steady_state(
    plan;
    method=:fixed_point,
    krylovdim=40,
    maxrestarts=20,
    return_info=true,
)
```

Both stationary routes validate trace, Hermiticity, positivity, and the true
Liouvillian residual. They do not symmetrize, clip, or repair an invalid
candidate.

### General shifted and deflated solves

[`no_jump_iterative_resolvent`](@ref) solves $A_{\lambda,\delta}x=b$ with
right-preconditioned recycled GMRES:

```julia
rhs = copy(source.data)
response = no_jump_iterative_resolvent(
    plan,
    rhs;
    shift=0.2,
    deflation=0.0,
    workspace,
    reuse=false,
)

x = response.solution
response.residual_inf
response.physical_residual_inf
```

Set `reuse=true` only for an intentional sequence of nearby systems. For every
positive shift, the paper proves that $\mathcal K R_\lambda^{\mathcal S}$ is a
strict contraction in the induced trace norm. At zero shift that contraction
becomes the CPTP map $\Phi$, so the same geometric convergence guarantee does
not apply.

Nested and reused no-jump-resolvent iterative solves request harmonic-Ritz
recycle extraction near the zero target. The `recycle_extraction` and `recycle_extraction_used`
diagnostics distinguish the request from the exact Rayleigh--Ritz fallback
used after an invariant Arnoldi breakdown.

### Nonzero slow modes

[`no_jump_iterative_liouvillian_spectrum`](@ref) applies thick-restarted Arnoldi to the
inverse of the trace-deflated shifted generator. Every outer action is an
inner recycled-GMRES solve. Transformed Ritz values $\nu$ are mapped back by

```math
\lambda=\mu-\frac1\nu.
```

```julia
modes = no_jump_iterative_liouvillian_spectrum(
    plan;
    nev=3,
    shift=0.0,
    deflation=1.0,
    krylovdim=48,
    inner_krylovdim=36,
    inner_atol=1e-11,
    inner_rtol=1e-9,
    candidate_oversampling=4,
    vectors=true,
)
```

The result contains only modes whose magnitude exceeds
`zero_exclusion_tolerance` and that pass fresh residual and tracelessness
checks against the original Liouvillian. This explicit zero test also removes
traceless stationary directions when the steady state is nonunique. Tighten
the inner tolerances before interpreting an unconverged outer calculation.
The default `candidate_oversampling=nothing` requests an automatic extra window
of `max(2, nev)` Ritz candidates. Increase it, together with `krylovdim`, when
several zero or rejected candidates occupy that window. Selected modes do not
certify a global Liouvillian gap or stationary-state multiplicity.

### Complex-shift inexact IRAM

For interior modes near a real or complex target, prepare a
[`TraceDeflatedShiftInvertPlan`](@ref). It applies

```math
\left[
\sigma-\mathcal L
-\delta\lvert I/D\rangle\langle\mathrm{Tr}\rvert
\right]^{-1}
```

without materializing that inverse. Here `sigma` is the requested shift and
`delta > 0` leaves every traceless nonzero mode unchanged while displacing the
traceful zero branch. The trace-one vector `I/D` is a numerical deflator; it
need not be the physical stationary state. Each transformed
operator action is an inexact, right-preconditioned GMRES solve using the
sectorwise no-jump resolvent. The outer eigensolver is a true implicit-QR
implicitly restarted Arnoldi method (IRAM); it is not a Krylov--Schur
implementation. The explicit `outer_restart=:iram` selector records that
contract and currently rejects every other value.

```julia
spectral_plan = TraceDeflatedShiftInvertPlan(
    plan; shift=-0.05 + 0.2im, deflation=1.0)
spectral_work = TraceDeflatedShiftInvertWorkspace(
    spectral_plan;
    outer_krylovdim=48,
    inner_krylovdim=36,
    inner_recycle_dim=0,
)

modes = trace_deflated_shiftinvert_spectrum(
    spectral_plan;
    nev=3,
    workspace=spectral_work,
    krylovdim=48,
    outer_restart=:iram,
    inner_rtol=1e-9,
    inner_initial_rtol=1e-3,
    adaptive_inner=true,
    mode_diagnostics=true,
    vectors=true,
)
```

The adaptive inner solve begins at the explicitly reported loose tolerance
and tightens monotonically after outer restarts. Its forcing signal is a fresh
residual for the original Liouvillian, rather than the transformed Ritz
residual. Inspect `inner_tolerance_history`, `inner_iterations`, and
`maximum_inner_residual_ratio` when tuning the nested calculation; the latter
is the worst achieved inner residual divided by the tolerance requested for
that solve. Set
`adaptive_inner=false` to use the final `inner_atol` and `inner_rtol` from the
first solve. The default `inner_recycle_dim=0` avoids treating different
Arnoldi right-hand sides as one linear sequence. Enabling recycling is an
explicit heuristic and requires both a positive `inner_recycle_dim` and
`reuse_inner=true`. When a prepared workspace is supplied, leaving either
inner-capacity keyword as `nothing` uses that workspace's fixed capacity.

Every returned eigenpair is re-evaluated against the original, undeflated
Liouvillian and must pass the physical residual and trace-null tests. Thus
inner-GMRES convergence and transformed Ritz convergence are candidate
criteria, not physical certificates. A complex shift does not inherit the
positive-real contraction guarantee of [`no_jump_resolvent`](@ref), and this
selected interior spectrum does not certify a global gap or uniqueness.

### Left modes and non-normality

With `mode_diagnostics=true`, the same inexact IRAM framework is applied to a
separately deflated adjoint problem. After computing and validating a physical
stationary state `rho_ss`, the left solve uses

```math
\left[
\overline{\sigma}-\mathcal L^\dagger
-\delta\lvert\mathrm{Tr}\rangle\langle\rho_{\mathrm{ss}}\rvert
\right]^{-1}.
```

This is intentionally not the literal adjoint of the forward numerical
deflator: the stationary-state functional preserves every nonzero left mode,
because such modes are orthogonal to `rho_ss`. The implementation globally
matches adjoint eigenvalues to conjugated right eigenvalues, then reports
paired left and right modes. For a simple eigenvalue, the scale-invariant
eigenvalue condition number is

```math
\kappa_\lambda
=\frac{\lVert l\rVert_2\lVert r\rVert_2}
       {\left\lvert l^\dagger r\right\rvert}.
```

After pairing, returned vectors are normalized with
`norm(right_vectors[:,j]) == 1` and, when the overlap is numerically
resolvable, `dot(left_vectors[:,j], right_vectors[:,j]) == 1`. Large
`condition_numbers` or small `reciprocal_condition_numbers` indicate sensitive
eigenvalues and non-normal dynamics. Inspect the separately recomputed
`right_residuals`, `left_residuals`, `trace_overlaps`, and
`stationary_overlaps` in `mode_diagnostics` before interpreting them.

Near-degenerate modes are additionally grouped by numerical eigenvalue
proximity. Each cluster reports singular values and a spectral-projector
conditioning estimate from its left/right subspace overlap. The routine does
not silently rotate clustered vectors, and neither a small overlap nor a
selected iterative spectrum proves that the full Liouvillian is defective.
`diagnostics_complete` requires successful eigenvalue pairing, resolvable
per-mode normalization, and full-rank cluster overlaps. Off-diagonal
biorthogonality is reported separately by `biorthogonality_error` rather than
being silently folded into that Boolean.
Use [`biorthogonal_mode_diagnostics`](@ref) directly when independently
computed right and adjoint spectra are already available. For broader
frequency-dependent non-normality probes, see [`resolvent_norm`](@ref) and
[`pseudospectral_abscissa`](@ref).

### Implicit Euler

One implicit step is

```math
\rho_{n+1}=(I-\Delta t\,\mathcal L)^{-1}\rho_n.
```

It is evaluated as a positive-shift resolvent with
$\lambda=1/\Delta t$. Use [`no_jump_iterative_implicit_euler_step!`](@ref) for a
preallocated step, or [`no_jump_iterative_implicit_euler`](@ref) for a strictly increasing
saved-time grid:

```julia
rho0 = iid_pure_state(basis, ComplexF64[1, 0])
evolution = no_jump_iterative_implicit_euler(
    plan, rho0, 0.0:0.02:0.2; workspace, atol=1e-10, rtol=1e-8)
```

This is a first-order stiff integrator. The positive-shift conditioning does
not remove time-discretization error; repeat the calculation with a smaller
step before interpreting physical results. The returned `generator_mode`
field makes an explicitly frozen driven source distinguishable from an
originally autonomous model.

## Dark states and zero-shift singularity

If $G$ has an eigenvalue on the imaginary axis, $R_0^{\mathcal S}$ is
singular. Under the theorem's unique-stationary-state assumption this is the
dark-state branch: the stationary state is a pure common dark eigenstate of
the jump operators. The zero-shift fixed-point and preconditioned stationary
routes therefore raise with guidance instead of regularizing the inverse.

A positive-shift no-jump resolvent remains defined. For the stationary state,
use a dark-state construction when it is known analytically, or an ordinary
[`stationary_state`](@ref) method that does not require $R_0^{\mathcal S}$.

## Resource and reliability checklist

- Plan and workspace constructors enforce the common 512 MiB default memory
  budget. `memory_budget=Inf` is the explicit opt-out.
- Reuse one immutable plan across calculations, and allocate one mutable
  workspace per concurrent task.
- Prefer `backend=:schur` unless the eigenvector conditioning reported in plan
  metadata has been checked.
- Inspect `physical_residual_inf`, `trace_error`, and state diagnostics for a
  stationary state; inspect original-Liouvillian residuals for slow modes.
- Refine both inner and outer Krylov controls for nested shift-invert spectra;
  for inexact IRAM, also inspect the recorded inner-tolerance schedule.
- Treat large paired-mode condition numbers as sensitivity diagnostics, not
  by themselves as proof of a Jordan block or defective generator.
- Refine the saved-time grid for implicit Euler.
- Treat `unique_steady_state=:assumed_not_certified` literally.

`unique_steady_state=:not_applicable` on a bare no-jump resolvent or an
implicit-Euler result means that no stationary-state claim was made.
`unique_steady_state=:assumed_not_certified` on a prepared no-jump-resolvent
iterative stationary or spectral calculation means that uniqueness is required by the selected
route but was not established by it.
