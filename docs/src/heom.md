# Permutationally invariant HEOM

The hierarchy equations of motion (HEOM) backend adds finite-memory bosonic
environments to an otherwise time-local PI system model. Every auxiliary
density operator (ADO) is stored in the same PI Schur coordinates as the
physical density operator. Consequently, the method enlarges the calculation
by the hierarchy size, but it never reconstructs a matrix of dimension
`d^N`.

The backend is intended for bath correlations represented by finite sums of
exponentials. [`HEOMBath`](@ref) remains the exact low-level convention.
Validated physical constructors additionally prepare Drude--Lorentz
Matsubara/Padé and underdamped-Brownian Matsubara expansions. Their truncation
and white-noise residue are explicit metadata; the library never silently
changes a user-supplied decomposition.

For a stochastic pure-state hierarchy of a *shared* Gaussian bath, see
[Permutationally invariant HOPS](hops.md). PI--HOPS replaces every density
ADO by a direct-sum Schur pseudo-ket and then averages independent
unnormalized root outer products. It has a different pathwise symmetry
requirement and statistical convergence limit; it is not an automatic
drop-in replacement for independent local baths.

## Convention

For bath `b`, the coefficients supplied to [`HEOMBath`](@ref) define the
left correlation

```math
C_b^L(t)=\sum_{k\in b}\ell_k e^{-\nu_k t},\qquad t\geq0.
```

The physical conjugate correlation is

```math
C_b^R(t)=C_b^L(t)^*.
```

Both are stored on one prepared pole list,
`C_b^R(t)=sum_k r_k exp(-nu_k*t)`. Real poles receive
`r_k=conj(ell_k)`. Exact complex-conjugate pole pairs are cross-paired. If a
complex pole has no exact conjugate partner, the constructor appends its
conjugate with zero left coefficient and the required right coefficient.
Thus a single input complex exponential normally becomes two prepared HEOM
poles while leaving `C_b^L` unchanged. This completion is exact rather than
tolerance based.

The advanced `right_coefficients` keyword supplies `r_k` explicitly on the
original pole list and disables completion. It is useful for already prepared
left/right decompositions, but the library then does not certify
`C_b^R=C_b^{L*}`; an inconsistent choice can destroy Hermiticity of the root
ADO and is never repaired.

The coupling `Q_b` is a fixed Hermitian PI operator. All coefficients and
`nu_k` may be complex, but `Re(nu_k)` must be strictly positive. Flatten all
prepared exponentials from all baths into one index `k`, and let `b(k)`
identify their bath. The conventional unscaled hierarchy implemented by
[`HEOMPlan`](@ref) is

```math
\begin{aligned}
\dot\rho_{\boldsymbol n}={}&
\mathcal L_S\rho_{\boldsymbol n}
-\sum_k n_k\nu_k\rho_{\boldsymbol n}\\
&-i\sum_k[Q_{b(k)},\rho_{\boldsymbol n+\boldsymbol e_k}]\\
&-i\sum_k n_k\left(
 \ell_k Q_{b(k)}\rho_{\boldsymbol n-\boldsymbol e_k}
 -r_k\rho_{\boldsymbol n-\boldsymbol e_k}Q_{b(k)}
\right).
\end{aligned}
```

`L_S` can itself contain Hamiltonian and Markovian dissipative terms. The root
ADO `rho_0` is the physical reduced state. Auxiliary ADOs generally have
neither unit trace, Hermiticity, nor positivity and must not be interpreted as
density matrices.

### Exact scaled coordinates

The default `scaling=:unscaled` preserves those coordinates. The opt-in
`scaling=:scaled` route stores

```math
\widehat\rho_{\boldsymbol n}=\rho_{\boldsymbol n}/s_{\boldsymbol n},
\qquad
s_{\boldsymbol n}=\prod_k\sqrt{n_k!\,a_k^{n_k}}.
```

The positive per-pole factor defaults to
`a_k=max(abs(ell_k),abs(r_k))`, with exactly one used when both coefficients
vanish. An explicit `scaling_factors` vector may instead be supplied; every
entry must be positive, finite, and exactly representable in the plan's real
scalar type. The transformed upward and downward factors are respectively

```math
\sqrt{(n_k+1)a_k},\qquad \sqrt{n_k/a_k}.
```

Thus the scaled generator is exactly `S^(-1) L_HEOM S`, not a truncation or
changed bath model. The root scale is one, and [`heom_ado`](@ref) multiplies
an auxiliary coordinate by [`heom_coordinate_scale`](@ref) before returning
the conventional unscaled ADO. At finite precision the constructor also
requires each standalone `s_n` needed for public ADO extraction to remain
positive and finite; it raises with depth/precision guidance rather than
overflowing silently. `scaling=:unscaled` remains fully compatible with the
original route.

The hierarchy retains `sum(n_k) <= D`, where `D=max_depth`. Upward ADOs beyond
that boundary are set to zero. This is the `terminator=:none` hard truncation.
With `terminator=:residue`, each bath's explicitly stored real discrepancy
`delta_b` adds the time-local correction

```math
2\delta_b\mathcal D[Q_b]\rho=-\delta_b[Q_b,[Q_b,\rho]]
```

to every ADO. An arbitrary [`HEOMBath`](@ref) defaults to zero residue; none is
inferred from its poles. Physical constructors calculate the documented
zero-frequency discrepancy. The correction is matrix-free, and the exact
adjoint uses its self-adjoint double-commutator form. A finite-depth and
finite-pole convergence study remains necessary even when the residue is used.

An optional positive `importance_cutoff` applies an additional, explicitly
heuristic approximation. For pole `k`, define

```math
w_k=\frac{\max(|\ell_k|,|r_k|)}{|\nu_k|^2},\qquad
q_k=\sqrt{\frac{w_k}{1+w_k}},\qquad
I_{\boldsymbol n}=\prod_k\frac{q_k^{n_k}}{\sqrt{n_k!}}.
```

The `:normalized_coupling_decay` policy retains `I_n >= importance_cutoff`
and requires every immediate parent to be retained. The result is therefore a
deterministic downward-closed order ideal; couplings to omitted upward ADOs
are zero. This score is a ranking heuristic, not an error bound. Zero cutoff
preserves the complete hierarchy and its original setup route. Inspect
[`heom_hierarchy_metadata`](@ref) and [`heom_ado_importances`](@ref), and
repeat the calculation at lower cutoffs.

For `K` exponential terms, the number of retained ADOs is exactly

```math
M=\binom{K+D}{D},
```

and the coordinate dimension is `M * length(basis)`. Setup checks this count
with exact integer arithmetic before allocation.

The hierarchy state therefore stores `M * n_PI` complex coordinates. For
positive `K` and `D`, the number of undirected nearest-hierarchy edges is

```math
E=K\binom{K+D-1}{D-1}.
```

The plan stores each edge once, together with two compact incident-edge
references and only `K*D` level-dependent scaling factors. It does not retain
dense `M`-by-`K` upward/downward index and factor tables. Topology indices use
32 bits whenever the prepared counts permit. One dense physical coupling
block is retained per selected PI sector and bath. PI compression removes the
system's `d^N` scaling; it does not remove the combinatorial hierarchy growth
or the dense cost of a large retained Schur block.

## Basic workflow

The following is a one-qubit exponentially correlated dephasing bath. The
same code works for a larger PI ensemble; only the PI basis and collective
coupling operator change.

```julia
using PermutationalInvariantDynamics

basis = PIBasis(1, 2)
spin = spin_matrices()
Q = collective_operator(basis, spin.jz)

# C(t) = 0.30 exp(-1.20t)
bath = HEOMBath(Q, 0.30, 1.20)
system = PIModel(basis, ())
plan = HEOMPlan(system, bath; max_depth=4)

rho0 = iid_pure_state(basis, ComplexF64[1, 1] / sqrt(2))
hierarchy0 = heom_initial_state(plan, rho0)
hierarchy_t = heom_evolve(plan, hierarchy0, (0.0, 0.7); steps=200)
rho_t = heom_reduced_state(hierarchy_t)
```

[`heom_initial_state`](@ref) constructs the conventional factorized initial
condition: `rho_0` is supplied by the user and every auxiliary ADO is zero.
Use [`heom_ado`](@ref) to inspect a detached auxiliary operator by position or
occupation vector.

```julia
first_auxiliary = heom_ado(hierarchy_t, [1])
indices = heom_multiindices(plan)
```

For several baths, pass a tuple. Each bath may contain several exponentials:

```julia
bath_z = HEOMBath(Qz, [c1, c2], [nu1, nu2])
bath_x = HEOMBath(Qx, [c3], [nu3])
plan = HEOMPlan(system, (bath_z, bath_x); max_depth=4)
```

The occupation-vector entries follow the prepared flattened order: all terms
of `bath_z`, including any automatically appended conjugate poles, followed
by all prepared terms of `bath_x`.

If a bath decomposition already provides both actions on one common pole
list, pass them explicitly:

```julia
bath = HEOMBath(Q, left, poles; right_coefficients=right)
```

Here `left`, `right`, and `poles` must have the same length. This route does
not append poles or impose a conjugacy relation.

## Physical bath constructors

[`drude_lorentz_bath`](@ref) uses units `hbar=k_B=1` and the convention

```math
J(\omega)=\frac{2\lambda\gamma\omega}{\gamma^2+\omega^2}.
```

For `decomposition=:matsubara`, it retains

```math
\begin{aligned}
\nu_0&=\gamma,&
c_0&=\lambda\gamma\left[\cot(\beta\gamma/2)-i\right],\\
\nu_k&=2\pi k/\beta,&
c_k&=\frac{4\lambda\gamma}{\beta}
      \frac{\nu_k}{\nu_k^2-\gamma^2}.
\end{aligned}
```

The Padé option uses the Hu--Xu--Yan tridiagonal-eigenvalue construction
([J. Chem. Phys. 134, 244106 (2011)](https://doi.org/10.1063/1.3602466)). Its
small symmetric eigensolves currently support Float32 and Float64 physical
parameters; use Matsubara for BigFloat. A coincident Drude and thermal pole is
rejected because the separate terms are singular and require a combined
analytic limit. Padé setup has the package-wide 512 MiB `memory_budget`; use
`Inf` only as an explicit opt-out after checking the quadratic eigensystem.
The Padé parameter routine was adapted from QuTiP's BSD-3-Clause HEOM bath
implementation and modified for the package's precision, validation, and
memory contracts. The exact revision, affected routine, modifications, and
complete upstream notice are recorded in
[`THIRD_PARTY_NOTICES.md`](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/blob/main/THIRD_PARTY_NOTICES.md).

```julia
bath = drude_lorentz_bath(
    Q, 0.3, 0.8, 2.0;
    matsubara_terms=4, decomposition=:pade)
plan = HEOMPlan(system, bath; max_depth=5, terminator=:residue)
```

[`underdamped_brownian_bath`](@ref) implements the explicitly different
spectral-density convention

```math
J(\omega)=\frac{\lambda^2\gamma\omega}
{(\omega_0^2-\omega^2)^2+\gamma^2\omega^2},
```

with $0 \lt \gamma \lt 2\omega_0$, its two conjugate damped-oscillator poles, and a
finite Matsubara tail. Here `coupling_strength=lambda` enters quadratically;
it is not the Drude reorganization-energy parameter. This distinction is
recorded in [`heom_bath_metadata`](@ref). Both physical constructors store the
real finite-tail discrepancy returned by [`heom_bath_residue`](@ref). The
residue may be signed for a finite approximation and is never clipped. These
bath and terminator conventions also follow the open-source HEOM formulation
described by Lambert *et al.*,
[Phys. Rev. Research 5, 013181 (2023)](https://doi.org/10.1103/PhysRevResearch.5.013181).

Physical-parameter inputs preserve Float32, Float64, or BigFloat on the
Matsubara route. Generated coefficients are rounded once in that selected
precision. Increase the number of expansion terms independently of hierarchy
depth and integration accuracy.

## Thermal and correlated preparation

[`heom_thermal_state`](@ref) starts from the bare-system Gibbs state. Its
three preparation modes make the approximation explicit:

```julia
factorized = heom_thermal_state(plan, H, beta)

relaxed = heom_thermal_state(
    plan, H, beta;
    preparation=:relaxation, relaxation_time=20.0, steps=4000)

stationary = heom_thermal_state(
    plan, H, beta;
    preparation=:stationary,
    preconditioner=:block, krylovdim=40)
```

The factorized route has zero auxiliary ADOs. Real-time relaxation produces a
correlated hierarchy after a user-selected finite time. The stationary route
uses the factorized hierarchy only as the initial GMRES guess. Neither route
is imaginary-time HEOM, and a stationary root is thermal only if the supplied
system generator and physical bath satisfy equilibrium assumptions. Known
physical-bath inverse-temperature metadata is checked exactly, but arbitrary
Markovian drives cannot be certified automatically.

## Identical independent local environments

A global [`HEOMBath`](@ref) coupled through `Q=sum_i q_i` describes a common
bath and contains cross-correlations between different particles. It is not
an implementation of `N` independent local baths. A true local-bath HEOM
requires symmetry-adapted occupations carrying both site and hierarchy labels;
those coordinates do not fit the current global-ADO `HEOMPlan`.

For a physically realizable collection of damped poles, use the exact
finite-cutoff supersite embedding described in [Local pseudomodes and PI
supersites](pseudomodes.md):

```julia
mode = BosonicPseudomode(
    3;
    label=:local_bath,
    frequency=omega_c,
    damping=kappa,
    thermal_occupation=nbar,
)

coupling = PseudomodeCoupling(
    Llocal;
    mode=:local_bath,
    strength=g,
)

embedding = pseudomode_model(
    N, Hlocal, mode;
    couplings=(coupling,),
)

basis = embedding.basis
model = embedding.model
```

Every system together with all of its local modes is one identical PI
supersite. `LocalJump` creates independent damping on every copy of a mode.
This is a time-local Markovian embedding in
`system ⊗ mode1 ⊗ mode2 ⊗ ...` local order, not global-ADO HEOM. At zero
occupation, one rotating-wave mode realizes the pole
`c=abs2(g)` and `nu=kappa/2 + im*frequency` through
`g*L*a' + conj(g)*L'*a`. Non-Hermitian `L`, Hermitian longitudinal
couplings, several modes, and counter-rotating terms are supported.

`pseudomode_model` lifts fixed local, collective, and symmetric `p`-body
system terms into the paired supersite ordering. Independent mode damping
usually requires every Schur sector; a restricted basis is accepted only when
`PIModel` validates it. The oscillator cutoffs are explicit approximations and
must be converged separately. An arbitrary fitted complex pole set need not
admit a positive damped-mode realization.

The historical single-mode convenience
[`independent_local_pseudomode_model`](@ref) remains available, but the
`BosonicPseudomode`/`PseudomodeCoupling` workflow is the general interface.

## Repeated and matrix-free calculations

[`HEOMPlan`](@ref) is immutable prepared data. [`HEOMWorkspace`](@ref) owns
only application scratch: a system-Liouvillian workspace and three PI-sized
vectors for left/right and residue double-commutator actions. Reuse one
workspace per task:

```julia
work = HEOMWorkspace(plan)
y = similar(hierarchy0.data)
apply!(y, plan, hierarchy0.data, work)  # autonomous system
```

One application views the hierarchy as an `n_PI`-by-`M` matrix and evaluates
the prepared system Liouvillian in bounded column batches. The packed edge
incidences are ordered by source ADO and bath. Consequently, `Q_b*rho_n` and
`rho_n*Q_b` are computed once for each nonempty `(n,b)` group and then
scattered to every pole edge in that group. Forward and adjoint application
share the same packed topology and use algebraically adjoint weights; this is
only a contraction reordering and does not alter the hierarchy equations.

Matrix right-hand sides are flattened as `n_PI` by
`(number of ADOs * number of right-hand sides)`. A driven system schedule is
prepared once for that complete application even when the flattened matrix is
processed in several bounded chunks. Prepare a wider bounded chunk up front
when it is reused often:

```julia
X = hcat(hierarchy0.data, hierarchy_probe_1, hierarchy_probe_2)
Y = similar(X)
batch_work = HEOMWorkspace(plan; batch_columns=size(X, 2))
apply!(Y, plan, X, 0.0, parameters, batch_work)
```

The system batch never exceeds the internal width bound. A default
`HEOMWorkspace(plan)` remains valid for arbitrary matrix widths; it uses more
chunks without growing its retained arrays. Forward and adjoint matrix
applications use the same layout.

[`HEOMEvolutionWorkspace`](@ref) additionally owns three hierarchy-sized
vectors for low-storage fixed-step RK4: a stage state, one derivative, and a
weighted stage accumulator. Passing it to [`heom_evolve!`](@ref) avoids
repeated allocations across a parameter or convergence scan.

```julia
evolution_work = HEOMEvolutionWorkspace(plan)
destination = copy(hierarchy0)
heom_evolve!(destination, plan, hierarchy0, (0.0, 0.5);
             steps=200, workspace=evolution_work)
```

### Instantaneous PI pulses

An ideal system unitary can be applied to every ADO without restarting the
hierarchy:

```julia
pulse = PIUnitaryPulse(basis, exp(-im * pi * spin.jx))
sequence = HierarchyPulseSequence(pulse_times, pulse)
states = heom_time_evolution(
    plan, rho0, times;
    steps_per_interval=8, pulses=sequence)
```

For a local matrix, `PIUnitaryPulse` prepares the Schur blocks of
$U^{\otimes N}$ directly; a compatible PI operator is also accepted. At an
event, every ADO is mapped as
$\rho_{\boldsymbol n}\mapsto U\rho_{\boldsymbol n}U^\dagger$. The root and
all memory auxiliaries therefore receive the same physical system pulse.

The fixed-step driver splits an interval exactly at each event and uses the
half-open convention `(start, stop]`. A pulse at a saved endpoint is applied
before that hierarchy state is saved and cannot be applied twice by adjacent
intervals. `apply_hierarchy_pulse!` exposes the same in-place operation for a
caller-managed `HEOMState` and task-owned `HEOMEvolutionWorkspace`.

Published tetrahedral, octahedral, and icosahedral Eulerian sequences are
available directly:

```julia
tau0 = 0.01
tedd = tetrahedral_pulse_sequence(basis, tau0)
oedd = octahedral_pulse_sequence(basis, tau0; cycles=2)
iedd = icosahedral_pulse_sequence(
    basis, tau0; start_time=0.5)
```

These constructors reproduce the TEDD, OEDD, and IEDD Cayley-graph words of
24, 48, and 120 events from Read, Serrano-Ensástiga, and Martin,
[*Quantum* **9**, 1661 (2025)](https://doi.org/10.22331/q-2025-03-12-1661).
Only two immutable axis--angle pulses are prepared for each schedule and
shared by reference across every event. The positive `tau0` is the free
interval before each pulse, so one cycle ends after `24tau0`, `48tau0`, or
`120tau0`, with the final cyclic pulse included at that endpoint.

The constructors implement ideal global spin rotations for any local spin
dimension. Their universal first-order single-spin ranges are respectively
$d\leq3$, $d\leq4$, and $d\leq6$. Claims about cancellation of interacting
spin Hamiltonians additionally require the anisotropy conditions stated in
the paper. A purely global rotation cannot remove a nontrivial rotationally
invariant interaction.

The plan scalar type is promoted from the prepared system, couplings, both
sets of bath coefficients, and frequencies. Sources, destinations, times, and step counts
must be representable without narrowing; compiled matrix-free system scratch
cannot be widened after preparation. Build the system and hierarchy at the
wider precision instead. Plans may be shared read-only, but each concurrent
application or evolution requires its own workspace.

Integer bath coefficients and frequencies are accepted only when their exact
values survive conversion to that prepared scalar type. Pass a suitably
precise `BigFloat` value (and prepare the system at matching precision) when a
large integer is not exactly representable; it is never silently rounded.
Explicit scaled-coordinate factors obey the same no-narrowing rule.

The bath owns a detached copy of its coupling, and a raw matrix supplied as
the system generator is copied into the plan. Subsequent mutation of either
caller object therefore cannot change prepared HEOM dynamics.

Hierarchy truncation is independent of the RK4 error. Use
[`heom_depth_convergence`](@ref) to compare successive depths in the
Hilbert--Schmidt norm of the final reduced state:

```julia
report = heom_depth_convergence(
    system, bath, rho0, (0.0, 0.7);
    depths=(2, 4, 6), steps=200,
    atol=1e-9, rtol=1e-7, consecutive=2)

collect(skipmissing(report.pairwise_errors))
report.pairwise_converged
report.converged
```

The system model is compiled once, and every truncated plan shares the same
prepared system and coupling blocks. Intermediate results retain only the PI
root states; `last(report.results).hierarchy` is the sole complete saved
hierarchy, while each raw result exposes its ADO count, dimension, trace drift,
and elapsed time. The standard [`ConvergenceStudyResult`](@ref) decision uses
the final `consecutive` requested depth pairs and never normalizes or repairs
either state. Repeat the report with a finer time step before attributing its
entire difference to hierarchy depth. The reported trace error is drift from
the supplied initial trace, not an assumption that the input was normalized.
Pass `scaling=:scaled` (and optional `scaling_factors`) to the source-based
overload, or use a scaled template plan; every prefix preserves the same
per-pole similarity transform.

[`heom_liouvillian`](@ref) adapts the plan to the package's synchronized
`MatrixFreeLiouvillian` interface. Its physical trace vector is nonzero only
on the root ADO:

```math
\mathrm{tr}_{\mathrm{HEOM}}(\boldsymbol\rho)
=\mathrm{tr}(\rho_{\boldsymbol0}).
```

This enables the existing matrix-free Krylov machinery without materializing
the full hierarchy generator. Both its forward and adjoint callbacks are
exact; explicit task-owned hot loops may call `apply!` and `apply_adjoint!`
with one `HEOMWorkspace`.

[`heom_steady_state`](@ref) is the direct trace-fixed GMRES convenience. For
larger hierarchies, `preconditioner=:block` extracts the common system block
once and represents every non-root ADO-diagonal `n_PI`-by-`n_PI` block by a
scalar shift. For `ComplexF32` and `ComplexF64`, one LAPACK Schur form normally
serves every well-conditioned shift, so retained storage is
`O(n_PI^2 + M)` even when all hierarchy decays differ. A reciprocal-condition
guard assigns an ordinary LU factor to any unsafe shift. Unsupported scalar
types use the generic LU route, sharing factors for exactly repeated shifts:

```julia
stationary_hierarchy = heom_steady_state(
    plan; preconditioner=:block,
    krylovdim=40, maxiter=800, atol=1e-10, rtol=1e-8)
rho_ss = heom_reduced_state(stationary_hierarchy)
```

Use [`heom_block_preconditioner`](@ref) directly when its setup will be reused
across solves. It omits only inter-ADO couplings from the preconditioning
approximation; it does not modify the generator or convergence test. Inspect
[`preconditioner_cost`](@ref) for the system right-hand sides, batched setup
calls, Schur/LU factorizations, guarded and fallback shifts, retained
coefficients, and per-application triangular solves. Every Schur application
checks a transformed residual and finite output. Its two scratch vectors are
lock-protected, so sharing one Schur preconditioner across concurrent solves is
safe but serialized; construct one per concurrent solve for parallel
throughput. `shift_backend=:lu` retains allocation-free concurrent application
without shared numerical scratch. No precision-narrowing shortcut is used.

Stationary solving requires an autonomous system generator. As for any finite
HEOM calculation, convergence of `rho_ss` with hierarchy depth remains the
user's responsibility.

For adaptive or stiff integration, [`heom_problem`](@ref) returns an in-place
`SciMLBase.ODEProblem` with its own captured HEOM workspace:

```julia
problem = heom_problem(plan, hierarchy0, (0.0, 10.0))
```

Choose and import a compatible SciML solver separately. Construct independent
problems for concurrent solves because each problem owns mutable scratch.

## Reliability checklist

Check separately:

1. convergence of the bath exponential decomposition;
2. consistency of the left/right correlation decomposition, especially when
   `right_coefficients` is explicit;
3. convergence in physical expansion terms and the effect of
   `terminator=:residue`;
4. convergence in `max_depth` and, when used, `importance_cutoff`;
5. convergence of the time step or Krylov residual;
6. preservation of `tr(heom_reduced_state(state))`;
7. Hermiticity and physicality of the root ADO.

The implementation never normalizes the hierarchy, clips eigenvalues,
symmetrizes invalid couplings, changes explicit bath coefficients, or inserts
a terminator without an explicit `terminator=:residue`. Default
conjugate-pole completion is part of the documented
correlation representation, not a state repair. The constructor rejects nonfinite coefficients, nondecaying
frequencies, non-Hermitian fixed couplings, incompatible PI bases, and scalar
types that would narrow the prepared system Liouvillian.

## Current limitations

- The implemented hierarchy is the unscaled or exactly similarity-scaled
  bosonic Gaussian-bath HEOM for fixed Hermitian system couplings. Fermionic
  sign hierarchies and non-Hermitian coupling pairs require a different
  construction.
- Each declared environment couples through one global PI operator `Q_b`.
  A common bath coupled to a collective observable is therefore supported,
  but this is not the specialized global-ADO hierarchy for `N` independent
  local non-Markovian baths. Use the finite-cutoff pseudomode embedding for a
  realizable damped pole.
- Factorized, finite real-time relaxed, and stationary correlated preparation
  routes are provided. Imaginary-time HEOM and a certificate that an arbitrary
  driven generator thermalizes to the interacting equilibrium are not.
- The residue terminator is a zero-frequency white-noise approximation.
  Low-temperature corrections beyond the retained physical expansion and
  Hamiltonian counterterms are not inferred.
- The number of ADOs grows as `binomial(K+D,D)`. PI compression removes the
  exponential system Hilbert space, not the combinatorial hierarchy growth
  caused by many bath poles and a large depth.
- Fixed-step RK4 remains the dependency-free dedicated time-domain solver.
  `heom_problem` supplies a SciML problem but no adaptive/stiff algorithm; add
  and import the solver package appropriate to the calculation.

The complete executable dephasing comparison is in
[`examples/pi_heom.jl`](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/blob/main/examples/pi_heom.jl),
with discussion in
[`examples/pi_heom.md`](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/blob/main/examples/pi_heom.md).
The ideal-pulse extension in
[`examples/nonmarkovian_dynamical_decoupling.jl`](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/blob/main/examples/nonmarkovian_dynamical_decoupling.jl)
applies CPMG and UDD4 events to every ADO, compares the reduced state with
PI--HOPS, and distinguishes the full-line one-pole Lorentzian correlation
from the physical positive-frequency spectral-density integral.

## API

```@docs
HEOMBath
drude_lorentz_bath
underdamped_brownian_bath
heom_bath_metadata
heom_bath_residue
HEOMPlan
HEOMWorkspace
HEOMEvolutionWorkspace
HEOMState
HEOMBlockPreconditioner
heom_block_preconditioner
heom_number_ados
heom_multiindices
heom_ado_importances
heom_hierarchy_metadata
heom_coordinate_scale
heom_initial_state
heom_thermal_state
heom_ado
heom_reduced_state
heom_evolve!
heom_evolve
heom_time_evolution
heom_problem
heom_depth_convergence
heom_liouvillian
heom_steady_state
```
