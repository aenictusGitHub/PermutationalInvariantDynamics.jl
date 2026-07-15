# API tiers and prepared analysis

The package exposes three API tiers. The tiers describe the recommended level
of abstraction and the expected rate of interface evolution; they do not
replace numerical convergence checks or the mathematical caveats of a method.

## Stable high-level API

These entry points are intended for ordinary research scripts and examples.
They validate basis ownership and return package-level state or result objects.

| Area | Recommended entry points |
|---|---|
| Representation | `PIBasis`, `PIState`, `PIOperator`, `exact_binomial`, `exact_multinomial`, Schur-block constructors, spin/state conveniences, `PIModel`, and the physical term constructors |
| Preparation | `compile`, `isautonomous`, `freeze` |
| Dynamics | `solve_dynamics`, `solve_populations`, `dynamics_problem`, `floquet_steady_state`, `quantum_trajectories` |
| Mean-field predictions | `MeanFieldPlan`, `solve_meanfield`, `meanfield_problem`, `meanfield_stationary_state`, `meanfield_stability` |
| Stationary and spectral analysis | `stationary_state`, `liouvillian_spectrum`, `pi_liouvillian_gap`, `diagnostics`, `recommend_solver` |
| Observables and information | `collective_expectation`, `collective_variance`, `qfi`, `qfim`, entropy and distance functions |
| Visualization | `schur_block_structure`, `visualize_schur_blocks`, `spin_husimi_q`, `spin_wigner`, `visualize_spin_phase_space`, and the density/Liouvillian/Floquet spectrum data and renderers |
| Reductions and entanglement | `one_body_rdm`, `reduced_state`, `reduced_state!`, `reduced_purity`, `negativity`, `partial_transpose_spectrum` |
| Validation | `state_diagnostics`, `positivity_diagnostics`, `validate_state` |

Prefer these commands in new code. In particular, `stationary_state` returns a
`PIState`-carrying result, while the older low-level `steady_state` interface
may return raw PI coordinates.

## Advanced reusable API

Advanced objects expose memory ownership and numerical algorithms explicitly.
They are supported and tested, but callers must obey their compatibility and
threading contracts.

- `OneBodyGeometry`, `PBodyGeometry`, `CollectiveObservablePlan`, and
  `ReductionPlan` own prepared representation data. They can only be used with
  the exact `PIBasis` object from which they were constructed.
- `LiouvillianPlan` and `CompiledPIModel` hold lowered model data;
  `LiouvillianWorkspace`, `EvolutionWorkspace`, `KrylovWorkspace`,
  `ArnoldiWorkspace`, `SymmetryProjectorWorkspace`, and `ReductionWorkspace`
  hold mutable scratch.
- `PopulationPlan` stores the certified sparse action on multiplicity-weighted
  Schur-diagonal probabilities. `PopulationWorkspace` owns its application and
  RK4 scratch. Certification is exact by default, compile-only coordinate
  lookup scales with the population dimension and is not retained, and scalar
  time-dependent rates reuse the same reduced matrices. Operator-valued
  schedules must be frozen before certification. Plan construction currently
  requires standalone `sqrt(f^nu)` coordinate scales to fit the working real
  type; model-derived plans also inherit the PI trace-vector bound. Direct
  population extraction and reconstruction instead use bounded prepared
  products.
- `InPlaceTimeOperator` describes a fixed-shape built-in operator schedule,
  including Appendix-D p-body operators.
  Its callback writes into the evaluated operator stored by each
  `LiouvillianWorkspace`; batched forward and adjoint calls evaluate it once
  per application stage rather than once per input column.
- `MeanFieldPlan` lowers supported physical terms to one-site contractions
  with at most `d^p` by `d^p` scratch matrices. `MeanFieldWorkspace` owns its
  RK and contraction scratch; use one workspace per concurrent task.
- `liouvillian`, `apply!`, `apply_adjoint!`, `evolve!`, `steady_state`, and the
  lower-level Krylov routines provide explicit control over storage and solver
  choices.

Prepared plans are read-only during application and may be shared between
tasks. Mutable workspaces must not be shared concurrently; allocate one per
task or thread.

Mean-field propagation is a controlled product-state closure, not exact PI
dynamics after correlations form. See [Mean-field predictions](meanfield.md)
for the finite-`N` and thermodynamic conventions and the supported term set.

`schur_block_structure` separates quantitative sector diagnostics from SVG
rendering. Reuse the returned structure when changing titles or colour scales;
matrix-free superoperator analysis needs one exact PI-coordinate application
per input coordinate. Young diagrams are a presentation option and do not
change or recompute that numerical structure. See
[Schur-block visualization](schur_visualization.md).

`pi_density_spectrum`, `liouvillian_spectrum_data`, and
`floquet_spectrum_data` similarly separate spectral computation from
presentation. Pass an existing compressed density result, eigenvalue result,
Floquet propagator, or multiplier vector whenever possible; repeated
`visualize_density_spectrum` or `visualize_spectrum` calls then perform no
eigensolve or time integration. Density spectra retain exact Schur
multiplicities rather than expanding to `d^N` entries.
Partial Krylov data retain their residual/convergence metadata and must not be
presented as a certified complete spectrum. See
[Spectral visualization](spectral_visualization.md).

`spin_husimi_q` and `spin_wigner` compute one normalized sphere per retained
qubit Schur sector. The aggregate grid is explicitly the angular marginal
over total spin; use `resolved=true` only when individual sector grids are
needed. Rendering a `SpinPhaseSpaceData` object never recomputes its transform.
See [Sector-resolved spin phase space](spin_phase_space.md).

### Extending physical terms

New `AbstractPITerm` subtypes use qualified dispatch hooks; no change to
`PIModel` or to its Liouvillian compiler is required. The hooks are
intentionally not exported, so an extension makes its dependency explicit.
For example, this custom collective channel delegates to the supported
`CollectiveJump` lowering:

```julia
const PID = PermutationalInvariantDynamics

struct MyCollectiveJump{O,R} <: AbstractPITerm
    operator::O
    rate::R
end

PID.term_operator(t::MyCollectiveJump) = t.operator
PID.term_rate(t::MyCollectiveJump) = t.rate
PID.body_order(::MyCollectiveJump) = 1
PID.term_scope(::MyCollectiveJump) = Val(:collective)
PID.term_process(::MyCollectiveJump) = Val(:jump)
PID.validate_term(t::MyCollectiveJump, basis::PIBasis) =
    PID.validate_term(CollectiveJump(t.operator; rate=t.rate), basis)
PID.compile_term(t::MyCollectiveJump, context) =
    PID.compile_term(CollectiveJump(t.operator; rate=t.rate), context)
```

`compile_term` must delegate to an equivalent built-in term. Compiled kernel
types and their application hooks are private and are not an extension
surface. A local sector-changing term must also define
`PID.required_sectors`; a term with a time-dependent operator or rate must
define `PID.rebuild_term(term, evaluated_operator, evaluated_rate)` so that
`freeze` can produce an autonomous term. Incomplete extensions fail during
model validation with an explicit error.

## Experimental research API

These routines are useful for specialized studies, but their signatures or
algorithms may evolve as the remaining numerical bottlenecks are addressed.

- `harmonic_arnoldi_spectrum` and matrix-free weak-symmetry projection target
  difficult interior modes. Residuals and spectral scope must be inspected;
  near-zero ordering is not a certified global-gap search.
- `implicitly_restarted_arnoldi_spectrum` (`method=:iram`) applies exact
  unwanted Ritz shifts through implicit QR for bounded-memory spectral-edge
  extraction.
- `jacobi_davidson_spectrum` (`method=:jd`) and
  `JacobiDavidsonWorkspace` provide hard
  invariant-subspace locking and optional preconditioned inexact correction
  solves for nonnormal or degenerate near-target clusters.
- `subduction_intertwiners` exposes the qudit subduction engine. Its default
  backend removes forbidden weights, assembles sparse simple-root equations,
  and obtains the exact prescribed nullity with SPQR; `algorithm=:dense`
  remains a small-problem reference. `littlewood_richardson_coefficient` uses
  an exact lattice-tableau recursion and constructs no intertwiners.
- Pseudospectral and response helpers are intended for moderate PI dimensions
  and require problem-dependent grid and tolerance convergence.

Experimental does not mean “unchecked”: these paths have regression tests.
It means that algorithm selection, diagnostic fields, or preallocation
interfaces may change as these restarted spectral algorithms and iterative
extreme-scale LR construction are exercised on larger research problems.

## Preparing repeated observables

`OneBodyGeometry` contains the CG contractions shared by every local matrix.
Use it while preparing several collective observables, then retain only the
smaller observable plans:

```julia
geometry = OneBodyGeometry(basis)
Jx = CollectiveObservablePlan(basis, sx/2; cache=geometry)
Jz = CollectiveObservablePlan(basis, sz/2; cache=geometry)

mx = collective_expectation(rho, Jx)
vx = collective_variance(rho, Jx)
F = qfim(rho, [sx/2, sz/2]; plans=(Jx, Jz))
```

A `CollectiveObservablePlan` retains one physical matrix per Schur sector,
roughly `sum(dim_nu^2)` complex numbers. It does not retain the larger
`OneBodyGeometry`. For a single contraction, pass `cache=geometry` directly;
for many states and one observable, the prepared plan avoids rebuilding the
collective blocks as well.

`one_body_rdm(rho; cache=geometry)` contracts every local matrix unit in one
geometry traversal. This is especially useful for repeated qudit marginals.

## Preparing repeated reductions

For a fixed particle bipartition, prepare its product-Schur data once:

```julia
reduction = ReductionPlan(basis, k)
work = ReductionWorkspace(reduction, rho)

rhoA = reduced_state(rho, k; plan=reduction, workspace=work)
pA = reduced_purity(rho, k; plan=reduction, workspace=work)
neg = negativity(rho, k; plan=reduction, workspace=work)
pt = partial_transpose_spectrum(rho, k; plan=reduction)

# Reuse the returned state as well as all contraction scratch.
rhoA_buffer = PIState(reduction.output_basis)
reduced_state!(rhoA_buffer, rho, reduction, work)
```

For qubits, the plan stores multiplicity-free SU(2) recoupling matrices. For
qudits, it stores every required Littlewood--Richardson intertwiner. Reuse can
remove nearly all repeated subduction setup, but it deliberately trades
retained RAM for speed. Qudit construction still performs sparse
rank-revealing factorizations and may have large temporary allocations.
Construct one plan per exact
`(basis,k)`, benchmark its setup and retained size, and avoid retaining plans
for bipartitions that are used only once.

`ReductionPlan` remains immutable and may be shared between tasks.
`ReductionWorkspace` owns the mutable product-block, multiplication,
partial-trace, partial-transpose, and reduced-block buffers. Use one workspace
per task. `reduced_state` still allocates its returned `PIState`, whereas
`reduced_state!` also reuses caller-owned output. Spectral routines necessarily
allocate their returned eigenvalue arrays.

## Validation before analysis

```julia
report = state_diagnostics(rho; atol=1e-12, rtol=1e-10)
validate_state(rho; trace_one=true, hermitian=true, positive=true)
```

Validation never normalizes, symmetrizes, or clips the input. Analysis
functions may average only skew-Hermitian components already shown to be
within the requested tolerance.
