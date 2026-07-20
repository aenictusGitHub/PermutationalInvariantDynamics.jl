# API tiers and prepared analysis

The package exposes three API tiers. The tiers describe the recommended level
of abstraction and the expected rate of interface evolution; they do not
replace numerical convergence checks or the mathematical caveats of a method.

For a first model, read [Getting started: from a model to a
solution](getting_started.md) before using this page to decide how far below
the high-level API your calculation needs to go.

## Stable high-level API

These entry points are intended for ordinary research scripts and examples.
They validate basis ownership and return package-level state or result objects.

| Area | Recommended entry points |
|---|---|
| Representation | `PIBasis`, `PIState`, `PIOperator`, `exact_binomial`, `exact_multinomial`, Schur-block constructors, spin/state conveniences, `PIModel`, and the physical term constructors |
| Preparation | `compile`, `compile_family`, `specialize`, `isautonomous`, `freeze` |
| Dynamics | `solve_dynamics` (including observable-only streaming), `solve_populations`, `dynamics_problem`, `floquet_steady_state`, `quantum_trajectories` (including online ensemble statistics), `trajectory_steady_state` |
| Mean-field predictions | `MeanFieldPlan`, `solve_meanfield`, `meanfield_problem`, `meanfield_stationary_state`, `meanfield_stability` |
| Stationary and spectral analysis | `stationary_state`, `liouvillian_spectrum`, `pi_liouvillian_gap`, `diagnostics`, `recommend_solver` |
| Observables and information | `collective_expectation`, `collective_variance`, `qfi`, `qfim`, `stabilizer_renyi_entropy`, `two_time_correlation`, `delayed_second_order_correlation`, `stationary_correlation_spectrum`, entropy and distance functions |
| Visualization | `schur_block_structure`, `visualize_schur_blocks`, `spin_husimi_q`, `spin_wigner`, `qudit_husimi_q`, `visualize_spin_phase_space`, and the density/Liouvillian/Floquet spectrum data and renderers |
| Reductions and entanglement | `one_body_rdm`, `reduced_state`, `reduced_state!`, `reduced_purity`, `negativity`, `partial_transpose_spectrum`, `ppt_mixture_test` |
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
- `PPTMixturePlan` owns the sparse real conic map for the PI qubit
  PPT-mixture test. It is tied to one exact basis and is safe to share across
  state scans; Clarabel solver state is call-local. Even for a
  sector-restricted input basis, the plan enforces equality in every Schur
  sector of the complete basis.
- `StabilizerRenyiPlan` owns normalized Krawtchouk transforms and stable
  binomial metadata for one exact qubit basis and precision.
  `StabilizerRenyiWorkspace` owns the `O(N^2)` mutable transform scratch.
  Reuse both across pure symmetric states, share only the plan between tasks,
  and read the [nonstabilizerness guide](nonstabilizerness.md) before
  interpreting the result. Mixed or multi-sector states are rejected rather
  than assigned the paper's pure-state measure.
- `LiouvillianPlan`, `CompiledPIModel`, and `CompiledPIModelFamily` hold lowered model data;
  `LiouvillianWorkspace`, `ThreadedLiouvillianWorkspace`,
  `EvolutionWorkspace`, `KrylovWorkspace`,
  `ArnoldiWorkspace`, `SymmetryProjectorWorkspace`, and `ReductionWorkspace`
  hold mutable scratch.
- `BlockGMRESWorkspace`, `MultiShiftGMRESWorkspace`,
  `RecycledGMRESWorkspace`, and `KrylovExpvWorkspace` reuse dominant
  full-coordinate storage for structured linear solves and exponential
  actions. Projected dense factorizations may still allocate, and every
  workspace is task-owned.
- `ParameterScanPlan` holds a copied parameter grid and callable model recipe;
  `ParameterScanWorkspace` holds continuation, family-Liouvillian, and solver
  scratch. Serial
  continuation, independent threaded scans, and optional process-distributed
  scans have deliberately different ownership contracts described in the
  [scan guide](parameter_scans.md).
- `HEOMPlan` stores the finite hierarchy topology and physical coupling data.
  `HEOMWorkspace` and `HEOMEvolutionWorkspace` separate application scratch
  from the three hierarchy-sized arrays used by low-storage forward RK4. Hard
  hierarchy truncation and bath-pole convergence remain explicit research
  choices.
- `QuditHusimiPlan` retains dense coherent vectors for a fixed point/sector
  set and exact basis. Reuse it across states; setup can dominate for many
  large irreps. `ConvergenceStudyResult` retains every requested refinement
  result, so extract compact estimates when histories are large.
- `TrajectoryPlan` holds one immutable fixed-operator jump lowering.
  `TrajectoryWorkspace` owns one path's integration buffers, while
  `TrajectoryBatchWorkspace` owns task-local workspaces and RNGs for repeated
  ensembles. One batch workspace must not serve concurrent ensemble calls.
- `CorrelationPlan` owns copied quantum-regression insertion blocks and may be
  shared read-only. `CorrelationWorkspace` owns the conditional state, block
  products, RK4 storage, and shifted-GMRES storage; use one per concurrent
  task. Time-domain correlations and resolvent spectra require an autonomous
  generator.
- `CompositeSuperoperator` is a read-only sum of tensor-factor maps.
  `CompositeSuperoperatorWorkspace` owns its full-coordinate buffers, small
  factor fibres, and nested PI application workspaces. Use explicit
  task-owned workspaces in hot or parallel loops; the convenience matrix-free
  wrapper serializes access to one compatibility workspace.
- `CompositeTrajectoryPlan` owns explicit fixed monitored channels, their
  assembled unconditional generator, and multiplicity-aware trace metadata.
  `CompositeTrajectoryWorkspace` shares two full channel buffers across all
  gains and losses; batch workspaces add one complete path workspace and RNG
  per active task. The supplied background excludes monitored dissipators.
- `PopulationPlan` stores the certified sparse action on multiplicity-weighted
  Schur-diagonal probabilities. `PopulationWorkspace` owns its application and
  three-array forward RK4 scratch. Certification is exact by default, compile-only coordinate
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
- `ThreadedLiouvillianWorkspace` partitions complete destination Schur sectors
  among task-private block scratch. `threaded_apply!` and its adjoint evaluate
  every operator/rate schedule once before spawning and never perform atomic
  output updates or scheduler-ordered reductions. It is useful only after benchmarking
  against ordinary application; small blocks can be dominated by task or
  nested-BLAS overhead. One task takes the ordinary inline fast path.
- `MeanFieldPlan` lowers supported physical terms to one-site contractions
  with at most `d^p` by `d^p` scratch matrices. `MeanFieldWorkspace` owns its
  three-matrix forward RK and contraction scratch; use one workspace per
  concurrent task.
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
- PI channels/tomography, portable checkpoints, joint symmetry projectors,
  implicit steady-state gradients, and checkpointed adjoint control are
  experimental research utilities. Channel and POVM certificates cover only
  the retained Schur algebra; control results require grid/Krylov convergence.
- Diffusive plans are immutable and shareable, but each homodyne/heterodyne
  realization requires a task-owned `DiffusiveWorkspace`. The normalized
  Euler--Maruyama step must be converged in `dt` and is not a finite-step
  positivity certificate.
- Confidence-controlled quantum-jump and diffusive ensembles use bounded
  Hermitian observables and simultaneous empirical-Bernstein checks at fixed
  batch boundaries. Their `converged` flag covers Monte Carlo sampling only,
  not integration-step, hierarchy, model, or finite-size convergence.
- `ordered_local_moments`, the versioned neutral cumulant payloads, and the
  optional QuantumCumulants adapter provide exact finite-`N` reference data
  for higher-order closures. The extension can also lower supported
  microscopic `PIModel` terms to QuantumCumulants 0.5 indexed equations;
  direct PI terms and ambiguous schedules are rejected. The selected order
  remains an approximation and still carries `d^(2k)` local-tensor storage.
- `WeakPIPseudoKet`, `WeakPITrajectoryPlan`, and the `weak_pi_*` trajectory
  functions provide an opt-in direct-sum Schur-irrep unraveling. Fixed local
  gains are decomposed into checked one-box Kraus branches for qubits and
  qudits. The pseudo-state and its path statistics depend on this unraveling
  convention and are not labeled-particle pure states.
  `weak_pi_trajectory_steady_state` streams density-valued path means without
  storing pseudo-ket histories; it returns a generally mixed `PIState` and its
  optional path-level uncertainty diagnostics. Fixed-step RK4 and continuous-
  hazard adaptive/event-driven timing are supported. Operator-valued and
  local-p-body jumps and composite pseudo-ket paths are not yet part of this
  surface.
- `CompositePIBasis`, `FiniteOperatorBasis`, factorized composite states and
  operators, `CompositeSuperoperator`, and `CompositeTrajectoryPlan` provide
  deterministic and density-valued quantum-jump dynamics of several PI
  ensembles and small finite auxiliaries without a global Kronecker
  superoperator. Monitored tensor-product channels are explicit and use
  task-owned fixed-step workspaces. Composite pseudo-kets, diffusive/event-
  driven paths, arbitrary gain maps, and implicit extensions of
  single-ensemble reduction routines remain outside this surface.

Experimental does not mean “unchecked”: these paths have regression tests.
It means that algorithm selection, diagnostic fields, or preallocation
interfaces may change as these restarted spectral algorithms and iterative
extreme-scale LR construction are exercised on larger research problems.

## Preparing repeated observables

`OneBodyGeometry` contains the CG contractions shared by every local matrix.
Its sparse logical cells are stored as packed offsets into contiguous tuple
arrays, so qudit caches do not retain one heap vector per empty cell.
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
partial-trace, partial-transpose, and reduced-block buffers, plus recoupling
matrices matched once to its working scalar type. Compact real qubit matrices
are therefore copied into a complex workspace, while already matching qudit
matrices are shared read-only. Use one workspace per task. `reduced_state`
still allocates its returned `PIState`, whereas `reduced_state!` also reuses
caller-owned output. Spectral routines necessarily allocate their returned
eigenvalue arrays.

## Validation before analysis

```julia
report = state_diagnostics(rho; atol=1e-12, rtol=1e-10)
validate_state(rho; trace_one=true, hermitian=true, positive=true)
```

Validation never normalizes, symmetrizes, or clips the input. Analysis
functions may average only skew-Hermitian components already shown to be
within the requested tolerance.
