# AGENTS.md

This is the repository-wide continuity guide for agents maintaining
`PermutationalInvariantDynamics.jl`. Read it before changing source code.

## How to use this guide

Keep this file focused on durable contracts: mathematical conventions,
architecture, ownership, precision, validation, and maintainer workflow. Do not
turn it into a release log, benchmark transcript, exhaustive API index, or copy
of an example guide.

Canonical detail lives elsewhere:

- User workflow and concepts: `docs/src/getting_started.md` and
  `docs/src/framework.md`.
- Public API tiers and complete index: `docs/src/api_tiers.md` and
  `docs/src/api_reference.md`.
- Algorithm derivations and implementation history: `IMPLEMENTATION_NOTES.md`.
- Planned research-scale work: `IMPLEMENTATION_PLAN.md`.
- Runnable-example inventory: `examples/README.md` and the paired example
  guides.
- Current performance gates, scaling harness, and comparison protocol:
  `benchmark/performance_regression.jl`, `benchmark/performance_audit.jl`,
  `benchmark/README.md`, and `docs/src/benchmarks.md`.

Before editing:

1. Inspect `git status --short` and preserve unrelated user changes.
2. Locate the relevant source, tests, docs, and example guide.
3. Identify the representation, precision, ownership, and memory contracts
   affected by the change.
4. Prefer the smallest implementation that preserves sparse and matrix-free
   equivalence.

After editing, run checks proportional to the risk, inspect the final diff, and
report the commands actually run. Never substitute historical test counts or
timings for current evidence.

## Non-negotiable rules

- Production code must not construct states, operators, superoperators, or
  transforms of size `d^N` or `d^(2N)`. Small dense reconstructions may exist
  only as explicitly bounded test oracles.
- Never silently truncate sectors, normalize a state, clip a spectrum, repair
  positivity, symmetrize invalid input, narrow precision, or change a requested
  algorithm to make a calculation succeed.
- Deterministic time-local generators may have negative rates; document that
  the resulting map need not be completely positive. Stochastic jump and
  diffusive rates must be finite, real, and nonnegative.
- Prepared plans are immutable and shareable. Mutable numerical scratch belongs
  to an explicit workspace owned by one task at a time.
- Sparse materialization and matrix-free application must lower from the same
  physical term plan and agree in forward and adjoint tests.
- Apply exact selection rules before allocating or looping. Caches are
  basis-owned, plan-owned, or call-local; never add an unprotected process-
  global mutable cache.
- Preserve generic scalar types where the backend supports them. A required
  nonzero value outside the requested scalar range must raise with wider-type
  guidance rather than become zero, `Inf`, or `NaN`.
- Optional dependencies must not alter core behavior or be imported by core
  source.

## Project purpose and references

The package simulates permutationally invariant (PI) open-system dynamics of
`N` identical `d`-level systems directly in the PI operator subspace.
Time-local generators are the core model. Finite-exponential PI--HEOM and
finite-cutoff PI supersites cover selected non-Markovian bosonic environments.

Primary mathematical source:

- `JPhysA58_ 275301(2025).pdf`
- Thierry Bastin and John Martin, *J. Phys. A* **58**, 275301 (2025)
- DOI: `10.1088/1751-8121/addfc1`

Published validation also uses maintainer-local copies of
`PRA94_033838(2016).pdf` and `PRA110_062208(2024).pdf`.

## Compatibility and repository hygiene

- Package/module: `PermutationalInvariantDynamics`.
- Supported Julia: 1.10 and later.
- Core dependencies: `LinearAlgebra`, `SparseArrays`, `Random`, and
  `SciMLBase`.
- Weak dependencies: `Clarabel`, `Distributed`, `Tables`, `Makie`,
  `QuantumCumulants`, `QuantumOptics`, `QuantumToolbox`, `JLD2`, and `HDF5`.
  Their methods belong in matching `ext/` modules.
- Current extension compatibility is Clarabel 0.11, Tables 1, Makie 0.21--0.24,
  QuantumCumulants 0.5, QuantumOptics 1, QuantumToolbox 0.45,
  JLD2 0.4--0.5, HDF5 0.16--0.17, and the supported Julia 1.10+
  `Distributed` stdlib. Update compat and optional smoke tests together.
- `Distributed` can load transitively through SciMLBase. Extension activation
  is not proof that the user requested remote workers.
- Do not commit generated root, `quality/`, `examples/`, `benchmark/`,
  `benchmark/comparison/*/`, `test/optional/`, or `notebooks/` manifests.
  Preserve the tracked `docs/Manifest.toml`.
- Public repository:
  `https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl`.
- License: `GPL-3.0-only`; the root `LICENSE` file is canonical.
- `COPYRIGHT.md`, `PROVENANCE.md`, and `THIRD_PARTY_NOTICES.md` are mandatory
  release records. Update them when authorship, an implementation source,
  unpublished research material, or an adapted third-party routine changes.
- Keep the QuTiP BSD-3-Clause snippet markers around the adapted Drude Padé
  routine and the CC-BY-4.0 markers around the published Platonic pulse data.
  Preserve both complete notices. A scholarly citation never substitutes for
  an upstream software or content license.
- Keep the pinned REUSE licensing CI job green. New or generated files must
  remain covered by `REUSE.toml` or a more specific valid SPDX declaration.
- Contributions use the DCO sign-off described in `CONTRIBUTING.md`. A
  maintainer must obtain any coauthor, employer, institution, or manuscript
  permission that automation cannot establish.
- The browser assistant emits substantial fixed templates and therefore
  labels generated Julia programs `GPL-3.0-only` without an output exception.
- Pin every external GitHub Action to a reviewed 40-character commit SHA.
  Dependabot is the update path; do not replace an immutable pin with a moving
  major-version tag.
- A source-package license audit is not a binary-distribution audit. A
  sysimage, container, application, or executable release needs an exact
  transitive dependency/artifact inventory, all required notices, and the
  corresponding-source/install-information review required by the GPL.

## Mandatory mathematical conventions

- A partition is a descending length-`d` tuple padded with zeros.
- Sector order is descending lexicographic partition order.
- GT-pattern order is ascending order of the stored entry tuple.
- Sector matrices flatten in Julia column-major order.
- The PI basis is equation (7):

  `F_nu^(W,W') = sum_T |nu,T,W><nu,T,W'| / sqrt(f^nu)`.

- Stored coefficient block: `C_nu`.
- Physical per-copy Schur block: `C_nu / sqrt(f^nu)`.
- Multiplicity-weighted analysis block: `sqrt(f^nu) * C_nu`.
- Trace: `sum_nu sqrt(f^nu) * tr(C_nu)`.
- Product block:
  `C_nu(A*B) = C_nu(A)C_nu(B)/sqrt(f^nu)`.
- Identity block: `sqrt(f^nu) * I`.
- Local computational labels are zero-based in the paper and one-based in
  Julia. Keep conversions explicit.
- For qubits, ascending stored GT-pattern order corresponds to
  `m=+j,+j-1,...,-j`.
- CG coefficients use the real Appendix-B convention, including the sign in
  equation (B.11).

### Exact combinatorial scaling

- `exact_binomial` and `exact_multinomial` are the public exact-`BigInt`
  routes. Multinomials use chained binomials, not factorial intermediates.
- Cancel large ratios as `Rational{BigInt}` before floating conversion. Use the
  checked binary-scaled helpers in `src/partitions.jl` when a ratio or square
  root is representable but its numerator or denominator is not.
- One-body scales use exact one-box branch weights. Appendix-D collective and
  gain scales use exact removal-path weights.
- Keep a native small-factor fast path. Exact or binary-scaled fallbacks must
  not slow ordinary small systems.
- `iid_pure_state` uses mode-centered conditional-binomial recurrences.
  Preserve structural zeros and never reintroduce a separately converted
  multinomial or an `Inf*0` product.
- State analysis should normally use `sqrt(f^nu) * C_nu`, whose trace is the
  sector population. Construct a physical per-copy block only for an API that
  explicitly returns one.
- Public `physical_block` must validate that its standalone divisor is
  representable even for a zero block. Internal fused contractions may use an
  exact prepared inverse when only the final product needs to be representable.

### Prepared one-box coefficient cache

- `OneBoxCGCache` is read-only, depth-bounded, tied to one exact `PIBasis`, and
  tied to its floating type and `BigFloat` precision where applicable.
- Direct `cgc` and `three_nu_symbol` calls validate scalar/precision
  compatibility and membership in the basis-owned reachable cache domain.
  Geometry and model routes additionally validate exact basis identity and
  sufficient depth.
- `max_depth=1` covers nonempty one-body geometry; p-body geometry needs depth
  at least `p`. Enforce the cache memory budget before retaining data.
- Geometry constructors without `coefficient_cache` retain the call-local
  route. Model lowering may share only the bounded transient cache documented
  in `docs/src/mathematics.md`; the finished plan retains packed geometry, not
  the primitive cache.
- `LiouvillianPlan`, `compile`, `liouvillian`, `steady_state`, and
  `compile_family` must preserve cache-independent values, phases, selection
  rules, and output precision.
- Do not add a generic Wigner-`3j`/`6j` table to this path. PI and Appendix-D
  kernels use one-box `U(d)` maps. Qubit reduction and negativity retain their
  complete SU(2) recoupling data inside `ReductionPlan`.

## Architecture and API tiers

The ordinary prepared workflow is:

`PIBasis -> PIModel -> compile -> CompiledPIModel -> solver/analysis`.

Related workflows are:

- Guided high-level study:
  `source + task -> PIStudy -> explain/check -> solve -> PIStudyResult`.
- Mean field:
  `(N,d,terms) -> MeanFieldPlan -> MeanFieldWorkspace -> one-site solver`.
- Certified populations:
  `PIModel -> PopulationPlan -> PopulationWorkspace -> solve_populations`.
- Composite deterministic dynamics:
  `factors -> CompositePIBasis -> CompositeSuperoperator -> workspace`.
- Composite stochastic dynamics:
  `background + monitored channels -> CompositeTrajectoryPlan -> workspace`.
- One shared/global pseudomode:
  `PI system + finite mode -> GlobalPseudomodeModel -> generator/workspace`.
- Parameter scans:
  `parameter grid + builder -> ParameterScanPlan -> workspace -> scan/resume`.
- Scalar-rate families:
  `PIModel -> compile_family -> CompiledPIModelFamily -> specialize`.
- Named affine generator families:
  `base PIModel + named term groups -> compile_affine_family -> specialize`.
- Reusable representation setup:
  `PIBasis -> prepare_geometry/PreparationCache -> immutable geometry bundle`.
- Verified deterministic studies:
  `PIExperiment -> plan/explain -> verified_solve -> versioned result archive`.
- Counting and inference:
  `TrajectoryPlan -> TiltedLiouvillianPlan -> cumulants`, and
  `model/observable builder -> inference problem -> fit/identifiability`.
- Identical local pseudomodes:
  `system + finite modes -> PISupersite -> pseudomode_model -> compile`.
- Bosonic PI--HEOM:
  `PI system + HEOMBaths -> HEOMPlan -> HEOMWorkspace -> solve`.
- Bosonic PI--HOPS:
  `PI Hamiltonian + HOPSBaths -> HOPSPlan -> HOPSWorkspace -> trajectories`.
- Ideal hierarchy control:
  `local/PI unitary or Platonic constructor -> HierarchyPulseSequence -> HEOM/HOPS`.

Prefer `PIStudy`/`solve`, `compile`, `solve_dynamics`, `stationary_state`,
`liouvillian_spectrum`, `diagnostics`, and `recommend_solver` in research
scripts. Lower-level `liouvillian`, `steady_state`, `apply!`, and Krylov APIs
are advanced interfaces. `docs/src/api_tiers.md` defines the intended stability
tier. `Workflow` is the curated stable namespace and must alias parent-module
bindings rather than wrap them. `Models` owns convention-tested recipes and
detached catalog metadata; examples should not fork their normalizations.
Guided checks never solve or materialize a source. Treat `PIDiagnosticIssue`
codes as the machine-readable contract; user interfaces must not parse their
human messages. `PIStudyResult.raw` always retains the original high-level
result, and `result_*` accessors never reconstruct missing output.

The static GitHub Pages model assistant lives in
`docs/src/model_code_generator.md` with dependency-free parser/emitter and UI
assets under `docs/src/assets/model_code_generator_*`. Its LaTeX subset must
lower through a typed whitelist AST, never `eval`, textual Julia
interpolation, or a remote service. Keep local versus collective jump
semantics explicit, default generated models to the complete PI basis and
memory-guarded automatic solver route, and test both parser rejection and
generated Julia syntax in `test/test_model_code_generator.jl`. The direct and
verified-experiment workflow selectors must reject unsupported combinations;
verified output lowers through `PIExperiment`, `explain_experiment`, and
`verified_solve`. Browser downloads contain Julia source, a normalized JSON
manifest, a README, and a Pluto notebook, with no executable deserialization.
Loading a manifest or URL fragment must reconstruct the typed configuration
and pass it through the same parser, physical checks, and resource checks as
manual input. Local autosave and share links remain browser-only and must not
contact a service. Its structured
architecture selector covers an ordinary PI ensemble, identical local
pseudomodes through `PISupersite`, and one shared pseudomode through
`GlobalPseudomodeModel`. Its typed calculation selector covers stationary
states/observables, state-free observable dynamics, selected Liouvillian
spectra, and certification-aware gaps. Ordinary PI and identical-local-
pseudomode stationary models may instead emit a term-resolved
`TrajectoryPlan`, an explicit product initial state, one fixed-capacity
`TrajectoryBatchWorkspace`, and `trajectory_steady_state`; the shared-global-
mode stationary combination remains unsupported. Observable dynamics may use
state-free PI/supersite trajectory statistics or a factorized
`CompositeTrajectoryPlan` for a shared mode. Shared-mode system jumps remain
unmonitored in that composite background and this must be stated explicitly.
Post-stationary purity, entropy, one-body RDM, and QFI always refer to the
physical systems: trace local or shared pseudomodes first. Selected spectral
values are not a certified global gap, and generated gap code must expose its
certification flags. Arbitrary composite tensor LaTeX remains unsupported
until it has a typed factor-and-cross-term schema; never approximate it with
free-form `kron` parsing.

The assistant's typed scan overlay supports direct deterministic stationary
states, stationary observables, and selected spectra for ordinary PI and
identical-local-pseudomode models. Axes are inclusive linear ranges over
existing model parameters; their Cartesian product is stored with the first
axis varying fastest. Generated scans use `ParameterScanPlan`, one task-owned
workspace, serial continuation, and a scalar-rate `CompiledPIModelFamily` fast
path when its stricter contract applies. Validate every finite grid point
before emission. Reject trajectories, dynamics, certified gaps, verified
experiments, optional post-stationary analyses, and shared-global-pseudomode
scans until each has a native typed source/output contract; never emit an ad
hoc loop or silently change the calculation.

`PIModel.terms` is a concrete immutable tuple. `LiouvillianPlan` owns prepared
read-only blocks, contractions, and rate descriptions. `LiouvillianWorkspace`
owns mutable scratch. Compatibility `mul!`/`action!` calls are synchronized;
parallel hot loops need explicit task-owned workspaces.

`compile_affine_family` lowers the union of one fixed base and named,
autonomous physical term groups once. Specialization must require exactly the
declared parameters, multiply complete generator contributions, preserve
ordinary compiled-family precision and selection rules, and return an
ordinary `SpecializedPIModel`. Negative coefficients are valid for
deterministic time-local generators; stochastic consumers must still reject
negative rates.

The first composite factor is the fastest coordinate. A factorized vector is
`kron(x_last,...,x_first)`, and the factorized map has reversed Kronecker
order. Never materialize the global Kronecker matrix in production.
`CompositeSuperoperatorBatchWorkspace` batches equal tensor fibers from
several right-hand sides and has immutable capacity; oversized input must
raise rather than grow hidden buffers. Forward and adjoint batch routes must
match repeated vector application. Prepared consumers of
`composite_matrixfree(S)` must recover a fresh composite workspace from the
wrapper's retained immutable plan; do not route those batches through its
synchronized columnwise compatibility callback.

Custom `AbstractPITerm` implementations extend the qualified hooks in
`src/terms.jl`. `compile_term` must delegate to an equivalent built-in term;
compiled-kernel internals are not an extension surface. Add end-to-end sparse,
matrix-free, adjoint, validation, and `freeze` coverage for every extension
pattern.

## Source map

- `src/PermutationalInvariantDynamics.jl`: include order and public exports.
- Representation core: `partitions.jl`, `gtpatterns.jl`, `cgc.jl`, `basis.jl`,
  `tensor_indices.jl`, `geometry.jl`, `pbody.jl`, and `operators.jl`.
- Model terms and lowering: `terms.jl`, `correlated_jumps.jl`, `spin.jl`,
  `vectorization.jl`, `liouvillian.jl`, `threaded_apply.jl`, and
  `compiled_families.jl`. `interchange.jl` owns dependency-neutral one-site
  operator import, and `affine_families.jl` owns named affine generator
  specialization. `source_protocol.jl` centralizes private basis,
  structured trace-functional, task-workspace, adjoint-capability, and
  matrix-free-only discovery for prepared linear-operator wrappers; extend
  those traits instead of adding consumer-local `isa` cascades.
  `solver_algorithms.jl` owns
  canonical solver symbols and compatibility aliases. `result_protocol.jl`
  owns explicit physical-time result lookup. `progress.jl` owns dependency-
  free progress events and cooperative cancellation. `result_outputs.jl` owns
  compact summaries, dependency-free result tables, and the common export dispatch;
  optional storage and plotting methods remain in extensions.
- Krylov, spectra, and symmetries: `krylov.jl`, `krylov_extensions.jl`,
  `spectra.jl`, `evans.jl`, `symmetries.jl`, and
  `restricted_symmetries.jl`, and `automatic_symmetries.jl`.
- State analysis: `observables.jl`, `entanglement.jl`,
  `reduction_sets.jl`, `prepared_artifacts.jl`, `local_factor_trace.jl`,
  `genuine_entanglement.jl`, `information.jl`, `nonstabilizerness.jl`,
  `symmetry_information.jl`, `populations.jl`, and
  `research_utilities.jl`.
- Deterministic dynamics and studies: `sciml.jl`, `evolution.jl`,
  `meanfield.jl`, `floquet.jl`, `response.jl`, `correlations.jl`,
  `highlevel.jl`, `guided_studies.jl`, `scans.jl`, `convergence.jl`,
  `experiments.jl`, `inference.jl`, `model_zoo.jl`, and
  `workflow_namespace.jl`.
- Non-Markovian and stochastic systems: `pseudomodes.jl`,
  `global_pseudomodes.jl`, `composite.jl`, `heom.jl`, `hops.jl`,
  `hierarchy_pulses.jl`, `bath_fitting.jl`, `trajectories.jl`,
  `batched_trajectories.jl`, `counting_statistics.jl`,
  `composite_trajectories.jl`,
  `weak_pi_trajectories.jl`, `diffusive.jl`, `adaptive_ensembles.jl`, and
  `distributed_api.jl`.
- Research utilities and optional bridges: `cumulants.jl`, `channels.jl`,
  `tomography.jl`, `checkpoints.jl`, `control.jl`, and `accelerators.jl`.
- Visualization and phase space: `phase_space.jl`, `qudit_phase_space.jl`,
  `visualization.jl`, `spectral_visualization.jl`, and
  `phase_space_visualization.jl`.

## Compilation, performance, and memory contracts

### Fixed kernels and matrix-free application

- A basis containing only `(N,0,...)` lowers collective one-body operators in
  the symmetric occupation basis, using
  `X_ab*sqrt(n_b*(n_a+1))`. The read-only geometry owns preconverted diagonal
  factors and packed transitions, so an `InPlaceTimeOperator` fill remains
  allocation-free for ordinary IEEE precision. Keep the `_needs_wide_collective`
  gate: cancellation-risk sizes retain the guarded general route. Mixed-sector
  collective-only models retain a private diagonal-only `OneBodyGeometry`
  containing `(sector,sector)` contractions. Local gains and conservative
  custom terms retain the complete sector-changing geometry.
- Fixed floating rates and `hbar` values participate in the immutable plan
  precision; exact rates retain the checked native-type path. Callable rates
  are tied to the prepared precision and must reject nonfinite, complex,
  wider, overflowing, or nonzero-underflowing evaluations. Fixed operators,
  in-place prototypes, and evaluated schedules must reject nonfinite
  coefficients before Schur lowering.
  If a fixed scalar widens a dense kernel, widen its blocks, loss matrices,
  and rectangular contractions once during preparation; do not leave mixed
  dense `mul!` calls to allocate packing buffers at application time. Preserve
  exact sparse support rather than widening sparse one-body kernels.
  `apply!` and `apply_adjoint!` require non-aliasing source and destination
  arrays, including partially overlapping views.
- Fixed collective one-body Schur blocks are assembled directly on exact CSC
  support in every sector; do not allocate a dense block and sparsify it
  afterwards. Fixed collective `K'K` blocks use sparse Gram products and remove
  exact zeros only. Driven blocks remain dense because their support may change
  at an evaluation. This representation choice must remain term-type-stable so
  `LiouvillianPlan` inference does not depend on the runtime sector list.
- Sparse materialization must form sparse Kronecker pieces from exact block
  support and reuse prepared `K'K` blocks. Never construct a dense
  `m^2`-by-`m^2` sector superoperator and sparsify it afterwards.
- Fixed `LocalJump` and safely scaled fixed `LocalPBodyJump` gains retain
  rectangular Schur contractions. Do not restore quartic PI-coordinate gain
  maps to matrix-free plans. Cancellation-risk p-body cases retain the guarded
  triplet fallback; explicit sparse materialization may expand the factors.
- `PBodyGeometry` retains every Appendix-D path isometry on exact CSC support
  over the combined `(centre pattern, local word)` coordinate. Its
  three-dimensional indexing interface is a compatibility view, not
  authorization to retain dense path tensors. Dynamic and static contractions
  must iterate the packed support directly. Cancellation-risk driven kernels
  use preallocated real certification scratch and accumulate both the value
  and absolute contribution bound on that same support; never restore a
  logical dense-index scan.
- Fixed one-body contractions may retain exact structural-zero supports when a
  setup-time arithmetic gate predicts a gain. Never use a numerical dropping
  tolerance. Dense fixed and driven contractions keep the dense rectangular
  path.
- Compatible fixed numeric kernels may fuse Hamiltonian blocks and
  anticommutator loss while keeping channel-resolved gains. When every input
  block is sparse, fusion must accumulate one CSC column at a time, visit
  terms in model order, and remove exact cancellations only; do not restore
  dense zero blocks. Preserve concrete tuple recursion and the
  Julia-1.10-stable default fusion dispatch.
- Trajectory, population, channel, and other channel-resolved consumers must
  use term-resolved lowering; never infer physical channels from a fused
  deterministic kernel.
- Batched application evaluates each schedule once and uses sectorwise matrix-
  matrix kernels where supported. Preserve vector/column fallbacks for custom
  callbacks and uncommon driven kernels.
- Batched conditional trajectories operate only on cohorts whose scheduler
  has already selected the same time and step. Fixed-capacity workspaces must
  reject growth; per-index RNG streams move with regrouped columns. Never
  replace independent intensity-capped trajectory scheduling with a hidden
  common minimum step.
- A complete vector action packs each immutable input Schur block once, before
  visiting the heterogeneous kernel tuple. The same contract applies to
  adjoint, conditional-trajectory, lowered restricted-symmetry, and
  multiworker threaded actions. Thread workers may share only the caller-
  packed read-only source; their mutable left/right/gain scratch remains
  task-owned.
- Fixed local one- and p-body gains lowered into Cartesian strong-symmetry
  coordinates retain sliced rectangular Schur contractions. Do not expand
  these gains into quartic reduced-coordinate triplets. Preserve prepared
  exact p-body scales in both forward and adjoint sandwiches.

### Threaded application

`ThreadedLiouvillianWorkspace` is an explicit opt-in for one vector action.
Each output Schur sector belongs to exactly one worker; prepared data and input
are read-only, and each worker owns block scratch. Preserve kernel order inside
a sector, evaluate schedules once before spawning, and do not use atomic output
updates or task-ordered reductions. One worker delegates to ordinary `apply!`.
Reject unsupported plan-less callbacks before spawning. Do not change global
BLAS threading. The workspace is guarded against concurrent reuse.

### Resource safeguards

- Long-running high-level workflows use dependency-free `ProgressEvent`
  records and cooperative `CancellationToken`s. Observe cancellation only at
  a documented safe unit boundary; never asynchronously interrupt a kernel
  while it owns mutable scratch. A workflow that cannot return a complete
  partial-result type raises `OperationCancelled` and documents the state of
  any caller-owned destination.
- Commands that can hide quadratic workspaces or histories use one 512 MiB
  default `memory_budget`; `Inf` is the only explicit opt-out.
- Validate before the guarded allocation. An explicit request must fail rather
  than reduce `nev`, drop output, narrow precision, accept nonconvergence, or
  choose another algorithm. An `:auto` route may choose a fully budgeted exact
  alternative.
- Count actual supplied workspace capacity, source-action scratch, result
  vectors, histories, observables, and predictable stochastic records.
- Keep the private source-memory traits semantically separate:
  `_performance_linear_operator_workspace_bytes` is a fresh task-owned
  workspace, `_performance_source_action_bytes` is only a per-application
  transient, and `_performance_batched_action_growth_bytes` is lazy retained
  batch growth. Consumers must compose the applicable pieces exactly once;
  scans multiply per-worker transients and growth by the active worker count.
- A prepared matrix-free Liouvillian already owns application scratch; do not
  charge a second generic action workspace inside nested Krylov accounting.
  Plan-less callbacks retain a conservative unknown-scratch allowance.
- Block methods must include block size in both Krylov and fixed-capacity
  operator-batch storage.
- Accelerator preflight must run on an already prepared source, account the
  simultaneous host assembly and device payload, check device index ranges,
  and initialize no hardware. Core must not claim a functional accelerator,
  materialize merely to probe capability, or hide per-action host transfers.
- `recommend_solver` separates setup, retained, solve, output, and peak
  estimates. Unknown builder, callback, or diagnostic allocations imply
  `safe_to_run=missing`, never `true`.
- Fixed standard kernels use exact-support contribution bounds for sparse
  materialization (`2m*nnz(H)`, `nnz(K)^2`, and `2m*nnz(K'K)` per sector).
  Dynamic, custom, or unknown kernels keep the dense-coordinate fallback.
  Include packed symmetric-occupation geometry in setup accounting and keep
  `compile`, compiled families, and `recommend_solver` estimates synchronized.
- Raw-model preparation estimates sum every distinct Appendix-D body order
  with any required one-body geometry. P-body bounds use the exact removal
  graph, GT-content selection rules, checked product dimensions, and
  coefficient-free structural support; terms of the same order share one
  geometry estimate.
- Mode-specific workspaces omit dominant unused arrays and must reject an
  incompatible operation instead of allocating the missing storage lazily.
- Prepared PI, composite, restricted, global-pseudomode, Floquet, and HEOM
  sources retain physical trace functionals on exact sparse support. Public
  APIs documented to return a dense trace vector remain dense compatibility
  routes; internal Krylov borders and preconditioners must not densify the
  functional merely to add its rank-one constraint.

Benchmark measurements and dated optimization history belong in
`IMPLEMENTATION_NOTES.md`. Regression thresholds live in the benchmark
scripts; do not copy machine-specific timings or allocation snapshots here.

## States, Schur blocks, populations, and spin helpers

### Schur construction

- `each_schur_block(A; representation=:physical)` returns detached physical
  blocks. `representation=:coefficient` returns mutable stored views.
- `operator_from_schur_blocks` and `state_from_schur_blocks` copy supplied
  blocks, accept arbitrary sector order, zero omitted sectors, and reject
  duplicates or wrong sizes. Optional output `T` must not narrow.
- `sector_metadata` retains exact `BigInt` multiplicities and Hilbert
  dimensions; qubit spin labels are exact rationals.

### Population backend

`PopulationPlan` is valid only after strict certification that the retained
GT-diagonal subspace is invariant. Its coordinate is

`p_(nu,W) = sqrt(f^nu) * (C_nu)_(W,W)`,

so `sum(p) == trace(rho)`.

- Default certification is structurally strict (`atol=rtol=0`). Explicit
  nonzero tolerances opt into approximate projection and must report
  `reason=:within_tolerance` when leakage is accepted.
- Test fixed autonomous components after their exact sum so cancellations are
  retained. Test scalar-driven components independently. Operator schedules
  remain inconclusive until frozen.
- Nonfinite generator data throw. Never discard a small coherence-generating
  term merely because another term sets a larger scale.
- The diagonal reverse lookup is population-sized and compile-time only.
  Reuse `PopulationPlan` and one task-owned `PopulationWorkspace`.
- Plan construction may require standalone `sqrt(f^nu)` scales even when
  `diagonal_populations` or `state_from_populations` can form a finite fused
  result. Preserve that distinction and raise with wider-type guidance.

### Spin and six-rate model

`spin_matrices` uses local order `|-j>,...,|j>`. For qubits this is
`(|g>,|e>)`. `dicke_state`, `dicke_operator`, `ghz_state`, and
`spin_coherent_state` must validate labels and infer precision without
narrowing.

`qubit_ensemble_model` implements the six PIQS channels with
`D[L]=L*rho*L' - {L'L,rho}/2`. Emission uses `j_-`, pumping uses `j_+`, and
dephasing uses `j_z=sigma_z/2`; one-qubit coherence therefore decays at half
the `dephasing` keyword rate. Omit fixed zero rates, retain callable rates, and
require explicit `T` when callable-only inputs cannot establish precision.

### Correlated reservoirs

`CorrelatedLocalJumps` and `CorrelatedCollectiveJumps` use a Hermitian PSD
Kossakowski matrix over fixed one-body channels. Fixed matrices are copied,
validated, and residual-Cholesky factorized once. In-place schedules keep all
evaluated matrix, factor, residual, effective-operator, block, and gain scratch
in the workspace and validate finiteness, Hermiticity, and PSD at every
evaluation. Preserve every strictly positive pivot except at explicit user
tolerance or arithmetic roundoff. The common scalar rate is independent of
the Kossakowski matrix. Checked correlated-rate wrappers must forward the
prototype precision used by compiled parameter families.

## Observables, information, entanglement, and reductions

- Expectations, entropies, distances, QFI/QFIM, symmetry-resolved information,
  phase space, and reductions should contract multiplicity-weighted blocks.
- Prepare `CollectiveObservablePlan` once per observable, `ReductionPlan`
  once per `(basis,k)`, and `CompositeReductionPlan` once per retained
  composite factor. Plans are tied to the exact basis; workspaces are
  task-owned.
- `PreparedGeometryBundle` may share a depth-bounded `OneBoxCGCache`,
  one-body geometry, selected p-body geometries, and a `ReductionPlanSet`.
  Validate exact basis identity, the retained basis layout, scalar type, and
  `BigFloat` precision/rounding before reuse. `PreparationCache` is explicit,
  user-owned, synchronized, and budgeted; never replace it with a process-
  global cache. Do not persist private geometry graphs until a versioned,
  reconstructing interchange schema can validate every convention.
  Because qudit LR/SPQR reduction setup has no allocation-free transient
  bound, a bundle containing qudit reductions must reject finite
  `memory_budget` before setup and require the explicit `Inf` opt-out.
- Repeated geometry-based one-body marginals use `OneBodyRDMWorkspace` and
  `one_body_rdm!`. The workspace owns one largest-sector
  multiplicity-weighted block and exact prepared scales; it is tied to one
  `OneBodyGeometry`, scalar type, and BigFloat context.
- `LocalFactorTracePlan` traces the same internal tensor factor from every
  supersite while retaining all particles. It owns normalized matrix-unit
  occupation transforms from the exact source basis to a complete kept-factor
  `PIBasis`. Retain only exact nonzero CSC support, selected with `iszero`;
  never introduce a numerical dropping tolerance. `LocalFactorTraceWorkspace`
  owns the short occupation vector. This is distinct from particle reduction
  through `ReductionPlan`. Construct CSC columns in exactly sized chunks so
  no unbudgeted growth capacity is hidden. Validate `Q'Q` through streamed
  sparse columns and linear scratch; never retain a quadratic Gram matrix.
  Enforce its sparse-transform setup budget and never replace it with
  local-string or full-Hilbert reconstruction.
- `ReductionWorkspace` modes (`:reduction`, `:negativity`, `:both`) omit
  incompatible buffers. A mode mismatch must raise rather than allocate.
- Reduced-state and purity paths contract
  `tr_B(U*C*U') = sum_q U_q*C*U_q'` one discarded-factor slice at a time.
  Do not restore the complete product block to `mode=:reduction`; negativity
  still requires it for partial transposition. Checked public calls validate
  by default. An explicit `check=false` is only for an enclosing workflow that
  already validated the state.
- A reduction workspace owns scalar-compatible recoupling data. Convert compact
  real qubit recouplers once per workspace. Prepared qudit plans retain LR
  intertwiners as exact-support CSC blocks split by the discarded-factor
  weight; a matching workspace shares them read-only and a wider workspace
  copies only their nonzeros. Do not restore mixed real/complex `mul!` calls
  that allocate packing scratch on Julia 1.10.
- General negativity has three exact routes: occupation branching for fully
  symmetric states, SU(2) recoupling for general qubits, and LR intertwiners for
  general qudits.
- `PPTMixturePlan` implements only the Novo--Moroder--Gühne PI qubit
  PPT-mixture SDP. It owns a sparse read-only conic map; Clarabel solver state
  is call-local. A restricted input basis still requires complete internal
  Schur-sector equality constraints with exact zeros in absent source sectors.
  Only a validated numerical non-PPT-mixture dual certificate detects GME for
  arbitrary `N`, within its reported tolerances; it is not an exact-arithmetic
  proof. PPT-mixture membership is also a biseparability certificate only for
  PI `N=3` (and the bipartite `N=2` consistency case). Never infer a conclusion
  from the raw scaled slack or solver status alone.
- `stabilizer_renyi_entropy` implements the second stabilizer Rényi entropy
  $M_2$ of Passarelli--Fazio--Lucignano only in its paper-defined domain:
  a validated pure qubit state with exact support in the fully symmetric
  `(N,0)` sector.  Do not apply its fourth Pauli moment to mixed or
  multi-sector PI states and call the result magic; the function must reject
  those inputs rather than project them.  A complete qubit basis is accepted
  when all other stored blocks are exactly zero.
- `StabilizerRenyiPlan` retains bounded normalized Krawtchouk transforms,
  hypergeometric modes, and logarithmic binomial data for one exact basis;
  `StabilizerRenyiWorkspace` owns its mutable transform arrays and is
  task-owned.  Evaluation is $O(N^4)$ with $O(N^3)$ retained plan data and
  $O(N^2)$ workspace.  It must never construct a $2^N$ state, a $4^N$
  Pauli list, or a table of representative Pauli matrices.  Keep the final
  multinomial contraction in log-sum-exp form and retain the independent
  Pauli second-moment/purity check.
- Product-Schur trace norms and reduced states use the exact weight
  `f^alpha * f^beta`. Qudit LR multiplicities are counted exactly; forbidden
  weights are removed before sparse generator constraints and SPQR nullspace
  recovery.
- Qudit LR setup can still require dense SPQR nullspace scratch, and the public
  `subduction_intertwiners` compatibility API returns detached dense matrices.
  A `ReductionPlan` must drop exact structural zeros before retention and keep
  discarded-factor blocks packed. Check every product-irrep dimension before
  allocation or reshape and retain that validated dimension in the packed
  map. Benchmark setup before caching many bipartitions, but never replace it
  with full-Hilbert reconstruction.
- Keep numerical-rank cutoffs sector-relative. Validate PSD before a rank
  decision. Roundoff handling inside a square root, logarithm, or support
  projector must not modify the input or returned spectrum.

## Visualization and phase space

Numerical extraction and rendering are separate layers. Rendering an existing
result must not trigger a solve, matrix-free probe, phase-space transform, or
full-Hilbert expansion.

`summarize` and `result_table` likewise inspect retained output only. Compact
tables exclude states, population vectors, eigenvectors, and nested
trajectory histories unless explicitly requested; an intrinsically
multiaxis output must raise instead of being flattened ambiguously.
Dependency-free `.pidrun` records are published from a same-filesystem staging
directory and retain exact PI states through the checkpoint schema. Optional
JLD2/HDF5 exports must state their portability limits and never narrow
heterogeneous numerical output silently. User metadata must not shadow schema,
package, result-type, or summary fields.

Every paired example guide embeds a curated expected-output PNG or SVG from
`docs/src/assets/example_figures/`. The source script's numerical assertions
are the regression; snapshots are illustrative and are never regenerated by
CI or Documenter. Update a snapshot only from a successful default run,
review it visually, keep stochastic seeds and convergence caveats explicit,
and commit no generated PDF.

`examples/catalog.toml` is the complete machine-readable inventory of paired
scripts and guides. `scripts/generate_example_gallery.jl` deterministically
renders the tracked browser-side gallery from that catalog; the catalog test
must fail when a runnable script is missing, duplicated, or points to an
absent reviewed output. `Models.find`, `Models.describe`, and `Models.example`
are read-only discovery helpers and never run or mutate an example.

### Schur blocks

- `schur_block_structure` computes numerical metadata and weights;
  `visualize_schur_blocks` and `save_schur_block_visualization` render them.
- Young diagrams label partition shapes, not distinguished standard tableaux.
  Tooltips report the exact number of tableaux.
- State/operator defaults measure physical blocks. Superoperator rows are
  output sectors and columns are input sectors in orthonormal PI coefficient
  coordinates.
- Exact matrix-free extraction costs one application per input PI coordinate.
  Freeze driven fallbacks once at the requested time before probing.
- `metric=:population` rejects appreciably complex or negative sector
  populations; it does not repair them.

### Density, Liouvillian, and Floquet spectra

- `pi_density_spectrum` is multiplicity-compressed by default. Keep exact
  degeneracies and negative eigenvalues; never expand a `d^N` list for a plot.
- `ComplexSpectrum` normalization preserves order, repeated roots, values, and
  available solver metadata.
- Liouvillian plots use `Re(lambda)=0` as the stability boundary. Floquet
  multiplier plots show the unit circle and `mu=1`.
- Multiplier-derived exponents use the principal `Log(mu)/T` branch. A zero
  multiplier raises; it is not omitted or regularized.
- A partial selected spectrum is not a certified global gap. Prefer passing an
  existing spectral result when residual/completeness metadata matters.

### Spin and qudit phase space

For qubit sector `nu`, use `rho_bar_nu=sqrt(f^nu)*C_nu`. Both Husimi-Q and
spin-Wigner sector densities integrate to the physical sector population.
Multi-sector aggregate data are an angular marginal, not one effective spin
irrep. Preserve the stored GT magnetic ordering and Wigner negativity.

`QuditHusimiPlan` lifts each supplied local unitary or Hermitian generator into
selected `U(d)` Schur irreps. The normalized-Haar sector value uses

`Q_nu(U) = dim(U_nu) * <nu,U|sqrt(f^nu) C_nu|nu,U>`.

The plan retains dense coherent-vector data per selected sector and point;
reuse it and budget setup before a large orbit sample. Unitary inputs rely on
LAPACK and are limited to Float32/Float64; use generator representation for
other supported scalar types. For `d=2`, normalized-Haar Q is `4pi` times the
spin-sphere density at the matching point.

## Mean field and cumulant bridge

### Mean-field closure

`MeanFieldPlan(model; limit=:finite)` and
`MeanFieldPlan(N,d,terms; limit=:finite)` evaluate the exact one-body
derivative at a product state; propagation closes generated marginals as
tensor powers. This is not exact finite PI dynamics once correlations form.

- Supported optimized terms include fixed one-body and symmetric p-body
  Hamiltonians, local one- and p-body jumps, and collective one- and p-body
  jumps. Direct PI blocks and terms without microscopic lowering must raise.
- Finite and thermodynamic combinatorics are explicit. Never infer or insert a
  Kac factor; use rates exactly as supplied.
- Product collective moments omit connected correlations.
- `MeanFieldPlan` owns copied read-only operators.
  `MeanFieldWorkspace` owns contractions and integrator stages. Reject
  incompatible precision rather than narrow. Fixed and evaluated rates must
  be finite and real; fixed operators and Hamiltonian `hbar` values must also
  be finite, and `hbar` must be nonzero.
- Fixed-point relaxation is basin-dependent and autonomous-only. Never return
  an unconverged default state. Stability lives on the real traceless-
  Hermitian tangent space.

### Higher-order cumulants

`ordered_local_moment` contracts distinct-particle local moments through
Appendix-D blocks and exact subset normalization. It retains local `d^(2k)`
data but never reconstructs `d^N`.

`CumulantModelPayload` and `CumulantBridgePayload` use neutral schema version
`1.0.0`. Adapters must preserve the particle-one-fastest tensor convention and
the standard dissipator, reject unsupported direct PI terms, and never invent
a microscopic model. QuantumCumulants lowering belongs entirely in its weak
extension and remains a selected-order approximation.

## Krylov solvers, scans, Floquet, and response

### Stationary and spectral solvers

- `steady_state(...; method=:krylov)` solves the trace-fixed system by
  matrix-free restarted GMRES. Reuse `KrylovWorkspace`.
- Spectrum routes include ordinary Arnoldi, thick-restarted harmonic Arnoldi
  near zero, exact-shift IRAM, hard-locking preconditioned Jacobi--Davidson,
  and thick-restarted block Arnoldi for clustered modes.
- Block Arnoldi is not block IRAM. Every reported block Ritz residual comes
  from a fresh full-space application.
- A partial nonnormal spectrum cannot certify missing modes. Gap routines must
  remove the full numerical stationary cluster and distinguish global from
  charge-resolved or partial estimates.
- All iterative workspaces derive storage precision from the operator and
  storage-bearing inputs. Recompile a prepared matrix-free model at wider
  precision instead of borrowing wider solver scratch.
- Sparse/dense steady-state factorizations keep their documented
  `ComplexF64` backend minimum and must reject wider floating shifts, initial
  vectors, or prepared scalar types rather than narrow them. Exact
  integer/rational components may use checked conversion, but overflow and
  nonzero underflow must raise. With basic diagnostics, `method=:auto` selects
  matrix-free Krylov before materialization when that factorization precision
  is unsupported.
- `KrylovWorkspace` stores real Givens rotations in the real component type
  of its Krylov scalar, never unconditionally in `Float64`; preserve this for
  both lower-precision speed and wider-precision reliability.
- Prepared Schur-sector preconditioners lower diagonal blocks directly from
  static plan kernels; only operator-scale probes apply the full Liouvillian.
  A `SpecializedPIModel` must evaluate those kernels with its bound family
  rates. Plan-less callbacks retain coordinate probing because their physical
  parameters cannot be inferred.
- Schur-sector preconditioners omit off-sector couplings only inside the
  preconditioner. Validate the final state with the raw Liouvillian residual.

### Parameter scans and continuation

- `ParameterScanPlan` owns copied parameters and immutable choices, not
  numerical scratch. `ParameterScanWorkspace` is task-local.
- A normal `parameter_scan` starts fresh. Only `resume_parameter_scan` installs
  a validated checkpoint seed. A failed point clears the warm start.
- Restart vectors, Ritz matrices, and recycled GCRO subspaces are copied at
  ownership boundaries and revalidated under every changed operator.
- Threaded/distributed scans require `continuation=false`. Threaded callbacks
  remain ordered with bounded buffering. Distributed chunks are deterministic
  but do not provide early remote cancellation.
- `save_outputs=false` must genuinely stream rather than retain state payloads.
- A scan uses one shared memory budget across compilation, workers, solver
  storage, outputs, and restart state. Count immutable family geometry once per
  process and mutable solver/application storage per worker.

### Convergence studies

`convergence_study` records refinement evidence, not an extrapolation or
solver certificate. Require the requested consecutive comparisons and reject
an unconverged inner result in that window. Across different retained bases,
compare a common observable, reduced state, or explicit embedding rather than
raw coefficient vectors. Sampling error, time-step bias, hierarchy depth,
Krylov dimension, and finite-size scaling are separate claims.

### Floquet and response

- `FloquetMap` is the reusable matrix-free period operator. Its adjoint is the
  reverse of the actual finite-step RK graph.
- `selected_floquet_multipliers` supports ordinary Arnoldi, IRAM,
  Jacobi--Davidson, and block Arnoldi (`:block_arnoldi`/`:block`).
- `FloquetBatchWorkspace` has fixed capacity. Forward mode retains only forward
  RK storage; full mode retains exact discrete-adjoint storage. An oversized
  batch raises instead of growing hidden arrays.
- `floquet_gap` must identify `mu=1`, reject an unstable remaining radius, and
  distinguish residual-certified partial estimates from complete global gaps.
- `ResponseWorkspace` separates linear and evolution storage. Resolvent norms
  solve both shifted systems and are converged estimates, not certified upper
  bounds. Singular or unconverged shifts raise.
- Correlation-time and susceptibility solves use physical trace-fixed tangent
  equations and report unscaled generator residuals.

## Symmetries and uniqueness

- Weak unitary symmetry projectors are complex-linear conjugation-charge
  projectors. Antiunitary covariance alone does not define such a block.
- LAPACK-backed symmetry projectors preserve `ComplexF32` or `ComplexF64`
  throughout lifting, masks, workspaces, and residual probes. Unsupported
  scalar types raise with explicit conversion guidance. Exact in-place
  projection is allowed; distinct overlapping views are not. Mixed-precision
  joint construction and exact/probed residual certification retain the
  least-precise source and working-dimension roundoff floors.
- `diagonal_symmetry_restriction` certifies separate strong ket/bra charges by
  exhaustive leakage checks. Compatible fixed prepared kernels lower directly
  into reduced Cartesian coordinates; unsupported kernels and non-Cartesian
  masks use the certified embedded fallback.
- Automatic strong-symmetry discovery is term resolved: every Hamiltonian and
  effective jump must commute separately. Its support-nullspace completeness
  claim is limited to diagonal binary-sign candidates with supported
  microscopic matrices. Reduction must retain every trace-bearing Hilbert
  charge explicitly; never select one stationary charge or infer strong
  commutation from weak covariance. Aggregate multi-charge guards count all
  retained outputs and per-sector solver peaks. Equal-charge workflows omit
  off-diagonal ket/bra decay blocks and must never call their spectral union
  global.
- Off-diagonal ket/bra blocks may have zero trace and are valid for decay modes,
  but trace-fixed stationary solving must reject them.
- Always validate a reduced mode with `restriction_full_residual`; that check
  intentionally needs ambient scratch.
- `matrixfree_symmetry_projector` results are charge-resolved. They may differ
  from the global gap.
- Evans reports use `true`, `false`, or `missing`. `missing` is inconclusive,
  not evidence for or against uniqueness. Weak covariance is not the same as
  strong term-by-term commutation.

## PI--HEOM and pseudomodes

`HEOMBath` stores a fixed Hermitian PI coupling and finite exponential left and
right bosonic correlations. Every pole must have finite positive real decay.
The low-level constructor does not infer a spectral density.

By default it completes the physical conjugate correlation: real poles use
conjugate coefficients, exact conjugate-pole pairs are cross-paired, and a
missing conjugate pole is appended with zero left coefficient. Explicit
`right_coefficients` instead supplies the same-pole right coefficients and
disables completion; an inconsistent explicit pair is never repaired.

- `drude_lorentz_bath` implements the documented Drude convention with
  Matsubara or Hu--Xu--Yan Padé terms. Padé uses Float32/Float64 LAPACK.
- `underdamped_brownian_bath` uses its separately documented lambda-squared
  convention. Do not conflate the two meanings of `lambda`.
- `HEOMPlan(max_depth=D)` retains occupation vectors with total depth at most
  `D`; with `K` terms the complete count is `binomial(K+D,D)`. Check ADO and
  coordinate counts before allocation.
- The root ADO is the reduced physical state. Auxiliary ADOs are generally not
  normalized, Hermitian, or positive and must be returned as `PIOperator`s.
- `terminator=:residue` applies only a stored checked residue. Importance
  pruning must be deterministic and downward-closed. Neither is an error
  certificate; converge decomposition, cutoff, and depth separately.
- `scaling=:scaled` is an exact diagonal similarity transform. Public ADO
  extraction returns unscaled physical coordinates and checks representable
  standalone scales.
- Plans store packed hierarchy edges and CSR-style incidence, not dense
  `nADO`-by-`K` neighbor tables. Forward and adjoint routes reuse bath-grouped
  contractions and bounded system batches.
- `HEOMWorkspace` pre-grows matrix-free PI system batch scratch to
  `min(nADO*batch_columns,16)` columns, with overflow-safe setup arithmetic.
  A default workspace uses `batch_columns=1`. Matrix right-hand sides flatten
  ADO and RHS columns, cap chunks at the retained capacity, and prepare driven
  system schedules once for the complete application; they must never grow
  batch storage while applying. Keep resource estimates synchronized with the
  retained capacity.
- `HEOMBlockPreconditioner` uses guarded Schur shifts for supported standard
  complex precision and LU fallbacks otherwise. Only the Schur backend has
  lock-protected shared scratch and serializes concurrent application; LU-only
  preconditioners do not share that scratch.
- `heom_steady_state` is autonomous-only and does not prove uniqueness or
  hierarchy convergence. Thermal relaxation/stationary preparation is not an
  imaginary-time HEOM certificate of interacting equilibrium.

### Hierarchy of pure states

`HOPSBath` stores a fixed PI coupling and one finite exponential correlation.
Unlike `HEOMBath`, its coupling may be non-Hermitian and it has no independent
right-correlation or white-residue field. `HOPSPlan` implements the linear
HOPS equation for shared Gaussian baths in the direct sum of retained Schur
irreps.

- Exact PI HOPS requires a common bath noise multiplying a PI coupling.
  Independent identical local colored noises break permutation symmetry on
  each realization and are not weak-PI pseudo-ket paths; use PI--HEOM or local
  pseudomode supersites.
- Each hierarchy node has `weak_pi_dimension(basis)` amplitudes, not
  `length(basis)` density coordinates. Plans reuse the packed HEOM hierarchy
  topology and exact scaled/unscaled similarity factors. Workspaces own all
  hierarchy, RK4, coupling, and noise scratch and are task-local.
- Apply each Schur block to all hierarchy nodes as one matrix--matrix kernel
  and retain bath-grouped edge lists. Reuse one hierarchy-sized action buffer:
  self-adjoint couplings combine both edge directions in one pass;
  non-Hermitian couplings apply `L`, accumulate downward edges, overwrite the
  buffer with `L'`, and then accumulate upward edges.
- Linear-HOPS roots are unnormalized. The physical estimator averages
  `|psi_0><psi_0|` before any normalization. Never normalize individual roots
  or average amplitudes when reconstructing a density operator.
- `PIUnitaryPulse` retains physical Schur blocks, never a PI-coordinate
  Kronecker superoperator. `HierarchyPulseSequence` events use `(start,stop]`
  semantics and act on every HEOM ADO or HOPS auxiliary while preserving
  hierarchy and colored-noise memory. Pulse-aware drivers split a fixed step
  at the exact event time. Never emulate a pulse by restarting from only the
  HEOM root or HOPS root pseudo-ket.
- `tetrahedral_pulse_sequence`, `octahedral_pulse_sequence`, and
  `icosahedral_pulse_sequence` implement the published 24-, 48-, and
  120-event Eulerian Cayley words with one equal free interval before every
  pulse and the cyclic pulse at the final endpoint. Each schedule retains
  references to only two immutable axis--angle pulses. These constructors
  model instantaneous control; finite-pulse Eulerian robustness requires an
  explicit time-dependent control Hamiltonian.
- The built-in stationary complex Ornstein--Uhlenbeck path is valid only when
  every retained exponential coefficient is real and nonnegative. A general
  complex/signed decomposition requires an explicit deterministic noise
  provider for the total covariance. Never take independent complex square
  roots of arbitrary residues.
- Combine exact duplicate poles only within the same bath, preserve
  first-occurrence order, and remove exact cancellations before constructing
  the hierarchy. Never combine equal poles belonging to independent baths.
- A positive importance cutoff may retain a hierarchy whose exact unpruned
  count exceeds `Int`. Bound both retained nodes and each pruning frontier
  from the memory budget before allocation; the retained order ideal remains
  deterministic and downward-closed.
- Fixed-step RK4 samples each prepared noise at the start, midpoint, and end
  of a step. Random draws occur in path preparation/advancement, never inside
  `hops_rhs!`; repeated conditioned RHS calls must be deterministic.
- BigFloat plans retain their required precision and rounding context.
  Workspace construction, conditioned application, OU advancement, root
  extraction, and ensemble reduction must re-enter that context rather than
  allocate at the caller's ambient precision.
- The hard hierarchy boundary and optional downward-closed importance cutoff
  are approximations. Converge time step, depth/cutoff, pole decomposition,
  and independent trajectory count separately.

### Finite local pseudomode supersites

`PISupersite` groups the physical system and all of its identical local
auxiliaries into one PI particle. Its internal factor order follows
`kron(factor1,factor2,...)`, with the last factor fastest. A pseudomode
supersite always labels the system first and the finite modes after it. This is
not `CompositePIBasis`: the latter tensors complete global operator spaces and
does not encode the pairing of system `i` with mode `i`.

- With system dimension `dS` and mode levels `r_mu`, the PI local dimension is
  `D=dS*prod(r_mu)` and the complete coordinate count is
  `binomial(N+D^2-1,N)`. Enforce construction and local-lift memory budgets;
  never form `D^N` or `D^(2N)`.
- `lift_system_pbody_operator` must interleave the auxiliary identity inside
  every particle, preserve exact sparse support, and use the ordering
  `(system1,modes1) tensor ... tensor (systemp,modesp)`. The grouped matrix
  `kron(Hsystem,Iaux^tensorp)` has the wrong ordering.
- `BosonicPseudomode` is a finite-cutoff specification. Frequency may be
  signed in a rotating frame; damping and thermal occupation are finite and
  nonnegative. Its rates use the package's standard dissipator, so mode
  amplitude decays at half the `damping` value.
- `PseudomodeCoupling` represents
  `g*L*a' + conj(g)*L'*a` plus the optional counter-rotating
  `h*L*a + conj(h)*L'*a'`. Complex strengths lower into Hermitian
  quadratures with real scalar rates. Do not combine them into a complex
  Hamiltonian rate or hide them inside a preweighted scan-dependent matrix.
- `pseudomode_model` keeps mode frequencies, couplings, damping, and lifted
  system terms separate so ordinary compilation can fuse compatible kernels
  and scalar-rate families can reuse geometry. Direct PI terms and operator
  schedules cannot be inferred as bare-system matrices by
  `lift_system_term`. Its prepared-site method must preserve exact basis
  identity and may override frequency, damping, and thermal-occupation values
  without rebuilding the cutoff layout. Fixed zero components are omitted by
  default; `retain_zero_terms=true` is the explicit prototype route when a
  later `compile_family` specialization must vary them.
- Independent mode loss/gain is a `LocalJump` and generally couples Schur
  sectors. The complete supersite basis is the normal route; any explicit
  sector restriction must pass the ordinary `PIModel` reachability checks.
- Product-state helpers tensor only one supersite and then call `iid_state` or
  `iid_pure_state`; `supersite_iid_state` also accepts an already correlated
  local system--auxiliary ket or density matrix. `pseudomode_trace_plan`
  groups all trailing modes and uses `LocalFactorTracePlan` to return a
  complete system-only PI basis. Reuse the read-only plan and one task-owned
  workspace for each exact cutoff/basis.
- A small top-level population is a useful cutoff warning, not a convergence
  certificate. Compare common system observables or mode-traced states across
  cutoffs; raw supersite coefficient vectors belong to different bases.

The global-ADO backend is limited to bosonic Gaussian HEOM with fixed
Hermitian global PI couplings. Fermionic sign hierarchies and non-Hermitian
coupling pairs are unsupported. Independent local baths are not represented by
one collective coupling because that would introduce cross-correlations.
Finite local pseudomodes are the safe PI-supersite route when a positive
damped-mode realization exists; they support several modes per system, but
every oscillator cutoff must be converged separately.

### One shared/global pseudomode

A mode shared by the complete ensemble is a finite global factor in
`CompositePIBasis(system_basis, mode_basis)`. It must not be placed inside a
`PISupersite`, because that would replicate the mode once per particle and
change the reservoir correlations. Conversely, a local mode paired with every
system must not be replaced by one global factor merely because the latter has
fewer coordinates.

- `GlobalPseudomodeModel` keeps the PI system as the first and fastest factor
  and one `FiniteOperatorBasis` mode as the second. Its coordinate dimension is
  `length(system_basis) * (nmax+1)^2`.
- `PseudomodeCoupling` still receives a one-particle matrix. The global builder
  lifts it to the collective sum and inserts no Kac, square-root-`N`, or other
  particle-number scaling.
- `background` excludes the explicit `damping_channels`; `generator` includes
  both. Pass the former plus the latter to composite trajectories so damping is
  counted exactly once.
- `global_pseudomode_workspace` is mutable task-owned scratch.
  `global_pseudomode_matrixfree` owns or reuses one synchronized workspace;
  parallel hot loops require one explicit workspace per task.
- Stationary solving is GMRES-only (including `AutoAlgorithm`); direct, dense,
  and shift-invert routes must reject rather than probe/materialize the full
  composite map. Workspace preflight includes every nested PI workspace and
  predictable fallback action transient. The matrix-free wrapper also charges
  its retained sparse trace-functional copy. BigFloat stationary and spectral
  solvers must execute entirely in the model's retained precision context,
  not only wrap individual operator applications.
- High-level selected spectra insert the factorized matrix-free wrapper before
  Arnoldi-family solving. Dense complete spectra of the global composite map
  remain unsupported.
- Product states and model-aware reductions preserve the prepared scalar and
  BigFloat context. Wider state inputs require rebuilding the model; never
  return composite coordinates which its prepared workspace cannot apply.
- `GlobalPseudomodeModel` retains one immutable `CompositeReductionPlan` for
  the system and one for the mode. Reuse those packed diagonal contractions;
  do not rebuild sector groups or a composite trace functional on every
  reduction.
- `trace_pseudomodes(rho,model)` contracts the global mode factor and returns
  the system `PIState`; `global_pseudomode_state` contracts the system factor.
  Do not use the local-supersite trace plan for this topology.
- The top-projector population belongs to the one shared mode and is not
  divided by `N`. Converge the cutoff using common observables or reduced
  states in addition to this boundary diagnostic.

## Deterministic and stochastic dynamics

### Deterministic evolution and streaming

Compile once before repeated evolution. Reuse `EvolutionWorkspace` with
`evolve!` for fixed-step RK propagation, `solve_dynamics` for typed high-level
results, and `dynamics_problem` for adaptive/stiff SciML integration.

`solve_dynamics(...; observables=...,save_states=false)` retains one mutable
state and returns observable series. Reject a state-free call with no requested
observable.

`ExpvAlgorithm` is an explicit autonomous-only high-level route. It reuses one
`KrylovExpvWorkspace` and one task-owned source-action workspace across output
intervals. A rejected slice must reuse its Arnoldi factorization and reevaluate
only the projected exponential; an accepted slice changes the state and
invalidates that factorization. Keep `:auto` on RK4 until a benchmarked
crossover policy is introduced. Reject driven generators and parameter-bearing
expv calls rather than freezing them implicitly.

### Density-valued quantum jumps

- `TrajectoryPlan` owns fixed operator/channel lowering. Workspaces and RNGs
  are task-owned; batch streams are indexed by global trajectory number.
- Fixed RK4 and adaptive/event-driven DOPRI use channel-resolved gain maps and
  preallocated hazard data. A task-owned effective-loss node cache evaluates
  scalar jump rates once per distinct physical node, reuses the same values for
  channel selection, and publishes validity only after successful preparation.
  Autonomous jump losses may remain cached; driven caches must be invalidated
  at every public path boundary so changed parameters cannot become stale.
  Require finite nonnegative stochastic rates.
- Local particle labels are unresolved. A density-valued local-jump path can be
  mixed even though its ensemble reproduces the PI master equation.
- State-free ensembles use prepared observable tuples and online Welford
  statistics. Do not create sampled `PIState`s in the hot loop. Disable pooled
  waiting-time storage when it is unnecessary.
- `trajectory_steady_state` streams post-burn-in samples, first averages within
  each path, then combines at least two independent path means. It reports
  sample uncertainty, residual, and trace error but never a convergence or
  uniqueness certificate.
- `BatchedConditionalPlan` shares term-resolved lowering across a matrix RHS.
  Its workspace has immutable capacity and must match repeated scalar drift,
  RK4, channel-intensity, and gain application. It does not choose time steps;
  callers may batch only paths already assigned the same physical time and
  step, and must preserve one RNG stream per global trajectory index.

### Weak-PI pseudo-kets

The weak-PI backend lives in `directsum_nu U_nu`, not the labeled-particle
Hilbert space. A sector slice represents

`C_nu = psi_nu * psi_nu' / sqrt(f^nu)`.

Relative phases between sectors are unphysical. Conversion from a density
state must reject any occupied multiplicity-weighted block whose numerical
rank exceeds one. Never normalize a pseudo-ket silently.

- Local one-body gains use checked one-box Kraus subduction and exact
  multiplicity ratios. Verify `sum K'*K == Q_nu` in every source sector.
- Plans support fixed collective, direct-PI, collective-p-body, and one-body
  `LocalJump` channels. Operator schedules, `LocalPBodyJump`, and composite
  pseudo-kets remain unsupported.
- Fixed and adaptive paths combine channel loss blocks before one sector
  matvec; only the selected event evaluates individual Kraus branches.
- Preserve the tuple-recursive hot kernel used for Julia 1.10 inference.
- `weak_pi_trajectory_steady_state` averages reconstructed sector-diagonal
  density blocks, never amplitudes or cross-sector outer products. Independent
  path means provide the primary uncertainty; optional batch means are only an
  autocorrelation diagnostic.

### Adaptive and diffusive ensembles

Adaptive jump, weak-PI, and diffusive ensembles stop only at deterministic
batch boundaries after all requested Hermitian observables and times satisfy
their simultaneous empirical-Bernstein criterion. The certificate covers
Monte Carlo sampling only. It does not cover integrator, finite-time,
unravelling, model, or hierarchy error. Reaching the path limit returns
`converged=false`.

Diffusive monitoring supports collective homodyne/heterodyne channels. The
unconditional model must already contain the Lindblad dissipator; monitoring
does not insert it. Observable streaming is Hermitian-only. Check
Euler--Maruyama step-size convergence and do not treat finite trace as a
positivity certificate.

### Composite systems

`CompositePIBasis` combines exact PI and small finite operator factors without
forming a global Kronecker matrix. Composite traces use joint diagonal
coordinates and exact multiplicity products.

`CompositeTrajectoryPlan` receives a trace-preserving background excluding the
monitored dissipators, then adds those channels exactly once. Do not pass a full
generator plus duplicate monitored jumps. Current composite paths are density-
valued fixed-step trajectories. Composite pseudo-kets, diffusive/event-driven
paths, arbitrary CP gains, Distributed batches, and implicit single-ensemble
reductions are not implemented.

## Correlations, channels, tomography, checkpoints, and control

- `CorrelationPlan` copies insertions once; `CorrelationWorkspace` owns
  propagation and shifted-GMRES scratch. Time-only mode must reject frequency
  operations instead of allocating lazily.
- `two_time_correlation` uses
  `tr(A*exp(L*tau)*(B*rho*R))`; unlike `expectation`, it does not implicitly
  adjoint `A`.
- Resolvent spectra return the connected one-sided transform. The FFT route is
  a finite-window uniform-grid transform, not the infinite-time resolvent.
- Channels, Choi tests, POVMs, and tomography certify only the retained PI
  coefficient algebra; they make no claim about arbitrary non-PI inputs.
- In-place channel application forbids source/destination aliasing.
- Checkpoints preserve exact basis metadata and never normalize or repair
  stored states. HDF5/JLD2 support is optional.
- Implicit steady-state gradients solve trace-fixed tangent equations.
  Checkpointed control gradients require a Hermitian terminal objective and
  time-step convergence.

## Verified studies, counting statistics, and inference

- `PIExperiment` is a deterministic PI-state orchestration layer. Planning
  and explanation are read-only; `verified_solve` retains separate physical,
  solver, and requested refinement evidence and never changes the requested
  algorithm. Constructor inputs with mutable numerical storage are
  defensively snapshotted. A refinement must preflight cumulative retained
  level outputs plus the next solve peak before its first solve. Archives
  contain versioned numerical data, inert flattened verification evidence,
  and provenance, not executable closures, workspaces, or solver state;
  schema 2 remains backward-readable with schema 1.
- `TiltedLiouvillianPlan` derives channel-resolved counting fields from a
  term-resolved `TrajectoryPlan`; never reconstruct channels from a fused
  deterministic kernel. Finite-time MGFs, SCGFs, cumulants, and Legendre data
  must retain their residual, discretization, and boundary diagnostics.
- Bath-correlation fitting reports quadrature, residual, rank, and candidate-
  selection limitations before constructing HEOM/HOPS baths. An unconverged
  or rank-deficient fit is rejected unless the caller explicitly opts into
  that condition for a convergence study.
- Parameter inference keeps parameter bounds, weighting, derivative route,
  solver evidence, Fisher rank, and local-identifiability limitations
  explicit. Every steady-state prediction requires `converged=true`; missing
  or false convergence is not accepted as data. A pseudocovariance for a
  rank-deficient problem is diagnostic, not an uncertainty certificate.

## Optional extension contracts

Only matching `ext/` modules may import weak dependencies. Core algorithms must
not dispatch on the accidental presence of an optional package.

- Tables exposes existing scalar/result columns lazily. It must not solve,
  transform, or expand data. Ordinary `Pkg.test()` covers Tables and
  Distributed because they are in the test target.
- Makie provides argument conversion for already computed result data. It must
  not invoke solvers or numerical transforms.
- Distributed owns one model/trajectory workspace per worker chunk and uses
  index-derived random streams. It has no adaptive stopping protocol and
  currently accepts model inputs rather than compiled task-local plans.
- QuantumCumulants follows the neutral payload contract and supported
  microscopic lowering rules.
- QuantumOptics and QuantumToolbox bridges accept only finite square one-site
  operators and copy their dense or sparse data without narrowing. Reject
  kets, superoperators, rectangular maps, and dimension mismatches; never
  import a many-body state or superoperator through these conveniences.
- JLD2 and HDF5 implement storage for the versioned checkpoint schema without
  changing that schema.
- Makie, QuantumCumulants, QuantumOptics, QuantumToolbox, JLD2, and HDF5
  changes require the isolated `test/optional` smoke suite.

## Documentation and public API

- `docs/src/getting_started.md` is the canonical task-oriented onboarding
  path. Keep its complete script synchronized with
  `examples/getting_started.jl` and the Home-page preview.
- `docs/src/framework.md` is the self-contained conceptual introduction.
- `docs/src/api_reference.md` is the complete alphabetical entry point.
- Every exported binding needs a source docstring and exactly one canonical
  `@docs` entry. `docs/make.jl` enforces zero undocumented exports and
  `checkdocs=:exports`.
- Qualify names that conflict with Base, such as
  `PermutationalInvariantDynamics.isvalid`.
- In tracked Markdown, write inline mathematics with `$...$` and display
  mathematics in fenced `math` blocks. This common syntax renders both in
  GitHub previews and through Documenter's configured KaTeX auto-renderer. Do
  not use double backticks, `\(...\)`, or `\[...\]` for Markdown mathematics.
  Source docstrings remain Julia-native and use double-backtick math; an
  unescaped `$` in an ordinary Julia string would interpolate. GitHub rejects
  `\operatorname`; use `\mathrm{tr}`, `\mathrm{Re}`, `\mathrm{diag}`, and
  analogous roman labels.
- Keep README, `CITATION.cff`, Documenter links, repository URL, and license
  synchronized. Do not add `date-released` before an actual release.
- `scripts/release_gate.jl` is the dependency-free, non-publishing metadata
  and repository-hygiene gate. It may validate a candidate or a dated release,
  but it must never create a tag, GitHub release, or registry request.
- The tracked `docs/Manifest.toml` is resolved for the current stable Julia;
  regenerate it before pinning documentation CI to an older Julia line.
- The README disclosure of substantial Codex assistance must remain. A human
  maintainer must understand every release candidate and obtain a green
  clean-checkout release run before registration.

Documentation deploys with the workflow's repository `GITHUB_TOKEN`.
`DOCUMENTER_KEY` is the write-enabled, repository-scoped deploy key used by
CompatHelper and TagBot so their pushes can trigger downstream workflows;
never commit its private half. Keep GitHub Actions permission to create
CompatHelper pull requests, and configure Pages from the generated `gh-pages`
branch after the first deployment. Coverage upload uses GitHub OIDC; preserve
both `id-token: write` on the test job and `use_oidc: true` on Codecov.

## Examples and published-model conventions

Every runnable `examples/*.jl` file has a same-basename Markdown guide. Keep
code, formulas, finite-size caveats, tolerances, figures, and output stems
synchronized. The exhaustive inventory belongs in `examples/README.md`, not
here.

Durable literature mappings:

- PRA 94, 033838 (2016): correlated emission is
  `gamma*D[J_-] + (gamma0-gamma)*sum_i D[sigma_-^(i)]`. The large pulse uses
  the certified population backend and must not create a `2^N` object.
- PRA 110, 062208 (2024): the dissipative LMG interaction is lowered as a
  one-body self term plus a symmetric two-body cross term with pair rate
  `2V/(N*j)`. Compare the unique finite-`N` parity-symmetric state to parity-
  even observables of a selected mean-field branch; never call the product
  closure exact correlated dynamics.
- The Debecker draft example is the PI uniform-all-pair specialization, not a
  reproduction of the nearest-neighbour periodic chain. One site is spin
  tensor a mode truncated at `nmax`, `Jpair=J/(N-1)`, the manuscript
  dissipator maps to package rate `2kappa`, and `g=sqrt(gamma*kappa)`. The
  longitudinal model has strong spin parity; certify leakage and validate the
  full residual after reduced solving. Its contour fits and tab-delimited
  boundary exports are empirical finite-grid guides, not thermodynamic phase
  boundaries. Preserve their detailed algorithm and metadata contract in the
  paired guide.
- Weak-PI trajectory examples compare ensemble-linear quantities across Kraus
  conventions. Do not identify individual paths with labeled-particle or
  differently rotated unravelings.
- Time-crystal examples are finite-size workflows unless they explicitly show
  size scaling. Preserve paper dissipator and ordered-pair normalization
  conventions stated in each guide.

Literature figures use the separate CairoMakie examples environment:

```sh
julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=examples examples/<name>.jl
```

CairoMakie is not a core dependency. The shared loader activates it only when
it is a direct dependency of the active project. Without CairoMakie, examples
must still run numerical assertions and skip only rendering. Keep plotting
after validation and render arrays already produced by the checked workflow.
`test/run_quick_examples.jl` executes the representative CI suite in fresh
Julia processes with `PID_EXAMPLE_RENDER=0`. Quick-suite selection must not
weaken an included example's numerical parameters, tolerances, or assertions.

## Verification workflow

Choose checks by changed surface. The following commands mirror the important
CI environments.

### Core package

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

For a focused local pass, set `PID_TEST_GROUPS` to one or more comma-separated
groups listed in `CONTRIBUTING.md`, for example
`PID_TEST_GROUPS=solvers,workflows`. The default and `all` retain the complete
suite and its normal file order. Focused groups are a development aid, not a
replacement for the complete release gate.

Run on Julia 1.10 and the current stable Julia before a release or after a
compiler-sensitive change.

### Threading and performance safeguards

```sh
JULIA_NUM_THREADS=4 julia --project=. benchmark/performance_regression.jl
julia --project=. benchmark/performance_audit.jl
julia --project=benchmark benchmark/cold_start.jl --mode quick
julia --project=benchmark benchmark/time_to_solution.jl --mode quick
julia --project=benchmark benchmark/batched_trajectories.jl
```

The regression script is the CI allocation/equivalence/thread-safety gate. The
global performance audit is a broader manual check; it deliberately avoids
brittle wall-clock thresholds. Cold-start measurements use fresh Julia
processes. Time-to-solution records setup, solve, validation, output, and
provenance separately; its analytic validation is mandatory. Run the
CI-parity threaded gate with Julia 1.10.

For reproducible internal scaling and cross-package comparisons, follow
`benchmark/README.md` and `benchmark/comparison/README.md`. Keep external
packages in isolated projects and processes. Compare collective dynamics only
after restricting the PID basis to the same spin irrep; independent local
jumps require the complete PI basis and full-Hilbert small-`N` references.
Record setup, hot application, retained memory, representation, provenance,
and numerical validation separately. Wall-clock results are never CI gates.
Internal-scaling schema 2 additionally isolates sparse-first and driven
collective phases for symmetric qubit and qutrit sectors, records structured
support estimates and a fixed-budget `:auto` probe, and uses `NA` for those
fields on all-sector rows. Cross-package adapters must reuse an existing CSC
matrix without copying it, identify the backend/action explicitly, and keep
the matched collective size grid synchronized across every adapter and guide.

### Documentation and quality

```sh
julia --project=docs -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
julia --project=docs docs/make.jl
julia --project=quality -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
julia --project=quality quality/quality.jl
```

Documentation CI uses the current stable Julia. The isolated quality project
requires Julia 1.12 or later even though the package itself supports Julia
1.10.

### Optional extensions

```sh
julia --project=test/optional -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
julia --project=test/optional test/optional/runtests.jl
```

Run the CI-parity optional suite with Julia 1.10.

### Examples

```sh
julia --project=. examples/getting_started.jl
julia --project=. examples/<changed-example>.jl
```

Use `--project=examples` when validating CairoMakie rendering. Refer to
`examples/README.md` rather than maintaining a second command inventory here.

### Representation-theory changes

1. Add a minimal identity or orthogonality test.
2. Compare with an analytical qubit case.
3. Add a small qudit case with a known result.
4. Run the complete suite.
5. Update mathematical docs and `IMPLEMENTATION_NOTES.md`.

### Handoff checklist

- Run tests proportional to every touched subsystem.
- Run documentation when exports or docstrings change.
- Run ordinary `Pkg.test()` for Tables or Distributed changes; run
  `test/optional` for Makie, QuantumCumulants, JLD2, or HDF5 changes.
- Run `git diff --check` and inspect `git status --short`.
- Remove generated root, `quality/`, `examples/`, `benchmark/`,
  `benchmark/comparison/*/`, `test/optional/`, and `notebooks/` manifests;
  preserve `docs/Manifest.toml`.
- Remove generated figures, checkpoints, and scan outputs unless they are
  intentional tracked fixtures.
- Confirm no unrelated user changes were overwritten.
- Report actual commands and outcomes, including checks not run.

## Known bounded limitations

- Sparse-SPQR LR nullspace setup and the public dense
  `subduction_intertwiners` output can exhaust memory for very large qudit
  irreps, even though prepared `ReductionPlan` intertwiners are packed.
- One extremely large Schur block still needs dense factorization for several
  PSD and spectral operations.
- Default uncached CG queries preserve the established real Float64 phase
  convention; explicit typed caches/geometries support checked wider
  precision. Do not describe CG geometry as globally Float64-only.
- Complete Liouvillian spectra and direct complete Floquet-map visualization
  are PI-dimensional dense operations. Selected Liouvillian and Floquet
  Arnoldi/IRAM/JD/block-Arnoldi routes are matrix-free.
- Qudit normalized-Haar Husimi-Q is implemented; generalized qudit Wigner and
  a dependency-free qudit-manifold renderer are not.
- Global PI--HEOM does not support fermionic sign hierarchies, non-Hermitian
  coupling pairs, or independent local baths. Linear PI--HOPS supports
  non-Hermitian shared-bath couplings but not independent local colored-noise
  realizations or a normalized nonlinear/Girsanov hierarchy. Finite local
  pseudomodes use the PI-supersite route.
- Composite stochastic evolution is density-valued fixed-step only; composite
  weak-PI pseudo-kets and diffusive/event-driven composite paths are absent.
- Evans certificates return `missing` for unsupported microscopic
  recouplings, custom/direct terms, or an exceeded memory budget.
- A coefficient-space trace functional cannot be stored when required
  `sqrt(f^nu)` entries exceed the chosen scalar range; compile at wider
  precision.
- Resource estimates cannot bound arbitrary user builders, callbacks, or
  diagnostic payloads. Preserve explicit assumption/exclusion metadata.

Consult `IMPLEMENTATION_PLAN.md` for research-scale extensions and
`IMPLEMENTATION_NOTES.md` for algorithmic rationale and measured optimization
history.

## Maintenance rules

- Keep representation-theory code explicit and commented; avoid opaque
  metaprogramming.
- Restricted bases must error when a requested local process reaches a missing
  sector.
- Give randomized Krylov and symmetry comparisons explicit initial vectors and
  fresh deterministic RNGs. Restart/breakdown recovery may consume RNG state.
- If a phase or equation convention is uncertain, re-read the source paper and
  add a small mathematical test. Never choose a convention only to satisfy an
  existing test.
- Update this file only when a durable repository contract changes. Put API
  tutorials in docs, example-specific procedures in paired guides, benchmark
  history in `IMPLEMENTATION_NOTES.md`, and current measurements in benchmark
  scripts.
