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
- Current performance gates and audit: `benchmark/performance_regression.jl`
  and `benchmark/performance_audit.jl`.

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
- Weak dependencies: `Distributed`, `Tables`, `Makie`, `QuantumCumulants`,
  `JLD2`, and `HDF5`. Their methods belong in matching `ext/` modules.
- Current extension compatibility is Tables 1, Makie 0.21--0.24,
  QuantumCumulants 0.5, JLD2 0.4--0.5, HDF5 0.16--0.17, and the supported
  Julia 1.10+ `Distributed` stdlib. Update compat and optional smoke tests
  together.
- `Distributed` can load transitively through SciMLBase. Extension activation
  is not proof that the user requested remote workers.
- Do not commit generated root, `quality/`, `examples/`, `test/optional/`, or
  `notebooks/` manifests. Preserve the tracked `docs/Manifest.toml`.
- Public repository:
  `https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl`.
- License: `GPL-3.0-only`; the root `LICENSE` file is canonical.

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

- Mean field:
  `(N,d,terms) -> MeanFieldPlan -> MeanFieldWorkspace -> one-site solver`.
- Certified populations:
  `PIModel -> PopulationPlan -> PopulationWorkspace -> solve_populations`.
- Composite deterministic dynamics:
  `factors -> CompositePIBasis -> CompositeSuperoperator -> workspace`.
- Composite stochastic dynamics:
  `background + monitored channels -> CompositeTrajectoryPlan -> workspace`.
- Parameter scans:
  `parameter grid + builder -> ParameterScanPlan -> workspace -> scan/resume`.
- Scalar-rate families:
  `PIModel -> compile_family -> CompiledPIModelFamily -> specialize`.
- Bosonic PI--HEOM:
  `PI system + HEOMBaths -> HEOMPlan -> HEOMWorkspace -> solve`.

Prefer `compile`, `solve_dynamics`, `stationary_state`,
`liouvillian_spectrum`, `diagnostics`, and `recommend_solver` in research
scripts. Lower-level `liouvillian`, `steady_state`, `apply!`, and Krylov APIs
are advanced interfaces. `docs/src/api_tiers.md` defines the intended stability
tier.

`PIModel.terms` is a concrete immutable tuple. `LiouvillianPlan` owns prepared
read-only blocks, contractions, and rate descriptions. `LiouvillianWorkspace`
owns mutable scratch. Compatibility `mul!`/`action!` calls are synchronized;
parallel hot loops need explicit task-owned workspaces.

The first composite factor is the fastest coordinate. A factorized vector is
`kron(x_last,...,x_first)`, and the factorized map has reversed Kronecker
order. Never materialize the global Kronecker matrix in production.

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
  `compiled_families.jl`. `source_protocol.jl` centralizes private basis,
  trace-vector, task-workspace, and adjoint-capability discovery for prepared
  linear-operator wrappers; extend those traits instead of adding consumer-
  local `isa` cascades. `solver_algorithms.jl` owns canonical solver symbols
  and compatibility aliases. `result_protocol.jl` owns explicit physical-time
  result lookup.
- Krylov, spectra, and symmetries: `krylov.jl`, `krylov_extensions.jl`,
  `spectra.jl`, `evans.jl`, `symmetries.jl`, and
  `restricted_symmetries.jl`.
- State analysis: `observables.jl`, `entanglement.jl`, `information.jl`,
  `symmetry_information.jl`, `populations.jl`, and `research_utilities.jl`.
- Deterministic dynamics and studies: `sciml.jl`, `evolution.jl`,
  `meanfield.jl`, `floquet.jl`, `response.jl`, `correlations.jl`,
  `highlevel.jl`, `scans.jl`, and `convergence.jl`.
- Non-Markovian and stochastic systems: `composite.jl`, `heom.jl`,
  `trajectories.jl`, `composite_trajectories.jl`,
  `weak_pi_trajectories.jl`, `diffusive.jl`, `adaptive_ensembles.jl`, and
  `distributed_api.jl`.
- Research utilities and optional bridges: `cumulants.jl`, `channels.jl`,
  `tomography.jl`, `checkpoints.jl`, and `control.jl`.
- Visualization and phase space: `phase_space.jl`, `qudit_phase_space.jl`,
  `visualization.jl`, `spectral_visualization.jl`, and
  `phase_space_visualization.jl`.

## Compilation, performance, and memory contracts

### Fixed kernels and matrix-free application

- Fixed `LocalJump` and safely scaled fixed `LocalPBodyJump` gains retain
  rectangular Schur contractions. Do not restore quartic PI-coordinate gain
  maps to matrix-free plans. Cancellation-risk p-body cases retain the guarded
  triplet fallback; explicit sparse materialization may expand the factors.
- Fixed one-body contractions may retain exact structural-zero supports when a
  setup-time arithmetic gate predicts a gain. Never use a numerical dropping
  tolerance. Dense fixed and driven contractions keep the dense rectangular
  path.
- Compatible fixed numeric kernels may fuse Hamiltonian blocks and
  anticommutator loss while keeping channel-resolved gains. Preserve concrete
  tuple recursion and the Julia-1.10-stable default fusion dispatch.
- Trajectory, population, channel, and other channel-resolved consumers must
  use term-resolved lowering; never infer physical channels from a fused
  deterministic kernel.
- Batched application evaluates each schedule once and uses sectorwise matrix-
  matrix kernels where supported. Preserve vector/column fallbacks for custom
  callbacks and uncommon driven kernels.

### Threaded application

`ThreadedLiouvillianWorkspace` is an explicit opt-in for one vector action.
Each output Schur sector belongs to exactly one worker; prepared data and input
are read-only, and each worker owns block scratch. Preserve kernel order inside
a sector, evaluate schedules once before spawning, and do not use atomic output
updates or task-ordered reductions. One worker delegates to ordinary `apply!`.
Reject unsupported plan-less callbacks before spawning. Do not change global
BLAS threading. The workspace is guarded against concurrent reuse.

### Resource safeguards

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
- `recommend_solver` separates setup, retained, solve, output, and peak
  estimates. Unknown builder, callback, or diagnostic allocations imply
  `safe_to_run=missing`, never `true`.
- Mode-specific workspaces omit dominant unused arrays and must reject an
  incompatible operation instead of allocating the missing storage lazily.

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
the Kossakowski matrix.

## Observables, information, entanglement, and reductions

- Expectations, entropies, distances, QFI/QFIM, symmetry-resolved information,
  phase space, and reductions should contract multiplicity-weighted blocks.
- Prepare `CollectiveObservablePlan` once per observable and `ReductionPlan`
  once per `(basis,k)`. Plans are tied to the exact basis; workspaces are
  task-owned.
- `ReductionWorkspace` modes (`:reduction`, `:negativity`, `:both`) omit
  incompatible buffers. A mode mismatch must raise rather than allocate.
- A reduction workspace owns scalar-compatible recoupling matrices. Convert
  compact real qubit recouplers once per workspace and share already matching
  qudit LR matrices read-only. Do not restore mixed real/complex `mul!` calls
  that allocate packing scratch on Julia 1.10.
- General negativity has three exact routes: occupation branching for fully
  symmetric states, SU(2) recoupling for general qubits, and LR intertwiners for
  general qudits.
- Product-Schur trace norms and reduced states use the exact weight
  `f^alpha * f^beta`. Qudit LR multiplicities are counted exactly; forbidden
  weights are removed before sparse generator constraints and SPQR nullspace
  recovery.
- Qudit LR setup and retained dense intertwiners can be large. Benchmark a
  plan before caching many bipartitions, but never replace it with full-Hilbert
  reconstruction.
- Keep numerical-rank cutoffs sector-relative. Validate PSD before a rank
  decision. Roundoff handling inside a square root, logarithm, or support
  projector must not modify the input or returned spectrum.

## Visualization and phase space

Numerical extraction and rendering are separate layers. Rendering an existing
result must not trigger a solve, matrix-free probe, phase-space transform, or
full-Hilbert expansion.

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
  incompatible precision rather than narrow.
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
- `KrylovWorkspace` stores real Givens rotations in the real component type
  of its Krylov scalar, never unconditionally in `Float64`; preserve this for
  both lower-precision speed and wider-precision reliability.
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
- `diagonal_symmetry_restriction` certifies separate strong ket/bra charges by
  exhaustive leakage checks. Compatible fixed prepared kernels lower directly
  into reduced Cartesian coordinates; unsupported kernels and non-Cartesian
  masks use the certified embedded fallback.
- Off-diagonal ket/bra blocks may have zero trace and are valid for decay modes,
  but trace-fixed stationary solving must reject them.
- Always validate a reduced mode with `restriction_full_residual`; that check
  intentionally needs ambient scratch.
- `matrixfree_symmetry_projector` results are charge-resolved. They may differ
  from the global gap.
- Evans reports use `true`, `false`, or `missing`. `missing` is inconclusive,
  not evidence for or against uniqueness. Weak covariance is not the same as
  strong term-by-term commutation.

## PI--HEOM and local pseudomodes

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
  `min(nADO,16)` columns so the first hierarchy application does not allocate.
  Keep resource estimates synchronized with this retained capacity.
- `HEOMBlockPreconditioner` uses guarded Schur shifts for supported standard
  complex precision and LU fallbacks otherwise. Only the Schur backend has
  lock-protected shared scratch and serializes concurrent application; LU-only
  preconditioners do not share that scratch.
- `heom_steady_state` is autonomous-only and does not prove uniqueness or
  hierarchy convergence. Thermal relaxation/stationary preparation is not an
  imaginary-time HEOM certificate of interacting equilibrium.

The global-ADO backend is limited to bosonic Gaussian HEOM with fixed
Hermitian global PI couplings. Fermionic sign hierarchies and non-Hermitian
coupling pairs are unsupported. Independent local baths are not represented by
one collective coupling because that would introduce cross-correlations.
`independent_local_pseudomode_model` is the safe finite-cutoff PI-supersite
route for one damped bosonic mode per site; converge `nmax` separately.

## Deterministic and stochastic dynamics

### Deterministic evolution and streaming

Compile once before repeated evolution. Reuse `EvolutionWorkspace` with
`evolve!` for fixed-step RK propagation, `solve_dynamics` for typed high-level
results, and `dynamics_problem` for adaptive/stiff SciML integration.

`solve_dynamics(...; observables=...,save_states=false)` retains one mutable
state and returns observable series. Reject a state-free call with no requested
observable.

### Density-valued quantum jumps

- `TrajectoryPlan` owns fixed operator/channel lowering. Workspaces and RNGs
  are task-owned; batch streams are indexed by global trajectory number.
- Fixed RK4 and adaptive/event-driven DOPRI use channel-resolved gain maps and
  preallocated hazard data. Evaluate scalar rates once per stage and require
  finite nonnegative stochastic rates.
- Local particle labels are unresolved. A density-valued local-jump path can be
  mixed even though its ensemble reproduces the PI master equation.
- State-free ensembles use prepared observable tuples and online Welford
  statistics. Do not create sampled `PIState`s in the hot loop. Disable pooled
  waiting-time storage when it is unnecessary.
- `trajectory_steady_state` streams post-burn-in samples, first averages within
  each path, then combines at least two independent path means. It reports
  sample uncertainty, residual, and trace error but never a convergence or
  uniqueness certificate.

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
- JLD2 and HDF5 implement storage for the versioned checkpoint schema without
  changing that schema.
- Makie, QuantumCumulants, JLD2, and HDF5 changes require the isolated
  `test/optional` smoke suite.

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
- Keep Markdown and docstring TeX compatible with GitHub. GitHub rejects
  `\operatorname`; use `\mathrm{tr}`, `\mathrm{Re}`, `\mathrm{diag}`, and
  analogous roman labels.
- Keep README, `CITATION.cff`, Documenter links, repository URL, and license
  synchronized. Do not add `date-released` before an actual release.
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
```

The regression script is the CI allocation/equivalence/thread-safety gate. The
global performance audit is a broader manual check; it deliberately avoids
brittle wall-clock thresholds. Run the CI-parity threaded gate with Julia 1.10.

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
- Remove generated root, `quality/`, `examples/`, `test/optional/`, and
  `notebooks/` manifests; preserve `docs/Manifest.toml`.
- Remove generated figures, checkpoints, and scan outputs unless they are
  intentional tracked fixtures.
- Confirm no unrelated user changes were overwritten.
- Report actual commands and outcomes, including checks not run.

## Known bounded limitations

- Sparse-SPQR LR nullspaces and retained dense intertwiners can exhaust memory
  for very large qudit irreps.
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
  coupling pairs, or independent local baths. Finite local pseudomodes use the
  PI-supersite route.
- Composite stochastic evolution is density-valued fixed-step only; composite
  weak-PI pseudo-kets and diffusive/event-driven composite paths are absent.
- Evans certificates return `missing` for unsupported microscopic
  recouplings, custom/direct terms, or an exceeded memory budget.
- A coefficient-space trace vector cannot be stored when required
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
