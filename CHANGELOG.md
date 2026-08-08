# Changelog

All notable public changes to `PermutationalInvariantDynamics.jl` are recorded
here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the package uses [Semantic Versioning](https://semver.org/).

## [0.1.0] - Unreleased

Initial public release candidate.

### Added

- General qubit and qudit permutationally invariant Schur-block states,
  operators, local/collective/p-body generators, and matrix-free dynamics.
- Compact orthogonal projectors onto arbitrary retained Schur isotypic sectors
  and the fully permutation-symmetric Hilbert-space sector.
- Short exact constructors for symmetric qudit occupation states,
  excitation-count Dicke and W states, qudit cat states, and normalized white
  states confined to one Schur sector or the fully symmetric sector.
- Physical pure-ket storage and preallocated Hamiltonian evolution in the
  sole fully symmetric irrep, including sparse one-body lifting, RK4 and
  Krylov exponential actions, direct expectations, rank-one density
  conversion, and ket-native local-factor traces.
- Certified Hilbert-block von Neumann entropy plans built from explicit
  partitions or diagonal strong symmetries, with strict off-block checks,
  projected-interpretation diagnostics, and blockwise cubic-work estimates.
- Stable full-state and reduced density moments of arbitrary positive integer
  order, with exact Schur-multiplicity scaling, binary block powering, and
  reusable memory-budgeted scratch; purity remains the dedicated second-order
  fast path.
- Prepared deterministic, stochastic, Floquet, mean-field, population,
  spectral, response, information-theoretic, entanglement, and visualization
  workflows.
- Prepared parameter scans with continuation, restart, streaming records, and
  optional threaded or distributed execution.
- A curated `Workflow` namespace, convention-tested `Models` recipes, named
  affine generator families, and strict one-site operator bridges for
  QuantumOptics.jl and QuantumToolbox.jl.
- A guided `PIStudy` front door with explain/check preflight,
  machine-readable diagnostics, a bounded `doctor()` smoke test, uniform
  result accessors, and preservation of the underlying expert result.
- Dependency-free progress events and cooperative cancellation for scans,
  fixed-step dynamics, HEOM, and individual HOPS paths, including resumable
  ordered scan prefixes.
- Compact result summaries and tables, common CSV/TSV and staged `.pidrun`
  exports, and optional Tables, Makie, JLD2, and HDF5 result integrations.
- Reproducible `PIExperiment` planning and verified deterministic solves with
  separate physical, solver, and refinement evidence plus a versioned,
  non-executable result archive.
- Matrix-free counting statistics and rare-event analysis, bath-correlation
  fitting for guarded HEOM/HOPS construction, and bounded parameter inference
  with implicit stationary sensitivities and identifiability diagnostics.
- Immutable prepared-geometry bundles, shared multi-bipartition reduction
  plans, and an explicit user-owned, memory-budgeted preparation cache.
- Fixed-capacity matrix-RHS conditional trajectory kernels with grouped jump
  gains and trajectory-index-stable random streams.
- Block, multi-shift, and recycled GMRES, plus adaptive matrix-free Krylov
  exponential actions and an autonomous high-level `ExpvAlgorithm` dynamics
  route with saved-state or observable-only output.
- PI hierarchy equations of motion for finite-exponential bosonic bath
  correlations, including propagation, stationary solving, and hierarchy-depth
  convergence reports.
- Linear PI hierarchy-of-pure-states trajectories for shared
  finite-exponential Gaussian baths, with direct-sum Schur amplitudes, exact
  scaled hierarchy coordinates, stationary Ornstein--Uhlenbeck noise,
  preallocated RK4 workspaces, and unnormalized density-ensemble
  reconstruction.
- Hierarchy-preserving ideal control pulses for PI--HEOM and PI--HOPS,
  including the published tetrahedral, octahedral, and icosahedral Eulerian
  dynamical-decoupling sequences.
- Exact paired PI supersites for identical systems with one or more
  finite-cutoff local pseudomodes, including automatic system-term lifting,
  matrix-free model construction, product states, and prepared mode tracing.
- One shared finite-cutoff global pseudomode as a factorized PI-system × mode
  composite, with collective rotating/counter-rotating coupling, thermal
  damping, deterministic and trajectory generators, matrix-free
  GMRES/Arnoldi workflows, and direct reduced states.
- The browser-only typed model-code generator now emits ordinary PI,
  identical-local-pseudomode, and shared-global-pseudomode workflows,
  including memory guards, oscillator-cutoff diagnostics, deterministic
  parameter scans, manifest round trips, browser-local autosave/share/undo,
  and auditable Pluto notebooks.
- Density-valued composite quantum-jump systems with explicit cross-factor
  channels, factorized gain/loss application, reproducible threaded batches,
  and online trajectory statistics.
- Exact-support sparse local-factor trace plans, packed
  `CompositeReductionPlan`s, preallocated `one_body_rdm!`, and
  discarded-factor-sliced particle reductions for repeated partial traces
  without full product-block scratch on reduced-state paths.
- Automated time-step, Krylov-dimension, hierarchy-depth, and sector-cutoff
  convergence studies; confidence-controlled quantum-jump and diffusive
  ensembles.
- Generalized qudit Husimi-Q data and optional Tables, Makie,
  QuantumCumulants, Distributed, JLD2, and HDF5 integrations.
- Literature validation examples, same-basename guides, package documentation,
  curated expected-output figures for every example guide, release automation,
  and a Pluto research-workflow notebook.
- A complete machine-readable example inventory, read-only model-recipe
  discovery, a searchable expected-output gallery, an architecture chooser,
  and short task-oriented installation and quickstart pages.
- Machine-readable example discovery, verified browser-generator bundles,
  fresh-process cold-start and complete time-to-solution harnesses, an optional
  PackageCompiler workload, and guarded accelerator capability/resource
  preflight without claiming an untested CUDA backend.

### Changed

- Literature figures now use resolved research grids and larger finite sizes,
  while `PID_EXAMPLE_QUICK=1` preserves lightweight executable checks. The
  boundary time-crystal scaling uses targeted matrix-free spectral solves for
  `N` through 40, and repeated rate scans reuse compiled Schur geometry.
- The identical-local-pseudomode example is now a self-contained generic
  all-to-all model with public pseudomode provenance; private-draft framing and
  obsolete named output assets were removed throughout the repository.
- Fixed one-body Schur lifts and compatible Hamiltonian/loss fusion now build
  exact sparse CSC support without dense sector-block intermediates.
- Density and weak-PI trajectories cache effective jump losses at distinct RK
  nodes, reusing endpoint data and autonomous rates while invalidating driven
  caches between public paths.
- Mixed-sector collective-only models retain diagonal one-box contractions
  instead of the sector-changing geometry required by local gain channels.
- Sensitivity systems, composite superoperators, and PI--HEOM generators now
  apply matrix right-hand sides through bounded, reusable batch workspaces;
  driven schedules are evaluated once per batched HEOM action.
- Prepared autonomous PI sources can opt into sector-parallel Krylov actions
  through a synchronized wrapper around task-owned threaded workspaces.
- Schur-sector GMRES preconditioners now lower diagonal blocks directly from
  prepared kernels, including specialized family rates, while plan-less
  callbacks retain the probing fallback.
- Appendix-D path isometries and qudit reduction intertwiners now retain exact
  sparse support. Model preflights include every distinct p-body geometry,
  driven cancellation checks traverse packed support with preallocated
  scratch, and product-irrep dimensions use checked indexing arithmetic.
- Counting-statistics source wrappers and SCGF curves now include hidden plan
  retention and retained reports in their memory guards; finite-time MGFs copy
  a tilted state only when `return_info=true`.
- Bath fitting infers precision from explicit floating-point fit controls. Its
  precision-local default is now expressed as `rtol=nothing` (still `1e-6` in
  the inferred type), avoiding accidental `Float32` widening while rejecting
  silent narrowing.
- Accelerator preflight accepts an explicit `rhs_kind`, reports the selected
  vector/matrix representation, and rejects backends whose transfer policy is
  not one explicit upload.

### Notes

- This version has not yet been released or registered. Replace `Unreleased`
  with the release date only after the exact release commit passes the clean
  release gate; add `date-released` to `CITATION.cff` at the same time.

[0.1.0]: https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/releases/tag/v0.1.0
