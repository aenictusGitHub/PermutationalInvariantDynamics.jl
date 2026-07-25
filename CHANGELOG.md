# Changelog

All notable public changes to `PermutationalInvariantDynamics.jl` are recorded
here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the package uses [Semantic Versioning](https://semver.org/).

## [0.1.0] - Unreleased

Initial public release candidate.

### Added

- General qubit and qudit permutationally invariant Schur-block states,
  operators, local/collective/p-body generators, and matrix-free dynamics.
- Prepared deterministic, stochastic, Floquet, mean-field, population,
  spectral, response, information-theoretic, entanglement, and visualization
  workflows.
- Prepared parameter scans with continuation, restart, streaming records, and
  optional threaded or distributed execution.
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
  identical-local-pseudomode, and shared-global-pseudomode stationary
  workflows, including memory guards and oscillator-cutoff diagnostics.
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

### Changed

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
- Schur-sector GMRES preconditioners now lower diagonal blocks directly from
  prepared kernels, including specialized family rates, while plan-less
  callbacks retain the probing fallback.
- Appendix-D path isometries and qudit reduction intertwiners now retain exact
  sparse support. Model preflights include every distinct p-body geometry,
  driven cancellation checks traverse packed support with preallocated
  scratch, and product-irrep dimensions use checked indexing arithmetic.

### Notes

- This version has not yet been released or registered. Replace `Unreleased`
  with the release date only after the exact release commit passes the clean
  release gate; add `date-released` to `CITATION.cff` at the same time.

[0.1.0]: https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/releases/tag/v0.1.0
