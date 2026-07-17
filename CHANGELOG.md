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
  exponential actions.
- PI hierarchy equations of motion for finite-exponential bosonic bath
  correlations, including propagation, stationary solving, and hierarchy-depth
  convergence reports.
- Density-valued composite quantum-jump systems with explicit cross-factor
  channels, factorized gain/loss application, reproducible threaded batches,
  and online trajectory statistics.
- Automated time-step, Krylov-dimension, hierarchy-depth, and sector-cutoff
  convergence studies; confidence-controlled quantum-jump and diffusive
  ensembles.
- Generalized qudit Husimi-Q data and optional Tables, Makie,
  QuantumCumulants, Distributed, JLD2, and HDF5 integrations.
- Literature validation examples, same-basename guides, package documentation,
  release automation, and a Pluto research-workflow notebook.

### Notes

- This version has not yet been released or registered. Replace `Unreleased`
  with the release date only after the exact release commit passes the clean
  release gate; add `date-released` to `CITATION.cff` at the same time.

[0.1.0]: https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/releases/tag/v0.1.0
