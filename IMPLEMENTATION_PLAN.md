# Implementation architecture

The package follows Bastin and Martin, J. Phys. A 58 (2025) 275301. Production
code never constructs Schur vectors or arrays of size `d^N` or `d^(2N)`.

## Layer graph

1. Exact partitions, GT patterns, dimensions, and Appendix-B CG coefficients.
2. Typed `PIBasis`, `PIState`, `PIOperator`, and normalization conventions.
3. One-body and Appendix-D p-body representation geometry.
4. Declarative immutable `PIModel` term tuples and dispatch-based lowering.
5. Immutable `LiouvillianPlan` plus caller-owned `LiouvillianWorkspace`.
6. Sparse and matrix-free adapters sharing the same lowered plan.
7. Dynamics, Krylov, symmetry, spectra, trajectories, Floquet, and response.
8. Prepared observable/reduction plans and sectorwise information analysis.
9. Geometry-free finite/thermodynamic product closure with a read-only
   `MeanFieldPlan` and per-task `MeanFieldWorkspace`.
10. High-level commands, typed results, Schur-block, compressed density-
    spectrum, and complex-spectrum diagnostics, dependency-free SVG rendering,
    documentation, and CI.

The main mathematical risks remain the Appendix-B phase convention, GT versus
computational ordering, square-root multiplicity factors in equations (27)
and (31), and Appendix-D path ordering. Each is covered by orthogonality,
analytical qubit, small-qudit, or dense-reference tests before use by a
generator.

## Architecture decisions

- Prepared mathematical data are immutable; mutable scratch belongs to an
  explicit workspace owned by one task.
- Compatibility matrix-free `mul!` calls are synchronized. Parallel hot loops
  use one workspace per task or thread.
- Autonomous and driven generators are distinct runtime contracts. Stationary
  linear algebra rejects driven generators; `freeze` requires an explicit time.
- Sparse and matrix-free backends must lower from the same term plan and pass
  precision-equivalence guards.
- State analysis validates trace, Hermiticity, and positivity before removing
  roundoff-level skew components. Invalid states are never silently repaired.
- Caches are basis-owned or call-owned; no unprotected global mutable cache is
  permitted.
- High-level model APIs return `PIState` or typed result objects. Raw coordinate
  solvers remain available as advanced interfaces.
- Mean-field lowering uses the same physical term conventions but does not
  depend on Schur geometry. Finite subset counts and leading thermodynamic
  counts are explicit, including ordered-overlap classes for collective
  p-body jumps, and rate normalization is never inferred.
- Schur-block extraction is a numerical diagnostic independent of rendering.
  State/operator weights may use physical `C_nu/sqrt(f^nu)` or stored
  coefficient blocks; superoperator rows are output sectors and columns are
  input sectors in PI coefficient coordinates.
- Matrix-free Schur-block extraction is exact and retains no global
  Liouvillian, but its setup cost is one application per input PI coordinate.
  Driven operator-valued fallbacks are frozen and lowered once at the requested
  time before probing. Text and SVG output add no plotting dependency.
- Density-spectrum rendering consumes multiplicity-compressed Schur-block
  eigenvalues and exact degeneracies without expanding the Hilbert space.
- Complex-spectrum extraction is independent of rendering. It preserves raw
  mode order, completeness, and Krylov convergence metadata. Liouvillian
  eigenvalues, Floquet multipliers, and principal-branch exponents share one
  SVG renderer with representation-specific stability boundaries.

## Remaining research-scale work

- Matrix-free iterative LR nullspaces beyond the implemented exact-count,
  weight-restricted sparse-SPQR backend for very large qudit irreps.
- Extreme-scale block/rational eigensolvers beyond the implemented exact-shift
  implicit-QR Arnoldi and hard-locking preconditioned Jacobi--Davidson paths.
- Matrix-free PSD oracles beyond the implemented shifted-Cholesky certificate
  for a single extremely large Schur block.
- Broader scalar-generic representation geometry beyond the current
  Float64-based CG convention.
