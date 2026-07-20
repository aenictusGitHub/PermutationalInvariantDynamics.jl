# Contributing

Thank you for improving `PermutationalInvariantDynamics.jl`. Start with
`AGENTS.md`: it records the mathematical normalization, precision rules,
plan/workspace ownership, dependency boundaries, and release gates that must
remain true across refactors.

## Development workflow

Instantiate and run the complete core suite from the repository root:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

For a focused edit, select one or more comma-separated test groups while
keeping the same `Pkg.test()` entry point:

```bash
PID_TEST_GROUPS=solvers,workflows julia --project=. -e 'using Pkg; Pkg.test()'
```

Available groups are `representation`, `lowering`, `solvers`, `dynamics`,
`nonmarkovian`, `stochastic`, `analysis`, `workflows`, `literature`, and
`visualization`. The default and `PID_TEST_GROUPS=all` run every group in the
normal regression order.

Run the allocation/thread-safety gates with multiple Julia threads:

```bash
JULIA_NUM_THREADS=4 julia --project=. benchmark/performance_regression.jl
```

Documentation and optional dependencies have isolated environments. See
`docs/Project.toml`, `quality/Project.toml`, and `test/optional/Project.toml`.
Do not commit generated root, quality, example, or optional-test manifests;
the tracked `docs/Manifest.toml` is the exception.

## Architectural expectations

- Keep PI production algorithms polynomial in the retained PI dimension;
  never construct full `d^N` states or `d^(2N)` superoperators.
- Prepared plans are immutable and shareable. Mutable workspaces belong to
  one task at a time and must be reusable in hot loops.
- Extend the private source traits in `src/source_protocol.jl` when adding a
  new linear-operator wrapper instead of adding `isa` chains to every solver.
- Add solver symbols and aliases through `src/solver_algorithms.jl`, then test
  the high-level command, resource recommendation, and parameter-scan route.
- Preserve scalar precision and explicit validation. Do not silently repair,
  normalize, truncate, or narrow user data.
- Keep non-core dependencies in package extensions.

Every exported binding needs a source docstring and one canonical `@docs`
entry. GitHub's math renderer does not accept `operatorname`; use roman labels
such as `\mathrm{tr}` instead.
