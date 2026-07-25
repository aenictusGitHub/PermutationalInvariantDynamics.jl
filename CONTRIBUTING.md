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
julia --startup-file=no --project=benchmark -e \
  'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
JULIA_NUM_THREADS=4 julia --startup-file=no --project=benchmark \
  benchmark/performance_regression.jl
```

Documentation and optional dependencies have isolated environments. See
`docs/Project.toml`, `quality/Project.toml`, `benchmark/Project.toml`, the
projects under `benchmark/comparison/`, and `test/optional/Project.toml`.
Do not commit generated root, quality, example, benchmark, comparison, or
optional-test manifests; the tracked `docs/Manifest.toml` is the exception.
The reproducible scaling and ecosystem-comparison commands, validation rules,
and interpretation limits are documented in `benchmark/README.md` and
`docs/src/benchmarks.md`.

Run the dependency-free release metadata check before proposing a versioned
release:

```bash
julia --startup-file=no scripts/release_gate.jl \
  --expect-version X.Y.Z --require-clean
```

The executable-example CI job is reproducible locally without CairoMakie:

```bash
julia --startup-file=no --project=. test/run_quick_examples.jl --shard 1/2
julia --startup-file=no --project=. test/run_quick_examples.jl --shard 2/2
```

See `docs/src/releasing.md` for the complete clean-checkout gate and the
maintainer-only General registration step.

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
entry. In `docs/src` and Documenter docstrings, use double backticks for
inline mathematics and fenced `math` blocks for displays; Julia Markdown does
not preserve `\(...\)`, `\[...\]`, or single-dollar delimiters for KaTeX.
Example guides rendered directly by GitHub use `$...$` inline and fenced
`math` blocks for displays. GitHub does not accept `\operatorname`; use roman
labels such as `\mathrm{tr}` instead.
