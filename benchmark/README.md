# Benchmarks

This directory separates seven complementary internal performance tasks:

- `performance_regression.jl` provides small deterministic CI correctness,
  equivalence, retained-storage, and hot-allocation gates. It deliberately
  avoids machine-dependent wall-clock thresholds.
- `performance_audit.jl` is the broad, human-readable warmed audit. It reports
  setup, retained memory, and representative hot operations for local
  inspection, but does not define CI timing requirements.
- `benchmarks.jl` defines the longer `BenchmarkTools` suite used during
  targeted optimization. It separates reusable preparation from hot
  explicit-workspace operations where the API supports that split.
- `internal_scaling.jl` produces machine-readable scaling tables for one
  stable four-family core workload. It does not replace the broader regression,
  audit, or detailed suites.
- `cold_start.jl` measures fresh-process Julia startup and package-load
  latency. Each row comes from a separate Julia process, and the package-load
  probe reports both external wall time and the interval surrounding `using`
  inside the child.
- `time_to_solution.jl` measures warmed end-to-end workflows with setup,
  solve, validation, and total phases reported separately for a stationary
  state, deterministic dynamics, streaming trajectories, and a prepared
  particle reduction.
- `batched_trajectories.jl` compares a fixed-capacity matrix-RHS conditional
  trajectory cohort with mathematically identical repeated scalar workspaces.
  It reports timing and warmed allocations without changing path scheduling.

The detailed suite includes an `N=64` symmetric collective group with separate
plan construction, sparse-first materialization, `:auto` compilation, sparse
and matrix-free actions, driven preallocated action, and prepared collective
observable measurements.

The CI regression and human-readable audit also include `N=64` structured
support, sparse/matrix-free equivalence, memory, and hot-allocation checks.

## Coverage by suite

The regression, audit, and detailed suites follow the current prepared APIs
across the main computational workflows. Case sizes and measurement effort
differ by suite:

| Workflow | Regression gate | Human-readable audit | Detailed `BenchmarkTools` suite | Internal scaling |
|---|---|---|---|---|
| Prepared PI core | sparse/matrix-free equivalence, forward/adjoint batch action, threading, allocation | setup, retained storage, scalar and batch action | separate setup, sparse, matrix-free, driven, threaded, and observable workloads | four fixed qubit/qutrit families |
| Time evolution and Floquet | low-storage RK4 and adaptive-expv storage checks | RK4, matrix-free Floquet-map setup, period/batch action, and stroboscopic evolution | prepared RK4 and Floquet setup/period/batch workloads | driven one-period action is not included |
| Appendix-D p-body and reductions | packed-support and allocation checks | p-body geometry/application, qudit reduction, local-factor trace | cached/uncached p-body setup and prepared reduction hot paths | not included |
| Identical local pseudomodes | supersite compile/application and spin-factor trace checks | prepared supersite construction, action, and reduction | reusable local-pseudomode setup and hot action | not included |
| Shared/global pseudomodes | matrix-free composite action and subsystem reduction checks | global-mode setup, action, and reduction | prepared global-pseudomode setup and hot action | not included |
| PI--HEOM | packed hierarchy forward/adjoint/batch checks | hierarchy setup and forward/adjoint/batch action | setup, prepared forward/adjoint/batch action, and preconditioner workloads | not included |
| PI--HOPS | conditioned hierarchy action and reused-ensemble checks | plan/workspace, conditioned action, and ensemble survey | plan, conditioned action, and reused-ensemble workloads | not included |
| Composite and stochastic systems | composite scalar/batch/adjoint action plus density, weak-PI, and composite trajectory gates | deterministic composite, batched, diffusive, density, weak-PI, and composite trajectory survey | reusable composite/batch and trajectory workloads | not included |
| Stationary and spectral solvers | adaptive-expv storage and symmetry-restricted workspace checks | GMRES, Arnoldi, harmonic Arnoldi, symmetry restriction, and preconditioner survey | reusable Krylov, preconditioned, recycled, and multi-shift workloads | not included |

The inventory describes code-path coverage, not equivalent problem sizes or
comparable timings between rows. In particular, stochastic and
non-Markovian cases use small deterministic seeds and cutoffs in regression,
while the audit and detailed suite are the appropriate places to increase
their workload.

## Prepare the benchmark environment

From the repository root, create the isolated environment with the local
checkout under development:

```sh
julia --startup-file=no --project=benchmark -e \
  'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
```

This creates `benchmark/Manifest.toml` locally. The manifest and generated
results are ignored and must not be committed. The repository intentionally
does not commit machine-specific benchmark timing tables.

Run the regression and audit in that isolated benchmark environment with:

```sh
JULIA_NUM_THREADS=4 julia --startup-file=no --project=benchmark \
  benchmark/performance_regression.jl
julia --startup-file=no --project=benchmark \
  benchmark/performance_audit.jl
julia --startup-file=no --project=benchmark \
  benchmark/batched_trajectories.jl
```

Load and run the longer suite from the isolated benchmark environment:

```sh
julia --startup-file=no --project=benchmark -e \
  'include("benchmark/benchmarks.jl"); display(run(SUITE; verbose=true))'
```

## Cold-start and end-to-end time to solution

Fresh-process latency is intentionally separate from warmed numerical
workloads:

```sh
julia --startup-file=no --project=benchmark \
  benchmark/cold_start.jl --mode quick
julia --startup-file=no --project=benchmark \
  benchmark/cold_start.jl --mode full
```

The startup probe launches Julia without loading the package. The load probe
starts another process, executes `using PermutationalInvariantDynamics`, and
constructs a tiny PI basis as a correctness smoke test. Child startup and
history files are disabled. The default child thread count is one; change it
explicitly with `--threads`. Discarded process warmups establish the ordinary
warm-precompile-cache measurement policy. They do not turn a child into a
reused process.

Run the four complete numerical workflows with:

```sh
julia --startup-file=no --project=benchmark \
  benchmark/time_to_solution.jl --mode quick
julia --startup-file=no --project=benchmark \
  benchmark/time_to_solution.jl --mode full
```

Every measured repetition reconstructs its basis, physical model, and
prepared resources in the `setup` phase. `solve` measures only the requested
numerical result, and `validation` compares it with an analytic state,
observable law, sampling-confidence bound, or GHZ marginal. A `total` row is
the sum of those three phases. The trajectory workload uses deterministic
index-derived seeds and online observable/jump statistics, so it does not
retain a state history merely for benchmarking. All high-level solves receive
the explicit memory budget selected by `--memory-budget-mib`; the reduction
case additionally enforces a conservative retained-setup bound before
constructing its plan.

Quick mode defaults to two measured repetitions after one discarded complete
warmup; full mode uses seven. These are raw repetitions, not
`BenchmarkTools` microbenchmarks. Both scripts accept `--samples`,
`--warmups`, `--output`, and `--dry-run`. The default ignored outputs are:

- `benchmark/results/cold_start_<mode>.tsv`;
- `benchmark/results/time_to_solution_<mode>.tsv`;
- a sibling `.metadata.tsv` for each result.

The sidecars record the active project and manifest hashes, Julia, CPU, BLAS,
threads, Git revision/worktree hash, phase policy, and run controls. Keep the
raw TSV and sidecar together. Do not commit machine-specific results.

The dependency-light parser and dry-run checks can be run independently:

```sh
julia --startup-file=no --project=benchmark \
  benchmark/test_productization_harnesses.jl
```

## Internal scaling benchmark

The quick matrix is deliberately conservative and is suitable for checking a
laptop setup:

```sh
julia --startup-file=no --project=benchmark \
  benchmark/internal_scaling.jl --mode quick
```

The full matrix extends the particle-number sweeps:

```sh
julia --startup-file=no --project=benchmark \
  benchmark/internal_scaling.jl --mode full
```

This runner remains the four-family schema-version-2 suite: both modes cover
all-sector qubits and qutrits plus fully symmetric qubits and qutrits. The
all-sector workload combines local and collective dissipation; the
symmetric-sector workload uses only sector-preserving collective terms.
Pseudomodes, HEOM, HOPS, composite dynamics, trajectories, p-body reductions,
and solver time-to-solution belong to the broader suites above and are not
silently added to this scaling schema.

Every operation receives explicit warm-up calls and is measured with
`BenchmarkTools`, one evaluation per sample. The script forces BLAS to one
thread so that results from machines with different BLAS defaults are easier
to interpret. Use `--samples`, `--seconds`, and `--warmups` to change the
measurement effort. Run separate processes when comparing thread counts.

The default output is
`benchmark/results/internal_scaling_<mode>.tsv`, accompanied by
`internal_scaling_<mode>.metadata.tsv`. Use `--output PATH` to select another
location. The result table records, separately:

- basis construction and matrix-free compilation setup time and allocation;
- hot explicit-workspace Liouvillian application time and allocation;
- for symmetric collective cases, sparse compilation/application and driven
  preallocated compilation/application;
- exact-support sparse estimates, actual nonzeros and retained bytes, and the
  backend selected by a fixed 512 MiB `:auto` probe;
- standalone component sizes and a non-double-counted retained total;
- PI, full-Hilbert, and full-Liouville dimensions for scaling plots;
- sparse-oracle agreement when the PI dimension is within the configured
  validation limit;
- trace-preservation and forward/adjoint duality checks for every case.

These fields form result schema version 2. Keep the sibling metadata file with
every table; metadata schema version 2 records the phase policies, active
project and selected manifest paths and SHA-256 hashes, and whether Julia's
startup file was disabled.

Standalone `Base.summarysize` columns include objects shared by that component
and therefore are not additive. `hot_retained_total_bytes` is the authoritative
non-double-counted retained footprint of the compiled model, explicit
workspace, and input/output vectors. BenchmarkTools' `allocated_bytes` fields
instead report transient allocation in the fastest measured call.

Sparse validation defaults to PI dimensions no larger than 5,000. Larger
cases still run the trace and adjoint checks. Change the bound explicitly with
`--validation-dimension-limit`; this affects validation cost, not the timed
setup or hot application measurements. Sparse and driven columns are `NA` for
all-sector rows: those phases target the occupation-space collective fast path,
while the ordinary matrix-free columns continue to cover every case.

Do not compare timings collected in different Julia processes without also
retaining the metadata files. They record Julia, CPU, BLAS, thread, Git, mode,
sampling, and resolved-environment information. A dirty checkout additionally
records a SHA-256 of the tracked diff plus every non-ignored untracked file.
Prefer a clean checkout for publication; otherwise archive the diff and verify
this hash. No generated result is a CI gate or a performance claim by itself.
