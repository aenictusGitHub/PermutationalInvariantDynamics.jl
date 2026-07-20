# Benchmarks

This directory separates four complementary performance tasks:

- `performance_regression.jl` provides CI allocation and correctness gates.
- `performance_audit.jl` is the broad, human-readable warmed audit.
- `benchmarks.jl` defines the longer `BenchmarkTools` suite used during
  targeted optimization.
- `internal_scaling.jl` produces reproducible scaling tables for publication
  and comparisons. It does not replace the regression gates.

The detailed suite includes an `N=64` symmetric collective group with separate
plan construction, sparse-first materialization, `:auto` compilation, sparse
and matrix-free actions, driven preallocated action, and prepared collective
observable measurements.

The CI regression and human-readable audit also include `N=64` structured
support, sparse/matrix-free equivalence, memory, and hot-allocation checks.

## Prepare the benchmark environment

From the repository root, create the isolated environment with the local
checkout under development:

```sh
julia --project=benchmark -e \
  'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
```

This creates `benchmark/Manifest.toml` locally. The manifest and generated
results are ignored and must not be committed.

## Internal scaling benchmark

The quick matrix is deliberately conservative and is suitable for checking a
laptop setup:

```sh
julia --project=benchmark benchmark/internal_scaling.jl --mode quick
```

The full matrix extends the particle-number sweeps:

```sh
julia --project=benchmark benchmark/internal_scaling.jl --mode full
```

Both modes cover all-sector qubits and qutrits plus fully symmetric qubits and
qutrits. The all-sector workload combines local and collective dissipation;
the symmetric-sector workload uses only sector-preserving collective terms.

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
every table; it records the schema version and phase policies explicitly.

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
and sampling information. A dirty checkout additionally records a SHA-256 of
the tracked diff plus every non-ignored untracked file. Prefer a clean checkout
for publication; otherwise archive the diff and verify this hash. No generated
result is a CI gate or a performance claim by itself.
