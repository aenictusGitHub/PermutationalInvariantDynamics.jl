# Results, tables, plots, and exports

High-level solvers return different numerical payloads because a stationary
state, a time series, a selected spectrum, and a parameter scan have genuinely
different storage requirements. The result-output API gives those objects a
small common surface without hiding their differences:

```julia
summary = summarize(result)
table = result_table(result)
save_result("observables.csv", result)
save_result("run.pidrun", result)
```

These operations inspect data already stored in `result`. They never repeat a
solve, compute a missing observable, reconstruct a full Hilbert-space state,
or silently validate/repair a density operator.

## Compact summaries

`summarize(result)` returns a named tuple suitable for logs and notebook
headers. Every summary contains `result_type` and `package_version`. Depending
on the result, it also reports stored evidence such as

- selected algorithm and stationary residual;
- sample range and number of retained states;
- trajectory count and Monte Carlo standard error;
- scan successes, failures, stopping, and cooperative-cancellation status;
- spectral completeness; or
- refinement convergence and its final pairwise error.

A missing field is reported only when the result explicitly represents it.
For example, `summarize` does not turn a selected spectrum into a certified
global gap and does not infer physicality from a trace residual.

```julia
steady = stationary_state(model;
    algorithm=GMRESAlgorithm(), return_info=true)

summary = summarize(steady)
summary.converged
summary.residual
summary.algorithm
```

## Dependency-free tables

`result_table(result)` returns a column-oriented `ResultTable`. Its columns
borrow result-owned vectors when possible:

```julia
solution = solve_dynamics(model, rho0, (0.0, 20.0);
    saveat=0.1,
    observables=(excitation=number_operator,),
    save_states=false)

table = result_table(solution)
table.columns.time
table.columns.excitation
first(table)  # a named-tuple row
```

The default table is deliberately compact. Density matrices, population
vectors, trajectory histories, and eigenvectors are not implicit columns.
For a workflow that genuinely needs nested Julia objects, request
`include_output=true` explicitly:

```julia
table_with_states = result_table(solution; include_output=true)
```

Some nested outputs are not tabular. In particular, a collection of complete
trajectory histories has two independent axes (path and time);
`include_output=true` therefore raises instead of flattening it ambiguously.
Request state-free online trajectory statistics, or save selected trajectory
states as explicit checkpoints.

Loading Tables.jl makes `ResultTable` a column table:

```julia
using Tables
rows = Tables.rowtable(result_table(solution))
```

The older direct Tables adapters for `ParameterScanResult`,
`ComplexSpectrum`, `QuditHusimiData`, and `ConvergenceStudyResult` remain
available.

## Export formats

The extension determines the default format:

| Extension | Contents |
|---|---|
| `.csv` | Compact result-table columns |
| `.tsv`, `.txt` | Compact tab-separated columns |
| `.pidrun` | New directory with versioned metadata, a compact table, and exact PI-state checkpoints owned directly by the result |
| `.jld2`, `.jld` | Julia-native object, summary, metadata, and detached columns; requires JLD2 |
| `.h5`, `.hdf5` | HDF5 summary, normalized text columns, and exact PI-state payloads; requires HDF5 |

For example:

```julia
save_result("scan.csv", scan_result)

save_result("steady.pidrun", steady;
    metadata=Dict(
        "model" => "driven ensemble",
        "parameter_set" => "figure 2",
    ))
```

A `.pidrun` path must not already exist. The package never recursively
replaces an existing directory. User metadata may not reuse schema or summary
keys such as `schema_version`, `result_type`, `package_version`, or
`residual`; conflicting keys raise instead of changing the archive's meaning.
Its layout is intentionally inspectable:

```text
steady.pidrun/
├── metadata.tsv
├── table.tsv
└── states/
    ├── index.tsv
    └── state_000001.pid
```

Each `.pid` file uses the reconstructing `PIStateCheckpoint` schema, including
the exact retained Schur sectors and scalar precision. A `.pidrun` record is a
convenient analysis export, not a universal restart format for every internal
workspace. Use `save_checkpoint` for a controlled state restart and
`save_experiment` for a complete verified-experiment archive.

CSV and TSV cells that are not scalar are represented explicitly as text.
Large output columns are excluded unless requested. JLD2 is Julia-native and
may require compatible package/type definitions to reload. HDF5 columns are
normalized to text to avoid narrowing heterogeneous scan metadata; exact PI
states are stored separately with the checkpoint payload convention.

## Plotting

With Makie loaded, a two-column numeric `ResultTable` can be plotted directly:

```julia
using CairoMakie
lines(result_table(solution))
```

Wider tables are intentionally not assigned an arbitrary ordinate. Choose the
columns:

```julia
table = result_table(solution)
lines(table.columns.time, table.columns.excitation)
```

`SpectrumResult` also converts directly to a complex-plane scatter plot. As
with the existing spectral and phase-space conversions, rendering uses only
stored numerical data.

## API

```@docs
PermutationalInvariantDynamics.ResultTable
PermutationalInvariantDynamics.summarize
PermutationalInvariantDynamics.result_table
PermutationalInvariantDynamics.PI_RESULT_ARCHIVE_VERSION
PermutationalInvariantDynamics.save_result
```
