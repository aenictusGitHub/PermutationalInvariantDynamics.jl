# Prepared steady-state parameter scan

Source: [`parameter_scan.jl`](parameter_scan.jl)

## Model

The example considers 12 qubits coupled independently to an emitting and a
pumping reservoir,

```math
\mathcal L\rho=\sum_i\mathcal D[\sigma_i^-]\rho
 +r\sum_i\mathcal D[\sigma_i^+]\rho.
```

The exact stationary excited-state fraction is ``r/(1+r)``. This gives a
compact validation of both the stationary solver and continuation machinery.

## Prepared scan

`ParameterScanPlan` owns the copied rate grid and a model builder, but no
compiled model or mutable solver workspace. The builder returns one `PIModel`
for each pumping rate. Every point therefore receives the correct prepared
rate kernels, while a serial scan reuses its compatible GMRES storage and uses
the preceding steady state as its initial iterate.

The example sets `save_outputs=false`. The streaming callback still sees each
new `PIState`, extracts the excitation fraction, and then lets the state be
released. Only scalar point metadata and one final restart seed remain in the
result. This prevents saved-state memory from growing as
`number of points * PI dimension`.

The first call intentionally stops after three points with `max_points=3`.
`resume_parameter_scan` validates the complete parameter grid and resumes from
the retained seed in a fresh `ParameterScanWorkspace`. No model builder,
callback, random generator, compiled plan, or Krylov workspace is embedded in
the checkpoint-neutral result.

The final `parameter_scan_columns` call provides dependency-free column data
for plotting or tabular analysis. The script checks the exact thermal
prediction and asserts that no state history was retained.

## Run

```sh
julia --project=. examples/parameter_scan.jl
```

For independent scans set `continuation=false` and use
`execution=:threads`. After `using Distributed` and adding workers, the
optional extension provides

```julia
distributed_parameter_scan(independent_plan)
```

for deterministic balanced chunks. Alternatively, a scheduler can assign
disjoint `indices` manually and combine returned objects with
`merge_parameter_scan_results`; per-index random seeds make spectral chunks
independent of scheduling. Distributed continuation is deliberately rejected,
and worker chunks complete before master-side stopping policies are applied.
