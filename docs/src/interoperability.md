# Optional ecosystem integrations

The core package remains lightweight. Optional Julia packages activate
extensions only after they are loaded; none changes the PI representation or
causes a full-Hilbert reconstruction.

## Tables.jl

After `using Tables`, `ParameterScanResult` is a lazy row table. Large states,
eigenvectors, and nested diagnostics are excluded from its implicit schema.
`ComplexSpectrum`, `QuditHusimiData`, and `ConvergenceStudyResult` expose
dependency-free numerical columns (including HEOM depth reports):

```julia
using Tables
columns = Tables.columntable(scan_result)
spectral_rows = Tables.rowtable(complex_spectrum)
```

Use the explicit `parameter_scan_rows(...; include_output=true)` route when a
sink genuinely needs large saved outputs.

Column adapters borrow result-owned vectors when possible and should be
treated as read-only views of the source object. Missing spectrum diagnostics
use an `O(1)` read-only logical column rather than allocating one `missing` per
mode. Qudit tabular data includes
the aggregate point-index/Q series, not the optional sector matrix.
Convergence tables include `estimate`; those entries may themselves be states
or arrays, so collecting the table is not automatically a compact export.
Choose a scalar estimate in the original refinement study when bounded output
is required.

## Makie

Loading Makie or CairoMakie activates plot conversions for existing numerical
objects:

```julia
using CairoMakie
scatter(complex_spectrum)       # Re(lambda), Im(lambda)
heatmap(spin_phase_space_data)  # theta, phi, density
heatmap(schur_structure)        # input/output sector weights
lines(convergence_report)       # refinement versus pairwise error
lines(qudit_husimi_data)        # supplied point index versus Q
```

Conversions never run a solver or phase-space transform. Axis labels,
palettes, logarithmic scales, and publication styling remain ordinary Makie
attributes. The dependency-free SVG renderers continue to be available when
Makie is absent.

Conversions prefer ranges, views, and lazy permutations. They expose the
aggregate spin/qudit data and raw Schur weights; resolved sectors, manifold
coordinates, residual annotations, and physical labels remain explicit user
plotting choices. Generic scalar layouts may require a safe allocating
fallback, so conversion should not be treated as an allocation-free solver
kernel.

## Distributed

`using Distributed` activates `distributed_parameter_scan`. Independent
points are assigned to deterministic, balanced worker chunks and merged in
global index order. Continuation scans remain serial because partitioning a
path-dependent warm start would change the computation. Builders, parameters,
algorithms, and diagnostics must serialize, and every selected worker must use
a compatible active project. Master callbacks require `save_outputs=true`;
all chunks finish before master-side stopping or failure policy is applied.

It also activates process-parallel quantum-jump and diffusive ensembles. Each
worker prepares geometry once for a deterministic contiguous chunk, while the
master assigns random streams by global trajectory index. Full saved results
must be serialized back to the master, so threaded or adaptive state-free
ensembles remain preferable when only statistics are required. Distributed
adaptive stopping is not implemented.

```@docs
PermutationalInvariantDynamics.distributed_quantum_trajectories
PermutationalInvariantDynamics.distributed_diffusive_trajectories
```

## QuantumCumulants, JLD2, and HDF5

The QuantumCumulants 0.5 extension supplies exact PI initial moments and
automatic microscopic symbolic lowering; see the
[cumulant bridge](cumulant_bridge.md). Symbolic spaces and indices are created
only inside the extension. Direct PI/nonmicroscopic terms and unevaluated
schedules are rejected, and the resulting selected-order closure remains an
approximation to check against exact PI moments.
JLD2 and HDF5 activate portable checkpoint writers described in the
[research utilities](research_utilities.md). All optional methods validate
their schema and reject ambiguous or narrowing conversions.

The repository's isolated optional-extension CI environment smoke-tests Makie
conversions, QuantumCumulants lowering, and JLD2/HDF5 checkpoint round trips
without adding any of these packages to the core dependency set.
