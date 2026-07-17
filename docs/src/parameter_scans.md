# Prepared parameter scans and continuation

Phase diagrams and finite-size studies usually solve a sequence of closely
related PI models. Reconstructing solver scratch at every point wastes memory
bandwidth, while retaining all states merely to continue the next solve can
consume far more memory than the Liouvillian application itself. The prepared
scan layer separates the immutable experiment description from task-owned
numerical scratch.

## Workflow

```julia
basis = PIBasis(20, 2)
sm = ComplexF64[0 1; 0 0]
sp = Matrix(adjoint(sm))

builder = pump -> PIModel(basis, (
    LocalJump(sm; rate=1.0),
    LocalJump(sp; rate=pump),
))

plan = ParameterScanPlan(0.05:0.05:0.5, builder;
    algorithm=GMRESAlgorithm(krylovdim=30),
    compile_options=(backend=:matrixfree,),
    continuation=true)

result = parameter_scan(plan)
```

The builder may accept either `(parameter)` or `(parameter, index)`. A second
constructor accepts `(parameters, prototype, remaker)` for code organized
around an immutable prototype. In both cases the callable must return a fresh
`PIModel` or an already `CompiledPIModel`. Compilation is performed once per
parameter when necessary: prepared kernel coefficients may depend on that
parameter and therefore cannot in general be shared safely.

`ParameterScanPlan` contains no numerical scratch. A
`ParameterScanWorkspace` owns the previous continuation seed and compatible
GMRES or Arnoldi arrays. One workspace may be reused sequentially; concurrent
tasks require one workspace each. A fresh `parameter_scan` call clears any old
continuation seed while retaining compatible solver arrays. Continue a
specific path only through `resume_parameter_scan`, which reloads the
validated checkpoint seed explicitly.

## Continuation and compatibility

Serial steady-state scans pass the preceding successful state to Krylov GMRES
or shift-invert. Spectral scans use a deterministic combination of the
previous selected Ritz vectors. Combining the vectors avoids an immediate
one-dimensional Arnoldi breakdown that would occur if its seed were an exact
single eigenvector.

Continuation is used only when all of the following match:

- particle number and local dimension;
- retained Schur partitions and PI coordinate dimension;
- numerical scalar type;
- steady-state versus spectrum task;
- the immediately preceding scan path.

Solver storage is independently reused when its task, dimension, scalar type,
and Krylov size match. A change in any of these properties rebuilds the
workspace instead of converting or truncating it.

This continuation reuses an initial state/vector and compatible ordinary
GMRES or Arnoldi arrays. It does not retain a GCRO invariant subspace. Use the
lower-level [`RecycledGMRESWorkspace`](@ref) explicitly when subspace recycling
between custom linear systems is the intended algorithm.

## Streaming and restart

Set `save_outputs=false` to avoid retaining every state or spectrum. A
callback still receives the complete live `ParameterScanPoint`:

```julia
callback = point -> begin
    point.status === :success && println(point.parameter, " ", point.residual)
    nothing
end

prefix = parameter_scan(plan; callback, max_points=25)
result = resume_parameter_scan(plan, prefix; callback)
```

A callback must return `nothing` to continue or `:stop` to produce a resumable
prefix. With
`save_restart=true`, the result retains at most one state or Ritz seed. Restart
seeds are copied when they cross workspace, result, resume, or merge ownership
boundaries, so callback-side changes to a streamed state and later changes to
a checkpoint cannot corrupt the next continuation point. The
result otherwise contains only copied parameters, point records, outputs
selected by the user, and plain metadata. It does not retain a builder,
compiled generator, workspace, callback, random generator, lock, or exception
backtrace.

Resume validates the task, complete parameter sequence, every stored index,
every indexed parameter, and the continuation/output/restart retention flags
before building a model. This prevents a resumed result from mixing retained
and discarded histories under misleading metadata. Failed points are retried
by default. A continuation result must be a successful prefix because warming
across an uncomputed branch would not represent the requested path. When an
extension evaluates no point or fails before producing a newer checkpoint,
the last validated prefix checkpoint remains available. A result made with
`save_restart=false` resumes cold, even if the supplied workspace was used by
another compatible-looking scan. Results without a saved seed report
`restart_index=0`; a nonzero restart index therefore always identifies an
owned usable checkpoint.

## Failure policy

Every failure creates a point with `status=:failed`, `converged=false`, and a
printable error type and message.

- `on_error=:stop` is the default and returns at the failed point;
- `on_error=:record` continues, but clears the continuation seed;
- `on_error=:throw` invokes the callback and then rethrows in serial mode.

No policy silently drops a parameter. Solver residual, trace error,
convergence, iteration count, compilation time, solution time, warm-start
status, workspace-reuse status, and nested solver diagnostics are recorded per
successful point.

## Threaded and distributed scans

Path-dependent continuation is deliberately incompatible with threaded
execution. Independent points use

```julia
independent = ParameterScanPlan(parameters, builder;
    continuation=false, algorithm=GMRESAlgorithm())
result = parameter_scan(independent; execution=:threads)
```

Each worker owns one workspace reused across its independent points, and every
point receives an index-derived deterministic random stream. An
acknowledgement protocol bounds live unsaved outputs and out-of-order callback
records by the worker count. Results and callbacks remain in parameter-index
order. The
builder, remaker, and user diagnostic must themselves be thread safe.

Multi-process schedulers can call `parameter_scan` with disjoint `indices` and
then use `merge_parameter_scan_results` when `continuation=false`. Merging
verifies the task, parameter sequence, stored index/value pairs, and absence
of duplicate indexes. Independently computed continuation chunks are rejected
because they do not prove a common restart path; use `resume_parameter_scan`
for that case. This explicit chunking keeps Distributed.jl or cluster
scheduling out of the package's core dependency set.

After loading the Distributed stdlib, the optional package extension performs
that partitioning directly:

```julia
using Distributed
addprocs(4; exeflags="--project=$(dirname(Base.active_project()))")

distributed = distributed_parameter_scan(independent)
```

The function deterministically assigns contiguous, balanced index chunks,
loads the package on every selected worker, evaluates all chunk failures as
records, and applies `on_error` plus callbacks on the master in global index
order. It rejects `continuation=true`. A master callback causes live numerical
outputs to cross process boundaries and is accepted only when
`save_outputs=true`, making that retained-memory cost explicit. For large
states, put the scalar reduction in `plan.diagnostic` and omit the callback.
All worker chunks finish before master-side callback stopping or `on_error`
selection is applied; these policies shorten the returned ordered prefix but
do not cancel already dispatched remote computation.
The plan and its captured builder/remaker must be serializable, and every
worker must activate an environment containing a compatible package version.

## Spectrum scans

Set `task=:spectrum`, select a high-level spectral algorithm, and optionally
retain eigenvectors:

```julia
plan = ParameterScanPlan(parameters, builder;
    task=:spectrum,
    algorithm=:iram,
    spectrum_target=:largest_real,
    nev=8,
    solver_options=(krylovdim=48, maxrestarts=20),
    save_vectors=false)
```

The continuation seed is retained even when the returned spectra omit their
vectors. Dense spectra accept the same scan interface but do not claim a warm
start or Krylov-workspace reuse.

For steady-state scans with `GMRESAlgorithm`, its `krylovdim` field is the
single source of workspace sizing. Repeating `krylovdim` in `solver_options`
is rejected instead of silently selecting one value.

## Dependency-free rows and columns

`parameter_scan_rows` returns named-tuple rows and
`parameter_scan_columns` returns a named tuple of vectors. They intentionally
exclude large outputs and nested diagnostics by default. Pass
`include_output=true` when those objects are required. These representations
are straightforward to adapt to Tables.jl without imposing it on core users.

Loading Tables.jl activates a lazy row-table interface directly on
`ParameterScanResult`:

```julia
using Tables
rows = Tables.rows(result)
columns = Tables.columntable(result)
```

Its schema is the same scalar metadata returned by `parameter_scan_rows`; it
never exposes `output` or nested `diagnostics` implicitly. Iterating rows is
lazy, while a column-table sink naturally allocates its output columns.

## API

```@docs
ParameterScanPlan
ParameterScanWorkspace
clear_parameter_scan_workspace!
ParameterScanPoint
ParameterScanResult
parameter_scan
resume_parameter_scan
merge_parameter_scan_results
distributed_parameter_scan
parameter_scan_rows
parameter_scan_columns
```
