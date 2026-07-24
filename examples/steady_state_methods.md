# Choosing a steady-state solver

Source: [`steady_state_methods.jl`](steady_state_methods.jl)

## Benchmark model

The script uses `N = 6` independently pumped and damped qubits. Its exact
stationary state is an iid diagonal product state, making solver error directly
measurable.

## Available methods

The small test model is compiled once with the sparse backend. The same
prepared model is then passed to `stationary_state` with typed algorithm
values:

- `DirectAlgorithm()`, a trace-constrained linear solve;
- `SVDAlgorithm()`, which selects the null singular vector;
- `EigenAlgorithm()`, which selects the traceful eigenvector nearest zero;
- `ShiftInvertAlgorithm(shift=-1e-3, maxiter=100)`, inverse iteration near
  the origin; and
- `GMRESAlgorithm(krylovdim=20, maxiter=200)`, a restarted matrix-free solve
  of the trace-fixed stationary equation.

With `return_info=true`, each call returns a `SteadyStateResult`: `result.state`
is a typed `PIState`, while `result.info` contains the method, residual, trace
error, iteration count, and convergence flag. The example reports those
quantities and the distance to the exact product state, validates each state
with `diagnostics`, and requires an exact-state error below `2e-8`.

A second shift-invert call uses the direct solution as `initial_state` and a
shift of `-1e-2`, demonstrating the typed-state warm-start path.

## Run

```sh
julia --project=. examples/steady_state_methods.jl
```

`DirectAlgorithm` is a robust choice for modest sparse problems. SVD is
diagnostic but usually more expensive; eigen and shift-invert methods are
useful when their spectral information or sparse factorization is appropriate.
Always judge a result by its Liouvillian residual and physical normalization,
not only by solver convergence status.

For large PI spaces, use `GMRESAlgorithm` and tune `krylovdim` and `maxiter`.
The stationary-state dispatcher uses matrix-free applications for GMRES even
when the supplied compiled model selected a sparse backend.

The final solve deliberately exposes the reusable solver internals. It compiles
a matrix-free model, allocates `KrylovWorkspace(..., 20)`, and constructs one
`schur_sector_preconditioner` with an expected reuse count of 10. Both objects
are passed through a preconditioned `GMRESAlgorithm`. The preconditioner stores
and factorizes only diagonal Schur-sector Liouville blocks; final convergence
is still checked against the unpreconditioned Liouvillian. The printed
`preconditioner_cost` metadata makes setup applications, factorization count,
storage, apply-cost estimate, and expected amortization visible.

For this compatible compiled PI model, `block_construction` is
`:prepared_kernels` and `setup_block_applications` is zero. The diagonal
blocks are lowered directly from the immutable physical term plan; only the
small operator-scale probes remain. A plan-less callback cannot expose that
structure and retains the coordinate-probing fallback.
