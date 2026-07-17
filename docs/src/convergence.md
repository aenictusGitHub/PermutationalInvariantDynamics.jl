# Numerical convergence studies

A solver's internal `converged=true` means that one discretized problem met
its own stopping rule. It does not establish convergence in a time step,
Krylov dimension, hierarchy depth, sector cutoff, closure order, or system
size. The refinement-study API records this separate layer of numerical
evidence without discarding the underlying solver reports.

## Generic refinement sequences

`convergence_study` evaluates all requested levels and compares neighbouring
estimates. The default requires agreement in the final two comparisons, so at
least three refinement levels are needed for a positive conclusion:

```julia
report = convergence_study([8, 16, 32, 64];
    parameter=:resolution,
    refinement_scale=n -> 1 / n,
    rtol=1e-6,
) do resolution
    result = solve_at_resolution(resolution)
    (
        value = observable(result),
        converged = result.converged,
        iterations = result.iterations,
    )
end

if report.converged
    estimate = convergence_estimate(report)
else
    @warn "resolution is not converged" reason=report.reason errors=report.pairwise_errors
end
```

The do-block is the `evaluator(level)`. It may return a number, array,
`PIState`, or a result carrying an `estimate`, `value`, `solution`, or `state`
property. For another schema, supply `estimate=result -> ...`.

The report retains:

- every refinement and unmodified evaluator result;
- the extracted estimates and diagnostics;
- inner-solver convergence flags when provided;
- pairwise errors and scale-aware tolerances;
- empirical rates when three successive `refinement_scale` values are
  geometrically spaced (otherwise the rate is `missing`);
- the first passing window and the status of the final window.

Because every raw result and estimate is retained, a generic study can be as
large as all of its saved histories combined. For a memory-bounded production
study, make the evaluator return compact diagnostics and use `estimate` to
extract only the common observable, reduced state, or final vector needed for
the comparison. The specialized HEOM-depth helper already retains complete
hierarchy data only at the finest depth.

An earlier passing window does not override a later disagreement. The result
then has `converged=false` and `reason=:refinement_not_stable`. Likewise, an
inner `converged=false` blocks the outer claim even when estimates happen to
agree.

This is a consistency report, not an automatic extrapolator or proof of an
asymptotic regime. Empirical orders are emitted only for compatible geometric
refinement scales and should be interpreted beside the raw errors.

For custom objects, provide their metric explicitly:

```julia
report = convergence_study(level -> solve(level), levels;
    estimate = result -> result.observables,
    distance = (a, b) -> maximum(abs, a .- b),
    estimate_norm = a -> maximum(abs, a),
)
```

All errors and norms must be finite nonnegative real values. Invalid numerical
data raise instead of being treated as unconverged finite data.

## Time-step convergence

`timestep_convergence` requires positive, strictly decreasing steps and uses
the step itself when estimating empirical order:

```julia
steps = [0.02, 0.01, 0.005, 0.0025]
report = timestep_convergence(steps; rtol=1e-5) do dt
    final_observable(dt)
end
```

For stochastic paths, use common random numbers when studying time-step bias;
otherwise sampling noise may dominate the pairwise difference. Sampling
confidence intervals and discretization convergence are separate claims.
The adaptive-ensemble `converged` flag controls Monte Carlo half-width only
and must not be reused as evidence that the trajectory integrator's step is
converged.

## Krylov-dimension convergence

Partial Ritz data and finite Krylov linear solves should be checked as their
subspace grows:

```julia
dimensions = [20, 30, 45, 60]
report = krylov_dimension_convergence(dimensions;
    estimate = result -> result.values,
    rtol=1e-7,
) do dimension
    krylov_liouvillian_spectrum(L;
        nev=4,
        krylovdim=dimension,
        require_convergence=false,
    )
end
```

The raw Arnoldi results and their residual flags remain in `report.results`.
The nominal rate axis is `1/dimension`; it is descriptive and is not an
assumption that Krylov error follows a power law. Unless three consecutive
dimensions form a geometric sequence, `observed_rates` therefore remains
`missing` while the pairwise convergence decision is still available.

## HEOM depth and sector cutoffs

Hierarchy convergence rebuilds the plan at every depth:

```julia
depth_report = hierarchy_depth_convergence([1, 2, 3, 4];
    estimate = result -> result.observable,
    rtol=1e-5,
) do depth
    solve_heom_at_depth(depth)
end
```

`sector_cutoff_convergence` assumes that a larger positive integer retains a
larger approximation space:

```julia
sector_report = sector_cutoff_convergence([2, 4, 6, 8]) do cutoff
    observable(solve_with_first_sectors(cutoff))
end
```

States represented in different truncated bases do not share a coordinate
space. Compare a common observable, reduced state, or an explicitly defined
embedding rather than subtracting their coefficient vectors directly.

If a model defines its cutoff in the opposite direction, use the generic API
and pass a strictly decreasing `refinement_scale` matching that convention.

## Consuming a result safely

`convergence_estimate(report)` raises unless the final window passed. This
makes it harder for downstream code to accidentally strip the convergence
status. An exploratory script may request the finest value with
`require_convergence=false`, but should retain and display the complete report
beside it.

Set `require_convergence=true` on `convergence_study` or a convenience wrapper
when an unconverged study should abort a production workflow.

## API

```@docs
ConvergenceStudyResult
convergence_study
convergence_estimate
timestep_convergence
krylov_dimension_convergence
hierarchy_depth_convergence
sector_cutoff_convergence
```
