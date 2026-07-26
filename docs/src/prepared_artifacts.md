# Reusing prepared representation data

Representation setup can dominate a parameter study even when every solve is
matrix free. One-body branching maps, Appendix-D removal paths, and
bipartition recouplers depend only on a PI basis and numerical coefficient
context. They should therefore be prepared once and reused.

```julia
using PermutationalInvariantDynamics

basis = PIBasis(12, 2)
geometry = prepare_geometry(
    basis;
    pbody_orders=(2, 3),
    reduction_ks=(1, 6),
    memory_budget=512 * 1024^2,
)

one_body = onebody_geometry(geometry)
pair_geometry = pbody_geometry(geometry, 2)
half_partition = reduction_plan(prepared_reductions(geometry), 6)
```

`PreparedGeometryBundle` is immutable and contains no mutable numerical
workspace. It can be shared read-only, while each task still owns the
workspaces used to apply operators or reductions. Every accessor checks exact
`PIBasis` identity. The bundle also snapshots the sector and GT-pattern
layout, and `BigFloat` bundles check the active precision and rounding mode.

The default `coefficient_cache=:auto` constructs one depth-bounded
`OneBoxCGCache` and shares it across the requested one- and many-body geometry.
Use `coefficient_cache=nothing` when a one-off call-local setup is preferable,
or supply an existing basis-owned coefficient cache.

## Explicit cache for recurring requests

When several parts of an application may independently request the same
preparation, use a user-owned cache:

```julia
store = PreparationCache(memory_budget=512 * 1024^2)

geometry_a = prepare_geometry!(
    store, basis;
    pbody_orders=(2, 3),
    reduction_ks=(1, 6),
)
geometry_b = prepare_geometry!(
    store, basis;
    pbody_orders=(2, 3),
    reduction_ks=(1, 6),
)

@assert geometry_a === geometry_b
preparation_cache_summary(store)
```

The key includes exact basis object identity, scalar type, `BigFloat`
precision and rounding mode, requested body orders, reduction sizes and
tolerance, and coefficient-cache selection. Cache construction and lookup are
synchronized. There is no process-global mutable cache.

The cache budget is checked before inserting a bundle. Its accounting
conservatively sums each bundle's standalone `summarysize`, so a basis shared
by two entries may be counted twice. `evict_prepared_geometry!` removes one
exact bundle, and `clear_preparation_cache!` drops all cache-owned references.

Qudit `ReductionPlanSet` setup can require sparse rank-revealing QR and dense
nullspace scratch for which no allocation-free upper bound is currently
available. A bundle containing qudit reductions therefore requires the
explicit `memory_budget=Inf` opt-out and rejects a finite budget before
constructing any geometry. Prepare one- and p-body data separately when that
opt-out is not acceptable.

## Why bundles are not persisted yet

`PreparedGeometryBundle` intentionally has no file save/load API. Its current
contents include private sparse branching layouts and recoupling structures
whose binary schema is not a public cross-version contract. Serializing those
objects through Julia serialization or JLD2 would risk accepting stale
coefficient conventions, package internals, or arithmetic contexts.

Reliable persistence requires a separate, versioned interchange schema with
explicit reconstruction and validation of every partition, GT-pattern order,
sparse support, scalar type, Julia/package version, and `BigFloat` context.
Until that schema exists, reconstructing a bundle is safer than presenting a
fragile disk cache as reusable research data. Mutable workspaces, callbacks,
locks, and solver state must never be added to such a schema.

## API

```@docs
PreparedGeometryBundle
prepare_geometry
validate_prepared_geometry
onebody_geometry
pbody_geometry
prepared_reductions
PreparationCache
prepare_geometry!
evict_prepared_geometry!
clear_preparation_cache!
preparation_cache_summary
```
