# Generalized qudit Husimi phase space

`qudit_husimi_q` extends the sector-resolved Husimi transform beyond qubits.
A phase-space point is a local unitary ``U\in U(d)`` (or a Hermitian generator
``H`` with ``U=\exp(-iH)``). The same local transformation is lifted into each
retained Schur irrep; no ``d^N`` state or operator is constructed.

For partition ``\nu``, the transform uses the extremal GT vector and the
multiplicity-weighted state block
``\bar\rho_\nu=\sqrt{f^\nu}C_\nu``:

```math
Q_\nu(U)=\dim(U_\nu)
\langle\nu,U|\bar\rho_\nu|\nu,U\rangle.
```

With normalized Haar measure, the integral of one sector density is exactly
its physical sector population. The aggregate integrates to the sum of the
selected populations. Numerical values are never clipped; an invalid input
state is rejected.

## Prepared use

```julia
using PermutationalInvariantDynamics, LinearAlgebra

basis = PIBasis(8, 3)
rho = maximally_mixed_state(basis)
generators = [Diagonal([x, -x, 0.0])
              for x in range(0, 2pi; length=64)]

plan = QuditHusimiPlan(basis, generators;
    representation=:generator)
q = qudit_husimi_q(rho, plan; resolved=true)
```

Reuse `QuditHusimiPlan` for multiple states on the same point set. Unitary
input uses a Schur factorization and supports Float32/Float64; the generator
route remains available for other supported scalar types. The result stores
one aggregate value per supplied point and, with `resolved=true`, a
sector-by-point matrix.

For selected sector `nu`, the plan retains
`dim(U_nu) * number_of_points` coherent-vector entries. Setup constructs the
lifted dense generator and its exponential for every sector/point pair, so
plan construction can dominate a large orbit sample even though it never
scales as `d^N`. The plan is tied to the exact `PIBasis`; share it read-only
and rebuild it rather than narrowing a state or point precision.

This API provides generalized Husimi-Q data, not a qudit Wigner transform.
The `U(d)` orbit has a redundant stabilizer and no single two-dimensional
coordinate chart is inferred. `QuditHusimiData` therefore preserves supplied
point order but not a manifold coordinate system; the optional Makie adapter
plots point index against aggregate Q. Use `sector_values` explicitly for
resolved custom plots.

For `d=2`, the normalized-Haar result equals ``4\pi`` times the package's
sphere-density `spin_husimi_q` convention at the corresponding spin-coherent
direction. See `examples/qudit_coherent_state_q_distribution.jl` for that sanity check and a qutrit
scan.

## API

```@docs
QuditHusimiPlan
QuditHusimiData
qudit_husimi_q
```
