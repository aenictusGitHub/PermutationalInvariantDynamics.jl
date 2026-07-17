# Exact PI reference for higher-order cumulant closures

[`cumulant_bridge.jl`](cumulant_bridge.jl) demonstrates how to use exact PI
local moments as initialization and validation data for a truncated cumulant
hierarchy. It compares a product factorization with both an exact product state
and a correlated GHZ state, then exports a neutral description of a model with
one- and two-body processes.

For distinct particles, the requested order-three table contains moments such
as

```math
M_{x z -}=\mathrm{tr}\!\left[\rho\,
  \sigma_x^{(1)}\sigma_z^{(2)}\sigma_-^{(3)}\right].
```

Permutation invariance makes the value independent of the selected distinct
particle labels and of their assignment to the local operators. The library
therefore stores only one canonical key for each multiset of operator labels.
For an alphabet of size `m`, exact order `k` needs
`binomial(m+k-1,k)` values rather than `m^k` values.

The exact contraction symmetrizes a `d^k` local tensor, applies the Appendix-D
Schur geometry, and divides by the exact subset count. No `d^N` state or
operator is constructed. The cost is nevertheless exponential in the chosen
closure order, as any general local order-`k` tensor contains `d^(2k)`
entries; keep `k` modest and reuse the returned table.

The first comparison must pass because the initial state is a tensor product.
For the GHZ state, the same factorization built from `one_body_rdm` misses
higher correlations, so `compare_cumulant_closure` reports a nonzero maximum
and root-mean-square error.

The final `CumulantBridgePayload` uses schema version `1.0.0`. It records:

- `N`, `d`, body order, process, scope, rates, and Hamiltonian `hbar` values;
- detached fixed operators or time-dependent operator prototypes/schedules;
- particle-one-fastest tensor ordering and the package dissipator convention;
- exact canonical moments through the selected order.

This payload is independent of any symbolic closure package. A direct PI term
is marked `microscopic=false`, because Schur blocks alone do not define a
unique local Hamiltonian or jump operator and the adapter must not invent one.

Run the example from the repository root with:

```sh
julia --project=. examples/cumulant_bridge.jl
```
