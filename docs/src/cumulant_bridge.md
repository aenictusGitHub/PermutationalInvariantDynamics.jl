# Higher-order cumulant bridge

The cumulant bridge supplies exact finite-`N` PI data for initializing,
validating, and convergence-testing higher-order moment closures. It is an
optional bridge rather than a second symbolic algebra system: the core package
extracts exact moments and microscopic metadata, while an external package
chooses symbolic operator and index spaces and performs the truncation.

## Exact ordered local moments

For local matrices ``A_{a_1},\ldots,A_{a_k}`` on distinct particles, define

```math
M_{a_1\ldots a_k}
=\mathrm{tr}\!\left(\rho A_{a_1}^{(1)}\cdots A_{a_k}^{(k)}\right).
```

For a PI state this equals the expectation of the permutation-symmetrized
`k`-site tensor summed over unordered subsets, divided by
``\binom{N}{k}``. [`ordered_local_moment`](@ref) performs that contraction
directly in Schur blocks. [`ordered_local_moments`](@ref) evaluates all
symmetry-canonical multisets from a labeled operator alphabet and reuses one
[`PBodyGeometry`](@ref) per order.

```julia
using PermutationalInvariantDynamics, LinearAlgebra

basis = PIBasis(8, 2)
rho = ghz_state(basis)
sx = ComplexF64[0 1; 1 0]
sz = ComplexF64[1 0; 0 -1]

moments = ordered_local_moments(rho, (x=sx, z=sz); order=3)
m_xxz = moments[:x, :x, :z]
```

Lookup order is immaterial, but the moment always refers to distinct sites.
Use an operator product as one alphabet entry when several operators act on
the same site. The largest local tensor has `d^(2k)` entries. This is
exponential in the deliberately selected closure order `k`, but it is
independent of the exponentially larger full-system dimension `d^N`.

## Neutral model schema

[`cumulant_model_payload`](@ref) exports built-in or custom term metadata
through the documented `AbstractPITerm` hooks. With `time=t`, schedules are
evaluated using `value_at(x,t,parameters)`; without a time, schedules and any
available prototypes are retained. [`cumulant_bridge_payload`](@ref) combines
this model description with an exact moment table.

Schema `1.0.0` fixes the following adapter conventions:

- particle one is the fastest tensor index;
- a p-body operator is summed over unordered distinct subsets;
- ordered local moments use distinct-particle/falling-factorial
  normalization;
- the dissipator is ``L\rho L^\dagger-
  \{L^\dagger L,\rho\}/2``;
- `microscopic=false` forbids automatic lowering of a direct PI term.

An adapter should reject a schema version it does not support. It must also
ask the user for a local realization of direct PI terms instead of guessing
one.

## Comparing a closure

[`compare_cumulant_closure`](@ref) accepts another moment table, a dictionary,
or a callable. It reports per-moment and aggregate errors and raises when the
closure omits a requested key.

```julia
sigma = ComplexF64[0.6 0.1; 0.1 0.4]
product = iid_state(basis, sigma)
exact = ordered_local_moments(product, (x=sx, z=sz); order=3)
closure = Dict(key => prod(LinearAlgebra.tr(
                   getproperty((x=sx, z=sz), label) * sigma)
                   for label in key)
               for key in keys(exact))
comparison = compare_cumulant_closure(exact, closure)
@assert comparison.within_tolerance
```

## Optional QuantumCumulants.jl adapter

`QuantumCumulants` is a weak dependency with compatibility restricted to the
supported 0.5 API line. After `import QuantumCumulants`, the package extension
maps neutral keys to symbolic averages supplied by the user:

```julia
import QuantumCumulants

# Construct these averages in the QuantumCumulants/SQA Hilbert and index
# spaces chosen for the model.
symbolic_map = Dict((:x,) => symbolic_average_x,
                    (:x, :z) => symbolic_average_xz)
u0 = quantumcumulants_initial_values(exact, symbolic_map)
```

The extension calls the public `QuantumCumulants.get_order` function and
rejects an order mismatch. It does not construct a symbolic Hamiltonian,
jumps, or indices automatically: those choices are model-specific, and the
QuantumCumulants 0.5 rewrite intentionally changed that surface. A research
adapter may use the neutral term payload to build its chosen symbols, call the
official `meanfield(...; order=k)` and `complete` workflow, and merge the
returned dictionary into its initial-value map.

The complete runnable dependency-free example is
[`examples/cumulant_bridge.jl`](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/blob/main/examples/cumulant_bridge.jl).

## API

```@docs
OrderedLocalMoments
ordered_local_moment
ordered_local_moments
CumulantTermPayload
CumulantModelPayload
CumulantBridgePayload
CumulantComparison
cumulant_model_payload
cumulant_bridge_payload
compare_cumulant_closure
quantumcumulants_initial_values
```
