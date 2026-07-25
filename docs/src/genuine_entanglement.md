# Genuine multipartite entanglement

`ppt_mixture_test` implements the permutationally invariant (PI)
positive-partial-transpose-mixture criterion of [Novo, Moroder, and Gühne,
*Phys. Rev. A* **88**, 012305
(2013)](https://doi.org/10.1103/PhysRevA.88.012305). It tests general PI qubit
states spanning several Schur sectors without constructing a `2^N` density
matrix.

This is a solver-backed entanglement test, so install and load the optional
[Clarabel.jl](https://github.com/oxfordcontrol/Clarabel.jl) dependency before
calling it:

```julia
using Pkg
Pkg.add("Clarabel") # once, in the active environment
```

```julia
using PermutationalInvariantDynamics
import Clarabel

basis = PIBasis(3, 2)
rho = ghz_state(basis)

result = ppt_mixture_test(rho)
result.classification
result.genuinely_multipartite_entangled
```

`import Clarabel` activates the package extension. It does not change the PI
representation or load a conic solver during ordinary core-package use.

## What the criterion proves

A state is biseparable when it is a convex mixture of states separable across
possibly different bipartitions. Every such state is a PPT mixture,

```math
\rho = \sum_{M\,|\,\overline M} P_{M\,|\,\overline M},
\qquad
P_{M\,|\,\overline M}\succeq 0,
\qquad
P_{M\,|\,\overline M}^{T_M}\succeq 0.
```

Consequently, proving that a state is *not* a PPT mixture certifies genuine
multipartite entanglement. The converse depends on particle number:

| Result | Certified conclusion |
|---|---|
| `classification == :gme_certified` | A validated numerical dual certificate establishes non-PPT-mixture membership within the reported tolerances and therefore detects genuine multipartite entanglement. |
| `classification == :ppt_mixture`, `N == 3` | For PI three-qubit states, PPT-mixture membership is equivalent to biseparability, so the state is not genuinely tripartite entangled. |
| `classification == :ppt_mixture`, `N >= 4` | The criterion did not detect genuine multipartite entanglement. It does **not** prove biseparability. |
| `classification == :inconclusive` | Neither returned numerical certificate passed validation; draw no entanglement conclusion. |

For the `N == 2` consistency case, PPT is also equivalent to separability.
The public result makes the asymmetry explicit: for `N >= 4`, a validated PPT
mixture has `genuinely_multipartite_entangled === missing` and
`biseparable === missing`, rather than misleading `false` and `true` values.
Only `:gme_certified` is a general-`N` numerical GME certificate. It is not an
exact-arithmetic proof: retain the reported residuals and tighten both solver
and certificate tolerances when a conclusion lies close to the boundary.

Do not classify a result from the sign of `scaled_margin` alone. The function
reports a conclusive classification only after independently checking the
returned primal Schur equality and PSD/PPT blocks, or the dual stationarity
and PSD conditions. A time limit, iteration limit, inaccurate optimizer
status, or failed post-solve check produces `:inconclusive`.

## PI Schur-block formulation

Permutation invariance reduces the exponentially many labeled bipartitions
to the cut sizes

```math
k\,|\,(N-k), \qquad k=1,\ldots,\left\lfloor N/2\right\rfloor.
```

For each cut, its unnormalized PPT operator is invariant under permutations
within both sides. It therefore decomposes into product-Schur blocks indexed
by partitions `alpha` of `k` and `beta` of `N-k`. The implementation stores
their multiplicity-weighted Hermitian matrices
$Y_{k,\alpha,\beta}$ and maximizes a common scaled slack $t$ subject to

```math
Y_{k,\alpha,\beta}\succeq t I,
\qquad
Y_{k,\alpha,\beta}^{T_A}\succeq t I.
```

A nonnegative feasible optimum is therefore a PPT-mixture decomposition; a
validated numerical separating dual establishes nonmembership within the
reported tolerances.

Clebsch--Gordan intertwiners assemble those product blocks into each total
Schur sector. In the package's equation-(7) coefficient convention, the
equality constraint is

```math
\sqrt{f^\lambda}\,C_\lambda
=\sum_{k,\alpha,\beta,r}
U_{\alpha\beta\rightarrow\lambda,r}^{\dagger}
Y_{k,\alpha,\beta}
U_{\alpha\beta\rightarrow\lambda,r}.
```

The left side is the multiplicity-weighted analysis block of the input
`PIState`. This scaling keeps the assembled equality finite whenever the
ordinary PI analysis block is representable. The optimized `scaled_margin`
has the same sign interpretation as the slack $s$ in Eq. (31) of the paper,
but its magnitude is not that unscaled $s$ because the product blocks use
the package's Schur normalization.

Clarabel uses real semidefinite cones. Each complex Hermitian PSD constraint
is therefore represented by its equivalent doubled real symmetric embedding.
This changes storage constants, not the feasible set.

## Reusing a plan

Preparing the Schur recouplings and sparse conic map can dominate a scan. Build
one immutable plan and reuse it for states constructed from the exact same
`PIBasis` object:

```julia
plan = PPTMixturePlan(basis)

results = map(states) do state
    ppt_mixture_test(state; plan=plan)
end
```

Solver state is call-local, so the plan itself is read-only and shareable.
The function validates every input state and never normalizes, truncates,
symmetrizes, or repairs it. The current implementation supports qubits and
`Float32`/`Float64` conic data; qudits and wider floating types are rejected
instead of being silently converted.

`atol` and `rtol` control input-state validation. Optimizer stopping tolerances
are set separately by `solver_atol` and `solver_rtol`, while
`certificate_atol` and `certificate_rtol` govern the independent post-solve
checks. Tighten both groups when auditing a point close to the PPT-mixture
boundary. Additional Clarabel settings can be supplied through
`solver_options`, except for settings controlled by explicit function
keywords.

## Restricted input sectors

A sector-restricted input basis does not justify dropping other sectors from
the decomposition variables. PPT operators for individual cuts can have
support outside the input state's retained sectors even though their PI sum
matches the state. `PPTMixturePlan` therefore constructs a complete internal
PI basis at the same `N`, and inserts exact zero equality targets for every
absent source sector.

This is a correctness safeguard. It also means that restricting the input
state's Schur sectors does not reduce the SDP dimensions:

```julia
restricted = PIBasis(6, 2; sectors=[(6, 0)])
plan = PPTMixturePlan(restricted) # internally enforces every N=6 sector
```

## Scaling and memory

The paper reduces the formulation to a polynomial number of objects: the
number of real parameters is $O(N^7)$, there are $O(N^3)$ product-Schur
matrix blocks, and the largest block dimension is $O(N^2)$. Polynomial does
not mean inexpensive; sparse conic factorization fill and conditioning remain
problem dependent.

`PPTMixturePlan` stores a sparse conic matrix and reports
`estimated_setup_bytes` and `estimated_solve_bytes`. Construction and solving
use the package's default 512 MiB `memory_budget`; a request that exceeds it
raises before the guarded allocation. Increase the budget only after checking
the estimates and available memory. `memory_budget=Inf` is the explicit
opt-out, not an automatic fallback, and the solver estimate cannot strictly
bound problem-dependent factorization fill.

For repeated states, reuse the plan. For increasing `N`, record solver status,
iterations, residual diagnostics, wall time, and peak memory rather than
assuming that polynomial asymptotics alone imply practical tractability.

## Result diagnostics

In addition to the three-valued scientific fields, `PPTMixtureResult` retains
the quantities needed to audit the numerical decision:

- `equality_residual`, `minimum_block_eigenvalue`, and
  `minimum_partial_transpose_eigenvalue` validate a returned primal
  decomposition;
- `dual_stationarity_residual` and `minimum_dual_cone_eigenvalue` validate a
  returned dual witness;
- `primal_objective`, `dual_objective`, `solver_status`, `iterations`, and
  `solve_time` describe the optimizer result;
- `certificate_atol`, `certificate_rtol`, and `message` record how that output
  was classified.

Keep the complete result with published scan data, especially for points near
the inferred boundary.

## API

```@docs
PPTMixturePlan
PPTMixtureResult
ppt_mixture_test
```
