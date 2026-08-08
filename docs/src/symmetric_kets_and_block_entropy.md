# Symmetric pure kets and block-resolved entropy

Two related tools avoid work that a density-first calculation does not need:

- [`SymmetricKet`](@ref) stores a genuine pure state when the dynamics remain
  in the sole fully symmetric Hilbert-space irrep;
- [`HilbertBlockEntropyPlan`](@ref) splits a PI density operator into certified
  Hilbert-space charge blocks before diagonalization.

They solve different problems. A symmetric ket is a state representation for
closed, fully symmetric pure-state dynamics. A Hilbert-block entropy plan is
an analysis object for a `PIState`, including mixed and multi-sector states.
Neither routine assumes a full tensor-product state of size $d^N$.

## Choose the right ket-like object

The package has three objects that contain Schur-irrep amplitudes, but only one
is an ordinary physical pure ket.

| Object | Meaning | Normalization and phases | Intended use |
|---|---|---|---|
| `SymmetricKet` | Physical ket in the single irrep $(N,0,\ldots,0)$ | Unit norm; every relative phase is physical | Closed Hamiltonian evolution, expectations, and algebraic partial traces |
| `WeakPIPseudoKet` | Auxiliary direct sum $\bigoplus_\nu U_\nu$ | Unit norm; phases between different Schur sectors are not physical | Weak-PI quantum-jump trajectories, including sector-changing local jumps |
| `HOPSRootKet` | Root amplitude of a linear HOPS hierarchy in the same direct-sum coordinates | Deliberately unnormalized; densities require an ensemble average of unnormalized outer products | Shared non-Markovian Gaussian baths |

Do not replace one object by another merely because their data are vectors. In
particular, a `SymmetricKet` cannot represent a mixed state or a pure state in
several Schur sectors. Conversely, a `WeakPIPseudoKet` is not a pure state of
the labeled-particle Hilbert space. See [Weak-PI pseudo-ket
trajectories](weak_pi_trajectories.md) and [PI--HOPS stochastic non-Markovian
dynamics](hops.md) for those two auxiliary conventions.

## Physical symmetric-ket workflow

Retain only the fully symmetric sector when constructing the basis. The local
Hamiltonian below is lifted directly to sparse symmetric-occupation support.

```julia
using LinearAlgebra
using PermutationalInvariantDynamics

N = 40
basis = PIBasis(N, 2; sectors=[(N, 0)])

plus_x = ComplexF64[1, 1] / sqrt(2)
psi0 = symmetric_product_ket(basis, plus_x)

omega = 0.7
hz = ComplexF64[1 0; 0 -1] / 2
plan = SymmetricKetHamiltonianPlan(
    basis, hz;
    representation=:local,
    rate=omega,
)

work = SymmetricKetWorkspace(plan)
psi_t = time_evolve_symmetric_ket(
    plan, psi0, (0.0, 1.0);
    steps=512,
    workspace=work,
)
validate_symmetric_ket(psi_t; atol=1e-10, rtol=1e-9)

# No density matrix is needed for this bare Hamiltonian-block expectation.
hz_mean = symmetric_ket_expectation(psi_t, plan)
```

When a Hamiltonian plan is supplied as the observable, the expectation uses
its lifted Hamiltonian block; the separate `rate/hbar` evolution scale is not
multiplied into the reported value.

The fixed-step driver never renormalizes the state. It validates the final
norm by default and raises when the requested `atol`/`rtol` is not met, so
integration error is exposed instead of hidden; `check=false` is an explicit
advanced opt-out. For an autonomous plan, use the
adaptive restarted-Arnoldi exponential action when it is faster than many RK4
steps:

```julia
krylov_work = KrylovExpvWorkspace(ComplexF64, length(psi0), 30)
result = krylov_time_evolve_symmetric_ket(
    plan, psi0, 1.0;
    workspace=krylov_work,
    atol=1e-11,
    rtol=1e-9,
)
psi_t = result.state
```

Callable rates are supported by the fixed-step action but rejected by the
single-exponential Krylov route. Operator-valued schedules and dissipative
terms belong in a density-matrix `PIModel` workflow.

Occupation basis vectors are available without scanning the symmetric block:

```julia
psi_occ = symmetric_occupation_ket(basis, (N - 3, 3))
```

Use [`symmetric_ket_density`](@ref) only when an analysis specifically needs a
`PIState`. The allocating conversion enforces the package's 512 MiB default
memory budget; `memory_budget=Inf` is an explicit opt-out, not a recommended
large-system route. The in-place form has no size-dependent allocation in its
ordinary finite-range contraction loop; its `memory_budget` governs the rare
wider-precision certification used when a product is numerically ambiguous.
The reverse
conversion [`symmetric_ket`](@ref) checks that
the input is a normalized rank-one state in the sole symmetric sector; it
never purifies a mixed state.

## Algebraic trace between local degrees of freedom

Suppose every particle has two two-level factors, so its local dimension is
$d=2\times2=4$. A `LocalFactorTracePlan` removes the same internal factor from
every particle. Its symmetric-ket overload evaluates only the source
coordinates requested by the exact sparse trace transform and does not first
allocate the $g\times g$ pure density block.

```julia
N = 4
joint_basis = PIBasis(N, 4; sectors=[(N, 0, 0, 0)])
psi_joint = symmetric_product_ket(
    joint_basis,
    ComplexF64[1, 0, 0, 0],
)

trace_plan = LocalFactorTracePlan(
    joint_basis, (2, 2);
    traced_factor=2,
)
trace_work = LocalFactorTraceWorkspace(trace_plan)
rho_first = local_factor_trace(
    psi_joint, trace_plan;
    workspace=trace_work,
    memory_budget=512 * 1024^2,
)

algebraic_entropy = von_neumann_entropy(rho_first; base=2)
```

The output is a physical PI density operator for the kept degree of freedom,
so its von Neumann entropy is the pure-state entanglement entropy between the
two local factor algebras. For a mixed joint state it is only the entropy of
the reduced state, not by itself an entanglement measure.
The same budget controls rare wider-precision certification of a requested
source contraction; ordinary exact-zero diagonals stay on the preallocated
native-precision path.

This workflow implements the representation-theoretic strategy motivated by
J. D. Wilson, J. T. Reilly, and M. J. Holland,
[*Efficient Polynomial-Scaled Determination of Algebraic Entanglement Entropy
Between Collective Degrees of Freedom*, arXiv:2603.00464v3
(2026)](https://arxiv.org/abs/2603.00464). Their central example decomposes a
fully symmetric SU(4) space under its SU(2) × SU(2) subgroup. The package
uses its general PI Schur representation and exact local-factor trace rather
than constructing a $4^N$ ket or a $2^N$ reduced density matrix.

## Prepared Hilbert-block entropy

`PIState` entropy is already evaluated sector by sector. If a state is also
block diagonal inside those sectors, a `HilbertBlockEntropyPlan` can reduce
the remaining eigensystems. A diagonal local unitary defines charge blocks:

```julia
charge = Diagonal(ComplexF64[1, -1])
entropy_plan = HilbertBlockEntropyPlan(rho_first.basis, charge)
entropy_work = HilbertBlockEntropyWorkspace(entropy_plan, Float64)

answer = block_von_neumann_entropy(
    rho_first, entropy_plan;
    base=2,
    workspace=entropy_work,
    return_info=true,
)

answer.value
answer.diagnostics.block_reason
answer.diagnostics.estimated_cubic_fraction
```

The constructor checks the local unitary and groups GT patterns by its
$N$-particle charge. An explicit partition can instead be supplied as
`sector => indices` pairs or named tuples with `sector`, `indices`, and an
optional `label`. Every index must appear exactly once; omissions and overlaps
raise.

If a model has already passed the term-resolved strong-symmetry checks, the
same grouping and its model certificate can be reused:

```julia
reduction = strong_symmetry_reduction(model)
entropy_plan = HilbertBlockEntropyPlan(reduction)
```

The model certificate does **not** certify an arbitrary state. Both
[`block_entropy_diagnostics`](@ref) and
[`block_von_neumann_entropy`](@ref) still check trace, Hermiticity,
positivity, and off-block leakage for the supplied `PIState`.

Exact block support is required by default. Passing nonzero `block_atol` or
`block_rtol` explicitly requests the entropy of the block-diagonal projection
when leakage is within tolerance. The diagnostics then report
`block_reason == :within_tolerance`; the library never silently drops those
coherences. Reuse one `HilbertBlockEntropyWorkspace` per task; it retains one
dense scratch matrix with the size of the largest prepared block and enforces
the standard memory budget at construction. Each entropy call also guards a
conservative bound for eigensolver transients, including when the workspace is
supplied by the caller.

## Scaling and practical limits

For the fully symmetric irrep,

```math
g=\binom{N+d-1}{d-1}.
```

A `SymmetricKet` stores $O(g)$ amplitudes, while its explicit `PIState` density
stores $O(g^2)$ coefficients. At fixed $d=4$, these are respectively
$O(N^3)$ and $O(N^6)$. Fixed one-body Hamiltonians use exact sparse
occupation support, but an already-lifted dense block can still cost
$O(g^2)$, and not every `PIModel` term or schedule has a ket-native route.

The ket-native local-factor trace avoids the $O(g^2)$ *source-density*
allocation, but the reusable `LocalFactorTracePlan`, its exact sparse
transform, and the reduced output density still consume memory. Prepare that
plan once, observe its memory guard, and reuse one task-owned workspace per
concurrent calculation.

For Hilbert blocks of sizes $n_b$ inside Schur sectors of sizes $m_\nu$, the
spectral work estimate changes from
$\sum_\nu m_\nu^3$ to $\sum_b n_b^3$. The diagnostics report both estimates;
they are operation-count proxies, not wall-clock guarantees. Small blocks can
be dominated by setup and dispatch overhead, so benchmark the repeated
analysis that matters to the study.

These APIs do not compute particle-bipartition negativity, replace a
dissipative density evolution by a pure Hamiltonian evolution, or discover an
arbitrary non-diagonal symmetry. Use `ReductionPlan`, an ordinary `PIModel`,
or an explicit Hilbert-block partition for those tasks.
