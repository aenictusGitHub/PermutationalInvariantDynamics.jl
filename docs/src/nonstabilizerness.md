# Nonstabilizerness of symmetric qubit states

The second stabilizer Rényi entropy quantifies how far a pure state is from
the stabilizer set. For an `N`-qubit pure state $\rho=|\psi\rangle\langle \psi|$, Passarelli, Fazio, and Lucignano define

```math
M_2(\rho) =
-\ln\!\left[
2^{-N}\sum_{P\in\mathcal P_N}
\bigl(\mathrm{tr}(\rho P)\bigr)^4
\right],
```

where $\mathcal P_N=\{I,X,Y,Z\}^{\otimes N}$ is the phase-free Pauli
basis. The default logarithm is natural, as in [Phys. Rev. A **110**, 022436
(2024)](https://doi.org/10.1103/PhysRevA.110.022436). Set `base=2` to report
the same quantity in bits.

`stabilizer_renyi_entropy` implements this definition for pure,
permutation-symmetric qubit states. It returns zero for a stabilizer state and
is positive for a nonstabilizer state, up to the documented numerical
tolerances.

## PI-efficient evaluation

A direct calculation enumerates $4^N$ Pauli strings. Permutation symmetry
reduces the distinct expectations to the Pauli occupation counts
$(n_I,n_X,n_Y,n_Z)$. The implementation goes further: it evaluates all
representatives with normalized Krawtchouk transforms of the symmetric Schur
block and stable hypergeometric weights.

The resulting costs are:

- $O(N^4)$ arithmetic for one state;
- $O(N^3)$ retained data in a reusable `StabilizerRenyiPlan`;
- $O(N^2)$ mutable scratch in a `StabilizerRenyiWorkspace`.

No state of size $2^N$ and no Pauli table of size $4^N$ is constructed.
Multinomial degeneracies are accumulated in the logarithmic domain, and the
same pass checks the pure-state Pauli second-moment identity as an internal
reliability diagnostic.

## One calculation

For one state, the convenience call prepares and releases its own plan and
workspace:

```julia
using PermutationalInvariantDynamics

N = 12
basis = PIBasis(N, 2; sectors=[(N, 0)])

# A tensor product of single-qubit T states.
ket = ComplexF64[1, cis(pi / 4)] / sqrt(2)
rho = iid_pure_state(basis, ket)

M2 = stabilizer_renyi_entropy(rho)
@assert isapprox(M2, N * log(4 / 3); rtol=1e-10)

M2_bits = stabilizer_renyi_entropy(rho; base=2)
```

The restricted basis is not required: a complete `PIBasis(N, 2)` is also
accepted when the state has support only in its fully symmetric sector.

## Repeated states on one basis

Prepare the immutable transform data once when scanning many pure symmetric
states:

```julia
plan = StabilizerRenyiPlan(basis; T=Float64)
workspace = StabilizerRenyiWorkspace(plan)

for phi in range(0, pi / 2; length=101)
    local_ket = ComplexF64[1, cis(phi)] / sqrt(2)
    state = iid_pure_state(basis, local_ket)
    value = stabilizer_renyi_entropy(
        state; plan=plan, workspace=workspace)
    # store or process value
end
```

The plan is read-only and may be shared. A workspace owns mutable matrices and
must be used by only one task at a time. Construct one workspace per concurrent
task. Both objects are tied to the exact `PIBasis` and numerical precision for
which they were built. Setup and workspace allocations obey the common
`memory_budget` safeguard; the default is 512 MiB and `Inf` is the explicit
opt-out.

## Scope and validation

This API deliberately follows the pure-state measure in the paper. It
therefore rejects:

- local dimensions other than two;
- mixed states, including a maximally mixed state;
- states with support outside the fully symmetric partition $(N,0)$;
- nonunit-trace, non-Hermitian, nonpositive, or nonfinite inputs;
- a plan or workspace built from a different basis or incompatible precision;
- an invalid logarithm base or a request exceeding the memory budget.

The function does not normalize, project, symmetrize, or clip its input. In
particular, applying the fourth Pauli-moment expression to a mixed state would
not define the nonstabilizerness measure used here and is not silently
accepted.

For a numerically prepared state, pass explicit `atol` and `rtol` only when
they represent a justified state-preparation error. These tolerances validate
purity, real Pauli expectations, and the Pauli second-moment identity; support
outside the symmetric sector remains an exact, non-tolerant requirement. They
do not alter the state.

## See also

- [Observables and quantum information](api/analysis.md)
- [API tiers and prepared analysis](api_tiers.md)
- [Framework and conventions](framework.md)
