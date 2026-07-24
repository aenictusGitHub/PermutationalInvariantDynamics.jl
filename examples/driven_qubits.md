# Driven qubits with local decay

Source: [`driven_qubits.jl`](driven_qubits.jl)

## Model

The example considers `N = 20` identical qubits with

\[
H=\frac{1}{2}\sum_i\sigma_i^x,\qquad
\dot\rho=-i[H,\rho]+0.1\sum_i\mathcal D[\sigma_i^-]\rho .
\]

Both sums are permutation invariant even though the decay channels are local.
The initial state is the product ground state.

## Prepared solution

The script constructs the model with `LocalHamiltonian` and `LocalJump`, then
calls

```julia
prepared = compile(model; backend=:matrixfree)
```

This lowers all fixed Schur geometry once without storing a Liouvillian
matrix. `solve_dynamics` propagates the typed `PIState` from `t = 0` to `1`
with fixed-step RK4, saves states every `0.25`, and uses 32 RK4 steps between
successive saved times.

Two reusable analysis objects are prepared before propagation:

- `CollectiveObservablePlan` evaluates the total excited-state population at
  every saved state;
- `OneBodyRDMWorkspace` computes the complete final one-qubit density matrix
  in one geometry traversal.

Both share one `OneBodyGeometry`. This specialized one-particle route avoids
preparing the more general SU(2)/Littlewood--Richardson bipartition geometry
needed by `ReductionPlan`; use the latter for reductions to two or more
particles, purity, or negativity.

The script also calls `diagnostics` on the compiled model and the final state.
It asserts that the initial and final states are valid and that the reduced
state has unit trace.

## Run

```sh
julia --project=. examples/driven_qubits.jl
```

The output reports the PI coordinate dimension, selected backend, excitation
fraction at all five saved times, final one-body state, trace error, and
minimum sector eigenvalue. Increase `steps_per_interval` to verify RK4
convergence before using the result quantitatively.
