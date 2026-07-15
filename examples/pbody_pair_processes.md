# Permutation-invariant pair processes

Source: [`pbody_pair_processes.jl`](pbody_pair_processes.jl)

## Model

For `N = 6`, the example combines a two-body Ising interaction, local pair
loss, and collective pair loss. Representative terms are

\[
\sum_{i<j}\sigma_i^z\sigma_j^z
=\frac{J_z^2-NI}{2},\qquad
L_{ij}=\sigma_i^-\sigma_j^- .
\]

The precise factor in the collective identity follows the `Jz` convention in
the source.

## Solution

Use `PBodyHamiltonian` for the two-site Hamiltonian kernel,
`LocalPBodyJump` for the sum over distinct local pairs, and
`CollectivePBodyJump` for the coherent sum of pair amplitudes. These terms
implement the Appendix-D combinatorics directly in PI space.

The model is first lowered with

```julia
prepared = compile(model; backend=:matrixfree)
workspace = LiouvillianWorkspace(prepared)
```

The explicit workspace makes repeated `apply!` calls reuse all matrix-free
scratch storage. This example intentionally accesses solver internals because
its purpose is backend validation: it assembles a sparse Liouvillian from the
same compiled plan, applies both backends to the initially excited `PIState`,
and compares their coordinate derivatives. It also wraps the derivative in a
typed `PIState` to evaluate its physical trace functional.

`diagnostics(prepared)` reports the selected backend and retained heap size.
The assertions require the pair-sum identity, backend action, and trace
derivative errors all to be below `1e-10`.

## Run

```sh
julia --project=. examples/pbody_pair_processes.jl
```

Agreement of the two actions tests the optimized Appendix-D kernel without
relying on a particular time integrator. Production dynamics normally pass
the compiled model directly to `solve_dynamics`; explicit sparse assembly is
used here only as a small-system reference.
