# Kitagawa-Ueda one-axis twisting

Source: [`kitagawa1993_one_axis_twisting.jl`](kitagawa1993_one_axis_twisting.jl)

## Model

For `N = 8` spins initially polarized along `+x`, the one-axis-twisting
Hamiltonian is

\[
H=\chi J_z^2.
\]

It is collective and conserves total spin, so evolution remains in the fully
symmetric sector. A standard analytic benchmark is

\[
\langle J_x(t)\rangle=\frac N2[\cos(\chi t)]^{N-1}.
\]

## Solution

Construct the collective quadratic Hamiltonian, prepare its matrix-free
generator once, and propagate the `+x` product state with the high-level
fixed-step solver:

```julia
prepared = compile(model; backend=:matrixfree)
solution = solve_dynamics(prepared, rho0, (first(times), last(times));
                          saveat=times, steps_per_interval=64)
```

The script builds one `OneBodyGeometry`, prepares `Jx` as a
`CollectiveObservablePlan`, and reuses its Schur blocks at every requested
time. It reports the maximum discrepancy from the exact mean spin. State
diagnostics validate the final trace, Hermiticity, and positivity without
modifying the propagated state.

The final one-particle purity demonstrates the analogous fixed-bipartition
workflow:

```julia
one_body = ReductionPlan(b, 1)
p1 = reduced_purity(last(solution), 1; plan=one_body)
```

For a state or parameter scan, the setup of both plans is paid once. A qudit
`ReductionPlan` can retain many dense Littlewood--Richardson intertwiners, so
it should be kept only when the same `(basis,k)` reduction is repeated.

## Run

```sh
julia --project=. examples/kitagawa1993_one_axis_twisting.jl
```

This benchmark tests coherent nonlinear dynamics independently of dissipative
terms. Spin squeezing can be obtained from the library's collective moments
and covariance functions using the same geometry or prepared spin plans.
Increase `steps_per_interval` to check fixed-step convergence after changing
the Hamiltonian strength or sampling grid.
