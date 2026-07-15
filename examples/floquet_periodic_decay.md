# Floquet dynamics with periodic decay

Source: [`floquet_periodic_decay.jl`](floquet_periodic_decay.jl)

## Model

Four qubits undergo independent spontaneous emission with a periodic rate

\[
\gamma(t)=0.4\left[1+0.7\cos(2\pi t/T)\right],\qquad T=2.
\]

The one-period quantum channel is the Floquet propagator
`U_F = T exp(∫_0^T L(t)dt)`.

## Solution

The time-dependent model is prepared once with
`compile(model; backend=:matrixfree)`. Its scalar rate is evaluated in the
preallocated Liouvillian kernels; no instantaneous sparse matrix is assembled.
The prepared model is passed to `floquet_propagator` with 40, 80, and 160 RK4
steps per period.

A second, constant-rate model is compiled with the sparse backend solely to
form the analytical reference

```julia
exact = exp(period * Matrix(liouvillian(constant_prepared)))
```

Because all Liouvillians here are proportional to the same decay generator,
they commute at different times. Consequently the numerical propagator can be
checked against `exp(T * L_average)`. The successive step counts demonstrate
the expected convergence.

At 160 steps the script also:

- reuses the already computed channel in
  `stationary_state(F - I; algorithm=SVDAlgorithm(), return_info=true)`,
  avoiding a second one-period integration while returning a typed periodic
  `PIState` and solver information;
- computes the leading multipliers and Floquet gap;
- applies the one-period channel four times to an initially excited state;
- reuses a `CollectiveObservablePlan` for the excited fraction; and
- validates the periodic state with `diagnostics`.

The assertions require a one-period error below `2e-8`, a periodic-state
residual below `1e-10`, and a valid density operator.

## Run

```sh
julia --project=. examples/floquet_periodic_decay.jl
```

For noncommuting drives, retain the step-doubling check; the averaged
Liouvillian is then generally not an exact reference.
