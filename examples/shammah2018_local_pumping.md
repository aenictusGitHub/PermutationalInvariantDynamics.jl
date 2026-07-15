# Independent pumping and emission

Source: [`shammah2018_local_pumping.jl`](shammah2018_local_pumping.jl)

## Model

Ten qubits experience independent emission at rate `γ↓ = 1` and pumping at
rate `γ↑ = 0.3`:

\[
\dot\rho=\gamma_\downarrow\sum_i\mathcal D[\sigma_i^-]\rho
+\gamma_\uparrow\sum_i\mathcal D[\sigma_i^+]\rho .
\]

The exact stationary state is the product of identical diagonal one-qubit
states with excited population `γ↑/(γ↓+γ↑)`.

## Solution

Add two `LocalJump` terms, prepare their common Schur geometry once, and call
the typed high-level stationary-state solver:

```julia
prepared = compile(model)
result = stationary_state(prepared; return_info=true)
numeric = result.state
```

The returned `SteadyStateResult` exposes the physical `PIState` together with
the selected method, convergence flag, stationarity residual, and trace error.
The example checks convergence and `diagnostics(numeric).valid`, constructs the
exact iid PI state independently, and compares their density operators. This
exercises multiple Schur sectors: restricting to the fully symmetric sector
would be incorrect for independent incoherent processes.

## Run

```sh
julia --project=. examples/shammah2018_local_pumping.jl
```

The trace, positivity, Liouvillian residual, and distance to the exact product
state provide complementary checks of the numerical solution.
