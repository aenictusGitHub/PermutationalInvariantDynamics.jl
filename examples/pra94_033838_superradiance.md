# Correlated superradiance from PRA 94, 033838

Source: [`pra94_033838_superradiance.jl`](pra94_033838_superradiance.jl)

## Model

For two atoms, correlated and independent emission can be separated as

\[
\mathcal L=\gamma\mathcal D[J_-]
+(\gamma_0-\gamma)\sum_i\mathcal D[\sigma_i^-].
\]

The example fixes `γ0 = 1` and tests several correlation strengths `γ`, from
independent to fully collective decay.

## Solution

`CollectiveJump` and `LocalJump` encode the two contributions without building
the four-dimensional computational-basis master equation. `compile` prepares
the Schur-space kernels once and selects the explicitly requested sparse
backend. Dense exponentiation of that small PI Liouvillian evolves the
initially excited state; this is intentional because the example performs a
pointwise validation against the analytic expressions in Eqs. 41–43 of the
article. For larger systems, pass the compiled model to `solve_dynamics`
instead of forming the exponential.

## Run

```sh
julia --project=. examples/pra94_033838_superradiance.jl
```

The limiting values `γ = 0` and `γ = γ0` are especially useful sanity checks
for rate and dissipator conventions.
