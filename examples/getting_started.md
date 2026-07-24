# Getting started: model to solution

Source: [`getting_started.jl`](getting_started.jl)

This is the canonical first runnable example. The fully explained version is
the documentation chapter
[Getting started: from a model to a solution](../docs/src/getting_started.md).

## Model

Eight identical qubits experience a transverse drive, independent emission,
and independent pumping:

```math
\dot\rho=-i\left[\frac{0.7}{2}\sum_i\sigma_i^x,\rho\right]
+0.12\sum_i\mathcal D[\sigma_i^-]\rho
+0.02\sum_i\mathcal D[\sigma_i^+]\rho.
```

The script follows the full prepared workflow:

```text
PIBasis → local matrices → PIModel → initial PIState → compile
        → solve_dynamics / stationary_state → diagnostics → convergence check
```

It samples the excitation fraction during fixed-step RK4 evolution, solves the
autonomous stationary state with residual metadata, validates both density
operators, and compares 8 with 16 RK4 steps between output times.

## Expected output

![Prepared PI dynamics and RK4 refinement](../docs/src/assets/example_figures/getting_started.png)

The left panel compares the excitation fraction obtained with 16 RK4 steps
per saved-time interval against the coarser 8-step result and the independently
computed stationary value. The right panel shows the pointwise difference
between the two time-step resolutions on a logarithmic scale. The preview was
generated with the default rates, particle number, time grid, and tolerances;
for quantitative use, repeat the refinement with smaller steps and check the
stationary residual and state diagnostics independently.

## Run

From the repository root:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'  # first checkout only
julia --project=. examples/getting_started.jl
```

The complete PI basis has 165 coordinates, compared with 65,536 entries in a
general full density operator. The numerical comparison checks integration
resolution; it is separate from the stationary residual and state-validity
checks.

Use the examples environment described in [`README.md`](README.md) to write
the optional PDF and PNG figure. A root-project run remains dependency-free
and skips only the rendering block when CairoMakie is unavailable.
