# Steady-state superradiance

Source: [`steady_superradiance.jl`](steady_superradiance.jl)

## Model

The Meiser-Thompson-Holland model combines collective emission and independent
repumping:

```math
\dot\rho=\Gamma_c\mathcal D[J_-]\rho
 +w\sum_i\mathcal D[\sigma_i^+]\rho .
```

The default figure uses `N = 20` and a 41-point logarithmic pump grid, with
$w=N\Gamma_c/2$ and $w=N\Gamma_c$ inserted exactly for 43 sampled states. It therefore resolves the
weak-pump, collective-emission peak, and saturated regimes instead of joining
only five isolated stationary states.

## Solution

Represent collective emission with `CollectiveJump` and local repumping with
`LocalJump`. Only their scalar rates vary, so `compile_family` prepares the
common Schur geometry once. For each `w`, `specialize` binds the new rates and
`stationary_state(...; algorithm=DirectAlgorithm())` solves the
trace-constrained PI linear system. The emitted intensity is

```math
I=\Gamma_c\langle J_+J_-\rangle,
```

and the excited population is contracted through a
`CollectiveObservablePlan`. The script also prints the enhancement relative
to independent emission and the large-`N` estimate `I_max ≃ N^2 Γc/8`.

## Makie figure

The optional three-panel CairoMakie figure shows the steady radiated
intensity, excited-state fraction, and collective enhancement across the pump
scan. The intensity panel includes the large-`N` peak estimate, while the
enhancement panel marks the independent-emitter value. The logarithmic pump
axis retains both the weak- and strong-pumping limits in one view. PDF and PNG
copies are saved as `steady_superradiance.*`.

## Run

```sh
julia --project=examples examples/steady_superradiance.jl
```

Set `PID_EXAMPLE_QUICK=1` for the lightweight `N = 10`, five-point smoke run.
Quick mode retains the same physical equations, normalizations, and numerical
checks.

Finite `N` shifts and rounds the optimum, so compare trends rather than
expecting the asymptotic maximum exactly.

## Expected output

![Expected steady superradiant intensity and enhancement versus pumping](../docs/src/assets/example_figures/steady_superradiance.png)

The points use the default finite ensemble; the asymptotic prediction is
included only as a reference.
