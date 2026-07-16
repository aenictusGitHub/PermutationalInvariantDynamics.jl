# Zhang--Mølmer superradiant quantum trajectories

Source:
[`zhang2018_superradiant_trajectories.jl`](zhang2018_superradiant_trajectories.jl)

## Published model

Zhang, Zhang, and Mølmer,
[*New J. Phys.* **20**, 112001 (2018)](https://doi.org/10.1088/1367-2630/aaec36),
introduced an efficient Dicke-basis Monte Carlo treatment of collectively and
individually decaying atoms. The decay-only specialization of their Eq. (1)
is, in this package's dissipator convention,

```math
\dot\rho=\Gamma_c\mathcal D[J_-]\rho
 +\gamma_l\sum_{i=1}^N\mathcal D[\sigma_-^{(i)}]\rho.
```

For an initially excited ensemble, Fig. 2(a,b) compares
``\gamma_l/\Gamma_c=1`` and ``10`` using ``N=50`` and 512 trajectories. Its two
radiated fluxes are

```math
I_c=\Gamma_c\langle J_+J_-\rangle,
\qquad
I_{\rm fs}=\gamma_l\left(N/2+\langle J_z\rangle\right).
```

`zhang2018_superradiance_model` and `zhang2018_radiation_operators` in
[`paper_models.jl`](paper_models.jl) keep these rate and spin conventions in
one place. The free Hamiltonian and collective Lamb shift in the complete
article model commute with the decay-only Dicke populations and hence do not
change this pulse comparison.

## What the example compares

The default ``N=10``, 256-trajectory run is a finite-size reproduction of the
published model, initial condition, rate ratios, and observables. It is sized
as a practical package regression rather than a digitization of Fig. 2. For
each rate ratio it compares two distinct numerical routes:

1. `PopulationPlan` certifies closure of the Schur/GT diagonal subspace. A
   dense exponential of its 36-coordinate generator gives the deterministic
   master-equation reference, instead of exponentiating the 286-coordinate PI
   Liouvillian.
2. `quantum_trajectories(...; algorithm=:event)` evolves the general PI
   conditional density operator with continuous jump times. The cavity and
   free-space means must agree pointwise with the master equation within six
   reported Monte Carlo standard errors plus a small numerical floor.

The example also checks normalization, the exact initial fluxes
``I_c(0)=N\Gamma_c`` and ``I_{\rm fs}(0)=N\gamma_l``, and the qualitative
contrast of the two panels: a collective burst for comparable rates and
free-space-dominated radiation for strong local decay.

To approach the paper's sampling parameters, change the constants at the top
of the script to `N = 50` and `ntrajectories = 512`, extend or refine the time
grid as needed, and converge the adaptive tolerances. This library stores PI
density operators at every requested time, so that run uses substantially
more memory than the paper's single-pseudo-state representation.

## Unravelling boundary

The deterministic master equation and all ensemble-linear observables are the
same, but the conditional records are not identical. Zhang *et al.* resolve a
local event into ``J\to J+s`` branches and propagate a pure symbolic Dicke
pseudo-state. This package combines the indistinguishable particle-local
outcomes into one completely positive PI gain map, so a local jump generally
produces a mixed conditional state. The example therefore compares ensemble
intensities, not individual ``(J,M)`` paths or trajectory variances.

Lloyd, Ziolkowska, and Keeling's recent
[permutation-symmetric trajectory method](https://arxiv.org/abs/2605.11103)
is especially relevant to this distinction and directly builds on the same PI
operator framework. Its published numerical cases all include a shared
bosonic cavity mode (and one uses a jump/diffusion hybrid), which the current
spin-only package does not represent. Reproducing those figures honestly
requires an auxiliary common-system Hilbert factor and sector-shift-resolved
local channels; they are documented as future scope rather than emulated here.

## Makie figure

The generated two-panel Makie figure corresponds to the two decay-rate ratios.
Each panel overlays the deterministic cavity and free-space intensities with
trajectory means and one-standard-error bands. It is saved as
`zhang2018_superradiant_trajectories.pdf` and `.png`, providing a direct visual
counterpart to the pointwise statistical assertions.

## Run

```sh
julia --project=examples examples/zhang2018_superradiant_trajectories.jl
```

The script prints the pulse maxima, their times, channel-resolved mean jump
counts, and the largest trajectory/master-equation discrepancy in standard-
error units for each rate ratio.
