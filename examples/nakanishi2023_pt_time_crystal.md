# Balanced-gain/loss dissipative time crystal

Source: [`nakanishi2023_pt_time_crystal.jl`](nakanishi2023_pt_time_crystal.jl)

Literature: Y. Nakanishi and T. Sasamoto, *Phys. Rev. A* **107**, L010201
(2023), [arXiv:2203.06672](https://arxiv.org/abs/2203.06672).

## Model and convention mapping

Equation (13) of the paper considers a collective spin \(S\),

\[
\mathcal L\rho=-ig[S_x,\rho]
+\frac{\kappa(1+p)}{S}\mathcal D_{\rm paper}[S_x^+]\rho
+\frac{\kappa(1-p)}{S}\mathcal D_{\rm paper}[S_x^-]\rho,
\qquad S_x^\pm=S_y\pm iS_z .
\]

The paper defines

\[
\mathcal D_{\rm paper}[L]\rho
=2L\rho L^\dagger-L^\dagger L\rho-\rho L^\dagger L,
\]

whereas `CollectiveJump` uses the standard convention

\[
\mathcal D_{\rm std}[L]\rho
=L\rho L^\dagger-\tfrac12\{L^\dagger L,\rho\}.
\]

Thus \(\mathcal D_{\rm paper}=2\mathcal D_{\rm std}\). For \(N\) qubits,
\(S=N/2\), so `nakanishi2023_pt_model` passes the library rates
\(4\kappa(1+p)/N\) and \(4\kappa(1-p)/N\) to the two collective jumps. This
factor-of-two conversion is essential when comparing the finite-size decay
rates.

All terms are collective and conserve total spin. The example therefore uses
only the fully symmetric partition `(N, 0)`; its Hilbert-space dimension is
\(N+1\), and its PI Liouville-space dimension is \((N+1)^2\).

## Exact balanced spectrum

For balanced gain and loss, \(p=0\), Eq. (14) gives the complete finite-\(N\)
spectrum

\[
\lambda_{l,q}=igq-\frac{4\kappa}{N}
\left[|q|+l(1+l+2|q|)\right],
\]

with \(q=-N,\ldots,N\) and \(l=0,\ldots,N-|q|\). These ranges produce exactly
\((N+1)^2\) eigenvalues, including degeneracies. The script diagonalizes the
small \(N=6\) PI Liouvillian and compares both spectra with a bipartite
multiset match. It does not sort complex eigenvalues and compare positions,
which would be fragile at degeneracies.

The \(l=0,\ |q|=1\) pair controls the slowest decay, hence

\[
\Delta_N=\frac{4\kappa}{N}.
\]

The script checks this value through `pi_liouvillian_gap`. It also solves for
the steady state and verifies the paper's balanced-case result
\(\rho_{\rm ss}=I_{N+1}/(N+1)\) inside the symmetric spin irrep.

## Matrix-free dynamics

The complete spectrum is a small-system validation, not the scalable route.
For \(N=24\), the example recompiles with `backend=:matrixfree` and evolves a
state initially polarized along \(+z\). The exact \(q=\pm1,\ l=0\) modes imply

\[
\frac{\langle S_z(t)\rangle}{S}
=e^{-4\kappa t/N}\cos(gt),
\]

which is compared directly with the preallocated RK4 evolution returned by
`solve_dynamics`.

At every finite \(N\), these oscillations still decay. What closes as
\(N^{-1}\) is their decay rate, so persistent oscillations are an
infinite-size statement. The exact spectrum and uniform-state assertions in
this example apply to the balanced case \(p=0\); they must not be reused for
\(p\ne0\).

## Makie figure

With CairoMakie available, the script generates a two-panel comparison. The
complex-plane panel overlays the complete `N=6` PI spectrum with every value
of Eq. (14), retaining exact degeneracies. The dynamics panel overlays the
matrix-free `N=24` magnetization with its exponentially damped analytical
curve. The visible damping is essential: persistent oscillations are not
claimed at either finite size.

The vector and raster outputs are saved as
`nakanishi2023_pt_time_crystal.pdf` and
`nakanishi2023_pt_time_crystal.png` in the configured example-figure
directory.

## Run

```sh
julia --project=examples examples/nakanishi2023_pt_time_crystal.jl
```

Running under the root package environment still performs the numerical
validation; without CairoMakie, it logs that the optional figure was skipped.
