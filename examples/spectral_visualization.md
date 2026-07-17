# Liouvillian and Floquet spectral visualization

Source: [`spectral_visualization.jl`](spectral_visualization.jl)

## Purpose

This example renders continuous-time Liouvillian eigenvalues and discrete-time
Floquet multipliers and exponents without a plotting dependency. It separates
the numerical data from rendering:

```julia
data = liouvillian_spectrum_data(prepared;
    nev=pi_dimension(prepared), algorithm=:dense)
figure = visualize_spectrum(data; title="Liouvillian spectrum")
```

The `ComplexSpectrum` object stores the values, their stability
classifications, completeness information, and representation convention.
Changing titles, limits, marker sizes, or legends therefore does not repeat an
eigensolve. `save_spectrum_visualization` writes the same SVG that an
SVG-capable notebook displays inline.

## Autonomous Liouvillian

Three qubits undergo a collective coherent drive and independent decay,

\[
\mathcal L(\rho)=-i\Omega[J_x,\rho]
+\gamma\sum_{n=1}^{3}\mathcal D[\sigma_-^{(n)}]\rho,
\qquad \Omega=0.8,\quad\gamma=0.35.
\]

The complete qubit PI basis has coordinate dimension 20 at `N=3`, making a
dense full-spectrum calculation an appropriate diagnostic. In the resulting
complex-plane diagram the horizontal axis is
\(\mathrm{Re}\,\lambda\), the decay or growth rate, and the vertical axis
is \(\mathrm{Im}\,\lambda\), the oscillation frequency. The vertical line
at `Re = 0` is the stability boundary. A trace-preserving relaxing generator
has a stationary eigenvalue at the origin and all other modes in the open
left half-plane. The script asserts both properties.

The example requests every PI eigenvalue explicitly:

```julia
liouvillian_data = liouvillian_spectrum_data(
    autonomous;
    target=:largest_real,
    nev=pi_dimension(autonomous),
    algorithm=:dense)
```

For a larger PI dimension, use a partial matrix-free Krylov or harmonic
Arnoldi spectrum near the stability boundary. Such a result must be marked and
described as partial: it cannot certify the absence of omitted modes or the
full stationary multiplicity. Increase the Krylov dimension until the
requested Ritz residuals have converged.

## Periodic scalar drive

The second model keeps the collective operator fixed but modulates its scalar
rate,

\[
\Omega(t)=0.9[1+0.45\cos(2\pi t/T)],\qquad T=2,
\]

while retaining the same local decay. Fixed-operator scalar time dependence
uses the preallocated compiled kernels. The one-period propagator
\(F=\mathcal T\exp\int_0^T\mathcal L(t)\,dt\) is computed with 128 and 256
RK4 steps; the script requires the two maps to agree before interpreting its
spectrum.

Floquet multipliers \(\mu_j\) are displayed in the complex plane. The unit
circle is the discrete-time stability boundary: a stable quantum channel has
\(|\mu_j|\leq1\), and trace preservation supplies a fixed multiplier
\(\mu=1\). Arguments give phases accumulated per period, while magnitudes
give stroboscopic damping.

The corresponding exponents use the principal complex logarithm,

\[
\xi_j=\frac{\mathrm{Log}\,\mu_j}{T},
\qquad -\frac{\pi}{T}\leq\mathrm{Im}\,\xi_j\leq\frac{\pi}{T}.
\]

Their real and imaginary axes again represent decay rates and frequencies,
but quasifrequencies outside that interval differ by an integer multiple of
\(2\pi/T\) and fold onto the same multiplier. The branch choice is therefore
part of the data convention, not evidence that other logarithm branches do
not exist. Julia preserves the signed-zero side of the negative-real branch
cut, so either endpoint can occur exactly on that cut.

## Reusing precomputed multipliers

The converged map is diagonalized once, and its stored values are reused:

```julia
multiplier_data = floquet_spectrum_data(F;
    period=T, representation=:multipliers)
multipliers = multiplier_data.values
exponent_data = floquet_spectrum_data(multipliers;
    input=:multipliers, period=T,
    representation=:exponents)
```

Both diagrams reuse the same multiplier vector. The example checks the fixed
mode, the closed-unit-disk condition, exponent stability, the principal
quasifrequency strip, and a direct multiset comparison with
`log.(multipliers) / T`.

The three SVG files are written only inside `mktempdir()`, checked, and removed
automatically. To retain a figure in a research workflow, pass a persistent
path explicitly; saving never recomputes its spectral data.

## Run

```sh
julia --project=. examples/spectral_visualization.jl
```
