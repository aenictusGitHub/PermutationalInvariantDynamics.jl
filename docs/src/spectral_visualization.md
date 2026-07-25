# Visualizing density, Liouvillian, and Floquet spectra

The spectral visualizers render dependency-free SVG while keeping spectral
computation separate from presentation. Liouvillian and Floquet data use
`ComplexSpectrum` and `visualize_spectrum`. Density operators instead use the
multiplicity-compressed result of `pi_density_spectrum` and the physically
distinct `visualize_density_spectrum`; density eigenvalues are not decay modes
or Floquet multipliers.

## Steady-state density spectrum

After solving a stationary state, diagonalize each physical Schur block once
and reuse the compressed result:

```julia
result = stationary_state(prepared; return_info=true)
rho_ss = result.state

density = pi_density_spectrum(rho_ss)
figure = visualize_density_spectrum(
    density;
    title="Steady-state density spectrum",
    show_degeneracies=true)
display(figure)
```

The horizontal coordinate is the compressed eigenmode rank and the vertical
coordinate is the unmodified real eigenvalue. Colours identify Schur sectors.
Each tooltip gives the partition, its within-sector eigenvalue index, and the
exact symmetric-group degeneracy. With `show_degeneracies=true`, an `×g`
annotation displays the same exact multiplicity next to a point.

The numerical result contains one eigenvalue per eigenvector of each physical
block, not one entry per full-Hilbert-space eigenvector. Its fields include
`values`, exact `BigInt` `degeneracies`, `sectors`, `sector_indices`, and
`total_dimension`. For a complete basis, `total_dimension == d^N`; for a
restricted basis it is the retained Hilbert dimension. The weighted identity

```julia
sum(g * value for (g, value) in zip(
    density.degeneracies, density.values)) ≈ 1
```

checks normalization without expanding the spectrum. Do not pass
`expanded=true` for visualization: expansion can require exponentially many
entries and discards the Schur-sector bookkeeping that makes the plot useful.

The convenience form

```julia
figure = visualize_density_spectrum(
    rho_ss; spectrum_kwargs=(atol=1e-12,))
```

computes the compressed spectrum once. Prefer passing a previously computed
result when its values are also used quantitatively. Visualization never clips
negative eigenvalues. Values below the presentation tolerance receive a red
outline; `presentation_atol` and `presentation_rtol` control only that styling
and do not repair the state or change any stored eigenvalue. Exact zero values
remain visible on the zero reference line.

## Liouvillian spectrum

For a modest PI dimension, obtain a complete spectrum through the existing
high-level solver and render it in the complex plane:

```julia
data = liouvillian_spectrum_data(
    prepared;
    target=:largest_real,
    nev=pi_dimension(prepared),
    algorithm=:dense,
)
figure = visualize_spectrum(data; title="Liouvillian spectrum")
display(figure)
```

The horizontal coordinate is the decay or growth rate
`real(lambda)` and the vertical coordinate is the oscillation frequency
`imag(lambda)`. The line `real(lambda)=0` is the stability boundary. Points
are classified, without modifying their values, as:

- `:stationary` when `lambda` is numerically zero;
- `:peripheral` when its real part is zero but its frequency is nonzero;
- `:decaying` in the open left half-plane; or
- `:unstable` in the open right half-plane.

The tolerance used for this visual classification is stored in
`data.tolerance`; it is not an eigensolver residual and is not used to clip or
move a mode.

For a computed Liouvillian source, solver tolerances use the usual `atol` and
`rtol` keywords, while `classification_atol` and `classification_rtol`
control only the marker classes. For an already computed value vector or
`SpectrumResult`, `atol` and `rtol` classify the stored values because no
solver is run. Residual arrays keep their own numerical precision.

For a large PI dimension, pass selected matrix-free modes that have already
been computed:

```julia
modes = krylov_liouvillian_spectrum(
    prepared.operator;
    nev=12,
    krylovdim=48,
    which=:LR,
    require_convergence=true,
)
data = liouvillian_spectrum_data(modes)
display(visualize_spectrum(data))
```

Residual and convergence arrays remain aligned with the original mode order.
A selected window is marked partial. Its rightmost displayed point is only
the displayed spectral abscissa: a partial ordinary, harmonic, IRAM, or
Jacobi--Davidson plot cannot by
itself certify the global gap, the full stationary multiplicity, or the
absence of an omitted unstable mode.

## Floquet multipliers

If the one-period propagator has already been converged, reuse it directly:

```julia
F = floquet_propagator(periodic_model, period; steps=256)
data = floquet_spectrum_data(
    F; period=period, representation=:multipliers)
display(visualize_spectrum(data; title="Floquet multipliers"))
```

The map `F` is diagonalized once when the data object is constructed. The SVG
shows the unit circle as the discrete-time stability boundary and marks the
fixed multiplier `mu=1`. Multipliers are classified as `:fixed`,
`:peripheral`, `:contracting`, or `:unstable`; values outside the unit disk
remain visible and are never projected back into it.

The source convenience form constructs the one-period map once and then
diagonalizes it:

```julia
data = floquet_spectrum_data(
    periodic_model, period;
    steps=256,
    representation=:multipliers,
)
```

Although the instantaneous Liouvillian can be matrix free, the current
Floquet propagator and its complete eigenspectrum are dense at the PI
coordinate dimension. Reuse a previously computed `F` or multiplier vector
when changing the figure or comparing logarithm branches.

## Principal-branch Floquet exponents

With a positive period, multipliers can be converted to exponents

```math
\xi_j=\mathrm{Log}(\mu_j)/T,
\qquad -\pi/T \leq \mathrm{Im}\,\xi_j \leq \pi/T.
```

```julia
multipliers = floquet_multipliers(F)
exponents = floquet_spectrum_data(
    multipliers;
    input=:multipliers,
    period=period,
    representation=:exponents,
    complete=true,
)
display(visualize_spectrum(exponents))
```

The exponent view shows `real(xi)=0` and the principal quasifrequency
boundaries `imag(xi)=±π/T`. Both endpoints are written because Julia's
principal complex logarithm distinguishes the signed-zero lips of the
negative-real branch cut; away from that cut the usual half-open convention
applies. A zero multiplier has no finite logarithm and is rejected with an
error; use the multiplier representation rather than silently omitting it or
replacing it by a finite value. Passing
`input=:exponents, representation=:multipliers` performs the inverse
conversion explicitly. Supplied exponent data with
`input=:exponents, representation=:exponents` are preserved verbatim and may
lie outside the principal zone; their default plot title therefore says only
“Floquet exponents.” The `metadata.branch` field is `:principal` only when
the library actually applied `Log` to multipliers.

## Reusing results and controlling the viewport

`liouvillian_spectrum_data` accepts a raw eigenvalue vector, a
`SpectrumResult`, or a spectral named tuple with `values` and optional
`residuals`, `converged`, and `dimension` fields. `floquet_spectrum_data`
similarly accepts a multiplier or exponent vector. Input order and repeated
eigenvalues are retained. If residuals accompany a converted Floquet vector,
they remain diagnostics of the input eigenproblem and are not rescaled; inspect
`data.metadata.residual_representation` for their units.

Rendering options include `title`, `width`, `height`, `marker_size`,
`show_legend`, `show_indices`, `xlimits`, and `ylimits`. Explicit limits do
not delete numerical data; the footer reports how many points lie outside the
viewport. Multiplier plots preserve equal aspect ratio so the unit circle is
geometrically meaningful.

```julia
focused = visualize_spectrum(
    data;
    xlimits=(-0.2, 0.02),
    ylimits=(-1.0, 1.0),
    marker_size=5,
    show_indices=true,
)
```

## Saving SVG

```julia
mktempdir() do directory
    path = joinpath(directory, "spectrum.svg")
    save_spectrum_visualization(path, figure)
end
```

For a density spectrum, use the corresponding writer:

```julia
save_density_spectrum_visualization(
    joinpath(directory, "density-spectrum.svg"), density_figure)
```

SVG text, titles, tooltips, axes, and reference boundaries are generated
without a plotting package. See the runnable
`examples/irrep_block_visualization.jl` example for a solved steady-state
Schur-population and compressed-density-spectrum workflow, and
`examples/spectral_visualization.jl` for complete Liouvillian, multiplier,
and exponent workflows with numerical checks.
