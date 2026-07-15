# Sector-resolved spin phase space

The spin phase-space functions turn a qubit PI density operator into Husimi-Q
or spin-Wigner data on the sphere without reconstructing its ``2^N`` Hilbert
space. They work one Schur block at a time and distinguish the total-spin
label from the angular coordinates.

## Why a PI state has several spheres

For qubits, a retained partition

```math
\nu=(\nu_1,\nu_2)
```

carries total spin ``j=(\nu_1-\nu_2)/2``, spin dimension ``n_j=2j+1``, and
symmetric-group multiplicity ``f^\nu``. A general PI state can occupy several
such sectors. It therefore does not belong to one spin-``j`` Hilbert space and
does not have one irreducible spin sphere.

The canonical result returned by this package is the discrete collection of
sector spheres. On each sphere the multiplicity-weighted Schur block

```math
\bar\rho_j=f^\nu\rho_j=\sqrt{f^\nu}\,C_\nu
```

has trace equal to the physical sector population ``p_j``. Here ``C_\nu`` is
the stored equation-(7) block and ``\rho_j=C_\nu/\sqrt{f^\nu}`` is the
physical block.

`SpinPhaseSpaceData.values` is the sum of the selected sector densities. It is
useful as a two-dimensional angular marginal, but it is explicitly obtained
by forgetting the discrete ``j`` label. It must not be interpreted as though
all sectors were one larger spin.

## Husimi-Q normalization

For a spin coherent state ``|\theta,\phi;j\rangle``, the package returns

```math
P_Q(j,\theta,\phi)=\frac{2j+1}{4\pi}
\langle\theta,\phi;j|\bar\rho_j|\theta,\phi;j\rangle.
```

This is a density with respect to
``d\Omega=\sin\theta\,d\theta\,d\phi``:

```math
\int P_Q(j,\Omega)d\Omega=p_j.
```

It is nonnegative for a physical input state. The implementation evaluates
the coherent amplitudes with a normalized recurrence about the binomial mode,
so it does not convert exponentially large binomial coefficients to floating
point. Azimuthal dependence is evaluated as a finite Fourier series.

## Spin-Wigner convention

`spin_wigner` uses the Agarwal multipole construction with orthonormal
Condon--Shortley polarization tensors ``T_{kq}`` and standard spherical
harmonics ``Y_{kq}``. With

```math
t_{kq}=\operatorname{tr}(T_{kq}^\dagger\bar\rho_j),
```

the normalized sector density is

```math
P_W(j,\theta,\phi)=
\sqrt{\frac{2j+1}{4\pi}}\sum_{k=0}^{2j}\sum_{q=-k}^{k}
t_{kq}Y_{kq}(\theta,\phi).
```

It also integrates to ``p_j``. A maximally mixed conditional block is the
uniform density ``p_j/(4\pi)``. For one qubit with Bloch vector ``r``, the
convention reduces to

```math
P_W(\mathbf n)=\frac{1+\sqrt{3}\,\mathbf r\cdot\mathbf n}{4\pi}.
```

Unlike the Q distribution, the Wigner distribution may be negative. Negative
values are retained numerically and displayed with a diverging palette; they
are never clipped or repaired.

## Computing and retaining sector data

```julia
using PermutationalInvariantDynamics

basis = PIBasis(4, 2)
rho = spin_coherent_state(basis, 0.8, 0.3)

q = spin_husimi_q(rho; ntheta=91, nphi=180, resolved=true)
w = spin_wigner(rho; ntheta=91, nphi=180, resolved=true)

q.values                    # aggregate, indexed [phi, theta]
q.sectors                   # retained partitions
q.populations               # exact phase-space integral of each sector
q.sector_values[1]          # first sector sphere
```

The default `resolved=false` streams one sector grid into the aggregate using
reusable scratch. This keeps retained phase-space RAM proportional to one
angular grid. Pass `resolved=true` only when the individual matrices will be
inspected or rendered. A subset can be selected explicitly:

```julia
symmetric = spin_husimi_q(
    rho; sectors=Partition((4, 0)), resolved=true)
```

The aggregate then integrates to the selected population, not necessarily
one. Sector selection never renormalizes the result.

The coherent-state Q evaluation costs
``O(n_\theta\sum_j(n_j^2+n_\phi n_j))``. Wigner multipole setup costs
``O(\sum_j n_j^3)`` and uses only sector-sized, call-local scratch; its grid
evaluation has the same Fourier scaling as Q. No global mutable geometry
cache or full Hilbert-space state is introduced.

## Dependency-free SVG

Numerical calculation and presentation are separate:

```julia
figure = visualize_spin_phase_space(
    w; title="Spin Wigner marginal")
display(figure)

sector_figure = visualize_spin_phase_space(
    w; sector=first(w.sectors), title="Symmetric-sector Wigner function")
```

The SVG is an equirectangular heatmap: azimuth ``\phi`` is horizontal and
polar angle ``\theta`` is vertical, with the north pole at the top. Rendering
a sector requires data computed with `resolved=true`. Husimi data use a
sequential color scale, while Wigner data use a zero-centered diverging scale.
Explicit `colorlimits` only change the mapping to colors; the numerical data
are retained unchanged.

The renderer groups equal-color cells into a bounded number of SVG paths and
has no plotting dependency. It requires regular, increasing coordinate grids
and rejects more than 100,000 cells rather than silently resampling. Save the
same notebook representation with
`save_spin_phase_space_visualization`.

See the paired runnable example `examples/spin_phase_space.jl` for a
mixed-sector state, normalization checks, sector rendering, and temporary SVG
output.
