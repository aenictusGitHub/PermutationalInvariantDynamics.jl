# PermutationalInvariantDynamics.jl

[![CI](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-stable-blue.svg)](https://aenictusgithub.github.io/PermutationalInvariantDynamics.jl/stable/)
[![Coverage](https://codecov.io/gh/aenictusGitHub/PermutationalInvariantDynamics.jl/graph/badge.svg)](https://codecov.io/gh/aenictusGitHub/PermutationalInvariantDynamics.jl)
[![License: GPL v3](https://img.shields.io/badge/license-GPL--3.0--only-blue.svg)](LICENSE)

Exact dynamics in the permutationally invariant operator subspace, based on
Bastin and Martin, *J. Phys. A* **58**, 275301 (2025).

General PI states may span every Schur sector; the package is not limited to
the fully symmetric Hilbert subspace. At fixed local dimension, the retained
PI operator space grows polynomially with particle number and production
algorithms never construct the full `d^N` Hilbert space.

## Installation

Until the package is registered, install it directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl")
```

For a local development checkout:

```julia
using Pkg
Pkg.develop(path="/path/to/PermutationalInvariantDynamics.jl")
```

After registration in Julia's General registry, use
`Pkg.add("PermutationalInvariantDynamics")`. Julia 1.10 and later are
supported.

## Documentation

Read the [hosted documentation](https://aenictusgithub.github.io/PermutationalInvariantDynamics.jl/stable/),
starting with the framework introduction. Its sources are also available in
[docs/src](docs/src). Then consult
the [architecture and efficient workflows](docs/src/architecture.md) and the
[complete public API index](docs/src/api_reference.md). Every exported binding
has the same source description in Julia's interactive help, for example
`?PIBasis` or `?stationary_state`.

```julia
using PermutationalInvariantDynamics

basis = PIBasis(20, 2)
sx = ComplexF64[0 1; 1 0]
sz = ComplexF64[1 0; 0 -1]
sm = ComplexF64[0 1; 0 0]
model = PIModel(basis, [
    LocalHamiltonian(0.5sx),
    LocalJump(sm; rate=0.1),
    CollectiveJump(sm; rate=0.02),
])
rho0 = iid_pure_state(basis, ComplexF64[1, 0])
prepared = compile(model)                 # geometry is built once
sol = solve_dynamics(prepared, rho0, (0.0, 20.0);
                     saveat=0.1, steps_per_interval=16)
Jz = CollectiveObservablePlan(basis, 0.5sz)
mz = [collective_expectation(rho, Jz) / basis.N for rho in sol]
```

For adaptive or stiff integration, add an OrdinaryDiffEq solver in the active
environment and use the SciML adapter:

```julia
using OrdinaryDiffEq
prob = dynamics_problem(prepared, rho0, (0.0, 20.0))
raw = solve(prob, Rodas5P(); saveat=0.1)
adaptive = PISolution(raw, basis)
```

The same prepared model supports stationary and spectral commands:

```julia
rho_ss = stationary_state(prepared)
slow = liouvillian_spectrum(prepared; target=:largest_real, nev=6)
diagnostics(rho_ss)
recommend_solver(model; task=:steady_state)
```

Stationary and spectral routines reject time-dependent generators. Select an
instant explicitly with `freeze(model; time=t, parameters=p)` or use Floquet
analysis for periodic dynamics.

For fast product-state predictions at much larger `N`, lower the same physical
terms to a one-site mean-field plan:

```julia
mf = MeanFieldPlan(model; limit=:finite)
sigma0 = ComplexF64[1 0; 0 0]
mf_states = solve_meanfield(mf, sigma0, (0.0, 20.0); saveat=0.1)
```

The finite-size closure evaluates the exact initial derivative of
`sigma^otimes N`; its later propagation neglects generated correlations. An
explicit thermodynamic mode is available for rates already scaled with `N`.
See `docs/src/meanfield.md` and `examples/meanfield_time_crystal.jl` for a
side-by-side PI comparison.

Steady-state Schur populations and Liouvillian couplings can be inspected
without a plotting dependency:

```julia
rho_ss = stationary_state(prepared)
populations = schur_block_structure(rho_ss; metric=:population)
display(visualize_schur_blocks(populations;
    show_values=true, show_young_diagrams=true))

generator_blocks = schur_block_structure(prepared)
display(visualize_schur_blocks(generator_blocks;
    scale=:log, show_young_diagrams=true))
```

Rows of a superoperator diagram are output sectors and columns are input
sectors. The axis thumbnails are the Young-diagram shapes of those sectors;
their tooltips report the exact number of standard Young tableaux rather than
selecting a non-canonical filling. Matrix-free analysis probes PI coordinates
only; see `docs/src/schur_visualization.md` for its cost and norm conventions.

The density-operator spectrum stays compressed by exact Schur multiplicities:

```julia
density = pi_density_spectrum(rho_ss)
display(visualize_density_spectrum(density; show_degeneracies=true))
```

This plots one eigenvalue per physical Schur-block eigenvector, not an
expanded `d^N` list.

Liouvillian eigenvalues and Floquet spectra use a corresponding reusable,
dependency-free SVG workflow:

```julia
spectrum = liouvillian_spectrum_data(prepared; nev=12)
display(visualize_spectrum(spectrum))

small_basis = PIBasis(3, 2)
period = 2pi
periodic_rate = (t, p) -> 0.1 * (1 + 0.5 * cos(2pi * t / period))
periodic_model = PIModel(
    small_basis, [LocalJump(sm; rate=periodic_rate)])
F = floquet_propagator(periodic_model, period; steps=256)
multipliers = floquet_spectrum_data(F; period=period)
display(visualize_spectrum(multipliers))
```

See `docs/src/spectral_visualization.md` for compressed density spectra,
stability boundaries, partial Krylov spectra, multiplier unit-circle plots,
and principal Floquet exponents.

Coefficient blocks use the orthonormal convention
`F = sum_T |nu,T,W><nu,T,W'| / sqrt(f^nu)`. Negative time-dependent rates are
accepted; a general negative-rate time-local generator need not produce a
completely positive dynamical map.

## Development

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
julia --project=. benchmark/performance_regression.jl
julia --project=. benchmark/performance_audit.jl
julia --project=docs -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate(); include("docs/make.jl")'
julia --project=quality -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
julia --project=quality quality/quality.jl
```

The [example index](examples/README.md) lists every runnable script and its
same-basename guide. See `docs/src/mathematics.md` for the mathematics,
`docs/src/meanfield.md` for product-state predictions, and
`docs/src/matrix_free_krylov.md` for large-scale solver choices.

## Development disclosure

OpenAI Codex has been used extensively to assist with implementation, tests,
documentation, performance work, and code audits. The maintainers define the
mathematical conventions and remain responsible for reviewing, understanding,
and validating every release candidate against analytical identities,
small-system reference calculations, and published results.

## Citation

Please cite both the software and the underlying framework paper. Machine-
readable metadata are provided in [`CITATION.cff`](CITATION.cff); the paper's
BibTeX entry is in [`CITATION.bib`](CITATION.bib).

## License

PermutationalInvariantDynamics.jl is distributed under the
[GNU General Public License version 3 only](LICENSE) (`GPL-3.0-only`).
