# PermutationalInvariantDynamics.jl

[![CI](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://aenictusgithub.github.io/PermutationalInvariantDynamics.jl/dev/)
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

For a cloned repository, instantiate the root environment once before running
its examples or tests:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

After registration in Julia's General registry, use
`Pkg.add("PermutationalInvariantDynamics")`. Julia 1.10 and later are
supported.

## Documentation

Read the [hosted documentation](https://aenictusgithub.github.io/PermutationalInvariantDynamics.jl/dev/),
starting with the step-by-step
[model-to-solution tutorial](https://aenictusgithub.github.io/PermutationalInvariantDynamics.jl/dev/getting_started/).
The browser-only
[PI model code generator](https://aenictusgithub.github.io/PermutationalInvariantDynamics.jl/dev/model_code_generator/)
turns a supported LaTeX subset into a commented stationary-state or
stationary-observable Julia program without sending formulas to a server.
Its sources are
also available in [docs/src](docs/src). Then consult the
[framework and physical conventions](docs/src/framework.md),
the [architecture and efficient workflows](docs/src/architecture.md) and the
[complete public API index](docs/src/api_reference.md). Every exported binding
has the same source description in Julia's interactive help, for example
`?PIBasis` or `?stationary_state`.

Prepared continuation scans, advanced matrix-free Krylov families, explicit
convergence reports, and deterministic PI--HEOM or stochastic PI--HOPS
non-Markovian dynamics are documented in
[parameter scans](docs/src/parameter_scans.md),
[Krylov extensions](docs/src/krylov_extensions.md),
[convergence reports](docs/src/convergence.md), and
[PI--HEOM](docs/src/heom.md) / [PI--HOPS](docs/src/hops.md). Identical systems
with identical independent
finite-cutoff local modes use the fully PI
[pseudomode-supersite workflow](docs/src/pseudomodes.md), while one mode shared
by the complete ensemble uses the factorized
[global-pseudomode workflow](docs/src/global_pseudomodes.md). Generalized qudit coherent-state Q data and
confidence-controlled stochastic stopping are covered by
[qudit phase space](docs/src/qudit_phase_space.md) and
[diffusive/trajectory monitoring](docs/src/diffusive_monitoring.md).
Optional Clarabel, Tables, Makie, Distributed, QuantumCumulants, JLD2, and
HDF5 adapters are summarized in the
[interoperability guide](docs/src/interoperability.md).
The Clarabel extension implements the polynomial-size PI qubit PPT-mixture
test described in the [genuine-multipartite-entanglement
guide](docs/src/genuine_entanglement.md), with the paper's necessary-and-
sufficient criterion for PI three-qubit states and a one-sided, validated
numerical GME certificate for larger systems.
Pure permutation-symmetric qubit states also support an `O(N^4)` second
stabilizer Rényi entropy calculation, with reusable `O(N^3)` transform data and
no enumeration of `4^N` Pauli strings. See the
[nonstabilizerness guide](docs/src/nonstabilizerness.md) for the prepared
workflow and its pure-state scope.
An executable prepared-workflow notebook and its isolated Pluto environment
are available under [notebooks](notebooks/README.md).
For reproducible internal scaling measurements and carefully scoped Julia
ecosystem comparisons, see the [benchmark guide](docs/src/benchmarks.md).

```julia
using PermutationalInvariantDynamics

basis = PIBasis(20, 2)
spin = spin_matrices()  # local order: (|g>, |e>)
model = PIModel(basis, [
    LocalHamiltonian(spin.jx),
    LocalJump(spin.jm; rate=0.1),
    CollectiveJump(spin.jm; rate=0.02),
])
rho0 = computational_product_state(basis, 1)
prepared = compile(model)                 # geometry is built once
sol = solve_dynamics(prepared, rho0, (0.0, 20.0);
                     saveat=0.1, steps_per_interval=16)
Jz = CollectiveObservablePlan(basis, spin.jz)
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

When only a few observables are needed, the fixed-step solver can discard the
sampled PI states and retain scalar series only:

```julia
series = solve_dynamics(
    prepared, rho0, (0.0, 20.0);
    saveat=0.1,
    observables=(magnetization=spin.jz,),
    save_states=false,
)
```

The same output policy is available for trajectory ensembles, where means,
variances, standard errors, and confidence intervals are accumulated online.
See the [streaming-output guide](docs/src/streaming_output.md).

For pure auxiliary paths in the direct sum of Schur irreps, the opt-in
`WeakPITrajectoryPlan` factorizes unresolved local gains into
sector-changing Kraus branches. These pseudo-kets are not labeled-particle
wavefunctions, but their ensemble average reproduces the same PI master
equation without constructing a `d^N` object. See the
[weak-PI trajectory guide](docs/src/weak_pi_trajectories.md).

For a shared structured Gaussian bath, `HOPSPlan` propagates one hierarchy of
those direct-sum Schur pseudo-kets per colored-noise realization. Linear-HOPS
roots remain unnormalized, and `hops_average` reconstructs the density from
their unnormalized outer products. See the
[PI--HOPS guide](docs/src/hops.md) for the exact shared-bath symmetry boundary
and the independent time-step, hierarchy-depth, and sampling refinements.

Autonomous models also support exact PI quantum regression without a dense
Liouvillian. `CorrelationPlan` and `CorrelationWorkspace` are reusable across
two-time correlations, delayed second-order correlations, connected
shifted-GMRES spectra, and finite-window FFTs; see the
[correlation guide](docs/src/correlations.md).

Collective homodyne and heterodyne records are available through preallocated
`DiffusivePlan`/`DiffusiveWorkspace` paths; the monitored collapse dissipator
must already be present in the unconditional model. See the
[diffusive-monitoring guide](docs/src/diffusive_monitoring.md).

PI channels, POVM sampling and tomography, versioned checkpoints, compressed
population metadata, simultaneous weak symmetries, and matrix-free control
gradients are collected in the experimental
[research-utilities guide](docs/src/research_utilities.md).

`CorrelatedLocalJumps` and `CorrelatedCollectiveJumps` accept a Hermitian
positive-semidefinite Kossakowski matrix for cross-correlated one-body noise.
Fixed channel matrices are factorized once, while `InPlaceTimeOperator`
provides a preallocated driven path. See the
[correlated-reservoir example](examples/correlated_reservoirs.md).

Several independently PI ensembles and small truncated auxiliaries can be
combined with `CompositePIBasis`. Factorized local and cross-system maps are
applied without forming the global Kronecker superoperator. Explicit
`CompositeJumpChannel`s add density-valued, preallocated quantum-jump
trajectories with reproducible threaded batches and online statistics. See
[composite systems](docs/src/composite_systems.md) and the
[stochastic composite example](examples/composite_quantum_trajectories.md).
One shared finite-cutoff cavity or reaction coordinate has a specialized
factorized builder, collective coupling convention, and direct system/mode
reductions; see [global pseudomodes](docs/src/global_pseudomodes.md) and the
[shared-cavity example](examples/global_pseudomode_cavity.md).

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
    small_basis, [LocalJump(spin.jm; rate=periodic_rate)])
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

See [CONTRIBUTING.md](CONTRIBUTING.md) for architectural contracts, focused
test groups, and the complete contributor workflow.

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
Release-candidate changes are recorded in the [changelog](CHANGELOG.md); the
[release guide](docs/src/releasing.md) is the maintainer gate for General
registration.

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
