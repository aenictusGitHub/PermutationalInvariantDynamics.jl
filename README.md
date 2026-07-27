# PermutationalInvariantDynamics.jl

[![CI](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://aenictusgithub.github.io/PermutationalInvariantDynamics.jl/dev/)
[![Coverage](https://codecov.io/gh/aenictusGitHub/PermutationalInvariantDynamics.jl/graph/badge.svg)](https://codecov.io/gh/aenictusGitHub/PermutationalInvariantDynamics.jl)
[![License: GPL v3](https://img.shields.io/badge/license-GPL--3.0--only-blue.svg)](LICENSE)
[![REUSE status](https://api.reuse.software/badge/github.com/aenictusGitHub/PermutationalInvariantDynamics.jl)](https://api.reuse.software/info/github.com/aenictusGitHub/PermutationalInvariantDynamics.jl)

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
supported. The [installation and environment guide](docs/src/installing.md)
covers isolated project environments, `doctor()`, optional integrations, and
useful problem reports.

## Choose your path

### First 5 minutes

Start from a convention-tested recipe and keep the complete PI basis:

```julia
using PermutationalInvariantDynamics

model = Models.driven_qubits(8)
spin = spin_matrices()
rho0 = computational_product_state(model.basis, 2)
prepared = compile(model; backend=:auto)
result = solve_dynamics(
    prepared, rho0, (0.0, 4.0);
    saveat=0.1, observables=(excited=spin.jp * spin.jm,),
    save_states=false,
)
steady = stationary_state(prepared; return_info=true)
@assert steady.info.converged
@assert diagnostics(steady.state).valid
```

Run [`examples/getting_started.jl`](examples/getting_started.jl) for the
complete version, including a time-step refinement and stationary residual.

### First paper

Pick a model with `Models.find(task=:steady_state)` or use the complete
[`examples/catalog.toml`](examples/catalog.toml), change only physically
documented parameters, and retain the example's analytical or published
reference check. Record solver metadata and refine the numerical control that
applies to the result: time step, Krylov dimension, hierarchy depth,
pseudomode cutoff, or trajectory count.

### Scaling up

Before a large run, call `recommend_solver(model; task=...)`, compile once,
stream observables instead of state histories, and reuse prepared plans and
task-owned workspaces. Use `compile_family` or `compile_affine_family` for
related parameter points, `prepare_geometry` or a user-owned
`PreparationCache` for recurring representation setup, `ReductionPlanSet` for
several marginals, and `threaded_matrixfree` only when one Liouvillian action
is large enough to benefit from explicit Julia-task parallelism. Use
`accelerator_preflight` to assess a future optional sparse upload; core does
not currently claim a functional CUDA backend. The
[architecture guide](docs/src/architecture.md) explains ownership and memory
budgets; the [benchmark guide](docs/src/benchmarks.md) provides reproducible
performance checks.

## Documentation

Start with the
[90-second quickstart](https://aenictusgithub.github.io/PermutationalInvariantDynamics.jl/dev/quickstart/)
or use the
[architecture chooser](https://aenictusgithub.github.io/PermutationalInvariantDynamics.jl/dev/choosing_workflow/)
to distinguish an ordinary PI ensemble, several PI factors, replicated local
pseudomodes, one shared pseudomode, PI--HEOM, and PI--HOPS.

Then choose the task:

- [build a custom model](https://aenictusgithub.github.io/PermutationalInvariantDynamics.jl/dev/getting_started/);
- [generate code from supported LaTeX](https://aenictusgithub.github.io/PermutationalInvariantDynamics.jl/dev/model_code_generator/);
- [find a runnable example](https://aenictusgithub.github.io/PermutationalInvariantDynamics.jl/dev/example_gallery/);
- [compute dynamics with bounded output](https://aenictusgithub.github.io/PermutationalInvariantDynamics.jl/dev/streaming_output/);
- [compute stationary states and spectra](https://aenictusgithub.github.io/PermutationalInvariantDynamics.jl/dev/api/solvers/);
- [run trajectories](https://aenictusgithub.github.io/PermutationalInvariantDynamics.jl/dev/weak_pi_trajectories/);
- [use local or shared pseudomodes](https://aenictusgithub.github.io/PermutationalInvariantDynamics.jl/dev/pseudomodes/);
- [use PI--HEOM or PI--HOPS](https://aenictusgithub.github.io/PermutationalInvariantDynamics.jl/dev/heom/);
- [inspect the complete public API](https://aenictusgithub.github.io/PermutationalInvariantDynamics.jl/dev/api_reference/).

Every exported binding also has Julia help, for example `?PIStudy`,
`?stationary_state`, or `?ReductionPlan`. The
[benchmark guide](docs/src/benchmarks.md) contains reproducible performance
and ecosystem-comparison protocols; advanced implementation detail remains in
the hosted architecture and mathematics chapters.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for architectural contracts, focused
test groups, and the complete contributor workflow.

```sh
julia --project=. -e 'using Pkg; Pkg.test()'
julia --startup-file=no --project=benchmark -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
julia --startup-file=no --project=benchmark benchmark/performance_regression.jl
julia --startup-file=no --project=benchmark benchmark/performance_audit.jl
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

## Funding

This project has received funding from the FWO and F.R.S.-FNRS under the
Excellence of Science (EOS) programme (EOS 40007526).

## License

PermutationalInvariantDynamics.jl is distributed under the
[GNU General Public License version 3 only](LICENSE) (`GPL-3.0-only`).
See [copyright and licensing](COPYRIGHT.md), [third-party
notices](THIRD_PARTY_NOTICES.md), and the [source and research provenance
ledger](PROVENANCE.md) for ownership, adapted-code attribution, generated-code
licensing, and release-review obligations.
