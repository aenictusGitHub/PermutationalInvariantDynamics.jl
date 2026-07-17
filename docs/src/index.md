# PermutationalInvariantDynamics.jl

`PermutationalInvariantDynamics.jl` simulates finite ensembles of `N`
identical `d`-level systems directly in their permutationally invariant (PI)
operator space. It supports time-local open dynamics, local and collective
dissipation, symmetric many-body processes, stationary states, spectra,
entanglement and information measures, trajectories, Floquet dynamics, and
mean-field predictions. Observable-only output, PI quantum regression, and
factorized deterministic dynamics of several PI ensembles are also available.

The exact PI representation scales polynomially with `N` at fixed `d` and
does not construct the full `d^N` Hilbert space in production algorithms.
General PI density operators may occupy **all** Schur sectors; they are not
restricted to the fully symmetric subspace.

## Installation

Until registration, install the package directly from its
[GitHub repository](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl):

```julia
using Pkg
Pkg.add(url="https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl")
```

For a local checkout under development:

```julia
using Pkg
Pkg.develop(path="/path/to/PermutationalInvariantDynamics.jl")
```

After the package is registered in Julia's General registry, installation is:

```julia
using Pkg
Pkg.add("PermutationalInvariantDynamics")
```

The package supports Julia 1.10 and later.

## Five-minute example

This example evolves 20 qubits subject to independent spontaneous emission.
The matrix `sm` maps the local state `|1>` to `|0>`; local labels become Julia
indices `1` and `2`.

```julia
using PermutationalInvariantDynamics

basis = PIBasis(20, 2)
sm = ComplexF64[0 1; 0 0]
sz = ComplexF64[1 0; 0 -1]

rho0 = iid_pure_state(basis, ComplexF64[0, 1])
model = PIModel(basis, [LocalJump(sm; rate=0.1)])
prepared = compile(model; backend=:auto)

solution = solve_dynamics(
    prepared, rho0, (0.0, 10.0);
    saveat=0.1, steps_per_interval=16,
    observables=(magnetization=sz / 2,),
    save_states=false,
)

magnetization = real.(solution.observables[:magnetization]) / basis.N

rho_ss = stationary_state(prepared)
report = diagnostics(rho_ss)
```

Compile a model once and reuse `prepared` for dynamics, stationary states,
spectra, and repeated analysis. Adaptive or stiff integration is available
through `dynamics_problem` and an installed SciML solver package.

## Where to continue

- [Framework introduction](framework.md) derives the PI Schur-block
  representation, scaling, model terms, and validity conditions.
- [Architecture and efficient workflows](architecture.md) explains sparse and
  matrix-free backends, preallocated workspaces, concurrency, and solver use.
- [API tiers and prepared analysis](api_tiers.md) separates recommended
  high-level commands from advanced research interfaces.
- [Streaming output](streaming_output.md), [weak-PI pseudo-ket
  trajectories](weak_pi_trajectories.md), [quantum regression and
  spectra](correlations.md), [higher-order cumulant closures](cumulant_bridge.md),
  [diffusive monitoring](diffusive_monitoring.md), [research utilities and
  control](research_utilities.md), and [composite systems](composite_systems.md) describe memory-conscious
  research extension workflows.
- [Complete public API index](api_reference.md) categorizes every exported
  function and type and links its full description.
- [Published validation](published_validation.md) and
  [Research examples](research_examples.md) connect runnable scripts to the
  literature.

## Mathematical source

The framework follows Thierry Bastin and John Martin,
*J. Phys. A: Mathematical and Theoretical* **58**, 275301 (2025),
[doi:10.1088/1751-8121/addfc1](https://doi.org/10.1088/1751-8121/addfc1).

Negative time-dependent rates are accepted for deterministic time-local
models. Such a generator need not define a completely positive evolution;
quantum-jump trajectories require nonnegative stochastic rates.

## License and development disclosure

The package is distributed under the
[GNU General Public License version 3 only](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/blob/main/LICENSE)
(`GPL-3.0-only`). OpenAI Codex has been used extensively to assist with
implementation, tests, documentation, performance work, and code audits. The
maintainers remain responsible for reviewing, understanding, and validating
every release candidate.
