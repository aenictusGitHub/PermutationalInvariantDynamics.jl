# PermutationalInvariantDynamics.jl

`PermutationalInvariantDynamics.jl` simulates finite ensembles of `N`
identical `d`-level systems directly in their permutationally invariant (PI)
operator space. It supports time-local open dynamics, local and collective
dissipation, symmetric many-body processes, stationary states, spectra,
entanglement and information measures, trajectories, Floquet dynamics, and
mean-field predictions. Observable-only output, PI quantum regression, and
factorized deterministic and density-valued stochastic dynamics of several PI
ensembles are also available.
Prepared continuation scans, confidence-controlled ensembles, generalized
qudit Husimi data, advanced matrix-free Krylov families, explicit convergence
reports, finite-exponential PI--HEOM, stochastic PI--HOPS for shared
structured baths, and exact PI supersites for identical finite-cutoff local
pseudomodes extend that workflow for larger research calculations. A separate
factorized embedding covers one finite-cutoff pseudomode shared by the
complete ensemble.

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

To run examples from a fresh repository checkout, instantiate its root
environment once:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

After the package is registered in Julia's General registry, installation is:

```julia
using Pkg
Pkg.add("PermutationalInvariantDynamics")
```

The package supports Julia 1.10 and later.

## Start here

For a first calculation, follow [Getting started: from a model to a
solution](getting_started.md). It explains, line by line, how to:

1. decide whether PI symmetry applies;
2. choose a basis and local-state convention;
3. translate a master equation into physical terms;
4. build and compile a reusable model;
5. compute dynamics or an autonomous stationary state;
6. read result objects, validate states, and check numerical convergence.

The runnable companion is
[`examples/getting_started.jl`](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/blob/main/examples/getting_started.jl).

## Five-minute preview

This compact version evolves eight driven qubits with independent emission
and pumping, records their excitation fraction, and computes the autonomous
stationary state. The getting-started chapter explains every choice.

```julia
using PermutationalInvariantDynamics

basis = PIBasis(8, 2)
spin = spin_matrices()  # local order: (|g>, |e>)
number = spin.jp * spin.jm

rho0 = computational_product_state(basis, 2)
model = PIModel(basis, (
    LocalHamiltonian(spin.jx; rate=0.7),
    LocalJump(spin.jm; rate=0.12),
    LocalJump(spin.jp; rate=0.02),
))
prepared = compile(model; backend=:auto)

solution = solve_dynamics(
    prepared, rho0, (0.0, 4.0);
    saveat=0.1, steps_per_interval=16,
    observables=(excited=number,),
    save_states=false,
)

excited_fraction = real.(solution.observables[:excited]) / basis.N

steady = stationary_state(prepared; return_info=true)
report = diagnostics(steady.state)
```

Compile a model once and reuse `prepared` for dynamics, stationary states,
spectra, and repeated analysis. `steady.info` carries the stationary residual
and convergence metadata. Adaptive or stiff integration is available through
`dynamics_problem` and an installed SciML solver package.

## Where to continue

- [Getting started](getting_started.md) is the task-oriented model-to-solution
  tutorial and troubleshooting guide.
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
  control](research_utilities.md), [composite systems](composite_systems.md),
  [global pseudomodes](global_pseudomodes.md), and [local pseudomode
  supersites](pseudomodes.md) describe memory-conscious research extension
  workflows.
- [Prepared parameter scans](parameter_scans.md), [advanced Krylov
  families](krylov_extensions.md), and [numerical convergence
  reports](convergence.md) cover continuation and explicit numerical evidence.
- [PI--HEOM](heom.md) documents the deterministic common-bath finite-memory
  convention, while [PI--HOPS](hops.md) propagates direct-sum Schur
  pseudo-ket hierarchies and reconstructs the density by Monte Carlo;
  [global pseudomodes](global_pseudomodes.md) cover one explicitly retained
  shared mode; [local pseudomodes](pseudomodes.md) cover identical independent
  finite-mode embeddings; and [qudit Husimi phase
  space](qudit_phase_space.md) describes generalized coherent-state Q data.
- [Optional ecosystem integrations](interoperability.md) records the precise
  Tables, Makie, Distributed, QuantumCumulants, JLD2, and HDF5 boundaries.
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
