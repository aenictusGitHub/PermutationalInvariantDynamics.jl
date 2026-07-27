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
complete ensemble. Matrix-free full counting statistics, finite-exponential
bath fitting, parameter inference, and verified experiment archives support
reproducible model-to-data studies.

The exact PI representation scales polynomially with `N` at fixed `d` and
does not construct the full `d^N` Hilbert space in production algorithms.
General PI density operators may occupy **all** Schur sectors; they are not
restricted to the fully symmetric subspace.

!!! tip "New here?"
    Run the [90-second quickstart](quickstart.md). If you are deciding between
    a PI ensemble, composite factors, local or shared pseudomodes, HEOM, and
    HOPS, use the [architecture chooser](choosing_workflow.md) first.

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

Choose the shortest path that matches the task:

- [90-second quickstart](quickstart.md): explain and solve a tested model
  through `PIStudy`.
- [Model code generator](model_code_generator.md): enter supported LaTeX
  ingredients and download a commented Julia script, manifest, README, and
  Pluto notebook.
- [Searchable example gallery](example_gallery.md): filter every runnable
  example by task, difficulty, runtime, and stochastic/deterministic method.
- [Getting started: from a model to a solution](getting_started.md): translate
  a custom equation term by term and learn the representation conventions.

The detailed getting-started chapter explains, line by line, how to:

1. decide whether PI symmetry applies;
2. choose a basis and local-state convention;
3. translate a master equation into physical terms;
4. build and compile a reusable model;
5. compute dynamics or an autonomous stationary state;
6. read result objects, validate states, and check numerical convergence.

The runnable companion is
[`examples/getting_started.jl`](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/blob/main/examples/getting_started.jl).

To start from formulas instead, the browser-only
[PI model code generator](model_code_generator.md) translates a documented
LaTeX subset into commented, memory-guarded stationary, dynamics, spectrum,
or gap programs. It keeps formulas in the browser and refuses ambiguous
local-versus-collective jump semantics. Its architecture selector also covers
identical local pseudomodes and one shared global pseudomode without exposing
free-form tensor code. Compatible state and dynamics calculations can use
deterministic prepared solvers or streaming quantum-trajectory statistics.
For supported deterministic PI calculations it can instead emit a typed
`PIExperiment`, show the resource/representation plan, verify the result, and
download the Julia program with a normalized JSON manifest, README, and Pluto
notebook.

## Five-minute preview

This compact version evolves eight driven qubits with independent emission
and pumping, records their excitation fraction, and computes the autonomous
stationary state. The getting-started chapter explains every choice.

```julia
using PermutationalInvariantDynamics

model = Models.driven_qubits(8)
basis = model.basis
spin = spin_matrices()  # local order: (|g>, |e>)
number = spin.jp * spin.jm

rho0 = computational_product_state(basis, 2)
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
- [PI model code generator](model_code_generator.md) creates a minimal
  stationary, observable-dynamics, selected-spectrum, or gap script from
  supported LaTeX ingredients. It includes the two finite-cutoff pseudomode
  embeddings, compatible trajectory routes, and optional physical-system
  purity, entropy, one-body-RDM, and collective-QFI analysis.
- [Framework introduction](framework.md) derives the PI Schur-block
  representation, scaling, model terms, and validity conditions.
- [Architecture and efficient workflows](architecture.md) explains sparse and
  matrix-free backends, preallocated workspaces, concurrency, and solver use.
- [Reusable prepared geometry](prepared_artifacts.md) covers immutable
  representation bundles and an explicit user-owned preparation cache.
- [API tiers and prepared analysis](api_tiers.md) separates recommended
  high-level commands from advanced research interfaces and documents the
  compact `Workflow` namespace.
- [Streaming output](streaming_output.md), [weak-PI pseudo-ket
  trajectories](weak_pi_trajectories.md), [quantum regression and
  spectra](correlations.md), [higher-order cumulant closures](cumulant_bridge.md),
  [diffusive monitoring](diffusive_monitoring.md), [research utilities and
  control](research_utilities.md), [composite systems](composite_systems.md),
  [global pseudomodes](global_pseudomodes.md), and [local pseudomode
  supersites](pseudomodes.md) describe memory-conscious research extension
  workflows.
- [Prepared parameter scans](parameter_scans.md), [advanced Krylov
  families](krylov_extensions.md), [optional accelerator
  preflight](accelerators.md), and [numerical convergence
  reports](convergence.md) cover continuation and explicit numerical evidence;
  [reproducible experiments](experiments.md) combine planning, verification,
  refinement, and portable numerical archives.
- [Counting statistics](counting_statistics.md), [bath-correlation
  fitting](bath_fitting.md), and [parameter inference](inference.md) cover
  rare-event, non-Markovian preparation, and model-to-data workflows.
- [Matrix-RHS trajectory cohorts](batched_trajectories.md) expose
  fixed-capacity batched kernels when an external scheduler has already
  grouped paths at the same time and step.
- [PI--HEOM](heom.md) documents the deterministic common-bath finite-memory
  convention, while [PI--HOPS](hops.md) propagates direct-sum Schur
  pseudo-ket hierarchies and reconstructs the density by Monte Carlo;
  [global pseudomodes](global_pseudomodes.md) cover one explicitly retained
  shared mode; [local pseudomodes](pseudomodes.md) cover identical independent
  finite-mode embeddings; and [qudit Husimi phase
  space](qudit_phase_space.md) describes generalized coherent-state Q data.
- [Optional ecosystem integrations](interoperability.md) records the precise
  QuantumOptics, QuantumToolbox, Tables, Makie, Distributed,
  QuantumCumulants, JLD2, and HDF5 boundaries.
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

## Funding

This project has received funding from the FWO and F.R.S.-FNRS under the
Excellence of Science (EOS) programme (EOS 40007526).

## License and development disclosure

The package is distributed under the
[GNU General Public License version 3 only](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/blob/main/LICENSE)
(`GPL-3.0-only`). OpenAI Codex has been used extensively to assist with
implementation, tests, documentation, performance work, and code audits. The
maintainers remain responsible for reviewing, understanding, and validating
every release candidate.
