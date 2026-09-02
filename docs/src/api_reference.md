# Complete public API index

This index covers every exported function, constructor, type, result object,
and compatibility alias in `PermutationalInvariantDynamics`. Entries link to
the source docstrings rendered on the categorized reference pages, so the
website and Julia's interactive `?name` help use the same descriptions.

Functions whose names end in `!` mutate caller-owned output or workspace
storage. Read [API tiers and prepared analysis](api_tiers.md) before using the
advanced workspace and solver interfaces directly.

```@docs
PermutationalInvariantDynamics
```

## Reference sections

| Section | Contents |
|---|---|
| [Representations, states, and models](api/representation.md) | Partitions, GT patterns, Schur blocks, state/operator construction, physical terms, Appendix-D processes, and vectorized superoperators |
| [Dynamics and evolution](api/dynamics.md) | Liouvillian preparation and application, SciML integration, fixed-step evolution, quantum trajectories, mean field, and Floquet dynamics |
| [Observables and quantum information](api/analysis.md) | Moments, correlations, squeezing, QFI/QFIM, spin phase space, reduced states, negativity, entropies, distances, and symmetry-resolved information |
| [Symmetric pure kets and block-resolved entropy](symmetric_kets_and_block_entropy.md) | Physical fully symmetric pure-state dynamics, ket-native local-factor traces, algebraic entropy, and prepared Hilbert charge blocks |
| [Nonstabilizerness of symmetric qubit states](nonstabilizerness.md) | Second stabilizer Rényi entropy, prepared Krawtchouk transforms, scope, and validation |
| [Genuine multipartite entanglement](genuine_entanglement.md) | PI qubit PPT-mixture plans, validated solver results, and the optional Clarabel backend |
| [Stationary states, spectra, and research solvers](api/solvers.md) | Direct and Krylov solvers, Liouvillian spectra and gaps, Evans tests, weak symmetries, response, memory estimates, and typed high-level commands |
| [No-jump-resolvent iterative solvers](no_jump_iterative_solvers.md) | Sectorwise Sylvester resolvents, stationary solvers, complex-shift trace-deflated inexact IRAM, paired left/right conditioning diagnostics, and implicit Euler |
| [Visualization](api/visualization.md) | Schur-block, density-spectrum, Liouvillian-spectrum, Floquet-spectrum, and spin phase-space SVG renderers |
| [Streaming output](streaming_output.md) | Observable-only deterministic evolution and state-free online trajectory statistics |
| [Matrix-RHS trajectory cohorts](batched_trajectories.md) | Fixed-capacity conditional propagation, intensities, grouped gains, and index-stable stochastic streams |
| [Weak-PI pseudo-ket trajectories](weak_pi_trajectories.md) | Direct-sum Schur-irrep pseudo-kets and sector-changing local Kraus branches |
| [Quantum regression and spectra](correlations.md) | Prepared two-time correlations, delayed intensity correlations, shifted-GMRES spectra, and finite-window FFTs |
| [Higher-order cumulant bridge](cumulant_bridge.md) | Exact distinct-site PI moments, neutral microscopic metadata, closure comparisons, and the optional QuantumCumulants adapter |
| [Research utilities and control](research_utilities.md) | Compressed spectral/population inspection, PI channels, tomography, checkpoints, joint symmetries, and gradients |
| [Diffusive monitoring](diffusive_monitoring.md) | Preallocated homodyne/heterodyne conditional PI dynamics, reproducible ensembles, and confidence-controlled stochastic stopping |
| [Qudit Husimi phase space](qudit_phase_space.md) | Sector-wise generalized coherent-state Q data for arbitrary local dimension |
| [Composite systems](composite_systems.md) | Multiple PI ensembles, finite auxiliary factors, preallocated tensor-product superoperators, and density-valued quantum jumps |
| [Global pseudomodes and shared cavities](global_pseudomodes.md) | One shared finite-cutoff mode, collective coupling, factorized dynamics, cutoff observables, and system/mode reductions |
| [Local pseudomodes and PI supersites](pseudomodes.md) | Identical finite-cutoff local modes, system-term lifting, matrix-free dynamics, cutoff checks, and prepared mode tracing |
| [Reusable prepared geometry](prepared_artifacts.md) | Immutable representation bundles and an explicit user-owned preparation cache |
| [Prepared parameter scans](parameter_scans.md) | Continuation, resumable point records, deterministic threaded scans, and tabular exports |
| [Progress and cancellation](progress.md) | Structured events, cooperative cancellation, and resumable scan boundaries |
| [Block, multi-shift, and recycled Krylov](krylov_extensions.md) | Thick-restarted block spectra, multiple-right-hand-side solves, shifted families, recycled subspaces, and exponential actions |
| [Optional accelerators](accelerators.md) | Device capability reports, sparse-upload resource preflight, and guarded extension contracts |
| [PI--HEOM non-Markovian dynamics](heom.md) | PI auxiliary-density hierarchies, matrix-free propagation, and stationary states |
| [PI--HOPS stochastic non-Markovian dynamics](hops.md) | Direct-sum Schur-irrep pure-state hierarchies, colored-noise trajectories, and Monte Carlo density reconstruction |
| [Numerical convergence reports](convergence.md) | Time-step, Krylov, HEOM-depth, and sector-cutoff refinement evidence |
| [Reproducible verified experiments](experiments.md) | Typed calculations, resource explanations, physical/solver/refinement evidence, and portable result archives |
| [Results, tables, plots, and exports](result_outputs.md) | Compact summaries, dependency-free result tables, common text/native exports, and optional Tables/Makie/JLD2/HDF5 integration |
| [Counting statistics and rare events](counting_statistics.md) | Matrix-free tilted generators, finite-time MGFs, SCGFs, cumulants, and discrete rate functions |
| [Bath-correlation fitting](bath_fitting.md) | Spectral quadrature, finite exponential fits, rank diagnostics, and guarded HEOM/HOPS bath construction |
| [Parameter inference](inference.md) | Bounded weighted fitting, implicit stationary sensitivities, Fisher matrices, and identifiability diagnostics |
| [Optional ecosystem integrations](interoperability.md) | QuantumOptics/QuantumToolbox operator bridges, Tables and Makie adapters, process-parallel scans/ensembles, symbolic cumulants, and checkpoint backends |

## Alphabetical index

```@index
Modules = [
    PermutationalInvariantDynamics,
    PermutationalInvariantDynamics.Models,
]
Pages = ["api/representation.md", "api/dynamics.md", "api/analysis.md", "api/solvers.md", "api/visualization.md", "api_tiers.md", "symmetric_kets_and_block_entropy.md", "result_outputs.md", "streaming_output.md", "progress.md", "batched_trajectories.md", "diffusive_monitoring.md", "weak_pi_trajectories.md", "correlations.md", "cumulant_bridge.md", "research_utilities.md", "composite_systems.md", "global_pseudomodes.md", "prepared_artifacts.md", "parameter_scans.md", "no_jump_iterative_solvers.md", "krylov_extensions.md", "accelerators.md", "heom.md", "hops.md", "bath_fitting.md", "counting_statistics.md", "experiments.md", "inference.md", "convergence.md", "qudit_phase_space.md", "nonstabilizerness.md", "genuine_entanglement.md", "interoperability.md"]
Order = [:type, :function, :constant, :macro]
```
