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
| [Nonstabilizerness of symmetric qubit states](nonstabilizerness.md) | Second stabilizer Rényi entropy, prepared Krawtchouk transforms, scope, and validation |
| [Genuine multipartite entanglement](genuine_entanglement.md) | PI qubit PPT-mixture plans, validated solver results, and the optional Clarabel backend |
| [Stationary states, spectra, and research solvers](api/solvers.md) | Direct and Krylov solvers, Liouvillian spectra and gaps, Evans tests, weak symmetries, response, memory estimates, and typed high-level commands |
| [Visualization](api/visualization.md) | Schur-block, density-spectrum, Liouvillian-spectrum, Floquet-spectrum, and spin phase-space SVG renderers |
| [Streaming output](streaming_output.md) | Observable-only deterministic evolution and state-free online trajectory statistics |
| [Weak-PI pseudo-ket trajectories](weak_pi_trajectories.md) | Direct-sum Schur-irrep pseudo-kets and sector-changing local Kraus branches |
| [Quantum regression and spectra](correlations.md) | Prepared two-time correlations, delayed intensity correlations, shifted-GMRES spectra, and finite-window FFTs |
| [Higher-order cumulant bridge](cumulant_bridge.md) | Exact distinct-site PI moments, neutral microscopic metadata, closure comparisons, and the optional QuantumCumulants adapter |
| [Research utilities and control](research_utilities.md) | Compressed spectral/population inspection, PI channels, tomography, checkpoints, joint symmetries, and gradients |
| [Diffusive monitoring](diffusive_monitoring.md) | Preallocated homodyne/heterodyne conditional PI dynamics, reproducible ensembles, and confidence-controlled stochastic stopping |
| [Qudit Husimi phase space](qudit_phase_space.md) | Sector-wise generalized coherent-state Q data for arbitrary local dimension |
| [Composite systems](composite_systems.md) | Multiple PI ensembles, finite auxiliary factors, preallocated tensor-product superoperators, and density-valued quantum jumps |
| [Prepared parameter scans](parameter_scans.md) | Continuation, resumable point records, deterministic threaded scans, and tabular exports |
| [Block, multi-shift, and recycled Krylov](krylov_extensions.md) | Thick-restarted block spectra, multiple-right-hand-side solves, shifted families, recycled subspaces, and exponential actions |
| [PI--HEOM non-Markovian dynamics](heom.md) | PI auxiliary-density hierarchies, matrix-free propagation, and stationary states |
| [Numerical convergence reports](convergence.md) | Time-step, Krylov, HEOM-depth, and sector-cutoff refinement evidence |
| [Optional ecosystem integrations](interoperability.md) | Tables and Makie data adapters, process-parallel scans/ensembles, symbolic cumulants, and checkpoint backends |

## Alphabetical index

```@index
Modules = [PermutationalInvariantDynamics]
Pages = ["api/representation.md", "api/dynamics.md", "api/analysis.md", "api/solvers.md", "api/visualization.md", "streaming_output.md", "diffusive_monitoring.md", "weak_pi_trajectories.md", "correlations.md", "cumulant_bridge.md", "research_utilities.md", "composite_systems.md", "parameter_scans.md", "krylov_extensions.md", "heom.md", "convergence.md", "qudit_phase_space.md", "nonstabilizerness.md", "genuine_entanglement.md", "interoperability.md"]
Order = [:type, :function, :constant, :macro]
```
