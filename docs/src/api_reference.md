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
| [Stationary states, spectra, and research solvers](api/solvers.md) | Direct and Krylov solvers, Liouvillian spectra and gaps, Evans tests, weak symmetries, response, memory estimates, and typed high-level commands |
| [Visualization](api/visualization.md) | Schur-block, density-spectrum, Liouvillian-spectrum, Floquet-spectrum, and spin phase-space SVG renderers |

## Alphabetical index

```@index
Modules = [PermutationalInvariantDynamics]
Pages = ["api/representation.md", "api/dynamics.md", "api/analysis.md", "api/solvers.md", "api/visualization.md"]
Order = [:type, :function, :constant, :macro]
```
