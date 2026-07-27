# Choose the right workflow

Use this page before constructing a basis. The decisive question is the
physical permutation action, not merely whether several coefficients happen
to be equal.

## One-screen architecture chooser

| Physical model | Exact package representation | Start with | Runnable example |
|:--|:--|:--|:--|
| All systems are interchangeable and the environment is time-local | One complete PI operator space | `PIModel` or `Models.*` | [`getting_started.jl`](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/blob/main/examples/getting_started.jl) |
| Several internally PI ensembles interact | Product of compressed PI factors | `CompositePIBasis` | [`composite_ensembles.jl`](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/blob/main/examples/composite_ensembles.jl) |
| Every system has its own identical finite-cutoff pseudomode | Permute complete system+mode supersites | `pseudomode_supersite` and `pseudomode_model` | [`all_to_all_xx_spin_local_pseudomodes.jl`](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/blob/main/examples/all_to_all_xx_spin_local_pseudomodes.jl) |
| One finite-cutoff mode is shared by the ensemble | PI system factor × finite-mode operator factor | `global_pseudomode_model` | [`global_pseudomode_cavity.jl`](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/blob/main/examples/global_pseudomode_cavity.jl) |
| A shared Gaussian bath correlation is a finite exponential sum and a density hierarchy is desired | One PI density operator per HEOM auxiliary | `HEOMPlan` | [`pi_heom.jl`](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/blob/main/examples/pi_heom.jl) |
| A shared Gaussian bath and stochastic pure-state hierarchy are desired | Direct-sum Schur pseudo-kets per HOPS auxiliary | `HOPSPlan` | [`pi_hops.jl`](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/blob/main/examples/pi_hops.jl) |
| Couplings distinguish particle labels, sites, distances, or graph edges | Generally not PI | Use a spatial method, or reformulate only if groups of genuinely interchangeable systems exist | — |

## Decision sequence

1. **Can every physical particle label be exchanged without changing the
   Hamiltonian, channels, initial state, or requested statistics?**
   If yes, use one PI ensemble. If only groups are interchangeable, use one PI
   factor per group. A nearest-neighbour ring is translation invariant but is
   not invariant under every permutation.
2. **Is an auxiliary mode replicated or shared?**
   Replicated identical modes belong inside each permuted supersite. One shared
   mode remains a separate factor. These embeddings are physically different
   even when their local coupling constants are equal.
3. **Is the retained environment finite-dimensional?**
   A chosen Fock cutoff uses a pseudomode embedding. A finite exponential bath
   decomposition without an explicit retained oscillator uses HEOM or HOPS.
4. **Is the requested result deterministic or statistical?**
   Density evolution, stationary states, spectra, and HEOM are deterministic.
   Quantum trajectories and HOPS require path-count and uncertainty
   convergence in addition to time-step and truncation checks.
5. **What output is actually needed?**
   Stream a small observable set whenever possible. Retain density histories,
   eigenvectors, hierarchy auxiliaries, or individual paths only when the
   scientific analysis requires them.

## Let the package explain the route

For a built model, the guided study interface combines validation, resource
planning, and execution:

```julia
using PermutationalInvariantDynamics

study = PIStudy(
    Models.driven_qubits(8);
    task=:steady_state,
)

explain(study)
result = solve(study)
```

The explanation is a plan, not a convergence certificate. The returned result
retains the selected algorithm and numerical diagnostics.

If the Hamiltonian and channels fit the documented subset, the
[PI model code generator](model_code_generator.md) performs the same
architecture choice interactively. It supports ordinary PI ensembles,
identical local pseudomodes, and one shared finite pseudomode; arbitrary
composite tensor formulas, HEOM, and HOPS still require the typed Julia API.

## Common wrong turns

- A fully symmetric initial state does not justify a symmetric-sector basis
  when local noise transfers population to other Schur sectors.
- “Local” and “collective” dissipation are different physical channels:
  $\sum_i\mathcal D[L_i]$ is not $\mathcal D[\sum_i L_i]$.
- Equal nearest-neighbour couplings do not imply full permutation symmetry.
- A pseudomode cutoff, HEOM depth, HOPS depth, trajectory count, and Krylov
  dimension are independent convergence controls.
- Automatic solver selection respects the declared memory budget, but it
  cannot certify that a physical truncation is converged.
