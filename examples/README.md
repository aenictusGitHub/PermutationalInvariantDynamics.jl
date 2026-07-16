# Examples

Every Julia example has a companion guide describing the physical model, its
permutationally invariant (PI) representation, and the numerical method used.

## Recommended workflow

New research scripts should prepare a model once and use the typed high-level
commands:

```julia
prepared = compile(model; backend=:auto)
states = solve_dynamics(prepared, rho0, (0.0, 10.0); saveat=0.1)
rho_ss = stationary_state(prepared)
slow_modes = liouvillian_spectrum(prepared; target=:largest_real, nev=6)

observable = CollectiveObservablePlan(model.basis, local_matrix)
values = [collective_expectation(rho, observable) for rho in states]
```

Use `diagnostics(prepared)` and `recommend_solver(model; task=...)` before a
large calculation. Prepare a `ReductionPlan(basis, k)` when the same marginal,
purity, or negativity is evaluated repeatedly. The examples that explicitly
compare sparse and matrix-free generators, matrix exponentials, or individual
Krylov algorithms intentionally use the lower-level API for those checks.

## Makie figures

The literature and trajectory examples use CairoMakie for publication-style
figures without making Makie a dependency of the core package. Every
standalone paper-specific example now has an optional numerical figure; the
generic backend and visualization demonstrations retain their dependency-free
output. Prepare the separate examples environment once from the repository
root:

```sh
julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
```

Then run a figure-producing example with, for example,

```sh
julia --project=examples examples/pra94_033838_superradiance.jl
```

Figures display in Makie-capable front ends and are saved as both PDF and PNG
under the ignored `examples/figures/` directory. Set
`PI_EXAMPLE_FIGURE_DIR=/path/to/output` to choose another destination. The
same scripts still perform all numerical assertions when run from the root
environment without CairoMakie; only the figure-generation step is skipped.

| Example | Guide | Topic |
|---|---|---|
| `driven_qubits.jl` | [Driven qubits](driven_qubits.md) | Coherent drive and local decay |
| `floquet_periodic_decay.jl` | [Floquet decay](floquet_periodic_decay.md) | Periodic Liouvillians |
| `gambetta2019_dissipative_discrete_time_crystal.jl` | [Dissipative discrete time crystal](gambetta2019_dissipative_discrete_time_crystal.md) | Floquet period-doubling precursor |
| `huelga1997_ramsey_dephasing.jl` | [Ramsey dephasing](huelga1997_ramsey_dephasing.md) | Independent dephasing |
| `iemini2018_boundary_time_crystal.jl` | [Boundary time crystal](iemini2018_boundary_time_crystal.md) | Gap closing and oscillatory modes |
| `kitagawa1993_one_axis_twisting.jl` | [One-axis twisting](kitagawa1993_one_axis_twisting.md) | Collective nonlinear dynamics |
| `meiser2009_steady_superradiance.jl` | [Steady superradiance](meiser2009_steady_superradiance.md) | Pumped superradiant steady states |
| `meanfield_time_crystal.jl` | [Mean-field time-crystal prediction](meanfield_time_crystal.md) | Finite-`N` product closure, thermodynamic prediction, and exact PI comparison |
| `morrison2008_cooperative_fluorescence.jl` | [Cooperative fluorescence](morrison2008_cooperative_fluorescence.md) | Exact driven-dissipative steady state |
| `nakanishi2023_pt_time_crystal.jl` | [PT-symmetric time crystal](nakanishi2023_pt_time_crystal.md) | Exact balanced-gain/loss spectrum and matrix-free dynamics |
| `paper_models.jl` | [Paper model constructors](paper_models.md) | Reusable literature models |
| `pbody_pair_processes.jl` | [Pair processes](pbody_pair_processes.md) | Appendix-D p-body terms |
| `piccitto2021_interacting_time_crystal.jl` | [Interacting boundary time crystal](piccitto2021_interacting_time_crystal.md) | Nonlinear collective-spin slow modes |
| `pra110_062208_lmg.jl` | [Dissipative LMG model](pra110_062208_lmg.md) | Exact finite PI versus finite-product and thermodynamic mean-field predictions |
| `pra94_033838_superradiance.jl` | [Correlated superradiance](pra94_033838_superradiance.md) | Two-atom analytic benchmark, `N=30` radiated pulse, and peak-state Schur blocks |
| `qubit_population_dynamics.jl` | [Certified qubit population dynamics](qubit_population_dynamics.md) | Six-rate model, reduced population evolution, and stationary populations |
| `quantum_trajectories.jl` | [Analytic Mølmer trajectory benchmark](quantum_trajectories.md) | Independent-emitter state, count, and no-jump laws |
| `schur_block_visualization.jl` | [Schur-block visualization](schur_block_visualization.md) | Steady-state blocks, Young-diagram labels, compressed density spectrum, and superoperator couplings as SVG |
| `spectral_visualization.jl` | [Spectral visualization](spectral_visualization.md) | Liouvillian eigenvalues and Floquet multiplier/exponent SVGs |
| `shammah2018_local_pumping.jl` | [Local pumping](shammah2018_local_pumping.md) | Exact thermal product state |
| `spin_phase_space.jl` | [Sector-resolved spin phase space](spin_phase_space.md) | Multi-sector Husimi-Q and spin-Wigner data with dependency-free SVG rendering |
| `steady_state_methods.jl` | [Steady-state solvers](steady_state_methods.md) | Typed solver choices, shift-invert, matrix-free GMRES, and preconditioning |
| `zhang2018_superradiant_trajectories.jl` | [Zhang--Mølmer superradiant trajectories](zhang2018_superradiant_trajectories.md) | Collective/local radiated pulses: trajectory ensemble versus population master equation |

Run a numerical example from the package root with:

```sh
julia --project=. examples/<name>.jl
```

Use `--project=examples` for the Makie-enabled literature scripts: the
year-named examples in the table, both `pra*.jl` validations,
`quantum_trajectories.jl`, and `meanfield_time_crystal.jl`. Each paired guide
describes its panels and output stem. Running the same script with
`--project=.` keeps all numerical assertions and skips only the optional
Makie block.

Fixed-step, Arnoldi, Floquet, and trajectory results must be convergence-tested
in their step size, Krylov dimension, period discretization, or ensemble size.
Every script keeps its published finite-size or analytical check next to the
library call so that API changes cannot silently alter the physical result.
