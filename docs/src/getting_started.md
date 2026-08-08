# Getting started: from a model to a solution

This page walks through one complete calculation and explains what each
object does. If this is your first use of the package, follow the steps in
order. The same pattern applies to qubits, qudits, closed dynamics, open
dynamics, and most of the advanced workflows:

```text
local basis and matrices
          ↓
physical PI terms → PIModel → compile once → solve → analyse → converge
```

The runnable version is
[`examples/getting_started.jl`](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/blob/main/examples/getting_started.jl).

## Before starting: does PI symmetry apply?

The exact PI representation is appropriate when all three statements hold:

1. Every constituent has the same local dimension `d`.
2. The initial density operator is invariant under relabelling constituents.
3. The generator treats constituent labels identically, so it preserves that
   invariance.

The constituents need not be bosons, and the state need not remain in the
fully symmetric Hilbert-space sector. `PIBasis(N, d)` retains every Schur
sector by default. That default is the safe choice for local incoherent
processes such as independent spontaneous emission.

If individual constituents have different frequencies, rates, or controls,
the model is generally not PI. Several separately PI ensembles can instead
be combined through the [composite-system workflow](composite_systems.md).

## The example model

Consider `N` identical qubits in the local basis

```math
|g\rangle = (1,0)^{\mathsf T},\qquad
|e\rangle = (0,1)^{\mathsf T}.
```

They experience a coherent transverse drive, independent emission, and
independent pumping:

```math
\frac{d\rho}{dt}
=-i\left[\frac{\Omega}{2}\sum_i\sigma_i^x,\rho\right]
+\gamma_\downarrow\sum_i\mathcal D[\sigma_i^-]\rho
+\gamma_\uparrow\sum_i\mathcal D[\sigma_i^+]\rho.
```

Here $\mathcal D[L]\rho=L\rho L^\dagger-\{L^\dagger L,\rho\}/2$.
The model is PI because every site receives the same Hamiltonian and the same
local rates.

## Step 1: create the PI basis

```julia
using PermutationalInvariantDynamics

N = 8
d = 2
basis = PIBasis(N, d)
```

`PIBasis` defines the Schur sectors, Gel'fand--Tsetlin patterns, and coordinate
offsets used by every state, operator, and model built from it. Construct it
once and reuse the exact same `basis` object.

The number of retained PI operator coordinates is

```julia
npi = pi_dimension(basis)  # 165 for N=8 qubits
```

By comparison, a general full density operator has `d^(2N) = 65_536`
coordinates here. Production PI algorithms never construct that full object.

## Step 2: define the local matrices

```julia
spin = spin_matrices()       # local order: (|g>, |e>)
sx = 2 * spin.jx
sm = spin.jm                 # |g><e|
sp = spin.jp                 # |e><g|
number = sp * sm             # |e><e|
```

These are only `d`-by-`d` one-particle matrices. The package lifts them into
the PI representation; do not construct an `N`-particle Kronecker product.

Matrix columns are input states and rows are output states. Therefore `sm[1,
2] = 1` maps the second local basis vector, `|e>`, to the first, `|g>`.
Julia uses one-based indices even when a paper labels the same states `0` and
`1`. The helper also fixes the package spin convention:
`spin.jz = diag(-1/2, 1/2)` in this order.

## Step 3: translate the master equation into terms

```julia
Omega = 0.7
gamma_down = 0.12
gamma_up = 0.02

terms = (
    LocalHamiltonian(spin.jx; rate=Omega),
    LocalJump(sm; rate=gamma_down),
    LocalJump(sp; rate=gamma_up),
)
```

Choose constructors from the physical meaning of the process:

| Physical contribution | Constructor |
|:--|:--|
| Identical one-particle Hamiltonian, `sum_i h_i` | `LocalHamiltonian(h)` |
| Independent identical jumps, `sum_i D[l_i]` | `LocalJump(l)` |
| One coherent summed jump, `D[sum_i L_i]` | `CollectiveJump(L)` |
| A Hamiltonian already stored as a `PIOperator` | `DirectPIHamiltonian(H)` |
| A permutation-symmetric `p`-particle process | `PBodyHamiltonian`, `LocalPBodyJump`, or `CollectivePBodyJump` |

The difference between local and collective jumps is essential. `LocalJump`
sums probabilities from unresolved independent channels and can move
population between Schur sectors. `CollectiveJump` sums amplitudes first and
then forms one dissipator.

The `rate` keyword multiplies the term exactly as supplied. The package does
not insert a factor of `N`, a Kac normalization, or a unit conversion.

## Step 4: build the declarative model

```julia
model = PIModel(basis, terms)

@assert isautonomous(model)
model_report = diagnostics(model)
```

`PIModel` records the physics but owns no mutable solver scratch. An
autonomous model has no explicit time dependence; it can be used for dynamics,
stationary states, and ordinary Liouvillian spectra.

`diagnostics(model)` reports the PI dimension, term information, autonomy,
and memory estimates. Model constructors also reject incompatible matrix
sizes, non-Hermitian Hamiltonians, and unsupported restricted-sector
transitions before a solve begins. For a large model, `recommend_solver` is
the assembly-free preflight; detailed model diagnostics may perform generator
validation work.

## Step 5: construct and validate the initial state

Start from the fully excited product state:

```julia
rho0 = computational_product_state(basis, 2)
validate_state(rho0)
```

`computational_product_state(basis, 2)` uses Julia's one-based local level and
constructs $(|e\rangle\langle e|)^{\otimes N}$ directly in PI coordinates.
It does not form a
`2^N` state vector. `validate_state` returns the state when its trace,
Hermiticity, and positivity checks pass; it throws instead of repairing an
invalid state.

Common alternatives are:

```julia
rho_same = iid_pure_state(basis, ComplexF64[0, 1])

mixed_local = ComplexF64[0.7 0; 0 0.3]
rho_mixed = iid_state(basis, mixed_local)

rho_two_excitations = dicke_state(basis, 2)
rho_w = w_state(basis)
rho_dicke_sector = dicke_state(basis, N / 2, 0)
rho_ghz = ghz_state(basis)
rho_cat = cat_state(basis; phase=pi / 5)
rho_coherent = spin_coherent_state(basis, pi / 3, pi / 4)
rho_symmetric_white = symmetric_maximally_mixed_state(basis)

qutrit_basis = PIBasis(4, 3)
rho_qutrit_counts = symmetric_occupation_state(qutrit_basis, (1, 2, 1))
```

An identical mixed product state generally occupies several Schur sectors;
that is still a PI state and is handled by the complete basis.
Occupation tuples use one-based local-level order and must sum to `N`.
`sector_maximally_mixed_state(basis, nu)` is the corresponding shortcut when
a normalized white state in one retained Schur sector is needed.

## Step 6: compile once

```julia
prepared = compile(model; backend=:auto)
prepared_report = diagnostics(prepared)
```

Compilation lowers the local matrices into reusable Schur-block geometry and
Liouvillian kernels. It does not evolve a state. Reuse `prepared` for every
initial state, time interval, stationary solve, or spectrum with the same
model.

`backend=:auto` is the recommended first choice. It selects a conservative
sparse or matrix-free representation from the actual compiled problem. For a
large calculation, inspect the recommendation before compiling:

```julia
advice = recommend_solver(model; task=:dynamics)
```

Inspect `advice.resources` to distinguish setup, retained plan, solver
workspace, and output storage. `advice.budget_status` is `:fits`, `:exceeds`,
`:unknown`, or `:disabled`; `safe_to_run` is intentionally `missing` when an
estimate cannot certify a fit. The high-level stationary, spectral, and
dynamics commands use a 512 MiB default budget, select a matrix-free route when
an automatic dense/direct choice is too large, and reject an explicitly
requested over-budget materialization before allocating it. Use
`memory_budget=Inf` only to opt out deliberately.

The recommendation is a transparent resource preflight, not a convergence
guarantee.
See [Architecture and efficient workflows](architecture.md) for explicit
backend and workspace control.

## Step 7A: compute time evolution and retain states

```julia
times = collect(0.0:0.1:4.0)

solution = solve_dynamics(
    prepared,
    rho0,
    (first(times), last(times));
    saveat=times,
    steps_per_interval=16,
    observables=(excited=number,),
    save_states=true,
)
```

The important arguments are:

| Argument | Meaning |
|:--|:--|
| `tspan=(t0,t1)` | Physical integration interval |
| `saveat` | Times returned to the user; explicit times must include both endpoints |
| `steps_per_interval` | Fixed RK4 steps taken between two consecutive saved times |
| `observables` | Named local matrices or `PIOperator`s sampled during propagation |
| `save_states` | Whether every sampled `PIState` is retained |

For an autonomous model, an adaptive matrix-free exponential action is also
available:

```julia
solution_expv = solve_dynamics(
    prepared,
    rho0,
    (first(times), last(times));
    saveat=times,
    algorithm=ExpvAlgorithm(
        krylovdim=30,
        atol=1e-11,
        rtol=1e-9,
    ),
    observables=(excited=number,),
    save_states=true,
)
```

This reuses one Arnoldi workspace and task-owned Liouvillian workspace between
saved times. It is often useful when output times are sparse relative to the
RK4 step required for accuracy. It is not a time-ordered exponential:
`ExpvAlgorithm` therefore rejects driven generators and nonempty
`parameters`. Use RK4 or `dynamics_problem` for those cases.

A local matrix passed as an observable denotes its collective sum. Thus the
stored excitation series is `sum_i <e_i|rho(t)|e_i>`:

```julia
excited_fraction = real.(solution.observables[:excited]) ./ N
rho_final = solution[end]
rho_at_two = state_at(solution, 2.0)
```

For a saved-grid result, physical-time lookup requires a stored time.
Iterating over `solution` iterates over the retained states when
`save_states=true`. Use `state(solution, i)` for a saved index and
`state_at(solution, t)` for a physical time; this distinction remains explicit
when `t` is an integer.

## Step 7B: retain only observables

If the state history is not needed, use the same prepared model with
`save_states=false`:

```julia
series = solve_dynamics(
    prepared,
    rho0,
    (first(times), last(times));
    saveat=times,
    steps_per_interval=16,
    observables=(excited=number,),
    save_states=false,
)

excited_fraction = real.(series.observables[:excited]) ./ N
@assert series.states === nothing
```

This keeps one evolving PI state plus the requested scalar arrays. It is the
preferred output policy for long evolutions or large parameter scans when no
later state-level analysis is planned.

The two result forms are:

| Call | Result contents |
|:--|:--|
| No `observables` | `DynamicsResult` with `times` and saved states |
| `observables=...`, `save_states=true` | `DynamicsStreamResult` with times, states, and observable arrays |
| `observables=...`, `save_states=false` | `DynamicsStreamResult` with times and observable arrays; `states === nothing` |

## Step 7C: compute the stationary state

Because this model is autonomous, its stationary state can be solved from the
same prepared object:

```julia
steady = stationary_state(prepared; return_info=true)
rho_ss = steady.state

@assert steady.info.converged
println("stationary residual = ", steady.info.residual)
println("stationary trace error = ", steady.info.trace_error)
validate_state(rho_ss)
```

Without `return_info=true`, `stationary_state(prepared)` returns the `PIState`
directly. Retain the result information when reporting numerical accuracy.
Solver convergence and state validity are separate checks, and neither proves
that the stationary state is unique. Study near-zero modes or use an explicit
uniqueness certificate when uniqueness is part of the conclusion.

Stationary-state and ordinary spectrum commands reject a genuinely driven
model. For a periodic generator, use Floquet dynamics. `freeze(model;
time=t, parameters=p)` answers only the instantaneous stationary question at
that chosen time; it is not the steady state of the driven evolution.

## Step 8: analyse the result

The same state can be inspected at several levels:

```julia
report = diagnostics(rho_final)
excited_final = collective_expectation(rho_final, number) / N
rho_one = one_body_rdm(rho_final)
purity_one = reduced_purity(rho_final, 1)
```

For a valid density operator, `report.valid` is true and the trace,
Hermiticity, and positivity errors lie within their reported tolerances.
`collective_expectation(rho, number)` returns the same total one-body
excitation measured during propagation. `one_body_rdm` instead returns the
normalized single-particle marginal.

Other analysis starts from the same `PIState`:

```julia
entropy = von_neumann_entropy(rho_ss)
fisher = qfi(rho_ss, spin.jx)
blocks = schur_block_structure(rho_ss; metric=:population)
```

Prepare `CollectiveObservablePlan` or `ReductionPlan` when repeating the same
analysis for many states. If one PI particle is itself a tensor product, such
as `spin tensor local_mode`, prepare `LocalFactorTracePlan` to remove the same
internal factor from every supersite before applying the ordinary
particle-bipartition tools.

## Step 9: check numerical convergence

PI compression is exact, but a fixed-step integrator still has a time-step
error. Repeat the observable calculation with more RK4 steps between saved
times:

```julia
coarse = solve_dynamics(
    prepared, rho0, (first(times), last(times));
    saveat=times, steps_per_interval=8,
    observables=(excited=number,), save_states=false)

fine = solve_dynamics(
    prepared, rho0, (first(times), last(times));
    saveat=times, steps_per_interval=16,
    observables=(excited=number,), save_states=false)

step_error = maximum(abs.(
    coarse.observables[:excited] .- fine.observables[:excited])) / N
```

Increase the resolution again until `step_error` is small compared with the
accuracy required by the physical conclusion. A small stationary residual,
a converged time step, a converged Krylov space, and a sufficiently large
trajectory ensemble are different checks; one never substitutes for another.
The [numerical-convergence guide](convergence.md) provides reusable refinement
reports.

## Complete runnable script

Putting the central steps together gives:

```julia
using PermutationalInvariantDynamics

N = 8
basis = PIBasis(N, 2)
spin = spin_matrices()
sm = spin.jm
sp = spin.jp
number = sp * sm

model = PIModel(basis, (
    LocalHamiltonian(spin.jx; rate=0.7),
    LocalJump(sm; rate=0.12),
    LocalJump(sp; rate=0.02),
))
rho0 = computational_product_state(basis, 2)
prepared = compile(model; backend=:auto)

times = collect(0.0:0.1:4.0)
solution = solve_dynamics(
    prepared, rho0, (first(times), last(times));
    saveat=times, steps_per_interval=16,
    observables=(excited=number,), save_states=true)

excited_fraction = real.(solution.observables[:excited]) ./ N
steady = stationary_state(prepared; return_info=true)

@assert diagnostics(solution[end]).valid
@assert steady.info.converged
@assert diagnostics(steady.state).valid
println("PI coordinates: ", pi_dimension(basis))
println("final excitation fraction: ", last(excited_fraction))
println("stationary residual: ", steady.info.residual)
```

## Common modifications

### Time-dependent scalar rates

A scalar schedule receives the current time and the user-supplied parameter
object:

```julia
drive = (t, p) -> p.Omega * cos(p.omega * t)
driven_model = PIModel(basis, (
    LocalHamiltonian(spin.jx; rate=drive),
    LocalJump(sm; rate=0.12),
))
driven = compile(driven_model; backend=:matrixfree)

driven_solution = solve_dynamics(
    driven, rho0, (0.0, 4.0);
    saveat=0.1, parameters=(Omega=0.7, omega=2.0))
```

Use `InPlaceTimeOperator` when the matrix itself, rather than only its scalar
coefficient, changes with time.

### Collective decay

Replace

```julia
LocalJump(sm; rate=gamma_down)
```

with

```julia
CollectiveJump(sm; rate=gamma_collective)
```

only when the physical collapse operator is
`J_- = sum_i sigma_i^-`. The two models have different interference and
Schur-sector structure.

### Qudits

For `d > 2`, use `PIBasis(N, d)`, `d`-component local states, and `d`-by-`d`
local matrices. The construction and solver sequence is unchanged, but PI
dimension grows more rapidly with `N`.

### Adaptive or stiff integration

Install a compatible SciML solver and use:

```julia
problem = dynamics_problem(prepared, rho0, (0.0, 4.0))
```

Then solve `problem` with the selected external algorithm. The package does
not add a heavy ODE solver dependency to its core.

### Other solution types

| Goal | Continue with |
|:--|:--|
| Quantum-jump records | `quantum_trajectories` and [streaming output](streaming_output.md) |
| Homodyne or heterodyne records | [Diffusive monitoring](diffusive_monitoring.md) |
| Periodically driven stationary response | `floquet_steady_state` and [spectral visualization](spectral_visualization.md) |
| Many related parameter points | [Prepared parameter scans](parameter_scans.md) |
| Very large product-state prediction | [Mean-field predictions](meanfield.md) |
| Finite-memory bosonic environment | [PI--HEOM](heom.md) |
| One cavity or pseudomode shared by the ensemble | [Global pseudomodes and shared cavities](global_pseudomodes.md) |
| Identical independent local pseudomodes | [Local pseudomodes and PI supersites](pseudomodes.md) |
| Several PI ensembles or a finite ancilla | [Composite systems](composite_systems.md) |

## Common errors and their fixes

| Symptom | Likely cause | Fix |
|:--|:--|:--|
| Local operator has the wrong size | Matrix dimension differs from `basis.d` | Supply a `d`-by-`d` matrix |
| A lowering operator raises the state | Row/column or local-level order is reversed | Write its action on each basis column explicitly |
| Restricted-sector compilation fails | A local process leaves the selected sectors | Start with the complete `PIBasis(N,d)` |
| Basis-ownership error | State, operator, plan, and model were built from different basis objects | Reuse the exact same `basis` instance |
| Hamiltonian construction fails | The supplied matrix is not Hermitian | Correct the matrix or use an explicitly documented unchecked constructor only for diagnostics |
| Stationary or spectral solve rejects the model | The generator is time dependent | Evolve it, use Floquet tools, or explicitly `freeze` one instant |
| A trajectory rejects a rate accepted by deterministic evolution | Jump probabilities require nonnegative rates | Use a nonnegative unraveling or deterministic evolution |
| `series[i]` throws after observable-only evolution | `save_states=false` deliberately discarded states | Read `series.observables`, or rerun with `save_states=true` |
| Results change when `steps_per_interval` increases | The fixed-step integration is not converged | Refine until the quantity of interest stabilizes |
| A SciML algorithm name is undefined | The corresponding solver package is not installed/imported | Add and import a compatible OrdinaryDiffEq package |

## Where to go next

- Read [Framework and physical conventions](framework.md) to understand the
  Schur--Weyl representation and equation-(7) normalization.
- Read [API tiers and prepared analysis](api_tiers.md) before choosing lower-
  level workspaces or solver internals.
- Use [Research examples](research_examples.md) to find a literature model or
  workflow close to your calculation.
- Consult the [complete public API](api_reference.md) for signatures and
  interactive-help descriptions.
