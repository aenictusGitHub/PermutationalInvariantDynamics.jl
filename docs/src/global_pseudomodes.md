# Global pseudomodes and shared cavities

A global pseudomode is one finite-cutoff bosonic mode coupled to a whole
permutationally invariant ensemble. Permutations act on the identical systems
but not on that distinguished mode. Typical examples are a shared cavity, a
reaction coordinate coupled collectively to the ensemble, or one damped
pseudomode realization of a common structured reservoir.

This topology is different from one independent pseudomode per particle.
Choose the topology from the physics before comparing coordinate counts.

## Representation and master equation

For a PI system basis with $n_{\mathrm{PI}}$ operator coordinates and a mode
with levels $0,\ldots,n_{\max}$, the package constructs

```julia
CompositePIBasis(system_basis, FiniteOperatorBasis(nmax + 1))
```

and stores

```math
n_{\mathrm{PI}}(n_{\max}+1)^2
```

coordinates. The system coordinate is the first and fastest factor. Factorized
maps are applied one tensor mode at a time; neither a $d^N$ system state nor
the global Kronecker superoperator is formed.

For a one-particle matrix $L$, `PseudomodeCoupling(L; strength=g)` uses the
collective operator

```math
J_L=\sum_{i=1}^N L_i
```

and the rotating-wave interaction

```math
H_{\mathrm{int}}=gJ_La^\dagger+g^*J_L^\dagger a.
```

An optional `counterrotating_strength=h` adds
$hJ_La+h^*J_L^\dagger a^\dagger$. No Kac or other $N$-dependent scaling
is inserted. Supply the scaling required by the model explicitly.

The mode part is

```math
-i[\omega a^\dagger a,\rho]
+\kappa(\bar n+1)\mathcal D[a]\rho
+\kappa\bar n\mathcal D[a^\dagger]\rho.
```

The package convention is
$\mathcal D[c]\rho=c\rho c^\dagger-\{c^\dagger c,\rho\}/2$. Consequently,
`damping=κ` makes a free mode amplitude decay at rate $\kappa/2$.

## Build a shared-mode model

The most explicit workflow starts from the bare PI system:

```julia
using LinearAlgebra
using PermutationalInvariantDynamics

N = 4
spin = spin_matrices()
system_basis = PIBasis(N, 2; sectors=[(N, 0)])
system_model = PIModel(system_basis, ())

mode = BosonicPseudomode(
    N + 1; label=:cavity, frequency=0.0, damping=0.2)
coupling = PseudomodeCoupling(
    spin.jm; mode=:cavity, strength=0.4)

model = global_pseudomode_model(
    system_model, mode; couplings=coupling)
rho0 = pseudomode_product_state(
    model, computational_product_state(system_basis, 2))
```

The `PIModel` overload preserves the exact system-basis object and is the
recommended route when the system already has local, collective, or
many-body terms. Two conveniences are equivalent:

```julia
same = pseudomode_model(system_model, mode; couplings=coupling)
same_alias = shared_pseudomode_model(
    system_model, mode; couplings=coupling)
```

The constructor from a one-particle Hamiltonian is

```julia
model = global_pseudomode_model(
    N, zeros(ComplexF64, 2, 2), mode;
    couplings=coupling, system_terms=())
```

For code that chooses the embedding from an option, use
`pseudomode_model(N, H, mode; topology=:global, ...)`. The same signature
defaults to `topology=:local`, so set the keyword explicitly when the mode is
shared.

`GlobalPseudomodeModel` exposes prepared read-only pieces:

| Field | Meaning |
|:--|:--|
| `background` | PI system generator, mode frequency, and coherent couplings; excludes damping jumps |
| `damping_channels` | Explicit loss and thermal-gain `CompositeJumpChannel`s |
| `generator` | Complete unconditional deterministic generator |
| `mode_operators` | Identity, annihilation, creation, number, parity, and top-level projector matrices |
| `coupling_operators` | Prepared collective system operators and quadratures |
| `system_reduction_plan` | Packed exact diagonal contraction retaining the PI system |
| `mode_reduction_plan` | Packed exact diagonal contraction retaining the finite mode |
| `resource_estimates` | Coordinate, coupling-map/operator, retained, setup, nested-workspace, and action-transient memory estimates |

The separation between `background` and `damping_channels` avoids counting
the same dissipator twice when constructing a `CompositeTrajectoryPlan`:

```julia
trajectory_plan = CompositeTrajectoryPlan(
    model.background, model.damping_channels...)
```

The `PIModel` overload accepts `T` to select a wider working type for an empty
system model or for mode and coupling data accompanying a system model already
lowered at that type. It does not retroactively widen a nonempty PI system
generator: rebuild those system terms at the target precision first. It rejects
a `T` that would narrow any input; omitting `T` promotes inputs without
narrowing. Product-state inputs may be narrower and are converted to the
prepared type, but wider inputs require rebuilding the model. Workspaces,
reductions, and stationary solvers preserve the prepared `BigFloat` precision
and rounding mode even when called from a narrower ambient precision context.

Preparation, matrix-free wrapping, product-state construction, and stationary
solving enforce `memory_budget` before their guarded allocations. The estimates
include retained model data, coupling maps and operators, task-owned workspace,
per-application transients, relevant state intermediates, and the matrix-free
wrapper's copied trace vector. `memory_budget=Inf` is the explicit opt-out;
the package does not silently narrow precision or switch to a dense method.

## Evolution, observables, and reductions

For a compact state history, evolve the factorized generator directly:

```julia
times = collect(range(0.0, 4.0; length=41))
states = time_evolution(
    model.generator, rho0, times; steps_per_interval=8)
```

Build observables as products of system and mode factors:

```julia
system_identity = identity_operator(system_basis)
mode_identity = model.mode_operators.identity
Jnumber = collective_operator(system_basis, spin.jp * spin.jm)

atoms = composite_tensor_operator(
    model.basis, Jnumber, mode_identity)
photons = composite_tensor_operator(
    model.basis, system_identity,
    model.mode_operators.number_operator)
top = composite_tensor_operator(
    model.basis, system_identity,
    model.mode_operators.top_projector)

atom_values = [real(expectation(rho, atoms)) for rho in states]
photon_values = [real(expectation(rho, photons)) for rho in states]
top_values = [real(expectation(rho, top)) for rho in states]
```

The mode-traced system remains a compressed `PIState`; the system-traced mode
is an ordinary dense matrix of size `(nmax+1) x (nmax+1)`:

```julia
rho_system = trace_pseudomodes(last(states), model)
rho_mode = global_pseudomode_state(last(states), model)

@assert rho_mode ≈ composite_reduced_state(last(states), 2)
```

These functions reuse the two model-owned `CompositeReductionPlan`s and
contract tensor-factor coordinates directly. They do not rebuild Schur-sector
groups, reconstruct the ensemble Hilbert space, normalize the result, or
repair an invalid state.

## Matrix-free application and solvers

`model.generator` is already a read-only `CompositeSuperoperator`. Explicit
task-owned scratch is useful in repeated or parallel applications:

```julia
work = global_pseudomode_workspace(model)
y = similar(rho0.data)
apply!(y, model, rho0.data, work)
```

Use one workspace per concurrent task. The synchronized convenience wrapper

```julia
L = global_pseudomode_matrixfree(model; workspace=work)
```

provides forward and adjoint actions for Krylov and response routines.
`liouvillian(model; representation=:matrixfree)` returns the same kind of
wrapper. Sparse materialization of the complete composite Kronecker
superoperator is intentionally unsupported.

For an autonomous model, prefer the typed high-level result:

```julia
steady = stationary_state(model; return_info=true)
rho_ss = steady.state
```

The advanced `steady_state(model; method=:krylov, ...)` route returns raw
coordinate-level solver output. Both use the matrix-free operator.
`AutoAlgorithm()` resolves to restarted GMRES; supported explicit choices are
`GMRESAlgorithm` and `RecycledGMRESAlgorithm`. Direct, dense, shift-invert,
and the PI-only Schur preconditioner are rejected for this composite
coordinate. A numerical residual establishes solver convergence, not
uniqueness or mode-cutoff convergence.

Selected decay modes use the same factorized route:

```julia
slow_modes = liouvillian_spectrum(
    model; algorithm=:arnoldi, nev=4)
```

Automatic spectral selection chooses a matrix-free Arnoldi family. Complete
dense diagonalization of the global composite map is intentionally
unsupported.

## Shared mode, replicated local modes, or HEOM?

| Physical model | Package representation | Main coordinate count | What must converge |
|:--|:--|:--|:--|
| One shared explicit mode | `GlobalPseudomodeModel` on `CompositePIBasis` | $n_{\mathrm{PI}}(n_{\max}+1)^2$ | Mode cutoff and numerical solver |
| One independent mode per identical system | `PISupersite` and local `pseudomode_model` | $\binom{N+[d(n_{\max}+1)]^2-1}{N}$ for one mode | Every local-mode cutoff and numerical solver |
| Common Gaussian bath represented by exponential correlations | `HEOMBath` and `HEOMPlan` | $n_{\mathrm{PI}}n_{\mathrm{ADO}}$ | Bath decomposition, hierarchy depth, and numerical solver |

The first two descriptions retain physical mode density matrices and
occupations. HEOM eliminates the bath into auxiliary density operators and
does not provide a physical single-mode state. Conversely, one damped
pseudomode represents a particular finite Markovian embedding, whereas HEOM
can retain several correlation poles without interpreting each ADO as a
physical oscillator.

A smaller shared-mode coordinate count is not an optimization of the local
model: it describes different bath correlations. The two coincide only in
special cases such as `N=1`.

## Cutoff and reliability checks

- Monitor `mode_operators.top_projector`. It measures the boundary population
  of the one shared mode; do not divide it by `N`.
- Compare common observables and reduced states after increasing `nmax`.
  A small top-level population is useful evidence, not a general error bound.
- Recheck time-step, Krylov, or stationary residual convergence independently
  of the cutoff.
- Restricted Schur sectors are valid only when every system term and
  collective coupling preserves them. Otherwise use a complete `PIBasis`.
- Deterministic time-local system rates may be negative, but stochastic
  damping channels require finite nonnegative rates.

The complete dependency-free example is
[`examples/global_pseudomode_cavity.jl`](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/blob/main/examples/global_pseudomode_cavity.jl).

## API

```@docs
GlobalPseudomodeModel
global_pseudomode_model
shared_pseudomode_model
global_pseudomode_workspace
global_pseudomode_matrixfree
global_pseudomode_state
global_pseudomode_state!
```
