# Composite PI systems

`CompositePIBasis` combines independently permutationally invariant ensembles
with small ordinary Hilbert-space factors. Typical uses include two atomic
species, several spatial ensembles, or PI emitters coupled to a truncated
cavity or distinguished ancilla. Each ensemble remains in its compressed PI
operator space: constructing a composite basis never reconstructs an
ensemble's `d^N` Hilbert space.

## Factors and coordinate order

A PI factor is an ordinary `PIBasis`. A finite factor is described by
`FiniteOperatorBasis(m)` and retains all `m^2` column-major operator
coordinates of that auxiliary system.

```julia
using PermutationalInvariantDynamics

atoms_a = PIBasis(2, 2)
atoms_b = PIBasis(1, 2)
ancilla = FiniteOperatorBasis(2; label=:ancilla)
basis = CompositePIBasis(atoms_a, atoms_b, ancilla)
```

The first declared factor is the fastest-varying flattened index. If `x1`,
`x2`, and `x3` are factor operator-coordinate vectors, the composite vector
is

```julia
kron(x3, kron(x2, x1))
```

This convention is also used by every factorized superoperator. It makes a
term with actions `(A1, A2, A3)` mathematically equal to
`kron(A3, kron(A2, A1))`, although that Kronecker matrix is never formed by
the matrix-free kernel.

## States, operators, and normalization

Construct factorized objects with `composite_tensor_state` and
`composite_tensor_operator`. PI state components must be `PIState`s; PI
operator components must be `PIOperator`s. A finite component is an ordinary
square matrix of the declared auxiliary dimension.

```julia
rho_a = iid_state(atoms_a, ComplexF64[1 0; 0 0])
rho_b = iid_state(atoms_b, ComplexF64[0 0; 0 1])
rho_c = ComplexF64[1 0; 0 0]
rho = composite_tensor_state(basis, rho_a, rho_b, rho_c)

I_a = identity_operator(atoms_a)
I_b = identity_operator(atoms_b)
Z_c = ComplexF64[1 0; 0 -1]
Z = composite_tensor_operator(basis, I_a, I_b, Z_c)

trace(rho)                 # product of the physical factor traces
expectation(rho, Z)        # tr(Z' * rho)
```

PI factors retain the package's equation-(7) coefficient convention and
finite factors use orthonormal matrix units. Their tensor product is therefore
again orthonormal. `trace` contracts only joint diagonal coordinates and
fuses exact Schur multiplicities; it does not allocate a full composite trace
vector. `composite_trace_vector` is available when a solver explicitly needs
that vector.

To trace all factors except one repeatedly, prepare the packed diagonal
contraction once:

```julia
reduction = CompositeReductionPlan(rho, 1)
rho_a = composite_reduced_state(rho, reduction)

# Reuse the result storage too.
rho_a_buffer = PIState(reduction.kept_basis)
composite_reduced_state!(rho_a_buffer, rho, reduction)
```

The plan stores only exact-support diagonal offsets and exact Schur
multiplicity scales. It is immutable, tied to the exact composite basis, and
safe to share between tasks; each call writes only to its caller-owned result.
For a retained finite factor, use a square matrix as the destination.

## Matrix-free generators

A `CompositeSuperoperatorTerm` is a product of factor maps, with `nothing`
denoting the identity on a factor. The convenience constructor names only the
active factors:

```julia
term = factorized_superoperator_term(
    basis,
    1 => action_a,
    3 => action_c;
    coefficient=0.25,
)
generator = CompositeSuperoperator(basis, term)
```

For a local PI master equation, compile it normally and lift the compiled
action. Basis provenance is checked by object identity.

```julia
sm = ComplexF64[0 1; 0 0]
local_model = PIModel(atoms_a, (LocalJump(sm; rate=0.1),))
local_action = compile(local_model; backend=:matrixfree)
local_term = local_superoperator_term(basis, 1, local_action)
```

Tensor-product interactions can be built from factor-coordinate left, right,
and sandwich maps. The higher-level constructors implement

```math
-i r[\bigotimes_f H_f,\rho]
```

and

```math
r\,\mathcal D[\bigotimes_f L_f](\rho),\qquad
\mathcal D[L](\rho)=L\rho L^\dagger-\frac12\{L^\dagger L,\rho\}.
```

```julia
Jx_a = collective_operator(atoms_a, ComplexF64[0 1; 1 0])
X_c = ComplexF64[0 1; 1 0]
interaction = composite_hamiltonian_superoperator(
    basis, 1 => Jx_a, 3 => X_c; rate=0.05)

Jm_b = collective_operator(atoms_b, sm)
loss = composite_dissipator_superoperator(
    basis, 2 => Jm_b, 3 => sm; rate=0.02)

generator = CompositeSuperoperator(basis, local_term) + interaction + loss
```

Hamiltonian factors are checked for Hermiticity by default. Hamiltonian and
dissipator rates must be real, but may be negative; as elsewhere in the
package, a negative dissipative rate need not generate a completely positive
map.

Use one explicit workspace per task in repeated applications or evolution:

```julia
y = similar(rho.data)
workspace = CompositeSuperoperatorWorkspace(generator, rho.data)
apply!(y, generator, rho.data, 0.0, nothing, workspace)

evolution_workspace = EvolutionWorkspace(generator, rho.data)
evolve!(y, generator, rho.data, (0.0, 1.0);
        steps=128, workspace=evolution_workspace)
```

The warmed tensor-mode application reuses full-vector buffers, factor fibers,
and any nested `LiouvillianWorkspace`. `composite_matrixfree(generator)` gives
a synchronized compatibility `MatrixFreeLiouvillian` for APIs that require
that wrapper; parallel hot loops should retain explicit task-owned composite
workspaces instead. Prepared package consumers such as
`sensitivity_problem` discover the retained immutable composite plan and
construct a fresh task-owned batch workspace. A bare compatibility matrix
product through the wrapper remains a synchronized column fallback.

Several right-hand sides use a separate fixed-capacity workspace:

```julia
X = hcat(rho.data, perturbation_1, perturbation_2)
Y = similar(X)
batch_work = CompositeSuperoperatorBatchWorkspace(
    generator; capacity=size(X, 2))

apply!(Y, generator, X, 0.0, nothing, batch_work)
```

The factor maps consume equal tensor fibers from all columns together.
`apply_adjoint!` uses the same workspace. Capacity is immutable: a wider
matrix raises instead of growing hidden buffers. This path is useful for block
Krylov methods and parameter sensitivities; it is not a global Kronecker
materialization.

## Density-valued composite quantum jumps

Monitored tensor-product channels are declared explicitly. The background
contains trace-preserving coherent and unmonitored physics and must exclude
the monitored dissipators:

```julia
background = composite_hamiltonian_superoperator(
    basis, 1 => Jx_a, 3 => X_c; rate=0.05)

channel = CompositeJumpChannel(
    basis, 2 => Jm_b, 3 => sm;
    rate=0.02,
    label=:joint_emission,
)
plan = CompositeTrajectoryPlan(background, channel)
```

The plan adds one complete dissipator per channel and exposes the resulting
unconditional generator through `composite_master_superoperator(plan)`. This
split is intentional: an arbitrary sum of superoperator terms does not retain
enough information to infer a unique physical unraveling, and accepting an
already complete generator plus the same channels would double-count their
gains.

For channel $k$ with $G_k[\rho]=J_k\rho J_k^\dagger$ and
$Q_k=J_k^\dagger J_k$, the normalized conditional equation is

```math
\dot\rho_c=\mathcal L_0(\rho_c)
-\frac12\sum_k\gamma_k\{Q_k,\rho_c\}
+\left(\sum_k\lambda_k\right)\rho_c,
\qquad
\lambda_k=\gamma_k\,\mathrm{tr}(Q_k\rho_c).
```

A selected jump maps $\rho_c$ to
$G_k[\rho_c]/\mathrm{tr}(G_k[\rho_c])$. The fixed operator maps remain
factorized. Scalar rates may be driven, but every evaluated trajectory rate
must be finite, real, nonnegative, and representable in the prepared
precision. Rate schedules used by threaded batches must be pure and thread
safe; mutable external callback state is not synchronized by the plan.

The fixed-step backend integrates each channel hazard with the same RK4
stages as the conditional state. A trial is shortened and recomputed whenever
the jump probability corresponding to its integrated total hazard would
exceed `max_jump_probability`; a rapidly changing rate therefore cannot
bypass the configured probability cap merely because it was small at the
beginning of the step.

```julia
workspace = CompositeTrajectoryWorkspace(plan, rho)
path = quantum_trajectory(
    plan, rho, 0.0:0.05:2.0;
    dt=0.005,
    workspace=workspace,
)

batch = CompositeTrajectoryBatchWorkspace(plan, rho)
paths = quantum_trajectories(
    plan, rho, 0.0:0.05:2.0, 1_000;
    dt=0.005,
    seed=11,
    threaded=true,
    workspace=batch,
)
mean_states = trajectory_average(paths)
```

Plans are immutable and shareable. A single-path workspace owns the RK4
stages, one background workspace, one shared pair of full tensor buffers, and
small channel-specific factor-fibre scratch. The number of full composite
buffers is therefore independent of the number of monitored channels. Batch
workers own distinct workspaces and RNGs; global trajectory-index seeds make
serial and threaded scheduling sample the same ordered paths.

With named Hermitian `CompositePIOperator` observables and
`save_states=false`, `quantum_trajectories` returns online means, variances,
confidence intervals, and optional jump statistics without storing sampled
states. Individual paths are density-valued and can be mixed when the
background contains unresolved local PI channels.

## Current scope

- Finite auxiliary factors retain all `m^2` operator coordinates, so a bosonic
  mode must first be truncated to a finite `m`.
- Cross terms are sums of tensor products of factor maps. General interactions
  should be decomposed into such terms rather than materialized as a composite
  Kronecker matrix.
- The current high-level `PIModel` compiler acts on one PI ensemble. Composite
  generators are assembled from compiled local actions and explicit cross
  maps as above.
- Composite trajectories currently support fixed tensor-product jump
  operators and fixed-step conditional integration. Callable jump operators,
  arbitrary CP gain maps, event-driven integration, diffusive monitoring,
  adaptive-confidence stopping, Distributed batches, and composite weak-PI
  pseudo-kets are not implemented.
- A finite bosonic mode needs an explicit truncation. Converge that truncation,
  the fixed time step, maximum jump probability, and trajectory count
  independently.
- Single-ensemble analysis routines such as Schur-sector negativity or the
  population backend are not implicitly generalized to composite states.
  Reduce or contract the intended factors explicitly when an analysis needs
  a subsystem state.

## API

```@docs
FiniteOperatorBasis
CompositePIBasis
CompositePIOperator
CompositePIState
CompositeReductionPlan
composite_tensor_operator
composite_tensor_state
composite_identity_operator
composite_trace_vector
composite_reduced_state
composite_reduced_state!
CompositeSuperoperatorTerm
factorized_superoperator_term
local_superoperator_term
CompositeSuperoperator
CompositeSuperoperatorWorkspace
CompositeSuperoperatorBatchWorkspace
factor_left_superoperator
factor_right_superoperator
factor_sandwich_superoperator
composite_hamiltonian_superoperator
composite_dissipator_superoperator
composite_matrixfree
CompositeJumpChannel
CompositeTrajectoryPlan
CompositeTrajectoryWorkspace
CompositeTrajectoryBatchWorkspace
CompositeQuantumTrajectory
composite_master_superoperator
```

See [`examples/composite_ensembles.jl`](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/blob/main/examples/composite_ensembles.jl)
for a complete small calculation with two PI ensembles and one finite
ancilla. [`examples/composite_quantum_trajectories.jl`](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/blob/main/examples/composite_quantum_trajectories.jl)
compares a cross-factor jump ensemble with the independently propagated
master equation.
