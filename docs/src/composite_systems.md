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
workspaces instead.

## Current scope

- Finite auxiliary factors retain all `m^2` operator coordinates, so a bosonic
  mode must first be truncated to a finite `m`.
- Cross terms are sums of tensor products of factor maps. General interactions
  should be decomposed into such terms rather than materialized as a composite
  Kronecker matrix.
- The current high-level `PIModel` compiler acts on one PI ensemble. Composite
  generators are assembled from compiled local actions and explicit cross
  maps as above.
- There is not yet a composite quantum-trajectory compiler. The existing PI
  trajectory API applies to a single PI model, not to `CompositeSuperoperator`.
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
composite_tensor_operator
composite_tensor_state
composite_identity_operator
composite_trace_vector
CompositeSuperoperatorTerm
factorized_superoperator_term
local_superoperator_term
CompositeSuperoperator
CompositeSuperoperatorWorkspace
factor_left_superoperator
factor_right_superoperator
factor_sandwich_superoperator
composite_hamiltonian_superoperator
composite_dissipator_superoperator
composite_matrixfree
```

See [`examples/composite_ensembles.jl`](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/blob/main/examples/composite_ensembles.jl)
for a complete small calculation with two PI ensembles and one finite
ancilla.
