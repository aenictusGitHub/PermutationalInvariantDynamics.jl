# Representations, states, and models

## Partitions, exact combinatorics, and representation theory

`exact_binomial` and `exact_multinomial` return `BigInt` values and are the
public route when a machine integer could overflow.  They do not choose a
floating approximation or a logarithmic representation.  PI construction
routines keep related ratios exact through cancellation and apply a
binary-scaled square root when needed; if the *final* nonzero factor is not
representable in the selected scalar type, the operation raises and the
caller must select a wider type.

```@docs
Partition
partitions
weight
length_nonzero
removable_corners
addable_corners
remove_corner
add_corner
minus_plus_neighbors
reachable_sectors
symmetric_group_dimension
unitary_group_dimension
commutant_dimension
exact_binomial
exact_multinomial
GTPattern
gt_patterns
PermutationalInvariantDynamics.isvalid
shape
content
gt_entry
triangular_shift
OneBoxCGCache
cgc
partition_triangle
three_nu_symbol
```

## PI bases, states, and one-body operators

```@docs
PIBasis
PIOperator
PIState
coefficient_block
physical_block
each_schur_block
operator_from_schur_blocks
state_from_schur_blocks
sector_metadata
sector_view
identity_operator
schur_sector_projector
fully_symmetric_projector
maximally_mixed_state
sector_maximally_mixed_state
symmetric_maximally_mixed_state
trace
purity
normalize!
ispositive
isphysical
positivity_diagnostics
state_diagnostics
validate_state
sector_population
sector_populations
basis_state
sector_density_matrix
iid_pure_state
iid_state
thermal_state
computational_product_state
symmetric_occupation_state
dicke_state
dicke_operator
w_state
cat_state
ghz_state
spin_coherent_state
spin_matrices
collective_spin
collective_block
collective_operator
mean_local_operator
local_kernel_element
local_kernel_operator
OneBodyGeometry
```

## Physical fully symmetric pure kets

These objects represent genuine physical pure states only when the basis
retains the single fully symmetric sector. They are distinct from
`WeakPIPseudoKet` and `HOPSRootKet`; see [Symmetric pure kets and
block-resolved entropy](../symmetric_kets_and_block_entropy.md) for the
comparison and scaling limits.

```@docs
SymmetricKet
symmetric_ket_dimension
validate_symmetric_ket
symmetric_occupation_ket
symmetric_product_ket
symmetric_ket_density
symmetric_ket_density!
symmetric_ket
```

## Appendix-D processes

```@docs
PBodyGeometry
pbody_collective_block
pbody_collective_operator
pbody_kernel_element
pbody_kernel_operator
```

## Supersites and pseudomode specifications

`PISupersite` groups the physical system and its local auxiliaries before
applying permutation symmetry. It is not a tensor product of global PI
operator spaces. The complete workflow, factor-ordering convention, cutoff
checks, and matrix-free examples are given in [Local pseudomodes and PI
supersites](../pseudomodes.md). One mode shared by the complete ensemble
instead uses a finite composite factor; see [Global pseudomodes and shared
cavities](../global_pseudomodes.md).

```@docs
PISupersite
supersite_tensor_operator
lift_supersite_operator
lift_system_operator
lift_system_pbody_operator
lift_system_term
supersite_iid_state
supersite_product_state
BosonicPseudomode
PseudomodeCoupling
pseudomode_supersite
lift_pseudomode_operator
pseudomode_operators
pseudomode_coupling_terms
pseudomode_damping_terms
pseudomode_model
independent_local_pseudomode_model
pseudomode_product_state
```

## Physical terms and models

```@docs
AbstractPITerm
InPlaceTimeOperator
LocalHamiltonian
CollectiveHamiltonian
LocalJump
CollectiveJump
CorrelatedLocalJumps
CorrelatedCollectiveJumps
DirectPIHamiltonian
DirectPIJump
PIModel
PBodyHamiltonian
LocalPBodyJump
CollectivePBodyJump
qubit_ensemble_model
```

## Vectorized superoperators and PI tests

```@docs
left_superoperator
right_superoperator
sandwich_superoperator
commutator_superoperator
dissipator_superoperator
is_pi_operator
is_pi_superoperator
is_permutationally_invariant
```
