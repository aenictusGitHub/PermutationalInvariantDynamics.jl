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
maximally_mixed_state
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
dicke_state
dicke_operator
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

## Appendix-D processes

```@docs
PBodyGeometry
pbody_collective_block
pbody_collective_operator
pbody_kernel_element
pbody_kernel_operator
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
