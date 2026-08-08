# Framework and physical conventions

`PermutationalInvariantDynamics.jl` simulates open dynamics of $N$
identical $d$-level systems when both the state and its evolution are
invariant under relabelling the particles. The reduction is exact: the package
keeps every degree of freedom in the permutationally invariant (PI) operator
space, rather than closing correlations or projecting onto a mean-field
ansatz. At fixed local dimension $d$, this space grows polynomially with
$N$, even though the full many-body Hilbert and Liouville spaces grow
exponentially.

The representation follows Thierry Bastin and John Martin,
*J. Phys. A: Math. Theor.* **58**, 275301 (2025),
[doi:10.1088/1751-8121/addfc1](https://doi.org/10.1088/1751-8121/addfc1).
This chapter introduces the physical assumptions, the Schur--Weyl blocks
stored by the package, and the recommended path from a model to research
observables.

If you prefer to construct and solve a model before studying the
representation, start with [Getting started: from a model to a
solution](getting_started.md), then return here for the mathematical
conventions behind each object.

## When the PI framework applies

Let

```math
\mathcal H=(\mathbb C^d)^{\otimes N}
```

and let $P_\pi$ permute the tensor factors according to
$\pi\in S_N$. An operator $A$ is permutationally invariant when

```math
P_\pi A P_\pi^\dagger=A
\qquad\text{for every }\pi\in S_N.
```

In particular, a PI density operator describes particles that are physically
equivalent under relabelling. This condition does **not** mean that the state
must be pure, that the particles must be bosons, or that the state must be
supported only in the fully symmetric Hilbert-space sector.

Define the permutation superoperator

```math
\mathcal P_\pi(X)=P_\pi X P_\pi^\dagger.
```

A sufficient and natural condition for a generator to preserve PI states is
permutation covariance,

```math
\mathcal P_\pi\mathcal L(t)
=\mathcal L(t)\mathcal P_\pi
\qquad\text{for every }\pi.
```

Thus, if $\rho(0)$ is PI, the solution remains PI. An individual local jump
operator need not itself be invariant: the sum of identical channels over
all particle labels is covariant as a whole. The package's built-in term
constructors encode precisely these PI sums.

The deterministic dynamics may be a general time-local Lindblad-like
equation,

```math
\frac{d\rho}{dt}
=-\frac{i}{\hbar}[H(t),\rho]
+\sum_k\gamma_k(t)\,\mathcal D[L_k(t)]\rho,
```

with

```math
\mathcal D[L]\rho
=L\rho L^\dagger
-\frac12\left(L^\dagger L\rho+\rho L^\dagger L\right).
```

Time-dependent real rates may be negative. This is useful for time-local
non-Markovian models, but the resulting map is not guaranteed to be
completely positive. Quantum-jump trajectories require nonnegative rates,
because negative rates do not define jump probabilities.

## Schur--Weyl decomposition

Schur--Weyl duality decomposes the many-body Hilbert space as

```math
(\mathbb C^d)^{\otimes N}
\simeq
\bigoplus_{\substack{\nu\vdash N\\\ell(\nu)\le d}}
S^\nu\otimes U^\nu(d).
```

Here $\nu=(\nu_1,\ldots,\nu_d)$ is a partition of $N$, padded with
zeros, and may be drawn as a Young diagram. The symmetric-group irrep
$S^\nu$ has dimension

```math
f^\nu=\dim S^\nu,
```

the number of standard Young tableaux of shape $\nu$. The unitary-group
irrep $U^\nu(d)$ has dimension

```math
g_\nu=\dim U^\nu(d).
```

The package labels a basis of $U^\nu(d)$ by Gel'fand--Tsetlin (GT)
patterns, denoted $W$ below. A Schur basis vector can therefore be written

```math
|\nu,T,W\rangle,
```

where $T=1,\ldots,f^\nu$ is the symmetric-group multiplicity label and
$W=1,\ldots,g_\nu$ is retained explicitly.

Schur's lemma implies that every PI operator acts identically on the
$f^\nu$ equivalent copies within a sector:

```math
A_{\mathrm{PI}}
=\bigoplus_\nu I_{S^\nu}\otimes R_\nu.
```

The small matrices $R_\nu$ contain all physical information. The package
constructs and evolves these blocks directly; production algorithms never
construct a Schur transform or a $d^N$-dimensional operator.

### The orthonormal PI basis

The numerical representation uses the orthonormal operator basis of equation
(7) of Bastin and Martin,

```math
F_\nu^{W,W'}
=\frac{1}{\sqrt{f^\nu}}
\sum_{T=1}^{f^\nu}
|\nu,T,W\rangle\langle\nu,T,W'|.
```

A PI operator is stored as

```math
A=\sum_{\nu,W,W'}(C_\nu)_{W,W'}F_\nu^{W,W'}.
```

`coefficient_block(A, nu)` returns the stored matrix $C_\nu$, while
`physical_block(A, nu)` returns the matrix acting on one Schur copy,

```math
R_\nu=\frac{C_\nu}{\sqrt{f^\nu}}.
```

For block-wise input and inspection, `each_schur_block(A)` iterates over
`partition => physical_block` pairs in the basis sector order.  The matching
constructors apply the normalization explicitly:

```julia
blocks = collect(each_schur_block(A))
B = operator_from_schur_blocks(A.basis, blocks)

# Stored equation-(7) blocks can instead be round-tripped explicitly.
C = operator_from_schur_blocks(
    A.basis,
    each_schur_block(A; representation=:coefficient);
    representation=:coefficient,
)
```

`state_from_schur_blocks` follows the same convention and optionally calls
`validate_state`; it never normalizes or repairs the supplied blocks.
`sector_metadata(basis)` reports each block dimension, flattened coordinate
range, exact symmetric-group multiplicity, and retained Hilbert-space
dimension without expanding the multiplicity copies.

This normalization makes the global PI coordinates orthonormal in the
Hilbert--Schmidt inner product:

```math
\mathrm{Tr}(A^\dagger B)
=\sum_\nu
\mathrm{tr}\!\left(C_\nu(A)^\dagger C_\nu(B)\right).
```

Several package conventions then follow immediately:

```math
\begin{aligned}
\mathrm{Tr}(A)
  &=\sum_\nu\sqrt{f^\nu}\,\mathrm{tr}(C_\nu),\\
C_\nu(AB)
  &=\frac{C_\nu(A)C_\nu(B)}{\sqrt{f^\nu}},\\
C_\nu(I)
  &=\sqrt{f^\nu}\,I,\\
\mathrm{Tr}(\rho^2)
  &=\sum_\nu\lVert C_\nu(\rho)\rVert_F^2.
\end{aligned}
```

For a density operator, the population of sector $\nu$ is

```math
p_\nu
=f^\nu\mathrm{tr}(R_\nu)
=\sqrt{f^\nu}\mathrm{tr}(C_\nu).
```

A valid state has Hermitian positive-semidefinite $R_\nu$ in every retained
sector and $\sum_\nu p_\nu=1$. `validate_state` checks these properties
without normalizing, symmetrizing, or clipping the input.

Young-diagram visualizations display the shape $\nu$, not an arbitrarily
chosen filled tableau. The $T$ label is summed over in $F_\nu^{W,W'}$, and
its exact multiplicity $f^\nu$ is retained analytically. See
[Schur-block visualization](schur_visualization.md) for the block and
rendering conventions.

### PI is broader than the symmetric sector

The fully symmetric Hilbert-space sector is only
$\nu=(N,0,\ldots,0)$, for which $f^\nu=1$. A pure identical product state
$|\psi\rangle^{\otimes N}$ lies in this sector, but an identical mixed
product state $\sigma^{\otimes N}$ generally has support in several Schur
sectors. Independent local dissipation can also transfer population between
sectors even though the resulting state remains PI.

Two projections that are sometimes both called the "permutation projector"
must be distinguished. The Hilbert-space projector onto invariant vectors is

```math
P_{\mathrm{sym}}
=\frac{1}{N!}\sum_{\pi\in S_N}P_\pi,
```

and selects only the fully symmetric sector. Construct it directly in PI
coordinates with `fully_symmetric_projector(basis)`. More generally,
`schur_sector_projector(basis, nu)` constructs the isotypic projector
$P_\nu$ with

```math
C_\nu(P_\nu)=\sqrt{f^\nu}\,I,
```

and all other blocks zero. These projectors are not trace-normalized;
`trace(P_nu)` is the Hilbert-space rank of the sector. For an existing state,
`sector_population(rho, nu)` evaluates $\mathrm{Tr}(P_\nu\rho)$
without allocating the projector.

By contrast, the Hilbert--Schmidt projector from arbitrary operators onto the
PI operator algebra is the permutation twirl

```math
\mathcal T_{\mathrm{PI}}(A)
=\frac{1}{N!}\sum_{\pi\in S_N}P_\pi A P_\pi^\dagger.
```

Every `PIState` and `PIOperator` is already in the range of this twirl, so it
acts as the identity on the package's compressed coordinates. Applying the
twirl to an arbitrary non-PI operator instead requires an external structured
representation or an exponentially large ambient input; the package does not
construct such a full-space object.

Consequently, `PIBasis(N, d)` retains all sectors by default. A restricted
basis is appropriate only when the model preserves the requested sector set.
For example, a purely collective model can often be studied in the symmetric
sector, whereas a generic `LocalJump` cannot. Model construction rejects a
restricted basis when a built-in local process requires omitted sectors; it
never truncates those transitions silently. Completeness is certified from the
exact Schur--Weyl PI-dimension identity, and local p-body closure is generated
from the `p`-box ancestors of the retained sectors and their descendants. Model validation therefore
does not enumerate every partition merely to validate a large restricted basis.

### Certified diagonal-population dynamics

Some generators preserve states that are diagonal in the stored GT-pattern
basis. In that case define the physical population of pattern `W` in sector
`nu` by

```math
p_{\nu,W}=\sqrt{f^\nu}(C_\nu)_{W,W}
             =f^\nu(R_\nu)_{W,W}.
```

These populations include the exact symmetric-group multiplicity and satisfy
`sum(p) == trace(rho)`. `PopulationPlan(model)` certifies that no diagonal
input can generate a retained off-diagonal Schur coordinate, then lowers the
model to a sparse reduced generator. The resulting coordinate count is
`sum(g_nu)`, rather than the general PI count `sum(g_nu^2)`. For qubits this
reduces the evolving vector from cubic to quadratic growth in `N`.

```julia
plan = PopulationPlan(model)
report = plan.invariance
p0 = diagonal_populations(rho0)
solution = solve_populations(plan, p0, (0.0, 10.0); saveat=0.1)
rho_final = state(solution, lastindex(solution))
```

Constructing the plan once and reading `plan.invariance` avoids compiling the
restriction twice; call `population_invariance(model)` only when no plan is
needed. The default certificate is structurally strict (`atol=rtol=0`), so a
weak but nonzero mixing term is never projected away. Supplying a nonzero
tolerance is an explicit approximate request recorded as
`reason=:within_tolerance` when it accepts nonzero leakage.

The certificate is deliberately basis dependent. Scalar time-dependent rates
on fixed operators are checked term by term and remain supported. A genuinely
operator-valued schedule returns an inconclusive report until the model is
`freeze`d at an explicit time. An appreciably coherent initial state is also
rejected by `diagonal_populations`; `check=false` is the explicit state-
projection route. No population path silently removes coherences or
normalizes a state.

The current `PopulationPlan` compiler forms the standalone `sqrt(f^nu)`
coordinate scale and the parent Liouvillian trace functional in the working
real type. Construction therefore raises when that factor is outside the
type's finite range; use wider coefficients for the plan. Direct population
extraction and reconstruction use prepared exact scaled products and may
still succeed whenever the requested final entries are representable.

### Spin and Dicke conveniences

`spin_matrices(d)` returns the standard spin-$j=(d-1)/2$ generators in the
ascending order $|-j>,...,|j>$. For qubits the package local order is
$(|g>,|e>)=(|-1/2>,|+1/2>)$. Thus `jm` is $|g><e|$ and `jx`, `jy`, and
`jz` are Pauli matrices divided by two. `collective_spin(basis, :x)` and the
other named components lower these matrices directly to Schur blocks.

The state helpers retain the same convention. `computational_product_state`
uses a one-based local level and works for qubits or qudits.
`symmetric_occupation_state(basis, (n1, ..., nd))` constructs a qudit
occupation state directly in the fully symmetric sector. The counts follow
the same one-based local-level order and must sum to `basis.N`. Its exact
combinatorial rank costs only $O(d)$ to evaluate, so the constructor does not
scan the symmetric block.

For qubits, `dicke_state(basis, k)` is the short excitation-count form and
`w_state(basis)` is its one-excitation specialization. The more general
`dicke_state(basis, j, m)` and `dicke_operator(basis, j, m, mp)` map spin
labels to the corresponding partition and GT-pattern weights. In a sector of
multiplicity $f^\nu$, their selected physical block entry is $1/f^\nu$:
the PI object is uniform over the indistinguishable multiplicity copies.

`cat_state(basis, a, b; phase=phi)` constructs a balanced cat state between
any two distinct qudit levels. `ghz_state` is the qubit levels-1-and-2
specialization, and `spin_coherent_state` constructs a qubit coherent-spin
state. None of these helpers forms a $d^N$ vector.

There are three distinct useful white states. `maximally_mixed_state(basis)`
is uniform over the complete Hilbert space represented by all retained
sectors. `sector_maximally_mixed_state(basis, nu)` is normalized on one Schur
isotypic sector, including all of its multiplicity copies, while
`symmetric_maximally_mixed_state(basis)` selects the fully symmetric sector.
All retain the caller's exact basis and raise if a requested sector is absent.

## Dimension and computational scaling

The different representations have the following dimensions:

| Space | Dimension |
|:--|:--|
| Many-body Hilbert space | $d^N$ |
| Full Liouville space | $d^{2N}$ |
| Complete PI operator space | $n_{\mathrm{PI}}=\sum_\nu g_\nu^2={N+d^2-1\choose N}$ |

At fixed $d$,

```math
n_{\mathrm{PI}}=O\!\left(N^{d^2-1}\right),
```

so the symmetry changes exponential growth into polynomial growth. For
qubits,

```math
n_{\mathrm{PI}}={N+3\choose 3}.
```

If only the fully symmetric sector is retained, its Hilbert-space irrep has

```math
g_{(N)}={N+d-1\choose N}
```

states and $g_{(N)}^2$ PI operator coordinates.

In this one-sector case, collective one-body operators are lifted directly in
the occupation basis through

```math
a_a^\dagger a_b\lvert\boldsymbol n\rangle
=\sqrt{n_b(n_a+1)}\,
 \lvert\boldsymbol n+\boldsymbol e_a-\boldsymbol e_b\rangle
\quad (a\ne b).
```

This avoids constructing the general one-box recoupling geometry. Fixed
collective terms retain their exact sparse block support, while driven terms
fill preallocated dense block scratch because their support may change with
time. Collective-only models spanning several Schur sectors prepare only
sector-diagonal one-box contractions. Models with local gains still use the
complete sector-changing one-box geometry.

Polynomial scaling does not make every operation inexpensive. A dense PI
Liouvillian contains $n_{\mathrm{PI}}^2$ entries, and a complete dense
eigendecomposition has cubic cost in $n_{\mathrm{PI}}$. The matrix-free
backend avoids materializing that matrix, but it still stores PI state and
Krylov vectors of length $n_{\mathrm{PI}}$. At fixed body order $p$, the
local contraction cost includes $d^{2p}$, and general-qudit reductions can
retain large packed Littlewood--Richardson intertwiner spaces and use a dense
temporary nullspace basis during setup. Consult
[Architecture and efficient workflows](architecture.md) and
[Matrix-free Krylov solvers](matrix_free_krylov.md) before a large scan.

### Exact combinatorics and representable numerical factors

Combinatorial coefficients can exceed a machine integer, or even
`floatmax(T)`, well before the corresponding PI calculation becomes too
large.  The public helpers return exact integers:

```julia
choose = exact_binomial(200, 100)            # BigInt
ways = exact_multinomial((80, 70, 50))       # BigInt
```

`symmetric_group_dimension`, `unitary_group_dimension`, and
`commutant_dimension` likewise return exact `BigInt` results.  These exact
functions do not return logarithms or floating approximations; storing and
printing an exceptionally large answer still costs space proportional to the
number of its digits.

Numerical PI kernels delay floating conversion until exact cancellations
have been performed.  One- and p-body Schur prefactors are formed from exact
rational branching/path weights rather than separately converting a large
binomial coefficient and large sector multiplicities.  Square roots are
evaluated after an exact power-of-two rescaling, so an intermediate
coefficient may exceed `floatmax(T)` when the final square root is still
representable.  When a large exact factor multiplies a small rate, block, or
contraction, their binary scales are combined before conversion; the factor
is never required to be representable by itself.  Ordinary representable
factors retain a direct native-arithmetic path. Cancellation-prone collective
blocks and static local-gain coordinates first check their native path sum,
then selectively recompute only the affected sector or sector-pair group with
guarded wider isometries and contractions before converting the checked result
back to the declared type. This includes severe cancellation when every exact
path-pair factor is individually representable, and large sums whose native
forward-error bound cannot certify even the final working-precision ulp. A widened value is certified
against a forward roundoff bound derived from its absolute path sum and
operation count. If it remains inside that uncertainty interval, the operation
raises rather than treating the interval as proof of an exact cancellation;
stable nonzero values still undergo the package's strict range checks. For an `InPlaceTimeOperator`,
caller-owned scratch fixes the scalar type in advance; a cancellation-prone
dynamic p-body block or local gain therefore raises with guidance to widen the
prototype instead of returning a working-precision discrepancy. Preallocated
one-body schedules follow the same rule once `N` exceeds the native geometry's
cancellation threshold. Static collective blocks and local-map elements can
instead widen only the requested result. Dynamic direct-PI schedules retain
their coefficient storage and apply the inverse Schur-multiplicity scale in a
fused form, so an unrepresentable standalone `sqrt(f^nu)` does not discard a
representable physical block.

PI operator multiplication likewise keeps its ordinary BLAS block product
when the intermediate is representable. If `C_A*C_B` overflows before the
required `1/sqrt(f^nu)` scaling, the exceptional path fuses that exact scale
with each product term and widens only an uncertified dot product or complex
component cancellation. Thus large-multiplicity identities remain idempotent without imposing
wider arithmetic on ordinary sectors. Physical Schur visualization metrics
also aggregate directly first; only a failed physical-range conversion causes
power-of-two entry scaling before the norm or SVD and joint restoration of the
binary and multiplicity scales.

Pure product-state amplitudes use normalized conditional
binomial recurrences, seeded near each binomial mode and advanced by adjacent
ratios, instead of evaluating a large multinomial coefficient separately
from the amplitude powers. One-site normalization is checked in at least the
recurrence precision, and its predicted tensor-power trace must remain within
the requested tolerance. Thus a low-precision norm that rounds to one cannot
silently become a badly normalized large-`N` state.

This handling preserves the requested storage type; it does not silently
change a `Float32` or `Float64` result to wider storage. If the final stored
value lies outside that type's exact nonzero finite range, the package raises
an `ArgumentError` rather than returning zero, `Inf`, or `NaN`. Individual
state amplitudes still follow ordinary rounding in their requested output
type. Supply wider numerical data, such as `BigFloat`, when the final stored
values themselves require the wider range.

State analyses avoid forming a per-copy block when it is unnecessary. They
use `sqrt(f^nu) * C_nu = f^nu * rho_nu`, whose trace is the sector
probability, for entropy, distances, collective information, reductions, and
spin phase space. Thus these bounded quantities can remain available after a
per-copy `physical_block`, purity, or Float64 coefficient-space trace vector
has become genuinely unrepresentable.

## Physical model terms

Let $h$, $\ell$, and $L$ be matrices on one particle, and let $h_p$,
$\ell_p$, and $L_p$ act on $p$ particles. For a subset
$S\subset\{1,\ldots,N\}$, a subscript $S$ denotes the corresponding
embedded operator. The built-in terms map to the following physical objects.

| Constructor | Contribution before the common scalar rate |
|:--|:--|
| `LocalHamiltonian(h)` | $H=\sum_i h^{(i)}$ |
| `CollectiveHamiltonian(h)` | $H=\sum_i h^{(i)}$ |
| `LocalJump(ell)` | $\sum_i\mathcal D[\ell^{(i)}]$ |
| `CollectiveJump(L)` | $\mathcal D[J]$, $J=\sum_iL^{(i)}$ |
| `CorrelatedLocalJumps((La,...), Gamma)` | $\sum_i\sum_{a,b}\Gamma_{ab}\mathcal D_{ab}[L_a^{(i)},L_b^{(i)}]$ |
| `CorrelatedCollectiveJumps((La,...), Gamma)` | $\sum_{a,b}\Gamma_{ab}\mathcal D_{ab}[J_a,J_b]$ |
| `PBodyHamiltonian(hp, p)` | $H_p=\sum_{\lvert S\rvert=p}(h_p)_S$ |
| `LocalPBodyJump(ellp, p)` | $\sum_{\lvert S\rvert=p}\mathcal D[(\ell_p)_S]$ |
| `CollectivePBodyJump(Lp, p)` | $\mathcal D[J_p]$, $J_p=\sum_{\lvert S\rvert=p}(L_p)_S$ |
| `DirectPIHamiltonian(H)` | Commutator with an existing `PIOperator` $H$ |
| `DirectPIJump(L)` | Dissipator of an existing PI operator $L$ |

`LocalHamiltonian` and `CollectiveHamiltonian` lower to the same identical
one-body Hamiltonian sum; the two names allow a model to express its physical
intent. Local and collective **jump** constructors are not equivalent. The
local channel is an incoherent sum of dissipators and can couple Schur
sectors. The collective channel first sums the amplitudes and then forms one
dissipator; it is sector diagonal.

The correlated constructors accept a Hermitian positive-semidefinite
Kossakowski matrix `Gamma`, with
$\mathcal D_{ab}[A,B]\,(\rho)=A\rho B^\dagger-\{B^\dagger A,\rho\}/2$.
A fixed matrix is validated and factorized once in
the small channel space; its effective independent jumps then use the same PI
kernels as `LocalJump` or `CollectiveJump`. `InPlaceTimeOperator` also accepts
a fixed-shape Kossakowski-matrix schedule. Its evaluated matrix and
factorization scratch belong to `LiouvillianWorkspace`, and every value is
revalidated before use.

The `rate` keyword multiplies the displayed contribution. Hamiltonian terms
also accept `hbar`. Operators supplied to a $p$-body constructor have size
$d^p\times d^p$ and must be invariant under permutations of their $p$
tensor factors. All subset sums are over unordered subsets. The package uses
rates exactly as supplied and never inserts a Kac or thermodynamic scaling
factor. When both a Hamiltonian `rate` and `hbar` are exact Integer/Rational
values, their quotient is cancelled exactly before checked floating
conversion. Ordinary floating inputs retain the direct division path, so
small fixed-precision models pay no multiprecision cost.

For time dependence, a scalar coefficient may be written as
`rate=(t, parameters) -> ...`. `InPlaceTimeOperator` provides a prepared,
fixed-shape schedule when the operator itself changes with time. Stationary
states and ordinary spectra require an autonomous generator. Use `freeze` only
when the stationary properties of one instantaneous generator are the
intended question; use direct time evolution or Floquet analysis for the
actual driven dynamics.

For the common six-channel qubit bath,
`qubit_ensemble_model` is a convention-explicit shorthand. Its local keywords
`emission`, `dephasing`, and `pumping` multiply dissipators of $j_-$,
$j_z$, and $j_+$ respectively; the three `collective_...` keywords use
the corresponding coherent sums. Because $j_z=sigma_z/2$, a single-qubit
off-diagonal element decays at half the `dephasing` keyword rate. Fixed zero
rates are omitted, so a collective-only model can use a symmetry-restricted
basis, while callable or negative rates are retained exactly as supplied.

## Package objects and ownership

The main data flow is

```text
PIBasis -> PIModel -> compile -> CompiledPIModel -> solver or analysis

(PI and finite factors) -> CompositePIBasis -> CompositeSuperoperator

(PI system, one shared mode) -> GlobalPseudomodeModel -> generator or workspace

(system, identical local modes) -> PISupersite -> pseudomode_model -> compile

(composite background, monitored jumps) -> CompositeTrajectoryPlan
                                         -> quantum_trajectories

(parameter grid, builder) -> ParameterScanPlan -> parameter_scan

(periodic compiled generator) -> FloquetMap -> multiplier or fixed-point solver

(autonomous compiled generator) -> ResponseWorkspace -> response or adjoint action

(PI system, exponential baths) -> scaled/unscaled HEOMPlan -> HEOM state or solver

(PI Hamiltonian, shared HOPSBaths) -> HOPSPlan -> HOPSWorkspace per path
                                               -> root-state ensemble
```

PI--HOPS has a stronger symmetry requirement than the averaged open-system
equation: every individual noise realization must preserve the PI
representation. The exact PI route therefore supports shared baths whose
coupling operators are collective/PI. Independent local colored noises are PI
only after ensemble averaging and generally leave the PI pseudo-ket space on
each path; use PI--HEOM or identical local pseudomode supersites for that
case. Linear HOPS propagates an unnormalized root pseudo-ket
$\psi_{\boldsymbol 0}$. Its density estimate is
$\mathbb E[|\psi_{\boldsymbol 0}\rangle\langle\psi_{\boldsymbol 0}|]$:
average the outer products without normalizing individual paths.

| Object | Role | Ownership rule |
|:--|:--|:--|
| `PIBasis` | Partitions, GT patterns, block offsets, and representation geometry labels | Share read-only |
| `PISupersite`, `BosonicPseudomode` | Exact factorization and finite-cutoff metadata for one identical system plus its local auxiliaries | Share read-only; one supersite basis is reused by every model at the same cutoffs |
| `PIState`, `PIOperator` | Dense vectors of orthonormal PI coefficients | Mutable value owned by the caller |
| `CompositePIBasis`, `CompositePIState`, `CompositePIOperator` | Tensor products of several PI spaces and finite auxiliary matrix spaces | Basis is shared read-only; state/operator data belong to the caller |
| `GlobalPseudomodeModel` | One PI system factor, one shared finite mode, collective couplings, and separated mode damping channels | Model and prepared maps are shared read-only; one global-pseudomode workspace per concurrent application |
| `PIModel` | Declarative immutable tuple of physical terms | Share read-only |
| `CompiledPIModel` | Prepared term lowering and sparse or matrix-free backend | Compile once and share read-only |
| `LiouvillianWorkspace` and solver workspaces | Mutable multiplication and Krylov scratch | One per concurrent task or thread |
| Advanced Krylov workspaces | Dominant bases/residual arrays for block, multi-shift, recycled, or exponential actions | One per concurrent solve; recycled state belongs to one ordered chain |
| `FloquetMap`, `FloquetWorkspace` | Immutable period/integration recipe and forward/adjoint RK scratch | Map shared with synchronized convenience calls; one explicit workspace per concurrent period action |
| `ResponseWorkspace` | Restarted-GMRES and/or exponential-action storage for one prepared source | Reuse sequentially; one workspace per concurrent response task |
| `ParameterScanPlan`, `ParameterScanWorkspace` | Immutable scan recipe and mutable continuation/solver scratch | Plan shared read-only; one workspace per serial caller or threaded worker |
| `HEOMPlan`, HEOM workspaces | Immutable scaled/unscaled ADO topology/couplings and application/RK4 scratch | Plan shared read-only; one workspace per concurrent application/evolution |
| `HOPSPlan`, `HOPSWorkspace` | Immutable shared-bath hierarchy geometry and mutable auxiliary pseudo-kets/noise/RK scratch | Plan shared read-only; one workspace and RNG per concurrent path |
| `WeakPITrajectoryPlan`, weak-PI workspaces | Immutable Schur-Kraus unraveling and task-owned path/batch scratch | Plan shared read-only; one workspace and RNG per concurrent worker |
| `QuditHusimiPlan` | Dense coherent vectors for one exact basis, point set, and sector selection | Share read-only across states; setup can dominate |
| `DiffusiveBatchPlan`, batch workspaces | Prepared trajectory request and worker-local path/RNG buffers | Plan shared read-only; workspace reused sequentially only |
| `ConvergenceStudyResult` | All raw refinement results, estimates, diagnostics, and decisions | Immutable record; memory includes every retained evaluator result |
| `CollectiveObservablePlan`, `LocalFactorTracePlan`, `ReductionPlan`, `CompositeReductionPlan` | Prepared observable, internal local-factor trace, particle-bipartition geometry, or composite-factor trace | Share read-only; tied to the exact basis object |
| `OneBodyRDMWorkspace`, `LocalFactorTraceWorkspace`, `ReductionWorkspace` | Mutable one-body, occupation, or product-Schur reduction scratch | One per concurrent task |
| `CorrelationPlan`, `CorrelationWorkspace` | Prepared quantum-regression insertions and their evolution/GMRES scratch | Plan shared read-only; one workspace per task |
| `CompositeSuperoperator`, `CompositeSuperoperatorWorkspace` | Sum of factorized maps and its tensor-fibre scratch | Generator shared read-only; one workspace per task |
| `CompositeTrajectoryPlan`, composite trajectory workspaces | Explicit monitored tensor-product channels and density-valued conditional evolution | Plan shared read-only; one workspace and RNG per concurrent path worker |

`compile(model; backend=:auto)` performs the expensive representation setup
once and chooses a conservative backend. For standard fixed kernels, the
sparse-memory bound uses their prepared exact block support; driven or custom
kernels with unknown support retain a safe dense-coordinate fallback. This
lets structured collective generators select sparse storage without charging
them for a fictitious dense PI matrix. Matrix-free application is preferable
when the resulting sparse matrix would still dominate memory, for driven
models, or for Krylov methods. Explicit `apply!` and `apply_adjoint!` calls
should reuse a caller-owned workspace in hot or parallel loops. The convenient
compatibility calls are synchronized and safe, but concurrent calls serialize.

When a small model needs several one-body/Appendix-D geometry families,
compilation automatically shares one bounded, transient `OneBoxCGCache`
during lowering. For repeated independent compilations, prepare a basis-owned
cache at the greatest required body order and pass it as `coefficient_cache`
to `LiouvillianPlan`, `compile`, `liouvillian`, `steady_state`, or
`compile_family`. It is read-only in use and never becomes process-global
state; incompatible basis, depth, or precision requests raise.

Prepared observables and reductions follow the same pattern: construct the
read-only plan once, then reuse it for many states. `LocalFactorTracePlan`
traces one internal tensor factor from every supersite and returns a complete
PI basis at the kept local dimension. Its prepared occupation transforms are
rectangular, exact-support sparse, and memory-guarded. `ReductionPlan` instead
traces or partially transposes groups of particles at fixed local dimension.
For a reduced state it contracts the discarded product factor one slice at a
time; only negativity needs the complete product block. Qudit
`ReductionPlan` objects can be much larger than collective-observable plans
because they may retain many subduction intertwiners. They store those maps as
exact-support sparse discarded-weight blocks and report the packed versus
dense-equivalent size in `plan.estimates`, but the weight-restricted SPQR setup
can still be costly. Benchmark setup and retained memory before caching many
reductions. The detailed stable, advanced, and experimental interfaces are listed in
[API tiers and prepared analysis](api_tiers.md).

## A complete qubit example

The following script defines driven qubits with independent and collective
decay, evolves an initially excited product state, samples a collective
observable, and computes the autonomous steady state.

```julia
using PermutationalInvariantDynamics

N = 12
basis = PIBasis(N, 2)  # all Schur sectors

spin = spin_matrices()  # local order: (|g>, |e>)

model = PIModel(basis, [
    LocalHamiltonian(spin.jx; rate=0.8),
    LocalJump(spin.jm; rate=0.10),
    CollectiveJump(spin.jm; rate=0.02 / N),
])

# Local level 2 is |e>; the constructor uses Julia's one-based index.
rho0 = computational_product_state(basis, 2)

# Representation geometry and Liouvillian kernels are prepared once.
prepared = compile(model; backend=:auto)

times = range(0.0, 10.0; length=101)
solution = solve_dynamics(
    prepared, rho0, (first(times), last(times));
    saveat=times,
    steps_per_interval=16,
    observables=(magnetization=spin.jz,),
    save_states=false,
)

# A local matrix in `observables` denotes its collective sum. Only the
# resulting scalar time series is retained here, not 101 PI state vectors.
magnetization = real.(solution.observables[:magnetization]) ./ N

rho_ss = stationary_state(prepared)
report = diagnostics(rho_ss)

println("PI coordinate dimension = ", pi_dimension(basis))
println("final magnetization per particle = ", last(magnetization))
println("steady-state trace error = ", report.trace_error)
```

The fixed-step solver is deliberately explicit about its temporal resolution;
increase `steps_per_interval` until the reported observables are converged.
For adaptive or stiff integration, construct `dynamics_problem(prepared,
rho0, tspan)` and solve it with a compatible SciML algorithm from an added
solver package. The paired scripts in [Research examples](research_examples.md)
show literature models and their numerical checks.

## Choosing a workflow

| Research task | Recommended entry point | Essential qualification |
|:--|:--|:--|
| Exact finite-$N$ PI dynamics | `compile`, then `solve_dynamics` | Converge the fixed time step |
| Adaptive or stiff dynamics | `dynamics_problem` | Add and configure a SciML solver |
| Autonomous steady state | `stationary_state` | Inspect residual, trace, and possible degeneracy |
| Slow modes or Liouvillian gap | `liouvillian_spectrum`, `pi_liouvillian_gap` | A selected Krylov window is not a complete spectrum |
| Periodic dynamics and selected multipliers | `floquet_map`, `selected_floquet_multipliers`, `floquet_steady_state` | Converge the one-period integration separately from Krylov residuals |
| Quantum-jump ensembles | `quantum_trajectories` | Rates nonnegative; converge tolerances and ensemble size |
| Trajectory estimate of an autonomous steady state | `trajectory_steady_state` | Converge settling time, within-path spacing, path integration, and independent-path count; residual is not a convergence proof |
| Confidence-controlled ensembles | `adaptive_quantum_trajectories`, `adaptive_weak_pi_quantum_trajectories`, `adaptive_diffusive_trajectories` | Sampling certificate only; separately converge the path integrator |
| Pure Schur pseudo-ket ensembles | `WeakPITrajectoryPlan`, `weak_pi_quantum_trajectories` | Auxiliary weak-PI states, not labeled-particle wavefunctions; fixed operators; converge fixed-step or event-driven controls |
| Weak-PI trajectory estimate of an autonomous steady state | `weak_pi_trajectory_steady_state` | Density outer products precede path averaging; converge burn-in, sampling window, selected path integrator, and independent-path count |
| Observable-only output | `solve_dynamics(...; observables=..., save_states=false)` | The state history is deliberately unavailable |
| State-free trajectory statistics | `quantum_trajectories(...; observables=..., save_states=false)` | Waiting-time output still scales with recorded jumps unless disabled |
| Two-time correlations and spectra | `CorrelationPlan`, `two_time_correlation`, `stationary_correlation_spectrum` | Autonomous QRT; converge RK4 or GMRES controls |
| Resolvents, adjoint evolution, and susceptibility | `ResponseWorkspace`, `resolvent_norm`, `adjoint_evolve`, `steady_state_susceptibility` | Matrix-free routes require an adjoint where applicable; iterative estimates retain residual diagnostics |
| Several PI ensembles or a finite ancilla | `CompositePIBasis`, `CompositeSuperoperator`, `CompositeTrajectoryPlan` | Cross maps and monitored gains are factorized; composite paths are density-valued |
| One shared cavity or explicit global pseudomode | `global_pseudomode_model` | The mode is one finite global factor coupled through a collective system operator; converge its cutoff |
| Identical systems with identical local pseudomodes | `pseudomode_supersite`, `pseudomode_model` | The complete system+local-modes tuple is one PI particle; converge every finite mode cutoff |
| Related steady states or spectra | `ParameterScanPlan`, `parameter_scan` | Serial continuation is path dependent; independent points may be threaded/distributed |
| Structured matrix-free linear families | `block_gmres`, `multishift_gmres`, `recycled_gmres`, `krylov_expv` | Inspect raw residuals/error estimates and reuse task-owned workspaces |
| Finite-memory bosonic bath | `HEOMBath`, `HEOMPlan`, `heom_evolve`, `heom_steady_state` | Prefer exact scaled ADOs when conditioning benefits; converge bath poles, hierarchy depth, and time/Krylov discretization separately |
| Shared-bath finite-memory pure-state ensemble | `HOPSBath`, `HOPSPlan`, `hops_average` | Coupling and noise must preserve PI on every path; average unnormalized root outer products and converge bath poles, hierarchy depth, time step, and sample count separately |
| Generalized qudit coherent-state Q | `QuditHusimiPlan`, `qudit_husimi_q` | Normalized Haar data on supplied points; no qudit Wigner convention inferred |
| Numerical refinement evidence | `convergence_study` and specialized wrappers | Final refinement agreement is distinct from inner solver convergence |
| Large-$N$ product prediction | `MeanFieldPlan`, `solve_meanfield` | Approximate after correlations develop |
| Repeated collective observable | `CollectiveObservablePlan` | Reuse one plan for many states |
| Trace a mode/ancilla inside every PI supersite | `LocalFactorTracePlan`, `LocalFactorTraceWorkspace` | Complete kept-factor PI output; exact-support sparse transforms have a setup memory guard |
| Repeated marginal or negativity | `ReductionPlan`, `ReductionWorkspace` | Setup can be large for qudits |
| Repeated one-factor composite trace | `CompositeReductionPlan` | Packed exact diagonal contraction; no full composite trace vector |

For product-state predictions at $N$ beyond a practical PI basis, the
mean-field layer evaluates

```math
\rho^{(p)}\approx\sigma^{\otimes p}
```

directly on one $d\times d$ density matrix. Its finite rule reproduces the
exact initial one-body derivative of a supported factorized state, but later
propagation discards generated correlations. It must not be described as
exact PI dynamics. See [Mean-field dynamics and
predictions](meanfield.md) for the finite and thermodynamic conventions.

## Conventions and reliability

The following conventions matter when comparing with analytical formulas or
manipulating coefficient data directly:

- Partitions are descending length-$d$ tuples padded with zeros and ordered
  in descending lexicographic order.
- GT patterns are ordered by their stored entry tuples; sector matrices are
  flattened in Julia column-major order.
- The paper labels local computational states $0,\ldots,d-1$, whereas Julia
  matrix indices are $1,\ldots,d$. The ordering of the supplied local
  matrices and state vectors must be consistent.
- Rates, Hamiltonian prefactors, and normalization factors are never changed
  implicitly. In particular, no Kac factor is inferred.
- Invalid states are rejected rather than normalized, symmetrized, or made
  positive by clipping eigenvalues.
- A complete dense spectrum and a selected Krylov spectrum answer different
  questions. Residuals, convergence flags, and spectral scope must accompany
  claims about slow modes or gaps.
- A finite product closure, a finite trajectory ensemble, and an
  under-resolved integrator are approximations even though the underlying PI
  representation is exact.

Remaining research-scale limitations are also structured rather than hidden:
large qudit irreps can make sparse Littlewood--Richardson factorization
expensive; a single very large Schur block still needs dense linear algebra
for some diagnostics; the Evans test can be inconclusive for unsupported
direct microscopic recouplings; and complete Floquet spectra or the dense
spectral-visualization source convenience still materialize the
PI-dimensional one-period map. Selected multiplier, periodic-state, and
stroboscopic calculations can instead use `FloquetMap` without that matrix.

Continue with [Architecture and efficient workflows](architecture.md) for
backend and concurrency details, [Mathematics and API](mathematics.md) for
analysis methods, [Published-model validation](published_validation.md) for
finite-size reference checks, and the [API reference](api_reference.md) for
individual signatures.
