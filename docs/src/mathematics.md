# Mathematics and API

For each partition `nu`, the stored matrix is `C_nu`; the physical block is
`C_nu/sqrt(f^nu)`. Flattening is by partition, then GT patterns, then column
major block order. Trace and multiplication implement equations (12) and (15).

The one-body geometry implements the CG products (B.10–B.11), the three-nu
dyadics (22), the local kernel (26–27), and collective blocks (31). Local and
collective dissipators implement equations (38) and (39).

## PI spectra

`pi_liouvillian_spectrum(L)` returns the complete spectrum of a static
Liouvillian in PI-coordinate space, sorted by decreasing real part by default.
It accepts a `PIModel`, sparse/dense matrix, or `MatrixFreeLiouvillian`;
`vectors=true` additionally returns sorted right eigenvectors. A complete
spectrum necessarily uses a dense PI-dimensional eigendecomposition, but it
never constructs the exponentially larger full Liouville representation.

`pi_liouvillian_gap(L)` computes the slowest nonstationary decay rate after
excluding every eigenvalue in a tolerance-controlled stationary cluster. It
therefore remains correct for degenerate steady-state manifolds, unlike simply
taking the second sorted eigenvalue. With `return_info=true` it reports the
controlling complex mode and oscillation frequency, stationary multiplicity,
spectral abscissa, stability flag, tolerance, and PI dimension. Unstable
generators raise by default; use `check_stability=false` for diagnostics.

Pass `symmetry=U` to exploit a verified unitary weak symmetry. The induced
Liouville action is diagonalized, equal charges are grouped, and the
Liouvillian spectrum is computed independently in each invariant block; the
global gap is then selected from their union. `return_info=true` adds
`symmetry_sectors`, containing each charge, block dimension, stationary
multiplicity, spectral abscissa, and sector gap. `symmetry=:auto` scans the
usual unitary candidates and uses the detected candidate with the largest
number of charge blocks. For an already assembled PI Liouvillian, also pass
`basis=basis`. Antiunitary covariance alone does not define complex-linear
charge eigenspaces and is therefore deliberately not used for this reduction.

`pi_density_spectrum(rho)` diagonalizes the physical Schur blocks and returns
`values`, exact `degeneracies`, `sectors`, and within-sector indices. Each
eigenvalue is stored once and its symmetric-group multiplicity is kept as an
exact integer, so the default output remains polynomial in `N` at fixed `d`.
Aliases are `pi_density_operator_spectrum` and `density_operator_spectrum`.
Set `expanded=true` only for small systems; `max_expanded_dimension` prevents
accidental exponential allocation. Pass the default compressed result to
`visualize_density_spectrum`; sector colours and exact degeneracy annotations
are presentation metadata and do not alter the eigenvalues.

## Evans uniqueness certificate

`evans_uniqueness(H,jumps)` implements the finite-dimensional Evans
commutant test by finding the joint nullspace of the vectorized commutators
with ``H``, every ``L_k``, and every ``L_k^\dagger``. A scalar commutant
certifies a unique stationary state for the trace-preserving GKSL semigroup.
The result reports the numerical commutant dimension and tolerance; use
`return_basis=true` to inspect its vectorized basis. This is an algebraic
certificate, distinct from estimating the zero-eigenvalue multiplicity of a
numerically assembled Liouvillian. See D. E. Evans, *Commun. Math. Phys.* 54,
293 (1977), DOI `10.1007/BF01614091`.

`evans_uniqueness(model)` avoids full `d^N` construction. It first applies
cheap exact certificates for one-particle, one-sector, locally irreducible,
and collective-only models. General built-in combinations of local,
collective, and Appendix-D p-body terms use the ket/bra-regrouped auxiliary
space with one local dimension `q=d^2`. On this space the positive operator

```math
K=\mathrm{ad}_H^\dagger\,\mathrm{ad}_H+
\sum_k\left(\mathrm{ad}_{L_k}^\dagger\,\mathrm{ad}_{L_k}+
\mathrm{ad}_{L_k^\dagger}^\dagger\,\mathrm{ad}_{L_k^\dagger}\right)
```

is permutation invariant. Its Schur blocks are constructed with the same
one- and p-body representation geometry as the dynamics, and their kernel
dimensions are weighted by the exact symmetric-group multiplicities. Thus
the reported dimension is the commutant dimension on the complete physical
Hilbert space, not merely in the PI operator subspace.

The auxiliary PI coordinate count is
`binomial(N+d^4-1,N)`, so this stronger test is polynomial at fixed `d` but can
still be expensive. `memory_budget` guards setup and returns
`unique=missing` before construction when its conservative estimate is too
large. Direct `PIOperator` terms and custom terms also return `missing` when
their microscopic ket/bra recoupling is unavailable. `return_basis=true`
returns kernel vectors compressed by auxiliary Schur sector and records each
sector multiplicity; it never expands those copies. Because `K` is a
normal-equation operator, the report distinguishes the requested
singular-value tolerance from the effective roundoff floor used for its
eigenvalue nullity.

`has_unique_steady_state_evans(model)` returns the resulting
`Bool`-or-`missing` field. The theorem requires a static GKSL generator, so
operator-valued time dependence, non-Hermitian Hamiltonian coefficients, and
negative or complex dissipative rates are rejected.

## Weak unitary and antiunitary symmetries

`check_liouvillian_symmetry(L,U;kind=:unitary,basis=basis)` tests the weak
covariance condition
``\mathcal L\mathcal U=\mathcal U\mathcal L`` with
``\mathcal U(\rho)=U\rho U^\dagger``. For `kind=:antiunitary`, it treats the
Hilbert-space antiunitary as ``UK`` in the chosen computational basis and
tests ``\mathcal L\mathcal U=\mathcal U\overline{\mathcal L}``, corresponding
to ``\mathcal U(\rho)=U\rho^*U^\dagger``. Reports include absolute and relative
residuals and the tolerance used. `is_liouvillian_symmetric` returns only the
Boolean result.

For PI inputs, a local `d×d` unitary is lifted as `U^⊗N` by exponentiating its
Hermitian generator inside each Schur block; neither a full `d^N` unitary nor
a full Liouville transformation is constructed. A `PIOperator` can specify
different unitary actions in the retained Schur sectors. For a full-space
Liouvillian, omit `basis` and pass its Hilbert-space unitary directly.

`usual_liouvillian_symmetries` checks named clock/phase and cyclic-shift
unitaries and complex conjugation. For qubits it additionally checks Pauli-X
and Pauli-Z parity and spin time reversal ``i\sigma_yK``. Custom named tuples,
dictionaries, or vectors of pairs can replace either candidate set. These are
weak superoperator symmetries; they do not assert that every Hamiltonian and
jump operator commutes individually with `U` (the stronger notion).

## Vectorization and permutation covariance

The package uses column-major vectorization. The exported constructors
`left_superoperator(A)`, `right_superoperator(A)`, and
`sandwich_superoperator(A,B)` implement, respectively,
``X\mapsto AX``, ``X\mapsto XA``, and ``X\mapsto AXB^\dagger`` through the
identities ``\mathrm{vec}(AXB^\dagger)=(\bar B\otimes A)
\mathrm{vec}(X)``. `commutator_superoperator` and
`dissipator_superoperator` build the corresponding Hamiltonian and Lindblad
pieces. Sparse PI Liouvillian assembly uses these same centralized identities.

For small full-space reference calculations, `is_pi_operator(A,N,d)` tests
``P_\pi A P_\pi^\dagger=A`` and `is_pi_superoperator(S,N,d)` tests covariance
of a Liouville matrix under ``X\mapsto P_\pi X P_\pi^\dagger``. It is
sufficient to check the `N-1` adjacent transpositions, so the implementation
permutes matrix indices directly and constructs neither permutation matrices
nor all `N!` group elements. These validation functions necessarily accept
objects of dimensions `d^N` and `d^(2N)`; they are not the production PI
simulation path. A `PIModel` returns `true` by construction.

## Prepared one-box Clebsch--Gordan coefficients

One-body contractions and Appendix-D paths repeatedly use the same real
one-box ``U(d)`` Clebsch--Gordan coefficients.  For several geometry setups on
one exact basis, prepare them once and pass the read-only cache explicitly:

```julia
coefficients = OneBoxCGCache(
    basis; max_depth=3, T=Float64, memory_budget=64 * 1024^2)
one_body = OneBodyGeometry(
    basis; T=Float64, coefficient_cache=coefficients)
three_body = PBodyGeometry(
    basis, 3; T=Float64, coefficient_cache=coefficients)

# The same cache can serve all compatible geometry families in one model.
prepared = compile(model; coefficient_cache=coefficients)
```

`max_depth=1` covers `OneBodyGeometry` when `N > 0` (the zero-particle case
uses depth zero); `PBodyGeometry(basis,p)` requires a cache prepared through
at least depth `p`. The cache is tied to the exact
`PIBasis` object and floating type, stores only the structurally admissible
one-box data within that depth, and checks its predicted storage against
`memory_budget` before allocating the coefficient tables. It may be shared by
sequential or concurrent read-only construction. Omitting `coefficient_cache`
preserves the
call-local path, which computes each transition once for that geometry and
retains no process-global mutable state.  That default is often preferable for
one small, one-off geometry because a standalone cache also has a setup cost;
the explicit cache pays off when its coefficients are reused.  For individual
coefficient queries, the same object can be passed as `cache=coefficients` to
`cgc` or `three_nu_symbol`.

Ordinary model compilation already uses this optimization automatically when
a model needs more than one one-box geometry family and its dimensions are
small: currently `d=2, N<=32`, `d=3, N<=8`, or `d=4, N<=5` (and the trivial
`d=1, N<=64` case).  These bounds also apply to a restricted basis with the
same `(N,d)`.  That automatic cache is bounded by 64 MiB, exists only during
lowering, and is released after the finished packed geometries have been
built.  A single geometry family keeps the lower-overhead call-local route.
Prepare and pass an explicit cache when several independent model compilations
should share the same coefficients or when a larger calculation has a
deliberately chosen memory budget.
The explicit `coefficient_cache` keyword is accepted by `LiouvillianPlan`,
`compile`, `liouvillian`, `steady_state`, and `compile_family`; each route
validates exact basis ownership, required removal depth, and scalar precision
before lowering any term.

This is deliberately not a table of generic Wigner ``3j`` or ``6j`` symbols.
The package's `three_nu_symbol` is a dyadic of two one-box ``U(d)``
coefficients, and Appendix-D isometries are products of the same one-box maps;
a Wigner-symbol table would therefore add SU(2)-specific storage without
accelerating the qudit geometry.  Qubit reduction and negativity instead
prepare the complete recoupling matrices they actually use inside
`ReductionPlan`, where their lifetime, precision, and memory cost remain
explicit.

## Appendix-D p-body processes

`PBodyGeometry(basis,p)` enumerates the allowed sequences of `p` removable
corners and caches the successive one-box CG isometries. These realize the
generalized partition triangle and generalized three-nu tensors of equations
(D.5)--(D.15), without constructing computational-space operators of size
`d^N`. Setup indexes each GT-pattern list once, caches every one-box edge
matrix shared by several removal paths, and propagates all centre patterns
through two preallocated buffers. It does not construct a pattern-amplitude
dictionary for every local word.

`pbody_collective_operator(basis,X,p)` represents
``\sum_{n_1<\cdots<n_p}X^{(n_1,\ldots,n_p)}``, while
`pbody_kernel_operator(basis,X,Y,p)` represents the local superoperator
``K_{X,Y}`` of equation (D.2). The term constructors are:

- `PBodyHamiltonian(H,p)` for the symmetric sum of `p`-particle Hamiltonians;
- `LocalPBodyJump(L,p)` for a sum of independent subset dissipators;
- `CollectivePBodyJump(L,p)` for the dissipator of the coherent subset sum.

The supplied `d^p × d^p` operator must be invariant under permutations of its
`p` tensor factors. Static sparse and preallocated matrix-free Liouvillians
support all three terms. At fixed `d` and `p`, path enumeration remains
polynomial in `N`; the unavoidable local contraction cost grows with `d^(2p)`.

## Tracing an internal factor of every supersite

`local_factor_trace` changes the local PI dimension while keeping all `N`
particles. For a local factorization
``\mathbb C^{d_1}\otimes\mathbb C^{d_2}``, it applies the same partial trace
inside every supersite. This is not the particle reduction performed by
`reduced_state`.

Let ``E_{ab}`` be the matrix-unit basis on the kept factor and let
``\mathbf n`` contain the occupation counts of its ``d_{\rm keep}^2``
letters. The normalized symmetric operator

```math
B_{\mathbf n}
=\frac{1}{\sqrt{M_{\mathbf n}}}
 \sum_{\text{distinct strings}}
 E_{r_1}\otimes\cdots\otimes E_{r_N},
\qquad
M_{\mathbf n}=\frac{N!}{\prod_r n_r!},
```

is one member of an orthonormal basis of the kept-factor PI operator space.
There are exactly
``\binom{N+d_{\rm keep}^2-1}{N}`` such occupations. If
``q_{\mathbf n}`` is its kept-factor Schur-coordinate vector and
``\ell_{\mathbf n}`` is the source-coordinate vector after the adjoint
insertion ``E_{ab}\mapsto E_{ab}\otimes I`` (or
``I\otimes E_{ab}``), then

```math
x_{\rm kept}
=Q L^\dagger x_{\rm source},
\qquad
Q=[q_{\mathbf n}],\quad L=[\ell_{\mathbf n}].
```

The columns are constructed by polarizing the same one-box Schur recurrence
used for tensor-power states. If ``P_{\lambda,\mathbf n}`` is the coefficient
of the corresponding matrix-unit monomial in the physical irrep block, its
stored equation-(7) block is

```math
C_{\lambda,\mathbf n}
=\sqrt{\frac{f^\lambda}{M_{\mathbf n}}}\,
 P_{\lambda,\mathbf n}.
```

Thus setup never enumerates ``d^N`` local words. A
`LocalFactorTracePlan` retains ``L`` and ``Q``; a
`LocalFactorTraceWorkspace` owns the one short occupation vector used by the
two matrix-vector contractions. The complete kept-factor output basis is
required even for a sector-restricted source because local tracing can
populate several output Young sectors.

## Entanglement negativity

`negativity(rho, k)` and `logarithmic_negativity(rho, k)` compute the exact
partial-transpose trace norm for general PI states. Fully symmetric states use
occupation-number branching into
`Sym^k(C^d) ⊗ Sym^(N-k)(C^d)`. General qubit states use multiplicity-free
SU(2) Clebsch--Gordan recoupling. General qudit states construct the matching
Littlewood--Richardson/subduction multiplicity spaces as nullspaces of U(d)
generator intertwining equations. All representations have polynomial size in
`N` at fixed `d`. LR multiplicities are counted exactly by lattice tableaux;
forbidden weights are removed before sparse simple-root assembly and SPQR
nullspace recovery. The factorization and retained dense intertwiner basis can
nevertheless become the practical bottleneck for larger `d` and partitions.
Large-spin qubit recoupling switches from an allocation-light factorial
formula to an exact-rational Racah recurrence before machine factorials can
overflow. Reduction and negativity contractions fuse exact Schur
multiplicities into the stored coefficient blocks, so a representable physical
answer is not lost merely because an intermediate per-copy block underflows.

`reduced_state(rho, k)` returns the `k`-particle marginal as another normalized
`PIState`. `reduced_purity(rho, k)` computes ``\mathrm{tr}(\rho_k^2)``, while
`reduced_purities(rho)` returns the values for every subsystem size. None of
these routines constructs the exponentially sized computational-basis
marginal.

`quantum_fisher_information(rho, G)` (or `qfi`) evaluates the exact mixed-state
spectral formula for a collective observable. Passing a local matrix constructs
``G=\sum_n G^{(n)}``; a `PIOperator` can be passed directly. Sector results are
weighted by their exact symmetric-group multiplicities, without constructing a
full-Hilbert-space matrix. The implementation diagonalizes the bounded
multiplicity-weighted blocks ``\sqrt{f^\nu}C_\nu`` rather than first forming the
possibly unrepresentable physical block ``C_\nu/\sqrt{f^\nu}``.

`quantum_fisher_information_matrix(rho, generators)` (alias `qfim`) extends
this to multiparameter unitary estimation. It diagonalizes every density block
once, shares collective geometry between local generators, and returns a real
symmetric matrix. Local matrices and existing `PIOperator` generators may be
mixed.

`collective_expectation(rho, X)`, `collective_variance(rho, X)`, and
`collective_moments(rho, X)` contract equation-(31) blocks directly. They avoid
assembling the flattened `collective_operator`; solution-wrapper methods return
the corresponding quantity along a trajectory.
For repeated observable, covariance-matrix, or scalar-QFI calls on the same basis, construct
`cache=OneBodyGeometry(rho.basis)` once and pass `cache=cache`; this avoids
rebuilding representation geometry without changing the contraction.

Run `julia --project=. benchmark/performance_audit.jl` for the global warmed
time/allocation report and its sparse-versus-matrix-free precision guards. The
BenchmarkTools workloads in `benchmark/benchmarks.jl` cover longer statistical
measurements and larger scaling studies.

## Information, metrology, and response

Sectorwise spectral functions provide `von_neumann_entropy`, `renyi_entropy`,
reduced entropies, mutual/conditional information, fidelity, Bures and trace
distances, and quantum relative entropy. Exact multiplicities are included.
Large multiplicities enter through exact binary scaling and logarithms, while
the ordinary representable case keeps the direct machine-arithmetic path.
Numerical-rank decisions are relative to each sector's own weighted spectrum,
so a small but physically relevant sector is not discarded by a tolerance set
by another sector.

Every density spectrum used by these functions is first checked for positivity
with the requested `atol` and `rtol`. Accepted nonpositive roundoff values may
then be treated as numerical zeros only inside a square root, logarithm, or
support test; neither the state nor any returned spectrum is clipped. Fidelity
uses this rule for its two square roots and corrects a final result to an
endpoint of `[0,1]` only inside a scale-aware arithmetic bound. A result farther
outside the interval raises, and the Bures distance does not mask it with an
unconditional clamp.

For `quantum_relative_entropy`, sigma's numerical support in a sector is
defined relative to that sector's weighted spectral scale (the floor is
`8n eps(T)` times that scale for an `n`-dimensional block). Rho's total weight
on the complementary projector is evaluated in one contraction. Positive
weight above the analogous numerical roundoff floor gives `Inf`; it is not
dropped through separate rho-eigenvalue or eigenvector-overlap cutoffs. The
same support rule is used by `relative_entropy_decomposition`.

## Symmetry-resolved information

`sector_resolved_entropy` returns the probability of every Schur sector and
the entropy of its normalized conditional irrep state. `entropy_decomposition`
splits the total entropy exactly into the Shannon entropy of the sector label,
conditional irrep entropy, and the entropy of the symmetric-group
multiplicity space. `relative_entropy_decomposition` provides the analogous
classical/intra-sector split for quantum relative entropy, using the same
sector-projector support test as the undecomposed quantity.

`sector_resolved_coherence` uses the stored GT basis and reports normalized
conditional relative-entropy coherence and its weighted contribution.
`symmetry_twirl` instead pinches within degenerate eigenspaces of a supplied
collective charge. The entropy increase is returned by
`relative_entropy_of_asymmetry` (`relative_entropy_of_symmetry`), while
`wigner_yanase_asymmetry` provides the skew-information measure. Local charges
are lowered to a `CollectiveObservablePlan`, so these routines can use finite
physical generator blocks even when storing the corresponding equation-(7)
`PIOperator` coefficients would overflow.

`sector_resolved_qfi` decomposes collective-generator QFI into conditional
sector QFIs. For general parameter tangents, `qfim_sector_decomposition`
separates classical Fisher information from changing sector probabilities and
the remaining intra-sector QFIM.

`charge_resolved_negativity` resolves negativity into eigenspaces of the
partial-transpose imbalance ``Q_A^T-Q_B``. It verifies the corresponding
charge symmetry rather than silently discarding off-block terms.
`number_resolved_negativity` uses the local charge `diag(0:d-1)`.

Collective covariance matrices support Kitagawa--Ueda and Wineland squeezing,
QFI entanglement-depth bounds, two-particle correlators, and normalized
second-order correlations. Multiparameter tools include SLD operators, their
commutator/compatibility matrix, mean Uhlmann curvature, QFIM from state
tangents, and classical Fisher information for PI POVMs.

Open-system response helpers expose decay modes and observable overlaps,
resolvent norms and grid pseudospectral abscissae, adjoint observable
evolution, integrated autocorrelation times, steady-state susceptibilities,
and SciML tangent problems. Dense spectral routines are intended for moderate
PI dimensions; they do not construct the full `d^N` Hilbert space.

`steady_state(model)` uses the exact equation-(7) trace functional in a
bordered, trace-constrained solve and checks stationarity and normalization.
For an explicitly supplied matrix use `steady_state(L; basis=basis)`, since an
unannotated matrix does not determine the physical trace convention. The SVD
method handles degenerate stationary manifolds and `return_info=true` reports
the residual, trace error, numerical nullity, and selected algorithm.

The `method` keyword selects the numerical algorithm:

- `:auto` tries the bordered direct solve and falls back to SVD;
- `:direct` uses only the trace-bordered sparse solve;
- `:svd` returns the minimum-norm trace-one member of a degenerate nullspace;
- `:eigen` uses the dense eigenvector nearest zero;
- `:shiftinvert` performs sparse inverse iteration near `shift`
  (`:shift_invert` and `:inverse_iteration` are aliases);
- `:krylov` (or `:gmres`) solves the trace-fixed stationary equation with
  restarted matrix-free GMRES.

Shift-invert accepts `shift`, `maxiter`, and `initial_state`. A small nonzero
shift is selected automatically if omitted. `atol` and `rtol` control the
stationarity test. `return_info=true` additionally reports the iteration
count, approximate eigenvalue, convergence flag, residual, and trace error.
Basic diagnostics do not perform an extra factorization. Request
`diagnostics=:nullity` explicitly when a dense SVD-based nullity estimate is
required.

## Prepared models and workspaces

`compile(model; backend=:auto)` builds fixed one- and p-body Schur data once
and returns a `CompiledPIModel`. The backend choice and conservative memory
estimate are available in `prepared.backend` and `prepared.estimates`.
Repeated and concurrent matrix-free calls should use caller-owned scratch:

```julia
prepared = compile(model; backend=:matrixfree)
work = LiouvillianWorkspace(prepared)
apply!(dest, prepared, source, t, parameters, work)
```

Use one workspace per task or thread. The compatibility `mul!` and `action!`
entry points are synchronized and safe, but serialize concurrent calls.
`apply_adjoint!` and batched matrix application use the same prepared kernels.

`CollectiveObservablePlan(basis, X)` stores collective Schur blocks for a
frequently sampled observable. Construct several plans from one
`OneBodyGeometry` and then pass the plan directly to collective expectation,
variance, QFI, or QFIM routines. `ReductionPlan(basis, k)` similarly reuses
qubit recouplers or qudit LR/subduction geometry through the `plan=` keyword
of reduced-state, purity, negativity, and partial-transpose routines.

`ReductionWorkspace(plan, rho)` adds caller-owned product-block, partial-trace,
partial-transpose, and reduced-sector scratch for repeated state scans. Pass it
through `workspace=` to `reduced_state`, `reduced_purity`, or `negativity`.
Use `mode=:reduction` to omit partial-transpose storage or
`mode=:negativity` to omit partial-trace and reduced-sector storage when a
balanced bipartition makes the general-purpose `mode=:both` workspace too
large. A mode-specific workspace rejects incompatible operations.
`reduced_state!(out, rho, plan, workspace)` also avoids allocating a new result.
Plans may be shared; use one mutable workspace per concurrent task.

Observable plans retain one Schur-sized matrix per sector. Qudit reduction
plans can be substantially larger because they retain all required LR
intertwiners. Setup uses exact tableau counts and weight-restricted sparse
simple-root constraints; SPQR can still dominate for large qudit irreps.
See [API tiers and prepared analysis](api_tiers.md) for usage and memory
tradeoffs.

## Periodic and Floquet dynamics

`time_evolve(L,rho,tspan)` propagates a density matrix after its Liouvillian
has been defined. Sparse/dense matrices and matrix-free Liouvillians share a
fixed-step fourth-order Runge--Kutta implementation. `evolve!` writes into a
supplied state or coordinate vector, and an `EvolutionWorkspace` reuses all
three full scratch arrays (stage state, derivative, and weighted accumulator)
across repeated calls. The classical RK4 formula and order are unchanged;
only its forward storage schedule differs. `time_evolution(L,rho,times)`
advances sequentially with one workspace. Step counts should be
convergence-tested for stiff generators; `dynamics_problem` remains available
for adaptive solvers.

## Quantum trajectories

`quantum_trajectory(model,rho0,times;dt=...)` implements a stochastic
quantum-jump unraveling entirely in PI coordinates. Collective and direct-PI
jumps use their usual gain maps. A local channel is unresolved over particle
labels and uses ``\sum_i L_i\rho L_i^\dagger``; consequently an individual PI
trajectory can be mixed even though its ensemble converges to the same master
equation. Appendix-D local and collective channels follow the same rule.

With `algorithm=:fixed`, the normalized no-jump equation is integrated with
preallocated RK4 stages and steps are shortened automatically to enforce
`max_jump_probability`. With `algorithm=:event` (also `:adaptive` or
`:event_driven`), an
embedded Dormand--Prince 5(4) solve advances the normalized state and its
integrated hazard together. Bisection locates each continuous hazard root, so
jump times are not tied to sampling-grid endpoints. `abstol`, `reltol`,
`dtmin`, `dtmax`, and `event_time_tolerance` control this adaptive path.
`TrajectoryPlan` compiles the immutable representation-theoretic geometry
once. `TrajectoryWorkspace` owns the mutable integrator scratch for one path,
and `TrajectoryBatchWorkspace` retains one such workspace and RNG per worker
for reuse across ensembles. Fixed-step RK4 uses three full-vector registers;
`mode=:fixed` omits the otherwise unused `k3` and `k4` registers plus six
adaptive-only vectors per worker. The default `mode=:full` supports both
algorithms and is required for continuous event-time integration. An empty
model accepts `T=Float32` (or another concrete real floating type) because it
has no term from which to infer that precision. During no-jump propagation,
the rate-weighted ``K^\dagger K`` Schur blocks are first combined into one
effective loss operator per sector. Individual channel intensities and a full
gain state are formed only when an actual jump must be selected. Accepted
Dormand--Prince stages supply a four-scalar quartic hazard interpolant, so
bisection locates the event without repeating full integration trials.
`quantum_trajectories` creates reproducible independent realizations and
`trajectory_average` returns the sampled ensemble density matrices.
For an autonomous fixed-operator unraveling,
`trajectory_steady_state(source, rho0; trajectories, settling_time, dt, ...)`
streams an approximate stationary density operator without retaining path
histories or jump records. If several post-settling samples are requested, it
first forms the time average on each path and then averages those independent
path means. Consequently correlated samples separated by `sampling_interval`
do not artificially increase the reported sample count. At least two paths
are required. The Hilbert--Schmidt sample spread and standard error reported
by `TrajectorySteadyStateResult` are computed across these independent path
means in the orthonormal PI coefficient coordinates. Optional Hermitian
observables use the same path-level sampling unit for their variances,
standard errors, and normal confidence intervals.

The returned Liouvillian residual, relative residual, and trace error diagnose
the averaged density operator but do not certify convergence or uniqueness.
Research use must independently increase the settling time, vary the
within-path spacing and sample count, converge fixed-step or event-driven
integration controls, and increase the number of trajectories. In a
non-unique stationary manifold the ensemble mean may depend on the initial
state. For a fixed Lindbladian its ensemble-mean evolution is independent of
the valid unraveling; the conditional-path distribution, Monte Carlo
variance, and finite-sample realization can depend on that unraveling. Driven
sources are rejected: use Floquet analysis for periodic dynamics or `freeze`
only when the instantaneous autonomous generator is the intended question.

The weak-PI pseudo-ket estimator follows the same path-level reduction, but
the nonlinear density conversion is performed first. For path ``r`` and its
post-settling samples,

```math
\overline C^{(r)}_\nu
=\frac{1}{K}\sum_k
\frac{|\psi^{(r)}_\nu(t_k)\rangle
      \langle\psi^{(r)}_\nu(t_k)|}{\sqrt{f^\nu}},
\qquad
\widehat C=\frac{1}{M}\sum_r\overline C^{(r)}.
```

No cross-sector outer product is physical, and amplitudes must not be averaged
before this map. `weak_pi_trajectory_steady_state` consequently returns a
mixed `PIState`. Its Hilbert--Schmidt uncertainty uses ``\|C\|_2`` in the
orthonormal PI coefficient basis and treats only the ``M`` path means as
independent samples.
Stochastic trajectories require finite, nonnegative real rates representable
in the prepared precision; negative time-local rates remain supported by
deterministic evolution but do not define this jump process. Time grids,
fixed-step `dt`, and explicitly supplied adaptive controls must likewise be
representable without narrowing. Defaults and every RK stage are formed in the
prepared real precision. Fixed-step `dt`, adaptive tolerances, and the number
of trajectories must be convergence-tested for research results.
Set `threaded=true` to use dynamically scheduled, task-owned workspaces. The
scheduler claims small chunks of paths to amortize atomic operations without
turning a long realization into a static load imbalance.
Random streams are seeded by trajectory index, so a fixed seed produces the
same ordered paths in serial and threaded execution. For repeated ensembles,
construct one `TrajectoryPlan` and `TrajectoryBatchWorkspace` and pass the
latter as `workspace=`; plans may be shared, whereas batch workspaces may only
be reused sequentially. Outer trajectory threading generally performs best
when BLAS is not itself using several threads, but the package never changes
the process-wide BLAS configuration.

The returned trajectories intentionally own independent time grids, jump
records, and saved PI states. Their unavoidable output payload therefore
scales as ``O(n_{\rm traj} n_{\rm save} n_{\rm PI})`` even though plan and
integrator storage are shared or reused. For large PI dimensions, save only
the times required for the analysis; retaining a dense time history can
dominate both the batch workspace and the simulation kernel's temporary
memory.

`jump_statistics` reports total and channel-resolved counts, unbiased count
variances, empirical rates, Fano factors, no-jump probability, and pooled
waiting times. `trajectory_observable_statistics` accepts named Hermitian
local matrices or PI operators and returns time-resolved Monte Carlo means,
unbiased variances, standard errors, and configurable normal confidence
intervals. `trajectory_statistics` combines both reports with the averaged PI
density matrices. Observable operators are assembled once and reused across
all realizations and sampling times.

Time-dependent scalar coefficients may be supplied as `rate=(t,p)->...` on
any fixed one-body, collective, direct-PI, or Appendix-D term. The matrix-free
Liouvillian caches its operator geometry and sector workspaces once; evaluating
the coefficient does not assemble an instantaneous sparse superoperator.
`InPlaceTimeOperator(prototype, update!)` likewise prepares every built-in
operator-valued term, including local, collective, direct-PI, and Appendix-D
p-body processes. Evaluated operators, Schur blocks, quadratic jump data, and
local p-body path contractions live in `LiouvillianWorkspace`; a raw function
remains the allocating compatibility path.

`floquet_propagator(model,T)` integrates the one-period PI map using a
preallocated fourth-order Runge--Kutta workspace. `floquet_multipliers`,
`floquet_exponents`, and `floquet_gap` analyze this map.
`floquet_steady_state` solves for its trace-normalized fixed point, while
`stroboscopic_evolution` and `floquet_evolve` apply the map for integer numbers
of periods. Propagator buffers derive their precision from the Liouvillian,
period, and time origin: a completely `Float32` problem stays `ComplexF32`.
Materialized and compatible custom operators can promote for wider time
inputs. A compiled matrix-free PI plan owns fixed-precision application
scratch, so a wider period or origin is rejected before allocating the
propagator; compile the model at `Float64` when `Float64` integration is
required. An integer time must be exactly representable in the selected
working precision. `steps` controls the fixed-step convergence and should be
checked for demanding or stiff protocols.

`floquet_gap(F, T; atol=...)` removes a fixed multiplier only when an
eigenvalue lies within `atol` of one. It raises when the supplied map has no
such fixed point instead of silently discarding an unrelated closest mode,
and also raises when a remaining multiplier is outside the unit disk by more
than `atol`. The returned decay rate follows the multiplier and period
precision; a one-dimensional map returns a precision-matched `NaN` because it
has no subleading mode.
