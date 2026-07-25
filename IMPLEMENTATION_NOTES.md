# Implementation notes

## Conventions

Partitions are descending, zero-padded tuples. Sectors and patterns are
lexicographically descending and ascending respectively; matrices flatten in
Julia column-major order. Coefficient blocks are equation-(7) Hilbert–Schmidt
coordinates. Physical Schur blocks equal `C/sqrt(f^nu)`.

The implementation uses the hook formula, Weyl formula, recursive interlacing
GT rows, equations (B.10)–(B.11) with exact rational products, equations
(22), (27), and (31), and equations (37)–(39) for one-body generators.

## Numerical methods and scaling

Representation geometry is cached independently of local matrices. Sparse
assembly applies the one-box sector selection rule. The basis dimension is
`binomial(N+d^2-1,N)`; no production routine constructs a `d^N` object.

Public `exact_binomial` and `exact_multinomial` return `BigInt` values;
sequential exact binomials evaluate a multinomial without a factorial
intermediate larger than the result.  Numerical kernels do not convert the
large integer factors independently.  They retain exact `Rational{BigInt}`
ratios through cancellation, and square-root ratios are power-of-two scaled
to a bounded interval before conversion, followed by `ldexp`.  This permits,
for example, a representable `sqrt(f^nu)` even when `f^nu` itself exceeds
`floatmax(T)`. A nonzero standalone combinatorial scale that overflows or
underflows the chosen concrete floating type raises instead of becoming
`Inf`, zero, or `NaN`; the caller must then use a wider scalar type. Final
state entries retain ordinary rounding semantics in their declared output
type.

One-box geometry uses the exact branching weight

`b(lambda -> mu) = |lambda| * f^mu / f^lambda`

computed from shifted partition rows in `O(d)`, without constructing either
hook dimension.  Its local-map prefactor is
`sqrt(b(lambda -> mu) * b(nu -> mu))`.  Along an Appendix-D removal path,

`a(path) = prod(b(step)) / p! = binomial(N,p) * f^center / f^endpoint`,

so collective path weights use `a(path)` and gain-pair weights use
`sqrt(a(left_path) * a(right_path))`.  The exact path data are prepared once;
ordinary hot dynamic application uses converted working-precision scales,
while exceptional large factors retain a bounded binary mantissa/exponent.
Pure product-state constructors use normalized conditional-binomial
recurrences.  Tail norms are formed with `hypot`; each one-dimensional
binomial amplitude is seeded at its mode, propagated outward by adjacent
ratios, normalized, and then combined with the component phases.  Float16 and
Float32 inputs use Float64 recurrence scratch, while Float64 and BigFloat
retain their working type.  This avoids `Inf*0` from a separately converted
multinomial coefficient and preserves exact structural zeros without using a
global logarithmic reconstruction. Their normalization check is evaluated in
the recurrence precision and propagated through all `N` tensor factors, so a
low-precision one-site norm that rounds to one cannot produce a badly
subnormalized large product state. Mixed IID construction applies the same
tensor-trace check and validates the represented trace without the former
tenfold low-precision tolerance inflation.

Trace and sector-population contractions fuse `sqrt(f^nu)` with the stored
coefficient trace when the multiplicity scale itself is outside range. The
ordinary direct multiply remains the fast path. Physical blocks and purities
whose nonzero per-copy values are genuinely below the requested scalar range
raise with wider-type guidance instead of returning zero. Maximally mixed
states use the exact retained Hilbert dimension and never form `d^N` in the
destination floating type.

`PIBasis`, GT-pattern containers, `PIState`, and `PIOperator` carry concrete
representation and scalar parameters. A `PIModel` stores its heterogeneous
terms as an immutable tuple. Custom terms use explicit dispatch hooks and
lower by delegating to an equivalent built-in term; reflective field-name
inspection and unknown-term fallbacks are deliberately rejected.

The generator pipeline separates immutable prepared data from mutable scratch.
`compile` produces a `CompiledPIModel` containing one `LiouvillianPlan`, a
selected sparse or matrix-free adapter, and transparent memory estimates.
Sparse assembly, scalar/batched application, and adjoint application all use
the same lowered term data. Caller-owned `LiouvillianWorkspace` objects make
the hot path preallocated and permit one-workspace-per-task concurrency.

The population-only pipeline lowers the same static kernels onto the physical
Schur-diagonal coordinates
`p_(nu,W)=sqrt(f^nu)(C_nu)_(W,W)`. Its default invariance test is exact at the
working scalar precision (`atol=rtol=0`); finite but nonzero tolerances are an
explicit approximate projection and are reported as `:within_tolerance`.
Autonomous kernels are accumulated sequentially before testing, preserving
exact cancellations without retaining every full-coordinate raw map at once.
Scalar-driven kernels are certified individually. Nonfinite entries, rates,
and tolerances throw. The reverse diagonal lookup is a population-sized
dictionary used only during compilation, so `PopulationPlan` retains no
`length(basis)` lookup. Its sparse reduced matrix, basis, and scalar rate
schedules are immutable; `PopulationWorkspace` owns three RK vectors plus one
independent kernel-application buffer. Its legacy `k3`/`k4` fields are empty
compatibility placeholders in newly constructed workspaces. High-level evolution promotes from the saved time grid, while the
in-place Float32 path rejects wider floating times.

The reduced compiler still uses a standalone `sqrt(f^nu)` coordinate map;
model-derived routes also inherit the coefficient-space trace-vector range
requirement from `LiouvillianPlan`. It raises when a required factor does not
fit the working real type. The separate population extraction/reconstruction
paths deliberately retain their prepared exact-scale products, so a finite
final entry need not pay this stricter plan-construction bound.

The standard qubit convenience layer uses local order
`(|g>,|e>)=(|-1/2>,|+1/2>)`. `qubit_ensemble_model` maps the six PIQS rates to
standard package dissipators of local/collective `j_-`, `j_z=sigma_z/2`, and
`j_+`. Spin and state constructors infer non-narrowing precision, validate
Dicke half-integers with overflow-safe arithmetic, and use checked exact
multiplicity conversion. Generic Schur block constructors similarly copy and
scale physical/coefficient blocks with optional non-narrowing output `T`,
while `sector_metadata` keeps exact multiplicity and retained Hilbert
dimensions.

The preferred high-level commands are `solve_dynamics`, `stationary_state`,
and `liouvillian_spectrum`; typed result objects keep PI basis metadata and
diagnostics attached. `recommend_solver` exposes its conservative memory and
dimension heuristic rather than hiding backend selection. Fixed-size isbits
arrays keep exact inline byte formulas; heap-backed BigFloat arrays use a
conservative bound at an explicit assumed precision, including scalar-bearing
one-body geometry tuples. GMRES counts real work in the real component type.
Model and compiled-model recommendations omit
one-body geometry when the term lowering does not use it, while provenance-
free inputs report a conservative geometry assumption. Exact Integer or
Rational Hamiltonian `rate/hbar` quotients are cancelled before floating
conversion; ordinary floating values keep the direct division path. The raw
coordinate and method-specific solver functions remain available for research
control.

Clustered selected spectra have an explicitly named thick-restarted block
Arnoldi route. `BlockArnoldiWorkspace` caches the search basis and its images,
deflates dependent directions, and reports only freshly recomputed full-space
residuals; it is not presented as block IRAM. High-level resource estimates
include the requested `block_size`. Period maps expose a fixed-capacity
`FloquetBatchWorkspace`: its forward mode evaluates RK4 blocks with three
scratch matrices beyond the destination, while full mode retains the reverse
stages needed for the exact discrete adjoint. Neither mode grows capacity in a
hot application; prepared Liouvillian sector-batch buffers are pre-grown by
the Floquet batch constructor. Sparse compiled sources skip that matrix-free
scratch, supplied-workspace guards count any later source-batch growth, and
the low-storage destination must use the map scalar type.

The mean-field pipeline is independent of Schur geometry. `MeanFieldPlan`
lowers the same physical term constructors to the finite product closure
`Tr_{2:N} L[sigma^otimes N]`; `MeanFieldWorkspace` owns all tensor-contraction
and RK4 scratch. Its largest matrices have dimensions `d^p` by `d^p` for the
largest supported body order and do not depend on the PI dimension. The
direct `(N,d,terms)` constructor does not construct a `PIBasis`. Finite mode
uses exact subset combinatorics. Thermodynamic mode retains
`N^(p-1)/(p-1)!` for Hamiltonian/local p-body terms. Collective p-body jumps
use exact finite ordered-subset overlap counts and retain the leading disjoint
class `N^(2p-1)/((p-1)!p!)` in thermodynamic mode. Their effective operators
are contracted into at most `d^p` by `d^p` work arrays; no two-subset union
operator is formed. The plan never infers Kac normalization from a rate.
Direct Schur terms and operator-valued drives remain rejected because their
required microscopic closure data are not represented unambiguously.

Schur-block diagnostics separate numerical extraction from presentation.
`schur_block_structure` returns a `SchurBlockStructure` containing sector
labels, irrep and vectorized-coordinate dimensions, raw block weights, an
absolute-threshold activity mask, and extraction metadata.
`visualize_schur_blocks` creates a `SchurBlockVisualization`, while
`save_schur_block_visualization` uses its dependency-free SVG renderer.
Thresholding and rendering normalization never modify the measured weights or
the source object.

SVG sector axes show Young-diagram partition shapes by default. State and
operator views need one top-axis shape per diagonal block; superoperator views
show shapes on both the output-row and input-column axes. The tooltip reports
the padded partition, U(d) irrep dimension, and exact symmetric-group
multiplicity `f^nu`, and states that the PI block sums over standard-tableau
labels rather than selecting one filling. Diagram cost is bounded: shapes up
to 64 boxes use individual cells, while larger partitions use at most 64
normalized row bands. `show_young_diagrams=false` retains the compact text-only
layout.

For a state or PI operator, tiles are diagonal in the Schur label. Physical
weights are measured from `C_nu/sqrt(f^nu)` and are the default; coefficient
weights are measured directly from stored `C_nu`. The state-only population
metric is `sqrt(f^nu)*tr(C_nu)` and validates reality and nonnegativity. A
superoperator is instead partitioned in orthonormal PI coefficient
coordinates, with output sectors on rows and input sectors on columns. Dense
and sparse inputs are scanned directly. Exact matrix-free weights require
`n_PI=length(basis)` applications, one for each input coordinate, using one
reusable source-appropriate workspace (`LiouvillianWorkspace` or
`SymmetryProjectorWorkspace`); this is deliberately a setup diagnostic, not a
time-stepping kernel. Driven inputs require an explicit evaluation time.
Arbitrary operator-valued fallback terms are frozen and lowered once at that
time before probing, avoiding one instantaneous sparse reconstruction per
coordinate. Frobenius accumulation uses stable scaled norms across dense,
sparse, and matrix-free paths. Trace norms use Julia's SVD-supported scalar
types and fail explicitly for unsupported types rather than narrowing
precision. Nonzero thresholds that underflow in the weight type are rejected.

Density-spectrum visualization consumes the existing compressed
`pi_density_spectrum` result directly. It plots one raw physical-block
eigenvalue per compressed rank, retains exact Schur-sector degeneracies and
labels, and never constructs the expanded Hilbert-space list. Presentation
tolerances only mark negative values; they do not clip or repair them.

Complex spectral visualization follows the same two-stage design.
`liouvillian_spectrum_data` and `floquet_spectrum_data` construct reusable
`ComplexSpectrum` objects; `visualize_spectrum` and
`save_spectrum_visualization` never recompute modes or a one-period map. Raw
order, degeneracies, convergence flags, Ritz residuals, and complete/partial
scope are retained. Liouvillian plots use the imaginary axis as their
stability boundary, Floquet multiplier plots use an equal-aspect unit circle,
and multiplier-derived exponent plots use the principal logarithm with
quasifrequency boundaries at `±π/T`. Classification changes only marker
metadata, never eigenvalues.
Explicit view limits report hidden-point counts rather than deleting data.

Matrix-free Liouvillian source convenience delegates to the existing selected
mode solvers without requesting eigenvectors. By contrast, a complete Floquet
spectrum remains a dense operation at the PI coordinate dimension because
`floquet_propagator` propagates every column and the multiplier calculation
diagonalizes the resulting map. Reuse a converged propagator or multiplier
vector for repeated views and exponent conversion.

Computed Liouvillian sources keep numerical solver tolerances (`atol`,
`rtol`) distinct from presentation tolerances (`classification_atol`,
`classification_rtol`). Precomputed result residuals are copied without
narrowing them to the eigenvalue scalar type.

Only multiplier-to-exponent conversion is tagged
`metadata.branch=:principal`; directly supplied exponents are kept verbatim.
Converted Floquet residuals are not rescaled and record their originating
representation in `metadata.residual_representation`.

Spin phase-space analysis follows the same numerical-data/presentation split.
For every qubit sector, the multiplicity-weighted block
`rho_bar_j=sqrt(f^nu)C_nu` has the physical sector population as its trace.
The Husimi density uses the spin coherent-state resolution of identity, while
the Agarwal Wigner density uses orthonormal Condon--Shortley polarization
tensors generated by diagonal recurrences and recursively evaluated spherical
harmonics. Both integrate sectorwise to that population; the aggregate is a
marginal over the discrete total-spin label. Q coherent amplitudes start at
the binomial mode, avoiding large binomial conversion. Unresolved calculation
reuses one angular grid, while resolved calculation intentionally retains one
grid per selected sector. The equirectangular SVG accepts only precomputed
data, groups cells into a bounded palette-sized path collection, and leaves
negative Wigner values and all raw samples unchanged.

## Streaming output, quantum regression, and composite spaces

Fixed-step deterministic streaming propagates one mutable PI coordinate
vector and contracts prepared observables at requested output times. The
trajectory analogue gives every worker one path workspace, one
`n_observable` by `n_time` buffer, and online Welford accumulators. Setting
`save_states=false` therefore avoids constructing sampled `PIState` histories;
the optional pooled waiting-time vector remains proportional to the number of
recorded inter-jump intervals. Index-derived seeds are unchanged, so streaming
changes retained output rather than the stochastic dynamics.

Prepared observables and deterministic output buffers are concrete tuples;
sampling does not index an abstract `Pair` vector or a heterogeneous public
dictionary in the hot loop. The public dictionary is assembled once after
propagation. Result types use exact union parameters for optional state or
trajectory histories, preserving the inferred legacy return and concrete
history element types. State-free trajectory output requires at least one
observable; jump summaries can be retained or disabled alongside it.

Quantum regression stores the physical left/right insertion blocks once. For
the public convention

`C_AB(tau) = tr(A * exp(L*tau) * (B*rho*R))`,

the readout vector contains the coefficients of `A'`, because the package's
ordinary coefficient dot product is `tr(A'X)`. Time-domain propagation reuses
one `EvolutionWorkspace`; its warmed explicit-workspace RK4 path allocates no
history or integration scratch. The stationary connected spectrum solves
`(im*omega*I-L)x = B*rho*R-rho*tr(B*rho*R)` with restarted GMRES; a rank-one
trace term removes the stationary null direction without changing its
trace-zero solution. The disconnected stationary component is a Dirac delta
and is rejected as an ordinary function. The separate radix-two FFT path uses
trapezoidal endpoints and represents a finite observation window.

A `CompositePIBasis` is the tensor product of several PI operator spaces and
small full matrix-unit spaces. Factor one is fastest, hence factorized data and
maps are ordered `kron(last,...,first)`. A PI factor keeps its equation-(7)
coordinates and a `FiniteOperatorBasis(m)` contributes exactly `m^2`
coordinates. Composite trace contracts Cartesian products of factor diagonal
coordinates and multiplies their exact Schur multiplicities only after the
joint block trace is formed. The tensor-mode application copies one strided
factor fibre into caller-owned scratch, applies its local map, scatters it
back, and alternates two full composite buffers across active factors. Thus a
sum of Kronecker-product maps is applied without retaining the global
Kronecker matrix. Compiled PI factor actions keep separate nested
`LiouvillianWorkspace`s. Recursive tuple traversal specializes heterogeneous
factor maps and terms, so a warmed explicit-workspace application remains
allocation-free.

Composite stochastic plans add explicit tensor-product monitored channels to
a trace-preserving background. Each channel prepares unit-rate sandwich,
left-`Q`, and right-`Q` factor maps. The conditional RHS evaluates its scalar
rate once, takes the physical intensity from the already-applied left-`Q`
term, adds both losses, and restores normalized trace with the total
intensity. A selected jump applies only its unscaled sandwich map because the
scalar rate cancels during normalization. Prepared joint-sector diagonal
lists and exact multiplicity scales contract traces without a standalone
composite trace vector. All channels share one pair of full tensor buffers per
task; only factor-fibre workspaces scale with channel count. Per-channel
hazards use the conditional RK4 stages, and a step that violates the jump-
probability cap is retried before its state update is committed. Fixed and
driven rate values share the deterministic composite coefficient checker, so
the stochastic and unconditional generators reject the same narrowing. The
layer does not infer an unraveling from arbitrary superoperator terms or
silently extend single-ensemble reductions to a composite state.

## Current limitations

The recursive dense Schur transform is intentionally confined to
`test/test_dense_schur_reference.jl`, where small `d^N` matrices provide an
independent oracle for CG unitarity, product-state blocks, and collective
operators. Production state construction uses the sector-only Schur-functor
recurrence and never forms this transform.

`iid_state` supports singular local density matrices directly through that
sector recurrence, without matrix logarithms or spectral regularization.

Population restriction is exact by default, but genuinely operator-valued
time schedules cannot be certified globally and must be frozen at an explicit
time. Arbitrary nonzero certification tolerances intentionally define an
approximate projection and must not be presented as an exact invariant
subspace. The reduced generator's sparsity is model dependent even though the
coordinate vector and compile-only lookup have population dimension.
Spin Husimi/Wigner transforms are currently qubit-only; a qudit phase space
would require a separately specified SU(d) coherent-state and kernel
convention rather than reusing the spin sphere silently.

Static matrix-free application uses immutable prepared term kernels and an
explicit `LiouvillianWorkspace`, without assembling the global Liouvillian.
Scalar, batched, and adjoint `apply!` calls reuse caller-owned scratch. The
compatibility `mul!` and `action!` entry points are synchronized for safety but
serialize concurrent calls; parallel work should use one workspace per task.
Time-dependent scalar rates on fixed operators use the same preallocated
kernels. `InPlaceTimeOperator(prototype, update!)` adds the same explicit-
workspace ownership for every built-in local, collective, direct, and
Appendix-D Hamiltonian or jump operator. Each application evaluates the
operator once, fills dynamic Schur blocks and local-gain data in task-local
buffers, and supports forward, adjoint, and prepare-once batched application.
Local p-body jumps retain only task-local rectangular path-pair contractions:
their gain is applied blockwise as `C*X*C'` (and `C'*X*C` for the adjoint)
with one reusable multiplication buffer. They do not retain an
`O(length(basis)^2)` PI-coordinate table. Plain operator-valued functions
retain the allocating instantaneous-sparse compatibility fallback.

Mean-field fixed-point relaxation is basin dependent and restricted to
autonomous plans. It reports convergence and residuals when requested and
throws instead of returning an unconverged state through the convenience
path. The Jacobian acts on the real traceless-Hermitian tangent space of size
`d^2-1`; it is a local stability diagnostic, not a global fixed-point or
limit-cycle search.

Floquet maps use preallocated RK4 stage matrices and sequential matrix-free
column application. Multipliers, principal-branch exponents, decay gaps,
trace-constrained periodic steady states, and stroboscopic trajectories are
available for moderate PI dimensions.
Direct density-matrix evolution from an assembled or matrix-free Liouvillian
uses reusable vector-valued RK4 workspaces. Sampled trajectories advance
sequentially, so neither stage arrays nor earlier time intervals are rebuilt.
PI quantum trajectories reuse the same static sector kernels and add
channel-resolved gain application. Local particle labels are deliberately
unresolved, yielding mixed PI conditional states while preserving the target
master equation at ensemble level. The fixed RK4 algorithm controls grid-based
jump timing with an explicit maximum jump probability. The adaptive event
algorithm instead integrates the normalized conditional state and hazard with
Dormand--Prince 5(4), then locates continuous jump times by a hazard-root
solve. Fixed steps or adaptive tolerances must be convergence-tested.
Trajectory statistics use online Welford accumulation for count and observable
variances. Reports include channel-resolved rates and Fano factors, pooled
waiting times, no-jump fractions, standard errors, and normal confidence
intervals; local observables are assembled into PI operators only once.

The separate weak-PI trajectory surface propagates normalized pseudo-kets in
`directsum_nu U_nu`. A sector slice represents the multiplicity-weighted
rank-one block `psi_nu*psi_nu'`, rather than a labeled-particle pure state.
Fixed collective/direct gains are split by source sector. Fixed one-body local
gains reuse the one-box geometry factorization: every common child partition
gives a rectangular Kraus map from one Schur irrep to another, with the
coefficient-space strength converted by `sqrt(f_output/f_input)`. Exact Schur
multiplicities are combined before floating conversion, and plan construction
checks `sum K'*K` against the prepared `Q` block for every channel/source
sector. This works for qubits and qudits without a dense Choi problem or a
`d^N` object.

`WeakPITrajectoryWorkspace` owns three-register fixed-step RK4 scratch,
channel/branch intensities, and selected-branch scratch. Full mode adds `k3`,
`k4`, and six Dormand--Prince/event-root vectors, while fixed mode omits them.
Density and weak-PI no-jump stages combine all rate-weighted `K'K` blocks
before one sector action; individual intensities are evaluated only for a
selected event. The adaptive path evolves the normalized pseudo-ket and
hazard with common 5(4) stages. Its quartic dense extension retains four
scalar hazard coefficients, bisects without new RHS evaluations, reconstructs
the root state once from the retained stages, and samples the same prepared
Kraus branches. Batch
workspaces retain one instance and RNG per task with trajectory-index-derived
seeds. Saved jump records carry
source, target, and one-box child partitions. Ensemble averages convert outer
products to PI coefficient blocks only during reduction. The backend rejects
mixed initial blocks, nonunit pseudo-kets, operator-valued schedules, local
p-body jumps, narrowing inputs, and invalid rates. It provides fixed-step
maximum-jump-probability and continuous-hazard adaptive integrators. A
state-free confidence-controlled ensemble contracts observables directly from
pseudo-kets and reports separately seeded trajectories as its effective
independent sample count. The history-free stationary estimator optionally
forms complete density batch means; their error assumes sufficiently long,
approximately independent batches and explicitly leaves burn-in and
finite-window bias uncontrolled. A different Kraus rotation preserves master
dynamics but may change path-level statistics.

The performance audit covers basis and generator construction, sparse and
matrix-free application, preallocated PI and mean-field evolution, collective
moments and covariances, QFI, entropy, and qudit reduced states. One-body
geometry is shared within compound observable calls. `CollectiveObservablePlan` retains
prepared Schur blocks for repeated expectations, variances, covariance, QFI,
and QFIM calls. Cached `one_body_rdm` contracts every local matrix unit in one
geometry traversal instead of rebuilding `d^2` collective operators.

Steady states use the exact PI trace vector and a bordered constrained solve,
with an SVD fallback for degenerate stationary spaces. Matrix-free restarted
GMRES, largest-real Arnoldi, thick-restarted harmonic extraction, Schur-sector
preconditioning, weak-symmetry projection, exact-shift implicit-QR restarting,
and preconditioned Jacobi--Davidson are implemented. Jacobi--Davidson hard
locks converged Schur directions and validates reconstructed right vectors
with full residuals. `pi_liouvillian_spectrum` exposes them as `method=:iram`
and `method=:jd`; only largest-real IRAM is accepted by the global gap API.
Positivity diagnostics use exact sector eigenspectra for small
LAPACK-supported blocks and shifted Cholesky certificates for large blocks or
generic scalar types. Failed Cholesky blocks receive a targeted eigmin or
pivoted-PSD check; accepted large states therefore avoid complete sector
diagonalization. A single very large Schur block still incurs dense cubic
factorization and is not covered by a genuinely matrix-free PSD oracle.

Ordinary, harmonic, and implicit-QR Arnoldi share a reusable
`ArnoldiWorkspace`. Harmonic restart reports retain per-cycle residual history
and prioritize converged modes in the thick subspace. Weak-symmetry projectors contain immutable Schur
data and a synchronized compatibility workspace; explicit
`SymmetryProjectorWorkspace` objects avoid serialization in parallel loops.
Schur preconditioners report setup applications, factor storage, apply cost,
and an estimated minimum reuse count through `preconditioner_cost`. The
experimental `JacobiDavidsonWorkspace` combines Arnoldi subspace storage and a
restarted-GMRES correction workspace; both it and implicit-QR Arnoldi remain
matrix-free.

Appendix-D processes use cached corner-removal paths and successive one-box CG
isometries. Collective p-body sums and local `K_X,Y` kernels share this
geometry; static sparse and matrix-free generators support p-body Hamiltonian,
local-jump, and collective-jump terms. Operators are checked for permutation
symmetry and restricted bases reject missing reachable sectors. Basis
completeness is tested with the exact Schur--Weyl coordinate-dimension identity,
while local p-body closure is generated from only the relevant Young-lattice
ancestors and descendants;
neither check scans all partitions of a large `N`.

Exact path factors are now represented as a direct working-precision scalar
for ordinary sizes and as a binary mantissa/exponent otherwise.  The latter is
applied jointly with a contraction (or with both gain-map contractions), so a
Kac-normalized result never forms `Inf*0` or squares a small contraction before
the exact factor can rescue it. Static collective blocks and local-gain
coordinates estimate cancellation from the absolute path sum and recompute
risky sectors or sector-pair groups in precision selected from the lost
path-weight bits. Widened local triplets reuse their path contractions for
every coordinate in the group. Preallocated dynamic blocks retain the native
allocation-free path; dynamic local gains with risky path scales are rejected
at compilation (including all-direct factors), while evaluated collective
blocks reject a cancellation-risk value with wider-prototype guidance when
fixed scratch precision cannot certify it.

One-body geometry is constructed lazily by `TermCompileContext`: models made
only of direct-PI or Appendix-D built-ins no longer pay its setup cost.  Custom
terms remain conservative and receive the one-body cache because their
delegated lowering is not known in advance.

## Published models

The correlated-emission model of PRA 94, 033838 (2016) and dissipative LMG
model of PRA 110, 062208 (2024) are implemented in `examples/paper_models.jl`.
The former is quantitatively checked against its analytical two-atom curves;
the latter is checked structurally and against its reported finite-size gap
trend and thermodynamic transition location.

## General bipartite negativity

Partial transposition uses product Schur--Weyl blocks. Qubits have a specialized
multiplicity-free SU(2) implementation. For qudits, Littlewood--Richardson
multiplicity spaces are obtained without enumerating SYT by solving the U(d)
generator intertwining equations. The intertwiner basis is immaterial because
the PI state is the identity on the matching symmetric-group multiplicity
space. LR multiplicities are counted exactly by lattice tableaux; forbidden
weights are removed before sparse simple-root assembly and SPQR nullspace
recovery. Prepared plans now retain exact-support discarded-weight blocks;
the rank-revealing factorization and temporary dense nullspace basis remain
the practical setup limits for very large qudit irreps.

Arbitrary particle marginals are returned by `reduced_state(rho, k)` as a new
equation-(7)-normalized `PIState` on `PIBasis(k,d)`. The implementation traces
product-Schur blocks and shares the SU(2)/Littlewood--Richardson conventions
used for negativity. `reduced_purity` is evaluated from this reduced PI state.
`ReductionPlan(basis,k)` retains the fixed product-Schur recouplers and can be
reused by marginals, purities, negativities, charge-resolved negativities, and
partial-transpose spectra. LR coefficients are counted exactly by lattice
tableaux; forbidden weights are removed before sparse simple-root assembly and
SPQR nullspace recovery. Very large qudit irreps can still require substantial
temporary and retained memory. `ReductionWorkspace(plan,rho)` separately owns
reusable product-block, multiplication, partial-trace, partial-transpose, and
reduced-sector scratch. It also converts compact real qubit recouplers once to
the workspace's complex scalar type, avoiding Julia 1.10's allocating
mixed-eltype matrix-multiplication fallback. Type-matched packed qudit LR
blocks are shared read-only; wider workspaces convert only stored nonzeros.
`reduced_state!` additionally
reuses the output state; eigensolver result vectors remain per-call
allocations. Plans stay immutable and shareable, while each concurrent task
requires its own workspace.

Collective-observable quantum Fisher information is evaluated sector by sector
from the eigenvalues of each physical density block. Each sector contribution
is multiplied by `f^nu`; no full-Hilbert-space density matrix or generator is
constructed. Local matrices passed to `qfi` are interpreted as collective sums.
The multiparameter extension `qfim` diagonalizes each density block once,
reuses collective geometry across local generators, and accumulates a real
symmetric QFI matrix with the same exact sector multiplicities.

Collective first and second moments can be contracted directly with
equation-(31) blocks through `collective_expectation`, `collective_variance`,
and `collective_moments`. These avoid assembling a flattened `PIOperator`.

Information-theory functions operate sectorwise with exact multiplicities.
Schur-resolved entropy, GT-basis coherence, collective-charge asymmetry,
sector-resolved QFI, tangent QFIM decompositions, and charge-resolved
negativity preserve normalized conditional-sector and multiplicity factors
explicitly.
Metrological diagnostics now include collective covariance/squeezing,
unitary-generator QFI/QFIM, tangent-state QFIM, SLD compatibility, and QFI
entanglement-depth bounds. Response tools use PI-coordinate Liouvillians for
decay modes, resolvents, adjoint evolution, integrated correlations, static
susceptibilities, and augmented tangent dynamics.

Liouvillian block assembly is expressed through centralized column-major
vectorization identities: left/right multiplication, sandwich maps,
commutators, and Lindblad dissipators. Full Hilbert- and Liouville-space
matrices can be checked for permutation invariance/covariance using adjacent
transpositions, without constructing permutation matrices or enumerating the
full symmetric group. These validation utilities are intended for small dense
or sparse reference models; production PI evolution remains Schur-block based.

The latest global audit also covers Appendix-D construction/application.
One-body sector connections and exact p-body path weights are cached inside
their geometry objects; repeated p-body terms share geometry by body order
during generator construction. Static matrix-free kernel collections are
concrete tuples, avoiding per-term dynamic dispatch in RHS loops. Trajectory
ensembles return concretely typed containers and reuse jump-count work arrays.
No global mutable geometry cache was introduced: the remaining front-loaded
CG construction cost stays explicitly owned by a basis/model or user cache.

Dominant iterative-solver and Floquet work arrays derive their complex
floating type from the operator and storage-bearing inputs rather than
imposing `ComplexF64`. Fully `Float32` problems therefore halve the dominant
Krylov/Floquet buffer storage. Wider inputs promote compatible materialized or
custom operators.
Compiled matrix-free PI plans own block scratch at their compiled precision;
their vector, batch, adjoint, Krylov, and Floquet paths therefore reject a
wider source before allocating large solver storage, and a model must be
compiled at the wider precision instead. Narrow destinations and incompatible
caller-owned workspaces likewise raise rather than truncate. Integer Floquet
times are accepted only when exactly representable. State validation also
reuses the Hermiticity result already computed by positivity diagnostics,
avoiding a duplicate Schur-block scan while preserving every validation
condition and invalid-input failure mode.

Floquet-gap extraction uses its public `atol` to certify the fixed multiplier
near one before selecting the subleading decay mode. A map without that fixed
point, or with a remaining multiplier outside the tolerated unit disk, raises
instead of losing its merely closest eigenvalue or hiding an instability. The
scalar result no longer widens `Float32` through a literal `0.0` clamp.

The CI matrix tests Julia 1.10 and current Julia on Linux and macOS, checks
public method ambiguities, runs four-thread allocation/thread-safety gates,
and builds documentation strictly. `benchmark/performance_regression.jl`
guards allocations and backend equivalence without unstable wall-clock
thresholds; `benchmark/performance_audit.jl` remains the human-readable timing
and RAM report.

Density-operator spectra are diagonalized sectorwise and retain exact Schur
multiplicities in compressed form. Complete Liouvillian spectra operate on the
PI-coordinate generator; matrix-free inputs are materialized only at the PI
dimension, since returning every eigenvalue is intrinsically a dense spectral
task.
The dedicated PI Liouvillian gap routine removes the complete numerical
stationary cluster before selecting the mode of largest real part. It reports
degenerate steady manifolds and oscillatory controlling modes explicitly and
rejects unstable generators unless diagnostic behavior is requested.
Verified unitary weak symmetries can further split the PI Liouville matrix into
charge eigenspaces before diagonalization. Sector spectra are recombined for
the exact global gap; automatic mode selects among the standard local
tensor-power candidates. Antiunitary symmetries are not treated as linear
charge decompositions.

Explicit Evans uniqueness uses stacked vectorized commutators. Model-level
checks never expand the full tensor-product Hilbert space: local
irreducibility is certified on one particle, single-sector models use their
Schur block, and collective multi-sector models are rejected by conserved
sector populations. Remaining built-in local, collective, and Appendix-D
p-body combinations are reduced to the kernel of the positive joint
commutator square on an auxiliary `PIBasis(N,d^2)`. Ket and bra tensor indices
are regrouped particle by particle before applying one- or p-body geometry;
Hamiltonian commutators are summed before squaring, while each collective jump
is summed before its two adjoint constraints are squared. Kernel dimensions
are multiplied by exact symmetric-group multiplicities, so the result counts
the full Hilbert-space commutant rather than only PI operators. A conservative
`memory_budget` guard returns `missing` before costly setup, as do direct or
custom PI terms without microscopic recoupling. The eigenvalue threshold
reports the normal-equation roundoff floor separately from the requested
singular-value tolerance.
Weak unitary and antiunitary symmetry checks operate directly on the
Liouvillian covariance relation. Local tensor-power symmetries are lifted
through Schur-block exponentiation, while antiunitary checks conjugate the
Liouvillian in the fixed real-CG computational/GT convention.

## Public documentation architecture

The documentation now has a self-contained framework introduction deriving
the PI covariance condition, Schur--Weyl decomposition, equation-(7) storage
normalization, polynomial dimension, term dictionary, and prepared workflow.
The public API is no longer an unfiltered `@autodocs` dump: explicit
categorized pages under `docs/src/api/` contain one canonical `@docs` entry
for every export, with an alphabetical index at `docs/src/api_reference.md`.
All exported bindings have source docstrings, including the formerly
misattached `iid_state`, `pi_liouvillian_gap`, and
`collective_covariance_matrix` descriptions and the density-spectrum aliases.

`docs/make.jl` fails when `Base.Docs.undocumented_names` finds a public binding
or when Documenter's `checkdocs=:exports` finds a documented export missing
from the manual. This keeps interactive Julia help and the hosted reference in
sync without duplicating descriptions in Markdown tables. The strict build
and the complete 2806-test suite passed after the population, spin/Schur, and
phase-space additions.

## Large-spin recoupling and one-body geometry

General-qubit subduction keeps the former allocation-light Racah formula for
doubled spins at most 32. Above that boundary the alternating Racah sum is
accumulated exactly as a `Rational{BigInt}` using a consecutive-term
recurrence. The coefficient is converted only once, as the checked square
root of its exact squared rational. This prevents the separate factorial
products from overflowing near doubled spin 99 and preserves the established
Condon--Shortley phase. Complete coupled-column orthogonality tests straddle
the doubled-spin 32/34 switch; lowering the cutoff prevents the several-ulp
factorial path error from reaching reduction tolerances.

`OneBodyGeometry` now prepares sparse child-to-parent one-box transitions once
per `(sector, removed partition)`. Content selection is applied before a CG
evaluation, and left/right transition lists are joined by their shared child
pattern. Structurally empty contraction cells share one read-only empty
vector. This replaces the former repeated parent-pair/child/local-label CG
loop; for a symmetric qubit `N=100` sector, warmed setup allocates below one
megabyte instead of multiple gigabytes. `_estimate_onebody_geometry` reports
exact structural counts and conservative incremental retained/setup byte
bounds without constructing geometry or evaluating a CG coefficient. For
qudits, candidate parents are counted separately in every GT weight space
before taking the left/right Cartesian product; the former `d^2 dim(mu)`
expression was not an upper bound when a weight had multiplicity.

For a fixed small irrep at enormous `N`, traceless collective generators can
be the `O(1)` difference of `O(N)` one-box branches. Ordinary sizes still use
the cached working-precision contraction. Once `N > 1/sqrt(eps(T))`, only the
requested sector is recomputed in the narrowest adequate wider precision;
BigFloat receives enough local guard bits for the actual particle count. The
result is checked while converting back to the declared output precision.

Qubit/qudit reduction and negativity paths no longer form an underflowing
physical parent block and multiply it by a huge subsystem multiplicity later.
They fuse the complete exact multiplicity ratio into the stored coefficient
block with `_checked_mul_sqrt_exact_ratio`. Reduced blocks are accumulated
directly in equation-(7) output coordinates, while partial-transpose blocks
used for trace norms are product-multiplicity weighted. Per-copy spectra that
are genuinely outside the requested scalar range raise rather than silently
returning zero. Product sectors with no retained parent coupling are omitted
from `ReductionPlan`, which is particularly important for large restricted
bases.

## Large-multiplicity state analysis and fused exact scaling

For a normalized state, the numerically bounded sector matrix is

`rho_bar_nu = sqrt(f^nu) C_nu = f^nu rho_nu`.

Its trace is the sector probability. Entropies, Rényi sums, fidelity and trace
distance, relative entropy, collective moments, QFI/QFIM, symmetry-resolved
quantities, one-body marginals, and spin phase space use this weighted matrix.
Schur-diagonal population extraction and population visualizations use the
same coordinates, and state reconstruction applies the inverse exact scale
jointly with each supplied population.
Entropy restores the multiplicity-space term with `p_nu log(f^nu)`, where the
logarithm is evaluated from a bounded binary mantissa. Numerical-rank tests
scale with each sector's own spectrum. Fixed-spin regression oracles through
`N=2100` cover multiplicities and square roots beyond Float64 while the
weighted answers remain finite.

Checked combinatorial scales have two tiers. A small, representable ratio
stores its native factor and follows the former single-multiply path.
Otherwise the implementation retains the exact numerator/denominator plus a
bounded binary mantissa/exponent; scalar, array, square-root, and two-factor
applications combine that scale with the numerical value before restoring the
exponent. Exact IEEE overflow and least-subnormal endpoint tests run only when
a rounded answer reaches an endpoint. A final value outside the declared type
raises rather than becoming zero or infinity.

Trace and sector-population contractions use the fused path and compensated
sector summation. `maximally_mixed_state` uses the exact retained Hilbert
dimension. Per-copy `physical_block`, purity, and a coefficient-space
Liouvillian trace vector still require their final stored values to exist in
the selected type and raise with wider-type guidance otherwise.
Stored-block multiplication attempts the original BLAS product first. If that
intermediate overflows before division by `sqrt(f^nu)`, a fused dot-product
fallback applies the exact inverse scale term by term and reserves guarded
wider multiplication for severe dot-product or intra-complex cancellation.
Physical Schur visualization
metrics similarly retain the direct aggregate-first path and scale entries
before the Frobenius norm or SVD only when that aggregate cannot be converted.
`ReductionWorkspace` retains prepared exact parent scales, scalar-matched
recouplers, and a parent-block buffer; warmed in-place marginal allocation is
below the prior small-system baseline instead of rebuilding BigInts or
allocating mixed-real/complex multiplication scratch on every call.

## Symmetric stabilizer Rényi entropy

`stabilizer_renyi_entropy` implements the pure-state second stabilizer Rényi
entropy of Passarelli--Fazio--Lucignano, not a mixed-state fourth-Pauli-moment
functional.  It consequently accepts only validated qubit density operators
with unit purity and exact stored support in the `(N,0)` Schur sector.  This is
intentional: applying the formula to the maximally mixed state would produce a
large number with no nonstabilizerness interpretation.  A complete basis is
still valid if every nonsymmetric block is structurally zero.

For each split `L=n_I+n_Z`, the implementation forms a bounded
hypergeometric-weighted rectangular block and transforms it as

`E_L = K_L H_L K_(N-L)^T`,

where `K_s[k,d]` is the normalized Krawtchouk polynomial coefficient.  The
recurrence is evaluated only to its midpoint and reflected with
`K_s[s-k,d]=(-1)^d*K_s[k,d]`; the middle odd-parity value is structural zero.
The `L` slices stream Pauli representatives directly into log-sum-exp fourth
and second moments.  Exact binomials are used only during plan setup to seed
hypergeometric mode probabilities; all retained setup factors are bounded.

This removes the paper's need to store `O(N^3)` representative Pauli matrices
and reduces a subsequent scalar `M_2` evaluation to `O(N^4)` time.  The
immutable plan retains the `O(N^3)` Krawtchouk tables, while a task-owned
workspace holds one `O(N^2)` probability table and five complex transform
buffers.  Do not replace its complex Krawtchouk workspace copies with mixed
real/complex `mul!` calls: Julia 1.10 can allocate conversion scratch there.
The Pauli second-moment identity is a non-negotiable transform diagnostic;
slightly negative `M_2` is corrected only when its fourth-moment violation is
already within that numerical bound, otherwise evaluation raises with
wider-precision guidance.

Static collective p-body blocks and local-gain sector-pair groups detect
ill-conditioned path cancellation and recompute only the affected data with
guarded wider precision. The gain check includes large direct path-pair
factors: representability of every factor does not certify their sum. Wide
triplet construction reuses each widened path contraction across all
coordinates in its sector pair. Native sums are widened whenever a conservative
`operation_count * eps(wide) * absolute_sum` forward-error bound cannot certify
their final ulp. If the widened result still lies inside that uncertainty
interval, the routine raises instead of treating the interval as proof of an
exact zero; stable nonzero entries retain strict output-range checks.
Preallocated dynamic p-body kernels cannot
widen caller-owned buffers, so cancellation-prone dynamic local p-body gains
are rejected at compilation and other dynamic p-body blocks apply the
analogous forward-error/ulp check before raising with guidance to widen the
`InPlaceTimeOperator` prototype. Mean-field collective
moments associate extensive factors with local expectations before
multiplication, and p-body expectations fuse the exact subset count. This
keeps Kac-normalized low-precision predictions finite without slowing the
ordinary direct branch.

The same audit covers one-body and direct-PI schedules. Public
`local_kernel_element` keeps its original native loop below the large-`N`
threshold and otherwise certifies the path sum before selectively rebuilding
the two retained sectors at wider precision. Guarded collective conversion
checks nonzero underflow and exact IEEE endpoints. A scheduled collective
one-body term is rejected at compilation when its fixed workspace cannot use
that wider geometry. Scheduled direct-PI terms prepare exact inverse
multiplicity scales once; ordinary sectors retain the previous direct division,
while a sector with unrepresentable `sqrt(f^nu)` fuses its exact inverse with
each stored coefficient.

## Coordinated performance pass (2026-07-19)

Six high-return changes now share the same immutable-plan/task-owned-workspace
contract.

Driven one-body `LocalJump` and `CorrelatedLocalJumps` schedules no longer
retain all representation-theoretically allowed PI coordinate pairs. For each
common child partition they prepare a rectangular contraction matrix and apply
the gain as `C*X*C'`; correlated schedules keep one such set per possible
effective Kossakowski channel and use only the evaluated rank. Forward,
adjoint, and matrix-batch routes share the same factorization. At `N=20`, the
audit reduced a warm driven application from roughly 103.7 ms to 0.168 ms and
combined retained plan/workspace memory from about 43.3 MiB to 0.355 MiB.

`CompiledPIModelFamily` compiles fixed operator geometry once and binds only
selected scalar rates in each `SpecializedPIModel`. Family parameter scans
reuse one task-owned `LiouvillianWorkspace` after the first point. A typed
`RecycledGMRESAlgorithm` retains a bounded GCRO space, but reapplies every
retained direction under each new trace-fixed operator and validates the raw
Liouvillian residual. Harmonic scans continue with the full selected Ritz
matrix. Stationary correlation spectra use bounded shared-Arnoldi multi-shift
batches, retrying only unconverged columns with restarted shifted GMRES; a
supplied workspace is never replaced outside the declared memory budget.

High-level sparse compilation, family specialization, and scan execution now
share a 512 MiB default preflight. Explicit materialization is rejected before
the sparse coordinate matrix is allocated; automatic family binding uses its
compile-time bounds and reuses a task-owned workspace when matrix-free is
selected. Scan preflight reserves the full ordinary or recycled Krylov basis,
transient Ritz/continuation copies, retained outputs, and all active local
workers before passing the remaining allowance downstream. Point diagnostics
separate the conservative known peak from unknown builder allocations and
user diagnostic payloads. `Inf` is retained as an explicit disabled state
rather than converted into a finite sentinel.

The scan auto policy applies the same dimension crossovers as the high-level
commands after including all workers and retained history: an over-budget
direct or dense candidate falls back to GMRES or Arnoldi, while an explicit
request remains an error. Resource output/restart/live-vector bounds retain
iterative result precision and conservatively reserve `ComplexF64` storage for
Float32 factorizing steady-state and dense-spectrum routes. Prepared steady
scans use the already retained operator directly, avoiding an unreported
sparse-to-matrix-free compatibility conversion.

Strong diagonal ket/bra charge restrictions lower compatible fixed prepared
Hamiltonian, collective-dissipator, and local-gain kernels into Cartesian
Schur rectangles after an exhaustive ambient leakage certificate. The hot
forward, adjoint, response, and Krylov paths then retain no ambient vectors.
Equal ket/bra selections share identical restricted block matrices. General
coordinate masks and unsupported kernels retain the embedded fallback;
`restriction_full_residual` deliberately uses separate ambient scratch to
audit the omitted coordinates.

Appendix-D path isometries cache each GT-pattern table and one-box transition
matrix, then propagate all centre columns through two reusable buffers for
each local word. A representative `N=6,d=3,p=3` setup fell from about 569 MiB
to 14.6 MiB allocated while retaining the same 0.95 MiB geometry. Word counts
and radix powers are checked before allocation.

PI--HEOM stores one packed edge list plus CSR-style ADO incidences instead of
dense neighbor/factor tables. Forward and adjoint traversals reuse `Q*rho`
and `rho*Q` once per source-ADO/bath group, and system applications use bounded
16-column batches. The stationary preconditioner extracts the system block
once. `ComplexF32`/`ComplexF64` use one guarded Schur form, the root LU, and
compact ADO shifts, with reciprocal-condition and transformed-residual checks;
unsafe shifts and generic precision fall back to duplicate-aware LU. The safe
route retains `O(nPI^2+nADO)` data. Its two Schur scratch vectors are locked,
so one shared preconditioner serializes concurrent applications.

## Prepared local-factor trace (2026-07-23)

`LocalFactorTracePlan` traces the same internal tensor factor from every PI
supersite while retaining all particles. For each normalized occupation of
kept-factor matrix units, setup constructs its complete output
Schur-coordinate column and the source column obtained from the adjoint
identity insertion. The exact map is their product `Q*L'`. This polarized
one-box recurrence visits PI occupations rather than the exponentially many
local strings, and it correctly maps a sector-restricted supersite basis into
a complete kept-factor basis.

The plan retains only the exact nonzero CSC support of its two transforms.
Column construction uses exact-sized per-column chunks and allocates the final
CSC arrays once after the total support is known, so setup does not hide an
unbudgeted vector-growth peak. Its 512 MiB preflight is evaluated from the
exact kept-factor PI dimension before constructing that basis, enumerating
occupations, or allocating the transforms. Repeated applications use two
matrix-vector products and one output-sized task-owned
`LocalFactorTraceWorkspace`. In particular, spin-only pseudomode observables
should trace the mode once in PI coordinates and then reuse the ordinary
qubit `ReductionPlan`; enumerating all lifted Pauli strings is retained only
as a small-system oracle.

## Paired supersites and local pseudomodes (2026-07-23)

`PISupersite` treats one physical system and all of its local finite
auxiliaries as one particle before applying Schur--Weyl compression. For
factor dimensions `(dS,r1,...,rM)`, the exact local dimension is
`D=dS*prod(r_mu)` and the complete operator coordinate count is
`binomial(N+D^2-1,N)`. This preserves the pairing between system `i` and its
own modes while avoiding every `D^N` construction; it is therefore distinct
from a tensor product of complete global `CompositePIBasis` factors.

System-only one-body matrices are lifted by local Kronecker products.
System-only `p`-body matrices use a sparse mixed-radix permutation that
inserts the auxiliary identity separately inside every particle:
`(system1,aux1) tensor ... tensor (systemp,auxp)`. The implementation retains
exact input support and preflights the resulting sparse arrays. This avoids
the grouped but physically incorrect `kron(Hsystem,Iaux^tensorp)` ordering.

`BosonicPseudomode` owns finite-cutoff ladder, number, parity, boundary, and
vacuum data. `PseudomodeCoupling` lowers complex rotating and
counter-rotating strengths into real rates multiplying Hermitian local
quadratures. `pseudomode_model` keeps the system Hamiltonian, every mode
frequency, coupling quadrature, local thermal damping channel, and lifted
system term separate. Consequently ordinary compilation can fuse compatible
kernels, while `compile_family` can reuse the exact Schur geometry across
scalar-rate scans. Product-state construction acts on one supersite before
the PI tensor-power recurrence, and `pseudomode_trace_plan` reuses the exact
local-factor trace to return the complete system-only PI state.

## One shared/global pseudomode (2026-07-23)

`GlobalPseudomodeModel` represents a physically distinct topology: the
many-particle system remains in one `PIBasis`, while a single oscillator is a
`FiniteOperatorBasis` factor. The resulting coordinate count is
`length(system_basis)*(nmax+1)^2`; the implementation never promotes the
oscillator into every PI particle or forms a full composite Kronecker
superoperator.

For a local matrix `L`, the builder first prepares the exact collective
operator `J_L=sum_i L_i`. With `K=g*J_L`, `X=a+a'`, and
`Y=-im*(a-a')`, the rotating interaction is lowered as
`((K+K') tensor X - im*(K-K') tensor Y)/2`. The counter-rotating interaction
uses the opposite sign on the `Y` term. This Hermitian-quadrature form feeds
the existing factorized commutator maps, preserves complex couplings, and
discards exact zero quadratures before allocating their maps.

The prepared `background` contains system dynamics, oscillator frequency, and
coherent coupling. Global thermal loss/gain channels are retained separately
for `CompositeTrajectoryPlan`, while `generator` includes them exactly once
for deterministic dynamics. Forward and adjoint composite application reuse
task-owned tensor-mode scratch. Factor reductions contract only traced
diagonal coordinates with exact Schur multiplicities, so both the system
`PIState` and the finite mode density matrix are obtained without a
many-particle reconstruction. The compatibility wrapper deliberately carries
no materializable PI plan; automatic stationary solving therefore selects
GMRES, and batched adjoints use the explicit callback. Memory preflight counts
the nested system `LiouvillianWorkspace` and the wrapper-owned trace copy.
Model-aware product states and reductions restore the retained BigFloat
context, and high-level stationary and selected-spectrum Krylov solves keep
that context for the complete solve rather than only individual applications.

## Shared-bath PI hierarchy of pure states (2026-07-23)

`HOPSPlan` lowers the linear hierarchy of pure states into the direct sum of
the Schur irreps retained by one exact `PIBasis`. Every hierarchy node has
`weak_pi_dimension(basis)` amplitudes. The implementation reuses HEOM's
downward-closed multi-index enumeration, exact scaled-coordinate similarity,
and packed parent/child topology, but applies Hamiltonians and bath couplings
to pure-state columns rather than PI density coordinates.

One right-hand-side evaluation batches all hierarchy nodes through each
prepared Schur block. Parent/child edges are grouped once by bath, and one
hierarchy-sized buffer serves both the forward and adjoint coupling actions.
Hermitian couplings combine both edge directions in one pass; non-Hermitian
couplings overwrite the buffer between downward and upward passes. The
fixed-step RK4 path keeps four low-storage stage hierarchies. For real nonnegative
exponential weights, exact stationary complex Ornstein--Uhlenbeck half-step
updates provide the start/midpoint/end noises without drawing inside the
conditioned RHS. Complex or signed decompositions require an externally
prepared total-covariance realization.

Exact duplicate poles are aggregated separately inside each bath before the
combinatorial hierarchy count. Edge coefficients are checked and packed at
setup rather than multiplied in the hot loop. Positive-cutoff enumeration
uses a budget-derived retained-node cap and a separately bounded frontier, so
a small downward-closed hierarchy remains constructible even when the
unpruned binomial count exceeds `Int`. BigFloat plans capture the maximum
input precision and active rounding mode; all workspace, OU, integration, and
reduction routes re-enter that context.

`hops_average` never retains auxiliary or root histories. It converts each
unnormalized root directly into PI coefficient blocks and applies an online
Welford reduction, using deterministic trajectory-index seeds and balanced
task-owned workspaces. A general PI density starts from a categorical
eigenensemble of its multiplicity-weighted Schur blocks. This construction
never samples multiplicity tableaux and never constructs a `d^N` object.
Linear roots are deliberately not normalized; normalized nonlinear/Girsanov
HOPS remains a distinct, unimplemented equation.

## Partial-trace kernels and prepared marginals (2026-07-23)

The internal-factor trace previously retained two dense rectangular
occupation transforms even though their support is structural and very
sparse. `LocalFactorTracePlan` now constructs one normalized occupation column
at a time and stores only values for which `iszero` is false in CSC form. No
numerical dropping tolerance is used. Setup preflight counts occupation
tuples, output-basis construction, recurrence caches, column scratch, growing
CSC storage, and sparse Gram/trace validation. The warmed application remains
two matrix-vector products with one task-owned occupation vector. On the
development machine at `N=4`, the `d=4 -> 2` retained transforms decreased
from about 2.19 MB to 35.4 KB and the `d=6 -> 2` case from about 46.1 MB to
217 KB; the corresponding warmed contractions were about 11 and 24 times
faster in that local measurement. Orthonormality validation now streams one
sparse Gram column through row adjacency and an output-sized stamped
accumulator. Its retained scratch is linear in `nnz(Q)+size(Q,2)`, rather than
the former quadratic Gram matrix.

Particle reduction now evaluates
`tr_B(U*C*U') = sum_q U_q*C*U_q'` directly. A reduction-only workspace
therefore owns only one `da x dim(parent)` product slice and no
`(da*db)^2` product block. Negativity keeps the complete product block because
partial transposition requires it. `reduced_state`, `reduced_state!`,
`reduced_purity`, and `reduced_purities` retain checked defaults; prepared
callers may validate once and request `check=false`. The batched purity helper
validates once for all requested sizes, and unprepared `k=0,N` endpoints avoid
building an otherwise unused reduction plan.

Geometry-based one-body marginals gained `OneBodyRDMWorkspace` and
`one_body_rdm!`. The workspace owns exact prepared multiplicity scales and one
largest-sector weighted block. This removes per-sector block construction;
the warmed machine-precision validate-once contraction is allocation-free in
the regression workload.

Composite factor traces gained immutable `CompositeReductionPlan`s. A plan
packs 0-based traced diagonal offsets, multiplicity-group boundaries, and
exact/binary-scaled Schur factors for one retained composite factor. Warmed
PI and finite-factor destinations allocate no memory in the regression
workload. `GlobalPseudomodeModel` retains one such plan for the PI system and
one for the shared mode, so repeated observable or cutoff analyses no longer
rebuild trace groups. A representative `N=6` qubit system with an
eight-level finite factor took roughly 0.8 microseconds for the system trace
and 1.5 microseconds for the mode trace on the development machine. These
times are dated local measurements; allocation, exact-support, and numerical
equivalence are the portable regression gates.

## Sparse-first fixed collective lowering (2026-07-24)

Fixed one-body Hamiltonians and jump operators now lower directly onto exact
CSC Schur support. The general one-box route evaluates candidate entries in
the established connection/contraction order, while the fully symmetric
occupation route emits only the diagonal and allowed occupation transitions.
Public `collective_block` remains dense, and driven operators retain dense
preallocated blocks because their support may change with time. Guarded-wide
large-\(N\) lowering keeps its certified dense fallback for the small feasible
irreps where cancellation requires wider arithmetic.

Collective jump losses form `K'K` from the retained sparse blocks and remove
only exact zeros. Fixed local one-body losses reuse the same sparse lift.
When all compatible fixed Hamiltonian or loss blocks are sparse, kernel fusion
uses a single Schur-vector accumulator per sector and constructs CSC columns
directly. Contributions to each coordinate are accumulated in model-term
order; exact cancellations are omitted without a dropping tolerance. Mixed
one-/p-body or direct-PI collections keep the existing dense fusion fallback.
Term-resolved trajectory and population plans remain separate from this
deterministic fusion path.

## Diagonal-only collective one-body geometry (2026-07-24)

Collective one-body Hamiltonians and jumps preserve every Schur sector. For a
mixed-sector model that contains no local one-body gain, compilation now builds
only `(sector,sector)` one-box contractions. It still shares the transition
tables and exact branch scales of `OneBodyGeometry`, but omits every
rectangular sector-changing contraction that only a local gain can consume.
The sole fully symmetric sector keeps its smaller occupation-number backend;
models containing `LocalJump`, correlated local gains, or conservative custom
terms retain the complete geometry.

The structural setup estimator applies the same diagonal-sector selection
before memory-budget enforcement, and high-level recommendations report that
smaller bound. Regression gates compare fixed and driven collective
Liouvillians with the complete-geometry oracle, require every retained key to
have equal input/output sectors, and verify that adding a local gain restores
the off-sector maps.

## High-level Krylov exponential dynamics (2026-07-24)

`solve_dynamics` retains fixed-step RK4 as its automatic default and now
accepts `ExpvAlgorithm` (or `:expv`/`:krylov_expv`) for autonomous generators.
It prepares one restarted-Arnoldi workspace and one task-owned source-action
workspace, then reuses both across saved intervals and observable-only
streaming. Its resource preflight counts the exact large-array payload
``n(m+4)+(m+1)m+(m+1)^2`` for `m=min(n,krylovdim)`, separately from prepared
Liouvillian scratch and returned output. Driven sources and parameter-bearing
applications raise rather than being treated as one fixed exponential.

The adaptive exponential kernel now retains the Arnoldi factorization after a
rejected slice. Because the full-space initial state is unchanged, retrying at
a shorter step only recomputes the small projected exponential. Accepted
slices invalidate the factorization as before. `arnoldi_factorizations`,
`trial_evaluations`, and `operator_applications` make that reuse visible
without changing the defect estimator or acceptance criterion.

## Trajectory effective-loss node cache (2026-07-24)

Density-valued and weak-PI trajectory workspaces now own one effective jump
loss cache keyed by the current physical integration node. Fixed RK4 therefore
prepares driven jump rates and combined loss blocks at only `t`, `t+h/2`, and
`t+h`; its duplicate midpoint and the endpoint intensity check reuse the
prepared values. Dormand--Prince likewise reuses its duplicate endpoint. The
channel-resolved intensity pass consumes the exact cached scales used in the
combined loss, avoiding a second callback evaluation and keeping channel sums
consistent with the no-jump drift.

Numeric jump rates are autonomous even when the Hamiltonian is driven and may
remain cached for the workspace lifetime. A driven jump cache is invalidated
at each public path boundary, so sequential workspace reuse with different
parameters cannot expose stale loss blocks. Cache validity is published only
after every callback and block accumulation succeeds. Fixed-seed stochastic
tests, callback-count tests at repeated RK nodes, and changed-parameter
workspace-reuse tests are the portable regression gates.

## Direct Schur-sector preconditioner lowering (2026-07-24)

Prepared autonomous PI Liouvillians now construct the block-diagonal
preconditioner directly from their immutable Schur kernels. Hamiltonian,
collective-gain, anticommutator-loss, same-sector local-gain, and Appendix-D
same-sector contractions are accumulated in the same order as the prepared
matrix-free action. Off-sector gain branches are omitted only inside the
preconditioner. The physical trace border, operator-scale normalization, and
optional regularization are then applied exactly as in the generic route.

Consequently a prepared plan needs only the reproducible scale probes, rather
than one full Liouvillian application per PI coordinate. Plan-less linear
operators retain the original probe oracle. Regression tests compare the
resulting preconditioner actions and term-resolved diagonal blocks against
that oracle and explicit sparse materialization. Metadata distinguishes scale
probes, block probes, and prepared-kernel construction so setup amortization
remains inspectable.

A `SpecializedPIModel` supplies its bound rate tuple while lowering the shared
family plan, so continuation scans avoid coordinate probing at every parameter
point. Its high-level stationary solve keeps the specialization as the
linear-operator source rather than erasing those parameters behind a detached
callback. A genuinely plan-less callback continues to use probing because no
reliable source protocol can recover its bound physical rates.

## Packed Appendix-D and qudit-reduction geometry (2026-07-24)

`PBodyGeometry` now retains each completed Appendix-D path as one CSC matrix
whose columns combine the centre pattern and local word. Construction still
uses the established two dense propagation buffers, but never retains the
zero-heavy endpoint-by-centre-by-word tensor. Fixed and driven contraction
kernels traverse exact CSC support; public three-index indexing remains
available through the internal array-compatible wrapper. The geometry's
`estimates` field records packed entries, dense-equivalent entries, retained
bytes, and dense payload bytes.

Raw-model memory preflight now sums a coefficient-free structural estimate
for every distinct body order together with the required one-body geometry.
The p-body estimate follows exact removal-path multiplicities and GT-content
selection rules, so it bounds packed CSC support without evaluating a CG
coefficient or falling back to a PI-coordinate-squared estimate. Combined
centre/word dimensions and retained entry counts use checked `Int` or exact
`BigInt` arithmetic. Cancellation-risk driven blocks own one preallocated real
matrix and accumulate their forward-error absolute sum while traversing the
same CSC nonzeros as the value kernel.

The public `subduction_intertwiners` result remains a vector of dense matrices.
`ReductionPlan` instead converts qudit LR maps into one exact-support CSC block
per discarded-factor label and releases the temporary dense nullspace result.
Reduced-state contractions consume each block directly, while negativity
forms the required complete product block from sparse support without
materializing a dense intertwiner. Type-matched workspaces share the immutable
blocks; wider workspaces convert only stored nonzeros. Qubit SU(2) recouplers
retain their existing dense small-block path. `ReductionPlan.estimates`
reports the packed and dense-equivalent sizes, and regression gates cover
machine precision, Float32 non-narrowing, BigFloat workspace conversion,
public dense compatibility, exact p-body values, retained support, and warmed
in-place allocation. Packed LR maps also retain their already checked product
dimension, so later allocation and reshape paths cannot reintroduce an
overflowing `da*db`.

## Batched sensitivities, composite maps, and HEOM (2026-07-24)

`sensitivity_problem` now applies the physical generator once to the complete
augmented state matrix `[rho, d_rho_1, ...]`. Built-in driven schedules are
therefore evaluated once per ODE right-hand side rather than once per
parameter, and static derivative generators receive one prepared task-owned
workspace when the problem is constructed. Custom vector-only callbacks keep
their column fallback.

`CompositeSuperoperatorBatchWorkspace` retains two full-coordinate matrices,
two small factor-fiber matrices per active factor, and batch-capable nested PI
workspaces. Its capacity is immutable. Forward and adjoint tensor-mode
applications batch corresponding fibers from all right-hand sides; a
contiguous dense first factor consumes every fiber in one matrix product.
Resource accounting scales every retained buffer and nested workspace with the
declared capacity. Although `composite_matrixfree` remains a plan-less
compatibility wrapper, its retained vector workspace identifies the exact
immutable composite plan. The private source protocol uses that identity to
construct fresh task-owned vector and batch workspaces for prepared consumers,
including sensitivities, without re-entering the synchronized column loop.

HEOM matrix right-hand sides are reinterpreted as one `n_PI`-by-
`(n_ADO*n_RHS)` system batch. Prepared PI system schedules are evaluated once
before bounded chunks, and forward/adjoint hierarchy couplings then scatter
within each RHS independently. Chunk width never exceeds the system batch
capacity already retained by `HEOMWorkspace`; application does not grow it.
The optional `batch_columns` constructor keyword pre-grows that bounded
capacity using overflow-safe setup arithmetic. The synchronized HEOM adapter
publishes matching forward and adjoint batch callbacks.

## Julia 1.10 sparse batch inference guard (2026-07-24)

The local-jump batched anticommutator keeps forward sparse blocks and adjoint
wrappers in separate outer branches. A loop-local runtime ternary joined
`SparseMatrixCSC` and `Adjoint` values and caused one small heap box per sector
application on Julia 1.10. In a Floquet RK4 graph that became four allocations
per step despite fully preallocated numerical buffers. Splitting the short
sector loop restores the allocation-free warmed forward and adjoint kernels
without changing operation order or arithmetic.

## Prepared scalar and solver precision guards (2026-07-24)

Liouvillian lowering now prepares every scalar rate through one checked
precision contract. Fixed floating rates and Hamiltonian `hbar` values promote
the immutable plan type; exact integer/rational data keep the native
working-type route after checked conversion or exact quotient cancellation.
Callable rates are tied to the prepared geometry precision and reject wider,
nonfinite, complex, overflowing, or nonzero-underflowing results. Compiled
families carry the prototype rate type into their shared scalar schedules and
validate each specialization before allocating a backend, including through
the nested checked-rate wrapper used by correlated reservoirs. When a fixed
floating scalar widens the plan precision, dense p-body/direct-PI blocks,
losses, and rectangular contractions are converted once at preparation;
sparse one-body support is retained. This avoids mixed dense `mul!` packing
allocations in forward, adjoint, and batched application. In-place operator
prototypes and callbacks, as well as fixed built-in operators, reject
nonfinite coefficients before Schur lowering, and
matrix-free Liouvillian application rejects aliased input/output storage
before clearing the destination. Mean-field lowering follows the same
finite-real rate and finite-operator rules.

Weak-unitary symmetry projectors retain `ComplexF32` when constructed from
Float32 data, including Schur lifting, masks, workspaces, joint intersections,
and residual probes. Mixed Float32/Float64 joint specifications are promoted
once before lifting. LAPACK-unsupported scalar types raise with explicit
conversion guidance instead of narrowing. Working-precision roundoff floors
are included in unitary, charge, commutation, rank, and exact/probed residual
checks, with mixed joint projectors retaining the least-precise source floor.
Exact in-place projection is supported through a sector copy; other
overlapping views are rejected.

Factorizing steady-state routes keep their intentional `ComplexF64` sparse-LU
minimum but validate shifts and initial vectors before materialization.
Unsupported prepared precision raises for an explicit factorizing method;
`:auto` with basic diagnostics selects the matrix-free Krylov route instead.
Neither a BigFloat shift nor a wider floating initial state is silently
converted to Float64. Exact integer/rational components use checked conversion
and reject overflow or nonzero underflow.
