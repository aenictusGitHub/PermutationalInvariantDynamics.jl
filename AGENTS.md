# AGENTS.md

This file is the continuity guide for agents maintaining
`PermutationalInvariantDynamics.jl`. Read it before changing source code.

## Project purpose

The package simulates time-local permutationally invariant (PI) open-system
dynamics of `N` identical `d`-level systems directly in the PI operator
subspace. Production algorithms must not construct objects of size `d^N` or
`d^(2N)`.

The primary mathematical source is the repository paper:

- `JPhysA58_ 275301(2025).pdf`
- Thierry Bastin and John Martin, *J. Phys. A* **58**, 275301 (2025)
- DOI: `10.1088/1751-8121/addfc1`

Published validation models additionally use:

- `PRA94_033838(2016).pdf` (maintainer-local reference copy)
- `PRA110_062208(2024).pdf` (maintainer-local reference copy)

## Compatibility and package hygiene

- Package/module: `PermutationalInvariantDynamics`
- Julia compatibility: 1.10 and later.
- Core dependencies: `LinearAlgebra`, `SparseArrays`, `Random`, `SciMLBase`.
- Do not commit a root `Manifest.toml`. `Pkg.instantiate()` and `Pkg.test()` may
  recreate it locally; remove it before handing work back.
- The isolated quality setup may similarly create `quality/Manifest.toml`;
  remove it after running Aqua/JET. Keep the tracked `docs/Manifest.toml`.
- Preserve generic numeric types where practical. Do not introduce a blanket
  `ComplexF64` restriction into new mathematical APIs.
- Never silently truncate sectors, normalize states, clip eigenvalues,
  symmetrize invalid inputs, or replace negative rates.

## Mandatory mathematical conventions

- A partition is a descending length-`d` tuple padded with zeros.
- Sector order: descending lexicographic partition order.
- GT-pattern order: ascending order of the stored entry tuple.
- Sector matrices flatten in Julia column-major order.
- The PI basis is equation (7):

  `F_nu^(W,W') = sum_T |nu,T,W><nu,T,W'| / sqrt(f^nu)`.

- Stored coefficient block: `C_nu`.
- Physical Schur block: `C_nu / sqrt(f^nu)`.
- Trace: `sum_nu sqrt(f^nu) * tr(C_nu)`.
- Product block: `C_nu(A*B) = C_nu(A)C_nu(B)/sqrt(f^nu)`.
- The identity block is `sqrt(f^nu) * I`.
- Local computational labels are zero-based in the paper but Julia matrix
  indices are one-based. Keep conversions explicit.
- In a qubit Schur block the ascending stored GT-pattern order corresponds to
  magnetic labels `m=+j,+j-1,...,-j`; phase-space code must not reverse this
  convention implicitly.
- CG coefficients use the real Appendix-B convention, including the sign in
  equation (B.11).

### Large combinatorial factors

- `exact_binomial` and `exact_multinomial` are the public exact-`BigInt`
  routes. The multinomial implementation uses chained binomials rather than
  factorial intermediates.
- Never convert a large numerator, denominator, binomial, or Schur
  multiplicity separately when the required ratio or square root can remain
  representable. Cancel with `Rational{BigInt}` first and use the checked
  binary-scaled helpers in `src/partitions.jl`.
- One-body scales use exact one-box branch weights. Appendix-D collective and
  gain scales use exact removal-path weights. Prepared scales retain an exact
  ratio plus a bounded binary mantissa/exponent only when the standalone
  factor is not representable; ordinary small factors keep the direct native
  multiply.
- `iid_pure_state` uses mode-centered conditional-binomial recurrences and
  must never reintroduce a separately converted multinomial coefficient or an
  `Inf*0` product. Exact zero local amplitudes remain structural zeros.
- State analysis should use the bounded multiplicity-weighted block
  `sqrt(f^nu) * C_nu`, whose trace is the sector population. Form the physical
  per-copy block only for APIs that explicitly return a per-copy object; it
  may be genuinely unrepresentable even when entropy, QFI, moments, reduced
  states, or phase-space densities are finite.
- Public `physical_block` must validate that its standalone equation-(7)
  divisor `sqrt(f^nu)` is representable even for an all-zero coefficient
  block. Internal contractions may instead use the checked fused inverse when
  only the final product needs to be representable.
- A standalone nonzero combinatorial scale outside the requested floating
  type must raise with wider-type guidance. Do not silently change a public
  result's scalar type.

## Source map

- `src/partitions.jl`: partitions, corners, exact hook/Weyl and
  binomial/multinomial values, exact branch/path weights, and checked scaled
  conversions, plus shared precision-aware retained-scalar byte estimates.
- `src/gtpatterns.jl`: immutable GT patterns and recursive enumeration.
- `src/cgc.jl`: one-box U(d) CG coefficients and three-nu symbols.
- `src/basis.jl`: PI basis, states/operators, normalization, checked Schur
  block access/construction, stable tensor-power amplitudes, iterators, and
  exact sector metadata.
- `src/tensor_indices.jl`: shared tensor-index dimension and permutation helpers.
- `src/geometry.jl`: cached one-body geometry, exact branching scales, and
  equations (27), (31).
- `src/pbody.jl`: Appendix-D paths, exact path weights, generalized
  contractions, and p-body kernels.
- `src/operators.jl`: PI algebra, initial states, kernel assembly.
- `src/terms.jl`: immutable model tuples, built-in term types, validation, and
  the dispatch-based custom-term extension contract.
- `src/spin.jl`: spin matrices, collective-spin and qubit state conveniences,
  Dicke operators, and the standard six-rate qubit ensemble model.
- `src/vectorization.jl`: `vec`-identity superoperators and full-space PI
  covariance tests.
- `src/liouvillian.jl`: compiled term lowering, immutable Liouvillian plans,
  explicit application workspaces, sparse/matrix-free adapters, and steady
  states.
- `src/krylov.jl`: restarted GMRES, Schur-sector preconditioning, ordinary and
  harmonic Arnoldi, exact-shift implicit-QR restarting, and hard-locking
  preconditioned Jacobi--Davidson.
- `src/spectra.jl`: complete PI Liouvillian spectra and multiplicity-compressed
  density-operator spectra.
- `src/evans.jl`: Evans commutant uniqueness certificates and efficient PI
  model specializations.
- `src/symmetries.jl`: weak unitary and antiunitary Liouvillian covariance
  checks in full or PI Liouville space.
- `src/observables.jl`: expectations, RDM and diagnostics.
- `src/entanglement.jl`: bipartite negativity, LR/subduction intertwiners,
  reduced-state purities, and preallocated reduction workspaces.
- `src/information.jl`: entropy and PI state-distinguishability measures.
- `src/symmetry_information.jl`: sector decompositions, coherence, asymmetry,
  and symmetry-resolved Fisher information.
- `src/sciml.jl`: in-place `ODEProblem` and solution wrapper.
- `src/meanfield.jl`: finite product-state closure, explicit thermodynamic
  combinatorics, preallocated one-site evolution, product observables, fixed
  points, and traceless-Hermitian stability analysis.
- `src/evolution.jl`: preallocated direct density-matrix propagation from an
  assembled or matrix-free Liouvillian.
- `src/trajectories.jl`: PI quantum-jump trajectories, reusable stochastic
  workspaces, ensembles, and density-matrix averaging.
- `src/floquet.jl`: preallocated one-period maps, multipliers, periodic states,
  and stroboscopic evolution.
- `src/response.jl`: modes, resolvents, adjoint evolution and sensitivities.
- `src/highlevel.jl`: typed algorithms/results, unified research commands,
  memory estimates, solver recommendations, diagnostics, and compact displays.
- `src/populations.jl`: strict Schur-diagonal invariance certificates,
  population-only generators, preallocated evolution, and stationary solves.
- `src/phase_space.jl`: sector-resolved qubit Husimi-Q and Agarwal spin-Wigner
  data without full-Hilbert reconstruction.
- `src/visualization.jl`: Schur-sector block measurements and dependency-free
  text/SVG visualization with Young-diagram labels for PI states, operators,
  and superoperators.
- `src/spectral_visualization.jl`: multiplicity-compressed density-spectrum
  views plus reusable Liouvillian/Floquet spectral data, stability
  classifications, and dependency-free SVG rendering.
- `src/phase_space_visualization.jl`: reusable dependency-free SVG rendering
  of aggregate or sector-resolved spin phase-space data.

## Architecture and public API tiers

The intended dependency flow is:

`PIBasis -> PIModel -> compile -> CompiledPIModel -> solver/analysis`.

Mean-field prediction has a parallel geometry-free flow:

`(N,d,terms) -> MeanFieldPlan -> MeanFieldWorkspace -> one-site solver`.

Certified population dynamics has a reduced flow available only when the
Schur-diagonal subspace is invariant:

`PIModel -> PopulationPlan -> PopulationWorkspace -> solve_populations`.

`PIModel.terms` is a concrete immutable tuple. `LiouvillianPlan` owns prepared,
read-only Schur blocks and gain maps; it must never own mutable numerical
scratch. `LiouvillianWorkspace` owns scratch and must be reused by only one
task at a time. Sparse materialization and matrix-free application lower from
the same plan. Use `apply!`/`apply_adjoint!` with an explicit workspace in hot
or parallel loops. Compatibility `mul!`/`action!` paths are synchronized and
safe, but concurrent calls serialize.

Prefer `compile`, `solve_dynamics`, `stationary_state`,
`liouvillian_spectrum`, `diagnostics`, and `recommend_solver` in ordinary
research scripts. Advanced code may use `liouvillian`, `steady_state`,
`apply!`, and Krylov routines directly. `docs/src/api_tiers.md` records the
stable, advanced, and experimental surfaces.

For product-state predictions, prefer `MeanFieldPlan`, `solve_meanfield`, and
`meanfield_problem`. The explicit `meanfield_rhs!` and `meanfield_evolve!`
paths require a caller-owned `MeanFieldWorkspace`; plans may be shared but
workspaces may not. Do not describe this closure as exact finite PI dynamics
after correlations have formed.

Custom `AbstractPITerm` implementations extend the qualified hooks documented
in `src/terms.jl`. Their `compile_term` method must delegate to an equivalent
built-in term; compiled kernel internals are not an extension surface. Add an
end-to-end sparse, matrix-free, adjoint, validation, and `freeze` test for each
new extension pattern.

## Public documentation and API index

`docs/src/framework.md` is the self-contained conceptual introduction: PI
covariance, Schur--Weyl sectors, equation-(7) normalization, scaling, physical
terms, the prepared workflow, and validity limits. Keep it suitable for a new
research user and keep its runnable qubit example synchronized with the
high-level API.

`docs/src/api_reference.md` is the complete alphabetical entry point. Detailed
descriptions are split into the explicit public-only pages under
`docs/src/api/`. Every exported binding must have a source docstring so the
website and Julia's `?name` help remain identical. `docs/make.jl` enforces both
`Base.Docs.undocumented_names(...; private=false) == []` and Documenter's
`checkdocs=:exports`; adding an export therefore requires adding its docstring
and one canonical `@docs` entry. Qualify names that conflict with Base, such as
`PermutationalInvariantDynamics.isvalid`.

The public repository is
`https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl` and the
package is licensed under `GPL-3.0-only`; the canonical full text is the root
`LICENSE` file. `CITATION.cff`, the README, and Documenter source/repository
links must remain synchronized with that URL and license. Do not add
`date-released` to citation metadata before the corresponding release exists.
Documentation deploys from GitHub Actions to the repository's GitHub Pages
site. Its workflow uses the current stable Julia line because the tracked
`docs/Manifest.toml` was resolved with Julia 1.12.6; regenerate that manifest
before pinning documentation CI to an older Julia release. The README contains
the required disclosure of substantial Codex assistance. A maintainer must
still review and understand every release
candidate and obtain a green clean-checkout release run before registration.
Repository setup must provide `CODECOV_TOKEN` and a write-enabled,
repository-scoped deploy-key private half as `DOCUMENTER_KEY`; never commit
either secret. The public deploy-key half belongs in the repository settings.
Enable GitHub Actions to create pull requests for CompatHelper, and select the
generated `gh-pages` branch as the Pages source after the first successful
documentation deployment. TagBot reuses `DOCUMENTER_KEY` so its tags can
trigger the versioned documentation workflow.

## Population backend, spin helpers, and Schur construction

`PopulationPlan` is available only when the retained GT-basis diagonal
subspace is invariant. Its public coordinate is the multiplicity-weighted
physical probability

`p_(nu,W) = sqrt(f^nu) * (C_nu)_(W,W) = f^nu * (R_nu)_(W,W)`,

so `sum(p) == trace(rho)`. Certification is structurally strict by default:
effective `atol=rtol=0`, nonfinite generator data throw, and a weak but
nonzero coherence-generating term must never be discarded because another
term sets a larger scale. Explicit nonzero tolerances opt into approximate
projection and produce `reason=:within_tolerance` when nonzero leakage is
accepted. Fixed autonomous components are checked after their exact sum, so
exact cancellations remain valid; scalar-driven components are checked
individually. Operator-valued schedules remain inconclusive until `freeze`d.

The compile-time diagonal-coordinate reverse lookup is a dictionary with one
entry per population, not a vector of length `length(basis)`, and it is not
retained by `PopulationPlan`. Reuse a plan and `PopulationWorkspace`; do not
call `population_invariance(model)` immediately before constructing the same
plan merely to inspect its report—read `plan.invariance`. In-place Float32
evolution rejects wider floating times; `solve_populations` promotes its saved
state storage from the time grid. One workspace may be reused sequentially,
never concurrently.

`PopulationPlan` construction currently materializes the standalone
`sqrt(f^nu)` population-coordinate scale in the working real type;
model-derived plans also inherit the parent coefficient-space trace
functional. It must raise with wider-type guidance when a required standalone
scale is outside that type's finite range. Do not infer from this
plan-construction bound that `diagonal_populations` or
`state_from_populations` must also fail: those paths use prepared exact scaled
products and may remain representable.

`qubit_ensemble_model` implements the six QuTiP/PIQS channels with the package
standard dissipator `D[L]=L rho L' - {L'L,rho}/2`: local and collective
emission use `j_-`, dephasing uses `j_z=sigma_z/2`, and pumping uses `j_+`.
Thus a one-qubit coherence decays at half the `dephasing` keyword rate. Fixed
zero rates are omitted so collective-only restricted-sector models work;
callable and negative rates are retained. Fixed numeric inputs determine the
default scalar type, while callable-only models require explicit `T` for a
non-Float64 precision. Direct PI Hamiltonian prototypes must be Hermitian
unless their constructor is deliberately called with `check=false`.

`spin_matrices` uses local order `|-j>,...,|j>`. `dicke_state` and
`dicke_operator` validate half-integer labels without machine-integer
overflow and use checked exact multiplicities; `ghz_state` and
`spin_coherent_state` infer floating precision without narrowing angle data.

`each_schur_block(A; representation=:physical)` returns detached physical
blocks, while `representation=:coefficient` returns mutable stored views.
`operator_from_schur_blocks` and `state_from_schur_blocks` accept pair
iterables in arbitrary sector order, copy inputs, zero unspecified sectors,
reject duplicates and wrong sizes, and accept an optional non-narrowing `T`
(including for an empty block collection). `sector_metadata` retains exact
`BigInt` multiplicities/Hilbert dimensions and reports exact rational spin for
qubits. Never replace the checked multiplicity conversion with a direct
floating `sqrt(T(f))`.

## Schur-block visualization

`schur_block_structure` returns numerical block metadata and weights;
`visualize_schur_blocks` wraps that structure for text or SVG display, and
`save_schur_block_visualization` writes the SVG without a plotting dependency.
The public result types are `SchurBlockStructure` and
`SchurBlockVisualization`.

By default the SVG includes the Young diagram of every retained partition;
use `show_young_diagrams=false` only for a compact legacy-style layout. State
and operator views show one shape per diagonal sector, while superoperator
rows and columns show output and input shapes respectively. A sector label is
a partition shape, not a distinguished standard tableau: the PI basis sums
over that multiplicity label. Tooltips therefore report the exact `f^nu`
number of standard tableaux and never imply a canonical filling. Rendering is
bounded for large `N`: exact boxes are used only for small diagrams and larger
shapes are represented by normalized row bands.

State and operator diagrams are diagonal in the retained Schur labels. Their
default `representation=:physical` measures blocks `C_nu/sqrt(f^nu)`;
`representation=:coefficient` instead measures the stored equation-(7)
coordinates `C_nu`. The state-only `metric=:population` uses
`sqrt(f^nu)*tr(C_nu)` and rejects appreciably complex or negative sector
populations rather than repairing them.

Superoperator weights are always expressed in orthonormal PI coefficient
coordinates. Rows are output sectors and columns are input sectors. Sparse or
dense matrices are scanned directly; exact matrix-free extraction costs one
application per input PI coordinate, hence `n_PI=length(basis)` applications,
and should be treated as setup work to compute once and reuse. A driven source
requires an explicit `time`. General operator-valued fallback terms are frozen
and lowered once at that time before the coordinate-probe loop, so an
instantaneous sparse operator is not rebuilt for every probe.

Matrix-free weak-symmetry projectors use one explicit
`SymmetryProjectorWorkspace` during extraction. Frobenius and population
metrics preserve the source real precision; `:trace_norm` relies on Julia's
SVD-supported scalar types and throws a clear `ArgumentError` for unsupported
types such as `BigFloat`. Thresholds are absolute, must be representable in
the weight type, and mark values at or below the cutoff inactive without
changing raw weights.

## Density, Liouvillian, and Floquet spectral visualization

`pi_density_spectrum` is the numerical source for density-operator plots. Its
default result is multiplicity-compressed: one physical Schur-block
eigenvalue, exact `BigInt` symmetric-group degeneracy, partition, and
within-sector index per entry. `visualize_density_spectrum` must retain that
object without expanding a `d^N` eigenvalue list; its rank axis is a compressed
index and its `total_dimension` is the retained Hilbert dimension (equal to
`d^N` only for a complete basis). Negative eigenvalues remain visible and are
never clipped or repaired by presentation tolerances. The public rendering
type and writer are `DensitySpectrumVisualization` and
`save_density_spectrum_visualization`.

`liouvillian_spectrum_data` and `floquet_spectrum_data` normalize existing
spectral results into `ComplexSpectrum` without changing order, repeated
roots, or numerical values. `visualize_spectrum` creates a
`SpectrumVisualization`; `visualize_liouvillian_spectrum` and
`visualize_floquet_spectrum` are convenience routes, while
`save_spectrum_visualization` writes the same notebook SVG.

Liouvillian axes are `Re(lambda)` and `Im(lambda)`, with `Re(lambda)=0` as the
stability boundary. Floquet multiplier plots preserve equal aspect and show
the unit circle and fixed point `mu=1`. Multiplier-derived exponent plots use
the principal `Log(mu)/T` branch and show `Im(xi)=±π/T`; zero multipliers must
raise rather than be omitted or regularized. Stability classes and view limits
never clip, move, or delete stored modes.

Prefer passing an existing `SpectrumResult` or selected-solver named tuple
(ordinary/harmonic/IRAM/JD),
Floquet propagator, or multiplier vector. Rendering a `ComplexSpectrum` must
never invoke an eigensolver or integration. Precomputed selected results retain
residual, convergence, dimension, and partial-scope metadata and must not be
presented as a certified global gap. The source convenience computes once
without eigenvectors but currently retains only its returned values; call the
selected solver first when residual metadata is required. Floquet source
convenience still constructs and diagonalizes a dense PI-dimensional one-period
map; no matrix-free Floquet eigensolver is implemented.

For a computed Liouvillian source, `atol`/`rtol` are solver tolerances and
`classification_atol`/`classification_rtol` affect only marker classes. For
precomputed vectors/results, `atol`/`rtol` classify stored modes because no
solver is invoked. Residual diagnostics retain their own scalar precision.
Floquet exponents obtained from multipliers carry `metadata.branch=:principal`;
supplied exponent vectors remain verbatim and are not labeled principal.
Residuals retained across a Floquet representation conversion remain in the
input problem's units, recorded by `metadata.residual_representation`.

## Sector-resolved spin phase space

`spin_husimi_q` and `spin_wigner` accept qubit `PIState`s and operate one Schur
block at a time. For sector `nu` with spin dimension `n_j=2j+1`, use the
multiplicity-weighted block

`rho_bar_j = f^nu R_nu = sqrt(f^nu) C_nu`,

whose trace is the physical sector population. The sphere-density
normalizations are

`P_Q = n_j/(4pi) <Omega,j|rho_bar_j|Omega,j>`

and

`P_W = sqrt(n_j/(4pi)) sum_(k,q) tr(T_kq' rho_bar_j) Y_kq(Omega)`.

Each sector sphere integrates to its population. `SpinPhaseSpaceData.values`
is their angular marginal after forgetting the discrete spin label; never
describe a multi-sector state as belonging to one effective spin irrep.
`resolved=false` reuses one grid scratch matrix, while `resolved=true` retains
one matrix per selected sector. Q uses a mode-centered coherent-amplitude
recurrence; Wigner multipoles use orthonormal Condon--Shortley polarization
tensors and retain negative values. Numerical data preserve Float32/BigFloat,
construct no `2^N` object, and use no global mutable cache.

`visualize_spin_phase_space` only renders existing numerical data. It uses a
dependency-free equirectangular SVG, a sequential Q palette, and a
zero-centered diverging Wigner palette. Sector rendering requires resolved
data; regular-grid and 100,000-cell checks apply only to rendering, not to the
transform. Color limits never clip or modify stored values.

## Entanglement and reduced states

`negativity(rho,k)` has three exact paths:

1. Fully symmetric states: occupation-number branching.
2. General qubits: multiplicity-free SU(2) recoupling.
3. General qudits: product-Schur blocks using Littlewood--Richardson
   intertwiner spaces obtained as nullspaces of U(d) generator equations.

Public related APIs include:

- `negativity`, `logarithmic_negativity`
- `subduction_intertwiners`, `littlewood_richardson_coefficient`
- `ReductionPlan`, `ReductionWorkspace`, `reduced_state`, `reduced_state!`,
  `reduced_purity`, `reduced_purities`
- `quantum_fisher_information` / `qfi` for collective generators
- `quantum_fisher_information_matrix` / `qfim` for multiparameter generators
- `collective_expectation`, `collective_variance`, `collective_moments`

For a product block labeled by subsystem partitions `(alpha,beta)`, trace-norm
and reduced-state contributions must be weighted by the exact multiplicity
`f^alpha * f^beta`. Qudit LR coefficients are counted exactly with lattice
tableaux; forbidden weights are removed before sparse simple-root assembly and
SPQR nullspace recovery. The factorization and retained dense intertwiners can
still be large, but must never be replaced with full Hilbert-space
reconstruction.

For repeated work, construct `CollectiveObservablePlan` once per observable
and `ReductionPlan` once per `(basis,k)`. Both are read-only and tied to the
exact `PIBasis` object used at construction. A qudit reduction plan can retain
many dense LR intertwiners: benchmark setup and retained RAM before caching a
large collection of bipartitions. `ReductionWorkspace(plan,rho)` owns the
mutable product-block, multiplication, partial-trace, partial-transpose, and
reduced-sector buffers; use one per task. It also owns scalar-type-matched
recoupling matrices for homogeneous hot-loop multiplication. Qubit plans keep
their compact real Float64 CG matrices and convert them once per workspace;
already matching qudit LR matrices are shared read-only rather than copied.
This separation is required on Julia 1.10, whose mixed real/complex `mul!`
fallback allocates substantial packing scratch. Pass the workspace through
`workspace=` or use `reduced_state!` to reuse caller-owned output as well.

## Mean-field closure

`MeanFieldPlan(model; limit=:finite)` and the geometry-free
`MeanFieldPlan(N,d,terms; limit=:finite)` evaluate
`Tr_{2:N} L[sigma^otimes N]`. The finite rule is the exact one-body derivative
at a product state; propagation closes all generated marginals as tensor
powers. It works for qubits and qudits and never constructs a PI basis or a
`d^N` state in the direct-constructor path.

Supported optimized terms are fixed one-body Hamiltonians, symmetric
`PBodyHamiltonian` terms, local one- and p-body jumps, and collective one-body
jumps. `CollectivePBodyJump` at arbitrary supported body order uses exact
ordered-subset overlap classes in finite mode and the leading disjoint class
in thermodynamic mode. Direct PI terms, operator-valued functions, and custom
terms without explicit microscopic lowering must raise. Scalar time-dependent
rates remain supported and may be negative.

Finite p-body rules use `binomial(N-1,p-1)`. Thermodynamic mode uses the
leading `N^(p-1)/(p-1)!`; for collective one-body jumps it replaces `N-1` by
`N` and drops the subleading one-site dissipator. Never infer or insert a Kac
factor: rates are used exactly as supplied. Product collective moments and
p-body expectations omit connected correlations.

`MeanFieldPlan` owns copied operators and precision-appropriate combinatorial
factors. `MeanFieldWorkspace` owns `d^p` by `d^p` contraction matrices and RK4
stages. The warmed explicit RHS is allocation-free for ordinary floating
types. Reject incompatible state, destination, or evaluated-rate precision
instead of silently narrowing it. Fixed-point relaxation is basin dependent,
requires an autonomous plan, and must not return an unconverged default state.
The Jacobian is real on the `d^2-1` traceless-Hermitian tangent space.

## Dynamics, trajectories, spectra, and symmetries

Compile a model once before repeated evolution. Deterministic evolution accepts
sparse or matrix-free Liouvillians. Reuse `EvolutionWorkspace` with `evolve!`
for repeated fixed-step RK4 propagation; use `solve_dynamics` for the typed
high-level fixed-step result and `dynamics_problem` when adaptive or stiff
SciML integration is required.

`steady_state(...; method=:krylov)` uses restarted GMRES on a rank-one
trace-fixed system and never materializes the Liouvillian. Reuse
`KrylovWorkspace`. `pi_liouvillian_spectrum/gap(...; method=:krylov)` use
matrix-free Arnoldi; converge Ritz residuals by increasing `krylovdim`.
Partial spectra cannot certify stationary multiplicities that fill the entire
requested eigenvalue window.

`method=:harmonic` performs thick-restarted harmonic extraction near zero;
`method=:iram` performs exact-shift implicit-QR Arnoldi at the spectral edge.
`jacobi_davidson_spectrum` (and spectrum `method=:jd`) supplies hard locking
and a reusable preconditioned correction workspace for difficult interior
clusters. The global gap API accepts largest-real IRAM, while JD is rejected
there because nearest-target selection cannot certify a global spectral edge.
`matrixfree_symmetry_projector` restricts it to a unitary conjugation charge
using sector-local Schur blocks; no symmetry superoperator or reduced
Liouvillian matrix is built. The resulting gap is charge-resolved and may
differ from the global gap. Antiunitary symmetries do not define these
complex-linear projectors.

Reuse `ArnoldiWorkspace` for repeated ordinary, harmonic, or IRAM solves. A
`MatrixFreeSymmetryProjector` has a synchronized compatibility workspace;
parallel hot loops must instead use one explicit `SymmetryProjectorWorkspace`
per task. Harmonic reports include restart history, retained/locked counts,
and scale-aware residual tolerances; convergence is still the caller's
responsibility for a partial nonnormal spectrum.

The dominant Krylov and Floquet work arrays use precision derived from the
Liouvillian and storage-bearing numerical inputs. A fully `Float32` problem
retains `ComplexF32` bases and stage arrays;
wider inputs promote compatible materialized and custom operators. Compiled
matrix-free PI plans own fixed-precision application scratch, so a wider
source, solver input, period, or time origin must raise before allocating
large solver storage; compile the model at the wider precision instead. A
narrow destination or incompatible explicit workspace must also raise rather
than truncate. Integer Floquet times are accepted only when exactly
representable in the selected precision.

`floquet_gap(F,T;atol=...)` must identify a multiplier within `atol` of one
and reject a remaining spectral radius larger than `1+atol`; it must never
delete an unrelated closest root or hide a genuine instability. A
one-dimensional map returns a scalar-precision `NaN` because it has no
subleading mode.

`schur_sector_preconditioner(L,basis)` builds reusable LU factors of the
diagonal Schur-sector blocks of the trace-fixed operator. Use
`preconditioner=:schur` for one-off construction, or pass the object explicitly
for scans. Its setup uses matrix-free applications and its final solution is
always validated with the unpreconditioned Liouvillian residual. Local terms'
off-sector couplings are deliberately omitted only from the preconditioner.
Inspect `preconditioner_cost(P)` before a scan: setup needs roughly one
Liouvillian application per PI coordinate plus scale probes and normally does
not amortize for a one-off solve.

Public dynamics and stochastic APIs include:

- `time_evolve`, `time_evolution`, `evolve!`, `EvolutionWorkspace`
- `floquet_propagator`, `floquet_steady_state`, `stroboscopic_evolution`
- `quantum_trajectory`, `quantum_trajectories`, `trajectory_average`
- `jump_statistics`, `trajectory_observable_statistics`, `trajectory_statistics`

PI trajectories use channel-resolved gain maps. Local particle labels are
unresolved, so an individual local-jump trajectory can be mixed even though
the ensemble converges to the PI master equation. Rates must be nonnegative
for stochastic evolution. `algorithm=:fixed` uses preallocated RK4 with a
maximum jump probability; `algorithm=:event`/`:adaptive` integrates the state
and hazard with Dormand--Prince 5(4) and locates continuous jump times by root
solving. Reuse one workspace sequentially or one per thread; convergence-test
fixed steps or adaptive tolerances and the ensemble size.

Public spectral and algebraic-diagnostic APIs include:

- `pi_liouvillian_spectrum`, `pi_density_spectrum`
- `pi_liouvillian_gap` / `liouvillian_gap`
- `evans_uniqueness`, `has_unique_steady_state_evans`
- `check_liouvillian_symmetry`, `is_liouvillian_symmetric`
- `usual_liouvillian_symmetries`

Density spectra retain exact Schur multiplicities in compressed form by
default. Complete Liouvillian spectra are dense PI-dimensional operations.
The gap routine removes the full numerical stationary cluster and can exploit
a supplied unitary weak symmetry or `symmetry=:auto`; antiunitary symmetry
alone does not define complex-linear charge blocks.

Evans reports use three outcomes: `true`, `false`, or `missing` when the
efficient theorem-based model test is inconclusive. Do not turn `missing` into
a uniqueness or non-uniqueness claim. Weak symmetry checks test covariance of
the Liouvillian, not strong term-by-term commutation.

## Published-model mapping

`examples/paper_models.jl` contains reusable constructors.

- PRA 94, 033838 (2016): correlated emission decomposes as
  `gamma*D[J_-] + (gamma0-gamma)*sum_i D[sigma_-^(i)]`.
  `examples/pra94_033838_superradiance.jl` compares Fig. 6 with equations
  (41)--(43), reaching approximately machine precision.
- PRA 110, 062208 (2024): the dissipative LMG Hamiltonian is constructed as a
  `DirectPIHamiltonian` proportional to `Jx^2-Jy^2`; individual and collective
  rate prefactors follow equation (3). The finite-N result must not be claimed
  to equal the thermodynamic mean-field curve.
- Additional examples cover Morrison--Parkins cooperative resonance
  fluorescence (including its exact steady state), Meiser--Holland
  steady-state superradiance, and four complementary time-crystal workflows:
  Iemini's free-spin BTC, Piccitto's interacting `p=2,q=1` BTC,
  Nakanishi--Sasamoto's exactly solvable balanced-gain/loss PT model, and the
  Gambetta dissipative Floquet-Rydberg protocol.
- `examples/meanfield_time_crystal.jl` compares the finite product closure,
  thermodynamic closure, and exact matrix-free PI dynamics for the balanced
  Nakanishi--Sasamoto model. Its finite/exact agreement is model specific and
  must not be generalized to correlated PI dynamics.
- Nakanishi--Sasamoto use `D_paper=2D_std`; their balanced symmetric-sector
  spectrum and gap `4kappa/N` are exact finite-size regression oracles.
  Piccitto use normalized Pauli magnetizations, so `Jz^2` lowers to a pair
  Hamiltonian of rate `-2omega_z/N` up to an identity, and `J+/-` give package
  collective rates `4Gamma_up/down/N`.
- Gambetta's written `sum_{i!=j}` interaction is an ordered-pair sum and maps
  to `PBodyHamiltonian(...; rate=2V0/N)`. Their caption has a documented
  mean-field-normalization ambiguity. The example follows the literal
  Hamiltonian, validates the `N=4` multiplier independently, and calls it only
  a finite-size precursor; published time-crystal claims require size scaling.

- `examples/qubit_population_dynamics.jl` validates the strict reduced
  population backend against full PI evolution and stationary solving for the
  six-rate qubit model. `examples/spin_phase_space.jl` validates multi-sector
  Q/Wigner normalization, exact multiplicities, coherent peaks, Wigner
  negativity, and dependency-free rendering.

Every runnable `examples/*.jl` file has a same-basename Markdown guide. Keep
the code, stated tolerances, and guide workflow synchronized. Current examples
compile each model once and prefer `solve_dynamics`, `stationary_state`, typed
algorithm/result objects, and prepared observable/reduction plans. Retain
low-level `liouvillian`, `apply!`, dense exponentiation, or complete spectra
only when the example explicitly validates a backend, solver, or published
small-system formula; explain that choice in the paired guide. Never replace a
published analytical or finite-size assertion merely to demonstrate a newer
API.

## Verification workflow

From the repository root:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
julia --project=. examples/pra94_033838_superradiance.jl
julia --project=. examples/pra110_062208_lmg.jl
julia --project=. examples/pbody_pair_processes.jl
julia --project=. examples/steady_state_methods.jl
julia --project=. examples/floquet_periodic_decay.jl
julia --project=. examples/quantum_trajectories.jl
julia --project=. examples/piccitto2021_interacting_time_crystal.jl
julia --project=. examples/nakanishi2023_pt_time_crystal.jl
julia --project=. examples/gambetta2019_dissipative_discrete_time_crystal.jl
julia --project=. examples/meanfield_time_crystal.jl
julia --project=. examples/schur_block_visualization.jl
julia --project=. examples/spectral_visualization.jl
julia --project=. examples/qubit_population_dynamics.jl
julia --project=. examples/spin_phase_space.jl
julia --project=. benchmark/performance_regression.jl
julia --project=. benchmark/performance_audit.jl
julia --project=docs docs/make.jl
julia --project=quality -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
julia --project=quality quality/quality.jl
```

The latest complete single-thread suites had **4037 passing tests** on both
Julia 1.10.11 and Julia 1.12.6 (2026-07-15), including exact large
combinatorics, scaled Schur
branch/path factors, stable large-`N` product amplitudes, the strict population
backend, six-rate qubit model, spin/state conveniences, Schur construction
helpers, and spin phase space. Treat that count as historical after further
changes and report actual command output. The strict documentation build and
its zero-undocumented-export gate passed after these additions. Aqua's 11
package gates, JET's three
public hot-path checks, and the public method-ambiguity check passed in the
preceding quality audit. All 22 example scripts have same-basename guides; the
new population-dynamics and spin-phase-space examples passed individually in
this integration pass, alongside the previously audited published-model,
mean-field, time-crystal, Schur/density-, and complex-spectrum examples.
Regression groups
cover exact combinatorics, CG orthogonality, PI algebra/generators, Appendix-D,
finite/thermodynamic mean-field closure and precision, Floquet and direct
evolution, trajectories/statistics, published models,
negativity and reduced states, QFI/QFIM, information measures, spectra, Evans
certificates, weak symmetries, symmetry-reduced gaps, response tools, and
Schur construction/metadata, strict population dynamics, spin conveniences,
spin phase space, Schur-block, compressed-density-spectrum, and complex-
spectrum extraction/rendering.

`benchmark/performance_audit.jl` is the dependency-free global time/RAM audit
with sparse-versus-matrix-free precision guards, including Appendix-D kernels.
`benchmark/benchmarks.jl` provides the more detailed BenchmarkTools suite.
`benchmark/performance_regression.jl` is the stable allocation, equivalence,
and shared-operator thread-safety gate; it deliberately avoids brittle wall
clock thresholds. The latest four-thread regression and global audit confirm
warmed explicit-workspace Liouvillian, dynamic local p-body forward/adjoint,
population application, and mean-field application are allocation-free, and
both full and reduced Liouvillian actions pass precision guards. Its current
allocation gates report 0 B for explicit vector application, population
application, and mean-field RHS; 128 B for a preallocated population RK4
evolution; 960 B for each three-column forward/adjoint batch; 11,648 B for
prepared collective moments; 127,960 B for plan-only reduction; and 31,552 B
for `reduced_state!` with both workspace and output reused. The global audit
also covers population-plan setup and Husimi-Q/Wigner transforms. The residual
reduction allocation is dominated by state validation/LAPACK scratch; do not
remove it through an implicit trust or positivity opt-out. For `n=1000,m=40`,
retained Float32 workspace sizes are 50.05%, 50.01%, and 50.02% of Float64 for
GMRES, Arnoldi, and Jacobi--Davidson respectively. The dominant remaining setup
cost is front-loaded CG/Schur geometry and sparse LR factorization; reuse
explicit plans and workspaces rather than adding global mutable caches.
Exact `BigInt`/rational work remains setup-only and is not retained in hot
Liouvillian workspaces. Large-`N` stability fallbacks are guarded so the
ordinary small-system kernels keep their native floating-point algorithms.
The matching Julia 1.10.11 four-thread gate reports 31,936 B for
`reduced_state!` after the workspace recouplers are converted once to the
working complex type; never restore mixed real/complex workspace `mul!` calls.

When changing representation theory:

1. Add a minimal identity/orthogonality test first.
2. Compare against an analytical qubit case.
3. Add a small qudit case with a known result.
4. Run the complete suite.
5. Update documentation and `IMPLEMENTATION_NOTES.md`.

## Closure status and bounded limitations

The previous closure checklist is implemented and regression-tested:

- A deliberately exponential recursive dense Schur transform exists only as a
  small-system test oracle; production `iid_state` uses a sector recurrence and
  accepts singular qubit/qudit inputs without logarithms or regularization.
- Large binomial/multinomial metadata are exact `BigInt`; Schur square roots,
  CG ratios, one-body branches, and Appendix-D path factors convert only after
  exact cancellation and binary scaling. Exact factors are fused with small
  numerical values before conversion, with a direct native branch for
  ordinary sizes. `iid_pure_state` uses normalized conditional-binomial
  recurrences and is regression-tested through symmetric `Float64` `N=1100`
  without forming an overflowing multinomial coefficient; accepted one-site
  normalization error is checked after amplification to `N` copies.
- PI operator products keep the ordinary BLAS-first block path. If their stored
  `C_A*C_B` intermediate overflows before division by `sqrt(f^nu)`, use the
  fused exact-scale fallback; guarded widening is only for a severely
  cancelled dot product or complex component in that exceptional block.
  Physical visualization norms similarly scale
  entries before aggregation only after the direct aggregate-first path fails.
- Entropy, Rényi entropy, distances, relative entropy, collective moments,
  QFI/QFIM, symmetry-resolved information, reduced states, negativity, and
  qubit phase space use multiplicity-weighted Schur blocks. Fixed-spin tests
  through `N=2100` cover sectors for which `sqrt(f^nu)` exceeds Float64.
  Numerical-rank cutoffs are sector relative. A per-copy physical block or
  purity that is itself outside the selected type's nonzero range raises.
  Spectral analyses validate PSD before making a numerical-rank decision.
  Mapping an accepted nonpositive roundoff eigenvalue to zero only inside a
  square root, logarithm, or support projector is an internal rank projection;
  it must never alter the input or a returned spectrum. Fidelity may move a
  computed endpoint into `[0,1]` only within its documented scale-aware
  roundoff bound and otherwise raises. Relative entropy tests support with one
  sector-level projector onto sigma's numerical nullspace; do not reintroduce
  per-eigenvalue rho or per-overlap cutoffs that can hide positive nullspace
  weight.
- General-qubit reduction keeps the small factorial Racah path through doubled
  spin 32 and switches above it to an exact-rational consecutive-term Racah
  recurrence. `ReductionWorkspace` owns prepared exact parent scales and a
  parent-block buffer; repeated in-place reductions must not rebuild BigInts.
- One-body geometry stages content-compatible child/parent transitions before
  evaluating CG coefficients. Its estimator counts qudit weight
  multiplicities structurally and reports conservative retained/setup bounds.
  Model validation detects complete bases from the exact Schur--Weyl PI
  dimension and generates local p-body closure from relevant Young-lattice
  ancestors and descendants; never restore an all-partition scan for a
  restricted basis.
  Large-`N` collective blocks, static p-body blocks, and static local p-body
  gain groups selectively recompute cancellation-prone data with guarded wider
  precision. Local-gain checks include severe sums of individually
  representable direct path-pair factors. A forward-error interval is not an
  exact-zero certificate: if a guarded-wide static sum remains unresolved, it
  raises rather than deleting the coordinate. A risky dynamic p-body block
  applies the same final-ulp certification and raises with guidance to widen
  its `InPlaceTimeOperator` prototype; dynamic local gains are rejected at
  compilation because their fixed contraction scratch cannot be widened
  safely. `mean_local_operator` scales physical blocks directly by
  `sqrt(f^nu)/N`, with an ordinary native branch, and must not materialize the
  extensive stored collective operator first.
- Public `local_kernel_element` risk-gates large-`N` one-box cancellation and
  selectively rebuilds only its retained endpoint sectors at wider precision;
  keep its ordinary loop unchanged. Its final `::S` assertion records the
  already-established scalar contract and is needed for Julia 1.10 to infer
  through the data-dependent guarded-wide branch. Preallocated one-body
  schedules instead reject a risky native geometry at compilation with
  wider-prototype guidance.
  Dynamic direct-PI blocks use prepared fused inverse Schur-multiplicity scales
  when `sqrt(f^nu)` is not representable, while ordinary sectors retain direct
  division without per-call exact-combinatoric setup.
- `InPlaceTimeOperator` provides caller-workspace operator schedules for every
  built-in local, collective, direct-PI, and Appendix-D term. Raw untyped
  functions remain the intentionally allocating compatibility path. Dynamic
  local p-body gain maps retain rectangular path-pair contractions and apply
  them blockwise as `C*X*C'`; never reintroduce the quadratic PI-coordinate
  `I/J/value` table into their plan or workspace.
- Large accepted Schur blocks use shifted Cholesky positivity certificates;
  LR construction uses exact tableau counts, weight restriction, sparse
  constraints, and SPQR.
- Exact-shift IRAM and hard-locking preconditioned Jacobi--Davidson complement
  harmonic Arnoldi, matrix-free weak-symmetry projection, and Schur-sector
  GMRES preconditioning.
- The auxiliary-PI Evans/Davies test handles combined built-in local,
  collective, Hamiltonian, and p-body generators; adaptive event-driven
  trajectories use continuous hazard roots; collective p-body mean field uses
  exact overlap combinatorics.
- Aqua and JET run in an isolated CI quality environment. Analysis routines
  preserve Float32 where their numerical backend supports it and reject an
  unavailable generic eigensolver explicitly rather than narrowing silently.
- Strict diagonal-population certification lowers autonomous cancellations
  exactly and time-dependent rates independently, then evolves only the
  population-dimensional vector. The six-rate qubit constructor, spin/state
  helpers, physical/coefficient Schur iterators and constructors, and
  multiplicity-aware spin Husimi-Q/Wigner analysis are public and regression-
  tested without full-Hilbert reconstruction.

Bounded research-scale limits remain: sparse-SPQR LR can exhaust memory for
very large qudit irreps; a single huge Schur block still needs dense Cholesky;
Evans reports `missing` for unsupported direct/custom microscopic recoupling or
an exceeded memory budget; CG geometry retains its documented Float64 phase
convention; and the Floquet eigensolver still materializes the PI-dimensional
one-period map. Spin phase-space transforms are currently qubit-only, while a
qudit phase-space convention remains future work. A coefficient-space trace
vector cannot be stored in a scalar type when `sqrt(f^nu)` itself exceeds that
type; Liouvillian construction then raises instead of retaining `Inf`, and the
model must use wider coefficients. Public memory estimates use exact inline
`sizeof(T)` accounting for fixed-size isbits scalars, but conservative
precision-aware retained-storage bounds for `BigFloat` and
`Complex{BigFloat}`. Other heap-backed types are explicitly sample-based, not
worst-case bounded. `recommend_solver` includes one-body geometry exactly for
model/compiled-model inputs and conservatively for inputs without term
provenance; preserve its reported assumption metadata. See `IMPLEMENTATION_PLAN.md`
and `IMPLEMENTATION_NOTES.md`.

## Maintenance rules

- Keep hard representation-theory code explicit and commented; avoid opaque
  metaprogramming.
- Apply selection rules before allocating or looping.
- Caches belong to a basis/model/operator or a call-local cache; never add an
  unprotected global mutable cache.
- Restricted bases must error when a requested local process reaches missing
  sectors.
- Negative time-dependent rates are valid input. Document that the resulting
  map need not be completely positive.
- Give randomized Krylov and symmetry tests an explicit initial vector and a
  fresh deterministic RNG for each compared call. Harmonic-Arnoldi breakdown
  recovery may consume the RNG even when the initial vector is fixed.
- If an equation or phase convention is uncertain, re-read the paper and add a
  small mathematical test. Never choose a convention only to satisfy a test.
