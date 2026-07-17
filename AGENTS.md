# AGENTS.md

This file is the continuity guide for agents maintaining
`PermutationalInvariantDynamics.jl`. Read it before changing source code.

## Project purpose

The package simulates permutationally invariant (PI) open-system dynamics of
`N` identical `d`-level systems directly in the PI operator subspace.
Time-local generators are the core model; finite-exponential PI--HEOM extends
the same coordinates to selected non-Markovian bosonic environments.
Production algorithms must not construct objects of size `d^N` or `d^(2N)`.

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
- Weak dependencies are `Distributed`, `Tables`, `Makie`,
  `QuantumCumulants`, `JLD2`, and `HDF5`. Keep their methods in package
  extensions. Core source must not import them directly. `Distributed` is a
  stdlib that SciMLBase may load transitively, so its extension can activate
  during core loading even without an explicit user `using Distributed`;
  never treat extension activation as proof that the user requested remote
  workers. The other weak dependencies must not be loaded by core source.
- Extension compatibility currently covers Tables 1, Makie 0.21--0.24,
  QuantumCumulants 0.5, JLD2 0.4--0.5, and HDF5 0.16--0.17; Distributed
  follows the supported Julia 1.10+ stdlib line. Update Project compat and
  optional smoke tests together.
- Do not commit a root `Manifest.toml`. `Pkg.instantiate()` and `Pkg.test()` may
  recreate it locally; remove it before handing work back.
- The isolated quality setup may similarly create `quality/Manifest.toml`;
  remove it after running Aqua/JET. Keep the tracked `docs/Manifest.toml`.
- The optional CairoMakie setup creates `examples/Manifest.toml`; it is ignored
  and must likewise be removed before handoff.
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
- `src/gtpatterns.jl`: immutable GT patterns, allocation-free content tuples,
  and recursive enumeration.
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
- `src/correlated_jumps.jl`: correlated local/collective one-body reservoirs,
  strict Kossakowski validation, and generic residual-Cholesky factorization.
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
- `src/krylov_extensions.jl`: block, shared-Arnoldi multi-shift, and recycled
  GMRES plus adaptive restarted exponential actions and their reusable
  workspaces.
- `src/spectra.jl`: complete PI Liouvillian spectra and multiplicity-compressed
  density-operator spectra.
- `src/evans.jl`: Evans commutant uniqueness certificates and efficient PI
  model specializations.
- `src/symmetries.jl`: weak unitary and antiunitary Liouvillian covariance
  checks in full or PI Liouville space, plus simultaneous commuting
  matrix-free symmetry projectors.
- `src/observables.jl`: expectations, RDM and diagnostics.
- `src/cumulants.jl`: exact distinct-particle local moments, versioned neutral
  microscopic-model payloads, closure comparisons, and the optional
  QuantumCumulants adapter entry point.
- `src/entanglement.jl`: bipartite negativity, LR/subduction intertwiners,
  reduced-state purities, and preallocated reduction workspaces.
- `src/information.jl`: entropy and PI state-distinguishability measures.
- `src/symmetry_information.jl`: sector decompositions, coherence, asymmetry,
  and symmetry-resolved Fisher information.
- `src/sciml.jl`: in-place `ODEProblem` and solution wrapper.
- `src/meanfield.jl`: finite product-state closure, explicit thermodynamic
  combinatorics, preallocated one-site evolution, product observables, fixed
  points, and traceless-Hermitian stability analysis.
- `src/composite.jl`: tensor products of independent PI operator spaces and
  finite auxiliary operator spaces, exact composite trace contractions,
  ownership-safe construction, and preallocated sums of factorized
  superoperators with dense contiguous-factor GEMM batching.
- `src/evolution.jl`: preallocated direct density-matrix propagation from an
  assembled or matrix-free Liouvillian.
- `src/heom.jl`: unscaled finite-exponential bosonic PI--HEOM plans,
  matrix-free application, fixed-step evolution, depth convergence, and
  trace-fixed stationary solving.
- `src/trajectories.jl`: PI quantum-jump trajectories, shared immutable
  trajectory plans, task-owned batch workspaces/RNGs, direct channel-intensity
  contractions, ensembles, and density-matrix averaging.
- `src/composite_trajectories.jl`: explicit tensor-product monitored jumps,
  density-valued composite conditional evolution, multiplicity-aware prepared
  traces, shared channel buffers, and reproducible task-owned batches.
- `src/weak_pi_trajectories.jl`: opt-in direct-sum Schur-irrep pseudo-kets,
  one-box Kraus subduction of local gains, preallocated fixed-step paths,
  batches, direct allocation-light ensemble reconstruction, and
  sector-transition statistics.
- `src/diffusive.jl`: preallocated collective homodyne/heterodyne conditional
  dynamics with shared heterodyne operator products, trajectory-indexed
  ensembles, observable output, and state averaging.
- `src/adaptive_ensembles.jl`: confidence-controlled state-free quantum-jump
  and diffusive ensembles with deterministic batching and online statistics.
- `src/distributed_api.jl`: extension entry points for process-parallel scans
  and stochastic ensembles; implementations live in the Distributed
  extension.
- `src/floquet.jl`: preallocated one-period maps, multipliers, periodic states,
  and stroboscopic evolution.
- `src/response.jl`: modes, resolvents, adjoint evolution and sensitivities.
- `src/correlations.jl`: prepared PI quantum regression, delayed intensity
  correlations, matrix-free shifted-GMRES spectra, and dependency-free
  finite-window FFT transforms.
- `src/highlevel.jl`: typed algorithms/results, unified research commands,
  memory estimates, solver recommendations, diagnostics, and compact displays.
- `src/scans.jl`: prepared steady-state/spectral parameter scans, continuation,
  resumable result records, deterministic threaded workers, and dependency-
  free tabular views.
- `src/convergence.jl`: explicit refinement studies for time steps, Krylov
  dimensions, hierarchy depths, and sector cutoffs.
- `src/populations.jl`: strict Schur-diagonal invariance certificates,
  population-only generators, preallocated evolution, and stationary solves.
- `src/research_utilities.jl`: compressed spectral/population inspection and
  shared validation helpers for the research-utility APIs.
- `src/channels.jl`: explicit and matrix-free PI channels, composition,
  adjoints, Kraus lowering, and retained-algebra CP/TP diagnostics.
- `src/tomography.jl`: PI POVM probabilities, sampling, and constrained
  maximum-likelihood tomography in retained coefficient coordinates.
- `src/checkpoints.jl`: versioned PI-state checkpoint payloads and optional
  HDF5/JLD2 storage extensions.
- `src/control.jl`: trace-fixed implicit steady-state gradients and
  checkpointed continuous-adjoint control gradients.
- `src/phase_space.jl`: sector-resolved qubit Husimi-Q and Agarwal spin-Wigner
  data without full-Hilbert reconstruction.
- `src/qudit_phase_space.jl`: prepared generalized `U(d)` coherent-state
  Husimi-Q data in selected Schur sectors.
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

Higher-order closure validation has a dependency-neutral flow:

`(PIModel, PIState, local alphabet) -> CumulantBridgePayload -> external closure`.

Certified population dynamics has a reduced flow available only when the
Schur-diagonal subspace is invariant:

`PIModel -> PopulationPlan -> PopulationWorkspace -> solve_populations`.

Composite deterministic dynamics has a separate factorized flow:

`(PIBasis/FiniteOperatorBasis factors) -> CompositePIBasis ->`
`CompositeSuperoperator -> CompositeSuperoperatorWorkspace -> evolve!`.

Composite density-valued stochastic dynamics uses:

`(CompositeSuperoperator background, CompositeJumpChannels) ->`
`CompositeTrajectoryPlan -> CompositeTrajectoryWorkspace ->`
`quantum_trajectory/quantum_trajectories`.

Prepared parameter studies use:

`(parameter grid, model builder) -> ParameterScanPlan ->`
`ParameterScanWorkspace -> parameter_scan/resume_parameter_scan`.

Finite-memory bosonic environments use:

`(PI system, HEOMBaths) -> HEOMPlan -> HEOMWorkspace/`
`HEOMEvolutionWorkspace -> apply!/heom_evolve/heom_steady_state`.

The first composite factor is the fastest coordinate. A factorized vector is
therefore `kron(x_last,...,x_first)`, and a factorized map has the reversed
Kronecker order. Never materialize that global Kronecker matrix in production.

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

`docs/src/getting_started.md` is the canonical task-oriented onboarding path:
PI applicability, local basis and matrices, physical terms, `PIModel`, initial
state, compilation, dynamics, stationary solving, result access, diagnostics,
and convergence. Keep its complete script synchronized with
`examples/getting_started.jl` and the compact Home-page preview. Beginner spin
examples use `spin_matrices()` in the package order `(|g>,|e>)` so the sign of
`jz` is never hidden by a hand-written Pauli convention.

`docs/src/framework.md` is the self-contained conceptual introduction: PI
covariance, Schur--Weyl sectors, equation-(7) normalization, scaling, physical
terms, the prepared workflow, and validity limits. Keep it suitable for a new
research user and keep its runnable qubit example synchronized with the
high-level API.

`docs/src/api_reference.md` is the complete alphabetical entry point. Detailed
descriptions are split into the explicit public-only pages under
`docs/src/api/` plus the streaming, diffusive-monitoring, weak-PI trajectory,
quantum-regression, cumulant-bridge, research-utilities, composite-system,
prepared-scan, advanced-Krylov, convergence, PI--HEOM, qudit-phase-space, and
optional-interoperability pages. Every exported binding must have a source
docstring so the website and Julia's `?name` help remain identical.
`docs/make.jl` enforces both
`Base.Docs.undocumented_names(...; private=false) == []` and Documenter's
`checkdocs=:exports`; adding an export therefore requires adding its docstring
and one canonical `@docs` entry. Qualify names that conflict with Base, such as
`PermutationalInvariantDynamics.isvalid`.

Keep TeX in tracked Markdown compatible with GitHub's math renderer. In
particular, GitHub rejects the `operatorname` command; use `\mathrm{tr}`, `\mathrm{Re}`,
`\mathrm{diag}`, and analogous roman labels instead, adding `\,` before a bare
argument when operator spacing would otherwise be lost. Apply the same rule to
source docstrings because Documenter may copy their mathematics into generated
pages.

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

`CorrelatedLocalJumps` and `CorrelatedCollectiveJumps` implement a Hermitian
positive-semidefinite Kossakowski matrix over fixed one-body operator
channels. Fixed matrices are copied, validated, and residual-Cholesky
factorized exactly once when the term is constructed; compilation lowers the
resulting effective channels through the existing local or collective PI
kernels. Do not diagonalize any `d^N` object or refactor a fixed channel
matrix during application. Raw matrix functions use the freeze/lower fallback.
An `InPlaceTimeOperator` Kossakowski schedule instead owns evaluated matrix,
factor, residual, effective-operator/block, and gain scratch in each
`LiouvillianWorkspace`; validate finiteness, Hermiticity, and PSD at every
evaluation. Plans remain read-only. Preserve every strictly positive pivot:
only nonpositive residuals at explicit user tolerance or arithmetic roundoff
may reduce numerical rank. The common scalar rate is independent of Gamma;
it must evaluate to a finite real number representable in the prepared
precision, and deterministic evolution still permits a negative time-local
rate.

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

## Sector-resolved spin and qudit phase space

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

`QuditHusimiPlan` generalizes only the coherent-state Husimi-Q transform. A
point is a local unitary `U` or a Hermitian generator `H` representing
`U=exp(-im*H)`. The same local transformation is lifted into every selected
Schur irrep and applied to its first, extremal GT vector. Sector `nu` uses

`Q_nu(U) = dim(U_nu) * <nu,U|sqrt(f^nu) C_nu|nu,U>`,

so normalized Haar integration returns that sector's physical population.
The `U(d)` parametrization is redundant because the fiducial vector has a
stabilizer; `QuditHusimiData.values` is indexed by the user-supplied point
order and is not a canonical low-dimensional coordinate chart.

A qudit plan is read-only, tied to the exact `PIBasis`, and retains one dense
coherent-vector matrix of size `dim(U_nu) * npoints` per selected sector.
Setup also constructs dense lifted generators and exponentials per selected
sector and point; reuse the plan across states and benchmark this setup before
a large orbit sample. `resolved=true` additionally retains a
`nsectors * npoints` value matrix. Unitary inputs use LAPACK Schur and are
limited to Float32/Float64; use `representation=:generator` for other
supported scalar types. The plan and state precision must combine without
narrowing. There is no generalized qudit Wigner transform or dependency-free
manifold renderer; the optional Makie conversion plots supplied point index
against the already computed Q value. For `d=2`, normalized-Haar Q is `4pi`
times the package's spin-sphere density at the corresponding coherent point.

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

## Higher-order cumulant bridge

`ordered_local_moment` evaluates a standard `tr(rho*A1^(1)*...*Ak^(k))`
moment on distinct particles.  It symmetrizes only the local `d^k` tensor,
contracts Appendix-D blocks with multiplicity-weighted state blocks, and
normalizes by the exact `binomial(N,k)` subset count.  Never replace this with
a `d^N` reconstruction.  `ordered_local_moments` stores one canonical multiset
key per PI-equivalent operator assignment and reuses one `PBodyGeometry` per
order.  Lookup order is immaterial, but products acting on one site must be
provided as one local matrix rather than mistaken for distinct-site moments.

`CumulantModelPayload` and `CumulantBridgePayload` use neutral schema version
`1.0.0`.  Adapters must check that version, preserve the particle-one-fastest
tensor convention and standard dissipator, and refuse to invent a microscopic
realization for a term marked `microscopic=false`.  An unevaluated allocating
operator schedule has no prototype; pass `time` and `parameters` to evaluate
it before symbolic lowering.  Payload matrices are detached copies.

QuantumCumulants is a weak dependency restricted to its supported 0.5 API
line. `quantumcumulants_initial_values` maps exact neutral keys onto
user-supplied symbolic averages and validates them with the official
`get_order` function. `quantumcumulants_model` additionally performs an
explicit optional-adapter lowering of microscopic `PIModel` terms: it creates
an `NLevelSpace`, distinct indices, permutation-symmetric p-body operators,
local/collective jump sums, unordered-subset factors, and fixed correlated
channels before calling the official `meanfield` workflow. Keep all such
symbolic construction in the extension, never in core.

Automatic lowering must reject direct PI terms because Schur blocks do not
specify a unique microscopic operator. Operator/rate schedules require an
explicit evaluation time and parameters; p-body matrices must be permutation
symmetric; custom seed operators remain the user's responsibility when the
default local transition set is too large. `complete` and `scale` are explicit
QuantumCumulants choices, and their output is still a selected-order cumulant
approximation. The exact PI moment backend remains independent of `d^N` but
necessarily retains `d^(2k)` local tensor data, so closure order is a bounded
research-scale parameter.

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

## Prepared scans, advanced Krylov, and convergence evidence

`ParameterScanPlan` owns a copied parameter container, a model builder or
prototype/remaker, immutable solver choices, and no compiled numerical
scratch. Every evaluated point must return a `PIModel` or `CompiledPIModel`;
models are compiled per point because parameter-dependent prepared kernels
cannot generally be reused. `ParameterScanWorkspace` owns continuation and
GMRES/Arnoldi scratch and is task-local. Reuse it sequentially only.

Every public `parameter_scan` invocation starts a fresh path: clear the old
continuation seed but retain compatible solver storage. Only
`resume_parameter_scan` may install a validated checkpoint seed. If a resume
extension is empty or fails before a newer success, retain the previous valid
prefix checkpoint. A checkpoint made with `save_restart=false` resumes cold,
regardless of stale workspace contents. Resume/merge validation requires the
same continuation, output, vector, and restart retention flags. Enforce
`restart_seed===nothing => restart_index==0`.

Serial continuation passes a preceding compatible state or deterministic
combination of selected Ritz vectors. Compatibility requires the exact
particle/local dimensions, retained partitions, PI coordinate dimension, and
scalar type. A failed point clears the warm start. Restart seeds are copied at
every workspace/result/resume/merge ownership boundary, so modifying a public
result cannot corrupt reusable scratch. `save_outputs=false` streams the live
output through a callback and retains only scalar records plus at most one
restart seed. Callbacks return only `nothing` or `:stop`; failures follow the
explicit `:stop`, `:record`, or `:throw` policy and interrupts are never
converted into failed parameter points.

Threaded scans require `continuation=false`. A bounded worker pool owns one
workspace per worker, uses index-derived random streams, and acknowledges
ordered results so the callback reorder buffer stays `O(nthreads)`. Builders,
remakers, and diagnostics must themselves be thread safe. The Distributed
extension assigns independent indexes to deterministic balanced contiguous
chunks and also forbids continuation. Its closures must serialize and every
worker must activate a compatible package environment. All remote chunks are
computed before master-side `on_error` or callback stopping is applied; this
is ordered result handling, not early cancellation of remote work. A master
callback is accepted only with `save_outputs=true`, because otherwise moving
and retaining every numerical output would defeat streaming. Prefer a scalar
worker-side diagnostic for large states.

Public merging of multiple scan chunks is restricted to
`continuation=false`; independently cold continuation chunks cannot be
presented as one path. `GMRESAlgorithm.krylovdim` owns steady-state scan
workspace sizing, so a duplicate `solver_options.krylovdim` must raise.

The advanced Krylov workspaces are mutable and task-owned. Their allocating
wrappers are conveniences; even mutating calls may allocate small projected
dense factorizations while reusing the dominant full-coordinate arrays.
`BlockGMRESWorkspace` stores up to `nrhs*(block_krylovdim+1)` basis vectors,
deflates dependent block directions, supports a fixed left preconditioner,
and validates every projected and raw residual. `MultiShiftGMRESWorkspace`
shares one unrestarted Arnoldi factorization across `(A-shift*I)x=b`; this
requires a zero common initial guess and no generic preconditioner. Increase
the retained dimension or solve unconverged shifts separately. Only an exact
Arnoldi closure is a happy breakdown; small nonzero remainders remain in the
basis. Projected least-squares systems use rank-revealing QR and must report
full-residual nonconvergence for inconsistent or rank-deficient cases.

`RecycledGMRESWorkspace` carries mutable GCRO `U,C` information along a
slowly varying operator sequence, rebuilds the retained image for every new
operator, and discards it when rank is lost. Use one workspace per ordered
continuation chain. Prepared parameter scans warm only their state/Ritz seed;
they do not automatically invoke recycled GMRES. `KrylovExpvWorkspace`
computes `exp(t*A)b` by accepted/rejected Arnoldi time slices. Its defect sum
is an error estimate, not a physical trace/Hermiticity/positivity repair; a
nonconverged partial result exposes `reached_time` only when the caller
explicitly disables convergence failure.

All these Krylov paths derive storage precision from the operator and
storage-bearing inputs. Explicit workspaces, shifts, targets, preconditioners,
and times must be representable without narrowing; a compiled matrix-free PI
operator cannot borrow wider scratch than its prepared application kernels.
Widen and recompile the model instead.

`convergence_study` records a deterministic refinement sequence rather than
asserting that an inner solver flag proves discretization convergence. The
final requested `consecutive` pairwise comparisons must pass, and any explicit
inner `converged=false` in that window blocks the result. Default PI-state and
operator distances are coefficient-space Hilbert--Schmidt norms and require
the exact same basis; compare a common observable, reduced state, or explicit
embedding across different sector cutoffs. Empirical rates are reported only
on compatible geometric refinement scales and are descriptive, not an
extrapolation certificate.

Generic reports retain every raw evaluator result and estimate. Extract a
compact observable when full histories would make the study itself the memory
bottleneck. `convergence_estimate` raises on an unconverged report by default.
For stochastic calculations, sampling confidence, time-step bias, hierarchy
depth, Krylov dimension, and finite-size scaling are separate claims; common
random numbers can reduce refinement noise but do not replace confidence
intervals.

## Permutationally invariant HEOM

`HEOMBath` represents a fixed Hermitian PI coupling `Q_b` and a finite left
correlation decomposition `C_b^L(t)=sum_k ell_k exp(-nu_k*t)`. By default it
prepares the conjugate correlation `C_b^R=C_b^{L*}` on the same pole list:
real poles use `conj(ell_k)`, exact complex-conjugate pairs are cross-paired,
and a missing conjugate pole is appended with zero left coefficient. The
advanced `right_coefficients` keyword instead supplies the same-pole right
coefficients explicitly and disables completion; an inconsistent explicit
pair can destroy root Hermiticity and must never be repaired silently.
Coefficients and frequencies may be complex, but every `Re(nu_k)` must be
finite and strictly positive. The coupling must use the exact system
`PIBasis`. No Drude, Matsubara, Padé, temperature, or spectral-density
decomposition is inferred.

`HEOMPlan(...; max_depth=D, terminator=:none)` implements the documented
unscaled bosonic hierarchy and retains occupation vectors with
`sum(n_k)<=D`. With `K` exponential terms, the exact ADO count is
`binomial(K+D,D)` and the coordinate count is that value times
`length(basis)`; both are checked before allocation. The root ADO is the
physical reduced state. Auxiliary ADOs use the same equation-(7) PI
coordinates but generally are neither normalized, Hermitian, nor positive and
must be returned as `PIOperator`s, not density states. The upward boundary is
hard truncated to zero; convergence in both hierarchy depth and bath
decomposition remains the user's responsibility.

The plan is immutable and shareable. `HEOMWorkspace` owns a system workspace
and two PI-sized coupling buffers. `HEOMEvolutionWorkspace` additionally owns
four RK4 stages and a temporary vector of the complete hierarchy size; use one
per concurrent task. `apply!` is matrix-free and supports a driven system,
while fixed-step evolution requires times and step counts exactly
representable in the plan's real precision. Sources and destinations must not
narrow the promoted plan scalar type. `heom_liouvillian` exposes a
synchronized matrix-free compatibility adapter; explicit parallel work should
call `apply!` with task-owned workspaces.

Bath couplings and raw matrix system generators are copied at preparation.
The plan stores separate left and right prepared coefficients. Accumulated
decays and occupation-weighted downward coefficients must remain finite in the
prepared scalar type; overflow raises with wider-precision guidance. Integer
bath data must convert exactly to that type or raise rather than round.

`heom_depth_convergence` reuses one prepared system and coupling blocks,
retains only reduced root states at intermediate depths, and keeps a complete
hierarchy only for the finest result. It isolates depth truncation at one
fixed RK4 discretization, so repeat a time-step study separately.
`heom_steady_state` is autonomous-only and uses the existing trace-fixed
matrix-free GMRES machinery; it does not establish uniqueness or depth
convergence. Current limits are fixed Hermitian global PI couplings,
factorized automatic initial conditions, hard truncation, and an unscaled
bosonic Gaussian hierarchy. Fermionic signs, independent local baths,
imaginary-time preparation, residue/counterterm inference, scaled ADOs, and
HEOM-specific adjoints or preconditioners are not implemented.

## Dynamics, trajectories, spectra, and symmetries

Compile a model once before repeated evolution. Deterministic evolution accepts
sparse or matrix-free Liouvillians. Reuse `EvolutionWorkspace` with `evolve!`
for repeated fixed-step RK4 propagation; use `solve_dynamics` for the typed
high-level fixed-step result and `dynamics_problem` when adaptive or stiff
SciML integration is required.

`solve_dynamics(...; observables=...,save_states=false)` evaluates named local
or PI observables at the saved times while retaining only one mutable state.
It returns `DynamicsStreamResult`; the default call remains a history-carrying
`DynamicsResult`. A state-free call with no observable is rejected rather than
performing work whose result is discarded.

State-free trajectory aggregation uses concrete prepared observable tuples and
task-local Welford accumulators. Threaded workers retain trajectory-indexed
random streams, but the pooled `waiting_times` vector is an unordered sample:
do not rely on its element order across scheduling choices. Disable jump
statistics when even this jump-count-scaled storage is unnecessary. At least
one observable is required with `save_states=false`; the no-observable route
preserves the legacy inferred vector return type rather than providing a
jump-only shorthand.

`adaptive_quantum_trajectories` and `adaptive_diffusive_trajectories` stop at
deterministic batch boundaries only after every requested Hermitian observable
at every saved time meets a finite-horizon simultaneous empirical-Bernstein
half-width. The bound uses prepared finite spectral-range estimates and
allocates the confidence budget across all planned checks; a very large
observable/time/check count can underflow that budget in low precision and
must raise with wider-precision guidance. Zero sampled variance is not by
itself a zero-error certificate. Reaching `max_trajectories` returns
`converged=false` and `stopping_reason=:maximum_trajectories`.

Adaptive ensembles retain online means, variances, confidence data, and a
small batch history, never state histories. Their prepared batch plans are
shareable and their batch workspaces/RNGs are task-owned. Global trajectory
indexes determine samples, so serial and threaded runs use the same paths at
each check boundary even though floating accumulator order can differ. The
confidence certificate controls Monte Carlo sampling only: separately
converge fixed/adaptive path integration, unraveling choices, and model
approximations. Optional jump statistics still retain pooled waiting times;
disable them when unnecessary. This stopping layer currently covers density-
valued PI quantum jumps and collective diffusive trajectories, not weak-PI
pseudo-kets or Distributed workers.

The legacy `quantum_trajectories` call must remain inference-stable and return
its concrete vector directly. Streaming result types use exact union type
parameters for optional histories; do not replace them with abstract history
fields. Deterministic sampling fills concrete tuple-aligned buffers and builds
the flexible public observable dictionary only after propagation, never in the
per-sample hot loop.

For autonomous two-time functions, `CorrelationPlan` copies insertion Schur
blocks once and `CorrelationWorkspace` owns propagation and GMRES scratch.
`two_time_correlation` uses the explicit convention
`tr(A*exp(L*tau)*(B*rho*R))`: unlike `expectation`, it does not implicitly
adjoint `A`. `stationary_correlation_spectrum` returns the connected complex
one-sided `exp(-im*omega*tau)` transform from shifted GMRES and rejects the
disconnected Dirac-delta contribution. `correlation_spectrum_fft` is a
finite-window, uniform-grid radix-two transform with trapezoidal endpoints;
it is not the infinite-time resolvent.

`CompositePIBasis` may contain several exact `PIBasis` factors and small
`FiniteOperatorBasis(m)` factors of size `m^2`. Composite traces contract only
joint diagonal coordinates with exact multiplicity products. Local compiled
PI actions retain exact basis provenance. Cross Hamiltonians and jumps are
sums of factor left/right/sandwich maps; use one
`CompositeSuperoperatorWorkspace` per task. Finite bosonic modes must be
truncated explicitly.

`CompositeJumpChannel` describes one fixed tensor-product monitored
operator. `CompositeTrajectoryPlan` receives a trace-preserving background
which must exclude those monitored dissipators, then assembles the complete
unconditional generator itself. Never infer an unraveling from arbitrary
`CompositeSuperoperator` terms or accept a full generator plus duplicate
channels. Scalar rates may be driven but must evaluate once per conditional
RHS to a finite, real, nonnegative value representable in the plan precision.
Driven callbacks used by threaded batches must be pure and thread safe.
The selected gain is applied unscaled because its rate cancels on
normalization. Fixed-step paths integrate per-channel hazards with the same
RK4 stages as the conditional state and retry without mutating the state when
the integrated jump probability exceeds its cap; never restore a start- or
endpoint-only cap for driven rates.
The `log1p`/`expm1` conversion at an accepted hazard limit may round one ulp
above the original floating-point probability cap; clamp only that one-ulp
inverse-roundtrip excess, and keep throwing for any larger invariant breach.

Prepared composite trajectory traces use joint-sector diagonal lists and
fused exact products of Schur multiplicities, not a standalone composite
trace vector. Every task owns one `CompositeTrajectoryWorkspace`; all
channels share its two full tensor buffers, while only small factor-fibre
scratch grows with channel count. Batch RNG streams are indexed by global
trajectory number. Current composite paths are density-valued fixed-step
trajectories. Composite pseudo-kets, diffusive/event-driven paths, arbitrary
CP gains, Distributed batches, and implicit single-ensemble reductions are
not implemented.

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
the Ritz-extraction path, search-space exhaustion, and scale-aware residual
tolerances. When the full ambient space or the exact Boolean-mask range of a
matrix-free symmetry projector is spanned, harmonic Arnoldi switches to
ordinary Rayleigh--Ritz extraction in that complete invariant space; never
infer this fallback from Arnoldi breakdown alone. Convergence is still the
caller's responsibility for a partial nonnormal spectrum.

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
- `TrajectoryPlan`, `TrajectoryWorkspace`, `TrajectoryBatchWorkspace`
- `CompositeJumpChannel`, `CompositeTrajectoryPlan`,
  `CompositeTrajectoryWorkspace`, `CompositeTrajectoryBatchWorkspace`
- `CompositeQuantumTrajectory`, `composite_master_superoperator`
- `DynamicsStreamResult`, `TrajectoryEnsembleResult`
- `AdaptiveTrajectoryResult`, `adaptive_quantum_trajectories`,
  `adaptive_diffusive_trajectories`
- `jump_statistics`, `trajectory_observable_statistics`, `trajectory_statistics`
- `WeakPIPseudoKet`, `WeakPITrajectoryPlan`, `WeakPITrajectoryWorkspace`
- `weak_pi_quantum_trajectory`, `weak_pi_quantum_trajectories`
- `weak_pi_trajectory_average`, `weak_pi_trajectory_statistics`

PI trajectories use channel-resolved gain maps. Local particle labels are
unresolved, so an individual local-jump trajectory can be mixed even though
the ensemble converges to the PI master equation. Rates must evaluate to
finite, nonnegative real values representable in the prepared precision for
stochastic evolution. Time grids and explicit integration controls must also
be representable without narrowing; defaults and adaptive stages retain that
precision. `algorithm=:fixed` uses preallocated RK4 with a
maximum jump probability; `algorithm=:event`/`:adaptive` integrates the state
and hazard with Dormand--Prince 5(4) and locates continuous jump times by root
solving. `TrajectoryPlan` owns one fixed-operator kernel lowering; its direct
`K'K` Schur contractions evaluate channel intensities without forming a gain
state, which is materialized only for a selected jump. A
`TrajectoryBatchWorkspace` shares that read-only plan while retaining one
mutable workspace and RNG per task. Batch scheduling dynamically claims small
chunks, and streams are seeded by trajectory index, so serial and threaded
results agree for a fixed seed. Reuse a batch workspace sequentially, never concurrently; do not
index mutable scratch by `Threads.threadid()`. Scalar rates may be driven, but
operator-valued schedules are rejected. An empty model must select a concrete
real precision through its initial state or `TrajectoryPlan(...; T=...)`.
Default returned histories scale as `O(npaths * nsave * nPI)`. With named
observables and `save_states=false`, `TrajectoryEnsembleResult` constructs no
sampled `PIState`: each worker retains one observable buffer and Welford
accumulator. Jump summaries are online, although pooled waiting times still
scale with the number of retained intervals and may be disabled with
`jump_statistics=false`. Convergence-test fixed steps or adaptive tolerances
and the ensemble size.

The weak-PI backend is a distinct opt-in unraveling in
`directsum_nu U_nu`, not a labeled-particle wavefunction. Its sector slice
`psi_nu` represents the coefficient block
`C_nu=psi_nu*psi_nu'/sqrt(f^nu)`; relative phases between different sectors
are unphysical. `weak_pi_pseudoket` must reject a density state unless every
occupied multiplicity-weighted block has numerical rank one, and
`WeakPIPseudoKet` construction must check finite unit norm without silently
normalizing input.

For a local one-body gain from source `nu` to output `lambda`, each common
one-box child `mu` supplies a Kraus matrix. The pseudo-density strength is the
coefficient gain strength multiplied by `sqrt(f^lambda/f^nu)`; construct that
ratio with checked exact multiplicities before its final square root. Every
prepared physical channel must verify `sum K'*K == Q_nu` separately in every
source sector. Split collective/direct jumps by source sector as well, so a
sampled branch never introduces interference between distinct central Schur
labels. This factorization is representation-generic and tested for qubits
and qutrits; it must never be replaced by a `d^N` transform or an unnecessary
dense Choi diagonalization.

`WeakPITrajectoryPlan` supports fixed collective/direct/collective-p-body
jumps and fixed one-body `LocalJump` channels. Operator-valued schedules and
`LocalPBodyJump` are rejected. Scalar rates may be driven but must remain
finite, real, nonnegative, and representable; Hamiltonian rates must be finite
and real. The current integrator is fixed-step RK4 with a maximum jump
probability. Plans are read-only, workspaces are task-owned, and batch streams
remain trajectory-index seeded. `WeakPIJumpRecord` retains channel, source,
target, and one-box child metadata. A Kraus rotation can change individual
paths and their variances while preserving the master equation, so compare
ensemble-linear quantities unless an unraveling convention is explicitly
matched. Keep the hot jump-kernel traversal specialized by recursively taking
`Base.tail` of its concrete tuple: retaining the full tuple and advancing a
`Val` index allocates on Julia 1.10 even though newer compilers optimize it.

The diffusive backend monitors collective PI channels only. The unconditional
model must already contain the corresponding Lindblad dissipator; monitoring
does not insert it. `DiffusivePlan` owns copied read-only kernels, while each
task needs its own `DiffusiveWorkspace`. Homodyne and
heterodyne observable output is restricted to Hermitian observables because
the streaming result stores real values. Euler--Maruyama step-size convergence
must be checked and a finite trace-preserving step is not a positivity
certificate. Batch random streams are indexed by trajectory number so serial
and threaded runs agree for a fixed seed.

Research-utility channels, POVMs, Choi tests, and tomography operate only on
the retained PI coefficient algebra; their CP/TP certificates do not make a
claim about arbitrary non-PI inputs. In-place channel application forbids
source/destination aliasing. Immutable channel, gradient, and joint-symmetry
plans may be shared, but mutable workspaces are task-owned. Joint projectors
require mutually commuting unitary symmetries; exact rank setup can require
one projected application per PI coordinate. Checkpoints preserve the exact
retained-sector basis and never normalize, clip, or repair stored states;
HDF5/JLD2 support is optional. Implicit steady-state derivatives solve a
trace-fixed tangent equation, and checkpointed control derivatives require a
Hermitian terminal objective and time-step convergence checks.

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

## Optional extension contracts

Optional packages activate only their matching `ext/` module. Keep extension
signatures documented in `docs/src/interoperability.md` and never make core
algorithms dispatch conditionally on whether an optional plotting/table
package happens to be loaded.

The Tables extension gives `ParameterScanResult` a lazy scalar-metadata row
schema that deliberately excludes numerical outputs and nested diagnostics.
`ComplexSpectrum`, `QuditHusimiData`, and `ConvergenceStudyResult` use column
access. Their ordinary columns borrow result-owned vectors and must be treated
as read-only; absent spectrum diagnostics use an `O(1)` logical missing
column rather than allocating one value per mode. Qudit tables expose only
aggregate point-index/Q columns; resolved sector matrices remain explicit
fields. Convergence tables include the
estimate column, which may itself contain arrays or states, so a collecting
sink can still retain large objects. Tables adapters must expose existing data
only and never trigger a solve, transform, or full-Hilbert expansion.

The Makie extension similarly provides only argument conversions for existing
`ComplexSpectrum`, `SpinPhaseSpaceData`, `SchurBlockStructure`,
`QuditHusimiData`, and `ConvergenceStudyResult` values. It must never run an
eigensolver, matrix-free probe, phase-space transform, or refinement study.
Its default plot type is intentionally minimal; metadata-aware labels,
manifold coordinates, sector selection, and publication styling remain user
choices. This package extension is separate from the examples-only
CairoMakie loader that saves paper figures.

The Distributed extension prepares one model/trajectory workspace per worker
chunk and assigns global index-derived random streams. Process-parallel jump
and diffusive calls return serialized path results to the master and currently
accept `PIModel` inputs, not already compiled task-local plans. They are useful
when full paths are required; threaded or adaptive state-free ensembles avoid
that transfer when only statistics are needed. There is no Distributed
adaptive stopping protocol. Worker ids must be unique active non-master
processes using a compatible project.

The QuantumCumulants extension follows the exact neutral payload and automatic
lowering rules in the cumulant section. JLD2 and HDF5 remain optional storage
backends for versioned checkpoint payloads; their availability must not alter
the core checkpoint schema or validation. The ordinary package test target
does not acquire QuantumCumulants: `test/test_cumulants.jl` runs its symbolic
smoke test only when an optional-test environment has explicitly loaded that
weak dependency. Changes to the adapter therefore require a separate
QuantumCumulants-0.5 smoke run in addition to `Pkg.test()`.

## Published-model mapping

`examples/paper_models.jl` contains reusable constructors.

- PRA 94, 033838 (2016): correlated emission decomposes as
  `gamma*D[J_-] + (gamma0-gamma)*sum_i D[sigma_-^(i)]`.
  `examples/pra94_033838_superradiance.jl` compares Fig. 6 with equations
  (41)--(43), reaching approximately machine precision. It also computes the
  `N=30`, `delta_gamma/gamma0=0.4` pulse with the certified 256-coordinate
  population backend and visualizes the peak state through physical
  Schur-sector populations and Young diagrams; it never materializes a
  length-`2^30` state vector or a `2^30`-by-`2^30` density matrix.
- PRA 110, 062208 (2024): the dissipative LMG Hamiltonian is lowered exactly
  as a one-body self term plus a symmetric two-body cross term, with pair rate
  `2V/(N*j)`. This equals `V*(Jx^2-Jy^2)/(N*j)` but preserves body-order
  provenance so the same model supports `MeanFieldPlan`. Individual and
  collective rate prefactors follow equation (3). The example compares the
  exact finite PI steady state, finite product closure, thermodynamic closure,
  and the analytical fixed point following equations (10). Equation (11) is
  instead the singular `gammaI=0` case. A unique finite-N steady state restores
  the Z2 symmetry, so compare its branch-independent `Z` and parity-even
  transverse pair order rather than its vanishing `X,Y` to one selected
  broken-symmetry branch. Never claim the finite product closure is exact
  correlated finite PI dynamics.
- `examples/quantum_trajectories.jl` is the foundational
  Dalibard--Castin--Mølmer / Mølmer--Castin--Dalibard independent-emitter
  regression. It checks the exact tensor-power density state, binomial
  excitation and photon counts, and no-jump law with statistical tolerances.
- Zhang--Zhang--Mølmer, NJP 20, 112001 (2018):
  `zhang2018_superradiant_trajectories.jl` compares cavity and free-space
  intensities for `gammaL/GammaC=1,10` against a certified population-space
  master equation. The default `N=10`, 256-path calculation is a finite-size
  regression of their `N=50`, 512-path Fig. 2 model, not a digitization.
  The default density-valued local CP gain gives generally mixed paths.
  `examples/weak_pi_trajectories.jl` separately samples its exact Schur Kraus
  branches as pure direct-sum pseudo-kets for the `gammaL/GammaC=1` case and
  compares equal fixed-step batches with both certified population and general
  matrix-free master evolution. Neither record should be identified
  path-by-path with a different Kraus convention or with labeled emitters.
- Lloyd--Ziolkowska--Keeling (2026) is directly relevant to future
  sector-shift-resolved PI trajectories. The single-ensemble Schur
  pseudo-ket factorization is now available. A finite truncated cavity can be
  represented through `FiniteOperatorBasis`, and density-valued cross-factor
  jumps are available, but the published cavity trajectories still need a
  composite pseudo-ket compiler. Do not claim a Keeling figure reproduction
  from the density-valued composite backend.
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
- `examples/streaming_output.jl` compares observable-only deterministic output
  and state-free online trajectories with an exact emission law.
- `examples/quantum_regression.jl` validates non-Hermitian QRT insertion,
  antibunching, shifted-GMRES spectra, and finite-window FFT data.
- `examples/weak_pi_trajectories.jl` compares Schur pseudo-kets,
  density-valued paths, certified population evolution, and general
  matrix-free PI evolution for the Zhang--Mølmer `gammaL/GammaC=1` decay
  case. Equal fixed-step trajectory batches support a descriptive warmed
  per-path timing, never a wall-clock regression gate. Its two Makie figures
  show flux confidence bands, ensemble-state error, sampled total-spin sector
  changes, exact qubit coordinate scaling through the paper's `N=50` size,
  and retained history for equal saved batches. `Base.summarysize` history
  bars exclude plans, worker scratch, and transient peak RAM; the full-Hilbert
  curves are formulas and no exponential object is constructed.
- `examples/composite_ensembles.jl` combines two PI factors with one finite
  auxiliary factor and checks local lifts, cross terms, and trace preservation.
- `examples/composite_quantum_trajectories.jl` compares density-valued
  cross-factor jump paths with the independently propagated unconditional
  generator, streams observable/jump statistics, and checks serial/threaded
  trajectory-index reproducibility.
- `examples/parameter_scan.jl` validates streamed GMRES continuation, restart
  ownership, resumption, and the exact thermal steady-state curve without
  retaining a state history.
- `examples/pi_heom.jl` checks collective exponential-bath dephasing against
  an analytic coherence law and compares hierarchy depths. Its state-level
  depth report is intentionally stricter than the displayed observable error.
- `examples/qudit_husimi.jl` validates qutrit aggregate/sector Husimi data,
  the normalized-Haar maximally mixed value, and the `d=2` normalization
  against the spin-sphere convention.

Every runnable `examples/*.jl` file has a same-basename Markdown guide. Keep
the code, stated tolerances, and guide workflow synchronized. Current examples
compile each model once and prefer `solve_dynamics`, `stationary_state`, typed
algorithm/result objects, and prepared observable/reduction plans. Retain
low-level `liouvillian`, `apply!`, dense exponentiation, or complete spectra
only when the example explicitly validates a backend, solver, or published
small-system formula; explain that choice in the paired guide. Never replace a
published analytical or finite-size assertion merely to demonstrate a newer
API.

The core package deliberately has no hard Makie dependency. The weak Makie
extension only converts already computed result data. Publication-style
figures use the separate `examples/Project.toml` environment and the optional
loader in `examples/utils/makie_support.jl`. From the repository root, prepare
that environment with
`julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'`.
Figure-enabled scripts display their `CairoMakie.Figure` and save PDF plus PNG
copies under ignored `examples/figures/`, or under
`ENV["PI_EXAMPLE_FIGURE_DIR"]` when set. Without CairoMakie they must still run
all numerical checks and skip only rendering. Never move CairoMakie into the
root dependencies merely for examples.
The shared loader activates CairoMakie only when it is a direct dependency of
the active project. Do not fall back to an unrelated global environment: its
transitive graphics stack may be incompatible with the package environment.
Every standalone paper-specific example uses this optional path. Its figure
must visualize arrays already produced by the checked numerical workflow,
retain the guide's finite-size/mean-field caveats, and have a unique stable
output stem. Keep the Makie block after the numerical assertions so plotting
cannot replace validation.

## Verification workflow

From the repository root:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
julia --project=. examples/getting_started.jl
julia --project=examples -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=examples examples/pra94_033838_superradiance.jl
julia --project=examples examples/pra110_062208_lmg.jl
julia --project=examples examples/huelga1997_ramsey_dephasing.jl
julia --project=examples examples/kitagawa1993_one_axis_twisting.jl
julia --project=examples examples/shammah2018_local_pumping.jl
julia --project=examples examples/morrison2008_cooperative_fluorescence.jl
julia --project=examples examples/meiser2009_steady_superradiance.jl
julia --project=examples examples/iemini2018_boundary_time_crystal.jl
julia --project=. examples/pbody_pair_processes.jl
julia --project=. examples/steady_state_methods.jl
julia --project=. examples/floquet_periodic_decay.jl
julia --project=examples examples/quantum_trajectories.jl
julia --project=examples examples/zhang2018_superradiant_trajectories.jl
julia --project=examples examples/piccitto2021_interacting_time_crystal.jl
julia --project=examples examples/nakanishi2023_pt_time_crystal.jl
julia --project=examples examples/gambetta2019_dissipative_discrete_time_crystal.jl
julia --project=examples examples/meanfield_time_crystal.jl
julia --project=. examples/schur_block_visualization.jl
julia --project=. examples/spectral_visualization.jl
julia --project=. examples/qubit_population_dynamics.jl
julia --project=. examples/spin_phase_space.jl
julia --project=. examples/streaming_output.jl
julia --project=. examples/quantum_regression.jl
julia --project=. examples/weak_pi_trajectories.jl
julia --project=. examples/composite_ensembles.jl
julia --project=. examples/composite_quantum_trajectories.jl
julia --project=. examples/correlated_reservoirs.jl
julia --project=. examples/wiseman_milburn_homodyne.jl
julia --project=. examples/cumulant_bridge.jl
julia --project=. examples/research_utilities.jl
julia --project=. examples/parameter_scan.jl
julia --project=. examples/pi_heom.jl
julia --project=. examples/qudit_husimi.jl
julia --project=. benchmark/performance_regression.jl
julia --project=. benchmark/performance_audit.jl
julia --project=docs docs/make.jl
julia --project=quality -e 'using Pkg; Pkg.develop(path=pwd()); Pkg.instantiate()'
julia --project=quality quality/quality.jl
```

The exact post-audit source passed **5166 assertions in 71 groups** under full
`Pkg.test()` runs on Julia 1.10.11 and Julia 1.12.6 (2026-07-17). This includes
advanced Krylov (80), PI--HEOM (95), prepared scans (97), Tables/Distributed scan
extensions (69), convergence reports (64), adaptive ensembles (23), and
qudit Husimi data (25), plus 83 composite-stochastic assertions, in addition
to all pre-existing regression groups.
The four principal new-feature groups also passed 336 focused assertions on
Julia 1.12.6. The isolated
optional environment passed 78 Makie, QuantumCumulants, JLD2, and HDF5
assertions. Aqua's 11 package gates and JET's
three public hot-path gates passed, as did the strict documentation build and
its zero-undocumented-export/checkdocs gates. The four-thread Julia 1.10
allocation/thread-safety regression and the dependency-free global
performance audit passed. Treat counts as historical after further changes
and report actual command output.

All runnable examples have same-basename guides. The prepared-scan, PI--HEOM,
and qudit-Husimi examples passed on Julia 1.10 after the final source changes;
root-project runs skip a globally installed CairoMakie unless it is a direct
dependency of the active examples environment. The earlier literature,
trajectory, population, phase-space, mean-field, time-crystal,
Schur/density-, and complex-spectrum examples remain covered by their
dedicated regression groups and prior executable audits.
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
with sparse-versus-matrix-free precision guards. It covers qubit and qudit
basis/geometry setup, deterministic and stochastic propagation, composite
maps, channels, quantum regression, Floquet action, reductions, Krylov,
phase-space transforms, and Appendix-D kernels.
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
prepared collective moments; 29,504 B for weak-PI averaging; 127,960 B for
plan-only reduction; and 31,552 B for `reduced_state!` with both workspace and
output reused. The Julia 1.10 gate reports 73,160 B for weak averaging and
31,936 B for the in-place reduction. Its composite stochastic gates report
112 B at the public conditional-RHS wrapper (0 B inside the function barrier),
32 B for a full preallocated RK4/hazard step, and 110,768 B for a reused
four-thread trajectory batch on Julia 1.12. The exact final Julia 1.10
threaded gate reports the same 112 B and 32 B wrapper/step allocations and
99,544 B for that batch. The global audit also covers
population-plan setup and Husimi-Q/Wigner transforms. The residual
reduction allocation is dominated by state validation/LAPACK scratch; do not
remove it through an implicit trust or positivity opt-out. For `n=1000,m=40`,
retained Float32 workspace sizes are 50.05%, 50.01%, and 50.02% of Float64 for
GMRES, Arnoldi, and Jacobi--Davidson respectively. The dominant remaining setup
cost is front-loaded CG/Schur geometry and sparse LR factorization; reuse
explicit plans and workspaces rather than adding global mutable caches.
The 2026-07-17 audit removed several measured setup/output bottlenecks without
weakening validation. For an `N=10`, 128-path, 11-time weak-PI ensemble,
reconstruction dropped from about 25.7 MiB/23 ms to 0.15 MiB/0.9 ms. A
455-coordinate, 16-period Floquet action dropped from 13.32 MiB/17.1 ms to
24.8 KiB/1.0 ms; negative periods retain the established inverse-power
fallback. Plan-local exact factorials reduce qubit `ReductionPlan` setup
allocation by 53--57% for `N=16:40` with bit-identical recouplers. Fresh
composite construction removes one full-coordinate copy, and contiguous dense
first-factor batching was 2.18 times faster in the measured tensor case.
`content(::GTPattern)` is allocation-free on Julia 1.10 and later. Accepted
trajectory steps snap to saved targets only within eight local ulps, preventing
both a roundoff microstep and any small-time violation of the requested step
bound. Heterodyne I/Q innovations reuse the same unmodified left/right
products.

Exact `BigInt`/rational work remains setup-only and is not retained in hot
Liouvillian workspaces. Large-`N` stability fallbacks are guarded so the
ordinary small-system kernels keep their native floating-point algorithms.
Never restore mixed real/complex workspace `mul!` calls.

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
- Block GMRES, shared-Arnoldi multi-shift GMRES, mutable GCRO recycling, and
  adaptive Krylov exponential action reuse dominant matrix-free storage while
  retaining raw-residual and partial-result diagnostics. Prepared scans add
  ownership-safe continuation, bounded threaded scheduling, resumable point
  records, and optional deterministic Distributed chunks; they do not silently
  substitute recycled GMRES for ordinary high-level solvers.
- Explicit convergence reports distinguish refinement agreement from inner
  solver convergence and guard extraction of an unconverged finest estimate.
  Specialized PI--HEOM depth reports reuse system/coupling preparation and
  keep only the finest complete hierarchy.
- The finite-exponential PI--HEOM backend stores every unscaled bosonic ADO in
  PI coordinates and applies the hard-truncated hierarchy matrix-free.
  Generalized qudit Husimi-Q uses selected Schur coherent vectors and
  normalized Haar measure without a `d^N` embedding.
- Confidence-controlled jump/diffusive ensembles use online simultaneous
  empirical-Bernstein stopping. Tables and Makie extensions expose existing
  result data, Distributed supplies independent process chunks, and the
  QuantumCumulants extension lowers supported microscopic terms without
  changing the dependency-neutral core schema.
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
- Observable-only deterministic output and state-free trajectory aggregation
  avoid saved PI histories while preserving index-derived random streams.
  Prepared quantum regression reuses the matrix-free Liouvillian for both RK4
  propagation and connected shifted-GMRES spectra.
- Weak-PI pseudo-ket trajectories use one vector per retained Schur irrep.
  Their checked one-box Kraus subduction resolves local sector changes for
  qubits and qudits without a full-Hilbert object; collective/direct branches
  remain sector preserving and ensemble averages reproduce the density PI
  gain and master evolution.
- Composite PI coordinates retain equation-(7) normalization independently
  in every ensemble factor. Their preallocated tensor-mode kernel applies
  sums of factor maps without constructing the global Kronecker matrix.
  Explicit tensor-product monitored channels now provide density-valued
  composite quantum jumps with exact multiplicity-aware traces and a shared
  full-buffer workspace; a composite weak-PI pseudo-ket compiler remains
  unavailable.

Bounded research-scale limits remain: sparse-SPQR LR can exhaust memory for
very large qudit irreps; a single huge Schur block still needs dense Cholesky;
Evans reports `missing` for unsupported direct/custom microscopic recoupling or
an exceeded memory budget; CG geometry retains its documented Float64 phase
convention; and the Floquet eigensolver still materializes the PI-dimensional
one-period map. A generalized normalized-Haar qudit Husimi-Q transform is
available, but the Agarwal Wigner transform and sphere-specific renderer
remain qubit-only. A coefficient-space trace vector cannot be stored in a
scalar type when `sqrt(f^nu)` itself exceeds that
type; Liouvillian construction then raises instead of retaining `Inf`, and the
model must use wider coefficients. Public memory estimates use exact inline
`sizeof(T)` accounting for fixed-size isbits scalars, but conservative
precision-aware retained-storage bounds for `BigFloat` and
`Complex{BigFloat}`. Other heap-backed types are explicitly sample-based, not
worst-case bounded. `recommend_solver` includes one-body geometry exactly for
model/compiled-model inputs and conservatively for inputs without term
provenance; preserve its reported assumption metadata. See `IMPLEMENTATION_PLAN.md`
and `IMPLEMENTATION_NOTES.md`.

The global performance audit also identified larger architectural follow-ups
that were deliberately not folded into the low-risk 2026-07-17 pass:
qudit `OneBodyGeometry` still retains many empty sector-pair cells and tiny
vectors; repeated diffusive batches still prepare time grids/observables per
path; ordinary Arnoldi, fixed-step trajectories, and time-only correlations
retain scratch needed only by their more advanced sibling algorithms; and a
balanced-bipartition `ReductionWorkspace` retains separate product and
partial-transpose matrices. Address these with packed geometry, explicit
batch/mode-specific workspaces, or a tested in-place transpose permutation,
not with global caches or validation opt-outs. A plan-less custom
`MatrixFreeLiouvillian` also materializes its adjoint on demand; repeated
adjoint work should use a prepared adjoint or a future explicit adjoint
callback.

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
