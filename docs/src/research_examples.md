# Research examples

## Choose an example by goal

If you are new to the package, begin with the fully explained
[model-to-solution tutorial](getting_started.md) and its
[`getting_started.jl`](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/blob/main/examples/getting_started.jl)
script. Then choose the closest workflow below.

| Goal | Suggested runnable example | What it adds |
|:--|:--|:--|
| First deterministic model | [`getting_started.jl`](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/blob/main/examples/getting_started.jl) | Basis, terms, compilation, dynamics, stationary state, diagnostics, and time-step refinement |
| Prepared observables and a reduced state | [`driven_qubits.jl`](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/blob/main/examples/driven_qubits.jl) | Shared one-body geometry, a prepared observable, and a specialized one-body marginal workspace |
| Compare stationary solvers | [`steady_state_methods.jl`](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/blob/main/examples/steady_state_methods.jl) | Direct, shift-invert, SVD, and prepared-preconditioned matrix-free GMRES paths |
| Keep observables but not states | [`streaming_output.jl`](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/blob/main/examples/streaming_output.jl) | Memory-light deterministic and stochastic output |
| Simulate jump records | [`quantum_trajectories.jl`](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/blob/main/examples/quantum_trajectories.jl) | Event-driven paths and analytical ensemble checks |
| Study periodic dynamics | [`floquet_periodic_decay.jl`](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/blob/main/examples/floquet_periodic_decay.jl) | Reusable matrix-free period maps, selected multipliers, and a periodic state |
| Compare density and Schur pseudo-ket paths | [`weak_pi_trajectories.jl`](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/blob/main/examples/weak_pi_trajectories.jl) | Fixed/event-driven weak-PI paths, confidence stopping, and stationary batch diagnostics |
| Combine ensembles or an ancilla | [`composite_ensembles.jl`](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/blob/main/examples/composite_ensembles.jl) | Factorized composite coordinates, cross-system maps, and fixed-capacity matrix-RHS actions |
| Couple the ensemble to one shared cavity | [`global_pseudomode_cavity.jl`](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/blob/main/examples/global_pseudomode_cavity.jl) | Collective Tavis--Cummings dynamics, factor reductions, and a mode-cutoff diagnostic |
| Add a finite-memory bath | [`pi_heom.jl`](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/blob/main/examples/pi_heom.jl) | Exactly scaled ADOs, depth convergence, fixed-capacity matrix-RHS actions, SciML construction, and a block-preconditioned steady solve |
| Unravel one shared structured bath | [`pi_hops.jl`](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/blob/main/examples/pi_hops.jl) | Direct-sum Schur pure-state hierarchy, stationary colored noise, and an HEOM/analytic comparison |
| Use a non-Hermitian HOPS coupling | [`pi_hops_collective_emission.jl`](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/blob/main/examples/pi_hops_collective_emission.jl) | Exact one-excitation collective emission, hard-depth comparison, prescribed noise, and conditioned hierarchy application |
| Start HOPS from a mixed state | [`pi_hops_mixed_multibath.jl`](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/blob/main/examples/pi_hops_mixed_multibath.jl) | Multiple shared baths, Schur spectral sampling, reusable batch workspaces, Monte Carlo errors, and pruning metadata |
| Embed one truncated pseudomode per spin | [`debecker2026_all_to_all_ising_pseudomodes.jl`](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/blob/main/examples/debecker2026_all_to_all_ising_pseudomodes.jl) | Uniform-all-pair PI supersites, trajectory estimates, spin-only negativity, a strong-parity-reduced x-GHZ steady solve, and cutoff checks |

Each script has a same-basename guide in the repository's `examples/`
directory. The sections below explain the specialized workflows and link to
their numerical conventions.

## PI research utilities

`examples/research_utilities.jl` demonstrates compressed spectral and
population inspection, a Kraus channel with retained-algebra CP/TP checks,
POVM sampling and tomography, versioned checkpoint round trips, and a joint
weak-symmetry projector. The paired [research-utilities guide](research_utilities.md)
records the scope and workspace conventions; none of these operations expands
the state to a `d^N` Hilbert-space matrix.

## Diffusive monitoring

`examples/wiseman_milburn_homodyne.jl` compares collective homodyne
trajectories with unconditional PI evolution and checks ensemble convergence.
The model already contains the monitored-channel dissipator; monitoring adds
only the conditioned innovation. See [Diffusive monitoring](diffusive_monitoring.md)
for homodyne/heterodyne conventions, task-owned workspaces, reproducible batch
streams, and Euler--Maruyama convergence requirements.

## Higher-order cumulant closures

`examples/cumulant_bridge.jl` exports exact distinct-site PI moments and
neutral microscopic term metadata, then compares a factorized closure with
product and correlated states.  The core bridge has no symbolic-algebra
dependency; an optional QuantumCumulants.jl 0.5 extension maps the exact
values onto user-supplied symbolic averages.

## Correlated Kossakowski reservoirs

`examples/correlated_reservoirs.jl` compares a correlated local bath with its
exact factorization into independent effective jumps, evolves the model
matrix-free, and checks a preallocated collective Kossakowski schedule:

```julia
gamma = mixing * mixing'
term = CorrelatedLocalJumps((sigma_minus, sigma_z), gamma; rate=0.08)
prepared = compile(PIModel(basis, (term,)); backend=:matrixfree)
```

The fixed factorization is only in the small reservoir-channel space.
Off-diagonal entries retain noise interference; no full `d^N` operator is
built. The complete guide is `examples/correlated_reservoirs.md`.

## Memory-light observable output

`examples/streaming_output.jl` compares history-carrying evolution with
observable-only deterministic output and state-free trajectory aggregation:

```julia
series = solve_dynamics(
    prepared, rho0, (0.0, 2.0);
    saveat=0.05,
    observables=(excitation=number,),
    save_states=false,
)
```

The deterministic result retains one evolving PI vector. The trajectory path
uses task-local Welford accumulators for means, unbiased variances, standard
errors, and confidence intervals, while preserving trajectory-indexed random
streams. Optional pooled waiting times still scale with the number of jumps;
disable jump statistics when they are not needed. See
[Streaming and observable-only output](streaming_output.md).

## Quantum-regression correlations and spectra

`examples/quantum_regression.jl` validates the PI quantum-regression path for
an incoherently pumped emitter. It compares a non-Hermitian first-order
correlation, delayed antibunching, and the connected optical spectrum with
closed forms:

```julia
plan = CorrelationPlan(prepared, adjoint(c), c)
work = CorrelationWorkspace(plan; krylovdim=20)
C = two_time_correlation(plan, rho_ss, delays; workspace=work)
S = stationary_correlation_spectrum(
    plan, rho_ss, frequencies; workspace=work)
```

The same matrix-free Liouvillian drives sequential RK4 samples and shifted
GMRES resolvents. The first readout operator is not implicitly adjointed, and
the stationary spectrum is the connected one-sided complex transform. See
[Quantum regression and optical correlations](correlations.md).

For general response calculations, `ResponseWorkspace` reuses the dominant
shifted-GMRES and exponential-action arrays across `resolvent_norm`,
`pseudospectral_abscissa`, `adjoint_evolve`,
`integrated_correlation_time`, and `steady_state_susceptibility`. The
matrix-free resolvent is a converged power/GMRES estimate rather than a
rigorous upper bound; every shifted solve must converge. Trace-fixed Poisson
and tangent solves report their physical residual and trace error. See
[Matrix-free Krylov solvers](matrix_free_krylov.md#Matrix-free-response-and-adjoint-analysis).

## Several PI ensembles and finite auxiliaries

`examples/composite_ensembles.jl` combines two independent compressed PI
ensembles with a finite two-level auxiliary factor. Local compiled
Liouvillians and tensor-product Hamiltonian/dissipator terms are applied as a
sum of factor maps:

```julia
basis = CompositePIBasis(ensemble_a, ensemble_b,
                         FiniteOperatorBasis(2))
generator = CompositeSuperoperator(basis, local_a, local_b) + interaction
work = CompositeSuperoperatorWorkspace(generator, rho.data)
apply!(destination, generator, rho.data, 0.0, nothing, work)
```

No global Kronecker superoperator or full Hilbert representation of either PI
ensemble is formed. Finite auxiliaries are explicitly truncated.
`examples/composite_quantum_trajectories.jl` additionally compares
density-valued cross-factor jump paths with the independently propagated
unconditional generator and demonstrates state-free statistics plus
trajectory-index reproducibility. See
[Composite PI systems](composite_systems.md).

## One shared cavity

`examples/global_pseudomode_cavity.jl` specializes the composite backend to
one finite cavity shared by a PI emitter ensemble. It evolves a damped
Tavis--Cummings model, extracts atomic and photon observables, traces either
factor directly, and monitors the highest cavity level without constructing a
global Kronecker superoperator. The mode is one distinguished global factor,
not one replicated auxiliary per emitter. See [Global pseudomodes and shared
cavities](global_pseudomodes.md).

## Sector-resolved spin phase space

`examples/spin_phase_space.jl` constructs a state occupying two total-spin
sectors and computes both its positive Husimi-Q density and signed
spin-Wigner quasidistribution. Each sector sphere integrates to its physical
Schur population, while the displayed aggregate is their angular marginal:

```julia
q = spin_husimi_q(rho; ntheta=81, nphi=160, resolved=true)
w = spin_wigner(rho; ntheta=81, nphi=160, resolved=true)

display(visualize_spin_phase_space(q))
display(visualize_spin_phase_space(
    w; sector=first(w.sectors)))
```

The script verifies multiplicity weighting, normalization, the coherent-state
peak, and Wigner negativity without constructing a full Hilbert-space state.
See [Sector-resolved spin phase space](spin_phase_space.md) for the precise
normalization and multipole convention.

`examples/qudit_husimi.jl` extends the coherent-state Q check to qutrit Schur
sectors. It reuses one `QuditHusimiPlan`, verifies normalized-Haar population
weighting and the maximally mixed density, and checks the `d=2` factor against
the spin-sphere convention. See [Qudit Husimi phase
space](qudit_phase_space.md); this is not a generalized qudit Wigner
transform.

## Prepared continuation and non-Markovian hierarchy examples

`examples/parameter_scan.jl` follows an exactly soluble emission/pumping
steady-state curve while retaining no state history. It compiles the fixed
geometry as one scalar-rate family, reuses a bounded GCRO recycle space,
stops, stores one ownership-safe continuation seed, resumes in a fresh
workspace, and exports compact columns. A setup-only
`sensitivity_problem` check exercises the same prepared matrix-RHS protocol.
Threaded and optional Distributed execution apply only after continuation is
disabled. See [Prepared parameter scans](parameter_scans.md).

`examples/pi_heom.jl` stores every ADO of a collectively dephased qubit
ensemble in exactly similarity-scaled PI coordinates and compares three
hierarchy depths with an exact exponential-bath coherence. It separately
demonstrates that convergence of one observable is weaker than convergence of
the complete reduced state. A one-step `heom_problem` check exposes the same
matrix-free right-hand side to SciML, while a small unique stationary model
demonstrates a reusable ADO-diagonal `HEOMBlockPreconditioner` and validates
fixed-capacity forward/adjoint matrix-RHS applications. Scaling changes
conditioning and stored auxiliary coordinates, not the root reduced state or
the hierarchy truncation. See [PI--HEOM](heom.md) and [Numerical convergence
reports](convergence.md).

`examples/pi_hops.jl` propagates the corresponding hierarchy of pure states
in direct-sum Schur-irrep coordinates, averages unnormalized root outer
products, and compares the result with both PI--HEOM and the analytic
collective-dephasing coherence. Its shared bath is PI on every noise
realization. Independent local colored noises are not replaced by a common
noise and remain an HEOM or local-pseudomode problem. See
[PI--HOPS](hops.md).

Two additional scripts expose the specialized HOPS interfaces.
`examples/pi_hops_collective_emission.jl` uses the non-Hermitian coupling
``L=J_-`` and a symmetric one-excitation state, for which depth one closes
exactly and the Lorentzian survival amplitude is analytical. It also
demonstrates a prescribed conditioned noise and deterministic `hops_rhs!`
application. `examples/pi_hops_mixed_multibath.jl` prepares a general mixed
PI state as a Schur spectral ensemble, combines two independent shared
dephasing baths, reuses `HOPSBatchWorkspace`, reports state-level Monte Carlo
errors, and inspects a setup-only importance-pruned hierarchy. Its pruning
score is explicitly not treated as an accuracy certificate.

These hierarchy examples still use a finite exponential decomposition and a
hard hierarchy boundary. Bath-decomposition error, depth error, integration
error, and (for HOPS) sampling error are separate refinements. The HOPS
backend supports fixed shared PI bath couplings, including non-Hermitian
couplings; it does not infer residue corrections, terminators, or the
specialized hierarchy for independent local non-Markovian baths.

## Uniform all-pair Ising spins with local pseudomodes

`examples/debecker2026_all_to_all_ising_pseudomodes.jl` demonstrates a
different finite-memory strategy: each spin and its truncated local
pseudomode are combined into one supersite of dimension
`d=2(nmax+1)`. The all-to-all interaction has the same coefficient for every
unordered spin pair, and every supersite has the same spin--mode coupling and
mode-loss channel. The enlarged model is therefore exactly PI and can use the
ordinary prepared Liouvillian workflow. At fixed cutoff the enlarged
spin-plus-pseudomode state follows a Markovian Lindblad equation, while
tracing the modes produces the intended finite-memory, generally
non-Markovian spin dynamics:

```julia
include("examples/paper_models.jl")
using .PaperModels

operators = debecker2026_pseudomode_operators(nmax)
basis = PIBasis(N, operators.dsite)
model = debecker2026_all_to_all_ising_pseudomode_model(
    basis, operators;
    Jpair=J / (N - 1), omega_c, gamma, kappa,
    coupling=:minus,
)
prepared = compile(model; backend=:matrixfree)
```

That snippet is the basic prepared workflow, not the integrator used for the
long manuscript-style panel. The executable propagates each
`omega_c*t=0:0.5:100` curve interval by interval with a matrix-free generator,
one reusable dimension-30 `KrylovExpvWorkspace`, and preallocated
`krylov_expv!`. This costs 6000 Liouvillian applications per curve and avoids
retaining a state history. A separate dimension-40 run with tighter
tolerances checks the broad `kappa/omega_c=20` curve; the current pointwise
`Cxx` difference is about `8e-16`. This replaces a fixed-step RK4 refinement,
which was not stable enough in that broad-decay regime.

At one stationary reference point the script calls both
`trajectory_steady_state` and `weak_pi_trajectory_steady_state`. The first
uses density-valued conditional paths; the second resolves the local gain into
one-box Schur Kraus branches and evolves direct-sum pseudo-kets. Each route
streams a post-settling density average within every independent path before
combining path means, and neither retains histories or jump logs. Comparisons
with the direct stationary solution include the full PI-coordinate error,
Hilbert--Schmidt Monte Carlo standard error, Liouvillian residual, `Cxx`, and
spin-only negativity. The colored stars on the stationary maps show the
weak-PI estimate. Burn-in time, within-path spacing, the selected fixed-step
or event-driven integration controls, path count, and pseudomode cutoff remain
separate convergence requirements.

The helper treats `Jpair` as the literal coefficient of every unordered pair;
the displayed `J` uses the explicit Kac choice `Jpair=J/(N-1)`. It also maps
the manuscript convention `D_paper=2D_package` to a local mode-jump rate
`2kappa` and sets the exchange strength to `sqrt(gamma*kappa)`. Both
`coupling=:minus` and `coupling=:z` are available. The latter retains a strong
spin-parity symmetry, so an unrestricted steady state need not be unique. For
that branch the script restricts both Schur-block indices to the initial
even-parity Hilbert support. This is a 204-coordinate stationary solve instead
of the full 816-coordinate default problem; an ordinary weak-conjugation
projector would retain both diagonal parity blocks and would not resolve the
stationary degeneracy. Every point checks strong-block leakage, parity, state
diagnostics, and the full `norm(L*rho)` residual. One adaptive matrix-free
`krylov_expv` propagation to `omega_c*t=1600` plus a half-time value remain as
independent reference checks rather than being repeated across the grid.

The longitudinal model also has the PI-compatible weak symmetry generated by
``\sigma_x\otimes(-1)^{a^\dagger a}`` on every supersite. The example certifies
it explicitly, but does not restrict the selected state to one of its charges:
for odd `N` that transformation exchanges the two strong spin-parity sectors
and a trivial-charge projection would alter the GHZ coherence.

The supplied manuscript studies a nearest-neighbour periodic chain. That
geometry is not invariant under arbitrary permutations, so this runnable
model is deliberately the uniform-all-pair specialization, not a reproduction
of the spatial chain. All distinct pairs share one `Cxx`; there is no distance
coordinate or spatial correlation length. The finite-size stationary heat
maps must likewise not be advertised as a thermodynamic phase diagram.

For the repeated stationary maps, the example prepares one
`LocalFactorTracePlan(basis,(2,levels); traced_factor=2)`. It traces every
pseudomode directly in PI coordinates, then uses `ReductionPlan` first to keep
two spins and again to evaluate their `1|1` negativity. A 16-Pauli-moment
reconstruction remains as an independent reference-point oracle. The plotted
quantity therefore measures two physical spins after tracing both
pseudomodes, rather than two spin--mode supersites. For longitudinal coupling
the example still uses full `N`-particle moments with the mode identity to
evaluate the relative-phase-optimized spin x-GHZ fidelity

```math
F_{\mathrm{GHZ}_x}^{\max}
=\frac{P_{+x}+P_{-x}}{2}
 +\left|\langle {+x}|^{\otimes N}\rho_{\mathrm{spin}}
                    |{-x}\rangle^{\otimes N}\right|.
```

The plotted `0.5` contour is the usual genuine-multipartite-entanglement
witness threshold. No full spin or spin--mode Hilbert matrix is formed.

The script also compares `nmax=1` and 2 and checks the wider calculation's
highest-level population for the `coupling=:minus` correlation curve. That
short cutoff comparison still uses fixed-step `solve_dynamics`; it does not by
itself certify either the long-sweep integration or the cutoff of the separate
longitudinal GHZ map.
The retained PI dimension is

```math
n_{\mathrm{PI}}=\binom{N+[2(n_{\max}+1)]^2-1}{N},
```

so cutoff growth must be estimated and convergence repeated for each claimed
observable. The optional Makie output contains a manuscript-style dynamics
panel, stationary correlation and spin-negativity maps, and a separate cutoff
figure. A third, manuscript-Fig.-4-style figure shows the parity-selected
x-GHZ witness, its `0.5` contour, and the maximum long-time residual. The
extracted `Cxx=0` and GHZ-witness boundaries are also written, independently
of Makie, as run-qualified tab-delimited text files beside the figures. Each
file retains the raw interpolated contour, the origin-constrained quadratic
candidate, and any available fallback candidate together with fit metadata.
Set `PI_EXAMPLE_FIGURE_DIR` to redirect both numerical and graphical output.
The
[complete example
guide](https://github.com/aenictusGitHub/PermutationalInvariantDynamics.jl/blob/main/examples/debecker2026_all_to_all_ising_pseudomodes.md)
records the model boundary, rate conversion, solver checks, and figure stems.

## Visualizing Liouvillian and Floquet spectra

`examples/spectral_visualization.jl` computes a complete `N=3` PI
Liouvillian spectrum and a converged one-period map for a periodically driven,
decaying qubit ensemble. The numerical results are retained independently of
their SVG presentation:

```julia
data = liouvillian_spectrum_data(
    prepared; nev=pi_dimension(prepared), algorithm=:dense)
display(visualize_spectrum(data))

F = floquet_propagator(periodic, period; steps=256)
multipliers = floquet_spectrum_data(F; period=period)
display(visualize_spectrum(multipliers))
```

The script checks the continuous-time stability half-plane, the Floquet fixed
multiplier and unit disk, RK4 step doubling, and the principal-log exponent
strip. It diagonalizes the converged map once, reuses its multiplier vector
for multiplier and exponent diagrams, and writes generated SVGs only inside a
temporary directory. See [Spectral visualization](spectral_visualization.md)
for the complete/partial-spectrum and branch conventions.

## Inspecting Schur-block structure

`examples/schur_block_visualization.jl` solves the full-rank `N=4` local
pump--decay steady state and compares it with its exact iid thermal form. It
then contrasts collective- and local-decay Liouvillians. Reusable state-sector
data are obtained with `schur_block_structure` and rendered separately:

```julia
structure = schur_block_structure(prepared;
    metric=:frobenius,
    threshold=1e-12)
figure = visualize_schur_blocks(structure;
    scale=:log,
    title="Local decay: Schur-sector coupling",
    show_young_diagrams=true)
save_schur_block_visualization("local_decay.svg", figure)
```

The Young-diagram thumbnails identify the input and output partitions. Their
tooltips retain the exact standard-tableau multiplicity `f^nu`; they do not
choose a particular tableau filling.

For the solved state, `metric=:population` displays physical sector trace
weights. The same state is diagonalized once, block by block, and its exact
Schur multiplicities remain compressed in the density-spectrum view:

```julia
density = pi_density_spectrum(rho_ss)
density_figure = visualize_density_spectrum(
    density; show_degeneracies=true)
save_density_spectrum_visualization("density.svg", density_figure)
```

The density plot contains one point per irrep eigenvector, coloured by sector,
with exact degeneracies in labels and tooltips. It does not create the
exponentially large `expanded=true` eigenvalue list.

State and operator structures are diagonal in the Schur labels. Their default
physical representation measures `C_nu/sqrt(f^nu)`; pass
`representation=:coefficient` to inspect stored equation-(7) blocks instead.
For a superoperator, rows are output sectors and columns are input sectors in
orthonormal PI coefficient coordinates. Consequently, an off-diagonal tile
shows a map between Schur sectors, not a state coherence or a transition
probability.

Sparse and dense superoperators are scanned directly. A matrix-free source is
probed exactly once for each of its `n_PI=length(basis)` input coordinates,
with reusable scratch and without retaining a global Liouvillian. This setup
cost should normally be paid once and the resulting `SchurBlockStructure`
reused. A driven source requires `time=...`; general operator-valued fallback
terms are frozen and lowered once at that time before all coordinate probes.

The returned figures display directly in SVG-capable notebooks and can be
written with `save_schur_block_visualization` and
`save_density_spectrum_visualization`. Both renderers are dependency-free; no
plotting package or full `d^N` state is constructed. See
[Schur-block visualization](schur_visualization.md) for the available metrics,
normalizations, extraction thresholds, and rendering options, and
[Spectral visualization](spectral_visualization.md) for the compressed density
spectrum convention.

## Certified Schur-diagonal population dynamics

`examples/qubit_population_dynamics.jl` constructs all six local and
collective qubit bath channels with `qubit_ensemble_model`. A diagonal spin
Hamiltonian and central Dicke initial state make the Schur-basis diagonal
subspace invariant. The script constructs the reduced plan once and inspects
its strict invariance certificate:

```julia
plan = PopulationPlan(model)
report = plan.invariance
p0 = diagonal_populations(rho0)
solution = solve_populations(plan, p0, (0.0, 2.0); saveat=0.25)
```

For `N=6`, only 16 physical GT-pattern populations are evolved instead of 84
general PI coordinates. The example propagates the complete PI density state
on the same grid and asserts agreement of every saved population vector. It
then calls `stationary_populations`, checks the reduced-generator residual and
normalization, reconstructs a validated `PIState`, and compares it with a
direct full-PI stationary solve. The paired guide
`examples/qubit_population_dynamics.md` gives the six-rate and multiplicity
conventions.

## Quantum trajectories against analytical and master-equation references

`examples/quantum_trajectories.jl` tests event-driven PI trajectories for
independent spontaneous emission. In addition to deterministic density-matrix
evolution, it uses the exact tensor-power state, binomial excitation/photon
counts, and no-jump probability as analytical Mølmer-method references. Every
stochastic assertion is scaled by the corresponding Monte Carlo standard
error.

`examples/zhang2018_superradiant_trajectories.jl` combines collective cavity
decay and individual free-space decay at the two rate ratios used in Fig. 2 of
Zhang, Zhang, and Mølmer, *New J. Phys.* **20**, 112001 (2018):

```julia
include("examples/paper_models.jl")
using .PaperModels

model = zhang2018_superradiance_model(
    N; GammaC=1.0, gammaL=10.0)
basis = model.basis
rho0 = iid_pure_state(basis, ComplexF64[0, 1])
radiation = zhang2018_radiation_operators(
    basis; GammaC=1.0, gammaL=10.0)
times = collect(range(0.0, 0.6; length=31))
trajectories = quantum_trajectories(
    model, rho0, times, 256;
    algorithm=:event, dt=0.02, dtmax=0.05, seed=2019)
statistics = trajectory_statistics(
    trajectories;
    observables=(cavity=radiation.cavity,
                 free_space=radiation.free_space),
    nchannels=2)
```

The two radiated intensities are checked pointwise against a certified
population-space master equation. The example guide explains why the
particle-unresolved local gain map has the same ensemble dynamics but a
different conditional record from the article's sector-shift-resolved pure
pseudo-state unraveling.

`examples/weak_pi_trajectories.jl` then exercises the separate opt-in
`WeakPITrajectoryPlan`. It factorizes that same local PI gain into complete
one-box subduction Kraus branches, propagates direct-sum Schur-irrep
pseudo-kets, and compares their ensemble with density-valued paths, certified
population dynamics, and general matrix-free PI evolution for
``\gamma_l/\Gamma_c=1``. Equal trajectory counts and fixed-step controls make
the warmed per-path timing descriptive of that run, but not a portable speed
certificate. The same prepared Schur-Kraus plan is also exercised with
adaptive Dormand--Prince propagation and continuous hazard roots. An
observable-only adaptive ensemble stops only at deterministic batch boundaries
when its simultaneous empirical-Bernstein targets pass or its trajectory cap
is reached.

The example also forms a history-free stationary density estimate from
event-driven pseudo-kets. It takes sector-block outer products before every
average, averages correlated samples within a path, and uses independently
seeded path means for its primary Monte Carlo standard error. Optional batch
means diagnose within-path autocorrelation under an explicitly approximate
independence assumption; they do not certify burn-in or finite-window bias.
Integrator controls, settling time, sampling window, and independent path
count therefore remain separate convergence checks.

The pseudo-kets are not labeled-particle wavefunctions, and their particular
Kraus record must not be identified path-by-path with another unraveling.
Composite cavity pseudo-ket trajectories remain outside this single-ensemble
backend.

With the examples-only CairoMakie environment active, both trajectory scripts
also render their numerical comparisons. The weak-PI example uses 95%
confidence bands and adds logarithmic state error, sampled sector-transition,
exact coordinate-scaling, and current-run timing panels. The separate
Zhang--Mølmer density-path figure places the two decay-rate ratios side by
side. PDF and PNG outputs are written without adding Makie to the package
dependency graph.

## Publication figures for the literature examples

All standalone paper-specific scripts use the optional loader in
`examples/utils/makie_support.jl`. Their numerical assertions, convergence
checks, and printed diagnostics run unchanged in the core environment; with
`--project=examples`, each script additionally displays a CairoMakie figure and
writes PDF and PNG copies under `examples/figures/` (or the directory selected
by `PI_EXAMPLE_FIGURE_DIR`).

The panels are organized by the quantity actually validated in each script:

- Huelga and Kitagawa--Ueda overlay exact metrology curves with PI dynamics;
  the latter also shows the one-spin purity generated by twisting.
- Shammah, Morrison--Parkins, and Meiser--Holland visualize steady-state
  sectors, exact-state observables, radiated intensity, and cooperative
  enhancement.
- Iemini, Piccitto, Nakanishi--Sasamoto, and Gambetta show finite-size slow
  modes, gaps or Floquet multipliers together with oscillatory dynamics. Their
  annotations explicitly distinguish finite-size precursors from a
  thermodynamic time-crystal claim.
- The finite/thermodynamic mean-field comparison overlays both closures with
  exact PI evolution and its analytical finite-`N` curve.

The Damanet, Pausch, Mølmer, and Zhang--Mølmer figures use the same convention
for the previously implemented pulse, mean-field, and trajectory comparisons.
Each same-basename guide describes the precise panels and saved-file stem.

## Appendix-D pair processes

`examples/pbody_pair_processes.jl`
combines a two-particle Ising interaction, independent pair loss, and a
collective pair-loss channel:

```julia
pair_loss = kron(sm, sm)
pair_interaction = kron(sz, sz)
model = PIModel(basis, [
    PBodyHamiltonian(pair_interaction, 2; rate=0.05),
    LocalPBodyJump(pair_loss, 2; rate=0.02),
    CollectivePBodyJump(pair_loss, 2; rate=0.001),
])
```

The script verifies the pair-sum identity, trace preservation, and equality of
sparse and matrix-free application. It deliberately exposes the advanced
workspace API for that backend check:

```julia
prepared = compile(model; backend=:matrixfree)
workspace = LiouvillianWorkspace(prepared)
Ls = liouvillian(prepared; representation=:sparse)
apply!(y, prepared, rho0.data, 0.0, nothing, workspace)
```

The explicit workspace makes repeated Appendix-D applications preallocated;
ordinary evolution can pass `prepared` directly to `solve_dynamics`.

## Comparing steady-state solvers

`examples/steady_state_methods.jl`
compares a trace-bordered solve, SVD, dense diagonalization, sparse
shift-invert, and restarted matrix-free GMRES against an exact product thermal
state.

```julia
prepared = compile(model; backend=:sparse)
result = stationary_state(prepared;
    algorithm=ShiftInvertAlgorithm(shift=-1e-3, maxiter=100),
    atol=1e-12,
    rtol=1e-10,
    return_info=true,
)
rho_ss = result.state
```

For parameter scans, the previous stationary state can be passed through
`initial_state`. The example reports residuals, trace errors, iteration counts,
reference-state errors, and indicative wall times. Timings include compilation
on a first Julia run and should not be interpreted as benchmarks.

For a solve that never assembles the Liouvillian, reuse both Krylov storage and
the Schur preconditioner across a scan:

```julia
prepared_mf = compile(model; backend=:matrixfree)
workspace = KrylovWorkspace(prepared_mf, 30)
P = schur_sector_preconditioner(prepared_mf, model.basis;
                                expected_reuses=10)
result = stationary_state(prepared_mf;
    algorithm=GMRESAlgorithm(krylovdim=30, maxiter=500,
                             preconditioner=P),
    workspace=workspace,
    return_info=true)
preconditioner_cost(P)
```

Selected slow modes and the gap use the same matrix-free action:

```julia
modes = pi_liouvillian_spectrum(model;
    method=:arnoldi, nev=6, krylovdim=40, vectors=true)
gap = pi_liouvillian_gap(model;
    method=:arnoldi, nev=6, krylovdim=40, return_info=true)
```

Increase `krylovdim` until every requested Ritz residual is converged. A
partial spectrum cannot certify a stationary multiplicity larger than the
requested `nev`; inspect `stationary_multiplicity_certified` in gap reports.

## Periodic decay and Floquet analysis

`examples/floquet_periodic_decay.jl`
uses a sinusoidally modulated local decay rate. It performs an RK4 convergence
study against the exact commuting one-period map and computes multipliers, the
Floquet gap, the periodic steady state, and stroboscopic populations.

```julia
rate = (t,p) -> gamma*(1 + a*cos(2pi*t/period))
model = PIModel(basis, [LocalJump(sm; rate=rate)])
prepared = compile(model; backend=:matrixfree)
F = floquet_map(prepared, period; steps=160)
selected = selected_floquet_multipliers(
    F; method=:arnoldi, which=:LM, nev=4,
    krylovdim=length(basis))
gap = floquet_gap(
    F; method=:arnoldi, nev=4,
    krylovdim=length(basis), return_info=true)
periodic = floquet_steady_state(
    F; method=:krylov, krylovdim=20, return_info=true)
states = stroboscopic_evolution(rho0, F, 4)
```

Every period action integrates only the supplied PI vector. Arnoldi/IRAM and
the trace-fixed GMRES solve therefore reuse the period map without
constructing a dense PI-dimensional channel. The script checks selected Ritz
residuals, the periodic-state residual, and a certified matrix-free parity
restriction. A small dense `floquet_propagator` remains only as an independent
reference for the complete multiplier set.

`floquet_gap(...; return_info=true)` distinguishes a residual-resolved
selected decay estimate from a certified global gap. The latter requires a
complete multiplier set; a partial Arnoldi/IRAM window must retain
`global_gap_certified=false`. Jacobi--Davidson can target a particular
multiplier but is not a global spectral-radius method.

Increase `steps` until the relevant map action, multiplier, or observable has
converged, independently of Krylov tolerances. Stiff protocols may require a
dedicated integration strategy rather than the fixed-step Floquet helper.

`examples/gambetta2019_dissipative_discrete_time_crystal.jl` applies the same
piecewise workflow to the fully connected dissipative Rydberg protocol of
Gambetta *et al.*, *Phys. Rev. Lett.* **122**, 015701 (2019). The short pulse
and long relaxation segment are integrated separately and composed in their
physical order. At `N=4`, the script checks the negative period-doubling
multiplier against an independent full-Hilbert-space calculation and performs
a step-doubling convergence check. This is explicitly a finite-size precursor;
the paper's time-crystalline conclusion comes from lifetime scaling at larger
`N`.

## Further literature models

`examples/morrison2008_cooperative_fluorescence.jl` implements Eq. (1) of
Morrison and Parkins, *Phys. Rev. A* **77**, 043810 (2008), and compares every
finite-size stationary state from `stationary_state` with their exact Eq. (2).
The script explicitly converts the package's `|g>,|e>` ordering to the spin
axes used in the paper and shares one `OneBodyGeometry` between two prepared
collective observables.

`examples/meiser2009_steady_superradiance.jl` implements Eq. (1) of Meiser and
Holland, *Phys. Rev. A* **81**, 033847 (2010). It compiles each parameter point,
uses a typed direct stationary-state solve, reports the collective intensity
of Eq. (2), and compares its maximum with the large-`N` prediction
``N^2\Gamma_c/8`` from Eq. (10).

`examples/iemini2018_boundary_time_crystal.jl` implements Eq. (2) of Iemini
*et al.*, *Phys. Rev. Lett.* **121**, 035301 (2018). It uses the high-level
`liouvillian_spectrum` ordering and contrasts the gapped
``\omega_0/\kappa<1`` regime with the time-crystalline
``\omega_0/\kappa>1`` regime, where finite-size oscillatory modes approach the
imaginary axis as `N` increases.

`examples/piccitto2021_interacting_time_crystal.jl` implements the nonlinear
`p=2,q=1` collective-spin model of Piccitto *et al.*, *Phys. Rev. B* **104**,
014307 (2021). Its Appendix-D pair Hamiltonian is algebraically identical to
the paper's normalized `Jz^2` term up to a scalar identity. Complete small
symmetric-sector spectra resolve the complex branch whose decay decreases
with size; the script does not infer the paper's approximate `N^-0.4` law from
only three small systems.

`examples/nakanishi2023_pt_time_crystal.jl` implements the balanced-gain/loss
model of Nakanishi and Sasamoto, *Phys. Rev. A* **107**, L010201 (2023). It
matches every finite-size eigenvalue to their exact Eq. (14), verifies the
gap `4kappa/N` and uniform irrep steady state, then checks the exact damped
magnetization with matrix-free evolution at a larger `N`.

`examples/meanfield_time_crystal.jl` lowers the same physical terms through
`MeanFieldPlan` and compares the finite product closure, its thermodynamic
limit, and exact matrix-free PI dynamics. The finite closure retains the
one-site dissipator and reproduces the damped magnetization in this balanced
model; the thermodynamic rule removes that subleading term and yields the
undamped oscillation. See [Mean-field predictions](meanfield.md) for the
general closure equations and supported term set.
