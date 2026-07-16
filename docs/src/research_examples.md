# Research examples

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
pseudo-state unraveling. It also records why current Keeling cavity examples
cannot yet be reproduced by this spin-only package.

With the examples-only CairoMakie environment active, both trajectory scripts
also render the numerical comparison and its one-standard-error bands. The
Mølmer benchmark adds a logarithmic state-error panel; the Zhang--Mølmer
figure places the two decay-rate ratios side by side. PDF and PNG outputs are
written without adding Makie to the package dependency graph.

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
    method=:krylov, nev=6, krylovdim=40, vectors=true)
gap = pi_liouvillian_gap(model;
    method=:krylov, nev=6, krylovdim=40, return_info=true)
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
F = floquet_propagator(prepared, period; steps=160)
periodic = stationary_state(F - I; basis=basis,
                            algorithm=SVDAlgorithm(), return_info=true)
rhoF = periodic.state
states = stroboscopic_evolution(rho0, F, 4)
```

Solving `F-I` reuses the already converged one-period channel. The convenience
`floquet_steady_state(model, period; ...)` remains useful when the propagator is
not otherwise needed.

Increase `steps` until the relevant propagator, multiplier, or observable has
converged. Stiff protocols may require a dedicated SciML integration rather
than the fixed-step Floquet helper.

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
