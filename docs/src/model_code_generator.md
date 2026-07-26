# PI model code generator

This browser-only assistant turns a small, explicit subset of LaTeX model
notation into a commented Julia program for PI stationary states, streamed
observable dynamics, selected Liouvillian spectra, or certification-aware gap
estimation. It supports an ordinary PI ensemble and two finite-cutoff
Markovian embeddings: one identical pseudomode per constituent or one
pseudomode shared by the whole ensemble. It uses physical package terms,
complete PI bases, prepared observable geometry, factorized composite
operators, explicit memory budgets, and deterministic or trajectory routes
where the corresponding package API is available.

The assistant deliberately does **not** send formulas to a server or a
language model. Translation is deterministic and restricted: unsupported or
ambiguous notation produces an error instead of guessed physics.

```@raw html
<div id="pid-code-generator">
  <div class="pid-privacy-banner">
    <span class="pid-privacy-mark">Local</span>
    <div>
      <strong>Your formulas stay in this browser.</strong>
      This page generates code but cannot execute Julia. Review the PI
      assumptions, channel semantics, numerical convergence, and any
      stationarity, uniqueness, or spectral-completeness claim before using a
      result.
    </div>
  </div>

  <div class="pid-layout">
    <form id="pid-generator-form" class="pid-panel">
      <div class="pid-panel-heading">
        <h2>1. Describe the model</h2>
        <label class="pid-field">
          <span class="pid-mini-label">Start from</span>
          <select id="pid-preset">
            <option value="driven">Driven qubits with local bath</option>
            <option value="collective">Local and collective decay</option>
            <option value="lmg">Collective LMG polynomial</option>
            <option value="qutrit">Spin-1 qutrit ensemble</option>
            <option value="dynamics">Streaming relaxation dynamics</option>
            <option value="trajectoryDynamics">Trajectory relaxation dynamics</option>
            <option value="spectrum">Near-zero Liouvillian spectrum</option>
            <option value="gap">Liouvillian gap diagnostics</option>
            <option value="localPseudomode">Identical local pseudomodes</option>
            <option value="globalPseudomode">One shared pseudomode</option>
          </select>
        </label>
      </div>

      <label class="pid-field pid-architecture-field">
        <span class="pid-label">System architecture</span>
        <select id="pid-architecture">
          <option value="pi">Ordinary PI ensemble</option>
          <option value="local-pseudomode">
            Identical local pseudomode per constituent
          </option>
          <option value="global-pseudomode">
            One pseudomode shared by the ensemble
          </option>
        </select>
        <span class="pid-hint">
          This choice fixes the tensor topology. Local modes become part of
          each permuted supersite; a shared mode remains a separate composite
          factor.
        </span>
      </label>

      <div class="pid-grid-two">
        <label class="pid-field">
          <span class="pid-label">Particles, N</span>
          <input id="pid-particle-count" type="number" min="1" step="1"
                 value="8" required>
        </label>
        <label class="pid-field">
          <span class="pid-label">Local dimension, d</span>
          <input id="pid-local-dimension" type="number" min="2" step="1"
                 value="2" required>
        </label>
      </div>

      <label class="pid-field">
        <span class="pid-label">Requested calculation</span>
        <select id="pid-calculation">
          <option value="steady-observable">
            Stationary state and expectation value
          </option>
          <option value="steady-state">Stationary density operator only</option>
          <option value="dynamics-observable">
            Time-dependent expectation value
          </option>
          <option value="liouvillian-spectrum">
            Selected Liouvillian spectrum
          </option>
          <option value="liouvillian-gap">Liouvillian gap</option>
        </select>
      </label>

      <label id="pid-method-section" class="pid-field">
        <span class="pid-label">Calculation method</span>
        <select id="pid-steady-method">
          <option value="deterministic">
            Deterministic density-operator route
          </option>
          <option value="trajectory">
            Quantum trajectories
          </option>
        </select>
        <span class="pid-hint">
          Stationary trajectories are available for ordinary PI ensembles and
          identical local pseudomodes. Observable trajectories are also
          available for a shared global mode. Deterministic shared-mode
          observable dynamics is intentionally rejected until it has a
          memory-guarded state-free streamer.
        </span>
      </label>

      <fieldset id="pid-initial-state-section"
                class="pid-subpanel" hidden>
        <legend>Initial product state</legend>
        <label class="pid-field">
          <span class="pid-label">Initial local level, 1…d</span>
          <input id="pid-initial-level" type="number"
                 min="1" step="1" value="1">
        </label>
        <span class="pid-hint">
          Every physical system starts in this one-based computational level.
          Local and shared pseudomodes start in vacuum. Test other PI initial
          states manually when strong symmetries make the result
          initial-state dependent.
        </span>
      </fieldset>

      <fieldset id="pid-trajectory-section"
                class="pid-subpanel" hidden>
        <legend>Quantum-trajectory controls</legend>
        <div class="pid-grid-two">
          <label class="pid-field">
            <span class="pid-label">Independent trajectories</span>
            <input id="pid-trajectory-count" type="number" min="2" step="1"
                   value="512">
          </label>
          <label class="pid-field">
            <span class="pid-label">Integrator step, dt</span>
            <input id="pid-trajectory-dt" type="number"
                   min="0" step="any" value="0.002">
          </label>
          <label class="pid-field">
            <span class="pid-label">Maximum jump probability</span>
            <input id="pid-trajectory-max-jump-probability" type="number"
                   min="0" max="1" step="any" value="0.02">
          </label>
          <label class="pid-field">
            <span class="pid-label">Random seed</span>
            <input id="pid-trajectory-seed" type="number"
                   min="0" step="1" value="2026">
          </label>
        </div>
        <div id="pid-trajectory-stationary-controls" class="pid-grid-two">
          <label class="pid-field">
            <span class="pid-label">Settling time</span>
            <input id="pid-trajectory-settling-time" type="number"
                   min="0" step="any" value="50">
          </label>
          <label class="pid-field">
            <span class="pid-label">Samples per trajectory</span>
            <input id="pid-trajectory-samples" type="number"
                   min="1" step="1" value="5">
          </label>
          <label class="pid-field">
            <span class="pid-label">Sampling interval</span>
            <input id="pid-trajectory-sampling-interval" type="number"
                   min="0" step="any" value="2">
          </label>
        </div>
        <span class="pid-hint">
          These are explicit starting values, not convergence guarantees.
          Increase the path count and reduce <code>dt</code>. For stationary
          estimates, also converge the settling time, sampling interval, and
          sampling window.
        </span>
      </fieldset>

      <fieldset id="pid-dynamics-section"
                class="pid-subpanel" hidden>
        <legend>Observable dynamics</legend>
        <div class="pid-grid-two">
          <label class="pid-field">
            <span class="pid-label">Start time</span>
            <input id="pid-dynamics-start-time" type="number"
                   step="any" value="0">
          </label>
          <label class="pid-field">
            <span class="pid-label">Final time</span>
            <input id="pid-dynamics-final-time" type="number"
                   step="any" value="10">
          </label>
          <label class="pid-field">
            <span class="pid-label">Output samples</span>
            <input id="pid-dynamics-samples" type="number"
                   min="2" step="1" value="101">
          </label>
          <label id="pid-dynamics-steps-field" class="pid-field">
            <span class="pid-label">RK4 steps per output interval</span>
            <input id="pid-dynamics-steps" type="number"
                   min="1" step="1" value="16">
          </label>
        </div>
        <span class="pid-hint">
          Deterministic dynamics streams one observable with matrix-free RK4.
          Trajectory dynamics instead reports the ensemble mean, standard
          error, and 95% confidence interval. Neither route retains state
          histories.
        </span>
      </fieldset>

      <fieldset id="pid-spectrum-section"
                class="pid-subpanel" hidden>
        <legend>Selected Liouvillian spectrum</legend>
        <div class="pid-grid-two">
          <label class="pid-field">
            <span class="pid-label">Spectral target</span>
            <select id="pid-spectrum-target">
              <option value="largest-real">Largest real part</option>
              <option value="near-zero">Near zero</option>
              <option value="largest-magnitude">Largest magnitude</option>
            </select>
          </label>
          <label class="pid-field">
            <span class="pid-label">Requested eigenvalues</span>
            <input id="pid-spectrum-nev" type="number"
                   min="1" step="1" value="6">
          </label>
          <label class="pid-field">
            <span class="pid-label">Random seed</span>
            <input id="pid-spectrum-seed" type="number"
                   min="0" step="1" value="2026">
          </label>
        </div>
        <span class="pid-hint">
          Automatic selection uses a complete dense spectrum only when it fits
          the budget; otherwise it uses a matrix-free selected eigensolver. A
          partial spectrum is not a certified global gap.
        </span>
      </fieldset>

      <fieldset id="pid-gap-section"
                class="pid-subpanel" hidden>
        <legend>Liouvillian gap</legend>
        <div class="pid-grid-two">
          <label class="pid-field">
            <span class="pid-label">Requested Ritz values</span>
            <input id="pid-gap-nev" type="number"
                   min="2" step="1" value="8">
          </label>
          <label class="pid-field">
            <span class="pid-label">Krylov dimension</span>
            <input id="pid-gap-krylovdim" type="number"
                   min="3" step="1" value="32">
          </label>
        </div>
        <span class="pid-hint">
          The Krylov dimension must exceed the Ritz count. The generated
          program prints stability, stationary multiplicity, and certification
          flags; never report the numerical gap as certified unless
          <code>gap_certified</code> is true.
        </span>
      </fieldset>

      <label class="pid-field">
        <span class="pid-label">Bare-system Hamiltonian H in LaTeX</span>
        <textarea id="pid-hamiltonian"
          class="pid-latex-input"
          spellcheck="false"
          placeholder="\Omega J_x + \chi J_z^2"></textarea>
        <span class="pid-hint">
          Use collective spin symbols J<sub>x,y,z,+,-</sub>, or an explicit
          sum such as \sum_i \sigma_x^{(i)}. Leave empty for a purely
          dissipative model. For a pseudomode architecture, do not add the
          oscillator frequency or system--mode coupling here; the structured
          controls below add them exactly once.
        </span>
      </label>

      <fieldset id="pid-pseudomode-section"
                class="pid-subpanel pid-pseudomode-panel" hidden>
        <legend>Finite-cutoff pseudomode</legend>
        <p id="pid-local-pseudomode-description"
           class="pid-architecture-note" hidden>
          Every constituent has its own identical mode. Permutations act on
          complete system+mode supersites, so the embedding remains one PI
          problem.
        </p>
        <p id="pid-global-pseudomode-description"
           class="pid-architecture-note" hidden>
          One mode is shared by all constituents. The generated solver keeps
          the PI system and oscillator as factorized composite coordinates.
        </p>

        <div class="pid-pseudomode-grid">
          <label class="pid-field">
            <span class="pid-label">Fock cutoff, nmax</span>
            <input id="pid-pseudomode-cutoff" type="number" min="0" step="1"
                   value="2">
            <span class="pid-hint">
              Retained levels are 0,…,nmax. Converge this approximation.
            </span>
          </label>
          <label class="pid-field">
            <span class="pid-label">Mode frequency</span>
            <input id="pid-pseudomode-frequency"
                   class="pid-latex-input" type="text" spellcheck="false"
                   value="\omega_c" placeholder="\omega_c">
          </label>
          <label class="pid-field">
            <span class="pid-label">Mode damping</span>
            <input id="pid-pseudomode-damping"
                   class="pid-latex-input" type="text" spellcheck="false"
                   value="\kappa" placeholder="\kappa">
          </label>
          <label class="pid-field">
            <span class="pid-label">Thermal occupation</span>
            <input id="pid-pseudomode-thermal-occupation"
                   class="pid-latex-input" type="text" spellcheck="false"
                   value="0" placeholder="nbar">
          </label>
          <label class="pid-field pid-span-two">
            <span class="pid-label">System coupling seed</span>
            <input id="pid-pseudomode-coupling-operator"
                   class="pid-latex-input" type="text" spellcheck="false"
                   value="\sigma_-" placeholder="\sigma_- or j_z">
            <span class="pid-hint">
              Enter one linear local spin operator. The selected architecture
              performs the corresponding local-supersite or collective lift.
            </span>
          </label>
          <label class="pid-field">
            <span class="pid-label">Rotating coupling strength</span>
            <input id="pid-pseudomode-coupling-strength"
                   class="pid-latex-input" type="text" spellcheck="false"
                   value="g" placeholder="g">
          </label>
          <label class="pid-field">
            <span class="pid-label">Counter-rotating strength</span>
            <input id="pid-pseudomode-counterrotating-strength"
                   class="pid-latex-input" type="text" spellcheck="false"
                   value="0" placeholder="0">
          </label>
        </div>
        <span class="pid-hint">
          With this package's dissipator convention, damping
          <code>kappa</code> makes the free-mode amplitude decay at
          <code>kappa/2</code>. The generated program reports the highest
          retained Fock-level population as a cutoff diagnostic.
        </span>
      </fieldset>

      <div class="pid-field">
        <span class="pid-label">Dissipative channels</span>
        <div id="pid-jump-list" class="pid-jump-list"></div>
        <div class="pid-button-row">
          <button id="pid-add-local-jump" type="button"
                  class="pid-button pid-button-quiet">+ Local channel</button>
          <button id="pid-add-collective-jump" type="button"
                  class="pid-button pid-button-quiet">+ Collective channel</button>
        </div>
        <span class="pid-hint">
          Enter only the bare-system jump seed. The selector distinguishes
          \sum_i D[l_i] from D[\sum_i l_i]; the assistant never guesses this
          physically important choice. In a pseudomode embedding, mode damping
          is set separately above.
        </span>
      </div>

      <label id="pid-observable-section" class="pid-field">
        <span class="pid-label">Observable in LaTeX</span>
        <input id="pid-observable"
               class="pid-latex-input"
               type="text"
               spellcheck="false"
               value="J_z/N"
               placeholder="J_z/N or J_x^2/N^2">
        <span class="pid-hint">
          A linear collective observable uses a prepared one-body contraction.
          A polynomial such as J_x^2 remains entirely in PI coordinates.
        </span>
      </label>

      <fieldset id="pid-analysis-section"
                class="pid-subpanel" hidden>
        <legend>Optional stationary-state analyses</legend>
        <div class="pid-analysis-grid">
          <label class="pid-check-field">
            <input id="pid-analysis-purity" type="checkbox">
            <span>System purity</span>
          </label>
          <label class="pid-check-field">
            <input id="pid-analysis-entropy" type="checkbox">
            <span>System von Neumann entropy</span>
          </label>
          <label class="pid-check-field">
            <input id="pid-analysis-one-body-rdm" type="checkbox">
            <span>One-body density matrix</span>
          </label>
          <label class="pid-field">
            <span class="pid-label">Collective QFI generator</span>
            <select id="pid-analysis-qfi-axis">
              <option value="none">None</option>
              <option value="x">Collective x axis</option>
              <option value="y">Collective y axis</option>
              <option value="z">Collective z axis</option>
            </select>
          </label>
        </div>
        <span class="pid-hint">
          For a pseudomode model these quantities refer to the physical systems
          after tracing out all retained pseudomodes. Purity, entropy, and QFI
          of a trajectory-averaged state are nonlinear plug-in estimates, not
          observables with the reported path standard error.
        </span>
      </fieldset>

      <label class="pid-field">
        <span class="pid-label">Numerical parameter values</span>
        <textarea id="pid-parameters"
          class="pid-latex-input"
          spellcheck="false"
          placeholder="\Omega = 0.7&#10;\gamma = 0.1"></textarea>
        <span class="pid-hint">
          One finite numerical assignment per line. Missing parameters receive
          an explicit <code>1.0 # TODO</code> placeholder in the output.
        </span>
      </label>

      <label class="pid-field">
        <span class="pid-label">Memory budget, MiB</span>
        <input id="pid-memory-budget" type="number"
               min="1" step="1" value="512" required>
        <span class="pid-hint">
          This explicit finite budget guards model preparation, task-owned
          workspaces, and requested output. Raise it deliberately only after
          inspecting the generated resource route.
        </span>
      </label>

      <button type="submit" class="pid-button pid-button-primary">
        Generate Julia code
      </button>
    </form>

    <section class="pid-panel" aria-labelledby="pid-output-heading">
      <div class="pid-panel-heading">
        <h2 id="pid-output-heading">2. Review and run</h2>
        <div class="pid-output-actions">
          <button id="pid-copy-code" type="button"
                  class="pid-button pid-button-quiet">Copy</button>
          <button id="pid-download-code" type="button"
                  class="pid-button pid-button-quiet">Download .jl</button>
        </div>
      </div>
      <div id="pid-generator-summary" class="pid-summary"></div>
      <div id="pid-generator-status" class="pid-status"
           role="status" aria-live="polite">
        Loading the generator…
      </div>
      <div class="pid-code-shell" role="region" tabindex="0"
           aria-labelledby="pid-output-heading">
        <pre id="pid-generated-code" class="pid-code"
             aria-label="Generated Julia code"></pre>
      </div>
      <ul id="pid-generator-warnings" class="pid-warning-list"></ul>
    </section>
  </div>
</div>
```

## Supported notation

The grammar is intentionally small enough that the generated physics is
auditable:

| LaTeX ingredient | Meaning in generated code |
|:--|:--|
| $J_x,J_y,J_z,J_\pm$ | Collective spin $J_a=\sum_i j_a^{(i)}$ |
| $j_x,j_y,j_z,j_\pm$ | One-site spin-$(d-1)/2$ matrix, accepted for local jump seeds |
| $\sigma_x,\sigma_y,\sigma_z,\sigma_\pm$ | Qubit Pauli/lowering/raising matrices; requires `d = 2` |
| $\sum_i \sigma_a^{(i)}$ | Explicit Pauli sum; for $a=x,y,z$, this is $2J_a$ |
| `+`, `-`, products, `/`, `^2` | Scalar combinations and collective polynomials |
| `\frac{a}{b}`, `\sqrt{a}` | Scalar fraction and square root |
| named Greek or ASCII scalars | Parameters defined in the numerical-value box |

The pseudomode frequency, damping, thermal occupation, and two coupling
strengths use the same real-scalar grammar. The system--mode coupling seed
must be one linear local operator $j_a$ or $\sigma_a$. The generated
interaction is the package `PseudomodeCoupling`: the rotating part is
$g L a^\dagger + g^* L^\dagger a$, with an optional counter-rotating
strength. The assistant does not infer a coupling normalization or Kac
scaling.

For Hamiltonians, linear collective expressions lower to
`LocalHamiltonian`, while collective polynomials such as $J_z^2$ use a
compressed `PIOperator` and `DirectPIHamiltonian`. A local channel lowers to
`LocalJump`; a collective channel lowers to `CollectiveJump`. No
$d^N$-dimensional state or operator is generated.

The rate box multiplies the standard package dissipator
$\mathcal D[L]\rho=L\rho L^\dagger-\{L^\dagger L,\rho\}/2$.
A coefficient placed inside $L$ is therefore squared by the dissipator:
write either rate `\gamma` with operator `j_-`, or rate `1` with operator
`\sqrt{\gamma}j_-`, but do not count the same rate twice.

An unsummed local observable such as $j_z$ is interpreted as the identical
one-site expectation
$\langle j_z^{(1)}\rangle=N^{-1}\langle J_z\rangle$. For clarity in papers,
prefer writing the normalization explicitly as $J_z/N$.

## Composite and pseudomode routes

The architecture selector fixes the physical topology:

| Architecture | Generated representation |
|:--|:--|
| Ordinary PI ensemble | Complete `PIBasis(N, d)` |
| Identical local pseudomodes | Complete PI basis of system+mode `PISupersite` objects |
| One shared pseudomode | PI system factor times one finite-mode operator factor |

For a local mode, the supersite dimension is
$D=d(n_{\max}+1)$, and the complete PI coordinate count grows as
$\binom{N+D^2-1}{N}$. Increasing the oscillator cutoff can therefore be
expensive even at modest $N$. For one shared mode, the composite coordinate
count is `pi_dimension(system_basis) * (nmax + 1)^2`; the generated code never
forms its global Kronecker superoperator.

Stationary-state programs for both pseudomode routes print the population of
the highest retained oscillator level. This is a useful warning signal, not a
convergence proof: repeat the calculation at larger `nmax` and compare the
observables or reduced states of interest. The global stationary route
evaluates a requested spin observable on
`trace_pseudomodes(rho_ss, embedding)`. The local route lifts the spin
observable into each system+mode supersite without reconstructing a
$D^N$-dimensional state.

## Calculation and method compatibility

The calculation and method selectors are independent because they answer
different questions: the first chooses the requested output, while the second
chooses deterministic density-operator propagation or stochastic
unravelling. The implemented combinations are:

| Calculation | Ordinary PI | Identical local modes | One shared mode |
|:--|:--|:--|:--|
| Deterministic stationary state or observable | Supported | Supported | Factorized matrix-free GMRES |
| Trajectory stationary state or observable | Supported | Supported | Not currently supported |
| Deterministic observable dynamics | State-free matrix-free stream | State-free PI-supersite stream | Not currently supported |
| Trajectory observable dynamics | Online path statistics | Online PI-supersite statistics | Factorized composite trajectories |
| Selected Liouvillian spectrum | Supported | Supported | Factorized matrix-free spectrum |
| Liouvillian gap | Krylov route with certification metadata | Same | Factorized matrix-free route |

The browser never silently changes an incompatible choice. It leaves the
selection visible and reports the core validation error, so the numerical
method is always an explicit user decision.

## Stationary states and optional analyses

For a deterministic stationary calculation, ordinary and local-pseudomode
programs prepare the Liouvillian and solve its trace-constrained null-vector
problem. The shared-global-mode program instead keeps the composite generator
factorized and passes the embedding directly to matrix-free GMRES. The
trajectory alternative prepares a channel-resolved `TrajectoryPlan`, reuses
a fixed-capacity `TrajectoryBatchWorkspace`, and calls
`trajectory_steady_state`. Post-settling states are reduced online, so no path
history is retained.

The trajectory route emits a reproducible seed and an explicit pure product
initial state. For a pseudomode model, each physical system starts in the
selected one-based local level and every finite mode starts in vacuum. The
stationary result reports the path-to-path Hilbert--Schmidt standard error
together with the Liouvillian residual, relative residual, and trace error.

Those numbers are diagnostics, not certificates. Independently converge the
settling time, fixed integration step, sampling interval and window, and
number of independent trajectories. Samples from the same trajectory are
correlated, so the estimator averages them within each path before computing
path-to-path uncertainty. Strong symmetries or a nonunique stationary
subspace can retain dependence on the initial product state; repeat the run
from other PI initial states when that distinction matters.

The history-free trajectory steady-state helper currently accepts ordinary
`PIModel`/`TrajectoryPlan` sources and the PI model produced by
`pseudomode_model`. It does not yet accept `GlobalPseudomodeModel`, so the
assistant rejects a trajectory request for one shared global mode instead of
silently emitting a different approximation.

Purity, base-two von Neumann entropy, the one-body density matrix, and
collective-axis QFI can be appended to either stationary route. For a
pseudomode architecture, these analyses refer to the physical-system state
after all retained modes have been traced out. The local-mode route prepares
and reuses an exact sparse `LocalFactorTracePlan`; it does not reconstruct the
supersite Hilbert space.

Purity, entropy, and QFI are nonlinear functions of a
trajectory-averaged density operator. Their values are therefore plug-in
estimates: the standard error reported for the state is not an uncertainty
bar for those nonlinear quantities. Converge them directly against independent
ensembles or path count.

## Streamed observable dynamics

Every dynamics program constructs an explicit computational product state and
a finite output grid. Deterministic ordinary-PI and local-pseudomode routes use
matrix-free RK4 through `solve_dynamics` with `save_states=false`; only the
requested observable series is retained. Refine the RK4 steps per output
interval until the curve is stable.

The trajectory route calls `quantum_trajectories` with fixed-capacity,
task-owned worker workspaces and online observable statistics. It prints time,
ensemble mean, standard error, and the selected confidence interval without
retaining trajectories or state histories. Refine both the fixed integration
step and the number of statistically independent paths.

Shared-mode trajectory dynamics monitors the explicit pseudomode damping
channels through a `CompositeTrajectoryPlan`. Bare-system jump terms remain in
the unconditional background and are not individually unravelled. This
distinction matters for jump-resolved physical interpretations even though the
unconditional evolution is retained.

## Spectra and gaps

The selected-spectrum calculation calls `liouvillian_spectrum` with an
explicit target, eigenvalue count, seed, and memory budget. Automatic selection
uses a complete dense spectrum only when its resource preflight fits; otherwise
it selects a matrix-free Krylov route. A selected set of modes is partial
unless the returned metadata establishes completeness, so it must not be
presented as a certified global gap.

The dedicated gap calculation instead calls `pi_liouvillian_gap` through a
largest-real Krylov route and prints the numerical gap together with stability,
stationary multiplicity, and certification flags. The Krylov dimension must
exceed the requested Ritz count. Treat the result as certified only when
`gap_certified` is true; a plausible positive number alone is insufficient.

## Boundaries of the assistant

The generator rejects site-dependent couplings, nearest-neighbour chains,
arbitrary coupling graphs, time-dependent coefficients, multiple pseudomodes
per constituent, and general LaTeX. The architecture selector covers the two
specific one-mode Markovian embeddings above; it does not parse arbitrary
generic composite tensor expressions such as `A \otimes B`. Those cases
require a deliberate translation using the [framework guide](framework.md),
[symmetric p-body terms](api/representation.md),
[local-pseudomode workflow](pseudomodes.md),
[shared-pseudomode workflow](global_pseudomodes.md), or the
[composite-system workflow](composite_systems.md).

The generator emits `Float64` model data. Wider or lower precision needs
explicitly typed matrices, geometry, tolerances, and a compatible
matrix-free solver; follow the [matrix-free Krylov guide](matrix_free_krylov.md)
rather than changing only one literal type.

The generated program uses `PIBasis(N, d)` rather than guessing a
fully-symmetric-sector restriction. It also does not guess a population-only
backend: those optimizations require structural certification by the library.
Stationary solves check convergence and validate the density operator;
trajectory calculations report statistical diagnostics; selected spectra
report solver metadata; and the gap route reports explicit certification
flags. None of these checks substitutes for the others. Use the
[Evans-unicity tools](api/solvers.md) when uniqueness is part of the physical
claim.

After downloading, run the program from an environment containing the
package:

```sh
julia --project=. generated_pi_deterministic_steady_observable.jl
```

Start a downloaded trajectory program with multiple threads to use its
preallocated path workers:

```sh
julia --threads=auto --project=. generated_pi_trajectory_dynamics_observable.jl
```

The generated filename records the architecture, method where applicable, and
calculation. For example, a selected spectrum uses
`generated_pi_liouvillian_spectrum.jl`.
