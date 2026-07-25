# PI model code generator

This browser-only assistant turns a small, explicit subset of LaTeX model
notation into a commented Julia program for a stationary PI calculation. It
supports an ordinary PI ensemble and two finite-cutoff Markovian embeddings:
one identical pseudomode per constituent or one pseudomode shared by the
whole ensemble. It uses physical package terms, complete PI bases, prepared
observable geometry, factorized composite operators, and memory-guarded
stationary solvers.

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
      assumptions, channel semantics, convergence, and stationary-state
      uniqueness before using a result.
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
        <select id="pid-target">
          <option value="expectation">Stationary state and expectation value</option>
          <option value="steady">Stationary density operator only</option>
        </select>
      </label>

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
| ``J_x,J_y,J_z,J_\pm`` | Collective spin ``J_a=\sum_i j_a^{(i)}`` |
| ``j_x,j_y,j_z,j_\pm`` | One-site spin-``(d-1)/2`` matrix, accepted for local jump seeds |
| ``\sigma_x,\sigma_y,\sigma_z,\sigma_\pm`` | Qubit Pauli/lowering/raising matrices; requires `d = 2` |
| ``\sum_i \sigma_a^{(i)}`` | Explicit Pauli sum; for ``a=x,y,z``, this is ``2J_a`` |
| `+`, `-`, products, `/`, `^2` | Scalar combinations and collective polynomials |
| `\frac{a}{b}`, `\sqrt{a}` | Scalar fraction and square root |
| named Greek or ASCII scalars | Parameters defined in the numerical-value box |

The pseudomode frequency, damping, thermal occupation, and two coupling
strengths use the same real-scalar grammar. The system--mode coupling seed
must be one linear local operator ``j_a`` or ``\sigma_a``. The generated
interaction is the package `PseudomodeCoupling`: the rotating part is
``g L a^\dagger + g^* L^\dagger a``, with an optional counter-rotating
strength. The assistant does not infer a coupling normalization or Kac
scaling.

For Hamiltonians, linear collective expressions lower to
`LocalHamiltonian`, while collective polynomials such as ``J_z^2`` use a
compressed `PIOperator` and `DirectPIHamiltonian`. A local channel lowers to
`LocalJump`; a collective channel lowers to `CollectiveJump`. No
``d^N``-dimensional state or operator is generated.

The rate box multiplies the standard package dissipator
``\mathcal D[L]\rho=L\rho L^\dagger-\{L^\dagger L,\rho\}/2``.
A coefficient placed inside ``L`` is therefore squared by the dissipator:
write either rate `\gamma` with operator `j_-`, or rate `1` with operator
`\sqrt{\gamma}j_-`, but do not count the same rate twice.

An unsummed local observable such as ``j_z`` is interpreted as the identical
one-site expectation
``\langle j_z^{(1)}\rangle=N^{-1}\langle J_z\rangle``. For clarity in papers,
prefer writing the normalization explicitly as ``J_z/N``.

## Composite and pseudomode routes

The architecture selector changes both the physical topology and the
generated numerical route:

| Architecture | Generated representation | Stationary route |
|:--|:--|:--|
| Ordinary PI ensemble | Complete `PIBasis(N, d)` | Automatic sparse/direct or matrix-free backend |
| Identical local pseudomodes | Complete PI basis of `PISupersite` objects | `pseudomode_model`, compiled once with the automatic memory guard |
| One shared pseudomode | PI system factor times one finite mode factor | `GlobalPseudomodeModel` with factorized, matrix-free GMRES |

For a local mode, the supersite dimension is
``D=d(n_{\max}+1)``, and the complete PI coordinate count grows as
``\binom{N+D^2-1}{N}``. Increasing the oscillator cutoff can therefore be
expensive even at modest ``N``. For one shared mode, the composite coordinate
count is `pi_dimension(system_basis) * (nmax + 1)^2`; the generated code never
forms its global Kronecker superoperator.

Both routes print the population of the highest retained oscillator level.
This is a useful warning signal, not a convergence proof: repeat the
calculation at larger `nmax` and compare the observables or reduced states of
interest. The global route evaluates a requested spin observable on
`trace_pseudomodes(rho_ss, embedding)`. The local route lifts the spin
observable into each system+mode supersite without reconstructing a
``D^N``-dimensional state.

## Boundaries of the assistant

The generator rejects site-dependent couplings, nearest-neighbour chains,
arbitrary coupling graphs, time-dependent coefficients, multiple pseudomodes
per constituent, and general LaTeX. The architecture selector covers the two
specific one-mode Markovian embeddings above; it does not parse arbitrary
generic composite tensor expressions such as `A \otimes B`. Those cases
require a deliberate translation using the [framework guide](framework.md),
[symmetric ``p``-body terms](api/representation.md),
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
The stationary solve checks convergence and validates the density operator,
but neither check proves uniqueness. Use the spectral and
[Evans-unicity tools](api/solvers.md) when uniqueness is part of the physical
claim.

After downloading, run the program from an environment containing the
package:

```sh
julia --project=. generated_pi_steady_state.jl
```
