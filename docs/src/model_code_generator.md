# PI model code generator

This browser-only assistant turns a small, explicit subset of LaTeX model
notation into a commented Julia program for a stationary PI calculation. It
uses physical package terms, the complete PI basis, prepared observable
geometry, the automatic sparse/matrix-free backend, and the high-level
memory-guarded stationary solver.

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
          </select>
        </label>
      </div>

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
        <span class="pid-label">Hamiltonian H in LaTeX</span>
        <textarea id="pid-hamiltonian"
          class="pid-latex-input"
          spellcheck="false"
          placeholder="\Omega J_x + \chi J_z^2"></textarea>
        <span class="pid-hint">
          Use collective spin symbols J<sub>x,y,z,+,-</sub>, or an explicit
          sum such as \sum_i \sigma_x^{(i)}. Leave empty for a purely
          dissipative model.
        </span>
      </label>

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
          Enter only the jump seed. The selector distinguishes
          \sum_i D[l_i] from D[\sum_i l_i]; the assistant never guesses this
          physically important choice.
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
      <div class="pid-code-shell">
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
| \(J_x,J_y,J_z,J_\pm\) | Collective spin \(J_a=\sum_i j_a^{(i)}\) |
| \(j_x,j_y,j_z,j_\pm\) | One-site spin-\((d-1)/2\) matrix, accepted for local jump seeds |
| \(\sigma_x,\sigma_y,\sigma_z,\sigma_\pm\) | Qubit Pauli/lowering/raising matrices; requires `d = 2` |
| \(\sum_i \sigma_a^{(i)}\) | Explicit Pauli sum; for \(a=x,y,z\), this is \(2J_a\) |
| `+`, `-`, products, `/`, `^2` | Scalar combinations and collective polynomials |
| `\frac{a}{b}`, `\sqrt{a}` | Scalar fraction and square root |
| named Greek or ASCII scalars | Parameters defined in the numerical-value box |

For Hamiltonians, linear collective expressions lower to
`LocalHamiltonian`, while collective polynomials such as \(J_z^2\) use a
compressed `PIOperator` and `DirectPIHamiltonian`. A local channel lowers to
`LocalJump`; a collective channel lowers to `CollectiveJump`. No
\(d^N\)-dimensional state or operator is generated.

The rate box multiplies the standard package dissipator
\(\mathcal D[L]\rho=L\rho L^\dagger-\{L^\dagger L,\rho\}/2\).
A coefficient placed inside \(L\) is therefore squared by the dissipator:
write either rate `\gamma` with operator `j_-`, or rate `1` with operator
`\sqrt{\gamma}j_-`, but do not count the same rate twice.

An unsummed local observable such as \(j_z\) is interpreted as the identical
one-site expectation
\(\langle j_z^{(1)}\rangle=N^{-1}\langle J_z\rangle\). For clarity in papers,
prefer writing the normalization explicitly as \(J_z/N\).

## Boundaries of the assistant

The generator rejects site-dependent couplings, nearest-neighbour chains,
arbitrary coupling graphs, time-dependent coefficients, tensor expressions,
and general LaTeX. Those cases require a deliberate translation using the
[framework guide](framework.md), [symmetric \(p\)-body terms](api/representation.md),
or [composite-system workflow](composite_systems.md).

This first version emits `Float64` model data. Wider or lower precision needs
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
