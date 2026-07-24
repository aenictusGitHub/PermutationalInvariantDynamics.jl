(function () {
  "use strict";
  if (typeof document === "undefined") return;

  const PRESETS = {
    driven: {
      N: 8,
      d: 2,
      target: "expectation",
      hamiltonian: String.raw`\frac{\Omega}{2}\sum_{i=1}^{N}\sigma_x^{(i)}`,
      jumps: [
        { kind: "local", operator: String.raw`\sigma_-`, rate: String.raw`\gamma_{\downarrow}` },
        { kind: "local", operator: String.raw`\sigma_+`, rate: String.raw`\gamma_{\uparrow}` },
      ],
      observable: "J_z/N",
      parameters: String.raw`\Omega = 0.7
\gamma_{\downarrow} = 0.12
\gamma_{\uparrow} = 0.02`,
    },
    collective: {
      N: 20,
      d: 2,
      target: "expectation",
      hamiltonian: String.raw`\omega J_z + \Omega J_x`,
      jumps: [
        { kind: "collective", operator: "J_-", rate: String.raw`\Gamma` },
        { kind: "local", operator: String.raw`\sigma_-`, rate: String.raw`\gamma` },
      ],
      observable: "J_z/N",
      parameters: String.raw`\omega = 1.0
\Omega = 0.4
\Gamma = 0.08
\gamma = 0.01`,
    },
    lmg: {
      N: 40,
      d: 2,
      target: "expectation",
      hamiltonian: String.raw`-\frac{2h}{N}J_z-\frac{2\lambda}{N}J_x^2`,
      jumps: [
        { kind: "collective", operator: "J_-", rate: String.raw`\Gamma/N` },
        { kind: "local", operator: String.raw`\sigma_-`, rate: String.raw`\gamma` },
      ],
      observable: "J_x^2/N^2",
      parameters: String.raw`h = 0.4
\lambda = 1.0
\Gamma = 0.2
\gamma = 0.01`,
    },
    qutrit: {
      N: 6,
      d: 3,
      target: "expectation",
      hamiltonian: String.raw`\Omega J_x + \Delta J_z`,
      jumps: [
        { kind: "local", operator: "j_-", rate: String.raw`\gamma` },
      ],
      observable: "J_z/N",
      parameters: String.raw`\Omega = 0.5
\Delta = 0.2
\gamma = 0.1`,
    },
  };

  function element(tag, options) {
    const node = document.createElement(tag);
    if (!options) return node;
    if (options.className) node.className = options.className;
    if (options.text) node.textContent = options.text;
    if (options.type) node.type = options.type;
    if (options.value !== undefined) node.value = options.value;
    if (options.placeholder) node.placeholder = options.placeholder;
    return node;
  }

  function initialize() {
    const root = document.getElementById("pid-code-generator");
    if (!root || !window.PIDModelCodeGenerator) return;
    const api = window.PIDModelCodeGenerator;
    const form = root.querySelector("#pid-generator-form");
    const jumpContainer = root.querySelector("#pid-jump-list");
    const output = root.querySelector("#pid-generated-code");
    const status = root.querySelector("#pid-generator-status");
    const warnings = root.querySelector("#pid-generator-warnings");
    const summary = root.querySelector("#pid-generator-summary");
    const copyButton = root.querySelector("#pid-copy-code");
    const downloadButton = root.querySelector("#pid-download-code");
    let generatedCode = "";

    function labelledControl(labelText, control) {
      const wrapper = element("label", { className: "pid-jump-control" });
      const label = element("span", { className: "pid-mini-label", text: labelText });
      wrapper.append(label, control);
      return wrapper;
    }

    function addJump(jump) {
      const row = element("div", { className: "pid-jump-row" });
      const kind = element("select");
      kind.setAttribute("aria-label", "Jump channel semantics");
      const local = element("option", {
        value: "local",
        text: "Independent local: sum_i D[l_i]",
      });
      const collective = element("option", {
        value: "collective",
        text: "Collective: D[sum_i l_i]",
      });
      kind.append(local, collective);
      kind.value = jump && jump.kind === "collective" ? "collective" : "local";

      const operator = element("input", {
        type: "text",
        value: jump ? jump.operator : String.raw`\sigma_-`,
        placeholder: String.raw`\sigma_- or J_-`,
      });
      operator.className = "pid-latex-input";
      operator.setAttribute("aria-label", "Jump operator in LaTeX");

      const rate = element("input", {
        type: "text",
        value: jump ? jump.rate : String.raw`\gamma`,
        placeholder: String.raw`\gamma`,
      });
      rate.className = "pid-latex-input pid-rate-input";
      rate.setAttribute("aria-label", "Jump rate in LaTeX");

      const remove = element("button", {
        type: "button",
        className: "pid-button pid-button-quiet pid-remove-jump",
        text: "Remove",
      });
      remove.setAttribute("aria-label", "Remove this jump channel");
      remove.addEventListener("click", function () {
        row.remove();
      });
      row.append(
        labelledControl("Channel", kind),
        labelledControl("Operator", operator),
        labelledControl("Rate", rate),
        remove,
      );
      jumpContainer.append(row);
    }

    function loadPreset(name) {
      const preset = PRESETS[name] || PRESETS.driven;
      root.querySelector("#pid-particle-count").value = preset.N;
      root.querySelector("#pid-local-dimension").value = preset.d;
      root.querySelector("#pid-target").value = preset.target;
      root.querySelector("#pid-hamiltonian").value = preset.hamiltonian;
      root.querySelector("#pid-parameters").value = preset.parameters;
      root.querySelector("#pid-observable").value = preset.observable;
      jumpContainer.replaceChildren();
      preset.jumps.forEach(addJump);
      updateObservableVisibility();
      generate();
    }

    function readJumps() {
      return Array.from(jumpContainer.querySelectorAll(".pid-jump-row")).map((row) => ({
        kind: row.querySelector("select").value,
        operator: row.querySelectorAll("input")[0].value,
        rate: row.querySelectorAll("input")[1].value,
      }));
    }

    function readConfiguration() {
      return {
        N: Number(root.querySelector("#pid-particle-count").value),
        d: Number(root.querySelector("#pid-local-dimension").value),
        target: root.querySelector("#pid-target").value,
        hamiltonian: root.querySelector("#pid-hamiltonian").value,
        jumps: readJumps(),
        observable: root.querySelector("#pid-observable").value,
        parameters: root.querySelector("#pid-parameters").value,
      };
    }

    function clearMessages() {
      warnings.replaceChildren();
      summary.replaceChildren();
      root.querySelectorAll("[aria-invalid='true']").forEach((node) => {
        node.removeAttribute("aria-invalid");
      });
    }

    function markField(field) {
      const mapping = {
        N: "#pid-particle-count",
        d: "#pid-local-dimension",
        hamiltonian: "#pid-hamiltonian",
        observable: "#pid-observable",
        parameters: "#pid-parameters",
      };
      const selector = mapping[field];
      let node = selector ? root.querySelector(selector) : null;
      const jumpMatch = /^jump (\d+)( rate)?$/.exec(field);
      if (!node && jumpMatch) {
        const row = jumpContainer.querySelectorAll(".pid-jump-row")[
          Number(jumpMatch[1]) - 1
        ];
        if (row) {
          node = row.querySelectorAll("input")[jumpMatch[2] ? 1 : 0];
        }
      }
      if (node) node.setAttribute("aria-invalid", "true");
    }

    function renderSummary(result) {
      const items = [
        `${result.summary.terms} compiled term${result.summary.terms === 1 ? "" : "s"}`,
        `${result.summary.jumps} jump channel${result.summary.jumps === 1 ? "" : "s"}`,
        result.summary.route,
      ];
      for (const text of items) {
        summary.append(element("span", { className: "pid-summary-chip", text }));
      }
    }

    function renderWarnings(resultWarnings) {
      for (const warning of resultWarnings) {
        warnings.append(element("li", { text: warning }));
      }
    }

    function generate() {
      clearMessages();
      try {
        const result = api.generate(readConfiguration());
        generatedCode = result.code;
        output.textContent = generatedCode;
        renderSummary(result);
        renderWarnings(result.warnings);
        status.className = "pid-status pid-status-success";
        status.textContent = "Code generated locally in your browser.";
        copyButton.disabled = false;
        downloadButton.disabled = false;
      } catch (error) {
        generatedCode = "";
        output.textContent = "# Correct the model input to generate Julia code.";
        copyButton.disabled = true;
        downloadButton.disabled = true;
        status.className = "pid-status pid-status-error";
        if (error instanceof api.GeneratorError) {
          status.textContent = `${error.field}: ${error.message}`;
          markField(error.field);
        } else {
          status.textContent = "Unexpected generator error. Please report this input as an issue.";
        }
      }
    }

    function updateObservableVisibility() {
      const visible = root.querySelector("#pid-target").value === "expectation";
      const section = root.querySelector("#pid-observable-section");
      section.hidden = !visible;
      root.querySelector("#pid-observable").required = visible;
    }

    async function copyCode() {
      if (!generatedCode) return;
      try {
        await navigator.clipboard.writeText(generatedCode);
        status.className = "pid-status pid-status-success";
        status.textContent = "Julia code copied to the clipboard.";
      } catch (_) {
        const helper = element("textarea");
        helper.value = generatedCode;
        helper.setAttribute("readonly", "");
        helper.className = "pid-copy-helper";
        document.body.append(helper);
        helper.select();
        const copied = typeof document.execCommand === "function" &&
          document.execCommand("copy");
        helper.remove();
        status.textContent = copied
          ? "Julia code copied to the clipboard."
          : "Clipboard access failed; select the code manually.";
      }
    }

    function downloadCode() {
      if (!generatedCode) return;
      const blob = new Blob([generatedCode], { type: "text/x-julia;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const link = element("a");
      link.href = url;
      link.download = "generated_pi_steady_state.jl";
      document.body.append(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    }

    form.addEventListener("submit", function (event) {
      event.preventDefault();
      generate();
    });
    root.querySelector("#pid-add-local-jump").addEventListener("click", function () {
      addJump({ kind: "local", operator: String.raw`\sigma_-`, rate: String.raw`\gamma` });
    });
    root.querySelector("#pid-add-collective-jump").addEventListener("click", function () {
      addJump({ kind: "collective", operator: "J_-", rate: String.raw`\Gamma` });
    });
    root.querySelector("#pid-preset").addEventListener("change", function (event) {
      loadPreset(event.target.value);
    });
    root.querySelector("#pid-target").addEventListener("change", updateObservableVisibility);
    copyButton.addEventListener("click", copyCode);
    downloadButton.addEventListener("click", downloadCode);

    loadPreset("driven");
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initialize);
  } else {
    initialize();
  }
})();
