(function () {
  "use strict";
  if (typeof document === "undefined") return;

  const PRESETS = {
    driven: {
      architecture: "pi",
      N: 8,
      d: 2,
      calculation: "steady-observable",
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
      architecture: "pi",
      N: 20,
      d: 2,
      calculation: "steady-observable",
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
      architecture: "pi",
      N: 40,
      d: 2,
      calculation: "steady-observable",
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
      architecture: "pi",
      N: 6,
      d: 3,
      calculation: "steady-observable",
      hamiltonian: String.raw`\Omega J_x + \Delta J_z`,
      jumps: [
        { kind: "local", operator: "j_-", rate: String.raw`\gamma` },
      ],
      observable: "J_z/N",
      parameters: String.raw`\Omega = 0.5
\Delta = 0.2
\gamma = 0.1`,
    },
    dynamics: {
      architecture: "pi",
      N: 8,
      d: 2,
      calculation: "dynamics-observable",
      steadyMethod: "deterministic",
      hamiltonian: String.raw`\Omega J_x`,
      jumps: [
        { kind: "local", operator: String.raw`\sigma_-`, rate: String.raw`\gamma` },
      ],
      observable: "J_z/N",
      parameters: String.raw`\Omega = 0.7
\gamma = 0.12`,
      initialState: { level: 2 },
      dynamics: {
        startTime: 0,
        finalTime: 5,
        samples: 51,
        stepsPerInterval: 16,
      },
    },
    trajectoryDynamics: {
      architecture: "pi",
      N: 8,
      d: 2,
      calculation: "dynamics-observable",
      steadyMethod: "trajectory",
      hamiltonian: String.raw`\Omega J_x`,
      jumps: [
        { kind: "local", operator: String.raw`\sigma_-`, rate: String.raw`\gamma` },
      ],
      observable: "J_z/N",
      parameters: String.raw`\Omega = 0.7
\gamma = 0.12`,
      initialState: { level: 2 },
      dynamics: {
        startTime: 0,
        finalTime: 5,
        samples: 51,
        stepsPerInterval: 16,
      },
      trajectory: {
        trajectories: 256,
        dt: 0.002,
        maxJumpProbability: 0.02,
        seed: 2026,
      },
    },
    spectrum: {
      architecture: "pi",
      N: 8,
      d: 2,
      calculation: "liouvillian-spectrum",
      steadyMethod: "deterministic",
      hamiltonian: String.raw`\Omega J_x`,
      jumps: [
        { kind: "local", operator: String.raw`\sigma_-`, rate: String.raw`\gamma` },
      ],
      observable: "J_z/N",
      parameters: String.raw`\Omega = 0.7
\gamma = 0.12`,
      spectrum: { target: "near-zero", nev: 6, seed: 2026 },
    },
    gap: {
      architecture: "pi",
      N: 8,
      d: 2,
      calculation: "liouvillian-gap",
      steadyMethod: "deterministic",
      hamiltonian: String.raw`\Omega J_x`,
      jumps: [
        { kind: "local", operator: String.raw`\sigma_-`, rate: String.raw`\gamma` },
      ],
      observable: "J_z/N",
      parameters: String.raw`\Omega = 0.7
\gamma = 0.12`,
      gap: { nev: 8, krylovdim: 32 },
    },
    localPseudomode: {
      architecture: "local-pseudomode",
      N: 3,
      d: 2,
      calculation: "steady-observable",
      hamiltonian: String.raw`\Omega J_x`,
      jumps: [],
      observable: "J_z/N",
      parameters: String.raw`\Omega = 0.4
\omega_c = 1.0
\kappa = 0.2
g = 0.15`,
      pseudomode: {
        nmax: 2,
        frequency: String.raw`\omega_c`,
        damping: String.raw`\kappa`,
        thermalOccupation: "0",
        couplingOperator: String.raw`\sigma_z`,
        couplingStrength: "g",
        counterrotatingStrength: "0",
      },
    },
    globalPseudomode: {
      architecture: "global-pseudomode",
      N: 12,
      d: 2,
      calculation: "steady-observable",
      hamiltonian: String.raw`\Delta J_z + \Omega J_x`,
      jumps: [
        { kind: "local", operator: String.raw`\sigma_-`, rate: String.raw`\gamma` },
      ],
      observable: "J_z/N",
      parameters: String.raw`\Delta = 0.2
\Omega = 0.5
\gamma = 0.01
\omega_c = 1.0
\kappa = 0.15
g = 0.1`,
      pseudomode: {
        nmax: 4,
        frequency: String.raw`\omega_c`,
        damping: String.raw`\kappa`,
        thermalOccupation: "0",
        couplingOperator: String.raw`\sigma_-`,
        couplingStrength: "g",
        counterrotatingStrength: "0",
      },
    },
  };

  const DEFAULT_PSEUDOMODE = {
    nmax: 2,
    frequency: String.raw`\omega_c`,
    damping: String.raw`\kappa`,
    thermalOccupation: "0",
    couplingOperator: String.raw`\sigma_-`,
    couplingStrength: "g",
    counterrotatingStrength: "0",
  };
  const DEFAULT_INITIAL_STATE = { level: 1 };
  const DEFAULT_TRAJECTORY = {
    trajectories: 512,
    dt: 0.002,
    maxJumpProbability: 0.02,
    seed: 2026,
    settlingTime: 50,
    samplesPerTrajectory: 5,
    samplingInterval: 2,
  };
  const DEFAULT_DYNAMICS = {
    startTime: 0,
    finalTime: 10,
    samples: 101,
    stepsPerInterval: 16,
  };
  const DEFAULT_SPECTRUM = {
    target: "largest-real",
    nev: 6,
    seed: 2026,
  };
  const DEFAULT_GAP = {
    nev: 8,
    krylovdim: 32,
  };
  const DEFAULT_ANALYSIS = {
    purity: false,
    entropy: false,
    oneBodyRDM: false,
    qfiAxis: "none",
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
    if (!root) return;
    document.documentElement.classList.add("pid-generator-page");
    if (!window.PIDModelCodeGenerator) return;
    const api = window.PIDModelCodeGenerator;
    const form = root.querySelector("#pid-generator-form");
    const jumpContainer = root.querySelector("#pid-jump-list");
    const output = root.querySelector("#pid-generated-code");
    const status = root.querySelector("#pid-generator-status");
    const warnings = root.querySelector("#pid-generator-warnings");
    const summary = root.querySelector("#pid-generator-summary");
    const copyButton = root.querySelector("#pid-copy-code");
    const downloadButton = root.querySelector("#pid-download-code");
    const bundleButton = root.querySelector("#pid-download-bundle");
    let generatedCode = "";
    let generatedBundle = null;

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
        text: "Independent local",
      });
      const collective = element("option", {
        value: "collective",
        text: "Collective",
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
      root.querySelector("#pid-architecture").value = preset.architecture || "pi";
      root.querySelector("#pid-particle-count").value = preset.N;
      root.querySelector("#pid-local-dimension").value = preset.d;
      root.querySelector("#pid-calculation").value =
        preset.calculation || "steady-observable";
      root.querySelector("#pid-workflow").value =
        preset.workflow || "direct-api";
      root.querySelector("#pid-steady-method").value =
        preset.steadyMethod || "deterministic";
      root.querySelector("#pid-hamiltonian").value = preset.hamiltonian;
      root.querySelector("#pid-parameters").value = preset.parameters;
      root.querySelector("#pid-observable").value = preset.observable;
      const initialState = preset.initialState || DEFAULT_INITIAL_STATE;
      root.querySelector("#pid-initial-level").value = initialState.level;
      const trajectory = Object.assign(
        {}, DEFAULT_TRAJECTORY, preset.trajectory || {},
      );
      root.querySelector("#pid-trajectory-count").value =
        trajectory.trajectories;
      root.querySelector("#pid-trajectory-dt").value = trajectory.dt;
      root.querySelector("#pid-trajectory-max-jump-probability").value =
        trajectory.maxJumpProbability;
      root.querySelector("#pid-trajectory-seed").value = trajectory.seed;
      root.querySelector("#pid-trajectory-settling-time").value =
        trajectory.settlingTime;
      root.querySelector("#pid-trajectory-samples").value =
        trajectory.samplesPerTrajectory;
      root.querySelector("#pid-trajectory-sampling-interval").value =
        trajectory.samplingInterval;
      const dynamics = Object.assign(
        {}, DEFAULT_DYNAMICS, preset.dynamics || {},
      );
      root.querySelector("#pid-dynamics-start-time").value =
        dynamics.startTime;
      root.querySelector("#pid-dynamics-final-time").value =
        dynamics.finalTime;
      root.querySelector("#pid-dynamics-samples").value = dynamics.samples;
      root.querySelector("#pid-dynamics-steps").value =
        dynamics.stepsPerInterval;
      const spectrum = Object.assign(
        {}, DEFAULT_SPECTRUM, preset.spectrum || {},
      );
      root.querySelector("#pid-spectrum-target").value = spectrum.target;
      root.querySelector("#pid-spectrum-nev").value = spectrum.nev;
      root.querySelector("#pid-spectrum-seed").value = spectrum.seed;
      const gap = Object.assign({}, DEFAULT_GAP, preset.gap || {});
      root.querySelector("#pid-gap-nev").value = gap.nev;
      root.querySelector("#pid-gap-krylovdim").value = gap.krylovdim;
      const analysis = Object.assign(
        {}, DEFAULT_ANALYSIS, preset.analysis || {},
      );
      root.querySelector("#pid-analysis-purity").checked = analysis.purity;
      root.querySelector("#pid-analysis-entropy").checked = analysis.entropy;
      root.querySelector("#pid-analysis-one-body-rdm").checked =
        analysis.oneBodyRDM;
      root.querySelector("#pid-analysis-qfi-axis").value = analysis.qfiAxis;
      root.querySelector("#pid-memory-budget").value =
        preset.memoryBudgetMiB || 512;
      const pseudomode = preset.pseudomode || DEFAULT_PSEUDOMODE;
      root.querySelector("#pid-pseudomode-cutoff").value = pseudomode.nmax;
      root.querySelector("#pid-pseudomode-frequency").value = pseudomode.frequency;
      root.querySelector("#pid-pseudomode-damping").value = pseudomode.damping;
      root.querySelector("#pid-pseudomode-thermal-occupation").value =
        pseudomode.thermalOccupation;
      root.querySelector("#pid-pseudomode-coupling-operator").value =
        pseudomode.couplingOperator;
      root.querySelector("#pid-pseudomode-coupling-strength").value =
        pseudomode.couplingStrength;
      root.querySelector("#pid-pseudomode-counterrotating-strength").value =
        pseudomode.counterrotatingStrength;
      jumpContainer.replaceChildren();
      preset.jumps.forEach(addJump);
      updateVisibility();
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
        architecture: root.querySelector("#pid-architecture").value,
        N: Number(root.querySelector("#pid-particle-count").value),
        d: Number(root.querySelector("#pid-local-dimension").value),
        calculation: root.querySelector("#pid-calculation").value,
        workflow: root.querySelector("#pid-workflow").value,
        steadyMethod: root.querySelector("#pid-steady-method").value,
        initialState: {
          level: Number(root.querySelector("#pid-initial-level").value),
        },
        trajectory: {
          trajectories:
            Number(root.querySelector("#pid-trajectory-count").value),
          settlingTime:
            Number(root.querySelector("#pid-trajectory-settling-time").value),
          dt: Number(root.querySelector("#pid-trajectory-dt").value),
          samplesPerTrajectory:
            Number(root.querySelector("#pid-trajectory-samples").value),
          samplingInterval:
            Number(root.querySelector(
              "#pid-trajectory-sampling-interval").value),
          maxJumpProbability:
            Number(root.querySelector(
              "#pid-trajectory-max-jump-probability").value),
          seed: Number(root.querySelector("#pid-trajectory-seed").value),
        },
        dynamics: {
          startTime:
            Number(root.querySelector("#pid-dynamics-start-time").value),
          finalTime:
            Number(root.querySelector("#pid-dynamics-final-time").value),
          samples: Number(root.querySelector("#pid-dynamics-samples").value),
          stepsPerInterval:
            Number(root.querySelector("#pid-dynamics-steps").value),
        },
        spectrum: {
          target: root.querySelector("#pid-spectrum-target").value,
          nev: Number(root.querySelector("#pid-spectrum-nev").value),
          seed: Number(root.querySelector("#pid-spectrum-seed").value),
        },
        gap: {
          nev: Number(root.querySelector("#pid-gap-nev").value),
          krylovdim: Number(root.querySelector("#pid-gap-krylovdim").value),
        },
        analysis: {
          purity: root.querySelector("#pid-analysis-purity").checked,
          entropy: root.querySelector("#pid-analysis-entropy").checked,
          oneBodyRDM:
            root.querySelector("#pid-analysis-one-body-rdm").checked,
          qfiAxis: root.querySelector("#pid-analysis-qfi-axis").value,
        },
        resources: {
          memoryBudgetMiB:
            Number(root.querySelector("#pid-memory-budget").value),
        },
        hamiltonian: root.querySelector("#pid-hamiltonian").value,
        jumps: readJumps(),
        observable: root.querySelector("#pid-observable").value,
        parameters: root.querySelector("#pid-parameters").value,
        pseudomode: {
          nmax: Number(root.querySelector("#pid-pseudomode-cutoff").value),
          frequency: root.querySelector("#pid-pseudomode-frequency").value,
          damping: root.querySelector("#pid-pseudomode-damping").value,
          thermalOccupation:
            root.querySelector("#pid-pseudomode-thermal-occupation").value,
          couplingOperator:
            root.querySelector("#pid-pseudomode-coupling-operator").value,
          couplingStrength:
            root.querySelector("#pid-pseudomode-coupling-strength").value,
          counterrotatingStrength:
            root.querySelector("#pid-pseudomode-counterrotating-strength").value,
        },
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
        architecture: "#pid-architecture",
        calculation: "#pid-calculation",
        workflow: "#pid-workflow",
        "calculation method": "#pid-steady-method",
        "steady-state method": "#pid-steady-method",
        N: "#pid-particle-count",
        d: "#pid-local-dimension",
        hamiltonian: "#pid-hamiltonian",
        observable: "#pid-observable",
        parameters: "#pid-parameters",
        "pseudomode cutoff": "#pid-pseudomode-cutoff",
        "pseudomode frequency": "#pid-pseudomode-frequency",
        "pseudomode damping": "#pid-pseudomode-damping",
        "pseudomode thermal occupation": "#pid-pseudomode-thermal-occupation",
        "pseudomode coupling operator": "#pid-pseudomode-coupling-operator",
        "pseudomode coupling strength": "#pid-pseudomode-coupling-strength",
        "pseudomode counter-rotating strength":
          "#pid-pseudomode-counterrotating-strength",
        trajectories: "#pid-trajectory-count",
        "initial local level": "#pid-initial-level",
        "settling time": "#pid-trajectory-settling-time",
        "trajectory dt": "#pid-trajectory-dt",
        "samples per trajectory": "#pid-trajectory-samples",
        "sampling interval": "#pid-trajectory-sampling-interval",
        "maximum jump probability":
          "#pid-trajectory-max-jump-probability",
        "trajectory seed": "#pid-trajectory-seed",
        "dynamics start time": "#pid-dynamics-start-time",
        "dynamics final time": "#pid-dynamics-final-time",
        "dynamics samples": "#pid-dynamics-samples",
        "steps per output interval": "#pid-dynamics-steps",
        "spectrum target": "#pid-spectrum-target",
        "spectrum eigenvalues": "#pid-spectrum-nev",
        "spectrum seed": "#pid-spectrum-seed",
        "gap eigenvalues": "#pid-gap-nev",
        "gap Krylov dimension": "#pid-gap-krylovdim",
        "QFI axis": "#pid-analysis-qfi-axis",
        "state analysis": "#pid-analysis-section",
        "memory budget": "#pid-memory-budget",
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
      const termDescription = result.summary.architecture === "pi"
        ? `${result.summary.terms} compiled term${result.summary.terms === 1 ? "" : "s"}`
        : `${result.summary.terms} bare-system term source${result.summary.terms === 1 ? "" : "s"}`;
      const coordinateDescription = result.summary.coordinates === null
        ? `coordinates: ${result.summary.coordinateFormula}`
        : result.summary.coordinates.length <= 24
          ? `${result.summary.coordinates} retained coordinates`
          : `${result.summary.coordinates.length}-digit exact coordinate count (see manifest)`;
      const items = [
        result.summary.topology,
        termDescription,
        `${result.summary.jumps} jump channel${result.summary.jumps === 1 ? "" : "s"}`,
        result.summary.workflow === "verified-experiment"
          ? "typed verified experiment"
          : "direct high-level API",
        result.summary.method,
        result.summary.cutoff === null || result.summary.cutoff === undefined
          ? null
          : `pseudomode cutoff nmax=${result.summary.cutoff}`,
        result.summary.route,
        coordinateDescription,
        result.summary.oneComplexVectorBytes === null
          ? null
          : result.summary.oneComplexVectorBytes.length <= 24
            ? `${result.summary.oneComplexVectorBytes} bytes per ComplexF64 vector`
            : `${result.summary.oneComplexVectorBytes.length}-digit one-vector byte count`,
      ].filter(Boolean);
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
        generatedBundle = result.bundle;
        output.textContent = generatedCode;
        renderSummary(result);
        renderWarnings(result.warnings);
        status.className = "pid-status pid-status-success";
        status.textContent = "Code generated locally in your browser.";
        copyButton.disabled = false;
        downloadButton.disabled = false;
        bundleButton.disabled = false;
      } catch (error) {
        generatedCode = "";
        generatedBundle = null;
        output.textContent = "# Correct the model input to generate Julia code.";
        copyButton.disabled = true;
        downloadButton.disabled = true;
        bundleButton.disabled = true;
        status.className = "pid-status pid-status-error";
        if (error instanceof api.GeneratorError) {
          status.textContent = `${error.field}: ${error.message}`;
          markField(error.field);
        } else {
          status.textContent = "Unexpected generator error. Please report this input as an issue.";
        }
      }
    }

    function updateVisibility() {
      const calculation = root.querySelector("#pid-calculation").value;
      const method = root.querySelector("#pid-steady-method").value;
      const architecture = root.querySelector("#pid-architecture").value;
      const stationary =
        calculation === "steady-state" ||
        calculation === "steady-observable";
      const dynamics = calculation === "dynamics-observable";
      const spectrum = calculation === "liouvillian-spectrum";
      const gap = calculation === "liouvillian-gap";
      const trajectory = method === "trajectory";

      function setRequired(section, visible, selector) {
        section.hidden = !visible;
        section.querySelectorAll(selector || "input, select").forEach((control) => {
          control.required = visible && control.type !== "checkbox";
        });
      }

      // Spectral calculations have only one valid deterministic method. If an
      // incompatible trajectory selection is retained while switching targets,
      // keep the selector visible so the core error can be corrected explicitly.
      root.querySelector("#pid-method-section").hidden =
        (spectrum || gap) && method === "deterministic";

      const needsInitialState = dynamics || (stationary && trajectory);
      setRequired(
        root.querySelector("#pid-initial-state-section"),
        needsInitialState,
      );

      const trajectoryVisible = trajectory && (stationary || dynamics);
      const trajectorySection = root.querySelector("#pid-trajectory-section");
      setRequired(trajectorySection, trajectoryVisible);
      const stationaryTrajectory =
        root.querySelector("#pid-trajectory-stationary-controls");
      stationaryTrajectory.hidden = !(trajectoryVisible && stationary);
      stationaryTrajectory.querySelectorAll("input").forEach((input) => {
        input.required = trajectoryVisible && stationary;
      });

      setRequired(root.querySelector("#pid-dynamics-section"), dynamics);
      const stepsField = root.querySelector("#pid-dynamics-steps-field");
      stepsField.hidden = dynamics && trajectory;
      root.querySelector("#pid-dynamics-steps").required =
        dynamics && !trajectory;

      setRequired(root.querySelector("#pid-spectrum-section"), spectrum);
      setRequired(root.querySelector("#pid-gap-section"), gap);

      const needsObservable =
        calculation === "steady-observable" || dynamics;
      const observableSection = root.querySelector("#pid-observable-section");
      observableSection.hidden = !needsObservable;
      root.querySelector("#pid-observable").required = needsObservable;

      const analysisSection = root.querySelector("#pid-analysis-section");
      const analysisSelected =
        root.querySelector("#pid-analysis-purity").checked ||
        root.querySelector("#pid-analysis-entropy").checked ||
        root.querySelector("#pid-analysis-one-body-rdm").checked ||
        root.querySelector("#pid-analysis-qfi-axis").value !== "none";
      // Keep an incompatible retained selection visible so it can be cleared;
      // changing the calculation never silently discards the user's analysis.
      analysisSection.hidden = !stationary && !analysisSelected;
      analysisSection.querySelectorAll("input, select").forEach((control) => {
        control.required = false;
      });

      const isPseudomode = architecture !== "pi";
      const panel = root.querySelector("#pid-pseudomode-section");
      panel.hidden = !isPseudomode;
      root.querySelector("#pid-local-pseudomode-description").hidden =
        architecture !== "local-pseudomode";
      root.querySelector("#pid-global-pseudomode-description").hidden =
        architecture !== "global-pseudomode";
      panel.querySelectorAll("input").forEach((input) => {
        input.required = isPseudomode;
      });
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

    function downloadArtifact(artifact) {
      const blob = new Blob(
        [artifact.contents], { type: artifact.mediaType },
      );
      const url = URL.createObjectURL(blob);
      const link = element("a");
      link.href = url;
      link.download = artifact.name;
      document.body.append(link);
      link.click();
      link.remove();
      window.setTimeout(function () {
        URL.revokeObjectURL(url);
      }, 1000);
    }

    function downloadCode() {
      if (!generatedBundle) return;
      downloadArtifact(generatedBundle.files[0]);
    }

    function downloadBundle() {
      if (!generatedBundle) return;
      // Keep the implementation dependency-free: each artifact is a normal
      // local browser download rather than a remotely assembled archive.
      for (const artifact of generatedBundle.files) {
        downloadArtifact(artifact);
      }
      status.className = "pid-status pid-status-success";
      status.textContent =
        "Experiment bundle downloaded: Julia script, JSON manifest, and README.";
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
    for (const selector of [
      "#pid-calculation",
      "#pid-steady-method",
      "#pid-architecture",
      "#pid-workflow",
    ]) {
      root.querySelector(selector).addEventListener("change", function () {
        updateVisibility();
        generate();
      });
    }
    copyButton.addEventListener("click", copyCode);
    downloadButton.addEventListener("click", downloadCode);
    bundleButton.addEventListener("click", downloadBundle);

    loadPreset("driven");
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initialize);
  } else {
    initialize();
  }
})();
