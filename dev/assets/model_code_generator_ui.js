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
  const DEFAULT_SCAN = {
    enabled: false,
    axes: [
      {
        parameter: String.raw`\Omega`,
        start: 0,
        stop: 1,
        points: 11,
      },
    ],
  };
  const STORAGE_KEY =
    "PermutationalInvariantDynamics.modelCodeGenerator.manifest.v1";
  const SHARE_PREFIX = "#pid-model=";

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
    const scanAxisContainer = root.querySelector("#pid-scan-axes");
    const output = root.querySelector("#pid-generated-code");
    const status = root.querySelector("#pid-generator-status");
    const warnings = root.querySelector("#pid-generator-warnings");
    const summary = root.querySelector("#pid-generator-summary");
    const copyButton = root.querySelector("#pid-copy-code");
    const downloadButton = root.querySelector("#pid-download-code");
    const plutoButton = root.querySelector("#pid-download-pluto");
    const bundleButton = root.querySelector("#pid-download-bundle");
    const loadManifestButton = root.querySelector("#pid-load-manifest");
    const manifestFile = root.querySelector("#pid-manifest-file");
    const shareButton = root.querySelector("#pid-copy-share-link");
    const undoButton = root.querySelector("#pid-undo-model");
    const resetButton = root.querySelector("#pid-reset-model");
    let generatedCode = "";
    let generatedBundle = null;
    let generatedManifest = null;
    let undoManifest = null;
    let suppressHistory = false;

    function labelledControl(labelText, control) {
      const wrapper = element("label", { className: "pid-jump-control" });
      const label = element("span", { className: "pid-mini-label", text: labelText });
      wrapper.append(label, control);
      return wrapper;
    }

    function labelledScanControl(labelText, control) {
      const wrapper = element("label", { className: "pid-scan-control" });
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

    function addScanAxis(axis) {
      const value = Object.assign({}, DEFAULT_SCAN.axes[0], axis || {});
      const row = element("div", { className: "pid-scan-axis-row" });

      const parameter = element("input", {
        type: "text",
        value: value.parameter,
        placeholder: String.raw`\gamma`,
      });
      parameter.className = "pid-latex-input pid-scan-parameter";
      parameter.setAttribute("aria-label", "Scanned parameter in LaTeX");

      const start = element("input", {
        type: "number",
        value: value.start,
      });
      start.className = "pid-scan-start";
      start.step = "any";
      start.setAttribute("aria-label", "Scan start value");

      const stop = element("input", {
        type: "number",
        value: value.stop,
      });
      stop.className = "pid-scan-stop";
      stop.step = "any";
      stop.setAttribute("aria-label", "Scan stop value");

      const points = element("input", {
        type: "number",
        value: value.points,
      });
      points.className = "pid-scan-points";
      points.min = "2";
      points.step = "1";
      points.setAttribute("aria-label", "Number of scan points");

      const remove = element("button", {
        type: "button",
        className: "pid-button pid-button-quiet pid-remove-scan-axis",
        text: "Remove",
      });
      remove.setAttribute("aria-label", "Remove this scan axis");
      remove.addEventListener("click", function () {
        row.remove();
      });

      row.append(
        labelledScanControl("Parameter", parameter),
        labelledScanControl("Start", start),
        labelledScanControl("Stop", stop),
        labelledScanControl("Points", points),
        remove,
      );
      scanAxisContainer.append(row);
    }

    function presetScan(preset) {
      if (preset.scan) {
        return {
          enabled: Boolean(preset.scan.enabled),
          axes: (preset.scan.axes || []).map((axis) => Object.assign({}, axis)),
        };
      }
      const firstAssignment = String(preset.parameters || "")
        .split(/\r?\n/)
        .map((line) => line.trim())
        .find((line) => line.includes("="));
      const parameter = firstAssignment
        ? firstAssignment.split("=", 1)[0].trim()
        : DEFAULT_SCAN.axes[0].parameter;
      return {
        enabled: false,
        axes: [Object.assign({}, DEFAULT_SCAN.axes[0], { parameter })],
      };
    }

    function applyConfiguration(preset) {
      root.querySelector("#pid-architecture").value = preset.architecture || "pi";
      root.querySelector("#pid-particle-count").value =
        preset.N === undefined ? 8 : preset.N;
      root.querySelector("#pid-local-dimension").value =
        preset.d === undefined ? 2 : preset.d;
      root.querySelector("#pid-calculation").value =
        preset.calculation || "steady-observable";
      root.querySelector("#pid-workflow").value =
        preset.workflow || "direct-api";
      root.querySelector("#pid-steady-method").value =
        preset.steadyMethod || "deterministic";
      root.querySelector("#pid-hamiltonian").value = preset.hamiltonian || "";
      root.querySelector("#pid-parameters").value = preset.parameters || "";
      root.querySelector("#pid-observable").value = preset.observable || "";
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
      const scan = presetScan(preset);
      root.querySelector("#pid-scan-enabled").checked = scan.enabled;
      scanAxisContainer.replaceChildren();
      (scan.axes.length ? scan.axes : DEFAULT_SCAN.axes).forEach(addScanAxis);
      root.querySelector("#pid-memory-budget").value =
        preset.memoryBudgetMiB || 512;
      const pseudomode = Object.assign(
        {}, DEFAULT_PSEUDOMODE, preset.pseudomode || {},
      );
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
      (preset.jumps || []).forEach(addJump);
      updateVisibility();
      generate();
    }

    function loadPreset(name) {
      applyConfiguration(PRESETS[name] || PRESETS.driven);
    }

    function readJumps() {
      return Array.from(jumpContainer.querySelectorAll(".pid-jump-row")).map((row) => ({
        kind: row.querySelector("select").value,
        operator: row.querySelectorAll("input")[0].value,
        rate: row.querySelectorAll("input")[1].value,
      }));
    }

    function readScanAxes() {
      function numericValue(input) {
        const raw = input.value.trim();
        return raw === "" ? "" : Number(raw);
      }
      return Array.from(
        scanAxisContainer.querySelectorAll(".pid-scan-axis-row"),
      ).map((row) => ({
        parameter: row.querySelector(".pid-scan-parameter").value,
        start: numericValue(row.querySelector(".pid-scan-start")),
        stop: numericValue(row.querySelector(".pid-scan-stop")),
        points: numericValue(row.querySelector(".pid-scan-points")),
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
        scan: {
          enabled: root.querySelector("#pid-scan-enabled").checked,
          axes: readScanAxes(),
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
        scan: "#pid-scan-enabled",
        "parameter scan": "#pid-scan-enabled",
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
      const scanMatch =
        /^scan (?:parameter|axis) (\d+)(?: (parameter|start|stop|points|range))?$/.exec(field);
      if (!node && scanMatch) {
        const row = scanAxisContainer.querySelectorAll(".pid-scan-axis-row")[
          Number(scanMatch[1]) - 1
        ];
        if (row) {
          const part = scanMatch[2] || "parameter";
          node = part === "range"
            ? row
            : row.querySelector(`.pid-scan-${part}`);
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
        Array.isArray(result.summary.scanAxes) &&
          result.summary.scanAxes.length > 0
          ? `${result.summary.scanAxes.length}-axis parameter scan`
          : null,
        result.summary.scanPoints === null ||
          result.summary.scanPoints === undefined
          ? null
          : `${result.summary.scanPoints} total scan point${
            result.summary.scanPoints === 1 ? "" : "s"
          }`,
        Array.isArray(result.summary.scanAxes) &&
          result.summary.scanAxes.length > 0
          ? `axes: ${result.summary.scanAxes.join(", ")}`
          : null,
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

    function serializedManifest(manifest) {
      return JSON.stringify(manifest);
    }

    function saveLocalManifest(manifest) {
      try {
        window.localStorage.setItem(
          STORAGE_KEY, serializedManifest(manifest),
        );
      } catch (_) {
        // Private browsing and locked-down browsers may disable persistent
        // storage. Generation remains completely functional without it.
      }
    }

    function encodeShareManifest(manifest) {
      const bytes = new TextEncoder().encode(serializedManifest(manifest));
      let binary = "";
      for (const byte of bytes) binary += String.fromCharCode(byte);
      return window.btoa(binary)
        .replace(/\+/g, "-")
        .replace(/\//g, "_")
        .replace(/=+$/g, "");
    }

    function decodeShareManifest(encoded) {
      const normalized = encoded
        .replace(/-/g, "+")
        .replace(/_/g, "/");
      const padding = "=".repeat((4 - (normalized.length % 4)) % 4);
      const binary = window.atob(normalized + padding);
      const bytes = Uint8Array.from(
        binary, (character) => character.charCodeAt(0),
      );
      return JSON.parse(new TextDecoder().decode(bytes));
    }

    function loadManifestObject(manifest, message) {
      try {
        const configuration = api.configurationFromManifest(manifest);
        applyConfiguration(configuration);
        status.className = "pid-status pid-status-success";
        status.textContent = message;
        return true;
      } catch (error) {
        status.className = "pid-status pid-status-error";
        status.textContent = error instanceof api.GeneratorError
          ? `${error.field}: ${error.message}`
          : "The manifest could not be loaded.";
        return false;
      }
    }

    function generate() {
      clearMessages();
      try {
        const result = api.generate(readConfiguration());
        if (
          !suppressHistory &&
          generatedManifest &&
          serializedManifest(generatedManifest) !==
            serializedManifest(result.manifest)
        ) {
          undoManifest = generatedManifest;
        }
        generatedCode = result.code;
        generatedBundle = result.bundle;
        generatedManifest = result.manifest;
        saveLocalManifest(generatedManifest);
        output.textContent = generatedCode;
        renderSummary(result);
        renderWarnings(result.warnings);
        status.className = "pid-status pid-status-success";
        status.textContent = "Code generated locally in your browser.";
        copyButton.disabled = false;
        downloadButton.disabled = false;
        plutoButton.disabled = false;
        bundleButton.disabled = false;
        shareButton.disabled = false;
        undoButton.disabled = undoManifest === null;
      } catch (error) {
        generatedCode = "";
        generatedBundle = null;
        output.textContent = "# Correct the model input to generate Julia code.";
        copyButton.disabled = true;
        downloadButton.disabled = true;
        plutoButton.disabled = true;
        bundleButton.disabled = true;
        shareButton.disabled = generatedManifest === null;
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
      const scanEnabled = root.querySelector("#pid-scan-enabled").checked;

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

      const scanSection = root.querySelector("#pid-scan-section");
      scanSection.hidden = !scanEnabled;
      scanSection.querySelectorAll("input").forEach((input) => {
        input.required = scanEnabled;
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

    async function copyText(text, successMessage) {
      try {
        await navigator.clipboard.writeText(text);
        status.className = "pid-status pid-status-success";
        status.textContent = successMessage;
      } catch (_) {
        const helper = element("textarea");
        helper.value = text;
        helper.setAttribute("readonly", "");
        helper.className = "pid-copy-helper";
        document.body.append(helper);
        helper.select();
        const copied = typeof document.execCommand === "function" &&
          document.execCommand("copy");
        helper.remove();
        status.textContent = copied
          ? successMessage
          : "Clipboard access failed; select the code manually.";
      }
    }

    async function copyCode() {
      if (!generatedCode) return;
      await copyText(generatedCode, "Julia code copied to the clipboard.");
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

    function downloadPluto() {
      if (!generatedBundle) return;
      const artifact = generatedBundle.files.find(
        (entry) => entry.name.endsWith("_pluto.jl"),
      );
      if (artifact) downloadArtifact(artifact);
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
        "Experiment bundle downloaded: Julia script, JSON manifest, README, and Pluto notebook.";
    }

    async function copyShareLink() {
      if (!generatedManifest) return;
      const url = new URL(window.location.href);
      url.hash = `pid-model=${encodeShareManifest(generatedManifest)}`;
      await copyText(
        url.toString(), "Shareable model link copied to the clipboard.",
      );
    }

    async function loadManifestFile() {
      const file = manifestFile.files && manifestFile.files[0];
      if (!file) return;
      try {
        if (file.size > 2 * 1024 * 1024) {
          throw new Error("manifest too large");
        }
        const manifest = JSON.parse(await file.text());
        loadManifestObject(manifest, `Loaded ${file.name}.`);
      } catch (_) {
        status.className = "pid-status pid-status-error";
        status.textContent =
          "The selected file is not a valid supported generator manifest.";
      } finally {
        manifestFile.value = "";
      }
    }

    function undoModel() {
      if (!undoManifest) return;
      const target = undoManifest;
      undoManifest = generatedManifest;
      suppressHistory = true;
      try {
        const configuration = api.configurationFromManifest(target);
        applyConfiguration(configuration);
      } finally {
        suppressHistory = false;
      }
      undoButton.disabled = undoManifest === null;
      status.className = "pid-status pid-status-success";
      status.textContent = "Restored the previous generated model.";
    }

    function resetModel() {
      loadPreset("driven");
      const url = new URL(window.location.href);
      url.hash = "";
      window.history.replaceState(null, "", url.toString());
      status.className = "pid-status pid-status-success";
      status.textContent = "Reset to the driven-qubit starter model.";
    }

    function restoreSession() {
      if (window.location.hash.startsWith(SHARE_PREFIX)) {
        try {
          const encoded = window.location.hash.slice(SHARE_PREFIX.length);
          if (loadManifestObject(
            decodeShareManifest(encoded),
            "Loaded the model from the share link.",
          )) return true;
        } catch (_) {
          status.className = "pid-status pid-status-error";
          status.textContent =
            "The model share link is invalid or no longer supported.";
        }
      }
      try {
        const stored = window.localStorage.getItem(STORAGE_KEY);
        if (stored) {
          if (loadManifestObject(
            JSON.parse(stored), "Restored the last local model.",
          )) return true;
        }
      } catch (_) {
        // Ignore unavailable or malformed local storage and use the starter.
      }
      return false;
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
    root.querySelector("#pid-add-scan-axis").addEventListener("click", function () {
      addScanAxis({ parameter: "" });
      updateVisibility();
    });
    root.querySelector("#pid-preset").addEventListener("change", function (event) {
      loadPreset(event.target.value);
    });
    for (const selector of [
      "#pid-calculation",
      "#pid-steady-method",
      "#pid-architecture",
      "#pid-workflow",
      "#pid-scan-enabled",
    ]) {
      root.querySelector(selector).addEventListener("change", function () {
        updateVisibility();
        generate();
      });
    }
    copyButton.addEventListener("click", copyCode);
    downloadButton.addEventListener("click", downloadCode);
    plutoButton.addEventListener("click", downloadPluto);
    bundleButton.addEventListener("click", downloadBundle);
    loadManifestButton.addEventListener("click", function () {
      manifestFile.click();
    });
    manifestFile.addEventListener("change", loadManifestFile);
    shareButton.addEventListener("click", copyShareLink);
    undoButton.addEventListener("click", undoModel);
    resetButton.addEventListener("click", resetModel);

    if (!restoreSession()) loadPreset("driven");
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initialize);
  } else {
    initialize();
  }
})();
