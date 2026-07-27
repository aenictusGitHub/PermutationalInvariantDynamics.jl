(function () {
  "use strict";

  const api = typeof PIDModelCodeGenerator !== "undefined"
    ? PIDModelCodeGenerator
    : require("../src/assets/model_code_generator_core.js");
  let passed = 0;

  function assert(condition, message) {
    if (!condition) throw new Error(message);
    passed += 1;
  }

  function assertIncludes(text, fragment, message) {
    assert(text.includes(fragment), `${message}\nMissing: ${fragment}\nIn:\n${text}`);
  }

  function assertRejects(action, field, message) {
    let caught = null;
    try {
      action();
    } catch (error) {
      caught = error;
    }
    assert(caught instanceof api.GeneratorError, `${message}: expected GeneratorError`);
    assert(caught.field === field, `${message}: expected field ${field}, got ${caught.field}`);
  }

  function assertRejectsScan(action, message) {
    let caught = null;
    try {
      action();
    } catch (error) {
      caught = error;
    }
    assert(caught instanceof api.GeneratorError, `${message}: expected GeneratorError`);
    assert(
      typeof caught.field === "string" && caught.field.includes("scan"),
      `${message}: expected a scan field, got ${caught.field}`,
    );
  }

  const driven = api.generate({
    N: 8,
    d: 2,
    target: "expectation",
    hamiltonian: String.raw`\frac{\Omega}{2}\sum_{i=1}^{N}\sigma_x^{(i)}`,
    jumps: [
      { kind: "local", operator: String.raw`\sigma_-`, rate: String.raw`\gamma_{\downarrow}` },
      { kind: "local", operator: String.raw`\sigma_+`, rate: String.raw`\gamma_{\uparrow}` },
      { kind: "collective", operator: "J_-", rate: String.raw`\Gamma` },
    ],
    observable: "J_z/N",
    parameters: String.raw`\Omega = 0.7
\gamma_{\downarrow} = 0.12
\gamma_{\uparrow} = 0.02
\Gamma = 0.01`,
  });

  // REUSE-IgnoreStart
  // These strings describe generated Julia output, not this test file.
  assertIncludes(
    driven.code,
    "# SPDX-FileCopyrightText: 2026 PermutationalInvariantDynamics.jl contributors",
    "generated Julia copyright identifier",
  );
  assertIncludes(
    driven.code,
    "# SPDX-License-Identifier: GPL-3.0-only",
    "generated Julia license identifier",
  );
  // REUSE-IgnoreEnd
  assertIncludes(
    driven.code,
    "with no option to use a later",
    "GPL-3.0-only generated-program notice",
  );
  assertIncludes(
    driven.bundle.files[2].contents,
    "GPL-3.0-only, without an output-license exception",
    "bundle README generated-program license",
  );
  assertIncludes(
    driven.bundle.files[2].contents,
    "The JSON file is descriptive metadata",
    "bundle README distinguishes the descriptive manifest",
  );
  assert(
    driven.bundle.files.length === 4,
    "the generated bundle includes a Pluto notebook",
  );
  assertIncludes(
    driven.bundle.files[3].contents,
    "### A Pluto.jl notebook ###",
    "Pluto notebook header",
  );
  assertIncludes(
    driven.bundle.files[3].contents,
    "using PermutationalInvariantDynamics",
    "Pluto notebook contains the generated calculation",
  );
  const restoredDrivenConfiguration =
    api.configurationFromManifest(driven.manifest);
  assert(
    restoredDrivenConfiguration.N === 8 &&
      restoredDrivenConfiguration.d === 2,
    "manifest round trip restores representation sizes",
  );
  assert(
    restoredDrivenConfiguration.jumps.length === 3 &&
      restoredDrivenConfiguration.scan.enabled === false,
    "manifest round trip restores channels and disabled scan state",
  );
  const restoredDriven = api.generate(restoredDrivenConfiguration);
  assert(
    restoredDriven.normalized.hamiltonian.trim() ===
      driven.normalized.hamiltonian.trim() &&
      restoredDriven.normalized.observable.trim() ===
        driven.normalized.observable.trim(),
    "manifest round trip preserves normalized formulas",
  );
  assertRejects(
    () => api.configurationFromManifest({ schema: "unknown/v1" }),
    "manifest",
    "unsupported manifest schema",
  );
  assertIncludes(driven.code, "LocalHamiltonian(", "one-body Hamiltonian lowering");
  assertIncludes(driven.code, "2 * spin.jx", "Pauli normalization");
  assertIncludes(driven.code, "LocalJump(jump_1;", "independent local jump");
  assertIncludes(driven.code, "CollectiveJump(jump_3;", "collective jump");
  assertIncludes(driven.code, "gamma_down = 0.12", "subscripted parameter");
  assertIncludes(driven.code, "CollectiveObservablePlan", "prepared observable");
  assertIncludes(driven.code, "backend=:auto", "automatic backend");
  assertIncludes(driven.code, "stationary_state(", "default deterministic stationary route");
  assert(
    !driven.code.includes("trajectory_steady_state("),
    "the default route must not silently select stochastic estimation",
  );
  assert(!driven.code.includes("TODO"), "provided parameters must not receive placeholders");
  assert(driven.summary.terms === 4, "term count");

  const trajectorySteady = api.generate({
    architecture: "pi",
    N: 3,
    d: 2,
    target: "steady",
    steadyMethod: "trajectory",
    hamiltonian: String.raw`\Omega J_x`,
    jumps: [
      { kind: "local", operator: String.raw`\sigma_-`, rate: String.raw`\gamma` },
    ],
    parameters: String.raw`\Omega=0.2
\gamma=0.1`,
    trajectory: {
      trajectories: 24,
      initialLevel: 2,
      settlingTime: 3.0,
      dt: 0.01,
      samplesPerTrajectory: 4,
      samplingInterval: 0.5,
      maxJumpProbability: 0.02,
      seed: 17,
    },
  });
  assertIncludes(
    trajectorySteady.code,
    "rho0 = computational_product_state(basis, INITIAL_LEVEL)",
    "trajectory route emits an explicit PI initial state",
  );
  assertIncludes(
    trajectorySteady.code,
    "TrajectoryPlan(model)",
    "trajectory route prepares channel-resolved kernels once",
  );
  assertIncludes(
    trajectorySteady.code,
    "trajectory_preflight = recommend_solver(",
    "trajectory route performs a guarded preparation preflight",
  );
  assertIncludes(
    trajectorySteady.code,
    "memory_budget=MEMORY_BUDGET",
    "trajectory estimator receives the declared memory budget",
  );
  assertIncludes(
    trajectorySteady.code,
    "TrajectoryBatchWorkspace(",
    "trajectory route reuses task-owned batch scratch",
  );
  assertIncludes(
    trajectorySteady.code,
    "trajectory_steady_state(",
    "trajectory stationary estimator",
  );
  assertIncludes(
    trajectorySteady.code,
    "return_info=true",
    "trajectory diagnostics are retained",
  );
  assertIncludes(
    trajectorySteady.code,
    "const TRAJECTORIES = 24",
    "trajectory count is emitted explicitly",
  );
  assertIncludes(
    trajectorySteady.code,
    "const INITIAL_LEVEL = 2",
    "trajectory initial level is emitted explicitly",
  );
  assertIncludes(
    trajectorySteady.code,
    "const SETTLING_TIME = 3.0",
    "trajectory settling time is emitted explicitly",
  );
  assertIncludes(
    trajectorySteady.code,
    "const TRAJECTORY_DT = 0.01",
    "trajectory time step is emitted explicitly",
  );
  assertIncludes(
    trajectorySteady.code,
    "const SAMPLES_PER_TRAJECTORY = 4",
    "within-path sample count is emitted explicitly",
  );
  assertIncludes(
    trajectorySteady.code,
    "const SAMPLING_INTERVAL = 0.5",
    "within-path sampling interval is emitted explicitly",
  );
  assertIncludes(
    trajectorySteady.code,
    "const MAX_JUMP_PROBABILITY = 0.02",
    "jump-probability guard is emitted explicitly",
  );
  assertIncludes(
    trajectorySteady.code,
    "const TRAJECTORY_SEED = 17",
    "trajectory seed is emitted explicitly",
  );
  assert(
    !trajectorySteady.code.includes("stationary_state("),
    "trajectory route must not also invoke a deterministic stationary solver",
  );
  assert(
    !trajectorySteady.code.includes("steady.info."),
    "trajectory result fields must not be confused with deterministic solver info",
  );

  const nonlinear = api.generate({
    N: 20,
    d: 2,
    target: "expectation",
    hamiltonian: String.raw`-\frac{2h}{N}J_z-\frac{2\lambda}{N}J_x^2`,
    jumps: [{ kind: "collective", operator: "J_-", rate: String.raw`\Gamma` }],
    observable: "J_x^2/N^2",
    parameters: String.raw`h=0.4
\lambda=1.0
\Gamma=0.2`,
  });
  assertIncludes(nonlinear.code, "DirectPIHamiltonian(H_collective)", "nonlinear PI Hamiltonian");
  assertIncludes(nonlinear.code, "Jx * Jx", "PIOperator powers use multiplication");
  assertIncludes(nonlinear.code, "(1.0 / (Float64(N) ^ 2)) * (observable_Jx * observable_Jx)", "PIOperator division becomes scalar multiplication");
  assertIncludes(nonlinear.code, "expectation(rho_ss, adjoint(observable))", "polynomial observable");
  assertIncludes(nonlinear.code, "OneBodyGeometry(basis)", "shared collective geometry");

  function hamiltonianLine(formula) {
    const code = api.generate({
      N: 3, d: 2, target: "steady", hamiltonian: formula,
      jumps: [{ kind: "local", operator: "j_-", rate: "1" }],
      parameters: "",
    }).code;
    return code.split("\n").find((line) => line.startsWith("H_collective ="));
  }
  assert(
    hamiltonianLine("-J_x^2") === "H_collective = (-(Jx * Jx))",
    "exponentiation must bind more tightly than unary minus",
  );
  assert(
    hamiltonianLine("-(J_x^2)") === "H_collective = (-(Jx * Jx))",
    "explicitly grouped negative square",
  );
  assert(
    hamiltonianLine("(-J_x)^2") === "H_collective = ((-Jx) * (-Jx))",
    "explicitly squared negative operator",
  );

  const qudit = api.generate({
    N: 5,
    d: 3,
    target: "steady",
    hamiltonian: String.raw`\Omega J_x`,
    jumps: [{ kind: "local", operator: "j_-", rate: String.raw`\gamma` }],
    parameters: String.raw`\Omega=1
\gamma=0.1`,
  });
  assertIncludes(qudit.code, "spin_matrices(d)", "qudit spin matrices");
  assertIncludes(qudit.code, "jump_1 = spin.jm", "qudit local lowering operator");

  const localPseudomode = api.generate({
    architecture: "local-pseudomode",
    N: 2,
    d: 2,
    target: "expectation",
    hamiltonian: String.raw`\Omega J_z + \chi J_x^2`,
    jumps: [
      { kind: "local", operator: String.raw`\sigma_-`, rate: String.raw`\gamma` },
      { kind: "collective", operator: "J_-", rate: String.raw`\Gamma` },
    ],
    observable: "J_x^2/N^2",
    pseudomode: {
      nmax: 1,
      frequency: String.raw`\omega_c`,
      damping: String.raw`\kappa`,
      thermalOccupation: "nbar",
      couplingOperator: String.raw`\sigma_-`,
      couplingStrength: "g",
      counterrotatingStrength: "g_cr",
    },
    parameters: String.raw`\Omega=0.5
\chi=0.02
\gamma=0.1
\Gamma=0.01
\omega_c=1
\kappa=0.2
nbar=0
g=0.15
g_cr=0`,
  });
  assertIncludes(localPseudomode.code, "BosonicPseudomode(", "local mode specification");
  assertIncludes(localPseudomode.code, "pseudomode_supersite(", "local supersite construction");
  assertIncludes(localPseudomode.code, "embedding = pseudomode_model(", "local embedding builder");
  assertIncludes(localPseudomode.code, "lift_system_operator(site, spin.jx", "nonlinear bare-system lift");
  assertIncludes(localPseudomode.code, "supersite_terms = (DirectPIHamiltonian(H_collective),)", "nonlinear supersite PI term");
  assertIncludes(localPseudomode.code, "system_terms=system_terms", "system channels are lifted by the builder");
  assertIncludes(localPseudomode.code, "backend=:auto", "local embedding keeps guarded automatic compilation");
  assertIncludes(localPseudomode.code, "mode_top_population", "local cutoff diagnostic");
  assertIncludes(localPseudomode.code, "observable_Jx = Jx", "system-only supersite observable reuses prepared polynomial component");
  assert(!localPseudomode.code.includes("collective_spin(basis"), "supersite must not be treated as a spin-(D-1)/2 object");
  assert(!localPseudomode.code.includes("kron("), "generator never emits a full tensor construction");
  assert(localPseudomode.summary.architecture === "local-pseudomode", "local topology summary");
  assert(localPseudomode.summary.cutoff === 1, "local cutoff summary");

  const localPseudomodeTrajectory = api.generate({
    architecture: "local-pseudomode",
    N: 2,
    d: 2,
    target: "steady",
    steadyMethod: "trajectory",
    hamiltonian: String.raw`\Omega J_x`,
    jumps: [],
    pseudomode: {
      nmax: 1,
      frequency: String.raw`\omega_c`,
      damping: String.raw`\kappa`,
      thermalOccupation: "0",
      couplingOperator: String.raw`\sigma_z`,
      couplingStrength: "g",
      counterrotatingStrength: "0",
    },
    parameters: String.raw`\Omega=0.2
\omega_c=1
\kappa=0.3
g=0.1`,
    trajectory: {
      trajectories: 8,
      initialLevel: 2,
      settlingTime: 2,
      dt: 0.01,
      samplesPerTrajectory: 2,
      samplingInterval: 0.5,
      maxJumpProbability: 0.02,
      seed: 9,
    },
  });
  assertIncludes(
    localPseudomodeTrajectory.code,
    "rho0 = pseudomode_product_state(",
    "local-pseudomode trajectory route emits a supersite product state",
  );
  assertIncludes(
    localPseudomodeTrajectory.code,
    "system_initial[INITIAL_LEVEL] = 1",
    "local-pseudomode trajectory route applies the selected system level",
  );
  assertIncludes(
    localPseudomodeTrajectory.code,
    "trajectory_plan = TrajectoryPlan(model)",
    "local-pseudomode trajectory route uses the embedded PI model",
  );
  assert(
    !localPseudomodeTrajectory.code.includes("stationary_state(") &&
      !localPseudomodeTrajectory.code.includes("prepared = compile("),
    "local-pseudomode trajectory route does not prepare a deterministic solve",
  );
  assert(
    localPseudomodeTrajectory.summary.route.includes("streaming path reduction"),
    "local-pseudomode trajectory route summary",
  );

  const globalPseudomode = api.generate({
    architecture: "global-pseudomode",
    N: 3,
    d: 2,
    target: "expectation",
    hamiltonian: String.raw`\Omega J_x + \chi J_z^2`,
    jumps: [{ kind: "local", operator: String.raw`\sigma_-`, rate: String.raw`\gamma` }],
    observable: "J_z/N",
    pseudomode: {
      nmax: 2,
      frequency: String.raw`\omega_c`,
      damping: String.raw`\kappa`,
      thermalOccupation: "0",
      couplingOperator: String.raw`\sigma_-`,
      couplingStrength: "g",
      counterrotatingStrength: "0",
    },
    parameters: String.raw`\Omega=0.5
\chi=0.02
\gamma=0.1
\omega_c=1
\kappa=0.2
g=0.15`,
  });
  assertIncludes(globalPseudomode.code, "system_model = PIModel(system_basis, system_terms)", "shared mode starts from an ordinary PI model");
  assertIncludes(globalPseudomode.code, "global_pseudomode_model(", "shared embedding builder");
  assertIncludes(globalPseudomode.code, "stationary_state(\n    embedding;", "specialized shared-mode stationary route");
  assertIncludes(globalPseudomode.code, "GMRESAlgorithm(krylovdim=40, maxiter=1000)", "bounded shared-mode GMRES settings");
  assertIncludes(globalPseudomode.code, "rho_system = trace_pseudomodes(rho_ss, embedding)", "packed system reduction");
  assertIncludes(globalPseudomode.code, "rho_mode = global_pseudomode_state", "packed mode reduction");
  assertIncludes(globalPseudomode.code, "LinearAlgebra.ishermitian(", "composite Hermiticity check");
  assertIncludes(
    globalPseudomode.code,
    "does not certify positivity of the complete composite state",
    "shared-mode code states the full-positivity limitation",
  );
  assert(
    globalPseudomode.warnings.some((warning) =>
      warning.includes("does not certify positivity")),
    "shared-mode warning states the full-positivity limitation",
  );
  assertIncludes(globalPseudomode.code, "mode_top_population", "global cutoff diagnostic");
  assert(!globalPseudomode.code.includes("compile(\n    embedding"), "shared model must not be compiled as an ordinary PI model");
  assert(!globalPseudomode.code.includes("kron("), "shared model stays factorized");
  assert(globalPseudomode.summary.route.includes("matrix-free GMRES"), "global route summary");

  const deterministicDynamics = api.generate({
    architecture: "pi",
    N: 2,
    d: 2,
    calculation: "dynamics-observable",
    steadyMethod: "deterministic",
    initialState: { level: 2 },
    hamiltonian: String.raw`\Omega J_x`,
    jumps: [{ kind: "local", operator: String.raw`\sigma_-`, rate: String.raw`\gamma` }],
    observable: "J_z/N",
    parameters: String.raw`\Omega=0.2
\gamma=0.1`,
    dynamics: { startTime: 0, finalTime: 0.1, samples: 3, stepsPerInterval: 2 },
    resources: { memoryBudgetMiB: 96 },
  });
  assertIncludes(deterministicDynamics.code, "const INITIAL_LEVEL = 2", "deterministic dynamics initial state");
  assertIncludes(deterministicDynamics.code, "const MEMORY_BUDGET = 96 * 1024^2", "custom memory budget");
  assertIncludes(deterministicDynamics.code, "backend=:matrixfree", "deterministic dynamics matrix-free compilation");
  assertIncludes(deterministicDynamics.code, "dynamics = solve_dynamics(", "deterministic observable dynamics");
  assertIncludes(deterministicDynamics.code, "save_states=false", "state-free deterministic output");
  assertIncludes(deterministicDynamics.code, "dynamics.observables[:observable]", "streamed deterministic observable");
  assert(deterministicDynamics.summary.calculation === "dynamics-observable", "deterministic dynamics summary");

  const localPseudomodeDynamics = api.generate({
    architecture: "local_pseudomode",
    N: 2,
    d: 2,
    calculation: "dynamics",
    initialState: { level: 1 },
    hamiltonian: String.raw`\Omega J_x`,
    jumps: [],
    observable: "J_z",
    pseudomode: {
      nmax: 1,
      frequency: String.raw`\omega_c`,
      damping: String.raw`\kappa`,
      thermalOccupation: "0",
      couplingOperator: String.raw`\sigma_z`,
      couplingStrength: "g",
      counterrotatingStrength: "0",
    },
    parameters: String.raw`\Omega=0.2
\omega_c=1
\kappa=0.3
g=0.1`,
    dynamics: { startTime: 0, finalTime: 0.1, samples: 3, stepsPerInterval: 2 },
  });
  assertIncludes(localPseudomodeDynamics.code, "rho0 = pseudomode_product_state(", "local-mode deterministic initial state");
  assertIncludes(localPseudomodeDynamics.code, "lift_system_operator(site, spin.jz", "local-mode physical observable lift");
  assertIncludes(localPseudomodeDynamics.code, "dynamics = solve_dynamics(", "local-mode deterministic dynamics");
  assertIncludes(localPseudomodeDynamics.code, "save_states=false", "local-mode state-free output");
  assert(localPseudomodeDynamics.summary.route.includes("PI-supersite dynamics"), "local-mode deterministic route summary");

  const trajectoryDynamics = api.generate({
    architecture: "pi",
    N: 2,
    d: 2,
    calculation: "transient",
    steadyMethod: "quantum_trajectories",
    initialState: { level: 2 },
    hamiltonian: String.raw`\Omega J_x`,
    jumps: [{ kind: "local", operator: String.raw`\sigma_-`, rate: String.raw`\gamma` }],
    observable: "J_z",
    parameters: String.raw`\Omega=0.2
\gamma=0.1`,
    trajectory: { trajectories: 3, dt: 0.01, maxJumpProbability: 0.02, seed: 41 },
    dynamics: { startTime: 0, finalTime: 0.1, samples: 3, stepsPerInterval: 2 },
  });
  assertIncludes(trajectoryDynamics.code, "TrajectoryPlan(model)", "trajectory dynamics channel plan");
  assertIncludes(trajectoryDynamics.code, "dynamics = quantum_trajectories(", "ordinary trajectory dynamics");
  assertIncludes(trajectoryDynamics.code, "dynamics_statistics.standard_error", "trajectory uncertainty output");
  assertIncludes(trajectoryDynamics.code, "save_states=false, jump_statistics=false", "trajectory histories remain disabled");
  assert(
    trajectoryDynamics.summary.method === "quantum-trajectory observable dynamics",
    "trajectory alias normalization",
  );

  const localPseudomodeTrajectoryDynamics = api.generate({
    architecture: "local-pseudomode",
    N: 2,
    d: 2,
    calculation: "dynamics_observable",
    steadyMethod: "trajectory",
    initialState: { level: 1 },
    hamiltonian: String.raw`\Omega J_x`,
    jumps: [],
    observable: "J_z",
    pseudomode: {
      nmax: 1,
      frequency: "1",
      damping: "0.3",
      thermalOccupation: "0",
      couplingOperator: String.raw`\sigma_z`,
      couplingStrength: "0.1",
      counterrotatingStrength: "0",
    },
    parameters: String.raw`\Omega=0.2`,
    trajectory: { trajectories: 3, dt: 0.01, maxJumpProbability: 0.02, seed: 43 },
    dynamics: { startTime: 0, finalTime: 0.1, samples: 3, stepsPerInterval: 2 },
  });
  assertIncludes(localPseudomodeTrajectoryDynamics.code, "pseudomode_product_state(", "local-mode trajectory initial state");
  assertIncludes(localPseudomodeTrajectoryDynamics.code, "TrajectoryBatchWorkspace(", "local-mode trajectory batch workspace");
  assertIncludes(localPseudomodeTrajectoryDynamics.code, "dynamics = quantum_trajectories(", "local-mode trajectory dynamics");
  assert(localPseudomodeTrajectoryDynamics.summary.route.includes("PI-supersite trajectories"), "local-mode trajectory route summary");

  const globalPseudomodeTrajectoryDynamics = api.generate({
    architecture: "global_pseudomode",
    N: 2,
    d: 2,
    calculation: "transient-observable",
    steadyMethod: "trajectory",
    initialState: { level: 1 },
    hamiltonian: String.raw`\Omega J_x`,
    jumps: [{ kind: "local", operator: String.raw`\sigma_-`, rate: "0.1" }],
    observable: "J_z",
    pseudomode: {
      nmax: 1,
      frequency: "1",
      damping: "0.3",
      thermalOccupation: "0",
      couplingOperator: String.raw`\sigma_-`,
      couplingStrength: "0.1",
      counterrotatingStrength: "0",
    },
    parameters: String.raw`\Omega=0.2`,
    trajectory: { trajectories: 3, dt: 0.01, maxJumpProbability: 0.02, seed: 47 },
    dynamics: { startTime: 0, finalTime: 0.1, samples: 3, stepsPerInterval: 2 },
  });
  assertIncludes(globalPseudomodeTrajectoryDynamics.code, "CompositeTrajectoryPlan(", "shared-mode composite trajectory plan");
  assertIncludes(globalPseudomodeTrajectoryDynamics.code, "CompositeTrajectoryBatchWorkspace(", "shared-mode trajectory workspace");
  assertIncludes(globalPseudomodeTrajectoryDynamics.code, "embedding.background, embedding.damping_channels...", "shared-mode monitored-channel split");
  assertIncludes(
    globalPseudomodeTrajectoryDynamics.code,
    "streaming_observable = observable",
    "shared-mode trajectories use the Hermitian observable without requiring a composite adjoint",
  );
  assertIncludes(globalPseudomodeTrajectoryDynamics.code, "dynamics = quantum_trajectories(", "shared-mode trajectory dynamics");
  assert(globalPseudomodeTrajectoryDynamics.summary.route.includes("factorized composite trajectories"), "shared-mode trajectory route summary");

  const selectedSpectrum = api.generate({
    architecture: "pi",
    N: 2,
    d: 2,
    calculation: "spectrum",
    hamiltonian: String.raw`\Omega J_x`,
    jumps: [{ kind: "local", operator: String.raw`\sigma_-`, rate: "0.1" }],
    parameters: String.raw`\Omega=0.2`,
    spectrum: { target: "largest-real", nev: 3, seed: 53 },
  });
  assertIncludes(selectedSpectrum.code, "spectrum = liouvillian_spectrum(", "ordinary selected spectrum");
  assertIncludes(selectedSpectrum.code, "target=:largest_real, nev=SPECTRUM_NEV", "spectrum target and count");
  assertIncludes(selectedSpectrum.code, "Random.MersenneTwister(SPECTRUM_SEED)", "reproducible spectral seed");
  assertIncludes(selectedSpectrum.code, "return_info=true", "spectrum metadata");
  assert(selectedSpectrum.summary.calculation === "liouvillian-spectrum", "spectrum alias normalization");

  const twoAxisScan = api.generate({
    architecture: "pi",
    N: 1,
    d: 2,
    calculation: "steady-observable",
    steadyMethod: "deterministic",
    hamiltonian: "",
    jumps: [
      {
        kind: "local",
        operator: String.raw`\sigma_-`,
        rate: String.raw`\gamma_{\downarrow}`,
      },
      {
        kind: "local",
        operator: String.raw`\sigma_+`,
        rate: String.raw`\gamma_{\uparrow}`,
      },
    ],
    observable: "J_z",
    parameters: "",
    scan: {
      enabled: true,
      axes: [
        {
          parameter: String.raw`\gamma_{\downarrow}`,
          start: 0.2,
          stop: 0.4,
          points: 2,
        },
        {
          parameter: String.raw`\gamma_{\uparrow}`,
          start: 0.1,
          stop: 0.3,
          points: 3,
        },
      ],
    },
  });
  assertIncludes(twoAxisScan.code, "ParameterScanPlan(", "typed parameter-scan plan");
  assertIncludes(twoAxisScan.code, "ParameterScanWorkspace()", "task-owned scan workspace");
  assertIncludes(twoAxisScan.code, "parameter_scan(", "native parameter-scan execution");
  assertIncludes(twoAxisScan.code, "scan_result", "named scan result");
  assertIncludes(twoAxisScan.code, "scan_rows", "tabular scan rows");
  assert(
    twoAxisScan.summary.scanPoints === 6,
    "Cartesian scan point count is reported",
  );
  assert(
    Array.isArray(twoAxisScan.summary.scanAxes) &&
      twoAxisScan.summary.scanAxes.join(",") === "gamma_down,gamma_up",
    "normalized scan axes are reported in order",
  );
  assert(
    twoAxisScan.manifest.calculation.scan.enabled === true,
    "scan manifest records enabled state",
  );
  assert(
    twoAxisScan.manifest.calculation.scan.ordering === "first-axis-fastest",
    "scan manifest records Cartesian ordering",
  );
  assert(
    twoAxisScan.manifest.calculation.scan.axes.length === 2,
    "scan manifest records both axes",
  );
  assert(
    twoAxisScan.manifest.calculation.scan.axes[0].parameter === "gamma_down" &&
      twoAxisScan.manifest.calculation.scan.axes[1].parameter === "gamma_up",
    "scan manifest preserves normalized axis order",
  );
  assert(
    twoAxisScan.manifest.calculation.scan.axes[1].points === 3,
    "scan manifest records axis resolution",
  );
  assert(
    !twoAxisScan.code.includes("TODO"),
    "scanned parameters use their first grid values as nominal values",
  );
  assert(
    twoAxisScan.bundle.stem.endsWith("_scan"),
    "scan bundle has an unambiguous filename",
  );

  const localPseudomodeScan = api.generate({
    architecture: "local-pseudomode",
    N: 1,
    d: 2,
    calculation: "steady-state",
    steadyMethod: "deterministic",
    hamiltonian: String.raw`\Omega J_z`,
    jumps: [],
    pseudomode: {
      nmax: 1,
      frequency: String.raw`\omega_c`,
      damping: String.raw`\kappa`,
      thermalOccupation: "0",
      couplingOperator: String.raw`\sigma_-`,
      couplingStrength: "g",
      counterrotatingStrength: "0",
    },
    parameters: String.raw`\Omega=0.2
\omega_c=1
\kappa=0.3
g=0.1`,
    scan: {
      enabled: true,
      axes: [
        { parameter: String.raw`\kappa`, start: 0.5, stop: 0.1, points: 3 },
      ],
    },
  });
  assertIncludes(
    localPseudomodeScan.code,
    "pseudomode_supersite(",
    "local-pseudomode scan prepares supersite geometry",
  );
  assertIncludes(
    localPseudomodeScan.code,
    "pseudomode_model(",
    "local-pseudomode scan rebuilds only model coefficients",
  );
  assertIncludes(
    localPseudomodeScan.code,
    "ParameterScanPlan(",
    "local-pseudomode scan uses native scan infrastructure",
  );
  assert(
    localPseudomodeScan.summary.scanPoints === 3,
    "descending local-pseudomode scan range is accepted",
  );
  assert(
    localPseudomodeScan.manifest.model.parameters.kappa === 0.5,
    "the first scan point overrides the nominal browser-validation value",
  );
  assert(
    localPseudomodeScan.warnings.some(
      (warning) => warning.includes("kappa") && warning.includes("overridden"),
    ),
    "overridden nominal parameter assignments are reported",
  );
  const restoredLocalScanConfiguration =
    api.configurationFromManifest(localPseudomodeScan.manifest);
  assert(
    restoredLocalScanConfiguration.architecture === "local-pseudomode" &&
      restoredLocalScanConfiguration.pseudomode.nmax === 1,
    "manifest round trip restores the local-pseudomode topology",
  );
  assert(
    restoredLocalScanConfiguration.scan.enabled === true &&
      restoredLocalScanConfiguration.scan.axes.length === 1 &&
      restoredLocalScanConfiguration.scan.axes[0].parameter === "kappa",
    "manifest round trip restores a normalized scan axis",
  );
  const restoredLocalScan = api.generate(restoredLocalScanConfiguration);
  assert(
    restoredLocalScan.summary.scanPoints === 3,
    "restored local-pseudomode scan preserves its Cartesian point count",
  );
  assert(
    restoredLocalScan.manifest.model.pseudomode.couplingOperator.trim() ===
      localPseudomodeScan.manifest.model.pseudomode.couplingOperator.trim(),
    "restored local-pseudomode scan preserves its coupling operator",
  );

  const spectralScan = api.generate({
    architecture: "pi",
    N: 1,
    d: 2,
    calculation: "liouvillian-spectrum",
    hamiltonian: String.raw`\Omega J_z`,
    jumps: [{ kind: "local", operator: String.raw`\sigma_-`, rate: "0.2" }],
    parameters: String.raw`\Omega=0.1`,
    spectrum: { target: "largest-real", nev: 2, seed: 61 },
    scan: {
      enabled: true,
      axes: [
        { parameter: String.raw`\Omega`, start: 0.1, stop: 0.3, points: 3 },
      ],
    },
  });
  assertIncludes(spectralScan.code, "task=:spectrum", "spectral scan task");
  assertIncludes(
    spectralScan.code,
    "spectrum_target=:largest_real",
    "spectral scan target",
  );
  assertIncludes(spectralScan.code, "save_vectors=false", "spectral scan omits Ritz vectors");

  const disabledScan = api.generate({
    architecture: "pi",
    N: 1,
    d: 2,
    calculation: "steady-state",
    hamiltonian: "0.1 J_z",
    jumps: [{ kind: "local", operator: String.raw`\sigma_-`, rate: "0.2" }],
    scan: { enabled: false, axes: [] },
  });
  assert(
    !disabledScan.code.includes("ParameterScanPlan("),
    "a disabled scan preserves the single-model workflow",
  );
  assert(
    disabledScan.manifest.calculation.scan === null,
    "a disabled scan is explicit in the normalized manifest",
  );

  const globalSelectedSpectrum = api.generate({
    architecture: "global-pseudomode",
    N: 2,
    d: 2,
    calculation: "liouvillian_spectrum",
    hamiltonian: String.raw`\Omega J_x`,
    jumps: [{ kind: "local", operator: String.raw`\sigma_-`, rate: "0.1" }],
    pseudomode: {
      nmax: 1,
      frequency: "1",
      damping: "0.3",
      thermalOccupation: "0",
      couplingOperator: String.raw`\sigma_-`,
      couplingStrength: "0.1",
      counterrotatingStrength: "0",
    },
    parameters: String.raw`\Omega=0.2`,
    spectrum: { target: "near_zero", nev: 3, seed: 59 },
  });
  assertIncludes(globalSelectedSpectrum.code, "spectrum = liouvillian_spectrum(\n    embedding;", "shared-mode selected spectrum");
  assertIncludes(globalSelectedSpectrum.code, "target=:near_zero", "shared-mode near-zero target");
  assert(globalSelectedSpectrum.summary.route.includes("factorized automatic matrix-free selected spectrum"), "shared-mode spectrum route summary");

  const ordinaryGap = api.generate({
    architecture: "pi",
    N: 2,
    d: 2,
    calculation: "gap",
    hamiltonian: String.raw`\Omega J_x`,
    jumps: [{ kind: "local", operator: String.raw`\sigma_-`, rate: "0.1" }],
    parameters: String.raw`\Omega=0.2`,
    gap: { nev: 3, krylovdim: 8 },
  });
  assertIncludes(ordinaryGap.code, "gap_source = model", "ordinary gap source");
  assertIncludes(ordinaryGap.code, "gap_result = pi_liouvillian_gap(", "ordinary gap solver");
  assertIncludes(ordinaryGap.code, "nev=GAP_NEV, krylovdim=GAP_KRYLOVDIM", "gap Krylov controls");
  assertIncludes(ordinaryGap.code, "gap_result.gap_certified", "gap certification output");
  assert(ordinaryGap.summary.calculation === "liouvillian-gap", "gap alias normalization");

  const globalGap = api.generate({
    architecture: "global-pseudomode",
    N: 2,
    d: 2,
    calculation: "liouvillian-gap",
    hamiltonian: String.raw`\Omega J_x`,
    jumps: [{ kind: "local", operator: String.raw`\sigma_-`, rate: "0.1" }],
    pseudomode: {
      nmax: 1,
      frequency: "1",
      damping: "0.3",
      thermalOccupation: "0",
      couplingOperator: String.raw`\sigma_-`,
      couplingStrength: "0.1",
      counterrotatingStrength: "0",
    },
    parameters: String.raw`\Omega=0.2`,
    gap: { nev: 3, krylovdim: 8 },
  });
  assertIncludes(globalGap.code, "gap_source = global_pseudomode_matrixfree(", "shared-mode factorized gap source");
  assertIncludes(globalGap.code, "gap_result = pi_liouvillian_gap(", "shared-mode gap solver");
  assert(globalGap.summary.route.includes("factorized matrix-free largest-real Krylov gap"), "shared-mode gap route summary");

  const stationaryAnalysis = api.generate({
    architecture: "pi",
    N: 2,
    d: 2,
    calculation: "steady_state",
    hamiltonian: String.raw`\Omega J_x`,
    jumps: [{ kind: "local", operator: String.raw`\sigma_-`, rate: "0.1" }],
    parameters: String.raw`\Omega=0.2`,
    analysis: { purity: true, entropy: true, oneBodyRDM: true, qfiAxis: "z" },
  });
  assertIncludes(stationaryAnalysis.code, "analysis_state = rho_ss", "ordinary analysis state");
  assertIncludes(stationaryAnalysis.code, "system_purity = purity(analysis_state)", "stationary purity");
  assertIncludes(stationaryAnalysis.code, "von_neumann_entropy(analysis_state; base=2)", "stationary entropy");
  assertIncludes(stationaryAnalysis.code, "one_body_density_matrix = one_body_rdm(", "stationary one-body reduction");
  assertIncludes(stationaryAnalysis.code, "qfi_value = qfi(", "stationary QFI");
  assertIncludes(
    stationaryAnalysis.code,
    "analysis_state, qfi_plan; atol=STATE_VALIDATION_TOL)",
    "stationary QFI validation tolerance",
  );

  const localStationaryAnalysis = api.generate({
    architecture: "local-pseudomode",
    N: 2,
    d: 2,
    calculation: "steady-state",
    hamiltonian: String.raw`\Omega J_x`,
    jumps: [],
    pseudomode: {
      nmax: 1,
      frequency: "1",
      damping: "0.3",
      thermalOccupation: "0",
      couplingOperator: String.raw`\sigma_z`,
      couplingStrength: "0.1",
      counterrotatingStrength: "0",
    },
    parameters: String.raw`\Omega=0.2`,
    analysis: { purity: true, entropy: true, oneBodyRDM: true, qfiAxis: "x" },
  });
  assertIncludes(localStationaryAnalysis.code, "analysis_trace_plan = pseudomode_trace_plan(", "local-mode analysis reduction plan");
  assertIncludes(localStationaryAnalysis.code, "analysis_state = trace_pseudomodes(", "local-mode physical-system analysis state");
  assertIncludes(localStationaryAnalysis.code, "analysis_basis = analysis_trace_plan.output_basis", "local-mode reduced basis");

  const globalStationaryAnalysis = api.generate({
    architecture: "global-pseudomode",
    N: 2,
    d: 2,
    calculation: "steady_observable",
    hamiltonian: String.raw`\Omega J_x`,
    jumps: [{ kind: "local", operator: String.raw`\sigma_-`, rate: "0.1" }],
    observable: "J_z",
    pseudomode: {
      nmax: 1,
      frequency: "1",
      damping: "0.3",
      thermalOccupation: "0",
      couplingOperator: String.raw`\sigma_-`,
      couplingStrength: "0.1",
      counterrotatingStrength: "0",
    },
    parameters: String.raw`\Omega=0.2`,
    analysis: { purity: true, entropy: true, oneBodyRDM: true, qfiAxis: "y" },
  });
  assertIncludes(globalStationaryAnalysis.code, "analysis_state = rho_system", "shared-mode reduced analysis state");
  assertIncludes(globalStationaryAnalysis.code, "analysis_basis = system_basis", "shared-mode physical-system basis");
  assertIncludes(globalStationaryAnalysis.code, "spin.jy; cache=analysis_geometry", "shared-mode selected QFI axis");

  assert(driven.summary.calculation === "steady-observable", "legacy expectation target alias");
  assert(trajectorySteady.summary.calculation === "steady-state", "legacy steady target alias");
  const legacyCalculationAliases = [
    ["transient", "dynamics-observable"],
    ["liouvillian_spectrum", "liouvillian-spectrum"],
    ["liouvillian_gap", "liouvillian-gap"],
  ];
  for (const [alias, canonical] of legacyCalculationAliases) {
    const aliasConfig = {
      N: 1,
      d: 2,
      calculation: alias,
      hamiltonian: "0.1 J_x",
      jumps: [{ kind: "local", operator: "j_-", rate: "0.2" }],
    };
    if (canonical === "dynamics-observable") {
      aliasConfig.observable = "J_z";
      aliasConfig.initialState = { level: 1 };
      aliasConfig.dynamics = { startTime: 0, finalTime: 0.1, samples: 2, stepsPerInterval: 1 };
    } else if (canonical === "liouvillian-spectrum") {
      aliasConfig.spectrum = { target: "largest_magnitude", nev: 1, seed: 1 };
    } else {
      aliasConfig.gap = { nev: 2, krylovdim: 3 };
    }
    assert(api.generate(aliasConfig).summary.calculation === canonical, `${alias} calculation alias`);
  }

  const pseudomodeCollision = api.generate({
    architecture: "global-pseudomode",
    N: 2,
    d: 2,
    target: "steady",
    hamiltonian: "",
    jumps: [],
    pseudomode: {
      nmax: 1,
      frequency: "mode",
      damping: "0.2",
      thermalOccupation: "0",
      couplingOperator: "j_-",
      couplingStrength: "coupling",
      counterrotatingStrength: "0",
    },
    parameters: "mode=1\ncoupling=0.1",
  });
  assertIncludes(pseudomodeCollision.code, "parameter_mode = 1.0", "mode name collision is renamed");
  assertIncludes(pseudomodeCollision.code, "parameter_coupling = 0.1", "coupling name collision is renamed");
  assertIncludes(pseudomodeCollision.code, "frequency=parameter_mode", "renamed mode parameter is used");
  assertIncludes(pseudomodeCollision.code, "strength=parameter_coupling", "renamed coupling parameter is used");

  const missing = api.generate({
    N: 3,
    d: 2,
    target: "steady",
    hamiltonian: String.raw`\Omega J_z`,
    jumps: [{ kind: "local", operator: String.raw`\sigma_-`, rate: String.raw`\gamma` }],
    parameters: String.raw`\Omega=0.2`,
  });
  assertIncludes(missing.code, "gamma = 1.0  # TODO", "missing parameter placeholder");
  assert(missing.warnings.some((warning) => warning.includes("gamma")), "missing parameter warning");

  const collision = api.generate({
    N: 3,
    d: 2,
    target: "steady",
    hamiltonian: "0.1 J_z",
    jumps: [{ kind: "local", operator: "j_-", rate: "jump_1" }],
    parameters: "jump_1=0.2",
  });
  assertIncludes(collision.code, "parameter_jump_1 = 0.2", "temporary-name collision is renamed");
  assertIncludes(collision.code, "rate=parameter_jump_1", "renamed rate is used");

  const inheritedName = api.generate({
    N: 3,
    d: 2,
    target: "steady",
    hamiltonian: "constructor J_z",
    jumps: [{ kind: "local", operator: "j_-", rate: "0.2" }],
    parameters: "constructor=0.1",
  });
  assertIncludes(inheritedName.code, "constructor = 0.1", "plain-object inherited name remains a scalar parameter");

  const wideIntegerPower = api.generate({
    N: 3,
    d: 2,
    target: "steady",
    hamiltonian: "(2^100) J_z",
    jumps: [{ kind: "local", operator: "j_-", rate: "0.2" }],
    parameters: "",
  });
  assertIncludes(wideIntegerPower.code, "(2.0 ^ 100)", "integer coefficient power uses Float64 arithmetic");

  const collectiveOnly = api.generate({
    N: 3,
    d: 2,
    target: "steady",
    hamiltonian: "0.1 J_z",
    jumps: [{ kind: "collective", operator: "J_-", rate: "0.2" }],
    parameters: "",
  });
  assert(
    collectiveOnly.warnings.some((warning) => warning.includes("Schur-sector")),
    "collective-only model warns about sector conservation",
  );

  const doubledRate = api.generate({
    N: 3,
    d: 2,
    target: "steady",
    hamiltonian: "0.1 J_z",
    jumps: [{ kind: "local", operator: String.raw`\sqrt{\gamma}j_-`, rate: String.raw`\gamma` }],
    parameters: String.raw`\gamma=0.2`,
  });
  assert(
    doubledRate.warnings.some((warning) => warning.includes("square")),
    "internal jump coefficient plus external rate warns about double counting",
  );

  assertRejectsScan(
    () => api.generate({
      N: 1,
      d: 2,
      calculation: "steady-state",
      hamiltonian: String.raw`\Omega J_z`,
      jumps: [{ kind: "local", operator: String.raw`\sigma_-`, rate: "0.2" }],
      parameters: String.raw`\Omega=0.1`,
      scan: { enabled: true, axes: [] },
    }),
    "an enabled scan requires at least one axis",
  );
  assertRejectsScan(
    () => api.generate({
      N: 1,
      d: 2,
      calculation: "steady-state",
      hamiltonian: String.raw`\Omega J_z`,
      jumps: [{ kind: "local", operator: String.raw`\sigma_-`, rate: "0.2" }],
      parameters: String.raw`\Omega=0.1`,
      scan: {
        enabled: true,
        axes: [{ parameter: String.raw`\gamma`, start: 0.1, stop: 0.2, points: 2 }],
      },
    }),
    "a scan axis must select a model parameter",
  );
  assertRejectsScan(
    () => api.generate({
      N: 1,
      d: 2,
      calculation: "steady-observable",
      hamiltonian: String.raw`\Omega J_z`,
      jumps: [{ kind: "local", operator: String.raw`\sigma_-`, rate: "0.2" }],
      observable: "a J_z",
      scan: {
        enabled: true,
        axes: [{ parameter: "a", start: 0.1, stop: 0.2, points: 2 }],
      },
    }),
    "observable-only scalars are not mistaken for model parameters",
  );
  assertRejectsScan(
    () => api.generate({
      N: 1,
      d: 2,
      calculation: "steady-state",
      hamiltonian: "",
      jumps: [{
        kind: "local",
        operator: String.raw`\sigma_-`,
        rate: String.raw`\gamma_{\downarrow}`,
      }],
      scan: {
        enabled: true,
        axes: [
          {
            parameter: String.raw`\gamma_{\downarrow}`,
            start: 0.1,
            stop: 0.2,
            points: 2,
          },
          {
            parameter: "gamma_down",
            start: 0.2,
            stop: 0.3,
            points: 2,
          },
        ],
      },
    }),
    "duplicate normalized scan axes are rejected",
  );
  assertRejectsScan(
    () => api.generate({
      N: 1,
      d: 2,
      calculation: "steady-state",
      hamiltonian: String.raw`\Omega J_z`,
      jumps: [{ kind: "local", operator: String.raw`\sigma_-`, rate: "0.2" }],
      scan: {
        enabled: true,
        axes: [{ parameter: String.raw`\Omega`, start: 0.1, stop: 0.2, points: 1 }],
      },
    }),
    "each scan axis requires at least two points",
  );
  assertRejectsScan(
    () => api.generate({
      N: 1,
      d: 2,
      calculation: "steady-state",
      hamiltonian: String.raw`\Omega J_z`,
      jumps: [{ kind: "local", operator: String.raw`\sigma_-`, rate: "0.2" }],
      scan: {
        enabled: true,
        axes: [{ parameter: String.raw`\Omega`, start: "", stop: 0.2, points: 2 }],
      },
    }),
    "blank scan endpoints are not coerced to zero",
  );
  assertRejectsScan(
    () => api.generate({
      N: 1,
      d: 2,
      calculation: "steady-state",
      hamiltonian: String.raw`\Omega J_z`,
      jumps: [{ kind: "local", operator: String.raw`\sigma_-`, rate: "0.2" }],
      scan: {
        enabled: true,
        axes: [{
          parameter: String.raw`\Omega`,
          start: Number.POSITIVE_INFINITY,
          stop: 0.2,
          points: 2,
        }],
      },
    }),
    "scan endpoints must be finite",
  );
  assertRejectsScan(
    () => api.generate({
      N: 1,
      d: 2,
      calculation: "steady-state",
      hamiltonian: String.raw`\Omega J_z`,
      jumps: [{ kind: "local", operator: String.raw`\sigma_-`, rate: "0.2" }],
      scan: {
        enabled: true,
        axes: [{ parameter: String.raw`\Omega`, start: 0.1, stop: 0.1, points: 2 }],
      },
    }),
    "a scan axis must span a nonzero interval",
  );
  assertRejectsScan(
    () => api.generate({
      N: 1,
      d: 2,
      calculation: "steady-state",
      hamiltonian: String.raw`\Omega J_z`,
      jumps: [{
        kind: "local",
        operator: String.raw`\sigma_-`,
        rate: String.raw`\gamma`,
      }],
      scan: {
        enabled: true,
        axes: [
          { parameter: String.raw`\Omega`, start: 0.1, stop: 0.2, points: 1001 },
          { parameter: String.raw`\gamma`, start: 0.1, stop: 0.2, points: 101 },
        ],
      },
    }),
    "the browser rejects an unbounded Cartesian grid",
  );
  assertRejectsScan(
    () => api.generate({
      N: 1,
      d: 2,
      calculation: "steady-state",
      steadyMethod: "trajectory",
      hamiltonian: String.raw`\Omega J_z`,
      jumps: [{ kind: "local", operator: String.raw`\sigma_-`, rate: "0.2" }],
      scan: {
        enabled: true,
        axes: [{ parameter: String.raw`\Omega`, start: 0.1, stop: 0.2, points: 2 }],
      },
    }),
    "trajectory stationary estimation is not silently scanned",
  );
  assertRejectsScan(
    () => api.generate({
      N: 1,
      d: 2,
      calculation: "dynamics-observable",
      hamiltonian: String.raw`\Omega J_z`,
      jumps: [{ kind: "local", operator: String.raw`\sigma_-`, rate: "0.2" }],
      observable: "J_z",
      initialState: { level: 1 },
      dynamics: { startTime: 0, finalTime: 1, samples: 2, stepsPerInterval: 1 },
      scan: {
        enabled: true,
        axes: [{ parameter: String.raw`\Omega`, start: 0.1, stop: 0.2, points: 2 }],
      },
    }),
    "dynamics scans are rejected until they have a typed native route",
  );
  assertRejectsScan(
    () => api.generate({
      N: 1,
      d: 2,
      calculation: "liouvillian-gap",
      hamiltonian: String.raw`\Omega J_z`,
      jumps: [{ kind: "local", operator: String.raw`\sigma_-`, rate: "0.2" }],
      gap: { nev: 2, krylovdim: 4 },
      scan: {
        enabled: true,
        axes: [{ parameter: String.raw`\Omega`, start: 0.1, stop: 0.2, points: 2 }],
      },
    }),
    "gap scans are rejected rather than losing certification semantics",
  );
  assertRejectsScan(
    () => api.generate({
      architecture: "global-pseudomode",
      N: 1,
      d: 2,
      calculation: "steady-state",
      hamiltonian: String.raw`\Omega J_z`,
      jumps: [{ kind: "local", operator: String.raw`\sigma_-`, rate: "0.2" }],
      pseudomode: {
        nmax: 1,
        frequency: "1",
        damping: "0.2",
        thermalOccupation: "0",
        couplingOperator: String.raw`\sigma_-`,
        couplingStrength: "0.1",
      },
      scan: {
        enabled: true,
        axes: [{ parameter: String.raw`\Omega`, start: 0.1, stop: 0.2, points: 2 }],
      },
    }),
    "shared-pseudomode scans are rejected explicitly",
  );
  assertRejectsScan(
    () => api.generate({
      workflow: "verified-experiment",
      architecture: "pi",
      N: 1,
      d: 2,
      calculation: "steady-state",
      hamiltonian: String.raw`\Omega J_z`,
      jumps: [{ kind: "local", operator: String.raw`\sigma_-`, rate: "0.2" }],
      scan: {
        enabled: true,
        axes: [{ parameter: String.raw`\Omega`, start: 0.1, stop: 0.2, points: 2 }],
      },
    }),
    "verified-experiment scans are rejected until archives describe the grid",
  );
  assertRejects(
    () => api.generate({
      architecture: "pi",
      N: 1,
      d: 2,
      calculation: "steady-state",
      hamiltonian: String.raw`\Omega J_z`,
      jumps: [{ kind: "local", operator: String.raw`\sigma_-`, rate: "0.2" }],
      analysis: { purity: true },
      scan: {
        enabled: true,
        axes: [{ parameter: String.raw`\Omega`, start: 0.1, stop: 0.2, points: 2 }],
      },
    }),
    "state analysis",
    "post-stationary state analyses require an explicit scan diagnostic",
  );
  assertRejects(
    () => api.generate({
      architecture: "pi",
      N: 1,
      d: 2,
      calculation: "steady-state",
      hamiltonian: "",
      jumps: [{
        kind: "local",
        operator: String.raw`\sigma_-`,
        rate: String.raw`\gamma`,
      }],
      scan: {
        enabled: true,
        axes: [
          { parameter: String.raw`\gamma`, start: 0.1, stop: -0.1, points: 3 },
        ],
      },
    }),
    "jump 1 rate",
    "jump rates are validated at every Cartesian scan point",
  );
  assertRejects(
    () => api.generate({
      architecture: "local-pseudomode",
      N: 1,
      d: 2,
      calculation: "steady-state",
      hamiltonian: "",
      jumps: [],
      pseudomode: {
        nmax: 1,
        frequency: "1",
        damping: String.raw`\kappa`,
        thermalOccupation: "0",
        couplingOperator: String.raw`\sigma_-`,
        couplingStrength: "0.1",
      },
      scan: {
        enabled: true,
        axes: [
          { parameter: String.raw`\kappa`, start: 0.1, stop: -0.1, points: 3 },
        ],
      },
    }),
    "pseudomode damping",
    "pseudomode damping is validated at every Cartesian scan point",
  );

  assertRejects(
    () => api.generate({
      N: 2,
      d: 2,
      calculation: "propagator",
      hamiltonian: "J_z",
      jumps: [],
    }),
    "calculation",
    "unknown calculation",
  );
  assertRejects(
    () => api.generate({
      N: 2,
      d: 2,
      calculation: "spectrum",
      steadyMethod: "trajectory",
      hamiltonian: "J_z",
      jumps: [{ kind: "local", operator: "j_-", rate: "0.1" }],
    }),
    "calculation method",
    "spectral calculations reject trajectory sampling",
  );
  assertRejects(
    () => api.generate({
      architecture: "global-pseudomode",
      N: 2,
      d: 2,
      calculation: "dynamics",
      steadyMethod: "deterministic",
      initialState: { level: 1 },
      hamiltonian: "J_z",
      jumps: [],
      observable: "J_z",
      pseudomode: {
        nmax: 1,
        frequency: "1",
        damping: "0.1",
        thermalOccupation: "0",
        couplingOperator: "j_-",
        couplingStrength: "0.1",
      },
    }),
    "calculation method",
    "shared-mode deterministic streaming is rejected explicitly",
  );
  assertRejects(
    () => api.generate({
      N: 2,
      d: 2,
      calculation: "dynamics",
      initialState: { level: 1 },
      hamiltonian: "J_z",
      jumps: [{ kind: "local", operator: "j_-", rate: "0.1" }],
      observable: "J_z",
      dynamics: { startTime: 0, finalTime: 1, samples: 3, stepsPerInterval: 2 },
      analysis: { purity: true },
    }),
    "state analysis",
    "state analyses reject nonstationary calculations",
  );
  assertRejects(
    () => api.generate({
      N: 2,
      d: 2,
      calculation: "dynamics",
      steadyMethod: "trajectory",
      initialState: { level: 1 },
      hamiltonian: "J_z",
      jumps: [{ kind: "local", operator: "j_-", rate: "0.1" }],
      observable: "J_+",
      trajectory: {
        trajectories: 2,
        dt: 0.01,
        maxJumpProbability: 0.02,
        seed: 1,
      },
      dynamics: { startTime: 0, finalTime: 1, samples: 3, stepsPerInterval: 2 },
    }),
    "observable",
    "trajectory streaming rejects a non-Hermitian observable",
  );
  assertRejects(
    () => api.generate({
      N: 2,
      d: 2,
      calculation: "dynamics",
      initialState: { level: 1 },
      hamiltonian: "J_z",
      jumps: [{ kind: "local", operator: "j_-", rate: "0.1" }],
      observable: "J_z",
      dynamics: { startTime: 1, finalTime: 1, samples: 3, stepsPerInterval: 2 },
    }),
    "dynamics final time",
    "dynamics interval must be ordered",
  );
  assertRejects(
    () => api.generate({
      N: 2,
      d: 2,
      calculation: "dynamics",
      initialState: { level: 1 },
      hamiltonian: "J_z",
      jumps: [{ kind: "local", operator: "j_-", rate: "0.1" }],
      observable: "J_z",
      dynamics: { startTime: 0, finalTime: 1, samples: 1, stepsPerInterval: 2 },
    }),
    "dynamics samples",
    "dynamics requires at least two output samples",
  );
  assertRejects(
    () => api.generate({
      N: 2,
      d: 2,
      calculation: "dynamics",
      initialState: { level: 1 },
      hamiltonian: "J_z",
      jumps: [{ kind: "local", operator: "j_-", rate: "0.1" }],
      observable: "J_z",
      dynamics: { startTime: 0, finalTime: 1, samples: 3, stepsPerInterval: 0 },
    }),
    "steps per output interval",
    "dynamics step count must be positive",
  );
  assertRejects(
    () => api.generate({
      N: 2,
      d: 2,
      calculation: "spectrum",
      hamiltonian: "J_z",
      jumps: [{ kind: "local", operator: "j_-", rate: "0.1" }],
      spectrum: { target: "smallest-real", nev: 2, seed: 1 },
    }),
    "spectrum target",
    "unknown selected-spectrum target",
  );
  assertRejects(
    () => api.generate({
      N: 2,
      d: 2,
      calculation: "spectrum",
      hamiltonian: "J_z",
      jumps: [{ kind: "local", operator: "j_-", rate: "0.1" }],
      spectrum: { target: "largest-real", nev: 0, seed: 1 },
    }),
    "spectrum eigenvalues",
    "spectrum eigenvalue count must be positive",
  );
  assertRejects(
    () => api.generate({
      N: 2,
      d: 2,
      calculation: "spectrum",
      hamiltonian: "J_z",
      jumps: [{ kind: "local", operator: "j_-", rate: "0.1" }],
      spectrum: { target: "largest-real", nev: 2, seed: -1 },
    }),
    "spectrum seed",
    "spectrum seed must be nonnegative",
  );
  assertRejects(
    () => api.generate({
      N: 2,
      d: 2,
      calculation: "gap",
      hamiltonian: "J_z",
      jumps: [{ kind: "local", operator: "j_-", rate: "0.1" }],
      gap: { nev: 2, krylovdim: 2 },
    }),
    "gap Krylov dimension",
    "gap Krylov dimension must exceed the Ritz count",
  );
  assertRejects(
    () => api.generate({
      N: 2,
      d: 2,
      calculation: "steady",
      hamiltonian: "J_z",
      jumps: [{ kind: "local", operator: "j_-", rate: "0.1" }],
      resources: { memoryBudgetMiB: 0 },
    }),
    "memory budget",
    "memory budget must be positive",
  );

  assertRejects(
    () => api.generate({
      N: 2,
      d: 2,
      target: "steady",
      hamiltonian: "J_z",
      jumps: [{ operator: "j_-", rate: "0.1" }],
    }),
    "jump 1 kind",
    "missing jump semantics are rejected instead of defaulting to local",
  );
  assertRejects(
    () => api.generate({
      N: 2,
      d: 2,
      target: "steady",
      hamiltonian: "J_z",
      jumps: [{ kind: "diagonal", operator: "j_-", rate: "0.1" }],
    }),
    "jump 1 kind",
    "unknown jump semantics are rejected",
  );

  const zeroRate = api.generate({
    N: 2,
    d: 2,
    target: "steady",
    hamiltonian: "J_z",
    jumps: [{ kind: "local", operator: "j_-", rate: 0 }],
  });
  assertIncludes(
    zeroRate.code,
    "LocalJump(jump_1; rate=0.0)",
    "a numeric zero jump rate is preserved",
  );

  assertRejects(
    () => api.generate({
      N: 4, d: 2, target: "steady", hamiltonian: "",
      jumps: [{ kind: "local", operator: "J_-", rate: "1" }],
    }),
    "jump 1",
    "local/collective mismatch",
  );
  assertRejects(
    () => api.generate({
      N: 4, d: 3, target: "steady", hamiltonian: String.raw`J_z`,
      jumps: [{ kind: "local", operator: String.raw`\sigma_-`, rate: "1" }],
    }),
    "jump 1",
    "Pauli symbols on a qudit",
  );
  assertRejects(
    () => api.generate({
      N: 4, d: 2, target: "steady", hamiltonian: String.raw`\Omega_1 J_x`,
      jumps: [{ kind: "local", operator: String.raw`\mathcal{D}[\sigma_-]`, rate: "1" }],
    }),
    "jump 1",
    "full dissipator in operator field",
  );
  assertRejects(
    () => api.generate({
      N: 4, d: 2, target: "steady",
      hamiltonian: String.raw`\Omega J_x; run(` + "`" + "malicious" + "`" + ")",
      jumps: [],
    }),
    "hamiltonian",
    "code-like input",
  );
  assertRejects(
    () => api.generate({
      N: 1e20, d: 2, target: "steady", hamiltonian: "J_z",
      jumps: [{ kind: "local", operator: "j_-", rate: "1" }],
    }),
    "N",
    "unsafe particle count",
  );
  assertRejects(
    () => api.generate({
      N: 3, d: 2, target: "steady", hamiltonian: "1e999 J_z",
      jumps: [{ kind: "local", operator: "j_-", rate: "1" }],
    }),
    "hamiltonian",
    "overflowing numeric literal",
  );
  assertRejects(
    () => api.generate({
      N: 3, d: 2, target: "steady", hamiltonian: "(1/0) J_z",
      jumps: [{ kind: "local", operator: "j_-", rate: "1" }],
    }),
    "hamiltonian",
    "nonfinite scalar subexpression",
  );
  assertRejects(
    () => api.generate({
      N: 4, d: 2, target: "steady", hamiltonian: "g J_z",
      jumps: [{ kind: "local", operator: "j_-", rate: "1" }],
      parameters: "g=N",
    }),
    "parameters",
    "parameter assignments cannot depend on N",
  );
  assertRejects(
    () => api.generate({
      N: 4, d: 2, target: "expectation", hamiltonian: "0.1 J_z",
      jumps: [{ kind: "local", operator: "j_-", rate: "1" }],
      observable: "J_x^0",
    }),
    "observable",
    "zero operator power",
  );
  assertRejects(
    () => api.generate({
      architecture: "tensor-network",
      N: 2, d: 2, target: "steady", hamiltonian: "J_z", jumps: [],
    }),
    "architecture",
    "unknown composite architecture",
  );
  assertRejects(
    () => api.generate({
      architecture: "global-pseudomode",
      N: 2,
      d: 2,
      target: "steady",
      steadyMethod: "trajectory",
      hamiltonian: "",
      jumps: [],
      trajectory: {
        trajectories: 8,
        initialLevel: 1,
        settlingTime: 2.0,
        dt: 0.01,
        samplesPerTrajectory: 2,
        samplingInterval: 0.5,
        maxJumpProbability: 0.02,
        seed: 3,
      },
      pseudomode: {
        nmax: 1,
        frequency: "1",
        damping: "0.1",
        thermalOccupation: "0",
        couplingOperator: "j_-",
        couplingStrength: "0.1",
        counterrotatingStrength: "0",
      },
    }),
    "steady-state method",
    "shared-pseudomode trajectory steady-state estimator is not silently approximated",
  );
  assertRejects(
    () => api.generate({
      architecture: "pi",
      N: 2, d: 2, target: "steady", steadyMethod: "trajectory",
      hamiltonian: "J_z",
      jumps: [{ kind: "local", operator: "j_-", rate: "0.1" }],
      trajectory: { trajectories: 8, initialLevel: 3 },
    }),
    "initial local level",
    "trajectory initial level must belong to the system local basis",
  );
  assertRejects(
    () => api.generate({
      architecture: "pi",
      N: 2, d: 2, target: "steady", steadyMethod: "trajectory",
      hamiltonian: "J_z",
      jumps: [{ kind: "local", operator: "j_-", rate: "0.1" }],
      trajectory: {
        trajectories: 8,
        initialLevel: 1,
        maxJumpProbability: 1,
      },
    }),
    "maximum jump probability",
    "trajectory jump-probability guard must lie below one",
  );
  assertRejects(
    () => api.generate({
      N: 2,
      d: 2,
      target: "steady",
      steadyMethod: "path-integral",
      hamiltonian: "J_z",
      jumps: [{ kind: "local", operator: "j_-", rate: "0.1" }],
    }),
    "steady-state method",
    "unknown steady-state method",
  );
  assertRejects(
    () => api.generate({
      N: 2,
      d: 2,
      target: "steady",
      steadyMethod: "trajectory",
      hamiltonian: "J_z",
      jumps: [{ kind: "local", operator: "j_-", rate: "0.1" }],
      trajectory: {
        trajectories: 1,
        initialLevel: 1,
        settlingTime: 2,
        dt: 0.01,
        samplesPerTrajectory: 2,
        samplingInterval: 0.5,
        maxJumpProbability: 0.02,
        seed: 1,
      },
    }),
    "trajectories",
    "trajectory estimator requires independent path statistics",
  );
  assertRejects(
    () => api.generate({
      N: 2,
      d: 2,
      target: "steady",
      steadyMethod: "trajectory",
      hamiltonian: "J_z",
      jumps: [{ kind: "local", operator: "j_-", rate: "0.1" }],
      trajectory: {
        trajectories: 8,
        initialLevel: 3,
        settlingTime: 2,
        dt: 0.01,
        samplesPerTrajectory: 2,
        samplingInterval: 0.5,
        maxJumpProbability: 0.02,
        seed: 1,
      },
    }),
    "initial local level",
    "trajectory initial level must match the local dimension",
  );
  assertRejects(
    () => api.generate({
      N: 2,
      d: 2,
      target: "steady",
      steadyMethod: "trajectory",
      hamiltonian: "J_z",
      jumps: [{ kind: "local", operator: "j_-", rate: "0.1" }],
      trajectory: {
        trajectories: 8,
        initialLevel: 1,
        settlingTime: 2,
        dt: 0.01,
        samplesPerTrajectory: 2,
        samplingInterval: 0.5,
        maxJumpProbability: 1,
        seed: 1,
      },
    }),
    "maximum jump probability",
    "trajectory jump-probability guard must be a probability",
  );
  assertRejects(
    () => api.generate({
      architecture: "local-pseudomode",
      N: 2, d: 2, target: "steady", hamiltonian: "", jumps: [],
      pseudomode: { nmax: -1, couplingOperator: "j_-", couplingStrength: "1" },
    }),
    "pseudomode cutoff",
    "negative pseudomode cutoff",
  );
  assertRejects(
    () => api.generate({
      architecture: "global-pseudomode",
      N: 2, d: 2, target: "steady", hamiltonian: "", jumps: [],
      pseudomode: {
        nmax: 1, frequency: "1", damping: "-0.1", thermalOccupation: "0",
        couplingOperator: "j_-", couplingStrength: "0.1",
      },
    }),
    "pseudomode damping",
    "negative pseudomode damping",
  );
  assertRejects(
    () => api.generate({
      architecture: "global-pseudomode",
      N: 2, d: 2, target: "steady", hamiltonian: "", jumps: [],
      pseudomode: {
        nmax: 1, frequency: "1", damping: "0.1", thermalOccupation: "-0.2",
        couplingOperator: "j_-", couplingStrength: "0.1",
      },
    }),
    "pseudomode thermal occupation",
    "negative thermal occupation",
  );
  assertRejects(
    () => api.generate({
      architecture: "global-pseudomode",
      N: 2, d: 2, target: "steady", hamiltonian: "", jumps: [],
      pseudomode: {
        nmax: 1, frequency: "J_z", damping: "0.1", thermalOccupation: "0",
        couplingOperator: "j_-", couplingStrength: "0.1",
      },
    }),
    "pseudomode frequency",
    "operator-valued pseudomode scalar",
  );
  assertRejects(
    () => api.generate({
      architecture: "global-pseudomode",
      N: 2, d: 2, target: "steady", hamiltonian: "", jumps: [],
      pseudomode: {
        nmax: 1, frequency: "1", damping: "0.1", thermalOccupation: "0",
        couplingOperator: "J_-", couplingStrength: "0.1",
      },
    }),
    "pseudomode coupling operator",
    "collective pseudomode seed",
  );
  assertRejects(
    () => api.generate({
      architecture: "local-pseudomode",
      N: 2, d: 3, target: "steady", hamiltonian: "", jumps: [],
      pseudomode: {
        nmax: 1, frequency: "1", damping: "0.1", thermalOccupation: "0",
        couplingOperator: String.raw`\sigma_-`, couplingStrength: "0.1",
      },
    }),
    "pseudomode coupling operator",
    "Pauli pseudomode seed on a qudit",
  );

  const output = `model code generator tests: ${passed} passed`;
  if (typeof print === "function") print(output);
  else console.log(output);
})();
