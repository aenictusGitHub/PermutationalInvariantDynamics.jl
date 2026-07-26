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
    assert(
      text.includes(fragment),
      `${message}\nMissing: ${fragment}\nIn:\n${text}`,
    );
  }

  function baseConfiguration(overrides) {
    return Object.assign({
      architecture: "pi",
      workflow: "direct-api",
      N: 2,
      d: 2,
      calculation: "steady-observable",
      steadyMethod: "deterministic",
      hamiltonian: String.raw`\Omega J_x`,
      jumps: [
        {
          kind: "local",
          operator: String.raw`\sigma_-`,
          rate: String.raw`\gamma`,
        },
      ],
      observable: "J_z/N",
      parameters: String.raw`\Omega=0.4
\gamma=0.1`,
      resources: { memoryBudgetMiB: 64 },
    }, overrides || {});
  }

  const direct = api.generate(baseConfiguration());
  assert(
    direct.summary.coordinates === "10",
    "complete N=2 qubit PI coordinate count",
  );
  assert(
    direct.summary.oneComplexVectorBytes === "160",
    "ComplexF64 vector lower bound",
  );
  assert(
    direct.resources.oneVectorFitsBudget === true,
    "one-vector budget comparison",
  );
  assert(
    direct.manifest.schema ===
      "permutational-invariant-dynamics/model-assistant/v1",
    "versioned manifest schema",
  );
  assert(
    JSON.parse(direct.manifestText).model.parameters.Omega === 0.4,
    "manifest is valid JSON with normalized parameters",
  );
  assert(
    direct.bundle.files.length === 3,
    "bundle contains Julia, manifest, and README files",
  );
  assert(
    direct.bundle.files[0].name.endsWith(".jl") &&
      direct.bundle.files[1].name.endsWith(".json") &&
      direct.bundle.files[2].name.endsWith("_README.txt"),
    "bundle file extensions",
  );
  assertIncludes(
    direct.code,
    "# Exact PI/composite coordinates: 10.",
    "generated resource header",
  );
  assertIncludes(
    direct.code,
    'println("retained coordinates = ", pi_dimension(basis))',
    "runtime representation report",
  );

  const verified = api.generate(baseConfiguration({
    workflow: "verified-experiment",
  }));
  assertIncludes(
    verified.code,
    "experiment = PIExperiment(",
    "typed experiment construction",
  );
  assertIncludes(
    verified.code,
    "experiment_plan = explain_experiment(experiment)",
    "explainable preflight",
  );
  assertIncludes(
    verified.code,
    "experiment_result = verified_solve(experiment)",
    "verified solve",
  );
  assertIncludes(
    verified.code,
    "experiment_result.provenance.structural_digest",
    "provenance digest",
  );
  assert(
    !verified.code.includes("steady = stationary_state("),
    "verified workflow does not run a second direct stationary solve",
  );
  assert(
    verified.summary.workflow === "verified-experiment",
    "workflow summary",
  );

  const localDynamics = api.generate(baseConfiguration({
    architecture: "local-pseudomode",
    workflow: "verified-experiment",
    N: 2,
    calculation: "dynamics-observable",
    hamiltonian: String.raw`\Omega J_x`,
    initialState: { level: 1 },
    dynamics: {
      startTime: 0,
      finalTime: 0.1,
      samples: 3,
      stepsPerInterval: 4,
    },
    pseudomode: {
      nmax: 1,
      frequency: String.raw`\omega_c`,
      damping: String.raw`\kappa`,
      thermalOccupation: "0",
      couplingOperator: String.raw`\sigma_z`,
      couplingStrength: "g",
      counterrotatingStrength: "0",
    },
    parameters: String.raw`\Omega=0.4
\gamma=0.1
\omega_c=1
\kappa=0.2
g=0.05`,
  }));
  assert(
    localDynamics.summary.coordinates === "136",
    "supersite coordinate count",
  );
  assertIncludes(
    localDynamics.code,
    "save_states=true, memory_budget=MEMORY_BUDGET",
    "verified dynamics retains bounded sampled states",
  );
  assertIncludes(
    localDynamics.code,
    "observable_values = experiment_result.observables[:observable]",
    "verified observable output",
  );

  for (const invalid of [
    baseConfiguration({
      workflow: "verified-experiment",
      steadyMethod: "trajectory",
    }),
    baseConfiguration({
      architecture: "global-pseudomode",
      workflow: "verified-experiment",
      pseudomode: {
        nmax: 1,
        frequency: "1",
        damping: "0.1",
        thermalOccupation: "0",
        couplingOperator: String.raw`\sigma_z`,
        couplingStrength: "0.1",
        counterrotatingStrength: "0",
      },
    }),
  ]) {
    let error = null;
    try {
      api.generate(invalid);
    } catch (caught) {
      error = caught;
    }
    assert(error instanceof api.GeneratorError, "unsupported verified route");
    assert(error.field === "workflow", "unsupported route identifies workflow");
  }

  if (typeof print === "function") {
    print(`model code generator productization tests: ${passed} passed`);
  } else {
    process.stdout.write(
      `model code generator productization tests: ${passed} passed\n`,
    );
  }
})();
