(function () {
  "use strict";
  const api = typeof PIDModelCodeGenerator !== "undefined"
    ? PIDModelCodeGenerator
    : require("../src/assets/model_code_generator_core.js");
  const result = api.generate({
    architecture: "local-pseudomode",
    N: 1,
    d: 2,
    calculation: "dynamics",
    steadyMethod: "deterministic",
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
    dynamics: {
      startTime: 0,
      finalTime: 0.02,
      samples: 3,
      stepsPerInterval: 2,
    },
    resources: { memoryBudgetMiB: 64 },
  });
  if (typeof print === "function") print(result.code);
  else process.stdout.write(result.code);
})();
