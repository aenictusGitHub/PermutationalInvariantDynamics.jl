(function () {
  "use strict";
  const api = typeof PIDModelCodeGenerator !== "undefined"
    ? PIDModelCodeGenerator
    : require("../src/assets/model_code_generator_core.js");
  const result = api.generate({
    architecture: "pi",
    N: 1,
    d: 2,
    calculation: "transient",
    steadyMethod: "quantum-trajectories",
    initialState: { level: 2 },
    hamiltonian: String.raw`\Omega J_x`,
    jumps: [
      { kind: "local", operator: String.raw`\sigma_-`, rate: String.raw`\gamma` },
    ],
    observable: "J_z",
    parameters: String.raw`\Omega=0.2
\gamma=0.1`,
    trajectory: {
      trajectories: 2,
      dt: 0.01,
      maxJumpProbability: 0.02,
      seed: 17,
    },
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
