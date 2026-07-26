(function () {
  "use strict";
  const api = typeof PIDModelCodeGenerator !== "undefined"
    ? PIDModelCodeGenerator
    : require("../src/assets/model_code_generator_core.js");
  const result = api.generate({
    architecture: "pi",
    N: 2,
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
      trajectories: 4,
      initialLevel: 1,
      settlingTime: 0.2,
      dt: 0.01,
      samplesPerTrajectory: 2,
      samplingInterval: 0.05,
      maxJumpProbability: 0.02,
      seed: 17,
    },
  });
  if (typeof print === "function") print(result.code);
  else process.stdout.write(result.code);
})();
