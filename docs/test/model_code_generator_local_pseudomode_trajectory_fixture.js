(function () {
  "use strict";
  const api = typeof PIDModelCodeGenerator !== "undefined"
    ? PIDModelCodeGenerator
    : require("../src/assets/model_code_generator_core.js");
  const result = api.generate({
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
\omega_c=1.0
\kappa=0.3
g=0.1`,
    trajectory: {
      trajectories: 4,
      initialLevel: 1,
      settlingTime: 0.1,
      dt: 0.01,
      samplesPerTrajectory: 1,
      samplingInterval: 0.05,
      maxJumpProbability: 0.02,
      seed: 23,
    },
  });
  if (typeof print === "function") print(result.code);
  else process.stdout.write(result.code);
})();
