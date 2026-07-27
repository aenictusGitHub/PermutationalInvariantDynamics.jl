(function () {
  "use strict";
  const api = typeof PIDModelCodeGenerator !== "undefined"
    ? PIDModelCodeGenerator
    : require("../src/assets/model_code_generator_core.js");
  const result = api.generate({
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
        {
          parameter: String.raw`\kappa`,
          start: 0.2,
          stop: 0.4,
          points: 2,
        },
      ],
    },
    resources: { memoryBudgetMiB: 64 },
  });
  if (typeof print === "function") print(result.code);
  else process.stdout.write(result.code);
})();
