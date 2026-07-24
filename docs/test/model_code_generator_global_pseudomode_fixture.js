(function () {
  "use strict";
  const api = typeof PIDModelCodeGenerator !== "undefined"
    ? PIDModelCodeGenerator
    : require("../src/assets/model_code_generator_core.js");
  const result = api.generate({
    architecture: "global-pseudomode",
    N: 3,
    d: 2,
    target: "expectation",
    hamiltonian: String.raw`\Omega J_x + \chi J_z^2`,
    jumps: [
      { kind: "local", operator: String.raw`\sigma_-`, rate: String.raw`\gamma` },
    ],
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
\omega_c=1.0
\kappa=0.2
g=0.15`,
  });
  if (typeof print === "function") print(result.code);
  else process.stdout.write(result.code);
})();
