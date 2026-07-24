(function () {
  "use strict";
  const api = typeof PIDModelCodeGenerator !== "undefined"
    ? PIDModelCodeGenerator
    : require("../src/assets/model_code_generator_core.js");
  const result = api.generate({
    architecture: "local-pseudomode",
    N: 2,
    d: 2,
    target: "expectation",
    hamiltonian: String.raw`\Omega J_z + \chi J_x^2`,
    jumps: [
      { kind: "local", operator: String.raw`\sigma_-`, rate: String.raw`\gamma` },
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
\omega_c=1.0
\kappa=0.2
nbar=0.0
g=0.15
g_cr=0.0`,
  });
  if (typeof print === "function") print(result.code);
  else process.stdout.write(result.code);
})();
