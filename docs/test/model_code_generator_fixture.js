(function () {
  "use strict";
  const api = typeof PIDModelCodeGenerator !== "undefined"
    ? PIDModelCodeGenerator
    : require("../src/assets/model_code_generator_core.js");
  const result = api.generate({
    N: 4,
    d: 2,
    target: "expectation",
    hamiltonian: String.raw`\frac{\Omega}{2}\sum_i\sigma_x^{(i)}+\chi J_z^2`,
    jumps: [
      { kind: "local", operator: String.raw`\sigma_-`, rate: String.raw`\gamma` },
      { kind: "collective", operator: "J_-", rate: String.raw`\Gamma` },
    ],
    observable: "J_z/N",
    parameters: String.raw`\Omega=0.5
\chi=0.02
\gamma=0.1
\Gamma=0.01`,
  });
  if (typeof print === "function") print(result.code);
  else process.stdout.write(result.code);
})();
