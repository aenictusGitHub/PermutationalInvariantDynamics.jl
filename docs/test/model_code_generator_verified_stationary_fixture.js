(function () {
  "use strict";
  const api = typeof PIDModelCodeGenerator !== "undefined"
    ? PIDModelCodeGenerator
    : require("../src/assets/model_code_generator_core.js");
  const result = api.generate({
    architecture: "pi",
    workflow: "verified-experiment",
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
    parameters: String.raw`\Omega=0.2
\gamma=0.4`,
  });
  if (typeof print === "function") print(result.code);
  else process.stdout.write(result.code);
})();
