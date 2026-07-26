(function () {
  "use strict";
  const api = typeof PIDModelCodeGenerator !== "undefined"
    ? PIDModelCodeGenerator
    : require("../src/assets/model_code_generator_core.js");
  const result = api.generate({
    architecture: "pi",
    workflow: "verified-experiment",
    N: 3,
    d: 2,
    calculation: "dynamics-observable",
    steadyMethod: "deterministic",
    initialState: { level: 2 },
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
    dynamics: {
      startTime: 0,
      finalTime: 0.1,
      samples: 3,
      stepsPerInterval: 4,
    },
  });
  if (typeof print === "function") print(result.code);
  else process.stdout.write(result.code);
})();
