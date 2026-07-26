(function () {
  "use strict";
  const api = typeof PIDModelCodeGenerator !== "undefined"
    ? PIDModelCodeGenerator
    : require("../src/assets/model_code_generator_core.js");
  const result = api.generate({
    architecture: "pi",
    N: 1,
    d: 2,
    calculation: "steady-state",
    hamiltonian: String.raw`\Omega J_x`,
    jumps: [
      { kind: "local", operator: String.raw`\sigma_-`, rate: String.raw`\gamma` },
    ],
    parameters: String.raw`\Omega=0.2
\gamma=0.1`,
    analysis: {
      purity: true,
      entropy: true,
      oneBodyRDM: true,
      qfiAxis: "z",
    },
    resources: { memoryBudgetMiB: 64 },
  });
  if (typeof print === "function") print(result.code);
  else process.stdout.write(result.code);
})();
