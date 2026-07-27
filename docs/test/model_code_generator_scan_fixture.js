(function () {
  "use strict";
  const api = typeof PIDModelCodeGenerator !== "undefined"
    ? PIDModelCodeGenerator
    : require("../src/assets/model_code_generator_core.js");
  const result = api.generate({
    architecture: "pi",
    N: 1,
    d: 2,
    calculation: "steady-observable",
    steadyMethod: "deterministic",
    hamiltonian: "",
    jumps: [
      {
        kind: "local",
        operator: String.raw`\sigma_-`,
        rate: String.raw`\gamma_{\downarrow}`,
      },
      {
        kind: "local",
        operator: String.raw`\sigma_+`,
        rate: String.raw`\gamma_{\uparrow}`,
      },
    ],
    observable: "J_z",
    parameters: "",
    scan: {
      enabled: true,
      axes: [
        {
          parameter: String.raw`\gamma_{\downarrow}`,
          start: 0.2,
          stop: 0.4,
          points: 2,
        },
        {
          parameter: String.raw`\gamma_{\uparrow}`,
          start: 0.1,
          stop: 0.3,
          points: 2,
        },
      ],
    },
    resources: { memoryBudgetMiB: 64 },
  });
  if (typeof print === "function") print(result.code);
  else process.stdout.write(result.code);
})();
