(function () {
  "use strict";
  const api = typeof PIDModelCodeGenerator !== "undefined"
    ? PIDModelCodeGenerator
    : require("../src/assets/model_code_generator_core.js");
  const result = api.generate({
    architecture: "pi",
    N: 1,
    d: 2,
    calculation: "liouvillian-spectrum",
    hamiltonian: String.raw`\Omega J_z`,
    jumps: [
      {
        kind: "local",
        operator: String.raw`\sigma_-`,
        rate: "0.2",
      },
    ],
    parameters: String.raw`\Omega=0.1`,
    spectrum: {
      target: "largest-real",
      nev: 2,
      seed: 61,
    },
    scan: {
      enabled: true,
      axes: [
        {
          parameter: String.raw`\Omega`,
          start: 0.1,
          stop: 0.3,
          points: 3,
        },
      ],
    },
    resources: { memoryBudgetMiB: 64 },
  });
  if (typeof print === "function") print(result.code);
  else process.stdout.write(result.code);
})();
