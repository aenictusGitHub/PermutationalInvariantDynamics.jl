(function () {
  "use strict";
  const api = typeof PIDModelCodeGenerator !== "undefined"
    ? PIDModelCodeGenerator
    : require("../src/assets/model_code_generator_core.js");
  const result = api.generate({
    N: 2,
    d: 2,
    calculation: "steady-observable",
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
\gamma=0.1`,
  });
  const notebook = result.bundle.files.find(
    (artifact) => artifact.name.endsWith("_pluto.jl"),
  );
  if (!notebook) throw new Error("generated Pluto notebook is missing");
  if (typeof print === "function") print(notebook.contents);
  else process.stdout.write(notebook.contents);
})();
