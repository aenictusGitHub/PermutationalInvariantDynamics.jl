(function () {
  "use strict";

  const api = typeof PIDModelCodeGenerator !== "undefined"
    ? PIDModelCodeGenerator
    : require("../src/assets/model_code_generator_core.js");
  let passed = 0;

  function assert(condition, message) {
    if (!condition) throw new Error(message);
    passed += 1;
  }

  function assertIncludes(text, fragment, message) {
    assert(text.includes(fragment), `${message}\nMissing: ${fragment}\nIn:\n${text}`);
  }

  function assertRejects(action, field, message) {
    let caught = null;
    try {
      action();
    } catch (error) {
      caught = error;
    }
    assert(caught instanceof api.GeneratorError, `${message}: expected GeneratorError`);
    assert(caught.field === field, `${message}: expected field ${field}, got ${caught.field}`);
  }

  const driven = api.generate({
    N: 8,
    d: 2,
    target: "expectation",
    hamiltonian: String.raw`\frac{\Omega}{2}\sum_{i=1}^{N}\sigma_x^{(i)}`,
    jumps: [
      { kind: "local", operator: String.raw`\sigma_-`, rate: String.raw`\gamma_{\downarrow}` },
      { kind: "local", operator: String.raw`\sigma_+`, rate: String.raw`\gamma_{\uparrow}` },
      { kind: "collective", operator: "J_-", rate: String.raw`\Gamma` },
    ],
    observable: "J_z/N",
    parameters: String.raw`\Omega = 0.7
\gamma_{\downarrow} = 0.12
\gamma_{\uparrow} = 0.02
\Gamma = 0.01`,
  });

  assertIncludes(driven.code, "LocalHamiltonian(", "one-body Hamiltonian lowering");
  assertIncludes(driven.code, "2 * spin.jx", "Pauli normalization");
  assertIncludes(driven.code, "LocalJump(jump_1;", "independent local jump");
  assertIncludes(driven.code, "CollectiveJump(jump_3;", "collective jump");
  assertIncludes(driven.code, "gamma_down = 0.12", "subscripted parameter");
  assertIncludes(driven.code, "CollectiveObservablePlan", "prepared observable");
  assertIncludes(driven.code, "backend=:auto", "automatic backend");
  assert(!driven.code.includes("TODO"), "provided parameters must not receive placeholders");
  assert(driven.summary.terms === 4, "term count");

  const nonlinear = api.generate({
    N: 20,
    d: 2,
    target: "expectation",
    hamiltonian: String.raw`-\frac{2h}{N}J_z-\frac{2\lambda}{N}J_x^2`,
    jumps: [{ kind: "collective", operator: "J_-", rate: String.raw`\Gamma` }],
    observable: "J_x^2/N^2",
    parameters: String.raw`h=0.4
\lambda=1.0
\Gamma=0.2`,
  });
  assertIncludes(nonlinear.code, "DirectPIHamiltonian(H_collective)", "nonlinear PI Hamiltonian");
  assertIncludes(nonlinear.code, "Jx * Jx", "PIOperator powers use multiplication");
  assertIncludes(nonlinear.code, "(1.0 / (Float64(N) ^ 2)) * (observable_Jx * observable_Jx)", "PIOperator division becomes scalar multiplication");
  assertIncludes(nonlinear.code, "expectation(rho_ss, adjoint(observable))", "polynomial observable");
  assertIncludes(nonlinear.code, "OneBodyGeometry(basis)", "shared collective geometry");

  function hamiltonianLine(formula) {
    const code = api.generate({
      N: 3, d: 2, target: "steady", hamiltonian: formula,
      jumps: [{ kind: "local", operator: "j_-", rate: "1" }],
      parameters: "",
    }).code;
    return code.split("\n").find((line) => line.startsWith("H_collective ="));
  }
  assert(
    hamiltonianLine("-J_x^2") === "H_collective = (-(Jx * Jx))",
    "exponentiation must bind more tightly than unary minus",
  );
  assert(
    hamiltonianLine("-(J_x^2)") === "H_collective = (-(Jx * Jx))",
    "explicitly grouped negative square",
  );
  assert(
    hamiltonianLine("(-J_x)^2") === "H_collective = ((-Jx) * (-Jx))",
    "explicitly squared negative operator",
  );

  const qudit = api.generate({
    N: 5,
    d: 3,
    target: "steady",
    hamiltonian: String.raw`\Omega J_x`,
    jumps: [{ kind: "local", operator: "j_-", rate: String.raw`\gamma` }],
    parameters: String.raw`\Omega=1
\gamma=0.1`,
  });
  assertIncludes(qudit.code, "spin_matrices(d)", "qudit spin matrices");
  assertIncludes(qudit.code, "jump_1 = spin.jm", "qudit local lowering operator");

  const localPseudomode = api.generate({
    architecture: "local-pseudomode",
    N: 2,
    d: 2,
    target: "expectation",
    hamiltonian: String.raw`\Omega J_z + \chi J_x^2`,
    jumps: [
      { kind: "local", operator: String.raw`\sigma_-`, rate: String.raw`\gamma` },
      { kind: "collective", operator: "J_-", rate: String.raw`\Gamma` },
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
\Gamma=0.01
\omega_c=1
\kappa=0.2
nbar=0
g=0.15
g_cr=0`,
  });
  assertIncludes(localPseudomode.code, "BosonicPseudomode(", "local mode specification");
  assertIncludes(localPseudomode.code, "pseudomode_supersite(", "local supersite construction");
  assertIncludes(localPseudomode.code, "embedding = pseudomode_model(", "local embedding builder");
  assertIncludes(localPseudomode.code, "lift_system_operator(site, spin.jx", "nonlinear bare-system lift");
  assertIncludes(localPseudomode.code, "supersite_terms = (DirectPIHamiltonian(H_collective),)", "nonlinear supersite PI term");
  assertIncludes(localPseudomode.code, "system_terms=system_terms", "system channels are lifted by the builder");
  assertIncludes(localPseudomode.code, "backend=:auto", "local embedding keeps guarded automatic compilation");
  assertIncludes(localPseudomode.code, "mode_top_population", "local cutoff diagnostic");
  assertIncludes(localPseudomode.code, "observable_Jx = Jx", "system-only supersite observable reuses prepared polynomial component");
  assert(!localPseudomode.code.includes("collective_spin(basis"), "supersite must not be treated as a spin-(D-1)/2 object");
  assert(!localPseudomode.code.includes("kron("), "generator never emits a full tensor construction");
  assert(localPseudomode.summary.architecture === "local-pseudomode", "local topology summary");
  assert(localPseudomode.summary.cutoff === 1, "local cutoff summary");

  const globalPseudomode = api.generate({
    architecture: "global-pseudomode",
    N: 3,
    d: 2,
    target: "expectation",
    hamiltonian: String.raw`\Omega J_x + \chi J_z^2`,
    jumps: [{ kind: "local", operator: String.raw`\sigma_-`, rate: String.raw`\gamma` }],
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
\omega_c=1
\kappa=0.2
g=0.15`,
  });
  assertIncludes(globalPseudomode.code, "system_model = PIModel(system_basis, system_terms)", "shared mode starts from an ordinary PI model");
  assertIncludes(globalPseudomode.code, "global_pseudomode_model(", "shared embedding builder");
  assertIncludes(globalPseudomode.code, "stationary_state(\n    embedding;", "specialized shared-mode stationary route");
  assertIncludes(globalPseudomode.code, "GMRESAlgorithm(krylovdim=40, maxiter=1000)", "bounded shared-mode GMRES settings");
  assertIncludes(globalPseudomode.code, "rho_system = trace_pseudomodes(rho_ss, embedding)", "packed system reduction");
  assertIncludes(globalPseudomode.code, "rho_mode = global_pseudomode_state", "packed mode reduction");
  assertIncludes(globalPseudomode.code, "LinearAlgebra.ishermitian(", "composite Hermiticity check");
  assertIncludes(globalPseudomode.code, "mode_top_population", "global cutoff diagnostic");
  assert(!globalPseudomode.code.includes("compile(\n    embedding"), "shared model must not be compiled as an ordinary PI model");
  assert(!globalPseudomode.code.includes("kron("), "shared model stays factorized");
  assert(globalPseudomode.summary.route.includes("matrix-free GMRES"), "global route summary");

  const pseudomodeCollision = api.generate({
    architecture: "global-pseudomode",
    N: 2,
    d: 2,
    target: "steady",
    hamiltonian: "",
    jumps: [],
    pseudomode: {
      nmax: 1,
      frequency: "mode",
      damping: "0.2",
      thermalOccupation: "0",
      couplingOperator: "j_-",
      couplingStrength: "coupling",
      counterrotatingStrength: "0",
    },
    parameters: "mode=1\ncoupling=0.1",
  });
  assertIncludes(pseudomodeCollision.code, "parameter_mode = 1.0", "mode name collision is renamed");
  assertIncludes(pseudomodeCollision.code, "parameter_coupling = 0.1", "coupling name collision is renamed");
  assertIncludes(pseudomodeCollision.code, "frequency=parameter_mode", "renamed mode parameter is used");
  assertIncludes(pseudomodeCollision.code, "strength=parameter_coupling", "renamed coupling parameter is used");

  const missing = api.generate({
    N: 3,
    d: 2,
    target: "steady",
    hamiltonian: String.raw`\Omega J_z`,
    jumps: [{ kind: "local", operator: String.raw`\sigma_-`, rate: String.raw`\gamma` }],
    parameters: String.raw`\Omega=0.2`,
  });
  assertIncludes(missing.code, "gamma = 1.0  # TODO", "missing parameter placeholder");
  assert(missing.warnings.some((warning) => warning.includes("gamma")), "missing parameter warning");

  const collision = api.generate({
    N: 3,
    d: 2,
    target: "steady",
    hamiltonian: "0.1 J_z",
    jumps: [{ kind: "local", operator: "j_-", rate: "jump_1" }],
    parameters: "jump_1=0.2",
  });
  assertIncludes(collision.code, "parameter_jump_1 = 0.2", "temporary-name collision is renamed");
  assertIncludes(collision.code, "rate=parameter_jump_1", "renamed rate is used");

  const inheritedName = api.generate({
    N: 3,
    d: 2,
    target: "steady",
    hamiltonian: "constructor J_z",
    jumps: [{ kind: "local", operator: "j_-", rate: "0.2" }],
    parameters: "constructor=0.1",
  });
  assertIncludes(inheritedName.code, "constructor = 0.1", "plain-object inherited name remains a scalar parameter");

  const wideIntegerPower = api.generate({
    N: 3,
    d: 2,
    target: "steady",
    hamiltonian: "(2^100) J_z",
    jumps: [{ kind: "local", operator: "j_-", rate: "0.2" }],
    parameters: "",
  });
  assertIncludes(wideIntegerPower.code, "(2.0 ^ 100)", "integer coefficient power uses Float64 arithmetic");

  const collectiveOnly = api.generate({
    N: 3,
    d: 2,
    target: "steady",
    hamiltonian: "0.1 J_z",
    jumps: [{ kind: "collective", operator: "J_-", rate: "0.2" }],
    parameters: "",
  });
  assert(
    collectiveOnly.warnings.some((warning) => warning.includes("Schur-sector")),
    "collective-only model warns about sector conservation",
  );

  const doubledRate = api.generate({
    N: 3,
    d: 2,
    target: "steady",
    hamiltonian: "0.1 J_z",
    jumps: [{ kind: "local", operator: String.raw`\sqrt{\gamma}j_-`, rate: String.raw`\gamma` }],
    parameters: String.raw`\gamma=0.2`,
  });
  assert(
    doubledRate.warnings.some((warning) => warning.includes("square")),
    "internal jump coefficient plus external rate warns about double counting",
  );

  assertRejects(
    () => api.generate({
      N: 4, d: 2, target: "steady", hamiltonian: "",
      jumps: [{ kind: "local", operator: "J_-", rate: "1" }],
    }),
    "jump 1",
    "local/collective mismatch",
  );
  assertRejects(
    () => api.generate({
      N: 4, d: 3, target: "steady", hamiltonian: String.raw`J_z`,
      jumps: [{ kind: "local", operator: String.raw`\sigma_-`, rate: "1" }],
    }),
    "jump 1",
    "Pauli symbols on a qudit",
  );
  assertRejects(
    () => api.generate({
      N: 4, d: 2, target: "steady", hamiltonian: String.raw`\Omega_1 J_x`,
      jumps: [{ kind: "local", operator: String.raw`\mathcal{D}[\sigma_-]`, rate: "1" }],
    }),
    "jump 1",
    "full dissipator in operator field",
  );
  assertRejects(
    () => api.generate({
      N: 4, d: 2, target: "steady",
      hamiltonian: String.raw`\Omega J_x; run(` + "`" + "malicious" + "`" + ")",
      jumps: [],
    }),
    "hamiltonian",
    "code-like input",
  );
  assertRejects(
    () => api.generate({
      N: 1e20, d: 2, target: "steady", hamiltonian: "J_z",
      jumps: [{ kind: "local", operator: "j_-", rate: "1" }],
    }),
    "N",
    "unsafe particle count",
  );
  assertRejects(
    () => api.generate({
      N: 3, d: 2, target: "steady", hamiltonian: "1e999 J_z",
      jumps: [{ kind: "local", operator: "j_-", rate: "1" }],
    }),
    "hamiltonian",
    "overflowing numeric literal",
  );
  assertRejects(
    () => api.generate({
      N: 3, d: 2, target: "steady", hamiltonian: "(1/0) J_z",
      jumps: [{ kind: "local", operator: "j_-", rate: "1" }],
    }),
    "hamiltonian",
    "nonfinite scalar subexpression",
  );
  assertRejects(
    () => api.generate({
      N: 4, d: 2, target: "steady", hamiltonian: "g J_z",
      jumps: [{ kind: "local", operator: "j_-", rate: "1" }],
      parameters: "g=N",
    }),
    "parameters",
    "parameter assignments cannot depend on N",
  );
  assertRejects(
    () => api.generate({
      N: 4, d: 2, target: "expectation", hamiltonian: "0.1 J_z",
      jumps: [{ kind: "local", operator: "j_-", rate: "1" }],
      observable: "J_x^0",
    }),
    "observable",
    "zero operator power",
  );
  assertRejects(
    () => api.generate({
      architecture: "tensor-network",
      N: 2, d: 2, target: "steady", hamiltonian: "J_z", jumps: [],
    }),
    "architecture",
    "unknown composite architecture",
  );
  assertRejects(
    () => api.generate({
      architecture: "local-pseudomode",
      N: 2, d: 2, target: "steady", hamiltonian: "", jumps: [],
      pseudomode: { nmax: -1, couplingOperator: "j_-", couplingStrength: "1" },
    }),
    "pseudomode cutoff",
    "negative pseudomode cutoff",
  );
  assertRejects(
    () => api.generate({
      architecture: "global-pseudomode",
      N: 2, d: 2, target: "steady", hamiltonian: "", jumps: [],
      pseudomode: {
        nmax: 1, frequency: "1", damping: "-0.1", thermalOccupation: "0",
        couplingOperator: "j_-", couplingStrength: "0.1",
      },
    }),
    "pseudomode damping",
    "negative pseudomode damping",
  );
  assertRejects(
    () => api.generate({
      architecture: "global-pseudomode",
      N: 2, d: 2, target: "steady", hamiltonian: "", jumps: [],
      pseudomode: {
        nmax: 1, frequency: "1", damping: "0.1", thermalOccupation: "-0.2",
        couplingOperator: "j_-", couplingStrength: "0.1",
      },
    }),
    "pseudomode thermal occupation",
    "negative thermal occupation",
  );
  assertRejects(
    () => api.generate({
      architecture: "global-pseudomode",
      N: 2, d: 2, target: "steady", hamiltonian: "", jumps: [],
      pseudomode: {
        nmax: 1, frequency: "J_z", damping: "0.1", thermalOccupation: "0",
        couplingOperator: "j_-", couplingStrength: "0.1",
      },
    }),
    "pseudomode frequency",
    "operator-valued pseudomode scalar",
  );
  assertRejects(
    () => api.generate({
      architecture: "global-pseudomode",
      N: 2, d: 2, target: "steady", hamiltonian: "", jumps: [],
      pseudomode: {
        nmax: 1, frequency: "1", damping: "0.1", thermalOccupation: "0",
        couplingOperator: "J_-", couplingStrength: "0.1",
      },
    }),
    "pseudomode coupling operator",
    "collective pseudomode seed",
  );
  assertRejects(
    () => api.generate({
      architecture: "local-pseudomode",
      N: 2, d: 3, target: "steady", hamiltonian: "", jumps: [],
      pseudomode: {
        nmax: 1, frequency: "1", damping: "0.1", thermalOccupation: "0",
        couplingOperator: String.raw`\sigma_-`, couplingStrength: "0.1",
      },
    }),
    "pseudomode coupling operator",
    "Pauli pseudomode seed on a qudit",
  );

  const output = `model code generator tests: ${passed} passed`;
  if (typeof print === "function") print(output);
  else console.log(output);
})();
