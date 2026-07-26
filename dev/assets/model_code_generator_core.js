(function (root, factory) {
  "use strict";
  const api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  }
  root.PIDModelCodeGenerator = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const VERSION = "1.4.0";
  const MAX_FORMULA_LENGTH = 2000;
  const JULIA_RESERVED = new Set([
    "baremodule", "begin", "break", "catch", "const", "continue", "do",
    "else", "elseif", "end", "export", "false", "finally", "for",
    "function", "global", "if", "import", "let", "local", "macro",
    "module", "quote", "return", "struct", "true", "try", "using",
    "while", "where", "mutable", "primitive", "abstract", "type",
  ]);
  const GENERATED_NAMES = new Set([
    "N", "d", "basis", "spin", "terms", "model", "prepared", "steady",
    "rho_ss", "observable", "observable_plan", "observable_value",
    "one_body_geometry", "H_collective", "pi", "im", "Inf", "NaN",
    "architecture", "MEMORY_BUDGET", "STEADY_ATOL", "STEADY_RTOL",
    "STATE_VALIDATION_TOL", "system_basis", "system_terms",
    "system_model", "H_system", "site", "embedding", "mode", "coupling",
    "supersite_terms", "rho_system", "rho_mode", "mode_operators",
    "mode_top_plan", "mode_top_population", "observable_one_body_geometry",
    "top_projector", "top_population", "nmax", "rho0", "system_initial",
    "trajectory_plan", "trajectory_workers", "trajectory_workspace",
    "TRAJECTORIES", "INITIAL_LEVEL", "SETTLING_TIME", "TRAJECTORY_DT",
    "SAMPLES_PER_TRAJECTORY", "SAMPLING_INTERVAL",
    "MAX_JUMP_PROBABILITY", "TRAJECTORY_SEED",
    "times", "dynamics", "dynamics_statistics", "observable_values",
    "observable_standard_error", "DYNAMICS_START_TIME",
    "DYNAMICS_FINAL_TIME", "DYNAMICS_SAMPLES", "STEPS_PER_INTERVAL",
    "spectrum", "spectrum_values", "SPECTRUM_NEV", "SPECTRUM_SEED",
    "gap_source", "gap_result", "GAP_NEV", "GAP_KRYLOVDIM",
    "analysis_state", "analysis_basis", "analysis_geometry",
    "analysis_trace_plan", "analysis_trace_workspace",
    "analysis_rdm_workspace", "one_body_density_matrix",
    "system_purity", "system_entropy", "qfi_plan", "qfi_value",
    "system_observable", "streaming_observable",
    "composite_trajectory_plan", "evolution_workspace", "current",
    "trajectory_preflight",
    "experiment", "experiment_plan", "experiment_result",
    "Random", "LinearAlgebra", "index", "time", "value",
  ]);
  const GREEK = new Set([
    "alpha", "beta", "gamma", "Gamma", "delta", "Delta", "epsilon",
    "varepsilon", "zeta", "eta", "theta", "Theta", "iota", "kappa",
    "lambda", "Lambda", "mu", "nu", "xi", "Xi", "rho", "tau",
    "upsilon", "Upsilon", "phi", "varphi", "Phi", "chi", "psi",
    "Psi", "omega", "Omega", "hbar",
  ]);
  const UNICODE_REPLACEMENTS = new Map([
    ["−", "-"], ["–", "-"], ["×", "*"], ["·", "*"], ["⋅", "*"],
    ["π", "\\pi"], ["Ω", "\\Omega"], ["ω", "\\omega"],
    ["Γ", "\\Gamma"], ["γ", "\\gamma"], ["κ", "\\kappa"],
    ["χ", "\\chi"], ["λ", "\\lambda"], ["Δ", "\\Delta"],
    ["δ", "\\delta"], ["η", "\\eta"], ["ϕ", "\\phi"], ["φ", "\\phi"],
    ["↑", "\\uparrow"], ["↓", "\\downarrow"],
  ]);

  const OPERATOR_ATOMS = Object.freeze(Object.assign(Object.create(null), {
    Jx: { family: "collective-spin", component: "x", local: "spin.jx", pi: "Jx" },
    Jy: { family: "collective-spin", component: "y", local: "spin.jy", pi: "Jy" },
    Jz: { family: "collective-spin", component: "z", local: "spin.jz", pi: "Jz" },
    Jp: { family: "collective-spin", component: "plus", local: "spin.jp", pi: "Jp" },
    Jm: { family: "collective-spin", component: "minus", local: "spin.jm", pi: "Jm" },
    SSx: { family: "collective-pauli", component: "x", local: "(2 * spin.jx)", pi: "(2 * Jx)" },
    SSy: { family: "collective-pauli", component: "y", local: "(2 * spin.jy)", pi: "(2 * Jy)" },
    SSz: { family: "collective-pauli", component: "z", local: "(2 * spin.jz)", pi: "(2 * Jz)" },
    SSp: { family: "collective-pauli", component: "plus", local: "spin.jp", pi: "Jp" },
    SSm: { family: "collective-pauli", component: "minus", local: "spin.jm", pi: "Jm" },
    jx: { family: "local-spin", component: "x", local: "spin.jx" },
    jy: { family: "local-spin", component: "y", local: "spin.jy" },
    jz: { family: "local-spin", component: "z", local: "spin.jz" },
    jp: { family: "local-spin", component: "plus", local: "spin.jp" },
    jm: { family: "local-spin", component: "minus", local: "spin.jm" },
    sigmax: { family: "local-pauli", component: "x", local: "(2 * spin.jx)" },
    sigmay: { family: "local-pauli", component: "y", local: "(2 * spin.jy)" },
    sigmaz: { family: "local-pauli", component: "z", local: "(2 * spin.jz)" },
    sigmap: { family: "local-pauli", component: "plus", local: "spin.jp" },
    sigmam: { family: "local-pauli", component: "minus", local: "spin.jm" },
  }));

  class GeneratorError extends Error {
    constructor(field, message) {
      super(message);
      this.name = "GeneratorError";
      this.field = field;
    }
  }

  function fail(field, message) {
    throw new GeneratorError(field, message);
  }

  function replaceUnicode(text) {
    let result = text;
    for (const [source, replacement] of UNICODE_REPLACEMENTS) {
      result = result.split(source).join(replacement);
    }
    return result;
  }

  function groupedArgument(text, start, field, command) {
    let index = start;
    while (index < text.length && /\s/.test(text[index])) index += 1;
    if (text[index] !== "{") {
      fail(field, `${command} must use braced arguments, for example \\frac{a}{b}.`);
    }
    let depth = 1;
    const contentStart = index + 1;
    index += 1;
    while (index < text.length && depth > 0) {
      if (text[index] === "{") depth += 1;
      if (text[index] === "}") depth -= 1;
      index += 1;
    }
    if (depth !== 0) fail(field, `Unbalanced braces after ${command}.`);
    return { content: text.slice(contentStart, index - 1), end: index };
  }

  function expandGroupedCommand(text, command, count, replacement, field) {
    let result = text;
    let guard = 0;
    while (result.includes(command)) {
      guard += 1;
      if (guard > 100) fail(field, `Too many nested ${command} commands.`);
      const position = result.lastIndexOf(command);
      let cursor = position + command.length;
      const groups = [];
      for (let index = 0; index < count; index += 1) {
        const group = groupedArgument(result, cursor, field, command);
        groups.push(group.content);
        cursor = group.end;
      }
      result = result.slice(0, position) + replacement(groups) + result.slice(cursor);
    }
    return result;
  }

  function expandWrapper(text, command, field) {
    return expandGroupedCommand(text, command, 1, (groups) => `(${groups[0]})`, field);
  }

  function axisName(axis) {
    if (axis === "+") return "p";
    if (axis === "-") return "m";
    return axis;
  }

  function replaceSummedOperators(text) {
    const index = String.raw`(?:_\s*(?:\{\s*i(?:\s*=\s*1)?\s*\}|i))`;
    const upper = String.raw`(?:\s*\^\s*(?:\{\s*N\s*\}|N))?`;
    const site = String.raw`(?:\s*\^\s*\{\s*\(\s*i\s*\)\s*\})?`;
    const sigma = new RegExp(
      String.raw`\\sum\s*${index}${upper}\s*\\sigma\s*_\s*(?:\{\s*([xyz+\-])\s*\}|([xyz+\-]))${site}`,
      "g",
    );
    const localSpin = new RegExp(
      String.raw`\\sum\s*${index}${upper}\s*j\s*_\s*(?:\{\s*([xyz+\-])\s*\}|([xyz+\-]))${site}`,
      "g",
    );
    let result = text.replace(sigma, (_, braced, plain) => ` SS${axisName(braced || plain)} `);
    result = result.replace(localSpin, (_, braced, plain) => ` J${axisName(braced || plain)} `);
    return result;
  }

  function sanitizeSubscript(subscript) {
    return subscript
      .replace(/\\downarrow/g, "down")
      .replace(/\\uparrow/g, "up")
      .replace(/\\plus/g, "plus")
      .replace(/\\minus/g, "minus")
      .replace(/\+/g, "plus")
      .replace(/-/g, "minus")
      .replace(/[^A-Za-z0-9]+/g, "_")
      .replace(/^_+|_+$/g, "");
  }

  function safeIdentifier(name) {
    let result = name.replace(/[^A-Za-z0-9_]/g, "_");
    if (!/^[A-Za-z_]/.test(result)) result = `parameter_${result}`;
    if (JULIA_RESERVED.has(result) || GENERATED_NAMES.has(result) ||
        /^jump_\d+$/.test(result)) {
      result = `parameter_${result}`;
    }
    return result;
  }

  function floatLiteral(value) {
    if (Object.is(value, -0)) return "-0.0";
    const text = String(value);
    return /[.eE]/.test(text) ? text : `${text}.0`;
  }

  function replaceLatexMacros(text, field) {
    return text.replace(
      /\\([A-Za-z]+)(?:\s*_\s*(?:\{([^{}]*)\}|([A-Za-z0-9+\-]+)))?/g,
      (whole, name, bracedSubscript, plainSubscript) => {
        if (name === "pi" && bracedSubscript === undefined && plainSubscript === undefined) {
          return "pi";
        }
        if (name === "uparrow" || name === "downarrow") return name === "uparrow" ? "up" : "down";
        if (!GREEK.has(name)) {
          fail(field, `Unsupported LaTeX command \\${name}. Use the documented spin symbols and ordinary named scalar parameters.`);
        }
        const subscript = bracedSubscript !== undefined ? bracedSubscript : plainSubscript;
        const suffix = subscript === undefined ? "" : sanitizeSubscript(subscript);
        return safeIdentifier(suffix ? `${name}_${suffix}` : name);
      },
    );
  }

  function normalizeLatex(source, field) {
    if (typeof source !== "string") fail(field, "Expected a LaTeX string.");
    if (source.length > MAX_FORMULA_LENGTH) {
      fail(field, `Formula is longer than ${MAX_FORMULA_LENGTH} characters.`);
    }
    let text = replaceUnicode(source.trim());
    if (!text) fail(field, "Formula is empty.");
    text = text
      .replace(/^\$+|\$+$/g, "")
      .replace(/\\\(|\\\)|\\\[|\\\]/g, "")
      .replace(/\\left|\\right/g, "")
      .replace(/\\langle|\\rangle/g, "")
      .replace(/\\(?:,|;|!|quad|qquad)/g, " ")
      .replace(/\\cdot|\\times/g, "*");
    if (/\\mathcal\s*\{\s*D\s*\}/.test(text)) {
      fail(field, "Enter the jump operator only; choose local or collective dissipation with the channel selector.");
    }
    if (/\\dagger|\\mathrm\s*\{\s*H\s*\}/.test(text)) {
      fail(field, "Adjoints are not part of the small supported grammar. Write the resulting spin expression explicitly.");
    }
    for (const command of ["\\mathrm", "\\mathsf", "\\mathbf"]) {
      text = expandWrapper(text, command, field);
    }
    for (const command of ["\\dfrac", "\\tfrac", "\\frac"]) {
      text = expandGroupedCommand(
        text, command, 2,
        (groups) => `((${groups[0]})/(${groups[1]}))`,
        field,
      );
    }
    text = expandGroupedCommand(text, "\\sqrt", 1, (groups) => `sqrt(${groups[0]})`, field);
    text = replaceSummedOperators(text);
    const site = String.raw`(?:\s*\^\s*\{\s*\(\s*i\s*\)\s*\})?`;
    text = text.replace(
      new RegExp(String.raw`J\s*_\s*(?:\{\s*([xyz+\-])\s*\}|([xyz+\-]))`, "g"),
      (_, braced, plain) => ` J${axisName(braced || plain)} `,
    );
    text = text.replace(
      new RegExp(String.raw`j\s*_\s*(?:\{\s*([xyz+\-])\s*\}|([xyz+\-]))${site}`, "g"),
      (_, braced, plain) => ` j${axisName(braced || plain)} `,
    );
    text = text.replace(
      new RegExp(String.raw`\\sigma\s*_\s*(?:\{\s*([xyz+\-])\s*\}|([xyz+\-]))${site}`, "g"),
      (_, braced, plain) => ` sigma${axisName(braced || plain)} `,
    );
    text = replaceLatexMacros(text, field);
    text = text
      .replace(/\^\s*\{\s*([0-9]+)\s*\}/g, "^$1")
      .replace(/[{}]/g, (brace) => brace === "{" ? "(" : ")");
    if (text.includes("\\")) {
      const command = text.match(/\\[A-Za-z]+|\\/);
      fail(field, `Unsupported LaTeX fragment ${command ? command[0] : "\\"}.`);
    }
    return text;
  }

  function token(type, value, position) {
    return { type, value, position };
  }

  function tokenize(normalized, field) {
    const raw = [];
    let index = 0;
    while (index < normalized.length) {
      const character = normalized[index];
      if (/\s/.test(character)) {
        index += 1;
        continue;
      }
      const number = normalized.slice(index).match(/^(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+\-]?\d+)?/);
      if (number) {
        const value = number[0].startsWith(".") ? `0${number[0]}` : number[0];
        const numericValue = Number(value);
        const mantissa = value.split(/[eE]/, 1)[0];
        if (!Number.isFinite(numericValue)) {
          fail(field, `Numeric literal ${value} is outside the finite Float64 range.`);
        }
        if (numericValue === 0 && /[1-9]/.test(mantissa)) {
          fail(field, `Numeric literal ${value} is outside the nonzero Float64 range.`);
        }
        raw.push(token("number", value, index));
        index += number[0].length;
        continue;
      }
      const identifier = normalized.slice(index).match(/^[A-Za-z_][A-Za-z0-9_]*/);
      if (identifier) {
        const name = identifier[0];
        const canonical = OPERATOR_ATOMS[name] || name === "N" || name === "pi" || name === "sqrt"
          ? name
          : safeIdentifier(name);
        raw.push(token("id", canonical, index));
        index += name.length;
        continue;
      }
      if ("+-*/^()".includes(character)) {
        raw.push(token(character === "(" || character === ")" ? "paren" : "op", character, index));
        index += 1;
        continue;
      }
      fail(field, `Unsupported character '${character}' near position ${index + 1}.`);
    }
    const result = [];
    function endsValue(entry) {
      return entry.type === "number" || entry.type === "id" ||
        (entry.type === "paren" && entry.value === ")");
    }
    function startsValue(entry) {
      return entry.type === "number" || entry.type === "id" ||
        (entry.type === "paren" && entry.value === "(");
    }
    for (const current of raw) {
      const previous = result[result.length - 1];
      const isFunctionCall = previous && previous.type === "id" &&
        previous.value === "sqrt" && current.type === "paren" && current.value === "(";
      if (previous && endsValue(previous) && startsValue(current) && !isFunctionCall) {
        result.push(token("op", "*", current.position));
      }
      result.push(current);
    }
    result.push(token("eof", "", normalized.length));
    return result;
  }

  function parseFormula(source, field) {
    const normalized = normalizeLatex(source, field);
    const tokens = tokenize(normalized, field);
    let cursor = 0;
    function peek() {
      return tokens[cursor];
    }
    function take() {
      const value = tokens[cursor];
      cursor += 1;
      return value;
    }
    function primary() {
      const current = take();
      if (current.type === "number") return { type: "number", value: current.value };
      if (current.type === "id") {
        if (current.value === "sqrt") {
          const opening = take();
          if (opening.type !== "paren" || opening.value !== "(") {
            fail(field, "sqrt must be followed by parentheses.");
          }
          const argument = sum();
          const closing = take();
          if (closing.type !== "paren" || closing.value !== ")") {
            fail(field, "Missing closing parenthesis after sqrt.");
          }
          return { type: "call", name: "sqrt", argument };
        }
        return { type: "id", name: current.value };
      }
      if (current.type === "paren" && current.value === "(") {
        const value = sum();
        const closing = take();
        if (closing.type !== "paren" || closing.value !== ")") {
          fail(field, "Missing closing parenthesis.");
        }
        return value;
      }
      fail(field, `Expected a number, parameter, or supported spin operator near position ${current.position + 1}.`);
    }

    // Mathematical precedence: exponentiation binds more tightly than unary
    // signs. Thus `-J_x^2` means `-(J_x^2)`, while `(-J_x)^2` retains the
    // explicitly parenthesized sign. The recursive exponent keeps `a^b^c`
    // right associative.
    function power() {
      const left = primary();
      if (peek().type === "op" && peek().value === "^") {
        take();
        return { type: "binary", op: "^", left, right: unary() };
      }
      return left;
    }

    function unary() {
      if (peek().type === "op" &&
          (peek().value === "+" || peek().value === "-")) {
        const operator = take().value;
        return { type: "unary", op: operator, argument: unary() };
      }
      return power();
    }

    function product() {
      let left = unary();
      while (peek().type === "op" &&
             (peek().value === "*" || peek().value === "/")) {
        const operator = take().value;
        left = { type: "binary", op: operator, left, right: unary() };
      }
      return left;
    }

    function sum() {
      let left = product();
      while (peek().type === "op" &&
             (peek().value === "+" || peek().value === "-")) {
        const operator = take().value;
        left = { type: "binary", op: operator, left, right: product() };
      }
      return left;
    }

    const ast = sum();
    if (peek().type !== "eof") {
      fail(field, `Unexpected token '${peek().value}' near position ${peek().position + 1}.`);
    }
    return { ast, normalized };
  }

  function mergeSets(first, second) {
    return new Set([...first, ...second]);
  }

  function analysis(kind, linear, families, components) {
    return { kind, linear, families, components };
  }

  function integerLiteral(node) {
    if (node.type !== "number") return null;
    const value = Number(node.value);
    return Number.isInteger(value) ? value : null;
  }

  function analyze(node, field) {
    if (node.type === "number") return analysis("scalar", true, new Set(), new Set());
    if (node.type === "id") {
      const atom = OPERATOR_ATOMS[node.name];
      if (!atom) return analysis("scalar", true, new Set(), new Set());
      return analysis("operator", true, new Set([atom.family]), new Set([atom.component]));
    }
    if (node.type === "unary") return analyze(node.argument, field);
    if (node.type === "call") {
      const argument = analyze(node.argument, field);
      if (argument.kind !== "scalar") fail(field, "sqrt may only act on a scalar coefficient.");
      return argument;
    }
    const left = analyze(node.left, field);
    const right = analyze(node.right, field);
    if (node.op === "+" || node.op === "-") {
      if (left.kind !== right.kind) {
        fail(field, "Cannot add a scalar and an operator.");
      }
      return analysis(
        left.kind,
        left.linear && right.linear,
        mergeSets(left.families, right.families),
        mergeSets(left.components, right.components),
      );
    }
    if (node.op === "*") {
      if (left.kind === "scalar" && right.kind === "scalar") {
        return analysis("scalar", true, new Set(), new Set());
      }
      if (left.kind === "operator" && right.kind === "operator") {
        return analysis(
          "operator", false,
          mergeSets(left.families, right.families),
          mergeSets(left.components, right.components),
        );
      }
      const operator = left.kind === "operator" ? left : right;
      return analysis("operator", operator.linear, operator.families, operator.components);
    }
    if (node.op === "/") {
      if (right.kind !== "scalar") fail(field, "Division by an operator is unsupported.");
      return left;
    }
    if (node.op === "^") {
      if (right.kind !== "scalar") fail(field, "The exponent must be a scalar integer.");
      const exponent = integerLiteral(node.right);
      if (exponent === null || exponent < 0) {
        fail(field, "Only nonnegative integer powers are supported.");
      }
      if (left.kind === "operator" && exponent !== 1 && exponent !== 2) {
        fail(field, "Operator powers must be one or two in the supported grammar.");
      }
      return analysis(
        left.kind,
        left.kind === "operator" ? exponent <= 1 && left.linear : true,
        left.families,
        left.components,
      );
    }
    fail(field, `Unsupported operation ${node.op}.`);
  }

  function isCollectiveFamily(family) {
    return family === "collective-spin" || family === "collective-pauli";
  }

  function isLocalFamily(family) {
    return family === "local-spin" || family === "local-pauli";
  }

  function requireFamilies(info, predicate, field, suggestion) {
    for (const family of info.families) {
      if (!predicate(family)) fail(field, suggestion);
    }
  }

  function emit(node, context) {
    if (node.type === "number") return floatLiteral(Number(node.value));
    if (node.type === "id") {
      const atom = OPERATOR_ATOMS[node.name];
      if (!atom) return node.name === "N" && context === "scalar" ?
        "Float64(N)" : node.name;
      if (context === "local") return atom.local;
      if (context === "collective" && atom.pi) return atom.pi;
      if (context === "observable-collective" && atom.pi) {
        return atom.pi.replace(/\bJ([xyzpm])\b/g, "observable_J$1");
      }
      throw new Error(`Internal code-generation context mismatch for ${node.name}.`);
    }
    if (node.type === "unary") {
      const argumentContext = analyze(node.argument, "internal").kind === "scalar" ?
        "scalar" : context;
      return `(${node.op}${emit(node.argument, argumentContext)})`;
    }
    if (node.type === "call") return `sqrt(${emit(node.argument, "scalar")})`;
    if (node.op === "/" && analyze(node.left, "internal").kind === "operator") {
      return `((1.0 / ${emit(node.right, "scalar")}) * ${emit(node.left, context)})`;
    }
    if (node.op === "^") {
      const exponent = integerLiteral(node.right);
      const leftIsOperator = analyze(node.left, "internal").kind === "operator";
      if (exponent === 1 && leftIsOperator) {
        return emit(node.left, context);
      }
      if (exponent === 2 && leftIsOperator) {
        const value = emit(node.left, context);
        return `(${value} * ${value})`;
      }
      if (exponent !== null) {
        const leftContext = analyze(node.left, "internal").kind === "scalar" ?
          "scalar" : context;
        return `(${emit(node.left, leftContext)} ^ ${exponent})`;
      }
    }
    const leftContext = analyze(node.left, "internal").kind === "scalar" ?
      "scalar" : context;
    const rightContext = analyze(node.right, "internal").kind === "scalar" ?
      "scalar" : context;
    return `(${emit(node.left, leftContext)} ${node.op} ${emit(node.right, rightContext)})`;
  }

  function visit(node, callback) {
    callback(node);
    if (node.type === "unary") visit(node.argument, callback);
    if (node.type === "call") visit(node.argument, callback);
    if (node.type === "binary") {
      visit(node.left, callback);
      visit(node.right, callback);
    }
  }

  function collectParameters(node, destination) {
    visit(node, (entry) => {
      if (entry.type === "id" && !OPERATOR_ATOMS[entry.name] &&
          entry.name !== "N" && entry.name !== "pi") {
        destination.add(entry.name);
      }
    });
  }

  function collectComponents(node, destination) {
    visit(node, (entry) => {
      if (entry.type !== "id") return;
      const atom = OPERATOR_ATOMS[entry.name];
      if (atom && isCollectiveFamily(atom.family)) destination.add(atom.component);
    });
  }

  function evaluateScalar(node, values, particleCount) {
    if (node.type === "number") return Number(node.value);
    if (node.type === "id") {
      if (node.name === "N") return particleCount;
      if (node.name === "pi") return Math.PI;
      return values.has(node.name) ? values.get(node.name) : null;
    }
    if (node.type === "unary") {
      const value = evaluateScalar(node.argument, values, particleCount);
      if (value === null) return null;
      return node.op === "-" ? -value : value;
    }
    if (node.type === "call") {
      const value = evaluateScalar(node.argument, values, particleCount);
      return value === null ? null : Math.sqrt(value);
    }
    const left = evaluateScalar(node.left, values, particleCount);
    const right = evaluateScalar(node.right, values, particleCount);
    if (left === null || right === null) return null;
    if (node.op === "+") return left + right;
    if (node.op === "-") return left - right;
    if (node.op === "*") return left * right;
    if (node.op === "/") return left / right;
    if (node.op === "^") return left ** right;
    return null;
  }

  function validateScalarSubexpressions(node, values, particleCount, field) {
    visit(node, (entry) => {
      if (analyze(entry, field).kind !== "scalar") return;
      const value = evaluateScalar(entry, values, particleCount);
      if (value === null || !Number.isFinite(value)) {
        fail(field, "A scalar subexpression is not finite with the supplied parameter values.");
      }
    });
  }

  function parseParameterValues(source) {
    const values = new Map();
    const code = new Map();
    const provided = new Set();
    const lines = String(source || "").split(/\r?\n/);
    for (let index = 0; index < lines.length; index += 1) {
      const line = lines[index].trim();
      if (!line || line.startsWith("#")) continue;
      const equals = line.indexOf("=");
      if (equals < 1 || line.indexOf("=", equals + 1) !== -1) {
        fail("parameters", `Parameter line ${index + 1} must have the form name = numeric_value.`);
      }
      const left = parseFormula(line.slice(0, equals), "parameters");
      if (left.ast.type !== "id" || OPERATOR_ATOMS[left.ast.name] ||
          left.ast.name === "N" || left.ast.name === "pi") {
        fail("parameters", `Left side of parameter line ${index + 1} must be one scalar name.`);
      }
      const right = parseFormula(line.slice(equals + 1), "parameters");
      const rightInfo = analyze(right.ast, "parameters");
      if (rightInfo.kind !== "scalar") {
        fail("parameters", `Value on parameter line ${index + 1} must be scalar.`);
      }
      let namedValue = null;
      visit(right.ast, (entry) => {
        if (entry.type === "id" && entry.name !== "pi") namedValue = entry.name;
      });
      if (namedValue !== null) {
        fail("parameters", `Value on parameter line ${index + 1} must be a numeric constant and cannot depend on ${namedValue}.`);
      }
      const value = evaluateScalar(right.ast, new Map(), 1);
      if (value === null || !Number.isFinite(value)) {
        fail("parameters", `Value on parameter line ${index + 1} must be a finite numeric constant.`);
      }
      if (values.has(left.ast.name)) {
        fail("parameters", `Parameter ${left.ast.name} is defined more than once.`);
      }
      values.set(left.ast.name, value);
      code.set(left.ast.name, floatLiteral(value));
      provided.add(left.ast.name);
    }
    return { values, code, provided };
  }

  function flattenAdditive(node, sign, destination) {
    if (node.type === "binary" && node.op === "+") {
      flattenAdditive(node.left, sign, destination);
      flattenAdditive(node.right, sign, destination);
    } else if (node.type === "binary" && node.op === "-") {
      flattenAdditive(node.left, sign, destination);
      flattenAdditive(node.right, -sign, destination);
    } else if (node.type === "unary" && node.op === "-") {
      flattenAdditive(node.argument, -sign, destination);
    } else {
      destination.push({ node, sign });
    }
  }

  function signedExpression(entries, context) {
    if (!entries.length) return null;
    return entries.map((entry, index) => {
      const expression = emit(entry.node, context);
      if (index === 0) return entry.sign < 0 ? `(-${expression})` : expression;
      return entry.sign < 0 ? ` - ${expression}` : ` + ${expression}`;
    }).join("");
  }

  function parseModel(config) {
    const particleCount = Number(config.N);
    const localDimension = Number(config.d);
    if (!Number.isSafeInteger(particleCount) || particleCount < 1) {
      fail("N", "N must be a positive integer.");
    }
    if (!Number.isSafeInteger(localDimension) || localDimension < 2) {
      fail("d", "The local dimension d must be an integer of at least two.");
    }
    const architectureAliases = new Map([
      ["pi", "pi"],
      ["local-pseudomode", "local-pseudomode"],
      ["local_pseudomode", "local-pseudomode"],
      ["global-pseudomode", "global-pseudomode"],
      ["global_pseudomode", "global-pseudomode"],
    ]);
    const requestedArchitecture = config.architecture === undefined
      ? "pi"
      : String(config.architecture);
    const architecture = architectureAliases.get(requestedArchitecture);
    if (!architecture) {
      fail(
        "architecture",
        "Choose an ordinary PI ensemble, identical local pseudomodes, or one shared global pseudomode.",
      );
    }
    const calculationAliases = new Map([
      ["steady", "steady-state"],
      ["steady-state", "steady-state"],
      ["steady_state", "steady-state"],
      ["expectation", "steady-observable"],
      ["steady-observable", "steady-observable"],
      ["steady_observable", "steady-observable"],
      ["dynamics", "dynamics-observable"],
      ["transient", "dynamics-observable"],
      ["transient-observable", "dynamics-observable"],
      ["dynamics-observable", "dynamics-observable"],
      ["dynamics_observable", "dynamics-observable"],
      ["spectrum", "liouvillian-spectrum"],
      ["liouvillian-spectrum", "liouvillian-spectrum"],
      ["liouvillian_spectrum", "liouvillian-spectrum"],
      ["gap", "liouvillian-gap"],
      ["liouvillian-gap", "liouvillian-gap"],
      ["liouvillian_gap", "liouvillian-gap"],
    ]);
    const requestedCalculation = config.calculation === undefined
      ? (config.target === undefined ? "expectation" : String(config.target))
      : String(config.calculation);
    const calculation = calculationAliases.get(requestedCalculation);
    if (!calculation) {
      fail(
        "calculation",
        "Choose a stationary state, stationary observable, observable dynamics, selected Liouvillian spectrum, or Liouvillian gap calculation.",
      );
    }
    const isStationary =
      calculation === "steady-state" || calculation === "steady-observable";
    const isDynamics = calculation === "dynamics-observable";
    const isSpectral =
      calculation === "liouvillian-spectrum" ||
      calculation === "liouvillian-gap";
    const target = calculation === "steady-state" ? "steady" :
      calculation === "steady-observable" ? "expectation" : calculation;
    const methodAliases = new Map([
      ["deterministic", "deterministic"],
      ["trajectory", "trajectory"],
      ["quantum-trajectories", "trajectory"],
      ["quantum_trajectories", "trajectory"],
    ]);
    const requestedSteadyMethod = config.steadyMethod === undefined
      ? "deterministic"
      : String(config.steadyMethod);
    const steadyMethod = methodAliases.get(requestedSteadyMethod);
    if (!steadyMethod) {
      fail(
        "steady-state method",
        "Choose the deterministic stationary solver or the quantum-trajectory estimator.",
      );
    }
    const workflowAliases = new Map([
      ["direct-api", "direct-api"],
      ["direct", "direct-api"],
      ["verified-experiment", "verified-experiment"],
      ["experiment", "verified-experiment"],
      ["verified", "verified-experiment"],
    ]);
    const requestedWorkflow = config.workflow === undefined
      ? "direct-api"
      : String(config.workflow);
    const workflow = workflowAliases.get(requestedWorkflow);
    if (!workflow) {
      fail(
        "workflow",
        "Choose the direct high-level API or the typed verified PIExperiment workflow.",
      );
    }
    if (isSpectral && steadyMethod !== "deterministic") {
      fail(
        "calculation method",
        "Liouvillian spectrum and gap calculations use deterministic prepared operators, not quantum trajectories.",
      );
    }
    if (
      isStationary &&
      steadyMethod === "trajectory" &&
      architecture === "global-pseudomode"
    ) {
      fail(
        "steady-state method",
        "The streaming trajectory steady-state estimator supports ordinary PI models and identical local-pseudomode supersites, not one shared global pseudomode.",
      );
    }
    if (
      isDynamics &&
      steadyMethod === "deterministic" &&
      architecture === "global-pseudomode"
    ) {
      fail(
        "calculation method",
        "Shared-mode deterministic dynamics currently lacks a memory-guarded state-free observable streamer. Select quantum trajectories, or use the documented advanced composite time-evolution API explicitly.",
      );
    }
    if (
      workflow === "verified-experiment" &&
      (steadyMethod !== "deterministic" ||
       isSpectral ||
       architecture === "global-pseudomode")
    ) {
      fail(
        "workflow",
        "The typed PIExperiment route currently supports deterministic stationary states and deterministic dynamics on ordinary PI or identical-local-pseudomode models. Use the direct API for trajectories, spectra, gaps, and a shared global mode.",
      );
    }

    function numericSetting(input, key, fallback, field, predicate, requirement) {
      const raw = input[key];
      const value = Number(
        raw === undefined || String(raw).trim() === "" ? fallback : raw,
      );
      if (!Number.isFinite(value) || !predicate(value)) {
        fail(field, requirement);
      }
      return value;
    }

    const trajectoryInput =
      config.trajectory && typeof config.trajectory === "object"
        ? config.trajectory
        : {};
    const initialInput =
      config.initialState && typeof config.initialState === "object"
        ? config.initialState
        : {};
    const initialKind = initialInput.kind === undefined
      ? "computational-product"
      : String(initialInput.kind);
    if (
      (isDynamics || steadyMethod === "trajectory") &&
      initialKind !== "computational-product"
    ) {
      fail(
        "initial state",
        "The assistant currently emits an explicit computational product state. Edit the generated typed PIState manually for a different preparation.",
      );
    }
    const initialLevelRaw = initialInput.level === undefined
      ? trajectoryInput.initialLevel
      : initialInput.level;
    const initialState = (isDynamics || steadyMethod === "trajectory")
      ? {
        kind: "computational-product",
        level: numericSetting(
          { level: initialLevelRaw },
          "level", 1, "initial local level",
          (value) =>
            Number.isSafeInteger(value) &&
            value >= 1 &&
            value <= localDimension,
          `The initial local level must be a one-based index in 1:${localDimension}.`,
        ),
      }
      : null;

    let trajectory = null;
    if (steadyMethod === "trajectory") {
      if (
        trajectoryInput.algorithm !== undefined &&
        String(trajectoryInput.algorithm) !== "fixed"
      ) {
        fail(
          "trajectory algorithm",
          "Generated trajectory calculations use the fixed, preallocated backend. Adaptive/event controls are not available for every supported architecture.",
        );
      }
      const trajectories = numericSetting(
        trajectoryInput, "trajectories", 512, "trajectories",
        (value) => Number.isSafeInteger(value) && value >= 2,
        "The number of trajectories must be a safe integer of at least two.",
      );
      const dt = numericSetting(
        trajectoryInput, "dt", 0.002, "trajectory dt",
        (value) => value > 0,
        "The trajectory time step must be finite and positive.",
      );
      const maxJumpProbability = numericSetting(
        trajectoryInput, "maxJumpProbability", 0.02,
        "maximum jump probability",
        (value) => value > 0 && value < 1,
        "The maximum jump probability must lie strictly between zero and one.",
      );
      const seed = numericSetting(
        trajectoryInput, "seed", 2026, "trajectory seed",
        (value) => Number.isSafeInteger(value) && value >= 0,
        "The trajectory seed must be a safe nonnegative integer.",
      );
      trajectory = {
        trajectories,
        dt,
        maxJumpProbability,
        seed,
      };
      if (isStationary) {
        trajectory.settlingTime = numericSetting(
          trajectoryInput, "settlingTime", 50, "settling time",
          (value) => value > 0,
          "The settling time must be finite and positive.",
        );
        trajectory.samplesPerTrajectory = numericSetting(
          trajectoryInput, "samplesPerTrajectory", 5,
          "samples per trajectory",
          (value) => Number.isSafeInteger(value) && value >= 1,
          "Samples per trajectory must be a safe positive integer.",
        );
        trajectory.samplingInterval = numericSetting(
          trajectoryInput, "samplingInterval", 2, "sampling interval",
          (value) => value > 0,
          "The sampling interval must be finite and positive.",
        );
      }
    }

    let dynamics = null;
    if (isDynamics) {
      const input = config.dynamics && typeof config.dynamics === "object"
        ? config.dynamics
        : {};
      if (input.saveStates === true) {
        fail(
          "dynamics output",
          "The assistant emits state-free observable dynamics to keep output memory bounded. Use the documented advanced API when a full state history is intentional.",
        );
      }
      const startTime = numericSetting(
        input, "startTime", 0, "dynamics start time",
        () => true,
        "The dynamics start time must be finite.",
      );
      const finalTime = numericSetting(
        input, "finalTime", 10, "dynamics final time",
        (value) => value > startTime,
        "The dynamics final time must be finite and larger than the start time.",
      );
      const samples = numericSetting(
        input, "samples", 101, "dynamics samples",
        (value) => Number.isSafeInteger(value) && value >= 2,
        "The number of output samples must be a safe integer of at least two.",
      );
      const stepsPerInterval = numericSetting(
        input, "stepsPerInterval", 16, "steps per output interval",
        (value) => Number.isSafeInteger(value) && value >= 1,
        "Steps per output interval must be a safe positive integer.",
      );
      dynamics = { startTime, finalTime, samples, stepsPerInterval };
    }

    let spectrum = null;
    if (calculation === "liouvillian-spectrum") {
      const input = config.spectrum && typeof config.spectrum === "object"
        ? config.spectrum
        : {};
      if (input.vectors === true) {
        fail(
          "spectrum vectors",
          "The assistant omits eigenvectors to keep selected-spectrum output bounded. Request vectors explicitly through the advanced spectral API.",
        );
      }
      const targetAliases = new Map([
        ["largest-real", "largest_real"],
        ["largest_real", "largest_real"],
        ["near-zero", "near_zero"],
        ["near_zero", "near_zero"],
        ["largest-magnitude", "largest_magnitude"],
        ["largest_magnitude", "largest_magnitude"],
      ]);
      const requestedTarget = input.target === undefined
        ? "largest-real"
        : String(input.target);
      const spectrumTarget = targetAliases.get(requestedTarget);
      if (!spectrumTarget) {
        fail(
          "spectrum target",
          "Choose largest real part, near zero, or largest magnitude.",
        );
      }
      spectrum = {
        target: spectrumTarget,
        nev: numericSetting(
          input, "nev", 6, "spectrum eigenvalues",
          (value) => Number.isSafeInteger(value) && value >= 1,
          "The requested eigenvalue count must be a safe positive integer.",
        ),
        seed: numericSetting(
          input, "seed", 2026, "spectrum seed",
          (value) => Number.isSafeInteger(value) && value >= 0,
          "The spectrum seed must be a safe nonnegative integer.",
        ),
      };
    }

    let gap = null;
    if (calculation === "liouvillian-gap") {
      const input = config.gap && typeof config.gap === "object"
        ? config.gap
        : {};
      gap = {
        nev: numericSetting(
          input, "nev", 8, "gap eigenvalues",
          (value) => Number.isSafeInteger(value) && value >= 2,
          "The gap calculation needs a safe integer of at least two Ritz values.",
        ),
        krylovdim: numericSetting(
          input, "krylovdim", 32, "gap Krylov dimension",
          (value) => Number.isSafeInteger(value) && value >= 3,
          "The gap Krylov dimension must be a safe integer of at least three.",
        ),
      };
      if (gap.krylovdim <= gap.nev) {
        fail(
          "gap Krylov dimension",
          "The gap Krylov dimension must be larger than the requested Ritz count.",
        );
      }
    }

    const resourceInput =
      config.resources && typeof config.resources === "object"
        ? config.resources
        : {};
    const memoryBudgetMiB = numericSetting(
      resourceInput, "memoryBudgetMiB", 512, "memory budget",
      (value) => Number.isSafeInteger(value) && value >= 1,
      "The memory budget must be a safe positive integer number of MiB.",
    );

    const analysisInput =
      config.analysis && typeof config.analysis === "object"
        ? config.analysis
        : {};
    const qfiAxes = new Set(["none", "x", "y", "z"]);
    const qfiAxis = analysisInput.qfiAxis === undefined
      ? "none"
      : String(analysisInput.qfiAxis);
    if (!qfiAxes.has(qfiAxis)) {
      fail("QFI axis", "Choose none, x, y, or z for the collective QFI generator.");
    }
    const analysis = {
      purity: analysisInput.purity === true,
      entropy: analysisInput.entropy === true,
      oneBodyRDM: analysisInput.oneBodyRDM === true,
      qfiAxis,
    };
    if (
      !isStationary &&
      (analysis.purity ||
       analysis.entropy ||
       analysis.oneBodyRDM ||
       analysis.qfiAxis !== "none")
    ) {
      fail(
        "state analysis",
        "Purity, entropy, one-body density matrices, and QFI require a stationary density-operator calculation.",
      );
    }

    const parameters = new Set();
    const hamiltonianComponents = new Set();
    const observableComponents = new Set();
    const warnings = [];
    if (isStationary && steadyMethod === "trajectory") {
      warnings.push(
        "A quantum-trajectory stationary state is a statistical estimate, not a convergence or uniqueness certificate.",
        "Converge the settling time, time step, sampling window, and independent path count separately.",
        "Samples from one path can be correlated; uncertainty is therefore estimated across independent path means.",
        "The generated computational product initial state can select a symmetry sector or stationary component. Repeat from other PI initial states when strong symmetries or multiple stationary states are possible.",
      );
      if (
        analysis.purity ||
        analysis.entropy ||
        analysis.qfiAxis !== "none"
      ) {
        warnings.push(
          "Purity, entropy, and QFI are nonlinear plug-in functionals of the trajectory-averaged state. Converge them against path count or independent ensembles; the reported state standard error is not their uncertainty bar.",
        );
      }
    } else if (isDynamics && steadyMethod === "trajectory") {
      warnings.push(
        "Trajectory curves contain both integration and Monte Carlo error. Converge the time step and independent path count separately.",
        "The generated route streams observable means and confidence intervals without retaining state histories.",
      );
    } else if (isStationary) {
      warnings.push(
        "The generated code checks convergence and state validity, but it does not certify uniqueness of the stationary state.",
      );
    } else if (calculation === "liouvillian-spectrum") {
      warnings.push(
        "A selected spectrum is partial unless the solver metadata proves completeness. Do not infer a certified global Liouvillian gap from these values alone.",
      );
    } else if (calculation === "liouvillian-gap") {
      warnings.push(
        "The iterative gap route reports certification flags. Treat the numerical gap as certified only when gap_certified is true.",
      );
    }
    if (workflow === "verified-experiment") {
      warnings.push(
        "The typed experiment route records an explainable resource plan, verification report, and deterministic provenance digest.",
      );
      if (isDynamics) {
        warnings.push(
          "Verified dynamics retains every requested PI state so physical-state checks can be performed. Reduce the output grid or use the direct state-free route when that history does not fit the declared memory budget.",
        );
      }
    }
    if (isStationary && architecture === "global-pseudomode") {
      warnings.push(
        "The generated shared-mode validation checks the full composite trace and Hermiticity plus the reduced system state; it does not certify positivity of the complete composite density operator.",
      );
    }

    let pseudomode = null;
    if (architecture !== "pi") {
      const input = config.pseudomode && typeof config.pseudomode === "object"
        ? config.pseudomode
        : {};
      const cutoff = Number(input.nmax === undefined ? 1 : input.nmax);
      if (!Number.isSafeInteger(cutoff) || cutoff < 0) {
        fail("pseudomode cutoff", "The pseudomode cutoff nmax must be a nonnegative integer.");
      }
      function scalarField(value, fallback, field) {
        const parsed = parseFormula(
          String(value === undefined || String(value).trim() === "" ? fallback : value),
          field,
        );
        if (analyze(parsed.ast, field).kind !== "scalar") {
          fail(field, "This pseudomode setting must be a real scalar expression.");
        }
        collectParameters(parsed.ast, parameters);
        return parsed;
      }
      const frequency = scalarField(input.frequency, "1", "pseudomode frequency");
      const damping = scalarField(input.damping, "0.1", "pseudomode damping");
      const thermalOccupation = scalarField(
        input.thermalOccupation, "0", "pseudomode thermal occupation",
      );
      const strength = scalarField(
        input.couplingStrength, "0.1", "pseudomode coupling strength",
      );
      const counterrotatingStrength = scalarField(
        input.counterrotatingStrength, "0",
        "pseudomode counter-rotating strength",
      );
      const couplingOperator = parseFormula(
        String(input.couplingOperator || String.raw`\sigma_-`),
        "pseudomode coupling operator",
      );
      const couplingInfo = analyze(
        couplingOperator.ast, "pseudomode coupling operator",
      );
      if (couplingInfo.kind !== "operator" || !couplingInfo.linear) {
        fail(
          "pseudomode coupling operator",
          "The coupling seed must be one linear one-site spin operator.",
        );
      }
      requireFamilies(
        couplingInfo, isLocalFamily, "pseudomode coupling operator",
        "Use a one-site j_a or sigma_a coupling seed. The embedding performs the collective or supersite lift.",
      );
      if (localDimension !== 2 &&
          [...couplingInfo.families].some((family) => family.includes("pauli"))) {
        fail(
          "pseudomode coupling operator",
          "Pauli sigma symbols require d = 2. Use a local spin-j symbol for qudits.",
        );
      }
      collectParameters(couplingOperator.ast, parameters);
      pseudomode = {
        nmax: cutoff,
        frequency,
        damping,
        thermalOccupation,
        couplingOperator,
        strength,
        counterrotatingStrength,
      };
      if (cutoff === 0) {
        warnings.push("nmax = 0 removes every pseudomode excitation; increase the cutoff for a nontrivial embedding.");
      }
      if (architecture === "local-pseudomode") {
        const exactSupersiteDimension =
          BigInt(localDimension) * (BigInt(cutoff) + 1n);
        warnings.push(
          `Each system has its own mode. The supersite dimension is d*(nmax+1) = ${exactSupersiteDimension}, so converge the cutoff and inspect the memory preflight.`,
        );
      } else {
        warnings.push(
          "The generated mode is one oscillator shared by the ensemble. This is physically different from one independent mode per system.",
        );
      }
      warnings.push(
        "The pseudomode cutoff is an approximation. Converge it and monitor the generated top-level population.",
      );
      warnings.push(
        "With the package dissipator convention, pseudomode damping kappa makes the free mode amplitude decay at kappa/2.",
      );
    }

    let hamiltonian = null;
    const hamiltonianText = String(config.hamiltonian || "").trim();
    if (hamiltonianText) {
      hamiltonian = parseFormula(hamiltonianText, "hamiltonian");
      const info = analyze(hamiltonian.ast, "hamiltonian");
      if (info.kind !== "operator") fail("hamiltonian", "The Hamiltonian must contain a supported spin operator.");
      requireFamilies(
        info, isCollectiveFamily, "hamiltonian",
        "A PI Hamiltonian must use J_a or an explicit sum_i sigma_a^(i), not an unsummed one-site operator.",
      );
      if (localDimension !== 2 && [...info.families].includes("collective-pauli")) {
        fail("hamiltonian", "Pauli sigma symbols require d = 2. Use spin-j symbols J_a for qudits.");
      }
      collectParameters(hamiltonian.ast, parameters);
      const hamiltonianEntries = [];
      flattenAdditive(hamiltonian.ast, 1, hamiltonianEntries);
      for (const entry of hamiltonianEntries) {
        if (!analyze(entry.node, "hamiltonian").linear) {
          collectComponents(entry.node, hamiltonianComponents);
        }
      }
      if (info.components.has("plus") || info.components.has("minus")) {
        warnings.push("The Hamiltonian contains raising/lowering operators; DirectPIHamiltonian will reject it unless the complete expression is Hermitian.");
      }
    }

    const jumps = [];
    const inputJumps = Array.isArray(config.jumps) ? config.jumps : [];
    for (let index = 0; index < inputJumps.length; index += 1) {
      const input = inputJumps[index];
      if (!input || !String(input.operator || "").trim()) continue;
      const field = `jump ${index + 1}`;
      if (input.kind !== "local" && input.kind !== "collective") {
        fail(
          `${field} kind`,
          "Choose explicitly between an independent local channel and one collective channel.",
        );
      }
      const kind = input.kind;
      const operator = parseFormula(String(input.operator), field);
      const operatorInfo = analyze(operator.ast, field);
      if (operatorInfo.kind !== "operator" || !operatorInfo.linear) {
        fail(field, "A jump must be a one-body spin operator or a linear combination of such operators.");
      }
      if (kind === "local") {
        requireFamilies(
          operatorInfo, isLocalFamily, field,
          "An independent local channel must use j_a or sigma_a. Use the collective selector for J_a.",
        );
      } else {
        requireFamilies(
          operatorInfo, isCollectiveFamily, field,
          "A collective channel must use J_a or sum_i sigma_a^(i).",
        );
      }
      if (localDimension !== 2 &&
          [...operatorInfo.families].some((family) => family.includes("pauli"))) {
        fail(field, "Pauli sigma symbols require d = 2. Use local j_a or collective J_a for qudits.");
      }
      const rawRate =
        input.rate === undefined || String(input.rate).trim() === ""
          ? "1"
          : String(input.rate);
      const rate = parseFormula(rawRate, `${field} rate`);
      const rateInfo = analyze(rate.ast, `${field} rate`);
      if (rateInfo.kind !== "scalar") fail(`${field} rate`, "A jump rate must be scalar.");
      const operatorParameters = new Set();
      collectParameters(operator.ast, operatorParameters);
      collectParameters(operator.ast, parameters);
      collectParameters(rate.ast, parameters);
      jumps.push({ kind, operator, rate, field, operatorParameters });
    }

    if (architecture === "pi" && !hamiltonian && !jumps.length) {
      fail("model", "Add a Hamiltonian or at least one dissipative channel.");
    }
    if (isStationary && !jumps.length && architecture === "pi") {
      warnings.push("A Hamiltonian-only generator normally has many stationary states; add dissipation or analyse the stationary subspace.");
    } else if (
      isStationary &&
      architecture === "global-pseudomode" &&
      !jumps.some((jump) => jump.kind === "local")
    ) {
      warnings.push("No independent local system channel was selected. Shared-mode damping and collective coupling preserve system Schur-sector populations, so the stationary state is generally nonunique across sectors.");
    } else if (
      isStationary &&
      architecture === "pi" &&
      jumps.length > 0 &&
      jumps.every((jump) => jump.kind === "collective")
    ) {
      warnings.push("Only collective channels were selected. With the complete PI basis, Schur-sector populations are conserved and the stationary state is generally nonunique.");
    }

    let observable = null;
    let observableInfo = null;
    let observableMode = null;
    if (
      calculation === "steady-observable" ||
      calculation === "dynamics-observable"
    ) {
      observable = parseFormula(String(config.observable || ""), "observable");
      observableInfo = analyze(observable.ast, "observable");
      if (observableInfo.kind !== "operator") fail("observable", "The observable must contain a supported spin operator.");
      const allCollective = [...observableInfo.families].every(isCollectiveFamily);
      const allLocal = [...observableInfo.families].every(isLocalFamily);
      if (allCollective) {
        observableMode = observableInfo.linear ? "collective-plan" : "pi-operator";
        if (observableMode === "pi-operator") {
          collectComponents(observable.ast, observableComponents);
        }
      } else if (allLocal && observableInfo.linear) {
        observableMode = "single-site-plan";
        warnings.push("An unsummed local observable is interpreted as the identical one-site expectation, (1/N) sum_i <a_i>.");
      } else {
        fail("observable", "Use either collective J_a operators or one unsummed local j_a/sigma_a operator, without mixing the conventions.");
      }
      if (localDimension !== 2 &&
          [...observableInfo.families].some((family) => family.includes("pauli"))) {
        fail("observable", "Pauli sigma symbols require d = 2.");
      }
      collectParameters(observable.ast, parameters);
      if (observableInfo.components.has("plus") || observableInfo.components.has("minus")) {
        if (isDynamics && steadyMethod === "trajectory") {
          fail(
            "observable",
            "Trajectory streaming requires a Hermitian observable. Use a real linear combination of x, y, and z components rather than raising/lowering symbols.",
          );
        }
        warnings.push("The selected observable may be non-Hermitian, so its expectation value can be complex.");
      }
      if (
        isDynamics &&
        steadyMethod === "trajectory" &&
        !observableInfo.linear
      ) {
        fail(
          "observable",
          "Trajectory streaming currently accepts only structurally Hermitian linear x/y/z observables. Use deterministic dynamics for a general collective polynomial.",
        );
      }
    }

    const values = parseParameterValues(config.parameters || "");
    for (const parameter of parameters) {
      if (!values.values.has(parameter)) {
        values.values.set(parameter, 1.0);
        values.code.set(parameter, "1.0");
        warnings.push(`Parameter ${parameter} was not assigned; generated code uses 1.0 with a TODO marker.`);
      }
    }
    for (const parameter of values.values.keys()) {
      if (!parameters.has(parameter)) {
        warnings.push(`Parameter ${parameter} is assigned but does not occur in the model.`);
      }
    }
    if (hamiltonian) {
      validateScalarSubexpressions(
        hamiltonian.ast, values.values, particleCount, "hamiltonian");
    }
    for (const jump of jumps) {
      validateScalarSubexpressions(
        jump.operator.ast, values.values, particleCount, jump.field);
      validateScalarSubexpressions(
        jump.rate.ast, values.values, particleCount, `${jump.field} rate`);
      const rate = evaluateScalar(jump.rate.ast, values.values, particleCount);
      if (rate !== null && (!Number.isFinite(rate) || rate < 0)) {
        fail(`${jump.field} rate`, "Jump rates in this assistant must evaluate to a finite nonnegative value.");
      }
      if (jump.operatorParameters.size && rate !== 1) {
        warnings.push(`${jump.field} contains a scalar parameter inside the jump operator and also has a nonunit rate. Confirm that the dissipator should square the internal coefficient.`);
      }
    }
    if (pseudomode) {
      const scalarFields = [
        ["pseudomode frequency", pseudomode.frequency, false],
        ["pseudomode damping", pseudomode.damping, true],
        ["pseudomode thermal occupation", pseudomode.thermalOccupation, true],
        ["pseudomode coupling strength", pseudomode.strength, false],
        [
          "pseudomode counter-rotating strength",
          pseudomode.counterrotatingStrength,
          false,
        ],
      ];
      for (const [field, parsedScalar, nonnegative] of scalarFields) {
        validateScalarSubexpressions(
          parsedScalar.ast, values.values, particleCount, field,
        );
        const value = evaluateScalar(
          parsedScalar.ast, values.values, particleCount,
        );
        if (value === null || !Number.isFinite(value) ||
            (nonnegative && value < 0)) {
          fail(
            field,
            nonnegative
              ? "This setting must evaluate to a finite nonnegative real value."
              : "This setting must evaluate to a finite real value.",
          );
        }
      }
      validateScalarSubexpressions(
        pseudomode.couplingOperator.ast,
        values.values,
        particleCount,
        "pseudomode coupling operator",
      );
    }
    if (observable) {
      validateScalarSubexpressions(
        observable.ast, values.values, particleCount, "observable");
    }
    if (steadyMethod === "trajectory") {
      if (architecture === "pi" && !jumps.length) {
        warnings.push(
          isStationary
            ? "No stochastic jump channel was selected. Hamiltonian-only trajectories do not provide dissipative relaxation to a stationary state."
            : "No stochastic jump channel was selected. Every path follows the same Hamiltonian dynamics, so a trajectory ensemble adds no information.",
        );
      } else if (architecture === "local-pseudomode" && !jumps.length) {
        const damping = evaluateScalar(
          pseudomode.damping.ast, values.values, particleCount,
        );
        if (damping === 0) {
          warnings.push(
            isStationary
              ? "The local pseudomodes have zero damping and no system jump was selected, so trajectories do not provide dissipative relaxation to a stationary state."
              : "The local pseudomodes have zero damping and no system jump was selected, so every trajectory follows the same deterministic dynamics.",
          );
        }
      } else if (isDynamics && architecture === "global-pseudomode") {
        warnings.push(
          "Shared-mode trajectories monitor pseudomode damping. Any bare-system jump channels remain in the unconditional background and are not individually unravelled.",
        );
      }
    }
    if (particleCount > 1000) {
      if (steadyMethod === "trajectory") {
        warnings.push(
          "Large N was requested. Inspect the PI coordinate count and benchmark one trajectory before launching the ensemble.",
        );
      } else if (isStationary) {
        warnings.push(
          "Large N was requested. Inspect recommend_solver(model; task=:steady_state) before running.",
        );
      } else {
        warnings.push(
          "Large N was requested. Inspect the generated memory preflight and benchmark one prepared operator action first.",
        );
      }
    }
    return {
      particleCount,
      localDimension,
      architecture,
      calculation,
      target,
      isStationary,
      isDynamics,
      isSpectral,
      steadyMethod,
      workflow,
      initialState,
      trajectory,
      dynamics,
      spectrum,
      gap,
      memoryBudgetMiB,
      analysis,
      hamiltonian,
      jumps,
      observable,
      observableInfo,
      observableMode,
      parameters,
      parameterValues: values,
      hamiltonianComponents,
      observableComponents,
      pseudomode,
      warnings,
    };
  }

  function componentDefinition(component) {
    const names = {
      x: ["Jx", ":x"],
      y: ["Jy", ":y"],
      z: ["Jz", ":z"],
      plus: ["Jp", ":plus"],
      minus: ["Jm", ":minus"],
    };
    return names[component];
  }

  function componentLocalExpression(component) {
    const names = {
      x: "spin.jx",
      y: "spin.jy",
      z: "spin.jz",
      plus: "spin.jp",
      minus: "spin.jm",
    };
    return names[component];
  }

  function observableComponentName(component) {
    const [name] = componentDefinition(component);
    return `observable_${name}`;
  }

  function boundedBinomial(n, k, maximumFactors) {
    let retained = k;
    if (retained < 0n || retained > n) return 0n;
    if (retained > n - retained) retained = n - retained;
    if (retained > maximumFactors) return null;
    let value = 1n;
    for (let index = 1n; index <= retained; index += 1n) {
      value = (value * (n - retained + index)) / index;
    }
    return value;
  }

  function resourceSummary(parsed) {
    const N = BigInt(parsed.particleCount);
    const d = BigInt(parsed.localDimension);
    const modeLevels = parsed.pseudomode
      ? BigInt(parsed.pseudomode.nmax) + 1n
      : 1n;
    const siteDimension = parsed.architecture === "local-pseudomode"
      ? d * modeLevels
      : d;
    const siteOperatorDimension = siteDimension * siteDimension;
    const coordinateFormula = parsed.architecture === "global-pseudomode"
      ? "binomial(N + d^2 - 1, N) * (nmax + 1)^2"
      : parsed.architecture === "local-pseudomode"
        ? "binomial(N + (d*(nmax + 1))^2 - 1, N)"
        : "binomial(N + d^2 - 1, N)";
    let coordinates = boundedBinomial(
      N + siteOperatorDimension - 1n, N, 256n,
    );
    if (coordinates !== null && parsed.architecture === "global-pseudomode") {
      coordinates *= modeLevels * modeLevels;
    }
    const oneComplexVectorBytes =
      coordinates === null ? null : 16n * coordinates;
    const memoryBudgetBytes =
      BigInt(parsed.memoryBudgetMiB) * 1024n * 1024n;
    const representation = parsed.architecture === "pi"
      ? "complete PI Schur basis of N d-level systems"
      : parsed.architecture === "local-pseudomode"
        ? "complete PI Schur basis of identical system+pseudomode supersites"
        : "factorized complete system PI basis and one finite-mode operator factor";
    return {
      representation,
      scalarType: "ComplexF64",
      coordinateFormula,
      exactCoordinateCount:
        coordinates === null ? null : coordinates.toString(),
      oneComplexVectorBytes:
        oneComplexVectorBytes === null
          ? null
          : oneComplexVectorBytes.toString(),
      memoryBudgetBytes: memoryBudgetBytes.toString(),
      oneVectorFitsBudget:
        oneComplexVectorBytes === null
          ? null
          : oneComplexVectorBytes <= memoryBudgetBytes,
      caveat:
        "One-vector storage is a lower bound, not a solver peak. The generated Julia preflight accounts for prepared geometry, workspaces, Krylov history, and requested output.",
    };
  }

  function generatedStem(parsed) {
    const architecture = parsed.architecture === "pi"
      ? "pi"
      : parsed.architecture.replace(/-/g, "_");
    const method =
      parsed.isStationary || parsed.isDynamics
        ? `${parsed.steadyMethod}_`
        : "";
    const workflow = parsed.workflow === "verified-experiment"
      ? "verified_"
      : "";
    return `generated_${architecture}_${workflow}${method}` +
      parsed.calculation.replace(/-/g, "_");
  }

  function manifestFor(parsed, summary, resources) {
    const parameterValues = Object.create(null);
    for (const parameter of parsed.parameters) {
      parameterValues[parameter] =
        parsed.parameterValues.values.get(parameter);
    }
    const manifest = {
      schema: "permutational-invariant-dynamics/model-assistant/v1",
      generatorVersion: VERSION,
      workflow: parsed.workflow,
      model: {
        architecture: parsed.architecture,
        particles: parsed.particleCount,
        localDimension: parsed.localDimension,
        hamiltonian: parsed.hamiltonian
          ? parsed.hamiltonian.normalized
          : "",
        jumps: parsed.jumps.map((jump) => ({
          kind: jump.kind,
          operator: jump.operator.normalized,
          rate: jump.rate.normalized,
        })),
        pseudomode: parsed.pseudomode
          ? {
            cutoff: parsed.pseudomode.nmax,
            frequency: parsed.pseudomode.frequency.normalized,
            damping: parsed.pseudomode.damping.normalized,
            thermalOccupation:
              parsed.pseudomode.thermalOccupation.normalized,
            couplingOperator:
              parsed.pseudomode.couplingOperator.normalized,
            couplingStrength: parsed.pseudomode.strength.normalized,
            counterrotatingStrength:
              parsed.pseudomode.counterrotatingStrength.normalized,
          }
          : null,
        parameters: parameterValues,
      },
      calculation: {
        task: parsed.calculation,
        method: parsed.steadyMethod,
        observable: parsed.observable ? parsed.observable.normalized : "",
        initialState: parsed.initialState,
        dynamics: parsed.dynamics,
        trajectory: parsed.trajectory,
        spectrum: parsed.spectrum,
        gap: parsed.gap,
        analysis: parsed.analysis,
      },
      representation: resources,
      route: summary.route,
      warnings: parsed.warnings.slice(),
    };
    return manifest;
  }

  function readmeFor(stem, parsed, resources) {
    const coordinateText = resources.exactCoordinateCount === null
      ? `${resources.coordinateFormula} (not expanded in the browser)`
      : `${resources.exactCoordinateCount} (${resources.coordinateFormula})`;
    return [
      "PermutationalInvariantDynamics.jl generated experiment",
      "======================================================",
      "",
      `Generator version: ${VERSION}`,
      `Workflow: ${parsed.workflow}`,
      `Architecture: ${parsed.architecture}`,
      `Calculation: ${parsed.calculation}`,
      `Representation: ${resources.representation}`,
      `Exact coordinate count: ${coordinateText}`,
      `Memory budget: ${resources.memoryBudgetBytes} bytes`,
      "",
      "Files",
      "-----",
      `${stem}.jl        Commented Julia program.`,
      `${stem}.json      Machine-readable normalized model and resource manifest.`,
      `${stem}_README.txt  This file.`,
      "",
      "License",
      "-------",
      "The generated Julia program contains template code from",
      "PermutationalInvariantDynamics.jl and is licensed under",
      "GPL-3.0-only, without an output-license exception. Redistribution or",
      "modification of that program must comply with version 3 of the GNU",
      "General Public License. The complete license is available in the",
      "package root LICENSE file and at:",
      "https://www.gnu.org/licenses/gpl-3.0.html",
      "",
      "The JSON file is descriptive metadata containing the normalized model",
      "and resource information. It contains no copied Julia or JavaScript",
      "program template. This README repeats the program's license so that",
      "the three separately downloaded bundle files remain understandable.",
      "",
      "Run",
      "---",
      `julia --project=. ${stem}.jl`,
      "",
      "The browser performs no solve. Review the physical PI assumption, jump",
      "semantics, parameter values, solver diagnostics, and convergence before",
      "using a result. One-vector storage is not the solver peak; the Julia",
      "program keeps the package memory preflight enabled.",
      "",
    ].join("\n");
  }

  function generate(config) {
    const parsed = parseModel(config);
    const resources = resourceSummary(parsed);
    if (resources.oneVectorFitsBudget === false) {
      parsed.warnings.push(
        "One ComplexF64 coordinate vector already exceeds the declared memory budget. The Julia preflight will reject any route requiring that vector; reduce the representation or raise the budget deliberately.",
      );
    } else if (resources.exactCoordinateCount === null) {
      parsed.warnings.push(
        "The exact coordinate formula is retained in the manifest, but this unusually large combinatorial count was not expanded in the browser. Inspect pi_dimension and the Julia resource preflight before running.",
      );
    }
    const lines = [];
    const isPseudomode = parsed.architecture !== "pi";
    const isLocalPseudomode = parsed.architecture === "local-pseudomode";
    const isGlobalPseudomode = parsed.architecture === "global-pseudomode";
    const isStationary = parsed.isStationary;
    const isDynamics = parsed.isDynamics;
    const isSpectrum = parsed.calculation === "liouvillian-spectrum";
    const isGap = parsed.calculation === "liouvillian-gap";
    const isTrajectory = parsed.steadyMethod === "trajectory";
    const isExperiment = parsed.workflow === "verified-experiment";
    const hamiltonianEntries = [];
    const linearHamiltonian = [];
    const nonlinearHamiltonianEntries = [];
    if (parsed.hamiltonian) {
      flattenAdditive(parsed.hamiltonian.ast, 1, hamiltonianEntries);
      for (const entry of hamiltonianEntries) {
        const info = analyze(entry.node, "hamiltonian");
        (info.linear ? linearHamiltonian : nonlinearHamiltonianEntries).push(entry);
      }
    }

    // REUSE-IgnoreStart
    // The SPDX lines below describe the generated Julia program, not this
    // JavaScript source file.
    lines.push("# SPDX-FileCopyrightText: 2026 PermutationalInvariantDynamics.jl contributors");
    lines.push("#");
    lines.push("# SPDX-License-Identifier: GPL-3.0-only");
    lines.push("#");
    lines.push("# This generated program contains template code from");
    lines.push("# PermutationalInvariantDynamics.jl and is licensed under version 3");
    lines.push("# of the GNU General Public License, with no option to use a later");
    lines.push("# version. This program comes with ABSOLUTELY NO WARRANTY.");
    lines.push("# See the package LICENSE file or");
    lines.push("# https://www.gnu.org/licenses/gpl-3.0.html.");
    lines.push("#");
    // REUSE-IgnoreEnd
    lines.push("# Generated by the PermutationalInvariantDynamics.jl model assistant.");
    lines.push("# Review the PI assumption: every constituent must be physically equivalent");
    lines.push("# and every local term/channel must act identically on all constituents.");
    lines.push("# Local sum_i D[l_i] and collective D[sum_i l_i] channels are different.");
    if (isLocalPseudomode) {
      lines.push("# Topology: every system has its own identical finite-cutoff pseudomode.");
      lines.push("# Permutations act on complete system+pseudomode supersites.");
    } else if (isGlobalPseudomode) {
      lines.push("# Topology: one finite-cutoff pseudomode is shared by the PI ensemble.");
      lines.push("# The coupling is collective and no Kac scaling is inserted automatically.");
    }
    lines.push(`# Representation: ${resources.representation}.`);
    lines.push(`# Coordinate formula: ${resources.coordinateFormula}.`);
    if (resources.exactCoordinateCount !== null) {
      lines.push(
        `# Exact PI/composite coordinates: ${resources.exactCoordinateCount}.`,
      );
      lines.push(
        `# One ComplexF64 coordinate vector: ${resources.oneComplexVectorBytes} bytes.`,
      );
    } else {
      lines.push("# The browser did not expand the exact coordinate count; Julia prints it below.");
    }
    lines.push("# One-vector storage is only a lower bound; the solver preflight remains authoritative.");
    if (isGlobalPseudomode) lines.push("using LinearAlgebra");
    if (isSpectrum) lines.push("using Random");
    lines.push("using PermutationalInvariantDynamics");
    lines.push("");
    lines.push("# One explicit budget guards preparation, solver scratch, and requested output.");
    lines.push(
      `const MEMORY_BUDGET = ${parsed.memoryBudgetMiB} * 1024^2`,
    );
    if (isStationary && !isTrajectory) {
      lines.push("const STEADY_ATOL = 1e-11");
      lines.push("const STEADY_RTOL = 1e-9");
    }
    if (isStationary || isExperiment) {
      lines.push("const STATE_VALIDATION_TOL = 1e-8");
    }
    if (parsed.initialState) {
      lines.push(`const INITIAL_LEVEL = ${parsed.initialState.level}`);
    }
    if (isTrajectory) {
      const trajectory = parsed.trajectory;
      lines.push("");
      lines.push("# Converge all trajectory controls before using the estimate quantitatively.");
      lines.push(`const TRAJECTORIES = ${trajectory.trajectories}`);
      lines.push(`const TRAJECTORY_DT = ${floatLiteral(trajectory.dt)}`);
      lines.push(
        `const MAX_JUMP_PROBABILITY = ${floatLiteral(trajectory.maxJumpProbability)}`,
      );
      lines.push(`const TRAJECTORY_SEED = ${trajectory.seed}`);
      if (isStationary) {
        lines.push(`const SETTLING_TIME = ${floatLiteral(trajectory.settlingTime)}`);
        lines.push(`const SAMPLES_PER_TRAJECTORY = ${trajectory.samplesPerTrajectory}`);
        lines.push(`const SAMPLING_INTERVAL = ${floatLiteral(trajectory.samplingInterval)}`);
      }
    }
    if (isDynamics) {
      lines.push("");
      lines.push(
        isExperiment
          ? "# Verified output is sampled at these physical times; retained states are memory guarded."
          : "# Output is streamed at these physical times; no state history is retained.",
      );
      lines.push(
        `const DYNAMICS_START_TIME = ${floatLiteral(parsed.dynamics.startTime)}`,
      );
      lines.push(
        `const DYNAMICS_FINAL_TIME = ${floatLiteral(parsed.dynamics.finalTime)}`,
      );
      lines.push(`const DYNAMICS_SAMPLES = ${parsed.dynamics.samples}`);
      if (!isTrajectory) {
        lines.push(
          `const STEPS_PER_INTERVAL = ${parsed.dynamics.stepsPerInterval}`,
        );
      }
    }
    if (isSpectrum) {
      lines.push("");
      lines.push(`const SPECTRUM_NEV = ${parsed.spectrum.nev}`);
      lines.push(`const SPECTRUM_SEED = ${parsed.spectrum.seed}`);
    }
    if (isGap) {
      lines.push("");
      lines.push(`const GAP_NEV = ${parsed.gap.nev}`);
      lines.push(`const GAP_KRYLOVDIM = ${parsed.gap.krylovdim}`);
    }
    if (isPseudomode || isTrajectory || isDynamics || isSpectrum || isGap) {
      lines.push("");
    }
    lines.push(`N = ${parsed.particleCount}`);
    lines.push(`d = ${parsed.localDimension}`);
    lines.push("spin = spin_matrices(d)");

    if (parsed.parameters.size) {
      lines.push("");
      lines.push("# Scalar parameters");
      for (const parameter of parsed.parameters) {
        const missing = !parsed.parameterValues.provided.has(parameter);
        const suffix = missing ? "  # TODO: replace this placeholder" : "";
        lines.push(`${parameter} = ${parsed.parameterValues.code.get(parameter)}${suffix}`);
      }
    }

    if (!isLocalPseudomode) {
      lines.push("");
      if (isGlobalPseudomode) {
        lines.push("system_basis = PIBasis(N, d)  # complete system PI basis");
      } else {
        lines.push("basis = PIBasis(N, d)  # complete PI basis; local noise may mix Schur sectors");
      }
    }

    const systemBasisName = isGlobalPseudomode ? "system_basis" : "basis";
    if (parsed.hamiltonianComponents.size && !isLocalPseudomode) {
      lines.push("");
      lines.push("# Shared one-body geometry avoids rebuilding nonlinear Hamiltonian maps.");
      lines.push(`one_body_geometry = OneBodyGeometry(${systemBasisName})`);
      for (const component of parsed.hamiltonianComponents) {
        const [name, symbol] = componentDefinition(component);
        lines.push(`${name} = collective_spin(${systemBasisName}, ${symbol}; cache=one_body_geometry)`);
      }
    }

    const termLines = [];
    if (!isLocalPseudomode) {
      if (linearHamiltonian.length) {
        const localExpression = signedExpression(linearHamiltonian, "local");
        termLines.push(`LocalHamiltonian(${localExpression}),`);
      }
      if (nonlinearHamiltonianEntries.length) {
        lines.push("");
        lines.push("# Nonlinear collective Hamiltonian stays entirely in compressed PI coordinates.");
        lines.push(`H_collective = ${signedExpression(nonlinearHamiltonianEntries, "collective")}`);
        termLines.push("DirectPIHamiltonian(H_collective),");
      }
    }

    if (parsed.jumps.length) {
      lines.push("");
      lines.push("# These are bare-system d-by-d jump seeds.");
      parsed.jumps.forEach((jump, index) => {
        const name = `jump_${index + 1}`;
        lines.push(`${name} = ${emit(jump.operator.ast, "local")}`);
        const constructor = jump.kind === "collective" ? "CollectiveJump" : "LocalJump";
        termLines.push(`${constructor}(${name}; rate=${emit(jump.rate.ast, "scalar")}),`);
      });
    }

    function emitTrajectorySolve(initialStateLines) {
      lines.push("");
      for (const line of initialStateLines) lines.push(line);
      lines.push("");
      lines.push("# Preflight preparation before allocating the term-resolved trajectory plan.");
      lines.push("trajectory_preflight = recommend_solver(");
      lines.push("    model; task=:dynamics, algorithm=:rk4,");
      lines.push("    samples=1, saved_states=0,");
      lines.push("    memory_budget=MEMORY_BUDGET,");
      lines.push(")");
      lines.push("trajectory_preflight.budget_status === :exceeds && error(");
      lines.push("    \"trajectory preparation exceeds MEMORY_BUDGET\")");
      lines.push("println(\"trajectory preparation budget status = \",");
      lines.push("    trajectory_preflight.budget_status)");
      lines.push("# Preserve term-resolved physical jump channels for trajectory sampling.");
      lines.push("trajectory_plan = TrajectoryPlan(model)");
      lines.push("# Fixed-capacity worker scratch is reused across all independent paths.");
      lines.push(
        "trajectory_workers = min(TRAJECTORIES, Threads.nthreads())",
      );
      lines.push("trajectory_workspace = TrajectoryBatchWorkspace(");
      lines.push("    trajectory_plan, rho0;");
      lines.push("    workers=trajectory_workers, mode=:fixed,");
      lines.push(")");
      lines.push("steady = trajectory_steady_state(");
      lines.push("    trajectory_plan, rho0;");
      lines.push("    trajectories=TRAJECTORIES,");
      lines.push("    settling_time=SETTLING_TIME,");
      lines.push("    dt=TRAJECTORY_DT,");
      lines.push("    samples_per_trajectory=SAMPLES_PER_TRAJECTORY,");
      lines.push("    sampling_interval=SAMPLING_INTERVAL,");
      lines.push("    max_jump_probability=MAX_JUMP_PROBABILITY,");
      lines.push("    algorithm=:fixed,");
      lines.push("    seed=TRAJECTORY_SEED,");
      lines.push("    threaded=trajectory_workers > 1,");
      lines.push("    workspace=trajectory_workspace,");
      lines.push("    memory_budget=MEMORY_BUDGET,");
      lines.push("    return_info=true,");
      lines.push(")");
    }

    function emitRepresentationSummary(coordinateExpression) {
      lines.push("");
      lines.push("# Report the retained representation before any expensive analysis.");
      lines.push(
        `println("representation = ${resources.representation}")`,
      );
      lines.push(`println("retained coordinates = ", ${coordinateExpression})`);
      lines.push("println(\"declared memory budget (bytes) = \", MEMORY_BUDGET)");
      lines.push("# The coordinate-vector size is not the solver peak.");
    }

    function emitDeterministicStationarySolve() {
      lines.push("");
      if (isExperiment) {
        lines.push("# Typed experiment planning explains the route before the solve.");
        lines.push("experiment = PIExperiment(");
        lines.push("    model; task=:steady_state, algorithm=AutoAlgorithm(),");
        lines.push("    memory_budget=MEMORY_BUDGET,");
        lines.push("    verification=VerificationSpec(");
        lines.push("        atol=STATE_VALIDATION_TOL,");
        lines.push("        rtol=STATE_VALIDATION_TOL,");
        lines.push("    ),");
        lines.push("    solver_options=(atol=STEADY_ATOL, rtol=STEADY_RTOL),");
        lines.push(
          `    metadata=(generator_version="${VERSION}", ` +
          `architecture="${parsed.architecture}"),`,
        );
        lines.push(")");
        lines.push("experiment_plan = explain_experiment(experiment)");
        lines.push("println(\"experiment plan = \", experiment_plan)");
        lines.push("experiment_result = verified_solve(experiment)");
        lines.push("println(\"verification report = \", experiment_result.report)");
        lines.push("println(\"provenance digest = \",");
        lines.push("    experiment_result.provenance.structural_digest)");
        lines.push("steady = experiment_result.solution");
        lines.push("# Optional: save_experiment(\"result.pidrun\", experiment_result)");
      } else {
        lines.push("# :auto selects the sparse/direct or matrix-free route from the compiled problem.");
        lines.push("# For a large run, inspect recommend_solver(model; task=:steady_state) first.");
        lines.push("prepared = compile(");
        lines.push("    model; backend=:auto, memory_budget=MEMORY_BUDGET)");
        lines.push("steady = stationary_state(");
        lines.push("    prepared;");
        lines.push("    atol=STEADY_ATOL, rtol=STEADY_RTOL,");
        lines.push("    return_info=true, memory_budget=MEMORY_BUDGET,");
        lines.push(")");
      }
    }

    function emitInitialState() {
      lines.push("");
      if (!isPseudomode) {
        lines.push("# Explicit PI product initial state in one selected local level.");
        lines.push("rho0 = computational_product_state(basis, INITIAL_LEVEL)");
      } else {
        lines.push("# Start every physical system in one selected level.");
        lines.push("system_initial = zeros(ComplexF64, d)");
        lines.push("system_initial[INITIAL_LEVEL] = 1");
        if (isLocalPseudomode) {
          lines.push("# Every local pseudomode starts in vacuum.");
          lines.push("rho0 = pseudomode_product_state(");
          lines.push("    site, system_initial; memory_budget=MEMORY_BUDGET)");
        } else {
          lines.push("# The shared pseudomode starts in vacuum.");
          lines.push("rho0 = pseudomode_product_state(");
          lines.push("    embedding, system_initial; memory_budget=MEMORY_BUDGET)");
        }
      }
    }

    function emitObservableOperator() {
      lines.push("");
      lines.push("# Construct the sampled observable only in compressed PI coordinates.");
      const observableBasis = isGlobalPseudomode ? "system_basis" : "basis";
      const canReuseGeometry = parsed.hamiltonianComponents.size > 0;
      const observableGeometry = canReuseGeometry
        ? "one_body_geometry"
        : "observable_one_body_geometry";
      if (!canReuseGeometry) {
        lines.push(
          `observable_one_body_geometry = OneBodyGeometry(${observableBasis})`,
        );
      }
      for (const component of parsed.observableComponents) {
        const [hamiltonianName, symbol] = componentDefinition(component);
        const observableName = observableComponentName(component);
        if (parsed.hamiltonianComponents.has(component)) {
          lines.push(`${observableName} = ${hamiltonianName}`);
        } else if (isLocalPseudomode) {
          lines.push(`${observableName} = collective_operator(`);
          lines.push("    basis,");
          lines.push(
            `    lift_system_operator(site, ${componentLocalExpression(component)}; ` +
            "memory_budget=MEMORY_BUDGET);",
          );
          lines.push(`    cache=${observableGeometry},`);
          lines.push(")");
        } else {
          lines.push(
            `${observableName} = ` +
            `collective_spin(${observableBasis}, ${symbol}; cache=${observableGeometry})`,
          );
        }
      }
      if (
        parsed.observableMode === "collective-plan" ||
        parsed.observableMode === "single-site-plan"
      ) {
        let localObservable = emit(parsed.observable.ast, "local");
        if (parsed.observableMode === "single-site-plan") {
          localObservable = `((${localObservable}) / N)`;
        }
        const preparedLocalObservable = isLocalPseudomode
          ? `lift_system_operator(site, ${localObservable}; ` +
            "memory_budget=MEMORY_BUDGET)"
          : localObservable;
        lines.push("observable = collective_operator(");
        lines.push(`    ${observableBasis}, ${preparedLocalObservable};`);
        lines.push(`    cache=${observableGeometry},`);
        lines.push(")");
      } else {
        lines.push(`observable = ${emit(parsed.observable.ast, "observable-collective")}`);
      }
      if (isGlobalPseudomode) {
        lines.push("system_observable = observable");
        lines.push("observable = composite_tensor_operator(");
        lines.push("    embedding.basis, system_observable,");
        lines.push("    embedding.mode_operators.identity,");
        lines.push(")");
      }
      lines.push("# Streaming contractions use dot(op, rho) = tr(op' * rho).");
      lines.push(
        isTrajectory
          ? "streaming_observable = observable  # trajectory observables are Hermitian"
          : "streaming_observable = adjoint(observable)",
      );
    }

    if (!isPseudomode) {
      lines.push("");
      lines.push("# A tuple keeps the compiled kernel types concrete.");
      lines.push("terms = (");
      for (const termLine of termLines) lines.push(`    ${termLine}`);
      lines.push(")");
      lines.push("model = PIModel(basis, terms)");
      emitRepresentationSummary("pi_dimension(basis)");
      if (isStationary) {
        if (isTrajectory) {
          emitTrajectorySolve([
            "# Use an explicit pure PI product state; change it to test initial-state dependence.",
            "rho0 = computational_product_state(basis, INITIAL_LEVEL)",
          ]);
        } else {
          emitDeterministicStationarySolve();
        }
      }
    } else {
      const pseudomode = parsed.pseudomode;
      lines.push("");
      lines.push("# The mode data are separate from the bare-system Hamiltonian and jumps.");
      lines.push("mode = BosonicPseudomode(");
      lines.push(`    ${pseudomode.nmax};`);
      lines.push("    label=:pseudomode,");
      lines.push(`    frequency=${emit(pseudomode.frequency.ast, "scalar")},`);
      lines.push(`    damping=${emit(pseudomode.damping.ast, "scalar")},`);
      lines.push(`    thermal_occupation=${emit(pseudomode.thermalOccupation.ast, "scalar")},`);
      lines.push("    memory_budget=MEMORY_BUDGET,");
      lines.push(")");
      lines.push("coupling = PseudomodeCoupling(");
      lines.push(`    ${emit(pseudomode.couplingOperator.ast, "local")};`);
      lines.push("    mode=:pseudomode,");
      lines.push(`    strength=${emit(pseudomode.strength.ast, "scalar")},`);
      lines.push(`    counterrotating_strength=${emit(pseudomode.counterrotatingStrength.ast, "scalar")},`);
      lines.push("    memory_budget=MEMORY_BUDGET,");
      lines.push(")");

      if (isLocalPseudomode) {
        lines.push("");
        lines.push("# One exact PI supersite contains system_i and its own local mode_i.");
        lines.push("site = pseudomode_supersite(");
        lines.push("    N, d, mode; memory_budget=MEMORY_BUDGET)");
        lines.push("basis = site.basis  # complete supersite PI basis");
        if (parsed.hamiltonianComponents.size) {
          lines.push("# Reuse this geometry for nonlinear system operators and diagnostics.");
          lines.push("one_body_geometry = OneBodyGeometry(basis)");
        }
        if (linearHamiltonian.length) {
          lines.push(`H_system = ${signedExpression(linearHamiltonian, "local")}`);
        } else {
          lines.push("H_system = zeros(ComplexF64, d, d)");
        }
        if (nonlinearHamiltonianEntries.length) {
          lines.push("");
          lines.push("# Lift bare-system generators before forming a supersite PI polynomial.");
          for (const component of parsed.hamiltonianComponents) {
            const [name] = componentDefinition(component);
            lines.push(`${name} = collective_operator(`);
            lines.push("    basis,");
            lines.push(
              `    lift_system_operator(site, ${componentLocalExpression(component)}; ` +
              "memory_budget=MEMORY_BUDGET);",
            );
            lines.push("    cache=one_body_geometry,");
            lines.push(")");
          }
          lines.push(`H_collective = ${signedExpression(nonlinearHamiltonianEntries, "collective")}`);
          lines.push("supersite_terms = (DirectPIHamiltonian(H_collective),)");
        } else {
          lines.push("supersite_terms = ()");
        }
        lines.push("system_terms = (");
        for (const termLine of termLines) lines.push(`    ${termLine}`);
        lines.push(")");
        lines.push("embedding = pseudomode_model(");
        lines.push("    site, H_system;");
        lines.push("    couplings=(coupling,),");
        lines.push("    system_terms=system_terms,");
        lines.push("    supersite_terms=supersite_terms,");
        lines.push("    memory_budget=MEMORY_BUDGET,");
        lines.push(")");
        lines.push("model = embedding.model");
        emitRepresentationSummary("pi_dimension(basis)");
        if (isStationary) {
          if (isTrajectory) {
            emitTrajectorySolve([
              "# Start every system in one selected level and every local mode in vacuum.",
              "system_initial = zeros(ComplexF64, d)",
              "system_initial[INITIAL_LEVEL] = 1",
              "rho0 = pseudomode_product_state(",
              "    site, system_initial; memory_budget=MEMORY_BUDGET)",
            ]);
          } else {
            emitDeterministicStationarySolve();
          }
        }
      } else {
        lines.push("");
        lines.push("# Build the ordinary PI system first, then attach one shared mode.");
        lines.push("system_terms = (");
        for (const termLine of termLines) lines.push(`    ${termLine}`);
        lines.push(")");
        lines.push("system_model = PIModel(system_basis, system_terms)");
        lines.push("embedding = global_pseudomode_model(");
        lines.push("    system_model, mode;");
        lines.push("    couplings=(coupling,),");
        lines.push("    memory_budget=MEMORY_BUDGET,");
        lines.push(")");
        emitRepresentationSummary("length(embedding.basis)");
        if (isStationary) {
          lines.push("");
          lines.push("# The shared-mode model is already factorized; its automatic route is matrix-free GMRES.");
          lines.push("# Do not compile or materialize its global Kronecker superoperator.");
          lines.push("steady = stationary_state(");
          lines.push("    embedding;");
          lines.push("    algorithm=GMRESAlgorithm(krylovdim=40, maxiter=1000),");
          lines.push("    atol=STEADY_ATOL, rtol=STEADY_RTOL,");
          lines.push("    return_info=true,");
          lines.push("    memory_budget=MEMORY_BUDGET,");
          lines.push(")");
        }
      }
    }

    if (isDynamics) {
      emitInitialState();
      emitObservableOperator();
      lines.push("");
      lines.push("times = collect(range(");
      lines.push("    DYNAMICS_START_TIME, DYNAMICS_FINAL_TIME;");
      lines.push("    length=DYNAMICS_SAMPLES,");
      lines.push("))");
      if (isTrajectory) {
        lines.push("");
        if (isGlobalPseudomode) {
          lines.push("# Only shared-mode damping is monitored; system jumps stay in the background.");
          lines.push("composite_trajectory_plan = CompositeTrajectoryPlan(");
          lines.push("    embedding.background, embedding.damping_channels...,");
          lines.push(")");
          lines.push("trajectory_workers = min(TRAJECTORIES, Threads.nthreads())");
          lines.push("trajectory_workspace = CompositeTrajectoryBatchWorkspace(");
          lines.push("    composite_trajectory_plan, rho0;");
          lines.push("    workers=trajectory_workers,");
          lines.push(")");
          lines.push("dynamics = quantum_trajectories(");
          lines.push("    composite_trajectory_plan, rho0, times, TRAJECTORIES;");
        } else {
          lines.push("# Preserve term-resolved physical jump channels for trajectory sampling.");
          lines.push("trajectory_plan = TrajectoryPlan(model)");
          lines.push("trajectory_workers = min(TRAJECTORIES, Threads.nthreads())");
          lines.push("trajectory_workspace = TrajectoryBatchWorkspace(");
          lines.push("    trajectory_plan, rho0;");
          lines.push("    workers=trajectory_workers, mode=:fixed,");
          lines.push(")");
          lines.push("dynamics = quantum_trajectories(");
          lines.push("    trajectory_plan, rho0, times, TRAJECTORIES;");
        }
        lines.push("    dt=TRAJECTORY_DT,");
        lines.push("    max_jump_probability=MAX_JUMP_PROBABILITY,");
        lines.push("    algorithm=:fixed,");
        lines.push("    seed=TRAJECTORY_SEED,");
        lines.push("    threaded=trajectory_workers > 1,");
        lines.push("    workspace=trajectory_workspace,");
        lines.push("    observables=(observable=streaming_observable,),");
        lines.push("    save_states=false, jump_statistics=false,");
        lines.push("    confidence=0.95, memory_budget=MEMORY_BUDGET,");
        lines.push(")");
        lines.push("dynamics_statistics =");
        lines.push("    dynamics.observables.observables[:observable]");
        lines.push("observable_values = dynamics_statistics.mean");
        lines.push("observable_standard_error = dynamics_statistics.standard_error");
        lines.push("");
        lines.push("for index in eachindex(times)");
        lines.push("    println(times[index], \"  \", observable_values[index], \"  \",");
        lines.push("        observable_standard_error[index], \"  \",");
        lines.push("        dynamics_statistics.lower[index], \"  \",");
        lines.push("        dynamics_statistics.upper[index])");
        lines.push("end");
        lines.push("# Columns: time, Monte Carlo mean, standard error, lower CI, upper CI.");
      } else {
        lines.push("");
        if (isExperiment) {
          lines.push("# Verified dynamics retains sampled PI states for physicality checks.");
          lines.push("# The memory preflight counts this history before propagation.");
          lines.push("experiment = PIExperiment(");
          lines.push("    model; task=:dynamics, initial_state=rho0,");
          lines.push("    algorithm=:rk4,");
          lines.push("    tspan=(DYNAMICS_START_TIME, DYNAMICS_FINAL_TIME),");
          lines.push("    saveat=times, steps_per_interval=STEPS_PER_INTERVAL,");
          lines.push("    observables=(observable=streaming_observable,),");
          lines.push("    save_states=true, memory_budget=MEMORY_BUDGET,");
          lines.push("    verification=VerificationSpec(");
          lines.push("        atol=STATE_VALIDATION_TOL,");
          lines.push("        rtol=STATE_VALIDATION_TOL,");
          lines.push("    ),");
          lines.push(
            `    metadata=(generator_version="${VERSION}", ` +
            `architecture="${parsed.architecture}"),`,
          );
          lines.push(")");
          lines.push("experiment_plan = explain_experiment(experiment)");
          lines.push("println(\"experiment plan = \", experiment_plan)");
          lines.push("experiment_result = verified_solve(experiment)");
          lines.push("println(\"verification report = \", experiment_result.report)");
          lines.push("println(\"provenance digest = \",");
          lines.push("    experiment_result.provenance.structural_digest)");
          lines.push("dynamics = experiment_result.solution");
          lines.push(
            "observable_values = experiment_result.observables[:observable]",
          );
          lines.push("# Optional: save_experiment(\"result.pidrun\", experiment_result)");
        } else {
          lines.push("# Matrix-free RK4 streams the observable and retains no sampled states.");
          lines.push("prepared = compile(");
          lines.push("    model; backend=:matrixfree, memory_budget=MEMORY_BUDGET)");
          lines.push("dynamics = solve_dynamics(");
          lines.push("    prepared, rho0, (DYNAMICS_START_TIME, DYNAMICS_FINAL_TIME);");
          lines.push("    saveat=times, steps_per_interval=STEPS_PER_INTERVAL,");
          lines.push("    observables=(observable=streaming_observable,),");
          lines.push("    save_states=false, memory_budget=MEMORY_BUDGET,");
          lines.push(")");
          lines.push("observable_values = dynamics.observables[:observable]");
        }
        lines.push("");
        lines.push("for (time, value) in zip(dynamics.times, observable_values)");
        lines.push("    println(time, \"  \", value)");
        lines.push("end");
        lines.push("# Columns: time, expectation value.");
      }
    } else if (isSpectrum) {
      const spectralSource = isGlobalPseudomode ? "embedding" : "model";
      lines.push("");
      lines.push("# The automatic route is dense only when the complete problem fits the budget;");
      lines.push("# otherwise it uses a selected matrix-free Krylov spectrum.");
      lines.push("spectrum = liouvillian_spectrum(");
      lines.push(`    ${spectralSource};`);
      lines.push(`    target=:${parsed.spectrum.target}, nev=SPECTRUM_NEV,`);
      lines.push("    algorithm=:auto, vectors=false, return_info=true,");
      lines.push("    rng=Random.MersenneTwister(SPECTRUM_SEED),");
      lines.push("    memory_budget=MEMORY_BUDGET,");
      lines.push(")");
      lines.push("spectrum_values = spectrum.values");
      lines.push("println(\"selected spectral algorithm = \", spectrum.info.selected_algorithm)");
      lines.push("println(\"selected Liouvillian eigenvalues:\")");
      lines.push("foreach(println, spectrum_values)");
      lines.push("# A partial selected spectrum is not, by itself, a certified global gap.");
    } else if (isGap) {
      lines.push("");
      if (isGlobalPseudomode) {
        lines.push("# Keep the shared-mode generator factorized and matrix free.");
        lines.push("gap_source = global_pseudomode_matrixfree(");
        lines.push("    embedding; memory_budget=MEMORY_BUDGET)");
      } else {
        lines.push("gap_source = model");
      }
      lines.push("# The largest-real Krylov route returns explicit certification metadata.");
      lines.push("gap_result = pi_liouvillian_gap(");
      lines.push("    gap_source; method=:krylov,");
      lines.push("    nev=GAP_NEV, krylovdim=GAP_KRYLOVDIM,");
      lines.push("    return_info=true, memory_budget=MEMORY_BUDGET,");
      lines.push(")");
      lines.push("println(\"Liouvillian gap = \", gap_result.gap)");
      lines.push("println(\"gap certified = \", gap_result.gap_certified)");
      lines.push("println(\"stationary multiplicity = \",");
      lines.push("    gap_result.stationary_multiplicity)");
      lines.push("println(\"stationary multiplicity certified = \",");
      lines.push("    gap_result.stationary_multiplicity_certified)");
      lines.push("println(\"stable = \", gap_result.stable)");
      lines.push("# Never report the value as certified unless gap_certified is true.");
    }

    if (isStationary) {
      if (!isTrajectory) {
        lines.push("steady.info.converged || error(\"stationary solver did not converge\")");
      }
      lines.push("rho_ss = steady.state");
      if (isGlobalPseudomode) {
      lines.push("# Packed model-owned reductions avoid any full-system reconstruction.");
      lines.push("# This does not certify positivity of the complete composite state.");
      lines.push("LinearAlgebra.ishermitian(");
      lines.push("    rho_ss; atol=STATE_VALIDATION_TOL, rtol=STATE_VALIDATION_TOL) ||");
      lines.push("    error(\"stationary composite state is not Hermitian\")");
      lines.push("rho_system = trace_pseudomodes(rho_ss, embedding)");
      lines.push("rho_mode = global_pseudomode_state(rho_ss, embedding)");
      lines.push("validate_state(");
      lines.push("    rho_system; atol=STATE_VALIDATION_TOL, rtol=STATE_VALIDATION_TOL)");
      lines.push("mode_top_population = real(rho_mode[end, end])");
      } else {
        if (isLocalPseudomode || isTrajectory) {
        lines.push("validate_state(");
        lines.push("    rho_ss; atol=STATE_VALIDATION_TOL, rtol=STATE_VALIDATION_TOL)");
        } else {
        lines.push("validate_state(rho_ss)  # checks trace, Hermiticity, and positivity");
        }
        if (isLocalPseudomode) {
        if (!parsed.hamiltonianComponents.size) {
          lines.push("# Prepare analysis geometry only after the stationary solve.");
          lines.push("one_body_geometry = OneBodyGeometry(basis)");
        }
        lines.push("mode_operators = only(embedding.mode_operators)");
        lines.push("mode_top_plan = CollectiveObservablePlan(");
        lines.push("    basis, mode_operators.top_projector; cache=one_body_geometry)");
        lines.push("mode_top_population =");
        lines.push("    real(collective_expectation(rho_ss, mode_top_plan)) / N");
        }
      }
      lines.push("");
      if (isGlobalPseudomode) {
      lines.push("println(\"PI system coordinates = \", pi_dimension(system_basis))");
      lines.push("println(\"composite coordinates = \", length(embedding.basis))");
      lines.push("println(\"composite trace = \", trace(rho_ss))");
      } else {
      lines.push("println(\"PI coordinates = \", pi_dimension(basis))");
      }
      if (isTrajectory) {
      lines.push("println(\"independent trajectories = \", steady.trajectory_count)");
      lines.push("println(\"samples per trajectory = \", steady.samples_per_trajectory)");
      lines.push("println(\"Hilbert--Schmidt sample spread = \", steady.sample_spread)");
      lines.push("println(\"Hilbert--Schmidt standard error = \", steady.standard_error)");
      lines.push("println(\"stationary residual = \", steady.residual)");
      lines.push("println(\"stationary relative residual = \", steady.relative_residual)");
      lines.push("println(\"stationary trace error = \", steady.trace_error)");
      } else {
      lines.push("println(\"stationary residual = \", steady.info.residual)");
      lines.push("println(\"stationary trace error = \", steady.info.trace_error)");
      }
      if (isPseudomode) {
      lines.push("println(\"pseudomode top-level population = \", mode_top_population)");
      }

      if (parsed.target === "expectation") {
      lines.push("");
      const observableBasis = isGlobalPseudomode ? "system_basis" : "basis";
      const observableState = isGlobalPseudomode ? "rho_system" : "rho_ss";
      const needsObservableGeometry =
        parsed.observableMode === "collective-plan" ||
        parsed.observableMode === "single-site-plan" ||
        parsed.observableComponents.size > 0;
      const canReuseHamiltonianGeometry =
        parsed.hamiltonianComponents.size > 0 && !isLocalPseudomode;
      let observableGeometry = "one_body_geometry";
      if (isLocalPseudomode) {
        observableGeometry = "one_body_geometry";
      } else if (needsObservableGeometry && !canReuseHamiltonianGeometry) {
        lines.push("# Observable geometry is prepared only after the stationary solve.");
        lines.push(`observable_one_body_geometry = OneBodyGeometry(${observableBasis})`);
        observableGeometry = "observable_one_body_geometry";
      }
      for (const component of parsed.observableComponents) {
        const [hamiltonianName, symbol] = componentDefinition(component);
        const observableName = observableComponentName(component);
        if (parsed.hamiltonianComponents.has(component)) {
          lines.push(`${observableName} = ${hamiltonianName}`);
        } else if (isLocalPseudomode) {
          lines.push(`${observableName} = collective_operator(`);
          lines.push("    basis,");
          lines.push(
            `    lift_system_operator(site, ${componentLocalExpression(component)}; ` +
            "memory_budget=MEMORY_BUDGET);",
          );
          lines.push(`    cache=${observableGeometry},`);
          lines.push(")");
        } else {
          lines.push(
            `${observableName} = ` +
            `collective_spin(${observableBasis}, ${symbol}; cache=${observableGeometry})`,
          );
        }
      }
      if (parsed.observableMode === "collective-plan") {
        lines.push("# The prepared plan is reusable for the same observable on further states.");
        const localObservable = emit(parsed.observable.ast, "local");
        const preparedLocalObservable = isLocalPseudomode
          ? `lift_system_operator(site, ${localObservable}; memory_budget=MEMORY_BUDGET)`
          : localObservable;
        lines.push("observable_plan = CollectiveObservablePlan(");
        lines.push(`    ${observableBasis}, ${preparedLocalObservable};`);
        lines.push(`    cache=${observableGeometry},`);
        lines.push(")");
        lines.push(`observable_value = collective_expectation(${observableState}, observable_plan)`);
      } else if (parsed.observableMode === "single-site-plan") {
        lines.push("# PI symmetry makes every one-site marginal expectation identical.");
        const localObservable = emit(parsed.observable.ast, "local");
        const preparedLocalObservable = isLocalPseudomode
          ? `lift_system_operator(site, ${localObservable}; memory_budget=MEMORY_BUDGET)`
          : localObservable;
        lines.push("observable_plan = CollectiveObservablePlan(");
        lines.push(`    ${observableBasis}, ${preparedLocalObservable};`);
        lines.push(`    cache=${observableGeometry},`);
        lines.push(")");
        lines.push(`observable_value = collective_expectation(${observableState}, observable_plan) / N`);
      } else {
        lines.push("# Polynomial collective observable, constructed only in PI coordinates.");
        lines.push(`observable = ${emit(parsed.observable.ast, "observable-collective")}`);
        lines.push("# expectation uses a Hilbert--Schmidt convention, hence the explicit adjoint");
        lines.push("# computes tr(observable * rho), also for non-Hermitian expressions.");
        lines.push(`observable_value = expectation(${observableState}, adjoint(observable))`);
      }
      lines.push("println(\"steady-state observable = \", observable_value)");
      }

      const stateAnalysisRequested =
        parsed.analysis.purity ||
        parsed.analysis.entropy ||
        parsed.analysis.oneBodyRDM ||
        parsed.analysis.qfiAxis !== "none";
      if (stateAnalysisRequested) {
        lines.push("");
        lines.push("# Optional analyses refer to the physical systems, not retained pseudomodes.");
        if (isLocalPseudomode) {
          lines.push("analysis_trace_plan = pseudomode_trace_plan(");
          lines.push("    site; memory_budget=MEMORY_BUDGET)");
          lines.push("analysis_trace_workspace =");
          lines.push("    LocalFactorTraceWorkspace(analysis_trace_plan)");
          lines.push("analysis_state = trace_pseudomodes(");
          lines.push("    rho_ss, site;");
          lines.push("    plan=analysis_trace_plan,");
          lines.push("    workspace=analysis_trace_workspace,");
          lines.push("    memory_budget=MEMORY_BUDGET, check=false,");
          lines.push(")");
          lines.push("analysis_basis = analysis_trace_plan.output_basis");
        } else if (isGlobalPseudomode) {
          lines.push("analysis_state = rho_system");
          lines.push("analysis_basis = system_basis");
        } else {
          lines.push("analysis_state = rho_ss");
          lines.push("analysis_basis = basis");
        }
        if (parsed.analysis.purity) {
          lines.push("system_purity = purity(analysis_state)");
          lines.push("println(\"physical-system purity = \", system_purity)");
        }
        if (parsed.analysis.entropy) {
          lines.push("system_entropy = von_neumann_entropy(analysis_state; base=2)");
          lines.push("println(\"physical-system entropy (bits) = \", system_entropy)");
        }
        if (
          parsed.analysis.oneBodyRDM ||
          parsed.analysis.qfiAxis !== "none"
        ) {
          lines.push("analysis_geometry = OneBodyGeometry(analysis_basis)");
        }
        if (parsed.analysis.oneBodyRDM) {
          lines.push("analysis_rdm_workspace = OneBodyRDMWorkspace(");
          lines.push("    analysis_geometry, analysis_state;");
          lines.push("    memory_budget=MEMORY_BUDGET,");
          lines.push(")");
          lines.push("one_body_density_matrix = one_body_rdm(");
          lines.push("    analysis_state;");
          lines.push("    workspace=analysis_rdm_workspace, check=false,");
          lines.push(")");
          lines.push("println(\"one-body density matrix =\")");
          lines.push("show(stdout, \"text/plain\", one_body_density_matrix)");
          lines.push("println()");
        }
        if (parsed.analysis.qfiAxis !== "none") {
          const qfiLocal = componentLocalExpression(parsed.analysis.qfiAxis);
          lines.push("qfi_plan = CollectiveObservablePlan(");
          lines.push(`    analysis_basis, ${qfiLocal}; cache=analysis_geometry)`);
          lines.push("qfi_value = qfi(");
          lines.push("    analysis_state, qfi_plan; atol=STATE_VALIDATION_TOL)");
          lines.push(
            `println("collective-${parsed.analysis.qfiAxis} QFI = ", qfi_value)`,
          );
        }
      }
      lines.push("");
      lines.push(
        isTrajectory
          ? "# Residual and Monte Carlo error are diagnostics, not convergence or uniqueness certificates."
          : "# Convergence does not by itself prove that the stationary state is unique.",
      );
    }

    const nonlinearHamiltonian =
      nonlinearHamiltonianEntries.length > 0;
    const userTermCount =
      (linearHamiltonian.length ? 1 : 0) +
      (nonlinearHamiltonianEntries.length ? 1 : 0) +
      parsed.jumps.length;
    const architectureLabels = {
      pi: "ordinary PI ensemble",
      "local-pseudomode": "identical local pseudomodes",
      "global-pseudomode": "one shared global pseudomode",
    };
    let route;
    if (isDynamics && isTrajectory && isGlobalPseudomode) {
      route = "factorized composite trajectories with online observable statistics";
    } else if (isDynamics && isTrajectory) {
      route = isLocalPseudomode
        ? "PI-supersite trajectories with online observable statistics"
        : "term-resolved PI trajectories with online observable statistics";
    } else if (isDynamics) {
      route = isLocalPseudomode
        ? "matrix-free PI-supersite dynamics with state-free observable output"
        : "matrix-free PI dynamics with state-free observable output";
    } else if (isSpectrum) {
      route = isGlobalPseudomode
        ? "factorized automatic matrix-free selected spectrum"
        : "memory-guarded automatic dense or matrix-free selected spectrum";
    } else if (isGap) {
      route = isGlobalPseudomode
        ? "factorized matrix-free largest-real Krylov gap"
        : "matrix-free largest-real Krylov gap with certification metadata";
    } else if (isLocalPseudomode) {
      route = isTrajectory
        ? "PI-supersite quantum trajectories with streaming path reduction"
        : "PI supersite with a memory-guarded automatic backend";
    } else if (isGlobalPseudomode) {
      route = "factorized composite model with matrix-free GMRES";
    } else if (isTrajectory) {
      route = "term-resolved PI quantum trajectories with streaming path reduction";
    } else {
      route = nonlinearHamiltonian
        ? "automatic backend with compressed collective PI operators"
        : "automatic backend with prepared one-body kernels";
    }
    if (isExperiment) {
      route =
        `typed PIExperiment planning and verified_solve over ${route}`;
    }
    const methodLabels = {
      "steady-state": isTrajectory
        ? "quantum-trajectory steady-state estimate"
        : "deterministic stationary-state solve",
      "steady-observable": isTrajectory
        ? "quantum-trajectory steady-state estimate"
        : "deterministic stationary-state solve",
      "dynamics-observable": isTrajectory
        ? "quantum-trajectory observable dynamics"
        : "deterministic observable dynamics",
      "liouvillian-spectrum": "selected Liouvillian spectrum",
      "liouvillian-gap": "certification-aware Liouvillian gap",
    };
    const code = `${lines.join("\n")}\n`;
    const summary = {
      terms: userTermCount,
      jumps: parsed.jumps.length,
      target: parsed.target,
      calculation: parsed.calculation,
      method: methodLabels[parsed.calculation],
      workflow: parsed.workflow,
      architecture: parsed.architecture,
      topology: architectureLabels[parsed.architecture],
      cutoff: parsed.pseudomode ? parsed.pseudomode.nmax : null,
      route,
      representation: resources.representation,
      coordinates: resources.exactCoordinateCount,
      coordinateFormula: resources.coordinateFormula,
      oneComplexVectorBytes: resources.oneComplexVectorBytes,
      memoryBudgetBytes: resources.memoryBudgetBytes,
    };
    const manifest = manifestFor(parsed, summary, resources);
    const stem = generatedStem(parsed);
    const manifestText = `${JSON.stringify(manifest, null, 2)}\n`;
    const readme = readmeFor(stem, parsed, resources);
    return {
      code,
      warnings: parsed.warnings,
      summary,
      manifest,
      manifestText,
      resources,
      bundle: {
        stem,
        files: [
          {
            name: `${stem}.jl`,
            mediaType: "text/x-julia;charset=utf-8",
            contents: code,
          },
          {
            name: `${stem}.json`,
            mediaType: "application/json;charset=utf-8",
            contents: manifestText,
          },
          {
            name: `${stem}_README.txt`,
            mediaType: "text/plain;charset=utf-8",
            contents: readme,
          },
        ],
      },
      normalized: {
        hamiltonian: parsed.hamiltonian ? parsed.hamiltonian.normalized : "",
        observable: parsed.observable ? parsed.observable.normalized : "",
        pseudomodeCoupling: parsed.pseudomode
          ? parsed.pseudomode.couplingOperator.normalized
          : "",
      },
    };
  }

  return {
    VERSION,
    GeneratorError,
    normalizeLatex,
    parseFormula,
    analyze,
    generate,
  };
});
