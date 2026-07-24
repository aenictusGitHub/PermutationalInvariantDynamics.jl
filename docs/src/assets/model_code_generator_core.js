(function (root, factory) {
  "use strict";
  const api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  }
  root.PIDModelCodeGenerator = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const VERSION = "1.0.0";
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
    const target = config.target === "steady" ? "steady" : "expectation";
    const parameters = new Set();
    const hamiltonianComponents = new Set();
    const observableComponents = new Set();
    const warnings = [
      "The generated code checks convergence and state validity, but it does not certify uniqueness of the stationary state.",
    ];

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
      const kind = input.kind === "collective" ? "collective" : "local";
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
      const rate = parseFormula(String(input.rate || "1"), `${field} rate`);
      const rateInfo = analyze(rate.ast, `${field} rate`);
      if (rateInfo.kind !== "scalar") fail(`${field} rate`, "A jump rate must be scalar.");
      const operatorParameters = new Set();
      collectParameters(operator.ast, operatorParameters);
      collectParameters(operator.ast, parameters);
      collectParameters(rate.ast, parameters);
      jumps.push({ kind, operator, rate, field, operatorParameters });
    }

    if (!hamiltonian && !jumps.length) {
      fail("model", "Add a Hamiltonian or at least one dissipative channel.");
    }
    if (!jumps.length) {
      warnings.push("A Hamiltonian-only generator normally has many stationary states; add dissipation or analyse the stationary subspace.");
    } else if (jumps.every((jump) => jump.kind === "collective")) {
      warnings.push("Only collective channels were selected. With the complete PI basis, Schur-sector populations are conserved and the stationary state is generally nonunique.");
    }

    let observable = null;
    let observableMode = null;
    if (target === "expectation") {
      observable = parseFormula(String(config.observable || ""), "observable");
      const observableInfo = analyze(observable.ast, "observable");
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
        warnings.push("The selected observable may be non-Hermitian, so its expectation value can be complex.");
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
    if (observable) {
      validateScalarSubexpressions(
        observable.ast, values.values, particleCount, "observable");
    }
    if (particleCount > 1000) {
      warnings.push("Large N was requested. Inspect recommend_solver(model; task=:steady_state) before running.");
    }
    return {
      particleCount,
      localDimension,
      target,
      hamiltonian,
      jumps,
      observable,
      observableMode,
      parameters,
      parameterValues: values,
      hamiltonianComponents,
      observableComponents,
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

  function generate(config) {
    const parsed = parseModel(config);
    const lines = [];
    lines.push("# Generated by the PermutationalInvariantDynamics.jl model assistant.");
    lines.push("# Review the PI assumption: every constituent must be physically equivalent");
    lines.push("# and every local term/channel must act identically on all constituents.");
    lines.push("# Local sum_i D[l_i] and collective D[sum_i l_i] channels are different.");
    lines.push("using PermutationalInvariantDynamics");
    lines.push("");
    lines.push(`N = ${parsed.particleCount}`);
    lines.push(`d = ${parsed.localDimension}`);
    lines.push("basis = PIBasis(N, d)  # complete PI basis; local noise may mix Schur sectors");
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

    if (parsed.hamiltonianComponents.size) {
      lines.push("");
      lines.push("# Shared one-body geometry avoids rebuilding nonlinear Hamiltonian maps.");
      lines.push("one_body_geometry = OneBodyGeometry(basis)");
      for (const component of parsed.hamiltonianComponents) {
        const [name, symbol] = componentDefinition(component);
        lines.push(`${name} = collective_spin(basis, ${symbol}; cache=one_body_geometry)`);
      }
    }

    const termLines = [];
    if (parsed.hamiltonian) {
      const entries = [];
      flattenAdditive(parsed.hamiltonian.ast, 1, entries);
      const linear = [];
      const nonlinear = [];
      for (const entry of entries) {
        const info = analyze(entry.node, "hamiltonian");
        (info.linear ? linear : nonlinear).push(entry);
      }
      if (linear.length) {
        const localExpression = signedExpression(linear, "local");
        termLines.push(`LocalHamiltonian(${localExpression}),`);
      }
      if (nonlinear.length) {
        lines.push("");
        lines.push("# Nonlinear collective Hamiltonian stays entirely in compressed PI coordinates.");
        lines.push(`H_collective = ${signedExpression(nonlinear, "collective")}`);
        termLines.push("DirectPIHamiltonian(H_collective),");
      }
    }

    if (parsed.jumps.length) {
      lines.push("");
      lines.push("# Jump seeds are d-by-d matrices; the term constructor performs the PI lift.");
      parsed.jumps.forEach((jump, index) => {
        const name = `jump_${index + 1}`;
        lines.push(`${name} = ${emit(jump.operator.ast, "local")}`);
        const constructor = jump.kind === "collective" ? "CollectiveJump" : "LocalJump";
        termLines.push(`${constructor}(${name}; rate=${emit(jump.rate.ast, "scalar")}),`);
      });
    }

    lines.push("");
    lines.push("# A tuple keeps the compiled kernel types concrete.");
    lines.push("terms = (");
    for (const termLine of termLines) lines.push(`    ${termLine}`);
    lines.push(")");
    lines.push("model = PIModel(basis, terms)");
    lines.push("");
    lines.push("# :auto selects the sparse/direct or matrix-free route from the compiled problem.");
    lines.push("# For a large run, inspect recommend_solver(model; task=:steady_state) first.");
    lines.push("prepared = compile(model; backend=:auto)");
    lines.push("steady = stationary_state(prepared; return_info=true)");
    lines.push("steady.info.converged || error(\"stationary solver did not converge\")");
    lines.push("rho_ss = steady.state");
    lines.push("validate_state(rho_ss)  # checks trace, Hermiticity, and positivity");
    lines.push("");
    lines.push("println(\"PI coordinates = \", pi_dimension(basis))");
    lines.push("println(\"stationary residual = \", steady.info.residual)");
    lines.push("println(\"stationary trace error = \", steady.info.trace_error)");

    if (parsed.target === "expectation") {
      lines.push("");
      const needsObservableGeometry =
        parsed.observableMode === "collective-plan" ||
        parsed.observableMode === "single-site-plan" ||
        parsed.observableComponents.size > 0;
      if (needsObservableGeometry && !parsed.hamiltonianComponents.size) {
        lines.push("# Observable geometry is prepared only after the stationary solve.");
        lines.push("one_body_geometry = OneBodyGeometry(basis)");
      }
      for (const component of parsed.observableComponents) {
        if (parsed.hamiltonianComponents.has(component)) continue;
        const [name, symbol] = componentDefinition(component);
        lines.push(`${name} = collective_spin(basis, ${symbol}; cache=one_body_geometry)`);
      }
      if (parsed.observableMode === "collective-plan") {
        lines.push("# The prepared plan is reusable for the same observable on further states.");
        lines.push(`observable_plan = CollectiveObservablePlan(basis, ${emit(parsed.observable.ast, "local")}; cache=one_body_geometry)`);
        lines.push("observable_value = collective_expectation(rho_ss, observable_plan)");
      } else if (parsed.observableMode === "single-site-plan") {
        lines.push("# PI symmetry makes every one-site marginal expectation identical.");
        lines.push(`observable_plan = CollectiveObservablePlan(basis, ${emit(parsed.observable.ast, "local")}; cache=one_body_geometry)`);
        lines.push("observable_value = collective_expectation(rho_ss, observable_plan) / N");
      } else {
        lines.push("# Polynomial collective observable, constructed only in PI coordinates.");
        lines.push(`observable = ${emit(parsed.observable.ast, "collective")}`);
        lines.push("# expectation uses a Hilbert--Schmidt convention, hence the explicit adjoint");
        lines.push("# computes tr(observable * rho_ss), also for non-Hermitian expressions.");
        lines.push("observable_value = expectation(rho_ss, adjoint(observable))");
      }
      lines.push("println(\"steady-state observable = \", observable_value)");
    }
    lines.push("");
    lines.push("# Convergence does not by itself prove that the stationary state is unique.");

    const nonlinearHamiltonian = parsed.hamiltonian &&
      !analyze(parsed.hamiltonian.ast, "hamiltonian").linear;
    return {
      code: `${lines.join("\n")}\n`,
      warnings: parsed.warnings,
      summary: {
        terms: termLines.length,
        jumps: parsed.jumps.length,
        target: parsed.target,
        route: nonlinearHamiltonian
          ? "automatic backend with compressed collective PI operators"
          : "automatic backend with prepared one-body kernels",
      },
      normalized: {
        hamiltonian: parsed.hamiltonian ? parsed.hamiltonian.normalized : "",
        observable: parsed.observable ? parsed.observable.normalized : "",
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
