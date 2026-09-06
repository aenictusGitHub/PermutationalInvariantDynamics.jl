(function () {
  "use strict";
  const api = typeof PIDModelCodeGenerator !== "undefined"
    ? PIDModelCodeGenerator : require("../src/assets/model_code_generator_core.js");
  let passed = 0;
  function assert(condition, message) {
    if (!condition) throw new Error(message);
    passed += 1;
  }
  function rejects(action, field) {
    let error;
    try { action(); } catch (caught) { error = caught; }
    assert(error instanceof api.GeneratorError && error.field === field, `Expected ${field} rejection; received ${error}`);
  }
  const base = {
    N: 2, d: 2, calculation: "steady-observable", hamiltonian: "J_x",
    jumps: [{ kind: "local", operator: "j_-", rate: "0.1" }], observable: "J_z/N",
  };
  for (const invalid of ["", "   ", null, false, true, [], [1], {}, Infinity, NaN]) {
    rejects(() => api.generate({ ...base, N: invalid }), "N");
    rejects(() => api.generate({ ...base, d: invalid }), "d");
    rejects(() => api.generate({ ...base, resources: { memoryBudgetMiB: invalid } }), "memory budget");
    rejects(() => api.generate({ ...base, calculation: "dynamics-observable", dynamics: { startTime: invalid } }), "dynamics start time");
    rejects(() => api.generate({ ...base, calculation: "liouvillian-spectrum", spectrum: { seed: invalid } }), "spectrum seed");
    rejects(() => api.generate({ ...base, steadyMethod: "trajectory", trajectory: { seed: invalid } }), "trajectory seed");
    rejects(() => api.generate({ ...base, architecture: "local-pseudomode", pseudomode: { nmax: invalid } }), "pseudomode cutoff");
    const manifest = api.generate(base).manifest;
    manifest.model.parameters = { unused: invalid };
    rejects(() => api.configurationFromManifest(manifest), "manifest parameter unused");
  }
  for (const setting of ["frequency", "damping", "thermalOccupation", "couplingStrength", "counterrotatingStrength", "couplingOperator"]) {
    const fields = {
      frequency: "frequency", damping: "damping", thermalOccupation: "thermal occupation",
      couplingStrength: "coupling strength", counterrotatingStrength: "counter-rotating strength",
      couplingOperator: "coupling operator",
    };
    rejects(() => api.generate({ ...base, architecture: "local-pseudomode", pseudomode: { [setting]: "" } }), `pseudomode ${fields[setting]}`);
  }
  rejects(() => api.generate({ ...base, jumps: [{ kind: "local", operator: "", rate: "1" }] }), "jump 1");
  rejects(() => api.generate({ ...base, jumps: [{ kind: "local", operator: "j_-", rate: "" }] }), "jump 1 rate");
  const dynamics = api.generate({ ...base, calculation: "dynamics-observable" });
  assert(dynamics.manifest.calculation.dynamics.startTime === 0, "omitted dynamics start retains documented default");
  assert(api.generate({ ...base, calculation: "liouvillian-spectrum", spectrum: { seed: 0 } }).manifest.calculation.spectrum.seed === 0, "explicit zero seed remains valid");
  assert(api.generate({ ...base, N: "2", d: "2" }).summary.coordinates === "10", "numeric form strings remain supported");
  const normalized = api.normalizeLatex(String.raw`\Omega J_x + \chi J_z^2`, "hamiltonian");
  assert(api.normalizeLatex(normalized, "hamiltonian") === normalized, "normalization is idempotent");

  const generated = api.generate(base);
  const archived = api.bundleArchive(generated.bundle);
  assert(archived.name === `${generated.bundle.stem}.zip` && archived.mediaType === "application/zip", "archive metadata");
  const bytes = archived.contents;
  const view = new DataView(bytes.buffer);
  const u16 = (offset) => view.getUint16(offset, true);
  const u32 = (offset) => view.getUint32(offset, true);
  const text = (start, count) => decodeURIComponent(Array.from(bytes.slice(start, start + count), (byte) => `%${byte.toString(16).padStart(2, "0")}`).join(""));
  const end = bytes.length - 22;
  assert(u32(end) === 0x06054b50 && u16(end + 10) === 4, "ZIP end record has all four artifacts");
  let entry = u32(end + 16);
  const directoryStart = entry;
  for (const file of generated.bundle.files) {
    assert(u32(entry) === 0x02014b50, "central directory entry");
    const local = u32(entry + 42);
    assert(u32(local) === 0x04034b50 && u16(local + 8) === 0, "stored local file header");
    const nameLength = u16(local + 26);
    assert(text(local + 30, nameLength) === file.name, "archive filename preserved");
    assert(text(local + 30 + nameLength, u32(local + 18)) === file.contents, "artifact text preserved byte for byte");
    assert(u32(local + 14) === u32(entry + 16), "local and central CRC agree");
    entry += 46 + u16(entry + 28);
  }
  assert(entry === end && entry - directoryStart === u32(end + 12), "central directory size and offsets");
  const repeated = api.bundleArchive(generated.bundle).contents;
  assert(bytes.length === repeated.length && bytes.every((byte, i) => byte === repeated[i]), "archive is reproducible");
  const known = api.bundleArchive({ stem: "crc", files: [{ name: "check.txt", contents: "123456789" }] }).contents;
  assert(new DataView(known.buffer).getUint32(14, true) === 0xcbf43926, "known CRC-32 check vector");
  const unicode = api.bundleArchive({ stem: "unicode", files: [{ name: "text.txt", contents: "α ≤ 𝛾\n" }] }).contents;
  assert(Array.from(unicode.slice(38, 50)).join(",") === "206,177,32,226,137,164,32,240,157,155,190,10", "UTF-8 includes supplementary Unicode characters");
  for (const name of ["../escape.jl", "/root.jl", "dir/file.jl", "dir\\file.jl"]) {
    rejects(() => api.bundleArchive({ stem: "safe", files: [{ name, contents: "x" }] }), "bundle");
  }
  rejects(() => api.bundleArchive({ stem: "safe", files: [{ name: "same", contents: "a" }, { name: "same", contents: "b" }] }), "bundle");
  rejects(() => api.bundleArchive({ stem: "safe", files: [{ name: "large", contents: "x".repeat(6 * 1024 * 1024) }] }), "bundle");
  const output = `model code generator reliability tests: ${passed} passed`;
  if (typeof print === "function") print(output); else console.log(output);
})();
