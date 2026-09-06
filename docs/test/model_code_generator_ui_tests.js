(function () {
  "use strict";
  // Exercise the real UI and page markup without a browser dependency in CI.
  // This fixture implements only DOM structure/events; parsing and generation
  // always use the production core. Visual behavior is checked in a browser.
  const node = typeof require === "function";
  const base = node ? require("path").join(__dirname, "../..") : ".";
  const source = (path) => node
    ? require("fs").readFileSync(`${base}/${path}`, "utf8")
    : readFile(`${base}/${path}`);
  const api = node ? require("../src/assets/model_code_generator_core.js") : PIDModelCodeGenerator;
  const uiPath = `${base}/docs/src/assets/model_code_generator_ui.js`;
  const markup = source("docs/src/model_code_generator.md").split("```@raw html\n")[1].split("```")[0];
  const key = "PermutationalInvariantDynamics.modelCodeGenerator.manifest.v1";
  let passed = 0;
  function assert(condition, message) {
    if (!condition) throw new Error(message);
    passed += 1;
  }
  function decode(text) {
    return text.replace(/&#(\d+);/g, (_, value) => String.fromCodePoint(Number(value)))
      .replace(/&quot;/g, '"').replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&amp;/g, "&");
  }
  class Element {
    constructor(tag) {
      this.tagName = tag.toUpperCase();
      this.attributes = new Map();
      this.children = [];
      this.listeners = [];
      this.classList = { add: (name) => { this.className += ` ${name}`; } };
    }
    setAttribute(name, value) { this.attributes.set(name, String(value)); }
    getAttribute(name) { return this.attributes.get(name) ?? null; }
    removeAttribute(name) { this.attributes.delete(name); }
    get value() {
      if (this.tagName === "SELECT") {
        const options = this.querySelectorAll("option");
        const value = this._value ?? (options[0] ? options[0].value : "");
        return options.some((option) => option.value === value) ? value : "";
      }
      return this._value ?? this.getAttribute("value") ?? (this.tagName === "TEXTAREA" ? this.textContent : "");
    }
    set value(value) { this._value = String(value); }
    get textContent() { return (this._text || "") + this.children.map((child) => child.textContent).join(""); }
    set textContent(text) { this.replaceChildren(); this._text = String(text); }
    append(...children) { for (const child of children) { child.parent = this; this.children.push(child); } }
    replaceChildren(...children) {
      for (const child of this.children) child.parent = null;
      this.children = []; this._text = ""; this.append(...children);
    }
    remove() {
      if (this.parent) this.parent.children.splice(this.parent.children.indexOf(this), 1);
      this.parent = null;
    }
    insertAdjacentElement(position, child) {
      if (position !== "afterend") throw new Error("Unexpected insertion position");
      this.parent.children.splice(this.parent.children.indexOf(this) + 1, 0, child);
      child.parent = this.parent;
    }
    matches(selector) {
      return selector.split(",").some((part) => {
        const match = part.trim();
        if (match[0] === "#") return this.id === match.slice(1);
        if (match[0] === ".") return this.className.split(/\s+/).includes(match.slice(1));
        const attr = /^\[([^=]+)='([^']*)'\]$/.exec(match);
        if (attr) return this.getAttribute(attr[1]) === attr[2];
        return this.tagName.toLowerCase() === match;
      });
    }
    querySelectorAll(selector) {
      return this.children.flatMap((child) => [
        ...(child.matches(selector) ? [child] : []), ...child.querySelectorAll(selector),
      ]);
    }
    querySelector(selector) { return this.querySelectorAll(selector)[0] || null; }
    addEventListener(type, handler, capture) { this.listeners.push({ type, handler, capture: !!capture }); }
    emit(type) {
      const event = { type, target: this, preventDefault() {} };
      const ancestors = [];
      for (let parent = this.parent; parent; parent = parent.parent) ancestors.push(parent);
      const results = [];
      const fire = (element, capture) => {
        for (const listener of element.listeners) {
          if (listener.type === type && listener.capture === capture) results.push(listener.handler(event));
        }
      };
      for (const ancestor of ancestors.slice().reverse()) fire(ancestor, true);
      fire(this, true); fire(this, false);
      for (const ancestor of ancestors) fire(ancestor, false);
      return Promise.all(results);
    }
    click() { if (!this.disabled) return this.emit("click"); }
    focus() { document.activeElement = this; }
    select() {}
  }
  for (const name of ["id", "type", "className"]) {
    const attribute = name === "className" ? "class" : name;
    Object.defineProperty(Element.prototype, name, {
      get() { return this.getAttribute(attribute) || ""; },
      set(value) { this.setAttribute(attribute, value); },
    });
  }
  function boot(manifest, hash) {
    const body = new Element("body");
    const stack = [body];
    const tokens = markup.match(/<!--[\s\S]*?-->|<[^>]*>|[^<]+/g);
    for (const token of tokens) {
      if (token.startsWith("<!--")) continue;
      if (token.startsWith("</")) { stack.pop(); continue; }
      if (!token.startsWith("<")) { stack[stack.length - 1]._text = (stack[stack.length - 1]._text || "") + decode(token); continue; }
      const tag = /^<([\w-]+)/.exec(token)[1];
      const child = new Element(tag);
      for (const attr of token.slice(tag.length + 1, -1).matchAll(/([\w-]+)(?:="([^"]*)")?/g)) {
        child.setAttribute(attr[1], decode(attr[2] || ""));
        if (["hidden", "disabled", "checked", "required"].includes(attr[1])) child[attr[1]] = true;
      }
      stack[stack.length - 1].append(child);
      if (!["input", "br", "hr"].includes(tag)) stack.push(child);
    }
    const storage = new Map(manifest ? [[key, JSON.stringify(manifest)]] : []);
    globalThis.document = {
      body, readyState: "complete", documentElement: new Element("html"),
      getElementById: (id) => body.querySelector(`#${id}`),
      createElement: (tag) => new Element(tag),
    };
    globalThis.window = {
      PIDModelCodeGenerator: api,
      localStorage: { getItem: (key) => storage.get(key), setItem: (key, value) => storage.set(key, value) },
      location: { hash: hash || "", href: "http://localhost/model_code_generator/" },
      atob: () => { throw new Error("invalid base64"); },
    };
    if (node) { delete require.cache[require.resolve(uiPath)]; require(uiPath); }
    else load(uiPath);
    return {
      q: (id) => body.querySelector(`#pid-${id}`),
      saved: () => JSON.parse(storage.get(key)),
      edit(id, value, type) {
        const element = this.q(id);
        element.value = value;
        return element.emit(type || "input");
      },
      generate() { return this.q("generator-form").emit("submit"); },
    };
  }
  const exports = ["copy-code", "download-code", "download-pluto", "download-bundle", "copy-share-link"];
  function exportsDisabled(app, expected, message) {
    assert(exports.every((name) => app.q(name).disabled === expected), message);
  }
  async function run() {
    let app = boot();
    exportsDisabled(app, false, "starter exports enabled");
    assert(app.q("generated-readme").textContent.includes("julia --project"), "run guide is available");
    const original = app.saved();
    await app.edit("particle-count", "9");
    exportsDisabled(app, true, "input invalidates every export, including share");
    assert(!app.q("generated-code").textContent.includes("using PermutationalInvariantDynamics"), "outdated code removed");
    assert(app.q("run-guide").hidden, "outdated guide hidden");
    assert(app.saved().model.particles === original.model.particles, "dirty edits do not overwrite saved model");
    await app.generate();
    exportsDisabled(app, false, "regeneration enables current exports");
    assert(app.saved().model.particles === 9, "regeneration saves current model");
    await app.edit("particle-count", "");
    await app.generate();
    exportsDisabled(app, true, "invalid generation disables sharing as well as downloads");
    assert(document.activeElement === app.q("particle-count"), "submission focuses invalid input");
    assert(app.q("particle-count").getAttribute("aria-describedby").includes("pid-field-error"), "error associated with input");
    assert(app.q("field-error").textContent.includes("positive integer"), "error displayed beside input");
    await app.edit("particle-count", "10");
    assert(!app.q("field-error"), "editing clears obsolete field error");
    await app.generate();
    await app.q("undo-model").click();
    assert(app.q("particle-count").value === "9", "undo restores previous successful generation");
    await app.q("add-local-jump").click();
    exportsDisabled(app, true, "adding a channel invalidates output");
    await app.generate();
    await app.q("jump-list").querySelector(".pid-remove-jump").click();
    exportsDisabled(app, true, "removing a channel invalidates output");
    await app.generate();
    await app.q("add-scan-axis").click();
    exportsDisabled(app, true, "adding an axis invalidates output");
    await app.generate();
    await app.q("scan-axes").querySelector(".pid-remove-scan-axis").click();
    exportsDisabled(app, true, "removing an axis invalidates output");

    app = boot();
    await app.edit("preset", "spectrum", "change");
    exportsDisabled(app, false, "selector regenerates after capture-phase invalidation");
    await app.edit("memory-budget", "64");
    await app.generate();
    const spectrum = app.saved();
    app = boot(spectrum);
    assert(app.q("memory-budget").value === "64", "restore preserves explicit memory budget");
    assert(app.q("spectrum-target").value === "near-zero", "manifest target maps to HTML option");
    assert(app.q("generated-code").textContent.includes("MEMORY_BUDGET = 64 * 1024^2"), "restored code preserves resource budget");
    await app.generate();
    assert(JSON.stringify(app.saved()) === JSON.stringify(spectrum), "restored form regenerates the same normalized manifest\n" + JSON.stringify([spectrum, app.saved()]));
    const before = app.q("generated-code").textContent;
    const invalid = JSON.parse(JSON.stringify(spectrum));
    invalid.model.hamiltonian = "eval(1)";
    app.q("manifest-file").files = [{ name: "invalid.json", size: 100, text: async () => JSON.stringify(invalid) }];
    await app.q("manifest-file").emit("change");
    assert(app.q("generated-code").textContent === before, "invalid imported physics preserves output");
    assert(app.q("memory-budget").value === "64", "invalid import preserves form");
    assert(JSON.stringify(app.saved()) === JSON.stringify(spectrum), "invalid import preserves local save");
    assert(app.q("generator-status").className.includes("error"), "invalid import reports failure");
    exportsDisabled(app, false, "current valid model remains exportable after rejected import");
    app = boot(spectrum, "#pid-model=broken");
    assert(app.q("generator-status").textContent.includes("share link is invalid"), "broken share error survives starter generation");
    assert(JSON.stringify(app.saved()) === JSON.stringify(spectrum), "broken share link preserves local save");
    assert(app.q("particle-count").value === "8", "broken link explicitly shows starter");

    const presets = app.q("preset").querySelectorAll("option").map((option) => option.value);
    for (const preset of presets) {
      app = boot();
      await app.edit("preset", preset, "change");
      exportsDisabled(app, false, `${preset}: preset generates`);
      const saved = app.saved();
      app = boot(saved);
      exportsDisabled(app, false, `${preset}: saved preset restores`);
      await app.generate();
      assert(JSON.stringify(app.saved()) === JSON.stringify(saved), `${preset}: restored form round trip`);
    }
  }
  run().then(() => {
    const message = `model code generator UI tests: ${passed} passed`;
    if (typeof print === "function") print(message); else console.log(message);
  }).catch((error) => { throw error; });
})();
