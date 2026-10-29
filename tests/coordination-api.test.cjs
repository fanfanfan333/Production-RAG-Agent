const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const ts = require("typescript");

function loadTs(file) {
  const module = { exports: {} };
  const code = ts.transpileModule(fs.readFileSync(file, "utf8"), {
    compilerOptions: { module: ts.ModuleKind.CommonJS },
  }).outputText;
  const localRequire = (name) => name.startsWith("@/")
    ? loadTs(path.resolve(__dirname, "..", name.slice(2) + ".ts")) : require(name);
  vm.runInThisContext(`(function(require, module, exports) {${code}\n})`)(localRequire, module, module.exports);
  return module.exports;
}

test("task submission forwards the user's credential and idempotency key", async (t) => {
  const original = global.fetch;
  t.after(() => { global.fetch = original; });
  let captured;
  global.fetch = async (url, init) => {
    captured = { url, init };
    return new Response(JSON.stringify({ task: { id: "one", status: "queued" } }), { status: 202 });
  };
  const { submitCoordinationTask } = loadTs(path.resolve(__dirname, "../lib/api/coordination.ts"));
  const task = await submitCoordinationTask("query", "code", "user-credential", "request-key");
  assert.equal(task.status, "queued");
  assert.equal(captured.init.headers["X-ForgeFlow-Token"], "user-credential");
  assert.equal(JSON.parse(captured.init.body).idempotency_key, "request-key");
});
