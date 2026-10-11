const { test } = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const ts = require("typescript");

// Run the actual TypeScript modules without adding a test framework.
function loadTs(file) {
  const module = { exports: {} };
  const code = ts.transpileModule(fs.readFileSync(file, "utf8"), {
    compilerOptions: { module: ts.ModuleKind.CommonJS },
  }).outputText;
  const localRequire = (name) => name.startsWith("@/")
    ? loadTs(path.resolve(__dirname, "..", name.slice(2) + ".ts"))
    : require(name);
  vm.runInThisContext(`(function(require, module, exports) {${code}\n})`)(localRequire, module, module.exports);
  return module.exports;
}

const { streamQuery } = loadTs(path.resolve(__dirname, "../lib/api/query.ts"));
const { getStreamApiBase } = loadTs(path.resolve(__dirname, "../lib/api/client.ts"));

test("streaming uses the same-origin proxy by default", () => {
  assert.equal(getStreamApiBase(), process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "/api");
});

async function runStream(t, content, split = false) {
  const original = global.fetch;
  t.after(() => { global.fetch = original; });
  const bytes = new TextEncoder().encode(content);
  global.fetch = async () => new Response(new ReadableStream({
    start(controller) {
      if (split) for (const byte of bytes) controller.enqueue(Uint8Array.of(byte));
      else controller.enqueue(bytes);
      controller.close();
    },
  }), { headers: { "content-type": "text/event-stream" } });
  const received = { tokens: [], done: [] };
  const promise = streamQuery({
    query: "question",
    onToken: (token) => received.tokens.push(token),
    onSources: () => {},
    onDone: (id) => received.done.push(id),
    onError: (error) => { throw error; },
  });
  return { promise, received };
}

test("truncated answers cannot be marked complete", async (t) => {
  const { promise, received } = await runStream(t, 'data: {"type":"token","content":"partial"}\n\n');
  await assert.rejects(promise, /中断/);
  assert.deepEqual(received.done, []);
  assert.deepEqual(received.tokens, ["partial"]);
});

test("UTF-8 byte splits and CRLF preserve answer and conversation ID", async (t) => {
  const { promise, received } = await runStream(t,
    'data: {"type":"token","content":"营业收入"}\r\n\r\ndata: {"type":"done","conversation_id":"c1"}\r\n\r\ndata: [DONE]\r\n', true);
  await promise;
  assert.deepEqual(received.tokens, ["营业收入"]);
  assert.deepEqual(received.done, ["c1"]);
});

test("completion marker without trailing newline is accepted", async (t) => {
  const { promise, received } = await runStream(t, "data: [DONE]");
  await promise;
  assert.equal(received.done.length, 1);
});
