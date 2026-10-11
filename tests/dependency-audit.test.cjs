const { test } = require("node:test");
const assert = require("node:assert/strict");
const { checkAudit } = require("../scripts/audit-dependencies.cjs");

const finding = { severity: "high", nodes: ["node_modules/braces"], via: [{ url: "https://github.com/advisories/GHSA-vfj7-8cjw-p6xm" }] };
const report = { metadata: {}, vulnerabilities: { braces: finding } };
const packages = { "node_modules/braces": { dev: true } };
const date = new Date("2026-10-10T00:00:00Z");

test("only the known development advisory is temporarily allowed", () => {
  assert.doesNotThrow(() => checkAudit(report, packages, date));
  assert.throws(() => checkAudit(report, {}, date));
  assert.throws(() => checkAudit(report, packages, new Date("2026-11-10T00:00:00Z")));
  assert.throws(() => checkAudit({ ...report, vulnerabilities: { braces: { ...finding, severity: "critical" } } }, packages, date));
  assert.throws(() => checkAudit({ ...report, vulnerabilities: { braces: { ...finding, via: [{ url: "https://example.com/new-advisory" }] } } }, packages, date));
  assert.throws(() => checkAudit({ error: "registry unavailable" }, packages, date));
});
