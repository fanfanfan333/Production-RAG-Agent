const { spawnSync } = require("node:child_process");
const lock = require("../package-lock.json");

function checkAudit(report, packages, now = new Date()) {
  if (report.error || !report.metadata || !report.vulnerabilities) {
    throw new Error("Dependency audit did not return a valid report");
  }
  const failures = [];
  for (const [name, finding] of Object.entries(report.vulnerabilities)) {
    if (!["high", "critical"].includes(finding.severity)) continue;
    // shortcut: braces has no upstream fix; review this development-only exception by 2026-11-10.
    const known = new Set(["braces", "micromatch", "fast-glob", "@next/eslint-plugin-next", "eslint-config-next"]);
    const permitted = now < new Date("2026-11-10T00:00:00Z") && finding.severity === "high"
      && known.has(name) && finding.nodes.length > 0
      && finding.nodes.every((path) => packages[path]?.dev === true)
      && finding.via.every((cause) => typeof cause === "string" ? known.has(cause)
        : cause.url === "https://github.com/advisories/GHSA-vfj7-8cjw-p6xm");
    if (!permitted) failures.push(name);
  }
  if (failures.length) throw new Error(`Dependency gate failed: ${failures.join(", ")}`);
}

if (require.main === module) {
  const result = spawnSync(process.platform === "win32" ? "npm.cmd" : "npm", ["audit", "--json", "--registry=https://registry.npmjs.org"], { encoding: "utf8", shell: process.platform === "win32" });
  if (result.error || ![0, 1].includes(result.status)) throw result.error || new Error(result.stderr);
  const report = JSON.parse(result.stdout);
  checkAudit(report, lock.packages);
  console.log("Dependency gate passed; known development-only braces advisory expires 2026-11-10.");
}

module.exports = { checkAudit };
