/**
 * tierBadgeText 最小检查（P0-1 三层标注）—— node 直接跑，无需测试框架：
 *
 *     node lib/documents/tier-label.check.ts
 *
 * 覆盖 PRD §6 的三条标注断言 + 「缺名退化、不出现 undefined / 裸分隔符」边界。
 * 非零退出码即失败。
 */

import { tierBadgeText } from "./tier-label.ts";

let failed = 0;

function eq(actual: string, expected: string, label: string): void {
  if (actual === expected) {
    console.log(`PASS ${label} → ${actual}`);
    return;
  }
  failed += 1;
  console.error(
    `FAIL ${label} → ${JSON.stringify(actual)}（期望 ${JSON.stringify(expected)}）`
  );
}

// PRD §6：department=公司名+部门名；tenant=公司名；private=「个人」
eq(tierBadgeText("department", "部门", "测试公司1", "营销部门"), "部门 · 测试公司1 · 营销部门", "部门文档");
eq(tierBadgeText("tenant", "公司", "测试公司1", null), "公司 · 测试公司1", "公司文档");
eq(tierBadgeText("private", "个人", "测试公司1", "营销部门"), "个人", "个人文档不带公司名");

// 边界：缺名退化为仅层级词，不得出现 undefined / 裸分隔符
eq(tierBadgeText("department", "部门", null, "营销部门"), "部门 · 营销部门", "缺公司名");
eq(tierBadgeText("department", "部门", "测试公司1", ""), "部门 · 测试公司1", "空部门名");
eq(tierBadgeText("tenant", "公司", "", null), "公司", "公司名缺失仅层级词");
eq(tierBadgeText("tenant", "公司", "   ", undefined), "公司", "纯空白公司名");
eq(tierBadgeText("tenant", "公司", "undefined", null), "公司", "字面量 undefined 视为缺失");
eq(tierBadgeText("tenant", "公司", "null", null), "公司", "字面量 null 视为缺失");
eq(tierBadgeText(undefined, undefined, undefined, undefined), "个人", "缺省参数");

// 不得残留裸分隔符（"· " 结尾 / "· ·" / 以 "·" 开头）
for (const [level, label, company, dept] of [
  ["department", "部门", null, null],
  ["department", "部门", null, "营销部门"],
  ["tenant", "公司", null, null],
] as const) {
  const text = tierBadgeText(level, label, company, dept);
  if (text.includes("undefined") || text.includes("· ·") || /·\s*$/.test(text) || text.startsWith("·")) {
    failed += 1;
    console.error(`FAIL 裸分隔符/脏值：${JSON.stringify(text)}`);
  } else {
    console.log(`PASS 无裸分隔符 → ${text}`);
  }
}

if (failed > 0) {
  console.error(`\n${failed} 项失败`);
  process.exitCode = 1;
} else {
  console.log("\n全部通过。");
}
