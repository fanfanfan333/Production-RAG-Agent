/**
 * 三层知识库徽标的文本组装（纯函数，**零依赖**）.
 *
 * 单独成模块的原因：这是"个人 / 部门 / 公司"标注的**唯一**拼接点，且零 import
 * 便于用 node 直接跑最小检查（见同目录 tier-label.check.ts），不必引入测试框架。
 *
 * 拼接规则（P0-1）：
 *   个人 → 「个人」                    （不显示公司名）
 *   公司 → 「公司 · 测试公司1」
 *   部门 → 「部门 · 测试公司1 · 营销部门」
 *
 * 边界：公司名 / 部门名缺失（null / 空串 / 纯空白 / 字面量 "undefined" "null"
 * "none"）一律**退化为仅层级词** —— 绝不渲染 `undefined` 或裸分隔符 `·`。
 */

export type BadgeTier = "private" | "department" | "tenant";

/** 层级 → 中文层级词（与后端 tenancy.ACCESS_LABELS 一致，仅作最后兜底）。 */
const TIER_WORDS: Record<BadgeTier, string> = {
  private: "个人",
  department: "部门",
  tenant: "公司",
};

/** 展示名清洗：空 / 纯空白 / 字面量占位符 → 空串（视为"缺失"）。 */
function cleanName(value?: string | null): string {
  const text = (value ?? "").trim();
  if (!text) return "";
  const lower = text.toLowerCase();
  if (lower === "undefined" || lower === "null" || lower === "none") return "";
  return text;
}

export function tierBadgeText(
  level?: BadgeTier,
  label?: string,
  companyName?: string | null,
  departmentName?: string | null
): string {
  const key: BadgeTier = level ?? "private";
  const base = cleanName(label) || TIER_WORDS[key] || TIER_WORDS.private;

  const company = cleanName(companyName);
  const department = cleanName(departmentName);

  // 层级词在前（保证"层级可辨识"），名字作为其后的补充信息。
  const extras: string[] = [];
  if (key === "department") {
    if (company) extras.push(company);
    if (department) extras.push(department);
  } else if (key === "tenant") {
    if (company) extras.push(company);
  }
  return [base, ...extras].join(" · ");
}
