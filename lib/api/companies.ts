import { apiFetch } from "@/lib/api/client";

/**
 * 公司注册表 API.
 *
 * 两个端点语义不同，刻意分开（与后端 /companies 路由一致）：
 *   fetchRegisteredCompanies —— 全部已注册公司（身份验证弹窗的公司下拉候选）
 *   fetchAccessibleCompanies —— 我可见范围内的公司 + 可见文档数
 *                                （文档页公司筛选候选，与列表/检索同源）
 */

export interface RegisteredCompany {
  companyId: string;
  companyName: string;
}

export interface AccessibleCompany extends RegisteredCompany {
  docCount: number;
}

/** 全部已注册公司（任何已登录用户可用；身份验证下拉候选）。 */
export async function fetchRegisteredCompanies(): Promise<RegisteredCompany[]> {
  const res = await apiFetch<{ items?: Record<string, unknown>[] }>("/companies");
  return (res.items ?? []).map((raw) => ({
    companyId: String(raw.company_id ?? ""),
    companyName: String(raw.company_name ?? raw.company_id ?? ""),
  }));
}

/**
 * 创建公司（平台管理员）.
 *
 * 公司名唯一（忽略大小写/空格变体），重名时后端 409「公司已存在」——
 * 错误由 :class:`ApiError` 抛出，调用方直接展示 ``err.message`` 即可。
 */
export async function createCompany(displayName: string): Promise<RegisteredCompany> {
  const raw = await apiFetch<Record<string, unknown>>("/companies", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ display_name: displayName }),
  });
  return {
    companyId: String(raw.company_id ?? ""),
    companyName: String(raw.company_name ?? ""),
  };
}

/**
 * 公司改名（平台管理员，须该公司创建者）.
 *
 * **``tenant_id`` 不变**：只改展示名，成员归属、文档归属、向量数据零迁移。
 */
export async function renameCompany(
  companyId: string,
  displayName: string
): Promise<RegisteredCompany> {
  const raw = await apiFetch<Record<string, unknown>>(
    `/companies/${encodeURIComponent(companyId)}`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ display_name: displayName }),
    }
  );
  return {
    companyId: String(raw.company_id ?? companyId),
    companyName: String(raw.company_name ?? displayName),
  };
}

/**
 * 我可见范围内的公司 + 各公司可见文档数（文档页公司筛选）。
 *
 * 平台管理员 = 其自建测试公司集合（无自建公司时为空列表）；
 * 普通成员 = 其所属公司。docCount 与 GET /documents?company_id= 同源。
 */
export async function fetchAccessibleCompanies(): Promise<AccessibleCompany[]> {
  const res = await apiFetch<Record<string, unknown>[] | { items?: Record<string, unknown>[] }>(
    "/companies/accessible"
  );
  const list = Array.isArray(res) ? res : (res.items ?? []);
  return list.map((raw) => ({
    companyId: String(raw.company_id ?? ""),
    companyName: String(raw.display_name ?? raw.company_name ?? raw.company_id ?? ""),
    docCount: Number(raw.doc_count ?? 0),
  }));
}
