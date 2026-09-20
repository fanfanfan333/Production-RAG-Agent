import { apiFetch } from "@/lib/api/client";
import type { CompanyDeletionImpact } from "@/lib/types";

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

// ── 删除公司（平台管理员，不可恢复）──────────────────────────────────────────

/** 后端 snake_case → 前端 camelCase（归一化只有一份，组件里不出现 snake_case）。 */
function normalizeCompanyImpact(raw: Record<string, unknown>): CompanyDeletionImpact {
  const company = (raw.company ?? {}) as Record<string, unknown>;
  const deleted = (raw.deleted ?? {}) as Record<string, unknown>;
  const kept = (raw.kept ?? {}) as Record<string, unknown>;
  return {
    company: {
      id: String(company.id ?? ""),
      name: String(company.name ?? ""),
      isTest: Boolean(company.is_test ?? false),
    },
    deleted: {
      members: Number(deleted.members ?? 0),
      documents: Number(deleted.documents ?? 0),
      documentsPrivate: Number(deleted.documents_private ?? 0),
      documentsDepartment: Number(deleted.documents_department ?? 0),
      documentsTenant: Number(deleted.documents_tenant ?? 0),
      conversations: Number(deleted.conversations ?? 0),
      messages: Number(deleted.messages ?? 0),
      collections: Number(deleted.collections ?? 0),
      feedback: Number(deleted.feedback ?? 0),
      vectors: Number(deleted.vectors ?? 0),
    },
    kept: {
      staffRequests: Number(kept.staff_requests ?? 0),
      shareRequests: Number(kept.share_requests ?? 0),
      badCases: Number(kept.bad_cases ?? 0),
      crossTenantDocuments: Number(kept.cross_tenant_documents ?? 0),
    },
  };
}

/**
 * 删除公司前的影响预检。权限与可删范围由后端判定（只有平台管理员可用）——
 * 前端据此决定按钮是否可点，但真正的守门永远在后端（预检本身也会 403 / 404）。
 */
export async function previewCompanyDeletion(
  companyId: string
): Promise<CompanyDeletionImpact> {
  const raw = await apiFetch<Record<string, unknown>>(
    `/companies/${encodeURIComponent(companyId)}/deletion-preview`
  );
  return normalizeCompanyImpact(raw);
}

/**
 * 删除整家公司（不可恢复）：员工账号、三级文档、会话与向量全删，只留审计留痕。
 * 删除可能耗时（清向量 + 磁盘），故放宽超时，参照 ``deleteStaffMember``。
 */
export async function deleteCompany(
  companyId: string
): Promise<{ message: string; impact: CompanyDeletionImpact }> {
  const raw = await apiFetch<Record<string, unknown>>(
    `/companies/${encodeURIComponent(companyId)}`,
    { method: "DELETE", timeout: 120000 }
  );
  return {
    message: String(raw.message ?? "公司已删除"),
    impact: normalizeCompanyImpact(raw),
  };
}
