import { apiFetch } from "@/lib/api/client";
import type {
  CompanyOption,
  GrantableRole,
  IdentityStatus,
  MemberDeletionImpact,
  StaffMember,
  StaffProfile,
  StaffRequestItem,
  StaffRequestStatus,
  StaffSummary,
} from "@/lib/types";

/**
 * 企业身份验证与企业管理员后台 API.
 *
 * 后端出参是 snake_case，这里集中做一次映射 —— 与 share.ts / normalize.ts
 * 的约定一致（归一化只有一份，组件里不出现 snake_case）。
 */

function normalizeItem(raw: Record<string, unknown>): StaffRequestItem {
  return {
    id: String(raw.id ?? ""),
    applicantUsername: String(raw.applicant_username ?? ""),
    applicantDisplayName: (raw.applicant_display_name ?? null) as string | null,
    applicantLabel: String(
      raw.applicant_label ?? raw.applicant_username ?? "未知成员"
    ),
    companyName: String(raw.company_name ?? ""),
    companyId: String(raw.company_id ?? ""),
    departmentName: String(raw.department_name ?? ""),
    departmentId: String(raw.department_id ?? ""),
    duty: String(raw.duty ?? ""),
    status: String(raw.status ?? "pending") as StaffRequestStatus,
    reviewerUsername: (raw.reviewer_username ?? null) as string | null,
    reviewerTitle: (raw.reviewer_title ?? null) as string | null,
    reviewerName: (raw.reviewer_name ?? null) as string | null,
    reviewerLabel: (raw.reviewer_label ?? null) as string | null,
    reviewComment: (raw.review_comment ?? null) as string | null,
    grantedRole: (raw.granted_role ?? null) as string | null,
    grantedRoleLabel: (raw.granted_role_label ?? null) as string | null,
    createdAt: (raw.created_at ?? null) as string | null,
    reviewedAt: (raw.reviewed_at ?? null) as string | null,
    applicantSeen: Boolean(raw.applicant_seen ?? false),
    isMine: Boolean(raw.is_mine ?? false),
    canReview: Boolean(raw.can_review ?? false),
  };
}

function normalizeMember(raw: Record<string, unknown>): StaffMember {
  return {
    id: String(raw.id ?? ""),
    username: String(raw.username ?? ""),
    displayName: (raw.display_name ?? null) as string | null,
    name: String(raw.name ?? raw.username ?? ""),
    role: String(raw.role ?? "employee"),
    roleLabel: String(raw.role_label ?? "普通员工"),
    isAdmin: Boolean(raw.is_admin ?? false),
    isActive: Boolean(raw.is_active ?? true),
    companyId: String(raw.company_id ?? "default"),
    companyName: String(raw.company_name ?? raw.company_id ?? "default"),
    departmentId: (raw.department_id ?? null) as string | null,
    departmentName: (raw.department_name ?? null) as string | null,
    jobTitle: (raw.job_title ?? null) as string | null,
    identityStatus: String(raw.identity_status ?? "none") as IdentityStatus,
    authSource: String(raw.auth_source ?? "local"),
    createdAt: (raw.created_at ?? null) as string | null,
  };
}

function normalizeProfile(raw: Record<string, unknown>): StaffProfile {
  return {
    id: String(raw.id ?? ""),
    username: String(raw.username ?? ""),
    name: String(raw.name ?? raw.username ?? ""),
    displayName: (raw.display_name ?? null) as string | null,
    role: String(raw.role ?? "employee"),
    roleLabel: String(raw.role_label ?? "普通员工"),
    isAdmin: Boolean(raw.is_admin ?? false),
    isActive: Boolean(raw.is_active ?? true),
    companyId: String(raw.company_id ?? "default"),
    companyName: String(raw.company_name ?? raw.company_id ?? "default"),
    departmentId: (raw.department_id ?? null) as string | null,
    departmentName: (raw.department_name ?? null) as string | null,
    jobTitle: (raw.job_title ?? null) as string | null,
    authSource: String(raw.auth_source ?? "local"),
    identityStatus: String(raw.identity_status ?? "none") as IdentityStatus,
    grantableRoles: Array.isArray(raw.grantable_roles)
      ? (raw.grantable_roles as string[]).map(String)
      : [],
    canReview: Boolean(raw.can_review ?? false),
    canAdminister: Boolean(raw.can_administer ?? false),
    latestRequest: raw.latest_request
      ? normalizeItem(raw.latest_request as Record<string, unknown>)
      : null,
    createdAt: (raw.created_at ?? null) as string | null,
  };
}

// ── 我的身份 ──────────────────────────────────────────────────────────────────

export async function fetchStaffProfile(): Promise<StaffProfile> {
  const raw = await apiFetch<Record<string, unknown>>("/staff/me");
  return normalizeProfile(raw);
}

export async function fetchStaffSummary(): Promise<StaffSummary> {
  const raw = await apiFetch<Record<string, unknown>>("/staff/summary");
  return {
    identityStatus: String(raw.identity_status ?? "none") as IdentityStatus,
    pendingForMe: Number(raw.pending_for_me ?? 0),
    myPending: Number(raw.my_pending ?? 0),
    myDecidedUnseen: Number(raw.my_decided_unseen ?? 0),
    canReview: Boolean(raw.can_review ?? false),
    canAdminister: Boolean(raw.can_administer ?? false),
    totalBadge: Number(raw.total_badge ?? 0),
  };
}

export async function fetchGrantableRoles(): Promise<GrantableRole[]> {
  const raw = await apiFetch<{ grantable_roles?: { value: string; label: string }[] }>(
    "/staff/config"
  );
  return (raw.grantable_roles ?? []).map((item) => ({
    value: String(item.value),
    label: String(item.label),
  }));
}

// ── 身份验证申请 ──────────────────────────────────────────────────────────────

export async function createStaffRequest(params: {
  companyName: string;
  departmentName: string;
  duty: string;
}): Promise<{ request: StaffRequestItem; message: string }> {
  const res = await apiFetch<{ request: Record<string, unknown>; message: string }>(
    "/staff/requests",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        company_name: params.companyName,
        department_name: params.departmentName,
        duty: params.duty,
      }),
    }
  );
  return { request: normalizeItem(res.request), message: res.message };
}

export async function getMyStaffRequests(): Promise<StaffRequestItem[]> {
  const res = await apiFetch<{ items: Record<string, unknown>[] }>(
    "/staff/requests/mine"
  );
  return (res.items ?? []).map(normalizeItem);
}

export async function getStaffInbox(): Promise<{
  items: StaffRequestItem[];
  pending: number;
}> {
  const res = await apiFetch<{ items: Record<string, unknown>[]; pending: number }>(
    "/staff/requests/inbox"
  );
  return {
    items: (res.items ?? []).map(normalizeItem),
    pending: Number(res.pending ?? 0),
  };
}

export async function reviewStaffRequest(params: {
  requestId: string;
  approve: boolean;
  reviewerTitle: string;
  reviewerName: string;
  role?: string;
  departmentName?: string;
  duty?: string;
  comment?: string;
}): Promise<{ message: string; request: StaffRequestItem }> {
  const res = await apiFetch<{ message: string; request: Record<string, unknown> }>(
    `/staff/requests/${params.requestId}/review`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        approve: params.approve,
        reviewer_title: params.reviewerTitle,
        reviewer_name: params.reviewerName,
        role: params.role ?? null,
        department_name: params.departmentName ?? null,
        duty: params.duty ?? null,
        comment: params.comment ?? null,
      }),
    }
  );
  return { message: res.message, request: normalizeItem(res.request) };
}

export async function cancelStaffRequest(requestId: string): Promise<void> {
  await apiFetch(`/staff/requests/${requestId}/cancel`, { method: "POST" });
}

export async function markStaffRequestsSeen(): Promise<number> {
  const res = await apiFetch<{ cleared: number }>("/staff/requests/mark-seen", {
    method: "POST",
  });
  return Number(res.cleared ?? 0);
}

// ── 企业管理后台 ──────────────────────────────────────────────────────────────

export async function getStaffMembers(params?: {
  companyId?: string;
  keyword?: string;
}): Promise<StaffMember[]> {
  const search = new URLSearchParams();
  if (params?.companyId) search.set("company_id", params.companyId);
  if (params?.keyword) search.set("keyword", params.keyword);
  const query = search.toString();
  const res = await apiFetch<{ items: Record<string, unknown>[] }>(
    `/staff/members${query ? `?${query}` : ""}`
  );
  return (res.items ?? []).map(normalizeMember);
}

export async function getStaffCompanies(): Promise<CompanyOption[]> {
  const res = await apiFetch<{ items: Record<string, unknown>[] }>("/staff/companies");
  return (res.items ?? []).map((raw) => ({
    companyId: String(raw.company_id ?? ""),
    companyName: String(raw.company_name ?? raw.company_id ?? ""),
    memberCount: Number(raw.member_count ?? 0),
  }));
}

export async function updateStaffMember(
  memberId: string,
  patch: {
    role?: string;
    departmentName?: string;
    jobTitle?: string;
    isActive?: boolean;
  }
): Promise<StaffMember> {
  const res = await apiFetch<{ member: Record<string, unknown>; message: string }>(
    `/staff/members/${memberId}`,
    {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        role: patch.role ?? null,
        department_name: patch.departmentName ?? null,
        job_title: patch.jobTitle ?? null,
        is_active: patch.isActive ?? null,
      }),
    }
  );
  return normalizeMember(res.member);
}

// ── 删除成员（账号注销） ──────────────────────────────────────────────────────

function normalizeImpact(raw: Record<string, unknown>): MemberDeletionImpact {
  const member = (raw.member ?? {}) as Record<string, unknown>;
  const deleted = (raw.deleted ?? {}) as Record<string, unknown>;
  const kept = (raw.kept ?? {}) as Record<string, unknown>;
  return {
    member: {
      id: String(member.id ?? ""),
      username: String(member.username ?? ""),
      name: String(member.name ?? member.username ?? ""),
      companyName: String(member.company_name ?? ""),
      departmentName: (member.department_name ?? null) as string | null,
      jobTitle: (member.job_title ?? null) as string | null,
      role: String(member.role ?? ""),
      roleLabel: String(member.role_label ?? ""),
    },
    deleted: {
      personalDocuments: Number(deleted.personal_documents ?? 0),
      conversations: Number(deleted.conversations ?? 0),
      messages: Number(deleted.messages ?? 0),
      collections: Number(deleted.collections ?? 0),
      feedback: Number(deleted.feedback ?? 0),
    },
    kept: {
      sharedDocuments: Number(kept.shared_documents ?? 0),
      staffRequests: Number(kept.staff_requests ?? 0),
      shareRequests: Number(kept.share_requests ?? 0),
      badCases: Number(kept.bad_cases ?? 0),
    },
  };
}

/**
 * 删除前的影响预检。权限与可删范围由后端判定 —— 前端据此决定按钮是否可点，
 * 但真正的守门永远在后端（预检本身也会 403）。
 */
export async function previewStaffMemberDeletion(
  memberId: string
): Promise<MemberDeletionImpact> {
  const raw = await apiFetch<Record<string, unknown>>(
    `/staff/members/${memberId}/deletion-preview`
  );
  return normalizeImpact(raw);
}

/** 删除成员账号：个人数据一并删除，其部门/公司知识库文档保留。 */
export async function deleteStaffMember(
  memberId: string
): Promise<{ message: string; impact: MemberDeletionImpact }> {
  const raw = await apiFetch<Record<string, unknown>>(
    `/staff/members/${memberId}`,
    { method: "DELETE", timeout: 120000 }
  );
  return {
    message: String(raw.message ?? "成员已删除"),
    impact: normalizeImpact(raw),
  };
}
