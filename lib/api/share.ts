import { apiFetch } from "@/lib/api/client";
import type {
  ShareIntent,
  ShareRequestItem,
  ShareRequestStatus,
  ShareRequestSummary,
  ShareTargetLevel,
} from "@/lib/types";

/**
 * 共享申请 API（「申请共享」+「查看申请」+ 审核）。
 *
 * 后端出参是 snake_case，这里集中做一次映射 —— 与 SSE / 历史回放的
 * normalize 约定一致（归一化只有一份）。
 */

function normalizeItem(raw: Record<string, unknown>): ShareRequestItem {
  const level = String(raw.target_level ?? "department");
  const intent = String(raw.intent ?? "publish") === "delete" ? "delete" : "publish";
  return {
    id: String(raw.id ?? ""),
    // 删除申请批准后文档没了 → document_id 为 null，前端据此显示「文档已删除」
    documentId: raw.document_id ? String(raw.document_id) : null,
    documentName: String(raw.document_name ?? ""),
    intent,
    intentLabel: String(
      raw.intent_label ?? (intent === "delete" ? "申请删除" : "申请共享")
    ),
    requesterUsername: String(raw.requester_username ?? ""),
    requesterDepartmentId: (raw.requester_department_id ?? null) as string | null,
    targetLevel: (level === "tenant" ? "tenant" : "department") as ShareTargetLevel,
    targetLabel: String(
      raw.target_label ?? (level === "tenant" ? "公司知识库" : "部门知识库")
    ),
    targetDepartmentId: (raw.target_department_id ?? null) as string | null,
    reason: (raw.reason ?? null) as string | null,
    status: String(raw.status ?? "pending") as ShareRequestStatus,
    reviewerUsername: (raw.reviewer_username ?? null) as string | null,
    reviewComment: (raw.review_comment ?? null) as string | null,
    createdAt: (raw.created_at ?? null) as string | null,
    reviewedAt: (raw.reviewed_at ?? null) as string | null,
    requesterSeen: Boolean(raw.requester_seen ?? false),
    isMine: Boolean(raw.is_mine ?? false),
    canReview: Boolean(raw.can_review ?? false),
  };
}

function normalizeSummary(raw: Record<string, unknown>): ShareRequestSummary {
  const scope = raw.review_scope;
  return {
    pendingForMe: Number(raw.pending_for_me ?? 0),
    myPending: Number(raw.my_pending ?? 0),
    myDecidedUnseen: Number(raw.my_decided_unseen ?? 0),
    reviewScope:
      scope === "company" || scope === "department" ? scope : null,
    canReview: Boolean(raw.can_review ?? false),
    totalBadge: Number(raw.total_badge ?? 0),
  };
}

export async function createShareRequest(params: {
  documentId: string;
  targetLevel: ShareTargetLevel;
  reason?: string;
}): Promise<ShareRequestItem> {
  const res = await apiFetch<{ request: Record<string, unknown>; message: string }>(
    "/share-requests",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        document_id: params.documentId,
        target_level: params.targetLevel,
        reason: params.reason ?? null,
      }),
    }
  );
  return normalizeItem(res.request);
}

export async function createDeleteRequest(params: {
  documentId: string;
  reason?: string;
}): Promise<ShareRequestItem> {
  // 目标层级由后端按文档当前层级推导（决定谁来审），前端不必猜。
  const res = await apiFetch<{ request: Record<string, unknown>; message: string }>(
    "/share-requests",
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        document_id: params.documentId,
        intent: "delete" satisfies ShareIntent,
        reason: params.reason ?? null,
      }),
    }
  );
  return normalizeItem(res.request);
}

export async function getMyShareRequests(
  status?: ShareRequestStatus
): Promise<ShareRequestItem[]> {
  const query = status ? `?status=${status}` : "";
  const res = await apiFetch<{ items: Record<string, unknown>[] }>(
    `/share-requests/mine${query}`
  );
  return (res.items ?? []).map(normalizeItem);
}

export async function getShareRequestInbox(): Promise<{
  items: ShareRequestItem[];
  pending: number;
}> {
  const res = await apiFetch<{ items: Record<string, unknown>[]; pending: number }>(
    "/share-requests/inbox"
  );
  return {
    items: (res.items ?? []).map(normalizeItem),
    pending: Number(res.pending ?? 0),
  };
}

export async function getShareRequestSummary(): Promise<ShareRequestSummary> {
  const raw = await apiFetch<Record<string, unknown>>("/share-requests/summary");
  return normalizeSummary(raw);
}

export async function markMyShareRequestsSeen(): Promise<number> {
  const res = await apiFetch<{ cleared: number }>("/share-requests/mark-seen", {
    method: "POST",
  });
  return Number(res.cleared ?? 0);
}

export async function reviewShareRequest(params: {
  requestId: string;
  approve: boolean;
  comment?: string;
}): Promise<{ message: string }> {
  return apiFetch<{ message: string }>(
    `/share-requests/${params.requestId}/review`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        approve: params.approve,
        comment: params.comment ?? null,
      }),
    }
  );
}

export async function cancelShareRequest(requestId: string): Promise<void> {
  await apiFetch(`/share-requests/${requestId}/cancel`, { method: "POST" });
}
