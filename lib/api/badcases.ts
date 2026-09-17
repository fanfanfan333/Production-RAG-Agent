import { apiFetch } from "@/lib/api/client";

/**
 * Bad Case 回流 API（持续监控闭环的"人"这一端）。
 *
 * 后端权限为 `audit.read`（admin / manager）。前端角色只有 admin / user，
 * 因此 UI 侧按 admin 展示，真正的权限校验以后端返回的 403 为准。
 */

export type BadCaseReason =
  | "citation_unsupported"
  | "evidence_refused"
  | "output_guard"
  | "feedback_down";

export type BadCaseStatus = "open" | "triaged" | "resolved" | "wontfix";

export type BadCaseSeverity = "high" | "medium" | "low";

/** 回流时一并落库的引用快照（只留定位所需字段）。 */
export interface BadCaseSourceSnapshot {
  document_id?: string | null;
  filename?: string | null;
  page_number?: number | null;
  line_start?: number | null;
  line_end?: number | null;
  content_type?: string | null;
  text_snippet?: string | null;
}

export interface BadCaseItem {
  id: string;
  reason: string;
  severity: string;
  intent: string | null;
  question: string;
  answer: string;
  status: string;
  resolution: string | null;
  tags: string | null;
  username: string | null;
  conversation_id: string | null;
  created_at: string;
  detail: Record<string, unknown> | null;
  sources: BadCaseSourceSnapshot[] | null;
}

export interface BadCaseListResponse {
  bad_cases: BadCaseItem[];
  total: number;
}

/** `metrics_snapshot()` 的派生比率（分母为 0 时为 null）。 */
export interface BadCaseMetricsRatios {
  evidence_refuse_rate?: number | null;
  citation_pass_rate?: number | null;
  citation_unsupported_rate?: number | null;
  citation_failed_per_check?: number | null;
  output_guard_change_rate?: number | null;
  refusal_rate?: number | null;
}

export interface BadCaseMetrics {
  counters?: Record<string, number>;
  latencies?: Record<string, { count: number; avg_ms: number; max_ms: number }>;
  ratios?: BadCaseMetricsRatios;
  generated_at?: string;
}

export interface BadCaseStatsResponse {
  total: number;
  by_reason: Record<string, number>;
  by_status: Record<string, number>;
  by_severity: Record<string, number>;
  metrics: BadCaseMetrics;
}

export interface BadCaseFilters {
  reason?: string;
  status?: string;
  severity?: string;
  limit?: number;
}

export interface BadCaseUpdatePayload {
  status?: BadCaseStatus;
  resolution?: string;
  tags?: string;
}

function buildQuery(filters: BadCaseFilters): string {
  const params = new URLSearchParams();
  if (filters.reason) params.set("reason", filters.reason);
  if (filters.status) params.set("status", filters.status);
  if (filters.severity) params.set("severity", filters.severity);
  if (filters.limit) params.set("limit", String(filters.limit));
  const qs = params.toString();
  return qs ? `?${qs}` : "";
}

export async function listBadCases(
  filters: BadCaseFilters = {}
): Promise<BadCaseListResponse> {
  return apiFetch<BadCaseListResponse>(`/badcases${buildQuery(filters)}`);
}

export async function updateBadCase(
  id: string,
  payload: BadCaseUpdatePayload
): Promise<BadCaseItem> {
  return apiFetch<BadCaseItem>(`/badcases/${id}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function getBadCaseStats(): Promise<BadCaseStatsResponse> {
  return apiFetch<BadCaseStatsResponse>("/badcases/stats");
}
