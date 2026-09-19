import { apiFetch } from "@/lib/api/client";

/**
 * RAG 质量监控 API（/quality/stats）.
 *
 * window=session 走进程内计数器（实时、重启归零）；
 * 24h/7d/30d/all 走 quality_events 表聚合（重启不丢、可看趋势）。
 * 比率分母为 0 时为 null，samples 给出各指标样本量 —— 面板据此把 "—"
 * 解释成"该窗口内暂无这类问答"，而不是显示一个无意义的空值。
 */

export type QualityWindow = "session" | "24h" | "7d" | "30d" | "all";

export interface QualityRatios {
  evidence_refuse_rate: number | null;
  citation_pass_rate: number | null;
  citation_unsupported_rate: number | null;
  output_guard_change_rate: number | null;
  refusal_rate: number | null;
}

export interface LatencyStats {
  avg: number | null;
  p95: number | null;
  max: number | null;
}

/** 后端原始返回：session 窗口的 latency_ms 是分阶段结构，这里先原样接住。 */
interface QualityStatsRaw {
  window: QualityWindow;
  source: "in_process" | "database";
  total_queries: number;
  ratios: Partial<QualityRatios>;
  samples: {
    evidence_gate: number;
    citations_checked: number;
    citation_answers: number;
    queries: number;
  };
  by_intent: Record<string, number>;
  latency_ms:
    | LatencyStats
    | Record<string, { avg_ms?: number | null; avg?: number | null; p95?: number | null; max_ms?: number | null; max?: number | null }>;
  citation_failure_reasons: Record<string, number>;
  generated_at: string;
}

export interface QualityStats {
  window: QualityWindow;
  source: "in_process" | "database";
  totalQueries: number;
  ratios: Partial<QualityRatios>;
  samples: QualityStatsRaw["samples"];
  byIntent: Record<string, number>;
  /** 整体时延（已归一化：两种窗口结构统一为 avg/p95/max）。 */
  latency: LatencyStats;
  /** 分阶段时延（仅 session 窗口有）。 */
  stageLatencies: Record<string, LatencyStats>;
  citationFailureReasons: Record<string, number>;
  generatedAt: string;
}

function normalizeLatency(raw: QualityStatsRaw["latency_ms"]): {
  latency: LatencyStats;
  stageLatencies: Record<string, LatencyStats>;
} {
  const entries = Object.entries(raw ?? {});
  const isPerStage = entries.some(
    ([, v]) => v && typeof v === "object" && ("avg_ms" in v || "max_ms" in v)
  );
  if (!isPerStage) {
    // 数据库窗口：{ avg, p95, max }
    const flat = raw as LatencyStats;
    return {
      latency: { avg: flat.avg ?? null, p95: flat.p95 ?? null, max: flat.max ?? null },
      stageLatencies: {},
    };
  }
  // session 窗口：{ retrieval: {avg_ms, max_ms}, total: {...} }
  const stageLatencies: Record<string, LatencyStats> = {};
  for (const [stage, v] of entries) {
    if (!v || typeof v !== "object") continue;
    const rec = v as Record<string, number | null | undefined>;
    stageLatencies[stage] = {
      avg: rec.avg_ms ?? rec.avg ?? null,
      p95: rec.p95 ?? null,
      max: rec.max_ms ?? rec.max ?? null,
    };
  }
  return {
    latency: stageLatencies["total"] ?? { avg: null, p95: null, max: null },
    stageLatencies,
  };
}

/** 重置结果：scope 说明清的是哪一层，deleted 是实际删除的行数（session 窗口为 0）。 */
export interface QualityResetResult {
  ok: boolean;
  window: QualityWindow;
  scope: "in_process" | "database";
  deleted: number;
  resetAt: string;
}

/**
 * 重置质量统计（不可逆，界面必须二次确认）.
 *
 * 语义刻意与面板当前窗口对齐：session 只清进程内计数器（不动数据库）；
 * 24h/7d/30d/all 删除 quality_events 中该窗口内的行。
 */
export async function resetQualityStats(
  window: QualityWindow
): Promise<QualityResetResult> {
  const raw = await apiFetch<{
    ok: boolean;
    window: QualityWindow;
    scope: "in_process" | "database";
    deleted: number;
    reset_at: string;
  }>(`/quality/reset?window=${window}`, { method: "POST" });
  return {
    ok: raw.ok,
    window: raw.window,
    scope: raw.scope,
    deleted: raw.deleted ?? 0,
    resetAt: raw.reset_at ?? "",
  };
}

export async function getQualityStats(
  window: QualityWindow
): Promise<QualityStats> {
  const raw = await apiFetch<QualityStatsRaw>(`/quality/stats?window=${window}`);
  const { latency, stageLatencies } = normalizeLatency(raw.latency_ms);
  return {
    window: raw.window,
    source: raw.source,
    totalQueries: raw.total_queries ?? 0,
    ratios: raw.ratios ?? {},
    samples: raw.samples ?? {
      evidence_gate: 0,
      citations_checked: 0,
      citation_answers: 0,
      queries: 0,
    },
    byIntent: raw.by_intent ?? {},
    latency,
    stageLatencies,
    citationFailureReasons: raw.citation_failure_reasons ?? {},
    generatedAt: raw.generated_at ?? "",
  };
}
