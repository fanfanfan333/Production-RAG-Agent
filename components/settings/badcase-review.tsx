"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { Activity, Loader2, RefreshCw, RotateCcw } from "lucide-react";
import {
  getBadCaseStats,
  type BadCaseStatsResponse,
} from "@/lib/api/badcases";
import {
  getQualityStats,
  resetQualityStats,
  type QualityStats,
  type QualityWindow,
} from "@/lib/api/quality";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";

/**
 * RAG 质量监控（设置页）.
 *
 * 上方是**运行期质量指标**：拒答率、引用通过率、引用存疑率、输出净化
 * 命中率等，用来判断"知识库答得准不准"。
 *
 * 此前指标只读进程内计数器 —— 后端一重启分母归零，整排比率显示 "—"，
 * 与下方来自数据库的 Bad Case 累计数自相矛盾。现在指标持久化到
 * quality_events 表，支持时间窗切换：
 *   本次运行（进程内，实时）/ 近 24 小时 / 近 7 天 / 近 30 天 / 全部。
 * 每张卡片带样本量（n=…），比率为 null 时明确显示"暂无数据"而不是 "—"。
 *
 * 原先下半部分还有一整块 Bad Case 审阅（筛选 + 逐条记录 + 引用校验明细 +
 * 定级/结案操作）—— 那部分是**运维侧的内部分诊台**，放在面向用户的系统设置里
 * 既啰嗦又容易让用户误以为"系统出错了"。按需求移除；后端 `/admin/badcases`
 * 接口与数据回流不受影响，将来要恢复在设置页挂载一个列表组件即可。
 */

const REASON_LABEL: Record<string, string> = {
  citation_unsupported: "引用不被支持",
  evidence_refused: "证据门控拒答",
  output_guard: "输出净化命中",
  feedback_down: "用户差评",
};

/** 引用校验失败原因（quality_events 聚合字段 → 中文） */
const CITATION_FAIL_LABEL: Record<string, string> = {
  unsupported: "原文未支持",
  hallucinated: "来源不存在",
  misattributed: "位置存疑",
  number_mismatch: "数字不一致",
  date_mismatch: "日期不一致",
};

/** 路由意图 → 中文 */
const INTENT_LABEL: Record<string, string> = {
  knowledge_qa: "知识问答",
  document_summary: "文档总结",
  general_chat: "通用闲聊",
  doc_relations: "关联分析",
  list_documents: "文档清单",
  document_agent: "文档生成",
};

const WINDOWS: { key: QualityWindow; label: string }[] = [
  { key: "session", label: "本次运行" },
  { key: "24h", label: "近 24 小时" },
  { key: "7d", label: "近 7 天" },
  { key: "30d", label: "近 30 天" },
  { key: "all", label: "全部" },
];

function pct(v: number | null | undefined): string {
  if (v === null || v === undefined) return "—";
  return `${(v * 100).toFixed(1)}%`;
}

function ms(v: number | null | undefined): string {
  if (v === null || v === undefined) return "—";
  if (v >= 1000) return `${(v / 1000).toFixed(1)}s`;
  return `${Math.round(v)}ms`;
}

export function BadCaseReview() {
  const [stats, setStats] = useState<BadCaseStatsResponse | null>(null);
  const [quality, setQuality] = useState<QualityStats | null>(null);
  const [window, setWindow] = useState<QualityWindow>("24h");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [resetting, setResetting] = useState(false);

  const load = useCallback(async (w: QualityWindow) => {
    setLoading(true);
    setError(null);
    try {
      const [badcase, qual] = await Promise.all([
        getBadCaseStats(),
        getQualityStats(w),
      ]);
      setStats(badcase);
      setQuality(qual);
    } catch (err) {
      setError(err instanceof Error ? err.message : "加载失败");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load(window);
  }, [load, window]);

  const windowLabel =
    WINDOWS.find((w) => w.key === window)?.label ?? window;

  /** 重置当前窗口的统计：清掉已累计的分子/分母，之后重新开始计数。 */
  const doReset = useCallback(async () => {
    setResetting(true);
    setError(null);
    setNotice(null);
    try {
      const res = await resetQualityStats(window);
      setConfirmOpen(false);
      setNotice(
        res.scope === "in_process"
          ? `已重置「${windowLabel}」的进程内计数（数据库历史未受影响）`
          : `已重置「${windowLabel}」的统计，清除 ${res.deleted} 条质量事件`
      );
      await load(window);
      // 4 秒后自动收起提示，不打断阅读
      setTimeout(() => setNotice(null), 4000);
    } catch (err) {
      setConfirmOpen(false);
      setError(
        err instanceof Error ? `重置失败：${err.message}` : "重置失败"
      );
    } finally {
      setResetting(false);
    }
  }, [load, window, windowLabel]);

  const ratios = quality?.ratios;
  const samples = quality?.samples;
  const metricCards = useMemo(
    () => [
      {
        label: "证据拒答率",
        value: ratios?.evidence_refuse_rate,
        sample: samples?.evidence_gate,
        sampleUnit: "次门控",
      },
      {
        label: "引用通过率",
        value: ratios?.citation_pass_rate,
        sample: samples?.citations_checked,
        sampleUnit: "条引用",
      },
      {
        label: "引用存疑率",
        value: ratios?.citation_unsupported_rate,
        sample: samples?.citation_answers,
        sampleUnit: "次回答",
      },
      {
        label: "输出净化命中率",
        value: ratios?.output_guard_change_rate,
        sample: samples?.queries,
        sampleUnit: "次问答",
      },
      {
        label: "整体拒答率",
        value: ratios?.refusal_rate,
        sample: samples?.queries,
        sampleUnit: "次问答",
      },
    ],
    [ratios, samples]
  );

  const intentEntries = Object.entries(quality?.byIntent ?? {});
  const failEntries = Object.entries(quality?.citationFailureReasons ?? {}).filter(
    ([, v]) => v > 0
  );

  return (
    <Card>
      <CardHeader className="flex flex-row items-start justify-between gap-4">
        <div>
          <CardTitle className="flex items-center gap-2">
            <Activity className="size-4 text-primary" />
            RAG 质量监控
          </CardTitle>
          <CardDescription>
            {quality?.source === "in_process"
              ? "本次运行的实时指标（进程内计数，重启归零）"
              : "历史聚合指标（持久化存储，重启不丢）"}
          </CardDescription>
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <Button
            size="sm"
            variant="outline"
            onClick={() => void load(window)}
            disabled={loading}
          >
            {loading ? (
              <Loader2 className="size-3.5 animate-spin" />
            ) : (
              <RefreshCw className="size-3.5" />
            )}
            刷新
          </Button>
          <Button
            size="sm"
            variant="destructive"
            onClick={() => setConfirmOpen(true)}
            disabled={loading || resetting}
          >
            {resetting ? (
              <Loader2 className="size-3.5 animate-spin" />
            ) : (
              <RotateCcw className="size-3.5" />
            )}
            重置
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* 时间窗切换：历史窗口重启不丢数据，是"指标全是 —"的修复 */}
        <div className="flex flex-wrap gap-1.5">
          {WINDOWS.map((w) => (
            <button
              key={w.key}
              type="button"
              onClick={() => setWindow(w.key)}
              className={cn(
                "rounded-full border px-2.5 py-1 text-xs transition-colors",
                window === w.key
                  ? "border-primary/60 bg-primary/10 font-medium text-primary"
                  : "border-border/60 text-muted-foreground hover:bg-accent/50"
              )}
            >
              {w.label}
            </button>
          ))}
        </div>

        {error && (
          <div className="rounded-xl border border-destructive/20 bg-destructive/5 px-4 py-3 text-sm text-destructive">
            {error}
          </div>
        )}
        {notice && (
          <div className="rounded-xl border border-emerald-500/25 bg-emerald-500/5 px-4 py-3 text-sm text-emerald-700 dark:text-emerald-400">
            {notice}
          </div>
        )}
        {loading && !quality ? (
          <Skeleton className="h-24 w-full" />
        ) : (
          <>
            <div className="grid gap-3 sm:grid-cols-3 lg:grid-cols-5">
              {metricCards.map((m) => (
                <div
                  key={m.label}
                  className="rounded-lg border border-border/60 bg-muted/30 px-3 py-2"
                >
                  <p className="text-xs text-muted-foreground">{m.label}</p>
                  <p className="mt-0.5 text-lg font-semibold tabular-nums">
                    {pct(m.value)}
                  </p>
                  <p className="text-[10px] text-muted-foreground/70">
                    {m.value === null || m.value === undefined
                      ? `该窗口内暂无${m.sampleUnit}`
                      : `n=${m.sample ?? 0} ${m.sampleUnit}`}
                  </p>
                </div>
              ))}
            </div>

            {/* 时延与意图分布 */}
            <div className="flex flex-wrap gap-x-6 gap-y-2 text-xs text-muted-foreground">
              <span>
                问答量{" "}
                <b className="text-foreground">{quality?.totalQueries ?? 0}</b> 次
              </span>
              <span>
                时延 avg <b className="text-foreground">{ms(quality?.latency.avg)}</b>
                {" / "}p95 <b className="text-foreground">{ms(quality?.latency.p95)}</b>
                {" / "}max <b className="text-foreground">{ms(quality?.latency.max)}</b>
              </span>
              {intentEntries.map(([k, v]) => (
                <span key={k}>
                  {INTENT_LABEL[k] ?? k}{" "}
                  <b className="text-foreground">{v}</b>
                </span>
              ))}
            </div>

            {/* 引用校验失败原因分布（只有出问题时才显示） */}
            {failEntries.length > 0 && (
              <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
                <span>引用失败原因：</span>
                {failEntries.map(([k, v]) => (
                  <span key={k}>
                    {CITATION_FAIL_LABEL[k] ?? k}{" "}
                    <b className="text-amber-600 dark:text-amber-300">{v}</b>
                  </span>
                ))}
              </div>
            )}

            <div className="flex flex-wrap gap-x-6 gap-y-2 border-t border-border/50 pt-3 text-xs text-muted-foreground">
              <span>
                累计回流 <b className="text-foreground">{stats?.total ?? 0}</b> 条
              </span>
              {Object.entries(stats?.by_reason ?? {}).map(([k, v]) => (
                <span key={k}>
                  {REASON_LABEL[k] ?? k} <b className="text-foreground">{v}</b>
                </span>
              ))}
            </div>
          </>
        )}
      </CardContent>

      {/* 重置是**不可逆**的删除，必须二次确认 */}
      <Dialog open={confirmOpen} onOpenChange={setConfirmOpen}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>重置「{windowLabel}」的质量统计？</DialogTitle>
            <DialogDescription>
              {window === "session" ? (
                <>
                  将清空<b className="text-foreground">进程内计数器</b>
                  （本次运行的实时指标）。数据库中的历史质量事件不受影响。
                </>
              ) : (
                <>
                  将删除该窗口内的
                  <b className="text-foreground">
                    {" "}
                    {quality?.totalQueries ?? 0} 条
                  </b>{" "}
                  质量事件记录，<b className="text-destructive">删除后不可恢复</b>
                  。之后这些比率会重新从零开始累计。
                </>
              )}
            </DialogDescription>
          </DialogHeader>
          <div className="flex justify-end gap-2">
            <Button
              size="sm"
              variant="outline"
              onClick={() => setConfirmOpen(false)}
              disabled={resetting}
            >
              取消
            </Button>
            <Button
              size="sm"
              variant="destructive"
              onClick={() => void doReset()}
              disabled={resetting}
            >
              {resetting && <Loader2 className="size-3.5 animate-spin" />}
              确认重置
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </Card>
  );
}
