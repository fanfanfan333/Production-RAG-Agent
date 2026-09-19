"use client";

import { useState, type ReactNode } from "react";
import {
  ChevronDown,
  FileText,
  ImageIcon,
  BookMarked,
  Table2,
  BarChart3,
  GitBranch,
  MonitorPlay,
  ExternalLink,
  Highlighter,
  MapPin,
  ShieldCheck,
  ShieldAlert,
  Sigma,
  Code2,
  CircleAlert,
} from "lucide-react";
import type {
  CitationCheckInfo,
  CitationVerdict,
  EvidenceSpan,
  QuerySource,
} from "@/lib/types";
import { IMAGE_TYPE_LABELS } from "@/lib/types";
import { getDocumentImageUrl } from "@/lib/api/documents";
import { cn } from "@/lib/utils";

/**
 * 引用校验"未通过项"的中文说明。
 *
 * 五项校验（存在 / 位置 / 支持 / 数字 / 日期）此前只在总览徽标的 title 里
 * 用一串编号表示（"位置存疑 2,3"），用户看到的是"某条存疑"却不知道该条
 * **到底哪一项没过**。逐条渲染它，引用才真的可核对。
 */
const CHECK_LABELS: Record<string, string> = {
  citation_exists: "来源不存在",
  position_correct: "位置存疑",
  supported: "原文未直接支持",
  numbers_consistent: "数字与原文不一致",
  dates_consistent: "日期与原文不一致",
};

function relevancePercent(score?: number): string | null {
  if (typeof score !== "number" || Number.isNaN(score)) return null;
  return `${Math.round(score * 100)}%`;
}

/** 引用卡片在页面里的锚点 id —— 正文上标点击后跳到这里。 */
export function sourceAnchorId(index1Based: number): string {
  return `source-${index1Based}`;
}

function isImageSource(source: QuerySource): boolean {
  return source.contentType === "image";
}

/**
 * 行号区间文案：优先用后端算好的 lineSpan，其次由 lineStart/lineEnd 推导。
 *
 * 位置信息是"细粒度引用"的基础 —— 引用卡片必须能说清"这是第几行"，
 * 用户才有机会回到原文核对。旧索引没有行号时返回 null，卡片自动降级。
 */
function lineSpanOf(source: QuerySource): string | null {
  if (source.lineSpan) return source.lineSpan;
  if (typeof source.lineStart === "number") {
    if (
      typeof source.lineEnd !== "number" ||
      source.lineEnd === source.lineStart
    ) {
      return String(source.lineStart);
    }
    return `${source.lineStart}-${source.lineEnd}`;
  }
  return null;
}

/**
 * 一句话溯源（后端 location_label 的产物）：《年报.pdf》第 3 页，第 12-28 行
 */
function locationSentence(source: QuerySource): string | null {
  if (source.location) return source.location;
  const name = source.documentName;
  if (!name) return null;
  const parts = [`《${name}》`];
  if (typeof source.page === "number") parts.push(`第 ${source.page} 页`);
  const span = lineSpanOf(source);
  if (span) parts.push(`第 ${span} 行`);
  return parts.join("，");
}

// ── 命中句（"答案实际用了哪几句"）────────────────────────────────────────────
//
// 引用此前只到"切片"这一级：卡片给出整块的行范围（第 83-105 行）＋整段原文，
// 用户看到 20 多行原文，却不知道答案用的是哪几句 —— 只能自己读完再猜。
// 下面这组函数把"命中句"落到可渲染的区间上：段落里是哪几句、这几句各自在
// 第几行。

/** 可高亮的字符区间（ratio 取该区间内命中句的最高重合率）。 */
type TextRange = { start: number; end: number; ratio: number };

/** 同一来源可能被多条答案句引用 → 先拉平再合并重叠区间。 */
function collectEvidence(verdicts: CitationVerdict[]): EvidenceSpan[] {
  return verdicts.flatMap((v) => v.evidence ?? []);
}

/**
 * 命中句 → 可高亮的字符区间.
 *
 * 后端同时给了"字符偏移"和"原句文本"，这里做一次自校验：偏移切出来的内容
 * 与文本一致才采用；不一致（历史回放口径不同、正文被上游裁剪过）就退化为
 * 按文本查找。定位失败的条目直接丢弃 —— 错位的高亮比没有高亮更糟，它会
 * 让人以为原文本来就是这么写的。
 */
function evidenceRanges(text: string, spans: EvidenceSpan[]): TextRange[] {
  if (!text || !spans.length) return [];
  const found: TextRange[] = [];
  for (const span of spans) {
    const probe = (span.text ?? "").trim();
    if (!probe) continue;
    let start = -1;
    if (
      Number.isInteger(span.start) &&
      span.end <= text.length &&
      text.slice(span.start, span.end).trim() === probe
    ) {
      start = span.start;
    } else {
      start = text.indexOf(probe);
    }
    if (start < 0) continue;
    found.push({ start, end: start + probe.length, ratio: span.ratio });
  }
  if (!found.length) return [];
  found.sort((a, b) => a.start - b.start);
  const merged: TextRange[] = [{ ...found[0] }];
  for (const r of found.slice(1)) {
    const last = merged[merged.length - 1];
    if (r.start <= last.end) {
      last.end = Math.max(last.end, r.end);
      last.ratio = Math.max(last.ratio, r.ratio);
    } else {
      merged.push({ ...r });
    }
  }
  return merged;
}

/** 命中句自身的行号区间（多条命中句取并集）；旧索引无行号时返回 null。 */
function hitLineSpan(spans: EvidenceSpan[]): string | null {
  const starts = spans
    .map((s) => s.line_start)
    .filter((n): n is number => typeof n === "number");
  if (!starts.length) return null;
  const ends = spans
    .map((s) => s.line_end ?? s.line_start)
    .filter((n): n is number => typeof n === "number");
  const lo = Math.min(...starts);
  const hi = ends.length ? Math.max(...ends) : lo;
  return lo === hi ? String(lo) : `${lo}-${hi}`;
}

/**
 * 命中句徽标：`命中 2 句 · 第 84-85 行`.
 *
 * 对"只标了行号"的直接补正：行号区间属于**整块切片**，用户要的是"答案用了
 * 哪几句"。这里给的是命中句自己的行号（由后端按命中句偏移换算）。
 */
function EvidenceBadge({ spans }: { spans: EvidenceSpan[] }) {
  if (!spans.length) return null;
  const span = hitLineSpan(spans);
  const ratios = spans.map((s) => `${Math.round((s.ratio ?? 0) * 100)}%`).join(" / ");
  return (
    <span
      className="inline-flex shrink-0 items-center gap-0.5 rounded-full bg-amber-500/15 px-2 py-0.5 text-[10px] font-medium text-amber-700 dark:text-amber-300"
      title={`命中句 = 答案实际依据的那几句原文，与被引句的内容词重合率 ${ratios}${
        span ? `；位于第 ${span} 行` : ""
      }`}
    >
      <Highlighter className="size-2.5" />
      命中 {spans.length} 句
      {span ? ` · 第 ${span} 行` : ""}
    </span>
  );
}

/**
 * 引用正文渲染：**收起态只显示命中句**，展开态给整段原文并高亮命中句.
 *
 * 这正是本次要解决的问题 —— 此前两种状态都只有整段原文，且行号只到"块"。
 * 没有任何命中句时（旧后端 / 与原句无内容交集）保持原样展示整段，不做
 * 任何标记：宁可不标，也不给用户一个错误的"依据在此"暗示。
 */
function EvidenceExcerpt({
  text,
  spans,
  expanded,
}: {
  text: string;
  spans: EvidenceSpan[];
  expanded: boolean;
}) {
  const ranges = evidenceRanges(text, spans);

  if (!ranges.length) {
    return (
      <span
        className={cn(
          "mt-1.5 block text-[11px] leading-relaxed text-muted-foreground",
          !expanded && "line-clamp-2"
        )}
      >
        {text}
      </span>
    );
  }

  // 收起：只列命中句本身 —— "答案就是靠这几句说的"
  if (!expanded) {
    return (
      <span className="mt-1.5 block space-y-1">
        {ranges.map((r, i) => (
          <span key={i} className="flex gap-1.5">
            <span className="mt-[3px] h-3 w-0.5 shrink-0 rounded-full bg-amber-500/70" />
            <span className="text-[11px] leading-relaxed text-foreground/85">
              {text.slice(r.start, r.end)}
            </span>
          </span>
        ))}
      </span>
    );
  }

  // 展开：整段原文 + 命中句高亮（核对上下文用）
  const nodes: ReactNode[] = [];
  let cursor = 0;
  ranges.forEach((r, i) => {
    if (r.start > cursor) nodes.push(text.slice(cursor, r.start));
    nodes.push(
      <mark
        key={`hit-${i}`}
        className="rounded bg-amber-400/25 px-0.5 text-foreground/90 dark:bg-amber-300/20"
      >
        {text.slice(r.start, r.end)}
      </mark>
    );
    cursor = r.end;
  });
  if (cursor < text.length) nodes.push(text.slice(cursor));

  return (
    <span className="mt-1.5 block text-[11px] leading-relaxed text-muted-foreground">
      {nodes}
    </span>
  );
}

/**
 * 这条引用**支撑了答案里的哪句话**.
 *
 * 引用是双向的：一段原文被引，是为了支撑某个结论。只给原文不给结论，
 * 用户仍要回头在答案里找"这句到底支撑了什么"。
 */
function SupportedSentences({
  sentences,
  expanded,
}: {
  sentences: string[];
  expanded: boolean;
}) {
  if (!sentences.length) return null;
  return (
    <span className="mt-1.5 block rounded-md bg-primary/5 px-2 py-1 text-[10px] leading-relaxed text-muted-foreground">
      <span className="font-medium text-foreground/70">
        本条支撑的结论{sentences.length > 1 ? `（${sentences.length}）` : ""}：
      </span>
      <span className={cn("block", !expanded && "line-clamp-1")}>
        {sentences.join("；")}
      </span>
    </span>
  );
}

/**
 * 图片类型 → 徽标的图标与配色.
 *
 * 这是三层图片处理在界面上的体现：一张图片到底被当成"表格""图表"还是
 * "流程图"处理（进而走的是 Table Parser 还是 Vision），用户一眼能看出来。
 */
const IMAGE_TYPE_STYLE: Record<
  string,
  { icon: typeof Table2; className: string; label: string }
> = {
  table: {
    icon: Table2,
    className: "bg-amber-500/15 text-amber-600 dark:text-amber-300",
    label: IMAGE_TYPE_LABELS.table,
  },
  // 公式与代码截图此前没有条目 → 落到 `?? IMAGE_TYPE_STYLE.photo` 兜底，
  // 在界面上被显示成"图片"。后端已经识别出 formula / code 并走了专用引擎，
  // 这里必须把它们如实呈现，否则"公式提取"这件事在 UI 上是不可见的。
  formula: {
    icon: Sigma,
    className: "bg-fuchsia-500/15 text-fuchsia-600 dark:text-fuchsia-300",
    label: IMAGE_TYPE_LABELS.formula,
  },
  code: {
    icon: Code2,
    className: "bg-slate-500/15 text-slate-600 dark:text-slate-300",
    label: IMAGE_TYPE_LABELS.code,
  },
  chart: {
    icon: BarChart3,
    className: "bg-sky-500/15 text-sky-600 dark:text-sky-300",
    label: IMAGE_TYPE_LABELS.chart,
  },
  diagram: {
    icon: GitBranch,
    className: "bg-emerald-500/15 text-emerald-600 dark:text-emerald-300",
    label: IMAGE_TYPE_LABELS.diagram,
  },
  screenshot: {
    icon: MonitorPlay,
    className: "bg-cyan-500/15 text-cyan-600 dark:text-cyan-300",
    label: IMAGE_TYPE_LABELS.screenshot,
  },
  photo: {
    icon: ImageIcon,
    className: "bg-violet-500/15 text-violet-600 dark:text-violet-300",
    label: IMAGE_TYPE_LABELS.photo,
  },
};

function TypeBadge({ source }: { source: QuerySource }) {
  if (isImageSource(source)) {
    const style = IMAGE_TYPE_STYLE[source.imageType ?? "photo"] ?? IMAGE_TYPE_STYLE.photo;
    const Icon = style.icon;
    return (
      <span
        className={cn(
          "inline-flex shrink-0 items-center gap-0.5 rounded-full px-2 py-0.5 text-[10px]",
          style.className
        )}
      >
        <Icon className="size-2.5" />
        {style.label}
      </span>
    );
  }
  if (source.contentType === "table") {
    // 表格图片被 Table Parser 还原成真表格后，content_type 也是 table，
    // 用不同文案把"图片里的表格"与"正文表格"区分开。
    const fromImage = source.imageType === "table";
    return (
      <span className="inline-flex shrink-0 items-center gap-0.5 rounded-full bg-amber-500/15 px-2 py-0.5 text-[10px] text-amber-600 dark:text-amber-300">
        <Table2 className="size-2.5" />
        {fromImage ? "表格·图片还原" : "表格"}
      </span>
    );
  }
  return null;
}

/**
 * 一句话溯源：`出自《年报.pdf》第 3 页，第 12-28 行`.
 *
 * 这是"溯源时用一句话标明这是检索文档的哪几行"在界面上的落点 ——
 * 行号来自后端切片时写入的位置信息，不是前端估算。
 */
function LocationLine({ source }: { source: QuerySource }) {
  const sentence = locationSentence(source);
  if (!sentence) return null;
  return (
    <span className="mt-1 flex items-center gap-1 text-[10px] text-muted-foreground/80">
      <MapPin className="size-2.5 shrink-0" />
      <span className="truncate">出自 {sentence}</span>
    </span>
  );
}

/** 引用校验徽标：已核验 / 存疑（Citation Verifier 的结论）。 */
function VerifiedBadge({ passed }: { passed: boolean | undefined }) {
  if (passed === undefined) return null;
  return passed ? (
    <span
      className="inline-flex shrink-0 items-center gap-0.5 rounded-full bg-emerald-500/15 px-2 py-0.5 text-[10px] text-emerald-600 dark:text-emerald-300"
      title="引用校验通过：来源存在、位置正确、原文支持该结论，数字与日期一致"
    >
      <ShieldCheck className="size-2.5" />
      已核验
    </span>
  ) : (
    <span
      className="inline-flex shrink-0 items-center gap-0.5 rounded-full bg-amber-500/15 px-2 py-0.5 text-[10px] text-amber-600 dark:text-amber-300"
      title="引用校验存疑：原文未能直接支持该结论，或数字/日期与原文不一致"
    >
      <ShieldAlert className="size-2.5" />
      存疑
    </span>
  );
}

/**
 * 存疑引用的**逐项失败原因**。
 *
 * 挂在引用卡片下方：告诉用户这条引用到底哪一项没过（来源不存在 / 位置存疑 /
 * 原文未直接支持 / 数字不一致 / 日期不一致），并把可核对的证据一并给出
 * —— 缺失的数字与日期、句子内容词在被引原文里的覆盖率。
 * "为什么标存疑"必须能自证，否则用户只能选择无视这个徽标。
 */
function VerdictDetail({ verdict }: { verdict: CitationVerdict }) {
  if (verdict.passed) return null;
  const reasons = verdict.failed_checks
    .map((k) => CHECK_LABELS[k] ?? k)
    .filter(Boolean);
  if (!reasons.length) return null;

  return (
    <span className="mt-1.5 block rounded-md border border-amber-300/50 bg-amber-50/60 px-2 py-1 text-[10px] leading-relaxed text-amber-800 dark:border-amber-700/40 dark:bg-amber-950/25 dark:text-amber-200">
      <span className="flex flex-wrap items-center gap-1">
        <CircleAlert className="size-2.5 shrink-0" />
        <span className="font-medium">校验未通过：</span>
        {reasons.map((r) => (
          <span
            key={r}
            className="rounded-full bg-amber-500/15 px-1.5 py-px font-medium"
          >
            {r}
          </span>
        ))}
      </span>
      {verdict.best_source ? (
        <span className="mt-0.5 block opacity-90">
          更像是出自来源 {verdict.best_source}
        </span>
      ) : null}
      {verdict.missing_numbers.length ? (
        <span className="mt-0.5 block opacity-90">
          原文中找不到的数字：{verdict.missing_numbers.join("、")}
        </span>
      ) : null}
      {verdict.missing_dates.length ? (
        <span className="mt-0.5 block opacity-90">
          原文中找不到的日期：{verdict.missing_dates.join("、")}
        </span>
      ) : null}
      <span className="mt-0.5 block opacity-80">
        句子内容词覆盖率 {Math.round((verdict.support_ratio ?? 0) * 100)}%
      </span>
    </span>
  );
}

/** Markdown 表格 → 等宽预览（让"结构化"这件事在界面上可见）。 */
function MarkdownTablePreview({ text }: { text: string }) {
  const lines = text.split("\n").filter((line) => line.trim().startsWith("|"));
  if (lines.length < 2) {
    return (
      <pre className="mt-1.5 overflow-x-auto rounded-md bg-muted/50 p-2 font-mono text-[11px] leading-relaxed text-muted-foreground">
        {text}
      </pre>
    );
  }
  const cells = (line: string) =>
    line
      .trim()
      .replace(/^\||\|$/g, "")
      .split("|")
      .map((cell) => cell.trim());
  const header = cells(lines[0]);
  const body = lines.slice(2).map(cells); // 第 2 行是 |---|---| 分隔行
  return (
    <span className="mt-1.5 block overflow-x-auto">
      <table className="w-full border-collapse text-[11px]">
        <thead>
          <tr>
            {header.map((cell, i) => (
              <th
                key={i}
                className="border border-border/60 bg-muted/60 px-2 py-1 text-left font-medium text-foreground/85"
              >
                {cell}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {body.map((row, r) => (
            <tr key={r}>
              {row.map((cell, c) => (
                <td
                  key={c}
                  className="border border-border/50 px-2 py-1 text-muted-foreground"
                >
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </span>
  );
}

/** 图片引用：直接展示检索命中的**原始图片**（部分2「返回原始图片」）。 */
function ImageSourceRow({
  source,
  index,
  expanded,
  onToggle,
  passing,
  failing,
}: {
  source: QuerySource;
  index: number;
  expanded: boolean;
  onToggle: () => void;
  /** 该来源下所有引用是否全部通过校验（无引用时为 undefined）。 */
  passing?: boolean;
  /** 未通过的逐条结论（一条来源可能被多句引用）。 */
  failing: CitationVerdict[];
}) {
  const score = relevancePercent(source.score);
  const src = getDocumentImageUrl(source.imageUrl);
  const [broken, setBroken] = useState(false);
  const style = IMAGE_TYPE_STYLE[source.imageType ?? "photo"] ?? IMAGE_TYPE_STYLE.photo;
  const TypeIcon = style.icon;
  const span = lineSpanOf(source);

  return (
    <div
      id={sourceAnchorId(index + 1)}
      className="scroll-mt-6 rounded-xl border border-violet-500/25 bg-background/50 p-3 transition-all hover:border-violet-500/40 hover:shadow-sm target:border-violet-500/60 target:ring-2 target:ring-violet-500/30"
    >
      <button
        type="button"
        onClick={onToggle}
        className="flex w-full gap-3 text-left"
      >
        <span className="flex size-5 shrink-0 items-center justify-center rounded-md bg-violet-500/15 text-[10px] font-semibold text-violet-600 dark:text-violet-300">
          {index + 1}
        </span>
        <span className="min-w-0 flex-1">
          <span className="flex flex-wrap items-center gap-2">
            <TypeIcon className="size-3.5 shrink-0 text-violet-500/80" />
            <span className="truncate text-xs font-medium text-foreground/90">
              {source.documentName ?? "未知文档"}
            </span>
            <TypeBadge source={source} />
            <VerifiedBadge passed={passing} />
            {typeof source.page === "number" ? (
              <span className="shrink-0 rounded-full bg-muted px-2 py-0.5 text-[10px] text-muted-foreground">
                第 {source.page} 页
              </span>
            ) : null}
            {span && (
              <span className="shrink-0 rounded-full bg-muted px-2 py-0.5 text-[10px] text-muted-foreground">
                第 {span} 行
              </span>
            )}
            {score && (
              <span className="ml-auto shrink-0 font-mono text-[10px] text-muted-foreground/70">
                {score}
              </span>
            )}
          </span>
          <LocationLine source={source} />
          {source.imageCaption && (
            <span
              className={cn(
                "mt-1.5 block text-[11px] leading-relaxed text-muted-foreground",
                !expanded && "line-clamp-2"
              )}
            >
              {source.imageCaption}
            </span>
          )}
          {source.vision && (
            <span
              className={cn(
                "mt-1 block rounded-md bg-violet-500/5 px-2 py-1 text-[11px] leading-relaxed text-violet-700/90 dark:text-violet-200/90",
                !expanded && "line-clamp-3"
              )}
            >
              视觉分析：{source.vision}
            </span>
          )}
        </span>
      </button>

      {/* 存疑时给出逐项失败原因（一条来源可能被多句引用 → 逐条列出） */}
      {failing.map((v, i) => (
        <VerdictDetail key={`${v.index}-${i}`} verdict={v} />
      ))}

      {expanded && src && !broken && (
        <div className="mt-2">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={src}
            alt={source.imageCaption || source.documentName || "文档图片"}
            onError={() => setBroken(true)}
            className="max-h-72 w-full rounded-lg border border-border/60 bg-muted/30 object-contain"
          />
          <a
            href={src}
            target="_blank"
            rel="noreferrer"
            className="mt-1.5 inline-flex items-center gap-1 text-[10px] text-violet-600 hover:underline dark:text-violet-300"
          >
            <ExternalLink className="size-2.5" />
            在新标签页打开原图
          </a>
        </div>
      )}
      {expanded && (!src || broken) && (
        <p className="mt-2 text-[11px] text-muted-foreground">
          原始图片不可用（可能未开启图片落盘，或文件已被清理）。
        </p>
      )}
    </div>
  );
}

function SourceRow({
  source,
  index,
  verdicts = [],
}: {
  source: QuerySource;
  index: number;
  /** 该来源对应的全部引用结论（同一来源可能被多句答案引用）。 */
  verdicts?: CitationVerdict[];
}) {
  const [expanded, setExpanded] = useState(false);
  const score = relevancePercent(source.score);
  const span = lineSpanOf(source);

  // 全部通过才算"已核验"：一条来源被两句引用、其中一句存疑时，整条标存疑
  // —— 否则用户看到"已核验"，而其中一条引用其实没被原文支持。
  const passing = verdicts.length ? verdicts.every((v) => v.passed) : undefined;
  // 命中句：答案实际依据的是哪几句原文（同一来源的多条引用取并集）
  const evidence = collectEvidence(verdicts);
  const sentences = Array.from(
    new Set(verdicts.map((v) => v.sentence).filter(Boolean))
  );
  const failing = verdicts.filter((v) => !v.passed);

  if (isImageSource(source)) {
    return (
      <ImageSourceRow
        source={source}
        index={index}
        expanded={expanded}
        onToggle={() => setExpanded((v) => !v)}
        passing={passing}
        failing={failing}
      />
    );
  }

  return (
    <div
      id={sourceAnchorId(index + 1)}
      className="scroll-mt-6 rounded-xl border border-border/50 bg-background/50 transition-all target:border-primary/50 target:ring-2 target:ring-primary/25"
    >
      <button
        type="button"
        onClick={() => setExpanded((v) => !v)}
        className="group flex w-full gap-3 rounded-xl p-3 text-left transition-all hover:bg-accent/40"
      >
        <span className="flex size-5 shrink-0 items-center justify-center rounded-md bg-primary/10 text-[10px] font-semibold text-primary">
          {index + 1}
        </span>
        <span className="min-w-0 flex-1">
          <span className="flex flex-wrap items-center gap-2">
            <FileText className="size-3.5 shrink-0 text-primary/60" />
            <span className="truncate text-xs font-medium text-foreground/90">
              {source.documentName ?? "未知文档"}
            </span>
            <TypeBadge source={source} />
            <VerifiedBadge passed={passing} />
            {typeof source.page === "number" ? (
              <span className="shrink-0 rounded-full bg-muted px-2 py-0.5 text-[10px] text-muted-foreground">
                第 {source.page} 页
              </span>
            ) : typeof source.pages === "number" && source.pages > 0 ? (
              <span className="shrink-0 rounded-full bg-muted px-2 py-0.5 text-[10px] text-muted-foreground">
                共 {source.pages} 页
              </span>
            ) : null}
            {span && (
              <span className="shrink-0 rounded-full bg-muted px-2 py-0.5 text-[10px] text-muted-foreground">
                第 {span} 行
              </span>
            )}
            {/* 命中句行号：把"整块 83-105 行"收紧到"实际用的第 84-85 行" */}
            <EvidenceBadge spans={evidence} />
            {score && (
              <span className="ml-auto shrink-0 font-mono text-[10px] text-muted-foreground/70">
                {score}
              </span>
            )}
          </span>
          {/* 一句话溯源：这是文档的哪几行 */}
          <LocationLine source={source} />
          {/* 引用正文：收起时只列命中句，展开时整段原文 + 高亮命中句 */}
          {source.chunkText &&
            (source.contentType === "table" && source.chunkText.includes("|") ? (
              // 表格类引用（正文表格或表格图片经 Table Parser 还原的结果）：
              // 直接把 Markdown 渲染成表格，而不是丢一段带竖线的纯文本。
              <MarkdownTablePreview text={source.chunkText} />
            ) : (
              <EvidenceExcerpt
                text={source.chunkText}
                spans={evidence}
                expanded={expanded}
              />
            ))}
          {/* 这条引用支撑了答案里的哪句话（引用是双向的） */}
          <SupportedSentences sentences={sentences} expanded={expanded} />
          {source.chunkText && source.chunkText.length > 120 && (
            <span className="mt-1 inline-block text-[10px] text-primary/60 opacity-0 transition-opacity group-hover:opacity-100">
              {expanded ? "只看命中句" : "展开全文与上下文"}
            </span>
          )}
        </span>
      </button>
      {/* 存疑时给出逐项失败原因（一条来源可能被多句引用 → 逐条列出） */}
      <div className="px-3 pb-3">
        {failing.map((v, i) => (
          <VerdictDetail key={`${v.index}-${i}`} verdict={v} />
        ))}
      </div>
    </div>
  );
}

/**
 * 无依据句的**显式标注**（#17 用户裁定：不静默，UI 明确标出）.
 *
 * 引用校验发现"找不到来源支持"的句子时，后端会移除该句的引用标记、并在答案
 * 末尾追加一行脚注 —— 但那只是一笔带过，用户看不出**具体哪句**没依据（静默）。
 * 这里把这些句子在界面上一句句标出来，且**不改动答案正文**（纯旁注）。
 *
 * 排除 `evidence_available === false` 的条目：来源拿不到 token/片段时
 * "未支持"只是无从判断，标出来会把中文句、图片块(vision)、表格块来源的
 * 句子误标 —— 宁可少标，不可错标（与后端 evidence_available 同源）。
 */
function UnsupportedSentences({ verdicts }: { verdicts: CitationVerdict[] }) {
  const seen = new Set<string>();
  const sentences = verdicts
    .filter((v) => !v.supported && v.evidence_available !== false)
    .map((v) => (v.sentence ?? "").trim())
    .filter((s) => {
      if (!s || seen.has(s)) return false;
      seen.add(s);
      return true;
    });
  if (!sentences.length) return null;
  return (
    <div className="mb-2 rounded-lg border border-amber-300/60 bg-amber-50/70 px-2.5 py-2 text-[11px] leading-relaxed text-amber-800 dark:border-amber-700/40 dark:bg-amber-950/30 dark:text-amber-200">
      <span className="flex items-start gap-1 font-medium">
        <ShieldAlert className="mt-px size-3.5 shrink-0" />
        <span>
          以下 {sentences.length} 句未找到来源支持（其引用标记已移除，请以原文为准）
        </span>
      </span>
      <ul className="mt-1 list-disc space-y-0.5 pl-4">
        {sentences.map((s, i) => (
          <li key={i} className="opacity-90">
            {s}
          </li>
        ))}
      </ul>
    </div>
  );
}

export function SourceCitations({
  sources,
  citationCheck,
  refused = false,
  refusalNote,
}: {
  sources: QuerySource[];
  /** Citation Verifier 的五项校验结论（可选；缺失时不显示校验徽标）。 */
  citationCheck?: CitationCheckInfo;
  /**
   * 本次答复是否为拒答。
   *
   * 拒答意味着"检索到了片段，但证据不足以支撑结论"，所以这些来源**不是**
   * 回答的依据。以前不管拒答与否都按同样样式展示，用户看到的就是
   * "系统说知识库里没找到资料"＋"旁边挂着 1 个引用来源"的自相矛盾画面。
   */
  refused?: boolean;
  /** 拒答时展示的说明文案（后端下发）。 */
  refusalNote?: string;
}) {
  // 拒答时默认收起：来源不是依据，展开只是为了排查"到底检索到了什么"
  const [open, setOpen] = useState(!refused);
  if (!sources.length) return null;

  const imageCount = sources.filter(isImageSource).length;

  // verdict.index 是 1-based 的 [Source N] 编号，与 sources 顺序一一对应。
  // 存成**列表**：同一条来源被多句答案引用时会有多条结论，以前只留最后一条
  // —— 于是"其中一句存疑"会被后一条覆盖成"已核验"，徽标与事实相反。
  const verdictsByIndex = new Map<number, CitationVerdict[]>();
  for (const v of citationCheck?.verdicts ?? []) {
    const list = verdictsByIndex.get(v.index);
    if (list) list.push(v);
    else verdictsByIndex.set(v.index, [v]);
  }

  const check = citationCheck;
  const showCheckSummary =
    check && check.overall !== "no_citations" && check.overall !== "refused_by_model";
  // 命中句总条数（引用卡片把引用从"块级"收紧到"句级"的量化体现）
  const hitSentenceCount = (check?.verdicts ?? []).reduce(
    (n, v) => n + (v.evidence?.length ?? 0),
    0
  );

  return (
    <div className="mt-3 border-t border-border/50 pt-3">
      {/* 拒答说明：把"为什么答不出来"和"这些来源不算依据"讲清楚 */}
      {refused && (
        <p className="mb-2 flex items-start gap-1.5 rounded-lg border border-amber-300/50 bg-amber-50/70 px-2.5 py-2 text-[11px] leading-relaxed text-amber-800 dark:border-amber-700/40 dark:bg-amber-950/30 dark:text-amber-200">
          <ShieldAlert className="mt-px size-3.5 shrink-0" />
          <span>
            {refusalNote ||
              "本次答复为拒答：检索到的片段不足以支撑结论，以下来源未被采用。"}
          </span>
        </p>
      )}
      {/* 无依据句显式标注（#17）：不静默 —— 无论引用列表是否展开都标出，
          且只做旁注、不改动答案正文。 */}
      <UnsupportedSentences verdicts={check?.verdicts ?? []} />
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-1.5 text-left"
      >
        <BookMarked className="size-3.5 text-muted-foreground" />
        <span className="text-xs font-medium text-muted-foreground">
          {refused ? "检索到的片段（未采用）" : "引用来源"}
        </span>
        <span className="rounded-full bg-muted px-1.5 py-px text-[10px] font-medium text-muted-foreground">
          {sources.length}
        </span>
        {imageCount > 0 && (
          <span className="inline-flex items-center gap-0.5 rounded-full bg-violet-500/15 px-1.5 py-px text-[10px] font-medium text-violet-600 dark:text-violet-300">
            <ImageIcon className="size-2.5" />
            {imageCount} 图
          </span>
        )}
        {/* 引用校验总览：把"这几条引用可不可信"压缩成一个徽标 */}
        {showCheckSummary && (
          <span
            className={cn(
              "inline-flex items-center gap-0.5 rounded-full px-1.5 py-px text-[10px] font-medium",
              check.overall === "verified"
                ? "bg-emerald-500/15 text-emerald-600 dark:text-emerald-300"
                : "bg-amber-500/15 text-amber-600 dark:text-amber-300"
            )}
            title={`引用校验：${check.passed}/${check.total} 条通过`}
          >
            {check.overall === "verified" ? (
              <ShieldCheck className="size-2.5" />
            ) : (
              <ShieldAlert className="size-2.5" />
            )}
            {check.overall === "verified"
              ? `${check.total} 条已核验`
              : `${check.total - check.passed} 条存疑`}
          </span>
        )}
        {/* 命中句总数：一眼看出"这批引用标到了多少句依据" */}
        {hitSentenceCount > 0 && (
          <span
            className="inline-flex items-center gap-0.5 rounded-full bg-amber-500/15 px-1.5 py-px text-[10px] font-medium text-amber-700 dark:text-amber-300"
            title="已回标每条引用实际依据的原文句子；点击任一来源可只看这几句"
          >
            <Highlighter className="size-2.5" />
            {hitSentenceCount} 句命中
          </span>
        )}
        <ChevronDown
          className={cn(
            "ml-auto size-3.5 text-muted-foreground/60 transition-transform",
            open && "rotate-180"
          )}
        />
      </button>

      {open && (
        <div className="mt-2 space-y-1.5">
          {sources.map((source, i) => (
            <SourceRow
              key={`${source.documentId ?? "src"}-${source.imageId ?? i}-${i}`}
              source={source}
              index={i}
              verdicts={verdictsByIndex.get(i + 1)}
            />
          ))}
        </div>
      )}
    </div>
  );
}
