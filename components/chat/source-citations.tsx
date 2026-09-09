"use client";

import { useState } from "react";
import {
  ChevronDown,
  FileText,
  BookMarked,
} from "lucide-react";
import type { QuerySource } from "@/lib/types";
import { cn } from "@/lib/utils";

function relevancePercent(score?: number): string | null {
  if (typeof score !== "number" || Number.isNaN(score)) return null;
  return `${Math.round(score * 100)}%`;
}

function SourceRow({
  source,
  index,
}: {
  source: QuerySource;
  index: number;
}) {
  const [expanded, setExpanded] = useState(false);
  const score = relevancePercent(source.score);

  return (
    <button
      type="button"
      onClick={() => setExpanded((v) => !v)}
      className="group flex w-full gap-3 rounded-xl border border-border/50 bg-background/50 p-3 text-left transition-all hover:border-primary/25 hover:bg-accent/40 hover:shadow-sm"
    >
      <span className="flex size-5 shrink-0 items-center justify-center rounded-md bg-primary/10 text-[10px] font-semibold text-primary">
        {index + 1}
      </span>
      <span className="min-w-0 flex-1">
        <span className="flex items-center gap-2">
          <FileText className="size-3.5 shrink-0 text-primary/60" />
          <span className="truncate text-xs font-medium text-foreground/90">
            {source.documentName ?? "未知文档"}
          </span>
          {typeof source.page === "number" ? (
            <span className="shrink-0 rounded-full bg-muted px-2 py-0.5 text-[10px] text-muted-foreground">
              第 {source.page} 页
            </span>
          ) : typeof source.pages === "number" && source.pages > 0 ? (
            <span className="shrink-0 rounded-full bg-muted px-2 py-0.5 text-[10px] text-muted-foreground">
              共 {source.pages} 页
            </span>
          ) : null}
          {score && (
            <span className="ml-auto shrink-0 font-mono text-[10px] text-muted-foreground/70">
              {score}
            </span>
          )}
        </span>
        {source.chunkText && (
          <span
            className={cn(
              "mt-1.5 block text-[11px] leading-relaxed text-muted-foreground",
              !expanded && "line-clamp-2"
            )}
          >
            {source.chunkText}
          </span>
        )}
        {source.chunkText && source.chunkText.length > 120 && (
          <span className="mt-1 inline-block text-[10px] text-primary/60 opacity-0 transition-opacity group-hover:opacity-100">
            {expanded ? "收起" : "展开全文"}
          </span>
        )}
      </span>
    </button>
  );
}

export function SourceCitations({ sources }: { sources: QuerySource[] }) {
  const [open, setOpen] = useState(true);
  if (!sources.length) return null;

  return (
    <div className="mt-3 border-t border-border/50 pt-3">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-1.5 text-left"
      >
        <BookMarked className="size-3.5 text-muted-foreground" />
        <span className="text-xs font-medium text-muted-foreground">
          引用来源
        </span>
        <span className="rounded-full bg-muted px-1.5 py-px text-[10px] font-medium text-muted-foreground">
          {sources.length}
        </span>
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
            <SourceRow key={`${source.documentId ?? "src"}-${i}`} source={source} index={i} />
          ))}
        </div>
      )}
    </div>
  );
}
