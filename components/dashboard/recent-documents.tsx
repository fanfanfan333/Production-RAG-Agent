"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import {
  AlertCircle,
  ArrowRight,
  Clock,
  FileSpreadsheet,
  FileText,
  FileType,
  Loader2,
  MessageSquare,
  Trash2,
} from "lucide-react";
import { toast } from "sonner";
import { deleteDocument } from "@/lib/api/documents";
import { useApp } from "@/lib/context/app-context";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import type { Document, DocumentStatus } from "@/lib/types";
import { formatBytes, formatRelativeTime } from "@/lib/utils";

const typeIcons: Record<string, typeof FileText> = {
  PDF: FileText,
  Markdown: FileType,
  表格: FileSpreadsheet,
  文档: FileText,
  文本: FileType,
  演示文稿: FileText,
};

const statusConfig: Record<
  DocumentStatus,
  { label: string; variant: "success" | "warning" | "destructive" }
> = {
  indexed: { label: "已索引", variant: "success" },
  processing: { label: "处理中", variant: "warning" },
  failed: { label: "失败", variant: "destructive" },
  already_exists: { label: "已存在", variant: "success" },
};

type FilterKey = "all" | "indexed" | "processing" | "failed";

interface RecentDocumentsProps {
  documents: Document[];
  loading?: boolean;
  error?: string | null;
  limit?: number;
}

/**
 * 最近文档列表（主界面互动优化）：
 * - 状态筛选 chips（全部 / 已索引 / 处理中 / 失败，按实际数量显示）
 * - 行内快捷操作：「针对此文档提问」一键带入对话草稿；删除带确认弹窗
 */
export function RecentDocuments({
  documents,
  loading,
  error,
  limit = 5,
}: RecentDocumentsProps) {
  const router = useRouter();
  const { refresh } = useApp();
  const [filter, setFilter] = useState<FilterKey>("all");
  const [pendingDelete, setPendingDelete] = useState<Document | null>(null);
  const [deleting, setDeleting] = useState(false);

  const counts = useMemo(() => {
    const c: Record<FilterKey, number> = {
      all: documents.length,
      indexed: 0,
      processing: 0,
      failed: 0,
    };
    for (const d of documents) {
      if (d.status === "indexed" || d.status === "already_exists") c.indexed++;
      else if (d.status === "processing") c.processing++;
      else if (d.status === "failed") c.failed++;
    }
    return c;
  }, [documents]);

  const filters: { key: FilterKey; label: string }[] = [
    { key: "all", label: `全部 ${counts.all}` },
    { key: "indexed", label: `已索引 ${counts.indexed}` },
    { key: "processing", label: `处理中 ${counts.processing}` },
    { key: "failed", label: `失败 ${counts.failed}` },
  ];

  const recent = useMemo(() => {
    const filtered =
      filter === "all"
        ? documents
        : documents.filter((d) =>
            filter === "indexed"
              ? d.status === "indexed" || d.status === "already_exists"
              : d.status === filter
          );
    return [...filtered]
      .sort((a, b) => b.uploadedAt.getTime() - a.uploadedAt.getTime())
      .slice(0, limit);
  }, [documents, filter, limit]);

  const askAbout = (doc: Document) => {
    const baseName = doc.name.replace(/\.[^.]+$/, "");
    router.push(
      `/chat?draft=${encodeURIComponent(
        `请介绍《${baseName}》的主要内容`
      )}`
    );
  };

  const confirmDelete = async () => {
    if (!pendingDelete) return;
    setDeleting(true);
    try {
      await deleteDocument(pendingDelete.id);
      toast.success(`已删除「${pendingDelete.name}」`);
      setPendingDelete(null);
      refresh();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "删除失败");
    } finally {
      setDeleting(false);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.2 }}
    >
      <Card className="h-full">
        <CardHeader>
          <CardTitle>最近文档</CardTitle>
          <CardDescription>
            最近上传的文档，可筛选状态或直接针对文档提问
          </CardDescription>
          <CardAction>
            <Button
              variant="ghost"
              size="sm"
              className="text-muted-foreground"
              asChild
            >
              <Link href="/documents">
                查看全部
                <ArrowRight className="size-3.5" />
              </Link>
            </Button>
          </CardAction>
        </CardHeader>
        <CardContent className="px-0">
          {/* 状态筛选 */}
          {!error && !loading && (
            <div className="flex flex-wrap gap-1.5 px-6 pb-3">
              {filters
                .filter((f) => f.key === "all" || counts[f.key] > 0)
                .map((f) => (
                  <button
                    key={f.key}
                    type="button"
                    onClick={() => setFilter(f.key)}
                    className={cn(
                      "rounded-full border px-3 py-1 text-xs transition-colors",
                      filter === f.key
                        ? "border-primary/50 bg-primary/10 font-medium text-primary"
                        : "border-border text-muted-foreground hover:border-foreground/25 hover:text-foreground"
                    )}
                    aria-pressed={filter === f.key}
                  >
                    {f.label}
                  </button>
                ))}
            </div>
          )}

          {error ? (
            <div className="flex items-center gap-3 px-6 py-8 text-sm text-destructive">
              <AlertCircle className="size-4 shrink-0" />
              <span>{error}</span>
            </div>
          ) : loading ? (
            <div className="space-y-4 px-6 py-2">
              {Array.from({ length: 3 }).map((_, i) => (
                <div key={i} className="flex items-center gap-4">
                  <Skeleton className="size-10 rounded-lg" />
                  <div className="flex-1 space-y-2">
                    <Skeleton className="h-4 w-48" />
                    <Skeleton className="h-3 w-32" />
                  </div>
                </div>
              ))}
            </div>
          ) : recent.length === 0 ? (
            <p className="px-6 py-8 text-center text-sm text-muted-foreground">
              {filter === "all"
                ? "暂无文档，先上传一个 PDF 开始使用吧。"
                : "该状态下暂无文档。"}
            </p>
          ) : (
            <ul className="divide-y divide-border/60">
              {recent.map((doc, index) => {
                const Icon = typeIcons[doc.type] ?? FileText;
                const status = statusConfig[doc.status];
                const canAsk =
                  doc.status === "indexed" || doc.status === "already_exists";

                return (
                  <motion.li
                    key={doc.id}
                    initial={{ opacity: 0, x: -8 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ duration: 0.3, delay: 0.25 + index * 0.05 }}
                    className="group flex items-center gap-4 px-6 py-4 transition-colors hover:bg-muted/40"
                  >
                    <div className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-muted">
                      {doc.status === "processing" ? (
                        <Loader2 className="size-4 animate-spin text-muted-foreground" />
                      ) : (
                        <Icon className="size-4 text-muted-foreground" />
                      )}
                    </div>
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm font-medium">{doc.name}</p>
                      <div className="mt-0.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
                        <span>{doc.type}</span>
                        <span>{formatBytes(doc.size)}</span>
                        <span>{doc.chunks} 个分块</span>
                      </div>
                    </div>
                    <div className="hidden shrink-0 items-center gap-3 sm:flex">
                      <span className="flex items-center gap-1 text-xs text-muted-foreground">
                        <Clock className="size-3" />
                        {formatRelativeTime(doc.uploadedAt)}
                      </span>
                      <Badge variant={status.variant}>{status.label}</Badge>
                    </div>

                    {/* 行内操作：hover 显示（触屏常显） */}
                    <div className="flex shrink-0 items-center gap-1 opacity-100 transition-opacity sm:opacity-0 sm:group-hover:opacity-100 sm:group-focus-within:opacity-100">
                      <Button
                        variant="ghost"
                        size="icon"
                        className="size-8"
                        title={canAsk ? "针对此文档提问" : "文档处理完成后可提问"}
                        aria-label={`针对 ${doc.name} 提问`}
                        disabled={!canAsk}
                        onClick={() => askAbout(doc)}
                      >
                        <MessageSquare className="size-3.5" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="size-8 text-muted-foreground hover:text-destructive"
                        title="删除文档"
                        aria-label={`删除 ${doc.name}`}
                        onClick={() => setPendingDelete(doc)}
                      >
                        <Trash2 className="size-3.5" />
                      </Button>
                    </div>

                    <Badge
                      variant={status.variant}
                      className="shrink-0 sm:hidden"
                    >
                      {status.label}
                    </Badge>
                  </motion.li>
                );
              })}
            </ul>
          )}
        </CardContent>
      </Card>

      {/* 删除确认 */}
      <Dialog
        open={!!pendingDelete}
        onOpenChange={(open) => {
          if (!open) setPendingDelete(null);
        }}
      >
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>删除文档</DialogTitle>
            <DialogDescription>
              确定要删除「{pendingDelete?.name}」吗？该文档的分块与向量索引将一并移除，操作不可恢复。
            </DialogDescription>
          </DialogHeader>
          <div className="flex justify-end gap-2">
            <Button
              variant="outline"
              onClick={() => setPendingDelete(null)}
              disabled={deleting}
            >
              取消
            </Button>
            <Button
              variant="destructive"
              onClick={confirmDelete}
              disabled={deleting}
            >
              {deleting && <Loader2 className="size-4 animate-spin" />}
              确认删除
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </motion.div>
  );
}
