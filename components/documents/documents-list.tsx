"use client";

import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import {
  AlertCircle,
  Clock,
  Eye,
  FileSpreadsheet,
  FileText,
  FileType,
  Loader2,
  Trash2,
} from "lucide-react";
import { toast } from "sonner";
import { deleteDocument, getDocumentChunks } from "@/lib/api/documents";
import { useApp } from "@/lib/context/app-context";
import { useDocuments } from "@/lib/hooks/use-documents";
import type { Document, DocumentChunksResponse, DocumentStatus } from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
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

/** 文档内容查看器：按分块展示索引后的文本（含页码） */
function DocumentViewer({ doc }: { doc: Document }) {
  const [data, setData] = useState<DocumentChunksResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // 弹窗打开时加载一次（组件随 key=doc.id 重挂载）
  useEffect(() => {
    let cancelled = false;
    getDocumentChunks(doc.id)
      .then((res) => {
        if (!cancelled) setData(res);
      })
      .catch((err) => {
        if (!cancelled)
          setError(err instanceof Error ? err.message : "加载文档内容失败");
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [doc.id]);

  return (
    <div className="max-h-[60vh] overflow-y-auto pr-1">
      {loading ? (
        <div className="space-y-3">
          {Array.from({ length: 3 }).map((_, i) => (
            <Skeleton key={i} className="h-20 w-full" />
          ))}
        </div>
      ) : error ? (
        <div className="flex items-center gap-2 py-6 text-sm text-destructive">
          <AlertCircle className="size-4 shrink-0" />
          {error}
        </div>
      ) : !data || data.chunks.length === 0 ? (
        <p className="py-6 text-center text-sm text-muted-foreground">
          该文档暂无索引内容（可能仍在处理中）
        </p>
      ) : (
        <div className="space-y-2.5">
          <p className="text-xs text-muted-foreground">
            共 {data.total} 个分块 · {data.pageCount} 页
          </p>
          {data.chunks.map((chunk) => (
            <div
              key={chunk.chunkIndex}
              className="rounded-lg border border-border/60 bg-muted/30 p-3"
            >
              <p className="mb-1 text-[10px] font-medium text-muted-foreground">
                分块 #{chunk.chunkIndex + 1} · 第 {chunk.pageNumber} 页
              </p>
              <p className="whitespace-pre-wrap text-xs leading-relaxed text-foreground/90">
                {chunk.text}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export function DocumentsList() {
  const { refresh } = useApp();
  const { documents, loading, error, refetch } = useDocuments({
    pollProcessing: true,
  });
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [viewingDoc, setViewingDoc] = useState<Document | null>(null);

  const handleDelete = async (id: string, name: string) => {
    setDeletingId(id);
    try {
      await deleteDocument(id);
      toast.success(`已删除「${name}」`);
      refresh();
      refetch();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "删除失败");
    } finally {
      setDeletingId(null);
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>全部文档</CardTitle>
        <CardDescription>
          查看已上传的全部文档，可打开查看内容或删除
        </CardDescription>
      </CardHeader>
      <CardContent className="px-0">
        {error ? (
          <div className="flex items-center gap-3 px-6 py-8 text-sm text-destructive">
            <AlertCircle className="size-4 shrink-0" />
            <span>{error}</span>
          </div>
        ) : loading ? (
          <div className="space-y-4 px-6 py-2">
            {Array.from({ length: 5 }).map((_, i) => (
              <div key={i} className="flex items-center gap-4">
                <Skeleton className="size-10 rounded-lg" />
                <div className="flex-1 space-y-2">
                  <Skeleton className="h-4 w-48" />
                  <Skeleton className="h-3 w-32" />
                </div>
              </div>
            ))}
          </div>
        ) : documents.length === 0 ? (
          <p className="px-6 py-8 text-center text-sm text-muted-foreground">
            尚未上传任何文档。
          </p>
        ) : (
          <ul className="divide-y divide-border/60">
            {documents.map((doc, index) => {
              const Icon = typeIcons[doc.type] ?? FileText;
              const status = statusConfig[doc.status];
              const canView =
                doc.status === "indexed" || doc.status === "already_exists";

              return (
                <motion.li
                  key={doc.id}
                  initial={{ opacity: 0, x: -8 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.3, delay: index * 0.03 }}
                  className="flex items-center gap-4 px-6 py-4 transition-colors hover:bg-muted/40"
                >
                  <div className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-muted">
                    <Icon className="size-4 text-muted-foreground" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-medium">{doc.name}</p>
                    <div className="mt-0.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
                      <span>{doc.type}</span>
                      <span>{formatBytes(doc.size)}</span>
                      <span>{doc.chunks} 个分块</span>
                      <span className="flex items-center gap-1">
                        <Clock className="size-3" />
                        {formatRelativeTime(doc.uploadedAt)}
                      </span>
                    </div>
                  </div>
                  <Badge variant={status.variant}>{status.label}</Badge>
                  {/* 查看文档内容 */}
                  <Button
                    variant="ghost"
                    size="icon"
                    className="shrink-0 text-muted-foreground hover:text-foreground"
                    title={canView ? "查看文档内容" : "文档处理完成后可查看"}
                    aria-label={`查看 ${doc.name}`}
                    disabled={!canView}
                    onClick={() => setViewingDoc(doc)}
                  >
                    <Eye className="size-4" />
                  </Button>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="shrink-0 text-muted-foreground hover:text-destructive"
                    title="删除文档"
                    disabled={deletingId === doc.id}
                    onClick={() => handleDelete(doc.id, doc.name)}
                  >
                    {deletingId === doc.id ? (
                      <Loader2 className="size-4 animate-spin" />
                    ) : (
                      <Trash2 className="size-4" />
                    )}
                  </Button>
                </motion.li>
              );
            })}
          </ul>
        )}
      </CardContent>

      {/* 文档内容查看弹窗 */}
      <Dialog
        open={!!viewingDoc}
        onOpenChange={(open) => {
          if (!open) setViewingDoc(null);
        }}
      >
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle className="truncate pr-6">
              {viewingDoc?.name}
            </DialogTitle>
            <DialogDescription>
              索引后的文档内容（按分块展示）
            </DialogDescription>
          </DialogHeader>
          {viewingDoc && (
            <DocumentViewer key={viewingDoc.id} doc={viewingDoc} />
          )}
        </DialogContent>
      </Dialog>
    </Card>
  );
}
