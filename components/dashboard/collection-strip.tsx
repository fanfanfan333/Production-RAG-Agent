"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import {
  CheckCircle2,
  Folder,
  Layers,
  Loader2,
  Plus,
  Trash2,
} from "lucide-react";
import { toast } from "sonner";
import { createCollection, deleteCollection } from "@/lib/api/collections";
import { useApp } from "@/lib/context/app-context";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { cn } from "@/lib/utils";
import type { Collection } from "@/lib/types";

interface CollectionStripProps {
  collections: Collection[];
  loading?: boolean;
  error?: string | null;
  /** Total documents across all collections (from document stats). */
  totalDocuments: number;
}

/**
 * 知识库分组切换条（主界面互动优化）：
 * 把「当前检索范围」从设置页搬到主界面，一键切换 / 新建知识库，
 * 切换后上传、提问、文档列表全部联动。
 */
export function CollectionStrip({
  collections,
  loading,
  error,
  totalDocuments,
}: CollectionStripProps) {
  const { activeCollectionId, setActiveCollectionId, refresh } = useApp();
  const [createOpen, setCreateOpen] = useState(false);
  const [newName, setNewName] = useState("");
  const [creating, setCreating] = useState(false);
  const [pendingDelete, setPendingDelete] = useState<{
    id: string;
    name: string;
  } | null>(null);
  const [deleting, setDeleting] = useState(false);

  // 知识库服务不可用时静默隐藏该区块，不打断工作台其余功能
  if (error) return null;

  const cards: { id: string | null; name: string; count: number }[] = [
    { id: null, name: "全部文档", count: totalDocuments },
    ...collections.map((c) => ({
      id: c.id,
      name: c.name,
      count: c.documentCount,
    })),
  ];

  const handleCreate = async () => {
    const name = newName.trim();
    if (!name) return;
    setCreating(true);
    try {
      const created = await createCollection(name);
      toast.success(`知识库「${created.name}」已创建`);
      setCreateOpen(false);
      setNewName("");
      // 创建后直接切换到新知识库，引导用户开始上传
      setActiveCollectionId(created.id);
      refresh();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "创建知识库失败");
    } finally {
      setCreating(false);
    }
  };

  const confirmDelete = async () => {
    if (!pendingDelete) return;
    setDeleting(true);
    try {
      await deleteCollection(pendingDelete.id);
      if (activeCollectionId === pendingDelete.id) {
        setActiveCollectionId(null);
      }
      toast.success(`知识库「${pendingDelete.name}」已删除`);
      setPendingDelete(null);
      refresh();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "删除知识库失败");
    } finally {
      setDeleting(false);
    }
  };

  return (
    <motion.section
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.1 }}
      aria-label="知识库分组"
    >
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-sm font-medium text-muted-foreground">
          知识库分组
          <span className="ml-2 text-xs text-muted-foreground/70">
            点击切换检索与上传范围
          </span>
        </h2>
      </div>

      <div className="flex gap-3 overflow-x-auto pb-1">
        {loading
          ? Array.from({ length: 3 }).map((_, i) => (
              <Skeleton
                key={i}
                className="h-[88px] w-[160px] shrink-0 rounded-xl"
              />
            ))
          : cards.map((card, index) => {
              const active = card.id === activeCollectionId;
              const Icon = card.id ? Folder : Layers;
              return (
                <motion.div
                  key={card.id ?? "all"}
                  role="button"
                  tabIndex={0}
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ duration: 0.3, delay: 0.1 + index * 0.04 }}
                  onClick={() => setActiveCollectionId(card.id)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      setActiveCollectionId(card.id);
                    }
                  }}
                  aria-pressed={active}
                  className={cn(
                    "group relative flex min-w-[150px] flex-1 cursor-pointer flex-col gap-2 rounded-xl border p-4 text-left transition-all",
                    active
                      ? "border-primary/50 bg-primary/5 shadow-xs"
                      : "border-border bg-card hover:border-foreground/25 hover:bg-muted/40"
                  )}
                >
                  <span className="flex items-center justify-between">
                    <Icon
                      className={cn(
                        "size-4",
                        active ? "text-primary" : "text-muted-foreground"
                      )}
                    />
                    <span className="flex items-center gap-1">
                      {active && (
                        <CheckCircle2 className="size-4 text-primary" />
                      )}
                      {/* 删除按钮：仅用户自建知识库可删（「全部文档」不可删） */}
                      {card.id && (
                        <span
                          role="button"
                          tabIndex={0}
                          aria-label={`删除知识库 ${card.name}`}
                          title="删除知识库"
                          className="rounded p-0.5 text-muted-foreground opacity-0 transition-opacity hover:text-destructive group-hover:opacity-100"
                          onClick={(e) => {
                            e.stopPropagation();
                            setPendingDelete({ id: card.id as string, name: card.name });
                          }}
                          onKeyDown={(e) => {
                            if (e.key === "Enter" || e.key === " ") {
                              e.stopPropagation();
                              setPendingDelete({ id: card.id as string, name: card.name });
                            }
                          }}
                        >
                          <Trash2 className="size-3.5" />
                        </span>
                      )}
                    </span>
                  </span>
                  <span className="min-w-0">
                    <span className="block truncate text-sm font-medium">
                      {card.name}
                    </span>
                    <span className="mt-0.5 block text-xs text-muted-foreground">
                      {card.count} 份文档
                    </span>
                  </span>
                </motion.div>
              );
            })}
        {!loading && (
          <button
            type="button"
            onClick={() => setCreateOpen(true)}
            className="flex min-w-[150px] flex-col items-center justify-center gap-1.5 rounded-xl border border-dashed border-border p-4 text-muted-foreground transition-colors hover:border-foreground/30 hover:text-foreground"
          >
            <Plus className="size-4" />
            <span className="text-xs">新建知识库</span>
          </button>
        )}
      </div>

      <Dialog open={createOpen} onOpenChange={setCreateOpen}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>新建知识库</DialogTitle>
            <DialogDescription>
              创建独立的文档分组，之后可按知识库上传文档与提问
            </DialogDescription>
          </DialogHeader>
          <Input
            autoFocus
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            placeholder="知识库名称，例如：产品部资料"
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.preventDefault();
                handleCreate();
              }
            }}
          />
          <div className="flex justify-end gap-2">
            <Button
              variant="outline"
              onClick={() => setCreateOpen(false)}
              disabled={creating}
            >
              取消
            </Button>
            <Button
              onClick={handleCreate}
              disabled={!newName.trim() || creating}
            >
              {creating && <Loader2 className="size-4 animate-spin" />}
              创建
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      {/* 删除知识库确认 */}
      <Dialog
        open={!!pendingDelete}
        onOpenChange={(open) => {
          if (!open) setPendingDelete(null);
        }}
      >
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>删除知识库</DialogTitle>
            <DialogDescription>
              确定要删除知识库「{pendingDelete?.name}」吗？库内文档不会被删除，
              只会解除与该知识库的关联，操作不可恢复。
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
    </motion.section>
  );
}
