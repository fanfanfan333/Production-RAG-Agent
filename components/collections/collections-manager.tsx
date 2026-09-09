"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import {
  AlertCircle,
  Check,
  FolderOpen,
  Loader2,
  Plus,
  Trash2,
} from "lucide-react";
import { toast } from "sonner";
import { assignDocumentToCollection } from "@/lib/api/documents";
import {
  createCollection,
  deleteCollection,
} from "@/lib/api/collections";
import { useApp } from "@/lib/context/app-context";
import { useCollections } from "@/lib/hooks/use-collections";
import { useDocuments } from "@/lib/hooks/use-documents";
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
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import { cn, formatRelativeTime } from "@/lib/utils";

export function CollectionsManager() {
  const { activeCollectionId, setActiveCollectionId, refresh } = useApp();
  const {
    collections,
    loading: collectionsLoading,
    error: collectionsError,
    refetch: refetchCollections,
  } = useCollections();
  const { documents, loading: documentsLoading, refetch: refetchDocuments } =
    useDocuments();
  const [newName, setNewName] = useState("");
  const [creating, setCreating] = useState(false);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [assigningId, setAssigningId] = useState<string | null>(null);

  const handleCreate = async () => {
    if (!newName.trim()) return;
    setCreating(true);
    try {
      const collection = await createCollection(newName.trim());
      toast.success(`集合「${collection.name}」已创建`);
      setNewName("");
      setDialogOpen(false);
      refresh();
      refetchCollections();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "创建失败");
    } finally {
      setCreating(false);
    }
  };

  const handleDeleteCollection = async (id: string, name: string) => {
    setDeletingId(id);
    try {
      await deleteCollection(id);
      if (activeCollectionId === id) setActiveCollectionId(null);
      toast.success(`集合「${name}」已删除`);
      refresh();
      refetchCollections();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "删除失败");
    } finally {
      setDeletingId(null);
    }
  };

  const handleAssign = async (documentId: string, collectionId: string | null) => {
    setAssigningId(documentId);
    try {
      await assignDocumentToCollection(documentId, collectionId);
      toast.success("文档已分配");
      refresh();
      refetchDocuments();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "分配失败");
    } finally {
      setAssigningId(null);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="text-sm text-muted-foreground">
          将文档组织到集合中，并可切换当前活跃的集合
        </p>
        <Dialog open={dialogOpen} onOpenChange={setDialogOpen}>
          <DialogTrigger asChild>
            <Button size="sm">
              <Plus className="size-4" />
              新建集合
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>创建集合</DialogTitle>
              <DialogDescription>
                将相关文档分组，便于定向问答
              </DialogDescription>
            </DialogHeader>
            <div className="flex gap-2">
              <Input
                placeholder="集合名称"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleCreate()}
              />
              <Button disabled={creating || !newName.trim()} onClick={handleCreate}>
                {creating ? (
                  <Loader2 className="size-4 animate-spin" />
                ) : (
                  "创建"
                )}
              </Button>
            </div>
          </DialogContent>
        </Dialog>
      </div>

      {collectionsError && (
        <div className="flex items-center gap-3 rounded-xl border border-destructive/20 bg-destructive/5 px-4 py-3 text-sm text-destructive">
          <AlertCircle className="size-4 shrink-0" />
          <span>{collectionsError}</span>
        </div>
      )}

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {collectionsLoading ? (
          Array.from({ length: 3 }).map((_, i) => (
            <Skeleton key={i} className="h-36 rounded-xl" />
          ))
        ) : collections.length === 0 ? (
          <Card className="sm:col-span-2 lg:col-span-3">
            <CardContent className="py-12 text-center text-sm text-muted-foreground">
              暂无集合。创建一个来组织你的文档。
            </CardContent>
          </Card>
        ) : (
          collections.map((collection, index) => {
            const isActive = activeCollectionId === collection.id;
            return (
              <motion.div
                key={collection.id}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.05 }}
              >
                <Card
                  className={cn(
                    "relative overflow-hidden transition-colors",
                    isActive && "border-primary/50 ring-1 ring-primary/20"
                  )}
                >
                  <CardHeader>
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <div className="flex size-9 items-center justify-center rounded-md bg-muted text-muted-foreground">
                          <FolderOpen className="size-4" />
                        </div>
                        <div>
                          <CardTitle className="text-base">
                            {collection.name}
                          </CardTitle>
                          <CardDescription>
                            {collection.documentCount} 篇文档 ·{" "}
                            {formatRelativeTime(collection.createdAt)}
                          </CardDescription>
                        </div>
                      </div>
                      {isActive && (
                        <Badge variant="success">
                          <Check className="size-3" />
                          活跃
                        </Badge>
                      )}
                    </div>
                  </CardHeader>
                  <CardContent className="flex gap-2">
                    <Button
                      size="sm"
                      variant={isActive ? "secondary" : "default"}
                      className="flex-1"
                      onClick={() =>
                        setActiveCollectionId(isActive ? null : collection.id)
                      }
                    >
                      {isActive ? "取消活跃" : "设为活跃"}
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      className="text-muted-foreground hover:text-destructive"
                      disabled={deletingId === collection.id}
                      onClick={() =>
                        handleDeleteCollection(collection.id, collection.name)
                      }
                    >
                      {deletingId === collection.id ? (
                        <Loader2 className="size-4 animate-spin" />
                      ) : (
                        <Trash2 className="size-4" />
                      )}
                    </Button>
                  </CardContent>
                </Card>
              </motion.div>
            );
          })
        )}
      </div>

      <Card>
        <CardHeader>
          <CardTitle>分配文档</CardTitle>
          <CardDescription>
            将已上传的文档关联到集合
          </CardDescription>
        </CardHeader>
        <CardContent className="px-0">
          {documentsLoading ? (
            <div className="space-y-3 px-6">
              {Array.from({ length: 4 }).map((_, i) => (
                <Skeleton key={i} className="h-12 w-full" />
              ))}
            </div>
          ) : documents.length === 0 ? (
            <p className="px-6 py-8 text-center text-sm text-muted-foreground">
              请先上传文档，再将其分配到集合。
            </p>
          ) : (
            <ul className="divide-y divide-border/60">
              {documents.map((doc) => (
                <li
                  key={doc.id}
                  className="flex flex-col gap-3 px-6 py-4 sm:flex-row sm:items-center sm:justify-between"
                >
                  <div className="min-w-0">
                    <p className="truncate text-sm font-medium">{doc.name}</p>
                    <p className="text-xs text-muted-foreground">
                      {doc.collectionId
                        ? `属于集合 ${
                            collections.find((c) => c.id === doc.collectionId)
                              ?.name ?? "未知集合"
                          }`
                        : "未分配"}
                    </p>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <Button
                      size="sm"
                      variant="outline"
                      disabled={assigningId === doc.id}
                      onClick={() => handleAssign(doc.id, null)}
                    >
                      未分配
                    </Button>
                    {collections.map((collection) => (
                      <Button
                        key={collection.id}
                        size="sm"
                        variant={
                          doc.collectionId === collection.id
                            ? "default"
                            : "outline"
                        }
                        disabled={assigningId === doc.id}
                        onClick={() => handleAssign(doc.id, collection.id)}
                      >
                        {collection.name}
                      </Button>
                    ))}
                  </div>
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
