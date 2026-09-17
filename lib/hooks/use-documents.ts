"use client";

import { useCallback, useEffect, useState } from "react";
import { getDocuments } from "@/lib/api/documents";
import { useApp } from "@/lib/context/app-context";
import type { AccessLevel, DashboardStats, Document } from "@/lib/types";

export function useDocuments(options?: {
  pollProcessing?: boolean;
  /** 三层知识库筛选：undefined/"all" = 全部；否则只看某一层。 */
  accessLevel?: AccessLevel | "all";
}) {
  const { refreshKey, activeCollectionId } = useApp();
  const accessLevel = options?.accessLevel ?? "all";
  const [documents, setDocuments] = useState<Document[]>([]);
  const [stats, setStats] = useState<DashboardStats>({
    totalDocuments: 0,
    totalChunks: 0,
    storageUsed: 0,
    storageLimit: 1024 * 1024 * 1024,
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refetch = useCallback(async () => {
    setError(null);
    try {
      const data = await getDocuments(activeCollectionId, accessLevel);
      setDocuments(data.documents);
      setStats(data.stats);
    } catch (err) {
      setError(err instanceof Error ? err.message : "加载文档失败");
    } finally {
      setLoading(false);
    }
  }, [activeCollectionId, accessLevel]);

  useEffect(() => {
    setLoading(true);
    refetch();
  }, [refetch, refreshKey]);

  useEffect(() => {
    if (!options?.pollProcessing) return;
    const hasProcessing = documents.some((doc) => doc.status === "processing");
    if (!hasProcessing) return;

    const interval = setInterval(refetch, 3000);
    return () => clearInterval(interval);
  }, [documents, options?.pollProcessing, refetch]);

  return { documents, stats, loading, error, refetch };
}
