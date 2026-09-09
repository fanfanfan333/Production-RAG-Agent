"use client";

import { useCallback, useEffect, useState } from "react";
import {
  deleteConversation,
  listConversations,
} from "@/lib/api/conversations";
import type { ConversationSummary } from "@/lib/types";

export function useConversations(pollIntervalMs?: number) {
  const [conversations, setConversations] = useState<ConversationSummary[]>(
    []
  );
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const refetch = useCallback(async () => {
    setError(null);
    try {
      const data = await listConversations();
      setConversations(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "加载历史对话失败");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refetch();
    if (!pollIntervalMs) return;
    const interval = setInterval(refetch, pollIntervalMs);
    return () => clearInterval(interval);
  }, [pollIntervalMs, refetch]);

  const remove = useCallback(
    async (id: string) => {
      await deleteConversation(id);
      setConversations((prev) => prev.filter((c) => c.id !== id));
    },
    []
  );

  return { conversations, loading, error, refetch, remove };
}
