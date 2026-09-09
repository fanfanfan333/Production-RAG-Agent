"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
import {
  MessageSquarePlus,
  Search,
  Trash2,
  X,
} from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { useConversations } from "@/lib/hooks/use-conversations";
import { useApp } from "@/lib/context/app-context";
import { cn } from "@/lib/utils";

function formatRelativeTime(iso?: string) {
  if (!iso) return "";
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "";
  const diff = Date.now() - then;
  const minutes = Math.floor(diff / 60000);
  if (minutes < 1) return "刚刚";
  if (minutes < 60) return `${minutes} 分钟前`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} 小时前`;
  const days = Math.floor(hours / 24);
  if (days < 30) return `${days} 天前`;
  return new Date(iso).toLocaleDateString();
}

/**
 * 历史对话侧栏 —— 高级灰版本.
 *
 * 相比主界面（浅色底）采用更深一级的灰阶面板（zinc-200/300 渐变），
 * 同色系分层营造高级感：无彩色图标、无表情，只靠灰度对比与留白。
 */
export function ConversationSidebar() {
  const { activeConversationId, setActiveConversationId, conversationVersion } =
    useApp();
  const { conversations, loading, error, refetch, remove } =
    useConversations();

  const [search, setSearch] = useState("");
  // 待确认删除的对话 id（两步确认，避免误删）
  const [confirmingId, setConfirmingId] = useState<string | null>(null);
  const [clearingAll, setClearingAll] = useState(false);
  const confirmTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Refresh the list whenever a turn is saved anywhere in the app
  // (new conversation created or existing one updated).
  const lastVersionRef = useRef(conversationVersion);
  useEffect(() => {
    if (lastVersionRef.current !== conversationVersion) {
      lastVersionRef.current = conversationVersion;
      refetch();
    }
  }, [conversationVersion, refetch]);

  // 确认按钮 3 秒后自动收回
  useEffect(() => {
    if (!confirmingId) return;
    confirmTimerRef.current = setTimeout(() => setConfirmingId(null), 3000);
    return () => {
      if (confirmTimerRef.current) clearTimeout(confirmTimerRef.current);
    };
  }, [confirmingId]);

  const filtered = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return conversations;
    return conversations.filter(
      (c) => c.title.toLowerCase().includes(q)
    );
  }, [conversations, search]);

  const handleSelect = (id: string) => {
    if (id === activeConversationId) return;
    setConfirmingId(null);
    setActiveConversationId(id);
  };

  const handleDelete = async (id: string) => {
    if (confirmingId !== id) {
      setConfirmingId(id);
      return;
    }
    setConfirmingId(null);
    try {
      await remove(id);
      if (id === activeConversationId) setActiveConversationId(null);
      toast.success("已删除该对话");
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "删除失败");
    }
  };

  const handleClearAll = async () => {
    if (!clearingAll) {
      setClearingAll(true);
      setTimeout(() => setClearingAll(false), 3000);
      return;
    }
    setClearingAll(false);
    const targets = conversations.map((c) => c.id);
    let failed = 0;
    for (const id of targets) {
      try {
        await remove(id);
      } catch {
        failed += 1;
      }
    }
    if (failed > 0) {
      toast.error(`${failed} 个对话删除失败，请重试`);
    } else {
      toast.success(`已清空 ${targets.length} 个对话`);
    }
    setActiveConversationId(null);
  };

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden rounded-2xl border border-zinc-300/70 bg-gradient-to-b from-zinc-200/90 via-zinc-200/70 to-zinc-300/60 text-zinc-800 shadow-lg shadow-zinc-400/20">
      {/* ── 头部 ── */}
      <div className="flex flex-row items-center justify-between border-b border-zinc-300/60 px-4 pb-3 pt-4">
        <div>
          <p className="text-sm font-semibold leading-tight text-zinc-800">
            历史对话
          </p>
          <p className="mt-0.5 text-[10px] leading-tight text-zinc-500">
            {conversations.length > 0 ? `${conversations.length} 个会话` : "暂无会话"}
          </p>
        </div>
        <div className="flex items-center gap-1">
          <Button
            variant="ghost"
            size="icon"
            className="size-7 text-zinc-500 hover:bg-zinc-400/30 hover:text-zinc-800"
            title="开始新对话"
            onClick={() => setActiveConversationId(null)}
          >
            <MessageSquarePlus className="size-4" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className={cn(
              "size-7 text-zinc-500 hover:bg-zinc-400/30",
              clearingAll ? "text-rose-600 hover:text-rose-700" : "hover:text-zinc-800"
            )}
            title={clearingAll ? "再次点击确认清空全部" : "清空全部对话"}
            disabled={conversations.length === 0}
            onClick={handleClearAll}
          >
            <Trash2 className="size-4" />
          </Button>
        </div>
      </div>

      {/* ── 搜索 ── */}
      {conversations.length > 3 && (
        <div className="px-3 pt-3">
          <div className="relative">
            <Search className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-zinc-400" />
            <input
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="搜索对话…"
              className="h-8 w-full rounded-lg border border-zinc-300/80 bg-white/70 pl-8 pr-7 text-xs text-zinc-800 placeholder:text-zinc-400 focus:border-zinc-400 focus:outline-none focus:ring-1 focus:ring-zinc-400/40"
            />
            {search && (
              <button
                type="button"
                aria-label="清除搜索"
                className="absolute right-2 top-1/2 -translate-y-1/2 text-zinc-400 hover:text-zinc-600"
                onClick={() => setSearch("")}
              >
                <X className="size-3.5" />
              </button>
            )}
          </div>
        </div>
      )}

      {/* ── 列表 ── */}
      <div className="min-h-0 flex-1 overflow-y-auto p-2">
        {loading ? (
          <div className="space-y-2 p-1">
            {[0, 1, 2].map((i) => (
              <div
                key={i}
                className="h-14 animate-pulse rounded-xl bg-zinc-300/50"
              />
            ))}
          </div>
        ) : error ? (
          <p className="p-3 text-xs text-rose-600">{error}</p>
        ) : filtered.length === 0 ? (
          <div className="flex flex-col items-center gap-2 px-4 py-10 text-center">
            {search ? (
              <p className="text-xs text-zinc-500">
                没有匹配「{search}」的对话
              </p>
            ) : (
              <p className="text-xs leading-relaxed text-zinc-500">
                还没有历史对话
                <br />
                发送第一条消息后就会出现在这里
              </p>
            )}
          </div>
        ) : (
          <ul className="space-y-1">
            {filtered.map((conv, index) => {
              const active = conv.id === activeConversationId;
              const confirming = confirmingId === conv.id;
              return (
                <motion.li
                  key={conv.id}
                  initial={{ opacity: 0, x: 8 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.2, delay: Math.min(index, 8) * 0.03 }}
                >
                  <div
                    role="button"
                    tabIndex={0}
                    onClick={() => handleSelect(conv.id)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        handleSelect(conv.id);
                      }
                    }}
                    className={cn(
                      "group flex w-full cursor-pointer items-start gap-2.5 rounded-xl border px-3 py-2.5 text-left transition-all",
                      active
                        ? "border-zinc-400/70 bg-white shadow-sm shadow-zinc-400/20"
                        : "border-transparent hover:border-zinc-300/80 hover:bg-white/50"
                    )}
                  >
                    <span
                      className={cn(
                        "mt-1.5 size-1.5 shrink-0 rounded-full transition-colors",
                        active ? "bg-zinc-700" : "bg-zinc-400"
                      )}
                    />
                    <span className="min-w-0 flex-1">
                      <span
                        className={cn(
                          "block truncate text-sm",
                          active
                            ? "font-medium text-zinc-900"
                            : "text-zinc-700 group-hover:text-zinc-900"
                        )}
                      >
                        {conv.title}
                      </span>
                      <span className="mt-0.5 block text-[11px] text-zinc-500">
                        {conv.messageCount} 条消息 ·{" "}
                        {formatRelativeTime(conv.updatedAt ?? conv.createdAt)}
                      </span>
                    </span>
                    <button
                      type="button"
                      aria-label="删除对话"
                      className={cn(
                        "mt-0.5 shrink-0 rounded-md p-1 transition-all",
                        confirming
                          ? "bg-rose-100 text-rose-600"
                          : "text-zinc-400 opacity-0 hover:bg-zinc-300/60 hover:text-rose-600 group-hover:opacity-100"
                      )}
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDelete(conv.id);
                      }}
                    >
                      {confirming ? (
                        <span className="text-[10px] font-medium">确认?</span>
                      ) : (
                        <Trash2 className="size-3.5" />
                      )}
                    </button>
                  </div>
                </motion.li>
              );
            })}
          </ul>
        )}
      </div>

      {/* ── 底部标识（纯文字，无图标） ── */}
      <div className="border-t border-zinc-300/60 px-4 py-2.5">
        <p className="text-[10px] text-zinc-500">
          粗排召回 + Cross-Encoder 精排 · 引用溯源
        </p>
      </div>
    </div>
  );
}
