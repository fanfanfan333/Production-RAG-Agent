"use client";

import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import { ArrowRight, History, MessageSquare } from "lucide-react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { useApp } from "@/lib/context/app-context";
import type { ConversationSummary } from "@/lib/types";

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

interface RecentConversationsProps {
  conversations: ConversationSummary[];
  loading?: boolean;
  error?: string | null;
}

export function RecentConversations({
  conversations,
  loading,
  error,
}: RecentConversationsProps) {
  const router = useRouter();
  const { setActiveConversationId } = useApp();

  const open = (id: string) => {
    setActiveConversationId(id);
    router.push("/chat");
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.25 }}
      className="h-full"
    >
      <Card className="h-full">
        <CardHeader className="flex flex-row items-center justify-between pb-2">
          <div>
            <CardTitle>历史对话</CardTitle>
            <CardDescription>
              点击任意记录可继续该对话提问
            </CardDescription>
          </div>
          <Button variant="ghost" size="sm" asChild>
            <Link href="/chat">
              查看全部
              <ArrowRight className="size-3.5" />
            </Link>
          </Button>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="space-y-2">
              {[0, 1, 2].map((i) => (
                <div key={i} className="h-12 animate-pulse rounded-lg bg-muted" />
              ))}
            </div>
          ) : error ? (
            <p className="text-sm text-destructive">{error}</p>
          ) : conversations.length === 0 ? (
            <div className="flex flex-col items-center gap-2 py-6 text-center">
              <History className="size-6 text-muted-foreground" />
              <p className="text-sm text-muted-foreground">
                还没有对话记录，在上方提问框提出第一个问题吧。
              </p>
            </div>
          ) : (
            <ul className="divide-y divide-border/60">
              {conversations.slice(0, 6).map((conv) => (
                <li key={conv.id}>
                  <button
                    type="button"
                    onClick={() => open(conv.id)}
                    className="flex w-full items-center gap-3 py-3 text-left transition-colors hover:bg-accent/50"
                  >
                    <span className="flex size-9 shrink-0 items-center justify-center rounded-md bg-muted text-muted-foreground">
                      <MessageSquare className="size-4" />
                    </span>
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-sm font-medium">
                        {conv.title}
                      </span>
                      <span className="mt-0.5 block text-xs text-muted-foreground">
                        {conv.messageCount} 条消息 ·{" "}
                        {formatRelativeTime(conv.updatedAt ?? conv.createdAt)}
                      </span>
                    </span>
                    <ArrowRight className="size-4 shrink-0 text-muted-foreground" />
                  </button>
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>
    </motion.div>
  );
}
