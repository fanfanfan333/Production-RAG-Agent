"use client";

import type { ReactNode } from "react";
import Link from "next/link";
import { motion } from "framer-motion";
import {
  AlertCircle,
  ArrowUpRight,
  FileText,
  HardDrive,
  Layers,
  MessageSquare,
  TrendingUp,
} from "lucide-react";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Skeleton } from "@/components/ui/skeleton";
import type {
  ConversationSummary,
  DashboardStats,
  Document,
} from "@/lib/types";
import { formatBytes } from "@/lib/utils";

const WEEK_MS = 7 * 24 * 60 * 60 * 1000;

interface StatsCardsProps {
  stats: DashboardStats;
  documents: Document[];
  conversations: ConversationSummary[];
  loading?: boolean;
  error?: string | null;
}

interface StatCardProps {
  label: string;
  value: string;
  sub?: ReactNode;
  icon: typeof FileText;
  href: string;
  loading?: boolean;
  delay?: number;
}

/** 可点击下钻的统计卡片：hover 抬升 + 右上角箭头提示可跳转。 */
function StatCard({
  label,
  value,
  sub,
  icon: Icon,
  href,
  loading,
  delay = 0,
}: StatCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay }}
    >
      <Link href={href} className="group block h-full outline-none" aria-label={label}>
        <Card className="h-full transition-all duration-200 group-hover:-translate-y-0.5 group-hover:border-foreground/25 group-hover:shadow-sm group-focus-visible:border-primary/50">
          <CardHeader className="flex flex-row items-center justify-between pb-2">
            <CardTitle className="text-[11px] font-medium uppercase tracking-[0.08em] text-muted-foreground">
              {label}
            </CardTitle>
            <span className="flex items-center gap-1">
              <Icon className="size-4 text-muted-foreground/70" />
              <ArrowUpRight className="size-3.5 text-muted-foreground/0 transition-all group-hover:translate-x-0.5 group-hover:-translate-y-0.5 group-hover:text-muted-foreground" />
            </span>
          </CardHeader>
          <CardContent>
            {loading ? (
              <Skeleton className="h-9 w-24" />
            ) : (
              <>
                <p className="text-3xl font-semibold tracking-tight tabular-nums">
                  {value}
                </p>
                {sub && (
                  <p className="mt-1 flex items-center gap-1 text-xs text-muted-foreground">
                    {sub}
                  </p>
                )}
              </>
            )}
          </CardContent>
        </Card>
      </Link>
    </motion.div>
  );
}

export function StatsCards({
  stats,
  documents,
  conversations,
  loading,
  error,
}: StatsCardsProps) {
  const storagePercent = stats.storageLimit
    ? Math.min(
        100,
        Math.round((stats.storageUsed / stats.storageLimit) * 100)
      )
    : 0;

  const weeklyDocs = documents.filter(
    (d) => Date.now() - d.uploadedAt.getTime() < WEEK_MS
  ).length;

  const weeklyConvs = conversations.filter((c) => {
    const t = c.updatedAt ?? c.createdAt;
    if (!t) return false;
    const time = new Date(t).getTime();
    return !Number.isNaN(time) && Date.now() - time < WEEK_MS;
  }).length;

  const avgChunks =
    stats.totalDocuments > 0
      ? Math.round(stats.totalChunks / stats.totalDocuments)
      : 0;

  if (error) {
    return (
      <div className="flex items-center gap-3 rounded-xl border border-destructive/20 bg-destructive/5 px-4 py-3 text-sm text-destructive">
        <AlertCircle className="size-4 shrink-0" />
        <span>{error}</span>
      </div>
    );
  }

  return (
    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
      <StatCard
        label="文档总数"
        value={stats.totalDocuments.toLocaleString()}
        sub={
          loading ? undefined : weeklyDocs > 0 ? (
            <>
              <TrendingUp className="size-3 text-emerald-600" />
              本周新增 {weeklyDocs} 份
            </>
          ) : (
            "近 7 天无新增"
          )
        }
        icon={FileText}
        href="/documents"
        loading={loading}
        delay={0.05}
      />
      <StatCard
        label="对话总数"
        value={conversations.length.toLocaleString()}
        sub={
          loading ? undefined : weeklyConvs > 0 ? (
            <>
              <TrendingUp className="size-3 text-emerald-600" />
              近 7 天 {weeklyConvs} 次
            </>
          ) : (
            "近 7 天无新对话"
          )
        }
        icon={MessageSquare}
        href="/chat"
        loading={loading}
        delay={0.1}
      />
      <StatCard
        label="分块总数"
        value={stats.totalChunks.toLocaleString()}
        sub={loading ? undefined : `平均每份文档 ${avgChunks} 块`}
        icon={Layers}
        href="/documents"
        loading={loading}
        delay={0.15}
      />

      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.2 }}
      >
        <Link
          href="/documents"
          className="group block h-full outline-none"
          aria-label="存储用量"
        >
          <Card className="h-full transition-all duration-200 group-hover:-translate-y-0.5 group-hover:border-foreground/25 group-hover:shadow-sm group-focus-visible:border-primary/50">
            <CardHeader className="flex flex-row items-center justify-between pb-2">
              <CardTitle className="text-[11px] font-medium uppercase tracking-[0.08em] text-muted-foreground">
                存储用量
              </CardTitle>
              <HardDrive className="size-4 text-muted-foreground/70" />
            </CardHeader>
            <CardContent className="space-y-3">
              {loading ? (
                <>
                  <Skeleton className="h-9 w-16" />
                  <Skeleton className="h-2.5 w-full" />
                </>
              ) : (
                <>
                  <div className="flex items-baseline justify-between">
                    <p className="text-3xl font-semibold tracking-tight tabular-nums">
                      {storagePercent}%
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {formatBytes(stats.storageUsed)} /{" "}
                      {formatBytes(stats.storageLimit)}
                    </p>
                  </div>
                  <Progress value={storagePercent} className="h-1.5" />
                </>
              )}
            </CardContent>
          </Card>
        </Link>
      </motion.div>
    </div>
  );
}
