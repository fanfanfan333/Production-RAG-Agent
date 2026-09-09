"use client";

import { AskHero } from "@/components/dashboard/ask-hero";
import { CollectionStrip } from "@/components/dashboard/collection-strip";
import { HealthStatus } from "@/components/dashboard/health-status";
import { QuickActions } from "@/components/dashboard/quick-actions";
import { RecentConversations } from "@/components/dashboard/recent-conversations";
import { RecentDocuments } from "@/components/dashboard/recent-documents";
import { StatsCards } from "@/components/dashboard/stats-cards";
import { UploadCard } from "@/components/dashboard/upload-card";
import { useCollections } from "@/lib/hooks/use-collections";
import { useConversations } from "@/lib/hooks/use-conversations";
import { useDocuments } from "@/lib/hooks/use-documents";

/**
 * 仪表盘工作台（主界面互动优化）：
 * 提问入口置顶 → 可点击统计 → 知识库切换 → 健康状态 →
 * 上传 + 快捷操作 → 最近文档（筛选/提问/删除）+ 历史对话。
 *
 * conversations / collections 数据在此提升，避免子组件重复请求。
 */
export function DashboardContent() {
  const { documents, stats, loading, error, refetch } = useDocuments({
    pollProcessing: true,
  });
  const {
    collections,
    loading: collectionsLoading,
    error: collectionsError,
  } = useCollections();
  const {
    conversations,
    loading: conversationsLoading,
    error: conversationsError,
  } = useConversations();

  return (
    <div className="space-y-6">
      {/* 快捷提问：无需跳转即可发起检索问答 */}
      <AskHero collections={collections} />

      {/* 可点击下钻的统计卡片 */}
      <StatsCards
        stats={stats}
        documents={documents}
        conversations={conversations}
        loading={loading}
        error={error}
      />

      {/* 知识库分组切换（错误时静默隐藏） */}
      <CollectionStrip
        collections={collections}
        loading={collectionsLoading}
        error={collectionsError}
        totalDocuments={stats.totalDocuments}
      />

      <HealthStatus />

      <div className="grid gap-6 lg:grid-cols-5">
        <div className="lg:col-span-3">
          <UploadCard disabled={false} onUploadComplete={refetch} />
        </div>
        <div className="lg:col-span-2">
          <QuickActions />
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-5">
        <div className="lg:col-span-3">
          <RecentDocuments
            documents={documents}
            loading={loading}
            error={error}
          />
        </div>
        <div className="lg:col-span-2">
          <RecentConversations
            conversations={conversations}
            loading={conversationsLoading}
            error={conversationsError}
          />
        </div>
      </div>
    </div>
  );
}
