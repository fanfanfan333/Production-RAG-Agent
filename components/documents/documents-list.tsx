"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import { motion } from "framer-motion";
import {
  AlertCircle,
  Building2,
  Clock,
  Eye,
  FileSpreadsheet,
  FileText,
  FileType,
  FileX2,
  FolderInput,
  Image as ImageIcon,
  Loader2,
  Lock,
  Share2,
  Sparkles,
  Trash2,
  Users,
} from "lucide-react";
import { toast } from "sonner";
import { deleteDocument, getDocumentChunks } from "@/lib/api/documents";
import {
  fetchAccessibleCompanies,
  type AccessibleCompany,
} from "@/lib/api/companies";
import { useApp } from "@/lib/context/app-context";
import { useAuth } from "@/lib/context/auth-context";
import { useDocuments } from "@/lib/hooks/use-documents";
import type {
  AccessLevel,
  Document,
  DocumentChunksResponse,
  DocumentStatus,
} from "@/lib/types";
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
import { Select, SelectItem } from "@/components/ui/select";
import { cn, describeIngestStage, formatBytes, formatRelativeTime } from "@/lib/utils";
import { AccessTierBadge, tierScopeName } from "@/components/documents/access-badge";
import {
  DocumentSharingDialog,
  type SharingDialogView,
} from "@/components/documents/document-sharing-dialog";

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

/** 三层知识库的视图切换：全部 / 个人 / 部门 / 公司。 */
const TIER_TABS: {
  key: AccessLevel | "all";
  label: string;
  icon: typeof Lock;
}[] = [
  { key: "all", label: "全部", icon: FileText },
  { key: "private", label: "个人知识库", icon: Lock },
  { key: "department", label: "部门知识库", icon: Users },
  { key: "tenant", label: "公司知识库", icon: Building2 },
];

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
        // 空结果有两种成因，文案必须分开 —— 此前一律显示"可能仍在处理中"，
        // 把"文档已入库完成、但没有任何分块对当前账号可见"也描述成处理中：
        // 用户会一直等一个永远不会出现的进度条。
        // ⚠️ 不写"部分内容因权限不可见"：后端刻意不返回占位符（占位符等于承认
        // "这里有东西被藏了"），前端也不该用文案把它说出来。
        <p className="py-6 text-center text-sm text-muted-foreground">
          {doc.status === "processing"
            ? "该文档仍在处理中，索引完成后即可预览原文。"
            : "暂无可预览的原文内容。若刚刚调整过这份文档的权限，请稍候重试或刷新页面。"}
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
  const { user } = useAuth();
  const router = useRouter();
  const [tier, setTier] = useState<AccessLevel | "all">("all");
  // 公司筛选（仅平台管理员可见）：候选 = GET /companies/accessible，
  // 与可见范围同源；默认「全部公司」。
  const isPlatformAdmin =
    user?.role === "admin" || Boolean(user?.permissions?.includes("*"));
  const [companies, setCompanies] = useState<AccessibleCompany[]>([]);
  const [companyId, setCompanyId] = useState("");
  const { documents, loading, error, refetch } = useDocuments({
    pollProcessing: true,
    accessLevel: tier,
    companyId: companyId || null,
  });

  useEffect(() => {
    if (!isPlatformAdmin) return;
    let cancelled = false;
    fetchAccessibleCompanies()
      .then((list) => {
        if (!cancelled) setCompanies(list);
      })
      .catch(() => {
        /* 候选拉取失败不阻塞文档列表本身 */
      });
    return () => {
      cancelled = true;
    };
  }, [isPlatformAdmin]);

  // admin 尚未创建任何测试公司：fail-closed（空列表 + 引导，不回退全平台）
  const adminHasNoCompany = isPlatformAdmin && companies.length === 0;
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [viewingDoc, setViewingDoc] = useState<Document | null>(null);
  const [sharingDoc, setSharingDoc] = useState<Document | null>(null);
  // 共享弹窗的落地视图：每个入口传入与之语义一致的落点，避免"点申请却落在发布段"
  //   request   「申请共享」入口 → 主视图，且把「申请共享」区提到最前
  //   delete    「申请删除」入口 → 主视图，且把「申请删除」区提到最前
  //   transfer  「转为部门文档」入口 → 直达部门选择二级视图
  //   main      「共享设置」入口 → 主视图常规顺序
  const [sharingView, setSharingView] = useState<SharingDialogView>("main");
  const openSharing = (doc: Document, view: SharingDialogView = "main") => {
    setSharingView(view);
    setSharingDoc(doc);
  };

  const counts = useMemo(() => {
    const base: Record<AccessLevel | "all", number> = {
      all: documents.length,
      private: 0,
      department: 0,
      tenant: 0,
    };
    documents.forEach((doc) => {
      const level = doc.accessLevel ?? "private";
      base[level] += 1;
    });
    return base;
  }, [documents]);

  const handleDelete = async (doc: Document) => {
    setDeletingId(doc.id);
    try {
      await deleteDocument(doc.id);
      toast.success(`已删除「${doc.name}」`);
      refresh();
      refetch();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "删除失败");
    } finally {
      setDeletingId(null);
    }
  };

  /**
   * 「总结此文档」：带着**确切文件名**跳到对话页自动发起总结。
   *
   * 用《》把文件名包起来，后端就能 100% 认得是点名了哪一份文档
   * （见 document_summary_node.resolve_summary_targets），
   * 不会退化成"总结全部文档"。
   */
  const handleSummarize = (doc: Document) => {
    router.push(`/chat?q=${encodeURIComponent(`总结《${doc.name}》`)}`);
  };

  return (
    <Card>
      <CardHeader className="gap-4">
        <div>
          <CardTitle>全部文档</CardTitle>
          <CardDescription>
            按「个人 / 部门 / 公司」三层知识库查看已上传的文档
          </CardDescription>
        </div>

        {/* 公司筛选器：仅平台管理员可见，候选 = 可见范围内的测试公司 */}
        {isPlatformAdmin && (
          <div className="flex items-center gap-2">
            <span className="flex items-center gap-1 text-xs text-muted-foreground">
              <Building2 className="size-3.5" />
              公司
            </span>
            <Select
              aria-label="按公司筛选"
              className="w-56"
              value={companyId}
              onChange={(e) => setCompanyId(e.target.value)}
            >
              <SelectItem value="">
                全部公司（{companies.reduce((sum, c) => sum + c.docCount, 0)}）
              </SelectItem>
              {companies.map((c) => (
                <SelectItem key={c.companyId} value={c.companyId}>
                  {c.companyName}（{c.docCount}）
                </SelectItem>
              ))}
            </Select>
          </div>
        )}

        {/* 三层知识库切换 */}
        <div className="flex flex-wrap items-center gap-1.5">
          {TIER_TABS.map((tab) => {
            const Icon = tab.icon;
            const active = tier === tab.key;
            return (
              <button
                key={tab.key}
                type="button"
                onClick={() => setTier(tab.key)}
                className={cn(
                  "inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1.5 text-xs font-medium transition-colors",
                  active
                    ? "border-primary/50 bg-primary/5 text-foreground"
                    : "border-border/60 text-muted-foreground hover:bg-muted/50 hover:text-foreground"
                )}
              >
                <Icon className="size-3.5" />
                {tab.label}
                {tab.key !== "all" && counts[tab.key] > 0 && (
                  <span className="rounded bg-muted px-1 text-[10px] text-muted-foreground">
                    {counts[tab.key]}
                  </span>
                )}
              </button>
            );
          })}
        </div>
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
            {adminHasNoCompany && !companyId && tier === "all"
              ? "暂无测试公司，请先创建公司。"
              : tier === "all"
                ? "尚未上传任何文档。"
                : `${tierScopeName(tier as AccessLevel)}中还没有文档。`}
          </p>
        ) : (
          <ul className="divide-y divide-border/60">
            {documents.map((doc, index) => {
              const Icon = typeIcons[doc.type] ?? FileText;
              const status = statusConfig[doc.status];
              const canView =
                doc.status === "indexed" || doc.status === "already_exists";
              const level = doc.accessLevel ?? "private";
              const canPublish =
                doc.canPublishDepartment || doc.canPublishCompany;
              // 有"不能直接发、只能申请"的层级 → 显示「申请共享」入口。
              //
              // 不再限定 `level === "private"`：部门负责人把**已发布到部门库**的
              // 文档提到公司库同样只能申请，而他在旧口径下（个人库 + 无发布权）
              // 拿不到这个入口 —— 界面里只有"发布到部门知识库"，向上无路。
              const canRequestShare =
                Boolean(doc.canRequestDepartment || doc.canRequestCompany) &&
                !doc.pendingShareRequest;

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
                    <div className="flex items-center gap-2">
                      <p className="truncate text-sm font-medium">{doc.name}</p>
                      <AccessTierBadge
                        level={level}
                        label={doc.accessLabel}
                        companyName={doc.tenantName}
                        departmentName={doc.departmentName}
                      />
                    </div>
                    <div className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
                      <span>{doc.type}</span>
                      <span>{formatBytes(doc.size)}</span>
                      <span>{doc.chunks} 个分块</span>
                      {/* 图片是独立检索对象（命中后回显原图），列表里要能看见 */}
                      {(doc.imageObjectCount ?? 0) > 0 && (
                        <span className="flex items-center gap-1">
                          <ImageIcon className="size-3" />
                          {doc.imageObjectCount} 张图可检索
                        </span>
                      )}
                      <span className="flex items-center gap-1">
                        <Clock className="size-3" />
                        {formatRelativeTime(doc.uploadedAt)}
                      </span>
                      {!doc.isOwner && doc.ownerUsername && (
                        <span className="flex items-center gap-1">
                          <Users className="size-3" />
                          由 {doc.ownerUsername} 上传
                        </span>
                      )}
                      {doc.pendingShareRequest && (
                        <span className="text-amber-600 dark:text-amber-400">
                          共享申请审核中
                        </span>
                      )}
                    </div>
                  </div>

                  <div className="flex shrink-0 flex-col items-end gap-1">
                    <Badge variant={status.variant}>{status.label}</Badge>
                    {/* 异步入库：把"处理中"讲清楚在哪个阶段，别只转圈 */}
                    {doc.status === "processing" &&
                      describeIngestStage(doc.currentStage, doc.progress) && (
                        <span className="text-[11px] text-muted-foreground">
                          {describeIngestStage(doc.currentStage, doc.progress)}
                        </span>
                      )}
                  </div>

                  {/* 「申请共享」入口：只要存在"只能申请、不能直接发"的层级就显示。
                      以 request 落点打开，让申请区直接置顶，不用再往下找 */}
                  {canRequestShare && (
                    <Button
                      variant="outline"
                      size="sm"
                      className="shrink-0 gap-1.5"
                      onClick={() => openSharing(doc, "request")}
                    >
                      <Share2 className="size-3.5" />
                      申请共享
                    </Button>
                  )}

                  {/* 发布与层级管理。两个入口可同时出现（部门负责人既有部门库的
                      发布权，又有公司库的申请权），弹窗里是同一份能力，不冲突 */}
                  {(canPublish || !canRequestShare) && (
                    <Button
                      variant="ghost"
                      size="icon"
                      className="shrink-0 text-muted-foreground hover:text-foreground"
                      title={canPublish ? "发布与共享设置" : "共享设置"}
                      aria-label={`共享设置 ${doc.name}`}
                      onClick={() => openSharing(doc)}
                    >
                      <Share2 className="size-4" />
                    </Button>
                  )}

                  {/* 公司级管理者：把这份已共享的文档改归到指定部门。
                      入口独立于「共享设置」，因为它是"选择目标部门"的动作，
                      而弹窗主视图里的层级按钮发的是操作者自己的部门。 */}
                  {doc.canTransferDepartment && (
                    <Button
                      variant="ghost"
                      size="icon"
                      className="shrink-0 text-muted-foreground hover:text-foreground"
                      title="转为部门文档"
                      aria-label={`转为部门文档 ${doc.name}`}
                      onClick={() => openSharing(doc, "transfer")}
                    >
                      <FolderInput className="size-4" />
                    </Button>
                  )}

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

                  {/* 总结此文档：只总结这一份（不带上其他文档） */}
                  <Button
                    variant="ghost"
                    size="icon"
                    className="shrink-0 text-muted-foreground hover:text-primary"
                    title={
                      canView
                        ? "只总结这一份文档"
                        : "文档处理完成并完成索引后可总结"
                    }
                    aria-label={`总结 ${doc.name}`}
                    disabled={!canView}
                    onClick={() => handleSummarize(doc)}
                  >
                    <Sparkles className="size-4" />
                  </Button>

                  {/* 无删除权但看得见该文档 → 走「申请删除」由上级审核。
                      与「申请共享」对称：自己没有的权限，通过申请向上要。
                      以 delete 落点打开，直达删除申请区。
                      已有待审申请时收口：后端会以 409 拒绝重复申请，且弹窗此时
                      也不再展示删除区，界面不该留一个注定失败的按钮。 */}
                  {doc.canRequestDelete &&
                    !doc.canDelete &&
                    !doc.pendingShareRequest && (
                      <Button
                        variant="outline"
                        size="sm"
                        className="shrink-0 gap-1.5"
                        title="你没有删除权限，可提交申请由上级审核"
                        onClick={() => openSharing(doc, "delete")}
                      >
                        <FileX2 className="size-3.5" />
                        申请删除
                      </Button>
                    )}

                  {/* 删除：无权限时禁用并把原因写在 title 上（不再点下去才 404） */}
                  <Button
                    variant="ghost"
                    size="icon"
                    className={cn(
                      "shrink-0 text-muted-foreground",
                      doc.canDelete && "hover:text-destructive"
                    )}
                    title={doc.canDelete ? "删除文档" : doc.deleteDeniedReason || "无权删除"}
                    aria-label={`删除 ${doc.name}`}
                    disabled={!doc.canDelete || deletingId === doc.id}
                    onClick={() => handleDelete(doc)}
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

      {/* 共享 / 发布 / 申请弹窗 */}
      <DocumentSharingDialog
        doc={sharingDoc}
        open={!!sharingDoc}
        initialView={sharingView}
        onOpenChange={(open) => {
          if (!open) setSharingDoc(null);
        }}
        onChanged={() => {
          refresh();
          refetch();
        }}
      />
    </Card>
  );
}
