"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  AlertCircle,
  Building2,
  CheckCircle2,
  Clock,
  FileX2,
  Loader2,
  Send,
  Users,
  XCircle,
} from "lucide-react";
import { toast } from "sonner";
import {
  cancelShareRequest,
  getMyShareRequests,
  getShareRequestInbox,
  markMyShareRequestsSeen,
  reviewShareRequest,
} from "@/lib/api/share";
import { useAuth } from "@/lib/context/auth-context";
import type { ShareRequestItem, ShareRequestStatus } from "@/lib/types";
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
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";

const STATUS_META: Record<
  ShareRequestStatus,
  {
    label: string;
    variant: "success" | "warning" | "destructive" | "secondary";
    icon: typeof Clock;
  }
> = {
  pending: { label: "待审核", variant: "warning", icon: Clock },
  approved: { label: "已通过", variant: "success", icon: CheckCircle2 },
  rejected: { label: "已拒绝", variant: "destructive", icon: XCircle },
  cancelled: { label: "已撤回", variant: "secondary", icon: XCircle },
};

function formatTime(value?: string | null): string {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function TargetTag({ item }: { item: ShareRequestItem }) {
  const isDelete = item.intent === "delete";
  const Icon = isDelete
    ? FileX2
    : item.targetLevel === "tenant"
      ? Building2
      : Users;
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 rounded-md border px-1.5 py-0.5 text-[11px]",
        isDelete
          ? "border-destructive/40 bg-destructive/5 text-destructive"
          : "border-border/60 bg-muted/40 text-muted-foreground"
      )}
    >
      <Icon className="size-3" />
      {isDelete ? `申请删除 · ${item.targetLabel}` : item.targetLabel}
    </span>
  );
}

/** 「我的申请」条目 —— 是否通过一眼可见。 */
function MyRequestRow({
  item,
  onCancel,
  cancelling,
}: {
  item: ShareRequestItem;
  onCancel: (item: ShareRequestItem) => void;
  cancelling: boolean;
}) {
  const meta = STATUS_META[item.status];
  const Icon = meta.icon;

  return (
    <div className="flex flex-col gap-2 px-6 py-4 transition-colors hover:bg-muted/30 sm:flex-row sm:items-center sm:gap-4">
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-2">
          <p className="truncate text-sm font-medium">{item.documentName}</p>
          <TargetTag item={item} />
          {!item.requesterSeen && item.status !== "pending" && (
            <span className="rounded bg-primary/10 px-1.5 py-0.5 text-[10px] font-medium text-primary">
              新结果
            </span>
          )}
        </div>
        <p className="mt-1 text-xs text-muted-foreground">
          提交于 {formatTime(item.createdAt)}
          {item.reviewerUsername && ` · 审核人 ${item.reviewerUsername}`}
          {item.reviewedAt && ` · ${formatTime(item.reviewedAt)}`}
        </p>
        {item.reason && (
          <p className="mt-1 line-clamp-2 text-xs text-muted-foreground">
            申请说明：{item.reason}
          </p>
        )}
        {item.reviewComment && (
          <p className="mt-1 text-xs text-foreground/80">
            审核意见：{item.reviewComment}
          </p>
        )}
      </div>

      <div className="flex shrink-0 items-center gap-2">
        <Badge variant={meta.variant} className="gap-1">
          <Icon className="size-3" />
          {meta.label}
        </Badge>
        {item.status === "pending" && (
          <Button
            variant="ghost"
            size="sm"
            className="text-muted-foreground hover:text-destructive"
            disabled={cancelling}
            onClick={() => onCancel(item)}
          >
            {cancelling ? <Loader2 className="size-3.5 animate-spin" /> : "撤回"}
          </Button>
        )}
      </div>
    </div>
  );
}

/** 「待我审核」条目 —— 低级权限者的申请在这里被同意 / 拒绝。 */
function InboxRow({
  item,
  onReview,
}: {
  item: ShareRequestItem;
  onReview: (item: ShareRequestItem, approve: boolean) => void;
}) {
  const meta = STATUS_META[item.status];
  const Icon = meta.icon;
  const pending = item.status === "pending";

  return (
    <div className="flex flex-col gap-2 px-6 py-4 transition-colors hover:bg-muted/30 sm:flex-row sm:items-center sm:gap-4">
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-2">
          <p className="truncate text-sm font-medium">{item.documentName}</p>
          <TargetTag item={item} />
        </div>
        <p className="mt-1 text-xs text-muted-foreground">
          申请人 {item.requesterUsername}
          {item.requesterDepartmentId && ` · 部门 ${item.requesterDepartmentId}`}
          {` · ${formatTime(item.createdAt)}`}
        </p>
        {item.reason && (
          <p className="mt-1 text-xs text-muted-foreground">
            申请说明：{item.reason}
          </p>
        )}
        {item.reviewComment && (
          <p className="mt-1 text-xs text-foreground/80">
            我的意见：{item.reviewComment}
          </p>
        )}
      </div>

      <div className="flex shrink-0 items-center gap-2">
        {pending ? (
          <>
            <Button
              size="sm"
              className="gap-1.5"
              onClick={() => onReview(item, true)}
            >
              <CheckCircle2 className="size-3.5" />
              同意
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="gap-1.5"
              onClick={() => onReview(item, false)}
            >
              <XCircle className="size-3.5" />
              拒绝
            </Button>
          </>
        ) : (
          <Badge variant={meta.variant} className="gap-1">
            <Icon className="size-3" />
            {meta.label}
          </Badge>
        )}
      </div>
    </div>
  );
}

export function ShareRequestsPanel() {
  const { user } = useAuth();
  const [tab, setTab] = useState<"mine" | "inbox">("mine");
  const [mine, setMine] = useState<ShareRequestItem[]>([]);
  const [inbox, setInbox] = useState<ShareRequestItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [cancellingId, setCancellingId] = useState<string | null>(null);
  const [reviewTarget, setReviewTarget] = useState<{
    item: ShareRequestItem;
    approve: boolean;
  } | null>(null);
  const [comment, setComment] = useState("");
  const [reviewing, setReviewing] = useState(false);

  const load = useCallback(async () => {
    setError(null);
    try {
      const [mineItems, inboxRes] = await Promise.all([
        getMyShareRequests(),
        getShareRequestInbox(),
      ]);
      setMine(mineItems);
      setInbox(inboxRes.items);
    } catch (err) {
      setError(err instanceof Error ? err.message : "加载申请记录失败");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  // 打开这一页即视为"看到了审核结果"，清掉主界面角标
  useEffect(() => {
    const hasUnseen = mine.some((item) => !item.requesterSeen && item.status !== "pending");
    if (!hasUnseen) return;
    markMyShareRequestsSeen()
      .then(() => {
        window.dispatchEvent(new Event("share-requests-changed"));
      })
      .catch(() => {
        /* 角标清理失败不影响主体功能 */
      });
  }, [mine]);

  const pendingInbox = useMemo(
    () => inbox.filter((item) => item.status === "pending"),
    [inbox]
  );

  const handleCancel = async (item: ShareRequestItem) => {
    setCancellingId(item.id);
    try {
      await cancelShareRequest(item.id);
      toast.success("申请已撤回");
      await load();
      window.dispatchEvent(new Event("share-requests-changed"));
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "撤回失败");
    } finally {
      setCancellingId(null);
    }
  };

  const submitReview = async () => {
    if (!reviewTarget) return;
    setReviewing(true);
    try {
      const res = await reviewShareRequest({
        requestId: reviewTarget.item.id,
        approve: reviewTarget.approve,
        comment: comment.trim() || undefined,
      });
      toast.success(res.message || "已处理");
      setReviewTarget(null);
      setComment("");
      await load();
      window.dispatchEvent(new Event("share-requests-changed"));
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "操作失败");
    } finally {
      setReviewing(false);
    }
  };

  const tabs = [
    { key: "mine" as const, label: "我的申请", count: mine.length },
    ...(user?.permissions?.includes("share.review.department") ||
    user?.permissions?.includes("share.review.company") ||
    user?.permissions?.includes("*")
      ? [
          {
            key: "inbox" as const,
            label: "待我审核",
            count: pendingInbox.length,
          },
        ]
      : []),
  ];

  return (
    <Card>
      <CardHeader className="gap-4">
        <div>
          <CardTitle>共享申请</CardTitle>
          <CardDescription>
            查看自己提交的申请是否通过；有审核权限时，可直接处理下级成员的申请
          </CardDescription>
        </div>

        <div className="flex flex-wrap items-center gap-1.5">
          {tabs.map((item) => (
            <button
              key={item.key}
              type="button"
              onClick={() => setTab(item.key)}
              className={cn(
                "inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1.5 text-xs font-medium transition-colors",
                tab === item.key
                  ? "border-primary/50 bg-primary/5 text-foreground"
                  : "border-border/60 text-muted-foreground hover:bg-muted/50 hover:text-foreground"
              )}
            >
              {item.label}
              {item.count > 0 && (
                <span className="rounded bg-muted px-1 text-[10px] text-muted-foreground">
                  {item.count}
                </span>
              )}
            </button>
          ))}
        </div>
      </CardHeader>

      <CardContent className="px-0">
        {error ? (
          <div className="flex items-center gap-3 px-6 py-8 text-sm text-destructive">
            <AlertCircle className="size-4 shrink-0" />
            <span>{error}</span>
          </div>
        ) : loading ? (
          <div className="space-y-3 px-6 py-2">
            {Array.from({ length: 3 }).map((_, i) => (
              <Skeleton key={i} className="h-16 w-full" />
            ))}
          </div>
        ) : tab === "mine" ? (
          mine.length === 0 ? (
            <p className="px-6 py-8 text-center text-sm text-muted-foreground">
              你还没有提交过共享申请。在「文档」页的个人文档上点击「申请共享」即可提交；
              需要删除一份自己没有删除权的部门/公司文档时，点击「申请删除」。
            </p>
          ) : (
            <div className="divide-y divide-border/60">
              {mine.map((item) => (
                <MyRequestRow
                  key={item.id}
                  item={item}
                  onCancel={handleCancel}
                  cancelling={cancellingId === item.id}
                />
              ))}
            </div>
          )
        ) : inbox.length === 0 ? (
          <p className="px-6 py-8 text-center text-sm text-muted-foreground">
            当前没有需要你审核的申请。
          </p>
        ) : (
          <div className="divide-y divide-border/60">
            {inbox.map((item) => (
              <InboxRow
                key={item.id}
                item={item}
                onReview={(target, approve) => {
                  setReviewTarget({ item: target, approve });
                  setComment("");
                }}
              />
            ))}
          </div>
        )}
      </CardContent>

      {/* 审核弹窗：同意 / 拒绝 + 审批意见 */}
      <Dialog
        open={!!reviewTarget}
        onOpenChange={(open) => {
          if (!open) setReviewTarget(null);
        }}
      >
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>
              {reviewTarget?.item.intent === "delete"
                ? reviewTarget?.approve
                  ? "同意删除申请"
                  : "拒绝删除申请"
                : reviewTarget?.approve
                  ? "同意共享申请"
                  : "拒绝共享申请"}
            </DialogTitle>
            <DialogDescription>
              {reviewTarget?.item.documentName} ·{" "}
              {reviewTarget?.item.targetLabel}
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4">
            <div className="rounded-lg border border-border/60 bg-muted/30 px-3.5 py-3 text-xs text-muted-foreground">
              <p>申请人：{reviewTarget?.item.requesterUsername}</p>
              {reviewTarget?.item.reason && (
                <p className="mt-1">说明：{reviewTarget.item.reason}</p>
              )}
              {reviewTarget?.approve &&
                (reviewTarget.item.intent === "delete" ? (
                  <p className="mt-2 font-medium text-destructive">
                    同意后，这份文档将从知识库中**永久删除**（含全部向量索引），
                    该范围内的成员将无法再检索到。此操作不可撤销。
                  </p>
                ) : (
                  <p className="mt-2 text-foreground/80">
                    同意后，这份文档会立即发布到{reviewTarget.item.targetLabel}
                    ，该范围内的成员即可检索到。
                  </p>
                ))}
            </div>

            <div className="space-y-1.5">
              <label className="text-xs font-medium text-muted-foreground">
                审批意见（可选，申请人可见）
              </label>
              <Textarea
                value={comment}
                onChange={(e) => setComment(e.target.value)}
                rows={3}
                maxLength={1000}
                placeholder="例如：确认可对外共享；或说明拒绝原因"
              />
            </div>

            <Button
              className={cn(
                "w-full gap-2",
                !reviewTarget?.approve && "bg-destructive hover:bg-destructive/90"
              )}
              disabled={reviewing}
              onClick={submitReview}
            >
              {reviewing ? (
                <Loader2 className="size-4 animate-spin" />
              ) : reviewTarget?.approve ? (
                <CheckCircle2 className="size-4" />
              ) : (
                <Send className="size-4" />
              )}
              {reviewTarget?.approve
                ? reviewTarget?.item.intent === "delete"
                  ? "确认同意并删除"
                  : "确认同意并发布"
                : "确认拒绝"}
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </Card>
  );
}
