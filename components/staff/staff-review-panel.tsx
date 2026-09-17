"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  AlertCircle,
  Building2,
  CheckCircle2,
  Clock,
  Loader2,
  Send,
  UserCheck,
  Users,
  XCircle,
} from "lucide-react";
import { toast } from "sonner";
import { fetchStaffProfile, getStaffInbox, reviewStaffRequest } from "@/lib/api/staff";
import { useAuth } from "@/lib/context/auth-context";
import type { StaffProfile, StaffRequestItem, StaffRequestStatus } from "@/lib/types";
import { ROLE_LABELS, roleRank } from "@/lib/types";
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
import { Input } from "@/components/ui/input";
import { Select, SelectItem } from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";

const STATUS_META: Record<
  StaffRequestStatus,
  { label: string; variant: "success" | "warning" | "destructive" | "secondary"; icon: typeof Clock }
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

/**
 * 身份验证审核队列（部门负责人及以上可见）.
 *
 * 「越级审核」体现在后端：审核人等级只要**严格高于**申请人，就能直接处理，
 * 不必等他直属上级先过一遍。前端不做额外限制，只把后端下发的 ``canReview``
 * 渲染出来 —— 能点的按钮与后端放行的范围永远一致。
 *
 * 同意 / 拒绝都必须填写「负责人 = 职务 + 名称」：产品要求审核结论后方要能
 * 看到是谁、以什么职务批的。表单默认用审核人自己的职务与姓名预填，减少输入。
 */
export function StaffReviewPanel() {
  const { user } = useAuth();
  const [items, setItems] = useState<StaffRequestItem[]>([]);
  const [profile, setProfile] = useState<StaffProfile | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<"pending" | "all">("pending");
  const [target, setTarget] = useState<{
    item: StaffRequestItem;
    approve: boolean;
  } | null>(null);

  const load = useCallback(async () => {
    setError(null);
    try {
      const [inbox, me] = await Promise.all([getStaffInbox(), fetchStaffProfile()]);
      setItems(inbox.items);
      setProfile(me);
    } catch (err) {
      setError(err instanceof Error ? err.message : "加载审核队列失败");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const pendingItems = useMemo(
    () => items.filter((item) => item.status === "pending"),
    [items]
  );
  const visible = tab === "pending" ? pendingItems : items;

  const handleDone = async () => {
    setTarget(null);
    await load();
    window.dispatchEvent(new Event("staff-requests-changed"));
  };

  return (
    <Card>
      <CardHeader className="gap-4">
        <div>
          <CardTitle>身份验证审核</CardTitle>
          <CardDescription>
            下级成员提交的企业身份申请在这里处理；你的等级高于申请人时可直接越级审核
          </CardDescription>
        </div>

        <div className="flex flex-wrap items-center gap-1.5">
          {[
            { key: "pending" as const, label: "待我审核", count: pendingItems.length },
            { key: "all" as const, label: "全部记录", count: items.length },
          ].map((item) => (
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
        ) : visible.length === 0 ? (
          <p className="px-6 py-8 text-center text-sm text-muted-foreground">
            {tab === "pending"
              ? "当前没有需要你审核的身份申请。"
              : "还没有任何身份申请记录。"}
          </p>
        ) : (
          <div className="divide-y divide-border/60">
            {visible.map((item) => (
              <ReviewRow
                key={item.id}
                item={item}
                onReview={(approve) => setTarget({ item, approve })}
              />
            ))}
          </div>
        )}
      </CardContent>

      <ReviewDialog
        target={target}
        profile={profile}
        currentUser={user}
        onClose={() => setTarget(null)}
        onDone={handleDone}
      />
    </Card>
  );
}

function ReviewRow({
  item,
  onReview,
}: {
  item: StaffRequestItem;
  onReview: (approve: boolean) => void;
}) {
  const meta = STATUS_META[item.status];
  const Icon = meta.icon;
  const pending = item.status === "pending";

  return (
    <div className="flex flex-col gap-3 px-6 py-4 transition-colors hover:bg-muted/30 sm:flex-row sm:items-center sm:gap-4">
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-2">
          <p className="truncate text-sm font-medium">{item.applicantLabel}</p>
          <span className="rounded bg-muted px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground">
            {item.applicantUsername}
          </span>
        </div>
        <p className="mt-1 flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-muted-foreground">
          <span className="inline-flex items-center gap-1">
            <Building2 className="size-3" />
            {item.companyName}
          </span>
          <span className="inline-flex items-center gap-1">
            <Users className="size-3" />
            {item.departmentName}
          </span>
          <span>职责：{item.duty}</span>
          <span>{formatTime(item.createdAt)}</span>
        </p>
        {/* 审核结论后方显示「负责人 = 职务 + 名称」 */}
        {item.reviewerLabel && (
          <p className="mt-1 text-xs text-foreground/80">
            负责人：{item.reviewerLabel}
            {item.grantedRoleLabel && ` · 授予「${item.grantedRoleLabel}」`}
            {item.reviewedAt && ` · ${formatTime(item.reviewedAt)}`}
          </p>
        )}
        {item.reviewComment && (
          <p className="mt-0.5 text-xs text-muted-foreground">
            审核意见：{item.reviewComment}
          </p>
        )}
      </div>

      <div className="flex shrink-0 items-center gap-2">
        {pending && item.canReview ? (
          <>
            <Button size="sm" className="gap-1.5" onClick={() => onReview(true)}>
              <CheckCircle2 className="size-3.5" />
              同意
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="gap-1.5"
              onClick={() => onReview(false)}
            >
              <XCircle className="size-3.5" />
              拒绝
            </Button>
          </>
        ) : pending && !item.canReview ? (
          <Badge variant="secondary" className="gap-1">
            <Clock className="size-3" />
            待上级审核
          </Badge>
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

function ReviewDialog({
  target,
  profile,
  currentUser,
  onClose,
  onDone,
}: {
  target: { item: StaffRequestItem; approve: boolean } | null;
  profile: StaffProfile | null;
  currentUser: {
    display_name?: string | null;
    username: string;
    job_title?: string | null;
    role?: string;
  } | null;
  onClose: () => void;
  onDone: () => void;
}) {
  const [title, setTitle] = useState("");
  const [name, setName] = useState("");
  const [role, setRole] = useState("employee");
  const [department, setDepartment] = useState("");
  const [duty, setDuty] = useState("");
  const [comment, setComment] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const grantable = useMemo(() => profile?.grantableRoles ?? [], [profile]);

  // 每次打开都用「我自己的职务 + 姓名」预填，并带上申请人的部门/职责
  useEffect(() => {
    if (!target) return;
    setTitle(currentUser?.job_title || profile?.jobTitle || "部门负责人");
    setName(currentUser?.display_name || currentUser?.username || "");
    setRole(grantable.includes("employee") ? "employee" : (grantable[0] ?? "employee"));
    setDepartment(target.item.departmentName);
    setDuty(target.item.duty);
    setComment("");
    setError(null);
  }, [target, currentUser, profile, grantable]);

  if (!target) return null;

  const { item, approve } = target;

  /** 可授予的角色必须严格低于审核人等级 —— 与后端 can_grant_role 同一规则。 */
  const reviewerRank = roleRank(currentUser?.role ?? profile?.role);
  const options = grantable.filter((value) => roleRank(value) < reviewerRank);

  const submit = async () => {
    setError(null);
    if (!title.trim()) {
      setError("请填写你的职务");
      return;
    }
    if (!name.trim()) {
      setError("请填写你的姓名");
      return;
    }
    if (approve && !role) {
      setError("请选择要授予的职责");
      return;
    }

    setSubmitting(true);
    try {
      const res = await reviewStaffRequest({
        requestId: item.id,
        approve,
        reviewerTitle: title.trim(),
        reviewerName: name.trim(),
        role: approve ? role : undefined,
        departmentName: approve ? department.trim() || undefined : undefined,
        duty: approve ? duty.trim() || undefined : undefined,
        comment: comment.trim() || undefined,
      });
      toast.success(res.message || "已处理");
      onDone();
    } catch (err) {
      setError(err instanceof Error ? err.message : "操作失败");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Dialog open onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-md">
        <DialogHeader>
          <DialogTitle>{approve ? "同意身份验证申请" : "拒绝身份验证申请"}</DialogTitle>
          <DialogDescription>
            {item.applicantLabel} · {item.companyName} · {item.departmentName}
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <div className="rounded-lg border border-border/60 bg-muted/30 px-3.5 py-3 text-[12px] text-muted-foreground">
            <p>申请人：{item.applicantLabel}（{item.applicantUsername}）</p>
            <p className="mt-1">
              申请归属：{item.companyName} · {item.departmentName}
            </p>
            <p className="mt-1">申请职责：{item.duty}</p>
            {approve && (
              <p className="mt-1.5 text-foreground/80">
                同意后该成员会加入上述公司与部门，并获得你选定职责的权限。
              </p>
            )}
          </div>

          {/* 负责人（职务 + 名称）——通过或拒绝都必须填写 */}
          <div className="space-y-1.5">
            <label className="text-[12px] font-medium text-muted-foreground">
              负责人（职务 + 名称，将显示在该申请记录后方）
            </label>
            <div className="grid grid-cols-2 gap-2">
              <Input
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                placeholder="职务，如 研发部负责人"
                maxLength={64}
                disabled={submitting}
              />
              <Input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="姓名，如 李四"
                maxLength={64}
                disabled={submitting}
              />
            </div>
          </div>

          {approve && (
            <>
              <div className="space-y-1.5">
                <label className="text-[12px] font-medium text-muted-foreground">
                  授予职责
                </label>
                <Select
                  value={role}
                  onChange={(e) => setRole(e.target.value)}
                  disabled={submitting}
                >
                  {options.length === 0 && (
                    <SelectItem value="employee">普通员工</SelectItem>
                  )}
                  {options.map((value) => (
                    <SelectItem key={value} value={value}>
                      {ROLE_LABELS[value] ?? value}
                    </SelectItem>
                  ))}
                </Select>
                <p className="text-[11px] text-muted-foreground">
                  只能授予低于你自己等级的职责（不能越权造出同级或更高权限）
                </p>
              </div>

              <div className="grid grid-cols-2 gap-2">
                <div className="space-y-1.5">
                  <label className="text-[12px] font-medium text-muted-foreground">
                    实际分配部门
                  </label>
                  <Input
                    value={department}
                    onChange={(e) => setDepartment(e.target.value)}
                    maxLength={128}
                    disabled={submitting}
                  />
                </div>
                <div className="space-y-1.5">
                  <label className="text-[12px] font-medium text-muted-foreground">
                    实际职责
                  </label>
                  <Input
                    value={duty}
                    onChange={(e) => setDuty(e.target.value)}
                    maxLength={128}
                    disabled={submitting}
                  />
                </div>
              </div>
            </>
          )}

          <div className="space-y-1.5">
            <label className="text-[12px] font-medium text-muted-foreground">
              审批意见（可选，申请人可见）
            </label>
            <Textarea
              value={comment}
              onChange={(e) => setComment(e.target.value)}
              rows={2}
              maxLength={1000}
              placeholder="例如：已核对工号，同意加入研发部"
            />
          </div>

          {error && (
            <p className="flex items-start gap-2 rounded-lg border border-destructive/20 bg-destructive/5 px-3 py-2 text-[12px] text-destructive">
              <AlertCircle className="mt-0.5 size-3.5 shrink-0" />
              {error}
            </p>
          )}

          <Button
            className={cn(
              "w-full gap-2",
              !approve && "bg-destructive hover:bg-destructive/90"
            )}
            disabled={submitting}
            onClick={submit}
          >
            {submitting ? (
              <Loader2 className="size-4 animate-spin" />
            ) : approve ? (
              <UserCheck className="size-4" />
            ) : (
              <Send className="size-4" />
            )}
            {approve ? "确认同意并开通权限" : "确认拒绝"}
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
