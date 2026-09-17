"use client";

import { useState } from "react";
import {
  ArrowUpRight,
  Building2,
  CheckCircle2,
  Clock,
  FileX2,
  Loader2,
  Lock,
  Send,
  Users,
} from "lucide-react";
import { toast } from "sonner";
import { createDeleteRequest, createShareRequest } from "@/lib/api/share";
import { updateDocumentVisibility } from "@/lib/api/documents";
import type { AccessLevel, Document, ShareTargetLevel } from "@/lib/types";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Textarea } from "@/components/ui/textarea";
import { cn } from "@/lib/utils";
import { AccessTierBadge, tierScopeName } from "@/components/documents/access-badge";

/**
 * 文档的共享面板：一条路径对应权限矩阵里的一行。
 *
 *   有权限（部门负责人/知识库管理员/企业管理员）
 *       → 直接「发布到部门知识库 / 公司知识库」，或把文档收回个人库
 *   无权限（普通员工）
 *       → 「申请共享」：选择目标层级 + 说明理由，提交给上级审核
 *   已有待审申请
 *       → 显示"审核中"，不再重复提交
 *
 * 界面不使用表情符号，全部用图标 + 克制的色彩表达状态。
 */
export function DocumentSharingDialog({
  doc,
  open,
  onOpenChange,
  onChanged,
}: {
  doc: Document | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onChanged?: () => void;
}) {
  const [target, setTarget] = useState<ShareTargetLevel>("department");
  const [reason, setReason] = useState("");
  const [deleteReason, setDeleteReason] = useState("");
  const [busy, setBusy] = useState<string | null>(null);

  if (!doc) return null;

  const current: AccessLevel = doc.accessLevel ?? "private";
  const canPublish = Boolean(doc.canPublishDepartment || doc.canPublishCompany);
  const pending = Boolean(doc.pendingShareRequest);
  // 能直接删就不用申请；两者互斥（后端能力字段同源，按钮状态与接口判定一致）
  const canRequestDelete = Boolean(doc.canRequestDelete && !doc.canDelete);

  const reset = () => {
    setReason("");
    setDeleteReason("");
    setBusy(null);
  };

  const close = (next: boolean) => {
    if (!next) reset();
    onOpenChange(next);
  };

  const publish = async (level: AccessLevel) => {
    setBusy(level);
    try {
      const res = await updateDocumentVisibility(doc.id, level);
      toast.success(res.message || `已更新为${tierScopeName(level)}`);
      onChanged?.();
      close(false);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "发布失败");
    } finally {
      setBusy(null);
    }
  };

  const submitRequest = async () => {
    setBusy("request");
    try {
      await createShareRequest({
        documentId: doc.id,
        targetLevel: target,
        reason: reason.trim() || undefined,
      });
      toast.success("申请已提交，可在「查看申请」跟踪审核进度");
      onChanged?.();
      close(false);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "提交申请失败");
    } finally {
      setBusy(null);
    }
  };

  const submitDeleteRequest = async () => {
    setBusy("delete");
    try {
      await createDeleteRequest({
        documentId: doc.id,
        reason: deleteReason.trim() || undefined,
      });
      toast.success("删除申请已提交，可在「查看申请」跟踪审核进度");
      onChanged?.();
      close(false);
    } catch (err) {
      toast.error(err instanceof Error ? err.message : "提交删除申请失败");
    } finally {
      setBusy(null);
    }
  };

  return (
    <Dialog open={open} onOpenChange={close}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle className="truncate pr-6">{doc.name}</DialogTitle>
          <DialogDescription>
            调整这份文档所在的知识库层级，或向上级申请共享 / 申请删除
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-5">
          {/* 当前层级 */}
          <div className="flex items-center justify-between rounded-lg border border-border/60 bg-muted/30 px-3.5 py-3">
            <div>
              <p className="text-xs text-muted-foreground">当前层级</p>
              <p className="mt-1 text-sm font-medium">{tierScopeName(current)}</p>
            </div>
            <AccessTierBadge level={current} label={doc.accessLabel} />
          </div>

          {/* 待审状态 */}
          {pending && (
            <div className="flex items-start gap-2 rounded-lg border border-amber-300/50 bg-amber-50/70 px-3.5 py-3 text-sm text-amber-800 dark:border-amber-700/40 dark:bg-amber-950/30 dark:text-amber-200">
              <Clock className="mt-0.5 size-4 shrink-0" />
              <span>
                这份文档已有一份待审核的共享申请，审核结果可在主界面「查看申请」中查看。
              </span>
            </div>
          )}

          {/* 有权限：直接发布 */}
          {canPublish && !pending && (
            <div className="space-y-2.5">
              <p className="text-xs font-medium text-muted-foreground">
                直接发布（你的角色具备发布权限）
              </p>

              {doc.canPublishDepartment && current !== "department" && (
                <TierAction
                  icon={Users}
                  title="发布到部门知识库"
                  description="同部门的成员按权限即可检索到这份文档"
                  loading={busy === "department"}
                  disabled={busy !== null}
                  onClick={() => publish("department")}
                />
              )}

              {doc.canPublishCompany && current !== "tenant" && (
                <TierAction
                  icon={Building2}
                  title="发布到公司知识库"
                  description="全公司成员按权限即可检索到这份文档"
                  loading={busy === "tenant"}
                  disabled={busy !== null}
                  onClick={() => publish("tenant")}
                />
              )}

              {current !== "private" && (
                <TierAction
                  icon={Lock}
                  title="收回至个人知识库"
                  description="仅你本人可见，已共享的成员将无法再检索到"
                  loading={busy === "private"}
                  disabled={busy !== null}
                  onClick={() => publish("private")}
                />
              )}
            </div>
          )}

          {/* 无权限：申请共享 */}
          {!canPublish && !pending && (
            <div className="space-y-3">
              <div className="space-y-1.5">
                <p className="text-xs font-medium text-muted-foreground">
                  申请共享到
                </p>
                <div className="grid grid-cols-2 gap-2">
                  <TargetOption
                    active={target === "department"}
                    icon={Users}
                    title="部门知识库"
                    hint="由部门负责人审核"
                    onClick={() => setTarget("department")}
                  />
                  <TargetOption
                    active={target === "tenant"}
                    icon={Building2}
                    title="公司知识库"
                    hint="由知识库管理员审核"
                    onClick={() => setTarget("tenant")}
                  />
                </div>
              </div>

              <div className="space-y-1.5">
                <label className="text-xs font-medium text-muted-foreground">
                  申请说明（可选）
                </label>
                <Textarea
                  value={reason}
                  onChange={(e) => setReason(e.target.value)}
                  placeholder="简要说明为什么需要共享，便于审核人判断"
                  rows={3}
                  maxLength={1000}
                />
              </div>

              <p className="text-xs text-muted-foreground">
                {doc.publishDeniedReason ||
                  "提交后由上一级权限的管理员审核，审核结果会出现在「查看申请」中。"}
              </p>

              <Button
                className="w-full gap-2"
                disabled={busy !== null}
                onClick={submitRequest}
              >
                {busy === "request" ? (
                  <Loader2 className="size-4 animate-spin" />
                ) : (
                  <Send className="size-4" />
                )}
                提交共享申请
              </Button>
            </div>
          )}

          {!canPublish && pending && (
            <p className="text-xs text-muted-foreground">
              审核通过后，文档会自动发布到目标知识库，无需再次操作。
            </p>
          )}

          {/* 没有删除权但看得见该文档 → 申请删除，由上级同意或拒绝 */}
          {canRequestDelete && !pending && (
            <div className="space-y-3 rounded-lg border border-border/60 bg-muted/20 px-3.5 py-3">
              <div className="flex items-start gap-2">
                <FileX2 className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
                <div className="space-y-1">
                  <p className="text-sm font-medium">申请删除</p>
                  <p className="text-xs leading-relaxed text-muted-foreground">
                    {doc.deleteDeniedReason ||
                      `${tierScopeName(current)}的文档由上级统一删除。你可以提交申请说明理由，由 ${
                        current === "tenant" ? "知识库管理员或企业管理员" : "部门负责人"
                      } 审核。`}
                  </p>
                </div>
              </div>

              <Textarea
                value={deleteReason}
                onChange={(e) => setDeleteReason(e.target.value)}
                placeholder="简要说明为什么需要删除这份文档，便于审核人判断"
                rows={3}
                maxLength={1000}
              />

              <Button
                variant="outline"
                className="w-full gap-2"
                disabled={busy !== null}
                onClick={submitDeleteRequest}
              >
                {busy === "delete" ? (
                  <Loader2 className="size-4 animate-spin" />
                ) : (
                  <FileX2 className="size-4" />
                )}
                提交删除申请
              </Button>
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}

function TierAction({
  icon: Icon,
  title,
  description,
  loading,
  disabled,
  onClick,
}: {
  icon: typeof Users;
  title: string;
  description: string;
  loading: boolean;
  disabled: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      className={cn(
        "flex w-full items-center gap-3 rounded-lg border border-border/60 px-3.5 py-3 text-left transition-colors",
        "hover:border-border hover:bg-muted/50",
        disabled && "cursor-not-allowed opacity-60"
      )}
    >
      <span className="flex size-8 shrink-0 items-center justify-center rounded-md bg-muted">
        {loading ? (
          <Loader2 className="size-4 animate-spin text-muted-foreground" />
        ) : (
          <Icon className="size-4 text-muted-foreground" />
        )}
      </span>
      <span className="min-w-0 flex-1">
        <span className="block text-sm font-medium">{title}</span>
        <span className="mt-0.5 block text-xs text-muted-foreground">
          {description}
        </span>
      </span>
      <ArrowUpRight className="size-4 shrink-0 text-muted-foreground" />
    </button>
  );
}

function TargetOption({
  active,
  icon: Icon,
  title,
  hint,
  onClick,
}: {
  active: boolean;
  icon: typeof Users;
  title: string;
  hint: string;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "flex flex-col items-start gap-1 rounded-lg border px-3 py-2.5 text-left transition-colors",
        active
          ? "border-primary/60 bg-primary/5"
          : "border-border/60 hover:bg-muted/50"
      )}
    >
      <span className="flex items-center gap-1.5 text-sm font-medium">
        <Icon className="size-3.5" />
        {title}
        {active && <CheckCircle2 className="size-3.5 text-primary" />}
      </span>
      <span className="text-[11px] text-muted-foreground">{hint}</span>
    </button>
  );
}
